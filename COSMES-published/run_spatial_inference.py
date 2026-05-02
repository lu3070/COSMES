#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COSMES — Core Spatial Intelligence & Reasoning
================================================
Unified spatial reasoning framework integrating Spatial-MLLM with VSI-Bench tasks,
featuring six innovative modules with offline CPU/GPU support.

Quick Start
-----------
  # Quick validation (CPU, 1 question, 4 frames)
  python run_spatial_inference.py --mode cpu --quick

  # CPU + all post-processing innovations
  python run_spatial_inference.py --mode cpu --geo-correct --hallucination-check --cross-validate

  # CPU + depth-aware sampling + edge enhancement + CoT
  python run_spatial_inference.py --mode cpu --depth-aware --edge-enhance --spatial-cot

  # GPU + multi-view ensemble (3 views) + uncertainty quantification
  python run_spatial_inference.py --mode gpu --multi-view 3 --uncertainty 2

Innovation Index
----------------
  --depth-aware       Depth + viewpoint change driven frame sampling
  --dual-branch       Semantic-geometric dual-branch prompt aggregation
  --spatial-cot       Spatial chain-of-thought structured prompting
  --cross-validate    Cross-task answer consistency validation
  --hallucination-check  Hallucination detection using rules and priors
  --lora-path         PEFT LoRA adapter loading
  --geo-correct       Geometric prior-based self-correction
  --uncertainty N     Bayesian uncertainty quantification via multiple sampling
  --edge-enhance      Laplacian edge / CLAHE contrast enhancement
  --multi-view N      Multi-view ensemble inference
  --cluster-sample    Histogram-based greedy farthest-point sampling
  --no-vggt           Pure vision-only backend (ablation baseline)
"""

import os
import sys
import re
import gc
import argparse
import statistics
import math
import torch
import time
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# ── Windows UTF-8 console support ────────────────────────────────────────────
if sys.platform == "win32":
    try:
        os.system("chcp 65001 >nul 2>&1")
    except Exception:
        pass

# ── Force offline mode (no network access) ──────────────────────────────────
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ── CPU mode: hide GPU, set thread count ────────────────────────────────────
_cpu_mode = False
for _i, _a in enumerate(sys.argv):
    if _a == "--cpu" or (_a == "--mode" and _i + 1 < len(sys.argv) and sys.argv[_i + 1] == "cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _cpu_mode = True
        break

if _cpu_mode:
    _ncpu = max(1, (os.cpu_count() or 4) // 2)
    os.environ.setdefault("OMP_NUM_THREADS", str(_ncpu))
    os.environ.setdefault("MKL_NUM_THREADS", str(_ncpu))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_ncpu))

# ── Add project root to sys.path and set working directory ──────────────────
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ_ROOT)
# Change working directory to ensure relative paths work consistently
os.chdir(_PROJ_ROOT)


# =============================================================================
# Utility functions: model path / memory management
# =============================================================================

def get_local_model_path(model_id: str = "Diankun/Spatial-MLLM-subset-sft",
                         custom_dir: str = None) -> str:
    """Resolve local model path, prefer custom_dir, otherwise search HF cache."""
    if custom_dir and os.path.isdir(custom_dir):
        if os.path.isfile(os.path.join(custom_dir, "config.json")):
            return custom_dir
        for d in os.listdir(custom_dir):
            p = os.path.join(custom_dir, d)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                return p
        return custom_dir
    hub = (os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
           or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    repo = os.path.join(hub, "models--" + model_id.replace("/", "--"), "snapshots")
    if os.path.isdir(repo):
        for rev in os.listdir(repo):
            snap = os.path.join(repo, rev)
            if os.path.isdir(snap) and os.path.isfile(os.path.join(snap, "config.json")):
                return snap
    return model_id


def apply_cpu_memory_optimizations(model, device: str):
    """CPU mode: force float32, set eval mode, and configure PyTorch threads."""
    if device != "cpu":
        return model
    model = model.to(torch.float32)
    model.eval()
    ncpu = max(1, (os.cpu_count() or 4) // 2)
    torch.set_num_threads(ncpu)
    torch.set_num_interop_threads(max(1, ncpu // 2))
    print(f"  [CPU] float32 | threads={ncpu}", flush=True)
    return model


def free_memory(device: str = "cpu"):
    """Free memory after inference: GC for CPU, empty_cache for GPU."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def cpu_friendly_generate(model, inputs: dict, max_new_tokens: int,
                           temperature: float = 0.1, top_p: float = 0.001):
    """
    CPU-specific generation function:
    - Limit to 256 tokens to avoid OOM
    - Use torch.inference_mode (no autograd graph)
    - Disable sampling at low temperature for speed
    """
    max_new_tokens = min(max_new_tokens, 256)
    do_sample = temperature > 0.05
    gen_kwargs: Dict[str, Any] = dict(max_new_tokens=max_new_tokens,
                                       do_sample=do_sample, use_cache=True)
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.inference_mode():
        return model.generate(**inputs, **gen_kwargs)


# =============================================================================
# Answer parsing
# =============================================================================

def _extract_tag(raw: str, tag: str) -> str:
    """Extract content between <tag>...</tag>, supports unclosed tags."""
    if not raw:
        return ""
    m = re.compile(rf"<{tag}\s*>(.*?)</{tag}\s*>", re.DOTALL | re.IGNORECASE).search(raw)
    if m:
        return m.group(1).strip()
    m = re.compile(rf"<{tag}\s*>(.*)", re.DOTALL | re.IGNORECASE).search(raw)
    return m.group(1).strip() if m else ""


def extract_thinking_and_answer(raw: str) -> Tuple[str, str]:
    """Extract (thinking, answer) from model output; use full text as answer if no <answer> tag."""
    if not raw:
        return "", ""
    raw = raw.strip()
    thinking = _extract_tag(raw, "thinking")
    answer = _extract_tag(raw, "answer")
    if not answer:
        body = re.sub(r"<thinking\s*>.*?(?:</thinking\s*>|$)", "", raw,
                      flags=re.DOTALL | re.IGNORECASE).strip()
        answer = body or raw
    # Clean short garbled strings: extract number only if truly garbled (no letters/decimal points)
    if answer and len(answer) <= 30:
        if (re.match(r"^[\w=#$%^&*@\-]+$", answer)        # Pure symbol string without .
                and not re.search(r"[a-zA-Z]", answer)    # No letters (preserve numeric answers)
                and not re.search(r"[\u4e00-\u9fff]", answer)):
            digits = re.sub(r"\D", "", answer)
            if digits:
                answer = digits
    return thinking, answer


def _try_num(s: str) -> Optional[float]:
    """Try to extract the first number from string."""
    m = re.search(r"-?\d+(?:\.\d+)?", (s or "").strip())
    return float(m.group()) if m else None


# =============================================================================
# Question lists: Standard + Spatial CoT versions
# =============================================================================

SPATIAL_QUESTIONS = [
    {
        "task": "Object Counting",
        "question": (
            "How many chair(s) are in this room? "
            "Please answer with only the numerical value within the <answer></answer> tags."
        ),
    },
    {
        "task": "Spatial Layout",
        "question": (
            "Describe the spatial layout of this room. "
            "Please answer within the <answer></answer> tags."
        ),
    },
    {
        "task": "Distance Estimation",
        "question": (
            "What is the approximate distance from the camera to the nearest object? "
            "Please answer with only the numerical value (meters) within the <answer></answer> tags."
        ),
    },
]

SPATIAL_COT_QUESTIONS = [
    {
        "task": "Object Counting [CoT]",
        "question": (
            "Think step by step, then answer: "
            "How many chair(s) are in this room? "
            "First reason briefly about what you see, then give only the numerical value "
            "within the <answer></answer> tags."
        ),
    },
    {
        "task": "Spatial Layout [CoT]",
        "question": (
            "Think step by step, then answer: "
            "Describe the spatial layout of this room. "
            "First reason briefly about the scene structure, then give your description "
            "within the <answer></answer> tags."
        ),
    },
    {
        "task": "Distance Estimation [CoT]",
        "question": (
            "Think step by step, then answer: "
            "What is the approximate distance from the camera to the nearest object? "
            "First reason briefly about the nearest object and depth cues, "
            "then give only the numerical value in meters within the <answer></answer> tags."
        ),
    },
]

_COT_MAP = {q["task"].replace(" [CoT]", ""): q for q in SPATIAL_COT_QUESTIONS}

# Dual-branch prompt prefixes
_PREFIX_SEMANTIC = "Focus on semantic content, objects, colors and scene category. "
_PREFIX_GEOMETRIC = "Focus on geometric structure, edges, depth and 3D spatial layout. "


# =============================================================================
# Multi-view ensemble aggregation
# =============================================================================

def aggregate_multi_view(all_results: List[List[Dict]]) -> List[Dict]:
    """
    Aggregate N view inference results:
    - Numeric tasks: take median
    - Descriptive tasks: take longest answer
    """
    if not all_results:
        return []
    N = len(all_results)
    Q = len(all_results[0])
    out = []
    for qi in range(Q):
        task = all_results[0][qi]["task"]
        answers = [all_results[v][qi]["answer"] for v in range(N)]
        raws = [all_results[v][qi].get("raw_output", "") for v in range(N)]
        times = [all_results[v][qi]["time_sec"] for v in range(N)]

        is_desc = "layout" in task.lower()
        if is_desc:
            best = max(answers, key=lambda a: len(a) if a else 0)
            best_raw = raws[answers.index(best)]
        else:
            nums = [_try_num(a) for a in answers if _try_num(a) is not None]
            if nums:
                med = statistics.median(nums)
                best = str(int(med)) if med == int(med) else f"{med:.2f}"
            else:
                best = Counter(answers).most_common(1)[0][0]
            best_raw = "\n\n".join(f"[View {v+1}] {raws[v]}" for v in range(N) if raws[v])

        out.append({
            "task": task,
            "question": all_results[0][qi]["question"],
            "thinking": "",
            "answer": best,
            "raw_output": best_raw,
            "time_sec": sum(times),
            "tokens": sum(all_results[v][qi].get("tokens", 0) for v in range(N)),
            "multi_view_answers": answers,
        })
    return out


# =============================================================================
# LoRA loading (training efficiency innovation)
# =============================================================================

def load_lora(model, lora_path: Optional[str]):
    """Load and merge LoRA weights using PEFT if lora_path is valid; skip otherwise."""
    if not lora_path:
        return model
    if not os.path.isdir(lora_path):
        print(f"[LoRA] Invalid path, skipping: {lora_path}")
        return model
    try:
        from peft import PeftModel  # type: ignore[import-not-found]
        print(f"[LoRA] Loading: {lora_path}", flush=True)
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
        model = model.merge_and_unload()
        print("[LoRA] Merged to model", flush=True)
    except ImportError:
        print("[LoRA] peft not installed, skipping (pip install peft)")
    except Exception as e:
        print(f"[LoRA] Load failed, skipping: {e}")
    return model


# =============================================================================
# Core generation functions
# =============================================================================

def _single_generate(model, processor, messages: list,
                     video_inputs_cached, vid_tensor,
                     device: str, max_new_tokens: int,
                     temperature: float = 0.1, top_p: float = 0.001,
                     is_pure_vl: bool = False) -> str:
    """
    Execute one inference generation.
    - CPU: call cpu_friendly_generate, release tensors after inference
    - GPU: standard torch.no_grad path
    - Don't pass videos_input when is_pure_vl=True (no-VGGT backend)
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], videos=video_inputs_cached,
                       padding=True, return_tensors="pt")
    if not is_pure_vl and vid_tensor is not None:
        inputs.update({"videos_input": vid_tensor})
    inputs = inputs.to(device)

    if device == "cpu":
        generated_ids = cpu_friendly_generate(
            model, inputs, max_new_tokens, temperature, top_p)
    else:
        with torch.no_grad():
            do_sample = temperature > 0.05
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                use_cache=True,
            )

    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], generated_ids)]
    result = processor.batch_decode(trimmed, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    text_out = result[0] if result else ""

    if device == "cpu":
        del inputs, generated_ids, trimmed
        free_memory("cpu")

    return text_out


def _build_msg(q_text: str, frames_pil, fps: float,
               video_path: str, nframes: int) -> list:
    """Build messages based on whether frame list is available."""
    if frames_pil is not None:
        return [{"role": "user", "content": [
            {"type": "video", "video": frames_pil, "sample_fps": fps},
            {"type": "text", "text": q_text},
        ]}]
    return [{"role": "user", "content": [
        {"type": "video", "video": video_path, "nframes": nframes},
        {"type": "text", "text": q_text},
    ]}]


# =============================================================================
# Main inference loop (supports CoT / dual-branch / uncertainty)
# =============================================================================

def run_questions(
    model, processor,
    questions: List[Dict],
    video_inputs_cached, vid_tensor,
    frames_pil, fps: float,
    device: str,
    max_new_tokens: int,
    max_desc_tokens: int,
    video_path: str,
    nframes: int,
    is_pure_vl: bool = False,
    use_cot: bool = False,
    dual_branch: bool = False,
    uncertainty_samples: int = 0,
) -> List[Dict]:
    """
    Run all questions on a set of video_inputs sequentially, return results list.
    Supports CoT prompting, dual-branch aggregation, Bayesian uncertainty quantification.
    """
    use_frames = frames_pil is not None
    results = []

    for idx, q in enumerate(questions):
        # CoT prompt replacement
        if use_cot:
            base = q["task"].replace(" [CoT]", "")
            q = _COT_MAP.get(base, q)

        task = q["task"]
        is_desc = "layout" in task.lower()
        tokens_lim = max(max_new_tokens, max_desc_tokens) if is_desc else max_new_tokens

        print(f"\n--- [{idx+1}/{len(questions)}] {task} ---", flush=True)
        t0 = time.time()

        if dual_branch:
            # ── Dual-branch: semantic + geometric inference, aggregate answers ──────────
            print("  [dual-branch] semantic...", flush=True)
            msg_sem = _build_msg(_PREFIX_SEMANTIC + q["question"],
                                 frames_pil, fps, video_path, nframes)
            raw_sem = _single_generate(model, processor, msg_sem,
                                       video_inputs_cached, vid_tensor,
                                       device, tokens_lim, is_pure_vl=is_pure_vl)

            print("  [dual-branch] geometric...", flush=True)
            msg_geo = _build_msg(_PREFIX_GEOMETRIC + q["question"],
                                 frames_pil, fps, video_path, nframes)
            raw_geo = _single_generate(model, processor, msg_geo,
                                       video_inputs_cached, vid_tensor,
                                       device, tokens_lim, is_pure_vl=is_pure_vl)

            _, ans_sem = extract_thinking_and_answer(raw_sem)
            _, ans_geo = extract_thinking_and_answer(raw_geo)
            thinking = ""

            if is_desc:
                answer = ans_sem if len(ans_sem) >= len(ans_geo) else ans_geo
            else:
                v1, v2 = _try_num(ans_sem), _try_num(ans_geo)
                if v1 is not None and v2 is not None:
                    avg = (v1 + v2) / 2
                    answer = str(int(round(avg))) if float(int(round(avg))) == avg else f"{avg:.2f}"
                else:
                    answer = ans_sem or ans_geo
            raw_out = f"[semantic]\n{raw_sem}\n\n[geometric]\n{raw_geo}"
            print(f"  sem={ans_sem!r} geo={ans_geo!r} => {answer!r}", flush=True)

        else:
            # ── Standard single-path inference ──────────────────────────────
            print("  generating...", flush=True)
            msg = _build_msg(q["question"], frames_pil, fps, video_path, nframes)
            raw_out = _single_generate(model, processor, msg,
                                       video_inputs_cached, vid_tensor,
                                       device, tokens_lim, is_pure_vl=is_pure_vl)
            thinking, answer = extract_thinking_and_answer(raw_out)

        elapsed = time.time() - t0

        # ── Uncertainty quantification: append N high-temp samples ──────
        uncertainty_info: Dict = {}
        if uncertainty_samples > 0 and not is_desc:
            print(f"  [uncertainty] {uncertainty_samples} extra samples (T=0.7)...", flush=True)
            extra = [answer]
            for _ in range(uncertainty_samples):
                msg_u = _build_msg(q["question"], frames_pil, fps, video_path, nframes)
                raw_u = _single_generate(model, processor, msg_u,
                                         video_inputs_cached, vid_tensor,
                                         device, tokens_lim,
                                         temperature=0.7, top_p=0.9,
                                         is_pure_vl=is_pure_vl)
                _, ans_u = extract_thinking_and_answer(raw_u)
                extra.append(ans_u)
            from postprocessing import compute_uncertainty, format_uncertainty_report
            unc = compute_uncertainty(extra, task)
            uncertainty_info = unc
            print("  " + format_uncertainty_report(unc).replace("\n", "\n  "), flush=True)
            if "mean" in unc:
                m = unc["mean"]
                answer = str(int(round(m))) if float(int(round(m))) == m else f"{m:.2f}"

        # ── Print results ────────────────────────────────────────────────────
        if thinking and not dual_branch:
            print(f"  thinking: {thinking[:120]}{'...' if len(thinking)>120 else ''}")
        print(f"  answer: {answer!r}")
        if not dual_branch:
            print(f"  raw: {raw_out[:200]}{'...' if len(raw_out)>200 else ''}")
        print(f"  time: {elapsed:.2f}s")

        results.append({
            "task": task,
            "question": q["question"],
            "thinking": thinking if not dual_branch else "",
            "answer": answer,
            "raw_output": raw_out,
            "time_sec": elapsed,
            "tokens": len(raw_out),
            "uncertainty": uncertainty_info,
        })

    return results


# =============================================================================
# Main inference function
# =============================================================================

def run_inference(
    video_path: str,
    # ── Basic settings ─────────────────────────────────────
    device: str = None,
    output_file: str = "cosmes_report.md",
    nframes: int = 8,
    max_new_tokens: int = 64,
    max_desc_tokens: int = 256,
    model_dir: str = None,
    questions: List[Dict] = None,
    # ── Frame sampling strategies ──────────────────────────
    use_space_aware: bool = False,
    depth_aware: bool = False,
    cluster_sample: bool = False,
    # ── Backend selection ───────────────────────────────────
    no_vggt: bool = False,
    # ── Multi-view ensemble ────────────────────────────────
    multi_view: int = 1,
    # ── Preprocessing ───────────────────────────────────────
    edge_enhance: bool = False,
    edge_mode: str = "edge",
    edge_alpha: float = 0.35,
    # ── Inference innovations ───────────────────────────────
    spatial_cot: bool = False,
    dual_branch: bool = False,
    uncertainty_samples: int = 0,
    # ── Post-processing ─────────────────────────────────────
    geo_correct: bool = False,
    hallucination_check: bool = False,
    cross_validate: bool = False,
    # ── LoRA ─────────────────────────────────────────────────
    lora_path: str = None,
) -> Tuple[List[Dict], str]:

    if questions is None:
        questions = SPATIAL_QUESTIONS

    # ── Device ───────────────────────────────────────────────────────────────
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] GPU not found, using CPU")
        device = "cpu"

    # ── Model path ───────────────────────────────────────────────────────────
    local_path = get_local_model_path("Diankun/Spatial-MLLM-subset-sft",
                                      custom_dir=model_dir)

    # ── Status tags ──────────────────────────────────────────────────────────
    sample_tag = ("depth-aware" if depth_aware else
                  "cluster" if cluster_sample else
                  "space-aware" if use_space_aware else "uniform")
    backend_tag = ("[no-VGGT]" if no_vggt else "[VGGT]")
    active = [k for k, v in {
        "depth-aware": depth_aware, "edge-enhance": edge_enhance,
        "CoT": spatial_cot, "dual-branch": dual_branch,
        f"uncertainty×{uncertainty_samples}": uncertainty_samples > 0,
        "geo-correct": geo_correct, "hallucination": hallucination_check,
        "cross-validate": cross_validate, "LoRA": bool(lora_path),
        f"multi-view×{multi_view}": multi_view > 1,
        "cluster": cluster_sample,
    }.items() if v]

    print("=" * 70)
    print(f"COSMES  {backend_tag}  sampling={sample_tag}  device={device}")
    if active:
        print(f"  active: {' | '.join(active)}")
    print(f"  video={video_path}")
    print(f"  model={local_path}")
    print("=" * 70)

    # ── 1. Load model ────────────────────────────────────────────────────────
    t_load = time.time()
    print("\n[1/4] Loading model...", flush=True)
    from qwen_vl_utils import process_vision_info

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    is_pure_vl = no_vggt  # no videos_input needed

    if no_vggt:
        from src.models import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            local_path, torch_dtype=dtype, attn_implementation="eager",
            local_files_only=True)
        processor = Qwen2_5_VLProcessor.from_pretrained(local_path, local_files_only=True)
        print(f"  no-VGGT pure-VL loaded", flush=True)
    else:
        from src.models import Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor
        model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
            local_path, torch_dtype=dtype, attn_implementation="eager",
            local_files_only=True)
        processor = Qwen2_5_VLProcessor.from_pretrained(local_path, local_files_only=True)
        print(f"  Spatial-MLLM+VGGT loaded", flush=True)

    model = model.to(device)
    if device == "cpu":
        model = apply_cpu_memory_optimizations(model, device)
    else:
        model.eval()

    model = load_lora(model, lora_path)

    load_time = time.time() - t_load
    print(f"  done ({load_time:.1f}s)", flush=True)
    free_memory(device)

    # ── 2. Frame sampling helper ─────────────────────────────────────────────
    def _get_frames(seed: int = 0):
        """Execute frame selection (with preprocessing), return (frames_pil, fps) or (None, None)."""
        fpil, fps = None, None
        if depth_aware:
            from space_aware_frames import get_depth_aware_frames_for_inference
            fpil, fidx, fps = get_depth_aware_frames_for_inference(video_path, nframes)
            print(f"  [depth-aware] indices={fidx[:8]}{'...' if len(fidx)>8 else ''}", flush=True)
        elif cluster_sample:
            from space_aware_frames import get_cluster_aware_frames_for_inference
            fpil, fidx, fps = get_cluster_aware_frames_for_inference(video_path, nframes)
            print(f"  [cluster] indices={fidx[:8]}{'...' if len(fidx)>8 else ''}", flush=True)
        elif use_space_aware:
            from space_aware_frames import get_space_aware_frames_for_inference
            fpil, fidx, fps = get_space_aware_frames_for_inference(video_path, nframes)
            print(f"  [space-aware] indices={fidx[:8]}{'...' if len(fidx)>8 else ''}", flush=True)
        elif multi_view > 1:
            from space_aware_frames import get_perturbed_frames_for_inference
            fpil, fidx, fps = get_perturbed_frames_for_inference(
                video_path, nframes, seed=seed, use_cluster=cluster_sample)
            print(f"  [multi-view seed={seed}] indices={fidx[:8]}", flush=True)

        if fpil and edge_enhance:
            from preprocessing import apply_preprocessing
            fpil = apply_preprocessing(fpil, mode=edge_mode, edge_alpha=edge_alpha)
            print(f"  [edge-enhance] mode={edge_mode} alpha={edge_alpha}", flush=True)
        return fpil, fps

    # ── 3. Video processing + inference ──────────────────────────────────────
    t_video = time.time()
    print(f"\n[2/4] Video processing (strategy={sample_tag})...", flush=True)

    if multi_view > 1:
        # ── Multi-view ensemble ────────────────────────────────────────────
        print(f"  multi-view: {multi_view} views", flush=True)
        all_results: List[List[Dict]] = []
        for v in range(multi_view):
            print(f"\n  === View {v+1}/{multi_view} ===", flush=True)
            fpil_v, fps_v = _get_frames(seed=v)
            msg_v = _build_msg(questions[0]["question"], fpil_v, fps_v, video_path, nframes)
            _, vid_inputs_v = process_vision_info(msg_v)
            vid_tensor_v = ((torch.stack(vid_inputs_v) / 255.0).to(device)
                            if not is_pure_vl else None)
            view_res = run_questions(
                model, processor, questions, vid_inputs_v, vid_tensor_v,
                fpil_v, fps_v, device, max_new_tokens, max_desc_tokens,
                video_path, nframes, is_pure_vl=is_pure_vl,
                use_cot=spatial_cot, dual_branch=dual_branch,
                uncertainty_samples=0,
            )
            all_results.append(view_res)

        video_time = time.time() - t_video
        results = aggregate_multi_view(all_results)
        total_infer = sum(r["time_sec"] for r in results)
        print("\n  [multi-view] aggregated:", flush=True)
        for r in results:
            print(f"    {r['task']}: {r['answer']}  views={r.get('multi_view_answers', [])}")

    else:
        # ── Single-path inference ────────────────────────────────────────────
        fpil, fps = _get_frames(seed=0)
        msg_vid = _build_msg(questions[0]["question"], fpil, fps, video_path, nframes)
        _, video_inputs_cached = process_vision_info(msg_vid)

        # Edge enhancement (uniform sampling path)
        if edge_enhance and fpil is None:
            from preprocessing import apply_preprocessing
            import numpy as np
            from PIL import Image as _PILImage
            raw_frames = [_PILImage.fromarray(
                (t.numpy().transpose(1, 2, 0) if t.shape[0] == 3
                 else t.numpy()).clip(0, 255).astype("uint8"), "RGB"
            ) for t in video_inputs_cached]
            fpil = apply_preprocessing(raw_frames, mode=edge_mode, edge_alpha=edge_alpha)
            msg_enh = _build_msg(questions[0]["question"], fpil, 1.0, video_path, nframes)
            _, video_inputs_cached = process_vision_info(msg_enh)
            fps = 1.0
            print(f"  [edge-enhance] uniform frames enhanced mode={edge_mode}", flush=True)

        video_time = time.time() - t_video
        print(f"  done ({video_time:.1f}s)", flush=True)

        vid_tensor = ((torch.stack(video_inputs_cached) / 255.0).to(device)
                      if not is_pure_vl else None)

        print("\n[3/4] Running inference...", flush=True)
        results = run_questions(
            model, processor, questions, video_inputs_cached, vid_tensor,
            fpil, fps, device, max_new_tokens, max_desc_tokens,
            video_path, nframes, is_pure_vl=is_pure_vl,
            use_cot=spatial_cot, dual_branch=dual_branch,
            uncertainty_samples=uncertainty_samples,
        )
        total_infer = sum(r["time_sec"] for r in results)

    # ── 4. Post-processing ─────────────────────────────────────────────────────
    consistency_report = ""
    rel_md = ""
    if geo_correct or hallucination_check or cross_validate:
        print("\n[4/4] Post-processing...", flush=True)
        from postprocessing import full_postprocess, build_reliability_summary_block
        results, consistency_report = full_postprocess(
            results,
            do_geo_correct=geo_correct,
            do_hallucination=hallucination_check,
            do_cross_validate=cross_validate,
        )
        for r in results:
            if r.get("correction_note"):
                print(f"  [geo-correct] {r['task']}: {r['correction_note']}")
            if r.get("hallucination_flags"):
                print(f"  [hallucination {r.get('hallucination_risk','')}] "
                      f"{r['task']}: {'; '.join(r['hallucination_flags'])}")
        if consistency_report:
            print(consistency_report)
        rel_md, rel_one_line = build_reliability_summary_block(
            results,
            consistency_report or "",
            do_geo=geo_correct,
            do_hall=hallucination_check,
            do_cross=cross_validate,
        )
        print(rel_one_line, flush=True)

    # ── Summary ────────────────────────────────────────────────────────────
    total_time = load_time + video_time + total_infer
    print("\n" + "=" * 70)
    print(f"  load={load_time:.1f}s  video={video_time:.1f}s  "
          f"infer={total_infer:.2f}s  total={total_time:.2f}s")
    if device == "cuda":
        print(f"  GPU peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print("=" * 70)

    # ── Generate report ─────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# COSMES Spatial Reasoning Report",
        f"Generated: {ts}",
        f"Device: {device} | Frames: {nframes} | Backend: {backend_tag} | Sampling: {sample_tag}",
        f"Innovations: {', '.join(active) if active else 'default'}",
        f"Video: {video_path}",
        f"Time: load={load_time:.1f}s video={video_time:.1f}s "
        f"infer={total_infer:.2f}s total={total_time:.2f}s",
        "", "## Results", "",
    ]
    for r in results:
        lines += [
            f"### {r['task']}",
            f"- **Q**: {r['question'][:120]}",
        ]
        if r.get("thinking"):
            lines.append(f"- **Thinking**: {r['thinking'][:300]}")
        lines.append(f"- **Answer**: {r['answer']}")
        if r.get("multi_view_answers"):
            lines.append(f"- **Per-view**: {r['multi_view_answers']}")
        if r.get("correction_note"):
            lines.append(f"- **Correction**: {r['correction_note']}")
        if r.get("hallucination_flags"):
            lines.append(f"- **Hallucination [{r.get('hallucination_risk','')}]**: "
                         + "; ".join(r["hallucination_flags"]))
        if r.get("uncertainty"):
            u = r["uncertainty"]
            lines.append(f"- **Uncertainty**: confidence={u.get('confidence_level','?')} "
                         f"mean={u.get('mean','?')} std={u.get('std','?')}")
        lines.append(f"- **Time**: {r['time_sec']:.2f}s")
        raw = (r.get("raw_output") or "").strip()
        if raw:
            preview = raw[:600] + ("..." if len(raw) > 600 else "")
            lines += ["- **Raw output**:", "```", preview, "```"]
        lines.append("")

    if consistency_report:
        lines += ["", "## Cross-task Consistency", "", consistency_report, ""]

    if rel_md:
        lines.append(rel_md)

    report = "\n".join(lines)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved: {output_file}")

    return results, report


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="COSMES – Core Spatial Intelligence & Reasoning",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Basic settings
    p.add_argument("--video", default="assets/arkitscenes_41069025.mp4")
    p.add_argument("--frames", type=int, default=8, help="number of frames (default 8)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-desc-tokens", type=int, default=256)
    p.add_argument("--output", default="cosmes_report.md")
    p.add_argument("--quick", action="store_true",
                   help="quick mode: 4 frames, 32 tokens, 1 question")
    p.add_argument("--model-dir", default=None, help="local model directory")
    p.add_argument("--mode", choices=["cpu", "gpu"], default=None)
    p.add_argument("--cpu", action="store_true", help="same as --mode cpu")

    # Frame sampling strategies
    p.add_argument("--space-aware", action="store_true")
    p.add_argument("--depth-aware", action="store_true",
                   help="[Innovation] depth proxy + viewpoint change driven sampling")
    p.add_argument("--cluster-sample", action="store_true",
                   help="histogram-based greedy farthest-point sampling")

    # Multi-view
    p.add_argument("--multi-view", type=int, default=1, metavar="N",
                   help="[Innovation] N-view ensemble inference (N>1)")

    # Backend
    p.add_argument("--no-vggt", action="store_true",
                   help="[Innovation] pure Qwen2.5-VL without VGGT (ablation baseline)")

    # LoRA
    p.add_argument("--lora-path", default=None,
                   help="[Innovation] PEFT LoRA adapter path")

    # Preprocessing
    p.add_argument("--edge-enhance", action="store_true",
                   help="[Innovation] edge/contrast preprocessing")
    p.add_argument("--edge-mode", choices=["edge", "clahe", "dual_geo", "none"],
                   default="edge")
    p.add_argument("--edge-alpha", type=float, default=0.35,
                   help="Laplacian blend alpha [0,1] (default 0.35)")

    # Inference paradigms
    p.add_argument("--spatial-cot", action="store_true",
                   help="[Innovation] spatial chain-of-thought prompts")
    p.add_argument("--dual-branch", action="store_true",
                   help="[Innovation] semantic+geometric dual-branch aggregation")

    # Uncertainty
    p.add_argument("--uncertainty", type=int, default=0, metavar="N",
                   help="[Innovation] N extra temperature samples for uncertainty (default 0)")

    # Post-processing
    p.add_argument("--geo-correct", action="store_true",
                   help="[Innovation] geometric self-correction with spatial priors")
    p.add_argument("--hallucination-check", action="store_true",
                   help="[Innovation] hallucination detection and flagging")
    p.add_argument("--cross-validate", action="store_true",
                   help="[Innovation] cross-task answer consistency check")

    args = p.parse_args()

    if not os.path.exists(args.video):
        print(f"[error] video not found: {args.video}")
        sys.exit(1)

    nframes = 4 if args.quick else args.frames
    max_tokens = 32 if args.quick else args.max_tokens
    questions = SPATIAL_QUESTIONS[:1] if args.quick else SPATIAL_QUESTIONS

    if args.cpu or args.mode == "cpu":
        device = "cpu"
    elif args.mode == "gpu":
        device = "cuda"
    else:
        device = None  # auto

    # CPU conservative parameters (memory saving)
    is_cpu = (device == "cpu") or (device is None and not torch.cuda.is_available())
    if is_cpu:
        if not args.quick and nframes > 4:
            print(f"[CPU] frames {nframes}->4 (memory limit, override with --frames)")
            nframes = 4
        if max_tokens > 128:
            print(f"[CPU] max-tokens {max_tokens}->128")
            max_tokens = 128
        if args.max_desc_tokens > 256:
            print(f"[CPU] max-desc-tokens {args.max_desc_tokens}->256")
            args.max_desc_tokens = 256
        if args.multi_view > 2:
            print(f"[CPU] multi-view {args.multi_view}->2")
            args.multi_view = 2
        if args.uncertainty > 2:
            print(f"[CPU] uncertainty {args.uncertainty}->2")
            args.uncertainty = 2

    if args.quick:
        print("[quick] 4 frames | 32 tokens | 1 question")

    run_inference(
        video_path=args.video,
        device=device,
        output_file=args.output,
        nframes=nframes,
        max_new_tokens=max_tokens,
        max_desc_tokens=args.max_desc_tokens,
        model_dir=args.model_dir,
        questions=questions,
        use_space_aware=args.space_aware,
        depth_aware=args.depth_aware,
        cluster_sample=args.cluster_sample,
        no_vggt=args.no_vggt,
        multi_view=args.multi_view,
        edge_enhance=args.edge_enhance,
        edge_mode=args.edge_mode,
        edge_alpha=args.edge_alpha,
        spatial_cot=args.spatial_cot,
        dual_branch=args.dual_branch,
        uncertainty_samples=args.uncertainty,
        geo_correct=args.geo_correct,
        hallucination_check=args.hallucination_check,
        cross_validate=args.cross_validate,
        lora_path=args.lora_path,
    )
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        import traceback
        print(f"\n[error] {e}")
        traceback.print_exc()
        sys.exit(1)
