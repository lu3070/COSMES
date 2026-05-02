#!/usr/bin/env python3
"""
Space-aware frame selection strategy from Spatial-MLLM paper.
Select keyframes based on scene changes/motion, ensuring uniform temporal distribution
while prioritizing frames with significant spatial changes. This provides better
visual coverage for spatial reasoning tasks (counting, layout, distance estimation).
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image


FRAME_FACTOR = 2  # Must match qwen_vl_utils, frames must be multiples of this factor


def _round_nframes(n: int) -> int:
    """Ensure nframes is a multiple of FRAME_FACTOR (required by qwen_vl_utils)."""
    return max(FRAME_FACTOR, (int(n) // FRAME_FACTOR) * FRAME_FACTOR)


def _load_video_metadata(video_path: str) -> Tuple[int, float]:
    """Return (total_frames, fps). Prefer decord, fallback to torchvision."""
    try:
        from decord import VideoReader
        vr = VideoReader(video_path)
        return len(vr), float(vr.get_avg_fps())
    except Exception:
        pass
    try:
        from torchvision.io import read_video
        from torchvision.io import VideoReader as TVVideoReader
        # torchvision read_video requires start_pts/end_pts
        video, _, info = read_video(video_path, pts_unit="sec", output_format="TCHW")
        total = video.size(0)
        fps = info.get("video_fps", 30.0)
        return total, float(fps)
    except Exception:
        raise RuntimeError(f"Cannot read video metadata: {video_path}")


def _read_frames_for_scores(video_path: str, max_frames: int = 128) -> np.ndarray:
    """
    Read frames for computing change scores (downsampled to save memory), shape (N, H, W) grayscale.
    """
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            raise ValueError("Video has no frames")
        step = max(1, total // max_frames)
        indices = list(range(0, total, step))[:max_frames]
        frames = vr.get_batch(indices).asnumpy()
        # (N, H, W, C) -> grayscale
        gray = np.dot(frames[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        return gray, indices, total
    except Exception:
        pass
    from torchvision.io import read_video
    video, _, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")
    total = video.size(0)
    if total <= 0:
        raise ValueError("Video has no frames")
    step = max(1, total // max_frames)
    indices = list(range(0, total, step))[:max_frames]
    frames = video[indices].numpy()
    gray = np.dot(frames.transpose(0, 2, 3, 1)[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    return gray, indices, total


def _frame_change_scores(gray: np.ndarray) -> np.ndarray:
    """
    Compute frame change scores relative to previous frame (L1 difference), first frame is 0.
    Returns float array of shape (len(gray),).
    """
    if len(gray) <= 1:
        return np.zeros(len(gray), dtype=np.float32)
    diff = np.abs(gray[1:].astype(np.float32) - gray[:-1].astype(np.float32))
    scores = np.mean(diff, axis=(1, 2))
    return np.concatenate([[0.0], scores])


def space_aware_frame_indices(
    video_path: str,
    nframes: int,
    total_frames: Optional[int] = None,
    video_fps: Optional[float] = None,
) -> List[int]:
    """
    Space-aware frame selection: prioritize frames with significant scene changes while ensuring temporal distribution.

    Strategy:
    1. If total_frames <= nframes, return [0,1,...,total_frames-1] (pad with last frame index if needed).
    2. Otherwise, uniformly sample ~2*nframes candidate frames across video, compute frame difference scores.
    3. Greedy selection among candidates: balance temporal uniformity and change magnitude, return nframes sorted indices.

    Returns:
        List of frame indices (integers, sorted), length nframes.
    """
    nframes = _round_nframes(nframes)
    if total_frames is None or video_fps is None:
        total_frames, video_fps = _load_video_metadata(video_path)
    total_frames = max(1, total_frames)

    if total_frames <= nframes:
        indices = list(range(total_frames))
        while len(indices) < nframes:
            indices.append(total_frames - 1)
        return indices[:nframes]

    try:
        gray, cand_indices, total = _read_frames_for_scores(video_path, max_frames=min(256, total_frames))
        change = _frame_change_scores(gray)
        # Map change scores back to virtual indices of full video: cand_indices[i] corresponds to change[i]
        # Select nframes indices in [0, total_frames-1]
        n_cand = len(cand_indices)
        if n_cand <= nframes:
            return _round_uniform_indices(0, total_frames - 1, nframes)

        # Normalize "time position" of each candidate to [0,1], and normalize change scores
        t_norm = np.linspace(0, 1, n_cand)
        c_norm = change.astype(np.float64)
        if c_norm.max() > c_norm.min():
            c_norm = (c_norm - c_norm.min()) / (c_norm.max() - c_norm.min())
        # Combined score: temporal uniformity + larger changes preferred
        score = 0.5 * t_norm + 0.5 * c_norm

        # Place nframes "target time points" on uniform grid across video, then select nearest candidate for each (slightly biased toward larger changes by combined score)
        target_ts = np.linspace(0, total_frames - 1, nframes)
        out = []
        used = set()
        for t in target_ts:
            best_i = None
            best_val = -1e9
            for i in range(n_cand):
                if i in used:
                    continue
                idx = cand_indices[i]
                dist = abs(idx - t)
                s = score[i] - 0.001 * dist
                if s > best_val:
                    best_val = s
                    best_i = i
            if best_i is not None:
                used.add(best_i)
                out.append(cand_indices[best_i])
        out = sorted(set(out))
        if len(out) < nframes:
            uniform = _round_uniform_indices(0, total_frames - 1, nframes)
            for u in uniform:
                if u not in out:
                    out.append(u)
                    if len(out) >= nframes:
                        break
            out = sorted(out)[:nframes]
        elif len(out) > nframes:
            out = _round_uniform_from_list(out, nframes)
        return out
    except Exception:
        return _round_uniform_indices(0, total_frames - 1, nframes)


def _round_uniform_indices(start: int, end: int, n: int) -> List[int]:
    """Uniformly sample n integer indices in [start, end]."""
    if end <= start or n <= 0:
        return [start] * n if n > 0 else []
    idx = np.linspace(start, end, n).round().astype(int).tolist()
    return idx


def _round_uniform_from_list(lst: List[int], n: int) -> List[int]:
    """Equidistantly sample n elements from sorted list."""
    if len(lst) <= n:
        return lst[:n]
    idx = np.linspace(0, len(lst) - 1, n).round().astype(int)
    return [lst[i] for i in idx]


def extract_frames_as_pil(
    video_path: str,
    indices: List[int],
) -> List[Image.Image]:
    """Extract frames from video by indices, return list of PIL.Image (RGB)."""
    if not indices:
        return []
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        out = []
        for i in indices:
            i = max(0, min(i, len(vr) - 1))
            frame = vr[i].asnumpy()
            if frame.ndim == 3 and frame.shape[2] >= 3:
                img = Image.fromarray(frame[:, :, :3].astype(np.uint8))
            else:
                img = Image.fromarray(frame.astype(np.uint8))
            if img.mode != "RGB":
                img = img.convert("RGB")
            out.append(img)
        return out
    except Exception:
        pass
    from torchvision.io import read_video
    video, _, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")
    total = video.size(0)
    out = []
    for i in indices:
        i = max(0, min(i, total - 1))
        frame = video[i].numpy()
        if frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[2] >= 3:
            frame = frame[:, :, :3]
        img = Image.fromarray(frame)
        if img.mode != "RGB":
            img = img.convert("RGB")
        out.append(img)
    return out


def get_space_aware_frames_for_inference(
    video_path: str,
    nframes: int,
) -> Tuple[List[Image.Image], List[int], float]:
    """
    Called by run_spatial_inference: return space-aware selected frames (PIL list), frame indices, and sample_fps.

    Returns:
        (frames_pil, frame_indices, sample_fps)
    """
    total_frames, video_fps = _load_video_metadata(video_path)
    nframes = _round_nframes(nframes)
    indices = space_aware_frame_indices(
        video_path,
        nframes,
        total_frames=total_frames,
        video_fps=video_fps,
    )
    frames_pil = extract_frames_as_pil(video_path, indices)
    duration_sec = total_frames / max(video_fps, 1e-6)
    sample_fps = len(frames_pil) / max(duration_sec, 1e-6)
    return frames_pil, indices, sample_fps


# ─────────────────────────────────────────────────────────────────────────────
# Innovation III: Histogram-Based Cluster Sampling
# Approach: Represent each frame by color histogram, use greedy "farthest point" strategy to select nframes with maximum visual diversity.
# Compared to space-aware L1 difference, histogram features are insensitive to local noise/motion blur, focusing more on scene content diversity.
# ─────────────────────────────────────────────────────────────────────────────


def compute_histogram_features(gray_frames: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """
    Compute grayscale histogram feature vector for each frame (normalized), return shape (N, n_bins).

        Args:
            gray_frames: Grayscale frame array, shape (N, H, W), range [0, 255].
            n_bins: Number of histogram bins, default 32.

        Returns:
            feats: Normalized histogram features, shape (N, n_bins).
    """
    n = len(gray_frames)
    feats = np.zeros((n, n_bins), dtype=np.float32)
    for i in range(n):
        hist, _ = np.histogram(gray_frames[i].ravel(), bins=n_bins, range=(0, 256))
        s = hist.sum()
        feats[i] = hist.astype(np.float32) / max(s, 1)
    return feats


def greedy_farthest_cluster_indices(
    feats: np.ndarray,
    n_select: int,
    candidate_indices: List[int],
) -> List[int]:
    """
    Greedy "farthest point" strategy: iteratively select frame indices farthest from selected set in feature space.
    Ensures selected frames have maximum visual diversity.

    Args:
        feats: Candidate frame features, shape (n_cand, D).
        n_select: Number of frames to select.
        candidate_indices: List of true indices in original video for candidates.

    Returns:
        Sorted list of selected frame indices (length n_select).
    """
    n_cand = len(feats)
    if n_cand <= n_select:
        return sorted(candidate_indices[:n_cand])

    selected_local = []
    remaining = list(range(n_cand))

    # First frame: select middle time frame, prefer scene middle over static start frame
    mid = n_cand // 2
    selected_local.append(mid)
    remaining.remove(mid)

    # Accumulated minimum distance array: dist_to_selected[i] = min distance from point i to selected set
    dist_to_selected = np.full(n_cand, np.inf, dtype=np.float64)
    for i in remaining:
        d = float(np.sum((feats[i] - feats[mid]) ** 2))
        dist_to_selected[i] = d

    while len(selected_local) < n_select and remaining:
        best_i = max(remaining, key=lambda i: dist_to_selected[i])
        selected_local.append(best_i)
        remaining.remove(best_i)
        for i in remaining:
            d = float(np.sum((feats[i] - feats[best_i]) ** 2))
            if d < dist_to_selected[i]:
                dist_to_selected[i] = d

    return sorted([candidate_indices[i] for i in selected_local])


def histogram_cluster_frame_indices(
    video_path: str,
    nframes: int,
    total_frames: Optional[int] = None,
    video_fps: Optional[float] = None,
    max_cand: int = 128,
    n_bins: int = 32,
) -> List[int]:
    """
    Histogram-based cluster sampling: select nframes with most diverse color/brightness distributions.

    Returns:
        List of frame indices (sorted), length nframes.
    """
    nframes = _round_nframes(nframes)
    if total_frames is None or video_fps is None:
        total_frames, video_fps = _load_video_metadata(video_path)
    total_frames = max(1, total_frames)

    if total_frames <= nframes:
        indices = list(range(total_frames))
        while len(indices) < nframes:
            indices.append(total_frames - 1)
        return indices[:nframes]

    try:
        gray, cand_indices, total = _read_frames_for_scores(video_path, max_frames=min(max_cand, total_frames))
        feats = compute_histogram_features(gray, n_bins=n_bins)
        selected = greedy_farthest_cluster_indices(feats, nframes, cand_indices)
        if len(selected) < nframes:
            uniform = _round_uniform_indices(0, total_frames - 1, nframes)
            for u in uniform:
                if u not in selected:
                    selected.append(u)
                if len(selected) >= nframes:
                    break
            selected = sorted(selected)[:nframes]
        return selected
    except Exception:
        return _round_uniform_indices(0, total_frames - 1, nframes)


def get_cluster_aware_frames_for_inference(
    video_path: str,
    nframes: int,
) -> Tuple[List[Image.Image], List[int], float]:
    """
    Called by run_spatial_inference (Innovation III): histogram cluster sampling version.
    Returns (frames_pil, frame_indices, sample_fps).
    """
    total_frames, video_fps = _load_video_metadata(video_path)
    nframes = _round_nframes(nframes)
    indices = histogram_cluster_frame_indices(
        video_path,
        nframes,
        total_frames=total_frames,
        video_fps=video_fps,
    )
    frames_pil = extract_frames_as_pil(video_path, indices)
    duration_sec = total_frames / max(video_fps, 1e-6)
    sample_fps = len(frames_pil) / max(duration_sec, 1e-6)
    return frames_pil, indices, sample_fps


def get_perturbed_frames_for_inference(
    video_path: str,
    nframes: int,
    seed: int = 0,
    use_cluster: bool = False,
) -> Tuple[List[Image.Image], List[int], float]:
    """
    For multi-view ensemble inference (Innovation I): generate perturbed keyframe subsets with different seeds for viewpoint diversity.

    Args:
        video_path: Video path.
        nframes: Target number of frames.
        seed: Random seed, different seeds produce different frame subsets.
        use_cluster: If True, use histogram clustering base (Innovation III); else use space-aware base.

    Returns:
        (frames_pil, frame_indices, sample_fps)
    """
    total_frames, video_fps = _load_video_metadata(video_path)
    nframes_rounded = _round_nframes(nframes)

    if use_cluster:
        indices = histogram_cluster_frame_indices(video_path, nframes_rounded, total_frames, video_fps)
    else:
        indices = space_aware_frame_indices(video_path, nframes_rounded, total_frames, video_fps)

    # Apply small perturbations (±offset) to selected indices using seed, introducing viewpoint diversity while preserving temporal order
    rng = np.random.RandomState(seed)
    offset_range = max(1, total_frames // (nframes_rounded * 4))
    perturbed = []
    for idx in indices:
        delta = int(rng.randint(-offset_range, offset_range + 1))
        new_idx = int(np.clip(idx + delta, 0, total_frames - 1))
        perturbed.append(new_idx)
    perturbed = sorted(set(perturbed))
    if len(perturbed) < nframes_rounded:
        uniform = _round_uniform_indices(0, total_frames - 1, nframes_rounded)
        for u in uniform:
            if u not in perturbed:
                perturbed.append(u)
            if len(perturbed) >= nframes_rounded:
                break
        perturbed = sorted(perturbed)[:nframes_rounded]

    frames_pil = extract_frames_as_pil(video_path, perturbed)
    duration_sec = total_frames / max(video_fps, 1e-6)
    sample_fps = len(frames_pil) / max(duration_sec, 1e-6)
    return frames_pil, perturbed, sample_fps


# ─────────────────────────────────────────────────────────────────────────────
# Frame Sampling Innovation: Depth + Viewpoint Jointly Driven Spatial-Aware Frame Selection
# Upgrade from pixel difference to Laplacian variance (geometric depth proxy) + inter-frame perspective difference (viewpoint change proxy)
# ─────────────────────────────────────────────────────────────────────────────


def compute_depth_proxy_scores(gray_frames: np.ndarray) -> np.ndarray:
    """
    Depth geometry proxy score: compute Laplacian variance for each frame. Higher values indicate richer edges/textures,
    implying more complex scene geometry (depth, contours). More focused on spatial geometry than pure L1 difference.

    Args:
        gray_frames: (N, H, W) grayscale frames, range [0, 255].

    Returns:
        scores: (N,) float, Laplacian variance for each frame.
    """
    scores = np.zeros(len(gray_frames), dtype=np.float64)
    for i, frame in enumerate(gray_frames):
        f = frame.astype(np.float32)
        # 3x3 Laplacian kernel
        lap = (
            -f[:-2, :-2] - f[:-2, 1:-1] - f[:-2, 2:]
            - f[1:-1, :-2] + 8 * f[1:-1, 1:-1] - f[1:-1, 2:]
            - f[2:, :-2] - f[2:, 1:-1] - f[2:, 2:]
        )
        scores[i] = float(np.var(lap))
    return scores


def compute_viewpoint_change_scores(gray_frames: np.ndarray) -> np.ndarray:
    """
    Viewpoint change proxy score: compute complement of normalized cross-correlation between adjacent frames.
    Higher values indicate larger viewpoint changes. Unlike L1 difference, cross-correlation is insensitive to
    global brightness shifts and better captures viewpoint/motion changes.

    Args:
        gray_frames: (N, H, W) grayscale frames.

    Returns:
        scores: (N,) float, first frame is 0, others are viewpoint change scores vs previous frame.
    """
    n = len(gray_frames)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        a = gray_frames[i - 1].astype(np.float64)
        b = gray_frames[i].astype(np.float64)
        # Normalized cross-correlation: 1 - NCC
        ma, mb = a.mean(), b.mean()
        sa = np.std(a) + 1e-6
        sb = np.std(b) + 1e-6
        ncc = float(np.mean((a - ma) * (b - mb)) / (sa * sb))
        scores[i] = 1.0 - ncc  # larger value means larger viewpoint difference
    return scores


def depth_viewpoint_frame_indices(
    video_path: str,
    nframes: int,
    total_frames: Optional[int] = None,
    video_fps: Optional[float] = None,
    max_cand: int = 128,
    w_depth: float = 0.5,
    w_viewpoint: float = 0.5,
) -> List[int]:
    """
    Depth + Viewpoint jointly driven spatial-aware frame selection.

    Combined score = w_depth * depth_norm + w_viewpoint * viewpoint_norm
    Prioritize frames with rich geometric information and large viewpoint changes while ensuring uniform temporal distribution.

    Args:
        w_depth: Weight for depth proxy score (default 0.5).
        w_viewpoint: Weight for viewpoint change score (default 0.5).

    Returns:
        List of frame indices (sorted), length nframes.
    """
    nframes = _round_nframes(nframes)
    if total_frames is None or video_fps is None:
        total_frames, video_fps = _load_video_metadata(video_path)
    total_frames = max(1, total_frames)

    if total_frames <= nframes:
        indices = list(range(total_frames))
        while len(indices) < nframes:
            indices.append(total_frames - 1)
        return indices[:nframes]

    try:
        gray, cand_indices, total = _read_frames_for_scores(video_path, max_frames=min(max_cand, total_frames))
        depth_scores = compute_depth_proxy_scores(gray)
        viewpoint_scores = compute_viewpoint_change_scores(gray)

        # Normalize to [0, 1]
        def _normalize(arr):
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                return (arr - mn) / (mx - mn)
            return np.zeros_like(arr)

        depth_norm = _normalize(depth_scores)
        viewpoint_norm = _normalize(viewpoint_scores)
        combined = w_depth * depth_norm + w_viewpoint * viewpoint_norm

        n_cand = len(cand_indices)
        if n_cand <= nframes:
            return _round_uniform_indices(0, total_frames - 1, nframes)

        # Select optimal candidate frame for each target time point on uniform temporal grid
        target_ts = np.linspace(0, total_frames - 1, nframes)
        out = []
        used = set()
        for t in target_ts:
            best_i = None
            best_val = -1e9
            for i in range(n_cand):
                if i in used:
                    continue
                idx = cand_indices[i]
                dist = abs(idx - t) / max(total_frames, 1)
                s = combined[i] - 0.5 * dist
                if s > best_val:
                    best_val = s
                    best_i = i
            if best_i is not None:
                used.add(best_i)
                out.append(cand_indices[best_i])

        out = sorted(set(out))
        if len(out) < nframes:
            uniform = _round_uniform_indices(0, total_frames - 1, nframes)
            for u in uniform:
                if u not in out:
                    out.append(u)
                if len(out) >= nframes:
                    break
            out = sorted(out)[:nframes]
        elif len(out) > nframes:
            out = _round_uniform_from_list(out, nframes)
        return out
    except Exception:
        return _round_uniform_indices(0, total_frames - 1, nframes)


def get_depth_aware_frames_for_inference(
    video_path: str,
    nframes: int,
    w_depth: float = 0.5,
    w_viewpoint: float = 0.5,
) -> Tuple[List[Image.Image], List[int], float]:
    """
    Called by run_spatial_inference (Frame Sampling Innovation): depth + viewpoint jointly driven frame selection.
    Returns (frames_pil, frame_indices, sample_fps).
    """
    total_frames, video_fps = _load_video_metadata(video_path)
    nframes = _round_nframes(nframes)
    indices = depth_viewpoint_frame_indices(
        video_path, nframes, total_frames, video_fps,
        w_depth=w_depth, w_viewpoint=w_viewpoint,
    )
    frames_pil = extract_frames_as_pil(video_path, indices)
    duration_sec = total_frames / max(video_fps, 1e-6)
    sample_fps = len(frames_pil) / max(duration_sec, 1e-6)
    return frames_pil, indices, sample_fps
