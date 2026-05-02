#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing and Reliability Innovation Module
=================================================
1. Geometric Self-Correction
   Automatically correct unreasonable outputs for counting and distance tasks based on prior ranges.

2. Bayesian Uncertainty Quantification
   Estimate answer distribution through multiple temperature samplings, providing mean ± std confidence intervals.

3. Hallucination Detection & Suppression
   Detect inconsistent answers through consistency rules and cross-task validation, flagging high-risk outputs.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import List, Dict, Any, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Spatial prior constraints (for self-correction)
# ─────────────────────────────────────────────────────────────────────────────

SPATIAL_PRIORS = {
    "count": {
        "min": 0,
        "max": 50,          # Max count for single category in indoor scenes
        "typical_max": 20,  # Suspicious if exceeding this value
    },
    "distance": {
        "min": 0.05,        # Minimum distance 5cm (unit: meters)
        "max": 30.0,        # Maximum indoor distance ~30m
        "typical_max": 10.0,
    },
}


def _try_parse_number(s: str) -> Optional[float]:
    """Try to extract the first number (integer or float) from string."""
    if not s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s.strip())
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Geometric Self-Correction
# ─────────────────────────────────────────────────────────────────────────────

def geo_correct_answer(answer: str, task: str) -> Tuple[str, str]:
    """
    Apply geometric self-correction to numeric answers based on spatial priors.

    Args:
        answer: Raw answer string from model.
        task: Task name (containing "count" or "distance").

    Returns:
        (corrected_answer, correction_note)
        correction_note is empty if no correction needed.
    """
    task_l = task.lower()
    is_count = "count" in task_l
    is_dist = "distance" in task_l

    if not (is_count or is_dist):
        return answer, ""

    val = _try_parse_number(answer)
    if val is None:
        return answer, ""

    prior_key = "count" if is_count else "distance"
    prior = SPATIAL_PRIORS[prior_key]

    note = ""
    corrected = val

    if val < prior["min"]:
        corrected = prior["min"]
        note = f"[geo-correct] value {val} < min {prior['min']}, corrected"
    elif val > prior["max"]:
        corrected = prior["max"]
        note = f"[geo-correct] value {val} > max {prior['max']}, clipped"
    elif val > prior["typical_max"]:
        note = f"[geo-correct warning] value {val} exceeds typical range [0, {prior['typical_max']}], please verify"

    if corrected != val:
        out_str = str(int(corrected)) if is_count else f"{corrected:.2f}"
        return out_str, note
    return answer, note


def geo_correct_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply geometric self-correction to all tasks in results list, update in-place with correction_note."""
    for r in results:
        corrected, note = geo_correct_answer(r.get("answer", ""), r.get("task", ""))
        r["answer_raw_before_correction"] = r["answer"]
        r["answer"] = corrected
        r["correction_note"] = note
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bayesian Uncertainty Quantification
# ─────────────────────────────────────────────────────────────────────────────

def compute_uncertainty(answers: List[str], task: str) -> Dict[str, Any]:
    """
    Compute uncertainty statistics for multiple sampled answers.

    Numeric tasks (count/distance): compute mean, std, 95% CI, coefficient of variation.
    Descriptive tasks: compute answer diversity (unique ratio).

    Args:
        answers: List of answer strings from multiple samplings.
        task: Task name.

    Returns:
        Dict containing mean, std, cv, ci_low, ci_high, confidence_level, etc.
    """
    task_l = task.lower()
    is_numeric = "count" in task_l or "distance" in task_l

    unc = {"n_samples": len(answers), "answers": answers}

    if is_numeric:
        nums = []
        for a in answers:
            v = _try_parse_number(a)
            if v is not None:
                nums.append(v)
        if len(nums) >= 2:
            mean_val = statistics.mean(nums)
            std_val = statistics.stdev(nums)
            cv = std_val / max(abs(mean_val), 1e-6)
            # Approximate 95% CI (t-distribution, use 1.96 * std/sqrt(n) for small samples)
            n = len(nums)
            t_factor = 1.96 if n >= 30 else {1: 12.7, 2: 4.30, 3: 3.18, 4: 2.78,
                                               5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
                                               9: 2.26, 10: 2.23}.get(n - 1, 2.0)
            margin = t_factor * std_val / math.sqrt(n)
            unc.update({
                "mean": round(mean_val, 3),
                "std": round(std_val, 3),
                "cv": round(cv, 3),
                "ci_low": round(mean_val - margin, 3),
                "ci_high": round(mean_val + margin, 3),
                "n_valid": n,
            })
            # Confidence level
            if cv < 0.1:
                unc["confidence_level"] = "High"
            elif cv < 0.3:
                unc["confidence_level"] = "Medium"
            else:
                unc["confidence_level"] = "Low"
        elif len(nums) == 1:
            unc.update({"mean": nums[0], "std": 0.0, "cv": 0.0,
                        "ci_low": nums[0], "ci_high": nums[0],
                        "n_valid": 1, "confidence_level": "Single sample"})
        else:
            unc["confidence_level"] = "Cannot quantify (no numeric answers)"
    else:
        unique_ratio = len(set(answers)) / max(len(answers), 1)
        unc["unique_ratio"] = round(unique_ratio, 3)
        unc["confidence_level"] = "High" if unique_ratio < 0.5 else "Medium" if unique_ratio < 0.8 else "Low"

    return unc


def format_uncertainty_report(unc: Dict[str, Any]) -> str:
    """Format uncertainty dict into human-readable string."""
    lines = [f"  Confidence: {unc.get('confidence_level', 'Unknown')}",
             f"  Samples: {unc.get('n_samples', 0)}"]
    if "mean" in unc:
        lines.append(f"  Mean ± Std: {unc['mean']} ± {unc['std']}")
        lines.append(f"  95% CI: [{unc.get('ci_low', '?')}, {unc.get('ci_high', '?')}]")
        lines.append(f"  CV: {unc.get('cv', '?')}")
    if "unique_ratio" in unc:
        lines.append(f"  Diversity: {unc['unique_ratio']:.0%}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hallucination Detection & Suppression
# ─────────────────────────────────────────────────────────────────────────────

HALLUCINATION_PATTERNS = [
    # Common spatial hallucination patterns
    (r"\b0\s*(chairs?|tables?|desks?)\b", "Claims zero objects, but scene likely has furniture"),
    (r"\b(\d{3,})\s*(chairs?|tables?|meters?|objects?)\b", "Abnormally large count/distance (>3 digits)"),
    (r"distance.*?\b(0\.0+|0)\s*(m|meter|meters)?\b", "Distance of zero is physically impossible"),
    (r"(infinite|infinity)\s*(distance)", "Infinite/non-existent distance"),
]

CROSS_TASK_CHECKS = [
    # (rule description, check function)
    ("Count is zero but layout description mentions furniture",
     lambda r: (
         "count" in r.get("task", "").lower()
     ) and r.get("answer", "").strip() == "0"),
]


def detect_hallucinations(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect common spatial hallucination patterns in inference results, append hallucination_flags field.

    Returns:
        results with hallucination_flags field added in-place.
    """
    for r in results:
        flags = []
        text = (r.get("answer", "") + " " + r.get("raw_output", "")).lower()

        # Pattern matching detection
        for pat, msg in HALLUCINATION_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                flags.append(msg)

        # Numeric prior boundary check
        task_l = r.get("task", "").lower()
        val = _try_parse_number(r.get("answer", ""))
        if val is not None:
            if "count" in task_l:
                if val < 0:
                    flags.append(f"Negative count ({val}) is invalid")
                elif val > SPATIAL_PRIORS["count"]["typical_max"]:
                    flags.append(f"Abnormally large count ({val} > {SPATIAL_PRIORS['count']['typical_max']})")
            elif "distance" in task_l:
                if val <= 0:
                    flags.append(f"Distance ≤ 0 ({val}) is invalid")
                elif val > SPATIAL_PRIORS["distance"]["max"]:
                    flags.append(f"Distance exceeds max indoor range ({val} > {SPATIAL_PRIORS['distance']['max']}m)")

        r["hallucination_flags"] = flags
        r["hallucination_risk"] = "High" if len(flags) >= 2 else ("Medium" if flags else "Low")
    return results


def cross_task_consistency_check(results: List[Dict[str, Any]]) -> str:
    """
    Cross-task consistency check: verify logical consistency between count, layout, and distance answers.

    Returns:
        Consistency report string.
    """
    report_lines = ["=== Cross-task Consistency Check ==="]

    # Collect answers from each task
    count_ans = None
    layout_ans = None
    dist_ans = None
    for r in results:
        tl = r.get("task", "").lower()
        if "count" in tl:
            count_ans = r.get("answer", "")
        elif "layout" in tl:
            layout_ans = r.get("answer", "")
        elif "distance" in tl:
            dist_ans = r.get("answer", "")

    issues = []

    # Rule 1: Count is zero but layout mentions furniture words
    furniture_words = ["chair", "table", "desk", "sofa", "bed", "cabinet"]
    if count_ans is not None and _try_parse_number(count_ans) == 0:
        if layout_ans and any(w in layout_ans.lower() for w in furniture_words):
            issues.append("[Conflict] count=0 but layout mentions furniture")

    # Rule 2: Very small distance but layout describes large room
    if dist_ans is not None:
        dist_val = _try_parse_number(dist_ans)
        if dist_val is not None and dist_val < 0.3:
            if layout_ans and any(w in layout_ans.lower() for w in ["large", "spacious", "big"]):
                issues.append(f"[Conflict] Very short distance ({dist_val}m) but layout describes large space")

    # Rule 3: Check consistency between count value and numbers in layout description
    if count_ans is not None and layout_ans is not None:
        count_val = _try_parse_number(count_ans)
        if count_val is not None:
            # Check if layout description contains significantly different numbers
            nums_in_layout = [float(m) for m in re.findall(r'\b\d+\b', layout_ans)]
            if nums_in_layout:
                min_n = min(nums_in_layout)
                max_n = max(nums_in_layout)
                if count_val > 0 and (count_val < min_n * 0.3 or count_val > max_n * 3):
                    issues.append(f"[potential conflict] count={count_val}, layout mentions numbers in [{min_n}, {max_n}]")

    if issues:
        report_lines += issues
    else:
        report_lines.append("[OK] No obvious cross-task conflicts found")

    return "\n".join(report_lines)


def build_reliability_summary_block(
    results: List[Dict[str, Any]],
    consistency_report: str,
    *,
    do_geo: bool,
    do_hall: bool,
    do_cross: bool,
) -> Tuple[str, str]:
    """
    Generate reliability summary block for Markdown output and print one line for eval script parsing.

    Evaluation conventions (consistent with paper tables):
    - Hallucination rate: when hallucination detection is enabled, count tasks triggering at least one rule / total tasks ×100%.
    - Geometric intervention rate: when geo-correct is enabled, count numeric tasks (count/distance) with non-empty correction_note containing "[geo-correct" / total numeric tasks ×100%.
    - Cross-task conflicts: when cross-validate is enabled, count lines containing "[Conflict]" or "[potential conflict]" in consistency report; null otherwise.
    """
    n = max(len(results), 1)
    flagged = sum(1 for r in results if r.get("hallucination_flags"))

    def _is_numeric_spatial_task(task: str) -> bool:
        tl = (task or "").lower()
        return "count" in tl or "distance" in tl

    numeric_rows = [r for r in results if _is_numeric_spatial_task(r.get("task", ""))]
    geo_denom = max(len(numeric_rows), 1)
    geo_iv = sum(
        1 for r in numeric_rows
        if r.get("correction_note") and "[geo-correct" in r["correction_note"]
    )
    cross_issues = (
        consistency_report.count("[Conflict]") + consistency_report.count("[potential conflict]")
        if do_cross else None
    )

    payload = {
        "hallucination_check": do_hall,
        "hall_flagged_tasks": flagged,
        "hall_task_total": n,
        "hall_rate_pct": round(100.0 * flagged / n, 2) if do_hall else None,
        "geo_correction_check": do_geo,
        "geo_intervention_tasks": geo_iv,
        "geo_numeric_task_total": geo_denom,
        "geo_rate_pct": round(100.0 * geo_iv / geo_denom, 2) if do_geo else None,
        "cross_task_check": do_cross,
        "cross_issue_count": cross_issues if do_cross else None,
    }

    json_blob = json.dumps(payload, ensure_ascii=False, indent=2)
    lines = [
        "",
        "## Reliability Summary",
        "",
        "<!-- COSMES_RELIABILITY_JSON",
        json_blob,
        "-->",
        "",
        "Operational definitions (single pass, three tasks):",
        f"- **Hallucination detection**: {'enabled' if do_hall else 'not run (report N/A)'}",
    ]
    if do_hall:
        lines.append(
            f"  - Tasks with $\\geq 1$ hallucination flag: **{flagged}/{n}** "
            f"({payload['hall_rate_pct']}%)"
        )
    lines.append(
        f"- **Geometric self-correction**: {'enabled' if do_geo else 'not run (report N/A)'}"
    )
    if do_geo:
        lines.append(
            f"  - Numeric tasks with clip/warn note: **{geo_iv}/{geo_denom}** "
            f"({payload['geo_rate_pct']}%)"
        )
    lines.append(
        f"- **Cross-task consistency**: {'enabled' if do_cross else 'not run (report N/A)'}"
    )
    if do_cross:
        lines.append(f"  - Contradiction / potential-conflict lines: **{cross_issues}**")

    def _fmt(v):
        return "NA" if v is None else str(v)

    one_line = (
        f"  [reliability-summary] "
        f"hall_pct={_fmt(payload['hall_rate_pct'])} geo_pct={_fmt(payload['geo_rate_pct'])} "
        f"cross_n={_fmt(cross_issues)}"
    )
    return "\n".join(lines), one_line


def full_postprocess(
    results: List[Dict[str, Any]],
    do_geo_correct: bool = True,
    do_hallucination: bool = True,
    do_cross_validate: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    One-click postprocessing: geometric self-correction + hallucination detection + cross-task consistency check.

    Returns:
        (processed_results, consistency_report)
    """
    if do_geo_correct:
        results = geo_correct_results(results)
    if do_hallucination:
        results = detect_hallucinations(results)
    consistency_report = ""
    if do_cross_validate:
        consistency_report = cross_task_consistency_check(results)
    return results, consistency_report
