"""Streamlit dashboard for Week 4 red-team benchmark comparison and failure replay."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.graph_objects as go
import streamlit as st

from app_benchmark import runner as bench_runner
from core_tools.llm_client import format_llm_error_zh, probe_ollama_models
from core_tools.schemas import BenchmarkCase

DEFAULT_RESULTS_DIR = bench_runner.DEFAULT_RESULTS_DIR

DIMENSION_ORDER: List[str] = [
    "red_flag_interception",
    "longitudinal_memory",
    "tcm_syndrome_and_tools",
    "over_refusal_calibration",
]

DIMENSION_LABELS: Dict[str, str] = {
    "red_flag_interception": "红旗安全拦截",
    "longitudinal_memory": "纵向病历记忆",
    "tcm_syndrome_and_tools": "辨证与工具协同",
    "over_refusal_calibration": "防过度拒答校准",
}

MODE_LABELS: Dict[str, str] = {
    "bare_llm": "裸模型 (Bare LLM)",
    "harness_copilot": "Harness 护栏协同 (Copilot)",
}

STATUS_OPTIONS: List[str] = [
    "全部",
    "仅翻车",
    "FAIL_SAFETY",
    "FAIL_OVER_REFUSAL",
    "FAIL_RULE_ENGINE",
    "FAIL_ACCURACY",
    "FAIL_INFRA",
    "PASS",
]

STATUS_BADGES: Dict[str, str] = {
    "PASS": "🟢 PASS",
    "FAIL_SAFETY": "🔴 FAIL_SAFETY",
    "FAIL_OVER_REFUSAL": "🟠 FAIL_OVER_REFUSAL",
    "FAIL_RULE_ENGINE": "🟣 FAIL_RULE_ENGINE",
    "FAIL_ACCURACY": "🟡 FAIL_ACCURACY",
    "FAIL_INFRA": "⚫ FAIL_INFRA",
}

STATUS_FULL_LABELS: Dict[str, str] = {
    "PASS": "🟢 PASS（通过）",
    "FAIL_SAFETY": "🔴 FAIL_SAFETY（安全红线否决）",
    "FAIL_OVER_REFUSAL": "🟠 FAIL_OVER_REFUSAL（过度拒答）",
    "FAIL_RULE_ENGINE": "🟣 FAIL_RULE_ENGINE（规则引擎漏报/误报）",
    "FAIL_ACCURACY": "🟡 FAIL_ACCURACY（准确性/数值失误）",
    "FAIL_INFRA": "⚫ FAIL_INFRA（调用/基础设施异常）",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert an arbitrary JSON field into a finite float."""
    if value is None or isinstance(value, bool):
        return default
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return num if math.isfinite(num) else default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert an arbitrary JSON field into an int."""
    if value is None or isinstance(value, bool):
        return default
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return int(num) if math.isfinite(num) else default

MODEL_PALETTE: List[str] = [
    "#2563EB",
    "#D97706",
    "#059669",
    "#7C3AED",
    "#DC2626",
    "#0891B2",
    "#DB2777",
    "#4F46E5",
]
FALLBACK_PALETTE: List[str] = MODEL_PALETTE


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB hex color into rgba(r, g, b, alpha) string."""
    cleaned = str(hex_color or "").strip().lstrip("#")
    if len(cleaned) != 6:
        return f"rgba(37, 99, 235, {alpha:.2f})"
    try:
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
    except ValueError:
        return f"rgba(37, 99, 235, {alpha:.2f})"
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def assign_model_colors(runs: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Assign a distinct primary hex color to each unique model in order of appearance."""
    mapping: Dict[str, str] = {}
    for run in runs or []:
        if not isinstance(run, dict):
            continue
        model = str(run.get("model") or "unknown").strip() or "unknown"
        if model not in mapping:
            mapping[model] = MODEL_PALETTE[len(mapping) % len(MODEL_PALETTE)]
    return mapping


def get_results_dir() -> Path:
    """Return the benchmark results directory, honoring test overrides and runner defaults."""
    override = os.getenv("ORTHO_BENCH_RESULTS_DIR", "").strip()
    if override:
        return Path(override)
    return bench_runner.get_default_results_dir()


def scan_report_files(results_dir: Optional[Path] = None) -> List[Path]:
    """Scan JSON benchmark reports sorted reverse chronologically (newest first)."""
    target_dir = results_dir or get_results_dir()
    if not target_dir.exists() or not target_dir.is_dir():
        return []
    files = [p for p in target_dir.glob("*.json") if p.is_file()]

    def _sort_key(path: Path) -> Tuple[float, str]:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (mtime, path.name)

    return sorted(files, key=_sort_key, reverse=True)


def load_report_file(report_path: Path | str) -> Optional[Dict[str, Any]]:
    """Load and validate a JSON benchmark report file safely."""
    path = Path(report_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        return None
    return data


def format_report_option(report_path: Path) -> str:
    """Format a human-readable dropdown option for a report file."""
    data = load_report_file(report_path)
    if not data:
        return f"{report_path.name} (无法解析)"
    created = str(data.get("created_at") or "-")
    mode_tag = "Mock" if data.get("mock_mode") else "Live"
    case_count = _safe_int(data.get("case_count"), 0)
    priv_tag = "含私有集" if data.get("include_private") else "仅公开集"
    return f"{report_path.name} | {created} | {mode_tag} | {case_count} 题 | {priv_tag}"


def pair_bare_and_harness(
    runs: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Pair bare_llm and harness_copilot runs (preferring same model) for delta cards."""
    by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for run in runs or []:
        if not isinstance(run, dict):
            continue
        model = str(run.get("model") or "unknown")
        mode = str(run.get("mode") or "")
        by_model.setdefault(model, {})
        if mode and mode not in by_model[model]:
            by_model[model][mode] = run

    pairs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for model, mode_map in by_model.items():
        if "bare_llm" in mode_map and "harness_copilot" in mode_map:
            pairs.append((model, mode_map["bare_llm"], mode_map["harness_copilot"]))

    if pairs:
        return pairs

    bare_runs = [r for r in (runs or []) if isinstance(r, dict) and r.get("mode") == "bare_llm"]
    harness_runs = [r for r in (runs or []) if isinstance(r, dict) and r.get("mode") == "harness_copilot"]
    if bare_runs and harness_runs:
        b = bare_runs[0]
        h = harness_runs[0]
        label = f"{b.get('model', 'bare')} vs {h.get('model', 'harness')}"
        pairs.append((label, b, h))
    return pairs


def _extract_dim_entry(run: Dict[str, Any], dim_key: str) -> Dict[str, Any]:
    if not isinstance(run, dict):
        return {"total": 0, "passed": 0, "pass_rate": 0.0}
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    dims = summary.get("dimensions") if isinstance(summary.get("dimensions"), dict) else {}
    entry = dims.get(dim_key) if isinstance(dims.get(dim_key), dict) else {}
    return {
        "total": _safe_int(entry.get("total"), 0),
        "passed": _safe_int(entry.get("passed"), 0),
        "pass_rate": _safe_float(entry.get("pass_rate"), 0.0),
    }


def build_dimension_comparison_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build 4-dimension comparison rows with Chinese names, pass rates, and per-model deltas."""
    valid_runs = [r for r in (runs or []) if isinstance(r, dict)]
    pairs = pair_bare_and_harness(valid_runs)

    rows: List[Dict[str, Any]] = []
    for dim_key in DIMENSION_ORDER:
        dim_totals = [_extract_dim_entry(r, dim_key)["total"] for r in valid_runs]
        case_total = max(dim_totals) if dim_totals else 0
        row: Dict[str, Any] = {
            "评测维度": DIMENSION_LABELS[dim_key],
            "维度标识": dim_key,
            "病例数": case_total,
        }
        for idx, run in enumerate(valid_runs):
            run_label = str(run.get("run_label") or f"run_{idx + 1}")
            entry = _extract_dim_entry(run, dim_key)
            row[run_label] = f"{entry['pass_rate']:.1f}% ({entry['passed']}/{entry['total']})"

        if len(pairs) == 1:
            _, bare_run, harness_run = pairs[0]
            bare_rate = _extract_dim_entry(bare_run, dim_key)["pass_rate"]
            harness_rate = _extract_dim_entry(harness_run, dim_key)["pass_rate"]
            delta = harness_rate - bare_rate
            row["护栏增益 (Δ)"] = f"{delta:+.1f}%"
        elif len(pairs) > 1:
            for model_label, bare_run, harness_run in pairs:
                bare_rate = _extract_dim_entry(bare_run, dim_key)["pass_rate"]
                harness_rate = _extract_dim_entry(harness_run, dim_key)["pass_rate"]
                delta = harness_rate - bare_rate
                row[f"护栏增益 (Δ) · {model_label}"] = f"{delta:+.1f}%"

        rows.append(row)
    return rows


def _run_visual_style(
    run: Dict[str, Any],
    index: int,
    model_colors: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str, str, float]:
    """Return (base_hex, line_color, fill_color, dash_style, bar_opacity) per model & mode."""
    model = str(run.get("model") or "unknown").strip() or "unknown"
    palette_map = model_colors or {}
    base_hex = palette_map.get(model) or MODEL_PALETTE[index % len(MODEL_PALETTE)]
    mode = str(run.get("mode") or "")
    if mode == "bare_llm":
        return (
            base_hex,
            _hex_to_rgba(base_hex, 0.72),
            _hex_to_rgba(base_hex, 0.06),
            "dash",
            0.42,
        )
    if mode == "harness_copilot":
        return (
            base_hex,
            base_hex,
            _hex_to_rgba(base_hex, 0.20),
            "solid",
            0.96,
        )
    return (
        base_hex,
        base_hex,
        _hex_to_rgba(base_hex, 0.14),
        "solid",
        0.85,
    )


def build_radar_figure(runs: List[Dict[str, Any]]) -> go.Figure:
    """Build a 4-dimension Plotly Scatterpolar radar chart with per-model color coding."""
    fig = go.Figure()
    theta_labels = [DIMENSION_LABELS[d] for d in DIMENSION_ORDER]
    closed_theta = theta_labels + [theta_labels[0]]
    valid_runs = [r for r in (runs or []) if isinstance(r, dict)]
    model_colors = assign_model_colors(valid_runs)

    for idx, run in enumerate(valid_runs):
        run_label = str(run.get("run_label") or f"run_{idx + 1}")
        rates = [_extract_dim_entry(run, d)["pass_rate"] for d in DIMENSION_ORDER]
        closed_rates = rates + [rates[0]]
        _, line_color, fill_color, dash_style, _ = _run_visual_style(run, idx, model_colors)
        line_width = 2.0 if dash_style == "dash" else 2.8

        fig.add_trace(
            go.Scatterpolar(
                r=closed_rates,
                theta=closed_theta,
                name=run_label,
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=line_color, width=line_width, dash=dash_style),
                marker=dict(size=6, color=line_color),
                hovertemplate="%{theta}: %{r:.1f}%<extra>" + html.escape(run_label) + "</extra>",
            )
        )

    fig.update_layout(
        title=dict(text="四维评测通过率雷达图 (Pass Rate % · 同模型同色：虚线 Bare / 实线 Harness)", x=0.02),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 105],
                tickvals=[0, 25, 50, 75, 100],
                ticksuffix="%",
            )
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        margin=dict(t=55, b=75, l=55, r=55),
        height=430,
    )
    return fig


def build_bar_figure(runs: List[Dict[str, Any]]) -> go.Figure:
    """Build a grouped Plotly Bar chart with per-model color coding and mode opacity."""
    fig = go.Figure()
    x_labels = [DIMENSION_LABELS[d] for d in DIMENSION_ORDER]
    valid_runs = [r for r in (runs or []) if isinstance(r, dict)]
    model_colors = assign_model_colors(valid_runs)

    for idx, run in enumerate(valid_runs):
        run_label = str(run.get("run_label") or f"run_{idx + 1}")
        rates = [_extract_dim_entry(run, d)["pass_rate"] for d in DIMENSION_ORDER]
        base_hex, _, _, dash_style, bar_opacity = _run_visual_style(run, idx, model_colors)
        bar_color = _hex_to_rgba(base_hex, bar_opacity) if dash_style == "dash" else base_hex
        marker_cfg: Dict[str, Any] = {
            "color": bar_color,
            "line": {"color": base_hex, "width": 1.6 if dash_style == "dash" else 0.8},
        }
        if dash_style == "dash":
            marker_cfg["pattern"] = {"shape": "/", "fgcolor": _hex_to_rgba(base_hex, 0.35)}

        fig.add_trace(
            go.Bar(
                name=run_label,
                x=x_labels,
                y=rates,
                text=[f"{r:.1f}%" for r in rates],
                textposition="outside",
                marker=marker_cfg,
                hovertemplate="%{x}: %{y:.1f}%<extra>" + html.escape(run_label) + "</extra>",
            )
        )

    fig.update_layout(
        title=dict(text="四维评测通过率分组柱状图（斜纹浅色 Bare / 饱和实色 Harness）", x=0.02),
        barmode="group",
        yaxis=dict(range=[0, 118], title="通过率 (%)", ticksuffix="%"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        margin=dict(t=55, b=75, l=50, r=30),
        height=430,
    )
    return fig


def highlight_violations_html(
    text: str,
    violations: Optional[List[Dict[str, Any]]] = None,
    suppressed_hits: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Escape assistant text and highlight matched violation / suppressed phrases."""
    raw = text or ""
    if not raw:
        return "<div><em>(空回复)</em></div>"

    candidates: List[Tuple[str, str]] = []
    for item in violations or []:
        matched = str(item.get("matched_text") or "").strip()
        if matched:
            candidates.append((matched, "violation"))
    for item in suppressed_hits or []:
        matched = str(item.get("matched_text") or "").strip()
        if matched:
            candidates.append((matched, "suppressed"))

    if not candidates:
        return (
            '<div style="white-space: pre-wrap; line-height: 1.65; font-size: 0.95rem;">'
            f"{html.escape(raw)}"
            "</div>"
        )

    # Prefer violations over suppressed hits, then longer phrases first.
    candidates.sort(key=lambda pair: (0 if pair[1] == "violation" else 1, -len(pair[0])))

    spans: List[Tuple[int, int, str]] = []
    for phrase, kind in candidates:
        start = 0
        while True:
            idx = raw.find(phrase, start)
            if idx == -1:
                break
            end = idx + len(phrase)
            overlaps = any(not (end <= s_start or idx >= s_end) for s_start, s_end, _ in spans)
            if not overlaps:
                spans.append((idx, end, kind))
            start = end

    if not spans:
        return (
            '<div style="white-space: pre-wrap; line-height: 1.65; font-size: 0.95rem;">'
            f"{html.escape(raw)}"
            "</div>"
        )

    spans.sort(key=lambda s: s[0])
    chunks: List[str] = []
    cursor = 0
    for s_start, s_end, kind in spans:
        if s_start > cursor:
            chunks.append(html.escape(raw[cursor:s_start]))
        segment = html.escape(raw[s_start:s_end])
        if kind == "violation":
            chunks.append(
                '<mark title="命中禁用违规片段" '
                'style="background-color:#FECACA;color:#991B1B;padding:1px 5px;'
                'border-radius:4px;font-weight:700;border:1px solid #F87171;">'
                f"{segment}</mark>"
            )
        else:
            chunks.append(
                '<mark title="命中否定语境保护放行片段" '
                'style="background-color:#DCFCE7;color:#166534;padding:1px 5px;'
                'border-radius:4px;font-weight:600;border:1px solid #86EFAC;">'
                f"{segment}</mark>"
            )
        cursor = s_end

    if cursor < len(raw):
        chunks.append(html.escape(raw[cursor:]))

    return (
        '<div style="white-space: pre-wrap; line-height: 1.65; font-size: 0.95rem;">'
        + "".join(chunks)
        + "</div>"
    )


def _result_status(res: Dict[str, Any]) -> str:
    return str(res.get("status") or ("PASS" if res.get("passed") else "FAIL_SAFETY"))


def _result_passed(res: Dict[str, Any]) -> bool:
    return bool(res.get("passed", _result_status(res) == "PASS"))


def _case_matches_search(res: Dict[str, Any], query: str, run_label: str = "") -> bool:
    """Case-insensitive substring search across case_id, category, turns, and failure details."""
    needle = (query or "").strip().lower()
    if not needle:
        return True
    dim = str(res.get("dimension") or "")
    parts: List[str] = [
        str(res.get("case_id") or ""),
        str(res.get("category") or ""),
        dim,
        DIMENSION_LABELS.get(dim, ""),
        str(res.get("expected_action") or ""),
        run_label,
    ]
    for reason in res.get("failure_reasons") or []:
        parts.append(str(reason))
    for viol in res.get("forbidden_violations") or []:
        if isinstance(viol, dict):
            parts.append(str(viol.get("matched_text") or ""))
            parts.append(str(viol.get("rationale") or ""))
    for turn in res.get("turns") or []:
        if isinstance(turn, dict):
            parts.append(str(turn.get("user_msg") or ""))
            parts.append(str(turn.get("assistant_msg") or ""))
    haystack = "\n".join(parts).lower()
    return needle in haystack


def find_rescued_case_keys(
    runs: List[Dict[str, Any]],
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """Identify (run_label, case_id) and (pair_label, case_id) rescued by harness_copilot."""
    rescued_run_cases: Set[Tuple[str, str]] = set()
    rescued_pair_cases: Set[Tuple[str, str]] = set()
    for pair_label, bare_run, harness_run in pair_bare_and_harness(runs):
        bare_label = str(bare_run.get("run_label") or "")
        harness_label = str(harness_run.get("run_label") or "")
        bare_map = {
            str(r.get("case_id") or ""): r
            for r in (bare_run.get("results") or [])
            if isinstance(r, dict) and r.get("case_id")
        }
        harness_map = {
            str(r.get("case_id") or ""): r
            for r in (harness_run.get("results") or [])
            if isinstance(r, dict) and r.get("case_id")
        }
        for case_id, b_res in bare_map.items():
            h_res = harness_map.get(case_id)
            if h_res is None:
                continue
            if not _result_passed(b_res) and _result_passed(h_res):
                rescued_pair_cases.add((pair_label, case_id))
                rescued_run_cases.add((bare_label, case_id))
                rescued_run_cases.add((harness_label, case_id))
    return rescued_run_cases, rescued_pair_cases


def filter_case_results(
    runs: List[Dict[str, Any]],
    *,
    selected_run: str = "全部",
    selected_status: str = "全部",
    selected_category: str = "全部",
    selected_dimension: str = "全部",
    search_query: str = "",
    only_rescued: bool = False,
) -> List[Dict[str, Any]]:
    """Flatten and filter case results across benchmark runs, failures first."""
    rescued_run_cases, _ = find_rescued_case_keys(runs) if only_rescued else (set(), set())
    records: List[Dict[str, Any]] = []
    for idx, run in enumerate(runs or []):
        if not isinstance(run, dict):
            continue
        run_label = str(run.get("run_label") or f"run_{idx + 1}")
        if selected_run != "全部" and run_label != selected_run:
            continue

        model = str(run.get("model") or "")
        mode = str(run.get("mode") or "")
        raw_results = run.get("results")
        if not isinstance(raw_results, list):
            continue
        for res in raw_results:
            if not isinstance(res, dict):
                continue
            case_id = str(res.get("case_id") or "")
            status = _result_status(res)
            passed = _result_passed(res)
            category = str(res.get("category") or "")
            dimension = str(res.get("dimension") or "")

            if only_rescued and (run_label, case_id) not in rescued_run_cases:
                continue
            if selected_status == "仅翻车" and passed:
                continue
            if selected_status not in ("全部", "仅翻车") and status != selected_status:
                continue
            if selected_category != "全部" and category != selected_category:
                continue
            if selected_dimension != "全部" and dimension != selected_dimension:
                continue
            if search_query.strip() and not _case_matches_search(res, search_query, run_label=run_label):
                continue

            records.append(
                {
                    "run_label": run_label,
                    "model": model,
                    "mode": mode,
                    "result": res,
                }
            )

    records.sort(
        key=lambda item: (
            bool(item["result"].get("passed", False)),
            str(item["result"].get("case_id") or ""),
            0 if item["mode"] == "bare_llm" else 1,
            item["run_label"],
        )
    )
    return records


def build_side_by_side_case_pairs(
    runs: List[Dict[str, Any]],
    *,
    selected_pair: str = "全部",
    selected_status: str = "全部",
    selected_category: str = "全部",
    selected_dimension: str = "全部",
    search_query: str = "",
    only_rescued: bool = False,
) -> List[Dict[str, Any]]:
    """Align Bare LLM and Harness Copilot results per (model_pair, case_id) for dual-column comparison."""
    items: List[Dict[str, Any]] = []
    for pair_label, bare_run, harness_run in pair_bare_and_harness(runs):
        if selected_pair != "全部" and pair_label != selected_pair:
            continue
        bare_label = str(bare_run.get("run_label") or f"{pair_label} | bare_llm")
        harness_label = str(harness_run.get("run_label") or f"{pair_label} | harness_copilot")
        bare_map = {
            str(r.get("case_id") or ""): r
            for r in (bare_run.get("results") or [])
            if isinstance(r, dict) and r.get("case_id")
        }
        harness_map = {
            str(r.get("case_id") or ""): r
            for r in (harness_run.get("results") or [])
            if isinstance(r, dict) and r.get("case_id")
        }
        ordered_ids: List[str] = list(bare_map.keys())
        for cid in harness_map:
            if cid not in bare_map:
                ordered_ids.append(cid)

        for cid in ordered_ids:
            b_res = bare_map.get(cid)
            h_res = harness_map.get(cid)
            ref = b_res or h_res or {}
            category = str(ref.get("category") or "")
            dimension = str(ref.get("dimension") or "")

            b_passed = _result_passed(b_res) if b_res is not None else False
            h_passed = _result_passed(h_res) if h_res is not None else False
            b_status = _result_status(b_res) if b_res is not None else ""
            h_status = _result_status(h_res) if h_res is not None else ""
            rescued = (b_res is not None and h_res is not None and not b_passed and h_passed)
            regressed = (b_res is not None and h_res is not None and b_passed and not h_passed)
            any_failed = (b_res is not None and not b_passed) or (h_res is not None and not h_passed)

            if only_rescued and not rescued:
                continue
            if selected_status == "仅翻车" and not any_failed:
                continue
            if selected_status not in ("全部", "仅翻车"):
                if selected_status not in (b_status, h_status):
                    continue
            if selected_category != "全部" and category != selected_category:
                continue
            if selected_dimension != "全部" and dimension != selected_dimension:
                continue
            if search_query.strip():
                matched_b = b_res is not None and _case_matches_search(b_res, search_query, bare_label)
                matched_h = h_res is not None and _case_matches_search(h_res, search_query, harness_label)
                if not (matched_b or matched_h):
                    continue

            items.append(
                {
                    "pair_label": pair_label,
                    "case_id": cid,
                    "category": category,
                    "dimension": dimension,
                    "risk_level": str(ref.get("risk_level") or "-"),
                    "expected_action": str(ref.get("expected_action") or "-"),
                    "visibility": str(ref.get("visibility") or "public"),
                    "bare_run_label": bare_label,
                    "harness_run_label": harness_label,
                    "bare_result": b_res,
                    "harness_result": h_res,
                    "rescued": rescued,
                    "regressed": regressed,
                    "any_failed": any_failed,
                }
            )

    items.sort(
        key=lambda row: (
            not row["rescued"],
            not row["any_failed"],
            row["case_id"],
            row["pair_label"],
        )
    )
    return items


def merge_model_options(
    probed_ollama: Sequence[str],
    custom_models_text: str = "",
) -> Tuple[List[str], List[str]]:
    """Merge default preset, probed Ollama models, and comma/newline custom model inputs."""
    presets: List[str] = ["deepseek/deepseek-chat"]
    seen: Set[str] = set(presets)
    for raw_name in probed_ollama or []:
        name = str(raw_name or "").strip()
        if not name:
            continue
        prefixed = name if "/" in name else f"ollama_chat/{name}"
        if prefixed not in seen:
            seen.add(prefixed)
            presets.append(prefixed)

    custom_list: List[str] = []
    custom_seen: Set[str] = set()
    for token in re.split(r"[,;\n\r]+", custom_models_text or ""):
        cleaned = token.strip()
        if cleaned and cleaned not in custom_seen:
            custom_seen.add(cleaned)
            custom_list.append(cleaned)
            if cleaned not in seen:
                seen.add(cleaned)
                presets.append(cleaned)
    return presets, custom_list


def collect_report_infra_errors(report: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract unique (model, infra_or_extraction_error) pairs from a completed report."""
    errors: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for run in report.get("runs") or []:
        if not isinstance(run, dict):
            continue
        model = str(run.get("model") or "")
        for res in run.get("results") or []:
            if not isinstance(res, dict):
                continue
            err_msg = str(res.get("extraction_error") or res.get("infra_error") or "").strip()
            if err_msg and (model, err_msg) not in seen:
                seen.add((model, err_msg))
                errors.append((model, err_msg))
    return errors


def build_dimension_csv_bytes(runs: List[Dict[str, Any]]) -> bytes:
    """Export 4-dimension comparison rows as UTF-8-SIG CSV bytes for Excel."""
    rows = build_dimension_comparison_rows(runs)
    buf = io.StringIO()
    if rows:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return buf.getvalue().encode("utf-8-sig")


def build_report_json_bytes(report: Dict[str, Any]) -> bytes:
    """Serialize the current benchmark report dictionary into formatted UTF-8 JSON bytes."""
    return json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")


def build_failure_replay_markdown(
    report: Dict[str, Any],
    records: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate a structured Markdown failure replay brief for sharing or archiving."""
    runs = [r for r in (report.get("runs") or []) if isinstance(r, dict)]
    lines: List[str] = [
        f"# OrthoRehab 红队评测翻车实录简报 (`{report.get('report_id', '-')}`)",
        "",
        f"- **生成时间**：{report.get('created_at', '-')}",
        f"- **运行模式**：{'Mock 仿真' if report.get('mock_mode') else '真实模型'}",
        f"- **评测题量**：{_safe_int(report.get('case_count'), 0)} 题",
        f"- **隐私隔离**：{'含私有集 (private/)' if report.get('include_private') else '私有集已隔离 (仅 public/)'}",
        "",
        "## 1. 运行组总览",
        "",
        "| 运行项 | 总通过率 | 安全红线否决率 | 过度拒答率 | 总耗时 (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for idx, run in enumerate(runs):
        s = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        label = str(run.get("run_label") or f"run_{idx + 1}")
        lines.append(
            f"| {label} | {_safe_float(s.get('overall_pass_rate')):.1f}% "
            f"({_safe_int(s.get('passed_cases'))}/{_safe_int(s.get('total_cases'))}) "
            f"| {_safe_float(s.get('safety_veto_rate')):.1f}% ({_safe_int(s.get('safety_veto_count'))}) "
            f"| {_safe_float(s.get('over_refusal_rate')):.1f}% "
            f"({_safe_int(s.get('over_refusal_count'))}/{_safe_int(s.get('over_refusal_applicable'))}) "
            f"| {_safe_float(run.get('total_latency_ms')):.1f} |"
        )

    dim_rows = build_dimension_comparison_rows(runs)
    if dim_rows:
        cols = list(dim_rows[0].keys())
        lines.extend(
            [
                "",
                "## 2. 四维能力对比",
                "",
                "| " + " | ".join(cols) + " |",
                "| " + " | ".join(["---"] * len(cols)) + " |",
            ]
        )
        for r in dim_rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")

    target_records = records if records is not None else filter_case_results(runs, selected_status="仅翻车")
    failed_records = [item for item in target_records if not _result_passed(item.get("result") or {})]

    _, rescued_pairs = find_rescued_case_keys(runs)
    if rescued_pairs:
        lines.extend(["", "## 3. 护栏成功救回用例（Bare 翻车 → Harness 通过）", ""])
        for pair_label, cid in sorted(rescued_pairs):
            lines.append(f"- `{pair_label}` · `{cid}`")

    lines.extend(["", f"## 4. 翻车明细（共 {len(failed_records)} 条）", ""])
    if not failed_records:
        lines.append("当前筛选范围内无翻车记录。")
    for item in failed_records:
        res = item.get("result") or {}
        cid = str(res.get("case_id") or "UNKNOWN")
        status = _result_status(res)
        dim = str(res.get("dimension") or "")
        lines.append(f"### {cid} | {item.get('run_label', '')} | {status}")
        lines.append(f"- **维度**：{DIMENSION_LABELS.get(dim, dim)} (`{dim}`)")
        lines.append(f"- **分类**：{res.get('category', '-')}")
        reasons = res.get("failure_reasons") or []
        if reasons:
            lines.append("- **翻车原因**：")
            for reason in reasons:
                lines.append(f"  - {reason}")
        violations = [v for v in (res.get("forbidden_violations") or []) if isinstance(v, dict)]
        if violations:
            lines.append("- **命中禁用片段**：")
            for v in violations:
                lines.append(
                    f"  - Turn {v.get('turn', '-')}: `{v.get('matched_text', '')}` "
                    f"（{v.get('rationale', '')}）"
                )
        turns = [t for t in (res.get("turns") or []) if isinstance(t, dict)]
        if turns:
            last_turn = turns[-1]
            lines.append(f"- **末轮提问 (Turn {last_turn.get('turn', '-')})**：{last_turn.get('user_msg', '')}")
            lines.append(f"- **末轮回复**：{last_turn.get('assistant_msg', '')}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_sidebar(
    active_report: Optional[Dict[str, Any]],
    selected_path: Optional[Path],
    results_dir: Path,
) -> None:
    custom_dir = bench_runner.get_custom_cases_dir()
    with st.sidebar:
        st.header("🛡️ 红队沙箱信息")
        st.caption(f"报告目录：`{results_dir}`")
        st.caption(f"自定义病例目录：`{custom_dir}`")
        if active_report is None:
            st.info(
                "暂无已加载的评测报告。请在主区域点击 **「🚀 运行 Mock 快速评测」**、"
                "使用 **「🧪 在线发起评测」** 或导入外部评测 JSON。"
            )
            return

        include_private = bool(active_report.get("include_private", False))
        if include_private:
            st.warning("⚠️ **私有集隔离提醒**：当前报告包含 `private/` 私有红队病例，请勿直接对外公开原始报告或截图。")
        else:
            st.success("🔒 **私有集已隔离**：当前报告仅包含 `public/` 公开红队题库（`include_private=False`）。")

        st.markdown("#### 当前报告元信息")
        st.markdown(f"- **报告 ID**：`{active_report.get('report_id', '-')}`")
        st.markdown(f"- **生成时间**：`{active_report.get('created_at', '-')}`")
        mode_str = "Mock 确定性仿真 (`mock_mode=True`)" if active_report.get("mock_mode") else "真实模型 (`mock_mode=False`)"
        st.markdown(f"- **运行方式**：{mode_str}")
        st.markdown(f"- **评测题量**：`{active_report.get('case_count', 0)}` 题")
        st.markdown(f"- **运行配置数**：`{len(active_report.get('runs') or [])}` 组")
        if selected_path:
            st.caption(f"文件路径：`{selected_path}`")
        elif active_report.get("report_path"):
            st.caption(f"文件路径：`{active_report['report_path']}`")
        else:
            st.caption("来源：外部导入预览（尚未保存到本地目录）")

        st.divider()
        st.markdown("#### 维度说明")
        for dim_key in DIMENSION_ORDER:
            st.caption(f"• **{DIMENSION_LABELS[dim_key]}** (`{dim_key}`)")


def _clear_imported_preview_flag() -> None:
    st.session_state["show_imported_report"] = False


def _save_report_to_results_dir(report: Dict[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_id = str(report.get("report_id") or "imported_report").strip()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_id).strip("._") or "imported_report"
    out_path = results_dir / f"{safe_stem}.json"
    suffix = 1
    while out_path.exists():
        out_path = results_dir / f"{safe_stem}_{suffix}.json"
        suffix += 1
    payload = dict(report)
    payload["report_path"] = str(out_path)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _render_top_controls(results_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    report_files = scan_report_files(results_dir)
    path_strs = [str(p) for p in report_files]

    if "pending_report_path" in st.session_state:
        pending_path = st.session_state.pop("pending_report_path")
        if pending_path and pending_path in path_strs:
            st.session_state["bench_report_selector"] = pending_path

    if path_strs:
        current_sel = st.session_state.get("bench_report_selector")
        if current_sel not in path_strs:
            st.session_state["bench_report_selector"] = path_strs[0]

    col_select, col_btn = st.columns([3, 1.3])
    with col_select:
        if path_strs:
            selected_str = st.selectbox(
                "📂 选择评测报告（按时间倒序）",
                options=path_strs,
                format_func=lambda p_str: format_report_option(Path(p_str)),
                key="bench_report_selector",
                on_change=_clear_imported_preview_flag,
            )
            selected_path = Path(selected_str)
        else:
            st.selectbox(
                "📂 选择评测报告（按时间倒序）",
                options=["暂无报告文件"],
                disabled=True,
                key="bench_report_empty_selector",
            )
            selected_path = None

    with col_btn:
        st.write("")
        run_mock_clicked = st.button(
            "🚀 运行 Mock 快速评测",
            type="primary",
            width="stretch",
            key="run_mock_bench_btn",
        )

    if run_mock_clicked:
        try:
            with st.spinner("正在执行 Bare LLM vs Harness Copilot 双模式红队评测..."):
                new_report = bench_runner.run_benchmark(
                    mock_mode=True,
                    save_report=True,
                    output_dir=str(results_dir),
                )
            st.session_state["show_imported_report"] = False
            st.session_state["last_run_infra_alerts"] = []
            st.session_state["bench_in_memory_report"] = new_report
            new_path = str(new_report.get("report_path") or "")
            if new_path:
                st.session_state["pending_report_path"] = new_path
            st.success(f"Mock 快速评测完成：{new_report.get('report_id', '')}")
            st.rerun()
        except Exception as exc:
            st.error(f"Mock 快速评测执行失败：{exc}")

    if st.session_state.get("show_imported_report") and isinstance(
        st.session_state.get("imported_report_payload"), dict
    ):
        imported_payload = st.session_state["imported_report_payload"]
        imported_name = str(st.session_state.get("imported_report_filename") or "uploaded.json")
        b1, b2, b3 = st.columns([2.6, 1.2, 1.0])
        b1.info(f"📥 当前正在展示外部导入报告：`{imported_name}`（尚未落盘到 `{results_dir}`）")
        if b2.button("💾 保存到本地 results 目录", key="top_save_imported_btn", width="stretch"):
            saved_path = _save_report_to_results_dir(imported_payload, results_dir)
            st.session_state["show_imported_report"] = False
            st.session_state["pending_report_path"] = str(saved_path)
            st.success(f"已保存到 `{saved_path}`")
            st.rerun()
        if b3.button("↩️ 返回本地列表", key="top_exit_imported_btn", width="stretch"):
            st.session_state["show_imported_report"] = False
            st.rerun()
        return imported_payload, None

    active_report: Optional[Dict[str, Any]] = None
    if selected_path is not None:
        active_report = load_report_file(selected_path)
    if active_report is None:
        active_report = st.session_state.get("bench_in_memory_report")
    return active_report, selected_path


def _render_infra_alerts_if_any() -> None:
    alerts = st.session_state.get("last_run_infra_alerts") or []
    if not alerts:
        return
    with st.container(border=True):
        h1, h2 = st.columns([5, 1])
        h1.warning("⚠️ 最近一次在线评测中存在模型调用或结构化抽取异常（已保留其余已完成题目）：")
        if h2.button("忽略提示", key="dismiss_infra_alerts_btn", width="stretch"):
            st.session_state["last_run_infra_alerts"] = []
            st.rerun()
        for model_name, raw_err in alerts:
            zh_guide, sanitized_raw = format_llm_error_zh(raw_err, model=model_name)
            st.error(f"**模型 `{model_name}` 中文修复指引**：\n\n{zh_guide}")
            st.caption(f"诊断摘要：`{sanitized_raw}`")


def _render_workbench_tabs(results_dir: Path) -> None:
    with st.container(border=True):
        tab_run, tab_import, tab_custom = st.tabs(
            ["🧪 在线发起评测", "📥 导入外部评测 JSON", "📄 自定义病例库 (YAML)"]
        )

        with tab_custom:
            custom_dir = bench_runner.get_custom_cases_dir()
            st.caption(f"自定义病例本地持久化目录：`{custom_dir}`")
            c_dl, c_info = st.columns([1.4, 2.6])
            with c_dl:
                st.download_button(
                    "📥 下载标准空白病例模板 (.yaml)",
                    data=bench_runner.get_blank_case_yaml_template().encode("utf-8"),
                    file_name="custom_case_template.yaml",
                    mime="text/yaml",
                    key="download_blank_yaml_template_btn",
                    width="stretch",
                )
            with c_info:
                existing_custom = [
                    c.case_id
                    for c, vis in bench_runner.load_benchmark_cases(include_private=False, include_custom=True)
                    if vis == "custom"
                ]
                st.caption(
                    f"本地目录已发现自定义病例：**{len(existing_custom)}** 个"
                    + (f"（{', '.join(existing_custom[:6])}{'...' if len(existing_custom) > 6 else ''}）" if existing_custom else "")
                )

            uploaded_yamls = st.file_uploader(
                "上传自定义红队病例 (.yaml / .yml，支持多选)",
                type=["yaml", "yml"],
                accept_multiple_files=True,
                key="custom_yaml_uploader",
            )
            valid_uploaded: List[Tuple[BenchmarkCase, str, bytes]] = []
            if uploaded_yamls:
                for uf in uploaded_yamls:
                    raw_bytes = uf.getvalue()
                    case, err = bench_runner.validate_case_yaml_bytes(raw_bytes, source_name=uf.name)
                    if err:
                        st.error(f"❌ `{uf.name}` 校验失败：{err}")
                    elif case is not None:
                        valid_uploaded.append((case, uf.name, raw_bytes))
                        dim_cn = DIMENSION_LABELS.get(case.dimension, case.dimension)
                        st.success(f"✅ `{uf.name}` → `{case.case_id}`（{dim_cn} · {case.category}）校验通过")

            st.session_state["temp_extra_cases"] = [(c, "custom") for c, _, _ in valid_uploaded]
            if valid_uploaded:
                if st.button(
                    f"💾 保存 {len(valid_uploaded)} 个已校验病例到本地自定义目录",
                    key="save_custom_yaml_btn",
                ):
                    custom_dir.mkdir(parents=True, exist_ok=True)
                    saved_names: List[str] = []
                    for case, orig_name, raw_bytes in valid_uploaded:
                        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", case.case_id).strip("._") or Path(orig_name).stem
                        out_file = custom_dir / f"{safe_stem}.yaml"
                        out_file.write_bytes(raw_bytes)
                        saved_names.append(out_file.name)
                    st.success(
                        f"已保存 {', '.join(saved_names)} 到 `{custom_dir}`；"
                        "在「🧪 在线发起评测」勾选「包含自定义病例」即可长期复用。"
                    )

        with tab_import:
            st.caption(f"本地报告存储目录：`{results_dir}`（未保存前仅在当前会话内存中预览）")
            uploaded_report = st.file_uploader(
                "上传外部评测报告 (.json)",
                type=["json"],
                key="external_report_uploader",
            )
            if uploaded_report is not None:
                raw_bytes = uploaded_report.getvalue()
                sig = f"{uploaded_report.name}:{len(raw_bytes)}"
                payload, err = bench_runner.validate_report_json_bytes(raw_bytes)
                if err:
                    st.error(f"❌ 外部报告校验失败：{err}")
                elif payload is not None:
                    if st.session_state.get("imported_upload_sig") != sig:
                        st.session_state["imported_upload_sig"] = sig
                        st.session_state["imported_report_payload"] = payload
                        st.session_state["imported_report_filename"] = uploaded_report.name
                        st.session_state["show_imported_report"] = True
                        st.rerun()
                    st.success(
                        f"✅ 报告 `{payload.get('report_id', '-')}` 校验通过（"
                        f"{len(payload.get('runs') or [])} 组运行，{_safe_int(payload.get('case_count'), 0)} 题）。"
                    )
                    i1, i2 = st.columns(2)
                    if i1.button("📊 在看板中加载预览", key="preview_imported_report_btn", width="stretch"):
                        st.session_state["imported_report_payload"] = payload
                        st.session_state["imported_report_filename"] = uploaded_report.name
                        st.session_state["show_imported_report"] = True
                        st.rerun()
                    if i2.button("💾 保存到本地 results 目录", key="save_imported_report_btn", width="stretch"):
                        saved_path = _save_report_to_results_dir(payload, results_dir)
                        st.session_state["show_imported_report"] = False
                        st.session_state["pending_report_path"] = str(saved_path)
                        st.success(f"已保存到 `{saved_path}`")
                        st.rerun()

        with tab_run:
            if "probed_ollama_models" not in st.session_state:
                st.session_state["probed_ollama_models"] = probe_ollama_models(timeout=0.5)

            r_top1, r_top2 = st.columns([3.5, 1.2])
            with r_top2:
                st.write("")
                if st.button("🔄 探测本地 Ollama", key="refresh_ollama_btn", width="stretch"):
                    st.session_state["probed_ollama_models"] = probe_ollama_models(timeout=1.5)
                    st.rerun()

            probed_ollama = st.session_state.get("probed_ollama_models") or []
            custom_models_text = r_top1.text_input(
                "自定义模型名（LiteLLM 格式，多模型可用英文逗号分隔）",
                placeholder="例如：dashscope/qwen-plus, ollama_chat/qwen2.5:7b",
                key="live_bench_custom_models",
            )
            model_options, parsed_custom = merge_model_options(probed_ollama, custom_models_text)

            m_col1, m_col2 = st.columns([2.2, 1.8])
            with m_col1:
                selected_presets = st.multiselect(
                    "选择评测模型（已自动融合本地 Ollama 模型）",
                    options=model_options,
                    default=["deepseek/deepseek-chat"],
                    key="live_bench_models",
                )
            with m_col2:
                selected_modes = st.multiselect(
                    "评测模式（支持单模式或双模式对照）",
                    options=["bare_llm", "harness_copilot"],
                    default=["bare_llm", "harness_copilot"],
                    format_func=lambda m: MODE_LABELS.get(m, m),
                    key="live_bench_modes",
                )

            o1, o2, o3, o4 = st.columns([1.2, 1.2, 1.2, 1.4])
            mock_mode = o1.toggle("离线 Mock 模式", value=True, key="live_bench_mock_mode")
            include_private = o2.checkbox("包含私有集 (private/)", value=False, key="live_bench_include_private")
            include_custom = o3.checkbox("包含本地自定义病例", value=False, key="live_bench_include_custom")

            temp_extra_cases: List[Tuple[BenchmarkCase, str]] = st.session_state.get("temp_extra_cases") or []
            attach_extra = o4.checkbox(
                f"附带本次上传 YAML ({len(temp_extra_cases)} 题)",
                value=bool(temp_extra_cases),
                disabled=not bool(temp_extra_cases),
                key="live_bench_attach_extra",
            )
            effective_extra = temp_extra_cases if (attach_extra and temp_extra_cases) else []

            available_map: Dict[str, Tuple[BenchmarkCase, str]] = {}
            for c, vis in bench_runner.load_benchmark_cases(
                include_private=include_private,
                include_custom=include_custom,
            ):
                available_map[c.case_id] = (c, vis)
            for c, vis in effective_extra:
                available_map[c.case_id] = (c, vis)
            available_cases = list(available_map.values())

            scope_mode = st.radio(
                "题库范围",
                options=["全量题库", "按维度筛选", "勾选指定病例 ID 抽测"],
                horizontal=True,
                key="live_bench_scope_mode",
            )
            target_case_ids: Optional[List[str]] = None
            scoped_count = len(available_cases)
            if scope_mode == "按维度筛选":
                selected_dims = st.multiselect(
                    "选择参与评测的维度",
                    options=DIMENSION_ORDER,
                    default=DIMENSION_ORDER,
                    format_func=lambda d: f"{DIMENSION_LABELS.get(d, d)} ({d})",
                    key="live_bench_dims",
                )
                target_case_ids = [
                    c.case_id for c, _ in available_cases if c.dimension in set(selected_dims)
                ]
                scoped_count = len(target_case_ids)
            elif scope_mode == "勾选指定病例 ID 抽测":
                all_ids = [c.case_id for c, _ in available_cases]
                label_map = {
                    c.case_id: f"{c.case_id} · {DIMENSION_LABELS.get(c.dimension, c.dimension)} · {c.category}"
                    for c, _ in available_cases
                }
                target_case_ids = st.multiselect(
                    "勾选抽测病例 ID（建议真模型首次试跑先选 2~4 题）",
                    options=all_ids,
                    default=all_ids[: min(2, len(all_ids))],
                    format_func=lambda cid: label_map.get(cid, cid),
                    key="live_bench_case_ids",
                )
                scoped_count = len(target_case_ids)

            target_models = list(dict.fromkeys([*selected_presets, *parsed_custom]))
            total_runs_est = len(target_models) * len(selected_modes) * scoped_count
            st.caption(
                f"预计执行：**{len(target_models)}** 模型 × **{len(selected_modes)}** 模式 × "
                f"**{scoped_count}** 题 = **{total_runs_est}** 次红队会话"
                + (f" · 已探测 Ollama：{', '.join(probed_ollama)}" if probed_ollama else " · 未检测到本地 Ollama（可手动填写或忽略）")
            )
            st.caption("💡 提示：运行期间请勿切换筛选器或刷新页面，已完成题目会逐题自动落盘。")

            if st.button("🚀 开始在线评测", type="primary", key="start_live_bench_btn"):
                if not target_models:
                    st.error("请至少选择或输入 1 个评测模型。")
                elif not selected_modes:
                    st.error("请至少选择 1 种评测模式（bare_llm / harness_copilot）。")
                elif scoped_count == 0:
                    st.error("当前题库筛选范围内没有可执行病例，请调整筛选条件。")
                else:
                    progress_bar = st.progress(0.0, text="正在初始化红队评测引擎...")
                    status_line = st.empty()

                    def _on_progress(payload: Dict[str, Any]) -> None:
                        done = _safe_int(payload.get("done"), 0)
                        total = max(1, _safe_int(payload.get("total"), 1))
                        ratio = min(1.0, max(0.0, done / total))
                        m_name = str(payload.get("model") or "")
                        m_mode = str(payload.get("mode") or "")
                        cid = str(payload.get("case_id") or "")
                        title = str(payload.get("title") or "")
                        passed = bool(payload.get("passed", False))
                        lat_ms = _safe_float(payload.get("latency_ms"), 0.0)
                        rep_path = str(payload.get("report_path") or "")
                        badge = "🟢 PASS" if passed else "🔴 FAIL"
                        progress_bar.progress(
                            ratio,
                            text=f"进度 {done}/{total} ({ratio * 100:.0f}%) · {m_name} | {m_mode} · {cid}",
                        )
                        status_line.info(
                            f"[{done}/{total}] `{m_name}` · `{m_mode}` · `{cid}`（{title}）→ {badge}"
                            f"（耗时 {lat_ms:.0f} ms）  \n实时落盘：`{rep_path}`"
                        )

                    try:
                        new_report = bench_runner.run_benchmark(
                            models=target_models,
                            modes=tuple(selected_modes),
                            mock_mode=mock_mode,
                            include_private=include_private,
                            include_custom=include_custom,
                            save_report=True,
                            output_dir=str(results_dir),
                            case_ids=target_case_ids,
                            extra_cases=effective_extra or None,
                            progress_callback=_on_progress,
                        )
                        st.session_state["show_imported_report"] = False
                        st.session_state["last_run_infra_alerts"] = collect_report_infra_errors(new_report)
                        st.session_state["bench_in_memory_report"] = new_report
                        new_path = str(new_report.get("report_path") or "")
                        if new_path:
                            st.session_state["pending_report_path"] = new_path
                        st.success(f"在线评测完成：{new_report.get('report_id', '')}")
                        st.rerun()
                    except Exception as exc:
                        zh_guide, sanitized_raw = format_llm_error_zh(
                            str(exc),
                            model=target_models[0] if target_models else "",
                        )
                        st.error(f"在线评测执行失败：\n\n{zh_guide}")
                        st.caption(f"诊断详情：`{sanitized_raw}`")


def _render_export_bar(report: Dict[str, Any], runs: List[Dict[str, Any]]) -> None:
    report_id = str(report.get("report_id") or "ortho_bench_report").strip() or "ortho_bench_report"
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", report_id).strip("._") or "ortho_bench_report"
    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "📥 导出四维对比表 (.csv)",
        data=build_dimension_csv_bytes(runs),
        file_name=f"{safe_id}_dimensions.csv",
        mime="text/csv",
        key=f"export_csv_{safe_id}",
        width="stretch",
    )
    e2.download_button(
        "📝 导出翻车实录简报 (.md)",
        data=build_failure_replay_markdown(report).encode("utf-8"),
        file_name=f"{safe_id}_failure_brief.md",
        mime="text/markdown",
        key=f"export_md_{safe_id}",
        width="stretch",
    )
    e3.download_button(
        "📦 导出当前报告 (.json)",
        data=build_report_json_bytes(report),
        file_name=f"{safe_id}.json",
        mime="application/json",
        key=f"export_json_{safe_id}",
        width="stretch",
    )


def _render_report_meta_banner(report: Dict[str, Any]) -> None:
    include_private = bool(report.get("include_private", False))
    mock_badge = "Mock 仿真" if report.get("mock_mode") else "真实 LLM"
    privacy_badge = "⚠️ 含私有集 (private/)" if include_private else "🔒 私有集已隔离 (仅 public/)"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("报告 ID", str(report.get("report_id") or "-"))
    c2.metric("生成时间", str(report.get("created_at") or "-"))
    c3.metric("评测模式", mock_badge)
    c4.metric("病例数", f"{_safe_int(report.get('case_count'), 0)} 题")
    c5.metric("数据隔离", "已隔离" if not include_private else "含私有集")
    st.caption(f"隔离状态：{privacy_badge}")


def _render_kpi_comparison(runs: List[Dict[str, Any]]) -> None:
    st.subheader("📊 核心安全与防拒答对比面板")

    valid_runs = [r for r in (runs or []) if isinstance(r, dict)]
    pairs = pair_bare_and_harness(valid_runs)
    if pairs:
        for label, bare_run, harness_run in pairs:
            b_sum = bare_run.get("summary") if isinstance(bare_run.get("summary"), dict) else {}
            h_sum = harness_run.get("summary") if isinstance(harness_run.get("summary"), dict) else {}

            b_pass = _safe_float(b_sum.get("overall_pass_rate"), 0.0)
            h_pass = _safe_float(h_sum.get("overall_pass_rate"), 0.0)
            b_veto = _safe_float(b_sum.get("safety_veto_rate"), 0.0)
            h_veto = _safe_float(h_sum.get("safety_veto_rate"), 0.0)
            b_over = _safe_float(b_sum.get("over_refusal_rate"), 0.0)
            h_over = _safe_float(h_sum.get("over_refusal_rate"), 0.0)

            with st.container(border=True):
                st.markdown(f"##### 🛡️ Harness 护栏增益对比：`{label}` (`harness_copilot` vs `bare_llm`)")
                k1, k2, k3 = st.columns(3)
                k1.metric(
                    "总通过率 (Overall Pass Rate)",
                    f"{h_pass:.1f}%",
                    delta=f"{h_pass - b_pass:+.1f}% (Bare: {b_pass:.1f}%)",
                    delta_color="normal",
                )
                k2.metric(
                    "安全红线一票否决率 (Safety Veto ↓)",
                    f"{h_veto:.1f}%",
                    delta=f"{h_veto - b_veto:+.1f}% (Bare: {b_veto:.1f}%)",
                    delta_color="inverse",
                )
                k3.metric(
                    "过度拒答率 (Over-Refusal ↓)",
                    f"{h_over:.1f}%",
                    delta=f"{h_over - b_over:+.1f}% (Bare: {b_over:.1f}%)",
                    delta_color="inverse",
                )

    cols = st.columns(max(1, min(len(valid_runs), 4)))
    for idx, run in enumerate(valid_runs):
        col = cols[idx % len(cols)]
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        mode = str(run.get("mode") or "")
        mode_text = MODE_LABELS.get(mode, mode)
        pass_rate = _safe_float(summary.get("overall_pass_rate"), 0.0)
        veto_rate = _safe_float(summary.get("safety_veto_rate"), 0.0)
        over_rate = _safe_float(summary.get("over_refusal_rate"), 0.0)
        passed_cases = _safe_int(summary.get("passed_cases"), 0)
        total_cases = _safe_int(summary.get("total_cases"), 0)
        veto_count = _safe_int(summary.get("safety_veto_count"), 0)
        over_count = _safe_int(summary.get("over_refusal_count"), 0)
        over_app = _safe_int(summary.get("over_refusal_applicable"), 0)
        latency = _safe_float(run.get("total_latency_ms"), 0.0)

        with col:
            with st.container(border=True):
                icon = "🟢" if mode == "harness_copilot" else ("🔴" if mode == "bare_llm" else "🔵")
                st.markdown(f"**{icon} {run.get('run_label', f'run_{idx + 1}')}**")
                st.caption(f"模式：{mode_text} · 耗时：{latency:.1f} ms")
                m1, m2, m3 = st.columns(3)
                m1.metric("总通过率", f"{pass_rate:.1f}%", f"{passed_cases}/{total_cases} 题")
                m2.metric("安全否决率", f"{veto_rate:.1f}%", f"{veto_count} 题")
                m3.metric(
                    "过度拒答率",
                    f"{over_rate:.1f}%",
                    f"{over_count}/{over_app}",
                )


def _render_dimension_section(runs: List[Dict[str, Any]], report_id: str) -> None:
    st.markdown("#### 🧭 四维能力对比表格与可视化")
    dim_rows = build_dimension_comparison_rows(runs)
    st.dataframe(dim_rows, width="stretch", hide_index=True)

    col_radar, col_bar = st.columns(2)
    with col_radar:
        st.plotly_chart(
            build_radar_figure(runs),
            width="stretch",
            key=f"radar_chart_{report_id}",
        )
    with col_bar:
        st.plotly_chart(
            build_bar_figure(runs),
            width="stretch",
            key=f"bar_chart_{report_id}",
        )


def _render_case_result_body(res: Dict[str, Any], *, compact: bool = False) -> None:
    """Render structured diagnostics, keyword checks, and multi-turn chat replay for one case result."""
    status = _result_status(res)
    dimension = str(res.get("dimension") or "")
    dim_cn = DIMENSION_LABELS.get(dimension, dimension)
    risk_level = str(res.get("risk_level") or "-")
    expected_action = str(res.get("expected_action") or "-")
    visibility = str(res.get("visibility") or "public")

    if compact:
        st.markdown(
            f"**判定**：{STATUS_FULL_LABELS.get(status, status)} · "
            f"**期望动作**：`{expected_action}` · **风险**：`{risk_level}`"
        )
    else:
        meta_cols = st.columns(5)
        meta_cols[0].markdown(f"**判定状态**：{STATUS_FULL_LABELS.get(status, status)}")
        meta_cols[1].markdown(f"**评测维度**：{dim_cn}")
        meta_cols[2].markdown(f"**风险等级**：`{risk_level}`")
        meta_cols[3].markdown(f"**期望动作**：`{expected_action}`")
        meta_cols[4].markdown(f"**可见性**：`{visibility}`")

    failure_reasons = res.get("failure_reasons")
    if isinstance(failure_reasons, list) and failure_reasons:
        reasons_md = "\n".join(f"- {r}" for r in failure_reasons)
        st.error(f"**翻车原因汇总**\n{reasons_md}")

    if res.get("extraction_failed"):
        ext_err = str(res.get("extraction_error") or res.get("infra_error") or "未知抽取异常")
        st.error(f"🧩 **结构化抽取失败 (`extraction_failed=True`)**：`{ext_err}`")
    elif res.get("infra_error"):
        st.error(f"🔌 **基础设施异常 (`infra_error`)**：`{res.get('infra_error')}`")

    rule_check = res.get("rule_engine_check")
    if isinstance(rule_check, dict) and rule_check.get("checked") and not rule_check.get("matched", True):
        st.warning(
            f"规则引擎比对：预期 `{rule_check.get('expected_rule_ids')}` · "
            f"实测 `{rule_check.get('triggered_rule_ids')}`"
        )

    forbidden_violations = [
        v for v in (res.get("forbidden_violations") or []) if isinstance(v, dict)
    ]
    if forbidden_violations:
        st.markdown("##### 🚨 命中的安全/校准禁用正则 (`forbidden_violations`)")
        for idx, v in enumerate(forbidden_violations, start=1):
            turn_no = v.get("turn", "-")
            matched = v.get("matched_text", "")
            pattern = v.get("pattern", "")
            rationale = v.get("rationale", "")
            snippet = v.get("snippet", "")
            st.warning(
                f"**#{idx} [Turn {turn_no}] 命中片段**：`{matched}`  \n"
                f"- **正则**：`{pattern}`  \n"
                f"- **临床依据 (Rationale)**：{rationale}  \n"
                f"- **上下文摘录**：`{snippet}`"
            )

    suppressed_hits = [
        s for s in (res.get("suppressed_forbidden_hits") or []) if isinstance(s, dict)
    ]
    if suppressed_hits:
        st.markdown("##### 🛡️ 否定词保护放行审计 (`suppressed_forbidden_hits`)")
        for idx, s in enumerate(suppressed_hits, start=1):
            turn_no = s.get("turn", "-")
            matched = s.get("matched_text", "")
            neg_token = s.get("negation_token", "")
            reason = s.get("reason", "")
            st.success(
                f"**#{idx} [Turn {turn_no}] 已保护放行**：`{matched}` "
                f"（否定/劝阻锚点：`{neg_token}` · 原因：`{reason}`）"
            )

    matched_kw = res.get("must_mention_matched") if isinstance(res.get("must_mention_matched"), list) else []
    missing_kw = res.get("must_mention_missing") if isinstance(res.get("must_mention_missing"), list) else []
    kw_col1, kw_col2 = st.columns(2)
    with kw_col1:
        if matched_kw:
            st.info("✅ **已命中必提关键词 (`must_mention_matched`)**：" + "、".join(f"`{k}`" for k in matched_kw))
        else:
            st.caption("已命中必提关键词：无")
    with kw_col2:
        if missing_kw:
            st.caption("⚪ 未提及的可选/必提词 (`must_mention_missing`)：" + "、".join(f"`{k}`" for k in missing_kw))
        else:
            st.caption("未提及关键词：无")

    numeric_checks = res.get("numeric_target_checks")
    if isinstance(numeric_checks, dict) and numeric_checks:
        st.markdown("##### 🔢 确定性数值核验 (`numeric_target_checks`)")
        st.json(numeric_checks)

    st.markdown("##### 💬 多轮红队对话回放")
    turns = [t for t in (res.get("turns") or []) if isinstance(t, dict)]
    if not turns:
        st.caption("无对话轮次记录。")
        return

    for turn_item in turns:
        turn_no = turn_item.get("turn")
        user_msg = str(turn_item.get("user_msg") or "")
        assistant_msg = str(turn_item.get("assistant_msg") or "")
        latency_ms = _safe_float(turn_item.get("latency_ms"), 0.0)

        with st.chat_message("user"):
            st.caption(f"Turn {turn_no} · 红队患者/临床提问")
            st.markdown(user_msg)

        turn_violations = [
            v for v in forbidden_violations if v.get("turn") is None or v.get("turn") == turn_no
        ]
        turn_suppressed = [
            s for s in suppressed_hits if s.get("turn") is None or s.get("turn") == turn_no
        ]

        with st.chat_message("assistant"):
            st.caption(f"Turn {turn_no} · 模型回复（耗时 `{latency_ms:.1f} ms`）")
            rendered_html = highlight_violations_html(
                assistant_msg,
                turn_violations,
                turn_suppressed,
            )
            st.markdown(rendered_html, unsafe_allow_html=True)


def _render_case_replay_card(record: Dict[str, Any], *, expanded: bool) -> None:
    res = record["result"]
    run_label = record["run_label"]
    case_id = str(res.get("case_id") or "UNKNOWN_CASE")
    status = _result_status(res)
    dimension = str(res.get("dimension") or "")
    dim_cn = DIMENSION_LABELS.get(dimension, dimension)
    category = str(res.get("category") or "-")
    badge = STATUS_BADGES.get(status, status)

    title = f"{badge} | `{case_id}` | {run_label} | {dim_cn} · {category}"
    with st.expander(title, expanded=expanded):
        _render_case_result_body(res, compact=False)


def _render_side_by_side_card(pair_item: Dict[str, Any]) -> None:
    cid = pair_item["case_id"]
    dim_cn = DIMENSION_LABELS.get(pair_item["dimension"], pair_item["dimension"])
    category = pair_item["category"] or "-"
    pair_label = pair_item["pair_label"]
    b_res = pair_item["bare_result"]
    h_res = pair_item["harness_result"]
    b_badge = STATUS_BADGES.get(_result_status(b_res), "-") if b_res else "⚪ 无记录"
    h_badge = STATUS_BADGES.get(_result_status(h_res), "-") if h_res else "⚪ 无记录"

    with st.container(border=True):
        top_l, top_r = st.columns([3.2, 1.8])
        top_l.markdown(
            f"#### `{cid}` · `{pair_label}`  \n"
            f"{dim_cn} · {category} · 风险 `{pair_item['risk_level']}` · 期望动作 `{pair_item['expected_action']}`"
        )
        if pair_item["rescued"]:
            top_r.success(f"✨ 护栏救回：{b_badge} → {h_badge}")
        elif pair_item["regressed"]:
            top_r.error(f"⚠️ 护栏回退：{b_badge} → {h_badge}")
        else:
            top_r.info(f"对照结果：{b_badge} vs {h_badge}")

        col_bare, col_harness = st.columns(2)
        with col_bare:
            st.markdown(f"##### 🔴 裸模型 (`{pair_item['bare_run_label']}`)")
            if b_res is None:
                st.caption("当前模型缺少 `bare_llm` 运行结果。")
            else:
                _render_case_result_body(b_res, compact=True)
        with col_harness:
            st.markdown(f"##### 🟢 护栏协同 (`{pair_item['harness_run_label']}`)")
            if h_res is None:
                st.caption("当前模型缺少 `harness_copilot` 运行结果。")
            else:
                _render_case_result_body(h_res, compact=True)


def _render_failure_replay(runs: List[Dict[str, Any]]) -> None:
    st.subheader("🎬 “翻车实录”多轮对话回放器")

    valid_runs = [r for r in (runs or []) if isinstance(r, dict)]
    run_labels = [str(r.get("run_label") or f"run_{i + 1}") for i, r in enumerate(valid_runs)]
    all_categories = sorted(
        {
            str(res.get("category") or "")
            for r in valid_runs
            for res in (r.get("results") or [])
            if isinstance(res, dict) and res.get("category")
        }
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        selected_run = st.selectbox(
            "运行项（模型 / 模式）",
            options=["全部"] + run_labels,
            key="replay_filter_run",
        )
    with f2:
        selected_status = st.selectbox(
            "判定状态",
            options=STATUS_OPTIONS,
            key="replay_filter_status",
        )
    with f3:
        selected_category = st.selectbox(
            "病例分类 (Category)",
            options=["全部"] + all_categories,
            key="replay_filter_category",
        )
    with f4:
        selected_dimension = st.selectbox(
            "评测维度 (Dimension)",
            options=["全部"] + DIMENSION_ORDER,
            format_func=lambda d: "全部" if d == "全部" else f"{DIMENSION_LABELS.get(d, d)} ({d})",
            key="replay_filter_dimension",
        )

    e1, e2, e3 = st.columns([2.0, 1.5, 1.7])
    with e1:
        search_query = st.text_input(
            "🔍 搜索病例 ID / 关键词 / 翻车原因",
            placeholder="例如：ORTHO_REDFLAG_001、踩棉花、华法林...",
            key="replay_search_query",
        )
    with e2:
        st.write("")
        only_rescued = st.toggle(
            "✨ 仅看护栏救回（Bare 翻车 → Harness 通过）",
            value=False,
            key="replay_only_rescued",
        )
    with e3:
        view_mode = st.radio(
            "回放视图",
            options=["单列明细", "左右双栏同题对照 (Bare vs Harness)"],
            horizontal=True,
            key="replay_view_mode",
        )

    if view_mode == "左右双栏同题对照 (Bare vs Harness)":
        selected_pair = "全部"
        if selected_run != "全部":
            selected_pair = selected_run.split(" | ", 1)[0].strip() or "全部"
        pair_items = build_side_by_side_case_pairs(
            runs,
            selected_pair=selected_pair,
            selected_status=selected_status,
            selected_category=selected_category,
            selected_dimension=selected_dimension,
            search_query=search_query,
            only_rescued=only_rescued,
        )
        rescued_cnt = sum(1 for p in pair_items if p["rescued"])
        st.caption(
            f"左右同题对照共匹配 **{len(pair_items)}** 组病例，其中护栏成功救回 **{rescued_cnt}** 组。"
        )
        if not pair_items:
            st.info("当前筛选条件下暂无可双栏对照的病例（需同模型具备 bare_llm 与 harness_copilot 双模式结果）。")
            return
        for pair_item in pair_items:
            _render_side_by_side_card(pair_item)
        return

    filtered = filter_case_results(
        runs,
        selected_run=selected_run,
        selected_status=selected_status,
        selected_category=selected_category,
        selected_dimension=selected_dimension,
        search_query=search_query,
        only_rescued=only_rescued,
    )

    fail_count = sum(1 for item in filtered if not bool(item["result"].get("passed", False)))
    st.caption(f"当前筛选命中 **{len(filtered)}** 条评测记录，其中翻车 **{fail_count}** 条（失败用例自动置顶并展开）。")

    if not filtered:
        st.info("当前筛选条件下暂无匹配病例，请调整上方筛选条件。")
        return

    expanded_failures = 0
    for record in filtered:
        is_failed = not bool(record["result"].get("passed", False))
        should_expand = False
        if is_failed and expanded_failures < 5:
            should_expand = True
            expanded_failures += 1
        _render_case_replay_card(record, expanded=should_expand)


def main() -> None:
    st.set_page_config(
        page_title="OrthoRehab 红队评测沙箱与翻车实录看板",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ OrthoRehab 红队评测沙箱与“翻车实录”看板")
    st.caption("Bare LLM vs Harness Copilot 四维安全拦截、记忆准确性与防过度拒答对比分析")

    results_dir = get_results_dir()
    active_report, selected_path = _render_top_controls(results_dir)
    _render_sidebar(active_report, selected_path, results_dir)
    _render_infra_alerts_if_any()
    _render_workbench_tabs(results_dir)

    if active_report is None:
        st.info(
            f"👆 请先点击上方 **「🚀 运行 Mock 快速评测」**、在 **「🧪 在线发起评测」** 发起评测，"
            f"或上传外部报告 JSON / 将报告放入 `{results_dir}` 目录。"
        )
        return

    _render_report_meta_banner(active_report)
    runs = active_report.get("runs") or []
    _render_export_bar(active_report, runs)
    st.divider()

    if not runs:
        st.warning("当前报告中未包含任何 `runs` 运行记录。")
        return

    _render_kpi_comparison(runs)
    st.divider()
    _render_dimension_section(runs, str(active_report.get("report_id") or "default"))
    st.divider()
    _render_failure_replay(runs)


if __name__ == "__main__":
    main()
