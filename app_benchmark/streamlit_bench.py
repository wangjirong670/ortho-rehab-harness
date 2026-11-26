"""Streamlit dashboard for Week 4 red-team benchmark comparison and failure replay."""

from __future__ import annotations

import html
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.graph_objects as go
import streamlit as st

from app_benchmark import runner as bench_runner

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "app_benchmark" / "results"

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

FALLBACK_PALETTE: List[str] = [
    "#2563EB",
    "#D97706",
    "#7C3AED",
    "#0891B2",
    "#4F46E5",
]


def get_results_dir() -> Path:
    """Return the benchmark results directory, honoring test and frozen-app overrides."""
    override = os.getenv("ORTHO_BENCH_RESULTS_DIR", "").strip()
    if override:
        return Path(override)
    env_results = os.getenv("ORTHO_RESULTS_DIR", "").strip()
    if env_results:
        return Path(env_results)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "results"
    return DEFAULT_RESULTS_DIR


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
    """Build 4-dimension comparison rows with Chinese names and pass rates."""
    valid_runs = [r for r in (runs or []) if isinstance(r, dict)]
    pairs = pair_bare_and_harness(valid_runs)
    primary_pair = pairs[0] if len(pairs) == 1 else None

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

        if primary_pair is not None:
            _, bare_run, harness_run = primary_pair
            bare_rate = _extract_dim_entry(bare_run, dim_key)["pass_rate"]
            harness_rate = _extract_dim_entry(harness_run, dim_key)["pass_rate"]
            delta = harness_rate - bare_rate
            row["护栏增益 (Δ)"] = f"{delta:+.1f}%"

        rows.append(row)
    return rows


def _run_visual_style(run: Dict[str, Any], index: int) -> Tuple[str, str, str]:
    mode = str(run.get("mode") or "")
    if mode == "bare_llm":
        return ("#E11D48", "rgba(225, 29, 72, 0.14)", "dash")
    if mode == "harness_copilot":
        return ("#059669", "rgba(5, 150, 105, 0.16)", "solid")
    color = FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)]
    return (color, "rgba(37, 99, 235, 0.12)", "solid")


def build_radar_figure(runs: List[Dict[str, Any]]) -> go.Figure:
    """Build a 4-dimension Plotly Scatterpolar radar chart comparing runs."""
    fig = go.Figure()
    theta_labels = [DIMENSION_LABELS[d] for d in DIMENSION_ORDER]
    closed_theta = theta_labels + [theta_labels[0]]

    for idx, run in enumerate(runs or []):
        if not isinstance(run, dict):
            continue
        run_label = str(run.get("run_label") or f"run_{idx + 1}")
        rates = [_extract_dim_entry(run, d)["pass_rate"] for d in DIMENSION_ORDER]
        closed_rates = rates + [rates[0]]
        line_color, fill_color, dash_style = _run_visual_style(run, idx)

        fig.add_trace(
            go.Scatterpolar(
                r=closed_rates,
                theta=closed_theta,
                name=run_label,
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=line_color, width=2.5, dash=dash_style),
                marker=dict(size=6, color=line_color),
                hovertemplate="%{theta}: %{r:.1f}%<extra>" + html.escape(run_label) + "</extra>",
            )
        )

    fig.update_layout(
        title=dict(text="四维评测通过率雷达图 (Pass Rate %)", x=0.02),
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
    """Build a grouped Plotly Bar chart comparing 4-dimension pass rates."""
    fig = go.Figure()
    x_labels = [DIMENSION_LABELS[d] for d in DIMENSION_ORDER]

    for idx, run in enumerate(runs or []):
        if not isinstance(run, dict):
            continue
        run_label = str(run.get("run_label") or f"run_{idx + 1}")
        rates = [_extract_dim_entry(run, d)["pass_rate"] for d in DIMENSION_ORDER]
        line_color, _, _ = _run_visual_style(run, idx)

        fig.add_trace(
            go.Bar(
                name=run_label,
                x=x_labels,
                y=rates,
                text=[f"{r:.1f}%" for r in rates],
                textposition="outside",
                marker_color=line_color,
                hovertemplate="%{x}: %{y:.1f}%<extra>" + html.escape(run_label) + "</extra>",
            )
        )

    fig.update_layout(
        title=dict(text="四维评测通过率分组柱状图", x=0.02),
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


def filter_case_results(
    runs: List[Dict[str, Any]],
    *,
    selected_run: str = "全部",
    selected_status: str = "全部",
    selected_category: str = "全部",
    selected_dimension: str = "全部",
) -> List[Dict[str, Any]]:
    """Flatten and filter case results across benchmark runs, failures first."""
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
            status = str(res.get("status") or ("PASS" if res.get("passed") else "FAIL_SAFETY"))
            passed = bool(res.get("passed", status == "PASS"))
            category = str(res.get("category") or "")
            dimension = str(res.get("dimension") or "")

            if selected_status == "仅翻车" and passed:
                continue
            if selected_status not in ("全部", "仅翻车") and status != selected_status:
                continue
            if selected_category != "全部" and category != selected_category:
                continue
            if selected_dimension != "全部" and dimension != selected_dimension:
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


def _render_sidebar(active_report: Optional[Dict[str, Any]], selected_path: Optional[Path]) -> None:
    with st.sidebar:
        st.header("🛡️ 红队沙箱信息")
        if active_report is None:
            st.info("暂无已加载的评测报告。请在主区域点击 **「🚀 运行 Mock 快速评测」** 生成首份报告。")
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
            st.caption(f"文件路径：`{selected_path.name}`")
        elif active_report.get("report_path"):
            st.caption(f"文件路径：`{Path(str(active_report['report_path'])).name}`")

        st.divider()
        st.markdown("#### 维度说明")
        for dim_key in DIMENSION_ORDER:
            st.caption(f"• **{DIMENSION_LABELS[dim_key]}** (`{dim_key}`)")


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
                run_kwargs: Dict[str, Any] = {"mock_mode": True, "save_report": True}
                if (
                    os.getenv("ORTHO_BENCH_RESULTS_DIR", "").strip()
                    or os.getenv("ORTHO_RESULTS_DIR", "").strip()
                    or getattr(sys, "frozen", False)
                ):
                    run_kwargs["output_dir"] = str(results_dir)
                new_report = bench_runner.run_benchmark(**run_kwargs)
            st.session_state["bench_in_memory_report"] = new_report
            new_path = str(new_report.get("report_path") or "")
            if new_path:
                st.session_state["pending_report_path"] = new_path
            st.success(f"Mock 快速评测完成：{new_report.get('report_id', '')}")
            st.rerun()
        except Exception as exc:
            st.error(f"Mock 快速评测执行失败：{exc}")

    active_report: Optional[Dict[str, Any]] = None
    if selected_path is not None:
        active_report = load_report_file(selected_path)
    if active_report is None:
        active_report = st.session_state.get("bench_in_memory_report")
    return active_report, selected_path


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


def _render_case_replay_card(record: Dict[str, Any], *, expanded: bool) -> None:
    res = record["result"]
    run_label = record["run_label"]
    case_id = str(res.get("case_id") or "UNKNOWN_CASE")
    status = str(res.get("status") or ("PASS" if res.get("passed") else "FAIL_SAFETY"))
    dimension = str(res.get("dimension") or "")
    dim_cn = DIMENSION_LABELS.get(dimension, dimension)
    category = str(res.get("category") or "-")
    risk_level = str(res.get("risk_level") or "-")
    expected_action = str(res.get("expected_action") or "-")
    visibility = str(res.get("visibility") or "public")
    badge = STATUS_BADGES.get(status, status)

    title = f"{badge} | `{case_id}` | {run_label} | {dim_cn} · {category}"
    with st.expander(title, expanded=expanded):
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

    filtered = filter_case_results(
        runs,
        selected_run=selected_run,
        selected_status=selected_status,
        selected_category=selected_category,
        selected_dimension=selected_dimension,
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
    _render_sidebar(active_report, selected_path)

    if active_report is None:
        st.info("👆 请先点击上方 **「🚀 运行 Mock 快速评测」** 生成对比报告，或将评测 JSON 放入 `app_benchmark/results/` 目录。")
        return

    _render_report_meta_banner(active_report)
    st.divider()

    runs = active_report.get("runs") or []
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
