"""Pure business logic and report-processing service for OrthoRehab red-team benchmark."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app_benchmark import runner as bench_runner
from core_tools.llm_client import format_llm_error_zh

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


def extract_violation_spans(
    text: str,
    result: Optional[Dict[str, Any]],
    *,
    turn: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return structured violation spans [{'start', 'end', 'text', 'reason'}] for safe React highlighting."""
    raw = text or ""
    if not raw or not isinstance(result, dict):
        return []

    raw_violations = result.get("forbidden_violations") or []
    candidates: List[Tuple[str, str]] = []
    for item in raw_violations:
        if not isinstance(item, dict):
            continue
        if turn is not None and item.get("turn") is not None and item.get("turn") != turn:
            continue
        matched = str(item.get("matched_text") or "").strip()
        if not matched:
            continue
        reason = str(item.get("rationale") or item.get("reason") or item.get("pattern") or "命中禁用违规片段").strip()
        candidates.append((matched, reason))

    if not candidates:
        return []

    candidates.sort(key=lambda pair: -len(pair[0]))

    spans: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []
    for phrase, reason in candidates:
        start = 0
        while True:
            idx = raw.find(phrase, start)
            if idx == -1:
                break
            end = idx + len(phrase)
            overlaps = any(not (end <= s_start or idx >= s_end) for s_start, s_end in occupied)
            if not overlaps:
                occupied.append((idx, end))
                spans.append(
                    {
                        "start": idx,
                        "end": end,
                        "text": raw[idx:end],
                        "reason": reason,
                    }
                )
            start = end

    spans.sort(key=lambda s: s["start"])
    return spans


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
            rescued = b_res is not None and h_res is not None and not b_passed and h_passed
            regressed = b_res is not None and h_res is not None and b_passed and not h_passed
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


def save_report_to_results_dir(report: Dict[str, Any], results_dir: Optional[Path] = None) -> Path:
    """Persist an imported benchmark report JSON safely into results_dir without overwriting existing files."""
    target_dir = results_dir or get_results_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_id = str(report.get("report_id") or "imported_report").strip()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_id).strip("._") or "imported_report"
    out_path = target_dir / f"{safe_stem}.json"
    suffix = 1
    while out_path.exists():
        out_path = target_dir / f"{safe_stem}_{suffix}.json"
        suffix += 1
    payload = dict(report)
    payload["report_path"] = str(out_path)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def list_reports_metadata(results_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return report summary metadata list for GET /api/v1/benchmark/reports."""
    items: List[Dict[str, Any]] = []
    for path in scan_report_files(results_dir):
        data = load_report_file(path)
        if not data:
            continue
        models: List[str] = []
        seen_models: Set[str] = set()
        for run in data.get("runs") or []:
            if isinstance(run, dict) and run.get("model"):
                m = str(run["model"])
                if m not in seen_models:
                    seen_models.add(m)
                    models.append(m)
        items.append(
            {
                "filename": path.name,
                "label": format_report_option(path),
                "created_at": str(data.get("created_at") or "-"),
                "mock_mode": bool(data.get("mock_mode", False)),
                "case_count": _safe_int(data.get("case_count"), 0),
                "include_private": bool(data.get("include_private", False)),
                "models": models,
            }
        )
    return items


def _build_turn_highlights(result: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Build {<turn_idx_str>: [span]} dictionary for all turns of a case result."""
    if not isinstance(result, dict):
        return {}
    highlights: Dict[str, List[Dict[str, Any]]] = {}
    turns = result.get("turns") or []
    for idx, turn_item in enumerate(turns):
        if not isinstance(turn_item, dict):
            continue
        turn_no = turn_item.get("turn")
        text = str(turn_item.get("assistant_msg") or "")
        highlights[str(idx)] = extract_violation_spans(text, result, turn=turn_no)
    return highlights


def build_report_detail_payload(filename: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble full structured benchmark report detail matching the REST API contract."""
    runs = [r for r in (report.get("runs") or []) if isinstance(r, dict)]
    raw_pairs = pair_bare_and_harness(runs)

    pairs_payload: List[Dict[str, Any]] = []
    for pair_label, bare_run, harness_run in raw_pairs:
        b_sum = bare_run.get("summary") if isinstance(bare_run.get("summary"), dict) else {}
        h_sum = harness_run.get("summary") if isinstance(harness_run.get("summary"), dict) else {}

        b_pass = _safe_float(b_sum.get("overall_pass_rate"), 0.0)
        h_pass = _safe_float(h_sum.get("overall_pass_rate"), 0.0)
        b_veto = _safe_float(b_sum.get("safety_veto_rate"), 0.0)
        h_veto = _safe_float(h_sum.get("safety_veto_rate"), 0.0)

        deltas: Dict[str, float] = {
            "overall_pass_rate": round(h_pass - b_pass, 2),
            "safety_veto_rate": round(h_veto - b_veto, 2),
        }
        for dim_key in DIMENSION_ORDER:
            b_dim = _extract_dim_entry(bare_run, dim_key)["pass_rate"]
            h_dim = _extract_dim_entry(harness_run, dim_key)["pass_rate"]
            deltas[dim_key] = round(h_dim - b_dim, 2)

        pairs_payload.append(
            {
                "pair_label": pair_label,
                "bare_run_label": str(bare_run.get("run_label") or f"{pair_label} | bare_llm"),
                "harness_run_label": str(harness_run.get("run_label") or f"{pair_label} | harness_copilot"),
                "bare_summary": b_sum,
                "harness_summary": h_sum,
                "deltas": deltas,
            }
        )

    raw_sbs = build_side_by_side_case_pairs(runs)
    side_by_side_cases: List[Dict[str, Any]] = []
    for item in raw_sbs:
        b_res = item.get("bare_result") if isinstance(item.get("bare_result"), dict) else {}
        h_res = item.get("harness_result") if isinstance(item.get("harness_result"), dict) else {}
        dim = str(item.get("dimension") or "")
        side_by_side_cases.append(
            {
                "pair_label": str(item.get("pair_label") or ""),
                "case_id": str(item.get("case_id") or ""),
                "category": str(item.get("category") or ""),
                "dimension": dim,
                "dimension_label": DIMENSION_LABELS.get(dim, dim),
                "risk_level": str(item.get("risk_level") or "-"),
                "expected_action": str(item.get("expected_action") or "-"),
                "visibility": str(item.get("visibility") or "public"),
                "bare_run_label": str(item.get("bare_run_label") or ""),
                "harness_run_label": str(item.get("harness_run_label") or ""),
                "rescued": bool(item.get("rescued")),
                "regressed": bool(item.get("regressed")),
                "any_failed": bool(item.get("any_failed")),
                "bare_result": b_res,
                "harness_result": h_res,
                "bare_highlights": _build_turn_highlights(b_res),
                "harness_highlights": _build_turn_highlights(h_res),
            }
        )

    infra_errors: List[Dict[str, str]] = []
    for model_name, raw_err in collect_report_infra_errors(report):
        zh_guide, sanitized_raw = format_llm_error_zh(raw_err, model=model_name)
        infra_errors.append(
            {
                "model": model_name,
                "error": sanitized_raw,
                "guide": zh_guide,
            }
        )

    return {
        "filename": filename,
        "meta": {
            "created_at": str(report.get("created_at") or "-"),
            "mock_mode": bool(report.get("mock_mode", False)),
            "case_count": _safe_int(report.get("case_count"), 0),
            "include_private": bool(report.get("include_private", False)),
        },
        "runs": runs,
        "pairs": pairs_payload,
        "dimension_rows": build_dimension_comparison_rows(runs),
        "side_by_side_cases": side_by_side_cases,
        "infra_errors": infra_errors,
    }
