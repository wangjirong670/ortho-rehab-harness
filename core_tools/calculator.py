"""Deterministic clinical scale calculator (Week 2).

Reads scale specifications from knowledge_base/scales_spec.yaml and computes
exact totals, subscales, normalized scores, severity bands, JOA improvement rates,
and clinical red-flag linkage.

Hard rule: NEVER silently default missing or invalid fields to 0.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core_tools.schemas import ScaleItem, ScaleSpec, ScalesCatalog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCALES_PATH = PROJECT_ROOT / "knowledge_base" / "scales_spec.yaml"

# Cervical JOA 17-point integer thresholds: <=12 triggers critical red-flag; 13-16 triggers caution linkage
JOA_CERVICAL_RED_FLAG_THRESHOLD = 12
JOA_CERVICAL_MILD_THRESHOLD = 16


def _is_valid_number(value: Any) -> bool:
    """Return True strictly for finite int/float values (explicitly rejecting bool)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _map_score_direction(yaml_direction: str) -> str:
    """Normalize YAML direction into UI-facing score_direction ('higher_is_worse' | 'lower_is_worse')."""
    if yaml_direction == "higher_is_worse":
        return "higher_is_worse"
    if yaml_direction == "higher_is_better":
        return "lower_is_worse"
    raise ValueError(f"Unsupported scale direction: {yaml_direction}")


@lru_cache(maxsize=4)
def load_scales_catalog(path: Optional[str] = None) -> ScalesCatalog:
    """Load and validate knowledge_base/scales_spec.yaml with caching."""
    target = Path(path) if path else DEFAULT_SCALES_PATH
    with target.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ScalesCatalog.model_validate(raw)


def list_available_scales(scales_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return metadata for all configured clinical scales."""
    catalog = load_scales_catalog(scales_path)
    return [
        {
            "scale_id": spec.scale_id,
            "name": spec.name,
            "version_note": spec.version_note,
            "score_direction": _map_score_direction(spec.direction),
            "min_total": spec.min_total,
            "max_total": spec.max_total,
            "item_count": len(spec.items),
            "required_fields": [item.item_id for item in spec.items if item.required],
        }
        for spec in catalog.scales
    ]


def _validate_item_value(item: ScaleItem, raw_val: Any) -> Optional[str]:
    """Validate a single item value; return an error description or None if valid."""
    if not _is_valid_number(raw_val):
        return (
            f"字段 '{item.item_id}' ({item.label}) 类型非法: "
            f"期望有限数值 (int/float)，实际收到 {type(raw_val).__name__} ({raw_val!r})"
        )

    val = float(raw_val)
    if val < item.min_value - 1e-9 or val > item.max_value + 1e-9:
        return (
            f"字段 '{item.item_id}' ({item.label}) 数值越界: "
            f"收到 {val:g}，允许范围为 [{item.min_value:g}, {item.max_value:g}]"
        )

    if item.allowed_values is not None:
        if not any(abs(val - allowed) <= 1e-6 for allowed in item.allowed_values):
            return (
                f"字段 '{item.item_id}' ({item.label}) 取值非法: "
                f"收到 {val:g}，仅允许离散值 {item.allowed_values}"
            )
    elif item.step is not None:
        offset = (val - item.min_value) / item.step
        if abs(offset - round(offset)) > 1e-6:
            return (
                f"字段 '{item.item_id}' ({item.label}) 不符合步长要求: "
                f"收到 {val:g}，步长应为 {item.step:g}"
            )

    return None


def _match_severity_band(spec: ScaleSpec, total_score: float) -> str:
    """Match a total score to its configured severity band."""
    for band in spec.severity_bands:
        if band.min_score - 1e-6 <= total_score <= band.max_score + 1e-6:
            return band.label
    # Handle tiny floating-point gaps between discrete band endpoints (e.g. VAS 3.05)
    sorted_bands = sorted(spec.severity_bands, key=lambda b: b.min_score)
    for band in sorted_bands:
        if total_score <= band.max_score + 1e-6:
            return band.label
    return sorted_bands[-1].label


def calculate_joa_improvement_rate(
    pre_score: Any,
    post_score: Any,
    max_score: float = 17.0,
    min_score: float = 0.0,
    step: Optional[float] = 1.0,
) -> Dict[str, Any]:
    """Compute Hirabayashi JOA improvement rate with strict type, range, step, and zero-division protection.

    Formula: (post_score - pre_score) / (max_score - pre_score) * 100
    """
    if not _is_valid_number(pre_score) or not _is_valid_number(post_score):
        return {
            "status": "error",
            "error_code": "INVALID_TYPE",
            "message": "术前评分 (pre_score) 与术后评分 (post_score) 必须为有限数值。",
            "improvement_rate_pct": None,
        }

    pre = float(pre_score)
    post = float(post_score)

    if pre < min_score - 1e-9 or pre > max_score + 1e-9:
        return {
            "status": "error",
            "error_code": "OUT_OF_RANGE",
            "message": f"术前评分 {pre:g} 超出合法范围 [{min_score:g}, {max_score:g}]。",
            "improvement_rate_pct": None,
        }
    if post < min_score - 1e-9 or post > max_score + 1e-9:
        return {
            "status": "error",
            "error_code": "OUT_OF_RANGE",
            "message": f"术后评分 {post:g} 超出合法范围 [{min_score:g}, {max_score:g}]。",
            "improvement_rate_pct": None,
        }

    if step is not None and step > 0:
        pre_offset = (pre - min_score) / step
        if abs(pre_offset - round(pre_offset)) > 1e-6:
            return {
                "status": "error",
                "error_code": "INVALID_STEP",
                "message": f"术前评分 (pre_score) {pre:g} 不符合步长要求（步长应为 {step:g}）。",
                "improvement_rate_pct": None,
            }
        post_offset = (post - min_score) / step
        if abs(post_offset - round(post_offset)) > 1e-6:
            return {
                "status": "error",
                "error_code": "INVALID_STEP",
                "message": f"术后评分 (post_score) {post:g} 不符合步长要求（步长应为 {step:g}）。",
                "improvement_rate_pct": None,
            }

    denominator = max_score - pre
    if abs(denominator) <= 1e-9:
        return {
            "status": "error",
            "error_code": "ZERO_DIVISION_PROTECTED",
            "message": (
                f"术前评分已达满分 ({max_score:g} 分)，分母 ({max_score:g} - pre_score) 为 0，"
                "无法计算平林法改善率（已触发除零保护）。"
            ),
            "pre_score": pre,
            "post_score": post,
            "max_score": max_score,
            "improvement_rate_pct": None,
        }

    rate = round((post - pre) / denominator * 100.0, 2)
    if rate >= 75.0:
        grade = "优 (≥75%)"
    elif rate >= 50.0:
        grade = "良 (50%-74.9%)"
    elif rate >= 25.0:
        grade = "可 (25%-49.9%)"
    else:
        grade = "差 (<25%)"

    return {
        "status": "ok",
        "pre_score": pre,
        "post_score": post,
        "max_score": max_score,
        "improvement_rate_pct": rate,
        "efficacy_grade": grade,
        "formula": f"({post:g} - {pre:g}) / ({max_score:g} - {pre:g}) * 100%",
    }


def calculate_scale(
    scale_id: str,
    params: Dict[str, Any],
    *,
    scales_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministically calculate a clinical scale score and interpretation.

    Args:
        scale_id: Configured scale identifier (e.g., 'VAS_PAIN', 'WOMAC_LIKERT_V3_1',
            'JOA_CERVICAL_17', 'JOA_LUMBAR_29').
        params: Mapping of item_id -> numeric value. Optional 'pre_score' may be supplied
            for JOA scales to also compute Hirabayashi improvement rate.
        scales_path: Optional custom path to scales_spec.yaml.

    Returns:
        Structured dict with status ('ok' | 'error'), score_direction, exact scores,
        severity band, and any red-flag linkage.
    """
    catalog = load_scales_catalog(scales_path)
    spec_map = {s.scale_id: s for s in catalog.scales}

    if scale_id not in spec_map:
        return {
            "status": "error",
            "scale_id": scale_id,
            "score_direction": None,
            "error_code": "UNKNOWN_SCALE",
            "message": f"未知量表 ID '{scale_id}'。可用量表: {sorted(spec_map.keys())}",
            "missing_fields": [],
            "invalid_fields": [],
        }

    spec = spec_map[scale_id]
    score_direction = _map_score_direction(spec.direction)

    if not isinstance(params, dict):
        return {
            "status": "error",
            "scale_id": spec.scale_id,
            "scale_name": spec.name,
            "score_direction": score_direction,
            "error_code": "INVALID_PARAMS_TYPE",
            "message": f"params 必须为 dict，实际收到 {type(params).__name__}",
            "missing_fields": [item.item_id for item in spec.items if item.required],
            "invalid_fields": [],
        }

    missing_fields: List[str] = []
    invalid_fields: List[Dict[str, str]] = []
    validated_values: Dict[str, float] = {}

    allowed_meta_keys = set()
    if spec.improvement_rate_formula:
        allowed_meta_keys.add("pre_score")

    known_item_ids = {item.item_id for item in spec.items}
    unknown_fields = sorted(
        key for key in params.keys() if key not in known_item_ids and key not in allowed_meta_keys
    )

    for item in spec.items:
        if item.item_id not in params or params[item.item_id] is None:
            if item.required:
                missing_fields.append(item.item_id)
            continue

        err_msg = _validate_item_value(item, params[item.item_id])
        if err_msg:
            invalid_fields.append({"field": item.item_id, "reason": err_msg})
        else:
            validated_values[item.item_id] = float(params[item.item_id])

    if spec.improvement_rate_formula and "pre_score" in params and params["pre_score"] is not None:
        pre_spec = ScaleItem(
            item_id="pre_score",
            label="术前基线总分",
            min_value=spec.min_total,
            max_value=spec.max_total,
            step=1.0,
            required=False,
        )
        pre_err = _validate_item_value(pre_spec, params["pre_score"])
        if pre_err:
            invalid_fields.append({"field": "pre_score", "reason": pre_err})

    if missing_fields or invalid_fields:
        msg_parts: List[str] = []
        if missing_fields:
            msg_parts.append(f"缺失必填项 ({len(missing_fields)} 项): {', '.join(missing_fields)}")
        if invalid_fields:
            msg_parts.append(
                f"字段取值或类型错误 ({len(invalid_fields)} 项): "
                + "; ".join(f["reason"] for f in invalid_fields)
            )
        return {
            "status": "error",
            "scale_id": spec.scale_id,
            "scale_name": spec.name,
            "score_direction": score_direction,
            "error_code": "VALIDATION_ERROR",
            "message": "；".join(msg_parts) + "。（严格校验模式：禁止缺失项默认补 0）",
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields,
            "unknown_fields": unknown_fields,
        }

    total_score = round(sum(validated_values.values()), 2)
    if total_score.is_integer():
        total_score_out: Any = int(total_score)
    else:
        total_score_out = total_score

    severity_band = _match_severity_band(spec, total_score)

    # Subscale breakdown
    subscale_scores: Dict[str, Dict[str, Any]] = {}
    if spec.subscales:
        for sub in spec.subscales:
            sub_val = round(
                sum(validated_values[item.item_id] for item in spec.items if item.subscale_id == sub.subscale_id),
                2,
            )
            sub_entry: Dict[str, Any] = {
                "label": sub.label,
                "score": int(sub_val) if sub_val.is_integer() else sub_val,
                "min_score": sub.min_score,
                "max_score": sub.max_score,
            }
            if sub.max_score > 0 and sub.min_score == 0:
                sub_entry["normalized_pct"] = round(sub_val / sub.max_score * 100.0, 2)
            subscale_scores[sub.subscale_id] = sub_entry

    result: Dict[str, Any] = {
        "status": "ok",
        "scale_id": spec.scale_id,
        "scale_name": spec.name,
        "version_note": spec.version_note,
        "score_direction": score_direction,
        "total_score": total_score_out,
        "min_total": spec.min_total,
        "max_total": spec.max_total,
        "severity_band": severity_band,
        "item_scores": validated_values,
        "red_flag_triggered": False,
        "red_flag_reason": None,
        "caution_triggered": False,
        "caution_reason": None,
        "linked_rule_ids": [],
        "references": list(spec.references),
    }

    if subscale_scores:
        result["subscale_scores"] = subscale_scores
    if unknown_fields:
        result["unknown_fields"] = unknown_fields

    # Scale-specific deterministic enrichments
    if spec.formula_type == "subscale_sum_with_normalized_pct":
        span = spec.max_total - spec.min_total
        result["normalized_score_pct"] = round((total_score - spec.min_total) / span * 100.0, 2)

    if spec.scale_id == "JOA_CERVICAL_17":
        if total_score <= JOA_CERVICAL_RED_FLAG_THRESHOLD + 1e-9:
            result["red_flag_triggered"] = True
            result["red_flag_reason"] = (
                f"颈椎 JOA 17 分制总分为 {total_score_out} 分（≤ {JOA_CERVICAL_RED_FLAG_THRESHOLD} 分阈值），"
                "提示中重度颈脊髓功能障碍，严禁颈椎旋转扳法/暴力正骨，建议尽快完善颈椎 MRI 并转诊脊柱骨科评估。"
            )
            result["caution_triggered"] = True
            result["caution_reason"] = result["red_flag_reason"]
            result["linked_rule_ids"] = ["RF_001"]
        elif total_score <= JOA_CERVICAL_MILD_THRESHOLD + 1e-9:
            result["caution_triggered"] = True
            result["caution_reason"] = (
                f"颈椎 JOA 17 分制总分为 {total_score_out} 分（13–16 分，轻度脊髓功能障碍），"
                "虽未达 12 分重危阈值，但仍存在轻度颈脊髓功能受累风险，仍严禁颈椎旋转扳法/暴力整复，建议结合影像学动态随访。"
            )
            result["linked_rule_ids"] = ["RF_001"]

    if spec.scale_id == "JOA_LUMBAR_29" and validated_values.get("bladder_score", 0) <= -6:
        result["red_flag_triggered"] = True
        result["red_flag_reason"] = (
            "腰椎 JOA 膀胱功能项为 -6 分（尿潴留/失禁），高度警惕马尾综合征（骨科急诊红旗），"
            "严禁推拿保守观察，须立即急诊脊柱外科评估。"
        )
        result["caution_triggered"] = True
        result["caution_reason"] = result["red_flag_reason"]
        result["linked_rule_ids"] = ["RF_002"]

    if spec.improvement_rate_formula and "pre_score" in params and params["pre_score"] is not None:
        result["improvement_rate"] = calculate_joa_improvement_rate(
            pre_score=params["pre_score"],
            post_score=total_score,
            max_score=spec.max_total,
            min_score=spec.min_total,
            step=1.0,
        )

    return result
