"""Unit tests for core_tools/calculator.py."""

import pytest

from core_tools.calculator import (
    calculate_joa_improvement_rate,
    calculate_scale,
    list_available_scales,
)


def _full_womac_params(value: int = 2) -> dict:
    return {
        "pain_walking": value,
        "pain_stairs": value,
        "pain_night": value,
        "pain_rest": value,
        "pain_standing": value,
        "stiffness_morning": value,
        "stiffness_later_day": value,
        "func_descending_stairs": value,
        "func_ascending_stairs": value,
        "func_rising_sitting": value,
        "func_standing": value,
        "func_bending": value,
        "func_walking_flat": value,
        "func_car": value,
        "func_shopping": value,
        "func_putting_socks": value,
        "func_rising_bed": value,
        "func_taking_off_socks": value,
        "func_lying_bed": value,
        "func_bath": value,
        "func_sitting": value,
        "func_toilet": value,
        "func_heavy_domestic": value,
        "func_light_domestic": value,
    }


def _full_joa_lumbar_params(bladder: int = 0) -> dict:
    return {
        "low_back_pain": 2,
        "leg_pain_tingling": 2,
        "gait": 2,
        "slr_test": 1,
        "sensory_disturbance": 2,
        "motor_disturbance": 2,
        "adl_turning_in_bed": 2,
        "adl_standing": 1,
        "adl_washing": 2,
        "adl_leaning_forward": 1,
        "adl_sitting": 1,
        "adl_lifting": 1,
        "adl_walking": 2,
        "bladder_score": bladder,
    }


def test_list_available_scales_covers_four_scales() -> None:
    scales = list_available_scales()
    ids = {s["scale_id"] for s in scales}
    assert ids == {"VAS_PAIN", "WOMAC_LIKERT_V3_1", "JOA_CERVICAL_17", "JOA_LUMBAR_29"}


@pytest.mark.parametrize(
    ("score", "expected_band"),
    [
        (0, "无痛"),
        (2.5, "轻度疼痛"),
        (3.5, "轻度疼痛"),
        (4.0, "中度疼痛"),
        (5.0, "中度疼痛"),
        (6.0, "中度疼痛"),
        (7.0, "重度疼痛"),
        (8.5, "重度疼痛"),
    ],
)
def test_vas_pain_valid_scores_and_direction(score: float, expected_band: str) -> None:
    res = calculate_scale("VAS_PAIN", {"pain_score": score})
    assert res["status"] == "ok"
    assert res["score_direction"] == "higher_is_worse"
    assert res["total_score"] == score
    assert res["severity_band"] == expected_band
    assert res["red_flag_triggered"] is False


def test_womac_likert_calculation_and_normalized_pct() -> None:
    res = calculate_scale("WOMAC_LIKERT_V3_1", _full_womac_params(2))
    assert res["status"] == "ok"
    assert res["score_direction"] == "higher_is_worse"
    assert res["total_score"] == 48
    assert res["normalized_score_pct"] == 50.0
    assert res["severity_band"] == "中度功能受限/症状"
    assert "项目自定义" in res["version_note"]
    assert res["subscale_scores"]["pain"]["score"] == 10
    assert res["subscale_scores"]["stiffness"]["score"] == 4
    assert res["subscale_scores"]["physical_function"]["score"] == 34


def test_joa_cervical_17_integer_thresholds_and_mild_caution() -> None:
    # Total = 2+2+1+1+1+2 = 9 (<= 12)
    res_low = calculate_scale(
        "JOA_CERVICAL_17",
        {
            "upper_limb_motor": 2,
            "lower_limb_motor": 2,
            "upper_limb_sensory": 1,
            "lower_limb_sensory": 1,
            "trunk_sensory": 1,
            "bladder_function": 2,
        },
    )
    assert res_low["status"] == "ok"
    assert res_low["score_direction"] == "lower_is_worse"
    assert res_low["total_score"] == 9
    assert res_low["severity_band"] == "中度脊髓功能障碍"
    assert res_low["red_flag_triggered"] is True
    assert res_low["caution_triggered"] is True
    assert "RF_001" in res_low["linked_rule_ids"]

    # Boundary: total = 12 (<= 12) triggers critical red flag; total = 13..16 triggers advisory caution
    res_12 = calculate_scale(
        "JOA_CERVICAL_17",
        {
            "upper_limb_motor": 3,
            "lower_limb_motor": 3,
            "upper_limb_sensory": 1,
            "lower_limb_sensory": 1,
            "trunk_sensory": 2,
            "bladder_function": 2,
        },
    )
    assert res_12["total_score"] == 12
    assert res_12["red_flag_triggered"] is True

    res_13 = calculate_scale(
        "JOA_CERVICAL_17",
        {
            "upper_limb_motor": 3,
            "lower_limb_motor": 3,
            "upper_limb_sensory": 2,
            "lower_limb_sensory": 1,
            "trunk_sensory": 2,
            "bladder_function": 2,
        },
    )
    assert res_13["total_score"] == 13
    assert res_13["red_flag_triggered"] is False
    assert res_13["caution_triggered"] is True
    assert "严禁颈椎旋转扳法" in res_13["caution_reason"]
    assert "RF_001" in res_13["linked_rule_ids"]

    # Score = 17 (normal) triggers neither red_flag nor caution
    res_17 = calculate_scale(
        "JOA_CERVICAL_17",
        {
            "upper_limb_motor": 4,
            "lower_limb_motor": 4,
            "upper_limb_sensory": 2,
            "lower_limb_sensory": 2,
            "trunk_sensory": 2,
            "bladder_function": 3,
        },
    )
    assert res_17["total_score"] == 17
    assert res_17["red_flag_triggered"] is False
    assert res_17["caution_triggered"] is False
    assert res_17["linked_rule_ids"] == []


def test_joa_pre_score_step_validation_rejects_fractional_scores() -> None:
    bad_rate = calculate_joa_improvement_rate(pre_score=16.5, post_score=13, max_score=17)
    assert bad_rate["status"] == "error"
    assert bad_rate["error_code"] == "INVALID_STEP"
    assert bad_rate["improvement_rate_pct"] is None

    bad_integrated = calculate_scale(
        "JOA_CERVICAL_17",
        {
            "upper_limb_motor": 3,
            "lower_limb_motor": 3,
            "upper_limb_sensory": 2,
            "lower_limb_sensory": 1,
            "trunk_sensory": 2,
            "bladder_function": 2,
            "pre_score": 16.5,
        },
    )
    assert bad_integrated["status"] == "error"
    assert bad_integrated["error_code"] == "VALIDATION_ERROR"
    assert any(f["field"] == "pre_score" for f in bad_integrated["invalid_fields"])


def test_joa_cervical_improvement_rate_and_zero_division_protection() -> None:
    # Normal Hirabayashi calculation: pre=9, post=15 => (15-9)/(17-9)*100 = 75.0%
    rate = calculate_joa_improvement_rate(pre_score=9, post_score=15, max_score=17)
    assert rate["status"] == "ok"
    assert rate["improvement_rate_pct"] == 75.0
    assert "优" in rate["efficacy_grade"]

    # Also test integrated optional pre_score inside calculate_scale
    integrated = calculate_scale(
        "JOA_CERVICAL_17",
        {
            "upper_limb_motor": 3,
            "lower_limb_motor": 4,
            "upper_limb_sensory": 2,
            "lower_limb_sensory": 2,
            "trunk_sensory": 1,
            "bladder_function": 3,
            "pre_score": 9,
        },
    )
    assert integrated["status"] == "ok"
    assert integrated["total_score"] == 15
    assert integrated["improvement_rate"]["improvement_rate_pct"] == 75.0

    # Zero-division protection when pre_score == max_score (17)
    zero_div = calculate_joa_improvement_rate(pre_score=17, post_score=17, max_score=17)
    assert zero_div["status"] == "error"
    assert zero_div["error_code"] == "ZERO_DIVISION_PROTECTED"
    assert zero_div["improvement_rate_pct"] is None


def test_joa_lumbar_29_calculation_and_discrete_bladder_validation() -> None:
    res = calculate_scale("JOA_LUMBAR_29", _full_joa_lumbar_params(bladder=0))
    assert res["status"] == "ok"
    assert res["score_direction"] == "lower_is_worse"
    assert res["total_score"] == 21
    assert res["severity_band"] == "轻度腰椎功能障碍"
    assert res["red_flag_triggered"] is False

    # Bladder score -6 triggers cauda equina linkage
    res_cauda = calculate_scale("JOA_LUMBAR_29", _full_joa_lumbar_params(bladder=-6))
    assert res_cauda["status"] == "ok"
    assert res_cauda["total_score"] == 15
    assert res_cauda["red_flag_triggered"] is True
    assert "RF_002" in res_cauda["linked_rule_ids"]

    # Disallowed intermediate bladder value (-2 not in [-6, -3, 0]) must fail
    bad_bladder = calculate_scale("JOA_LUMBAR_29", _full_joa_lumbar_params(bladder=-2))
    assert bad_bladder["status"] == "error"
    assert bad_bladder["invalid_fields"][0]["field"] == "bladder_score"


def test_strict_validation_forbids_silent_zero_fill_and_invalid_inputs() -> None:
    # Missing required fields in WOMAC -> MUST return error, never default to 0
    partial_womac = _full_womac_params(1)
    del partial_womac["pain_night"]
    del partial_womac["func_toilet"]
    res_missing = calculate_scale("WOMAC_LIKERT_V3_1", partial_womac)
    assert res_missing["status"] == "error"
    assert set(res_missing["missing_fields"]) == {"pain_night", "func_toilet"}
    assert "total_score" not in res_missing

    # Explicit None treated as missing
    res_none = calculate_scale("VAS_PAIN", {"pain_score": None})
    assert res_none["status"] == "error"
    assert res_none["missing_fields"] == ["pain_score"]

    # Out of range value
    res_oob = calculate_scale("VAS_PAIN", {"pain_score": 12})
    assert res_oob["status"] == "error"
    assert res_oob["invalid_fields"][0]["field"] == "pain_score"

    # Invalid step (VAS allows 0.5 step, 3.3 is invalid)
    res_step = calculate_scale("VAS_PAIN", {"pain_score": 3.3})
    assert res_step["status"] == "error"
    assert res_step["invalid_fields"][0]["field"] == "pain_score"

    # Reject bool masquerading as int and string types
    res_bool = calculate_scale("VAS_PAIN", {"pain_score": True})
    assert res_bool["status"] == "error"
    res_str = calculate_scale("VAS_PAIN", {"pain_score": "5"})
    assert res_str["status"] == "error"

    # Unknown scale ID
    res_unknown = calculate_scale("NON_EXISTENT_SCALE", {"a": 1})
    assert res_unknown["status"] == "error"
    assert res_unknown["error_code"] == "UNKNOWN_SCALE"
