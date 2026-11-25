"""Tests for Week 3 desktop copilot (prompts, Mock pipeline, recalculation, and Streamlit AppTest)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(PROJECT_ROOT / "app_copilot" / "streamlit_copilot.py")

from app_copilot.prompts import (
    DEMO_CASES,
    CopilotExtractedNote,
    build_draft_messages,
    build_extraction_messages,
)
from app_copilot.streamlit_copilot import (
    recalculate_scale_in_state,
    run_copilot_analysis,
)
from core_tools.calculator import calculate_scale
from core_tools.rule_scanner import scan_clinical_risks


def test_build_extraction_messages_contains_all_scales_and_strict_rules() -> None:
    raw = "某男，45岁，腰痛1周，VAS 4 分。"
    messages = build_extraction_messages(raw)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    sys_content = messages[0]["content"]
    for scale_id in ("VAS_PAIN", "WOMAC_LIKERT_V3_1", "JOA_CERVICAL_17", "JOA_LUMBAR_29"):
        assert scale_id in sys_content
    for item_id in ("pain_score", "pain_walking", "upper_limb_motor", "bladder_score"):
        assert item_id in sys_content
    assert "严禁脑补" in sys_content or "严禁默认" in sys_content
    assert "JSON" in sys_content
    assert raw in messages[1]["content"]


def test_build_draft_messages_injects_deterministic_constraints_and_scales() -> None:
    note = CopilotExtractedNote(
        patient_summary="某男，58岁",
        chief_complaint="走路踩棉花感2周",
        history_of_present_illness="渐进性双手精细动作障碍",
    ).model_dump()
    risk_report = scan_clinical_risks(raw_text="患者步态踩棉花感，扣纽扣不灵。")
    joa_res = calculate_scale(
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

    ward_msgs = build_draft_messages("ward_round", note, risk_report, [joa_res])
    ward_text = "\n".join(m["content"] for m in ward_msgs)
    assert "RF_001" in ward_text
    assert "颈椎旋转扳法" in ward_text
    assert "JOA_CERVICAL_17" in ward_text
    assert "9" in ward_text
    assert "中度脊髓功能障碍" in ward_text
    assert "分值越低风险" in ward_text
    assert "量表联动红旗" in ward_text

    rehab_msgs = build_draft_messages("rehab_education", note, risk_report, [joa_res])
    rehab_text = "\n".join(m["content"] for m in rehab_msgs)
    assert "分阶段康复宣教要点" in rehab_text
    assert "RF_001" in rehab_text

    with pytest.raises(ValueError):
        build_draft_messages("invalid_mode", note, risk_report, [joa_res])


def test_demo_case_1_cervical_myelopathy_and_joa_linkage(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    state = run_copilot_analysis(
        DEMO_CASES["cervical_myelopathy_joa"]["text"],
        draft_mode="ward_round",
        mock_mode=True,
    )

    alert_ids = {a["rule_id"] for a in state["risk_report"]["alerts"]}
    assert "RF_001" in alert_ids
    assert "RF_002" not in alert_ids

    joa_results = [r for r in state["scale_results"] if r.get("scale_id") == "JOA_CERVICAL_17"]
    assert len(joa_results) == 1
    joa = joa_results[0]
    assert joa["status"] == "ok"
    assert joa["total_score"] == 9
    assert joa["total_score"] <= 12.5
    assert joa["red_flag_triggered"] is True
    assert "RF_001" in joa["linked_rule_ids"]

    unified_sources = {a.get("source") for a in state["unified_alerts"]}
    assert "scale_linkage" in unified_sources
    assert "RF_001" in state["draft"]["content"]
    assert "严禁" in state["draft"]["content"]


def test_demo_case_2_warfarin_notoginseng_and_aspirin_washout(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    state = run_copilot_analysis(
        DEMO_CASES["warfarin_notoginseng_washout"]["text"],
        draft_mode="ward_round",
        mock_mode=True,
    )

    alert_ids = {a["rule_id"] for a in state["risk_report"]["alerts"]}
    assert "DC_001" in alert_ids
    assert "DC_003" not in alert_ids

    suppressed = state["risk_report"]["suppressed_matches"]
    assert any(
        s["rule_id"] == "DC_003"
        and "阿司匹林" in s["term"]
        and s["reason"] in ("drug_stopped_beyond_washout", "text_drug_washout")
        for s in suppressed
    )

    scale_map = {r["scale_id"]: r for r in state["scale_results"]}
    assert scale_map["VAS_PAIN"]["status"] == "ok"
    assert scale_map["VAS_PAIN"]["total_score"] == 5
    assert scale_map["WOMAC_LIKERT_V3_1"]["status"] == "error"
    assert len(scale_map["WOMAC_LIKERT_V3_1"]["missing_fields"]) == 21


def test_demo_case_3_negative_red_flags_no_false_positive(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    state = run_copilot_analysis(
        DEMO_CASES["sedentary_low_back_pain"]["text"],
        draft_mode="rehab_education",
        mock_mode=True,
    )

    assert state["risk_report"]["alerts"] == []
    assert state["unified_alerts"] == []
    assert state["risk_report"]["risk_level"] == "none"

    suppressed_terms = {s["term"] for s in state["risk_report"]["suppressed_matches"]}
    assert {"踩棉花感", "鞍区麻木", "二便失禁", "夜间静息痛", "关节红肿热痛"}.issubset(
        suppressed_terms
    )

    assert len(state["scale_results"]) == 1
    vas = state["scale_results"][0]
    assert vas["scale_id"] == "VAS_PAIN"
    assert vas["status"] == "ok"
    assert vas["total_score"] == 3
    assert vas["severity_band"] == "轻度疼痛"
    assert "分阶段康复宣教要点" in state["draft"]["content"]


def test_incomplete_scale_errors_without_zero_fill_then_recalculates_with_linkage(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    # Partial cervical JOA note: omits 4 items AND omits "踩棉花感" so only JOA threshold can trigger RF_001
    partial_note = (
        "某男，55岁，颈肩部酸胀1个月，无发热，无二便失禁。"
        "颈椎 JOA 17 分制初评仅记录：上肢运动功能 2 分，下肢运动功能 2 分（感觉与膀胱项待补充）。"
    )
    state = run_copilot_analysis(partial_note, draft_mode="ward_round", mock_mode=True)

    assert state["unified_alerts"] == []
    assert len(state["scale_results"]) == 1
    incomplete = state["scale_results"][0]
    assert incomplete["scale_id"] == "JOA_CERVICAL_17"
    assert incomplete["status"] == "error"
    assert "total_score" not in incomplete
    assert set(incomplete["missing_fields"]) == {
        "upper_limb_sensory",
        "lower_limb_sensory",
        "trunk_sensory",
        "bladder_function",
    }

    # Complete the missing fields so total = 2+2+1+1+1+2 = 9 <= 12.5
    completed_params = {
        "upper_limb_motor": 2,
        "lower_limb_motor": 2,
        "upper_limb_sensory": 1,
        "lower_limb_sensory": 1,
        "trunk_sensory": 1,
        "bladder_function": 2,
    }
    updated_state = recalculate_scale_in_state(
        state,
        "JOA_CERVICAL_17",
        completed_params,
        scale_index=0,
        refresh_draft=True,
        mock_mode=True,
    )

    # Original context preserved
    assert updated_state["raw_text"] == state["raw_text"]
    assert updated_state["extracted_note"]["chief_complaint"] == state["extracted_note"]["chief_complaint"]

    # Scale now passes & triggers red-flag linkage at top banner
    recalc_joa = updated_state["scale_results"][0]
    assert recalc_joa["status"] == "ok"
    assert recalc_joa["total_score"] == 9
    assert recalc_joa["red_flag_triggered"] is True
    assert any(a["source"] == "scale_linkage" and "RF_001" in a["rule_id"] for a in updated_state["unified_alerts"])
    assert "9" in updated_state["draft"]["content"]
    assert "RF_001" in updated_state["draft"]["content"]


def test_streamlit_apptest_smoke_all_demo_cases(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception

    # Sidebar compliance warning & demo buttons present
    assert len(at.sidebar.warning) >= 1
    demo_keys = [b.key for b in at.sidebar.button]
    for case_id in DEMO_CASES:
        assert f"demo_btn_{case_id}" in demo_keys

    # 1. Click Demo Case 1 and run analysis
    at.sidebar.button(key="demo_btn_cervical_myelopathy_joa").click().run()
    assert not at.exception
    assert "踩棉花感" in at.text_area(key="raw_note_input").value

    at.button(key="run_analysis_btn").click().run()
    assert not at.exception
    error_texts = "\n".join(el.value for el in at.error)
    assert "RF_001" in error_texts
    assert "JOA_CERVICAL_17" in error_texts

    # 2. Click Demo Case 2 and run analysis
    at.sidebar.button(key="demo_btn_warfarin_notoginseng_washout").click().run()
    assert not at.exception
    at.button(key="run_analysis_btn").click().run()
    assert not at.exception
    error_texts_2 = "\n".join(el.value for el in at.error)
    assert "DC_001" in error_texts_2
    assert "DC_003" not in error_texts_2
    warning_texts_2 = "\n".join(el.value for el in at.warning)
    assert "WOMAC_LIKERT_V3_1" in warning_texts_2

    # 3. Click Demo Case 3 and run analysis (no top errors, green success banner)
    at.sidebar.button(key="demo_btn_sedentary_low_back_pain").click().run()
    assert not at.exception
    at.button(key="run_analysis_btn").click().run()
    assert not at.exception
    assert len(at.error) == 0
    assert any("安全扫描通过" in el.value for el in at.success)


def test_streamlit_apptest_manual_tab2_recalculation_triggers_top_red_flag(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception

    partial_note = (
        "某男，56岁，颈肩僵硬3周，无发热，无二便失禁。"
        "颈椎 JOA 17 分制仅记录：上肢运动功能 2 分，下肢运动功能 2 分。"
    )
    at.text_area(key="raw_note_input").set_value(partial_note)
    at.button(key="run_analysis_btn").click().run()
    assert not at.exception
    assert len(at.error) == 0
    assert any("JOA_CERVICAL_17" in el.value for el in at.warning)

    # Complete missing JOA fields in Tab 2 (2+2+1+1+1+2 = 9 <= 12.5)
    at.number_input(key="fix_scale_1_0_JOA_CERVICAL_17_upper_limb_sensory").set_value(1.0)
    at.number_input(key="fix_scale_1_0_JOA_CERVICAL_17_lower_limb_sensory").set_value(1.0)
    at.number_input(key="fix_scale_1_0_JOA_CERVICAL_17_trunk_sensory").set_value(1.0)
    at.number_input(key="fix_scale_1_0_JOA_CERVICAL_17_bladder_function").set_value(2.0)
    at.button(key="recalc_err_1_0_JOA_CERVICAL_17").click().run()
    assert not at.exception

    # Top safety banner must now display JOA linkage red flag without losing context
    error_texts = "\n".join(el.value for el in at.error)
    assert "RF_001" in error_texts
    assert "JOA_CERVICAL_17" in error_texts

    # Refresh Tab 3 draft directly from Tab 2 sync button
    at.button(key="sync_draft_from_tab2_1").click().run()
    assert not at.exception
    assert any("RF_001" in el.value for el in at.markdown)
