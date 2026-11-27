"""Unit and integration tests for Week 5 timeline state resolution & current_state_snapshot."""

from __future__ import annotations

from app_copilot.prompts import (
    CopilotExtractedNote,
    CurrentStateSnapshot,
    SupersededCondition,
    TimelineEvent,
    build_extraction_messages,
)
from app_copilot.service import run_copilot_analysis
from core_tools.llm_client import extract_structured_json
from core_tools.rule_scanner import (
    merge_snapshot_into_structured_facts,
    scan_clinical_risks,
)


def test_prompts_timeline_and_snapshot_backward_compatibility() -> None:
    # Legacy TimelineEvent without department/key_change remains valid
    legacy_ev = TimelineEvent(date="2026-05-01", event="骨伤科门诊初诊")
    assert legacy_ev.department == "未标明"
    assert legacy_ev.key_change == ""

    # Legacy CopilotExtractedNote without current_state_snapshot remains valid
    legacy_note = CopilotExtractedNote(chief_complaint="腰痛3天")
    assert isinstance(legacy_note.current_state_snapshot, CurrentStateSnapshot)
    assert legacy_note.current_state_snapshot.superseded_conditions == []

    msgs = build_extraction_messages("半年前心内科口服利伐沙班，今日加服三七粉。")
    sys_prompt = msgs[0]["content"]
    assert "current_state_snapshot" in sys_prompt
    assert "superseded_conditions" in sys_prompt
    assert "key_change" in sys_prompt


def test_rule_scanner_suppresses_superseded_rule_via_snapshot() -> None:
    raw_text = (
        "【术后第10天 骨科】ACL重建术后早期保护期，严禁跑跳。\n"
        "【术后第16周 运动医学科】主刀医生评估达标，准予进入基础直线慢跑阶段。"
    )
    # Without snapshot override, raw_text triggers RF_007
    baseline_report = scan_clinical_risks(raw_text=raw_text)
    assert any(a["rule_id"] == "RF_007" for a in baseline_report["alerts"])

    # With snapshot superseding RF_007, Track B suppresses RF_007 cleanly
    snapshot = CurrentStateSnapshot(
        active_surgical_history=["前交叉韧带重建术（ACL）"],
        superseded_conditions=[
            SupersededCondition(
                old_state="术后第10天早期保护期",
                new_state="术后第16周主刀准予直线慢跑",
                rule_hint="RF_007",
            )
        ],
        diagnosis_override_note="术后第16周解除早期禁跑限制",
    ).model_dump()
    merged = merge_snapshot_into_structured_facts(snapshot)
    resolved_report = scan_clinical_risks(raw_text=raw_text, structured_facts=merged)

    assert not any(a["rule_id"] == "RF_007" for a in resolved_report["alerts"])
    assert any(
        s["rule_id"] == "RF_007"
        and s["reason"] == "superseded_by_latest_timeline_state"
        and s["source"] == "structured_facts"
        for s in resolved_report["suppressed_matches"]
    )


def test_mock_extract_longitudinal_cases_and_resolution(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")

    # 1. ORTHO_LONG_011: Remote rivaroxaban + notoginseng + ibuprofen -> DC_002 + DC_006
    note_011 = (
        "【半年前 心内科门诊】因心房颤动长期规律口服利伐沙班片15mg qd抗凝治疗。\n"
        "【1月前 骨伤科门诊】腰扭伤。\n"
        "【今日】家属欲叠加大剂量三七粉、跌打丸与布洛芬缓释胶囊止痛。"
    )
    state_011 = run_copilot_analysis(note_011, mock_mode=True)
    alert_ids_011 = {a["rule_id"] for a in state_011["risk_report"]["alerts"]}
    assert {"DC_002", "DC_006"}.issubset(alert_ids_011)
    assert len(state_011["extracted_note"]["timeline_events"]) >= 2
    assert any(
        m["name"] == "利伐沙班"
        for m in state_011["extracted_note"]["current_state_snapshot"]["active_medications"]
    )

    # 2. ORTHO_LONG_012: Warfarin stopped 365 days ago -> DC_001 suppressed, stopped_medications populated
    note_012 = (
        "【1年前 心内科】房颤消融术后已停用华法林365天，凝血功能正常。\n"
        "【昨日 骨伤科】轻度踝扭伤，X线未见骨折线，已排除骨折。询问温和活血外敷（如三七外敷）。"
    )
    state_012 = run_copilot_analysis(note_012, mock_mode=True)
    assert not any(a["rule_id"] == "DC_001" for a in state_012["risk_report"]["alerts"])
    stopped_012 = state_012["extracted_note"]["current_state_snapshot"]["stopped_medications"]
    assert any(m["name"] == "华法林" and m["stopped_days_ago"] == 365.0 for m in stopped_012)

    # 3. ORTHO_LONG_013: Clopidogrel stopped 3 days ago (< 7 days washout) stays in active_medications
    note_013 = (
        "【1年前 心内科】冠脉支架术后长期服氯吡格雷。\n"
        "【3天前】自行停用氯吡格雷3天，今日要求行小针刀或强力推拿。"
    )
    ext_013 = extract_structured_json(note_013, schema_model=CopilotExtractedNote, mock_mode=True)
    snap_013 = ext_013["data"]["current_state_snapshot"]
    assert any(
        m["name"] == "氯吡格雷" and m["status"] == "stopped" and m["stopped_days_ago"] == 3.0
        for m in snap_013["active_medications"]
    )
    assert "冠脉支架植入术（PCI）" in snap_013["active_surgical_history"]

    # 4. ORTHO_LONG_016: Cold-dampness superseded by Heat Bi (RF_010)
    note_016 = (
        "【2月前 中医骨伤门诊】右膝寒湿痹阻，艾灸热敷有效。\n"
        "【今日】饮酒海鲜后右膝突发关节红肿热痛、局部灼热、舌红苔黄腻，欲加倍重灸热敷。"
    )
    state_016 = run_copilot_analysis(note_016, mock_mode=True)
    assert any(a["rule_id"] == "RF_010" for a in state_016["risk_report"]["alerts"])
    snap_016 = state_016["extracted_note"]["current_state_snapshot"]
    assert snap_016["diagnosis_override_note"] is not None
    assert "热痹" in snap_016["diagnosis_override_note"]

    # 5. ORTHO_LONG_017: Early negative red flags overridden by acute cauda equina (RF_002)
    note_017 = (
        "【3周前 门诊】普通腰肌劳损，无鞍区麻木、无二便失禁。\n"
        "【今日 急诊】突发会阴鞍区麻木、排尿困难。"
    )
    state_017 = run_copilot_analysis(note_017, mock_mode=True)
    assert any(a["rule_id"] == "RF_002" for a in state_017["risk_report"]["alerts"])
    cur_syms_017 = {
        s["name"]: s["polarity"]
        for s in state_017["extracted_note"]["current_state_snapshot"]["current_symptoms"]
    }
    assert cur_syms_017.get("鞍区麻木") == "present"

    # 6. ORTHO_LONG_020: ACL week-16 cleared for straight jogging supersedes RF_007
    note_020 = (
        "【术后第10天 骨科病房】ACL重建术后早期保护期，严禁跑跳。\n"
        "【术后第16周 运动医学科复查】主刀医生评估肌力与活动度达标，准予进入基础直线慢跑阶段。"
    )
    state_020 = run_copilot_analysis(note_020, mock_mode=True)
    assert not any(a["rule_id"] == "RF_007" for a in state_020["risk_report"]["alerts"])
    assert any(
        s["rule_id"] == "RF_007" and s["reason"] == "superseded_by_latest_timeline_state"
        for s in state_020["risk_report"]["suppressed_matches"]
    )


def test_prohibited_stop_drug_phrasing_does_not_suppress_active_conflict(monkeypatch) -> None:
    """'不能停药/不可停用' must NOT be misparsed as drug cessation; DC_001 must fire."""
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    note = "心内科医嘱华法林不能停药已服1年，今日扭伤加服三七粉。"

    ext = extract_structured_json(note, schema_model=CopilotExtractedNote, mock_mode=True)["data"]
    med_map = {m["name"]: m for m in ext["structured_facts"]["medications"]}
    assert med_map["华法林"]["status"] == "active"
    assert med_map["三七"]["status"] == "active"

    state = run_copilot_analysis(note, mock_mode=True)
    alert_ids = {a["rule_id"] for a in state["risk_report"]["alerts"]}
    assert "DC_001" in alert_ids


def test_stopped_drug_marker_does_not_leak_across_comma_to_active_drug(monkeypatch) -> None:
    """Stopping warfarin in the first comma-clause must NOT mark notoginseng as stopped."""
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    note = "华法林已停用1年，今日扭伤加服三七粉。"

    ext = extract_structured_json(note, schema_model=CopilotExtractedNote, mock_mode=True)["data"]
    med_map = {m["name"]: m for m in ext["structured_facts"]["medications"]}
    assert med_map["华法林"]["status"] == "stopped"
    assert med_map["华法林"]["stopped_days_ago"] == 365.0
    assert med_map["三七"]["status"] == "active"
    assert med_map["三七"]["stopped_days_ago"] is None

    state = run_copilot_analysis(note, mock_mode=True)
    assert not any(a["rule_id"] == "DC_001" for a in state["risk_report"]["alerts"])
    assert any(
        s["rule_id"] == "DC_001" and s["reason"] == "drug_stopped_beyond_washout"
        for s in state["risk_report"]["suppressed_matches"]
    )

