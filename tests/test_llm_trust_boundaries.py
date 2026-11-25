"""Adversarial regression tests for P0-2 LLM structured-facts & timeline trust boundaries."""

from __future__ import annotations

from core_tools.llm_client import extract_structured_json
from core_tools.rule_scanner import merge_snapshot_into_structured_facts, scan_clinical_risks


def _alert_map(report: dict) -> dict[str, dict]:
    return {a["rule_id"]: a for a in report["alerts"]}


def test_critical_rules_cannot_be_suppressed_by_llm_fields() -> None:
    # 1. superseded_conditions trying to waive critical RF_001 / RF_002 must be rejected
    text_rf1 = "走路踩棉花感，霍夫曼征阳性，扣纽扣不灵。"
    rep_rf1 = scan_clinical_risks(
        raw_text=text_rf1,
        structured_facts={
            "superseded_conditions": [
                {"old_state": "脊髓受压", "new_state": "已缓解", "rule_hint": "RF_001"}
            ]
        },
    )
    alerts_rf1 = _alert_map(rep_rf1)
    assert "RF_001" in alerts_rf1
    assert rep_rf1["has_critical_risk"] is True
    assert alerts_rf1["RF_001"]["conflict_with_text"] is True
    assert alerts_rf1["RF_001"]["needs_review"] is True

    # 2. Untrusted schema-outside superseded_rule_ids bypass (including lowercase) must be ignored
    rep_rf1_bypass = scan_clinical_risks(
        raw_text=text_rf1,
        structured_facts={"superseded_rule_ids": ["rf_001", "RF_002", "NON_EXISTENT_999"]},
    )
    assert "RF_001" in _alert_map(rep_rf1_bypass)
    assert rep_rf1_bypass["has_critical_risk"] is True

    # 3. absent polarity cannot silence unnegated affirmative critical symptom in text
    rep_absent = scan_clinical_risks(
        raw_text="走路踩棉花感。",
        structured_facts={"symptoms": [{"name": "踩棉花感", "polarity": "absent"}]},
    )
    alerts_absent = _alert_map(rep_absent)
    assert "RF_001" in alerts_absent
    assert alerts_absent["RF_001"]["conflict_with_text"] is True
    assert alerts_absent["RF_001"]["needs_review"] is True

    # 4. Old negation cannot override a newer affirmative symptom via absent
    rep_progress = scan_clinical_risks(
        raw_text="【3周前】普通腰痛，无鞍区麻木。\n【今日】突发鞍区麻木、尿潴留。",
        structured_facts={"symptoms": [{"name": "鞍区麻木", "polarity": "absent"}]},
    )
    alerts_progress = _alert_map(rep_progress)
    assert "RF_002" in alerts_progress
    assert alerts_progress["RF_002"]["conflict_with_text"] is True

    # 5. Corroborated latest negation + absent CAN suppress legitimately
    rep_resolved = scan_clinical_risks(
        raw_text="【3周前】疑踩棉花感。\n【今日】症状完全消失，否认踩棉花感，霍夫曼征阴性。",
        structured_facts={"symptoms": [{"name": "踩棉花感", "polarity": "absent"}]},
    )
    assert "RF_001" not in _alert_map(rep_resolved)

    # 6. Hallucinated stopped_days_ago=180 when raw_text has active warfarin -> DC_001 retained + conflict
    rep_drug = scan_clinical_risks(
        raw_text="长期口服华法林，今日加服三七粉。",
        structured_facts={
            "medications": [
                {"name": "华法林", "status": "stopped", "stopped_days_ago": 180},
                {"name": "三七粉", "status": "active"},
            ]
        },
    )
    alerts_drug = _alert_map(rep_drug)
    assert "DC_001" in alerts_drug
    assert alerts_drug["DC_001"]["conflict_with_text"] is True
    assert alerts_drug["DC_001"]["needs_review"] is True

    # 7. Hallucinated stopped_days_ago=180 when raw_text says "切勿自行停用华法林" -> DC_001 retained + conflict
    rep_drug_neg = scan_clinical_risks(
        raw_text="医生叮嘱切勿自行停用华法林，已规律服用6个月，今日扭伤加服三七粉。",
        structured_facts={
            "medications": [
                {"name": "华法林", "status": "stopped", "stopped_days_ago": 180},
                {"name": "三七粉", "status": "active"},
            ]
        },
    )
    assert "DC_001" in _alert_map(rep_drug_neg)
    assert _alert_map(rep_drug_neg)["DC_001"]["conflict_with_text"] is True


def test_acl_question_cannot_supersede_rf007_in_scanner_or_mock() -> None:
    poc5_text = (
        "【术后第10天 骨科】右膝ACL重建联合半月板缝合术后，早期保护期，严禁跑跳，患者已自行丢拐全负重。\n"
        "【今日】家属询问：批准跑步要等到半年后吗？"
    )
    # Even if an LLM hallucinates superseded_conditions for RF_007, scanner refuses without non-interrogative clearance
    rep_manual = scan_clinical_risks(
        raw_text=poc5_text,
        structured_facts={
            "superseded_conditions": [
                {
                    "old_state": "ACL重建术后早期保护期",
                    "new_state": "批准跑步",
                    "rule_hint": "RF_007",
                }
            ]
        },
    )
    assert "RF_007" in _alert_map(rep_manual)
    assert _alert_map(rep_manual)["RF_007"]["conflict_with_text"] is True

    # Also test full Mock extraction -> merge -> scan pipeline
    ext = extract_structured_json(poc5_text, mock_mode=True)
    assert ext["status"] == "ok"
    merged = merge_snapshot_into_structured_facts(
        ext["data"].get("current_state_snapshot"),
        fallback_facts=ext["data"].get("structured_facts"),
    )
    rep_mock = scan_clinical_risks(raw_text=poc5_text, structured_facts=merged)
    assert "RF_007" in _alert_map(rep_mock)


def test_negated_clearance_phrasing_cannot_supersede_rf007() -> None:
    negated_clearance = (
        "【术后第8周 复查】ACL重建术后评估尚未达标，暂不准予直线慢跑。"
    )
    rep = scan_clinical_risks(
        raw_text=negated_clearance,
        structured_facts={
            "superseded_conditions": [
                {
                    "old_state": "ACL重建术后早期保护期",
                    "new_state": "准予直线慢跑",
                    "rule_hint": "RF_007",
                }
            ]
        },
    )
    alerts = _alert_map(rep)
    assert "RF_007" in alerts
    assert alerts["RF_007"]["conflict_with_text"] is True
    assert alerts["RF_007"]["needs_review"] is True
