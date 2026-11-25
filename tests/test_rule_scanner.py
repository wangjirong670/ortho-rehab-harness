"""Unit tests for core_tools/rule_scanner.py (dual-track red flags & drug conflicts)."""

from core_tools.rule_scanner import (
    check_negation_context,
    load_red_flags_catalog,
    scan_clinical_risks,
)


def _alert_ids(report: dict) -> set[str]:
    return {a["rule_id"] for a in report["alerts"]}


# ==============================================================================
# Track A: Negation window & anti-false-positive tests
# ==============================================================================


def test_negation_suppresses_false_positive_red_flags() -> None:
    text = (
        "患者30岁，久坐后腰颈酸胀。查体：否认踩棉花感，霍夫曼征阴性；"
        "无鞍区麻木、无二便失禁，大小便正常；无夜间静息痛；"
        "X线未见骨折线，排除骨折；下肢血管彩超未见血栓，Homans征阴性；"
        "无关节红肿热痛，局部不红不热，无发热。"
    )
    report = scan_clinical_risks(raw_text=text)
    assert report["status"] == "ok"
    assert report["alerts"] == []
    assert report["risk_level"] == "none"
    suppressed_terms = {item["term"] for item in report["suppressed_matches"]}
    assert {"踩棉花感", "鞍区麻木", "二便失禁", "夜间静息痛", "关节红肿热痛"}.issubset(
        suppressed_terms
    )
    assert report["summary"]["suppressed_count"] >= 5


def test_prohibitive_phrasing_is_recognized_as_negation_context() -> None:
    rf_catalog = load_red_flags_catalog()
    reply = "疑似脊髓受压时切忌在家自行复位，也禁止去推拿馆做旋转复位。"
    idx = reply.index("在家自行复位")
    is_neg, reason, token = check_negation_context(
        reply,
        idx,
        idx + len("在家自行复位"),
        global_negation_terms=rf_catalog.global_negation_terms,
    )
    assert is_neg is True
    assert reason == "text_negation_window"
    assert token == "切忌"


def test_negation_does_not_leak_across_comma_clauses() -> None:
    # Leading "无发热" must NOT suppress "踩棉花感" in the next clause!
    text = "患者无发热，近2周走路踩棉花感明显，扣纽扣不灵。"
    report = scan_clinical_risks(raw_text=text)
    assert "RF_001" in _alert_ids(report)
    assert report["has_critical_risk"] is True
    assert report["risk_level"] == "critical"


# ==============================================================================
# Track A: True positive clinical risk hits
# ==============================================================================


def test_true_positive_red_flags_and_drug_conflicts() -> None:
    # Cervical myelopathy + warfarin & notoginseng conflict
    text = "患者近两周走路踩棉花感，系扣子困难。房颤长期口服华法林，今自行加服三七粉和跌打丸消肿。"
    report = scan_clinical_risks(raw_text=text)
    ids = _alert_ids(report)
    assert "RF_001" in ids
    assert "DC_001" in ids
    assert report["has_critical_risk"] is True
    for alert in report["alerts"]:
        assert alert["matched_evidence"]
        assert alert["clinical_action"]
        assert alert["reference"]


def test_cauda_equina_and_acute_gout_hits() -> None:
    text_ce = "搬重物后剧烈腰痛伴鞍区麻木、尿潴留6小时。"
    report_ce = scan_clinical_risks(raw_text=text_ce)
    assert "RF_002" in _alert_ids(report_ce)
    assert report_ce["alerts"][0]["clinical_action"] == "EMERGENCY_REFER"

    text_gout = "昨晚饮酒后右足第一跖趾关节红肿热痛、局部灼热拒按，舌红苔黄腻。"
    report_gout = scan_clinical_risks(raw_text=text_gout)
    assert "RF_010" in _alert_ids(report_gout)


def test_single_nsaid_does_not_self_conflict_but_dual_nsaids_do() -> None:
    # Single NSAID alone should NOT trigger DC_004 (duplicate NSAIDs)
    single_report = scan_clinical_risks(raw_text="右膝疼痛，目前仅口服布洛芬缓释胶囊镇痛。")
    assert "DC_004" not in _alert_ids(single_report)

    # Two distinct oral NSAIDs MUST trigger DC_004
    dual_report = scan_clinical_risks(raw_text="右膝疼痛，患者同时口服洛索洛芬钠片与塞来昔布胶囊。")
    assert "DC_004" in _alert_ids(dual_report)


# ==============================================================================
# Track A & B: Drug washout window & structured facts override
# ==============================================================================


def test_text_drug_stopped_beyond_washout_is_suppressed() -> None:
    # Warfarin washout is 7 days; stopped 6 months ago (180 days) -> must NOT alert
    text = "既往有房颤史，已停用华法林半年，近日扭伤后口服三七粉。"
    report = scan_clinical_risks(raw_text=text)
    assert "DC_001" not in _alert_ids(report)
    assert any(
        s["rule_id"] == "DC_001" and s["reason"] == "text_drug_washout"
        for s in report["suppressed_matches"]
    )


def test_text_drug_stopped_within_washout_still_alerts() -> None:
    # Warfarin washout is 7 days; stopped only 2 days ago -> MUST still alert!
    text = "患者2天前刚停用华法林，今日自行口服三七粉活血。"
    report = scan_clinical_risks(raw_text=text)
    assert "DC_001" in _alert_ids(report)


def test_track_b_structured_facts_polarity_and_washout_precedence() -> None:
    # Structured absent symptom overrides raw_text mention; stopped_days_ago > washout suppresses DC_001
    raw_text = "病历提及踩棉花感待评估；既往曾用华法林，现用三七粉。"
    structured_facts = {
        "symptoms": [
            {"name": "踩棉花感", "polarity": "absent"},
        ],
        "medications": [
            {"name": "华法林", "status": "stopped", "stopped_days_ago": 180},
            {"name": "三七粉", "status": "active"},
        ],
    }
    report = scan_clinical_risks(raw_text=raw_text, structured_facts=structured_facts)
    assert "RF_001" not in _alert_ids(report)
    assert "DC_001" not in _alert_ids(report)
    assert any(
        s["rule_id"] == "RF_001" and s["source"] == "structured_facts"
        for s in report["suppressed_matches"]
    )
    assert any(
        s["rule_id"] == "DC_001" and s["reason"] == "drug_stopped_beyond_washout"
        for s in report["suppressed_matches"]
    )


def test_track_b_structured_facts_active_and_in_washout_triggers_alerts() -> None:
    # Even with empty raw_text, Track B accurately triggers RF_001 and DC_001 (stopped 3 days <= 7 days)
    structured_facts = {
        "symptoms": [
            {"name": "踩棉花感", "polarity": "present"},
        ],
        "medications": [
            {"name": "华法林", "status": "stopped", "stopped_days_ago": 3},
            {"name": "跌打丸", "status": "active"},
        ],
    }
    report = scan_clinical_risks(raw_text="", structured_facts=structured_facts)
    ids = _alert_ids(report)
    assert "RF_001" in ids
    assert "DC_001" in ids
    assert all(a["source"] == "structured_facts" for a in report["alerts"])
