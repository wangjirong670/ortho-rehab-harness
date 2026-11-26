"""Unit tests for core_tools/rule_scanner.py (dual-track red flags & drug conflicts)."""

import pytest

from core_tools.rule_scanner import (
    check_negation_context,
    find_stop_drug_marker,
    is_context_negated,
    load_red_flags_catalog,
    parse_duration_days,
    scan_clinical_risks,
    strip_quoted_segments,
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
    # Corroborated absent symptom (with latest text negation) & valid stopped_days_ago > washout suppress cleanly
    raw_text = "既往曾怀疑踩棉花感，今日查体否认踩棉花感；已停用华法林半年，现用三七粉。"
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


# ==============================================================================
# P0-1 / P0-3 / P1-7 / P2-1 / P2-3 Adversarial Regression Tests
# ==============================================================================


@pytest.mark.parametrize(
    ("text", "expected_rule_id"),
    [
        ("患者无明显诱因出现踩棉花感，伴扣纽扣不灵。", "RF_001"),
        ("双下肢无力伴走路踩棉花感。", "RF_001"),
        ("不排除马尾综合征。", "RF_002"),
        ("尚不能排除脊髓型颈椎病。", "RF_001"),
        ("不能排除脊髓受压。", "RF_001"),
        ("并非无大小便失禁。", "RF_002"),
        ("鞍区麻木未见缓解。", "RF_002"),
        ("尿潴留未见好转。", "RF_002"),
        ("此前无明显不适近1周出现踩棉花感。", "RF_001"),
        ("无外伤史但近1周出现鞍区麻木。", "RF_002"),
        ("无外伤但出现鞍区麻木。", "RF_002"),
        ("毫无疑问该患者近日出现排尿困难。", "RF_002"),
        ("否认高血压糖尿病：今突发尿潴留。", "RF_002"),
        ("霍夫曼征阳性双侧腱反射正常。", "RF_001"),
        ("MRI排除骨折后出现尿潴留。", "RF_002"),
        ("否认发热盗汗伴夜间静息痛加重。", "RF_004"),
        ("此前无明显不适，近1周出现踩棉花感。", "RF_001"),
    ],
)
def test_negation_window_keeps_affirmative_red_flags(text: str, expected_rule_id: str) -> None:
    report = scan_clinical_risks(raw_text=text)
    assert expected_rule_id in _alert_ids(report), f"Expected {expected_rule_id} for: {text}"


@pytest.mark.parametrize(
    "text",
    [
        "否认踩棉花感，霍夫曼征阴性。",
        "无二便失禁",
        "未出现发热或踩棉花感。",
        "未见新发鞍区麻木。",
    ],
)
def test_negation_window_controls_remain_negative(text: str) -> None:
    report = scan_clinical_risks(raw_text=text)
    assert report["alerts"] == [], f"Unexpected alerts for negative control: {text}"


def test_stop_marker_scope_and_resume() -> None:
    # 1. Cross-drug isolation: warfarin stopped 3 months, switched to rivaroxaban + notoginseng -> DC_002
    rep1 = scan_clinical_risks(raw_text="华法林已停3月改用利伐沙班抗凝，今日自行加服三七粉。")
    assert "DC_002" in _alert_ids(rep1)
    assert "DC_001" not in _alert_ids(rep1)

    # 2. Negated stop phrasing with regular use duration -> DC_001 must fire
    rep2 = scan_clinical_risks(raw_text="医生叮嘱切勿自行停用华法林，已规律服用6个月，今日扭伤加服三七粉。")
    assert "DC_001" in _alert_ids(rep2)

    # 3. Stopped then resumed -> DC_001 must fire
    rep3 = scan_clinical_risks(raw_text="华法林停药10天后已于昨日恢复服用，今日加服三七粉。")
    assert "DC_001" in _alert_ids(rep3)

    # 4. Function-level stop marker & planned stop exclusions
    assert find_stop_drug_marker("千万别停用华法林") is None
    assert find_stop_drug_marker("华法林没停药") is None
    assert find_stop_drug_marker("建议择期停用华法林") is None
    assert find_stop_drug_marker("别停药") is None
    assert find_stop_drug_marker("至嘱勿自行停用") is None
    assert find_stop_drug_marker("华法林停药10天后已于昨日恢复服用") is None
    assert find_stop_drug_marker("已停用华法林半年") in ("停用", "已停")

    # 5. Duration parser fixes
    assert parse_duration_days("一周半") == 10.5
    assert parse_duration_days("1个半月") == 45.0
    assert parse_duration_days("一个半月") == 45.0
    assert parse_duration_days("72小时") == 3.0

    # 6. Washout boundary preserved (7 days alerts, 8 days suppressed, vague duration conservative)
    assert "DC_001" in _alert_ids(scan_clinical_risks(raw_text="停用华法林7天，今日口服三七粉。"))
    assert "DC_001" not in _alert_ids(scan_clinical_risks(raw_text="停用华法林8天，今日口服三七粉。"))
    assert "DC_001" in _alert_ids(scan_clinical_risks(raw_text="前天刚停用华法林，今日口服三七粉。"))


@pytest.mark.parametrize(
    ("text", "expected_rule_id"),
    [
        ("腰突患者近2天出现尿失禁，伴会阴麻木。", "RF_002"),
        ("腰椎管狭窄，近1周二便功能障碍，小便费力。", "RF_002"),
        ("查体：Hoffmann(+)，双侧病理征(+)，腱反射亢进。", "RF_001"),
        ("这两天脚底发飘像踩海绵，扣扣子费劲，筷子拿不稳。", "RF_001"),
        ("搬重物后腰痛，屁股发木解不出尿半天了。", "RF_002"),
        ("女，70岁，T值-3.2，腰背痛，想正骨。", "RF_009"),
        ("腰痛伴新发足下垂3天，无法背伸。", "RF_002"),
    ],
)
def test_p1_7_red_flag_keyword_and_regex_gaps(text: str, expected_rule_id: str) -> None:
    report = scan_clinical_risks(raw_text=text)
    assert expected_rule_id in _alert_ids(report)


@pytest.mark.parametrize(
    "text",
    [
        "30岁程序员，久坐后屁股麻，没有腿麻，走路有力气，大小便正常，腰有点酸。",
        "学生，上课常憋尿，最近腰酸，请问怎么护腰？",
        "开会久坐会阴有点闷，小便正常，腰肌紧张。",
    ],
)
def test_p1_5_colloquial_negative_counterexamples_zero_false_positives(text: str) -> None:
    report = scan_clinical_risks(raw_text=text)
    assert report["alerts"] == []


def test_p2_1_false_positives_suppressed_and_p2_3_new_rules() -> None:
    # P2-1: remote trauma & long-term post-ACL & misclassified tramadol/codeine combo
    assert "RF_003" not in _alert_ids(scan_clinical_risks(raw_text="2年前车祸外伤史，已痊愈，现颈肩僵痛。"))
    assert "RF_007" not in _alert_ids(scan_clinical_risks(raw_text="ACL重建术后2年，现腰痛。"))
    assert "RF_007" not in _alert_ids(scan_clinical_risks(raw_text="前叉重建术后两年以上，现偶有酸胀。"))
    assert "DC_004" not in _alert_ids(scan_clinical_risks(raw_text="口服氨酚双氢可待因联合布洛芬镇痛。"))

    # P2-3: DC_011 (antithrombotic active or in-washout + invasive procedure)
    assert "DC_011" in _alert_ids(scan_clinical_risks(raw_text="房颤长期口服华法林，今日要求小针刀松解。"))
    assert "DC_011" in _alert_ids(scan_clinical_risks(raw_text="冠脉支架术后停用氯吡格雷3天，今日要求行小针刀松解。"))
    assert "DC_011" in _alert_ids(scan_clinical_risks(raw_text="氯吡格雷停药7天，要求小针刀治疗。"))
    assert "DC_011" not in _alert_ids(scan_clinical_risks(raw_text="氯吡格雷停药8天，要求小针刀治疗。"))

    # P2-3: RF_011..RF_016
    assert "RF_011" in _alert_ids(scan_clinical_risks(raw_text="3年前行颈椎前路融合内固定术（ACDF），今落枕想做旋转斜扳。"))
    assert "RF_012" in _alert_ids(scan_clinical_risks(raw_text="强直性脊柱炎呈竹节样脊柱改变，腰背僵硬要求正骨。"))
    assert "RF_013" in _alert_ids(scan_clinical_risks(raw_text="下肢深静脉血栓（DVT）患者今日突发胸痛伴呼吸困难、咯血。"))
    assert "RF_014" in _alert_ids(scan_clinical_risks(raw_text="昨晚颈椎旋转复位正骨后今晨突发眩晕伴复视、构音障碍。"))
    assert "RF_015" in _alert_ids(scan_clinical_risks(raw_text="腰椎间盘突出伴双下肢进行性肌力下降3天。"))
    assert "RF_016" in _alert_ids(scan_clinical_risks(raw_text="患者怀孕16周，腰痛要求腰骶部拔罐和口服布洛芬。"))


def test_exported_negation_and_quote_helpers() -> None:
    quoted = "很多患者想「明天再去」，但马尾综合征不能等，请立即急诊。"
    stripped = strip_quoted_segments(quoted)
    assert "明天再去" not in stripped

    safe_reply = "踩棉花感提示脊髓受压，严禁在家自行复位，也不可以去推拿馆做旋转复位。"
    idx = safe_reply.index("去推拿馆做旋转复位")
    assert is_context_negated(safe_reply, idx, idx + len("去推拿馆做旋转复位")) is True

    bypass_reply = "无需手术可以去推拿馆做旋转复位，禁忌谈不上。"
    idx2 = bypass_reply.index("去推拿馆做旋转复位")
    assert is_context_negated(bypass_reply, idx2, idx2 + len("去推拿馆做旋转复位")) is False


@pytest.mark.parametrize(
    "text",
    [
        "颈部推拿后出现眩晕、复视、走路不稳",
        "颈部正骨后突发剧烈枕颈痛、眩晕及复视",
        "颈椎扳法后突发眩晕、饮水呛咳及吞咽困难",
        "颈部按摩后新发步态不稳、构音障碍",
    ],
)
def test_r2_rf014_vertebral_artery_dissection_triggers(text: str) -> None:
    report = scan_clinical_risks(raw_text=text)
    assert "RF_014" in _alert_ids(report), f"Expected RF_014 for: {text}"
    assert report["has_critical_risk"] is True


@pytest.mark.parametrize(
    "text",
    [
        "昨日颈部推拿后感觉轻松，无头晕无复视",
        "昨日颈部推拿后感觉轻松，无头晕、复视",
        "昨日颈部按摩后未出现眩晕、走路不稳",
        "未行髋关节置换，未做ACL重建。",
        "未行手术，未做过ACL重建，未接受脊柱融合。",
        "未曾做ACL重建，未曾行髋关节置换，未实施内固定。",
        "排除骨折，无踩棉花感，否认二便失禁。",
    ],
)
def test_r2_surgical_procedure_negations_and_rf014_controls_zero_fp(text: str) -> None:
    report = scan_clinical_risks(raw_text=text)
    assert report["alerts"] == [], f"Unexpected alerts for negative text: {text}"


def test_r2_negation_additions_do_not_break_worsening_or_walking_positives() -> None:
    # Ensure 未行(?!走) does not suppress walking-related myelopathy signs and 未见缓解/好转/改善 stay positive
    assert "RF_001" in _alert_ids(scan_clinical_risks(raw_text="患者长期卧床未行走，现出现踩棉花感。"))
    assert "RF_002" in _alert_ids(scan_clinical_risks(raw_text="尿潴留未见改善。"))
    assert "RF_002" in _alert_ids(scan_clinical_risks(raw_text="鞍区麻木未见缓解。"))
