"""Unit and UI tests for v0.2.0 desktop copilot UX upgrades (Line 2)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from app_copilot.service import (
    build_clinical_handover_text,
    detect_potential_pii,
    extract_text_from_uploaded_file,
    markdown_to_plain_text,
    parse_discrete_score_options,
    remove_scale_from_state,
    run_copilot_analysis,
    run_fast_safety_scan,
    sanitize_clinical_pii,
    summarize_contraindicated_capsules,
)
from core_tools.calculator import load_scales_catalog
from core_tools.llm_client import format_llm_error_zh, probe_ollama_models

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_format_llm_error_zh_covers_all_common_failures_and_redacts_keys() -> None:
    zh_auth, raw_auth = format_llm_error_zh(
        "AuthenticationError: Incorrect API key provided: sk-secret123456",
        model="deepseek/deepseek-chat",
    )
    assert "API Key" in zh_auth
    assert "401" in zh_auth
    assert "sk-secret123456" not in raw_auth
    assert "sk-***" in raw_auth

    zh_conn, _ = format_llm_error_zh(
        "APIConnectionError: [WinError 10061] No connection could be made to localhost:11434",
        model="ollama_chat/qwen2.5:7b",
    )
    assert "Ollama" in zh_conn
    assert "ollama serve" in zh_conn

    zh_404, _ = format_llm_error_zh(
        "NotFoundError: model 'qwen2.5:7b' not found (404), try pulling it first",
        model="ollama_chat/qwen2.5:7b",
    )
    assert "ollama pull qwen2.5:7b" in zh_404

    zh_timeout, _ = format_llm_error_zh(
        "APITimeoutError: Request timed out after 60s",
        model="ollama_chat/qwen2.5:7b",
    )
    assert "LLM_TIMEOUT_SECONDS" in zh_timeout
    assert "180" in zh_timeout

    zh_json, _ = format_llm_error_zh(
        "JSON_PARSE_ERROR: 模型返回内容无法解析为合法 JSON",
        model="ollama_chat/qwen2.5:3b",
    )
    assert "JSON Schema" in zh_json


def test_probe_ollama_models_success_and_silent_fallback(monkeypatch) -> None:
    class _FakeResp:
        status = 200

        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"models": [{"name": "qwen2.5:7b"}, {"name": "deepseek-r1:8b"}]}
            ).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=1.5: _FakeResp())
    models = probe_ollama_models("http://localhost:11434/v1")
    assert models == ["ollama_chat/qwen2.5:7b", "ollama_chat/deepseek-r1:8b"]

    def _fail_urlopen(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _fail_urlopen)
    assert probe_ollama_models("http://localhost:11434") == []


def test_sanitize_clinical_pii_protects_medical_terms_and_redacts_explicit_pii() -> None:
    raw = (
        "姓名：张三，某男，58岁，住院号：ZY2026088，手机号 13812345678，"
        "身份证号 11010119680512341X。\n"
        "既往长期口服华法林，外用葛根汤与三七粉，主诉左膝痛3周，右踝扭伤2天。"
    )
    detected = detect_potential_pii(raw)
    assert detected["person_name"] == 1
    assert detected["record_id"] == 1
    assert detected["phone"] == 1
    assert detected["id_card"] == 1
    assert detected["total"] == 4

    cleaned, counts = sanitize_clinical_pii(raw)
    assert counts == detected
    assert "张三" not in cleaned
    assert "ZY2026088" not in cleaned
    assert "13812345678" not in cleaned
    assert "11010119680512341X" not in cleaned
    assert "[姓名已脱敏]" in cleaned
    assert "[编号已脱敏]" in cleaned
    assert "[手机号已脱敏]" in cleaned
    assert "[身份证号已脱敏]" in cleaned

    # Medical terms and anonymized phrases must remain completely intact
    for medical_term in ("华法林", "葛根汤", "三七粉", "左膝痛", "右踝扭伤", "某男，58岁"):
        assert medical_term in cleaned

    # Clean anonymized text yields 0 detections
    assert detect_potential_pii(cleaned)["total"] == 0


def test_markdown_to_plain_text_and_handover_export(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    md = (
        "# 上级医师查房汇报提纲\n\n"
        "> **脱敏患者画像**：某男，58岁\n\n"
        "## 一、安全红旗\n"
        "- **绝对禁忌**：`严禁颈椎旋转扳法`（参考[指南](https://example.com)）\n"
        "---\n"
        "*声明：仅限脱敏辅助*"
    )
    plain = markdown_to_plain_text(md)
    assert "#" not in plain
    assert "**" not in plain
    assert "`" not in plain
    assert "https://example.com" not in plain
    assert "上级医师查房汇报提纲" in plain
    assert "• 绝对禁忌：严禁颈椎旋转扳法（参考指南）" in plain

    state = run_copilot_analysis(
        "某男，58岁，走路踩棉花感2周。颈椎JOA：上肢运动2分，下肢运动2分，上肢感觉1分，下肢感觉1分，躯干感觉1分，膀胱功能2分。",
        mock_mode=True,
    )
    handover = build_clinical_handover_text(state)
    assert "【OrthoRehab 骨伤/康复临床交班摘要（脱敏辅助）】" in handover
    assert "RF_001" in handover
    assert "9 / 17" in handover


def test_extract_text_from_uploaded_txt_and_docx_zero_dependency() -> None:
    txt_bytes = "某女，62岁，左膝疼痛2周。".encode("utf-8-sig")
    assert extract_text_from_uploaded_file("note.txt", txt_bytes) == "某女，62岁，左膝疼痛2周。"

    # Build minimal valid .docx zip archive in memory
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>某男，55岁，颈肩僵硬1月。</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>VAS 疼痛评分 4 分。</w:t></w:r></w:p>"
        "</w:body>"
        "</w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", doc_xml.encode("utf-8"))

    extracted = extract_text_from_uploaded_file("record.docx", buf.getvalue())
    assert "某男，55岁，颈肩僵硬1月。" in extracted
    assert "VAS 疼痛评分 4 分。" in extracted


def test_fast_safety_scan_capsules_and_scale_helpers() -> None:
    state = run_fast_safety_scan("患者近两周走路踩棉花感，同时长期口服华法林并自服三七粉。")
    assert state["fast_scan_only"] is True
    assert state["scale_results"] == []
    ids = {a["rule_id"] for a in state["unified_alerts"]}
    assert {"RF_001", "DC_001"}.issubset(ids)

    capsules = summarize_contraindicated_capsules(state["unified_alerts"])
    assert "🚫 旋转扳法/暴力正骨" in capsules
    assert "🚫 活血药/止痛药叠加" in capsules

    catalog = load_scales_catalog()
    womac = next(s for s in catalog.scales if s.scale_id == "WOMAC_LIKERT_V3_1")
    womac_opts = parse_discrete_score_options(womac, womac.items[0])
    assert womac_opts == {0.0: "0 无", 1.0: "1 轻度", 2.0: "2 中度", 3.0: "3 重度", 4.0: "4 极重"}

    # Remove scale card cleanly updates unified_alerts and draft_stale
    full_state = run_copilot_analysis(
        "某男，58岁，颈肩僵硬。上肢运动2分，下肢运动2分，上肢感觉1分，下肢感觉1分，躯干感觉1分，膀胱功能2分。",
        mock_mode=True,
    )
    assert len(full_state["scale_results"]) == 1
    removed = remove_scale_from_state(full_state, 0)
    assert removed["scale_results"] == []
    assert removed["extracted_note"]["scale_params"] == []
    assert removed["draft_stale"] is True
