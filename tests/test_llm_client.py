"""Unit tests for core_tools/llm_client.py (Mock Mode + LiteLLM routing + security hardening)."""

import json
import sys
import types
from typing import List

import pytest
from pydantic import BaseModel

from app_copilot.prompts import CopilotExtractedNote, build_extraction_messages
from core_tools.llm_client import (
    LLMClient,
    LLMConfig,
    _extract_raw_note_body,
    chat_completion,
    extract_structured_json,
)
from core_tools.rule_scanner import scan_clinical_risks


class _SimpleExtractedNote(BaseModel):
    chief_complaint: str
    symptoms: List[dict]
    medications: List[dict]


def test_mock_chat_completion_requires_no_api_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    resp = chat_completion(
        messages=[
            {"role": "user", "content": "最近走路踩棉花感，想去推拿馆做颈椎旋转扳法正骨可以吗？"}
        ]
    )
    assert resp["mock"] is True
    assert resp["model"].startswith("mock/")
    assert "严禁" in resp["content"] or "脊髓" in resp["content"]


def test_mock_extract_structured_json_end_to_end_with_rule_scanner(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    note = (
        "58岁男性，颈肩僵硬2月，走路踩棉花感，否认二便失禁。"
        "房颤史，停用华法林半年，昨日自行服三七粉。VAS: 6"
    )
    extracted = extract_structured_json(note, schema_model=_SimpleExtractedNote)
    assert extracted["status"] == "ok"
    assert extracted["mock"] is True

    facts = extracted["data"]
    # End-to-end deterministic scan using Mock-extracted structured facts
    report = scan_clinical_risks(raw_text=note, structured_facts=facts)
    alert_ids = {a["rule_id"] for a in report["alerts"]}
    assert "RF_001" in alert_ids
    assert "RF_002" not in alert_ids
    assert "DC_001" not in alert_ids


def test_litellm_routing_and_json_fence_stripping(monkeypatch) -> None:
    captured_kwargs = {}

    def _fake_completion(**kwargs):
        captured_kwargs.update(kwargs)
        msg = types.SimpleNamespace(content='```json\n{"chief_complaint": "腰痛", "symptoms": [], "medications": []}\n```')
        choice = types.SimpleNamespace(message=msg)
        usage = types.SimpleNamespace(prompt_tokens=12, completion_tokens=18, total_tokens=30)
        return types.SimpleNamespace(choices=[choice], usage=usage)

    fake_litellm = types.SimpleNamespace(completion=_fake_completion)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    client = LLMClient(
        config=LLMConfig(
            default_model="dashscope/qwen-plus",
            temperature=0.05,
            timeout_seconds=30,
            max_retries=1,
            mock_mode=False,
        )
    )
    res = client.extract_structured_json("30岁久坐腰痛", schema_model=_SimpleExtractedNote)
    assert res["status"] == "ok"
    assert res["mock"] is False
    assert res["data"]["chief_complaint"] == "腰痛"
    assert captured_kwargs["model"] == "dashscope/qwen-plus"
    assert captured_kwargs["response_format"] == {"type": "json_object"}


def test_api_base_not_leaked_to_other_providers(monkeypatch) -> None:
    """P0-4: OPENAI_API_BASE must NEVER leak to non-OpenAI providers."""
    captured_calls = []

    def _fake_completion(**kwargs):
        captured_calls.append(dict(kwargs))
        msg = types.SimpleNamespace(content="ok")
        choice = types.SimpleNamespace(message=msg)
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return types.SimpleNamespace(choices=[choice], usage=usage)

    fake_litellm = types.SimpleNamespace(completion=_fake_completion)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    monkeypatch.setenv("OPENAI_API_BASE", "https://relay.example-third-party.com/v1")
    for env_var in (
        "DEEPSEEK_API_BASE",
        "DASHSCOPE_API_BASE",
        "ZAI_API_BASE",
        "ZHIPUAI_API_BASE",
        "MOONSHOT_API_BASE",
        "ANTHROPIC_API_BASE",
        "GEMINI_API_BASE",
    ):
        monkeypatch.delenv(env_var, raising=False)

    client = LLMClient(config=LLMConfig.from_env())

    non_openai_models = [
        "deepseek/deepseek-chat",
        "dashscope/qwen-plus",
        "zai/glm-4-plus",
        "moonshot/moonshot-v1-8k",
        "anthropic/claude-3-5-sonnet-latest",
        "gemini/gemini-2.5-pro",
    ]
    for model_name in non_openai_models:
        captured_calls.clear()
        client.chat_completion([{"role": "user", "content": "hi"}], model=model_name, mock_mode=False)
        assert len(captured_calls) == 1
        assert "api_base" not in captured_calls[0], f"OPENAI_API_BASE leaked to {model_name}"

    openai_compatible_models = [
        "openai/gpt-4o-mini",
        "custom_openai/my-local-model",
        "ollama/qwen2.5:14b",
        "vllm/llama3-8b",
        "hosted_vllm/qwen2.5",
        "gpt-4o",
    ]
    for model_name in openai_compatible_models:
        captured_calls.clear()
        client.chat_completion([{"role": "user", "content": "hi"}], model=model_name, mock_mode=False)
        assert captured_calls[0].get("api_base") == "https://relay.example-third-party.com/v1"

    # Explicit api_base argument from caller is respected even for non-OpenAI providers
    captured_calls.clear()
    client.chat_completion(
        [{"role": "user", "content": "hi"}],
        model="deepseek/deepseek-chat",
        mock_mode=False,
        api_base="https://explicit-deepseek.example.com/v1",
    )
    assert captured_calls[0].get("api_base") == "https://explicit-deepseek.example.com/v1"

    # Provider-specific *_API_BASE env vars work independently of OPENAI_API_BASE
    provider_env_cases = [
        ("deepseek/deepseek-chat", "DEEPSEEK_API_BASE", "https://ds.example.com/v1"),
        ("dashscope/qwen-plus", "DASHSCOPE_API_BASE", "https://dash.example.com/v1"),
        ("zai/glm-4-plus", "ZAI_API_BASE", "https://zai.example.com/v1"),
        ("moonshot/moonshot-v1-8k", "MOONSHOT_API_BASE", "https://moon.example.com/v1"),
        ("anthropic/claude-3-5-sonnet-latest", "ANTHROPIC_API_BASE", "https://anth.example.com/v1"),
        ("gemini/gemini-2.5-pro", "GEMINI_API_BASE", "https://gem.example.com/v1"),
    ]
    for model_name, env_name, env_url in provider_env_cases:
        monkeypatch.setenv(env_name, env_url)
        captured_calls.clear()
        client.chat_completion([{"role": "user", "content": "hi"}], model=model_name, mock_mode=False)
        assert captured_calls[0].get("api_base") == env_url


def test_litellm_exception_caught_and_redacted(monkeypatch) -> None:
    """P1-6: litellm.completion exceptions must not crash unhandled and must redact secrets."""
    def _failing_completion(**kwargs):
        raise RuntimeError("401 Unauthorized: incorrect API key provided: sk-secret123456abcd")

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_failing_completion))
    client = LLMClient(config=LLMConfig(default_model="deepseek/deepseek-chat", mock_mode=False))

    chat_res = client.chat_completion([{"role": "user", "content": "test"}], mock_mode=False)
    assert chat_res["status"] == "error"
    assert chat_res["error_code"] == "LLM_CALL_ERROR"
    assert "sk-secret123456abcd" not in chat_res["message"]
    assert "sk-***" in chat_res["message"]

    ext_res = client.extract_structured_json("test", schema_model=CopilotExtractedNote, mock_mode=False)
    assert ext_res["status"] == "error"
    assert ext_res["error_code"] == "LLM_CALL_ERROR"
    assert "sk-secret123456abcd" not in ext_res["message"]


@pytest.mark.parametrize(
    "dirty_payload",
    [
        '[{"chief_complaint": "腰痛"}]',
        '{"chief_complaint": "腰痛", "structured_facts": {"symptoms": [{"name": "踩棉花感", "polarity": "positive"}], "medications": []}}',
        '{"chief_complaint": "腰痛", "scale_params": "VAS 5分"}',
        '{"chief_complaint": "腰痛", "timeline_events": [{"date": "今日", "event": null}]}',
        '{"chief_complaint": "腰痛", "structured_facts": {"symptoms": [], "medications": [{"name": "华法林", "status": "stopped", "stopped_days_ago": -3}]}}',
        '{"chief_complaint": "腰痛", "structured_facts": {"symptoms": [], "medications": [{"name": "华法林", "status": "stopped", "stopped_days_ago": NaN}]}}',
        '{"chief_complaint": "腰痛", "structured_facts": {"symptoms": {"踩棉花感": "present"}, "medications": []}}',
    ],
)
def test_dirty_llm_outputs_trigger_single_repair_retry_and_return_structured_error(
    monkeypatch, dirty_payload: str
) -> None:
    """P1-6: All 7 dirty output shapes fail gracefully with structured error after 1 repair retry."""
    call_count = 0

    def _dirty_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        msg = types.SimpleNamespace(content=dirty_payload)
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice], usage=None)

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_dirty_completion))
    client = LLMClient(config=LLMConfig(default_model="deepseek/deepseek-chat", mock_mode=False))

    res = client.extract_structured_json("患者腰痛", schema_model=CopilotExtractedNote, mock_mode=False)
    assert call_count == 2  # initial call + 1 targeted repair retry
    assert res["status"] == "error"
    assert res["error_code"] == "SCHEMA_VALIDATION_ERROR"
    assert res["data"] is None
    assert res.get("retried") is True


def test_targeted_repair_retry_recovers_valid_json(monkeypatch) -> None:
    """P1-6: ValidationError on attempt 1 is repaired on attempt 2 with error summary."""
    seen_messages = []

    def _recovering_completion(**kwargs):
        seen_messages.append(kwargs["messages"])
        if len(seen_messages) == 1:
            content = '{"chief_complaint": "颈痛", "structured_facts": {"symptoms": [{"name": "踩棉花感", "polarity": "positive"}]}}'
        else:
            content = '{"chief_complaint": "颈痛", "structured_facts": {"symptoms": [{"name": "踩棉花感", "polarity": "present"}]}}'
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_recovering_completion))
    client = LLMClient(config=LLMConfig(default_model="deepseek/deepseek-chat", mock_mode=False))

    res = client.extract_structured_json("颈痛踩棉花感", schema_model=CopilotExtractedNote, mock_mode=False)
    assert len(seen_messages) == 2
    assert "校验错误摘要" in seen_messages[1][-1]["content"]
    assert res["status"] == "ok"
    assert res.get("retried") is True
    assert res["data"]["structured_facts"]["symptoms"][0]["polarity"] == "present"


def test_reasoning_think_block_and_example_json_preceding_real_json(monkeypatch) -> None:
    """P1-6: Reasoning <think> blocks or earlier example JSON must not hijack final JSON extraction."""
    raw_output = (
        '<think>先看一个示例格式：{"chief_complaint": "错误示例", "symptoms": "bad_type"}\n'
        '```json\n{"broken": true}\n```</think>\n'
        '以下为正式抽取结果：\n'
        '```json\n'
        '{"chief_complaint": "真实颈肩痛", "structured_facts": {"symptoms": [{"name": "踩棉花感", "polarity": "present"}], "medications": []}}\n'
        '```'
    )

    def _thinking_completion(**kwargs):
        msg = types.SimpleNamespace(content=raw_output)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_thinking_completion))
    client = LLMClient(config=LLMConfig(default_model="deepseek/deepseek-reasoner", mock_mode=False))

    res = client.extract_structured_json("颈肩痛踩棉花感", schema_model=CopilotExtractedNote, mock_mode=False)
    assert res["status"] == "ok"
    assert res["data"]["chief_complaint"] == "真实颈肩痛"
    assert res["data"]["structured_facts"]["symptoms"][0]["name"] == "踩棉花感"


def test_llm_config_from_env_invalid_numeric_fallbacks(monkeypatch) -> None:
    """P2-4: Non-numeric env vars must safely fall back to defaults instead of crashing."""
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "abc")
    monkeypatch.setenv("LLM_MAX_RETRIES", "bad")
    monkeypatch.setenv("LLM_TEMPERATURE", "NaN")
    cfg = LLMConfig.from_env()
    assert cfg.timeout_seconds == 60
    assert cfg.max_retries == 2
    assert cfg.temperature == 0.1

    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("LLM_TIMEOUT", "not_a_number")
    cfg2 = LLMConfig.from_env()
    assert cfg2.timeout_seconds == 60


def test_mock_acl_question_poc_does_not_supersede_rf007(monkeypatch) -> None:
    """P0-2 PoC #5: Family asking about running in 6 months must NOT clear RF_007."""
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    poc_note = (
        "【术后第10天 骨科】右膝ACL重建联合半月板缝合术后，早期保护期，严禁跑跳，患者已自行丢拐全负重。\n"
        "【今日】家属询问：批准跑步要等到半年后吗？"
    )
    msgs = build_extraction_messages(poc_note)
    assert _extract_raw_note_body(msgs[1]["content"]) == poc_note

    ext = extract_structured_json(msgs, schema_model=CopilotExtractedNote, mock_mode=True)
    assert ext["status"] == "ok"
    snap = ext["data"]["current_state_snapshot"]
    assert snap["superseded_conditions"] == []

    report = scan_clinical_risks(raw_text=poc_note, structured_facts=ext["data"]["structured_facts"])
    assert any(a["rule_id"] == "RF_007" for a in report["alerts"])


def test_mock_drug_cessation_scope_negation_and_resume(monkeypatch) -> None:
    """P0-3: Cross-drug switch, negated stop markers, planned stops, and resumed drugs must not suppress active conflicts."""
    monkeypatch.setenv("LLM_MOCK_MODE", "true")

    # 1. Cross-drug switch: warfarin stopped 3 months ago, switched to rivaroxaban
    note_switch = "华法林已停3月改用利伐沙班抗凝，今日自行加服三七粉。"
    ext_switch = extract_structured_json(note_switch, schema_model=CopilotExtractedNote, mock_mode=True)["data"]
    meds_switch = {m["name"]: m for m in ext_switch["structured_facts"]["medications"]}
    assert meds_switch["华法林"]["status"] == "stopped"
    assert meds_switch["华法林"]["stopped_days_ago"] == 90.0
    assert meds_switch["利伐沙班"]["status"] == "active"
    assert meds_switch["利伐沙班"]["stopped_days_ago"] is None

    # 2. Negated stop marker with intervening words + active duration ("已规律服用6个月")
    for neg_note in (
        "医生叮嘱切勿自行停用华法林，已规律服用6个月，今日扭伤加服三七粉。",
        "千万别停用华法林，已规律服用6个月，今日扭伤加服三七粉。",
        "华法林没停药，已规律服用6个月，今日扭伤加服三七粉。",
        "建议择期停用华法林，今日扭伤加服三七粉。",
    ):
        ext_neg = extract_structured_json(neg_note, schema_model=CopilotExtractedNote, mock_mode=True)["data"]
        meds_neg = {m["name"]: m for m in ext_neg["structured_facts"]["medications"]}
        assert meds_neg["华法林"]["status"] == "active", f"Failed for: {neg_note}"
        assert meds_neg["华法林"]["stopped_days_ago"] is None

    # 3. Stopped then resumed
    note_resume = "华法林停药10天后已于昨日恢复服用，今日加服三七粉。"
    ext_resume = extract_structured_json(note_resume, schema_model=CopilotExtractedNote, mock_mode=True)["data"]
    meds_resume = {m["name"]: m for m in ext_resume["structured_facts"]["medications"]}
    assert meds_resume["华法林"]["status"] == "active"
    assert meds_resume["华法林"]["stopped_days_ago"] is None
