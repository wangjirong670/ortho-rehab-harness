"""Unit tests for core_tools/llm_client.py (Mock Mode + LiteLLM routing)."""

import sys
import types
from typing import List

from pydantic import BaseModel

from core_tools.llm_client import (
    LLMClient,
    LLMConfig,
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
