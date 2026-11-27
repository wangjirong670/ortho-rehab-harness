"""Tests for v0.4.0 dual-mode lossless draft cache and local case workspace (CRUD, revisions, API)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from app_copilot.case_store import (
    MAX_REVISIONS,
    append_followup_visit,
    delete_case,
    get_case_archives_dir,
    import_case_archive,
    list_cases,
    load_case,
    restore_case_revision,
    save_case,
    update_case_metadata,
)
from app_copilot.service import (
    format_copilot_response,
    recalculate_scale_in_state,
    regenerate_draft_in_state,
    remove_scale_from_state,
    run_copilot_analysis,
)
from backend.main import app
from core_tools.llm_client import LLMClient, LLMConfig


class _CountingMockClient(LLMClient):
    """Spy client counting real LLM draft generation calls in Mock mode."""

    def __init__(self) -> None:
        super().__init__(config=LLMConfig(mock_mode=True))
        self.extract_calls = 0
        self.chat_calls = 0

    def extract_structured_json(self, prompt_or_messages, **kwargs):
        self.extract_calls += 1
        return super().extract_structured_json(prompt_or_messages, **kwargs)

    def chat_completion(self, messages, **kwargs):
        self.chat_calls += 1
        return super().chat_completion(messages, **kwargs)


def test_dual_mode_draft_lossless_cache_and_stale_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    spy = _CountingMockClient()

    note = (
        "某男，58岁，颈肩僵硬3月，走路踩棉花感2周，无二便失禁。"
        "颈椎 JOA 17 分制：上肢运动功能 2 分，下肢运动功能 2 分。"
    )
    # 1. Initial analysis generates ward_round draft (1 extract + 1 chat)
    state = run_copilot_analysis(note, draft_mode="ward_round", mock_mode=True, client=spy)
    assert spy.extract_calls == 1
    assert spy.chat_calls == 1
    assert state["draft_mode"] == "ward_round"
    assert state["draft_cache_hit"] is False
    assert "ward_round" in state["drafts_by_mode"]
    assert state["draft_stale_by_mode"]["ward_round"] is False
    assert state["drafts_by_mode"]["ward_round"]["plain_text"]
    assert state["drafts_by_mode"]["ward_round"]["generated_at"]

    # 2. Switch to rehab_education -> cache miss -> generates rehab_education (chat_calls == 2)
    state_rehab = regenerate_draft_in_state(
        state, draft_mode="rehab_education", force=False, mock_mode=True, client=spy
    )
    assert spy.chat_calls == 2
    assert state_rehab["draft_cache_hit"] is False
    assert state_rehab["draft_mode"] == "rehab_education"
    assert set(state_rehab["drafts_by_mode"].keys()) == {"ward_round", "rehab_education"}
    assert state_rehab["draft_stale_by_mode"] == {"ward_round": False, "rehab_education": False}

    # 3. Switch back to ward_round with force=False -> ZERO LLM call (cache hit!)
    state_back = regenerate_draft_in_state(
        state_rehab, draft_mode="ward_round", force=False, mock_mode=True, client=spy
    )
    assert spy.chat_calls == 2  # NO extra LLM call!
    assert state_back["draft_cache_hit"] is True
    assert state_back["draft_mode"] == "ward_round"
    assert "上级医师查房汇报提纲" in state_back["draft"]["content"]

    # 4. Switch to rehab_education with force=False -> ZERO LLM call (cache hit!)
    state_rehab_2 = regenerate_draft_in_state(
        state_back, draft_mode="rehab_education", force=False, mock_mode=True, client=spy
    )
    assert spy.chat_calls == 2
    assert state_rehab_2["draft_cache_hit"] is True
    assert "分阶段康复宣教要点" in state_rehab_2["draft"]["content"]

    # 5. Force regenerate rehab_education -> calls LLM once (chat_calls == 3)
    state_forced = regenerate_draft_in_state(
        state_rehab_2, draft_mode="rehab_education", force=True, mock_mode=True, client=spy
    )
    assert spy.chat_calls == 3
    assert state_forced["draft_cache_hit"] is False

    # 6. Recalculating scale with refresh_draft=True updates ONLY target mode; other mode becomes stale
    state_recalc = recalculate_scale_in_state(
        state_forced,
        "JOA_CERVICAL_17",
        {
            "upper_limb_motor": 2,
            "lower_limb_motor": 2,
            "upper_limb_sensory": 1,
            "lower_limb_sensory": 1,
            "trunk_sensory": 1,
            "bladder_function": 2,
        },
        scale_index=0,
        refresh_draft=True,
        draft_mode="ward_round",
        mock_mode=True,
        client=spy,
    )
    assert spy.chat_calls == 4  # Only 1 call for ward_round, did NOT burn call for rehab_education
    assert state_recalc["draft_stale_by_mode"]["ward_round"] is False
    assert state_recalc["draft_stale_by_mode"]["rehab_education"] is True

    # Switching to stale rehab_education with force=False still returns cached draft without calling LLM, preserving stale=True
    state_stale_switch = regenerate_draft_in_state(
        state_recalc, draft_mode="rehab_education", force=False, mock_mode=True, client=spy
    )
    assert spy.chat_calls == 4
    assert state_stale_switch["draft_cache_hit"] is True
    assert state_stale_switch["draft_stale"] is True

    # Removing scale marks all cached modes stale
    state_removed = remove_scale_from_state(state_recalc, 0)
    assert state_removed["draft_stale"] is True
    assert state_removed["draft_stale_by_mode"] == {"ward_round": True, "rehab_education": True}


def test_case_store_crud_followup_restore_import_and_path_safety(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archives_dir = tmp_path / "case_archives"
    monkeypatch.setenv("ORTHO_CASE_ARCHIVES_DIR", str(archives_dir))
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    assert get_case_archives_dir() == archives_dir

    # Path traversal / invalid case_id blocked
    for bad_id in ("../secret", "a/b", "a\\b", "bad.id", "x" * 65):
        with pytest.raises(ValueError):
            save_case(raw_text="腰痛", case_id=bad_id)
        with pytest.raises(ValueError):
            load_case(bad_id)
    with pytest.raises(ValueError):
        load_case("")

    raw_1 = "某男，58岁，走路踩棉花感2周，无二便失禁。"
    st_1 = run_copilot_analysis(raw_1, draft_mode="ward_round", mock_mode=True)
    saved = save_case(
        raw_text=raw_1,
        draft_mode="ward_round",
        analysis_state=st_1,
        revision_note="初次分析归档",
    )
    cid = saved["case_id"]
    assert saved["schema_version"] == 1
    assert saved["revision_count"] == 1
    assert len(saved["revisions"]) == 1
    assert "极高危红旗" in saved["tags"]
    assert len(saved["title"]) <= 36

    # Silent update (record_revision=False) after switching draft does NOT increment revision_count
    st_2 = regenerate_draft_in_state(st_1, draft_mode="rehab_education", mock_mode=True)
    updated_silent = save_case(
        case_id=cid,
        raw_text=raw_1,
        draft_mode="rehab_education",
        analysis_state=st_2,
        record_revision=False,
    )
    assert updated_silent["revision_count"] == 1
    assert updated_silent["revisions"][0]["draft_modes_ready"] == ["ward_round", "rehab_education"]

    # Update title and tags
    renamed = update_case_metadata(cid, title="脊髓型颈椎病随访档案", tags=["重点随访"])
    assert renamed["title"] == "脊髓型颈椎病随访档案"
    assert renamed["tags"] == ["重点随访"]

    # Append follow-up visit
    followed = append_followup_visit(cid, "复查颈椎MRI示C4-5椎间盘突出压迫硬膜囊。", visit_date="2026-11-27")
    assert followed["revision_count"] == 2
    assert "【2026-11-27 复诊随访】" in followed["raw_text"]
    assert "复查颈椎MRI" in followed["raw_text"]
    assert followed["analysis_state"]["draft_stale"] is True

    # Restore revision 1
    restored = restore_case_revision(cid, 1)
    assert restored["revision_count"] == 3
    assert "复查颈椎MRI" not in restored["raw_text"]
    assert restored["analysis_state"]["draft_stale"] is False

    # Create a second low-risk case and test search + high_risk_only filter
    raw_low = "某男，30岁，久坐腰肌酸胀，无踩棉花感，无二便失禁，无发热，VAS 3 分。"
    st_low = run_copilot_analysis(raw_low, draft_mode="rehab_education", mock_mode=True)
    saved_low = save_case(raw_text=raw_low, draft_mode="rehab_education", analysis_state=st_low)

    all_list = list_cases()
    assert len(all_list["cases"]) == 2
    high_only = list_cases(high_risk_only=True)
    assert [c["case_id"] for c in high_only["cases"]] == [cid]
    search_hit = list_cases(q="脊髓型颈椎病")
    assert [c["case_id"] for c in search_hit["cases"]] == [cid]

    # Import collision allocates a new case_id without overwriting original
    exported = load_case(cid)
    imported = import_case_archive(exported)
    assert imported["case_id"] != cid
    assert imported["title"] == exported["title"]

    # Revision cap keeps at most MAX_REVISIONS
    for i in range(MAX_REVISIONS + 5):
        save_case(case_id=saved_low["case_id"], raw_text=f"{raw_low} v{i}", record_revision=True)
    capped = load_case(saved_low["case_id"])
    assert len(capped["revisions"]) == MAX_REVISIONS

    # Delete archive
    del_res = delete_case(imported["case_id"])
    assert del_res == {"deleted": True, "case_id": imported["case_id"]}
    with pytest.raises(FileNotFoundError):
        load_case(imported["case_id"])


def test_cases_and_draft_cache_fastapi_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    monkeypatch.setenv("ORTHO_CASE_ARCHIVES_DIR", str(tmp_path / "case_archives"))
    monkeypatch.setenv("ORTHO_AUTH_TOKEN", "case-secret-token")
    monkeypatch.setenv("ORTHO_APP_HOME", str(tmp_path / "app_home"))
    (tmp_path / "app_home").mkdir(parents=True, exist_ok=True)

    client = TestClient(app)
    headers = {"Authorization": "Bearer case-secret-token"}

    # Auth enforced on /api/v1/cases
    assert client.get("/api/v1/cases").status_code == 401

    # 1. Analyze + switch draft mode + cache hit on /api/v1/copilot/draft/regenerate
    ana = client.post(
        "/api/v1/copilot/analyze",
        headers=headers,
        json={"raw_text": "某男，58岁，走路踩棉花感2周，无发热。", "draft_mode": "ward_round", "mock_mode": True},
    )
    assert ana.status_code == 200
    st = ana.json()
    assert st["draft_cache_hit"] is False
    assert "ward_round" in st["drafts_by_mode"]

    gen_rehab = client.post(
        "/api/v1/copilot/draft/regenerate",
        headers=headers,
        json={"analysis_state": st, "draft_mode": "rehab_education", "force": False, "mock_mode": True},
    )
    assert gen_rehab.status_code == 200
    st2 = gen_rehab.json()
    assert st2["draft_cache_hit"] is False
    assert set(st2["drafts_by_mode"].keys()) == {"ward_round", "rehab_education"}

    hit_ward = client.post(
        "/api/v1/copilot/draft/regenerate",
        headers=headers,
        json={"analysis_state": st2, "draft_mode": "ward_round", "force": False, "mock_mode": True},
    )
    assert hit_ward.status_code == 200
    assert hit_ward.json()["draft_cache_hit"] is True
    assert hit_ward.json()["draft_mode"] == "ward_round"

    # 2. Save case -> list -> get -> patch -> followup -> restore -> export -> import -> delete
    save_resp = client.post(
        "/api/v1/cases",
        headers=headers,
        json={
            "raw_text": st2["raw_text"],
            "draft_mode": st2["draft_mode"],
            "analysis_state": st2,
            "revision_note": "FastAPI 归档测试",
        },
    )
    assert save_resp.status_code == 200
    case_obj = save_resp.json()
    cid = case_obj["case_id"]

    list_resp = client.get("/api/v1/cases?high_risk_only=true", headers=headers)
    assert list_resp.status_code == 200
    assert any(c["case_id"] == cid for c in list_resp.json()["cases"])

    patch_resp = client.patch(
        f"/api/v1/cases/{cid}",
        headers=headers,
        json={"title": "颈椎病例API测试", "tags": ["API", "重点"]},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["title"] == "颈椎病例API测试"

    fu_resp = client.post(
        f"/api/v1/cases/{cid}/followup",
        headers=headers,
        json={"visit_date": "2026-11-28", "followup_text": "步态较前稍稳，拟进一步查MRI。"},
    )
    assert fu_resp.status_code == 200
    assert "2026-11-28 复诊随访" in fu_resp.json()["raw_text"]

    res_resp = client.post(f"/api/v1/cases/{cid}/restore/1", headers=headers)
    assert res_resp.status_code == 200
    assert "2026-11-28 复诊随访" not in res_resp.json()["raw_text"]

    exp_resp = client.get(f"/api/v1/cases/{cid}/export", headers=headers)
    assert exp_resp.status_code == 200
    assert exp_resp.json()["case_id"] == cid

    imp_resp = client.post(
        "/api/v1/cases/import",
        headers=headers,
        files={"file": ("backup.json", exp_resp.content, "application/json")},
    )
    assert imp_resp.status_code == 200
    imported_id = imp_resp.json()["case_id"]
    assert imported_id != cid

    # Path traversal rejected with 400/404
    assert client.get("/api/v1/cases/..%2Fbad", headers=headers).status_code in (400, 404)

    assert client.delete(f"/api/v1/cases/{imported_id}", headers=headers).status_code == 200
    assert client.get(f"/api/v1/cases/{imported_id}", headers=headers).status_code == 404
