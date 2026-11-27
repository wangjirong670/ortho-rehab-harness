"""End-to-end REST & SSE tests for /api/v1/patients and /api/v1/patients/{id}/sessions/*."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

from backend.main import app


def _parse_sse(text: str) -> List[Tuple[str, Dict]]:
    events: List[Tuple[str, Dict]] = []
    cur_event = ""
    data_lines: List[str] = []
    for raw in text.splitlines():
        if raw.startswith(":"):
            continue
        if raw == "":
            if cur_event and data_lines:
                events.append((cur_event, json.loads("\n".join(data_lines))))
            cur_event = ""
            data_lines = []
            continue
        if raw.startswith("event:"):
            cur_event = raw.split(":", 1)[1].strip()
        elif raw.startswith("data:"):
            data_lines.append(raw.split(":", 1)[1].strip())
    if cur_event and data_lines:
        events.append((cur_event, json.loads("\n".join(data_lines))))
    return events


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    monkeypatch.delenv("ORTHO_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ORTHO_APP_HOME", str(tmp_path / "app_home"))
    monkeypatch.setenv("ORTHO_CASE_ARCHIVES_DIR", str(tmp_path / "case_archives"))
    (tmp_path / "app_home").mkdir(parents=True, exist_ok=True)
    (tmp_path / "case_archives").mkdir(parents=True, exist_ok=True)
    return TestClient(app)


def test_patient_sessions_sse_e2e_inheritance_education_scales_and_trace(
    api_client: TestClient,
) -> None:
    # 1. Create patient with display_name
    create_pt = api_client.post(
        "/api/v1/patients",
        json={
            "patient_id": "PT_E2E_01",
            "display_name": "302床 张某",
            "body": "中年患者，颈肩部酸紧。",
        },
    )
    assert create_pt.status_code == 201
    pt_data = create_pt.json()
    assert pt_data["patient_id"] == "PT_E2E_01"
    assert pt_data["revision"] == 1

    # 2. Create Session 1 (initial visit triggering RF_001)
    s1_resp = api_client.post(
        "/api/v1/patients/PT_E2E_01/sessions",
        json={"title": "初诊脊髓症状排查"},
    )
    assert s1_resp.status_code == 201
    s1_id = s1_resp.json()["session_id"]

    with api_client.stream(
        "POST",
        f"/api/v1/patients/PT_E2E_01/sessions/{s1_id}/chat",
        json={
            "content": "患者颈椎不适，近期走路踩棉花感、双手扣纽扣困难，想做颈椎旋转扳法可以吗？",
            "output_mode": "full",
            "mock_mode": True,
        },
    ) as stream_resp:
        assert stream_resp.status_code == 200
        raw_sse = "".join(stream_resp.iter_text())

    sse_events = _parse_sse(raw_sse)
    ev_names = [name for name, _ in sse_events]
    assert "step" in ev_names
    assert "alert" in ev_names
    assert "content" in ev_names
    assert "verification" in ev_names
    assert ev_names[-1] == "done"
    # Alert must arrive BEFORE content (D10)
    assert ev_names.index("alert") < ev_names.index("content")

    # Verify RF_001 was auto-persisted into PATIENT.md
    pt_after_s1 = api_client.get("/api/v1/patients/PT_E2E_01").json()
    active_rfs = [
        rf for rf in pt_after_s1["card"]["active_red_flags"] if rf["status"] == "active"
    ]
    assert any(rf["id"] == "RF_001" for rf in active_rfs)

    # Silent delete via PUT must return 400
    bad_card = dict(pt_after_s1["card"])
    bad_card["active_red_flags"] = []
    bad_put = api_client.put(
        "/api/v1/patients/PT_E2E_01",
        json={"card": bad_card, "body": pt_after_s1["body"]},
    )
    assert bad_put.status_code == 400

    # Commit medication change
    commit_resp = api_client.post(
        "/api/v1/patients/PT_E2E_01/commit",
        json={
            "changes": {
                "add_medications": [
                    {"name": "阿司匹林", "status": "active", "source": "clinician"}
                ]
            }
        },
    )
    assert commit_resp.status_code == 200
    assert commit_resp.json()["revision"] >= 3

    # 3. Create Session 2 (follow-up without re-mentioning RF_001 -> must inherit RF_001)
    s2_resp = api_client.post(
        "/api/v1/patients/PT_E2E_01/sessions",
        json={"title": "复诊追踪"},
    )
    assert s2_resp.status_code == 201
    s2_id = s2_resp.json()["session_id"]

    with api_client.stream(
        "POST",
        f"/api/v1/patients/PT_E2E_01/sessions/{s2_id}/chat",
        json={
            "content": "今日复诊脖子还有点僵，没有新发热。",
            "output_mode": "full",
            "mock_mode": True,
        },
    ) as s2_stream:
        assert s2_stream.status_code == 200
        s2_sse = _parse_sse("".join(s2_stream.iter_text()))

    alert_ev = next(payload for name, payload in s2_sse if name == "alert")
    assert "RF_001" in alert_ev.get("inherited_red_flags", [])
    assert any(
        a.get("rule_id") == "RF_001" and a.get("inherited_from_card")
        for a in alert_ev.get("unified_alerts", [])
    )

    # 4. Generate standalone patient education document
    edu_resp = api_client.post(
        f"/api/v1/patients/PT_E2E_01/sessions/{s2_id}/education",
        json={"mock_mode": True},
    )
    assert edu_resp.status_code == 200
    edu_data = edu_resp.json()
    assert edu_data["mode"] == "rehab_education"
    assert edu_data["safety_check"]["passed"] is True
    assert "RF_001" in edu_data["inherited_red_flags"]

    # 5. Deterministic scale entry (partial missing must NOT zero-fill; severe JOA triggers RF_001)
    partial_joa = api_client.post(
        f"/api/v1/patients/PT_E2E_01/sessions/{s2_id}/scales",
        json={"scale_id": "JOA_CERVICAL_17", "updated_params": {"upper_limb_motor": 2}},
    )
    assert partial_joa.status_code == 200
    partial_sc = partial_joa.json()["scale_result"]
    assert partial_sc["status"] == "error"
    assert partial_sc["error_code"] == "VALIDATION_ERROR"
    assert "lower_limb_motor" in partial_sc["missing_fields"]

    full_joa = api_client.post(
        f"/api/v1/patients/PT_E2E_01/sessions/{s2_id}/scales",
        json={
            "scale_id": "JOA_CERVICAL_17",
            "updated_params": {
                "upper_limb_motor": 1,
                "lower_limb_motor": 1,
                "upper_limb_sensory": 1,
                "lower_limb_sensory": 1,
                "trunk_sensory": 1,
                "bladder_function": 2,
            },
        },
    )
    assert full_joa.status_code == 200
    full_sc = full_joa.json()["scale_result"]
    assert full_sc["status"] == "ok"
    assert full_sc["total_score"] == 7.0
    assert full_sc["red_flag_triggered"] is True

    # 6. Audit trace & history restore check
    trace_resp = api_client.get(f"/api/v1/patients/PT_E2E_01/sessions/{s2_id}/trace")
    assert trace_resp.status_code == 200
    trace_types = {ev["type"] for ev in trace_resp.json()["events"]}
    assert {"session_meta", "user_input", "extraction", "scan", "scale", "generation", "verification", "education"}.issubset(trace_types)

    hist_resp = api_client.get("/api/v1/patients/PT_E2E_01/history")
    assert hist_resp.status_code == 200
    assert len(hist_resp.json()["revisions"]) >= 3

    restore_resp = api_client.post("/api/v1/patients/PT_E2E_01/restore/1")
    assert restore_resp.status_code == 200
    assert restore_resp.json()["card"]["active_red_flags"] == []
