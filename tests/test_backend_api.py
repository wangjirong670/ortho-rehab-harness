"""API contract, auth, SSE streaming, and desktop launcher tests for FastAPI backend."""

from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

import desktop_app
import launcher
from app_benchmark.service import extract_violation_spans
from backend.main import app
from tests.test_bench_ui import build_sample_report


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    monkeypatch.delenv("ORTHO_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ORTHO_BENCH_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("ORTHO_APP_HOME", str(tmp_path / "app_home"))
    (tmp_path / "app_home").mkdir(parents=True, exist_ok=True)
    return TestClient(app)


def test_health_and_optional_bearer_auth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    monkeypatch.setenv("ORTHO_AUTH_TOKEN", "secret-token-xyz")
    monkeypatch.setenv("ORTHO_APP_MODE", "local_desktop")
    monkeypatch.setenv("ORTHO_APP_HOME", str(tmp_path))

    c = TestClient(app)
    health_resp = c.get("/api/v1/health")
    assert health_resp.status_code == 200
    health = health_resp.json()
    assert health["status"] == "ok"
    assert health["version"] == "0.4.0"
    assert health["mode"] == "local_desktop"
    assert health["auth_enabled"] is True
    assert health["mock_mode"] is True
    assert isinstance(health["default_model"], str)
    assert isinstance(health["frontend_ready"], bool)

    # Protected endpoints must reject missing or invalid token
    unauth = c.get("/api/v1/copilot/demo-cases")
    assert unauth.status_code == 401

    bad_auth = c.get(
        "/api/v1/kb/overview",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert bad_auth.status_code == 401

    ok_auth = c.get(
        "/api/v1/copilot/demo-cases",
        headers={"Authorization": "Bearer secret-token-xyz"},
    )
    assert ok_auth.status_code == 200
    assert len(ok_auth.json()["cases"]) >= 3


def test_copilot_upload_pii_fast_scan_analyze_scales_and_handover(client: TestClient) -> None:
    # 1. Demo cases
    demo_resp = client.get("/api/v1/copilot/demo-cases")
    assert demo_resp.status_code == 200
    cases = demo_resp.json()["cases"]
    assert len(cases) >= 3
    first_case = cases[0]
    assert {"id", "title", "tag", "raw_text", "default_draft_mode"}.issubset(first_case.keys())

    # 2. Upload .txt and .docx
    up_txt = client.post(
        "/api/v1/copilot/upload",
        files={"file": ("note.txt", "某男，58岁，颈肩僵硬。".encode("utf-8"), "text/plain")},
    )
    assert up_txt.status_code == 200
    assert up_txt.json()["filename"] == "note.txt"
    assert up_txt.json()["char_count"] > 0

    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>某女，64岁，右膝痛。</w:t></w:r></w:p></w:body>"
        "</w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", doc_xml.encode("utf-8"))
    up_docx = client.post(
        "/api/v1/copilot/upload",
        files={"file": ("case.docx", buf.getvalue(), "application/octet-stream")},
    )
    assert up_docx.status_code == 200
    assert "某女，64岁，右膝痛。" in up_docx.json()["text"]

    # 3. PII detect & sanitize
    raw_pii = "姓名：张三，手机号 13812345678，住院号：ZY2026001，走路踩棉花感2周。"
    pii_det = client.post("/api/v1/copilot/pii/detect", json={"text": raw_pii})
    assert pii_det.status_code == 200
    det_json = pii_det.json()
    assert det_json["has_pii"] is True
    assert len(det_json["findings"]) == 3

    pii_san = client.post("/api/v1/copilot/pii/sanitize", json={"text": raw_pii})
    assert pii_san.status_code == 200
    san_json = pii_san.json()
    assert san_json["changed"] is True
    assert "张三" not in san_json["sanitized_text"]
    assert "13812345678" not in san_json["sanitized_text"]
    assert "踩棉花感" in san_json["sanitized_text"]

    # 4. Fast scan
    fast_resp = client.post(
        "/api/v1/copilot/fast-scan",
        json={"raw_text": "患者走路踩棉花感2周，同时服用华法林和三七粉。"},
    )
    assert fast_resp.status_code == 200
    fast_data = fast_resp.json()
    assert fast_data["fast_scan_only"] is True
    assert len(fast_data["unified_alerts"]) >= 2
    assert len(fast_data["contraindicated_capsules"]) >= 1

    # 5. Full analyze (partial JOA -> recalculate -> draft regenerate -> remove -> handover)
    partial_note = (
        "某男，58岁，颈肩酸胀1个月，无发热，无二便失禁。"
        "颈椎 JOA 17 分制：上肢运动功能 2 分，下肢运动功能 2 分。"
    )
    analyze_resp = client.post(
        "/api/v1/copilot/analyze",
        json={"raw_text": partial_note, "draft_mode": "ward_round", "mock_mode": True},
    )
    assert analyze_resp.status_code == 200
    state = analyze_resp.json()
    assert state["fast_scan_only"] is False
    assert state["draft_mode"] == "ward_round"
    assert state["scale_results"][0]["status"] == "error"

    recalc_resp = client.post(
        "/api/v1/copilot/scales/recalculate",
        json={
            "analysis_state": state,
            "scale_id": "JOA_CERVICAL_17",
            "updated_params": {
                "upper_limb_motor": 2,
                "lower_limb_motor": 2,
                "upper_limb_sensory": 1,
                "lower_limb_sensory": 1,
                "trunk_sensory": 1,
                "bladder_function": 2,
            },
            "scale_index": 0,
            "refresh_draft": True,
            "mock_mode": True,
        },
    )
    assert recalc_resp.status_code == 200
    recalc_state = recalc_resp.json()
    assert recalc_state["scale_results"][0]["status"] == "ok"
    assert recalc_state["scale_results"][0]["total_score"] == 9
    assert recalc_state["draft_stale"] is False
    assert any("RF_001" in str(a.get("rule_id")) for a in recalc_state["unified_alerts"])

    regen_resp = client.post(
        "/api/v1/copilot/draft/regenerate",
        json={"analysis_state": recalc_state, "draft_mode": "rehab_education", "mock_mode": True},
    )
    assert regen_resp.status_code == 200
    regen_state = regen_resp.json()
    assert regen_state["draft_mode"] == "rehab_education"
    assert "分阶段康复宣教要点" in regen_state["draft"]["content"]

    handover_resp = client.post(
        "/api/v1/copilot/handover",
        json={"analysis_state": regen_state},
    )
    assert handover_resp.status_code == 200
    assert "OrthoRehab" in handover_resp.json()["handover_text"]
    assert len(handover_resp.json()["draft_plain_text"]) > 10

    remove_resp = client.post(
        "/api/v1/copilot/scales/remove",
        json={"analysis_state": regen_state, "scale_index": 0},
    )
    assert remove_resp.status_code == 200
    assert remove_resp.json()["scale_results"] == []
    assert remove_resp.json()["draft_stale"] is True


def test_benchmark_reports_import_export_highlights_and_sse_stream(
    client: TestClient, tmp_path: Path
) -> None:
    # 1. Extract violation spans unit check
    sample_report = build_sample_report(report_id="api_bench_20261126")
    bare_case_0 = sample_report["runs"][0]["results"][0]
    spans = extract_violation_spans(
        bare_case_0["turns"][1]["assistant_msg"],
        bare_case_0,
    )
    assert len(spans) == 1
    assert spans[0]["text"] == "在家自行复位"
    assert spans[0]["start"] < spans[0]["end"]
    assert "脊髓型颈椎病" in spans[0]["reason"]

    # 2. Import report JSON
    import_resp = client.post(
        "/api/v1/benchmark/reports/import",
        files={
            "file": (
                "sample.json",
                json.dumps(sample_report, ensure_ascii=False).encode("utf-8"),
                "application/json",
            )
        },
    )
    assert import_resp.status_code == 200
    filename = import_resp.json()["filename"]
    assert filename.endswith(".json")

    # 3. List reports
    list_resp = client.get("/api/v1/benchmark/reports")
    assert list_resp.status_code == 200
    reports = list_resp.json()["reports"]
    assert any(r["filename"] == filename for r in reports)

    # 4. Get report detail (pairs, deltas, side_by_side_cases, highlights)
    detail_resp = client.get(f"/api/v1/benchmark/reports/{filename}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["filename"] == filename
    assert detail["meta"]["case_count"] == 4
    assert len(detail["pairs"]) == 1
    deltas = detail["pairs"][0]["deltas"]
    assert deltas["overall_pass_rate"] == 75.0
    assert deltas["safety_veto_rate"] == -25.0
    assert deltas["red_flag_interception"] == 100.0
    assert len(detail["dimension_rows"]) == 4

    sbs_rf001 = next(c for c in detail["side_by_side_cases"] if c["case_id"] == "ORTHO_REDFLAG_001")
    assert sbs_rf001["rescued"] is True
    assert len(sbs_rf001["bare_highlights"]["1"]) == 1
    assert sbs_rf001["bare_highlights"]["1"][0]["text"] == "在家自行复位"
    assert sbs_rf001["harness_highlights"]["1"] == []

    # 5. Export CSV / JSON / Markdown
    for fmt, expected_type in (
        ("csv", "text/csv"),
        ("json", "application/json"),
        ("markdown", "text/markdown"),
    ):
        exp = client.get(f"/api/v1/benchmark/reports/{filename}/export?format={fmt}")
        assert exp.status_code == 200
        assert expected_type in exp.headers["content-type"]
        assert len(exp.content) > 10

    # 6. Run live mock benchmark and consume SSE progress + done
    run_resp = client.post(
        "/api/v1/benchmark/runs",
        json={
            "models": ["deepseek/deepseek-chat"],
            "modes": ["bare_llm", "harness_copilot"],
            "mock_mode": True,
            "include_private": False,
        },
    )
    assert run_resp.status_code == 200
    job_id = run_resp.json()["job_id"]

    events: list[tuple[str, Dict[str, Any]]] = []
    with client.stream("GET", f"/api/v1/benchmark/runs/{job_id}/stream") as stream_resp:
        assert stream_resp.status_code == 200
        cur_event = ""
        for line in stream_resp.iter_lines():
            if line.startswith("event: "):
                cur_event = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                data_obj = json.loads(line[len("data: ") :])
                events.append((cur_event, data_obj))

    progress_events = [d for e, d in events if e == "progress"]
    done_events = [d for e, d in events if e == "done"]
    assert len(progress_events) >= 2
    assert progress_events[-1]["completed"] == progress_events[-1]["total"]
    assert len(done_events) == 1
    assert done_events[0]["filename"].endswith(".json")


def test_kb_overview_and_settings_endpoints(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 1. KB overview
    kb_resp = client.get("/api/v1/kb/overview")
    assert kb_resp.status_code == 200
    kb_data = kb_resp.json()
    assert len(kb_data["red_flags"]["rules"]) >= 10
    assert len(kb_data["drug_conflicts"]["conflicts"]) >= 5
    assert len(kb_data["scales"]["scales"]) == 4
    assert "STOP_AND_REFER" in kb_data["labels"]["clinical_actions"]

    # 2. Settings GET & PUT (.env preservation of unpassed keys)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-initial-secret-9999")
    monkeypatch.setenv("COPILOT_DEFAULT_MODEL", "deepseek/deepseek-chat")
    env_path = tmp_path / "app_home" / ".env"
    env_path.write_text(
        "COPILOT_DEFAULT_MODEL=deepseek/deepseek-chat\n"
        "DEEPSEEK_API_KEY=sk-initial-secret-9999\n"
        "LLM_MOCK_MODE=true\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("backend.routers.settings.probe_ollama_models", lambda timeout=0.6: [])
    get_res = client.get("/api/v1/settings/llm")
    assert get_res.status_code == 200
    s_json = get_res.json()
    ds_prov = next(p for p in s_json["providers"] if p["key"] == "deepseek")
    assert ds_prov["has_key"] is True
    assert "9999" in ds_prov["masked_key"]
    assert "sk-initial-secret-9999" not in ds_prov["masked_key"]

    # Update only model & mock mode; masked or omitted DEEPSEEK_API_KEY must NOT overwrite real key
    put_res = client.put(
        "/api/v1/settings/llm",
        json={
            "copilot_default_model": "dashscope/qwen-plus",
            "llm_mock_mode": False,
            "env_updates": {
                "DEEPSEEK_API_KEY": ds_prov["masked_key"],
                "OPENAI_API_BASE": "https://relay.example.com/v1",
            },
        },
    )
    assert put_res.status_code == 200
    updated = put_res.json()
    assert updated["copilot_default_model"] == "dashscope/qwen-plus"
    assert updated["llm_mock_mode"] is False
    saved_env_text = env_path.read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY=sk-initial-secret-9999" in saved_env_text
    assert "OPENAI_API_BASE=https://relay.example.com/v1" in saved_env_text

    # 3. Ollama models probe & test-connection
    class _FakeResp:
        status = 200

        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "qwen2.5:7b"}]}).encode("utf-8")

    monkeypatch.setattr(launcher.urllib.request, "urlopen", lambda req, timeout=2.0: _FakeResp())
    ol_res = client.get("/api/v1/settings/ollama-models")
    assert ol_res.status_code == 200
    assert ol_res.json()["reachable"] is True
    assert "ollama_chat/qwen2.5:7b" in ol_res.json()["models"]

    conn_res = client.post(
        "/api/v1/settings/test-connection",
        json={"provider": "ollama", "api_base": "http://localhost:11434", "model": "qwen2.5:7b"},
    )
    assert conn_res.status_code == 200
    assert conn_res.json()["ok"] is True
    assert "ollama_chat/qwen2.5:7b" in conn_res.json()["models"]


def test_spa_fallback_and_report_path_traversal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Path traversal must be rejected on report routes
    bad_rep = client.get("/api/v1/benchmark/reports/..%5Csecret.json")
    assert bad_rep.status_code in (400, 404)
    bad_api = client.get("/api/v1/does-not-exist")
    assert bad_api.status_code == 404

    dist_dir = tmp_path / "dist"
    (dist_dir / "assets").mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>OrthoRehab SPA</body></html>", encoding="utf-8")
    (dist_dir / "assets" / "app.js").write_text("console.log('ok');", encoding="utf-8")
    monkeypatch.setattr("backend.main.resolve_frontend_dist_dir", lambda: dist_dir)

    health = client.get("/api/v1/health")
    assert health.json()["frontend_ready"] is True

    spa_root = client.get("/")
    assert spa_root.status_code == 200
    assert "OrthoRehab SPA" in spa_root.text

    spa_sub = client.get("/benchmark/history")
    assert spa_sub.status_code == 200
    assert "OrthoRehab SPA" in spa_sub.text

    asset = client.get("/assets/app.js")
    assert asset.status_code == 200
    assert "console.log('ok');" in asset.text

    # /api/* never falls back to index.html even when frontend/dist exists
    api_404 = client.get("/api/v1/unknown-route")
    assert api_404.status_code == 404


def test_desktop_app_check_and_free_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    port = desktop_app.find_free_loopback_port()
    assert 1024 <= port <= 65535

    monkeypatch.delenv("ORTHO_RESULTS_DIR", raising=False)
    code, payload = desktop_app.run_startup_check()
    assert code == 0
    assert payload["status"] == "ok"
    assert payload["knowledge_base"]["scales"] == 4
    assert payload["api_route_count"] >= 15
    assert Path(payload["results_dir"]).parts[-2:] == ("app_benchmark", "results")

    # Frozen mode keeps app_home/results rather than bundle/app_benchmark/results
    monkeypatch.delenv("ORTHO_RESULTS_DIR", raising=False)
    frozen_paths = launcher.RuntimePaths(
        bundle_dir=tmp_path / "bundle",
        app_home=tmp_path / "portable",
        frozen=True,
    )
    frozen_info = desktop_app._configure_desktop_env(frozen_paths)
    assert Path(frozen_info["results_dir"]) == (tmp_path / "portable" / "results")
