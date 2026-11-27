"""Unit tests for PatientCard PATIENT.md store and JSONL trace session_store."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app_copilot import patient_store, session_store
from core_tools.schemas import PatientCard, PatientRedFlagItem


def test_patient_card_schema_and_resolved_reason_validation() -> None:
    with pytest.raises(ValidationError):
        PatientRedFlagItem(
            id="RF_001",
            since="2026-11-27",
            status="resolved",
            resolved_reason="   ",
        )

    with pytest.raises(ValidationError):
        PatientCard(patient_id="../escape")


def test_patient_md_roundtrip_history_and_prompt_anonymization(tmp_path: Path) -> None:
    archives = tmp_path / "case_archives"
    card, body, raw_md = patient_store.create_patient(
        display_name="302床 张明伟",
        patient_id="PT_302_TEST",
        body="302床 张明伟主诉颈肩痛3周，既往冠脉支架术后长期口服氯吡格雷。",
        archives_dir=archives,
    )
    assert card.revision == 1
    assert "PATIENT.r001.md" in [
        p.name for p in (archives / "PT_302_TEST" / ".history").glob("*.md")
    ]
    assert "---" in raw_md

    # build_prompt_context must deterministically remove display_name & tokens
    prompt_ctx = patient_store.build_prompt_context(card, body)
    assert "302床 张明伟" not in prompt_ctx
    assert "张明伟" not in prompt_ctx

    # Auto apply red flag RF_001
    updated_card, added = patient_store.apply_red_flags_auto(
        "PT_302_TEST",
        [{"rule_id": "RF_001", "name": "颈脊髓受压风险"}],
        archives_dir=archives,
    )
    assert added == ["RF_001"]
    assert updated_card.revision == 2

    # Silent removal of active red flag must be forbidden
    stripped = updated_card.model_copy(deep=True)
    stripped.active_red_flags = []
    with pytest.raises(ValueError, match="禁止静默删除有效红旗"):
        patient_store.save_patient("PT_302_TEST", stripped, body, archives_dir=archives)

    # Explicit resolution with resolved_reason succeeds and records revision r003
    committed_card, _, _ = patient_store.commit_changes(
        "PT_302_TEST",
        {
            "resolve_red_flags": [{"id": "RF_001", "resolved_reason": "MRI 排除脊髓压迫"}],
            "add_medications": [
                {"name": "氯吡格雷", "status": "active", "start_date": "2025-10-01", "source": "clinician"}
            ],
            "add_surgical_history": ["冠脉支架植入术"],
        },
        archives_dir=archives,
    )
    assert committed_card.revision == 3
    assert any(rf.status == "resolved" and rf.resolved_at for rf in committed_card.active_red_flags)

    revs = patient_store.list_history("PT_302_TEST", archives_dir=archives)
    assert [r["revision"] for r in revs] == [3, 2, 1]

    # Restore back to r002 creates r004 with RF_001 active again
    restored, _, _ = patient_store.restore_revision("PT_302_TEST", 2, archives_dir=archives)
    assert restored.revision == 4
    assert any(rf.id == "RF_001" and rf.status == "active" for rf in restored.active_red_flags)


def test_session_store_append_replay_markdown_and_path_traversal(tmp_path: Path) -> None:
    archives = tmp_path / "case_archives"
    patient_store.create_patient(display_name="患者乙", patient_id="PT_SESSION_01", archives_dir=archives)

    with pytest.raises(ValueError):
        session_store.validate_session_id("../evil")

    created = session_store.create_session("PT_SESSION_01", title="初诊评估", archives_dir=archives)
    sid = created["session_id"]

    session_store.append_event(
        "PT_SESSION_01",
        sid,
        "user_input",
        {"content": "右膝疼痛3天，昨日扭伤。", "output_mode": "full"},
        turn_id="turn_1",
        render_md=False,
        archives_dir=archives,
    )
    session_store.append_event(
        "PT_SESSION_01",
        sid,
        "scan",
        {
            "risk_report": {"alerts": [], "suppressed_matches": []},
            "unified_alerts": [],
        },
        turn_id="turn_1",
        render_md=False,
        archives_dir=archives,
    )
    session_store.append_event(
        "PT_SESSION_01",
        sid,
        "generation",
        {
            "content": "建议先行制动护养并排查X线。",
            "raw_content": "建议先行制动护养并排查X线。",
            "model": "mock/deepseek",
            "blocked": False,
        },
        turn_id="turn_1",
        render_md=False,
        archives_dir=archives,
    )
    session_store.append_event(
        "PT_SESSION_01",
        sid,
        "card_diff",
        {"add_surgical_history": ["右膝关节镜探查史"]},
        turn_id="turn_1",
        render_md=True,
        archives_dir=archives,
    )

    state = session_store.replay_session("PT_SESSION_01", sid, archives_dir=archives)
    assert state["title"] == "初诊评估"
    assert len(state["messages"]) == 2
    assert len(state["pending_card_diffs"]) == 1
    assert state["event_count"] == 5

    # Read-only .md exists and reflects trace content
    md_file = archives / "PT_SESSION_01" / "sessions" / f"{sid}.md"
    assert md_file.is_file()
    md_text = md_file.read_text(encoding="utf-8")
    assert "仅供只读查阅" in md_text
    assert "右膝疼痛3天" in md_text

    # Scale recalculation event marks prior assistant messages stale + adds system message
    session_store.append_event(
        "PT_SESSION_01",
        sid,
        "scale",
        {
            "scale_result": {"scale_id": "VAS_PAIN", "status": "ok", "total_score": 6.0},
            "mark_stale": True,
            "system_note": "VAS 量表已手动更新为 6 分。",
        },
        turn_id="scale_1",
        render_md=True,
        archives_dir=archives,
    )
    replayed_after_scale = session_store.replay_session("PT_SESSION_01", sid, archives_dir=archives)
    assert replayed_after_scale["messages"][1]["stale"] is True
    assert replayed_after_scale["messages"][-1]["role"] == "system"
    assert "VAS 量表已手动更新" in replayed_after_scale["messages"][-1]["content"]
