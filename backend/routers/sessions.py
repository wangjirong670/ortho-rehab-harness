"""Clinical encounter session REST & SSE endpoints (/api/v1/patients/{patient_id}/sessions/*)."""

from __future__ import annotations

import json
import queue
import threading
import uuid
from typing import Any, Dict, Generator, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app_copilot.patient_store import apply_red_flags_auto, load_patient
from app_copilot.service import detect_potential_pii_details, sanitize_error_message
from app_copilot.session_store import (
    append_event,
    create_session,
    list_sessions,
    read_trace_events,
    replay_session,
    sync_session_markdown,
)
from core_tools.calculator import calculate_scale
from core_tools.harness_core import (
    TurnContext,
    collect_unified_alerts,
    generate_patient_education,
    run_turn,
)
from core_tools.llm_client import format_llm_error_zh

router = APIRouter(prefix="/patients/{patient_id}/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    title: str = ""


class ChatTurnRequest(BaseModel):
    content: str = Field(..., min_length=1)
    model: Optional[str] = None
    thinking: Literal["off", "low", "medium", "high"] = "off"
    output_mode: Literal["full", "screen"] = "full"
    mock_mode: Optional[bool] = None
    pii_action: Literal["none", "sanitized", "override_raw"] = "none"


class EducationRequest(BaseModel):
    model: Optional[str] = None
    mock_mode: Optional[bool] = None
    turn_id: Optional[str] = None


class SessionScaleRequest(BaseModel):
    scale_id: str = Field(..., min_length=1)
    updated_params: Dict[str, Any] = Field(default_factory=dict)


def _format_sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("")
def get_patient_sessions(patient_id: str) -> Dict[str, Any]:
    try:
        sessions = list_sessions(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"patient_id": patient_id, "sessions": sessions}


@router.post("", status_code=201)
def create_patient_session(patient_id: str, req: CreateSessionRequest) -> Dict[str, Any]:
    try:
        return create_session(patient_id, title=req.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{session_id}")
def get_session_state(patient_id: str, session_id: str) -> Dict[str, Any]:
    try:
        return replay_session(patient_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{session_id}/trace")
def get_session_trace(
    patient_id: str,
    session_id: str,
    turn_id: Optional[str] = Query(default=None),
    download: bool = Query(default=False),
) -> Any:
    try:
        events = read_trace_events(patient_id, session_id, turn_id=turn_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if download:
        jsonl_body = "\n".join(json.dumps(ev, ensure_ascii=False) for ev in events) + "\n"
        return Response(
            content=jsonl_body,
            media_type="application/x-ndjson; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{session_id}.trace.jsonl"',
            },
        )
    return {
        "patient_id": patient_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "events": events,
    }


@router.post("/{session_id}/chat")
def chat_in_session(patient_id: str, session_id: str, req: ChatTurnRequest) -> StreamingResponse:
    cleaned_content = (req.content or "").strip()
    if not cleaned_content:
        raise HTTPException(status_code=400, detail="输入内容不能为空。")

    try:
        patient_card, patient_body, _ = load_patient(patient_id)
        session_state = replay_session(patient_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    turn_id = f"turn_{uuid.uuid4().hex[:8]}"
    pii_details = detect_potential_pii_details(cleaned_content)

    append_event(
        patient_id,
        session_id,
        "user_input",
        {
            "content": cleaned_content,
            "model": req.model,
            "thinking": req.thinking,
            "output_mode": req.output_mode,
        },
        turn_id=turn_id,
        render_md=False,
    )
    append_event(
        patient_id,
        session_id,
        "pii_check",
        {
            "has_pii": bool(pii_details.get("has_pii")),
            "item_count": len(pii_details.get("items") or []),
            "action": req.pii_action,
        },
        turn_id=turn_id,
        render_md=False,
    )

    event_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()

    def _emit_cb(ev_name: str, payload: Dict[str, Any]) -> None:
        enriched = dict(payload)
        enriched.setdefault("turn_id", turn_id)
        if ev_name == "alert":
            _, newly_saved_rfs = apply_red_flags_auto(
                patient_id,
                enriched.get("unified_alerts") or [],
                source="extracted",
            )
            enriched["auto_saved_red_flags"] = newly_saved_rfs
        event_queue.put({"event": ev_name, "data": enriched})

    def _worker() -> None:
        try:
            ctx = TurnContext(
                patient_card=patient_card,
                patient_body=patient_body,
                history=list(session_state.get("history_for_llm") or []),
                patient_id=patient_id,
                session_id=session_id,
            )
            result = run_turn(
                ctx,
                cleaned_content,
                model=req.model,
                thinking=req.thinking,
                output_mode=req.output_mode,
                emit=_emit_cb,
                mock_mode=req.mock_mode,
                response_style="clinical_chat",
            )

            # Persist trace events strictly matching session_store contract
            append_event(
                patient_id,
                session_id,
                "extraction",
                {
                    "extracted_note": result.extracted_note,
                    "extraction_meta": result.extraction_meta,
                },
                turn_id=turn_id,
                render_md=False,
            )
            append_event(
                patient_id,
                session_id,
                "scan",
                {
                    "risk_report": result.risk_report,
                    "unified_alerts": result.unified_alerts,
                    "inherited_red_flags": result.inherited_red_flags,
                },
                turn_id=turn_id,
                render_md=False,
            )
            append_event(
                patient_id,
                session_id,
                "scale",
                {
                    "scale_results": result.scale_results,
                    "unified_alerts": result.unified_alerts,
                },
                turn_id=turn_id,
                render_md=False,
            )
            append_event(
                patient_id,
                session_id,
                "generation",
                {
                    "content": result.content,
                    "raw_content": result.raw_content,
                    "reasoning": result.reasoning,
                    "reasoning_verification": result.reasoning_verification,
                    "model": result.model,
                    "mock": result.mock,
                    "output_mode": result.output_mode,
                    "blocked": result.blocked,
                    "verification": result.verification,
                    "usage": result.usage,
                    "latency_ms": result.latency_ms,
                },
                turn_id=turn_id,
                render_md=False,
            )
            append_event(
                patient_id,
                session_id,
                "verification",
                result.verification,
                turn_id=turn_id,
                render_md=False,
            )

            diff_event_payload: Dict[str, Any] = {}
            if result.card_diff:
                diff_ev = append_event(
                    patient_id,
                    session_id,
                    "card_diff",
                    result.card_diff,
                    turn_id=turn_id,
                    render_md=False,
                )
                diff_event_payload = dict(result.card_diff)
                diff_event_payload["diff_id"] = f"diff_{diff_ev['seq']}"
                diff_event_payload["turn_id"] = turn_id

            sync_session_markdown(patient_id, session_id)

            if result.reasoning:
                event_queue.put(
                    {
                        "event": "reasoning",
                        "data": {
                            "turn_id": turn_id,
                            "reasoning": result.reasoning,
                            "reasoning_verification": result.reasoning_verification,
                        },
                    }
                )

            # Content is pushed ONLY AFTER verification has completed
            event_queue.put(
                {
                    "event": "content",
                    "data": {
                        "turn_id": turn_id,
                        "content": result.content,
                        "model": result.model,
                        "mock": result.mock,
                        "output_mode": result.output_mode,
                        "blocked": result.blocked,
                        "latency_ms": result.latency_ms,
                    },
                }
            )
            event_queue.put(
                {
                    "event": "verification",
                    "data": {
                        "turn_id": turn_id,
                        **result.verification,
                    },
                }
            )
            if diff_event_payload:
                event_queue.put({"event": "card_diff", "data": diff_event_payload})
            event_queue.put(
                {
                    "event": "done",
                    "data": {
                        "turn_id": turn_id,
                        "patient_id": patient_id,
                        "session_id": session_id,
                        "blocked": result.blocked,
                        "alert_count": len(result.unified_alerts),
                    },
                }
            )
        except Exception as exc:
            zh_err, tech_err = format_llm_error_zh(str(exc), model=req.model or "")
            event_queue.put(
                {
                    "event": "error",
                    "data": {
                        "turn_id": turn_id,
                        "message": f"{zh_err}（{sanitize_error_message(tech_err)}）",
                    },
                }
            )
        finally:
            event_queue.put(None)

    threading.Thread(target=_worker, daemon=True).start()

    def _sse_stream() -> Generator[str, None, None]:
        while True:
            try:
                item = event_queue.get(timeout=15.0)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if item is None:
                break
            yield _format_sse(item["event"], item["data"])

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{session_id}/education")
def generate_session_education(
    patient_id: str,
    session_id: str,
    req: EducationRequest,
) -> Dict[str, Any]:
    try:
        patient_card, patient_body, _ = load_patient(patient_id)
        session_state = replay_session(patient_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    latest_assistant = next(
        (m.get("content", "") for m in reversed(session_state.get("messages") or []) if m.get("role") == "assistant"),
        "",
    )
    ctx = TurnContext(
        patient_card=patient_card,
        patient_body=patient_body,
        history=list(session_state.get("history_for_llm") or []),
        patient_id=patient_id,
        session_id=session_id,
    )
    try:
        edu = generate_patient_education(
            ctx,
            latest_reply=str(latest_assistant or ""),
            extracted_note=session_state.get("latest_extracted_note") or {},
            risk_report=session_state.get("latest_risk_report") or {},
            scale_results=session_state.get("latest_scale_results") or [],
            unified_alerts=session_state.get("latest_alerts") or [],
            model=req.model,
            mock_mode=req.mock_mode,
        )
    except Exception as exc:
        zh_err, tech_err = format_llm_error_zh(str(exc), model=req.model or "")
        raise HTTPException(
            status_code=502,
            detail=f"{zh_err}\n（技术细节：{sanitize_error_message(tech_err)}）",
        ) from exc

    turn_id = req.turn_id or f"edu_{uuid.uuid4().hex[:8]}"
    append_event(
        patient_id,
        session_id,
        "education",
        {
            "content": edu["content"],
            "raw_content": edu.get("raw_content", ""),
            "blocked": edu["blocked"],
            "model": edu["model"],
            "verification": edu["safety_check"],
        },
        turn_id=turn_id,
        render_md=False,
    )
    append_event(
        patient_id,
        session_id,
        "verification",
        edu["safety_check"],
        turn_id=turn_id,
        render_md=True,
    )
    return {
        "patient_id": patient_id,
        "session_id": session_id,
        "turn_id": turn_id,
        **edu,
    }


@router.post("/{session_id}/scales")
def recalculate_session_scale(
    patient_id: str,
    session_id: str,
    req: SessionScaleRequest,
) -> Dict[str, Any]:
    try:
        load_patient(patient_id)
        session_state = replay_session(patient_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    clean_params = {k: v for k, v in (req.updated_params or {}).items() if v is not None}
    calc_res = calculate_scale(req.scale_id, clean_params)
    calc_res["input_params"] = dict(clean_params)

    existing_scales = list(session_state.get("latest_scale_results") or [])
    replaced = False
    for idx, item in enumerate(existing_scales):
        if item.get("scale_id") == calc_res.get("scale_id"):
            existing_scales[idx] = calc_res
            replaced = True
            break
    if not replaced:
        existing_scales.append(calc_res)

    unified_alerts = collect_unified_alerts(
        session_state.get("latest_risk_report") or {},
        existing_scales,
    )
    _, newly_added = apply_red_flags_auto(patient_id, unified_alerts, source="system")

    turn_id = f"scale_{uuid.uuid4().hex[:8]}"
    append_event(
        patient_id,
        session_id,
        "scale",
        {
            "scale_result": calc_res,
            "scale_results": existing_scales,
            "unified_alerts": unified_alerts,
            "mark_stale": True,
            "system_note": f"医师已手动重算量表 {calc_res.get('scale_id')}，历史模型回复已标为待核对过期。",
        },
        turn_id=turn_id,
        render_md=False,
    )
    append_event(
        patient_id,
        session_id,
        "human_action",
        {
            "action": "recalculate_scale",
            "scale_id": calc_res.get("scale_id"),
            "status": calc_res.get("status"),
            "auto_saved_red_flags": newly_added,
        },
        turn_id=turn_id,
        render_md=True,
    )
    updated_state = replay_session(patient_id, session_id)
    return {
        "scale_result": calc_res,
        "auto_saved_red_flags": newly_added,
        "session": updated_state,
    }


@router.post("/{session_id}/diffs/{diff_id}/dismiss")
def dismiss_session_card_diff(
    patient_id: str,
    session_id: str,
    diff_id: str,
) -> Dict[str, Any]:
    try:
        load_patient(patient_id)
        replay_session(patient_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    append_event(
        patient_id,
        session_id,
        "human_action",
        {
            "action": "dismiss_card_diff",
            "diff_id": diff_id,
        },
        render_md=True,
    )
    return {"session": replay_session(patient_id, session_id)}
