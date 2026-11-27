"""Local clinical case archive workspace REST endpoints (/api/v1/cases/*)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app_copilot.case_store import (
    append_followup_visit,
    delete_case,
    import_case_archive,
    list_cases,
    load_case,
    restore_case_revision,
    save_case,
    update_case_metadata,
)

router = APIRouter(prefix="/cases", tags=["cases"])


class SaveCaseRequest(BaseModel):
    case_id: Optional[str] = None
    title: Optional[str] = None
    tags: Optional[List[str]] = None
    raw_text: str = Field(..., min_length=1)
    draft_mode: str = "ward_round"
    analysis_state: Optional[Dict[str, Any]] = None
    revision_note: Optional[str] = None
    record_revision: bool = True


class UpdateCaseMetaRequest(BaseModel):
    title: Optional[str] = None
    tags: Optional[List[str]] = None


class FollowupVisitRequest(BaseModel):
    visit_date: Optional[str] = None
    followup_text: str = Field(..., min_length=1)


@router.get("")
def get_cases_list(
    q: str = Query(default=""),
    high_risk_only: bool = Query(default=False),
) -> Dict[str, Any]:
    return list_cases(q=q, high_risk_only=high_risk_only)


@router.post("/import")
async def import_case_endpoint(request: Request) -> Dict[str, Any]:
    content_type = (request.headers.get("content-type") or "").lower()
    payload: Any = None
    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded = form.get("file")
        if uploaded is None or not hasattr(uploaded, "read"):
            raise HTTPException(status_code=400, detail="请上传有效的病例档案 JSON 文件。")
        raw_bytes = await uploaded.read()
        try:
            payload = json.loads(raw_bytes.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"导入文件不是合法的 UTF-8 JSON：{exc}") from exc
    else:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"请求体不是合法的 JSON：{exc}") from exc

    try:
        return import_case_archive(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{case_id}")
def get_case_detail(case_id: str) -> Dict[str, Any]:
    try:
        return load_case(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("")
def create_or_update_case(req: SaveCaseRequest) -> Dict[str, Any]:
    try:
        return save_case(
            raw_text=req.raw_text,
            draft_mode=req.draft_mode,
            analysis_state=req.analysis_state,
            case_id=req.case_id,
            title=req.title,
            tags=req.tags,
            revision_note=req.revision_note,
            record_revision=req.record_revision,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/{case_id}")
def patch_case_metadata(case_id: str, req: UpdateCaseMetaRequest) -> Dict[str, Any]:
    try:
        return update_case_metadata(case_id, title=req.title, tags=req.tags)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{case_id}/followup")
def add_case_followup(case_id: str, req: FollowupVisitRequest) -> Dict[str, Any]:
    try:
        return append_followup_visit(
            case_id,
            req.followup_text,
            visit_date=req.visit_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{case_id}/restore/{rev}")
def restore_case_rev(case_id: str, rev: int) -> Dict[str, Any]:
    try:
        return restore_case_revision(case_id, rev)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{case_id}")
def remove_case(case_id: str) -> Dict[str, Any]:
    try:
        return delete_case(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{case_id}/export")
def export_case_archive(case_id: str) -> Response:
    try:
        archive = load_case(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    body = json.dumps(archive, ensure_ascii=False, indent=2).encode("utf-8")
    encoded_name = quote(f"{archive['case_id']}.json")
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
    )
