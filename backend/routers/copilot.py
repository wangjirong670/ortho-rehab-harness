"""Copilot REST endpoints (/api/v1/copilot/*)."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app_copilot.service import (
    build_clinical_handover_text,
    detect_potential_pii_details,
    extract_text_from_uploaded_file,
    format_copilot_response,
    get_demo_cases_payload,
    markdown_to_plain_text,
    recalculate_scale_in_state,
    regenerate_draft_in_state,
    remove_scale_from_state,
    run_copilot_analysis,
    run_fast_safety_scan,
    sanitize_clinical_pii_payload,
    sanitize_error_message,
)
from core_tools.llm_client import format_llm_error_zh

router = APIRouter(prefix="/copilot", tags=["copilot"])


class PiiTextRequest(BaseModel):
    text: str = ""


class FastScanRequest(BaseModel):
    raw_text: str = Field(..., min_length=1)


class AnalyzeRequest(BaseModel):
    raw_text: str = Field(..., min_length=1)
    draft_mode: Literal["ward_round", "rehab_education"] = "ward_round"
    model: Optional[str] = None
    mock_mode: Optional[bool] = None


class ScaleRecalculateRequest(BaseModel):
    analysis_state: Dict[str, Any]
    scale_id: str = Field(..., min_length=1)
    updated_params: Dict[str, Any] = Field(default_factory=dict)
    scale_index: Optional[int] = None
    refresh_draft: bool = False
    draft_mode: Optional[Literal["ward_round", "rehab_education"]] = None
    model: Optional[str] = None
    mock_mode: Optional[bool] = None


class ScaleRemoveRequest(BaseModel):
    analysis_state: Dict[str, Any]
    scale_index: int


class DraftRegenerateRequest(BaseModel):
    analysis_state: Dict[str, Any]
    draft_mode: Literal["ward_round", "rehab_education"] = "ward_round"
    model: Optional[str] = None
    mock_mode: Optional[bool] = None


class HandoverRequest(BaseModel):
    analysis_state: Dict[str, Any]


@router.get("/demo-cases")
def get_demo_cases() -> Dict[str, Any]:
    return {"cases": get_demo_cases_payload()}


@router.post("/upload")
async def upload_clinical_note(file: UploadFile = File(...)) -> Dict[str, Any]:
    raw_bytes = await file.read()
    filename = file.filename or "note.txt"
    try:
        text = extract_text_from_uploaded_file(filename, raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "filename": filename,
        "text": text,
        "char_count": len(text),
    }


@router.post("/pii/detect")
def detect_pii(req: PiiTextRequest) -> Dict[str, Any]:
    return detect_potential_pii_details(req.text)


@router.post("/pii/sanitize")
def sanitize_pii(req: PiiTextRequest) -> Dict[str, Any]:
    return sanitize_clinical_pii_payload(req.text)


@router.post("/fast-scan")
def fast_scan(req: FastScanRequest) -> Dict[str, Any]:
    try:
        state = run_fast_safety_scan(req.raw_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return format_copilot_response(state)


@router.post("/analyze")
def analyze_note(req: AnalyzeRequest) -> Dict[str, Any]:
    try:
        state = run_copilot_analysis(
            req.raw_text,
            draft_mode=req.draft_mode,
            model=req.model,
            mock_mode=req.mock_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        zh_err, tech_err = format_llm_error_zh(str(exc), model=req.model or "")
        raise HTTPException(
            status_code=502,
            detail=f"{zh_err}\n（技术细节：{sanitize_error_message(tech_err)}）",
        ) from exc
    return format_copilot_response(state)


@router.post("/scales/recalculate")
def recalculate_scale(req: ScaleRecalculateRequest) -> Dict[str, Any]:
    try:
        state = recalculate_scale_in_state(
            req.analysis_state,
            req.scale_id,
            req.updated_params,
            scale_index=req.scale_index,
            refresh_draft=req.refresh_draft,
            draft_mode=req.draft_mode,
            model=req.model,
            mock_mode=req.mock_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        zh_err, tech_err = format_llm_error_zh(str(exc), model=req.model or "")
        raise HTTPException(
            status_code=502,
            detail=f"{zh_err}\n（技术细节：{sanitize_error_message(tech_err)}）",
        ) from exc
    return format_copilot_response(state)


@router.post("/scales/remove")
def remove_scale(req: ScaleRemoveRequest) -> Dict[str, Any]:
    state = remove_scale_from_state(req.analysis_state, req.scale_index)
    return format_copilot_response(state)


@router.post("/draft/regenerate")
def regenerate_draft(req: DraftRegenerateRequest) -> Dict[str, Any]:
    try:
        state = regenerate_draft_in_state(
            req.analysis_state,
            draft_mode=req.draft_mode,
            model=req.model,
            mock_mode=req.mock_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        zh_err, tech_err = format_llm_error_zh(str(exc), model=req.model or "")
        raise HTTPException(
            status_code=502,
            detail=f"{zh_err}\n（技术细节：{sanitize_error_message(tech_err)}）",
        ) from exc
    return format_copilot_response(state)


@router.post("/handover")
def export_handover(req: HandoverRequest) -> Dict[str, Any]:
    handover_text = build_clinical_handover_text(req.analysis_state)
    draft_content = str((req.analysis_state.get("draft") or {}).get("content") or "")
    draft_plain_text = markdown_to_plain_text(draft_content)
    return {
        "handover_text": handover_text,
        "draft_plain_text": draft_plain_text,
    }
