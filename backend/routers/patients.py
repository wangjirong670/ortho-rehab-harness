"""Patient workspace card REST endpoints (/api/v1/patients/*)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app_copilot.patient_store import (
    commit_changes,
    create_patient,
    delete_patient,
    list_history,
    list_patients,
    load_patient,
    restore_revision,
    save_patient,
)
from core_tools.schemas import PatientCard

router = APIRouter(prefix="/patients", tags=["patients"])


class CreatePatientRequest(BaseModel):
    patient_id: Optional[str] = None
    display_name: str = Field(default="", max_length=64)
    body: str = ""
    card: Optional[Dict[str, Any]] = None


class UpdatePatientRequest(BaseModel):
    card: Dict[str, Any]
    body: str = ""


class CommitPatientRequest(BaseModel):
    changes: Dict[str, Any] = Field(default_factory=dict)


def _format_patient_response(card: PatientCard, body: str, raw_markdown: str) -> Dict[str, Any]:
    card_dict = card.model_dump(mode="json")
    return {
        "patient_id": card.patient_id,
        "display_name": card.display_name,
        "revision": card.revision,
        "card": card_dict,
        "body": body,
        "raw_markdown": raw_markdown,
    }


@router.get("")
def get_patients_list() -> Dict[str, Any]:
    return {"patients": list_patients()}


@router.post("", status_code=201)
def create_patient_endpoint(req: CreatePatientRequest) -> Dict[str, Any]:
    try:
        card, body, raw_md = create_patient(
            display_name=req.display_name,
            patient_id=req.patient_id,
            body=req.body,
            card_data=req.card,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _format_patient_response(card, body, raw_md)


@router.get("/{patient_id}")
def get_patient_detail(patient_id: str) -> Dict[str, Any]:
    try:
        card, body, raw_md = load_patient(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _format_patient_response(card, body, raw_md)


@router.put("/{patient_id}")
def update_patient_card(patient_id: str, req: UpdatePatientRequest) -> Dict[str, Any]:
    try:
        card, body, raw_md = save_patient(patient_id, req.card, req.body)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"患者档案卡校验失败：{exc}") from exc
    return _format_patient_response(card, body, raw_md)


@router.post("/{patient_id}/commit")
def commit_patient_changes(patient_id: str, req: CommitPatientRequest) -> Dict[str, Any]:
    try:
        card, body, raw_md = commit_changes(patient_id, req.changes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"档案变更提交失败：{exc}") from exc
    return _format_patient_response(card, body, raw_md)


@router.get("/{patient_id}/history")
def get_patient_history(patient_id: str) -> Dict[str, Any]:
    try:
        revisions = list_history(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"patient_id": patient_id, "revisions": revisions}


@router.post("/{patient_id}/restore/{rev}")
def restore_patient_version(patient_id: str, rev: int) -> Dict[str, Any]:
    try:
        card, body, raw_md = restore_revision(patient_id, rev)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _format_patient_response(card, body, raw_md)


@router.delete("/{patient_id}")
def delete_patient_endpoint(patient_id: str) -> Dict[str, Any]:
    try:
        deleted = delete_patient(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"未找到患者档案：{patient_id}")
    return {"deleted": True, "patient_id": patient_id}
