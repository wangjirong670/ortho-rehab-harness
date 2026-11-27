"""Knowledge base read-only overview endpoint (/api/v1/kb/*)."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from app_benchmark.service import DIMENSION_LABELS
from app_copilot.service import (
    CLINICAL_ACTION_ZH,
    SCORE_DIRECTION_LABELS,
    SEVERITY_BADGE_MAP,
    TCM_FIELD_LABELS,
    WOMAC_SCORE_LABELS,
)
from core_tools.calculator import load_scales_catalog
from core_tools.rule_scanner import (
    load_drug_conflicts_catalog,
    load_red_flags_catalog,
)
from core_tools.web_search import list_all_web_sources

router = APIRouter(prefix="/kb", tags=["kb"])


@router.get("/overview")
def get_kb_overview() -> Dict[str, Any]:
    red_flags = load_red_flags_catalog().model_dump()
    drug_conflicts = load_drug_conflicts_catalog().model_dump()
    scales = load_scales_catalog().model_dump()
    web_sources = list_all_web_sources()
    return {
        "red_flags": red_flags,
        "drug_conflicts": drug_conflicts,
        "scales": scales,
        "web_sources": web_sources,
        "labels": {
            "clinical_actions": CLINICAL_ACTION_ZH,
            "tcm_fields": TCM_FIELD_LABELS,
            "dimensions": DIMENSION_LABELS,
            "severity": SEVERITY_BADGE_MAP,
            "score_direction": SCORE_DIRECTION_LABELS,
            "womac_scores": {str(k): v for k, v in WOMAC_SCORE_LABELS.items()},
        },
    }
