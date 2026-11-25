"""Pydantic v2 schema definitions for shared medical YAML assets and benchmark cases.

Enforces structural validity, regex syntax safety, scale arithmetic self-consistency,
and unique IDs before any knowledge base file enters runtime tools.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

ReviewStatus = Literal["draft_for_clinician_review", "clinician_verified"]
SeverityLevel = Literal["critical", "high", "medium", "low"]
ExpectedAction = Literal[
    "STOP_AND_REFER",
    "EMERGENCY_REFER",
    "STOP_AND_VERIFY_PROTOCOL",
    "STOP_AND_MODIFY_PLAN",
    "STOP_AND_VERIFY_MEDS",
    "CAUTION_AND_PROTECT",
    "COMPUTE_WITH_TOOL",
    "ALLOW_CONSERVATIVE_GUIDANCE",
]
BenchmarkDimension = Literal[
    "red_flag_interception",
    "longitudinal_memory",
    "tcm_syndrome_and_tools",
    "over_refusal_calibration",
]


# ==============================================================================
# 1. Red Flags Schema (knowledge_base/red_flags.yaml)
# ==============================================================================


def _validate_keyword_or_regex(entry: str, field_name: str) -> str:
    """Validate literal keyword or 'regex:<pattern>' entry."""
    if not entry or not entry.strip():
        raise ValueError(f"{field_name} contains an empty keyword entry")
    if entry.startswith("regex:"):
        pattern = entry[6:]
        if not pattern:
            raise ValueError(f"{field_name} has empty pattern after 'regex:'")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex '{pattern}' in {field_name}: {exc}") from exc
    return entry


class TriggerSymptoms(BaseModel):
    required_any: List[str] = Field(min_length=1)
    supporting_any: List[str] = Field(default_factory=list)

    @field_validator("required_any", "supporting_any")
    @classmethod
    def validate_symptom_patterns(cls, items: List[str], info) -> List[str]:
        return [_validate_keyword_or_regex(item, info.field_name) for item in items]


class RedFlagRule(BaseModel):
    rule_id: str = Field(pattern=r"^RF_\d{3,}$")
    name: str = Field(min_length=2)
    category: str = Field(min_length=2)
    severity: Literal["critical", "high", "medium"]
    review_status: ReviewStatus
    trigger_symptoms: TriggerSymptoms
    negation_guards: List[str] = Field(default_factory=list)
    contraindicated_actions: List[str] = Field(min_length=1)
    recommended_action: ExpectedAction
    clinical_alert: str = Field(min_length=5)
    clinical_rationale: str = Field(min_length=5)
    references: List[str] = Field(min_length=1)

    @field_validator("negation_guards")
    @classmethod
    def validate_negation_guards(cls, items: List[str]) -> List[str]:
        return [_validate_keyword_or_regex(item, "negation_guards") for item in items]


class RedFlagsCatalog(BaseModel):
    version: str
    domain: str
    description: str
    global_negation_terms: List[str] = Field(min_length=1)
    rules: List[RedFlagRule] = Field(min_length=1)

    @field_validator("global_negation_terms")
    @classmethod
    def validate_global_negation_terms(cls, items: List[str]) -> List[str]:
        return [_validate_keyword_or_regex(item, "global_negation_terms") for item in items]

    @model_validator(mode="after")
    def ensure_unique_rule_ids(self) -> "RedFlagsCatalog":
        ids = [r.rule_id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate rule_id detected in red_flags.yaml")
        return self


# ==============================================================================
# 2. Drug Conflicts Schema (knowledge_base/drug_conflicts.yaml)
# ==============================================================================


class DrugGroup(BaseModel):
    label: str = Field(min_length=2)
    keywords: List[str] = Field(min_length=1)
    washout_window_days: Optional[int] = Field(default=None, ge=0)

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, items: List[str]) -> List[str]:
        return [_validate_keyword_or_regex(item, "keywords") for item in items]


class DrugConflictRule(BaseModel):
    conflict_id: str = Field(pattern=r"^DC_\d{3,}$")
    name: str = Field(min_length=2)
    severity: Literal["critical", "high", "medium"]
    review_status: ReviewStatus
    group_a: DrugGroup
    group_b: DrugGroup
    risk_mechanism: str = Field(min_length=5)
    clinical_alert: str = Field(min_length=5)
    suggested_handling: str = Field(min_length=5)
    references: List[str] = Field(min_length=1)


class DrugConflictsCatalog(BaseModel):
    version: str
    domain: str
    description: str
    conflicts: List[DrugConflictRule] = Field(min_length=1)

    @model_validator(mode="after")
    def ensure_unique_conflict_ids(self) -> "DrugConflictsCatalog":
        ids = [c.conflict_id for c in self.conflicts]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate conflict_id detected in drug_conflicts.yaml")
        return self


# ==============================================================================
# 3. Clinical Scales Schema (knowledge_base/scales_spec.yaml)
# ==============================================================================


class ScaleSubscale(BaseModel):
    subscale_id: str = Field(min_length=2)
    label: str = Field(min_length=2)
    min_score: float
    max_score: float
    item_count: int = Field(ge=1)

    @model_validator(mode="after")
    def check_bounds(self) -> "ScaleSubscale":
        if self.min_score > self.max_score:
            raise ValueError(f"Subscale {self.subscale_id}: min_score > max_score")
        return self


class ScaleItem(BaseModel):
    item_id: str = Field(min_length=2)
    label: str = Field(min_length=2)
    subscale_id: Optional[str] = None
    min_value: float
    max_value: float
    step: Optional[float] = Field(default=1.0, gt=0)
    allowed_values: Optional[List[float]] = None
    required: bool = True

    @model_validator(mode="after")
    def check_item_range(self) -> "ScaleItem":
        if self.min_value > self.max_value:
            raise ValueError(f"Item {self.item_id}: min_value > max_value")
        if self.allowed_values:
            if min(self.allowed_values) < self.min_value or max(self.allowed_values) > self.max_value:
                raise ValueError(f"Item {self.item_id}: allowed_values out of [min_value, max_value]")
        return self


class SeverityBand(BaseModel):
    label: str = Field(min_length=1)
    min_score: float
    max_score: float

    @model_validator(mode="after")
    def check_band_bounds(self) -> "SeverityBand":
        if self.min_score > self.max_score:
            raise ValueError(f"SeverityBand {self.label}: min_score > max_score")
        return self


class ScaleSpec(BaseModel):
    scale_id: str = Field(pattern=r"^[A-Z0-9_]+$")
    name: str = Field(min_length=2)
    version_note: str = Field(min_length=5)
    review_status: ReviewStatus
    direction: Literal["higher_is_worse", "higher_is_better"]
    min_total: float
    max_total: float
    formula_type: Literal[
        "direct_sum",
        "subscale_sum_with_normalized_pct",
        "direct_sum_with_improvement_rate",
        "subscale_sum_with_improvement_rate",
    ]
    subscales: List[ScaleSubscale] = Field(default_factory=list)
    items: List[ScaleItem] = Field(min_length=1)
    severity_bands: List[SeverityBand] = Field(min_length=1)
    improvement_rate_formula: Optional[str] = None
    references: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_arithmetic_consistency(self) -> "ScaleSpec":
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError(f"Scale {self.scale_id}: duplicate item_id detected")

        calc_min = sum(item.min_value for item in self.items)
        calc_max = sum(item.max_value for item in self.items)
        if abs(calc_min - self.min_total) > 1e-6 or abs(calc_max - self.max_total) > 1e-6:
            raise ValueError(
                f"Scale {self.scale_id}: item range sum [{calc_min}, {calc_max}] "
                f"!= declared [{self.min_total}, {self.max_total}]"
            )

        if self.subscales:
            subscale_map = {s.subscale_id: s for s in self.subscales}
            if len(subscale_map) != len(self.subscales):
                raise ValueError(f"Scale {self.scale_id}: duplicate subscale_id detected")
            grouped: Dict[str, List[ScaleItem]] = {sid: [] for sid in subscale_map}
            for item in self.items:
                if item.subscale_id not in subscale_map:
                    raise ValueError(
                        f"Scale {self.scale_id}: item '{item.item_id}' has unknown subscale_id '{item.subscale_id}'"
                    )
                grouped[item.subscale_id].append(item)
            for sid, spec in subscale_map.items():
                members = grouped[sid]
                if len(members) != spec.item_count:
                    raise ValueError(
                        f"Scale {self.scale_id} subscale '{sid}': expected {spec.item_count} items, got {len(members)}"
                    )
                s_min = sum(m.min_value for m in members)
                s_max = sum(m.max_value for m in members)
                if abs(s_min - spec.min_score) > 1e-6 or abs(s_max - spec.max_score) > 1e-6:
                    raise ValueError(
                        f"Scale {self.scale_id} subscale '{sid}': item sum [{s_min}, {s_max}] "
                        f"!= declared [{spec.min_score}, {spec.max_score}]"
                    )

        for band in self.severity_bands:
            if band.min_score < self.min_total or band.max_score > self.max_total:
                raise ValueError(
                    f"Scale {self.scale_id}: band '{band.label}' [{band.min_score}, {band.max_score}] "
                    f"exceeds total range [{self.min_total}, {self.max_total}]"
                )
        return self


class ScalesCatalog(BaseModel):
    version: str
    domain: str
    description: str
    scales: List[ScaleSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def ensure_unique_scale_ids(self) -> "ScalesCatalog":
        ids = [s.scale_id for s in self.scales]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate scale_id detected in scales_spec.yaml")
        return self


# ==============================================================================
# 4. Benchmark Test Case Schema (app_benchmark/test_cases/**/*.yaml)
# ==============================================================================


class TurnMessage(BaseModel):
    turn: int = Field(ge=1)
    user_msg: str = Field(min_length=3)


class ForbiddenPattern(BaseModel):
    pattern: str = Field(min_length=1)
    ignore_if_negated: bool = True
    rationale: Optional[str] = None

    @field_validator("pattern")
    @classmethod
    def validate_regex(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"Invalid regex '{value}': {exc}") from exc
        return value


class DeterministicRules(BaseModel):
    must_mention_any: List[str] = Field(min_length=1)
    forbidden_patterns: List[Union[str, ForbiddenPattern]] = Field(default_factory=list)
    expected_action: ExpectedAction

    @field_validator("forbidden_patterns")
    @classmethod
    def normalize_and_validate_patterns(
        cls, items: List[Union[str, ForbiddenPattern]]
    ) -> List[ForbiddenPattern]:
        normalized: List[ForbiddenPattern] = []
        for entry in items:
            if isinstance(entry, str):
                normalized.append(ForbiddenPattern(pattern=entry, ignore_if_negated=True))
            else:
                normalized.append(entry)
        return normalized


class ScoringRubric(BaseModel):
    safety_pass_required: bool
    check_over_refusal: bool


class BenchmarkCase(BaseModel):
    case_id: str = Field(pattern=r"^[A-Z0-9_]+$")
    category: str = Field(min_length=2)
    dimension: BenchmarkDimension
    risk_level: SeverityLevel
    review_status: ReviewStatus = "draft_for_clinician_review"
    patient_profile: str = Field(min_length=5)
    expected_rule_ids: List[str] = Field(default_factory=list)
    expected_scale_id: Optional[str] = None
    expected_numeric_targets: Optional[Dict[str, float]] = None
    multi_turn_script: List[TurnMessage] = Field(min_length=1)
    deterministic_rules: DeterministicRules
    scoring_rubric: ScoringRubric

    @model_validator(mode="after")
    def check_turn_sequence(self) -> "BenchmarkCase":
        expected_turns = list(range(1, len(self.multi_turn_script) + 1))
        actual_turns = [t.turn for t in self.multi_turn_script]
        if actual_turns != expected_turns:
            raise ValueError(f"Case {self.case_id}: turn sequence must be {expected_turns}, got {actual_turns}")
        return self
