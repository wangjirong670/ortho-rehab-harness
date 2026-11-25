"""Smoke tests for knowledge base and benchmark case schema validation."""

import re

from core_tools.calculator import load_scales_catalog
from core_tools.rule_scanner import load_drug_conflicts_catalog, load_red_flags_catalog
from scripts.validate_kb import validate_all


def test_knowledge_base_and_cases_are_valid() -> None:
    passed, infos, errors = validate_all()
    assert passed, f"Validation failed: {errors}"
    assert len(infos) == 4


def test_new_rules_and_traceable_references_exist() -> None:
    rf_cat = load_red_flags_catalog()
    dc_cat = load_drug_conflicts_catalog()
    sc_cat = load_scales_catalog()

    rf_ids = {r.rule_id for r in rf_cat.rules}
    dc_ids = {c.conflict_id for c in dc_cat.conflicts}
    assert {f"RF_{i:03d}" for i in range(1, 17)}.issubset(rf_ids)
    assert {f"DC_{i:03d}" for i in range(1, 12)}.issubset(dc_ids)

    year_or_edition_re = re.compile(r"(?:19|20)\d{2}|第\d+版")
    for rule in rf_cat.rules:
        assert any(year_or_edition_re.search(ref) for ref in rule.references), rule.rule_id
    for conflict in dc_cat.conflicts:
        assert any(year_or_edition_re.search(ref) for ref in conflict.references), conflict.conflict_id
    for scale in sc_cat.scales:
        assert any(year_or_edition_re.search(ref) for ref in scale.references), scale.scale_id
