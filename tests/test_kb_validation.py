"""Smoke tests for knowledge base and benchmark case schema validation."""

from scripts.validate_kb import validate_all


def test_knowledge_base_and_cases_are_valid() -> None:
    passed, infos, errors = validate_all()
    assert passed, f"Validation failed: {errors}"
    assert len(infos) == 4
