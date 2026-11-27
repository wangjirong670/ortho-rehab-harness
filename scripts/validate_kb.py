"""Validate all shared knowledge base YAMLs and benchmark test cases.

Usage:
    python scripts/validate_kb.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Set, Tuple

import yaml
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_tools.schemas import (  # noqa: E402
    BenchmarkCase,
    DrugConflictsCatalog,
    RedFlagsCatalog,
    ScalesCatalog,
    WebSourcesCatalog,
)


def _load_yaml(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_all(project_root: Path = PROJECT_ROOT) -> Tuple[bool, List[str], List[str]]:
    """Validate knowledge_base/ and app_benchmark/test_cases/.

    Returns:
        (passed, info_messages, error_messages)
    """
    infos: List[str] = []
    errors: List[str] = []

    kb_dir = project_root / "knowledge_base"
    known_rule_ids: Set[str] = set()
    known_scale_ids: Set[str] = set()

    # 1. Red Flags
    rf_path = kb_dir / "red_flags.yaml"
    try:
        rf_catalog = RedFlagsCatalog.model_validate(_load_yaml(rf_path))
        known_rule_ids.update(r.rule_id for r in rf_catalog.rules)
        infos.append(f"[OK] red_flags.yaml: {len(rf_catalog.rules)} 条红旗规则校验通过")
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        errors.append(f"[FAIL] red_flags.yaml: {exc}")

    # 2. Drug Conflicts
    dc_path = kb_dir / "drug_conflicts.yaml"
    try:
        dc_catalog = DrugConflictsCatalog.model_validate(_load_yaml(dc_path))
        known_rule_ids.update(c.conflict_id for c in dc_catalog.conflicts)
        infos.append(f"[OK] drug_conflicts.yaml: {len(dc_catalog.conflicts)} 组用药冲突规则校验通过")
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        errors.append(f"[FAIL] drug_conflicts.yaml: {exc}")

    # 3. Scales Spec
    scales_path = kb_dir / "scales_spec.yaml"
    try:
        scales_catalog = ScalesCatalog.model_validate(_load_yaml(scales_path))
        known_scale_ids.update(s.scale_id for s in scales_catalog.scales)
        scale_summary = ", ".join(f"{s.scale_id}({len(s.items)}项)" for s in scales_catalog.scales)
        infos.append(f"[OK] scales_spec.yaml: {len(scales_catalog.scales)} 个量表校验通过 -> {scale_summary}")
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        errors.append(f"[FAIL] scales_spec.yaml: {exc}")

    # 3b. Whitelisted Web Sources
    ws_path = kb_dir / "web_sources.yaml"
    try:
        ws_catalog = WebSourcesCatalog.model_validate(_load_yaml(ws_path))
        infos.append(f"[OK] web_sources.yaml: {len(ws_catalog.sources)} 个权威医学检索来源校验通过")
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        errors.append(f"[FAIL] web_sources.yaml: {exc}")

    # 4. Benchmark Test Cases (public + private if present)
    tc_root = project_root / "app_benchmark" / "test_cases"
    yaml_files = sorted(list(tc_root.rglob("*.yaml")) + list(tc_root.rglob("*.yml")))
    if not yaml_files:
        errors.append(f"[FAIL] 未在 {tc_root} 找到任何评测用例 YAML")
    else:
        seen_case_ids: Set[str] = set()
        dim_counts = {}
        for case_file in yaml_files:
            rel_path = case_file.relative_to(project_root)
            try:
                case = BenchmarkCase.model_validate(_load_yaml(case_file))
            except (OSError, yaml.YAMLError, ValidationError) as exc:
                errors.append(f"[FAIL] {rel_path}: {exc}")
                continue

            if case.case_id in seen_case_ids:
                errors.append(f"[FAIL] {rel_path}: 重复的 case_id '{case.case_id}'")
            seen_case_ids.add(case.case_id)

            if case_file.stem != case.case_id:
                errors.append(
                    f"[FAIL] {rel_path}: 文件名 '{case_file.stem}' 与 case_id '{case.case_id}' 不一致"
                )

            for ref_rule in case.expected_rule_ids:
                if known_rule_ids and ref_rule not in known_rule_ids:
                    errors.append(
                        f"[FAIL] {rel_path}: expected_rule_ids 引用了不存在的规则 '{ref_rule}'"
                    )

            if case.expected_scale_id and known_scale_ids and case.expected_scale_id not in known_scale_ids:
                errors.append(
                    f"[FAIL] {rel_path}: expected_scale_id 引用了不存在的量表 '{case.expected_scale_id}'"
                )

            dim_counts[case.dimension] = dim_counts.get(case.dimension, 0) + 1

        if not errors:
            dim_text = ", ".join(f"{k}={v}" for k, v in sorted(dim_counts.items()))
            infos.append(f"[OK] benchmark test_cases: {len(yaml_files)} 道用例校验通过 ({dim_text})")

    return len(errors) == 0, infos, errors


def main() -> int:
    passed, infos, errors = validate_all()
    for line in infos:
        print(line)
    for err in errors:
        print(err, file=sys.stderr)
    if passed:
        print("\nALL CHECKS PASSED.")
        return 0
    print(f"\nVALIDATION FAILED ({len(errors)} error(s)).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
