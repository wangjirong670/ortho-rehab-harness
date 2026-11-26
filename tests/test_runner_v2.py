"""Unit tests for Week 6 v0.2.0 runner interface extensions (tests/test_runner_v2.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from app_benchmark import runner
from core_tools.schemas import BenchmarkCase


def test_get_custom_cases_dir_resolution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ORTHO_APP_HOME", raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert runner.get_custom_cases_dir() == runner.PRIVATE_CASES_DIR

    app_home = tmp_path / "ortho_home"
    monkeypatch.setenv("ORTHO_APP_HOME", str(app_home))
    assert runner.get_custom_cases_dir() == app_home / "custom_cases"

    monkeypatch.delenv("ORTHO_APP_HOME", raising=False)
    fake_exe = tmp_path / "dist" / "OrthoRehab.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe), raising=False)
    assert runner.get_custom_cases_dir() == fake_exe.parent / "custom_cases"


def test_blank_template_and_validate_case_yaml_bytes() -> None:
    tpl = runner.get_blank_case_yaml_template()
    assert isinstance(tpl, str) and "CUSTOM_ORTHO_001" in tpl
    case, err = runner.validate_case_yaml_bytes(tpl.encode("utf-8"), source_name="template.yaml")
    assert err is None
    assert isinstance(case, BenchmarkCase)
    assert case.case_id == "CUSTOM_ORTHO_001"

    # Invalid UTF-8
    c_bad_enc, err_enc = runner.validate_case_yaml_bytes(b"\xff\xfe\x00", source_name="bad_enc.yaml")
    assert c_bad_enc is None and err_enc is not None and "UTF-8" in err_enc

    # Empty payload
    c_empty, err_empty = runner.validate_case_yaml_bytes(b"   \n", source_name="empty.yaml")
    assert c_empty is None and err_empty is not None and "为空" in err_empty

    # Bad YAML syntax
    c_syntax, err_syntax = runner.validate_case_yaml_bytes(b"case_id: [unclosed", source_name="bad.yaml")
    assert c_syntax is None and err_syntax is not None and "YAML" in err_syntax

    # Non-dict root
    c_list, err_list = runner.validate_case_yaml_bytes("- item1\n- item2\n".encode("utf-8"))
    assert c_list is None and err_list is not None and "键值对象" in err_list

    # Pydantic schema error
    bad_schema_yaml = yaml.safe_dump({"case_id": "bad-lowercase", "category": "测试"})
    c_schema, err_schema = runner.validate_case_yaml_bytes(bad_schema_yaml.encode("utf-8"))
    assert c_schema is None and err_schema is not None and "校验失败" in err_schema


def test_validate_report_json_bytes() -> None:
    valid_payload = {
        "report_id": "bench_20261126_120000",
        "runs": [
            {
                "model": "deepseek/deepseek-chat",
                "mode": "harness_copilot",
                "results": [],
                "summary": {"total_cases": 0, "passed_cases": 0},
            }
        ],
    }
    parsed, err = runner.validate_report_json_bytes(
        json.dumps(valid_payload, ensure_ascii=False).encode("utf-8")
    )
    assert err is None
    assert parsed is not None and parsed["report_id"] == "bench_20261126_120000"

    # Empty / syntax / missing fields
    assert runner.validate_report_json_bytes(b"")[1] is not None
    assert runner.validate_report_json_bytes(b"{not-json")[1] is not None
    assert runner.validate_report_json_bytes(b"[]")[1] is not None
    assert runner.validate_report_json_bytes(json.dumps({"runs": []}).encode("utf-8"))[1] is not None
    assert (
        runner.validate_report_json_bytes(
            json.dumps({"report_id": "x", "runs": "bad"}).encode("utf-8")
        )[1]
        is not None
    )
    assert (
        runner.validate_report_json_bytes(
            json.dumps({"report_id": "x", "runs": [{"model": "m"}]}).encode("utf-8")
        )[1]
        is not None
    )


def test_load_benchmark_cases_custom_and_dedup(tmp_path: Path, monkeypatch) -> None:
    private_dir = tmp_path / "private"
    custom_dir = tmp_path / "app_home" / "custom_cases"
    private_dir.mkdir(parents=True)
    custom_dir.mkdir(parents=True)

    private_case = yaml.safe_load(runner.get_blank_case_yaml_template())
    private_case["case_id"] = "PRIV_CASE_001"
    (private_dir / "PRIV_CASE_001.yaml").write_text(
        yaml.safe_dump(private_case, allow_unicode=True),
        encoding="utf-8",
    )

    custom_case = yaml.safe_load(runner.get_blank_case_yaml_template())
    custom_case["case_id"] = "CUST_CASE_002"
    (custom_dir / "CUST_CASE_002.yaml").write_text(
        yaml.safe_dump(custom_case, allow_unicode=True),
        encoding="utf-8",
    )
    # Duplicate case_id with private_dir should be deduped cleanly
    dup_case = yaml.safe_load(runner.get_blank_case_yaml_template())
    dup_case["case_id"] = "PRIV_CASE_001"
    (custom_dir / "PRIV_DUP.yaml").write_text(
        yaml.safe_dump(dup_case, allow_unicode=True),
        encoding="utf-8",
    )

    monkeypatch.setattr(runner, "PRIVATE_CASES_DIR", private_dir)

    # 1. When private_dir == custom_dir (dev mode), do NOT load twice
    monkeypatch.delenv("ORTHO_APP_HOME", raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    both_same = runner.load_benchmark_cases(include_private=True, include_custom=True)
    priv_hits = [(c.case_id, vis) for c, vis in both_same if c.case_id == "PRIV_CASE_001"]
    assert priv_hits == [("PRIV_CASE_001", "private")]

    # 2. When ORTHO_APP_HOME points to custom_cases, include_custom loads custom visibility & dedups
    monkeypatch.setenv("ORTHO_APP_HOME", str(tmp_path / "app_home"))
    loaded = runner.load_benchmark_cases(include_private=True, include_custom=True)
    vis_map = {c.case_id: vis for c, vis in loaded}
    assert vis_map["PRIV_CASE_001"] == "private"
    assert vis_map["CUST_CASE_002"] == "custom"
    assert len([c for c, _ in loaded if c.case_id == "PRIV_CASE_001"]) == 1


def test_run_benchmark_filtering_extra_cases_and_progress_callback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_MOCK_MODE", "true")

    extra_case, err = runner.validate_case_yaml_bytes(
        runner.get_blank_case_yaml_template().encode("utf-8")
    )
    assert err is None and extra_case is not None

    # Override an existing case_id + add a brand new custom case
    override_001 = extra_case.model_copy(
        update={
            "case_id": "ORTHO_REDFLAG_001",
            "category": "覆盖测试类别",
        }
    )

    events: List[Dict[str, Any]] = []
    report = runner.run_benchmark(
        models=["deepseek/deepseek-chat"],
        modes=("harness_copilot",),
        mock_mode=True,
        save_report=True,
        output_dir=tmp_path,
        extra_cases=[(override_001, "custom"), (extra_case, "custom")],
        case_ids=["ORTHO_REDFLAG_001", "CUSTOM_ORTHO_001"],
        progress_callback=events.append,
    )

    assert report["case_count"] == 2
    assert len(report["runs"]) == 1
    result_ids = [r["case_id"] for r in report["runs"][0]["results"]]
    assert result_ids == ["ORTHO_REDFLAG_001", "CUSTOM_ORTHO_001"]
    assert report["runs"][0]["results"][0]["category"] == "覆盖测试类别"

    # Progress callback verification
    assert len(events) == 2
    for idx, ev in enumerate(events, start=1):
        assert set(ev.keys()) >= {
            "done",
            "total",
            "model",
            "mode",
            "case_id",
            "title",
            "passed",
            "latency_ms",
            "report_path",
        }
        assert ev["done"] == idx
        assert ev["total"] == 2
        assert ev["model"] == "deepseek/deepseek-chat"
        assert ev["mode"] == "harness_copilot"
        assert ev["report_path"] == report["report_path"]

    assert report["report_path"] is not None
    assert Path(report["report_path"]).exists()

    # Category filter test
    cat_report = runner.run_benchmark(
        models=["deepseek/deepseek-chat"],
        modes=("bare_llm",),
        mock_mode=True,
        save_report=False,
        categories=["骨伤手法禁忌-红旗拦截"],
    )
    assert cat_report["case_count"] == 1
    assert cat_report["runs"][0]["results"][0]["case_id"] == "ORTHO_REDFLAG_001"
