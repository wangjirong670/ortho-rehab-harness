"""Unit tests for Week 5 benchmark UI helpers: multi-model colors, deltas, rescue filter, side-by-side, and exports."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from streamlit.testing.v1 import AppTest

from app_benchmark.streamlit_bench import (
    DIMENSION_ORDER,
    assign_model_colors,
    build_bar_figure,
    build_dimension_comparison_rows,
    build_dimension_csv_bytes,
    build_failure_replay_markdown,
    build_radar_figure,
    build_report_json_bytes,
    build_side_by_side_case_pairs,
    collect_report_infra_errors,
    filter_case_results,
    find_rescued_case_keys,
    merge_model_options,
)
from tests.test_bench_ui import BENCH_APP_PATH, build_sample_report


def _build_multi_model_report() -> Dict[str, Any]:
    """Construct a 2-model x 2-mode report with distinct rescue and regression patterns."""
    base = build_sample_report(report_id="multi_model_20261126")
    runs_m1 = base["runs"]
    runs_m1[0]["model"] = "deepseek/deepseek-chat"
    runs_m1[0]["run_label"] = "deepseek/deepseek-chat | bare_llm"
    runs_m1[1]["model"] = "deepseek/deepseek-chat"
    runs_m1[1]["run_label"] = "deepseek/deepseek-chat | harness_copilot"

    m2_bare = json.loads(json.dumps(runs_m1[0]))
    m2_bare["model"] = "ollama_chat/qwen2.5:7b"
    m2_bare["run_label"] = "ollama_chat/qwen2.5:7b | bare_llm"
    m2_bare["summary"]["overall_pass_rate"] = 50.0
    m2_bare["summary"]["passed_cases"] = 2
    m2_bare["summary"]["dimensions"]["red_flag_interception"] = {
        "total": 1,
        "passed": 1,
        "pass_rate": 100.0,
    }
    m2_bare["results"][0]["status"] = "PASS"
    m2_bare["results"][0]["passed"] = True
    m2_bare["results"][0]["failure_reasons"] = []
    m2_bare["results"][0]["forbidden_violations"] = []

    m2_harness = json.loads(json.dumps(runs_m1[1]))
    m2_harness["model"] = "ollama_chat/qwen2.5:7b"
    m2_harness["run_label"] = "ollama_chat/qwen2.5:7b | harness_copilot"
    m2_harness["summary"]["overall_pass_rate"] = 75.0
    m2_harness["summary"]["passed_cases"] = 3
    m2_harness["summary"]["dimensions"]["tcm_syndrome_and_tools"] = {
        "total": 1,
        "passed": 0,
        "pass_rate": 0.0,
    }
    m2_harness["results"][2]["status"] = "FAIL_ACCURACY"
    m2_harness["results"][2]["passed"] = False
    m2_harness["results"][2]["failure_reasons"] = ["数值计算偏差"]

    return {
        "report_id": "multi_model_20261126",
        "report_path": "",
        "created_at": "2026-11-26 16:30:00",
        "mock_mode": False,
        "include_private": False,
        "case_count": 4,
        "runs": [*runs_m1, m2_bare, m2_harness],
    }


def test_multi_model_colors_and_mode_styles() -> None:
    report = _build_multi_model_report()
    runs = report["runs"]
    colors = assign_model_colors(runs)

    assert len(colors) == 2
    assert colors["deepseek/deepseek-chat"] != colors["ollama_chat/qwen2.5:7b"]

    radar = build_radar_figure(runs)
    assert len(radar.data) == 4
    # Same model shares base RGB across bare/harness, while bare uses dash and harness uses solid.
    assert radar.data[0].line.dash == "dash"
    assert radar.data[1].line.dash == "solid"
    assert radar.data[2].line.dash == "dash"
    assert radar.data[3].line.dash == "solid"
    # Different models do not collide in solid trace colors.
    assert radar.data[1].line.color != radar.data[3].line.color

    bar = build_bar_figure(runs)
    assert len(bar.data) == 4
    assert bar.data[0].marker.pattern.shape == "/"
    assert bar.data[1].marker.color == colors["deepseek/deepseek-chat"]
    assert bar.data[3].marker.color == colors["ollama_chat/qwen2.5:7b"]
    assert bar.data[1].marker.color != bar.data[3].marker.color


def test_dimension_comparison_rows_preserve_per_model_deltas() -> None:
    report = _build_multi_model_report()
    rows = build_dimension_comparison_rows(report["runs"])
    assert len(rows) == 4
    assert [r["维度标识"] for r in rows] == DIMENSION_ORDER

    col_m1 = "护栏增益 (Δ) · deepseek/deepseek-chat"
    col_m2 = "护栏增益 (Δ) · ollama_chat/qwen2.5:7b"
    assert col_m1 in rows[0]
    assert col_m2 in rows[0]
    assert rows[0][col_m1] == "+100.0%"
    assert rows[0][col_m2] == "+0.0%"
    assert rows[2][col_m1] == "+100.0%"
    assert rows[2][col_m2] == "+0.0%"


def test_rescued_filter_search_and_side_by_side_pairs() -> None:
    report = _build_multi_model_report()
    runs = report["runs"]

    rescued_run_cases, rescued_pair_cases = find_rescued_case_keys(runs)
    assert ("deepseek/deepseek-chat", "ORTHO_REDFLAG_001") in rescued_pair_cases
    assert ("deepseek/deepseek-chat", "ORTHO_SCALE_009") in rescued_pair_cases
    assert ("deepseek/deepseek-chat", "ORTHO_CALIBRATION_010") in rescued_pair_cases
    # ORTHO_LONGITUDINAL_007 passed in both bare & harness, so not "rescued"
    assert ("deepseek/deepseek-chat", "ORTHO_LONGITUDINAL_007") not in rescued_pair_cases
    # For ollama_chat/qwen2.5:7b, ORTHO_REDFLAG_001 passed in bare already; only ORTHO_CALIBRATION_010 was rescued
    assert ("ollama_chat/qwen2.5:7b", "ORTHO_REDFLAG_001") not in rescued_pair_cases
    assert ("ollama_chat/qwen2.5:7b", "ORTHO_CALIBRATION_010") in rescued_pair_cases
    assert ("deepseek/deepseek-chat | bare_llm", "ORTHO_REDFLAG_001") in rescued_run_cases

    rescued_records = filter_case_results(runs, only_rescued=True)
    # 3 rescued cases for m1 (6 records) + 1 rescued case for m2 (2 records) = 8 records
    assert len(rescued_records) == 8

    keyword_records = filter_case_results(runs, search_query="踩棉花")
    assert len(keyword_records) >= 2
    assert all(r["result"]["case_id"] == "ORTHO_REDFLAG_001" for r in keyword_records)

    pairs_rescued = build_side_by_side_case_pairs(runs, only_rescued=True)
    assert len(pairs_rescued) == 4
    assert all(p["rescued"] for p in pairs_rescued)

    pairs_m2_scale = build_side_by_side_case_pairs(
        runs,
        selected_pair="ollama_chat/qwen2.5:7b",
        search_query="ORTHO_SCALE_009",
    )
    assert len(pairs_m2_scale) == 1
    assert pairs_m2_scale[0]["rescued"] is False
    assert pairs_m2_scale[0]["any_failed"] is True


def test_export_helpers_csv_markdown_and_json() -> None:
    report = _build_multi_model_report()
    runs = report["runs"]

    csv_bytes = build_dimension_csv_bytes(runs)
    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    csv_text = csv_bytes.decode("utf-8-sig")
    reader = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(reader) == 4
    assert "护栏增益 (Δ) · deepseek/deepseek-chat" in reader[0]
    assert "护栏增益 (Δ) · ollama_chat/qwen2.5:7b" in reader[0]

    md_text = build_failure_replay_markdown(report)
    assert "multi_model_20261126" in md_text
    assert "## 3. 护栏成功救回用例" in md_text
    assert "ORTHO_REDFLAG_001" in md_text
    assert "ORTHO_CALIBRATION_010" in md_text

    json_bytes = build_report_json_bytes(report)
    parsed = json.loads(json_bytes.decode("utf-8"))
    assert parsed["report_id"] == "multi_model_20261126"
    assert len(parsed["runs"]) == 4


def test_merge_model_options_and_collect_infra_errors() -> None:
    options, custom = merge_model_options(
        ["ollama_chat/qwen2.5:7b", "deepseek/deepseek-chat", "llama3.1:8b"],
        "dashscope/qwen-plus, ollama_chat/qwen2.5:7b\nopenai/gpt-4o-mini",
    )
    assert options[0] == "deepseek/deepseek-chat"
    assert "ollama_chat/qwen2.5:7b" in options
    assert "ollama_chat/llama3.1:8b" in options
    assert "dashscope/qwen-plus" in options
    assert "openai/gpt-4o-mini" in options
    assert len(options) == len(set(options))
    assert custom == ["dashscope/qwen-plus", "ollama_chat/qwen2.5:7b", "openai/gpt-4o-mini"]

    report = build_sample_report()
    report["runs"][0]["results"][0]["infra_error"] = "Connection refused 11434"
    errs = collect_report_infra_errors(report)
    assert errs == [("mock-model", "Connection refused 11434")]


def test_apptest_rescued_toggle_and_side_by_side_view(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "bench_multi.json"
    sample = _build_multi_model_report()
    sample["report_path"] = str(report_path)
    report_path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("ORTHO_BENCH_RESULTS_DIR", str(tmp_path))

    at = AppTest.from_file(BENCH_APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception

    # Switch to side-by-side comparison + enable rescued-only toggle
    at.radio(key="replay_view_mode").set_value("左右双栏同题对照 (Bare vs Harness)").run()
    assert not at.exception
    at.toggle(key="replay_only_rescued").set_value(True).run()
    assert not at.exception
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "ORTHO_REDFLAG_001" in markdown_text
    assert "ORTHO_CALIBRATION_010" in markdown_text
