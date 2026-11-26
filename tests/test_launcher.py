"""Tests for desktop launcher, frozen path resolution, --check-env, and release packaging whitelist."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import launcher
from app_benchmark import runner as bench_runner
from app_benchmark import streamlit_bench
from core_tools import llm_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_build_release_module():
    spec = importlib.util.spec_from_file_location(
        "ortho_packaging_build_release",
        PROJECT_ROOT / "packaging" / "build_release.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_release = _load_build_release_module()


def test_resolve_runtime_paths_dev_and_frozen(tmp_path: Path) -> None:
    dev_paths = launcher.resolve_runtime_paths(frozen=False, file_path=PROJECT_ROOT / "launcher.py")
    assert dev_paths.frozen is False
    assert dev_paths.bundle_dir == PROJECT_ROOT
    assert dev_paths.app_home == PROJECT_ROOT

    meipass_dir = tmp_path / "bundle_internal"
    app_dir = tmp_path / "portable_app"
    exe_path = app_dir / "OrthoRehab-TCM-Harness.exe"
    frozen_paths = launcher.resolve_runtime_paths(
        frozen=True,
        meipass=str(meipass_dir),
        executable=str(exe_path),
    )
    assert frozen_paths.frozen is True
    assert frozen_paths.bundle_dir == meipass_dir.resolve()
    assert frozen_paths.app_home == app_dir.resolve()


def test_configure_runtime_env_and_frozen_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home = tmp_path / "dist_app"
    app_home.mkdir()
    (app_home / ".env").write_text("COPILOT_DEFAULT_MODEL=mock/custom-test\n", encoding="utf-8")

    monkeypatch.delenv("ORTHO_APP_HOME", raising=False)
    monkeypatch.delenv("ORTHO_RESULTS_DIR", raising=False)
    monkeypatch.delenv("ORTHO_BENCH_RESULTS_DIR", raising=False)
    monkeypatch.delenv("COPILOT_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("LLM_MOCK_MODE", raising=False)

    info = launcher.configure_runtime_env(PROJECT_ROOT, app_home, mock=True)
    assert info["env_exists"] is True
    assert info["env_loaded"] is True
    assert info["results_dir"] == str(app_home / "results")
    assert llm_client._resolve_env_file() == app_home / ".env"
    assert bench_runner.get_default_results_dir() == app_home / "results"
    assert streamlit_bench.get_results_dir() == app_home / "results"

    cfg = llm_client.LLMConfig.from_env()
    assert cfg.default_model == "mock/custom-test"
    assert cfg.mock_mode is True


def test_build_streamlit_argv_and_child_command(tmp_path: Path) -> None:
    copilot_argv = launcher.build_streamlit_argv(
        "copilot",
        bundle_dir=tmp_path,
        port=8511,
        headless=True,
        frozen=True,
    )
    assert copilot_argv[:3] == [
        "streamlit",
        "run",
        str(tmp_path / "app_copilot" / "streamlit_copilot.py"),
    ]
    assert "--global.developmentMode=false" in copilot_argv
    assert "--browser.gatherUsageStats=false" in copilot_argv
    assert "--server.port=8511" in copilot_argv
    assert "--server.headless=true" in copilot_argv
    assert "--server.fileWatcherType=none" in copilot_argv

    bench_argv = launcher.build_streamlit_argv(
        "benchmark",
        bundle_dir=tmp_path,
        port=None,
        headless=False,
        frozen=False,
    )
    assert bench_argv[2] == str(tmp_path / "app_benchmark" / "streamlit_bench.py")
    assert "--server.port=8502" in bench_argv
    assert "--server.headless=false" in bench_argv
    assert "--server.fileWatcherType=none" not in bench_argv

    frozen_cmd = launcher.build_child_command(
        "copilot",
        port=8501,
        headless=True,
        mock=True,
        frozen=True,
        executable="C:\\app\\OrthoRehab-TCM-Harness.exe",
    )
    assert frozen_cmd == [
        "C:\\app\\OrthoRehab-TCM-Harness.exe",
        "--mode",
        "copilot",
        "--port",
        "8501",
        "--headless",
        "--mock",
    ]


def test_ensure_env_file_and_initial_mock_preference(tmp_path: Path) -> None:
    app_home = tmp_path / "app"
    bundle_dir = tmp_path / "bundle"
    app_home.mkdir()
    bundle_dir.mkdir()
    (bundle_dir / ".env.example").write_text("LLM_MOCK_MODE=false\n", encoding="utf-8")

    # First run without .env defaults to Mock=True for zero-config experience
    assert launcher.read_initial_mock_preference(app_home) is True

    created = launcher.ensure_env_file(app_home, bundle_dir)
    assert created == app_home / ".env"
    assert created.read_text(encoding="utf-8") == "LLM_MOCK_MODE=false\n"
    assert launcher.read_initial_mock_preference(app_home) is False
    assert launcher.read_initial_mock_preference(app_home, cli_mock=True) is True


def test_check_env_mode_succeeds() -> None:
    paths = launcher.resolve_runtime_paths(frozen=False, file_path=PROJECT_ROOT / "launcher.py")
    code, payload = launcher.run_env_check(paths)
    assert code == 0
    assert payload["status"] == "ok"
    assert payload["knowledge_base"]["red_flags"] >= 16
    assert payload["knowledge_base"]["drug_conflicts"] >= 11
    assert payload["knowledge_base"]["scales"] >= 4
    assert payload["knowledge_base"]["public_cases"] >= 20

    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "launcher.py"), "--check-env"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    cli_payload = json.loads(proc.stdout)
    assert cli_payload["status"] == "ok"
    assert cli_payload["scripts"]["copilot"] is True
    assert cli_payload["scripts"]["benchmark"] is True


def test_packaging_whitelist_and_security_audit(tmp_path: Path) -> None:
    assert build_release.is_forbidden_bundle_path(".env") is True
    assert build_release.is_forbidden_bundle_path(".env.local") is True
    assert build_release.is_forbidden_bundle_path(".env.example") is False
    assert build_release.is_forbidden_bundle_path("app_benchmark/test_cases/private/secret_case.yaml") is True
    assert build_release.is_forbidden_bundle_path("app_benchmark/test_cases/private/README.md") is False
    assert build_release.is_forbidden_bundle_path("app_benchmark/results/bench_20261126.json") is True
    assert build_release.is_forbidden_bundle_path("core_tools/__pycache__/schemas.cpython-311.pyc") is True
    assert build_release.is_forbidden_bundle_path(".git/config") is True
    assert build_release.is_forbidden_bundle_path(".venv/pyvenv.cfg") is True
    assert build_release.is_forbidden_bundle_path("knowledge_base/red_flags.yaml") is False

    bundle_files = build_release.collect_whitelisted_bundle_files(PROJECT_ROOT)
    rel_paths = [src.relative_to(PROJECT_ROOT).as_posix() for src, _ in bundle_files]
    assert "knowledge_base/red_flags.yaml" in rel_paths
    assert "app_benchmark/test_cases/public/ORTHO_REDFLAG_001.yaml" in rel_paths
    assert "app_benchmark/test_cases/private/README.md" in rel_paths
    assert not any(p == ".env" or p.endswith(".pyc") for p in rel_paths)
    assert not any(
        p.startswith("app_benchmark/test_cases/private/") and not p.endswith(("README.md", ".gitkeep"))
        for p in rel_paths
    )
    assert not any(p.startswith("app_benchmark/results/") for p in rel_paths)

    # Audit catches forbidden files injected into dist directory
    fake_dist = tmp_path / "OrthoRehab-TCM-Harness"
    (fake_dist / "_internal" / "app_benchmark" / "test_cases" / "private").mkdir(parents=True)
    (fake_dist / "_internal" / "app_benchmark" / "test_cases" / "private" / "README.md").write_text("ok", encoding="utf-8")
    (fake_dist / "results").mkdir()
    assert build_release.audit_dist_directory(fake_dist) == []

    (fake_dist / ".env").write_text("SECRET=1", encoding="utf-8")
    (fake_dist / "_internal" / "app_benchmark" / "test_cases" / "private" / "leak.yaml").write_text("x: 1", encoding="utf-8")
    (fake_dist / "_internal" / "app_benchmark" / "results").mkdir(parents=True)
    (fake_dist / "_internal" / "app_benchmark" / "results" / "old.json").write_text("{}", encoding="utf-8")
    violations = build_release.audit_dist_directory(fake_dist)
    assert ".env" in violations
    assert "_internal/app_benchmark/test_cases/private/leak.yaml" in violations
    assert "_internal/app_benchmark/results/old.json" in violations
