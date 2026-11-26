"""Build Windows --onedir PyInstaller distribution and release ZIP for OrthoRehab-TCM-Harness.

Usage:
    python packaging/build_release.py --version v0.1.0
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

APP_NAME = "OrthoRehab-TCM-Harness"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PACKAGES: Tuple[str, ...] = (
    "streamlit",
    "litellm",
    "openai",
    "pydantic",
    "pydantic_core",
    "plotly",
    "packaging",
    "requests",
    "tiktoken",
    "pyyaml",
    "python-dotenv",
    "certifi",
    "altair",
    "pyarrow",
    "narwhals",
    "click",
    "httpx",
    "aiohttp",
    "jsonschema",
    "uvicorn",
    "starlette",
    "fastapi",
    "anyio",
    "rich",
    "tornado",
    "tenacity",
)

DATA_PACKAGES: Tuple[str, ...] = (
    "streamlit",
    "plotly",
    "litellm",
    "tiktoken_ext",
    "certifi",
    "altair",
)

EXPLICIT_HIDDEN_IMPORTS: Tuple[str, ...] = (
    "streamlit.runtime.scriptrunner.magic_funcs",
    "streamlit.web.cli",
    "streamlit.web.bootstrap",
    "streamlit.runtime.caching",
    "tiktoken_ext.openai_public",
    "tiktoken_ext",
    "yaml",
    "dotenv",
    "pydantic",
    "plotly",
    "litellm",
    "openai",
    "tkinter",
    "tkinter.ttk",
    "tkinter.messagebox",
)

_ALLOWED_PRIVATE_FILENAMES = frozenset({"README.md", ".gitkeep"})


def normalize_version(version: str) -> str:
    """Validate and normalize release version tag (e.g. v0.1.0)."""
    cleaned = (version or "").strip()
    if not cleaned:
        raise ValueError("Release version must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cleaned):
        raise ValueError(f"Invalid release version '{cleaned}'")
    return cleaned


def is_forbidden_bundle_path(rel_posix: str, *, allow_third_party_pem: bool = False) -> bool:
    """Return True if a relative POSIX path must NEVER enter bundle or release ZIP."""
    norm = rel_posix.strip().replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    norm = norm.lstrip("/")
    if not norm:
        return False

    parts = norm.split("/")
    filename = parts[-1]

    # Cache, VCS, virtualenv, build outputs
    forbidden_dirs = {"__pycache__", ".git", ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", "release_artifacts"}
    if any(part in forbidden_dirs for part in parts):
        return True

    if filename.endswith((".pyc", ".pyo", ".secret", ".key")):
        return True
    if filename.endswith(".pem") and not allow_third_party_pem:
        return True

    # Secrets / .env files (keep .env.example only)
    if filename == ".env" or (filename.startswith(".env.") and filename != ".env.example"):
        return True
    if norm.endswith(".streamlit/secrets.toml"):
        return True

    # Private benchmark cases: allow ONLY README.md and .gitkeep
    if "app_benchmark/test_cases/private/" in norm or norm.startswith("test_cases/private/"):
        if filename not in _ALLOWED_PRIVATE_FILENAMES:
            return True

    # Historical benchmark JSON reports / outputs must never be bundled
    if (
        norm.startswith("app_benchmark/results/")
        or norm.startswith("app_benchmark/outputs/")
        or norm.startswith("app_benchmark/reports/")
        or "/results/" in f"/{norm}"
    ):
        if not norm.endswith("/results"):
            return True

    return False


def collect_whitelisted_bundle_files(project_root: Path = PROJECT_ROOT) -> List[Tuple[Path, str]]:
    """Collect strictly whitelisted business source/asset files for PyInstaller datas."""
    entries: List[Tuple[Path, str]] = []

    def _add(src: Path, dest_dir: str) -> None:
        if not src.is_file():
            return
        rel = src.relative_to(project_root).as_posix()
        if is_forbidden_bundle_path(rel):
            raise RuntimeError(f"Refusing to bundle forbidden path: {rel}")
        entries.append((src, dest_dir))

    for p in sorted((project_root / "knowledge_base").glob("*.yaml")):
        _add(p, "knowledge_base")

    for pkg in ("core_tools", "app_copilot", "app_benchmark"):
        for p in sorted((project_root / pkg).glob("*.py")):
            _add(p, pkg)

    for p in sorted((project_root / "app_benchmark" / "test_cases" / "public").glob("*.yaml")):
        _add(p, "app_benchmark/test_cases/public")

    private_dir = project_root / "app_benchmark" / "test_cases" / "private"
    for fname in sorted(_ALLOWED_PRIVATE_FILENAMES):
        candidate = private_dir / fname
        if candidate.is_file():
            _add(candidate, "app_benchmark/test_cases/private")

    env_example = project_root / ".env.example"
    if env_example.is_file():
        _add(env_example, ".")

    return entries


def collect_outer_release_files(project_root: Path = PROJECT_ROOT) -> List[Tuple[Path, str]]:
    """Collect user-facing files placed directly alongside OrthoRehab-TCM-Harness.exe."""
    specs = [
        (project_root / ".env.example", ".env.example"),
        (project_root / "LICENSE", "LICENSE"),
        (project_root / "packaging" / "README_RELEASE.txt", "README_RELEASE.txt"),
    ]
    missing = [str(src) for src, _ in specs if not src.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required outer release file(s): {missing}")
    return specs


def audit_dist_directory(dist_dir: Path) -> List[str]:
    """Scan built distribution directory and return any forbidden relative paths."""
    violations: List[str] = []
    if not dist_dir.exists():
        return [f"Missing distribution directory: {dist_dir}"]

    for path in sorted(dist_dir.rglob("*")):
        rel = path.relative_to(dist_dir).as_posix()
        # Strip PyInstaller's _internal prefix before applying repo-level forbidden rules
        in_internal = rel.startswith("_internal/")
        logical_rel = rel[len("_internal/"):] if in_internal else rel
        if path.is_dir():
            if path.name in {"__pycache__", ".git", ".venv", "venv", ".pytest_cache"}:
                violations.append(rel)
            continue
        if logical_rel == "results":
            continue
        repo_owned = logical_rel.startswith(("knowledge_base/", "core_tools/", "app_copilot/", "app_benchmark/"))
        if is_forbidden_bundle_path(logical_rel, allow_third_party_pem=(in_internal and not repo_owned)):
            violations.append(rel)
    return violations


def generate_pyinstaller_spec(
    *,
    project_root: Path = PROJECT_ROOT,
    spec_path: Path,
    bundle_files: Sequence[Tuple[Path, str]],
) -> Path:
    """Write a deterministic PyInstaller --onedir .spec file."""
    datas_literal = repr([(str(src), dest) for src, dest in bundle_files])
    metadata_literal = repr(list(METADATA_PACKAGES))
    data_pkgs_literal = repr(list(DATA_PACKAGES))
    hidden_literal = repr(list(EXPLICIT_HIDDEN_IMPORTS))
    launcher_literal = repr(str(project_root / "launcher.py"))
    pathex_literal = repr([str(project_root)])

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
import importlib.metadata as _md
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

binaries = []
for _bpkg in ("litellm",):
    try:
        binaries += collect_dynamic_libs(_bpkg)
    except Exception:
        pass

datas = {datas_literal}
for _pkg in {metadata_literal}:
    try:
        _md.distribution(_pkg)
        datas += copy_metadata(_pkg)
    except _md.PackageNotFoundError:
        pass

for _dpkg in {data_pkgs_literal}:
    try:
        datas += collect_data_files(_dpkg)
    except Exception:
        pass

hiddenimports = {hidden_literal}
for _app_pkg in (
    "core_tools",
    "app_copilot",
    "app_benchmark",
    "litellm.rust_bridge",
    "litellm.litellm_core_utils",
    "tiktoken_ext",
):
    try:
        hiddenimports += collect_submodules(_app_pkg)
    except Exception:
        pass

try:
    hiddenimports += collect_submodules(
        "streamlit",
        filter=lambda name: not name.startswith("streamlit.testing"),
    )
except Exception:
    pass

a = Analysis(
    [{launcher_literal}],
    pathex={pathex_literal},
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=["pytest", "_pytest", "IPython", "jupyter"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name={APP_NAME!r},
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name={APP_NAME!r},
)
"""
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(spec_content, encoding="utf-8")
    return spec_path


def compute_sha256(file_path: Path) -> str:
    """Compute lowercase SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def package_zip_and_checksum(
    *,
    dist_dir: Path,
    release_dir: Path,
    version: str,
) -> Tuple[Path, Path]:
    """Archive dist_dir into release_artifacts/OrthoRehab-TCM-Harness-<ver>-win64.zip + SHA256SUMS.txt."""
    release_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"{APP_NAME}-{version}-win64.zip"
    zip_path = release_dir / zip_name
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Ensure empty results/ folder is preserved in archive
        results_arc = f"{APP_NAME}/results/"
        zf.writestr(zipfile.ZipInfo(results_arc), b"")

        for item in sorted(dist_dir.rglob("*")):
            rel = item.relative_to(dist_dir).as_posix()
            arcname = f"{APP_NAME}/{rel}"
            if item.is_dir():
                if rel == "results":
                    continue
                zf.writestr(zipfile.ZipInfo(f"{arcname}/"), b"")
            else:
                zf.write(item, arcname)

    sha256_hex = compute_sha256(zip_path)
    sums_path = release_dir / "SHA256SUMS.txt"
    sums_path.write_text(f"{sha256_hex}  {zip_name}\n", encoding="utf-8")
    return zip_path, sums_path


def build_release(
    *,
    version: str,
    project_root: Path = PROJECT_ROOT,
    python_exe: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """Run full PyInstaller --onedir build, security audit, ZIP creation, and SHA256SUMS."""
    ver = normalize_version(version)
    py_cmd = python_exe or sys.executable

    dist_root = project_root / "dist"
    build_root = project_root / "build"
    dist_dir = dist_root / APP_NAME
    release_dir = project_root / "release_artifacts"
    spec_path = build_root / f"{APP_NAME}.spec"

    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    bundle_files = collect_whitelisted_bundle_files(project_root)
    generate_pyinstaller_spec(
        project_root=project_root,
        spec_path=spec_path,
        bundle_files=bundle_files,
    )

    cmd = [
        py_cmd,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(dist_root),
        "--workpath",
        str(build_root / "pyinstaller_work"),
        str(spec_path),
    ]
    print(f"[build_release] Running PyInstaller: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(project_root), check=True)

    exe_path = dist_dir / f"{APP_NAME}.exe"
    if not exe_path.is_file():
        raise FileNotFoundError(f"Expected built executable not found: {exe_path}")

    for src, dest_name in collect_outer_release_files(project_root):
        shutil.copyfile(src, dist_dir / dest_name)

    (dist_dir / "results").mkdir(parents=True, exist_ok=True)

    violations = audit_dist_directory(dist_dir)
    if violations:
        raise RuntimeError(f"Security audit failed! Forbidden files in dist: {violations}")

    zip_path, sums_path = package_zip_and_checksum(
        dist_dir=dist_dir,
        release_dir=release_dir,
        version=ver,
    )
    print(f"[build_release] Built exe:  {exe_path}")
    print(f"[build_release] Built zip:  {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"[build_release] SHA256SUMS: {sums_path}")
    return exe_path, zip_path, sums_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Windows Release ZIP for OrthoRehab-TCM-Harness")
    parser.add_argument("--version", default="v0.1.0", help="Release version tag (e.g. v0.1.0)")
    args = parser.parse_args(argv)
    build_release(version=args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
