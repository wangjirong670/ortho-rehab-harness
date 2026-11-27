"""Build Windows --onedir PyInstaller distribution and release ZIP for OrthoRehab-TCM-Harness.

Usage:
    python packaging/build_release.py --version v0.4.0
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

APP_NAME = "OrthoRehab-TCM-Harness"
DEFAULT_VERSION = "v0.4.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ICON_REL = Path("packaging") / "assets" / "app_icon.ico"

METADATA_PACKAGES: Tuple[str, ...] = (
    "fastapi",
    "uvicorn",
    "starlette",
    "httpx",
    "python-multipart",
    "pywebview",
    "pythonnet",
    "clr-loader",
    "proxy_tools",
    "bottle",
    "litellm",
    "openai",
    "pydantic",
    "pydantic_core",
    "packaging",
    "requests",
    "tiktoken",
    "pyyaml",
    "python-dotenv",
    "certifi",
    "click",
    "anyio",
    "jsonschema",
    "aiohttp",
)

DATA_PACKAGES: Tuple[str, ...] = (
    "litellm",
    "tiktoken_ext",
    "certifi",
)

COLLECT_ALL_PACKAGES: Tuple[str, ...] = (
    "webview",
    "pythonnet",
    "clr_loader",
    "proxy_tools",
    "bottle",
)

EXPLICIT_HIDDEN_IMPORTS: Tuple[str, ...] = (
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "starlette",
    "httpx",
    "python_multipart",
    "multipart",
    "tiktoken_ext.openai_public",
    "tiktoken_ext",
    "yaml",
    "dotenv",
    "pydantic",
    "litellm",
    "openai",
    "launcher",
    "desktop_app",
)

EXCLUDED_MODULES: Tuple[str, ...] = (
    "pytest",
    "_pytest",
    "IPython",
    "jupyter",
    "streamlit",
    "plotly",
    "altair",
    "pyarrow",
    "tornado",
    "pandas",
    "matplotlib",
)

_ALLOWED_PRIVATE_FILENAMES = frozenset({"README.md", ".gitkeep"})
_DEPRECATED_STREAMLIT_FILES = frozenset({"streamlit_copilot.py", "streamlit_bench.py"})


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

    # Cache, VCS, virtualenv, frontend node_modules/.vite anywhere; build outputs & runtime dirs at root
    always_forbidden_dirs = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".vite",
    }
    if any(part in always_forbidden_dirs for part in parts):
        return True

    root_forbidden_dirs = {"dist", "build", "release_artifacts", "logs", "custom_cases", "case_archives"}
    if parts[0] in root_forbidden_dirs:
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
    frontend_dist_dir = project_root / "frontend" / "dist"
    frontend_index = frontend_dist_dir / "index.html"
    if not frontend_index.is_file():
        raise FileNotFoundError(
            f"Missing built frontend SPA ({frontend_index}). Please run 'npm ci && npm run build' inside frontend/ first."
        )

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
            if p.name in _DEPRECATED_STREAMLIT_FILES:
                continue
            _add(p, pkg)

    backend_dir = project_root / "backend"
    if backend_dir.is_dir():
        for p in sorted(backend_dir.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            _add(p, p.parent.relative_to(project_root).as_posix())

    for p in sorted(frontend_dist_dir.rglob("*")):
        if p.is_file():
            _add(p, p.parent.relative_to(project_root).as_posix())

    for entry_py in ("launcher.py", "desktop_app.py"):
        candidate = project_root / entry_py
        if candidate.is_file():
            _add(candidate, ".")

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

    icon_file = project_root / APP_ICON_REL
    if icon_file.is_file():
        _add(icon_file, "packaging/assets")

    return entries


def _render_shield_rgba(size: int) -> bytes:
    """Render a crisp medical shield + white cross RGBA icon bitmap without Pillow."""
    buf = bytearray(size * size * 4)
    # Colors (B, G, R, A)
    teal_dark = (110, 118, 15, 255)   # #0F766E
    teal_mid = (136, 148, 13, 255)    # #0D9488
    teal_border = (166, 184, 20, 255) # #14B8A6
    white = (255, 255, 255, 255)
    amber = (36, 191, 251, 255)       # #FBBF24 accent bar

    for y in range(size):
        ny = (y + 0.5) / size
        # Shield half-width profile by normalized vertical coordinate
        if ny < 0.10 or ny > 0.92:
            half_w = -1.0
        elif ny <= 0.50:
            half_w = 0.38
        else:
            t = (ny - 0.50) / 0.42
            half_w = 0.38 * max(0.0, 1.0 - t * t)

        for x in range(size):
            nx = (x + 0.5) / size
            dx = abs(nx - 0.5)
            idx = ((size - 1 - y) * size + x) * 4  # bottom-up DIB row order

            if half_w <= 0 or dx > half_w:
                continue

            # Border detection
            is_border = (
                ny < 0.145
                or dx > half_w - 0.045
                or (ny > 0.50 and half_w - dx < 0.055)
            )
            # Medical cross inside center
            in_v_arm = dx <= 0.075 and 0.25 <= ny <= 0.67
            in_h_arm = abs(ny - 0.46) <= 0.075 and dx <= 0.21
            in_accent = abs(ny - 0.74) <= 0.025 and dx <= 0.10

            if in_v_arm or in_h_arm:
                color = white
            elif in_accent and not is_border:
                color = amber
            elif is_border:
                color = teal_border
            elif nx < 0.5:
                color = teal_dark
            else:
                color = teal_mid

            buf[idx:idx + 4] = bytes(color)

    return bytes(buf)


def generate_app_icon(icon_path: Path, sizes: Sequence[int] = (16, 32, 48, 64)) -> Path:
    """Generate a multi-resolution Windows .ico file using pure Python stdlib."""
    images: List[Tuple[int, bytes]] = []
    for sz in sizes:
        xor_data = _render_shield_rgba(sz)
        row_bytes = ((sz + 31) // 32) * 4
        and_data = bytearray(row_bytes * sz)
        for y_bottom in range(sz):
            for x in range(sz):
                alpha = xor_data[(y_bottom * sz + x) * 4 + 3]
                if alpha < 128:
                    and_data[y_bottom * row_bytes + (x // 8)] |= 1 << (7 - (x % 8))

        header = struct.pack(
            "<IIIHHIIIIII",
            40,
            sz,
            sz * 2,
            1,
            32,
            0,
            len(xor_data) + len(and_data),
            0,
            0,
            0,
            0,
        )
        images.append((sz, header + xor_data + bytes(and_data)))

    count = len(images)
    ico_dir = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries = bytearray()
    payload = bytearray()
    for sz, dib in images:
        dim_byte = 0 if sz >= 256 else sz
        entries += struct.pack("<BBBBHHII", dim_byte, dim_byte, 0, 0, 1, 32, len(dib), offset)
        payload += dib
        offset += len(dib)

    icon_path.parent.mkdir(parents=True, exist_ok=True)
    icon_path.write_bytes(ico_dir + bytes(entries) + bytes(payload))
    return icon_path


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
            if path.name in {"__pycache__", ".git", ".venv", "venv", ".pytest_cache", "node_modules", ".vite", "streamlit"}:
                violations.append(rel)
            continue
        if logical_rel == "results":
            continue
        if path.name in _DEPRECATED_STREAMLIT_FILES:
            violations.append(rel)
            continue
        repo_owned = logical_rel.startswith(
            ("knowledge_base/", "core_tools/", "app_copilot/", "app_benchmark/", "backend/", "frontend/")
        )
        if is_forbidden_bundle_path(logical_rel, allow_third_party_pem=(in_internal and not repo_owned)):
            violations.append(rel)
    return violations


def generate_pyinstaller_spec(
    *,
    project_root: Path = PROJECT_ROOT,
    spec_path: Path,
    bundle_files: Sequence[Tuple[Path, str]],
    icon_path: Optional[Path] = None,
) -> Path:
    """Write a deterministic PyInstaller --onedir .spec file targeting desktop_app.py."""
    datas_literal = repr([(str(src), dest) for src, dest in bundle_files])
    metadata_literal = repr(list(METADATA_PACKAGES))
    data_pkgs_literal = repr(list(DATA_PACKAGES))
    collect_all_literal = repr(list(COLLECT_ALL_PACKAGES))
    hidden_literal = repr(list(EXPLICIT_HIDDEN_IMPORTS))
    excludes_literal = repr(list(EXCLUDED_MODULES))
    entry_literal = repr(str(project_root / "desktop_app.py"))
    pathex_literal = repr([str(project_root)])
    resolved_icon = icon_path if (icon_path and icon_path.is_file()) else None
    icon_literal = repr(str(resolved_icon)) if resolved_icon else "None"

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
import importlib.metadata as _md
from PyInstaller.utils.hooks import (
    collect_all,
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
for _all_pkg in {collect_all_literal}:
    try:
        _d, _b, _h = collect_all(_all_pkg)
        datas += _d
        binaries += _b
        hiddenimports += _h
    except Exception:
        pass

for _app_pkg in (
    "backend",
    "core_tools",
    "uvicorn",
    "fastapi",
    "starlette",
    "litellm.rust_bridge",
    "litellm.litellm_core_utils",
    "tiktoken_ext",
):
    try:
        hiddenimports += collect_submodules(_app_pkg)
    except Exception:
        pass

for _filtered_pkg in ("app_copilot", "app_benchmark"):
    try:
        hiddenimports += collect_submodules(
            _filtered_pkg,
            filter=lambda name: not name.endswith(("streamlit_copilot", "streamlit_bench")),
        )
    except Exception:
        pass

a = Analysis(
    [{entry_literal}],
    pathex={pathex_literal},
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={excludes_literal},
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon={icon_literal},
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

    icon_path = project_root / APP_ICON_REL
    if not icon_path.is_file():
        generate_app_icon(icon_path)

    bundle_files = collect_whitelisted_bundle_files(project_root)
    generate_pyinstaller_spec(
        project_root=project_root,
        spec_path=spec_path,
        bundle_files=bundle_files,
        icon_path=icon_path,
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
    parser.add_argument("--version", default=DEFAULT_VERSION, help=f"Release version tag (e.g. {DEFAULT_VERSION})")
    args = parser.parse_args(argv)
    build_release(version=args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
