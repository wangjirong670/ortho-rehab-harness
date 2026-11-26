"""Shared dependencies, runtime config, CORS, and optional Bearer token authentication."""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException, Request, status

import launcher


def init_runtime_environment() -> launcher.RuntimePaths:
    """Resolve runtime paths and load .env if not yet initialized."""
    paths = launcher.resolve_runtime_paths()
    if not paths.frozen and not os.getenv("ORTHO_RESULTS_DIR", "").strip():
        dev_results_dir = paths.bundle_dir / "app_benchmark" / "results"
        if dev_results_dir.is_dir():
            os.environ["ORTHO_RESULTS_DIR"] = str(dev_results_dir)
    launcher.configure_runtime_env(paths.bundle_dir, paths.app_home)
    return paths


def get_auth_token() -> str:
    """Return configured ORTHO_AUTH_TOKEN (empty means auth disabled)."""
    return os.getenv("ORTHO_AUTH_TOKEN", "").strip()


def get_cors_origins() -> List[str]:
    """Parse ORTHO_CORS_ORIGINS (comma-separated) or return local dev defaults."""
    raw = os.getenv("ORTHO_CORS_ORIGINS", "").strip()
    if raw:
        origins = [item.strip() for item in raw.split(",") if item.strip()]
        if origins:
            return origins
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


def get_app_mode() -> str:
    """Return runtime mode ('local_desktop' or 'server')."""
    raw = os.getenv("ORTHO_APP_MODE", "server").strip()
    return "local_desktop" if raw == "local_desktop" else "server"


def resolve_frontend_dist_dir() -> Path:
    """Locate compiled frontend/dist directory across dev and frozen modes."""
    paths = launcher.resolve_runtime_paths()
    for base in (paths.bundle_dir, paths.app_home):
        candidate = base / "frontend" / "dist"
        if candidate.is_dir():
            return candidate
    return paths.bundle_dir / "frontend" / "dist"


def verify_bearer_token(request: Request) -> None:
    """Enforce Authorization: Bearer <token> on /api/v1/* when ORTHO_AUTH_TOKEN is non-empty."""
    expected = get_auth_token()
    if not expected:
        return

    auth_header = request.headers.get("Authorization", "").strip()
    provided: Optional[str] = None
    if auth_header:
        parts = auth_header.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            provided = parts[1].strip()
    elif request.url.path.endswith("/stream"):
        # Browser EventSource cannot attach custom Authorization headers.
        query_token = request.query_params.get("token", "").strip()
        if query_token:
            provided = query_token

    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未授权访问：请提供有效的 Authorization: Bearer <token> 凭证。",
            headers={"WWW-Authenticate": "Bearer"},
        )
