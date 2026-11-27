"""FastAPI main application entrypoint for OrthoRehab-TCM-Harness."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.deps import (
    get_app_mode,
    get_auth_token,
    get_cors_origins,
    init_runtime_environment,
    resolve_frontend_dist_dir,
    verify_bearer_token,
)
from backend.routers import benchmark, cases, copilot, kb, settings
from core_tools.llm_client import LLMConfig

APP_VERSION = "0.3.0"


@asynccontextmanager
async def _lifespan(_: FastAPI):
    init_runtime_environment()
    yield


def create_app() -> FastAPI:
    init_runtime_environment()

    application = FastAPI(
        title="OrthoRehab-TCM-Harness API",
        version=APP_VERSION,
        description="中医骨伤与康复确定性护栏副驾 · 红队安全评测沙箱 REST & SSE 后端服务",
        lifespan=_lifespan,
    )

    cors_origins = get_cors_origins()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/v1/health")
    def health_check() -> Dict[str, Any]:
        cfg = LLMConfig.from_env()
        dist_dir = resolve_frontend_dist_dir()
        frontend_ready = (dist_dir / "index.html").is_file()
        return {
            "status": "ok",
            "version": APP_VERSION,
            "mode": get_app_mode(),
            "auth_enabled": bool(get_auth_token()),
            "mock_mode": bool(cfg.mock_mode),
            "default_model": cfg.default_model,
            "frontend_ready": frontend_ready,
        }

    auth_deps = [Depends(verify_bearer_token)]
    application.include_router(copilot.router, prefix="/api/v1", dependencies=auth_deps)
    application.include_router(cases.router, prefix="/api/v1", dependencies=auth_deps)
    application.include_router(benchmark.router, prefix="/api/v1", dependencies=auth_deps)
    application.include_router(kb.router, prefix="/api/v1", dependencies=auth_deps)
    application.include_router(settings.router, prefix="/api/v1", dependencies=auth_deps)

    @application.get("/{full_path:path}")
    def serve_spa_or_static(full_path: str) -> Any:
        cleaned = (full_path or "").lstrip("/")
        if cleaned == "api" or cleaned.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found.")

        dist_dir = resolve_frontend_dist_dir().resolve()
        index_file = dist_dir / "index.html"
        if not index_file.is_file():
            raise HTTPException(
                status_code=404,
                detail="前端尚未构建（缺少 frontend/dist/index.html），请先在 frontend/ 下执行 npm run build。",
            )

        if cleaned:
            candidate = (dist_dir / cleaned).resolve()
            if candidate.is_relative_to(dist_dir) and candidate.is_file():
                return FileResponse(candidate)

        return FileResponse(index_file)

    return application


app = create_app()
