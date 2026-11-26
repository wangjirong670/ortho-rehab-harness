"""LLM configuration, .env persistence, and connectivity test endpoints (/api/v1/settings/*)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import launcher
from app_copilot.service import PRESET_MODELS
from core_tools.llm_client import LLMConfig, _resolve_env_file, probe_ollama_models

router = APIRouter(prefix="/settings", tags=["settings"])


class UpdateLLMSettingsRequest(BaseModel):
    copilot_default_model: Optional[str] = None
    benchmark_models: Optional[str] = None
    llm_mock_mode: Optional[bool] = None
    env_updates: Optional[Dict[str, Optional[str]]] = None


class TestConnectionRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: Optional[str] = None


def _mask_secret(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    if len(cleaned) <= 8:
        return "***"
    return f"{cleaned[:4]}...{cleaned[-4:]}"


def _is_masked_placeholder(value: str, current_secret: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True
    if current_secret and stripped == _mask_secret(current_secret):
        return True
    if "..." in stripped or "***" in stripped:
        return True
    return False


def _build_llm_settings_payload() -> Dict[str, Any]:
    cfg = LLMConfig.from_env()
    ollama_models = probe_ollama_models(timeout=0.6)
    preset_models = list(dict.fromkeys([*PRESET_MODELS, *ollama_models]))

    providers: List[Dict[str, Any]] = []
    for preset in launcher.PROVIDER_PRESETS.values():
        raw_key = os.getenv(preset.api_key_env, "").strip() if preset.api_key_env else ""
        raw_base = os.getenv(preset.api_base_env, "").strip() if preset.api_base_env else ""
        providers.append(
            {
                "key": preset.key,
                "label": preset.label,
                "default_model": preset.default_model,
                "model_suggestions": list(preset.model_suggestions),
                "api_key_env": preset.api_key_env,
                "api_base_env": preset.api_base_env,
                "has_key": bool(raw_key),
                "masked_key": _mask_secret(raw_key),
                "api_base": raw_base or preset.default_base,
                "needs_key": preset.needs_key,
            }
        )

    return {
        "copilot_default_model": cfg.default_model,
        "benchmark_models": os.getenv(
            "BENCHMARK_MODELS",
            "deepseek/deepseek-chat,dashscope/qwen-plus,openai/gpt-4o-mini",
        ),
        "llm_mock_mode": bool(cfg.mock_mode),
        "preset_models": preset_models,
        "providers": providers,
        "ollama_models": ollama_models,
        "compliance_notice": launcher.COMPLIANCE_NOTICE,
    }


@router.get("/llm")
def get_llm_settings() -> Dict[str, Any]:
    return _build_llm_settings_payload()


@router.put("/llm")
def update_llm_settings(req: UpdateLLMSettingsRequest) -> Dict[str, Any]:
    updates: Dict[str, str] = {}
    if req.copilot_default_model is not None and req.copilot_default_model.strip():
        updates["COPILOT_DEFAULT_MODEL"] = req.copilot_default_model.strip()
    if req.benchmark_models is not None:
        updates["BENCHMARK_MODELS"] = req.benchmark_models.strip()
    if req.llm_mock_mode is not None:
        updates["LLM_MOCK_MODE"] = "true" if req.llm_mock_mode else "false"

    if req.env_updates:
        for raw_key, raw_val in req.env_updates.items():
            key = str(raw_key or "").strip()
            if not key or raw_val is None:
                continue
            val_str = str(raw_val).strip()
            if key.endswith("_API_KEY"):
                current_val = os.getenv(key, "").strip()
                if _is_masked_placeholder(val_str, current_val):
                    continue
            updates[key] = val_str

    if updates:
        env_path = _resolve_env_file()
        paths = launcher.resolve_runtime_paths()
        template_path: Optional[Path] = None
        for candidate in (paths.app_home / ".env.example", paths.bundle_dir / ".env.example"):
            if candidate.is_file():
                template_path = candidate
                break

        launcher.upsert_env_file(env_path, updates, template_path=template_path)
        for key, val in updates.items():
            if val:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    return _build_llm_settings_payload()


@router.post("/test-connection")
def test_connection(req: TestConnectionRequest) -> Dict[str, Any]:
    provider = (req.provider or "").strip().lower()
    if provider not in launcher.PROVIDER_PRESETS:
        raise HTTPException(status_code=400, detail=f"未知模型服务商：{req.provider}")

    preset = launcher.PROVIDER_PRESETS[provider]
    api_key = (req.api_key or "").strip()
    if not api_key and preset.api_key_env:
        api_key = os.getenv(preset.api_key_env, "").strip()
    elif preset.api_key_env and _is_masked_placeholder(api_key, os.getenv(preset.api_key_env, "")):
        api_key = os.getenv(preset.api_key_env, "").strip()

    api_base = (req.api_base or "").strip()
    if not api_base and preset.api_base_env:
        api_base = os.getenv(preset.api_base_env, "").strip() or preset.default_base

    model = (req.model or "").strip() or preset.default_model

    t0 = time.perf_counter()
    ok, msg = launcher.test_provider_connectivity(
        provider,
        api_key=api_key,
        api_base=api_base,
        model=model,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    models: List[str] = []
    if provider == "ollama" and ok:
        tag_ok, tag_names, _ = launcher.probe_ollama_tags(api_base or preset.default_base, timeout=2.0)
        if tag_ok:
            models = [launcher.normalize_ollama_model_name(name) for name in tag_names]

    return {
        "ok": bool(ok),
        "message": msg,
        "latency_ms": latency_ms,
        "models": models,
    }


@router.get("/ollama-models")
def list_ollama_models() -> Dict[str, Any]:
    api_base = os.getenv("OLLAMA_API_BASE", "").strip() or "http://localhost:11434"
    ok, names, _ = launcher.probe_ollama_tags(api_base, timeout=1.5)
    models = [launcher.normalize_ollama_model_name(name) for name in names] if ok else []
    return {
        "models": models,
        "reachable": bool(ok),
    }
