"""Unified desktop launcher for OrthoRehab-TCM-Harness (GUI / Copilot / Benchmark / Env Check).

Supports:
- Default GUI mode (``--mode gui``): Chinese Tkinter control panel with one-click
  launch/stop for Project B Desktop Copilot (8501) and Project A Red-Team Sandbox (8502),
  graphical LLM/Ollama connection wizard, automatic port fallback, per-service file logging,
  crash log diagnostics, environment self-check button, and results folder shortcut.
- CLI Streamlit modes (``--mode copilot`` / ``--mode benchmark``): programmatically invokes
  ``streamlit.web.cli.main()`` with ``--global.developmentMode=false`` and
  ``--browser.gatherUsageStats=false`` for PyInstaller frozen compatibility.
- Environment self-check (``--check-env``): validates knowledge_base YAMLs and core imports.
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import json
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, TextIO, Tuple

from dotenv import dotenv_values, load_dotenv

APP_TITLE = "OrthoRehab-TCM-Harness 桌面工作台"
APP_SUBTITLE = "中医骨伤与康复确定性护栏副驾 · 红队安全评测沙箱"
DEFAULT_COPILOT_PORT = 8501
DEFAULT_BENCHMARK_PORT = 8502

COMPLIANCE_NOTICE = (
    "⚠️ 医疗安全与合规边界声明：本软件仅用于医学教育、红队安全评测研究与脱敏临床文本辅助整理，"
    "不接入任何医院信息系统（HIS/EMR），不构成医疗器械，严禁用于直接面向患者下达医疗诊断或开具处方。"
    "内置规则均需由执业中医骨伤/康复/药学医师审校后再作辅助参考。"
)

MODE_SCRIPT_MAP: Dict[str, Tuple[str, str, int]] = {
    "copilot": ("app_copilot", "streamlit_copilot.py", DEFAULT_COPILOT_PORT),
    "benchmark": ("app_benchmark", "streamlit_bench.py", DEFAULT_BENCHMARK_PORT),
}

MANAGED_ENV_KEYS: Tuple[str, ...] = (
    "COPILOT_DEFAULT_MODEL",
    "BENCHMARK_MODELS",
    "JUDGE_MODEL",
    "LLM_MOCK_MODE",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_API_BASE",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_API_BASE",
    "ZHIPUAI_API_KEY",
    "ZAI_API_BASE",
    "ZHIPUAI_API_BASE",
    "MOONSHOT_API_KEY",
    "MOONSHOT_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OLLAMA_API_BASE",
    "OLLAMA_CHAT_API_BASE",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_API_BASE",
    "GEMINI_API_KEY",
    "GEMINI_API_BASE",
    "LLM_TIMEOUT_SECONDS",
    "LLM_TIMEOUT",
    "LLM_MAX_RETRIES",
    "LLM_TEMPERATURE",
)


@dataclass(frozen=True)
class ProviderPreset:
    """Configuration preset for the graphical LLM & connection wizard."""

    key: str
    label: str
    default_model: str
    model_suggestions: Tuple[str, ...]
    api_key_env: str
    api_base_env: str
    default_base: str
    default_models_endpoint: str
    needs_key: bool


PROVIDER_PRESETS: Dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        key="deepseek",
        label="DeepSeek (深度求索)",
        default_model="deepseek/deepseek-chat",
        model_suggestions=("deepseek/deepseek-chat", "deepseek/deepseek-reasoner"),
        api_key_env="DEEPSEEK_API_KEY",
        api_base_env="DEEPSEEK_API_BASE",
        default_base="",
        default_models_endpoint="https://api.deepseek.com/models",
        needs_key=True,
    ),
    "dashscope": ProviderPreset(
        key="dashscope",
        label="通义千问 (阿里云百炼)",
        default_model="dashscope/qwen-plus",
        model_suggestions=("dashscope/qwen-plus", "dashscope/qwen-max", "dashscope/qwen-turbo"),
        api_key_env="DASHSCOPE_API_KEY",
        api_base_env="DASHSCOPE_API_BASE",
        default_base="",
        default_models_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/models",
        needs_key=True,
    ),
    "openai": ProviderPreset(
        key="openai",
        label="OpenAI / 兼容中转接口",
        default_model="openai/gpt-4o-mini",
        model_suggestions=("openai/gpt-4o-mini", "openai/gpt-4o"),
        api_key_env="OPENAI_API_KEY",
        api_base_env="OPENAI_API_BASE",
        default_base="",
        default_models_endpoint="https://api.openai.com/v1/models",
        needs_key=True,
    ),
    "ollama": ProviderPreset(
        key="ollama",
        label="本地 Ollama (离线隐私部署)",
        default_model="ollama_chat/qwen2.5:7b",
        model_suggestions=("ollama_chat/qwen2.5:7b", "ollama_chat/qwen2.5:14b"),
        api_key_env="",
        api_base_env="OLLAMA_API_BASE",
        default_base="http://localhost:11434",
        default_models_endpoint="http://localhost:11434/api/tags",
        needs_key=False,
    ),
}


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved bundle (read-only code/assets) and user-writable application home."""

    bundle_dir: Path
    app_home: Path
    frozen: bool


def resolve_runtime_paths(
    *,
    frozen: Optional[bool] = None,
    meipass: Optional[str] = None,
    executable: Optional[str] = None,
    file_path: Optional[Path] = None,
) -> RuntimePaths:
    """Resolve BUNDLE_DIR (sys._MEIPASS when frozen) and APP_HOME (exe directory when frozen)."""
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    anchor_file = (file_path or Path(__file__)).resolve()

    if is_frozen:
        raw_meipass = meipass if meipass is not None else getattr(sys, "_MEIPASS", None)
        raw_exec = executable if executable is not None else sys.executable
        app_home = Path(raw_exec).resolve().parent
        bundle_dir = Path(raw_meipass).resolve() if raw_meipass else app_home
        return RuntimePaths(bundle_dir=bundle_dir, app_home=app_home, frozen=True)

    dev_root = anchor_file.parent
    return RuntimePaths(bundle_dir=dev_root, app_home=dev_root, frozen=False)


def configure_runtime_env(
    bundle_dir: Path,
    app_home: Path,
    *,
    mock: Optional[bool] = None,
) -> Dict[str, Any]:
    """Ensure sys.path, .env loading from APP_HOME, and ORTHO_* env vars are configured."""
    bundle_str = str(bundle_dir)
    if bundle_str not in sys.path:
        sys.path.insert(0, bundle_str)

    os.environ["ORTHO_APP_HOME"] = str(app_home)
    os.environ.setdefault("ORTHO_RESULTS_DIR", str(app_home / "results"))

    env_file = app_home / ".env"
    env_loaded = False
    if env_file.is_file():
        env_loaded = bool(load_dotenv(env_file, override=False))

    if mock is True:
        os.environ["LLM_MOCK_MODE"] = "true"
    elif mock is False:
        os.environ["LLM_MOCK_MODE"] = "false"

    return {
        "bundle_dir": str(bundle_dir),
        "app_home": str(app_home),
        "results_dir": os.environ["ORTHO_RESULTS_DIR"],
        "env_file": str(env_file),
        "env_exists": env_file.is_file(),
        "env_loaded": env_loaded,
    }


def _collect_managed_env_keys(paths: RuntimePaths) -> Set[str]:
    """Collect managed LLM env keys from built-in defaults and .env.example."""
    keys = set(MANAGED_ENV_KEYS)
    for candidate in (paths.app_home / ".env.example", paths.bundle_dir / ".env.example"):
        if candidate.is_file():
            for key in dotenv_values(candidate).keys():
                if key:
                    keys.add(key)
    return keys


def build_child_env(
    paths: RuntimePaths,
    *,
    base_env: Optional[Mapping[str, str]] = None,
    use_mock: bool,
    tracked_keys: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """Build subprocess env with APP_HOME/.env force-applied and deleted/empty keys cleared."""
    env: Dict[str, str] = dict(base_env if base_env is not None else os.environ)
    env["ORTHO_APP_HOME"] = str(paths.app_home)
    env["ORTHO_RESULTS_DIR"] = str(paths.app_home / "results")

    env_file = paths.app_home / ".env"
    managed_keys = _collect_managed_env_keys(paths)
    if tracked_keys:
        managed_keys.update(k for k in tracked_keys if k)

    if env_file.is_file():
        parsed = dotenv_values(env_file)
        for key in managed_keys:
            val = parsed.get(key)
            if val is None or not str(val).strip():
                env.pop(key, None)
            else:
                env[key] = str(val)
        for key, val in parsed.items():
            if not key:
                continue
            if val is None or not str(val).strip():
                env.pop(key, None)
            else:
                env[key] = str(val)

    env["LLM_MOCK_MODE"] = "true" if use_mock else "false"
    return env


def build_streamlit_argv(
    mode: str,
    *,
    bundle_dir: Path,
    port: Optional[int] = None,
    headless: bool = False,
    frozen: Optional[bool] = None,
) -> List[str]:
    """Build deterministic sys.argv for programmatic streamlit.web.cli.main() launch."""
    if mode not in MODE_SCRIPT_MAP:
        raise ValueError(f"Unsupported Streamlit mode '{mode}'. Expected one of {tuple(MODE_SCRIPT_MAP)}")

    pkg_dir, script_name, default_port = MODE_SCRIPT_MAP[mode]
    script_path = bundle_dir / pkg_dir / script_name
    resolved_port = int(port) if port is not None else default_port
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)

    argv = [
        "streamlit",
        "run",
        str(script_path),
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
        f"--server.port={resolved_port}",
        "--server.address=127.0.0.1",
        f"--server.headless={'true' if headless else 'false'}",
    ]
    if is_frozen:
        argv.append("--server.fileWatcherType=none")
    return argv


def build_child_command(
    mode: str,
    *,
    port: int,
    headless: bool = True,
    mock: bool = False,
    frozen: Optional[bool] = None,
    executable: Optional[str] = None,
    script_file: Optional[Path] = None,
) -> List[str]:
    """Build subprocess command for launching copilot/benchmark from the GUI."""
    if mode not in MODE_SCRIPT_MAP:
        raise ValueError(f"Unsupported child mode '{mode}'")

    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    exe_path = executable or sys.executable
    if is_frozen:
        cmd = [exe_path, "--mode", mode, "--port", str(port)]
    else:
        launcher_path = str((script_file or Path(__file__)).resolve())
        cmd = [exe_path, launcher_path, "--mode", mode, "--port", str(port)]

    if headless:
        cmd.append("--headless")
    if mock:
        cmd.append("--mock")
    return cmd


def ensure_env_file(app_home: Path, bundle_dir: Path) -> Path:
    """Create APP_HOME/.env from .env.example if missing, returning the target Path."""
    target = app_home / ".env"
    if target.exists():
        return target

    app_home.mkdir(parents=True, exist_ok=True)
    candidates = [
        app_home / ".env.example",
        bundle_dir / ".env.example",
    ]
    for src in candidates:
        if src.is_file():
            shutil.copyfile(src, target)
            return target

    target.write_text(
        "# OrthoRehab-TCM-Harness 配置文件\n"
        "COPILOT_DEFAULT_MODEL=deepseek/deepseek-chat\n"
        "LLM_MOCK_MODE=true\n"
        "DEEPSEEK_API_KEY=\n",
        encoding="utf-8",
    )
    return target


def upsert_env_file(
    env_path: Path,
    updates: Mapping[str, str],
    *,
    template_path: Optional[Path] = None,
) -> Path:
    """Update or insert key=value entries in .env while preserving existing comments."""
    if not env_path.exists():
        env_path.parent.mkdir(parents=True, exist_ok=True)
        if template_path is not None and template_path.is_file():
            shutil.copyfile(template_path, env_path)
        else:
            env_path.write_text("", encoding="utf-8")

    raw_lines = env_path.read_text(encoding="utf-8").splitlines()
    remaining = {str(k): str(v) for k, v in updates.items() if k}
    new_lines: List[str] = []
    seen_keys: Set[str] = set()

    for line in raw_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key_part = stripped.split("=", 1)[0].strip()
            if key_part in remaining:
                if key_part not in seen_keys:
                    new_lines.append(f"{key_part}={remaining[key_part]}")
                    seen_keys.add(key_part)
                continue
        new_lines.append(line)

    missing_keys = [k for k in remaining if k not in seen_keys]
    if missing_keys:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        for key in missing_keys:
            new_lines.append(f"{key}={remaining[key]}")

    content = "\n".join(new_lines) + ("\n" if new_lines else "")
    env_path.write_text(content, encoding="utf-8")
    return env_path


def normalize_ollama_model_name(model_name: str) -> str:
    """Ensure local Ollama model names carry the ollama_chat/ LiteLLM prefix."""
    cleaned = (model_name or "").strip()
    if not cleaned:
        return "ollama_chat/qwen2.5:7b"
    if cleaned.startswith("ollama_chat/"):
        return cleaned
    if cleaned.startswith("ollama/"):
        return f"ollama_chat/{cleaned[len('ollama/'):]}"
    return f"ollama_chat/{cleaned}"


def build_wizard_env_updates(
    provider: str,
    *,
    model: str,
    api_key: str = "",
    api_base: str = "",
) -> Dict[str, str]:
    """Build .env key-value updates for a selected provider from the model wizard."""
    if provider not in PROVIDER_PRESETS:
        raise ValueError(f"Unsupported provider '{provider}'")

    preset = PROVIDER_PRESETS[provider]
    clean_model = (model or "").strip() or preset.default_model
    clean_base = (api_base or "").strip()
    clean_key = (api_key or "").strip()

    if provider == "ollama":
        return {
            "COPILOT_DEFAULT_MODEL": normalize_ollama_model_name(clean_model),
            "OLLAMA_API_BASE": clean_base or preset.default_base,
            "LLM_TIMEOUT_SECONDS": "180",
            "LLM_MOCK_MODE": "false",
        }

    updates: Dict[str, str] = {
        "COPILOT_DEFAULT_MODEL": clean_model,
        preset.api_key_env: clean_key,
        "LLM_MOCK_MODE": "false",
    }
    if preset.api_base_env:
        updates[preset.api_base_env] = clean_base
    return updates


def probe_ollama_tags(
    api_base: str = "http://localhost:11434",
    *,
    timeout: float = 2.0,
) -> Tuple[bool, List[str], str]:
    """Query local Ollama /api/tags endpoint and return (ok, model_names, message_zh)."""
    base = (api_base or "http://localhost:11434").strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        return False, [], "Ollama 服务地址需以 http:// 或 https:// 开头。"

    url = f"{base}/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
    except urllib.error.URLError as exc:
        return False, [], f"无法连接本地 Ollama（{url}）：{exc.reason}。请确认 Ollama 服务已启动。"
    except TimeoutError:
        return False, [], f"连接 Ollama 超时（{url}），请检查服务是否运行。"
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        return False, [], f"读取 Ollama 模型列表失败：{exc}"

    raw_models = data.get("models") if isinstance(data, dict) else None
    names: List[str] = []
    if isinstance(raw_models, list):
        for item in raw_models:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
    if not names:
        return True, [], "已连接 Ollama 服务，但尚未拉取模型（可执行 `ollama pull qwen2.5:7b`）。"
    return True, names, f"Ollama 连接正常，发现 {len(names)} 个本地模型。"


def test_provider_connectivity(
    provider: str,
    *,
    api_key: str = "",
    api_base: str = "",
    model: str = "",
    timeout: float = 4.0,
) -> Tuple[bool, str]:
    """Validate provider configuration and perform lightweight connectivity check."""
    if provider not in PROVIDER_PRESETS:
        return False, f"未知服务商：{provider}"

    preset = PROVIDER_PRESETS[provider]
    clean_base = (api_base or "").strip()
    clean_key = (api_key or "").strip()

    if clean_base and not clean_base.startswith(("http://", "https://")):
        return False, "API Base URL 格式无效，必须以 http:// 或 https:// 开头。"

    if provider == "ollama":
        ok, names, msg = probe_ollama_tags(clean_base or preset.default_base, timeout=timeout)
        if not ok:
            return False, msg
        target_name = normalize_ollama_model_name(model or preset.default_model).split("/", 1)[-1]
        if names and target_name not in names:
            return True, f"{msg} 注意当前填写的 `{target_name}` 尚未在本地列表内，可点击「刷新」选取已有模型或执行 `ollama pull {target_name}`。"
        return True, msg

    if not clean_key:
        return False, f"请先填写 {preset.api_key_env}。"
    if any(ch.isspace() for ch in clean_key):
        return False, "API Key 中包含空白字符，请检查是否误复制了空格或换行。"

    probe_url = f"{clean_base.rstrip('/')}/models" if clean_base else preset.default_models_endpoint
    req = urllib.request.Request(
        probe_url,
        headers={
            "Authorization": f"Bearer {clean_key}",
            "Accept": "application/json",
            "User-Agent": "OrthoRehab-TCM-Harness/0.2",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return True, f"连通性测试通过（HTTP {resp.status}），API Key 与接口地址有效。"
            return True, f"接口可达（HTTP {resp.status}），配置格式校验通过。"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, f"鉴权失败（HTTP {exc.code}）：API Key 无效或权限不足，请检查后重试。"
        if exc.code in (404, 405):
            return True, f"服务器可达（HTTP {exc.code}，中转端点可能未开放 /models），配置格式校验通过。"
        return False, f"接口返回异常状态（HTTP {exc.code}），请核对 Base URL 与服务状态。"
    except urllib.error.URLError as exc:
        return False, f"无法连接接口地址（{probe_url}）：{exc.reason}。请检查网络、代理或 Base URL。"
    except TimeoutError:
        return False, f"连接接口超时（{probe_url}），请检查网络或代理设置。"
    except OSError as exc:
        return False, f"连接探测异常：{exc}"


def read_initial_mock_preference(app_home: Path, cli_mock: bool = False) -> bool:
    """Determine initial GUI Mock toggle: True on first run (no .env), else follow .env/CLI."""
    if cli_mock:
        return True
    env_file = app_home / ".env"
    if not env_file.is_file():
        return True
    parsed = dotenv_values(env_file)
    raw = str(parsed.get("LLM_MOCK_MODE") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if local TCP port can be bound without active listeners."""
    if port <= 0 or port > 65535:
        return False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        if probe.connect_ex((host, port)) == 0:
            return False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(
    preferred_port: int,
    *,
    exclude_ports: Optional[Iterable[int]] = None,
    host: str = "127.0.0.1",
) -> int:
    """Return preferred_port if free, else fallback to next available port (e.g. 8501->8511)."""
    excluded = set(exclude_ports or ())
    candidates: List[int] = [preferred_port]
    candidates.extend(range(preferred_port + 10, preferred_port + 30))
    candidates.extend(range(preferred_port + 1, preferred_port + 10))

    for candidate in candidates:
        if candidate in excluded or candidate > 65535:
            continue
        if is_port_available(candidate, host=host):
            return candidate

    raise RuntimeError(f"未能找到可用本地端口（起始端口：{preferred_port}）")


def read_log_tail(log_path: Path, max_lines: int = 15) -> str:
    """Read the last N lines from a UTF-8 service log file safely."""
    if not log_path.is_file():
        return "（暂无日志输出）"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"（读取日志失败：{exc}）"
    if not lines:
        return "（日志文件为空）"
    return "\n".join(lines[-max_lines:])


def hide_console_window_if_gui(frozen: bool) -> None:
    """Hide auto-spawned Windows console box ONLY in standalone frozen GUI mode."""
    if sys.platform != "win32" or not frozen:
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        hwnd = kernel32.GetConsoleWindow()
        if not hwnd:
            return
        pid_buf = (ctypes.c_uint32 * 4)()
        attached_count = kernel32.GetConsoleProcessList(pid_buf, 4)
        # Hide when double-clicked (single process attached) or fallback standalone frozen GUI
        if attached_count <= 1:
            user32.ShowWindow(hwnd, 0)
    except Exception:
        pass


def resolve_app_icon_path(paths: RuntimePaths) -> Optional[Path]:
    """Locate packaging/assets/app_icon.ico in bundle or dev workspace."""
    for base in (paths.bundle_dir, paths.app_home):
        candidate = base / "packaging" / "assets" / "app_icon.ico"
        if candidate.is_file():
            return candidate
    return None


def run_env_check(paths: Optional[RuntimePaths] = None) -> Tuple[int, Dict[str, Any]]:
    """Verify bundled YAML knowledge base, public test cases, and critical runtime imports."""
    runtime = paths or resolve_runtime_paths()
    env_info = configure_runtime_env(runtime.bundle_dir, runtime.app_home)

    required_modules = (
        "yaml",
        "pydantic",
        "streamlit",
        "plotly",
        "litellm",
        "core_tools.schemas",
        "core_tools.calculator",
        "core_tools.rule_scanner",
        "core_tools.llm_client",
        "core_tools.harness_core",
        "app_copilot.prompts",
        "app_copilot.service",
        "app_copilot.patient_store",
        "app_copilot.session_store",
        "app_benchmark.evaluator",
        "app_benchmark.runner",
        "backend.main",
    )

    imported: Dict[str, str] = {}
    errors: List[str] = []

    for mod_name in required_modules:
        try:
            mod = importlib.import_module(mod_name)
            imported[mod_name] = getattr(mod, "__file__", "built-in") or "ok"
        except Exception as exc:
            errors.append(f"import {mod_name}: {type(exc).__name__}: {exc}")

    kb_summary: Dict[str, Any] = {}
    if not errors:
        try:
            from app_benchmark.runner import load_benchmark_cases
            from core_tools.calculator import load_scales_catalog
            from core_tools.rule_scanner import (
                load_drug_conflicts_catalog,
                load_red_flags_catalog,
            )

            rf_cat = load_red_flags_catalog()
            dc_cat = load_drug_conflicts_catalog()
            sc_cat = load_scales_catalog()
            public_cases = load_benchmark_cases(include_private=False)

            kb_summary = {
                "red_flags": len(rf_cat.rules),
                "drug_conflicts": len(dc_cat.conflicts),
                "scales": len(sc_cat.scales),
                "public_cases": len(public_cases),
            }
        except Exception as exc:
            errors.append(f"knowledge_base_load: {type(exc).__name__}: {exc}")

    copilot_exists = (runtime.bundle_dir / "app_copilot" / "service.py").is_file()
    bench_exists = (runtime.bundle_dir / "app_benchmark" / "runner.py").is_file()
    if not copilot_exists:
        errors.append("missing app_copilot/service.py")
    if not bench_exists:
        errors.append("missing app_benchmark/runner.py")

    payload: Dict[str, Any] = {
        "status": "ok" if not errors else "error",
        "frozen": runtime.frozen,
        **env_info,
        "scripts": {
            "copilot": copilot_exists,
            "benchmark": bench_exists,
        },
        "knowledge_base": kb_summary,
        "imports_count": len(imported),
        "errors": errors,
    }
    return (0 if not errors else 1), payload


def launch_streamlit_mode(
    mode: str,
    *,
    paths: Optional[RuntimePaths] = None,
    port: Optional[int] = None,
    headless: bool = False,
    mock: bool = False,
) -> int:
    """Configure frozen runtime environment and launch Streamlit in the current process."""
    runtime = paths or resolve_runtime_paths()
    configure_runtime_env(
        runtime.bundle_dir,
        runtime.app_home,
        mock=True if mock else None,
    )
    argv = build_streamlit_argv(
        mode,
        bundle_dir=runtime.bundle_dir,
        port=port,
        headless=headless,
        frozen=runtime.frozen,
    )
    script_path = Path(argv[2])
    if not script_path.is_file():
        print(f"[ERROR] Streamlit target script not found: {script_path}", file=sys.stderr)
        return 1

    sys.argv = argv
    from streamlit.web import cli as st_cli

    try:
        st_cli.main()
        return 0
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 0


def _terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Cleanly terminate a spawned child process and any grandchildren on Windows/POSIX."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=5,
                check=False,
            )
        else:
            proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _probe_streamlit_health(port: int, timeout_s: float = 1.0) -> bool:
    """Return True when Streamlit health endpoint responds 'ok'."""
    url = f"http://127.0.0.1:{port}/_stcore/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore").strip().lower()
            return resp.status == 200 and body == "ok"
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _open_directory_in_os(target_dir: Path) -> None:
    """Open a local folder in OS file explorer."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(target_dir))  # type: ignore[attr-defined]
    else:
        webbrowser.open(target_dir.as_uri())


def run_gui(paths: RuntimePaths, *, cli_mock: bool = False) -> int:
    """Launch the Chinese Tkinter desktop control panel."""
    hide_console_window_if_gui(paths.frozen)

    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk

    configure_runtime_env(paths.bundle_dir, paths.app_home)

    class DesktopControlPanel:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title(APP_TITLE)
            self.root.geometry("720x560")
            self.root.minsize(680, 520)
            self.root.configure(bg="#F8FAFC")

            self._apply_icon(self.root)

            self.processes: Dict[str, subprocess.Popen[Any]] = {}
            self.log_handles: Dict[str, TextIO] = {}
            self.active_ports: Dict[str, int] = {}
            self.user_stopping: Set[str] = set()
            self.tracked_env_keys: Set[str] = set()

            self.mock_var = tk.BooleanVar(
                value=read_initial_mock_preference(paths.app_home, cli_mock=cli_mock)
            )
            self.env_hint_var = tk.StringVar()
            self.status_vars: Dict[str, tk.StringVar] = {
                "copilot": tk.StringVar(value=f"⚪ 未启动（默认端口 {DEFAULT_COPILOT_PORT}）"),
                "benchmark": tk.StringVar(value=f"⚪ 未启动（默认端口 {DEFAULT_BENCHMARK_PORT}）"),
            }
            self.log_var = tk.StringVar(
                value=f"工作目录：{paths.app_home} | 就绪，可一键启动下方工作台。"
            )
            self.buttons: Dict[str, tk.Button] = {}
            self.open_buttons: Dict[str, tk.Button] = {}

            self._build_ui()
            self._refresh_env_hint()
            atexit.register(self.stop_all)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self._schedule_health_poll()

        def _apply_icon(self, window: tk.Misc) -> None:
            icon_path = resolve_app_icon_path(paths)
            if icon_path is not None:
                try:
                    window.iconbitmap(str(icon_path))  # type: ignore[attr-defined]
                except Exception:
                    pass

        def _refresh_env_hint(self) -> None:
            env_file = paths.app_home / ".env"
            if not env_file.is_file():
                self.env_hint_var.set(
                    "提示：当前目录尚无 .env，已默认勾选离线 Mock 模式；点击「模型与连接向导」即可可视化配置真实模型。"
                )
                return

            parsed = dotenv_values(env_file)
            self.tracked_env_keys.update(k for k in parsed.keys() if k)
            model = str(parsed.get("COPILOT_DEFAULT_MODEL") or "未指定").strip()
            mock_str = str(parsed.get("LLM_MOCK_MODE") or "false").strip().lower()
            mode_desc = "Mock 演示" if mock_str in {"1", "true", "yes", "on"} else "真实模型"
            self.env_hint_var.set(
                f"提示：已加载同级 .env（默认模型：{model} | .env 模式：{mode_desc}）；每次启动将自动覆盖生效。"
            )

        def _build_ui(self) -> None:
            style = ttk.Style()
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass

            container = tk.Frame(self.root, bg="#F8FAFC", padx=20, pady=16)
            container.pack(fill="both", expand=True)

            tk.Label(
                container,
                text=APP_TITLE,
                font=("Microsoft YaHei UI", 16, "bold"),
                fg="#0F172A",
                bg="#F8FAFC",
                anchor="w",
            ).pack(fill="x")
            tk.Label(
                container,
                text=APP_SUBTITLE,
                font=("Microsoft YaHei UI", 10),
                fg="#475569",
                bg="#F8FAFC",
                anchor="w",
                pady=2,
            ).pack(fill="x")

            notice_frame = tk.Frame(
                container,
                bg="#FEF2F2",
                highlightbackground="#FCA5A5",
                highlightthickness=1,
                padx=12,
                pady=10,
            )
            notice_frame.pack(fill="x", pady=(10, 12))
            tk.Label(
                notice_frame,
                text=COMPLIANCE_NOTICE,
                font=("Microsoft YaHei UI", 9),
                fg="#991B1B",
                bg="#FEF2F2",
                justify="left",
                wraplength=650,
            ).pack(fill="x")

            opt_frame = tk.LabelFrame(
                container,
                text=" 运行模式与配置 ",
                font=("Microsoft YaHei UI", 9, "bold"),
                fg="#1E293B",
                bg="#FFFFFF",
                padx=14,
                pady=10,
            )
            opt_frame.pack(fill="x", pady=(0, 12))

            tk.Checkbutton(
                opt_frame,
                text="离线 Mock 演示模式 (LLM_MOCK_MODE=true，无需 API Key 即可零配置体验)",
                variable=self.mock_var,
                font=("Microsoft YaHei UI", 9),
                fg="#0F172A",
                bg="#FFFFFF",
                activebackground="#FFFFFF",
                selectcolor="#FFFFFF",
            ).pack(anchor="w")

            tk.Label(
                opt_frame,
                textvariable=self.env_hint_var,
                font=("Microsoft YaHei UI", 8),
                fg="#64748B",
                bg="#FFFFFF",
                anchor="w",
                wraplength=640,
                justify="left",
            ).pack(fill="x", pady=(3, 0))

            svc_frame = tk.LabelFrame(
                container,
                text=" 工作台服务启动区 ",
                font=("Microsoft YaHei UI", 9, "bold"),
                fg="#1E293B",
                bg="#FFFFFF",
                padx=14,
                pady=12,
            )
            svc_frame.pack(fill="x", pady=(0, 12))

            self._add_service_row(
                svc_frame,
                mode="copilot",
                default_port=DEFAULT_COPILOT_PORT,
                btn_text=f"🩺 启动骨伤/康复桌面副驾 ({DEFAULT_COPILOT_PORT})",
            )
            self._add_service_row(
                svc_frame,
                mode="benchmark",
                default_port=DEFAULT_BENCHMARK_PORT,
                btn_text=f"🛡️ 启动红队安全评测沙箱 ({DEFAULT_BENCHMARK_PORT})",
            )

            util_frame = tk.Frame(container, bg="#F8FAFC")
            util_frame.pack(fill="x", pady=(2, 10))

            for text, cmd, bg_color, fg_color in (
                ("⚙️ 模型与连接向导", self._open_model_wizard, "#0F766E", "#FFFFFF"),
                ("📝 用记事本编辑 .env", self._open_env_file, "#E2E8F0", "#0F172A"),
                ("📂 打开评测结果目录", self._open_results_dir, "#E2E8F0", "#0F172A"),
                ("🩺 环境自检", self._run_self_check_modal, "#E2E8F0", "#0F172A"),
            ):
                tk.Button(
                    util_frame,
                    text=text,
                    command=cmd,
                    font=("Microsoft YaHei UI", 9, "bold" if bg_color != "#E2E8F0" else "normal"),
                    bg=bg_color,
                    fg=fg_color,
                    relief="flat",
                    padx=11,
                    pady=7,
                    cursor="hand2",
                ).pack(side="left", padx=(0, 8))

            footer = tk.Frame(container, bg="#E2E8F0", padx=10, pady=7)
            footer.pack(fill="x", side="bottom")
            tk.Label(
                footer,
                textvariable=self.log_var,
                font=("Microsoft YaHei UI", 8),
                fg="#334155",
                bg="#E2E8F0",
                anchor="w",
                wraplength=660,
            ).pack(fill="x")

        def _add_service_row(
            self,
            parent: tk.Widget,
            *,
            mode: str,
            default_port: int,
            btn_text: str,
        ) -> None:
            row_frame = tk.Frame(parent, bg="#FFFFFF")
            row_frame.pack(fill="x", pady=5)

            btn = tk.Button(
                row_frame,
                text=btn_text,
                command=lambda m=mode, p=default_port: self._toggle_service(m, p),
                font=("Microsoft YaHei UI", 10, "bold"),
                bg="#0F766E" if mode == "copilot" else "#1D4ED8",
                fg="#FFFFFF",
                relief="flat",
                width=30,
                pady=7,
                cursor="hand2",
            )
            btn.pack(side="left")
            self.buttons[mode] = btn

            tk.Label(
                row_frame,
                textvariable=self.status_vars[mode],
                font=("Microsoft YaHei UI", 9),
                fg="#334155",
                bg="#FFFFFF",
                padx=12,
                anchor="w",
            ).pack(side="left", fill="x", expand=True)

            open_btn = tk.Button(
                row_frame,
                text="🌐 打开页面",
                command=lambda m=mode, p=default_port: self._open_service_browser(m, p),
                font=("Microsoft YaHei UI", 8),
                bg="#F1F5F9",
                fg="#334155",
                relief="flat",
                padx=10,
                pady=5,
                cursor="hand2",
                state="disabled",
            )
            open_btn.pack(side="right")
            self.open_buttons[mode] = open_btn

        def _open_service_browser(self, mode: str, default_port: int) -> None:
            port = self.active_ports.get(mode, default_port)
            webbrowser.open(f"http://127.0.0.1:{port}")

        def _set_button_running(self, mode: str, port: int, running: bool, *, ready: bool = False) -> None:
            btn = self.buttons[mode]
            open_btn = self.open_buttons[mode]
            default_port = MODE_SCRIPT_MAP[mode][2]
            if running:
                label = (
                    f"⏹️ 停止骨伤/康复桌面副驾 ({port})"
                    if mode == "copilot"
                    else f"⏹️ 停止红队安全评测沙箱 ({port})"
                )
                btn.configure(text=label, bg="#B91C1C")
                open_btn.configure(state="normal" if ready else "disabled")
            else:
                label = (
                    f"🩺 启动骨伤/康复桌面副驾 ({default_port})"
                    if mode == "copilot"
                    else f"🛡️ 启动红队安全评测沙箱 ({default_port})"
                )
                btn.configure(
                    text=label,
                    bg="#0F766E" if mode == "copilot" else "#1D4ED8",
                )
                open_btn.configure(state="disabled")

        def _close_log_handle(self, mode: str) -> None:
            handle = self.log_handles.pop(mode, None)
            if handle is not None:
                try:
                    handle.flush()
                    handle.close()
                except Exception:
                    pass

        def _show_crash_dialog(self, mode: str, port: int, returncode: Optional[int]) -> None:
            log_path = paths.app_home / "logs" / f"{mode}.log"
            tail_text = read_log_tail(log_path, max_lines=15)
            service_name = "桌面副驾" if mode == "copilot" else "红队评测沙箱"

            dlg = tk.Toplevel(self.root)
            dlg.title(f"{service_name} 启动异常")
            dlg.geometry("620x420")
            dlg.configure(bg="#F8FAFC")
            self._apply_icon(dlg)
            dlg.transient(self.root)

            frame = tk.Frame(dlg, bg="#F8FAFC", padx=16, pady=14)
            frame.pack(fill="both", expand=True)

            tk.Label(
                frame,
                text=f"❌ {service_name}（端口 {port}）意外退出（退出码：{returncode}）",
                font=("Microsoft YaHei UI", 11, "bold"),
                fg="#991B1B",
                bg="#F8FAFC",
                anchor="w",
            ).pack(fill="x")

            tips = (
                "常见排查建议：\n"
                "1. 若使用真实大模型，请检查「模型与连接向导」或 .env 中的 API Key / Ollama 服务状态；\n"
                "2. 若提示端口或防火墙拦截，请允许本程序通过 Windows Defender 防火墙本地回环访问；\n"
                "3. 可点击「🩺 环境自检」核对内置知识库与依赖完整性。"
            )
            tk.Label(
                frame,
                text=tips,
                font=("Microsoft YaHei UI", 9),
                fg="#334155",
                bg="#F8FAFC",
                justify="left",
                anchor="w",
            ).pack(fill="x", pady=(6, 8))

            tk.Label(
                frame,
                text=f"日志末尾 15 行（{log_path}）：",
                font=("Microsoft YaHei UI", 8, "bold"),
                fg="#475569",
                bg="#F8FAFC",
                anchor="w",
            ).pack(fill="x")

            box = scrolledtext.ScrolledText(frame, height=11, font=("Consolas", 9), wrap="word")
            box.insert("1.0", tail_text)
            box.configure(state="disabled")
            box.pack(fill="both", expand=True, pady=(4, 10))

            btn_bar = tk.Frame(frame, bg="#F8FAFC")
            btn_bar.pack(fill="x")
            tk.Button(
                btn_bar,
                text="📂 打开日志目录",
                command=lambda: _open_directory_in_os(paths.app_home / "logs"),
                font=("Microsoft YaHei UI", 9),
                bg="#E2E8F0",
                fg="#0F172A",
                relief="flat",
                padx=12,
                pady=6,
                cursor="hand2",
            ).pack(side="left")
            tk.Button(
                btn_bar,
                text="关闭",
                command=dlg.destroy,
                font=("Microsoft YaHei UI", 9),
                bg="#0F766E",
                fg="#FFFFFF",
                relief="flat",
                padx=14,
                pady=6,
                cursor="hand2",
            ).pack(side="right")

        def _toggle_service(self, mode: str, default_port: int) -> None:
            proc = self.processes.get(mode)
            if proc is not None and proc.poll() is None:
                self.user_stopping.add(mode)
                bound_port = self.active_ports.get(mode, default_port)
                _terminate_process_tree(proc)
                self.processes.pop(mode, None)
                self.active_ports.pop(mode, None)
                self._close_log_handle(mode)
                self._set_button_running(mode, default_port, False)
                self.status_vars[mode].set(f"⚪ 已停止（默认端口 {default_port}）")
                self.log_var.set(f"已停止 {mode} 服务（原端口 {bound_port}）。")
                return

            self.user_stopping.discard(mode)
            self._refresh_env_hint()

            other_ports = [p for m, p in self.active_ports.items() if m != mode]
            try:
                actual_port = find_available_port(default_port, exclude_ports=other_ports)
            except Exception as exc:
                messagebox.showerror("端口分配失败", str(exc))
                return

            use_mock = bool(self.mock_var.get())
            cmd = build_child_command(
                mode,
                port=actual_port,
                headless=True,
                mock=use_mock,
                frozen=paths.frozen,
            )
            env = build_child_env(
                paths,
                use_mock=use_mock,
                tracked_keys=self.tracked_env_keys,
            )
            if (paths.app_home / ".env").is_file():
                self.tracked_env_keys.update(k for k in dotenv_values(paths.app_home / ".env").keys() if k)

            logs_dir = paths.app_home / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"{mode}.log"
            self._close_log_handle(mode)
            try:
                log_fh = log_path.open("w", encoding="utf-8", errors="replace")
            except OSError as exc:
                messagebox.showerror("日志创建失败", f"无法写入日志文件 {log_path}：{exc}")
                return

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(paths.app_home),
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            except Exception as exc:
                log_fh.close()
                messagebox.showerror("启动失败", f"无法启动 {mode} 服务：{exc}")
                return

            self.log_handles[mode] = log_fh
            self.processes[mode] = proc
            self.active_ports[mode] = actual_port
            self._set_button_running(mode, actual_port, True, ready=False)
            self.status_vars[mode].set(f"🟡 正在启动...（端口 {actual_port}）")
            if actual_port != default_port:
                self.log_var.set(
                    f"端口 {default_port} 已被占用，已自动顺延至 {actual_port}；正在拉起 {mode} 服务（Mock={use_mock}）..."
                )
            else:
                self.log_var.set(f"正在拉起 {mode} 服务（端口 {actual_port}，Mock={use_mock}）...")

            threading.Thread(
                target=self._wait_and_open_browser,
                args=(mode, actual_port, proc),
                daemon=True,
            ).start()

        def _wait_and_open_browser(
            self,
            mode: str,
            port: int,
            proc: subprocess.Popen[Any],
        ) -> None:
            deadline = time.monotonic() + 30.0
            ready = False
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                if _probe_streamlit_health(port, timeout_s=0.8):
                    ready = True
                    break
                time.sleep(0.4)

            def _on_ui() -> None:
                if self.processes.get(mode) is not proc:
                    return
                if proc.poll() is not None:
                    code = proc.returncode
                    self.processes.pop(mode, None)
                    self.active_ports.pop(mode, None)
                    self._close_log_handle(mode)
                    self._set_button_running(mode, port, False)
                    if mode in self.user_stopping:
                        self.user_stopping.discard(mode)
                        return
                    self.status_vars[mode].set(f"🔴 启动异常退出（端口 {port}）")
                    self.log_var.set(f"{mode} 子进程异常退出（退出码 {code}），请查看弹窗日志。")
                    self._show_crash_dialog(mode, port, code)
                    return

                if ready:
                    url = f"http://127.0.0.1:{port}"
                    self._set_button_running(mode, port, True, ready=True)
                    self.status_vars[mode].set(f"🟢 运行中：{url}")
                    self.log_var.set(f"{mode} 已就绪，正在打开默认浏览器：{url}")
                    webbrowser.open(url)
                else:
                    self._set_button_running(mode, port, True, ready=True)
                    self.status_vars[mode].set(f"🟡 启动较慢，可点「打开页面」重试（端口 {port}）")

            try:
                self.root.after(0, _on_ui)
            except tk.TclError:
                pass

        def _schedule_health_poll(self) -> None:
            for mode in ("copilot", "benchmark"):
                proc = self.processes.get(mode)
                if proc is not None and proc.poll() is not None:
                    code = proc.returncode
                    port = self.active_ports.pop(mode, MODE_SCRIPT_MAP[mode][2])
                    self.processes.pop(mode, None)
                    self._close_log_handle(mode)
                    self._set_button_running(mode, port, False)
                    if mode in self.user_stopping:
                        self.user_stopping.discard(mode)
                        self.status_vars[mode].set(f"⚪ 已停止（默认端口 {MODE_SCRIPT_MAP[mode][2]}）")
                    else:
                        self.status_vars[mode].set(f"🔴 已意外退出（退出码 {code}）")
                        self.log_var.set(f"{mode} 运行中断（退出码 {code}）。")
                        self._show_crash_dialog(mode, port, code)
            self.root.after(1500, self._schedule_health_poll)

        def _open_model_wizard(self) -> None:
            dlg = tk.Toplevel(self.root)
            dlg.title("模型与连接向导 (.env 配置)")
            dlg.geometry("560x430")
            dlg.configure(bg="#F8FAFC")
            self._apply_icon(dlg)
            dlg.transient(self.root)
            dlg.grab_set()

            env_file = paths.app_home / ".env"
            existing = dotenv_values(env_file) if env_file.is_file() else {}
            existing_model = str(existing.get("COPILOT_DEFAULT_MODEL") or "").strip()

            initial_provider = "deepseek"
            if existing_model.startswith(("ollama_chat/", "ollama/")):
                initial_provider = "ollama"
            elif existing_model.startswith("dashscope/"):
                initial_provider = "dashscope"
            elif existing_model.startswith("openai/"):
                initial_provider = "openai"

            provider_var = tk.StringVar(value=initial_provider)
            model_var = tk.StringVar(
                value=existing_model or PROVIDER_PRESETS[initial_provider].default_model
            )
            key_var = tk.StringVar()
            base_var = tk.StringVar()
            show_key_var = tk.BooleanVar(value=False)
            test_status_var = tk.StringVar(value="填写后可点击「测试连通性」，再保存写入同级 .env。")

            body = tk.Frame(dlg, bg="#F8FAFC", padx=18, pady=14)
            body.pack(fill="both", expand=True)

            tk.Label(
                body,
                text="⚙️ 大模型与本地 Ollama 连接向导",
                font=("Microsoft YaHei UI", 12, "bold"),
                fg="#0F172A",
                bg="#F8FAFC",
                anchor="w",
            ).pack(fill="x", pady=(0, 8))

            # Provider selector
            prov_row = tk.Frame(body, bg="#F8FAFC")
            prov_row.pack(fill="x", pady=4)
            tk.Label(prov_row, text="1. 选择模型服务商：", width=16, anchor="w", bg="#F8FAFC", font=("Microsoft YaHei UI", 9, "bold")).pack(side="left")

            label_to_key = {p.label: k for k, p in PROVIDER_PRESETS.items()}
            key_to_label = {k: p.label for k, p in PROVIDER_PRESETS.items()}
            prov_label_var = tk.StringVar(value=key_to_label[initial_provider])
            prov_combo = ttk.Combobox(
                prov_row,
                textvariable=prov_label_var,
                values=list(label_to_key.keys()),
                state="readonly",
            )
            prov_combo.pack(side="left", fill="x", expand=True)

            # Model selector
            mod_row = tk.Frame(body, bg="#F8FAFC")
            mod_row.pack(fill="x", pady=6)
            tk.Label(mod_row, text="2. 模型名称：", width=16, anchor="w", bg="#F8FAFC", font=("Microsoft YaHei UI", 9, "bold")).pack(side="left")
            model_combo = ttk.Combobox(mod_row, textvariable=model_var)
            model_combo.pack(side="left", fill="x", expand=True)
            refresh_btn = tk.Button(
                mod_row,
                text="🔄 刷新 Ollama",
                font=("Microsoft YaHei UI", 8),
                bg="#E2E8F0",
                fg="#0F172A",
                relief="flat",
                padx=8,
                pady=3,
                cursor="hand2",
            )
            refresh_btn.pack(side="left", padx=(6, 0))

            # API Key row
            key_row = tk.Frame(body, bg="#F8FAFC")
            key_row.pack(fill="x", pady=6)
            tk.Label(key_row, text="3. API Key：", width=16, anchor="w", bg="#F8FAFC", font=("Microsoft YaHei UI", 9, "bold")).pack(side="left")
            key_entry = tk.Entry(key_row, textvariable=key_var, show="*", font=("Consolas", 9))
            key_entry.pack(side="left", fill="x", expand=True)
            key_toggle = tk.Checkbutton(
                key_row,
                text="显示",
                variable=show_key_var,
                bg="#F8FAFC",
                command=lambda: key_entry.configure(show="" if show_key_var.get() else "*"),
            )
            key_toggle.pack(side="left", padx=(6, 0))

            # Base URL row
            base_row = tk.Frame(body, bg="#F8FAFC")
            base_row.pack(fill="x", pady=6)
            tk.Label(base_row, text="4. 接口 Base URL：", width=16, anchor="w", bg="#F8FAFC", font=("Microsoft YaHei UI", 9, "bold")).pack(side="left")
            base_entry = tk.Entry(base_row, textvariable=base_var, font=("Consolas", 9))
            base_entry.pack(side="left", fill="x", expand=True)

            note_lbl = tk.Label(
                body,
                text="",
                font=("Microsoft YaHei UI", 8),
                fg="#475569",
                bg="#F8FAFC",
                anchor="w",
                justify="left",
                wraplength=510,
            )
            note_lbl.pack(fill="x", pady=(4, 8))

            status_box = tk.Label(
                body,
                textvariable=test_status_var,
                font=("Microsoft YaHei UI", 9),
                fg="#1E293B",
                bg="#E2E8F0",
                padx=10,
                pady=8,
                anchor="w",
                justify="left",
                wraplength=500,
            )
            status_box.pack(fill="x", pady=(4, 10))

            def _sync_provider_fields(*_: Any) -> None:
                pkey = label_to_key.get(prov_label_var.get(), "deepseek")
                provider_var.set(pkey)
                preset = PROVIDER_PRESETS[pkey]
                model_combo.configure(values=list(preset.model_suggestions))
                if not model_var.get().strip() or not model_var.get().startswith(pkey.split("_")[0]):
                    if existing_model and (
                        (pkey == "ollama" and existing_model.startswith(("ollama_chat/", "ollama/")))
                        or existing_model.startswith(f"{pkey}/")
                    ):
                        model_var.set(existing_model)
                    else:
                        model_var.set(preset.default_model)

                if preset.needs_key:
                    key_entry.configure(state="normal")
                    key_toggle.configure(state="normal")
                    key_var.set(str(existing.get(preset.api_key_env) or ""))
                    base_var.set(str(existing.get(preset.api_base_env) or preset.default_base))
                    refresh_btn.configure(state="disabled")
                    note_lbl.configure(
                        text=f"官方直连可留空 Base URL；保存时将写入 {preset.api_key_env} 并自动关闭 LLM_MOCK_MODE。"
                    )
                else:
                    key_var.set("")
                    key_entry.configure(state="disabled")
                    key_toggle.configure(state="disabled")
                    base_var.set(str(existing.get("OLLAMA_API_BASE") or preset.default_base))
                    refresh_btn.configure(state="normal")
                    note_lbl.configure(
                        text="本地 Ollama 无需 API Key；保存时自动补齐 ollama_chat/ 前缀并设置 LLM_TIMEOUT_SECONDS=180。"
                    )

            def _refresh_ollama_models() -> None:
                test_status_var.set("正在探测本地 Ollama 已安装模型...")
                base_url = base_var.get().strip() or "http://localhost:11434"

                def _worker() -> None:
                    ok, names, msg = probe_ollama_tags(base_url)

                    def _done() -> None:
                        if ok and names:
                            formatted = [normalize_ollama_model_name(n) for n in names]
                            model_combo.configure(values=formatted)
                            if model_var.get().strip() not in formatted:
                                model_var.set(formatted[0])
                        test_status_var.set(msg)

                    try:
                        dlg.after(0, _done)
                    except tk.TclError:
                        pass

                threading.Thread(target=_worker, daemon=True).start()

            refresh_btn.configure(command=_refresh_ollama_models)

            def _on_test_conn() -> None:
                pkey = provider_var.get()
                test_status_var.set("正在测试连通性，请稍候...")

                def _worker() -> None:
                    ok, msg = test_provider_connectivity(
                        pkey,
                        api_key=key_var.get(),
                        api_base=base_var.get(),
                        model=model_var.get(),
                    )

                    def _done() -> None:
                        prefix = "✅ " if ok else "❌ "
                        test_status_var.set(prefix + msg)

                    try:
                        dlg.after(0, _done)
                    except tk.TclError:
                        pass

                threading.Thread(target=_worker, daemon=True).start()

            def _on_save() -> None:
                pkey = provider_var.get()
                preset = PROVIDER_PRESETS[pkey]
                if preset.needs_key and not key_var.get().strip():
                    messagebox.showwarning("缺少 API Key", f"请先填写 {preset.api_key_env}，或改选本地 Ollama。", parent=dlg)
                    return

                updates = build_wizard_env_updates(
                    pkey,
                    model=model_var.get(),
                    api_key=key_var.get(),
                    api_base=base_var.get(),
                )
                template = (
                    paths.app_home / ".env.example"
                    if (paths.app_home / ".env.example").is_file()
                    else paths.bundle_dir / ".env.example"
                )
                try:
                    saved_path = upsert_env_file(paths.app_home / ".env", updates, template_path=template)
                except OSError as exc:
                    messagebox.showerror("保存失败", f"无法写入 {paths.app_home / '.env'}：{exc}", parent=dlg)
                    return

                self.tracked_env_keys.update(updates.keys())
                self.mock_var.set(False)
                self._refresh_env_hint()
                self.log_var.set(f"已保存配置到 {saved_path}（默认模型：{updates['COPILOT_DEFAULT_MODEL']}），Mock 已自动取消勾选。")
                messagebox.showinfo("配置已保存", f"已写入 {saved_path}！\n重启对应服务即可按新配置运行。", parent=dlg)
                dlg.destroy()

            prov_combo.bind("<<ComboboxSelected>>", _sync_provider_fields)
            _sync_provider_fields()
            if initial_provider == "ollama":
                _refresh_ollama_models()

            btn_row = tk.Frame(body, bg="#F8FAFC")
            btn_row.pack(fill="x", side="bottom")
            tk.Button(
                btn_row,
                text="🔌 测试连通性",
                command=_on_test_conn,
                font=("Microsoft YaHei UI", 9),
                bg="#E2E8F0",
                fg="#0F172A",
                relief="flat",
                padx=12,
                pady=7,
                cursor="hand2",
            ).pack(side="left")

            tk.Button(
                btn_row,
                text="取消",
                command=dlg.destroy,
                font=("Microsoft YaHei UI", 9),
                bg="#E2E8F0",
                fg="#0F172A",
                relief="flat",
                padx=12,
                pady=7,
                cursor="hand2",
            ).pack(side="right", padx=(8, 0))

            tk.Button(
                btn_row,
                text="💾 保存配置到 .env",
                command=_on_save,
                font=("Microsoft YaHei UI", 9, "bold"),
                bg="#0F766E",
                fg="#FFFFFF",
                relief="flat",
                padx=14,
                pady=7,
                cursor="hand2",
            ).pack(side="right")

        def _run_self_check_modal(self) -> None:
            self.log_var.set("正在执行冻结环境自检（加载知识库 YAML 与核心模块）...")

            def _worker() -> None:
                code, payload = run_env_check(paths)

                def _show() -> None:
                    kb = payload.get("knowledge_base") or {}
                    summary = (
                        f"状态：{'✅ 通过 (exit 0)' if code == 0 else '❌ 异常 (exit 1)'}\n"
                        f"运行态：{'PyInstaller 冻结版' if payload.get('frozen') else '源码开发态'}\n"
                        f"工作目录：{payload.get('app_home')}\n"
                        f"结果目录：{payload.get('results_dir')}\n"
                        f"红旗规则：{kb.get('red_flags', '-')} 条 | 用药冲突：{kb.get('drug_conflicts', '-')} 组 | "
                        f"量表：{kb.get('scales', '-')} 套 | 公开用例：{kb.get('public_cases', '-')} 题\n"
                        f"已验证模块数：{payload.get('imports_count', 0)}"
                    )
                    if payload.get("errors"):
                        summary += "\n\n错误信息：\n" + "\n".join(payload["errors"])
                    self.log_var.set("环境自检完成：" + ("全部通过" if code == 0 else "发现异常"))
                    if code == 0:
                        messagebox.showinfo("环境自检结果", summary, parent=self.root)
                    else:
                        messagebox.showerror("环境自检失败", summary, parent=self.root)

                try:
                    self.root.after(0, _show)
                except tk.TclError:
                    pass

            threading.Thread(target=_worker, daemon=True).start()

        def _open_env_file(self) -> None:
            try:
                env_path = ensure_env_file(paths.app_home, paths.bundle_dir)
                self._refresh_env_hint()
                self.log_var.set(f"已打开配置文件：{env_path}（保存后下次启动服务立即生效）")
                if sys.platform == "win32":
                    subprocess.Popen(["notepad.exe", str(env_path)])
                else:
                    webbrowser.open(env_path.as_uri())
            except Exception as exc:
                messagebox.showerror("打开失败", f"无法创建或打开 .env 文件：{exc}")

        def _open_results_dir(self) -> None:
            try:
                results_dir = Path(os.getenv("ORTHO_RESULTS_DIR", str(paths.app_home / "results")))
                _open_directory_in_os(results_dir)
                self.log_var.set(f"已打开评测结果目录：{results_dir}")
            except Exception as exc:
                messagebox.showerror("打开失败", f"无法打开结果目录：{exc}")

        def stop_all(self) -> None:
            for mode, proc in list(self.processes.items()):
                self.user_stopping.add(mode)
                _terminate_process_tree(proc)
                self._close_log_handle(mode)
            self.processes.clear()
            self.active_ports.clear()

        def _on_close(self) -> None:
            self.stop_all()
            self.root.destroy()

    root = tk.Tk()
    DesktopControlPanel(root)
    root.mainloop()
    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for launcher.py."""
    parser = argparse.ArgumentParser(
        description="OrthoRehab-TCM-Harness 统一桌面启动器"
    )
    parser.add_argument(
        "--mode",
        choices=["gui", "copilot", "benchmark"],
        default="gui",
        help="启动模式：gui（默认桌面控制面板）/ copilot（副驾）/ benchmark（红队沙箱）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Streamlit 服务端口（copilot 默认 8501，benchmark 默认 8502）",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="以无头模式启动 Streamlit（不自动弹出浏览器）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="强制开启离线 Mock 演示模式（LLM_MOCK_MODE=true）",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="执行冻结环境快速自检（加载知识库 YAML 与核心模块并输出 JSON）",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    paths = resolve_runtime_paths()

    if args.check_env:
        code, payload = run_env_check(paths)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return code

    if args.mode in ("copilot", "benchmark"):
        return launch_streamlit_mode(
            args.mode,
            paths=paths,
            port=args.port,
            headless=args.headless,
            mock=args.mock,
        )

    return run_gui(paths, cli_mock=args.mock)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
