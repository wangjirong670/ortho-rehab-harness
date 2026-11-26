"""Native desktop window (PyWebView) and server launcher for OrthoRehab-TCM-Harness."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import uvicorn

import launcher

APP_WINDOW_TITLE = "OrthoRehab-TCM-Harness · 中医骨伤临床护栏工作台"
APP_WINDOW_WIDTH = 1380
APP_WINDOW_HEIGHT = 880
APP_WINDOW_MIN_SIZE = (1100, 720)


def find_free_loopback_port(preferred_port: int = 0) -> int:
    """Bind 127.0.0.1 to find an available local TCP port."""
    if preferred_port > 0 and launcher.is_port_available(preferred_port, host="127.0.0.1"):
        return preferred_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(url: str, *, timeout_s: float = 15.0) -> bool:
    """Poll /api/v1/health until HTTP 200 {'status': 'ok'} or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    if isinstance(payload, dict) and payload.get("status") == "ok":
                        return True
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(0.15)
    return False


def _configure_desktop_env(
    paths: launcher.RuntimePaths,
    *,
    mock: Optional[bool] = None,
) -> Dict[str, Any]:
    """Configure runtime env, pointing dev-mode ORTHO_RESULTS_DIR to app_benchmark/results."""
    if not paths.frozen and not os.getenv("ORTHO_RESULTS_DIR", "").strip():
        dev_results_dir = paths.bundle_dir / "app_benchmark" / "results"
        if dev_results_dir.is_dir():
            os.environ["ORTHO_RESULTS_DIR"] = str(dev_results_dir)
    return launcher.configure_runtime_env(paths.bundle_dir, paths.app_home, mock=mock)


def run_startup_check() -> Tuple[int, Dict[str, Any]]:
    """Validate dependencies, FastAPI routes, knowledge base catalogs, and configuration."""
    paths = launcher.resolve_runtime_paths()
    env_info = _configure_desktop_env(paths)

    required_modules = (
        "fastapi",
        "uvicorn",
        "httpx",
        "python_multipart",
        "webview",
        "pydantic",
        "yaml",
        "litellm",
        "app_copilot.service",
        "app_benchmark.service",
        "backend.main",
    )

    errors: list[str] = []
    imported: Dict[str, str] = {}
    for mod_name in required_modules:
        try:
            mod = importlib.import_module(mod_name)
            imported[mod_name] = getattr(mod, "__file__", "built-in") or "ok"
        except Exception as exc:
            errors.append(f"import {mod_name}: {type(exc).__name__}: {exc}")

    kb_summary: Dict[str, Any] = {}
    routes_summary: list[str] = []
    frontend_ready = False

    if not errors:
        try:
            from backend.deps import resolve_frontend_dist_dir
            from backend.main import app
            from core_tools.calculator import load_scales_catalog
            from core_tools.rule_scanner import (
                load_drug_conflicts_catalog,
                load_red_flags_catalog,
            )

            rf_cat = load_red_flags_catalog()
            dc_cat = load_drug_conflicts_catalog()
            sc_cat = load_scales_catalog()
            kb_summary = {
                "red_flags": len(rf_cat.rules),
                "drug_conflicts": len(dc_cat.conflicts),
                "scales": len(sc_cat.scales),
            }

            openapi_paths = set((app.openapi() or {}).get("paths", {}).keys())
            direct_paths = {
                getattr(route, "path", "")
                for route in app.routes
                if getattr(route, "path", "")
            }
            routes_summary = sorted(
                p for p in (openapi_paths | direct_paths) if p.startswith("/api/v1")
            )
            required_routes = {
                "/api/v1/health",
                "/api/v1/copilot/demo-cases",
                "/api/v1/copilot/analyze",
                "/api/v1/benchmark/reports",
                "/api/v1/kb/overview",
                "/api/v1/settings/llm",
            }
            missing_routes = sorted(required_routes - set(routes_summary))
            if missing_routes:
                errors.append(f"missing_routes: {missing_routes}")

            dist_dir = resolve_frontend_dist_dir()
            frontend_ready = (dist_dir / "index.html").is_file()
        except Exception as exc:
            errors.append(f"check_failed: {type(exc).__name__}: {exc}")

    payload = {
        "status": "ok" if not errors else "error",
        "version": "0.3.0",
        **env_info,
        "frontend_ready": frontend_ready,
        "knowledge_base": kb_summary,
        "api_route_count": len(routes_summary),
        "imports_count": len(imported),
        "errors": errors,
    }
    return (0 if not errors else 1), payload


class _BackgroundUvicorn:
    """Manage a uvicorn.Server running in a background thread with graceful shutdown."""

    def __init__(self, app: Any, host: str, port: int) -> None:
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            use_colors=False,
        )
        self.server = uvicorn.Server(config)
        self.server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=timeout_s)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OrthoRehab-TCM-Harness 原生桌面与服务端启动器"
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="仅启动 FastAPI/Uvicorn 后端服务（不打开桌面窗口）",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="启动本地服务并在系统默认浏览器中打开",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="绑定监听主机（默认读取 ORTHO_HOST，否则 127.0.0.1）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="绑定监听端口（默认读取 ORTHO_PORT，--server-only 默认 8000，桌面模式自动选空闲端口）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="强制开启离线 Mock 模式 (LLM_MOCK_MODE=true)",
    )
    parser.add_argument(
        "--check",
        "--check-env",
        dest="check",
        action="store_true",
        help="仅校验依赖、路由与知识库配置后退出（兼容 --check-env 别名）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="将 --check 自检结果 JSON 同步写入指定文件路径",
    )
    return parser


def _ensure_standard_streams(app_home: Path) -> Optional[Any]:
    """Redirect None stdout/stderr (PyInstaller console=False) to desktop_app.log."""
    if sys.stdout is not None and sys.stderr is not None:
        return None
    logs_dir = app_home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "desktop_app.log"
    stream = log_path.open("a", encoding="utf-8", buffering=1, errors="replace")
    if sys.stdout is None:
        sys.stdout = stream  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = stream  # type: ignore[assignment]
    return stream


def _write_check_result(
    app_home: Path,
    payload: Dict[str, Any],
    *,
    output_path: Optional[str] = None,
) -> Path:
    """Persist --check/--check-env JSON to app_home/logs/check_result.json (and optional --output)."""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    default_path = app_home / "logs" / "check_result.json"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text(serialized, encoding="utf-8")

    if output_path:
        custom_path = Path(output_path)
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        custom_path.write_text(serialized, encoding="utf-8")
    return default_path


def _show_native_error_dialog(title: str, message: str) -> None:
    """Display a Windows native error dialog without requiring Tkinter."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
    except Exception:
        pass


def _log_desktop_error(
    app_home: Path,
    message: str,
    *,
    exc: Optional[BaseException] = None,
    show_dialog: bool = False,
) -> Path:
    """Write startup error details to app_home/logs/desktop_app.log and stderr."""
    logs_dir = app_home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "desktop_app.log"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[{timestamp}] [ERROR] {message}"]
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
        if tb:
            lines.append(tb)
    with log_path.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write("\n".join(lines) + "\n\n")

    print(f"[ERROR] {message}", file=sys.stderr)
    if show_dialog:
        _show_native_error_dialog(
            "OrthoRehab-TCM-Harness 启动失败",
            f"{message}\n\n详细错误日志已写入：\n{log_path}",
        )
    return log_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    paths = launcher.resolve_runtime_paths()
    _ensure_standard_streams(paths.app_home)
    _configure_desktop_env(
        paths,
        mock=True if args.mock else None,
    )

    if args.check:
        code, payload = run_startup_check()
        _write_check_result(paths.app_home, payload, output_path=args.output)
        if sys.stdout is not None:
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return code

    if args.server_only:
        os.environ["ORTHO_APP_MODE"] = "server"
        host = args.host or os.getenv("ORTHO_HOST", "").strip() or "127.0.0.1"
        env_port = os.getenv("ORTHO_PORT", "").strip()
        port = args.port if args.port is not None else (int(env_port) if env_port else 8000)
        use_colors = False if (paths.frozen or sys.stderr is None or not sys.stderr.isatty()) else None

        from backend.main import app

        uvicorn.run(app, host=host, port=port, log_level="info", use_colors=use_colors)
        return 0

    os.environ["ORTHO_APP_MODE"] = "local_desktop"
    is_native_gui = not args.browser

    env_port = os.getenv("ORTHO_PORT", "").strip()
    preferred_port = args.port if args.port is not None else (int(env_port) if env_port else 0)
    bg_server: Optional[_BackgroundUvicorn] = None

    try:
        port = find_free_loopback_port(preferred_port)
        base_url = f"http://127.0.0.1:{port}"

        from backend.main import app

        bg_server = _BackgroundUvicorn(app, host="127.0.0.1", port=port)
        bg_server.start()

        if not wait_for_health(f"{base_url}/api/v1/health", timeout_s=15.0):
            bg_server.stop()
            _log_desktop_error(
                paths.app_home,
                "本地后端服务启动超时或健康检查失败，请检查端口占用或配置日志。",
                show_dialog=(is_native_gui and paths.frozen),
            )
            return 1

        if args.browser:
            print(f"OrthoRehab-TCM-Harness 已就绪：{base_url}（按 Ctrl+C 停止服务）")
            webbrowser.open(base_url)
            try:
                while bg_server.thread.is_alive():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
            finally:
                bg_server.stop()
            return 0

        import webview

        window = webview.create_window(
            title=APP_WINDOW_TITLE,
            url=base_url,
            width=APP_WINDOW_WIDTH,
            height=APP_WINDOW_HEIGHT,
            min_size=APP_WINDOW_MIN_SIZE,
        )

        def _on_closed() -> None:
            if bg_server is not None:
                bg_server.stop()

        window.events.closed += _on_closed
        webview.start()
        return 0
    except Exception as exc:
        _log_desktop_error(
            paths.app_home,
            f"桌面工作台启动失败：{type(exc).__name__}: {exc}",
            exc=exc,
            show_dialog=(is_native_gui and paths.frozen),
        )
        return 1
    finally:
        if bg_server is not None:
            bg_server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
