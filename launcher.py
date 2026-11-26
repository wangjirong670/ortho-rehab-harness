"""Unified desktop launcher for OrthoRehab-TCM-Harness (GUI / Copilot / Benchmark / Env Check).

Supports:
- Default GUI mode (``--mode gui``): Chinese Tkinter control panel with one-click
  launch/stop for Project B Desktop Copilot (8501) and Project A Red-Team Sandbox (8502),
  offline Mock toggle, .env initialization, and results folder shortcut.
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
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        "app_copilot.prompts",
        "app_copilot.streamlit_copilot",
        "app_benchmark.evaluator",
        "app_benchmark.runner",
        "app_benchmark.streamlit_bench",
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

    copilot_exists = (runtime.bundle_dir / "app_copilot" / "streamlit_copilot.py").is_file()
    bench_exists = (runtime.bundle_dir / "app_benchmark" / "streamlit_bench.py").is_file()
    if not copilot_exists:
        errors.append("missing app_copilot/streamlit_copilot.py")
    if not bench_exists:
        errors.append("missing app_benchmark/streamlit_bench.py")

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


def run_gui(paths: RuntimePaths, *, cli_mock: bool = False) -> int:
    """Launch the Chinese Tkinter desktop control panel."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    configure_runtime_env(paths.bundle_dir, paths.app_home)

    class DesktopControlPanel:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title(APP_TITLE)
            self.root.geometry("680x510")
            self.root.minsize(640, 480)
            self.root.configure(bg="#F8FAFC")

            self.processes: Dict[str, subprocess.Popen[Any]] = {}
            self.mock_var = tk.BooleanVar(
                value=read_initial_mock_preference(paths.app_home, cli_mock=cli_mock)
            )
            self.status_vars: Dict[str, tk.StringVar] = {
                "copilot": tk.StringVar(value="⚪ 未启动（端口 8501）"),
                "benchmark": tk.StringVar(value="⚪ 未启动（端口 8502）"),
            }
            self.log_var = tk.StringVar(
                value=f"工作目录：{paths.app_home} | 就绪，可一键启动下方工作台。"
            )
            self.buttons: Dict[str, ttk.Button] = {}

            self._build_ui()
            atexit.register(self.stop_all)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self._schedule_health_poll()

        def _build_ui(self) -> None:
            style = ttk.Style()
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass

            container = tk.Frame(self.root, bg="#F8FAFC", padx=20, pady=16)
            container.pack(fill="both", expand=True)

            # Header
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

            # Compliance notice box
            notice_frame = tk.Frame(
                container,
                bg="#FEF2F2",
                highlightbackground="#FCA5A5",
                highlightthickness=1,
                padx=12,
                pady=10,
            )
            notice_frame.pack(fill="x", pady=(10, 14))
            tk.Label(
                notice_frame,
                text=COMPLIANCE_NOTICE,
                font=("Microsoft YaHei UI", 9),
                fg="#991B1B",
                bg="#FEF2F2",
                justify="left",
                wraplength=610,
            ).pack(fill="x")

            # Mode options card
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

            env_hint = (
                "提示：当前工作目录尚无 .env，已默认勾选离线 Mock 模式。"
                if not (paths.app_home / ".env").is_file()
                else "提示：已检测到同级 .env 配置；取消勾选后将按 .env 中的真实模型配置运行。"
            )
            tk.Label(
                opt_frame,
                text=env_hint,
                font=("Microsoft YaHei UI", 8),
                fg="#64748B",
                bg="#FFFFFF",
                anchor="w",
            ).pack(fill="x", pady=(2, 0))

            # Services card
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
                port=DEFAULT_COPILOT_PORT,
                btn_text="🩺 启动骨伤/康复桌面副驾 (8501)",
                row=0,
            )
            self._add_service_row(
                svc_frame,
                mode="benchmark",
                port=DEFAULT_BENCHMARK_PORT,
                btn_text="🛡️ 启动红队安全评测沙箱 (8502)",
                row=1,
            )

            # Utility buttons
            util_frame = tk.Frame(container, bg="#F8FAFC")
            util_frame.pack(fill="x", pady=(2, 10))

            tk.Button(
                util_frame,
                text="⚙️ 打开 / 初始化 .env 配置文件",
                command=self._open_env_file,
                font=("Microsoft YaHei UI", 9),
                bg="#E2E8F0",
                fg="#0F172A",
                relief="flat",
                padx=12,
                pady=7,
                cursor="hand2",
            ).pack(side="left", padx=(0, 10))

            tk.Button(
                util_frame,
                text="📂 打开评测结果目录 (results/)",
                command=self._open_results_dir,
                font=("Microsoft YaHei UI", 9),
                bg="#E2E8F0",
                fg="#0F172A",
                relief="flat",
                padx=12,
                pady=7,
                cursor="hand2",
            ).pack(side="left")

            # Status footer
            footer = tk.Frame(
                container,
                bg="#E2E8F0",
                padx=10,
                pady=6,
            )
            footer.pack(fill="x", side="bottom")
            tk.Label(
                footer,
                textvariable=self.log_var,
                font=("Microsoft YaHei UI", 8),
                fg="#334155",
                bg="#E2E8F0",
                anchor="w",
                wraplength=620,
            ).pack(fill="x")

        def _add_service_row(
            self,
            parent: tk.Widget,
            *,
            mode: str,
            port: int,
            btn_text: str,
            row: int,
        ) -> None:
            row_frame = tk.Frame(parent, bg="#FFFFFF")
            row_frame.pack(fill="x", pady=5)

            btn = tk.Button(
                row_frame,
                text=btn_text,
                command=lambda m=mode, p=port: self._toggle_service(m, p),
                font=("Microsoft YaHei UI", 10, "bold"),
                bg="#0F766E" if mode == "copilot" else "#1D4ED8",
                fg="#FFFFFF",
                relief="flat",
                width=30,
                pady=7,
                cursor="hand2",
            )
            btn.pack(side="left")
            self.buttons[mode] = btn  # type: ignore[assignment]

            tk.Label(
                row_frame,
                textvariable=self.status_vars[mode],
                font=("Microsoft YaHei UI", 9),
                fg="#334155",
                bg="#FFFFFF",
                padx=12,
                anchor="w",
            ).pack(side="left", fill="x", expand=True)

            tk.Button(
                row_frame,
                text="🌐 打开页面",
                command=lambda p=port: webbrowser.open(f"http://127.0.0.1:{p}"),
                font=("Microsoft YaHei UI", 8),
                bg="#F1F5F9",
                fg="#334155",
                relief="flat",
                padx=8,
                pady=5,
                cursor="hand2",
            ).pack(side="right")

        def _set_button_running(self, mode: str, port: int, running: bool) -> None:
            btn = self.buttons[mode]
            if running:
                label = (
                    f"⏹️ 停止骨伤/康复桌面副驾 ({port})"
                    if mode == "copilot"
                    else f"⏹️ 停止红队安全评测沙箱 ({port})"
                )
                btn.configure(text=label, bg="#B91C1C")
            else:
                label = (
                    f"🩺 启动骨伤/康复桌面副驾 ({port})"
                    if mode == "copilot"
                    else f"🛡️ 启动红队安全评测沙箱 ({port})"
                )
                btn.configure(
                    text=label,
                    bg="#0F766E" if mode == "copilot" else "#1D4ED8",
                )

        def _toggle_service(self, mode: str, port: int) -> None:
            proc = self.processes.get(mode)
            if proc is not None and proc.poll() is None:
                _terminate_process_tree(proc)
                self.processes.pop(mode, None)
                self._set_button_running(mode, port, False)
                self.status_vars[mode].set(f"⚪ 已停止（端口 {port}）")
                self.log_var.set(f"已停止 {mode} 服务（端口 {port}）。")
                return

            use_mock = bool(self.mock_var.get())
            cmd = build_child_command(
                mode,
                port=port,
                headless=True,
                mock=use_mock,
                frozen=paths.frozen,
            )
            env = os.environ.copy()
            env["ORTHO_APP_HOME"] = str(paths.app_home)
            env["ORTHO_RESULTS_DIR"] = str(paths.app_home / "results")
            env["LLM_MOCK_MODE"] = "true" if use_mock else "false"

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(paths.app_home),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
            except Exception as exc:
                messagebox.showerror("启动失败", f"无法启动 {mode} 服务：{exc}")
                return

            self.processes[mode] = proc
            self._set_button_running(mode, port, True)
            self.status_vars[mode].set(f"🟡 正在启动...（端口 {port}）")
            self.log_var.set(f"正在拉起 {mode} 服务（端口 {port}，Mock={use_mock}）...")

            threading.Thread(
                target=self._wait_and_open_browser,
                args=(mode, port, proc),
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
                if proc.poll() is not None:
                    self.processes.pop(mode, None)
                    self._set_button_running(mode, port, False)
                    self.status_vars[mode].set(f"🔴 启动异常退出（端口 {port}）")
                    self.log_var.set(f"{mode} 子进程已退出（退出码 {proc.returncode}）。")
                    return
                if ready:
                    url = f"http://127.0.0.1:{port}"
                    self.status_vars[mode].set(f"🟢 运行中：{url}")
                    self.log_var.set(f"{mode} 已就绪，正在打开默认浏览器：{url}")
                    webbrowser.open(url)
                else:
                    self.status_vars[mode].set(f"🟡 启动耗时较长，请稍后点击「打开页面」")

            try:
                self.root.after(0, _on_ui)
            except tk.TclError:
                pass

        def _schedule_health_poll(self) -> None:
            for mode, port in (
                ("copilot", DEFAULT_COPILOT_PORT),
                ("benchmark", DEFAULT_BENCHMARK_PORT),
            ):
                proc = self.processes.get(mode)
                if proc is not None and proc.poll() is not None:
                    self.processes.pop(mode, None)
                    self._set_button_running(mode, port, False)
                    self.status_vars[mode].set(f"⚪ 已停止（端口 {port}）")
            self.root.after(1500, self._schedule_health_poll)

        def _open_env_file(self) -> None:
            try:
                env_path = ensure_env_file(paths.app_home, paths.bundle_dir)
                self.log_var.set(f"已打开配置文件：{env_path}")
                if sys.platform == "win32":
                    subprocess.Popen(["notepad.exe", str(env_path)])
                else:
                    webbrowser.open(env_path.as_uri())
            except Exception as exc:
                messagebox.showerror("打开失败", f"无法创建或打开 .env 文件：{exc}")

        def _open_results_dir(self) -> None:
            try:
                results_dir = Path(os.getenv("ORTHO_RESULTS_DIR", str(paths.app_home / "results")))
                results_dir.mkdir(parents=True, exist_ok=True)
                self.log_var.set(f"已打开评测结果目录：{results_dir}")
                if sys.platform == "win32":
                    os.startfile(str(results_dir))  # type: ignore[attr-defined]
                else:
                    webbrowser.open(results_dir.as_uri())
            except Exception as exc:
                messagebox.showerror("打开失败", f"无法打开结果目录：{exc}")

        def stop_all(self) -> None:
            for proc in list(self.processes.values()):
                _terminate_process_tree(proc)
            self.processes.clear()

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
