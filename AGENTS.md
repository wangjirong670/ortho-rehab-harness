# AGENTS.md — OrthoRehab-TCM-Harness 仓库协作与工程规范

本文件为所有在本仓库工作的 AI Agent 与开发者提供统一的架构边界、临床安全铁律、Windows 工程避坑指南与常用验收命令。

---

## 一、项目定位与 `v0.3.0` 架构分层

`OrthoRehab-TCM-Harness` 是面向**中医骨伤科与康复科**的共生双端确定性临床护栏系统，自 `v0.3.0` 起采用前后端分离单窗口原生桌面架构（**Single Source of Truth**）：

```text
knowledge_base/*.yaml (16 条红旗 / 11 组用药冲突 / 4 套骨伤量表)
        +
core_tools/ (schemas / calculator / rule_scanner / llm_client)
        ├──▶ app_copilot/service.py  (临床副驾编排 + 输出确定性复检)
        └──▶ app_benchmark/service.py & runner.py & evaluator.py (红队 20 题评测)
                └──▶ backend/ (FastAPI REST & SSE)
                        └──▶ frontend/ (React + Vite + Tailwind 玻璃态 SPA)
                                └──▶ desktop_app.py (PyWebView 原生单窗口主入口)
```

- **入口约定**：`desktop_app.py` 是唯一主入口（支持原生桌面、`--browser`、`--server-only`、`--check`、`--mock`）。
- **旧版弃用说明**：`app_copilot/streamlit_copilot.py`、`app_benchmark/streamlit_bench.py` 与旧 Tkinter 双网页控制面板自 `v0.3.0` 起已弃用，不得再打入发布包；`launcher.py` 仅作为底层运行时路径解析与 `.env` 加载工具模块保留，勿删除。

---

## 二、核心安全与工程红线（必须严格遵守）

1. **确定性外置（严禁交由概率模型拍脑袋）**：
   - 量表计算（`core_tools/calculator.py`）：`VAS_PAIN`、`WOMAC_LIKERT_V3_1`、`JOA_CERVICAL_17`、`JOA_LUMBAR_29` 必须由确定性代码核算，**严禁缺项静默补 `0`**，且 `JOA_CERVICAL_17 <= 12.5` 必须自动联动 `RF_001`。
   - 红旗与用药冲突（`core_tools/rule_scanner.py`）：必须执行双轨扫描（否定词窗口 + 停药洗脱期 `washout_window_days` + `current_state_snapshot` 时间线消解），并在文书输出侧经过 `verify_draft_against_alerts` 二次硬核复检。
2. **否定词边界与防绕过约束**：
   - **严禁将单字 `"不"` 直接加入宽窗口否定词表 `global_negation_terms`**（否则会放行“如果不舒服可以按摩”等高危条件句）。
   - 任何涉及 `core_tools/rule_scanner.py` 或 `app_benchmark/evaluator.py` 的否定判定改动，**必须同步补充正反双向回归单测**（既要防止“不坐矮凳/勿热敷/不能停药”被误杀或误抑制，也要封堵“不得不/不妨/不如”等双重否定或委婉绕过）。
3. **数据脱敏与公私隔离**：
   - 代码、示例、测试用例与截图中仅允许使用虚构脱敏数据，严禁出现真实患者姓名、手机号、身份证号、住院号或可识别医院信息。
   - `.gitignore` 与 `packaging/build_release.py` 必须严格屏蔽 `.env`、`.env.*`（保留 `.env.example`）、`*.secret`、`*.key`、`app_benchmark/test_cases/private/*`（仅留 `README.md` 与 `.gitkeep`）、`results/`、`logs/`、`custom_cases/`、`frontend/node_modules/`、`frontend/dist/`、`dist/`、`build/`、`release_artifacts/`。
4. **仓库密钥零留存规范**：
   - **严禁在源码根目录 `.env` 或仓库内 `dist/` 目录中长期存放真实云端 API Key**（防止误打包或测试意外消耗真实额度）。
   - 真人使用真实云端 API 实测绿色版时，应将 `release_artifacts/` 产物**独立解压至仓库外目录**再配置密钥运行；仓库内日常开发与自动化回归测试默认走 `LLM_MOCK_MODE=true` 或本机离线 Ollama。
5. **医学资产可审计**：
   - 修改 `knowledge_base/*.yaml` 或 `app_benchmark/test_cases/` 后，必须运行 `.\.venv\Scripts\python.exe scripts/validate_kb.py` 并确认 `ALL CHECKS PASSED`。

---

## 三、Windows 环境、打包与发布避坑指南

- **PowerShell 5.1 语法**：不支持 `&&` 链式连接，必须使用 `; if ($LASTEXITCODE -eq 0) { ... }`；注意原生 PowerShell cmdlet（如 `Copy-Item`、`Remove-Item`）不刷新 `$LASTEXITCODE`，不要在 cmdlet 后紧接 `if ($LASTEXITCODE -eq 0)`。
- **零黑框 Windows GUI 子系统（`console=False`）**：
  - `packaging/build_release.py` 必须保持 `console=False`（PE `Subsystem = 2 / IMAGE_SUBSYSTEM_WINDOWS_GUI`），避免 Windows 11 双击弹出 Terminal 黑框；
  - `desktop_app.py` 必须保留 `_ensure_standard_streams()`，在 `sys.stdout/sys.stderr is None` 时将其重定向至 `app_home/logs/desktop_app.log` 且 `.isatty()` 返回 `False`，`uvicorn` 传 `use_colors=False`，防止冻结态报 `NoneType has no attribute isatty`；`--check` 同步将 JSON 落盘至 `app_home/logs/check_result.json`。
- **前后端打包顺序**：只要修改过 `frontend/`，必须先在 `frontend/` 执行 `npm run build` 生成 `frontend/dist/`，再运行 `packaging/build_release.py`。
- **GitHub Actions 与推送约束**：
  - 若本机 `gh` token 缺少 `workflow` OAuth scope，不要将 `.github/workflows/release.yml` 改动随普通代码一起 `git push`（会被 GitHub 拒推），应先同步写入 `packaging/release.yml.template` 由用户在网页端手动同步。
  - **未经用户明确要求，严禁擅自执行 `git push` 或打 Git tag 触发发版**。

---

## 四、常用开发、校验与打包命令速查

> 所有 `pytest` 回归测试应在确认未残留真实云端 API Key 的干净环境下运行（优先走 Mock 或隔离环境变量）。

```powershell
# 1. 知识库与评测用例 Schema 全量校验
.\.venv\Scripts\python.exe scripts/validate_kb.py

# 2. 全仓库自动化测试（含确定性引擎、评测器、FastAPI 接口与打包审计）
.\.venv\Scripts\python.exe -m pytest -q

# 3. 路由与运行环境快速自检（输出并落盘至 logs/check_result.json）
.\.venv\Scripts\python.exe desktop_app.py --check

# 4. 源码模式启动原生桌面工作台（离线 Mock）
.\.venv\Scripts\python.exe desktop_app.py --mock

# 5. 构建前端 SPA 并打包本地 Windows 绿色版
cd frontend ; npm ci ; if ($LASTEXITCODE -eq 0) { npm run build } ; cd ..
.\.venv\Scripts\python.exe packaging/build_release.py --version v0.3.0
```
