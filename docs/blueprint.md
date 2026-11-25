# OrthoRehab-TCM-Harness 项目蓝图与工程约束记录

本文档固化项目初始规划，并记录第 1 周架构评审中明确的关键防坑约束。

## 1. 核心架构原则

1. **一套资产，双端复用**：`knowledge_base/` 与 `core_tools/` 同时服务于桌面副驾 (`app_copilot`) 与红队评测沙箱 (`app_benchmark`)。
2. **确定性外置**：量表算分、硬红旗拦截、中西医用药冲突判定全部由 Python 确定性执行，LLM 仅负责结构化抽取与临床自然语言组织。
3. **数据脱敏与公私隔离**：评测用例拆分为 `app_benchmark/test_cases/public/`（公开示例）与 `app_benchmark/test_cases/private/`（防泄漏私有集，`.gitignore` 屏蔽）。

## 2. 关键防坑工程约束

### 2.1 否定词与时态防误杀（规则扫描 + 评测判分）
- 临床病历常见“否认踩棉花感”“无二便失禁”“停用华法林3月”等表述。
- **副驾侧 (`rule_scanner.py`)**：优先基于 LLM 抽取出的结构化事实（含 `polarity: present/absent/uncertain` 与 `status: active/stopped/unknown`）触发确定性规则，再辅以带中文否定词窗口的文本兜底扫描。
- **评测侧 (`evaluator.py`)**：`forbidden_patterns` 均声明 `ignore_if_negated`，避免将安全回复（如“切忌在家自行复位”）因命中子串而误杀判负。

### 2.2 量表版本显式化 (`scales_spec.yaml`)
- `VAS_PAIN`：0-10 分制视觉模拟疼痛评分。
- `WOMAC_LIKERT_V3_1`：Likert 5 级评分制（疼痛 0-20、僵硬 0-8、功能 0-68，总分 0-96，可选 0-100 标准化分）。
- `JOA_CERVICAL_17`：颈椎 JOA 17 分制（含 Hirabayashi 改善率公式）。
- `JOA_LUMBAR_29`：腰椎 JOA 29 分制（含膀胱功能负分项与改善率公式）。

### 2.3 医学资产双人复核标记
- `knowledge_base/*.yaml` 中每条规则均带 `review_status`（`draft_for_clinician_review` / `clinician_verified`），便于临床人员逐条核验并追踪进度。

## 3. 6 周双轨路线图

- **第 1 周**：仓库骨架、首批 10 条红旗规则、10 组用药冲突、量表规范、前 10 道评测陷阱题与 Schema 校验脚本。
- **第 2 周**：`core_tools/` 确定性计算器、带否定词窗口的规则扫描器、LiteLLM 多模型客户端与单元测试。
- **第 3 周**：`app_copilot/` v0.1 上线（脱敏病历结构化 → 确定性算分 → 红旗/用药警示条 → 查房提纲/宣教草稿）。
- **第 4 周**：`app_benchmark/` v0.1 上线（扩充至 20+ 题，批量跑 3-4 个主流模型并反向测试项目 B）。
- **第 5 周**：跨诊次纵向病历时间线状态抽取 + 10-15 道长病历陷阱回归测试。
- **第 6 周**：整理翻车实录、脱敏清理、完善文档与开源发布。
