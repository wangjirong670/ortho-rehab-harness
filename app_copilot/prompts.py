"""Prompt templates, extraction schema, and demo cases for Project B desktop copilot (Week 3)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from core_tools.calculator import load_scales_catalog

DraftMode = Literal["ward_round", "rehab_education"]

DRAFT_MODE_LABELS: Dict[str, str] = {
    "ward_round": "上级医师查房汇报提纲",
    "rehab_education": "分阶段康复宣教要点",
}


class TimelineEvent(BaseModel):
    date: str = Field(default="未标明时间")
    department: str = Field(default="未标明")
    event: str = Field(min_length=1)
    key_change: str = Field(default="")

    @field_validator("date", mode="before")
    @classmethod
    def _normalize_date(cls, value: Any) -> str:
        if value is None or not str(value).strip():
            return "未标明时间"
        return str(value).strip()

    @field_validator("department", mode="before")
    @classmethod
    def _normalize_department(cls, value: Any) -> str:
        if value is None or not str(value).strip():
            return "未标明"
        return str(value).strip()

    @field_validator("key_change", mode="before")
    @classmethod
    def _normalize_key_change(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


class SymptomFact(BaseModel):
    name: str = Field(min_length=1)
    polarity: Literal["present", "absent", "uncertain"] = "present"

    @field_validator("polarity", mode="before")
    @classmethod
    def _normalize_polarity(cls, value: Any) -> str:
        if value is None:
            return "uncertain"
        return str(value).strip().lower()


class MedicationFact(BaseModel):
    name: str = Field(min_length=1)
    status: Literal["active", "stopped", "unknown"] = "unknown"
    stopped_days_ago: Optional[float] = Field(default=None, ge=0)

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> str:
        if value is None:
            return "unknown"
        return str(value).strip().lower()


class StructuredFacts(BaseModel):
    symptoms: List[SymptomFact] = Field(default_factory=list)
    medications: List[MedicationFact] = Field(default_factory=list)


class SupersededCondition(BaseModel):
    old_state: str = Field(min_length=1)
    new_state: str = Field(min_length=1)
    rule_hint: Optional[str] = None

    @field_validator("rule_hint", mode="before")
    @classmethod
    def _normalize_rule_hint(cls, value: Any) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        return str(value).strip().upper()


class CurrentStateSnapshot(BaseModel):
    """Chronologically resolved latest clinical state after longitudinal conflict resolution."""

    active_medications: List[MedicationFact] = Field(default_factory=list)
    stopped_medications: List[MedicationFact] = Field(default_factory=list)
    current_symptoms: List[SymptomFact] = Field(default_factory=list)
    active_surgical_history: List[str] = Field(default_factory=list)
    superseded_conditions: List[SupersededCondition] = Field(default_factory=list)
    diagnosis_override_note: Optional[str] = None

    @field_validator("diagnosis_override_note", mode="before")
    @classmethod
    def _normalize_override_note(cls, value: Any) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        return str(value).strip()


class ScaleParamEntry(BaseModel):
    scale_id: str = Field(min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("scale_id", mode="before")
    @classmethod
    def _normalize_scale_id(cls, value: Any) -> str:
        return str(value).strip().upper()


class CopilotExtractedNote(BaseModel):
    """Pydantic v2 schema for structured extraction from anonymized clinical notes."""

    patient_summary: str = Field(default="未提及")
    chief_complaint: str = Field(default="未提及")
    history_of_present_illness: str = Field(default="未提及")
    tcm_four_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    physical_exam: List[str] = Field(default_factory=list)
    imaging_summary: str = Field(default="未提及")
    timeline_events: List[TimelineEvent] = Field(default_factory=list)
    structured_facts: StructuredFacts = Field(default_factory=StructuredFacts)
    current_state_snapshot: CurrentStateSnapshot = Field(default_factory=CurrentStateSnapshot)
    scale_params: List[ScaleParamEntry] = Field(default_factory=list)

    @field_validator("current_state_snapshot", mode="before")
    @classmethod
    def _coerce_snapshot(cls, value: Any) -> Any:
        if value is None:
            return {}
        return value

    @field_validator(
        "patient_summary",
        "chief_complaint",
        "history_of_present_illness",
        "imaging_summary",
        mode="before",
    )
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        if value is None or not str(value).strip():
            return "未提及"
        return str(value).strip()

    @field_validator("tcm_four_diagnostics", mode="before")
    @classmethod
    def _coerce_dict(cls, value: Any) -> Dict[str, Any]:
        if value is None or not isinstance(value, dict):
            return {}
        return value

    @field_validator("physical_exam", mode="before")
    @classmethod
    def _coerce_physical_exam(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @field_validator("timeline_events", "scale_params", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> List[Any]:
        if value is None:
            return []
        return list(value)


def _format_scales_catalog_for_prompt() -> str:
    """Dynamically render exact scale_id and item_id definitions from scales_spec.yaml."""
    catalog = load_scales_catalog()
    lines: List[str] = []
    for spec in catalog.scales:
        direction_hint = (
            "分值越高越重" if spec.direction == "higher_is_worse" else "分值越高功能越好/越低越重"
        )
        lines.append(
            f"- scale_id: `{spec.scale_id}` | 名称: {spec.name} | 总分范围: [{spec.min_total:g}, {spec.max_total:g}] ({direction_hint})"
        )
        if spec.improvement_rate_formula:
            lines.append("  - 可选附加字段 `pre_score`: 术前基线总分（仅在原文明确提供术前评分时填写）")
        for item in spec.items:
            constraint = (
                f"允许离散值 {item.allowed_values}"
                if item.allowed_values is not None
                else f"范围 [{item.min_value:g}, {item.max_value:g}], 步长 {item.step:g}"
            )
            lines.append(f"  - `{item.item_id}`: {item.label}（{constraint}）")
    return "\n".join(lines)


def build_extraction_messages(raw_text: str) -> List[Dict[str, str]]:
    """Build strict Chinese extraction prompt messages with dynamic scale catalog."""
    scales_block = _format_scales_catalog_for_prompt()
    system_prompt = f"""你是中医骨伤科与康复科桌面临床辅助副驾的结构化信息抽取器。
请从用户提供的**脱敏临床病历文本**中提取结构化事实，并**仅输出一个合法的 JSON 对象**（不要输出任何 Markdown 解释文字）。

### 输出 JSON 字段规范（严格对齐 CopilotExtractedNote）
1. `patient_summary` (string): 脱敏基本画像（如性别、年龄段、职业/基础疾病背景；严禁编造姓名、住院号、身份证号等隐私信息）。
2. `chief_complaint` (string): 主诉摘要。
3. `history_of_present_illness` (string): 现病史摘要（发病诱因、演变过程、既往处理）。
4. `tcm_four_diagnostics` (object): 中医骨伤四诊要点，建议包含键：
   - `inspection` (望诊), `auscultation_olfaction` (闻诊), `inquiry` (问诊), `palpation` (切诊/专科触诊), `tongue` (舌象), `pulse` (脉象), `syndrome_pattern` (辨证要点)。未提及填"未提及"。
5. `physical_exam` (array[string]): 专科查体要点列表（如压痛部位、病理征、肌力、关节活动度等）。
6. `imaging_summary` (string): 影像学/检验摘要（X线、CT、MRI、骨密度、彩超、INR 等）。
7. `timeline_events` (array[object]): 跨诊次关键事件时间线，按时间先后严格排列。每项包含：
   - `"date"` (string): 时间点（如 `"2025-11-10"`、`"半年前"`、`"术后第16周"`、`"本次复诊"`）
   - `"department"` (string): 科室或就诊场景（如 `"心内科"`、`"骨伤科"`、`"运动医学科"`，未标明填 `"未标明"`）
   - `"event"` (string): 该次就诊/节点的关键临床事件摘要
   - `"key_change"` (string): 关键变化或状态转折（如 `"已停用华法林365天"`、`"由寒痹转为热痹"`、`"主刀准予直线慢跑"`，无则填 `""`）
8. `structured_facts` (object): 供确定性安全扫描器轨道 B 使用的结构化临床事实：
   - `symptoms` (array[object]): 每项 `{{"name": "症状或体征名称", "polarity": "present" | "absent" | "uncertain"}}`。
     * 极性标注铁律：若原文为"否认/无/未见/排除/阴性"（如"否认踩棉花感"、"无二便失禁"、"霍夫曼征阴性"），**必须**提取并标注 `polarity: "absent"`，绝不可标为 `"present"` 或遗漏。
   - `medications` (array[object]): 每项 `{{"name": "药品或中成药名称", "status": "active" | "stopped" | "unknown", "stopped_days_ago": float | null}}`。
     * 停药与洗脱窗口铁律：若原文明确说明某药已停用及停药时长（如"停用半年"≈ 180.0、"停用3个月"≈ 90.0、"停用2天"≈ 2.0），必须标 `status: "stopped"` 并准确换算填入 `stopped_days_ago`；当前在用/刚加服标 `"active"`。
9. `current_state_snapshot` (object): 经**跨诊次时间线状态抽取与冲突消解**后的最新有效状态快照，严格按时间顺序用最新状态覆盖过期旧状态：
   - `active_medications` (array[MedicationFact]): 当前仍在用及**停药未满洗脱期**（如氯吡格雷停药仅3天）的药物清单。
   - `stopped_medications` (array[MedicationFact]): 已明确停用且**超过洗脱期**的药物清单（含 `stopped_days_ago`）。
   - `current_symptoms` (array[SymptomFact]): 经时间线消解后的最新有效症状极性（若早期阴性但最新诊次新发马尾或热痹红肿热痛，必须标为 `"present"`）。
   - `active_surgical_history` (array[string]): 仍具长期临床约束的既往手术/内固定/肿瘤史（如 THA、ACDF、ACL重建、冠脉支架、恶性肿瘤术后）。
   - `superseded_conditions` (array[object]): 已被最新诊次推翻或已过保护期的旧状态，每项 `{{"old_state": "...", "new_state": "...", "rule_hint": "RF_007" | null}}`（例如术后第16周获主刀准予慢跑从而解除早期保护期 `RF_007`，或2月前寒痹已被本次热痹推翻）。
   - `diagnosis_override_note` (string | null): 简要说明最新诊次是否推翻了既往诊断/分期（未发生推翻填 `null`）。
10. `scale_params` (array[object]): 每项 `{{"scale_id": "<精确量表ID>", "params": {{<item_id>: <数值>}}}}`。
   * **量表分值抽取绝对禁令（严禁脑补 / 严禁默认填 0）**：
     - 只能使用下方知识库清单中精确的 `scale_id` 和合法 `item_id`。
     - **严格只填写病历原文中明确给出具体数值的子项**！
     - 原文未提及或未记录的子项**必须完全省略**，绝不允许推测、补齐或默认填 `0`！
     - 若原文仅给出某量表的部分子项得分，只输出已明确提及的那些字段即可（后续将由确定性计算器提示缺失项供医师手动补录）；若完全未提及任何量表数值，`scale_params` 输出空数组 `[]`。

### 合法量表与条目 ID 清单（动态读取自 knowledge_base/scales_spec.yaml）
{scales_block}
"""

    user_prompt = (
        "请对以下脱敏临床病历文本进行结构化抽取，严格按照 System 约束仅输出一个合法 JSON 对象：\n\n"
        f"【病历原文开始】\n{raw_text.strip()}\n【病历原文结束】"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _format_alerts_for_prompt(risk_report: Dict[str, Any], scale_results: List[Dict[str, Any]]) -> str:
    """Format deterministic red flags, drug conflicts, and scale-linked red flags as hard constraints."""
    lines: List[str] = []
    alerts = risk_report.get("alerts", []) or []
    for alert in alerts:
        rule_id = alert.get("rule_id", "UNKNOWN")
        name = alert.get("name", "")
        severity = alert.get("severity", "high")
        evidence = "、".join(str(x) for x in alert.get("matched_evidence", []) or [])
        clinical_alert = alert.get("clinical_alert", "")
        lines.append(f"- [{rule_id} | 严重度: {severity}] {name}")
        if evidence:
            lines.append(f"  - 命中依据: {evidence}")
        if clinical_alert:
            lines.append(f"  - 临床警示: {clinical_alert}")
        if alert.get("contraindicated_actions"):
            prohibited = "；".join(str(x) for x in alert["contraindicated_actions"])
            lines.append(f"  - 绝对禁忌操作 (contraindicated_actions): {prohibited}")
        if alert.get("suggested_handling"):
            lines.append(f"  - 处置建议 (suggested_handling): {alert['suggested_handling']}")
        if alert.get("reference"):
            lines.append(f"  - 规则依据: {'；'.join(str(x) for x in alert['reference'])}")

    for res in scale_results:
        if res.get("status") == "ok" and res.get("red_flag_triggered"):
            linked = "、".join(res.get("linked_rule_ids") or []) or res.get("scale_id", "")
            lines.append(
                f"- [量表联动红旗 | 关联规则: {linked}] {res.get('scale_name', res.get('scale_id'))} "
                f"总分 = {res.get('total_score')} 分"
            )
            lines.append(f"  - 联动警示: {res.get('red_flag_reason', '')}")

    if not lines:
        return "【无命中红旗/用药冲突】当前确定性双轨扫描与量表阈值未触发高危红旗或用药冲突。"
    return "\n".join(lines)


def _format_scales_for_prompt(scale_results: List[Dict[str, Any]]) -> str:
    """Format deterministic scale outputs into prompt text."""
    if not scale_results:
        return "【暂无量表计算结果】病历中未提供可计算的量表参数。"

    lines: List[str] = []
    for res in scale_results:
        scale_id = res.get("scale_id", "UNKNOWN")
        scale_name = res.get("scale_name", scale_id)
        direction = res.get("score_direction")
        direction_zh = (
            "分值越高风险/症状越重"
            if direction == "higher_is_worse"
            else ("分值越低风险/功能障碍越重" if direction == "lower_is_worse" else "未知方向")
        )
        if res.get("status") == "ok":
            line = (
                f"- {scale_name} (`{scale_id}`): 总分 **{res.get('total_score')}** "
                f"(理论范围 {res.get('min_total')}~{res.get('max_total')}), "
                f"分级: **{res.get('severity_band')}**, 方向说明: {direction_zh}"
            )
            if "normalized_score_pct" in res:
                line += f", 0-100 标准化分: {res['normalized_score_pct']}%"
            if res.get("red_flag_triggered"):
                line += f", **触发量表联动红旗** ({', '.join(res.get('linked_rule_ids', []))})"
            lines.append(line)
            if res.get("subscale_scores"):
                sub_parts = [
                    f"{sub.get('label', sid)}={sub.get('score')}分"
                    for sid, sub in res["subscale_scores"].items()
                ]
                lines.append(f"  - 分维度得分: {'；'.join(sub_parts)}")
            if res.get("improvement_rate") and res["improvement_rate"].get("status") == "ok":
                ir = res["improvement_rate"]
                lines.append(
                    f"  - 平林法改善率: {ir['improvement_rate_pct']}% ({ir.get('efficacy_grade', '')})"
                )
        else:
            lines.append(
                f"- [未完成算分] {scale_name} (`{scale_id}`): 方向说明: {direction_zh}；"
                f"缺失或非法字段导致暂未出分（{res.get('message', '')}）。严禁自行估算或补 0！"
            )
    return "\n".join(lines)


def build_draft_messages(
    mode: str,
    extracted_note: Dict[str, Any],
    risk_report: Dict[str, Any],
    scale_results: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build high-constraint clinical draft generation prompt injecting deterministic outputs."""
    if mode not in DRAFT_MODE_LABELS:
        raise ValueError(f"Unsupported draft mode '{mode}'. Expected one of {sorted(DRAFT_MODE_LABELS)}")

    mode_label = DRAFT_MODE_LABELS[mode]
    hard_constraints_text = _format_alerts_for_prompt(risk_report, scale_results)
    scales_text = _format_scales_for_prompt(scale_results)

    if mode == "ward_round":
        structure_guide = (
            "请输出适合骨伤/康复科医生向**上级医师汇报查房**的 Markdown 提纲，建议结构：\n"
            "1. **病例核心摘要与跨诊次时间线**\n"
            "2. **中医四诊与专科查体/影像要点**\n"
            "3. **确定性量表评估结果**（原样引用精确总分、分级与方向说明）\n"
            "4. **安全红旗与中西医用药冲突硬约束**（若命中必须置前高亮，逐条列出禁忌与依据）\n"
            "5. **查房拟请示问题与下一步安全检查/诊疗思路**\n"
            "6. **临床辅助边界声明**"
        )
    else:
        structure_guide = (
            "请输出适合临床医护人员向患者或家属讲解的**分阶段康复宣教要点** Markdown 草稿，建议结构：\n"
            "1. **当前风险提示与绝对禁忌红线**（若存在红旗或用药冲突，必须作为第一条醒目警告，明确哪些动作/手法/药物绝不可自行使用）\n"
            "2. **当前评估状态解读**（基于确定性量表总分与分级客观解读）\n"
            "3. **分阶段康复与日常护养要点**（结合红旗约束区分急性/安全评估期与稳定恢复期；若当前为高危红旗或急诊指征，明确要求先由专科排查后再进入主动训练）\n"
            "4. **用药核查与复诊警示信号**\n"
            "5. **合规边界声明**"
        )

    system_prompt = (
        f"你是中医骨伤科与康复科专业临床文书副驾，当前任务是生成【{mode_label}】（【文书模式：{mode}】）。\n"
        "### 不可违背的确定性安全与合规铁律\n"
        "1. **严禁违背红旗与禁忌约束**：下文“确定性安全硬约束”区块列出的所有 `rule_id`、绝对禁忌操作 (`contraindicated_actions`) "
        "与用药处置建议 (`suggested_handling`) 均来自 Python 确定性规则引擎，你必须无条件遵守，**严禁在任何建议中推荐被禁止的手法、理疗、运动或合并用药**。\n"
        "2. **严禁篡改或脑补量表分值**：所有量表总分、分级、方向说明及是否缺失均以“确定性量表计算结果”为准，不得擅自修改数字或给未完成量表编造分数。\n"
        "3. **角色边界声明**：仅限脱敏病历整理与临床思维辅助参考，严禁直接对患者下达最终医疗诊断或开具处方。\n"
        "4. 输出格式为规范、条理清晰的中文 Markdown。"
    )

    note_snapshot = json.dumps(extracted_note, ensure_ascii=False, indent=2)
    user_prompt = (
        f"请生成【{mode_label}】（mode={mode}）。\n\n"
        f"### 一、确定性安全硬约束（绝对不可违背）\n{hard_constraints_text}\n\n"
        f"### 二、确定性量表计算结果\n{scales_text}\n\n"
        f"### 三、结构化脱敏病历上下文\n```json\n{note_snapshot}\n```\n\n"
        f"### 四、文书组织要求\n{structure_guide}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


DEMO_CASES: Dict[str, Dict[str, str]] = {
    "cervical_myelopathy_joa": {
        "case_id": "cervical_myelopathy_joa",
        "title": "演示病例 1：颈椎踩棉花感 + 颈椎 JOA 评估",
        "button_label": "🚨 病例 1：颈椎踩棉花感 + JOA",
        "description": "验证 RF_001 脊髓受压红旗 + JOA_CERVICAL_17 (≤12.5分) 联动告警与手法绝对禁忌。",
        "default_mode": "ward_round",
        "text": (
            "某男，58岁，退休教师（脱敏门诊记录 DEMO-001）。\n"
            "主诉：颈肩部酸胀僵硬3个月，步态踩棉花感、双手扣纽扣不灵2周。\n"
            "现病史：3个月前无明显诱因出现颈肩僵硬；2周前逐渐出现走路发飘、步态踩棉花感，伴有持物易落及扣纽扣不灵；"
            "外院患者曾询问能否做颈椎旋转正骨推拿，暂未处理。今日来诊，仅轻度尿频感，无尿潴留，无二便失禁，无发热，无夜间静息痛。\n"
            "中医四诊：望诊神清、颈部活动受限、步态略呈痉挛步态；闻诊语声清晰；问诊双上肢麻木、握力下降、夜寐欠安；"
            "切诊颈夹脊压痛，舌质淡暗、苔薄白，脉弦细。辨证属肝肾不足、气滞血瘀型颈痹（疑似脊髓型颈椎病）。\n"
            "专科查体：C4-C6棘突旁压痛，霍夫曼征阳性，双上肢痛觉减退，双膝腱反射亢进。\n"
            "影像学摘要：外院颈椎X线提示C4-C6椎体骨质增生、椎间隙变窄（尚未行颈椎MRI）。\n"
            "量表评估（颈椎 JOA 17 分制）：上肢运动功能 2 分，下肢运动功能 2 分，上肢感觉功能 1 分，"
            "下肢感觉功能 1 分，躯干感觉功能 1 分，膀胱功能 2 分。"
        ),
    },
    "warfarin_notoginseng_washout": {
        "case_id": "warfarin_notoginseng_washout",
        "title": "演示病例 2：跨诊次华法林 + 三七粉 + 已停用阿司匹林",
        "button_label": "💊 病例 2：华法林 + 三七粉 + 洗脱期",
        "description": "验证 DC_001 抗凝药+活血中药冲突、已停用半年的阿司匹林进洗脱期过滤及 WOMAC 缺项拦截。",
        "default_mode": "ward_round",
        "text": (
            "某女，67岁（脱敏跨诊次随访记录 DEMO-002）。\n"
            "2025-11-10 心内科记录：因心房颤动长期规范口服华法林抗凝治疗（当前仍在用，最近复查INR 2.3）；"
            "既往曾服阿司匹林，因故于半年前停用（阿司匹林已停用半年，约180天）。\n"
            "2026-05-08 骨伤门诊记录：3天前不慎扭伤右膝关节，右膝酸痛伴屈伸不利，局部不红不热，无发热；"
            "X线未见骨折线，排除骨折；否认踩棉花感、无鞍区麻木、无二便失禁。患者昨日起自行加服三七粉活血消肿。\n"
            "中医四诊：望诊右膝轻度肿胀、皮色如常，舌暗红、苔薄白，脉弦涩。辨证属气滞血瘀证。\n"
            "量表评估（记录尚不完整）：VAS 评分 5 分；WOMAC 部分条目记录：平地行走时疼痛 2 分，上下楼梯时疼痛 3 分，"
            "早晨醒来时关节僵硬 2 分（其余日常活动功能条目待补录）。"
        ),
    },
    "sedentary_low_back_pain": {
        "case_id": "sedentary_low_back_pain",
        "title": "演示病例 3：明确阴性红旗的普通久坐腰痛",
        "button_label": "🟢 病例 3：久坐腰痛（阴性防误杀）",
        "description": "验证否定词防误杀（零告警 + suppressed_matches 审计记录）及 VAS 正常算分。",
        "default_mode": "rehab_education",
        "text": (
            "某男，32岁，软件工程师（脱敏门诊记录 DEMO-003）。\n"
            "主诉：长期久坐办公后腰部酸胀不适3周，适当起身活动后稍缓解。\n"
            "现病史：近3周因连续伏案工作出现下腰部酸沉困痛，晨起略僵硬、活动10分钟后减轻；"
            "否认外伤史及肿瘤病史，目前未服用抗凝药、止痛药或任何活血中成药。\n"
            "专科查体与红旗筛查：腰肌轻度紧张压痛，否认踩棉花感，霍夫曼征阴性；"
            "无鞍区麻木、无二便失禁，大小便正常；无夜间静息痛，无发热；无关节红肿热痛，局部不红不热；"
            "X线未见骨折线，排除骨折。\n"
            "中医四诊：望诊精神可、腰椎生理曲度略直，舌淡红、苔薄白，脉缓。辨证属劳损气滞型腰痛。\n"
            "量表评估：VAS 疼痛评分 3 分。"
        ),
    },
}
