"""Pure business logic and deterministic orchestration service for OrthoRehab desktop copilot."""

from __future__ import annotations

import copy
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from app_copilot.prompts import (
    DEMO_CASES,
    DRAFT_MODE_LABELS,  # @deprecated: remove in P1
    CopilotExtractedNote,
    build_draft_messages,
    build_extraction_messages,
)
from core_tools.calculator import (
    calculate_scale,
    list_available_scales,
    load_scales_catalog,
)
from core_tools.harness_core import (
    TurnContext,
    collect_unified_alerts,
    generate_patient_education,
    run_turn,
    verify_draft_against_alerts,
)
from core_tools.llm_client import (
    LLMClient,
    LLMConfig,
    format_llm_error_zh,
    probe_ollama_models,
)
from core_tools.rule_scanner import (
    load_drug_conflicts_catalog,
    load_red_flags_catalog,
    merge_snapshot_into_structured_facts,
    scan_clinical_risks,
)

PRESET_MODELS: List[str] = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-reasoner",
    "dashscope/qwen-plus",
    "openai/gpt-4o-mini",
    "openai/o3-mini",
    "ollama_chat/qwen2.5:7b",
    "zai/glm-4-plus",
    "moonshot/moonshot-v1-8k",
    "anthropic/claude-3-7-sonnet-20250219",
    "gemini/gemini-2.5-pro",
]

SCORE_DIRECTION_LABELS: Dict[str, str] = {
    "higher_is_worse": "分值越高风险/症状越重（高分更重）",
    "lower_is_worse": "分值越低风险/功能障碍越重（低分更危重）",
}

SEVERITY_BADGE_MAP: Dict[str, str] = {
    "critical": "🔴 CRITICAL（极高危）",
    "high": "🟠 HIGH（高危）",
    "medium": "🟡 MEDIUM（中危）",
    "low": "🔵 LOW（低危）",
}

TCM_FIELD_LABELS: Dict[str, str] = {
    "inspection": "望诊",
    "auscultation_olfaction": "闻诊",
    "inquiry": "问诊",
    "palpation": "切诊 / 专科触诊",
    "tongue": "舌象",
    "pulse": "脉象",
    "syndrome_pattern": "辨证要点",
}

CLINICAL_ACTION_ZH: Dict[str, str] = {
    "STOP_AND_REFER": "立即停诊并转专科评估",
    "EMERGENCY_REFER": "立即急诊转诊（红线急救）",
    "STOP_AND_VERIFY_PROTOCOL": "暂停操作并核对手术医师康复方案",
    "STOP_AND_MODIFY_PLAN": "停止原方案并调整治疗计划",
    "STOP_AND_VERIFY_MEDS": "暂停叠加并立即核查用药清单",
    "CAUTION_AND_PROTECT": "高危警示与保护性体位管理",
    "COMPUTE_WITH_TOOL": "交由确定性量表工具核算",
    "ALLOW_CONSERVATIVE_GUIDANCE": "红旗阴性，可给予保守护养指导",
}

POLARITY_ZH: Dict[str, str] = {
    "present": "阳性",
    "absent": "阴性",
    "uncertain": "待明确",
}

MED_STATUS_ZH: Dict[str, str] = {
    "active": "在用",
    "stopped": "已停药",
    "unknown": "状态待核实",
}

SOURCE_LABEL_ZH: Dict[str, str] = {
    "text_scan": "原文规则扫描",
    "structured_facts": "结构化事实核查",
    "hybrid": "双轨联合命中",
    "scale_linkage": "确定性量表阈值联动",
}

WOMAC_SCORE_LABELS: Dict[int, str] = {
    0: "0 无",
    1: "1 轻度",
    2: "2 中度",
    3: "3 重度",
    4: "4 极重",
}

_PII_NAME_EXCLUSION = frozenset(
    {"某男", "某女", "男性", "女性", "未知", "匿名", "已脱敏", "待完善", "未提及", "中年", "老年", "青年"}
)
_PII_EXPLICIT_NAME_RE = re.compile(
    r"((?:患者姓名|姓名|病人|患者|联系人|家属)\s*[:：]\s*)([\u4e00-\u9fa5·]{2,4})"
)
_PII_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_PII_ID_CARD_RE = re.compile(
    r"(?<!\d)([1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)"
)
_PII_RECORD_ID_RE = re.compile(
    r"((?:住院号|病历号|门诊号|就诊卡号|医保号|床号)\s*[:：]\s*)([A-Za-z0-9-]{2,24})"
)


def describe_score_direction(score_direction: Optional[str]) -> str:
    """Return explicit Chinese clinical interpretation of a scale's score direction."""
    if not score_direction:
        return "方向未定义"
    return SCORE_DIRECTION_LABELS.get(score_direction, score_direction)


def sanitize_error_message(message: Any) -> str:
    """Redact API keys, Bearer tokens, and secret query params before UI rendering."""
    text = str(message or "")
    text = re.sub(r"sk-[A-Za-z0-9_*-]+", "sk-***", text)
    text = re.sub(r"AIza[0-9A-Za-z_-]{10,}", "AIza***", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1***", text, flags=re.IGNORECASE)
    text = re.sub(
        r"((?:api[_-]?key|secret[_-]?key|access[_-]?token|authorization)\s*[:=]\s*[\"']?)[^\s\"',;]+",
        r"\1***",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"([?&](?:api_key|key|token)=)[^&\s]+", r"\1***", text, flags=re.IGNORECASE)
    return text


def detect_potential_pii(text: str) -> Dict[str, int]:
    """Detect high-confidence Chinese PII tokens only in safe explicit contexts (never flagging medical terms)."""
    src = text or ""
    name_count = 0
    for m in _PII_EXPLICIT_NAME_RE.finditer(src):
        val = m.group(2).strip()
        if val and not val.startswith("某") and val not in _PII_NAME_EXCLUSION:
            name_count += 1

    phone_count = len(_PII_PHONE_RE.findall(src))
    id_count = len(_PII_ID_CARD_RE.findall(src))
    record_count = len(_PII_RECORD_ID_RE.findall(src))
    return {
        "person_name": name_count,
        "phone": phone_count,
        "id_card": id_count,
        "record_id": record_count,
        "total": name_count + phone_count + id_count + record_count,
    }


def _mask_pii_sample(kind: str, val: str) -> str:
    if kind == "person_name":
        return val[0] + "*" * max(1, len(val) - 1)
    if kind == "phone" and len(val) == 11:
        return f"{val[:3]}****{val[-4:]}"
    if kind == "id_card" and len(val) >= 8:
        return f"{val[:4]}**********{val[-4:]}"
    if len(val) >= 4:
        return f"{val[:2]}***{val[-2:]}"
    return "***"


def detect_potential_pii_details(text: str) -> Dict[str, Any]:
    """Return structured PII findings with type, Chinese label, count, and masked samples."""
    src = text or ""
    name_samples: List[str] = []
    for m in _PII_EXPLICIT_NAME_RE.finditer(src):
        val = m.group(2).strip()
        if val and not val.startswith("某") and val not in _PII_NAME_EXCLUSION:
            name_samples.append(_mask_pii_sample("person_name", val))

    phone_samples = [_mask_pii_sample("phone", m.group(1)) for m in _PII_PHONE_RE.finditer(src)]
    id_samples = [_mask_pii_sample("id_card", m.group(1)) for m in _PII_ID_CARD_RE.finditer(src)]
    record_samples = [_mask_pii_sample("record_id", m.group(2)) for m in _PII_RECORD_ID_RE.finditer(src)]

    findings: List[Dict[str, Any]] = []
    specs = (
        ("person_name", "显式姓名", name_samples),
        ("phone", "手机号码", phone_samples),
        ("id_card", "身份证号", id_samples),
        ("record_id", "住院/病历编号", record_samples),
    )
    for kind, label, samples in specs:
        if samples:
            findings.append(
                {
                    "type": kind,
                    "label": label,
                    "count": len(samples),
                    "samples": samples[:5],
                }
            )
    return {
        "has_pii": bool(findings),
        "findings": findings,
    }


def sanitize_clinical_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Redact explicit patient names, 11-digit phones, 18-digit IDs, and record IDs without harming medical terms."""
    src = text or ""
    counts = {"person_name": 0, "phone": 0, "id_card": 0, "record_id": 0}

    def _replace_name(match: re.Match[str]) -> str:
        prefix, val = match.group(1), match.group(2).strip()
        if not val or val.startswith("某") or val in _PII_NAME_EXCLUSION:
            return match.group(0)
        counts["person_name"] += 1
        return f"{prefix}[姓名已脱敏]"

    def _replace_phone(_: re.Match[str]) -> str:
        counts["phone"] += 1
        return "[手机号已脱敏]"

    def _replace_id(_: re.Match[str]) -> str:
        counts["id_card"] += 1
        return "[身份证号已脱敏]"

    def _replace_record(match: re.Match[str]) -> str:
        counts["record_id"] += 1
        return f"{match.group(1)}[编号已脱敏]"

    cleaned = _PII_EXPLICIT_NAME_RE.sub(_replace_name, src)
    cleaned = _PII_ID_CARD_RE.sub(_replace_id, cleaned)
    cleaned = _PII_PHONE_RE.sub(_replace_phone, cleaned)
    cleaned = _PII_RECORD_ID_RE.sub(_replace_record, cleaned)
    counts["total"] = sum(counts.values())
    return cleaned, counts


def sanitize_clinical_pii_payload(text: str) -> Dict[str, Any]:
    """Sanitize PII and return API-friendly dict with sanitized_text, changed, and summary."""
    cleaned, counts = sanitize_clinical_pii(text)
    changed = cleaned != (text or "") and counts["total"] > 0
    if counts["total"] > 0:
        summary = (
            f"已完成本地脱敏：姓名 {counts['person_name']} 处、手机号 {counts['phone']} 处、"
            f"身份证号 {counts['id_card']} 处、住院/病历号 {counts['record_id']} 处。"
        )
    else:
        summary = "未检测到需要替换的显式隐私字段（未改动医学词汇）。"
    return {
        "sanitized_text": cleaned,
        "changed": changed,
        "summary": summary,
    }


def markdown_to_plain_text(md_text: str) -> str:
    """Strip Markdown formatting into clean plain text suitable for EMR/WeChat/DingTalk pasting."""
    if not md_text:
        return ""
    lines_out: List[str] = []
    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            lines_out.append("")
            continue
        if re.fullmatch(r"[-*_]{3,}", stripped):
            lines_out.append("----------------------------------------")
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"__([^_]+)__", r"\1", line)
        line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"^(\s*)[-*]\s+", r"\1• ", line)
        lines_out.append(line)

    collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(lines_out))
    return collapsed.strip()


def build_clinical_handover_text(state: Dict[str, Any]) -> str:
    """Build a consolidated plain-text handover sheet combining red flags, scale scores, and draft."""
    lines: List[str] = [
        "【OrthoRehab 骨伤/康复临床交班摘要（脱敏辅助）】",
        "",
        "一、安全红旗与用药冲突摘要",
    ]
    alerts = state.get("unified_alerts") or []
    if not alerts:
        lines.append("• 安全扫描通过：未检出红旗征或高危中西医用药冲突。")
    else:
        for idx, alert in enumerate(alerts, start=1):
            rid = alert.get("rule_id", "")
            name = alert.get("name", "")
            action_zh = CLINICAL_ACTION_ZH.get(
                str(alert.get("clinical_action", "")), str(alert.get("clinical_action", ""))
            )
            lines.append(f"{idx}. [{rid}] {name} | 建议动作：{action_zh}")
            if alert.get("clinical_alert"):
                lines.append(f"   警示：{alert['clinical_alert']}")
            if alert.get("contraindicated_actions"):
                lines.append("   绝对禁忌：" + "；".join(str(x) for x in alert["contraindicated_actions"]))

    lines.extend(["", "二、确定性量表计算结果"])
    scales = state.get("scale_results") or []
    if not scales:
        lines.append("• 暂无量表评估结果。")
    else:
        for res in scales:
            sname = res.get("scale_name") or res.get("scale_id") or ""
            if res.get("status") == "ok":
                lines.append(
                    f"• {sname}：总分 {res.get('total_score')} / {res.get('max_total'):g}（{res.get('severity_band', '')}）"
                )
            else:
                lines.append(f"• {sname}：暂未完成算分（{res.get('message', '')}）")

    lines.extend(["", "三、临床文书草稿（纯文本）"])
    draft = state.get("draft") or {}
    if draft.get("blocked"):
        lines.append("⚠️ 原生成草稿因违背安全约束已被拦截，请先依据上方红旗提示核查。")
    else:
        lines.append(markdown_to_plain_text(draft.get("content", "")) or "暂无生成文书。")

    return "\n".join(lines).strip()


def extract_text_from_uploaded_file(filename: str, raw_bytes: bytes) -> str:
    """Extract plain UTF-8 clinical text from .txt, .md, or .docx files using only stdlib."""
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                xml_bytes = zf.read("word/document.xml")
        except Exception as exc:
            raise ValueError(f"无法读取 .docx 文件内容（请确认文件未加密且未损坏）：{exc}") from exc

        root = ET.fromstring(xml_bytes)
        ns_w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs: List[str] = []
        for p_elem in root.iter(f"{ns_w}p"):
            chunks = [t.text for t in p_elem.iter(f"{ns_w}t") if t.text]
            joined = "".join(chunks).strip()
            if joined:
                paragraphs.append(joined)
        return "\n".join(paragraphs).strip()

    if suffix and suffix not in (".txt", ".md"):
        raise ValueError(f"不支持的文件格式 '{suffix}'，仅支持 .txt、.md 或 .docx 文件。")

    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw_bytes.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码，请将病历另存为 UTF-8 编码的 .txt / .md 或 .docx 文件。")


def summarize_contraindicated_capsules(unified_alerts: List[Dict[str, Any]]) -> List[str]:
    """Generate concise prohibited-action capsule tags for compact top-screen rendering."""
    tags: List[str] = []
    seen = set()

    def _add(tag: str) -> None:
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)

    for alert in unified_alerts or []:
        blob = " ".join(
            [str(alert.get("name", "")), str(alert.get("clinical_alert", ""))]
            + [str(x) for x in (alert.get("contraindicated_actions") or [])]
        )
        if any(k in blob for k in ("旋转扳法", "斜扳", "掰脖子", "暴力正骨", "踩跷")):
            _add("🚫 旋转扳法/暴力正骨")
        if any(k in blob for k in ("艾灸", "重灸", "热敷", "烤灯", "热疗")):
            _add("🚫 艾灸/热敷热疗")
        if any(k in blob for k in ("三七", "活血", "跌打丸", "NSAIDs", "同服", "叠服", "药酒")):
            _add("🚫 活血药/止痛药叠加")
        if any(k in blob for k in ("推拿", "揉捏", "按摩")):
            _add("🚫 患处推拿揉按")
        if any(k in blob for k in ("牵引",)):
            _add("🚫 盲目脊柱牵引")
        if any(k in blob for k in ("弃拐", "全负重", "跑跳", "开链")):
            _add("🚫 早期负重/跑跳")
        if any(k in blob for k in ("屈髋", "矮凳", "翘二郎腿", "内旋")):
            _add("🚫 屈髋>90°/盘腿内旋")
        if any(k in blob for k in ("小针刀", "封闭")):
            _add("🚫 有创小针刀/封闭")
        if any(k in blob for k in ("急诊", "马尾", "深静脉血栓")):
            _add("🚑 严禁拖延保守观察")

    return tags[:8]


def _enrich_draft_entry(draft_result: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure a draft entry always contains plain_text and generated_at fields."""
    entry = dict(draft_result or {})
    content = str(entry.get("content") or "")
    entry["plain_text"] = markdown_to_plain_text(content)
    if not entry.get("generated_at"):
        entry["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return entry


def _resolve_draft_caches(state: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, bool]]:
    """Extract and normalize drafts_by_mode + draft_stale_by_mode, seeding legacy states if needed."""
    raw_cache = state.get("drafts_by_mode")
    drafts_by_mode: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_cache, dict):
        for mode_key, draft_val in raw_cache.items():
            if mode_key in DRAFT_MODE_LABELS and isinstance(draft_val, dict) and draft_val.get("content"):
                drafts_by_mode[mode_key] = _enrich_draft_entry(draft_val)

    raw_stale = state.get("draft_stale_by_mode")
    draft_stale_by_mode: Dict[str, bool] = {}
    if isinstance(raw_stale, dict):
        for mode_key, is_stale in raw_stale.items():
            if mode_key in DRAFT_MODE_LABELS:
                draft_stale_by_mode[mode_key] = bool(is_stale)

    # Legacy compatibility: seed cache from top-level draft if present on a non-fast-scan state
    if not drafts_by_mode and not state.get("fast_scan_only"):
        top_draft = state.get("draft")
        if isinstance(top_draft, dict) and top_draft.get("content"):
            top_mode = str(top_draft.get("mode") or state.get("draft_mode") or "ward_round")
            if top_mode in DRAFT_MODE_LABELS:
                drafts_by_mode[top_mode] = _enrich_draft_entry(top_draft)
                draft_stale_by_mode.setdefault(top_mode, bool(state.get("draft_stale", False)))

    for mode_key in drafts_by_mode:
        draft_stale_by_mode.setdefault(mode_key, False)

    return drafts_by_mode, draft_stale_by_mode


def format_copilot_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a copilot analysis state dict to match the CopilotAnalysisResponse API contract."""
    unified_alerts = list(state.get("unified_alerts") or [])
    drafts_by_mode, draft_stale_by_mode = _resolve_draft_caches(state)
    raw_draft = state.get("draft") or {}
    active_draft = _enrich_draft_entry(raw_draft) if isinstance(raw_draft, dict) and raw_draft else {}
    active_mode = str(state.get("draft_mode") or active_draft.get("mode") or "ward_round")
    if active_draft and active_mode in DRAFT_MODE_LABELS and active_mode not in drafts_by_mode and not state.get("fast_scan_only"):
        drafts_by_mode[active_mode] = copy.deepcopy(active_draft)
        draft_stale_by_mode.setdefault(active_mode, bool(state.get("draft_stale", False)))

    return {
        "raw_text": str(state.get("raw_text") or ""),
        "fast_scan_only": bool(state.get("fast_scan_only", False)),
        "extracted_note": state.get("extracted_note") or {},
        "risk_report": state.get("risk_report") or {},
        "scale_results": list(state.get("scale_results") or []),
        "unified_alerts": unified_alerts,
        "contraindicated_capsules": summarize_contraindicated_capsules(unified_alerts),
        "draft": active_draft,
        "draft_mode": active_mode,
        "draft_stale": bool(state.get("draft_stale", False)),
        "draft_plain_text": str(active_draft.get("plain_text") or ""),
        "drafts_by_mode": drafts_by_mode,
        "draft_stale_by_mode": draft_stale_by_mode,
        "draft_cache_hit": bool(state.get("draft_cache_hit", False)),
        "trace": state.get("trace") or {},
    }


def run_fast_safety_scan(raw_text: str) -> Dict[str, Any]:
    """Execute zero-LLM Track A deterministic red-flag and drug-conflict screening via unified run_turn()."""
    cleaned_text = (raw_text or "").strip()
    if not cleaned_text:
        raise ValueError("输入病历文本不能为空。")

    turn_res = run_turn(
        TurnContext(),
        cleaned_text,
        output_mode="screen",
    )
    placeholder_draft = _enrich_draft_entry(
        {
            "mode": "ward_round",
            "mode_label": DRAFT_MODE_LABELS["ward_round"],
            "content": "⚡ 当前为「极速安全筛查」结果，仅运行确定性规则引擎（零 LLM 等待）。如需生成量表算分与文书草稿，请点击上方「🚀 开始分析」。",
            "blocked": False,
            "safety_check": turn_res.verification,
            "model": "deterministic-only",
            "mock": True,
            "usage": {},
        }
    )
    return {
        "raw_text": cleaned_text,
        "fast_scan_only": True,
        "extracted_note": turn_res.extracted_note,
        "risk_report": turn_res.risk_report,
        "scale_results": [],
        "unified_alerts": turn_res.unified_alerts,
        "contraindicated_capsules": summarize_contraindicated_capsules(turn_res.unified_alerts),
        "draft": placeholder_draft,
        "draft_mode": "ward_round",
        "draft_stale": False,
        "draft_plain_text": placeholder_draft["plain_text"],
        "drafts_by_mode": {},
        "draft_stale_by_mode": {},
        "draft_cache_hit": False,
        "trace": {
            "extraction_model": "deterministic-only",
            "extraction_mock": True,
            "draft_model": "deterministic-only",
            "draft_mock": True,
            "draft_usage": {},
        },
    }


def evaluate_scale_entries(
    scale_params: Any,
    *,
    scales_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Deterministically evaluate extracted scale entries without defaulting missing fields to 0."""
    if not isinstance(scale_params, list):
        return []
    results: List[Dict[str, Any]] = []
    for entry in scale_params:
        if not isinstance(entry, dict):
            continue
        scale_id = str(entry.get("scale_id", "")).strip()
        if not scale_id:
            continue
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        calc_res = calculate_scale(scale_id, params, scales_path=scales_path)
        calc_res["input_params"] = dict(params)
        results.append(calc_res)
    return results


def generate_draft_document(
    mode: str,
    extracted_note: Dict[str, Any],
    risk_report: Dict[str, Any],
    scale_results: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    mock_mode: Optional[bool] = None,
    client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Generate ward-round outline or staged rehab education draft with hard constraints and output verification."""
    active_client = client or LLMClient()
    messages = build_draft_messages(mode, extracted_note, risk_report, scale_results)
    resp = active_client.chat_completion(messages, model=model, mock_mode=mock_mode)
    if resp.get("status") == "error":
        raise RuntimeError(sanitize_error_message(resp.get("message") or "文书草稿生成失败。"))

    unified_alerts = collect_unified_alerts(risk_report, scale_results)
    safety_check = verify_draft_against_alerts(resp.get("content", ""), unified_alerts)
    is_blocked = bool(safety_check.get("blocked"))
    safe_content = (
        "⚠️ **安全拦截：模型生成的文书草稿涉嫌违背当前激活的安全红旗/禁忌约束，已被确定性输出复检拦截，禁止渲染。**\n\n"
        f"- 拦截原因：{safety_check.get('message', '')}"
        if is_blocked
        else resp["content"]
    )
    return {
        "mode": mode,
        "mode_label": DRAFT_MODE_LABELS.get(mode, mode),
        "content": safe_content,
        "blocked": is_blocked,
        "safety_check": safety_check,
        "model": resp["model"],
        "mock": resp["mock"],
        "usage": resp.get("usage", {}),
        "messages": messages,
    }


def run_copilot_analysis(
    raw_text: str,
    draft_mode: str = "ward_round",
    model: Optional[str] = None,
    mock_mode: Optional[bool] = None,
    *,
    client: Optional[LLMClient] = None,
    scales_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full deterministic + LLM copilot pipeline via unified run_turn()."""
    cleaned_text = (raw_text or "").strip()
    if not cleaned_text:
        raise ValueError("输入病历文本不能为空。")
    if draft_mode not in DRAFT_MODE_LABELS:
        raise ValueError(f"不支持的文书模式: {draft_mode}")

    active_client = client or LLMClient()
    turn_res = run_turn(
        TurnContext(),
        cleaned_text,
        model=model,
        output_mode="full",
        client=active_client,
        mock_mode=mock_mode,
        scales_path=scales_path,
        response_style=draft_mode,
    )

    draft_result = _enrich_draft_entry(
        {
            "mode": draft_mode,
            "mode_label": DRAFT_MODE_LABELS.get(draft_mode, draft_mode),
            "content": turn_res.content,
            "blocked": turn_res.blocked,
            "safety_check": turn_res.verification,
            "model": turn_res.model,
            "mock": turn_res.mock,
            "usage": turn_res.usage,
            "messages": turn_res.messages,
        }
    )

    return {
        "raw_text": cleaned_text,
        "fast_scan_only": False,
        "extracted_note": turn_res.extracted_note,
        "risk_report": turn_res.risk_report,
        "scale_results": turn_res.scale_results,
        "unified_alerts": turn_res.unified_alerts,
        "contraindicated_capsules": summarize_contraindicated_capsules(turn_res.unified_alerts),
        "draft": draft_result,
        "draft_mode": draft_mode,
        "draft_stale": False,
        "draft_plain_text": draft_result["plain_text"],
        "drafts_by_mode": {draft_mode: copy.deepcopy(draft_result)},
        "draft_stale_by_mode": {draft_mode: False},
        "draft_cache_hit": False,
        "trace": {
            "extraction_model": turn_res.extraction_meta.get("model"),
            "extraction_mock": turn_res.extraction_meta.get("mock"),
            "extraction_raw_content": turn_res.extraction_meta.get("raw_content"),
            "draft_model": draft_result.get("model"),
            "draft_mock": draft_result.get("mock"),
            "draft_usage": draft_result.get("usage", {}),
        },
    }


def recalculate_scale_in_state(
    analysis_state: Dict[str, Any],
    scale_id: str,
    updated_params: Dict[str, Any],
    *,
    scale_index: Optional[int] = None,
    refresh_draft: bool = False,
    draft_mode: Optional[str] = None,
    model: Optional[str] = None,
    mock_mode: Optional[bool] = None,
    client: Optional[LLMClient] = None,
    scales_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Recalculate a single scale inside existing analysis state and sync top-level alerts."""
    new_state = copy.deepcopy(analysis_state)
    clean_params = {k: v for k, v in (updated_params or {}).items() if v is not None}

    new_res = calculate_scale(scale_id, clean_params, scales_path=scales_path)
    new_res["input_params"] = dict(clean_params)

    scale_results: List[Dict[str, Any]] = list(new_state.get("scale_results") or [])
    scale_params: List[Dict[str, Any]] = list(
        (new_state.get("extracted_note") or {}).get("scale_params") or []
    )

    if scale_index is not None and 0 <= scale_index < len(scale_results):
        scale_results[scale_index] = new_res
        if scale_index < len(scale_params):
            scale_params[scale_index] = {"scale_id": scale_id, "params": dict(clean_params)}
        else:
            scale_params.append({"scale_id": scale_id, "params": dict(clean_params)})
    else:
        existing_idx = next(
            (i for i, item in enumerate(scale_results) if item.get("scale_id") == scale_id),
            None,
        )
        if existing_idx is not None:
            scale_results[existing_idx] = new_res
            if existing_idx < len(scale_params):
                scale_params[existing_idx] = {"scale_id": scale_id, "params": dict(clean_params)}
        else:
            scale_results.append(new_res)
            scale_params.append({"scale_id": scale_id, "params": dict(clean_params)})

    new_state["scale_results"] = scale_results
    new_state.setdefault("extracted_note", {})["scale_params"] = scale_params
    unified_alerts = collect_unified_alerts(
        new_state.get("risk_report", {}),
        scale_results,
    )
    new_state["unified_alerts"] = unified_alerts
    new_state["contraindicated_capsules"] = summarize_contraindicated_capsules(unified_alerts)

    target_mode = draft_mode or new_state.get("draft_mode", "ward_round")
    new_state["draft_mode"] = target_mode

    drafts_by_mode, draft_stale_by_mode = _resolve_draft_caches(new_state)
    for cached_mode in list(drafts_by_mode.keys()):
        draft_stale_by_mode[cached_mode] = True

    if refresh_draft:
        draft_result = _enrich_draft_entry(
            generate_draft_document(
                target_mode,
                new_state["extracted_note"],
                new_state.get("risk_report", {}),
                scale_results,
                model=model,
                mock_mode=mock_mode,
                client=client,
            )
        )
        drafts_by_mode[target_mode] = copy.deepcopy(draft_result)
        draft_stale_by_mode[target_mode] = False
        new_state["draft"] = draft_result
        new_state["draft_stale"] = False
        new_state["draft_plain_text"] = draft_result["plain_text"]
        new_state["draft_cache_hit"] = False
        new_state.setdefault("trace", {})["draft_model"] = draft_result.get("model")
        new_state["trace"]["draft_mock"] = draft_result.get("mock")
        new_state["trace"]["draft_usage"] = draft_result.get("usage", {})
    else:
        new_state["draft_stale"] = True
        new_state["draft_cache_hit"] = False

    new_state["drafts_by_mode"] = drafts_by_mode
    new_state["draft_stale_by_mode"] = draft_stale_by_mode
    return new_state


def remove_scale_from_state(
    analysis_state: Dict[str, Any],
    scale_index: int,
) -> Dict[str, Any]:
    """Remove a scale card by index and synchronize unified_alerts + draft_stale flag."""
    new_state = copy.deepcopy(analysis_state)
    scale_results = list(new_state.get("scale_results") or [])
    scale_params = list((new_state.get("extracted_note") or {}).get("scale_params") or [])

    if 0 <= scale_index < len(scale_results):
        scale_results.pop(scale_index)
    if 0 <= scale_index < len(scale_params):
        scale_params.pop(scale_index)

    new_state["scale_results"] = scale_results
    new_state.setdefault("extracted_note", {})["scale_params"] = scale_params
    unified_alerts = collect_unified_alerts(
        new_state.get("risk_report", {}),
        scale_results,
    )
    new_state["unified_alerts"] = unified_alerts
    new_state["contraindicated_capsules"] = summarize_contraindicated_capsules(unified_alerts)

    drafts_by_mode, draft_stale_by_mode = _resolve_draft_caches(new_state)
    for cached_mode in list(drafts_by_mode.keys()):
        draft_stale_by_mode[cached_mode] = True
    new_state["drafts_by_mode"] = drafts_by_mode
    new_state["draft_stale_by_mode"] = draft_stale_by_mode
    new_state["draft_stale"] = True
    new_state["draft_cache_hit"] = False
    return new_state


def regenerate_draft_in_state(
    analysis_state: Dict[str, Any],
    draft_mode: str = "ward_round",
    *,
    force: bool = False,
    model: Optional[str] = None,
    mock_mode: Optional[bool] = None,
    client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Switch or regenerate draft document inside existing analysis state with lossless per-mode caching."""
    if draft_mode not in DRAFT_MODE_LABELS:
        raise ValueError(f"不支持的文书模式: {draft_mode}")
    new_state = copy.deepcopy(analysis_state)
    drafts_by_mode, draft_stale_by_mode = _resolve_draft_caches(new_state)

    if not force and draft_mode in drafts_by_mode:
        cached_draft = _enrich_draft_entry(copy.deepcopy(drafts_by_mode[draft_mode]))
        drafts_by_mode[draft_mode] = copy.deepcopy(cached_draft)
        mode_stale = bool(draft_stale_by_mode.get(draft_mode, False))
        new_state["draft"] = cached_draft
        new_state["draft_mode"] = draft_mode
        new_state["draft_stale"] = mode_stale
        new_state["draft_plain_text"] = cached_draft["plain_text"]
        new_state["drafts_by_mode"] = drafts_by_mode
        new_state["draft_stale_by_mode"] = draft_stale_by_mode
        new_state["draft_cache_hit"] = True
        return new_state

    draft_result = _enrich_draft_entry(
        generate_draft_document(
            draft_mode,
            new_state.get("extracted_note") or {},
            new_state.get("risk_report") or {},
            list(new_state.get("scale_results") or []),
            model=model,
            mock_mode=mock_mode,
            client=client,
        )
    )
    drafts_by_mode[draft_mode] = copy.deepcopy(draft_result)
    draft_stale_by_mode[draft_mode] = False

    new_state["draft"] = draft_result
    new_state["draft_mode"] = draft_mode
    new_state["draft_stale"] = False
    new_state["draft_plain_text"] = draft_result["plain_text"]
    new_state["drafts_by_mode"] = drafts_by_mode
    new_state["draft_stale_by_mode"] = draft_stale_by_mode
    new_state["draft_cache_hit"] = False
    new_state.setdefault("trace", {})["draft_model"] = draft_result.get("model")
    new_state["trace"]["draft_mock"] = draft_result.get("mock")
    new_state["trace"]["draft_usage"] = draft_result.get("usage", {})
    return new_state


def format_active_medication_stop_display(med: Dict[str, Any]) -> str:
    """Format active/in-washout medication cessation duration for Copilot UI display."""
    stopped_days = med.get("stopped_days_ago")
    if stopped_days is not None:
        return f"{stopped_days:g} 天（洗脱期内）"
    if med.get("status") == "stopped":
        return "已停药，时长未明（按洗脱期内处理）"
    return "在用 / 未停药"


def parse_discrete_score_options(spec: Any, item: Any) -> Optional[Dict[float, str]]:
    """Build Chinese score option labels for discrete small-range integer scale items."""
    scale_id = getattr(spec, "scale_id", "")
    if getattr(item, "allowed_values", None) is not None:
        mapping: Dict[float, str] = {}
        parenthetical = re.search(r"[（(]([^（）()]+)[）)]", str(getattr(item, "label", "")))
        if parenthetical:
            for part in re.split(r"[，,、;；]+", parenthetical.group(1)):
                kv = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[:=：]\s*(.+?)\s*$", part)
                if kv:
                    mapping[float(kv.group(1))] = f"{float(kv.group(1)):g} {kv.group(2).strip()}"
        for val in item.allowed_values:
            fv = float(val)
            mapping.setdefault(fv, f"{fv:g} 分")
        return mapping

    step = float(getattr(item, "step", None) or 1.0)
    min_v = float(getattr(item, "min_value", 0.0))
    max_v = float(getattr(item, "max_value", 0.0))
    if step != 1.0 or (max_v - min_v) > 4.0:
        return None

    if scale_id == "WOMAC_LIKERT_V3_1":
        return {float(k): v for k, v in WOMAC_SCORE_LABELS.items() if min_v <= k <= max_v}

    if scale_id == "JOA_LUMBAR_29":
        mapping = {}
        parenthetical = re.search(r"[（(]([^（）()]+)[）)]", str(getattr(item, "label", "")))
        if parenthetical:
            for part in re.split(r"[，,、;；]+", parenthetical.group(1)):
                kv = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[:=：]\s*(.+?)\s*$", part)
                if kv:
                    mapping[float(kv.group(1))] = f"{float(kv.group(1)):g} {kv.group(2).strip()}"
        cur = int(min_v)
        while cur <= int(max_v):
            mapping.setdefault(float(cur), f"{cur} 分")
            cur += 1
        return mapping

    return None
