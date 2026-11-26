"""Streamlit UI and pure orchestration logic for Project B desktop copilot (Week 3)."""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app_copilot.prompts import (
    DEMO_CASES,
    DRAFT_MODE_LABELS,
    CopilotExtractedNote,
    build_draft_messages,
    build_extraction_messages,
)
from core_tools.calculator import (
    calculate_scale,
    list_available_scales,
    load_scales_catalog,
)
from core_tools.llm_client import LLMClient, LLMConfig
from core_tools.rule_scanner import (
    load_red_flags_catalog,
    merge_snapshot_into_structured_facts,
    scan_clinical_risks,
)

PRESET_MODELS: List[str] = [
    "deepseek/deepseek-chat",
    "dashscope/qwen-plus",
    "openai/gpt-4o-mini",
    "zai/glm-4-plus",
    "moonshot/moonshot-v1-8k",
    "anthropic/claude-3-5-sonnet-latest",
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


_HIGH_RISK_DRAFT_ACTION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("旋转扳法/正骨复位", r"(?:颈椎|腰椎)?旋转(?:扳法|复位|整复)|斜扳|踩跷|正骨|整脊|高速低幅整复|掰脖子|在家自行复位"),
    ("推拿按摩", r"推拿|按摩|揉按|揉捏(?:腓肠肌)?|重手法"),
    ("热敷/重灸/热疗", r"热敷|重灸|艾灸|烤灯|红外线|超短波|短波|透热理疗"),
    ("提前弃拐/全负重/跑跳", r"(?:提前)?(?:丢拐|弃拐)(?:全)?负重|全负重|跑跳|急停变向|绑沙袋|开链(?:抗阻)?伸膝"),
    ("有创操作(小针刀/封闭)", r"小针刀|针刀|封闭注射|激素封闭|拔罐放血"),
    ("脱位高危体位", r"屈髋(?:大于|超过|>)\s*90|坐矮凳|翘二郎腿|盘腿打坐|大幅度内旋"),
    ("禁忌脊柱牵引", r"(?:大重量|暴力|颈椎|腰椎|脊柱)牵引"),
    ("冲突药物继续同服", r"(?:继续|同时|联合|一起|共同)(?:同服|服用|合用|叠加|吃)|继续同服|同时服用|联合服用|继续叠加|一起吃"),
)

_DRAFT_PROHIBITION_LINE_HEADER_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?(?:绝对禁忌操作|绝对禁忌|禁忌清单|禁忌操作|禁忌项目|安全红线|严禁以下)"
)
_DRAFT_PRE_PROHIBITION_RE = re.compile(
    r"(?:严禁|禁止|切勿|切忌|绝不可|绝不能|不得|不可以|不可|不能|不要|不建议|不推荐|不予|不宜|避免|禁忌|暂缓|停止|暂停|杜绝|严防|勿|忌)"
)
_DRAFT_POST_PROHIBITION_RE = re.compile(
    r"(?:是不?可取的|是不允许的|均属禁忌|属禁忌|均禁止|均严禁|不可以|不行|不可|不得|禁止|严禁|禁忌|应避免|需避免|要避免|切勿尝试)"
)
_DRAFT_AFFIRMATIVE_RE = re.compile(
    r"(?<![不未勿禁绝莫别免])(?:"
    r"可行|可以(?!导致|引起|诱发|造成|加重|引发|致|使)"
    r"|可做|可尝试|可适当|建议|安排|推荐|予以|去做|试试|直接做|直接行"
    r"|继续|同服|一起吃|就行|即可|就好|也行|没问题|尽管|不妨|照常|没关系"
    r"|松一下|来一遍|加上去|能做|能行|准予|批准|允许"
    r"|可(?!能|导致|引起|诱发|造成|加重|引发|致|使|增|见|知|取|谓|惜|怕|疑|靠|怜|叹|爱|笑|恨|悲|喜|恶)"
    r")"
)
_DRAFT_RIGHT_AFFIRMATIVE_RE = re.compile(
    r"(?:可行|可以(?!导致|引起|诱发|造成|加重|引发|致|使)|就行|即可|就好|也行|无妨|试试|没问题|没关系|照常|松一下|来一遍|加上去)"
)
_DRAFT_DOUBLE_NEG_PROHIBITION_RE = re.compile(
    r"(?:并非|不是|无需|不必|谈不上|没有)[^，。；\n]{0,6}?(?:禁止|严禁|禁忌|不可|不能)"
)
_DRAFT_NEG_BREAK_RE = re.compile(r"(?:但是|但|然而|不过|而是|改为|改做|改用)")
_RISK_CONSEQUENCE_WARNING_RE = re.compile(
    r"(?:导致|引起|诱发|造成|加重|引发|脱落|血栓脱落|肺栓塞|脊髓损伤|截瘫|骨折|出血|血肿|脱位|危险|风险)"
)
_DRAFT_SECTION_HEADING_RE = re.compile(r"^\s*#{2,6}\s+(.*)$", flags=re.MULTILINE)
_DRAFT_SKIP_SECTION_TITLE_RE = re.compile(
    r"(?:安全红旗|用药冲突硬约束|确定性安全|安全边界与绝对禁忌|绝对禁忌|当前风险提示与绝对禁忌|"
    r"确定性量表|量表评估结果|当前功能与疼痛评估|跨诊次时间线|病例核心摘要|四诊与专科查体|四诊/影像要点)"
)
_DRAFT_CITATION_LINE_RE = re.compile(
    r"^\s*(?:"
    r"[-*]\s*\[(?:RF_\d+|DC_\d+|量表联动[^\]]+)\]"
    r"|[-*]\s*(?:\*\*)?(?:命中依据|规则依据|临床警示|绝对禁忌操作|处置建议|联动警示|联动提示)\b"
    r"|[-*]\s*\*\*(?:[0-9]{4}[-/.年]|术后|昨日|今日|半年前|\d+个?[天周月年]前|既往|本次|昨晚|个人记录)[^*]*\*\*\s*[:：]"
    r")"
)
_DRAFT_PATIENT_REPORT_PREFIX_RE = re.compile(
    r"^(?:[-*]\s*)?(?:患者|家属|本人)?(?:自诉|主诉|诉|曾询问|询问|今日要求|要求|欲|想提前|打算|准备|既往|网上看|朋友推荐|非要我去|上次|当时医嘱)"
)
_DRAFT_HISTORICAL_EFFECTIVE_RE = re.compile(r"既往[^，。；\n]{0,16}有效")
_CONTRA_PREFIX_NOISE_RE = re.compile(
    r"^(?:未排除感染前的|未经手术医师许可(?:提前)?|未经影像学确诊的|非专业环境|术后早期|早期|局部|患肢|大面积多片|盲目|暴力|强行)"
)
_CONTRA_GENERIC_STOPWORDS: FrozenSet[str] = frozenset(
    {"训练", "动作", "运动", "治疗", "药物", "用药", "外敷", "观察", "颈椎", "腰椎", "胸椎", "脊柱", "关节", "髋关节", "膝关节", "患肢", "局部"}
)


def _extract_alert_action_terms(alerts: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Collect (rule_id, rule_name, regex_pattern) tuples from active alerts and high-risk action library."""
    items: List[Tuple[str, str, str]] = []
    seen_patterns = set()

    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        rule_id = str(alert.get("rule_id") or "ALERT")
        rule_name = str(alert.get("name") or "")
        for raw_action in alert.get("contraindicated_actions") or []:
            action_text = str(raw_action or "").strip()
            if not action_text:
                continue
            for part in re.split(r"[（()）/\s、，,;；或及与]+", action_text):
                token = _CONTRA_PREFIX_NOISE_RE.sub("", part.strip()).strip()
                if len(token) >= 2 and token not in _CONTRA_GENERIC_STOPWORDS:
                    pat = re.escape(token)
                    if (rule_id, pat) not in seen_patterns:
                        seen_patterns.add((rule_id, pat))
                        items.append((rule_id, rule_name, pat))

    active_ids = ", ".join(
        str(a.get("rule_id") or "") for a in alerts if isinstance(a, dict) and a.get("rule_id")
    ) or "ACTIVE_ALERT"
    for label, pat in _HIGH_RISK_DRAFT_ACTION_PATTERNS:
        items.append((active_ids, label, pat))
    return items


def _iter_checkable_draft_lines(draft_text: str) -> List[str]:
    """Filter out deterministic constraint/timeline/citation blocks when draft has Markdown sections."""
    has_sections = bool(_DRAFT_SECTION_HEADING_RE.search(draft_text))
    if not has_sections:
        return [ln for ln in draft_text.splitlines() if ln.strip()]

    checkable: List[str] = []
    in_preamble = True
    in_blockquote_block = False
    skip_section = False

    for line in draft_text.splitlines():
        stripped = line.strip()
        if not stripped:
            in_blockquote_block = False
            continue

        sec_m = _DRAFT_SECTION_HEADING_RE.match(stripped)
        if sec_m:
            in_preamble = False
            in_blockquote_block = False
            skip_section = bool(_DRAFT_SKIP_SECTION_TITLE_RE.search(sec_m.group(1)))
            continue

        if in_preamble:
            if stripped.startswith("# "):
                in_blockquote_block = False
                continue
            if stripped.startswith(">"):
                in_blockquote_block = True
                continue
            if in_blockquote_block:
                continue

        if skip_section:
            continue

        if _DRAFT_CITATION_LINE_RE.match(stripped):
            continue

        checkable.append(line)
    return checkable


def _sentence_has_risk_warning_exemption(sent: str) -> bool:
    """Return True if sentence states a risk consequence warning without affirmative recommendation."""
    if not _RISK_CONSEQUENCE_WARNING_RE.search(sent):
        return False

    for adv_part in re.split(r"(?:但是|但|然而|不过)", sent):
        if not adv_part.strip():
            continue
        for sub in re.split(r"[；;，,]+", adv_part):
            s = sub.strip()
            if not s:
                continue
            if _DRAFT_PRE_PROHIBITION_RE.search(s) or _DRAFT_POST_PROHIBITION_RE.search(s):
                continue
            for aff_m in _DRAFT_AFFIRMATIVE_RE.finditer(s):
                right_ctx = s[aff_m.end() :]
                # Ignore causal/possibility phrasing such as "可以使血栓脱落" or "可能导致脊髓损伤"
                if re.match(r"^\s*(?:导致|引起|诱发|造成|加重|引发|放大|增加|使)", right_ctx):
                    continue
                return False
    return True


def _is_patient_complaint_restatement(sent: str) -> bool:
    """Return True when a sentence only restates patient request or historical efficacy without clinician approval."""
    s = sent.strip()
    if not s:
        return False
    if not (_DRAFT_PATIENT_REPORT_PREFIX_RE.search(s) or _DRAFT_HISTORICAL_EFFECTIVE_RE.search(s)):
        return False
    if re.search(r"[？?]\s*$", s):
        return True
    for adv_part in re.split(r"(?:但是|但|然而|不过)", s):
        for sub in re.split(r"[；;，,]+", adv_part):
            if _DRAFT_PRE_PROHIBITION_RE.search(sub):
                continue
            if _DRAFT_AFFIRMATIVE_RE.search(sub):
                return False
    return True


_DRAFT_TRAILING_APPROVAL_RE = re.compile(
    r"^(?:那么|那|则|即|当前|今日|今天|目前|也|都|还是)?\s*"
    r"(?:可行|可以(?:安排|试|试试|做|行)?|安排(?:一下)?|建议(?:可行|安排|尝试|做)?|推荐|予以|就行|即可|就好|也行|没问题|没关系|试试|照常|松一下|来一遍|加上去)[。！!]?$"
)


def _analyze_draft_subclause(
    subclause: str,
    match_start: int,
    match_end: int,
    *,
    allow_prohibition_line_header: bool = False,
    next_subclause: str = "",
) -> bool:
    """Return True iff the matched action is positively permitted/recommended and NOT negated."""
    left = subclause[:match_start]
    right = subclause[match_end:]
    matched_text = subclause[match_start:match_end]

    is_double_neg = bool(_DRAFT_DOUBLE_NEG_PROHIBITION_RE.search(left))
    if not is_double_neg:
        neg_matches = list(_DRAFT_PRE_PROHIBITION_RE.finditer(left))
        if neg_matches:
            last_neg = neg_matches[-1]
            between = left[last_neg.end() :]
            if not _DRAFT_NEG_BREAK_RE.search(between):
                return False

        if _DRAFT_POST_PROHIBITION_RE.search(right):
            return False

        if allow_prohibition_line_header and not _DRAFT_AFFIRMATIVE_RE.search(left):
            return False

    # Principle 3: positive permission/recommendation context must govern the matched action
    if _DRAFT_AFFIRMATIVE_RE.search(left) or _DRAFT_RIGHT_AFFIRMATIVE_RE.search(right):
        return True

    # Self-affirmative compound drug phrases (e.g., "继续同服", "一起吃") when not negated or questioned
    if re.search(r"(?:继续同服|同服|一起吃|照常吃)", matched_text) and not re.search(r"[？?]\s*$", subclause):
        return True

    # Adjacent pure-approval follow-up clause (e.g., "患者要求做颈椎旋转扳法，可以安排。")
    if next_subclause and _DRAFT_TRAILING_APPROVAL_RE.match(next_subclause.strip()):
        return True

    return False


def verify_draft_against_alerts(
    draft_text: str,
    alerts: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Deterministic post-generation draft verification against active safety alerts.

    Enforces three principles:
    1. Only inspect self-authored recommendation/education segments (skip deterministic constraint,
       timeline, citation blocks, and pure patient-request restatements when sectioned).
    2. Exempt risk-consequence warning sentences that carry no affirmative recommendation.
    3. Flag violations only when an affirmative permission/recommendation context governs the
       contraindicated action without negation.
    """
    active_alerts = [a for a in (alerts or []) if isinstance(a, dict)]
    if not active_alerts or not (draft_text or "").strip():
        return {"passed": True, "blocked": False, "violations": [], "message": ""}

    action_specs = _extract_alert_action_terms(active_alerts)
    violations: List[Dict[str, str]] = []
    seen_violation_keys = set()

    for line in _iter_checkable_draft_lines(draft_text):
        has_header = bool(_DRAFT_PROHIBITION_LINE_HEADER_RE.search(line))
        # Preserve trailing punctuation per sentence so interrogatives / warnings remain intact
        sentences = [s for s in re.findall(r"[^。！？\n]+[。！？]*", line) if s.strip()]
        for sent_idx, sent in enumerate(sentences):
            if _sentence_has_risk_warning_exemption(sent):
                continue
            if _is_patient_complaint_restatement(sent):
                continue

            sent_body = re.sub(r"[。！？]+$", "", sent)
            adv_parts = re.split(r"(?:但是|但|然而|不过)", sent_body)
            for adv_idx, adv_part in enumerate(adv_parts):
                if not adv_part.strip():
                    continue
                allow_header_here = has_header and sent_idx == 0 and adv_idx == 0
                subclauses = re.split(r"[；;，,]+", adv_part)
                for sub_idx, sub in enumerate(subclauses):
                    if not sub.strip():
                        continue
                    next_sub = subclauses[sub_idx + 1] if sub_idx + 1 < len(subclauses) else ""
                    for rule_id, rule_label, pat in action_specs:
                        for m in re.finditer(pat, sub, flags=re.IGNORECASE):
                            if not _analyze_draft_subclause(
                                sub,
                                m.start(),
                                m.end(),
                                allow_prohibition_line_header=allow_header_here,
                                next_subclause=next_sub,
                            ):
                                continue
                            matched_action = m.group(0)
                            v_key = (sub.strip(), matched_action)
                            if v_key in seen_violation_keys:
                                continue
                            seen_violation_keys.add(v_key)
                            violations.append(
                                {
                                    "rule_id": rule_id,
                                    "rule_name": rule_label,
                                    "matched_action": matched_action,
                                    "snippet": sub.strip()[:120],
                                    "reason": f"草稿疑似非否定放行禁忌操作 '{matched_action}'（与激活告警 {rule_id} 冲突）",
                                }
                            )

    if violations:
        summary = "；".join(v["reason"] for v in violations[:3])
        return {
            "passed": False,
            "blocked": True,
            "violations": violations,
            "message": f"安全拦截：生成草稿违背确定性安全禁忌，已阻止渲染。{summary}",
        }
    return {"passed": True, "blocked": False, "violations": [], "message": ""}


def collect_unified_alerts(
    risk_report: Dict[str, Any],
    scale_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate rule-scanner alerts and deterministic scale linkage red flags."""
    unified: List[Dict[str, Any]] = []
    for alert in risk_report.get("alerts", []) or []:
        item = dict(alert)
        item.setdefault("source_label", "双轨规则扫描")
        unified.append(item)

    rf_map = {r.rule_id: r for r in load_red_flags_catalog().rules}
    for res in scale_results:
        if res.get("status") != "ok":
            continue
        is_red_flag = bool(res.get("red_flag_triggered"))
        is_caution = bool(res.get("caution_triggered"))
        if not is_red_flag and not is_caution:
            continue

        linked_ids = list(res.get("linked_rule_ids") or [])
        linked_rule = rf_map.get(linked_ids[0]) if linked_ids else None
        rule_id_label = ", ".join(linked_ids) if linked_ids else f"SCALE_{res.get('scale_id', '')}"
        if is_red_flag:
            unified.append(
                {
                    "rule_id": rule_id_label,
                    "rule_type": "scale_linkage",
                    "name": f"量表阈值联动红旗：{res.get('scale_name', res.get('scale_id'))}",
                    "category": "量表与红旗联动预警",
                    "severity": "critical",
                    "source": "scale_linkage",
                    "source_label": f"确定性量表联动 ({res.get('scale_id')})",
                    "matched_evidence": [
                        f"{res.get('scale_id')} 总分 = {res.get('total_score')} 分（{res.get('severity_band', '')}）"
                    ],
                    "clinical_action": linked_rule.recommended_action if linked_rule else "STOP_AND_REFER",
                    "clinical_alert": res.get("red_flag_reason") or (linked_rule.clinical_alert if linked_rule else ""),
                    "contraindicated_actions": list(linked_rule.contraindicated_actions) if linked_rule else [],
                    "reference": list(res.get("references") or [])
                    + (list(linked_rule.references) if linked_rule else []),
                }
            )
        else:
            unified.append(
                {
                    "rule_id": rule_id_label,
                    "rule_type": "scale_caution_linkage",
                    "name": f"量表轻度异常联动提示：{res.get('scale_name', res.get('scale_id'))}",
                    "category": "量表与红旗联动提示",
                    "severity": "medium",
                    "source": "scale_linkage",
                    "source_label": f"确定性量表轻度提示联动 ({res.get('scale_id')})",
                    "matched_evidence": [
                        f"{res.get('scale_id')} 总分 = {res.get('total_score')} 分（{res.get('severity_band', '')}）"
                    ],
                    "clinical_action": "CAUTION_AND_PROTECT",
                    "clinical_alert": res.get("caution_reason") or (linked_rule.clinical_alert if linked_rule else ""),
                    "contraindicated_actions": list(linked_rule.contraindicated_actions) if linked_rule else [],
                    "reference": list(res.get("references") or [])
                    + (list(linked_rule.references) if linked_rule else []),
                }
            )
    return unified


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
    """Execute the full 4-step deterministic + LLM copilot pipeline without UI dependencies."""
    cleaned_text = (raw_text or "").strip()
    if not cleaned_text:
        raise ValueError("输入病历文本不能为空。")
    if draft_mode not in DRAFT_MODE_LABELS:
        raise ValueError(f"不支持的文书模式: {draft_mode}")

    active_client = client or LLMClient()

    # Step 1: Structured clinical extraction validated by CopilotExtractedNote
    extraction_messages = build_extraction_messages(cleaned_text)
    extract_res = active_client.extract_structured_json(
        extraction_messages,
        schema_model=CopilotExtractedNote,
        model=model,
        mock_mode=mock_mode,
    )
    if extract_res.get("status") != "ok" or not isinstance(extract_res.get("data"), dict):
        raise RuntimeError(extract_res.get("message") or "病历结构化抽取失败。")
    extracted_note: Dict[str, Any] = extract_res["data"]

    # Step 2: Deterministic dual-track safety scan (merging timeline snapshot into Track B)
    merged_facts = merge_snapshot_into_structured_facts(
        extracted_note.get("current_state_snapshot"),
        fallback_facts=extracted_note.get("structured_facts"),
    )
    risk_report = scan_clinical_risks(
        raw_text=cleaned_text,
        structured_facts=merged_facts,
    )

    # Step 3: Deterministic scale calculation (never defaults missing items to 0)
    scale_results = evaluate_scale_entries(
        extracted_note.get("scale_params", []),
        scales_path=scales_path,
    )

    # Step 4: High-constraint draft generation
    draft_result = generate_draft_document(
        draft_mode,
        extracted_note,
        risk_report,
        scale_results,
        model=model,
        mock_mode=mock_mode,
        client=active_client,
    )

    unified_alerts = collect_unified_alerts(risk_report, scale_results)
    return {
        "raw_text": cleaned_text,
        "extracted_note": extracted_note,
        "risk_report": risk_report,
        "scale_results": scale_results,
        "unified_alerts": unified_alerts,
        "draft": draft_result,
        "draft_mode": draft_mode,
        "draft_stale": False,
        "trace": {
            "extraction_model": extract_res.get("model"),
            "extraction_mock": extract_res.get("mock"),
            "extraction_raw_content": extract_res.get("raw_content"),
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
    new_state["unified_alerts"] = collect_unified_alerts(
        new_state.get("risk_report", {}),
        scale_results,
    )

    target_mode = draft_mode or new_state.get("draft_mode", "ward_round")
    new_state["draft_mode"] = target_mode

    if refresh_draft:
        draft_result = generate_draft_document(
            target_mode,
            new_state["extracted_note"],
            new_state.get("risk_report", {}),
            scale_results,
            model=model,
            mock_mode=mock_mode,
            client=client,
        )
        new_state["draft"] = draft_result
        new_state["draft_stale"] = False
        new_state.setdefault("trace", {})["draft_model"] = draft_result.get("model")
        new_state["trace"]["draft_mock"] = draft_result.get("mock")
        new_state["trace"]["draft_usage"] = draft_result.get("usage", {})
    else:
        new_state["draft_stale"] = True

    return new_state


# ==============================================================================
# Streamlit UI Rendering Helpers
# ==============================================================================


def _apply_demo_case(case_id: str) -> None:
    case = DEMO_CASES[case_id]
    st.session_state["raw_note_input"] = case["text"]
    st.session_state["active_demo_id"] = case_id
    st.session_state["draft_mode_selector"] = case.get("default_mode", "ward_round")
    st.session_state["copilot_state"] = None


def _clear_workspace() -> None:
    st.session_state["raw_note_input"] = ""
    st.session_state["active_demo_id"] = None
    st.session_state["copilot_state"] = None


def _render_sidebar(default_config: LLMConfig) -> tuple[str, bool]:
    with st.sidebar:
        st.header("⚙️ 副驾配置与演示数据")
        st.warning(
            "⚠️ **合规安全边界**\n\n"
            "仅限**脱敏病历**与临床思维辅助，**不直接对患者下诊断或开具处方**。"
            "严禁输入真实姓名、住院号、身份证号等患者隐私标识。"
        )

        st.subheader("模型与运行模式")
        default_model = default_config.default_model
        model_options = PRESET_MODELS + ["自定义模型..."]
        default_idx = (
            PRESET_MODELS.index(default_model)
            if default_model in PRESET_MODELS
            else len(model_options) - 1
        )
        selected_preset = st.selectbox(
            "选择 LiteLLM 模型",
            options=model_options,
            index=default_idx,
            key="model_preset_select",
        )
        if selected_preset == "自定义模型...":
            custom_default = default_model if default_model not in PRESET_MODELS else ""
            chosen_model = (
                st.text_input(
                    "自定义 LiteLLM 模型名",
                    value=custom_default,
                    placeholder="例如: ollama/qwen2.5:14b",
                    key="custom_model_input",
                ).strip()
                or default_model
            )
        else:
            chosen_model = selected_preset

        mock_mode = st.toggle(
            "离线 Mock 模式 (LLM_MOCK_MODE)",
            value=default_config.mock_mode,
            key="mock_mode_toggle",
            help="开启后无需 API Key 即可跑通完整抽取、双轨安全扫描、确定性算分与草稿流程。",
        )
        if mock_mode:
            st.info("🧪 当前处于本地 Mock 模式，所有确定性规则与算分均真实执行。")

        st.divider()
        st.subheader("🧪 一键填充演示病例（纯虚构脱敏）")
        for case_id, case_info in DEMO_CASES.items():
            st.button(
                case_info["button_label"],
                key=f"demo_btn_{case_id}",
                width="stretch",
                on_click=_apply_demo_case,
                args=(case_id,),
                help=case_info["description"],
            )

    return chosen_model, mock_mode


def _render_top_safety_banner(state: Dict[str, Any]) -> None:
    unified_alerts = state.get("unified_alerts") or []
    risk_report = state.get("risk_report") or {}
    suppressed = risk_report.get("suppressed_matches") or []

    if unified_alerts:
        has_high_or_critical = any(str(a.get("severity", "high")) in ("critical", "high") for a in unified_alerts)
        if has_high_or_critical:
            st.error(
                f"🚨 **高危临床警示**：共触发 **{len(unified_alerts)}** 项红旗 / 用药冲突 / 量表联动告警，"
                "请在任何诊疗或康复建议前先核查安全红线！"
            )
        else:
            st.warning(
                f"⚠️ **临床安全提示**：共触发 **{len(unified_alerts)}** 项提示级联动约束，"
                "请留意禁忌操作并动态随访！"
            )
        for alert in unified_alerts:
            severity = str(alert.get("severity", "high"))
            badge = SEVERITY_BADGE_MAP.get(severity, severity.upper())
            rule_id = alert.get("rule_id", "UNKNOWN")
            title = alert.get("name", "")
            review_tags: List[str] = []
            if alert.get("needs_review"):
                review_tags.append("⚠️ 需人工复核 (needs_review)")
            if alert.get("conflict_with_text"):
                review_tags.append("⚠️ 结构化与原文冲突 (conflict_with_text)")
            tag_suffix = f" | {' · '.join(review_tags)}" if review_tags else ""
            with st.container(border=True):
                header_msg = (
                    f"**[{rule_id}] {title}** | 严重度：`{badge}` | 来源：`{alert.get('source_label', alert.get('source', ''))}`{tag_suffix}"
                )
                if severity in ("critical", "high"):
                    st.error(header_msg)
                else:
                    st.warning(header_msg)
                if review_tags:
                    conflict_detail = alert.get("conflict_detail") or alert.get("review_reason") or "模型结构化事实与病历原文证据存在潜在冲突或边界否定抑制，请临床医师人工核实原文！"
                    st.warning(f"🔎 **人工复核提示**：{conflict_detail}")
                evidence = "；".join(str(x) for x in alert.get("matched_evidence") or [])
                st.markdown(f"- **命中证据 (`matched_evidence`)**：{evidence or '见结构化事实'}")
                st.markdown(f"- **建议动作 (`clinical_action`)**：`{alert.get('clinical_action', '')}`")
                st.markdown(f"- **临床警示 (`clinical_alert`)**：{alert.get('clinical_alert', '')}")
                if alert.get("contraindicated_actions"):
                    contra = "；".join(str(x) for x in alert["contraindicated_actions"])
                    st.markdown(f"- **绝对禁忌操作**：{contra}")
                if alert.get("suggested_handling"):
                    st.markdown(f"- **处置建议**：{alert['suggested_handling']}")
                if alert.get("risk_mechanism"):
                    st.markdown(f"- **风险机制**：{alert['risk_mechanism']}")
                refs = "；".join(str(x) for x in alert.get("reference") or [])
                st.caption(f"📚 教材 / 指南依据：{refs or '未注明'}")
    else:
        st.success("✅ **安全扫描通过**：确定性双轨扫描与量表阈值未命中任何红旗征或高危用药冲突。")

    with st.expander(
        f"🛡️ 误报拦截审计记录（否定词与洗脱期过滤 `suppressed_matches`: {len(suppressed)} 条）",
        expanded=False,
    ):
        if suppressed:
            rows = [
                {
                    "规则ID": s.get("rule_id", ""),
                    "类型": s.get("rule_type", ""),
                    "命中词": s.get("term", ""),
                    "拦截原因": s.get("reason", ""),
                    "来源": s.get("source", ""),
                    "详情": s.get("detail", ""),
                }
                for s in suppressed
            ]
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.caption("本次扫描无被否定词或药物洗脱期拦截的候选条目。")


def _render_tab_structured_note(state: Dict[str, Any]) -> None:
    note = state.get("extracted_note") or {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 基本画像与主诉")
        st.markdown(f"- **脱敏画像**：{note.get('patient_summary', '未提及')}")
        st.markdown(f"- **主诉**：{note.get('chief_complaint', '未提及')}")
        st.markdown(f"- **现病史摘要**：{note.get('history_of_present_illness', '未提及')}")
        st.markdown("#### 影像学 / 检验摘要")
        st.info(note.get("imaging_summary", "未提及"))
    with col2:
        st.markdown("#### 中医骨伤四诊要点")
        tcm = note.get("tcm_four_diagnostics") or {}
        if tcm:
            for k, v in tcm.items():
                label = TCM_FIELD_LABELS.get(k, k)
                st.markdown(f"- **{label}**：{v}")
        else:
            st.caption("未提取到中医四诊信息。")
        st.markdown("#### 专科查体要点")
        exams = note.get("physical_exam") or []
        if exams:
            for ex in exams:
                st.markdown(f"- {ex}")
        else:
            st.caption("暂无专科查体记录。")

    st.divider()
    st.markdown("#### 🕒 跨诊次时间线演变 (`timeline_events`)")
    timeline = note.get("timeline_events") or []
    if timeline:
        for idx, ev in enumerate(timeline, start=1):
            date_str = ev.get("date", "未标明时间")
            dept_str = ev.get("department") or "未标明"
            event_str = ev.get("event", "")
            key_change = (ev.get("key_change") or "").strip()
            with st.container(border=True):
                st.markdown(f"**{idx}. `{date_str}`** · 科室/场景：`{dept_str}`")
                st.markdown(event_str)
                if key_change:
                    st.caption(f"🔀 关键变化 / 状态转折：{key_change}")
    else:
        st.caption("暂无明确时间线事件。")

    st.divider()
    st.markdown("#### 📌 当前状态快照 (`current_state_snapshot`) — 时间线冲突消解结果")
    snapshot = note.get("current_state_snapshot") or {}
    override_note = snapshot.get("diagnosis_override_note")
    if override_note:
        st.info(f"⚡ **最新状态推翻 / 转折提示 (`diagnosis_override_note`)**：{override_note}")

    snap_c1, snap_c2 = st.columns(2)
    with snap_c1:
        st.markdown("**💊 当前在用 / 未满洗脱期药物 (`active_medications`)**")
        active_meds = snapshot.get("active_medications") or []
        if active_meds:
            st.dataframe(
                [
                    {
                        "药物": m.get("name", ""),
                        "状态": m.get("status", ""),
                        "停药天数": (
                            f"{m['stopped_days_ago']:g} 天（洗脱期内）"
                            if m.get("stopped_days_ago") is not None
                            else ("已停用（天数未明，按洗脱期内处理）" if m.get("status") == "stopped" else "在用 / 未停药")
                        ),
                    }
                    for m in active_meds
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("当前无高风险在用或洗脱期内药物。")

    with snap_c2:
        st.markdown("**🛑 已停用且超洗脱期药物 (`stopped_medications`)**")
        stopped_meds = snapshot.get("stopped_medications") or []
        if stopped_meds:
            st.dataframe(
                [
                    {
                        "药物": m.get("name", ""),
                        "状态": m.get("status", "stopped"),
                        "已停药天数": (
                            f"{m['stopped_days_ago']:g} 天"
                            if m.get("stopped_days_ago") is not None
                            else "已停用"
                        ),
                    }
                    for m in stopped_meds
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("暂无已停用且过洗脱期的历史药物。")

    surg_history = snapshot.get("active_surgical_history") or []
    if surg_history:
        st.markdown("**🦴 长期手术 / 内固定 / 病史约束 (`active_surgical_history`)**")
        for sh in surg_history:
            st.markdown(f"- `{sh}`")

    superseded_list = snapshot.get("superseded_conditions") or []
    if superseded_list:
        st.markdown("**🔄 已被最新诊次推翻 / 已过保护期的旧状态 (`superseded_conditions`)**")
        st.dataframe(
            [
                {
                    "过期旧状态 (old_state)": item.get("old_state", ""),
                    "最新状态 (new_state)": item.get("new_state", ""),
                    "解除规则提示 (rule_hint)": item.get("rule_hint") or "—",
                }
                for item in superseded_list
            ],
            width="stretch",
            hide_index=True,
        )

    st.divider()
    facts = note.get("structured_facts") or {}
    c_sym, c_med = st.columns(2)
    with c_sym:
        st.markdown("#### 结构化症状极性 (`symptoms`)")
        syms = facts.get("symptoms") or []
        if syms:
            st.dataframe(
                [
                    {"症状/体征": s.get("name", ""), "极性 (polarity)": s.get("polarity", "")}
                    for s in syms
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("未识别到结构化红旗相关症状。")
    with c_med:
        st.markdown("#### 结构化用药状态 (`medications`)")
        meds = facts.get("medications") or []
        if meds:
            st.dataframe(
                [
                    {
                        "药物/制剂": m.get("name", ""),
                        "状态 (status)": m.get("status", ""),
                        "已停药天数": (
                            f"{m['stopped_days_ago']:g} 天"
                            if m.get("stopped_days_ago") is not None
                            else "—"
                        ),
                    }
                    for m in meds
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("未识别到结构化药物记录。")


def _render_scale_manual_inputs(
    spec: Any,
    existing_params: Dict[str, Any],
    key_prefix: str,
) -> Dict[str, Any]:
    """Render dynamic manual input widgets for all items of a scale without defaulting missing to 0."""
    collected: Dict[str, Any] = {}
    cols = st.columns(2)
    for idx, item in enumerate(spec.items):
        col = cols[idx % 2]
        raw_val = existing_params.get(item.item_id)
        item_key = f"{key_prefix}_{item.item_id}"
        with col:
            if item.allowed_values is not None:
                options = [None] + [float(v) for v in item.allowed_values]
                current = float(raw_val) if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool) else None
                default_opt_idx = options.index(current) if current in options else 0
                val = st.selectbox(
                    f"{item.label} (`{item.item_id}`)",
                    options=options,
                    index=default_opt_idx,
                    format_func=lambda v: "⚠️ 未填写（请补录）" if v is None else f"{v:g} 分",
                    key=item_key,
                )
            else:
                step = float(item.step or 1.0)
                min_v = float(item.min_value)
                max_v = float(item.max_value)
                default_num: Optional[float] = None
                if (
                    isinstance(raw_val, (int, float))
                    and not isinstance(raw_val, bool)
                    and min_v <= float(raw_val) <= max_v
                ):
                    default_num = float(raw_val)
                val = st.number_input(
                    f"{item.label} (`{item.item_id}` | {min_v:g}~{max_v:g})",
                    min_value=min_v,
                    max_value=max_v,
                    value=default_num,
                    step=step,
                    placeholder="留空表示未填写（严禁默认补0）",
                    key=item_key,
                )
            if val is not None:
                collected[item.item_id] = float(val)

    if spec.improvement_rate_formula:
        raw_pre = existing_params.get("pre_score")
        default_pre: Optional[float] = None
        if (
            isinstance(raw_pre, (int, float))
            and not isinstance(raw_pre, bool)
            and float(spec.min_total) <= float(raw_pre) <= float(spec.max_total)
        ):
            default_pre = float(raw_pre)
        pre_val = st.number_input(
            f"可选：术前基线总分 (`pre_score` | 用于计算改善率，{spec.min_total:g}~{spec.max_total:g})",
            min_value=float(spec.min_total),
            max_value=float(spec.max_total),
            value=default_pre,
            step=1.0,
            placeholder="仅需计算改善率时填写",
            key=f"{key_prefix}_pre_score",
        )
        if pre_val is not None:
            collected["pre_score"] = float(pre_val)

    return collected


def _render_tab_scales(
    state: Dict[str, Any],
    *,
    chosen_model: str,
    mock_mode: bool,
) -> None:
    catalog = load_scales_catalog()
    spec_map = {s.scale_id: s for s in catalog.scales}
    scale_results = state.get("scale_results") or []
    run_id = st.session_state.get("analysis_run_id", 0)

    if state.get("draft_stale"):
        st.info("ℹ️ 量表结果已更新，Tab 3 文书草稿尚未同步刷新。您可点击下方按钮立即刷新草稿。")
        if st.button("🔄 依据最新量表结果刷新 Tab 3 文书草稿", key=f"sync_draft_from_tab2_{run_id}"):
            try:
                with st.spinner("正在依据最新量表与红旗结果重新生成文书草稿..."):
                    draft_res = generate_draft_document(
                        state.get("draft_mode", "ward_round"),
                        state["extracted_note"],
                        state["risk_report"],
                        state["scale_results"],
                        model=chosen_model,
                        mock_mode=mock_mode,
                    )
                    state["draft"] = draft_res
                    state["draft_stale"] = False
                    state.setdefault("trace", {})["draft_model"] = draft_res.get("model")
                    state["trace"]["draft_mock"] = draft_res.get("mock")
                    st.session_state["copilot_state"] = state
                st.rerun()
            except Exception as exc:
                st.error(f"刷新文书草稿失败：{sanitize_error_message(exc)}")

    if not scale_results:
        st.info("病历文本中未检测到明确量表条目分值，可在下方“手动新增量表评估”选择量表计算。")

    for idx, res in enumerate(scale_results):
        scale_id = res.get("scale_id", "")
        spec = spec_map.get(scale_id)
        direction_text = describe_score_direction(res.get("score_direction"))

        with st.container(border=True):
            if res.get("status") == "ok":
                st.subheader(f"📊 {res.get('scale_name', scale_id)} (`{scale_id}`)")
                m1, m2, m3 = st.columns(3)
                m1.metric("确定性总分", f"{res.get('total_score')} / {res.get('max_total'):g}")
                m2.metric("严重程度分级", str(res.get("severity_band", "")))
                if "normalized_score_pct" in res:
                    m3.metric("0-100 标准化分", f"{res['normalized_score_pct']}%")
                else:
                    m3.metric("理论分值范围", f"{res.get('min_total'):g} ~ {res.get('max_total'):g}")

                st.caption(f"🧭 **风险方向说明**：{direction_text}")

                if res.get("subscale_scores"):
                    sub_rows = [
                        {
                            "维度": sub.get("label", sid),
                            "得分": sub.get("score"),
                            "满分": sub.get("max_score"),
                            "标准化百分比": f"{sub['normalized_pct']}%" if "normalized_pct" in sub else "—",
                        }
                        for sid, sub in res["subscale_scores"].items()
                    ]
                    st.dataframe(sub_rows, width="stretch", hide_index=True)

                if res.get("improvement_rate"):
                    ir = res["improvement_rate"]
                    if ir.get("status") == "ok":
                        st.success(
                            f"📈 平林法 JOA 改善率：**{ir['improvement_rate_pct']}%**（{ir.get('efficacy_grade', '')}）"
                            f" | 公式：`{ir.get('formula', '')}`"
                        )
                    else:
                        st.warning(f"⚠️ 改善率计算提示：{ir.get('message', '')}")

                if res.get("red_flag_triggered"):
                    st.error(f"🚨 **量表联动红旗**：{res.get('red_flag_reason', '')}")

                if spec is not None:
                    with st.expander("✏️ 核对或修改条目分值并重算", expanded=False):
                        updated_params = _render_scale_manual_inputs(
                            spec,
                            res.get("input_params") or res.get("item_scores") or {},
                            key_prefix=f"edit_scale_{run_id}_{idx}_{scale_id}",
                        )
                        if st.button("重新计算量表", key=f"recalc_ok_{run_id}_{idx}_{scale_id}"):
                            new_state = recalculate_scale_in_state(
                                state,
                                scale_id,
                                updated_params,
                                scale_index=idx,
                                refresh_draft=False,
                            )
                            st.session_state["copilot_state"] = new_state
                            st.rerun()
            else:
                st.subheader(f"⚠️ {res.get('scale_name', scale_id)} (`{scale_id}`) — 待补录或修正")
                st.caption(f"🧭 **风险方向说明**：{direction_text}")
                missing = res.get("missing_fields") or []
                invalid = res.get("invalid_fields") or []
                warn_lines = [
                    f"**{res.get('scale_name', scale_id)} (`{scale_id}`) 未完成确定性算分：{res.get('message', '')}**"
                ]
                if missing and spec is not None:
                    label_map = {it.item_id: it.label for it in spec.items}
                    missing_desc = "、".join(f"`{fid}` ({label_map.get(fid, fid)})" for fid in missing)
                    warn_lines.append(f"- **缺失必填字段 ({len(missing)} 项)**：{missing_desc}")
                if invalid:
                    invalid_desc = "；".join(f"`{f.get('field')}`: {f.get('reason')}" for f in invalid)
                    warn_lines.append(f"- **越界或非法字段 ({len(invalid)} 项)**：{invalid_desc}")
                st.warning("\n".join(warn_lines))

                if spec is not None:
                    st.markdown("**请手动补录以下条目分值（未填写条目将保持缺失校验，绝不默认填 0）：**")
                    updated_params = _render_scale_manual_inputs(
                        spec,
                        res.get("input_params") or {},
                        key_prefix=f"fix_scale_{run_id}_{idx}_{scale_id}",
                    )
                    if st.button(
                        "重新计算量表",
                        type="primary",
                        key=f"recalc_err_{run_id}_{idx}_{scale_id}",
                    ):
                        new_state = recalculate_scale_in_state(
                            state,
                            scale_id,
                            updated_params,
                            scale_index=idx,
                            refresh_draft=False,
                        )
                        st.session_state["copilot_state"] = new_state
                        st.rerun()

    st.divider()
    with st.expander("➕ 手动新增量表评估（4 套标准骨伤/康复量表）", expanded=not bool(scale_results)):
        available = list_available_scales()
        scale_labels = {s["scale_id"]: f"{s['name']} ({s['scale_id']})" for s in available}
        chosen_add_id = st.selectbox(
            "选择要新增评估的量表",
            options=[s["scale_id"] for s in available],
            format_func=lambda sid: scale_labels.get(sid, sid),
            key=f"add_scale_select_{run_id}",
        )
        add_spec = spec_map.get(chosen_add_id)
        if add_spec is not None:
            st.caption(
                f"🧭 **风险方向说明**：{describe_score_direction('higher_is_worse' if add_spec.direction == 'higher_is_worse' else 'lower_is_worse')} | {add_spec.version_note}"
            )
            add_params = _render_scale_manual_inputs(
                add_spec,
                {},
                key_prefix=f"add_scale_{run_id}_{chosen_add_id}",
            )
            if st.button("计算并加入评估", key=f"add_scale_btn_{run_id}_{chosen_add_id}"):
                new_state = recalculate_scale_in_state(
                    state,
                    chosen_add_id,
                    add_params,
                    scale_index=None,
                    refresh_draft=False,
                )
                st.session_state["copilot_state"] = new_state
                st.rerun()


def _render_tab_draft(
    state: Dict[str, Any],
    *,
    chosen_model: str,
    mock_mode: bool,
) -> None:
    current_draft = state.get("draft") or {}
    current_mode = state.get("draft_mode", "ward_round")
    run_id = st.session_state.get("analysis_run_id", 0)

    c1, c2 = st.columns([2, 1])
    with c1:
        target_mode = st.radio(
            "选择文书生成模式",
            options=list(DRAFT_MODE_LABELS.keys()),
            index=list(DRAFT_MODE_LABELS.keys()).index(current_mode),
            format_func=lambda m: DRAFT_MODE_LABELS[m],
            horizontal=True,
            key=f"tab3_draft_mode_{run_id}",
        )
    with c2:
        st.write("")
        regenerate_clicked = st.button(
            "🔄 重新生成 / 切换文书草稿",
            type="primary",
            width="stretch",
            key=f"tab3_regen_btn_{run_id}",
        )

    if regenerate_clicked or target_mode != current_mode:
        try:
            with st.spinner(f"正在生成【{DRAFT_MODE_LABELS[target_mode]}】..."):
                new_draft = generate_draft_document(
                    target_mode,
                    state["extracted_note"],
                    state["risk_report"],
                    state["scale_results"],
                    model=chosen_model,
                    mock_mode=mock_mode,
                )
                state["draft"] = new_draft
                state["draft_mode"] = target_mode
                state["draft_stale"] = False
                state.setdefault("trace", {})["draft_model"] = new_draft.get("model")
                state["trace"]["draft_mock"] = new_draft.get("mock")
                state["trace"]["draft_usage"] = new_draft.get("usage", {})
                st.session_state["copilot_state"] = state
                current_draft = new_draft
        except Exception as exc:
            st.error(f"文书草稿生成失败：{sanitize_error_message(exc)}")

    if state.get("draft_stale"):
        st.warning("⚠️ 量表结果已更新，当前草稿仍基于更新前的数据，请点击上方按钮刷新。")

    draft_content = current_draft.get("content", "")
    st.caption(
        f"当前模式：**{DRAFT_MODE_LABELS.get(state.get('draft_mode', 'ward_round'))}** | "
        f"模型：`{current_draft.get('model', '')}` | Mock：`{current_draft.get('mock', False)}`"
    )
    if current_draft.get("blocked"):
        safety_check = current_draft.get("safety_check") or {}
        violations = safety_check.get("violations") or []
        detail_lines = [
            f"- **[{v.get('rule_id', '')}]** 命中违规动作 `{v.get('matched_action', '')}`：{v.get('snippet', '')}"
            for v in violations
        ]
        st.error(
            "🛑 **安全拦截：生成的文书草稿违背确定性安全红旗/禁忌约束，已阻止渲染！**\n\n"
            + ("\n".join(detail_lines) if detail_lines else safety_check.get("message", ""))
        )
        return

    with st.container(border=True):
        st.markdown(draft_content)

    st.markdown("#### 📋 Markdown 源码（点击右上角一键复制）")
    st.code(draft_content, language="markdown")


def _render_tab_trace(state: Dict[str, Any]) -> None:
    trace = state.get("trace") or {}
    st.markdown("#### 模型与运行元信息")
    st.json(
        {
            "extraction_model": trace.get("extraction_model"),
            "extraction_mock": trace.get("extraction_mock"),
            "draft_model": trace.get("draft_model"),
            "draft_mock": trace.get("draft_mock"),
            "draft_usage": trace.get("draft_usage"),
            "risk_summary": (state.get("risk_report") or {}).get("summary"),
            "unified_alert_count": len(state.get("unified_alerts") or []),
            "scale_count": len(state.get("scale_results") or []),
        }
    )

    with st.expander("1️⃣ 原始结构化提取 JSON (`extracted_note`)", expanded=False):
        st.json(state.get("extracted_note") or {})

    with st.expander("2️⃣ 双轨规则扫描完整 JSON (`risk_report`)", expanded=True):
        st.json(state.get("risk_report") or {})

    with st.expander("3️⃣ 确定性量表计算结果 JSON (`scale_results`)", expanded=False):
        st.json(state.get("scale_results") or [])


def main() -> None:
    st.set_page_config(
        page_title="OrthoRehab 骨伤/康复桌面干活副驾",
        page_icon="🩺",
        layout="wide",
    )

    default_config = LLMConfig.from_env()
    st.session_state.setdefault("raw_note_input", "")
    st.session_state.setdefault("copilot_state", None)
    st.session_state.setdefault("analysis_run_id", 0)
    st.session_state.setdefault("draft_mode_selector", "ward_round")

    chosen_model, mock_mode = _render_sidebar(default_config)

    st.title("🩺 OrthoRehab 骨伤 / 康复桌面干活副驾")
    st.caption(
        "脱敏病历结构化 → 双轨确定性红旗/用药冲突扫描 → 严校验量表算分 → 高约束查房/康复宣教文书"
    )

    st.text_area(
        "脱敏病历 / 跨诊次记录原文",
        key="raw_note_input",
        height=210,
        placeholder="请粘贴已脱敏的骨伤/康复病历文本，或点击左侧边栏演示病例按钮一键填充...",
    )

    col_mode, col_run, col_clear = st.columns([2, 1, 1])
    with col_mode:
        selected_mode = st.selectbox(
            "初始文书草稿类型",
            options=list(DRAFT_MODE_LABELS.keys()),
            format_func=lambda m: DRAFT_MODE_LABELS[m],
            key="draft_mode_selector",
        )
    with col_run:
        st.write("")
        run_clicked = st.button(
            "🚀 开始分析",
            type="primary",
            width="stretch",
            key="run_analysis_btn",
        )
    with col_clear:
        st.write("")
        st.button(
            "🧹 清空",
            width="stretch",
            key="clear_btn",
            on_click=_clear_workspace,
        )

    if run_clicked:
        raw_input = st.session_state.get("raw_note_input", "")
        if not raw_input.strip():
            st.warning("请先输入脱敏病历文本，或从左侧边栏选择一键填充演示病例。")
        else:
            try:
                with st.spinner("正在执行结构化抽取、双轨安全扫描、确定性量表计算与草稿生成..."):
                    result_state = run_copilot_analysis(
                        raw_input,
                        draft_mode=selected_mode,
                        model=chosen_model,
                        mock_mode=mock_mode,
                    )
                st.session_state["copilot_state"] = result_state
                st.session_state["analysis_run_id"] = st.session_state.get("analysis_run_id", 0) + 1
            except Exception as exc:
                st.error(f"分析执行失败：{sanitize_error_message(exc)}")

    state = st.session_state.get("copilot_state")
    if not state:
        st.info("👆 请点击上方 **「开始分析」** 按钮（可先在左侧边栏选择任意演示病例一键填充）。")
        return

    st.divider()
    _render_top_safety_banner(state)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Tab 1：结构化病历与时间线",
            "Tab 2：确定性量表计算",
            "Tab 3：文书草稿",
            "Tab 4：执行轨迹 (Trace)",
        ]
    )
    with tab1:
        _render_tab_structured_note(state)
    with tab2:
        _render_tab_scales(state, chosen_model=chosen_model, mock_mode=mock_mode)
    with tab3:
        _render_tab_draft(state, chosen_model=chosen_model, mock_mode=mock_mode)
    with tab4:
        _render_tab_trace(state)


if __name__ == "__main__":
    main()
