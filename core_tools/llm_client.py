"""Unified multi-provider LLM client built on LiteLLM with local Mock Mode (Week 2).

Reads configuration from .env (via python-dotenv) and supports OpenAI, DeepSeek,
DashScope (Qwen), Zhipu GLM, Moonshot, Anthropic, Gemini, and OpenAI-compatible
custom endpoints.

When LLM_MOCK_MODE=true (or mock_mode=True), runs completely offline without any
API key and returns schema-valid structured clinical extractions and replies.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Type, Union

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from core_tools.calculator import load_scales_catalog
from core_tools.rule_scanner import (
    _extract_clause_windows,
    check_negation_context,
    find_stop_drug_marker,
    load_drug_conflicts_catalog,
    load_red_flags_catalog,
    parse_duration_days,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


_OPENAI_COMPATIBLE_PROVIDERS: FrozenSet[str] = frozenset(
    {"openai", "custom_openai", "ollama", "vllm", "hosted_vllm"}
)
_OPENAI_UNPREFIXED_PREFIXES: Tuple[str, ...] = ("gpt-", "o1", "o3", "o4", "chatgpt-")
_PROVIDER_API_BASE_ENV_MAP: Dict[str, Tuple[str, ...]] = {
    "deepseek": ("DEEPSEEK_API_BASE",),
    "dashscope": ("DASHSCOPE_API_BASE",),
    "zai": ("ZAI_API_BASE", "ZHIPUAI_API_BASE"),
    "zhipu": ("ZHIPUAI_API_BASE", "ZAI_API_BASE"),
    "moonshot": ("MOONSHOT_API_BASE",),
    "anthropic": ("ANTHROPIC_API_BASE",),
    "gemini": ("GEMINI_API_BASE",),
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(
    name: str,
    default: int,
    *,
    min_value: Optional[int] = None,
    fallback_env: Optional[str] = None,
) -> int:
    raw = os.getenv(name)
    if (raw is None or not raw.strip()) and fallback_env:
        raw = os.getenv(fallback_env)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw.strip())
    except ValueError:
        return default
    if min_value is not None and val < min_value:
        return default
    return val


def _env_float(
    name: str,
    default: float,
    *,
    min_value: Optional[float] = None,
) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        val = float(raw.strip())
    except ValueError:
        return default
    if math.isnan(val) or math.isinf(val):
        return default
    if min_value is not None and val < min_value:
        return default
    return val


def _sanitize_error_message(message: str) -> str:
    """Redact API keys, Bearer tokens, and secret query params from error messages."""
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
    return text


def _is_openai_compatible_model(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    if "/" in normalized:
        provider = normalized.split("/", 1)[0].strip()
        return provider in _OPENAI_COMPATIBLE_PROVIDERS
    return normalized.startswith(_OPENAI_UNPREFIXED_PREFIXES)


def _resolve_api_base(
    chosen_model: str,
    config_api_base: Optional[str],
    call_kwargs_overrides: Dict[str, Any],
) -> Optional[str]:
    """Resolve api_base strictly per provider so OPENAI_API_BASE never leaks to other vendors."""
    if "api_base" in call_kwargs_overrides:
        explicit = call_kwargs_overrides["api_base"]
        return str(explicit).strip() if explicit else None

    if _is_openai_compatible_model(chosen_model):
        candidate = config_api_base or os.getenv("OPENAI_API_BASE") or None
        return str(candidate).strip() if candidate else None

    normalized = (chosen_model or "").strip().lower()
    provider = normalized.split("/", 1)[0].strip() if "/" in normalized else ""
    if not provider:
        return None
    env_names = _PROVIDER_API_BASE_ENV_MAP.get(provider)
    if env_names is None and re.fullmatch(r"[a-z0-9_]+", provider):
        env_names = (f"{provider.upper()}_API_BASE",)
    for env_name in env_names or ():
        val = os.getenv(env_name)
        if val and val.strip():
            return val.strip()
    return None


@dataclass
class LLMConfig:
    default_model: str = "deepseek/deepseek-chat"
    temperature: float = 0.1
    timeout_seconds: int = 60
    max_retries: int = 2
    mock_mode: bool = False
    api_base: Optional[str] = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            default_model=os.getenv("COPILOT_DEFAULT_MODEL", "deepseek/deepseek-chat"),
            temperature=_env_float("LLM_TEMPERATURE", 0.1, min_value=0.0),
            timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 60, min_value=1, fallback_env="LLM_TIMEOUT"),
            max_retries=_env_int("LLM_MAX_RETRIES", 2, min_value=0),
            mock_mode=_env_bool("LLM_MOCK_MODE", default=False),
            api_base=os.getenv("OPENAI_API_BASE") or None,
        )


def _strip_thinking_blocks(raw_text: str) -> str:
    """Remove <think>...</think> or <reasoning>...</reasoning> blocks from reasoning model outputs."""
    if not raw_text:
        return ""
    stripped = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE)
    stripped = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _extract_json_candidates(raw_text: str) -> List[str]:
    """Collect candidate JSON strings preferring the final valid object after thinking blocks."""
    if not raw_text:
        return []

    def _scan_text(text: str) -> List[str]:
        found: List[str] = []
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        for block in reversed(fenced):
            b = block.strip()
            if b:
                found.append(b)

        decoder = json.JSONDecoder()
        balanced: List[str] = []
        idx = 0
        length = len(text)
        while idx < length:
            ch = text[idx]
            if ch in "{[":
                try:
                    _, end_idx = decoder.raw_decode(text, idx)
                    balanced.append(text[idx:end_idx].strip())
                    idx = end_idx
                    continue
                except json.JSONDecodeError:
                    pass
            idx += 1
        for item in reversed(balanced):
            if item:
                found.append(item)
        return found

    cleaned_no_think = _strip_thinking_blocks(raw_text)
    candidates = _scan_text(cleaned_no_think)
    if not candidates and cleaned_no_think != raw_text.strip():
        candidates = _scan_text(raw_text)

    if not candidates and cleaned_no_think:
        candidates.append(cleaned_no_think)

    deduped: List[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _strip_json_fences(raw_text: str) -> str:
    """Extract clean JSON object/array string from LLM output (ignoring <think> and earlier examples)."""
    candidates = _extract_json_candidates(raw_text)
    return candidates[0] if candidates else raw_text.strip()


def _extract_raw_note_body(user_text: str) -> str:
    """Extract raw clinical note body from nonce-delimited JSON or legacy delimiters."""
    nonce_match = re.search(
        r"<<<CLINICAL_DATA_([0-9a-fA-F]+)_START>>>\s*([\s\S]*?)\s*<<<CLINICAL_DATA_\1_END>>>",
        user_text,
    )
    if nonce_match:
        payload = nonce_match.group(2).strip()
        try:
            decoded = json.loads(payload)
            if isinstance(decoded, str):
                return decoded.strip()
        except json.JSONDecodeError:
            pass
        return payload

    start_tag = "【病历原文开始】"
    end_tag = "【病历原文结束】"
    s_idx = user_text.find(start_tag)
    e_idx = user_text.rfind(end_tag)
    if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
        return user_text[s_idx + len(start_tag) : e_idx].strip()
    return user_text.strip()


def _extract_scale_params_mock(clean_text: str) -> List[Dict[str, Any]]:
    """Extract explicitly stated clinical scale item scores without ever fabricating missing values."""
    scale_candidates: List[Dict[str, Any]] = []
    catalog = load_scales_catalog()
    spec_map = {s.scale_id: s for s in catalog.scales}

    # 1. VAS_PAIN
    vas_patterns = (
        r"(?:VAS(?:\s*(?:疼痛)?评分)?|视觉模拟(?:疼痛)?评分(?:\s*\(VAS\))?)\s*(?:[:：约大概为]\s*|\s+)(\d+(?:\.\d+)?)\s*分?",
        r"(?:VAS(?:\s*(?:疼痛)?评分)?|视觉模拟(?:疼痛)?评分)\s*(\d+(?:\.\d+)?)\s*分",
    )
    for pat in vas_patterns:
        vas_match = re.search(pat, clean_text, flags=re.IGNORECASE)
        if vas_match:
            scale_candidates.append(
                {"scale_id": "VAS_PAIN", "params": {"pain_score": float(vas_match.group(1))}}
            )
            break

    # 2. JOA_CERVICAL_17 (only populate items explicitly scored in text)
    cervical_patterns = {
        "upper_limb_motor": r"上肢运动(?:功能)?\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
        "lower_limb_motor": r"下肢运动(?:功能)?\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
        "upper_limb_sensory": r"上肢感觉(?:功能)?\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
        "lower_limb_sensory": r"下肢感觉(?:功能)?\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
        "trunk_sensory": r"躯干感觉(?:功能)?\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
    }
    cervical_params: Dict[str, float] = {}
    for item_id, pat in cervical_patterns.items():
        m = re.search(pat, clean_text)
        if m:
            raw_num = m.group(1) or m.group(2)
            cervical_params[item_id] = float(raw_num)

    bladder_match = re.search(
        r"膀胱功能(?:评分)?\s*(?:[:：为得]\s*(-?\d+(?:\.\d+)?)\s*分?|(-?\d+(?:\.\d+)?)\s*分)",
        clean_text,
    )
    has_lumbar_joa_hint = bool(re.search(r"腰椎\s*JOA|JOA\s*29|29\s*分制", clean_text, flags=re.IGNORECASE))
    has_cervical_joa_hint = bool(re.search(r"颈椎\s*JOA|JOA\s*17|17\s*分制", clean_text, flags=re.IGNORECASE))

    if bladder_match:
        bladder_val = float(bladder_match.group(1) or bladder_match.group(2))
        if bladder_val >= 0 and (cervical_params or has_cervical_joa_hint or not has_lumbar_joa_hint):
            cervical_params["bladder_function"] = bladder_val

    pre_cervical_match = re.search(
        r"术前\s*(?:颈椎)?\s*JOA(?:\s*17\s*分制)?\s*(?:[:：为]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
        clean_text,
        flags=re.IGNORECASE,
    )
    if pre_cervical_match and (cervical_params or has_cervical_joa_hint):
        cervical_params["pre_score"] = float(pre_cervical_match.group(1) or pre_cervical_match.group(2))

    if cervical_params:
        scale_candidates.append({"scale_id": "JOA_CERVICAL_17", "params": cervical_params})

    # 3. JOA_LUMBAR_29 (only populate items explicitly scored in text)
    lumbar_patterns = {
        "low_back_pain": r"下腰痛\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
        "leg_pain_tingling": r"下肢痛(?:和/或麻木)?\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
        "gait": r"步行能力\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
        "slr_test": r"直腿抬高(?:试验)?\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
        "sensory_disturbance": r"感觉障碍\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
        "motor_disturbance": r"运动障碍\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)",
    }
    lumbar_params: Dict[str, float] = {}
    for item_id, pat in lumbar_patterns.items():
        m = re.search(pat, clean_text)
        if m:
            lumbar_params[item_id] = float(m.group(1) or m.group(2))
    if bladder_match:
        bladder_val = float(bladder_match.group(1) or bladder_match.group(2))
        if bladder_val < 0 or (has_lumbar_joa_hint and not has_cervical_joa_hint):
            lumbar_params["bladder_score"] = bladder_val
    if lumbar_params:
        scale_candidates.append({"scale_id": "JOA_LUMBAR_29", "params": lumbar_params})

    # 4. WOMAC_LIKERT_V3_1 (match exact Chinese item labels with explicit numeric scores)
    womac_spec = spec_map.get("WOMAC_LIKERT_V3_1")
    if womac_spec:
        womac_params: Dict[str, float] = {}
        sorted_items = sorted(womac_spec.items, key=lambda it: len(it.label), reverse=True)
        for item in sorted_items:
            pat = (
                re.escape(item.label)
                + r"\s*(?:[:：为得]\s*(\d+(?:\.\d+)?)\s*分?|(\d+(?:\.\d+)?)\s*分)"
            )
            m = re.search(pat, clean_text)
            if m:
                womac_params[item.item_id] = float(m.group(1) or m.group(2))
        if womac_params:
            scale_candidates.append({"scale_id": "WOMAC_LIKERT_V3_1", "params": womac_params})

    return scale_candidates


def _extract_section_field(clean_text: str, prefixes: tuple[str, ...]) -> Optional[str]:
    for prefix in prefixes:
        m = re.search(rf"{prefix}\s*[:：]\s*([^\n]+)", clean_text)
        if m:
            return m.group(1).strip()
    return None


def _split_clinical_visits(clean_text: str) -> List[Dict[str, str]]:
    """Split multi-visit longitudinal notes into chronological visit segments."""
    header_re = re.compile(r"【([^】]{2,40})】")
    headers = list(header_re.finditer(clean_text))
    if headers:
        visits: List[Dict[str, str]] = []
        for idx, match in enumerate(headers):
            hdr_text = match.group(1).strip()
            body_start = match.end()
            body_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(clean_text)
            body = clean_text[body_start:body_end].strip()
            visits.append({"header": hdr_text, "body": body, "full": f"【{hdr_text}】{body}".strip()})
        return visits

    # Fallback: split by explicit lines or sentences with date/time markers
    visits = []
    for raw_line in re.split(r"[\n]+", clean_text):
        line = raw_line.strip()
        if line:
            visits.append({"header": "", "body": line, "full": line})
    return visits or [{"header": "", "body": clean_text, "full": clean_text}]


def _detect_department(text: str) -> str:
    dept_keywords = (
        "心内科",
        "脊柱外科",
        "运动医学科",
        "肿瘤外科",
        "肿瘤科",
        "消化科",
        "中医骨伤门诊",
        "骨伤门诊",
        "骨伤科",
        "康复科",
        "骨科病房",
        "骨科",
        "推拿馆",
        "社区药店",
        "社区",
        "外院",
        "急诊",
        "门诊",
        "个人记录",
    )
    for dept in dept_keywords:
        if dept in text:
            return dept
    return "未标明"


def _detect_key_change(text: str) -> str:
    is_question = any(q in text for q in ("？", "?", "吗", "询问", "能否", "是否", "要等到"))
    if (
        not is_question
        and not re.search(r"(?:未|不|严禁|禁止|不可|不能)[^，。；;\n]{0,4}?(?:准予|批准|达标)", text)
        and any(k in text for k in ("准予", "批准", "达标"))
        and any(k in text for k in ("慢跑", "跑步"))
    ):
        return "阶段康复达标，主刀准予解除早期禁跑限制"
    if any(k in text for k in ("红肿热痛", "红肿灼热", "舌红苔黄腻", "热痹")):
        return "证型/病情转为热痹急性炎性期（禁重灸热敷）"
    if any(k in text for k in ("鞍区麻木", "排尿困难", "尿潴留", "大小便失禁")):
        return "新发马尾神经受压红旗征（推翻前期良性诊断）"
    if find_stop_drug_marker(text):
        days = parse_duration_days(text)
        if days is not None:
            return f"停药状态变更（已停约 {days:g} 天）"
        return "存在停药操作记录"
    if any(k in text for k in ("夜间静息痛", "体重明显下降", "骨转移")):
        return "出现肿瘤骨转移高危红旗征象"
    if any(k in text for k in ("黑便", "柏油样", "上腹")):
        return "出现疑似消化道出血征象（需排查NSAIDs叠加）"
    if any(k in text for k in ("全髋关节置换", "THA", "ACDF", "融合内固定", "ACL重建", "支架", "乳腺癌", "肺癌")):
        return "关键既往手术/长期约束史"
    if any(k in text for k in ("加服", "自购", "叠加", "三七", "跌打丸", "布洛芬", "塞来昔布", "利伐沙班", "华法林")):
        return "重点药物暴露/加药节点"
    return ""


def _extract_timeline_events_mock(clean_text: str) -> List[Dict[str, str]]:
    events: List[Dict[str, str]] = []
    time_line_re = re.compile(
        r"(\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?|\d+个?[天周月年]前|半年前|昨日|今日(?:复诊)?|本周|近\d+个?[天周月]|术后第?\d+[天周月]|第\d+周)"
    )

    header_re = re.compile(r"【([^】]{2,40})】")
    if header_re.search(clean_text):
        for visit in _split_clinical_visits(clean_text):
            hdr = visit["header"]
            body = visit["body"]
            m = time_line_re.search(hdr) or time_line_re.search(body)
            date_str = m.group(1) if m else (hdr.split()[0] if hdr else "未标明时间")
            dept_str = _detect_department(hdr + " " + body[:30])
            summary_src = body or hdr
            events.append(
                {
                    "date": date_str,
                    "department": dept_str,
                    "event": summary_src[:140],
                    "key_change": _detect_key_change(hdr + " " + body),
                }
            )
    else:
        for raw_line in re.split(r"[\n。；;]+", clean_text):
            line = raw_line.strip()
            if not line:
                continue
            m = time_line_re.search(line)
            if m:
                events.append(
                    {
                        "date": m.group(1),
                        "department": _detect_department(line),
                        "event": line[:140],
                        "key_change": _detect_key_change(line),
                    }
                )
    if not events:
        events.append(
            {
                "date": "本次记录",
                "department": _detect_department(clean_text),
                "event": clean_text[:120] or "脱敏病历导入",
                "key_change": _detect_key_change(clean_text),
            }
        )
    return events[:12]


def _extract_surgical_history_mock(clean_text: str) -> List[str]:
    history: List[str] = []
    patterns = (
        ("全髋关节置换术（THA）", r"(?:全髋关节置换(?:术)?|THA术后|人工股骨头置换)"),
        ("颈椎前路融合内固定术（ACDF）", r"(?:颈椎前路(?:减压植骨)?融合内固定(?:术)?|ACDF)"),
        ("前交叉韧带重建术（ACL）", r"(?:前交叉韧带重建(?:术)?|ACL重建(?:术后)?|前叉术后)"),
        ("冠脉支架植入术（PCI）", r"(?:冠脉支架(?:植入)?(?:术后)?|PCI支架)"),
        ("恶性肿瘤手术史", r"(?:乳腺癌|肺癌|前列腺癌)(?:病史|手术史|术后|改良根治术后)"),
    )
    for label, pat in patterns:
        if re.search(pat, clean_text, flags=re.IGNORECASE):
            history.append(label)
    return history


def _mock_extract_clinical_facts(user_text: str) -> Dict[str, Any]:
    """Deterministically synthesize a realistic structured clinical note extraction in Mock Mode."""
    clean_text = _extract_raw_note_body(user_text)
    rf_catalog = load_red_flags_catalog()
    dc_catalog = load_drug_conflicts_catalog()
    visits = _split_clinical_visits(clean_text)

    # Chronological symptom polarity resolution: latest visit mention overrides earlier polarity
    symptoms: List[Dict[str, str]] = []
    seen_symptoms = set()
    for rule in rf_catalog.rules:
        for kw in rule.trigger_symptoms.required_any + rule.trigger_symptoms.supporting_any:
            is_regex_kw = kw.startswith("regex:")
            pattern_str = kw[6:] if is_regex_kw else re.escape(kw)
            if not is_regex_kw and kw in seen_symptoms:
                continue
            latest_polarity: Optional[str] = None
            matched_sym_name: str = kw
            for visit in visits:
                seg_text = visit["full"]
                try:
                    seg_matches = list(re.finditer(pattern_str, seg_text, flags=re.IGNORECASE))
                except re.error:
                    continue
                if not seg_matches:
                    continue
                seg_present = False
                for match in seg_matches:
                    matched_sym_name = match.group(0) if is_regex_kw else kw
                    # Skip hypothetical questions about treatments (e.g., "询问能否做颈椎旋转正骨推拿")
                    sent_l = max(seg_text.rfind("。", 0, match.start()), seg_text.rfind("\n", 0, match.start())) + 1
                    sent_r = seg_text.find("。", match.end())
                    sent_slice = seg_text[sent_l : (sent_r if sent_r != -1 else len(seg_text))]
                    if (
                        not is_regex_kw
                        and kw in rule.trigger_symptoms.supporting_any
                        and kw not in rule.trigger_symptoms.required_any
                        and any(q in sent_slice for q in ("询问能否", "能否做", "可以做", "？", "?"))
                    ):
                        continue
                    is_neg, _, _ = check_negation_context(
                        seg_text,
                        match.start(),
                        match.end(),
                        global_negation_terms=rf_catalog.global_negation_terms,
                        negation_guards=rule.negation_guards,
                    )
                    if not is_neg:
                        seg_present = True
                        break
                latest_polarity = "present" if seg_present else "absent"

            if latest_polarity is not None and matched_sym_name not in seen_symptoms:
                seen_symptoms.add(matched_sym_name)
                symptoms.append({"name": matched_sym_name, "polarity": latest_polarity})

    # Build drug washout lookup from catalog
    drug_washout_map: Dict[str, int] = {}
    all_drug_keywords: List[str] = []
    for conflict in dc_catalog.conflicts:
        for grp in (conflict.group_a, conflict.group_b):
            w_days = grp.washout_window_days if grp.washout_window_days is not None else 0
            for kw in grp.keywords:
                all_drug_keywords.append(kw)
                drug_washout_map[kw.lower()] = max(drug_washout_map.get(kw.lower(), 0), w_days)

    medications: List[Dict[str, Any]] = []
    seen_meds = set()

    historical_markers = ("既往曾服", "既往服用", "既往口服", "曾服", "曾用", "以前服", "吃过", "用过", "服过")
    current_active_markers = ("仍在用", "现服", "加服", "拟服", "恢复", "长期口服", "长期服用", "长期规律口服", "至今未停")
    unique_drug_keywords = list(dict.fromkeys(all_drug_keywords))
    negated_stop_re = re.compile(
        r"(?:至今未|并未|尚未|没有|没|不能|不可|不得|切勿|不要|切忌|严禁|禁止|别|莫|勿|未|不)"
        r"[^，,。！？；;\n]{0,6}?(?:停用|停药|停服|停掉|停止服用|停止使用|已停|停)"
    )
    planned_stop_re = re.compile(
        r"(?:建议|拟|计划|准备|择期|考虑|打算)[^，,。！？；;\n]{0,6}?(?:停用|停药|停服|停掉|停止服用|停止使用)"
    )
    switch_drug_re = re.compile(r"(?:改用|换用|换成|改为|替换为|调整为)")
    resume_drug_re = re.compile(r"(?<!未)(?:恢复|重新|续用|再次服用|复用|继续服用|继续口服|改回)")

    for kw in unique_drug_keywords:
        matches = list(re.finditer(re.escape(kw), clean_text, flags=re.IGNORECASE))
        if not matches or kw in seen_meds:
            continue

        other_drugs = [
            od for od in unique_drug_keywords
            if od.lower() != kw.lower() and od.lower() not in kw.lower() and kw.lower() not in od.lower()
        ]
        other_drugs_re = (
            re.compile("|".join(re.escape(od) for od in sorted(other_drugs, key=len, reverse=True)), flags=re.IGNORECASE)
            if other_drugs
            else None
        )

        mention_states: List[Dict[str, Any]] = []
        for match in matches:
            left_win, right_win, full_clause = _extract_clause_windows(
                clean_text, match.start(), match.end(), pre_window=16, post_window=22
            )
            scoped_left = left_win
            if other_drugs_re:
                last_other_left = None
                for om in other_drugs_re.finditer(scoped_left):
                    last_other_left = om
                if last_other_left is not None:
                    scoped_left = scoped_left[last_other_left.end() :]
            switch_m = None
            for sm in switch_drug_re.finditer(scoped_left):
                switch_m = sm
            if switch_m is not None:
                scoped_left = scoped_left[switch_m.end() :]

            scoped_right = right_win
            if other_drugs_re:
                om_right = other_drugs_re.search(scoped_right)
                if om_right is not None:
                    scoped_right = scoped_right[: om_right.start()]

            local_snippet = scoped_left + clean_text[match.start() : match.end()] + scoped_right
            has_negated_or_planned_stop = bool(
                negated_stop_re.search(local_snippet) or planned_stop_re.search(local_snippet)
            )
            sanitized_stop_snippet = planned_stop_re.sub("", negated_stop_re.sub("", local_snippet))
            matched_stop_marker = find_stop_drug_marker(sanitized_stop_snippet)
            is_stopped = bool(matched_stop_marker) and not has_negated_or_planned_stop

            if has_negated_or_planned_stop and not is_stopped:
                mention_states.append(
                    {"status": "active", "stopped_days_ago": None, "priority": 2, "explicit_active": True}
                )
                continue

            # Check whether the drug was resumed after a stop mention in the same sentence
            resumed_after_stop = False
            if is_stopped:
                sentence_tail = clean_text[match.start() :]
                sent_end_m = re.search(r"[。！？；;\n]", sentence_tail)
                sentence_scope = sentence_tail[: sent_end_m.start()] if sent_end_m else sentence_tail
                if other_drugs_re:
                    om_sent = other_drugs_re.search(sentence_scope[len(kw) :])
                    if om_sent is not None:
                        sentence_scope = sentence_scope[: len(kw) + om_sent.start()]
                stop_pos = sanitized_stop_snippet.find(matched_stop_marker) if matched_stop_marker else -1
                if stop_pos != -1 and resume_drug_re.search(sentence_scope):
                    # Verify resume appears after the stop marker in the combined local+sentence window
                    combined_stop_window = scoped_left + sentence_scope
                    stop_idx_comb = combined_stop_window.find(matched_stop_marker)
                    if stop_idx_comb != -1 and resume_drug_re.search(combined_stop_window[stop_idx_comb + len(matched_stop_marker) :]):
                        resumed_after_stop = True

            if resumed_after_stop:
                mention_states.append(
                    {"status": "active", "stopped_days_ago": None, "priority": 2, "explicit_active": True}
                )
                continue

            stopped_days = None
            if is_stopped:
                # Avoid attributing active-taking duration (e.g. "已规律服用6个月") to cessation
                dur_src = re.sub(r"(?:已?规律服用|长期服用|已服用|连服|服用)\s*\d*[^，,。！？；;\n]{0,8}", "", local_snippet)
                stopped_days = parse_duration_days(dur_src)
                if stopped_days is None:
                    # Only look forward within the same comma-clause and before any other drug or switch/taking verb
                    tail = clean_text[match.end() :]
                    boundary_m = re.search(r"[，,。！？；;\n]", tail)
                    forward_scope = tail[: boundary_m.start()] if boundary_m else tail
                    if other_drugs_re:
                        om_fwd = other_drugs_re.search(forward_scope)
                        if om_fwd is not None:
                            forward_scope = forward_scope[: om_fwd.start()]
                    sw_fwd = switch_drug_re.search(forward_scope)
                    if sw_fwd is not None:
                        forward_scope = forward_scope[: sw_fwd.start()]
                    if not re.search(r"(?:规律服用|仍在服|继续服|恢复服用)", forward_scope):
                        stopped_days = parse_duration_days(forward_scope[:32])

            if is_stopped:
                mention_states.append(
                    {"status": "stopped", "stopped_days_ago": stopped_days, "priority": 2, "explicit_active": False}
                )
                continue

            is_neg, _, _ = check_negation_context(
                clean_text,
                match.start(),
                match.end(),
                global_negation_terms=rf_catalog.global_negation_terms,
            )
            if is_neg:
                continue

            is_historical_only = (
                any(hm in local_snippet for hm in historical_markers)
                or ("？" in full_clause and not any(cm in local_snippet for cm in ("一起吃", "同服", "加服", "叠加")))
            ) and not any(
                cm in local_snippet for cm in ("仍在用", "现服", "加服", "恢复", "长期口服", "长期服用", "长期规律口服", "至今未停")
            )
            is_explicit_active = any(cm in local_snippet for cm in current_active_markers)
            if is_historical_only and not is_explicit_active:
                mention_states.append(
                    {"status": "historical", "stopped_days_ago": None, "priority": 0, "explicit_active": False}
                )
            else:
                mention_states.append(
                    {
                        "status": "active",
                        "stopped_days_ago": None,
                        "priority": 1,
                        "explicit_active": is_explicit_active,
                    }
                )

        if not mention_states:
            continue

        # Prefer explicit active/stopped over ambiguous historical mention; use latest non-historical state,
        # but do not let a vague non-explicit mention override an earlier explicit stopped state.
        non_hist = [m for m in mention_states if m["status"] != "historical"]
        if non_hist:
            last_stopped_idx = max(
                (idx for idx, m in enumerate(non_hist) if m["status"] == "stopped"),
                default=-1,
            )
            if last_stopped_idx != -1 and not any(
                m.get("explicit_active") for m in non_hist[last_stopped_idx + 1 :]
            ):
                chosen = non_hist[last_stopped_idx]
            else:
                chosen = non_hist[-1]
        else:
            chosen = mention_states[-1]
        final_status = "unknown" if chosen["status"] == "historical" else chosen["status"]

        seen_meds.add(kw)
        medications.append(
            {
                "name": kw,
                "status": final_status,
                "stopped_days_ago": chosen["stopped_days_ago"],
            }
        )

    # Split medications into active_medications (active/unknown/in-washout) vs stopped_medications (beyond washout)
    active_medications: List[Dict[str, Any]] = []
    stopped_medications: List[Dict[str, Any]] = []
    for med in medications:
        w_days = drug_washout_map.get(med["name"].lower(), 0)
        s_days = med.get("stopped_days_ago")
        if (
            med["status"] == "stopped"
            and isinstance(s_days, (int, float))
            and not isinstance(s_days, bool)
            and s_days > w_days
        ):
            stopped_medications.append(dict(med))
        else:
            active_medications.append(dict(med))

    active_surgical_history = _extract_surgical_history_mock(clean_text)
    present_sym_names = {s["name"] for s in symptoms if s["polarity"] == "present"}

    # Chronological conflict resolution & superseded conditions
    superseded_conditions: List[Dict[str, Any]] = []
    override_notes: List[str] = []

    has_acl = any("ACL" in h for h in active_surgical_history)
    latest_visit = visits[-1] if visits else {"header": "", "body": clean_text, "full": clean_text}
    latest_header = latest_visit.get("header", "")
    latest_body = latest_visit.get("body", "") or latest_visit.get("full", "")
    has_mature_acl_header = bool(
        re.search(
            r"(?:术后\s*第?\s*(?:1[6-9]|[2-9]\d)\s*周|术后\s*(?:4|5|6|7|8|9|1[0-2])\s*个月|术后半年|术后\s*[12]\s*年|第\s*(?:1[6-9]|[2-9]\d)\s*周)",
            latest_header,
        )
    )
    has_affirmative_jog_clearance = False
    if has_acl and has_mature_acl_header:
        for sent in re.split(r"[。！？\n]+", latest_body):
            s = sent.strip()
            if not s:
                continue
            is_question_sent = any(q in s for q in ("？", "?", "吗", "询问", "能否", "是否", "要等到", "可以吗"))
            is_negated_sent = bool(re.search(r"(?:未|不|严禁|禁止|不可|不能)[^，。；]{0,6}?(?:准予|批准|达标|允许)", s))
            if (
                not is_question_sent
                and not is_negated_sent
                and any(k in s for k in ("准予", "批准", "达标", "允许"))
                and any(k in s for k in ("慢跑", "跑步"))
            ):
                has_affirmative_jog_clearance = True
                break

    if has_affirmative_jog_clearance:
        superseded_conditions.append(
            {
                "old_state": "ACL重建术后第10天早期保护期（严禁跑跳与高负荷训练）",
                "new_state": "术后第16周复查肌力与活动度达标，主刀医生准予进入基础直线慢跑阶段",
                "rule_hint": "RF_007",
            }
        )
        override_notes.append(
            "最新术后第16周复查已获主刀评估达标并准予直线慢跑，解除早期 RF_007 禁跑限制。"
        )

    if any(k in clean_text for k in ("寒湿痹阻", "寒痹")) and (
        {"关节红肿热痛", "局部灼热", "热痹", "急性痛风"} & present_sym_names
        or "红肿热痛" in clean_text
        or "红肿灼热" in clean_text
    ):
        superseded_conditions.append(
            {
                "old_state": "既往膝关节寒湿痹阻（艾灸热敷有效）",
                "new_state": "本次转为热痹/急性炎性充血期（红肿灼热、舌红苔黄腻）",
                "rule_hint": None,
            }
        )
        override_notes.append(
            "最新诊次已由既往寒湿痹阻转为热痹急性炎性期，既往艾灸热敷方案已推翻且当前严禁重灸热敷。"
        )

    if any(k in clean_text for k in ("腰肌劳损", "轻度坐骨神经痛", "红旗阴性")) and (
        {"鞍区麻木", "会阴区麻木", "尿潴留", "排尿困难", "大小便失禁", "二便失禁"} & present_sym_names
    ):
        superseded_conditions.append(
            {
                "old_state": "前期普通腰肌劳损/轻度坐骨神经痛（红旗阴性）",
                "new_state": "本次突发会阴鞍区麻木与排尿困难，高度怀疑马尾综合征急诊红旗",
                "rule_hint": None,
            }
        )
        override_notes.append(
            "最新诊次突发马尾神经受压红旗征，已推翻前期良性腰肌劳损判断，须立即急诊脊柱外科评估。"
        )

    for med in stopped_medications:
        if any(k in med["name"] for k in ("华法林", "利伐沙班", "阿哌沙班", "达比加群", "氯吡格雷", "阿司匹林")):
            superseded_conditions.append(
                {
                    "old_state": f"既往服用 {med['name']} 抗栓治疗状态",
                    "new_state": f"已停药 {med['stopped_days_ago']:g} 天（超过洗脱期，不再视为在用冲突）",
                    "rule_hint": None,
                }
            )
            override_notes.append(
                f"既往抗栓药物 {med['name']} 已停用 {med['stopped_days_ago']:g} 天并超过洗脱期，解除抗凝出血禁忌锚定。"
            )

    if any("肿瘤" in h for h in active_surgical_history) and (
        "夜间静息痛" in present_sym_names or "体重明显下降" in present_sym_names
    ):
        override_notes.append(
            "结合既往恶性肿瘤手术史及新发夜间静息痛/消瘦，警惕脊柱骨转移红旗征，严禁按普通腰背痛行牵引正骨。"
        )

    current_state_snapshot = {
        "active_medications": active_medications,
        "stopped_medications": stopped_medications,
        "current_symptoms": symptoms,
        "active_surgical_history": active_surgical_history,
        "superseded_conditions": superseded_conditions,
        "diagnosis_override_note": "；".join(override_notes) if override_notes else None,
    }

    scale_candidates = _extract_scale_params_mock(clean_text)

    first_line = next((ln.strip() for ln in clean_text.splitlines() if ln.strip()), "")
    patient_summary = first_line[:80] if first_line else "脱敏病例（未提供基本信息）"
    chief_complaint = _extract_section_field(clean_text, ("主诉",)) or (clean_text[:80] if clean_text else "未提供主诉")
    hpi = _extract_section_field(clean_text, ("现病史",)) or clean_text[:220] or "未提及"
    imaging_summary = _extract_section_field(clean_text, ("影像学摘要", "影像检查", "影像学"))
    if not imaging_summary:
        img_hits = [
            seg.strip()
            for seg in re.split(r"[。\n；;]+", clean_text)
            if any(k in seg for k in ("X线", "MRI", "CT", "彩超", "超声", "骨密度", "INR"))
        ]
        imaging_summary = "；".join(img_hits) if img_hits else "病历暂未提供明确影像学报告"

    tongue_m = re.search(r"(舌[^，。；;\n]+)", clean_text)
    pulse_m = re.search(r"(脉[^，。；;\n]+)", clean_text)
    syndrome_m = re.search(r"((?:辨证|证属)[^。\n]+)", clean_text)
    tcm_line = _extract_section_field(clean_text, ("中医四诊", "四诊"))
    tcm_four = {
        "inspection": tcm_line or "见病历望诊描述（Mock 抽取）",
        "auscultation_olfaction": "语声、气味未见特殊异常记录",
        "inquiry": chief_complaint,
        "palpation": _extract_section_field(clean_text, ("专科查体与红旗筛查", "专科查体", "查体")) or "建议核对专科触诊压痛记录",
        "tongue": tongue_m.group(1) if tongue_m else "未明确记录",
        "pulse": pulse_m.group(1) if pulse_m else "未明确记录",
        "syndrome_pattern": syndrome_m.group(1) if syndrome_m else "待结合四诊辨证",
    }

    exam_line = _extract_section_field(clean_text, ("专科查体与红旗筛查", "专科查体", "查体"))
    physical_exam: List[str] = []
    if exam_line:
        physical_exam = [part.strip() for part in re.split(r"[；;，,]+", exam_line) if part.strip()]
    else:
        physical_exam = [s["name"] for s in symptoms if s["polarity"] == "present"]
    if not physical_exam:
        physical_exam = ["详见病历查体记录"]

    structured_facts = {"symptoms": symptoms, "medications": medications}
    timeline_events = _extract_timeline_events_mock(clean_text)

    return {
        "patient_summary": patient_summary,
        "chief_complaint": chief_complaint,
        "history_of_present_illness": hpi,
        "tcm_four_diagnostics": tcm_four,
        "physical_exam": physical_exam,
        "imaging_summary": imaging_summary,
        "timeline_events": timeline_events,
        "structured_facts": structured_facts,
        "current_state_snapshot": current_state_snapshot,
        "scale_params": scale_candidates,
        # Backward compatibility fields for existing tests / consumers
        "symptoms": symptoms,
        "medications": medications,
        "scale_candidates": scale_candidates,
        "mock_notice": "当前由 LLM_MOCK_MODE 本地模拟生成，仅供流程验证与离线开发。",
    }


def _mock_draft_reply(messages: List[Dict[str, str]]) -> str:
    """Synthesize a high-constraint Markdown clinical draft obeying deterministic alerts and scores."""
    all_text = "\n".join(m.get("content", "") for m in messages)
    user_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")

    mode = "rehab_education" if "【文书模式：rehab_education】" in all_text or "mode=rehab_education" in user_text else "ward_round"

    def _extract_block(start_hdr: str, end_hdr: str) -> str:
        s_idx = user_text.find(start_hdr)
        if s_idx == -1:
            return ""
        s_idx += len(start_hdr)
        e_idx = user_text.find(end_hdr, s_idx)
        return (user_text[s_idx:e_idx] if e_idx != -1 else user_text[s_idx:]).strip()

    constraints_block = _extract_block("### 一、确定性安全硬约束（绝对不可违背）", "### 二、确定性量表计算结果")
    scales_block = _extract_block("### 二、确定性量表计算结果", "### 三、结构化脱敏病历上下文")

    note_data: Dict[str, Any] = {}
    json_match = re.search(r"### 三、结构化脱敏病历上下文\s*```json\s*([\s\S]*?)```", user_text)
    if json_match:
        try:
            note_data = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            note_data = {}

    has_risk = bool(constraints_block) and "【无命中红旗/用药冲突】" not in constraints_block
    patient_summary = note_data.get("patient_summary", "脱敏病例")
    chief_complaint = note_data.get("chief_complaint", "见病历原文")
    imaging_summary = note_data.get("imaging_summary", "暂无")
    tcm = note_data.get("tcm_four_diagnostics") or {}
    syndrome = tcm.get("syndrome_pattern", "待四诊合参")
    timeline = note_data.get("timeline_events") or []
    timeline_md = "\n".join(f"- **{ev.get('date', '时间未明')}**：{ev.get('event', '')}" for ev in timeline) or "- 本次脱敏记录"

    if mode == "ward_round":
        lines = [
            "# 上级医师查房汇报提纲（Mock 辅助草稿）",
            "",
            f"> **脱敏患者画像**：{patient_summary}",
            f"> **主诉摘要**：{chief_complaint}",
            "",
            "## 一、安全红旗与中西医用药冲突硬约束",
        ]
        if has_risk:
            lines.extend(
                [
                    "⚠️ **确定性规则引擎已触发以下高危约束，汇报与诊疗方案严禁违背：**",
                    constraints_block,
                ]
            )
        else:
            lines.append("✅ 当前确定性双轨扫描未命中红旗征或高危中西医用药冲突。")

        lines.extend(
            [
                "",
                "## 二、确定性量表评估结果",
                scales_block or "暂无完整量表评分。",
                "",
                "## 三、跨诊次时间线与四诊/影像要点",
                f"- **中医辨证要点**：{syndrome}（舌象：{tcm.get('tongue', '未详')}；脉象：{tcm.get('pulse', '未详')}）",
                f"- **影像/检验摘要**：{imaging_summary}",
                "**关键时间线：**",
                timeline_md,
                "",
                "## 四、查房拟请示与下一步安全处置方向",
            ]
        )
        if has_risk:
            lines.extend(
                [
                    "- 严格落实上述绝对禁忌清单，**暂缓并严禁一切触发禁忌的手法整复、热疗或冲突活血用药叠加**。",
                    "- 建议请上级医师复核红旗指征：必要时优先完善专科影像学检查（如颈/腰椎 MRI）或 INR/凝血及药学评估，明确手术或专科转诊指征。",
                ]
            )
        else:
            lines.extend(
                [
                    "- 已完成红旗否定筛查，可结合局部肌肉劳损与辨证要点开展规范保守康复方案。",
                    "- 建议随访疼痛与功能量表变化，动态观察有无新发神经受压或红旗症状。",
                ]
            )
        lines.extend(
            [
                "",
                "---",
                "*声明：本提纲由 OrthoRehab 桌面副驾（Mock 模式）生成，仅限脱敏病历整理与临床思维辅助，不替代执业医师临床决策。*",
            ]
        )
        return "\n".join(lines)

    # mode == "rehab_education"
    lines = [
        "# 分阶段康复宣教要点（Mock 辅助草稿）",
        "",
        f"> **适用对象画像（脱敏）**：{patient_summary}",
        "",
        "## 一、安全边界与绝对禁忌提示",
    ]
    if has_risk:
        lines.extend(
            [
                "🚨 **当前存在高危警示信号，请务必先行遵守以下安全红线：**",
                constraints_block,
                "",
                "- **在专科医师完成针对性排查与评估前，严禁自行接受或尝试上述禁忌清单中的任何推拿扳法、理疗或活血药物叠加。**",
            ]
        )
    else:
        lines.append("✅ 当前未检测到高危红旗或用药冲突，可在安全前提下按阶段开展温和康复护养。")

    lines.extend(
        [
            "",
            "## 二、当前功能与疼痛评估解读",
            scales_block or "暂无量化量表结果。",
            "",
            "## 三、分阶段康复与日常护养要点",
        ]
    )
    if has_risk:
        lines.extend(
            [
                "- **第一阶段（当前优先：安全制动与专科复核期）**：保持病变部位稳定与保护性制动，避免盲目颈腰部旋转、大幅度负重或自行加服未经审核的活血止痛成药/药酒。",
                "- **第二阶段（专科明确安全后）**：仅在主诊医师/康复治疗师确认排除高危进展并制定个体化方案后，方可循序渐进开展低负荷保护性训练。",
            ]
        )
    else:
        lines.extend(
            [
                "- **第一阶段（急性缓解期 / 第 1-2 周）**：减少连续久坐超过 45 分钟，注意保持脊柱中立位与腰背保暖，可在无痛范围内进行轻柔肌肉放松与呼吸调节。",
                "- **第二阶段（功能巩固期 / 第 2-6 周）**：逐步加强核心肌群稳定性训练（如温和桥式、麦肯基/猫式伸展等无痛动作），循序渐进提高日常耐力。",
                "- **日常复诊预警信号**：若后续出现下肢放射痛麻木加重、踩棉花感、大小便异常或夜间静息痛，请立即停止锻炼并及时骨科就诊。",
            ]
        )
    lines.extend(
        [
            "",
            "---",
            "*合规提示：本宣教要点仅作脱敏病例与康复思维辅助，不构成直接面向患者的诊断或处方。*",
        ]
    )
    return "\n".join(lines)


def _mock_chat_reply(messages: List[Dict[str, str]]) -> str:
    """Generate a clinically reasonable mock assistant reply based on the latest user turn."""
    combined_all = "\n".join(m.get("content", "") for m in messages)
    if (
        "【文书模式：ward_round】" in combined_all
        or "【文书模式：rehab_education】" in combined_all
        or ("确定性安全硬约束" in combined_all and "确定性量表计算结果" in combined_all)
    ):
        return _mock_draft_reply(messages)

    combined_user = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    if any(k in combined_user for k in ("踩棉花", "脊髓", "扳法", "掰脖子")):
        return (
            "【Mock 临床安全提示】您描述的步态踩棉花感与精细动作障碍高度提示颈脊髓受压（疑似脊髓型颈椎病）。"
            "此类情况严禁进行任何颈椎旋转扳法、暴力正骨或在家自行复位，以免诱发急性脊髓损伤。"
            "建议尽快前往脊柱骨科完善颈椎 MRI 检查并由专科医师评估治疗方案。"
        )
    if any(k in combined_user for k in ("鞍区麻木", "大小便失禁", "尿潴留", "马尾")):
        return (
            "【Mock 临床急诊提示】会阴/鞍区麻木伴排尿困难属于马尾综合征高危急诊红旗征象，"
            "切勿贴膏药观察或推拿拖延至次日，请立即前往医院急诊骨科/脊柱外科行紧急 MRI 评估。"
        )
    if any(k in combined_user for k in ("华法林", "利伐沙班", "双抗", "阿司匹林")) and any(
        k in combined_user for k in ("三七", "跌打丸", "活血", "洛索洛芬", "塞来昔布")
    ):
        return (
            "【Mock 用药安全提示】抗凝药或双联抗血小板药物与活血化瘀中药/口服 NSAIDs 存在显著出血相互作用，"
            "并非间隔两小时服用即可消除风险。请勿自行叠加服用，建议立即联系主诊医师核查 INR 及出血风险并调整镇痛消肿方案。"
        )
    if any(k in combined_user for k in ("红肿", "发烫", "痛风", "第一跖趾")) and any(
        k in combined_user for k in ("艾灸", "热敷", "重灸")
    ):
        return (
            "【Mock 辨证禁忌提示】局部明显红肿灼热、剧痛拒按符合急性痛风/热痹急性期表现，属于实热充血阶段，"
            "严禁局部重灸、热敷或搓揉以免加重炎症渗出与疼痛。建议制动抬高患肢并及时就医核查血尿酸与炎症指标。"
        )
    return (
        "【Mock 临床辅助回复】已收到您提供的脱敏临床信息。对于明确无外伤、无下肢放射痛麻木、"
        "二便正常且无夜间静息痛与发热的单纯久坐腰肌酸胀，可先注意改善久坐姿势、定时起身活动及温和核心肌群拉伸；"
        "若后续出现症状加重或新增麻木无力等红旗表现，请及时骨科就诊。"
    )


class LLMClient:
    """Unified LiteLLM wrapper supporting multi-provider routing and offline Mock Mode."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()

    def _is_mock(self, mock_mode: Optional[bool]) -> bool:
        if mock_mode is not None:
            return mock_mode
        return self.config.mock_mode

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        mock_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a unified chat completion via LiteLLM or local Mock Mode."""
        chosen_model = model or self.config.default_model
        use_mock = self._is_mock(mock_mode)

        if use_mock:
            content = _mock_chat_reply(messages)
            return {
                "status": "ok",
                "content": content,
                "model": f"mock/{chosen_model}",
                "mock": True,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "raw": None,
            }

        try:
            import litellm  # type: ignore[import-not-found]
        except ImportError as exc:
            return {
                "status": "error",
                "error_code": "LITELLM_NOT_INSTALLED",
                "message": "未安装 litellm。请运行 `pip install litellm` 安装，或在 .env 中设置 LLM_MOCK_MODE=true 使用本地 Mock 模式。",
                "content": "",
                "model": chosen_model,
                "mock": False,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "raw": None,
            }

        call_kwargs: Dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "timeout": self.config.timeout_seconds,
            "num_retries": self.config.max_retries,
        }
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            call_kwargs["response_format"] = response_format

        extra_kwargs = dict(kwargs)
        resolved_api_base = _resolve_api_base(chosen_model, self.config.api_base, extra_kwargs)
        extra_kwargs.pop("api_base", None)
        call_kwargs.update(extra_kwargs)
        if resolved_api_base:
            call_kwargs["api_base"] = resolved_api_base

        try:
            response = litellm.completion(**call_kwargs)
            content = response.choices[0].message.content or ""
        except Exception as exc:
            return {
                "status": "error",
                "error_code": "LLM_CALL_ERROR",
                "message": _sanitize_error_message(f"{type(exc).__name__}: {exc}"),
                "content": "",
                "model": chosen_model,
                "mock": False,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "raw": None,
            }

        usage_obj = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0,
            "completion_tokens": getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0,
            "total_tokens": getattr(usage_obj, "total_tokens", 0) if usage_obj else 0,
        }
        return {
            "status": "ok",
            "content": content,
            "model": chosen_model,
            "mock": False,
            "usage": usage,
            "raw": response,
        }

    @staticmethod
    def _parse_and_validate_json(
        raw_content: str,
        schema_model: Optional[Type[BaseModel]],
    ) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        candidates = _extract_json_candidates(raw_content)
        parsed_objects: List[Any] = []
        last_json_err: Optional[Exception] = None
        for cand in candidates:
            try:
                parsed_objects.append(json.loads(cand))
            except json.JSONDecodeError as exc:
                last_json_err = exc

        if not parsed_objects:
            err_detail = str(last_json_err or "未找到合法 JSON 对象")
            return None, "JSON_PARSE_ERROR", f"模型返回内容无法解析为合法 JSON: {err_detail}"

        if schema_model is None:
            return parsed_objects[0], None, None

        last_val_err: Optional[Exception] = None
        for obj in parsed_objects:
            try:
                validated = schema_model.model_validate(obj)
                return validated.model_dump(), None, None
            except (ValidationError, TypeError, ValueError) as exc:
                last_val_err = exc

        return None, "SCHEMA_VALIDATION_ERROR", f"模型输出未通过 Schema 校验: {last_val_err}"

    def extract_structured_json(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        *,
        schema_model: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        mock_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Extract and parse a structured JSON object (optionally validated by Pydantic)."""
        if isinstance(prompt_or_messages, str):
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append(
                    {
                        "role": "system",
                        "content": "你是中医骨伤与康复科结构化病历信息抽取助手，请仅输出合法 JSON，不要附带额外解释。",
                    }
                )
            messages.append({"role": "user", "content": prompt_or_messages})
            user_text = prompt_or_messages
        else:
            messages = list(prompt_or_messages)
            user_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")

        chosen_model = model or self.config.default_model
        use_mock = self._is_mock(mock_mode)

        if use_mock:
            data = _mock_extract_clinical_facts(user_text)
            if schema_model is not None:
                try:
                    validated = schema_model.model_validate(data)
                    data = validated.model_dump()
                except (ValidationError, TypeError, ValueError) as exc:
                    return {
                        "status": "error",
                        "error_code": "SCHEMA_VALIDATION_ERROR",
                        "message": f"Mock 输出未通过 Schema 校验: {exc}",
                        "data": None,
                        "model": f"mock/{chosen_model}",
                        "mock": True,
                        "raw_content": json.dumps(data, ensure_ascii=False),
                    }
            return {
                "status": "ok",
                "data": data,
                "model": f"mock/{chosen_model}",
                "mock": True,
                "raw_content": json.dumps(data, ensure_ascii=False),
            }

        resp = self.chat_completion(
            messages=messages,
            model=chosen_model,
            temperature=temperature,
            response_format={"type": "json_object"},
            mock_mode=False,
            **kwargs,
        )
        if resp.get("status") == "error":
            return {
                "status": "error",
                "error_code": resp.get("error_code", "LLM_CALL_ERROR"),
                "message": resp.get("message", "LLM 调用失败"),
                "data": None,
                "model": resp.get("model", chosen_model),
                "mock": False,
                "raw_content": "",
            }

        data, err_code, err_msg = self._parse_and_validate_json(resp["content"], schema_model)
        if data is not None:
            return {
                "status": "ok",
                "data": data,
                "model": resp["model"],
                "mock": False,
                "raw_content": resp["content"],
            }

        # Targeted single repair retry on JSON parse or Pydantic ValidationError in real LLM mode
        repair_messages = list(messages) + [
            {"role": "assistant", "content": resp["content"]},
            {
                "role": "user",
                "content": (
                    "你刚才输出的 JSON 未通过格式或 Schema 校验，请严格修复错误并仅输出一个合法的 JSON 对象"
                    "（不要输出 Markdown 解释或思考过程）。\n"
                    f"校验错误摘要：{(err_msg or '')[:600]}"
                ),
            },
        ]
        repair_resp = self.chat_completion(
            messages=repair_messages,
            model=chosen_model,
            temperature=temperature,
            response_format={"type": "json_object"},
            mock_mode=False,
            **kwargs,
        )
        if repair_resp.get("status") == "error":
            return {
                "status": "error",
                "error_code": repair_resp.get("error_code", "LLM_CALL_ERROR"),
                "message": repair_resp.get("message", "修复重试调用失败"),
                "data": None,
                "model": repair_resp.get("model", chosen_model),
                "mock": False,
                "raw_content": resp["content"],
            }

        repaired_data, repaired_err_code, repaired_err_msg = self._parse_and_validate_json(
            repair_resp["content"], schema_model
        )
        if repaired_data is not None:
            return {
                "status": "ok",
                "data": repaired_data,
                "model": repair_resp["model"],
                "mock": False,
                "raw_content": repair_resp["content"],
                "retried": True,
            }

        return {
            "status": "error",
            "error_code": repaired_err_code or err_code or "SCHEMA_VALIDATION_ERROR",
            "message": _sanitize_error_message(repaired_err_msg or err_msg or "结构化抽取校验失败"),
            "data": None,
            "model": repair_resp["model"],
            "mock": False,
            "raw_content": repair_resp["content"],
            "retried": True,
        }


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    mock_mode: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Module-level convenience wrapper for LLMClient.chat_completion()."""
    return LLMClient().chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        mock_mode=mock_mode,
        **kwargs,
    )


def extract_structured_json(
    prompt_or_messages: Union[str, List[Dict[str, str]]],
    *,
    schema_model: Optional[Type[BaseModel]] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    mock_mode: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Module-level convenience wrapper for LLMClient.extract_structured_json()."""
    return LLMClient().extract_structured_json(
        prompt_or_messages=prompt_or_messages,
        schema_model=schema_model,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        mock_mode=mock_mode,
        **kwargs,
    )
