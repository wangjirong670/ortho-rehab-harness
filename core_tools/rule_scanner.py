"""Deterministic dual-track clinical risk scanner (Week 2).

Track A (Text Scan + Negation & Washout Window):
    Scans raw clinical text against knowledge_base/red_flags.yaml and
    knowledge_base/drug_conflicts.yaml with rule-specific negation_guards,
    bidirectional negation windows (respecting Chinese clause boundaries),
    and drug cessation duration parsing against washout_window_days.

Track B (Structured Facts Polarity & Medication Status):
    Prioritizes structured clinical facts (symptom polarity: present/absent,
    medication status: active/stopped + stopped_days_ago). Explicit absent
    symptoms or medications stopped beyond washout_window_days override
    text-level hits and are logged into suppressed_matches for auditability.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from core_tools.schemas import (
    DrugConflictRule,
    DrugConflictsCatalog,
    RedFlagRule,
    RedFlagsCatalog,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RED_FLAGS_PATH = PROJECT_ROOT / "knowledge_base" / "red_flags.yaml"
DEFAULT_DRUG_CONFLICTS_PATH = PROJECT_ROOT / "knowledge_base" / "drug_conflicts.yaml"

SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}

# Strong clause boundaries that stop negation scope propagation
STRONG_BOUNDARY_CHARS = "。！？；;\n"
SOFT_CLAUSE_SEPARATORS = "，,"

# Additional prohibitive/cautionary negation terms (e.g., "切忌在家自行复位")
PROHIBITION_TERMS = (
    "切忌",
    "严禁",
    "禁止",
    "不要",
    "不能",
    "不可",
    "切勿",
    "勿",
    "避免",
    "不宜",
    "禁忌",
)

POST_NEGATION_TERMS = (
    "阴性",
    "正常",
    "(-)",
    "（-）",
    "已排除",
    "未见",
    "不存在",
    "无异常",
)

STOP_DRUG_MARKERS = (
    "停用",
    "停药",
    "停服",
    "停掉",
    "已停",
    "早已停",
    "未再服用",
    "未服用",
    "停止服用",
    "停止使用",
)

_NEGATED_STOP_MARKER_RE = re.compile(
    r"(?:至今未|并未|尚未|没有|不能|不可|不得|切勿|不要|切忌|严禁|禁止|勿|未|不)\s*"
    r"(?:停用|停药|停服|停掉|停止服用|停止使用)"
)


def find_stop_drug_marker(snippet: str) -> Optional[str]:
    """Return a matched cessation marker only when not negated (e.g. excludes '不能停药'/'至今未停用')."""
    if not snippet:
        return None
    cleaned = _NEGATED_STOP_MARKER_RE.sub("", snippet)
    return next((m for m in STOP_DRUG_MARKERS if m in cleaned), None)

CN_NUM_MAP = {
    "零": 0,
    "半": 0.5,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

DURATION_PATTERN = re.compile(
    r"(?P<num>\d+(?:\.\d+)?|半|[一二两三四五六七八九十]{1,3})\s*"
    r"(?P<unit>天|日|周|星期|个月|月|年)"
)


@lru_cache(maxsize=4)
def load_red_flags_catalog(path: Optional[str] = None) -> RedFlagsCatalog:
    target = Path(path) if path else DEFAULT_RED_FLAGS_PATH
    with target.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return RedFlagsCatalog.model_validate(raw)


@lru_cache(maxsize=4)
def load_drug_conflicts_catalog(path: Optional[str] = None) -> DrugConflictsCatalog:
    target = Path(path) if path else DEFAULT_DRUG_CONFLICTS_PATH
    with target.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return DrugConflictsCatalog.model_validate(raw)


def _parse_cn_or_arabic_number(token: str) -> Optional[float]:
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        pass
    if token in CN_NUM_MAP:
        return float(CN_NUM_MAP[token])
    if token.startswith("十") and len(token) == 2 and token[1] in CN_NUM_MAP:
        return 10.0 + CN_NUM_MAP[token[1]]
    if len(token) == 2 and token[1] == "十" and token[0] in CN_NUM_MAP:
        return float(CN_NUM_MAP[token[0]] * 10)
    if len(token) == 3 and token[1] == "十" and token[0] in CN_NUM_MAP and token[2] in CN_NUM_MAP:
        return float(CN_NUM_MAP[token[0]] * 10 + CN_NUM_MAP[token[2]])
    return None


def parse_duration_days(text_snippet: str) -> Optional[float]:
    """Parse Chinese/Arabic duration phrases (e.g., '半年', '3个月', '2周', '10天') into days."""
    if "半年" in text_snippet:
        return 180.0
    match = DURATION_PATTERN.search(text_snippet)
    if not match:
        return None
    val = _parse_cn_or_arabic_number(match.group("num"))
    if val is None:
        return None
    unit = match.group("unit")
    if unit in ("天", "日"):
        return val
    if unit in ("周", "星期"):
        return val * 7.0
    if unit in ("月", "个月"):
        return val * 30.0
    if unit == "年":
        return val * 365.0
    return None


def _extract_clause_windows(
    text: str,
    start: int,
    end: int,
    pre_window: int = 12,
    post_window: int = 8,
) -> Tuple[str, str, str]:
    """Extract clause-bounded left/right context windows around [start, end)."""
    # Full clause bounded by strong punctuation
    clause_start = start
    while clause_start > 0 and text[clause_start - 1] not in STRONG_BOUNDARY_CHARS:
        clause_start -= 1
    clause_end = end
    while clause_end < len(text) and text[clause_end] not in STRONG_BOUNDARY_CHARS:
        clause_end += 1
    full_clause = text[clause_start:clause_end]

    # Left window: stop at strong boundary OR soft comma (to avoid "无发热，踩棉花感" leaking)
    left_limit = max(clause_start, start - pre_window)
    left_raw = text[left_limit:start]
    for sep in SOFT_CLAUSE_SEPARATORS:
        sep_idx = left_raw.rfind(sep)
        if sep_idx != -1:
            left_raw = left_raw[sep_idx + 1 :]

    # Right window: stop at strong boundary OR soft comma
    right_limit = min(clause_end, end + post_window)
    right_raw = text[end:right_limit]
    for sep in SOFT_CLAUSE_SEPARATORS:
        sep_idx = right_raw.find(sep)
        if sep_idx != -1:
            right_raw = right_raw[:sep_idx]

    return left_raw, right_raw, full_clause


def check_negation_context(
    text: str,
    start: int,
    end: int,
    *,
    global_negation_terms: List[str],
    negation_guards: Optional[List[str]] = None,
    pre_window: int = 12,
    post_window: int = 8,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Determine whether a match span [start, end) is negated in clinical context.

    Returns:
        (is_negated, reason_code, matched_guard_or_term)
    """
    left_win, right_win, full_clause = _extract_clause_windows(
        text, start, end, pre_window=pre_window, post_window=post_window
    )
    local_span = left_win + text[start:end] + right_win
    local_lower = local_span.lower()

    # 1. Rule-specific negation_guards take precedence when overlapping the local span
    for guard in negation_guards or []:
        g_lower = guard.lower()
        if g_lower in local_lower:
            return True, "text_negation_guard", guard

    # 2. Pre-window negation & prohibition terms
    pre_terms = sorted(set(global_negation_terms) | set(PROHIBITION_TERMS), key=len, reverse=True)
    for term in pre_terms:
        if term in left_win:
            return True, "text_negation_window", term

    # 3. Post-window negation markers (e.g., "Hoffmann征阴性", "踩棉花感(-)")
    for term in POST_NEGATION_TERMS:
        if term in right_win:
            return True, "text_negation_window", term

    # 4. Parallel enumeration guard: e.g. "否认头晕、踩棉花感、二便失禁"
    if "、" in full_clause:
        clause_start_idx = start
        while clause_start_idx > 0 and text[clause_start_idx - 1] not in STRONG_BOUNDARY_CHARS:
            clause_start_idx -= 1
        rel_start = start - clause_start_idx
        sub_before = full_clause[:rel_start]
        # Only traverse across '、' (not across '，' unless no positive verb intervenes)
        last_comma = max(sub_before.rfind("，"), sub_before.rfind(","))
        enum_segment = sub_before[last_comma + 1 :]
        for term in pre_terms:
            if term in enum_segment:
                return True, "text_negation_window", term

    return False, None, None


def _check_drug_stopped_in_text(
    text: str,
    start: int,
    end: int,
    washout_window_days: Optional[int],
) -> Tuple[bool, Optional[float], Optional[str]]:
    """Inspect local clause around a drug mention to see if it was stopped beyond washout."""
    left_win, right_win, _ = _extract_clause_windows(text, start, end, pre_window=12, post_window=14)
    local_snippet = left_win + text[start:end] + right_win

    matched_marker = find_stop_drug_marker(local_snippet)
    if not matched_marker:
        return False, None, None

    stopped_days = parse_duration_days(local_snippet)
    washout = washout_window_days if washout_window_days is not None else 0

    if stopped_days is not None:
        if stopped_days > washout:
            return True, stopped_days, f"文本提示 '{matched_marker}' 约 {stopped_days:g} 天，已超过洗脱期 ({washout} 天)"
        # Still inside washout window -> DO NOT suppress!
        return False, stopped_days, f"文本提示 '{matched_marker}' 仅 {stopped_days:g} 天，仍在洗脱期内 (≤ {washout} 天)"

    # Explicit cessation marker without duration: only suppress if washout == 0;
    # otherwise stay conservative (safer for clinical red-flag harness) unless "早已停" / "长期停用".
    if matched_marker in ("早已停", "未服用", "未再服用") or "长期停" in local_snippet:
        return True, None, f"文本明确提示 '{matched_marker}'（非当前在用）"
    return False, None, f"文本提示 '{matched_marker}' 但未明确停药时长，按保守原则仍保留预警"


def _find_keyword_matches(text: str, keywords: List[str]) -> List[Tuple[str, int, int]]:
    """Find case-insensitive literal occurrences of keywords in text."""
    matches: List[Tuple[str, int, int]] = []
    if not text:
        return matches
    for kw in keywords:
        if not kw:
            continue
        for m in re.finditer(re.escape(kw), text, flags=re.IGNORECASE):
            matches.append((kw, m.start(), m.end()))
    return matches


def _term_matches_keyword(fact_name: str, keyword: str) -> bool:
    """Bidirectional normalized substring match between a structured fact name and a rule keyword."""
    a = fact_name.strip().lower()
    b = keyword.strip().lower()
    if not a or not b:
        return False
    return a in b or b in a


def _normalize_structured_facts(
    structured_facts: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Normalize structured_facts into canonical symptoms and medications lists."""
    if not structured_facts or not isinstance(structured_facts, dict):
        return [], []

    symptoms: List[Dict[str, Any]] = []
    for item in structured_facts.get("symptoms", []) or []:
        if isinstance(item, str):
            symptoms.append({"name": item, "polarity": "present"})
        elif isinstance(item, dict) and item.get("name"):
            symptoms.append(
                {
                    "name": str(item["name"]),
                    "polarity": str(item.get("polarity", "present")).lower(),
                }
            )

    medications: List[Dict[str, Any]] = []
    for item in structured_facts.get("medications", []) or []:
        if isinstance(item, str):
            medications.append({"name": item, "status": "active", "stopped_days_ago": None})
        elif isinstance(item, dict) and item.get("name"):
            medications.append(
                {
                    "name": str(item["name"]),
                    "status": str(item.get("status", "active")).lower(),
                    "stopped_days_ago": item.get("stopped_days_ago"),
                }
            )

    return symptoms, medications


def _extract_superseded_rules(
    structured_facts: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Extract superseded rule IDs and human-readable override details from structured_facts."""
    superseded: Dict[str, str] = {}
    if not structured_facts or not isinstance(structured_facts, dict):
        return superseded

    for cond in structured_facts.get("superseded_conditions", []) or []:
        if not isinstance(cond, dict):
            continue
        rule_hint = str(cond.get("rule_hint") or "").strip().upper()
        if not rule_hint:
            continue
        old_state = str(cond.get("old_state") or "既往旧状态").strip()
        new_state = str(cond.get("new_state") or "最新时间线状态").strip()
        superseded[rule_hint] = f"最新时间线状态已覆盖旧约束（{old_state} → {new_state}）"

    for rid in structured_facts.get("superseded_rule_ids", []) or []:
        rule_id = str(rid or "").strip().upper()
        if rule_id and rule_id not in superseded:
            superseded[rule_id] = f"规则 {rule_id} 已被最新时间线状态明确解除或推翻"

    return superseded


def merge_snapshot_into_structured_facts(
    current_state_snapshot: Optional[Dict[str, Any]],
    fallback_facts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map CopilotExtractedNote.current_state_snapshot into Track B structured_facts input.

    Preserves full backward compatibility with legacy `structured_facts` while letting
    chronologically resolved `current_symptoms`, `active_medications`, `stopped_medications`,
    and `superseded_conditions` take precedence.
    """
    base = dict(fallback_facts) if isinstance(fallback_facts, dict) else {}
    snapshot = dict(current_state_snapshot) if isinstance(current_state_snapshot, dict) else {}

    if not snapshot:
        return {
            "symptoms": list(base.get("symptoms") or []),
            "medications": list(base.get("medications") or []),
            **(
                {"superseded_rule_ids": list(base.get("superseded_rule_ids") or [])}
                if base.get("superseded_rule_ids")
                else {}
            ),
            **(
                {"superseded_conditions": list(base.get("superseded_conditions") or [])}
                if base.get("superseded_conditions")
                else {}
            ),
        }

    # 1. Symptoms: snapshot.current_symptoms overrides fallback_facts.symptoms by normalized name
    merged_symptoms: List[Dict[str, Any]] = []
    seen_sym_names: Set[str] = set()
    for item in snapshot.get("current_symptoms") or []:
        if isinstance(item, dict) and item.get("name"):
            key = str(item["name"]).strip().lower()
            if key and key not in seen_sym_names:
                seen_sym_names.add(key)
                merged_symptoms.append(
                    {
                        "name": str(item["name"]).strip(),
                        "polarity": str(item.get("polarity", "present")).lower(),
                    }
                )
    for item in base.get("symptoms") or []:
        if isinstance(item, dict) and item.get("name"):
            key = str(item["name"]).strip().lower()
            if key and key not in seen_sym_names:
                seen_sym_names.add(key)
                merged_symptoms.append(
                    {
                        "name": str(item["name"]).strip(),
                        "polarity": str(item.get("polarity", "present")).lower(),
                    }
                )
        elif isinstance(item, str) and item.strip():
            key = item.strip().lower()
            if key not in seen_sym_names:
                seen_sym_names.add(key)
                merged_symptoms.append({"name": item.strip(), "polarity": "present"})

    # 2. Medications: combine active_medications (incl. in-washout) + stopped_medications, then fallback
    merged_meds: List[Dict[str, Any]] = []
    seen_med_names: Set[str] = set()
    for bucket_key, default_status in (
        ("active_medications", "active"),
        ("stopped_medications", "stopped"),
    ):
        for item in snapshot.get(bucket_key) or []:
            if isinstance(item, dict) and item.get("name"):
                key = str(item["name"]).strip().lower()
                if key and key not in seen_med_names:
                    seen_med_names.add(key)
                    merged_meds.append(
                        {
                            "name": str(item["name"]).strip(),
                            "status": str(item.get("status") or default_status).lower(),
                            "stopped_days_ago": item.get("stopped_days_ago"),
                        }
                    )
    for item in base.get("medications") or []:
        if isinstance(item, dict) and item.get("name"):
            key = str(item["name"]).strip().lower()
            if key and key not in seen_med_names:
                seen_med_names.add(key)
                merged_meds.append(
                    {
                        "name": str(item["name"]).strip(),
                        "status": str(item.get("status", "active")).lower(),
                        "stopped_days_ago": item.get("stopped_days_ago"),
                    }
                )
        elif isinstance(item, str) and item.strip():
            key = item.strip().lower()
            if key not in seen_med_names:
                seen_med_names.add(key)
                merged_meds.append({"name": item.strip(), "status": "active", "stopped_days_ago": None})

    # 3. Superseded conditions & rule hints
    superseded_conditions = list(
        snapshot.get("superseded_conditions") or base.get("superseded_conditions") or []
    )
    superseded_rule_ids: List[str] = []
    for cond in superseded_conditions:
        if isinstance(cond, dict) and cond.get("rule_hint"):
            rid = str(cond["rule_hint"]).strip().upper()
            if rid and rid not in superseded_rule_ids:
                superseded_rule_ids.append(rid)
    for rid in list(snapshot.get("superseded_rule_ids") or []) + list(base.get("superseded_rule_ids") or []):
        r_clean = str(rid or "").strip().upper()
        if r_clean and r_clean not in superseded_rule_ids:
            superseded_rule_ids.append(r_clean)

    result: Dict[str, Any] = {
        "symptoms": merged_symptoms,
        "medications": merged_meds,
        "superseded_conditions": superseded_conditions,
        "superseded_rule_ids": superseded_rule_ids,
    }
    if snapshot.get("active_surgical_history"):
        result["active_surgical_history"] = list(snapshot["active_surgical_history"])
    if snapshot.get("diagnosis_override_note"):
        result["diagnosis_override_note"] = snapshot["diagnosis_override_note"]
    return result


def _scan_red_flags(
    raw_text: str,
    symptoms: List[Dict[str, Any]],
    catalog: RedFlagsCatalog,
    superseded_rules: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    alerts: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    superseded_map = superseded_rules or {}

    present_symptoms = [s["name"] for s in symptoms if s["polarity"] == "present"]
    absent_symptoms = [s["name"] for s in symptoms if s["polarity"] == "absent"]

    for rule in catalog.rules:
        req_keywords = rule.trigger_symptoms.required_any
        sup_keywords = rule.trigger_symptoms.supporting_any

        # --- Track B: Structured facts evaluation ---
        struct_req_hits: List[str] = []
        struct_sup_hits: List[str] = []
        struct_absent_hits: List[str] = []

        for s_name in present_symptoms:
            if any(_term_matches_keyword(s_name, kw) for kw in req_keywords):
                struct_req_hits.append(s_name)
            elif any(_term_matches_keyword(s_name, kw) for kw in sup_keywords):
                struct_sup_hits.append(s_name)

        for s_name in absent_symptoms:
            if any(_term_matches_keyword(s_name, kw) for kw in req_keywords):
                struct_absent_hits.append(s_name)
                suppressed.append(
                    {
                        "rule_id": rule.rule_id,
                        "rule_type": "red_flag",
                        "term": s_name,
                        "reason": "symptom_explicitly_absent",
                        "source": "structured_facts",
                        "detail": f"结构化事实明确标记症状 '{s_name}' 为阴性/不存在 (polarity=absent)",
                    }
                )

        # --- Track A: Text scan with negation window ---
        text_req_hits: List[str] = []
        text_sup_hits: List[str] = []

        for kw, start, end in _find_keyword_matches(raw_text, req_keywords):
            # If Track B explicitly marked this symptom absent, suppress text hit
            if any(_term_matches_keyword(abs_name, kw) for abs_name in struct_absent_hits):
                suppressed.append(
                    {
                        "rule_id": rule.rule_id,
                        "rule_type": "red_flag",
                        "term": kw,
                        "reason": "overridden_by_structured_absent",
                        "source": "structured_facts",
                        "detail": f"文本命中 '{kw}'，但结构化事实明确为阴性，已抑制告警",
                    }
                )
                continue

            is_neg, neg_reason, neg_token = check_negation_context(
                raw_text,
                start,
                end,
                global_negation_terms=catalog.global_negation_terms,
                negation_guards=rule.negation_guards,
            )
            if is_neg:
                suppressed.append(
                    {
                        "rule_id": rule.rule_id,
                        "rule_type": "red_flag",
                        "term": kw,
                        "reason": neg_reason or "text_negation_window",
                        "source": neg_reason or "text_negation_window",
                        "detail": f"命中否定语境 '{neg_token}'，已抑制 '{kw}' 误报",
                    }
                )
            else:
                if kw not in text_req_hits:
                    text_req_hits.append(kw)

        for kw, start, end in _find_keyword_matches(raw_text, sup_keywords):
            is_neg, _, _ = check_negation_context(
                raw_text,
                start,
                end,
                global_negation_terms=catalog.global_negation_terms,
                negation_guards=rule.negation_guards,
            )
            if not is_neg and kw not in text_sup_hits:
                text_sup_hits.append(kw)

        # Trigger condition: at least one required_any hit from Track B or Track A
        if struct_req_hits or text_req_hits:
            if rule.rule_id in superseded_map:
                candidate_terms = "、".join(dict.fromkeys(struct_req_hits + text_req_hits))
                suppressed.append(
                    {
                        "rule_id": rule.rule_id,
                        "rule_type": "red_flag",
                        "term": candidate_terms or rule.name,
                        "reason": "superseded_by_latest_timeline_state",
                        "source": "structured_facts",
                        "detail": superseded_map[rule.rule_id],
                    }
                )
                continue

            if struct_req_hits and text_req_hits:
                source = "hybrid"
            elif struct_req_hits:
                source = "structured_facts"
            else:
                source = "text_scan"

            evidence = list(dict.fromkeys(struct_req_hits + text_req_hits + struct_sup_hits + text_sup_hits))
            alerts.append(
                {
                    "rule_id": rule.rule_id,
                    "rule_type": "red_flag",
                    "name": rule.name,
                    "category": rule.category,
                    "severity": rule.severity,
                    "source": source,
                    "matched_evidence": evidence,
                    "clinical_action": rule.recommended_action,
                    "clinical_alert": rule.clinical_alert,
                    "contraindicated_actions": list(rule.contraindicated_actions),
                    "clinical_rationale": rule.clinical_rationale,
                    "reference": list(rule.references),
                }
            )

    return alerts, suppressed


def _collect_group_hits(
    raw_text: str,
    medications: List[Dict[str, Any]],
    keywords: List[str],
    washout_window_days: Optional[int],
    conflict_id: str,
    global_negation_terms: List[str],
    suppressed: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collect active/in-washout hits for a drug group across Track B and Track A."""
    active_hits: List[Dict[str, Any]] = []
    suppressed_terms: Set[str] = set()

    # 1. Track B structured medications first (authoritative when present)
    for med in medications:
        med_name = med["name"]
        matched_kw = next((kw for kw in keywords if _term_matches_keyword(med_name, kw)), None)
        if not matched_kw:
            continue

        status = med.get("status", "active")
        stopped_days = med.get("stopped_days_ago")
        washout = washout_window_days if washout_window_days is not None else 0

        if status == "stopped":
            if isinstance(stopped_days, (int, float)) and not isinstance(stopped_days, bool):
                if stopped_days > washout:
                    suppressed_terms.add(matched_kw.lower())
                    suppressed_terms.add(med_name.lower())
                    suppressed.append(
                        {
                            "rule_id": conflict_id,
                            "rule_type": "drug_conflict",
                            "term": med_name,
                            "reason": "drug_stopped_beyond_washout",
                            "source": "structured_facts",
                            "detail": (
                                f"药物 '{med_name}' 已停用 {stopped_days:g} 天，"
                                f"超过洗脱期 ({washout} 天)，已过滤既往用药冲突"
                            ),
                        }
                    )
                    continue
                # Stopped but still inside washout window
                active_hits.append(
                    {
                        "entity": med_name,
                        "keyword": matched_kw,
                        "source": "structured_facts",
                        "note": f"停药仅 {stopped_days:g} 天，仍在洗脱期内 (≤ {washout} 天)",
                    }
                )
                continue
            else:
                # Stopped without days specified -> keep conservative warning or suppress if washout==0
                if washout == 0:
                    suppressed_terms.add(matched_kw.lower())
                    suppressed_terms.add(med_name.lower())
                    suppressed.append(
                        {
                            "rule_id": conflict_id,
                            "rule_type": "drug_conflict",
                            "term": med_name,
                            "reason": "drug_stopped_no_washout_required",
                            "source": "structured_facts",
                            "detail": f"药物 '{med_name}' 状态为已停用且无洗脱期要求",
                        }
                    )
                    continue
                active_hits.append(
                    {
                        "entity": med_name,
                        "keyword": matched_kw,
                        "source": "structured_facts",
                        "note": f"标记已停药但未提供停药天数，按洗脱期 ({washout} 天) 保守预警",
                    }
                )
                continue

        if status in ("active", "unknown"):
            active_hits.append(
                {
                    "entity": med_name,
                    "keyword": matched_kw,
                    "source": "structured_facts",
                    "note": "当前在用" if status == "active" else "用药状态未明（保守纳入）",
                }
            )

    # 2. Track A raw text scan (respecting Track B overrides, negation, and cessation phrases)
    seen_entities = {h["entity"].lower() for h in active_hits}
    for kw, start, end in _find_keyword_matches(raw_text, keywords):
        kw_lower = kw.lower()
        if kw_lower in suppressed_terms or any(kw_lower in st or st in kw_lower for st in suppressed_terms):
            continue
        if kw_lower in seen_entities or any(kw_lower in se or se in kw_lower for se in seen_entities):
            continue

        # Check explicit drug cessation in text
        is_stopped, stopped_days, stop_detail = _check_drug_stopped_in_text(
            raw_text, start, end, washout_window_days
        )
        if is_stopped:
            suppressed.append(
                {
                    "rule_id": conflict_id,
                    "rule_type": "drug_conflict",
                    "term": kw,
                    "reason": "text_drug_washout",
                    "source": "text_drug_washout",
                    "detail": stop_detail or f"'{kw}' 已停药超过洗脱期",
                }
            )
            continue

        # Check negation (e.g., "否认服用华法林", "未使用抗凝药")
        is_neg, neg_reason, neg_token = check_negation_context(
            raw_text,
            start,
            end,
            global_negation_terms=global_negation_terms,
            negation_guards=None,
        )
        if is_neg:
            suppressed.append(
                {
                    "rule_id": conflict_id,
                    "rule_type": "drug_conflict",
                    "term": kw,
                    "reason": neg_reason or "text_negation_window",
                    "source": neg_reason or "text_negation_window",
                    "detail": f"命中否定语境 '{neg_token}'，已抑制药物 '{kw}' 误报",
                }
            )
            continue

        seen_entities.add(kw_lower)
        active_hits.append(
            {
                "entity": kw,
                "keyword": kw,
                "source": "text_scan",
                "note": stop_detail or "文本检出",
            }
        )

    return active_hits


def _has_distinct_pair(hits_a: List[Dict[str, Any]], hits_b: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Ensure group_a and group_b match two distinct entities (prevents single drug self-conflict)."""
    for ha in hits_a:
        ea = ha["entity"].strip().lower()
        ka = ha["keyword"].strip().lower()
        for hb in hits_b:
            eb = hb["entity"].strip().lower()
            kb = hb["keyword"].strip().lower()
            if ea == eb or ka == kb or ea in eb or eb in ea:
                continue
            return ha, hb
    return None


def _scan_drug_conflicts(
    raw_text: str,
    medications: List[Dict[str, Any]],
    catalog: DrugConflictsCatalog,
    global_negation_terms: List[str],
    superseded_rules: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    alerts: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    superseded_map = superseded_rules or {}

    for rule in catalog.conflicts:
        hits_a = _collect_group_hits(
            raw_text=raw_text,
            medications=medications,
            keywords=rule.group_a.keywords,
            washout_window_days=rule.group_a.washout_window_days,
            conflict_id=rule.conflict_id,
            global_negation_terms=global_negation_terms,
            suppressed=suppressed,
        )
        hits_b = _collect_group_hits(
            raw_text=raw_text,
            medications=medications,
            keywords=rule.group_b.keywords,
            washout_window_days=rule.group_b.washout_window_days,
            conflict_id=rule.conflict_id,
            global_negation_terms=global_negation_terms,
            suppressed=suppressed,
        )

        pair = _has_distinct_pair(hits_a, hits_b)
        if not pair:
            continue

        ha, hb = pair
        if rule.conflict_id in superseded_map:
            suppressed.append(
                {
                    "rule_id": rule.conflict_id,
                    "rule_type": "drug_conflict",
                    "term": f"{ha['entity']} + {hb['entity']}",
                    "reason": "superseded_by_latest_timeline_state",
                    "source": "structured_facts",
                    "detail": superseded_map[rule.conflict_id],
                }
            )
            continue

        sources = {h["source"] for h in (hits_a + hits_b)}
        if sources == {"structured_facts"}:
            source = "structured_facts"
        elif sources == {"text_scan"}:
            source = "text_scan"
        else:
            source = "hybrid"

        evidence = list(
            dict.fromkeys(
                [f"{ha['entity']}（{ha['note']}）", f"{hb['entity']}（{hb['note']}）"]
            )
        )
        alerts.append(
            {
                "rule_id": rule.conflict_id,
                "rule_type": "drug_conflict",
                "name": rule.name,
                "category": "中西医结合用药冲突",
                "severity": rule.severity,
                "source": source,
                "matched_evidence": evidence,
                "clinical_action": "STOP_AND_VERIFY_MEDS",
                "clinical_alert": rule.clinical_alert,
                "suggested_handling": rule.suggested_handling,
                "risk_mechanism": rule.risk_mechanism,
                "reference": list(rule.references),
            }
        )

    return alerts, suppressed


def scan_clinical_risks(
    raw_text: str = "",
    structured_facts: Optional[Dict[str, Any]] = None,
    *,
    red_flags_path: Optional[str] = None,
    drug_conflicts_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan clinical text and/or structured facts for red flags and drug conflicts.

    Args:
        raw_text: Unstructured clinical note or conversation text.
        structured_facts: Optional dict with 'symptoms' (list of {name, polarity}),
            'medications' (list of {name, status, stopped_days_ago}), and optional
            'superseded_rule_ids' / 'superseded_conditions' from timeline resolution.
        red_flags_path: Optional custom path to red_flags.yaml.
        drug_conflicts_path: Optional custom path to drug_conflicts.yaml.

    Returns:
        Unified deterministic risk report containing alerts, suppressed_matches,
        overall risk_level, and counts.
    """
    rf_catalog = load_red_flags_catalog(red_flags_path)
    dc_catalog = load_drug_conflicts_catalog(drug_conflicts_path)

    symptoms, medications = _normalize_structured_facts(structured_facts)
    superseded_rules = _extract_superseded_rules(structured_facts)
    safe_text = raw_text or ""

    rf_alerts, rf_suppressed = _scan_red_flags(
        safe_text, symptoms, rf_catalog, superseded_rules=superseded_rules
    )
    dc_alerts, dc_suppressed = _scan_drug_conflicts(
        safe_text,
        medications,
        dc_catalog,
        rf_catalog.global_negation_terms,
        superseded_rules=superseded_rules,
    )

    all_alerts = sorted(
        rf_alerts + dc_alerts,
        key=lambda item: (-SEVERITY_RANK.get(item["severity"], 0), item["rule_id"]),
    )
    all_suppressed = rf_suppressed + dc_suppressed

    if not all_alerts:
        risk_level = "none"
    else:
        risk_level = all_alerts[0]["severity"]

    return {
        "status": "ok",
        "has_critical_risk": any(a["severity"] == "critical" for a in all_alerts),
        "risk_level": risk_level,
        "alerts": all_alerts,
        "suppressed_matches": all_suppressed,
        "summary": {
            "total_alerts": len(all_alerts),
            "red_flag_count": len(rf_alerts),
            "drug_conflict_count": len(dc_alerts),
            "suppressed_count": len(all_suppressed),
        },
    }
