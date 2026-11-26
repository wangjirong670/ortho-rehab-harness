"""Deterministic dual-track clinical risk scanner (hardened).

Track A (Text Scan + Negation & Washout Window):
    Scans raw clinical text against knowledge_base/red_flags.yaml and
    knowledge_base/drug_conflicts.yaml with rule-specific negation_guards,
    bidirectional clause-bounded negation windows (with double-negation,
    onset-verb, and composite-word protections), regex keyword support,
    and scoped drug cessation duration parsing against washout_window_days.

Track B (Structured Facts Polarity & Medication Status):
    Incorporates structured clinical facts while enforcing deterministic
    trust boundaries: critical rules can never be suppressed by LLM timeline
    overrides, absent/stopped claims conflicting with raw text are flagged
    with conflict_with_text=True and needs_review=True, and only whitelisted
    staging rules with non-interrogative text evidence may be superseded.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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

# Only non-critical staging-protection rules may ever be superseded by timeline resolution.
SUPERSEDABLE_RULE_WHITELIST: frozenset[str] = frozenset({"RF_007"})

# Strong clause boundaries that stop negation scope propagation
STRONG_BOUNDARY_CHARS = "。！？；;\n"
SOFT_CLAUSE_SEPARATORS = "，,"

# Left-window reset tokens (comma, colon, semicolon, adversative/temporal shifts)
_LEFT_WINDOW_RESET_RE = re.compile(
    r"(?:但是|但|然而|今日|现在|(?<![出发表实体再重呈])现|：|:|；|;|，|,)"
)
_RIGHT_CLAUSE_CUT_RE = re.compile(r"[,，；;：:]")

# Additional prohibitive/cautionary negation terms (e.g., "切忌在家自行复位")
PROHIBITION_TERMS = (
    "切忌",
    "严禁",
    "禁止",
    "不要",
    "不可以",
    "不能",
    "不可",
    "不得",
    "切勿",
    "勿",
    "避免",
    "不宜",
    "禁忌",
)

# Built-in regex guards for negation terms when callers pass raw literal strings
_LITERAL_NEGATION_OVERRIDES: Dict[str, re.Pattern[str]] = {
    "无": re.compile(r"(?<!毫)无(?!力|诱因|疑|毫|法|效|故|谓|菌|痛|非|论|需|须|怪|妨|奈|聊|辜|可|所谓)"),
    "无明显": re.compile(r"无明显(?!诱因)"),
    "排除": re.compile(r"(?<!待)(?<!未)(?<!需)(?<!不)(?<!难)排除"),
    "未行": re.compile(r"未行(?!走)"),
}

# Double-negation patterns that force affirmative interpretation (e.g. "不排除马尾综合征", "尚不能排除")
_DOUBLE_NEGATION_RE = re.compile(
    r"(?:(?:(?:不能|尚不能|未能|无法|尚无法|难以)?不|(?:不能|尚不能|未能|无法|尚无法|难以))(?:排除|除外)|(?:并非|不是|绝非)(?:无|没有))"
)

# Onset / progression verbs between negation and keyword invalidate negation
_ONSET_VERB_RE = re.compile(r"(?:出现|突发|发生|新发|加重|开始|进展为|继发|诱发)")
_RIGHT_WORSENING_RE = re.compile(
    r"^[\s，,、]*(?:仍|依然|持续|进行性|明显)?(?:加重|恶化|进展|频发|加剧)"
)

# Positive modals / affirmative suffixes used when checking replies/drafts (P1-2 / P0-5)
_POSITIVE_MODAL_BREAK_RE = re.compile(r"(?:可以|能|行|尽管|直接|试试|去做|安排|一起吃|同服|松解)")
_RIGHT_AFFIRM_BREAK_RE = re.compile(
    r"^[^，。；！？\n]{0,8}?(?:即可|就行|没问题|也不迟|都可以|安全)"
)

# Post-window negation patterns
_POST_POLARITY_RE = re.compile(r"(?:阴性|正常|\(-\)|（-）|无异常)")
_POST_EXCLUSION_RE = re.compile(
    r"(?:已排除|不存在|(?:未见|未(?!见))(?!(?:明显|完全|彻底)?(?:缓解|好转|改善|消退|消失|恢复|减轻|控制)))"
)
_POST_PROHIBITION_RE = re.compile(
    r"(?:都是|也是|绝)?(?:不可以的|不可以|不行|不可取|不能|不得|禁止|严禁|切勿)"
)
_AFFIRMATIVE_TAIL_RE = re.compile(r"(?:阳性|\(\+\)|（\+）|亢进)$")
_BENIGN_PRE_POLARITY_GAP_RE = re.compile(r"^[\s:：是为呈均较前基本完全大致尚可]{0,3}$")
_BENIGN_PRE_EXCLUSION_GAP_RE = re.compile(r"^[\s:：是为均较前基本可暂目前]{0,4}$")
_BENIGN_PRE_PROHIBITION_GAP_RE = re.compile(r"^[^，。；！？\n]{0,12}?(?:都|也|均)?(?:是)?$")

_QUOTED_SEGMENT_RE = re.compile(r"「[^」]*」|『[^』]*』|“[^”]*”|\"[^\"]*\"|‘[^’]*’")

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
    "早已停",
    "未再服用",
    "停止服用",
    "停止使用",
    "未服用",
    "停用",
    "停药",
    "停服",
    "停掉",
    "已停",
)

_NEGATED_STOP_MARKER_RE = re.compile(
    r"(?:至今未|并未|尚未|没有|不能|不可|不得|切勿|不要|切忌|严禁|禁止|勿|未|别|莫|没|不)"
    r"[^，。；：、！？\n]{0,4}?"
    r"(?:停用|停药|停服|停掉|停止服用|停止使用)"
)

_PLANNED_STOP_MARKER_RE = re.compile(
    r"(?:建议|拟|计划|准备|择期|考虑|打算|欲|需|需要|应该|应当)"
    r"[^，。；：、！？\n]{0,4}?"
    r"(?:停用|停药|停服|停掉|停止服用|停止使用|停)"
)

_RESUME_DRUG_MARKER_RE = re.compile(
    r"(?:恢复(?:服用|用药|使用|吃)?|重新(?:服用|用药|使用|启用|吃)?|续用|再次服用|再次使用|再服|复用|复服|继续服用)"
)
_NEGATED_RESUME_PREFIX_RE = re.compile(r"(?:未|没|切勿|不要|不能|不可|禁止|严禁)[^，。；\n]{0,3}$")
_DRUG_SWITCH_RE = re.compile(r"(?:改用|换用|换成|改为|替换为|改服|接替)")

# Timeline supersede clearance markers
_TIMELINE_CLEARANCE_MARKERS = (
    "已缓解",
    "达标",
    "准予",
    "批准",
    "康复评估通过",
    "通过",
    "重返运动",
    "解除",
)
_INTERROGATIVE_HINT_RE = re.compile(r"[?？]|(?:询问|吗|呢|能否|可以吗|是不是|要等到)")

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
    r"(?:(?P<half_before>个半)|(?P<unit_prefix>个))?\s*"
    r"(?P<unit>小时|天|日|周|星期|月|年)\s*"
    r"(?P<half_after>半)?"
)


def strip_quoted_segments(text: str, *, preserve_length: bool = False) -> str:
    """Remove or mask quoted segments (e.g., 「明天再去」, “可以按摩”) before scanning."""
    if not text:
        return ""
    if preserve_length:
        return _QUOTED_SEGMENT_RE.sub(lambda m: " " * (m.end() - m.start()), text)
    return _QUOTED_SEGMENT_RE.sub("", text)


def _has_active_resume_marker(text: str) -> bool:
    """Return True if text contains a non-negated medication resumption phrase."""
    for m in _RESUME_DRUG_MARKER_RE.finditer(text):
        prefix = text[max(0, m.start() - 6) : m.start()]
        if _NEGATED_RESUME_PREFIX_RE.search(prefix):
            continue
        return True
    return False


def find_stop_drug_marker(snippet: str) -> Optional[str]:
    """Return a matched cessation marker only when not negated, planned, or resumed."""
    if not snippet:
        return None
    cleaned = _NEGATED_STOP_MARKER_RE.sub("", snippet)
    cleaned = _PLANNED_STOP_MARKER_RE.sub("", cleaned)

    best_marker: Optional[str] = None
    best_pos = -1
    best_end = -1
    for marker in STOP_DRUG_MARKERS:
        idx = cleaned.rfind(marker)
        if idx > best_pos:
            best_pos = idx
            best_end = idx + len(marker)
            best_marker = marker

    if best_marker is None:
        return None

    tail = cleaned[best_end:]
    if _has_active_resume_marker(tail):
        return None
    return best_marker


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


def _collect_all_drug_keywords(catalog: Optional[DrugConflictsCatalog] = None) -> List[str]:
    cat = catalog or load_drug_conflicts_catalog()
    kws: List[str] = []
    for conflict in cat.conflicts:
        for kw in conflict.group_a.keywords + conflict.group_b.keywords:
            if kw and not kw.startswith("regex:"):
                kws.append(kw)
    return list(dict.fromkeys(kws))


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
    """Parse Chinese/Arabic duration phrases (e.g., '半年', '一周半', '1个半月', '72小时') into days."""
    if not text_snippet:
        return None
    if "半年" in text_snippet:
        return 180.0
    match = DURATION_PATTERN.search(text_snippet)
    if not match:
        return None
    num_token = match.group("num")
    val = _parse_cn_or_arabic_number(num_token)
    if val is None:
        return None
    if (match.group("half_before") or match.group("half_after")) and num_token != "半":
        val += 0.5
    unit = match.group("unit")
    if unit == "小时":
        return round(val / 24.0, 4)
    if unit in ("天", "日"):
        return val
    if unit in ("周", "星期"):
        return val * 7.0
    if unit == "月":
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
    clause_start = start
    while clause_start > 0 and text[clause_start - 1] not in STRONG_BOUNDARY_CHARS:
        clause_start -= 1
    clause_end = end
    while clause_end < len(text) and text[clause_end] not in STRONG_BOUNDARY_CHARS:
        clause_end += 1
    full_clause = text[clause_start:clause_end]

    # Left window: stop at strong boundary OR any left-reset separator/transition
    left_limit = max(clause_start, start - pre_window)
    left_raw = text[left_limit:start]
    last_reset_end = 0
    for m in _LEFT_WINDOW_RESET_RE.finditer(left_raw):
        last_reset_end = m.end()
    if last_reset_end > 0:
        left_raw = left_raw[last_reset_end:]

    # Right window: stop at strong boundary OR comma/semicolon/colon
    right_limit = min(clause_end, end + post_window)
    right_raw = text[end:right_limit]
    cut_m = _RIGHT_CLAUSE_CUT_RE.search(right_raw)
    if cut_m:
        right_raw = right_raw[: cut_m.start()]

    return left_raw, right_raw, full_clause


def _compile_negation_pattern(term: str) -> re.Pattern[str]:
    if term.startswith("regex:"):
        return re.compile(term[6:], flags=re.IGNORECASE)
    if term in _LITERAL_NEGATION_OVERRIDES:
        return _LITERAL_NEGATION_OVERRIDES[term]
    return re.compile(re.escape(term), flags=re.IGNORECASE)


def _match_left_negation(
    left_win: str,
    pre_terms: Sequence[str],
    *,
    right_win: str = "",
    break_on_positive_modals: bool = False,
) -> Optional[str]:
    """Find a valid negation term in left_win that is not invalidated by onset verbs or modals."""
    if not left_win:
        return None
    for term in pre_terms:
        if not term:
            continue
        pat = _compile_negation_pattern(term)
        matches = list(pat.finditer(left_win))
        if not matches:
            continue
        # Inspect the rightmost occurrence of this negation pattern in left_win
        m = matches[-1]
        matched_token = m.group(0)
        after_neg = left_win[m.end() :]

        # Check onset verbs occurring AFTER the negation token
        if _ONSET_VERB_RE.search(after_neg):
            is_direct_neg_onset = (
                matched_token in ("未见", "未", "并未", "尚未", "至今未", "未曾", "没有", "无")
                and bool(re.match(r"^(?:明显)?(?:出现|新发|发生)$", after_neg.strip()))
            )
            if not is_direct_neg_onset:
                continue

        if break_on_positive_modals:
            if _POSITIVE_MODAL_BREAK_RE.search(after_neg):
                continue
            if _RIGHT_AFFIRM_BREAK_RE.search(right_win):
                continue

        return matched_token
    return None


def _match_post_negation(
    matched_text: str,
    right_win: str,
    *,
    allow_post_prohibition: bool = False,
) -> Optional[str]:
    """Match post-window negation without misclassifying adjacent clinical signs."""
    if not right_win:
        return None

    # 1. Exclusion/absence markers (e.g., "骨折已排除", "血栓未见", excluding "未见缓解/好转")
    for m in _POST_EXCLUSION_RE.finditer(right_win):
        gap = right_win[: m.start()]
        if _BENIGN_PRE_EXCLUSION_GAP_RE.match(gap):
            return m.group(0)

    # 2. Polarity markers (阴性/正常/(-)/无异常) - never override explicit "+/阳性" matches
    if not _AFFIRMATIVE_TAIL_RE.search(matched_text):
        for m in _POST_POLARITY_RE.finditer(right_win):
            gap = right_win[: m.start()]
            if _BENIGN_PRE_POLARITY_GAP_RE.match(gap):
                return m.group(0)

    # 3. Post-position prohibitive phrasing (for reply/draft safety checks)
    if allow_post_prohibition:
        for m in _POST_PROHIBITION_RE.finditer(right_win):
            gap = right_win[: m.start()]
            if _BENIGN_PRE_PROHIBITION_GAP_RE.match(gap):
                return m.group(0)

    return None


def _match_negation_guard(guard: str, local_span: str) -> bool:
    if not guard:
        return False
    if guard.startswith("regex:"):
        return bool(re.search(guard[6:], local_span, flags=re.IGNORECASE))
    return guard.lower() in local_span.lower()


def check_negation_context(
    text: str,
    start: int,
    end: int,
    *,
    global_negation_terms: Sequence[str],
    negation_guards: Optional[Sequence[str]] = None,
    extra_negation_terms: Optional[Sequence[str]] = None,
    pre_window: int = 12,
    post_window: int = 8,
    break_on_positive_modals: bool = False,
    allow_post_prohibition: bool = False,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Determine whether a match span [start, end) is negated in clinical context.

    Returns:
        (is_negated, reason_code, matched_guard_or_term)
    """
    if not text or start < 0 or end > len(text) or start >= end:
        return False, None, None

    matched_text = text[start:end]
    left_win, right_win, full_clause = _extract_clause_windows(
        text, start, end, pre_window=pre_window, post_window=post_window
    )
    local_span = left_win + matched_text + right_win

    # 0. Force affirmative on double negation or explicit right-hand worsening
    if _DOUBLE_NEGATION_RE.search(left_win) or _DOUBLE_NEGATION_RE.search(right_win):
        return False, None, None
    if _RIGHT_WORSENING_RE.match(right_win):
        return False, None, None

    # 1. Rule-specific negation_guards take precedence when overlapping the local span
    for guard in negation_guards or []:
        if _match_negation_guard(guard, local_span):
            return True, "text_negation_guard", guard

    # 2. Pre-window negation & prohibition terms
    all_pre_terms = set(global_negation_terms) | set(PROHIBITION_TERMS) | set(extra_negation_terms or ())
    pre_terms = sorted(all_pre_terms, key=len, reverse=True)
    matched_pre = _match_left_negation(
        left_win,
        pre_terms,
        right_win=right_win,
        break_on_positive_modals=break_on_positive_modals,
    )
    if matched_pre:
        return True, "text_negation_window", matched_pre

    # 3. Post-window negation markers (e.g., "Hoffmann征阴性", "踩棉花感(-)")
    matched_post = _match_post_negation(
        matched_text,
        right_win,
        allow_post_prohibition=allow_post_prohibition,
    )
    if matched_post:
        return True, "text_negation_window", matched_post

    # 4. Parallel enumeration guard: e.g. "否认头晕、踩棉花感、二便失禁"
    if "、" in full_clause:
        clause_start_idx = start
        while clause_start_idx > 0 and text[clause_start_idx - 1] not in STRONG_BOUNDARY_CHARS:
            clause_start_idx -= 1
        rel_start = start - clause_start_idx
        sub_before = full_clause[:rel_start]
        last_reset = 0
        for m in _LEFT_WINDOW_RESET_RE.finditer(sub_before):
            last_reset = m.end()
        enum_segment = sub_before[last_reset:]
        if not _DOUBLE_NEGATION_RE.search(enum_segment):
            matched_enum = _match_left_negation(
                enum_segment,
                pre_terms,
                right_win=right_win,
                break_on_positive_modals=break_on_positive_modals,
            )
            if matched_enum:
                return True, "text_negation_window", matched_enum

    return False, None, None


def is_context_negated(
    text: str,
    start: int,
    end: int,
    *,
    global_negation_terms: Optional[Sequence[str]] = None,
    extra_negation_terms: Optional[Sequence[str]] = None,
    negation_guards: Optional[Sequence[str]] = None,
    pre_window: int = 16,
    post_window: int = 14,
    break_on_positive_modals: bool = True,
    allow_post_prohibition: bool = True,
) -> bool:
    """Public boolean helper for evaluator & draft verifier to test if [start, end) is negated."""
    terms = (
        list(global_negation_terms)
        if global_negation_terms is not None
        else list(load_red_flags_catalog().global_negation_terms)
    )
    is_neg, _, _ = check_negation_context(
        text,
        start,
        end,
        global_negation_terms=terms,
        negation_guards=negation_guards,
        extra_negation_terms=extra_negation_terms,
        pre_window=pre_window,
        post_window=post_window,
        break_on_positive_modals=break_on_positive_modals,
        allow_post_prohibition=allow_post_prohibition,
    )
    return is_neg


def _extract_scoped_drug_clause(
    text: str,
    start: int,
    end: int,
    other_drug_keywords: Sequence[str],
) -> Tuple[str, str]:
    """Extract comma/semicolon/switch-bounded clause around drug [start, end) and sentence tail.

    Prevents stop markers or durations from leaking across other drugs or comma clauses.
    """
    clause_start = start
    while clause_start > 0 and text[clause_start - 1] not in "。！？；;\n，,：:":
        clause_start -= 1
    clause_end = end
    while clause_end < len(text) and text[clause_end] not in "。！？；;\n，,：:":
        clause_end += 1

    left_part = text[clause_start:start]
    right_part = text[end:clause_end]

    # Cut at drug transition verbs (e.g. "华法林已停3月改用利伐沙班")
    switch_matches = list(_DRUG_SWITCH_RE.finditer(left_part))
    if switch_matches:
        left_part = left_part[switch_matches[-1].end() :]
    right_switch = _DRUG_SWITCH_RE.search(right_part)
    if right_switch:
        right_part = right_part[: right_switch.start()]

    # Cut at any intervening distinct drug keyword
    current_drug_lower = text[start:end].lower()
    for other_kw in other_drug_keywords:
        okw_lower = other_kw.lower()
        if not okw_lower or okw_lower in current_drug_lower or current_drug_lower in okw_lower:
            continue
        for m in re.finditer(re.escape(other_kw), left_part, flags=re.IGNORECASE):
            left_part = left_part[m.end() :]
        m_right = re.search(re.escape(other_kw), right_part, flags=re.IGNORECASE)
        if m_right:
            right_part = right_part[: m_right.start()]

    scoped_clause = left_part + text[start:end] + right_part

    # Sentence tail after clause_end (bounded by sentence end and other drug mentions) for resume check
    sent_end = clause_end
    while sent_end < len(text) and text[sent_end] not in "。！？\n":
        sent_end += 1
    sentence_tail = text[clause_end:sent_end]
    for other_kw in other_drug_keywords:
        okw_lower = other_kw.lower()
        if not okw_lower or okw_lower in current_drug_lower or current_drug_lower in okw_lower:
            continue
        m_tail = re.search(re.escape(other_kw), sentence_tail, flags=re.IGNORECASE)
        if m_tail:
            sentence_tail = sentence_tail[: m_tail.start()]

    return scoped_clause, sentence_tail


def _check_drug_stopped_in_text(
    text: str,
    start: int,
    end: int,
    washout_window_days: Optional[int],
    other_drug_keywords: Optional[Sequence[str]] = None,
) -> Tuple[bool, Optional[float], Optional[str], bool]:
    """Inspect scoped clause around a drug mention to see if it was validly stopped.

    Returns:
        (suppressed_beyond_washout, stopped_days, detail, has_valid_stop_marker)
    """
    drug_kws = other_drug_keywords if other_drug_keywords is not None else _collect_all_drug_keywords()
    scoped_clause, sentence_tail = _extract_scoped_drug_clause(text, start, end, drug_kws)

    matched_marker = find_stop_drug_marker(scoped_clause)
    if not matched_marker:
        return False, None, None, False

    if _has_active_resume_marker(sentence_tail):
        return False, None, "文本提示停药后已恢复服用，按当前在用处理", False

    stopped_days = parse_duration_days(scoped_clause)
    washout = washout_window_days if washout_window_days is not None else 0

    if stopped_days is not None:
        if stopped_days > washout:
            return (
                True,
                stopped_days,
                f"文本提示 '{matched_marker}' 约 {stopped_days:g} 天，已超过洗脱期 ({washout} 天)",
                True,
            )
        return (
            False,
            stopped_days,
            f"文本提示 '{matched_marker}' 仅 {stopped_days:g} 天，仍在洗脱期内 (≤ {washout} 天)",
            True,
        )

    if matched_marker in ("早已停", "未服用", "未再服用") or "长期停" in scoped_clause:
        return True, None, f"文本明确提示 '{matched_marker}'（非当前在用）", True
    return False, None, f"文本提示 '{matched_marker}' 但未明确停药时长，按保守原则仍保留预警", True


def _find_keyword_matches(text: str, keywords: List[str]) -> List[Tuple[str, int, int]]:
    """Find case-insensitive literal or 'regex:' occurrences of keywords in text."""
    matches: List[Tuple[str, int, int]] = []
    if not text:
        return matches
    for kw in keywords:
        if not kw:
            continue
        if kw.startswith("regex:"):
            pattern = kw[6:]
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                matches.append((m.group(0), m.start(), m.end()))
        else:
            for m in re.finditer(re.escape(kw), text, flags=re.IGNORECASE):
                matches.append((kw, m.start(), m.end()))
    return matches


def _term_matches_keyword(fact_name: str, keyword: str) -> bool:
    """Normalized match between a structured fact name and a rule keyword (literal or regex:)."""
    a = fact_name.strip().lower()
    b = keyword.strip()
    if not a or not b:
        return False
    if b.startswith("regex:"):
        return bool(re.search(b[6:], fact_name, flags=re.IGNORECASE))
    b_low = b.lower()
    return a in b_low or b_low in a


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


_NEGATED_CLEARANCE_PREFIX_RE = re.compile(
    r"(?:尚未|暂未|暂不|仍未|仍不|并未|未能|不可|不得|禁止|严禁|拒绝|没有|未|不|无)[^，,;；:：、\n]{0,4}$"
)
_NEGATED_CLEARANCE_SUFFIX_RE = re.compile(r"^(?:失败|不了|未果)")


def _has_valid_supersede_text_evidence(raw_text: str, rule_id: str) -> bool:
    """Verify the latest clinical visit/paragraph contains non-negated, non-interrogative clearance."""
    if not raw_text or not raw_text.strip():
        return False
    blocks = [b.strip() for b in re.split(r"\n+|(?=【)", raw_text) if b.strip()]
    if not blocks:
        return False

    visit_blocks = [b for b in blocks if b.startswith("【")]
    target_block = visit_blocks[-1] if visit_blocks else blocks[-1]

    for sent in re.split(r"[。！!\n]+", target_block):
        sent = sent.strip()
        if not sent:
            continue
        if _INTERROGATIVE_HINT_RE.search(sent):
            continue
        if rule_id == "RF_007" and "严禁跑跳" in sent and "早期保护期" in sent:
            continue
        for marker in _TIMELINE_CLEARANCE_MARKERS:
            for m in re.finditer(re.escape(marker), sent):
                left_ctx = sent[max(0, m.start() - 8) : m.start()]
                sep_idx = max((left_ctx.rfind(c) for c in "，,;；:：、"), default=-1)
                if sep_idx != -1:
                    left_ctx = left_ctx[sep_idx + 1 :]
                right_ctx = sent[m.end() : m.end() + 4]
                if _NEGATED_CLEARANCE_PREFIX_RE.search(left_ctx):
                    continue
                if _NEGATED_CLEARANCE_SUFFIX_RE.search(right_ctx):
                    continue
                return True
    return False


def _extract_superseded_rules(
    structured_facts: Optional[Dict[str, Any]],
    *,
    raw_text: str = "",
    rule_severity_map: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Extract validated supersede overrides and rejected/untrusted override attempts.

    Enforces P0-2 trust boundaries:
    - Arbitrary `superseded_rule_ids` outside schema are always ignored.
    - Critical rules can NEVER be superseded.
    - Only whitelisted rules (e.g. RF_007) with valid text clearance evidence may be superseded.
    """
    valid_superseded: Dict[str, str] = {}
    rejected_attempts: Dict[str, str] = {}
    if not structured_facts or not isinstance(structured_facts, dict):
        return valid_superseded, rejected_attempts

    sev_map = rule_severity_map or {}

    for cond in structured_facts.get("superseded_conditions", []) or []:
        if not isinstance(cond, dict):
            continue
        rule_hint = str(cond.get("rule_hint") or "").strip().upper()
        if not rule_hint:
            continue
        old_state = str(cond.get("old_state") or "既往旧状态").strip()
        new_state = str(cond.get("new_state") or "最新时间线状态").strip()

        if rule_hint not in sev_map:
            rejected_attempts[rule_hint] = f"拒绝解除未知规则 ID '{rule_hint}'（不在规则目录内）"
            continue
        if sev_map[rule_hint] == "critical" or rule_hint not in SUPERSEDABLE_RULE_WHITELIST:
            rejected_attempts[rule_hint] = (
                f"信任边界拦截：规则 {rule_hint} (severity={sev_map[rule_hint]}) 不可被结构化字段解除"
            )
            continue
        if not _has_valid_supersede_text_evidence(raw_text, rule_hint):
            rejected_attempts[rule_hint] = (
                f"信任边界拦截：最新诊次原文缺乏明确的核准/缓解/达标非问句证据，拒绝解除 {rule_hint}"
            )
            continue

        valid_superseded[rule_hint] = f"最新时间线状态已覆盖旧约束（{old_state} → {new_state}）"

    for rid in structured_facts.get("superseded_rule_ids", []) or []:
        rule_id = str(rid or "").strip().upper()
        if rule_id and rule_id not in valid_superseded and rule_id not in rejected_attempts:
            rejected_attempts[rule_id] = (
                f"信任边界拦截：忽略未在 Schema 定义的原始 superseded_rule_ids 绕过请求 ({rule_id})"
            )

    return valid_superseded, rejected_attempts


def merge_snapshot_into_structured_facts(
    current_state_snapshot: Optional[Dict[str, Any]],
    fallback_facts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map CopilotExtractedNote.current_state_snapshot into Track B structured_facts input."""
    base = dict(fallback_facts) if isinstance(fallback_facts, dict) else {}
    snapshot = dict(current_state_snapshot) if isinstance(current_state_snapshot, dict) else {}

    if not snapshot:
        return {
            "symptoms": list(base.get("symptoms") or []),
            "medications": list(base.get("medications") or []),
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

    # 3. Superseded conditions (only whitelisted rule_hints propagate as hints; validated again at scan time)
    superseded_conditions = list(
        snapshot.get("superseded_conditions") or base.get("superseded_conditions") or []
    )

    result: Dict[str, Any] = {
        "symptoms": merged_symptoms,
        "medications": merged_meds,
        "superseded_conditions": superseded_conditions,
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
    rejected_supersedes: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    alerts: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    superseded_map = superseded_rules or {}
    rejected_map = rejected_supersedes or {}

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

        # --- Track A: Text scan with negation window ---
        text_req_hits: List[str] = []
        text_sup_hits: List[str] = []
        conflict_reasons: List[str] = []

        raw_req_matches = sorted(_find_keyword_matches(raw_text, req_keywords), key=lambda m: m[1])
        # Track negation polarity per keyword mention (sorted chronologically by text offset)
        evaluated_req: List[Tuple[str, int, int, bool, Optional[str], Optional[str]]] = []
        for kw, start, end in raw_req_matches:
            is_neg, neg_reason, neg_token = check_negation_context(
                raw_text,
                start,
                end,
                global_negation_terms=catalog.global_negation_terms,
                negation_guards=rule.negation_guards,
            )
            evaluated_req.append((kw, start, end, is_neg, neg_reason, neg_token))

        # Check whether an absent structured claim is corroborated by the latest text mention
        for abs_name in struct_absent_hits:
            matching_evals = [e for e in evaluated_req if _term_matches_keyword(abs_name, e[0])]
            if not matching_evals or matching_evals[-1][3]:
                suppressed.append(
                    {
                        "rule_id": rule.rule_id,
                        "rule_type": "red_flag",
                        "term": abs_name,
                        "reason": "symptom_explicitly_absent",
                        "source": "structured_facts",
                        "detail": f"结构化事实明确标记症状 '{abs_name}' 为阴性/不存在 (polarity=absent)",
                    }
                )

        for kw, start, end, is_neg, neg_reason, neg_token in evaluated_req:
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
                continue

            # Text hit is affirmative! Check if Track B falsely claimed absent
            matched_absent = [
                abs_name for abs_name in struct_absent_hits if _term_matches_keyword(abs_name, kw)
            ]
            if matched_absent:
                # Only allow Track B absent override if a later mention of the same symptom in text was negated
                same_sym_evals = [e for e in evaluated_req if _term_matches_keyword(matched_absent[0], e[0])]
                latest_is_negated = bool(same_sym_evals and same_sym_evals[-1][3])
                if latest_is_negated:
                    suppressed.append(
                        {
                            "rule_id": rule.rule_id,
                            "rule_type": "red_flag",
                            "term": kw,
                            "reason": "overridden_by_structured_absent",
                            "source": "structured_facts",
                            "detail": f"既往提及 '{kw}'，最新原文已存在否定表述且结构化事实为阴性，已抑制告警",
                        }
                    )
                    continue
                conflict_reasons.append(
                    f"原文明确存在阳性表述 '{kw}'，与结构化字段 polarity=absent 冲突，拒绝静默抑制"
                )

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

        if struct_req_hits or text_req_hits:
            if rule.rule_id in superseded_map and rule.severity != "critical":
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

            if rule.rule_id in rejected_map:
                conflict_reasons.append(rejected_map[rule.rule_id])

            if struct_req_hits and text_req_hits:
                source = "hybrid"
            elif struct_req_hits:
                source = "structured_facts"
            else:
                source = "text_scan"

            evidence = list(dict.fromkeys(struct_req_hits + text_req_hits + struct_sup_hits + text_sup_hits))
            has_conflict = bool(conflict_reasons)
            alert_entry: Dict[str, Any] = {
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
                "conflict_with_text": has_conflict,
                "needs_review": has_conflict,
            }
            if has_conflict:
                alert_entry["conflict_details"] = list(dict.fromkeys(conflict_reasons))
            alerts.append(alert_entry)

    return alerts, suppressed


def _collect_group_hits(
    raw_text: str,
    medications: List[Dict[str, Any]],
    keywords: List[str],
    washout_window_days: Optional[int],
    conflict_id: str,
    global_negation_terms: List[str],
    suppressed: List[Dict[str, Any]],
    other_drug_keywords: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Collect active/in-washout hits for a drug group across Track B and Track A."""
    active_hits: List[Dict[str, Any]] = []
    suppressed_terms: Set[str] = set()
    conflict_notes: Dict[str, str] = {}

    drug_kws = other_drug_keywords if other_drug_keywords is not None else _collect_all_drug_keywords()

    # Pre-inspect raw text mentions of each drug to enforce P0-2 trust boundary on stopped_days_ago
    text_mentions = _find_keyword_matches(raw_text, keywords)

    # 1. Track B structured medications
    for med in medications:
        med_name = med["name"]
        matched_kw = next((kw for kw in keywords if _term_matches_keyword(med_name, kw)), None)
        if not matched_kw:
            continue

        status = med.get("status", "active")
        stopped_days = med.get("stopped_days_ago")
        washout = washout_window_days if washout_window_days is not None else 0

        if status == "stopped":
            # Check if raw_text mentions this drug; if so, require valid text stop evidence
            same_drug_text_spans = [
                (kw, s, e)
                for kw, s, e in text_mentions
                if _term_matches_keyword(med_name, kw)
            ]
            text_stop_verified = True
            text_conflict_msg: Optional[str] = None
            if same_drug_text_spans:
                span_checks = [
                    _check_drug_stopped_in_text(raw_text, s, e, washout_window_days, drug_kws)
                    for _, s, e in same_drug_text_spans
                ]
                # Use the last mention's stop status
                last_suppressed, last_days, _, has_marker = span_checks[-1]
                if not has_marker:
                    text_stop_verified = False
                    text_conflict_msg = (
                        f"原文提及 '{med_name}' 但附近无有效停药标记，拒绝采信结构化 stopped={stopped_days}"
                    )
                elif (
                    isinstance(stopped_days, (int, float))
                    and not isinstance(stopped_days, bool)
                    and stopped_days > washout
                    and not last_suppressed
                ):
                    text_stop_verified = False
                    text_conflict_msg = (
                        f"原文停药时长 ({last_days} 天) 未超过洗脱期 ({washout} 天)，与结构化 stopped_days_ago={stopped_days} 冲突"
                    )

            if not text_stop_verified:
                conflict_notes[med_name.lower()] = text_conflict_msg or "停药状态与原文冲突"
                conflict_notes[matched_kw.lower()] = conflict_notes[med_name.lower()]
                active_hits.append(
                    {
                        "entity": med_name,
                        "keyword": matched_kw,
                        "source": "hybrid",
                        "note": "结构化停药声明与原文冲突（保守按在用纳入复核）",
                        "conflict_with_text": True,
                        "conflict_detail": conflict_notes[med_name.lower()],
                    }
                )
                continue

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

    # 2. Track A raw text scan (respecting Track B overrides, negation, and scoped cessation)
    seen_entities = {h["entity"].lower() for h in active_hits}
    for kw, start, end in text_mentions:
        kw_lower = kw.lower()
        if kw_lower in suppressed_terms or any(kw_lower in st or st in kw_lower for st in suppressed_terms):
            continue
        if kw_lower in seen_entities or any(kw_lower in se or se in kw_lower for se in seen_entities):
            continue

        is_stopped, _, stop_detail, _ = _check_drug_stopped_in_text(
            raw_text, start, end, washout_window_days, drug_kws
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

        is_neg, neg_reason, neg_token = check_negation_context(
            raw_text,
            start,
            end,
            global_negation_terms=global_negation_terms,
            negation_guards=None,
        )
        # If negation came from a negated stop-drug directive (e.g. "切勿自行停用华法林"), the drug is ACTIVE!
        if is_neg:
            scoped_clause, _ = _extract_scoped_drug_clause(raw_text, start, end, drug_kws)
            if _NEGATED_STOP_MARKER_RE.search(scoped_clause):
                is_neg = False
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


def _has_distinct_pair(
    hits_a: List[Dict[str, Any]],
    hits_b: List[Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
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
    rejected_supersedes: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    alerts: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    superseded_map = superseded_rules or {}
    rejected_map = rejected_supersedes or {}
    all_drug_kws = _collect_all_drug_keywords(catalog)

    for rule in catalog.conflicts:
        hits_a = _collect_group_hits(
            raw_text=raw_text,
            medications=medications,
            keywords=rule.group_a.keywords,
            washout_window_days=rule.group_a.washout_window_days,
            conflict_id=rule.conflict_id,
            global_negation_terms=global_negation_terms,
            suppressed=suppressed,
            other_drug_keywords=all_drug_kws,
        )
        hits_b = _collect_group_hits(
            raw_text=raw_text,
            medications=medications,
            keywords=rule.group_b.keywords,
            washout_window_days=rule.group_b.washout_window_days,
            conflict_id=rule.conflict_id,
            global_negation_terms=global_negation_terms,
            suppressed=suppressed,
            other_drug_keywords=all_drug_kws,
        )

        pair = _has_distinct_pair(hits_a, hits_b)
        if not pair:
            continue

        ha, hb = pair
        if rule.conflict_id in superseded_map and rule.severity != "critical":
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

        conflict_details: List[str] = []
        for h in (ha, hb):
            if h.get("conflict_with_text") and h.get("conflict_detail"):
                conflict_details.append(str(h["conflict_detail"]))
        if rule.conflict_id in rejected_map:
            conflict_details.append(rejected_map[rule.conflict_id])

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
        has_conflict = bool(conflict_details)
        alert_entry: Dict[str, Any] = {
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
            "conflict_with_text": has_conflict,
            "needs_review": has_conflict,
        }
        if has_conflict:
            alert_entry["conflict_details"] = list(dict.fromkeys(conflict_details))
        alerts.append(alert_entry)

    return alerts, suppressed


def scan_clinical_risks(
    raw_text: str = "",
    structured_facts: Optional[Dict[str, Any]] = None,
    *,
    red_flags_path: Optional[str] = None,
    drug_conflicts_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan clinical text and/or structured facts for red flags and drug conflicts."""
    rf_catalog = load_red_flags_catalog(red_flags_path)
    dc_catalog = load_drug_conflicts_catalog(drug_conflicts_path)

    rule_severity_map = {r.rule_id: r.severity for r in rf_catalog.rules}
    rule_severity_map.update({c.conflict_id: c.severity for c in dc_catalog.conflicts})

    safe_text = raw_text or ""
    symptoms, medications = _normalize_structured_facts(structured_facts)
    superseded_rules, rejected_supersedes = _extract_superseded_rules(
        structured_facts,
        raw_text=safe_text,
        rule_severity_map=rule_severity_map,
    )

    rf_alerts, rf_suppressed = _scan_red_flags(
        safe_text,
        symptoms,
        rf_catalog,
        superseded_rules=superseded_rules,
        rejected_supersedes=rejected_supersedes,
    )
    dc_alerts, dc_suppressed = _scan_drug_conflicts(
        safe_text,
        medications,
        dc_catalog,
        rf_catalog.global_negation_terms,
        superseded_rules=superseded_rules,
        rejected_supersedes=rejected_supersedes,
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
