"""Local clinical case archive workspace storage with atomic writes and revision history."""

from __future__ import annotations

import copy
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import launcher
from app_copilot.service import DRAFT_MODE_LABELS, format_copilot_response

CASE_SCHEMA_VERSION = 1
MAX_REVISIONS = 20
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def get_case_archives_dir() -> Path:
    """Resolve local case archive directory (env ORTHO_CASE_ARCHIVES_DIR or app_home/case_archives)."""
    env_dir = os.getenv("ORTHO_CASE_ARCHIVES_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    paths = launcher.resolve_runtime_paths()
    return paths.app_home / "case_archives"


def validate_case_id(case_id: str) -> str:
    """Enforce strict case_id format to prevent path traversal."""
    cleaned = (case_id or "").strip()
    if not CASE_ID_PATTERN.fullmatch(cleaned):
        raise ValueError("非法的病例档案 ID（仅允许 1-64 位字母、数字、下划线或短横线）。")
    return cleaned


def generate_case_id() -> str:
    """Generate a unique timestamped case archive identifier."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"CASE_{stamp}_{suffix}"


def _resolve_case_file(case_id: str, archives_dir: Optional[Path] = None) -> Path:
    safe_id = validate_case_id(case_id)
    base_dir = (archives_dir or get_case_archives_dir()).resolve()
    target = (base_dir / f"{safe_id}.json").resolve()
    if not target.is_relative_to(base_dir):
        raise ValueError("非法的病例档案文件路径。")
    return target


def _atomic_write_json(target: Path, payload: Dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex[:6]}.tmp")
    raw_text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        tmp_path.write_text(raw_text, encoding="utf-8")
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def derive_case_title(
    explicit_title: Optional[str],
    raw_text: str,
    analysis_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Derive a concise <=36 char case title from explicit input, chief_complaint, or raw_text."""
    cleaned_explicit = (explicit_title or "").strip()
    if cleaned_explicit:
        return cleaned_explicit[:36]

    note = (analysis_state or {}).get("extracted_note") or {}
    cc = str(note.get("chief_complaint") or "").strip()
    if cc and cc not in ("未提及", "未提供主诉"):
        first_part = re.split(r"[\n。；;]+", cc)[0].strip()
        if first_part:
            return first_part[:36]

    first_line = next((ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()), "")
    if first_line:
        first_sent = re.split(r"[。；;]+", first_line)[0].strip() or first_line
        return first_sent[:36]

    return "未命名脱敏病例"


def _merge_case_tags(
    custom_tags: Optional[List[str]],
    analysis_state: Optional[Dict[str, Any]],
) -> List[str]:
    tags: List[str] = []
    seen = set()

    def _add(tag: str) -> None:
        t = str(tag or "").strip()
        if t and t not in seen:
            seen.add(t)
            tags.append(t)

    state = analysis_state or {}
    alerts = state.get("unified_alerts") or []
    has_critical = any(str(a.get("severity", "")).lower() == "critical" for a in alerts)
    has_high = any(str(a.get("severity", "")).lower() == "high" for a in alerts)
    has_drug_conflict = any(str(a.get("rule_type", "")) == "drug_conflict" for a in alerts)
    if has_critical:
        _add("极高危红旗")
    elif has_high:
        _add("高危警示")
    if has_drug_conflict:
        _add("用药冲突")

    scales = state.get("scale_results") or []
    if any(s.get("status") == "error" for s in scales):
        _add("量表待补录")
    elif scales:
        _add("量表已算分")

    drafts_by_mode = state.get("drafts_by_mode") or {}
    if len(drafts_by_mode) >= 2:
        _add("双文书已就绪")
    elif len(drafts_by_mode) == 1:
        _add("单文书已生成")
    elif state.get("fast_scan_only"):
        _add("极速筛查")

    for t in custom_tags or []:
        _add(t)

    return tags[:12]


def _normalize_analysis_state(
    analysis_state: Optional[Dict[str, Any]],
    raw_text: str,
    draft_mode: str,
) -> Dict[str, Any]:
    if not analysis_state or not isinstance(analysis_state, dict):
        return {
            "raw_text": raw_text,
            "fast_scan_only": False,
            "extracted_note": {},
            "risk_report": {},
            "scale_results": [],
            "unified_alerts": [],
            "contraindicated_capsules": [],
            "draft": {},
            "draft_mode": draft_mode,
            "draft_stale": False,
            "draft_plain_text": "",
            "drafts_by_mode": {},
            "draft_stale_by_mode": {},
            "draft_cache_hit": False,
            "trace": {},
        }
    normalized = format_copilot_response(copy.deepcopy(analysis_state))
    if raw_text and not normalized.get("raw_text"):
        normalized["raw_text"] = raw_text
    if draft_mode in DRAFT_MODE_LABELS and not normalized.get("draft_mode"):
        normalized["draft_mode"] = draft_mode
    return normalized


def _build_revision_entry(
    rev_num: int,
    *,
    note: str,
    raw_text: str,
    analysis_state: Dict[str, Any],
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    alerts = analysis_state.get("unified_alerts") or []
    drafts_by_mode = analysis_state.get("drafts_by_mode") or {}
    return {
        "rev": rev_num,
        "timestamp": timestamp or _now_iso(),
        "note": (note or f"保存版本 v{rev_num}").strip(),
        "raw_text": raw_text,
        "alert_count": len(alerts),
        "draft_modes_ready": list(drafts_by_mode.keys()),
        "analysis_state": copy.deepcopy(analysis_state),
    }


def build_case_summary(archive: Dict[str, Any]) -> Dict[str, Any]:
    """Extract lightweight list summary fields from a full archive dict."""
    state = archive.get("analysis_state") or {}
    alerts = state.get("unified_alerts") or []
    critical_count = sum(1 for a in alerts if str(a.get("severity", "")).lower() == "critical")
    scales = state.get("scale_results") or []
    drafts_by_mode = state.get("drafts_by_mode") or {}
    raw_text = str(archive.get("raw_text") or "")
    snippet = re.sub(r"\s+", " ", raw_text).strip()[:80]
    return {
        "case_id": archive.get("case_id", ""),
        "title": archive.get("title", ""),
        "tags": list(archive.get("tags") or []),
        "created_at": archive.get("created_at", ""),
        "updated_at": archive.get("updated_at", ""),
        "revision_count": int(archive.get("revision_count") or len(archive.get("revisions") or [])),
        "alert_count": len(alerts),
        "critical_count": critical_count,
        "scale_count": len(scales),
        "draft_modes_ready": list(drafts_by_mode.keys()),
        "snippet": snippet,
    }


def list_cases(
    q: str = "",
    high_risk_only: bool = False,
    *,
    archives_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """List saved case summaries sorted reverse-chronologically by updated_at."""
    target_dir = archives_dir or get_case_archives_dir()
    summaries: List[Dict[str, Any]] = []
    query = (q or "").strip().lower()

    if target_dir.is_dir():
        for path in sorted(target_dir.glob("*.json")):
            try:
                archive = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(archive, dict) or not archive.get("case_id"):
                    continue
            except (OSError, json.JSONDecodeError):
                continue

            summary = build_case_summary(archive)
            if high_risk_only:
                alerts = (archive.get("analysis_state") or {}).get("unified_alerts") or []
                is_high = summary["critical_count"] > 0 or any(
                    str(a.get("severity", "")).lower() in ("critical", "high") for a in alerts
                )
                if not is_high:
                    continue

            if query:
                haystack = " ".join(
                    [
                        str(summary["case_id"]),
                        str(summary["title"]),
                        str(summary["snippet"]),
                        " ".join(summary["tags"]),
                        str(archive.get("raw_text") or ""),
                    ]
                ).lower()
                if query not in haystack:
                    continue

            summaries.append(summary)

    summaries.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return {
        "archives_dir": str(target_dir),
        "cases": summaries,
    }


def load_case(case_id: str, *, archives_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load a single full case archive by case_id (raises FileNotFoundError if absent)."""
    target = _resolve_case_file(case_id, archives_dir)
    if not target.is_file():
        raise FileNotFoundError(f"未找到病例档案 '{case_id}'。")
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"病例档案 '{case_id}' 文件格式损坏。")
    return data


def save_case(
    *,
    raw_text: str,
    draft_mode: str = "ward_round",
    analysis_state: Optional[Dict[str, Any]] = None,
    case_id: Optional[str] = None,
    title: Optional[str] = None,
    tags: Optional[List[str]] = None,
    revision_note: Optional[str] = None,
    record_revision: bool = True,
    archives_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create or update a local case archive atomically."""
    cleaned_raw = (raw_text or "").strip()
    if not cleaned_raw:
        raise ValueError("病例原文不能为空。")

    safe_mode = draft_mode if draft_mode in DRAFT_MODE_LABELS else "ward_round"
    target_dir = archives_dir or get_case_archives_dir()
    now_iso = _now_iso()

    existing: Optional[Dict[str, Any]] = None
    if case_id and str(case_id).strip():
        safe_id = validate_case_id(str(case_id))
        target = _resolve_case_file(safe_id, target_dir)
        if target.is_file():
            existing = load_case(safe_id, archives_dir=target_dir)
    else:
        safe_id = generate_case_id()
        target = _resolve_case_file(safe_id, target_dir)

    normalized_state = _normalize_analysis_state(analysis_state, cleaned_raw, safe_mode)
    resolved_title = derive_case_title(
        title if title is not None else (existing.get("title") if existing else None),
        cleaned_raw,
        normalized_state,
    )
    existing_custom_tags = existing.get("tags") if (existing and tags is None) else tags
    resolved_tags = _merge_case_tags(existing_custom_tags, normalized_state)

    if existing is None:
        rev_num = 1
        revisions = [
            _build_revision_entry(
                rev_num,
                note=revision_note or "首次创建病例档案",
                raw_text=cleaned_raw,
                analysis_state=normalized_state,
                timestamp=now_iso,
            )
        ]
        created_at = now_iso
        revision_count = 1
    else:
        created_at = str(existing.get("created_at") or now_iso)
        revisions = list(existing.get("revisions") or [])
        revision_count = int(existing.get("revision_count") or len(revisions) or 1)
        if record_revision:
            revision_count += 1
            revisions.insert(
                0,
                _build_revision_entry(
                    revision_count,
                    note=revision_note or "更新病历分析状态",
                    raw_text=cleaned_raw,
                    analysis_state=normalized_state,
                    timestamp=now_iso,
                ),
            )
            revisions = revisions[:MAX_REVISIONS]
        elif revisions:
            # In-place silent update on latest state (e.g. cached draft tab switch) without spamming revisions
            latest_rev = int(revisions[0].get("rev") or revision_count)
            revisions[0] = _build_revision_entry(
                latest_rev,
                note=str(revisions[0].get("note") or "更新状态"),
                raw_text=cleaned_raw,
                analysis_state=normalized_state,
                timestamp=now_iso,
            )

    archive = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": safe_id,
        "title": resolved_title,
        "tags": resolved_tags,
        "created_at": created_at,
        "updated_at": now_iso,
        "revision_count": revision_count,
        "raw_text": cleaned_raw,
        "draft_mode": safe_mode,
        "analysis_state": normalized_state,
        "revisions": revisions,
    }
    _atomic_write_json(target, archive)
    return archive


def update_case_metadata(
    case_id: str,
    *,
    title: Optional[str] = None,
    tags: Optional[List[str]] = None,
    archives_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Update only title and/or custom tags of an existing case archive."""
    target = _resolve_case_file(case_id, archives_dir)
    if not target.is_file():
        raise FileNotFoundError(f"未找到病例档案 '{case_id}'。")
    archive = load_case(case_id, archives_dir=archives_dir)
    if title is not None:
        cleaned_title = title.strip()
        if not cleaned_title:
            raise ValueError("档案标题不能为空。")
        archive["title"] = cleaned_title[:64]
    if tags is not None:
        archive["tags"] = [str(t).strip() for t in tags if str(t).strip()][:12]
    archive["updated_at"] = _now_iso()
    _atomic_write_json(target, archive)
    return archive


def append_followup_visit(
    case_id: str,
    followup_text: str,
    *,
    visit_date: Optional[str] = None,
    archives_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Append a cross-visit follow-up note to raw_text, record a revision, and save atomically."""
    cleaned_followup = (followup_text or "").strip()
    if not cleaned_followup:
        raise ValueError("复诊随访内容不能为空。")

    target = _resolve_case_file(case_id, archives_dir)
    if not target.is_file():
        raise FileNotFoundError(f"未找到病例档案 '{case_id}'。")
    archive = load_case(case_id, archives_dir=archives_dir)

    date_label = (visit_date or "").strip() or datetime.now().strftime("%Y-%m-%d")
    base_raw = str(archive.get("raw_text") or "").rstrip()
    merged_raw = f"{base_raw}\n\n【{date_label} 复诊随访】\n{cleaned_followup}".strip()

    now_iso = _now_iso()
    state = copy.deepcopy(archive.get("analysis_state") or {})
    state["raw_text"] = merged_raw
    # Mark cached drafts stale because clinical timeline changed
    state["draft_stale"] = True
    stale_map = dict(state.get("draft_stale_by_mode") or {})
    for m in list((state.get("drafts_by_mode") or {}).keys()):
        stale_map[m] = True
    state["draft_stale_by_mode"] = stale_map

    revisions = list(archive.get("revisions") or [])
    new_rev = int(archive.get("revision_count") or len(revisions) or 0) + 1
    revisions.insert(
        0,
        _build_revision_entry(
            new_rev,
            note=f"追加新诊次（{date_label}）",
            raw_text=merged_raw,
            analysis_state=state,
            timestamp=now_iso,
        ),
    )

    archive["raw_text"] = merged_raw
    archive["analysis_state"] = state
    archive["revision_count"] = new_rev
    archive["revisions"] = revisions[:MAX_REVISIONS]
    archive["updated_at"] = now_iso
    _atomic_write_json(target, archive)
    return archive


def restore_case_revision(
    case_id: str,
    rev: int,
    *,
    archives_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Restore an archive to a specific historical revision number."""
    target = _resolve_case_file(case_id, archives_dir)
    if not target.is_file():
        raise FileNotFoundError(f"未找到病例档案 '{case_id}'。")
    archive = load_case(case_id, archives_dir=archives_dir)
    revisions = list(archive.get("revisions") or [])
    matched = next((item for item in revisions if int(item.get("rev", -1)) == int(rev)), None)
    if matched is None:
        raise FileNotFoundError(f"档案 '{case_id}' 中不存在历史版本 v{rev}。")

    restored_raw = str(matched.get("raw_text") or archive.get("raw_text") or "").strip()
    restored_state = copy.deepcopy(matched.get("analysis_state") or {})
    restored_mode = str(restored_state.get("draft_mode") or archive.get("draft_mode") or "ward_round")
    if restored_mode not in DRAFT_MODE_LABELS:
        restored_mode = "ward_round"
    normalized_state = _normalize_analysis_state(restored_state, restored_raw, restored_mode)

    now_iso = _now_iso()
    new_rev = int(archive.get("revision_count") or len(revisions) or 0) + 1
    revisions.insert(
        0,
        _build_revision_entry(
            new_rev,
            note=f"恢复至历史版本 v{rev}",
            raw_text=restored_raw,
            analysis_state=normalized_state,
            timestamp=now_iso,
        ),
    )

    archive["raw_text"] = restored_raw
    archive["draft_mode"] = restored_mode
    archive["analysis_state"] = normalized_state
    archive["tags"] = _merge_case_tags(archive.get("tags"), normalized_state)
    archive["revision_count"] = new_rev
    archive["revisions"] = revisions[:MAX_REVISIONS]
    archive["updated_at"] = now_iso
    _atomic_write_json(target, archive)
    return archive


def delete_case(case_id: str, *, archives_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Delete a case archive from disk (raises FileNotFoundError if absent)."""
    target = _resolve_case_file(case_id, archives_dir)
    if not target.is_file():
        raise FileNotFoundError(f"未找到病例档案 '{case_id}'。")
    target.unlink()
    return {"deleted": True, "case_id": validate_case_id(case_id)}


def import_case_archive(
    payload: Dict[str, Any],
    *,
    archives_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate and import an exported case archive JSON without overwriting existing case_ids."""
    if not isinstance(payload, dict):
        raise ValueError("导入的病例档案必须为 JSON 对象。")

    raw_text = str(payload.get("raw_text") or "").strip()
    if not raw_text:
        raise ValueError("导入档案缺少有效病历原文 (`raw_text`)。")

    target_dir = archives_dir or get_case_archives_dir()
    candidate_id = str(payload.get("case_id") or "").strip()
    if not candidate_id or not CASE_ID_PATTERN.fullmatch(candidate_id):
        resolved_id = generate_case_id()
    else:
        existing_path = _resolve_case_file(candidate_id, target_dir)
        resolved_id = generate_case_id() if existing_path.exists() else candidate_id

    draft_mode = str(payload.get("draft_mode") or "ward_round")
    if draft_mode not in DRAFT_MODE_LABELS:
        draft_mode = "ward_round"

    normalized_state = _normalize_analysis_state(payload.get("analysis_state"), raw_text, draft_mode)
    title = derive_case_title(payload.get("title"), raw_text, normalized_state)
    tags = _merge_case_tags(payload.get("tags"), normalized_state)

    now_iso = _now_iso()
    raw_revisions = payload.get("revisions")
    revisions: List[Dict[str, Any]] = []
    if isinstance(raw_revisions, list):
        for idx, item in enumerate(raw_revisions[:MAX_REVISIONS], start=1):
            if not isinstance(item, dict):
                continue
            r_raw = str(item.get("raw_text") or raw_text).strip()
            r_state = _normalize_analysis_state(item.get("analysis_state"), r_raw, draft_mode)
            revisions.append(
                _build_revision_entry(
                    int(item.get("rev") or idx),
                    note=str(item.get("note") or f"导入版本 v{idx}"),
                    raw_text=r_raw,
                    analysis_state=r_state,
                    timestamp=str(item.get("timestamp") or now_iso),
                )
            )

    if not revisions:
        revisions = [
            _build_revision_entry(
                1,
                note="从外部 JSON 导入",
                raw_text=raw_text,
                analysis_state=normalized_state,
                timestamp=now_iso,
            )
        ]

    rev_count = max(int(payload.get("revision_count") or len(revisions)), len(revisions))
    archive = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": resolved_id,
        "title": title,
        "tags": tags,
        "created_at": str(payload.get("created_at") or now_iso),
        "updated_at": now_iso,
        "revision_count": rev_count,
        "raw_text": raw_text,
        "draft_mode": draft_mode,
        "analysis_state": normalized_state,
        "revisions": revisions[:MAX_REVISIONS],
    }
    _atomic_write_json(_resolve_case_file(resolved_id, target_dir), archive)
    return archive
