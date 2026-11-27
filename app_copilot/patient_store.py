"""Patient workspace card persistence (case_archives/<patient_id>/PATIENT.md).

Stores each patient as a folder containing:
- PATIENT.md (YAML front matter validated by PatientCard + freeform clinical body)
- .history/PATIENT.rNNN.md (immutable revision snapshots)
- sessions/ (JSONL trace + read-only Markdown views)
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

import launcher
from core_tools.schemas import (
    PatientCard,
    PatientMedicationItem,
    PatientRedFlagItem,
    PatientScaleRecord,
)

PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_FRONTMATTER_RE = re.compile(r"^\ufeff?---\r?\n(.*?)\r?\n---\r?\n?(.*)$", re.DOTALL)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _today_str() -> str:
    return date.today().isoformat()


def get_case_archives_dir() -> Path:
    """Resolve local case_archives root directory (env ORTHO_CASE_ARCHIVES_DIR or app_home/case_archives)."""
    env_dir = os.getenv("ORTHO_CASE_ARCHIVES_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    paths = launcher.resolve_runtime_paths()
    return paths.app_home / "case_archives"


def validate_patient_id(patient_id: str) -> str:
    """Enforce strict patient_id format to prevent path traversal."""
    cleaned = (patient_id or "").strip()
    if not PATIENT_ID_PATTERN.fullmatch(cleaned):
        raise ValueError("非法的患者档案 ID（仅允许 1-64 位字母、数字、下划线或短横线）。")
    return cleaned


def generate_patient_id() -> str:
    """Generate a unique timestamped patient identifier."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"PT_{stamp}_{suffix}"


def _resolve_patient_dir(patient_id: str, archives_dir: Optional[Path] = None) -> Path:
    safe_id = validate_patient_id(patient_id)
    base_dir = (archives_dir or get_case_archives_dir()).resolve()
    target = (base_dir / safe_id).resolve()
    if not target.is_relative_to(base_dir) or target == base_dir:
        raise ValueError("非法的患者档案目录路径。")
    return target


def _resolve_patient_md_path(patient_id: str, archives_dir: Optional[Path] = None) -> Path:
    return _resolve_patient_dir(patient_id, archives_dir) / "PATIENT.md"


def _resolve_history_dir(patient_id: str, archives_dir: Optional[Path] = None) -> Path:
    return _resolve_patient_dir(patient_id, archives_dir) / ".history"


def _atomic_write_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def serialize_patient_md(card: PatientCard, body: str = "") -> str:
    """Serialize PatientCard + clinical body into YAML front matter Markdown."""
    payload = card.model_dump(mode="json", exclude_none=True)
    yaml_text = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    cleaned_body = (body or "").strip()
    if cleaned_body:
        return f"---\n{yaml_text}\n---\n\n{cleaned_body}\n"
    return f"---\n{yaml_text}\n---\n"


def parse_patient_md(raw_md: str) -> Tuple[PatientCard, str]:
    """Parse and Pydantic-validate PATIENT.md content into (PatientCard, body)."""
    text = raw_md or ""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError("PATIENT.md 格式错误：缺少合法的 YAML front matter（--- ... ---）。")
    yaml_block, body = match.group(1), match.group(2)
    try:
        loaded = yaml.safe_load(yaml_block)
    except yaml.YAMLError as exc:
        raise ValueError(f"PATIENT.md YAML 解析失败：{exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("PATIENT.md YAML front matter 必须是键值对象。")
    card = PatientCard.model_validate(loaded)
    return card, body.strip()


def _assert_no_silent_red_flag_removal(old_card: PatientCard, new_card: PatientCard) -> None:
    """Ensure active red flags cannot be silently dropped without an explicit resolved_reason."""
    old_active = {
        rf.id: rf
        for rf in old_card.active_red_flags
        if rf.status == "active"
    }
    if not old_active:
        return

    now_ts = _now_iso()
    new_by_id: Dict[str, List[PatientRedFlagItem]] = {}
    for rf in new_card.active_red_flags:
        new_by_id.setdefault(rf.id, []).append(rf)

    for rf_id in old_active:
        entries = new_by_id.get(rf_id) or []
        if not entries:
            raise ValueError(
                f"禁止静默删除有效红旗 {rf_id}，请将其状态设为 resolved 并填写解除理由。"
            )
        has_active = any(e.status == "active" for e in entries)
        if has_active:
            continue
        # Otherwise all entries for this rule are resolved; ensure resolved_reason exists and fill resolved_at
        resolved_entries = [e for e in entries if e.status == "resolved"]
        if not resolved_entries or not any((e.resolved_reason or "").strip() for e in resolved_entries):
            raise ValueError(f"解除红旗 {rf_id} 必须填写非空的 resolved_reason。")
        for entry in resolved_entries:
            if not (entry.resolved_at or "").strip():
                entry.resolved_at = now_ts


def _write_revision_snapshot(
    patient_id: str,
    card: PatientCard,
    body: str,
    *,
    archives_dir: Optional[Path] = None,
) -> Path:
    history_dir = _resolve_history_dir(patient_id, archives_dir)
    rev_file = history_dir / f"PATIENT.r{card.revision:03d}.md"
    _atomic_write_text(rev_file, serialize_patient_md(card, body))
    return rev_file


def create_patient(
    display_name: str = "",
    *,
    patient_id: Optional[str] = None,
    body: str = "",
    card_data: Optional[Dict[str, Any]] = None,
    archives_dir: Optional[Path] = None,
) -> Tuple[PatientCard, str, str]:
    """Create a new patient workspace folder with initial PATIENT.md and revision r001."""
    pid = validate_patient_id(patient_id) if patient_id else generate_patient_id()
    patient_dir = _resolve_patient_dir(pid, archives_dir)
    md_path = patient_dir / "PATIENT.md"
    if md_path.exists():
        raise ValueError(f"患者档案已存在：{pid}")

    now_ts = _now_iso()
    init_payload: Dict[str, Any] = dict(card_data or {})
    init_payload["patient_id"] = pid
    if display_name:
        init_payload["display_name"] = display_name
    elif "display_name" not in init_payload:
        init_payload["display_name"] = pid
    init_payload["created_at"] = now_ts
    init_payload["updated_at"] = now_ts
    init_payload["revision"] = 1

    card = PatientCard.model_validate(init_payload)
    patient_dir.mkdir(parents=True, exist_ok=True)
    (patient_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (patient_dir / ".history").mkdir(parents=True, exist_ok=True)

    raw_md = serialize_patient_md(card, body)
    _atomic_write_text(md_path, raw_md)
    _write_revision_snapshot(pid, card, body, archives_dir=archives_dir)
    return card, body.strip(), raw_md


def load_patient(
    patient_id: str,
    *,
    archives_dir: Optional[Path] = None,
) -> Tuple[PatientCard, str, str]:
    """Load and validate PATIENT.md for a patient workspace."""
    md_path = _resolve_patient_md_path(patient_id, archives_dir)
    if not md_path.is_file():
        raise FileNotFoundError(f"未找到患者档案：{patient_id}")
    raw_md = md_path.read_text(encoding="utf-8")
    card, body = parse_patient_md(raw_md)
    if card.patient_id != validate_patient_id(patient_id):
        card.patient_id = validate_patient_id(patient_id)
    return card, body, raw_md


def list_patients(*, archives_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List all patient folders (containing PATIENT.md) sorted by updated_at descending."""
    base_dir = (archives_dir or get_case_archives_dir()).resolve()
    if not base_dir.exists():
        return []

    items: List[Dict[str, Any]] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        if not PATIENT_ID_PATTERN.fullmatch(child.name):
            continue
        md_file = child / "PATIENT.md"
        if not md_file.is_file():
            continue
        try:
            card, _ = parse_patient_md(md_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        sessions_dir = child / "sessions"
        session_count = (
            sum(1 for f in sessions_dir.glob("*.trace.jsonl") if f.is_file())
            if sessions_dir.is_dir()
            else 0
        )
        active_rfs = [rf for rf in card.active_red_flags if rf.status == "active"]
        items.append(
            {
                "patient_id": card.patient_id or child.name,
                "display_name": card.display_name or child.name,
                "active_red_flag_count": len(active_rfs),
                "active_red_flag_ids": [rf.id for rf in active_rfs],
                "session_count": session_count,
                "revision": card.revision,
                "created_at": card.created_at or "",
                "updated_at": card.updated_at or "",
            }
        )

    items.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
    return items


def save_patient(
    patient_id: str,
    card: PatientCard | Dict[str, Any],
    body: str,
    *,
    archives_dir: Optional[Path] = None,
    allow_red_flag_revert: bool = False,
) -> Tuple[PatientCard, str, str]:
    """Validate and atomically persist PATIENT.md + append a new .history revision."""
    safe_id = validate_patient_id(patient_id)
    old_card, _, _ = load_patient(safe_id, archives_dir=archives_dir)

    if isinstance(card, PatientCard):
        candidate_dict = card.model_dump(mode="python")
    elif isinstance(card, dict):
        candidate_dict = copy.deepcopy(card)
    else:
        raise ValueError("患者档案卡格式非法。")

    candidate_dict["patient_id"] = safe_id
    candidate_dict["created_at"] = old_card.created_at or _now_iso()
    candidate_dict["updated_at"] = _now_iso()
    candidate_dict["revision"] = int(old_card.revision or 1) + 1

    new_card = PatientCard.model_validate(candidate_dict)
    if not allow_red_flag_revert:
        _assert_no_silent_red_flag_removal(old_card, new_card)

    cleaned_body = (body or "").strip()
    raw_md = serialize_patient_md(new_card, cleaned_body)
    _atomic_write_text(_resolve_patient_md_path(safe_id, archives_dir), raw_md)
    _write_revision_snapshot(safe_id, new_card, cleaned_body, archives_dir=archives_dir)
    return new_card, cleaned_body, raw_md


def apply_red_flags_auto(
    patient_id: str,
    alerts: Iterable[Dict[str, Any]],
    *,
    source: str = "extracted",
    archives_dir: Optional[Path] = None,
) -> Tuple[PatientCard, List[str]]:
    """Automatically persist newly triggered red flags/conflicts to PATIENT.md (append-only for active rules)."""
    safe_id = validate_patient_id(patient_id)
    card, body, _ = load_patient(safe_id, archives_dir=archives_dir)
    updated_card = card.model_copy(deep=True)
    existing_active_ids = {
        rf.id for rf in updated_card.active_red_flags if rf.status == "active"
    }

    added_ids: List[str] = []
    seen_new = set()
    today = _today_str()
    valid_source = source if source in ("clinician", "extracted", "system") else "extracted"

    for alert in alerts or []:
        if not isinstance(alert, dict):
            continue
        raw_rule_id = str(alert.get("rule_id") or "").strip().upper()
        if not raw_rule_id:
            continue
        # Scale linkage can hold comma-separated IDs like "RF_001"
        sub_ids = [part.strip().upper() for part in raw_rule_id.split(",") if part.strip()]
        for rid in sub_ids:
            if rid in existing_active_ids or rid in seen_new:
                continue
            seen_new.add(rid)
            added_ids.append(rid)
            updated_card.active_red_flags.append(
                PatientRedFlagItem(
                    id=rid,
                    name=str(alert.get("name") or "").strip() or None,
                    since=today,
                    status="active",
                    source=valid_source,  # type: ignore[arg-type]
                )
            )

    if not added_ids:
        return card, []

    saved_card, _, _ = save_patient(safe_id, updated_card, body, archives_dir=archives_dir)
    return saved_card, added_ids


def commit_changes(
    patient_id: str,
    changes: Dict[str, Any],
    *,
    archives_dir: Optional[Path] = None,
) -> Tuple[PatientCard, str, str]:
    """Commit previewed clinical card changes and/or explicit red-flag resolutions with audit trail."""
    safe_id = validate_patient_id(patient_id)
    card, body, _ = load_patient(safe_id, archives_dir=archives_dir)
    updated = card.model_copy(deep=True)
    new_body = body

    if "display_name" in changes and changes["display_name"] is not None:
        updated.display_name = str(changes["display_name"]).strip()[:64]

    # Explicit red-flag resolutions (require resolved_reason)
    for res_item in changes.get("resolve_red_flags") or []:
        if not isinstance(res_item, dict):
            continue
        target_id = str(res_item.get("id") or "").strip().upper()
        reason = str(res_item.get("resolved_reason") or "").strip()
        if not target_id or not reason:
            raise ValueError("解除红旗必须同时提供 id 与 resolved_reason。")
        matched = False
        now_ts = _now_iso()
        for rf in updated.active_red_flags:
            if rf.id == target_id and rf.status == "active":
                rf.status = "resolved"
                rf.resolved_reason = reason
                rf.resolved_at = now_ts
                matched = True
        if not matched:
            raise ValueError(f"未找到处于有效状态的红旗：{target_id}")

    # Medications upsert
    meds_by_name: Dict[str, int] = {m.name: idx for idx, m in enumerate(updated.medications)}
    for med_raw in (changes.get("add_medications") or []) + (changes.get("update_medications") or []):
        if not isinstance(med_raw, dict):
            continue
        med_obj = PatientMedicationItem.model_validate(med_raw)
        if med_obj.name in meds_by_name:
            updated.medications[meds_by_name[med_obj.name]] = med_obj
        else:
            meds_by_name[med_obj.name] = len(updated.medications)
            updated.medications.append(med_obj)

    # Allergies & surgical_history append (dedup)
    for allergy in changes.get("add_allergies") or []:
        val = str(allergy or "").strip()
        if val and val not in updated.allergies:
            updated.allergies.append(val)

    for surgery in changes.get("add_surgical_history") or []:
        val = str(surgery or "").strip()
        if val and val not in updated.surgical_history:
            updated.surgical_history.append(val)

    # Scale history append
    existing_scale_keys = {
        (s.scale_id, s.date, s.total_score) for s in updated.scale_history
    }
    for sc_raw in changes.get("add_scale_history") or []:
        if not isinstance(sc_raw, dict):
            continue
        sc_obj = PatientScaleRecord.model_validate(sc_raw)
        key = (sc_obj.scale_id, sc_obj.date, sc_obj.total_score)
        if key not in existing_scale_keys:
            existing_scale_keys.add(key)
            updated.scale_history.append(sc_obj)

    if isinstance(changes.get("body"), str):
        new_body = changes["body"].strip()
    elif isinstance(changes.get("append_body"), str) and changes["append_body"].strip():
        addition = changes["append_body"].strip()
        new_body = f"{new_body}\n\n{addition}".strip() if new_body else addition

    return save_patient(safe_id, updated, new_body, archives_dir=archives_dir)


def list_history(
    patient_id: str,
    *,
    archives_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List recorded revision snapshots under .history/PATIENT.rNNN.md."""
    safe_id = validate_patient_id(patient_id)
    # Ensure patient exists
    load_patient(safe_id, archives_dir=archives_dir)
    history_dir = _resolve_history_dir(safe_id, archives_dir)
    if not history_dir.is_dir():
        return []

    revs: List[Dict[str, Any]] = []
    for path in sorted(history_dir.glob("PATIENT.r*.md")):
        m = re.match(r"^PATIENT\.r(\d+)\.md$", path.name)
        if not m:
            continue
        rev_num = int(m.group(1))
        try:
            card, _ = parse_patient_md(path.read_text(encoding="utf-8"))
            revs.append(
                {
                    "revision": rev_num,
                    "updated_at": card.updated_at or "",
                    "active_red_flag_count": sum(1 for rf in card.active_red_flags if rf.status == "active"),
                    "active_red_flag_ids": [rf.id for rf in card.active_red_flags if rf.status == "active"],
                    "medication_count": len(card.medications),
                }
            )
        except Exception:
            continue

    revs.sort(key=lambda item: int(item["revision"]), reverse=True)
    return revs


def restore_revision(
    patient_id: str,
    rev: int,
    *,
    archives_dir: Optional[Path] = None,
) -> Tuple[PatientCard, str, str]:
    """Restore PATIENT.md from .history/PATIENT.rNNN.md as a new auditable revision."""
    safe_id = validate_patient_id(patient_id)
    rev_int = int(rev)
    if rev_int < 1:
        raise ValueError("版本号必须为正整数。")
    history_file = _resolve_history_dir(safe_id, archives_dir) / f"PATIENT.r{rev_int:03d}.md"
    if not history_file.is_file():
        raise FileNotFoundError(f"未找到修订版本 r{rev_int:03d}")

    target_card, target_body = parse_patient_md(history_file.read_text(encoding="utf-8"))
    return save_patient(
        safe_id,
        target_card,
        target_body,
        archives_dir=archives_dir,
        allow_red_flag_revert=True,
    )


def delete_patient(patient_id: str, *, archives_dir: Optional[Path] = None) -> bool:
    """Delete an entire patient workspace folder safely."""
    patient_dir = _resolve_patient_dir(patient_id, archives_dir)
    if not patient_dir.is_dir():
        return False
    shutil.rmtree(patient_dir)
    return True


def _compute_stopped_days(med: PatientMedicationItem) -> Optional[float]:
    if med.stopped_days_ago is not None:
        return float(med.stopped_days_ago)
    if med.stop_date:
        try:
            stop_d = date.fromisoformat(med.stop_date[:10])
            delta = (date.today() - stop_d).days
            if delta >= 0:
                return float(delta)
        except ValueError:
            return None
    return None


def build_patient_structured_facts(card: Optional[PatientCard]) -> Dict[str, Any]:
    """Convert PatientCard medications & surgical history into Track B structured_facts."""
    if card is None:
        return {"symptoms": [], "medications": [], "surgical_history": [], "superseded_rules": []}

    meds: List[Dict[str, Any]] = []
    for med in card.medications:
        stopped_days = _compute_stopped_days(med)
        meds.append(
            {
                "name": med.name,
                "status": med.status,
                "stopped_days_ago": stopped_days,
            }
        )
    return {
        "symptoms": [],
        "medications": meds,
        "surgical_history": list(card.surgical_history),
        "superseded_rules": [],
    }


def _strip_display_name(text: str, display_name: str) -> str:
    """Deterministically remove patient display_name before building any LLM prompt."""
    result = text or ""
    cleaned_name = (display_name or "").strip()
    if len(cleaned_name) >= 2:
        result = result.replace(cleaned_name, "[患者标识已脱敏]")
        # Also strip trailing name part such as "302床 张某" -> "张某"
        for token in re.split(r"[\s,，/]+", cleaned_name):
            tok = token.strip()
            if len(tok) >= 2 and not re.fullmatch(r"\d+床?", tok):
                result = result.replace(tok, "[患者姓名已脱敏]")
    return result


def build_prompt_context(card: Optional[PatientCard], body: str = "") -> str:
    """Build de-identified longitudinal prompt context from PATIENT.md (never includes display_name)."""
    if card is None and not (body or "").strip():
        return ""

    sections: List[str] = []
    display_name = card.display_name if card is not None else ""

    if card is not None:
        active_rfs = [rf for rf in card.active_red_flags if rf.status == "active"]
        resolved_rfs = [rf for rf in card.active_red_flags if rf.status == "resolved"]

        if active_rfs:
            rf_lines = [
                f"- [{rf.id}] {rf.name or '临床高危红旗'}（持续生效，起始：{rf.since}，来源：{rf.source}）"
                for rf in active_rfs
            ]
            sections.append("【患者档案卡有效红旗（只增不静默删）】\n" + "\n".join(rf_lines))

        if resolved_rfs:
            res_lines = [
                f"- [{rf.id}] {rf.name or rf.id}（已手动解除，原因：{rf.resolved_reason or '已记录'}）"
                for rf in resolved_rfs
            ]
            sections.append("【患者档案卡已解除红旗记录】\n" + "\n".join(res_lines))

        if card.medications:
            med_lines: List[str] = []
            for m in card.medications:
                days = _compute_stopped_days(m)
                extra = []
                if m.start_date:
                    extra.append(f"启用:{m.start_date}")
                if m.stop_date:
                    extra.append(f"停用:{m.stop_date}")
                if days is not None:
                    extra.append(f"已停{days:g}天")
                extra_txt = f"（{'，'.join(extra)}）" if extra else ""
                med_lines.append(f"- {m.name} [status={m.status}]{extra_txt}")
            sections.append("【患者长期/既往用药史】\n" + "\n".join(med_lines))

        if card.allergies:
            sections.append("【过敏史】" + "、".join(card.allergies))

        if card.surgical_history:
            sections.append("【手术及重要病史】" + "、".join(card.surgical_history))

        if card.scale_history:
            sc_lines = [
                f"- {s.date} | {s.scale_id}: {s.total_score} 分（{s.severity_band or s.status}）"
                for s in card.scale_history
            ]
            sections.append("【历史量表轨迹】\n" + "\n".join(sc_lines))

    sanitized_body = _strip_display_name((body or "").strip(), display_name)
    if sanitized_body:
        sections.append("【患者档案补充摘要】\n" + sanitized_body)

    return _strip_display_name("\n\n".join(sections), display_name).strip()
