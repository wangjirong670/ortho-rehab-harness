"""Event-sourced clinical session persistence (.trace.jsonl + read-only .md).

Each clinical encounter/session under case_archives/<patient_id>/sessions/ consists of:
- <session_id>.trace.jsonl: append-only source of truth for full UI replay & audit trail
- <session_id>.md: read-only rendered clinical summary derived strictly from trace events
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

from app_copilot.patient_store import (
    _atomic_write_text,
    _resolve_patient_dir,
    load_patient,
    validate_patient_id,
)

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

TRACE_EVENT_TYPES: FrozenSet[str] = frozenset(
    {
        "session_meta",
        "user_input",
        "pii_check",
        "extraction",
        "scan",
        "scale",
        "web_fetch",
        "generation",
        "verification",
        "human_action",
        "card_diff",
        "education",
    }
)

_TRACE_WRITE_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def validate_session_id(session_id: str) -> str:
    """Enforce strict session_id format to prevent path traversal."""
    cleaned = (session_id or "").strip()
    if not SESSION_ID_PATTERN.fullmatch(cleaned):
        raise ValueError("非法的会话 ID（仅允许 1-80 位字母、数字、下划线或短横线）。")
    return cleaned


def _slugify_title(title: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (title or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    return cleaned[:24]


def generate_session_id(title: str = "") -> str:
    """Generate a deterministic-format session id (<YYYY-MM-DD>_<slug>_<HHMMSS>_<4hex>)."""
    today = date.today().isoformat()
    stamp = datetime.now().strftime("%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    slug = _slugify_title(title)
    if slug:
        candidate = f"{today}_{slug}_{stamp}_{suffix}"
    else:
        candidate = f"{today}_visit_{stamp}_{suffix}"
    return candidate[:80]


def _resolve_sessions_dir(patient_id: str, archives_dir: Optional[Path] = None) -> Path:
    return _resolve_patient_dir(patient_id, archives_dir) / "sessions"


def _resolve_trace_path(
    patient_id: str,
    session_id: str,
    *,
    archives_dir: Optional[Path] = None,
) -> Path:
    safe_sid = validate_session_id(session_id)
    sessions_dir = _resolve_sessions_dir(patient_id, archives_dir).resolve()
    target = (sessions_dir / f"{safe_sid}.trace.jsonl").resolve()
    if not target.is_relative_to(sessions_dir):
        raise ValueError("非法的会话轨迹文件路径。")
    return target


def _resolve_markdown_path(
    patient_id: str,
    session_id: str,
    *,
    archives_dir: Optional[Path] = None,
) -> Path:
    safe_sid = validate_session_id(session_id)
    sessions_dir = _resolve_sessions_dir(patient_id, archives_dir).resolve()
    target = (sessions_dir / f"{safe_sid}.md").resolve()
    if not target.is_relative_to(sessions_dir):
        raise ValueError("非法的会话摘要文件路径。")
    return target


def read_trace_events(
    patient_id: str,
    session_id: str,
    *,
    turn_id: Optional[str] = None,
    archives_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Read all trace events from <session_id>.trace.jsonl (optionally filtered by turn_id)."""
    load_patient(patient_id, archives_dir=archives_dir)
    trace_path = _resolve_trace_path(patient_id, session_id, archives_dir=archives_dir)
    if not trace_path.is_file():
        raise FileNotFoundError(f"未找到会话轨迹：{session_id}")

    events: List[Dict[str, Any]] = []
    for raw_line in trace_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            if turn_id is not None and item.get("turn_id") != turn_id:
                continue
            events.append(item)
    return events


def append_event(
    patient_id: str,
    session_id: str,
    event_type: str,
    payload: Dict[str, Any],
    *,
    turn_id: Optional[str] = None,
    render_md: bool = True,
    archives_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Append a single immutable event to <session_id>.trace.jsonl and optionally refresh .md."""
    if event_type not in TRACE_EVENT_TYPES:
        raise ValueError(f"非法的轨迹事件类型：{event_type}")

    safe_pid = validate_patient_id(patient_id)
    safe_sid = validate_session_id(session_id)
    trace_path = _resolve_trace_path(safe_pid, safe_sid, archives_dir=archives_dir)

    with _TRACE_WRITE_LOCK:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        existing_lines = (
            [ln for ln in trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if trace_path.is_file()
            else []
        )
        seq = len(existing_lines) + 1
        event: Dict[str, Any] = {
            "seq": seq,
            "ts": _now_iso(),
            "turn_id": turn_id,
            "type": event_type,
            "payload": copy.deepcopy(payload) if isinstance(payload, dict) else {"value": payload},
        }
        line_str = json.dumps(event, ensure_ascii=False)
        with trace_path.open("a", encoding="utf-8") as fh:
            fh.write(line_str + "\n")

    if render_md:
        sync_session_markdown(safe_pid, safe_sid, archives_dir=archives_dir)
    return event


def create_session(
    patient_id: str,
    title: str = "",
    *,
    session_id: Optional[str] = None,
    archives_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Initialize a new clinical session under case_archives/<patient_id>/sessions/."""
    safe_pid = validate_patient_id(patient_id)
    load_patient(safe_pid, archives_dir=archives_dir)

    sid = validate_session_id(session_id) if session_id else generate_session_id(title)
    trace_path = _resolve_trace_path(safe_pid, sid, archives_dir=archives_dir)
    if trace_path.exists():
        raise ValueError(f"会话已存在：{sid}")

    clean_title = (title or "").strip() or f"诊疗记录 {date.today().isoformat()}"
    append_event(
        safe_pid,
        sid,
        "session_meta",
        {
            "patient_id": safe_pid,
            "session_id": sid,
            "title": clean_title,
            "created_at": _now_iso(),
        },
        render_md=True,
        archives_dir=archives_dir,
    )
    return replay_session(safe_pid, sid, archives_dir=archives_dir)


def replay_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reconstruct complete SessionState deterministically from append-only trace events."""
    session_id = ""
    patient_id = ""
    title = "诊疗会话"
    created_at = ""
    updated_at = ""

    messages: List[Dict[str, Any]] = []
    history_for_llm: List[Dict[str, str]] = []
    latest_extracted_note: Dict[str, Any] = {}
    latest_risk_report: Dict[str, Any] = {}
    latest_alerts: List[Dict[str, Any]] = []
    latest_scale_results: List[Dict[str, Any]] = []
    pending_card_diffs: List[Dict[str, Any]] = []
    educations: List[Dict[str, Any]] = []
    turn_artifacts: Dict[str, Dict[str, Any]] = {}

    for ev in events:
        etype = str(ev.get("type") or "")
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        ts = str(ev.get("ts") or "")
        turn_id = ev.get("turn_id")
        if ts:
            updated_at = ts

        if turn_id:
            turn_artifacts.setdefault(str(turn_id), {})

        if etype == "session_meta":
            session_id = str(payload.get("session_id") or session_id)
            patient_id = str(payload.get("patient_id") or patient_id)
            title = str(payload.get("title") or title)
            created_at = str(payload.get("created_at") or ts or created_at)
        elif etype == "user_input":
            content = str(payload.get("content") or "")
            messages.append(
                {
                    "role": "user",
                    "content": content,
                    "turn_id": turn_id,
                    "ts": ts,
                    "output_mode": payload.get("output_mode", "full"),
                    "model": payload.get("model"),
                    "thinking": payload.get("thinking", "off"),
                }
            )
            history_for_llm.append({"role": "user", "content": content})
        elif etype == "extraction":
            note = payload.get("extracted_note")
            if isinstance(note, dict):
                latest_extracted_note = copy.deepcopy(note)
                if turn_id:
                    turn_artifacts[str(turn_id)]["extracted_note"] = copy.deepcopy(note)
        elif etype == "scan":
            if isinstance(payload.get("risk_report"), dict):
                latest_risk_report = copy.deepcopy(payload["risk_report"])
            if isinstance(payload.get("unified_alerts"), list):
                latest_alerts = copy.deepcopy(payload["unified_alerts"])
            if turn_id:
                turn_artifacts[str(turn_id)]["alerts"] = copy.deepcopy(latest_alerts)
        elif etype == "scale":
            if isinstance(payload.get("scale_results"), list):
                latest_scale_results = copy.deepcopy(payload["scale_results"])
            elif isinstance(payload.get("scale_result"), dict):
                updated_one = copy.deepcopy(payload["scale_result"])
                sid_val = updated_one.get("scale_id")
                replaced = False
                for idx, existing in enumerate(latest_scale_results):
                    if existing.get("scale_id") == sid_val:
                        latest_scale_results[idx] = updated_one
                        replaced = True
                        break
                if not replaced:
                    latest_scale_results.append(updated_one)
            if isinstance(payload.get("unified_alerts"), list):
                latest_alerts = copy.deepcopy(payload["unified_alerts"])
            if payload.get("mark_stale"):
                for msg in messages:
                    if msg.get("role") == "assistant":
                        msg["stale"] = True
                messages.append(
                    {
                        "role": "system",
                        "content": str(
                            payload.get("system_note")
                            or "量表分值已手动更新并重算，此前模型回复可能已过期。"
                        ),
                        "turn_id": turn_id,
                        "ts": ts,
                    }
                )
            if turn_id:
                turn_artifacts[str(turn_id)]["scale_results"] = copy.deepcopy(latest_scale_results)
        elif etype == "generation":
            content = str(payload.get("content") or "")
            raw_content = str(payload.get("raw_content") or content)
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "raw_content": raw_content,
                "reasoning": str(payload.get("reasoning") or ""),
                "reasoning_verification": copy.deepcopy(
                    payload.get("reasoning_verification")
                    or {"checked": False, "has_conflict": False, "violations": [], "message": ""}
                ),
                "turn_id": turn_id,
                "ts": ts,
                "model": payload.get("model"),
                "mock": bool(payload.get("mock", False)),
                "output_mode": payload.get("output_mode", "full"),
                "blocked": bool(payload.get("blocked", False)),
                "verification": copy.deepcopy(payload.get("verification") or {}),
                "alerts": copy.deepcopy(turn_artifacts.get(str(turn_id), {}).get("alerts", latest_alerts)),
                "scale_results": copy.deepcopy(
                    turn_artifacts.get(str(turn_id), {}).get("scale_results", latest_scale_results)
                ),
                "stale": False,
            }
            messages.append(assistant_msg)
            history_for_llm.append({"role": "assistant", "content": content})
        elif etype == "verification":
            # Attach final verification metadata onto matching assistant turn if present
            if turn_id:
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("turn_id") == turn_id:
                        msg["verification"] = copy.deepcopy(payload)
                        msg["blocked"] = bool(payload.get("blocked", msg.get("blocked", False)))
                        break
        elif etype == "card_diff":
            if payload:
                diff_entry = copy.deepcopy(payload)
                diff_entry.setdefault("diff_id", f"diff_{ev.get('seq')}")
                diff_entry.setdefault("turn_id", turn_id)
                diff_entry.setdefault("ts", ts)
                pending_card_diffs.append(diff_entry)
        elif etype == "human_action":
            action = str(payload.get("action") or "")
            if action == "commit_card_diff":
                target_diff_id = payload.get("diff_id")
                if target_diff_id:
                    pending_card_diffs = [
                        d for d in pending_card_diffs if d.get("diff_id") != target_diff_id
                    ]
                else:
                    pending_card_diffs = []
            elif action == "dismiss_card_diff":
                target_diff_id = payload.get("diff_id")
                pending_card_diffs = [
                    d for d in pending_card_diffs if d.get("diff_id") != target_diff_id
                ]
            elif action == "resolve_red_flag":
                resolved_id = str(payload.get("rule_id") or "").strip().upper()
                if resolved_id:
                    latest_alerts = [
                        a for a in latest_alerts if str(a.get("rule_id") or "").upper() != resolved_id
                    ]
        elif etype == "education":
            educations.append(
                {
                    "ts": ts,
                    "turn_id": turn_id,
                    "content": str(payload.get("content") or ""),
                    "blocked": bool(payload.get("blocked", False)),
                    "model": payload.get("model"),
                    "verification": copy.deepcopy(payload.get("verification") or {}),
                }
            )

    return {
        "session_id": session_id,
        "patient_id": patient_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": messages,
        "history_for_llm": history_for_llm,
        "latest_extracted_note": latest_extracted_note,
        "latest_risk_report": latest_risk_report,
        "latest_alerts": latest_alerts,
        "latest_scale_results": latest_scale_results,
        "pending_card_diffs": pending_card_diffs,
        "educations": educations,
        "event_count": len(events),
    }


def replay_session(
    patient_id: str,
    session_id: str,
    *,
    archives_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load trace events from disk and reconstruct full SessionState."""
    events = read_trace_events(patient_id, session_id, archives_dir=archives_dir)
    state = replay_events(events)
    if not state.get("session_id"):
        state["session_id"] = validate_session_id(session_id)
    if not state.get("patient_id"):
        state["patient_id"] = validate_patient_id(patient_id)
    return state


def render_markdown(events: List[Dict[str, Any]]) -> str:
    """Render append-only trace events into a human-readable read-only Markdown view."""
    state = replay_events(events)
    lines: List[str] = [
        f"# 临床诊疗会话记录：{state.get('title') or state.get('session_id')}",
        "",
        "> 本文件由 `.trace.jsonl` 自动渲染生成，仅供只读查阅；请勿手动编辑。",
        "",
        f"- **患者档案 ID**：`{state.get('patient_id', '')}`",
        f"- **会话 ID**：`{state.get('session_id', '')}`",
        f"- **创建时间**：{state.get('created_at', '-')}",
        f"- **事件总数**：{state.get('event_count', 0)}",
        "",
    ]

    if state.get("latest_alerts"):
        lines.append("## 确定性安全告警汇总")
        for alert in state["latest_alerts"]:
            lines.append(
                f"- **[{alert.get('rule_id', '')}] {alert.get('name', '')}**"
                f"（严重度：{alert.get('severity', '')}）"
            )
        lines.append("")

    if state.get("latest_scale_results"):
        lines.append("## 确定性量表计算结果")
        for res in state["latest_scale_results"]:
            if res.get("status") == "ok":
                lines.append(
                    f"- {res.get('scale_name', res.get('scale_id'))}: "
                    f"**{res.get('total_score')}** 分（{res.get('severity_band', '')}）"
                )
            else:
                lines.append(
                    f"- {res.get('scale_name', res.get('scale_id'))}: 未完成算分（{res.get('message', '')}）"
                )
        lines.append("")

    lines.append("## 对话与诊疗轨迹")
    for msg in state.get("messages") or []:
        role = msg.get("role")
        ts = msg.get("ts") or ""
        if role == "user":
            lines.append(f"### 临床输入（{ts}）")
            lines.append(str(msg.get("content") or ""))
            lines.append("")
        elif role == "assistant":
            model_tag = f" | 模型 `{msg.get('model')}`" if msg.get("model") else ""
            stale_tag = " | [已过期]" if msg.get("stale") else ""
            blocked_tag = " | [已拦截]" if msg.get("blocked") else ""
            lines.append(f"### 护栏辅助回复（{ts}{model_tag}{blocked_tag}{stale_tag}）")
            lines.append(str(msg.get("content") or ""))
            lines.append("")
        else:
            lines.append(f"> 系统提示（{ts}）：{msg.get('content', '')}")
            lines.append("")

    if state.get("educations"):
        lines.append("## 独立患者宣教文书")
        for idx, edu in enumerate(state["educations"], start=1):
            lines.append(f"### 宣教文书 #{idx}（{edu.get('ts', '')}）")
            lines.append(str(edu.get("content") or ""))
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def sync_session_markdown(
    patient_id: str,
    session_id: str,
    *,
    archives_dir: Optional[Path] = None,
) -> Path:
    """Re-render and atomically write <session_id>.md from <session_id>.trace.jsonl."""
    events = read_trace_events(patient_id, session_id, archives_dir=archives_dir)
    md_text = render_markdown(events)
    md_path = _resolve_markdown_path(patient_id, session_id, archives_dir=archives_dir)
    _atomic_write_text(md_path, md_text)
    return md_path


def list_sessions(
    patient_id: str,
    *,
    archives_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List all sessions under case_archives/<patient_id>/sessions/ sorted newest first."""
    safe_pid = validate_patient_id(patient_id)
    load_patient(safe_pid, archives_dir=archives_dir)
    sessions_dir = _resolve_sessions_dir(safe_pid, archives_dir)
    if not sessions_dir.is_dir():
        return []

    summaries: List[Dict[str, Any]] = []
    for trace_file in sessions_dir.glob("*.trace.jsonl"):
        if not trace_file.is_file():
            continue
        sid = trace_file.name[: -len(".trace.jsonl")]
        if not SESSION_ID_PATTERN.fullmatch(sid):
            continue
        try:
            events = read_trace_events(safe_pid, sid, archives_dir=archives_dir)
            state = replay_events(events)
        except Exception:
            continue
        alerts = state.get("latest_alerts") or []
        turn_count = sum(1 for m in state.get("messages") or [] if m.get("role") == "user")
        summaries.append(
            {
                "session_id": sid,
                "patient_id": safe_pid,
                "title": state.get("title") or sid,
                "created_at": state.get("created_at") or "",
                "updated_at": state.get("updated_at") or "",
                "turn_count": turn_count,
                "alert_count": len(alerts),
                "has_critical": any(
                    str(a.get("severity") or "").lower() == "critical" for a in alerts
                ),
                "pending_diff_count": len(state.get("pending_card_diffs") or []),
            }
        )

    summaries.sort(key=lambda s: str(s.get("updated_at") or s.get("created_at") or ""), reverse=True)
    return summaries
