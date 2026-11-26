"""Streamlit UI and pure orchestration logic for Project B desktop copilot (Week 3)."""

from __future__ import annotations

import copy
import io
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
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
from app_copilot.service import (
    CLINICAL_ACTION_ZH,
    MED_STATUS_ZH,
    POLARITY_ZH,
    PRESET_MODELS,
    SCORE_DIRECTION_LABELS,
    SEVERITY_BADGE_MAP,
    SOURCE_LABEL_ZH,
    TCM_FIELD_LABELS,
    WOMAC_SCORE_LABELS,
    build_clinical_handover_text,
    collect_unified_alerts,
    describe_score_direction,
    detect_potential_pii,
    evaluate_scale_entries,
    extract_text_from_uploaded_file,
    format_active_medication_stop_display,
    generate_draft_document,
    markdown_to_plain_text,
    parse_discrete_score_options,
    recalculate_scale_in_state,
    remove_scale_from_state,
    run_copilot_analysis,
    run_fast_safety_scan,
    sanitize_clinical_pii,
    sanitize_error_message,
    summarize_contraindicated_capsules,
    verify_draft_against_alerts,
)
from core_tools.calculator import (
    calculate_scale,
    list_available_scales,
    load_scales_catalog,
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


def _record_analysis_history(state: Dict[str, Any], *, label_prefix: str = "") -> None:
    """Keep the most recent 5 anonymized analysis runs in session state."""
    history: List[Dict[str, Any]] = list(st.session_state.get("analysis_history") or [])
    note = state.get("extracted_note") or {}
    snippet = (
        str(note.get("chief_complaint") or "").strip()
        or str(state.get("raw_text") or "").strip()[:32]
        or "脱敏病历"
    )[:24]
    alert_cnt = len(state.get("unified_alerts") or [])
    ts = datetime.now().strftime("%H:%M:%S")
    title = f"{label_prefix}{ts} | {snippet} ({alert_cnt}项警示)"
    entry = {"title": title, "state": copy.deepcopy(state)}
    history.insert(0, entry)
    st.session_state["analysis_history"] = history[:5]


def _restore_history_entry(index: int) -> None:
    history = st.session_state.get("analysis_history") or []
    if 0 <= index < len(history):
        restored = copy.deepcopy(history[index]["state"])
        st.session_state["copilot_state"] = restored
        st.session_state["raw_note_input"] = restored.get("raw_text", "")
        st.session_state["analysis_run_id"] = st.session_state.get("analysis_run_id", 0) + 1


def _render_sidebar(default_config: LLMConfig) -> tuple[str, bool]:
    with st.sidebar:
        st.header("⚙️ 副驾配置与演示数据")
        st.warning(
            "⚠️ **合规安全边界**\n\n"
            "仅限**脱敏病历**与临床思维辅助，**不直接对患者下诊断或开具处方**。"
            "严禁输入真实姓名、住院号、身份证号等患者隐私标识。"
        )

        st.subheader("模型与运行模式")
        if "probed_ollama_models" not in st.session_state:
            st.session_state["probed_ollama_models"] = probe_ollama_models(timeout=0.4)

        col_m1, col_m2 = st.columns([3, 1])
        with col_m2:
            if st.button("🔄", key="refresh_ollama_btn", help="刷新本地 Ollama 已安装模型列表"):
                st.session_state["probed_ollama_models"] = probe_ollama_models(timeout=1.5)

        ollama_models: List[str] = list(st.session_state.get("probed_ollama_models") or [])
        merged_presets: List[str] = list(PRESET_MODELS)
        for om in ollama_models:
            if om not in merged_presets:
                merged_presets.append(om)

        default_model = default_config.default_model
        model_options = merged_presets + ["自定义模型..."]
        default_idx = (
            merged_presets.index(default_model)
            if default_model in merged_presets
            else len(model_options) - 1
        )
        with col_m1:
            selected_preset = st.selectbox(
                "选择大模型",
                options=model_options,
                index=default_idx,
                key="model_preset_select",
            )
        if ollama_models:
            st.caption(f"🟢 已探测到本地 Ollama 模型 {len(ollama_models)} 个")

        if selected_preset == "自定义模型...":
            custom_default = default_model if default_model not in merged_presets else ""
            chosen_model = (
                st.text_input(
                    "自定义模型名",
                    value=custom_default,
                    placeholder="例如: ollama_chat/qwen2.5:7b",
                    key="custom_model_input",
                ).strip()
                or default_model
            )
        else:
            chosen_model = selected_preset

        mock_mode = st.toggle(
            "离线演示模式（Mock）",
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

        history = st.session_state.get("analysis_history") or []
        if history:
            st.divider()
            st.subheader("🕒 最近分析记录（保留 5 条）")
            for idx, item in enumerate(history):
                st.button(
                    item["title"],
                    key=f"history_restore_btn_{idx}",
                    width="stretch",
                    on_click=_restore_history_entry,
                    args=(idx,),
                )

    return chosen_model, mock_mode


def _render_single_alert_card(alert: Dict[str, Any]) -> None:
    severity = str(alert.get("severity", "high"))
    badge = SEVERITY_BADGE_MAP.get(severity, severity.upper())
    rule_id = alert.get("rule_id", "UNKNOWN")
    title = alert.get("name", "")
    raw_source = str(alert.get("source", ""))
    source_label = alert.get("source_label") or SOURCE_LABEL_ZH.get(raw_source, raw_source)
    raw_action = str(alert.get("clinical_action", ""))
    action_zh = CLINICAL_ACTION_ZH.get(raw_action, raw_action)

    review_tags: List[str] = []
    if alert.get("needs_review"):
        review_tags.append("⚠️ 需人工复核 (needs_review)")
    if alert.get("conflict_with_text"):
        review_tags.append("⚠️ 结构化与原文冲突 (conflict_with_text)")
    tag_suffix = f" | {' · '.join(review_tags)}" if review_tags else ""

    with st.container(border=True):
        header_msg = (
            f"**[{rule_id}] {title}** | 严重度：`{badge}` | 来源：`{source_label}`{tag_suffix}"
        )
        if severity in ("critical", "high"):
            st.error(header_msg)
        else:
            st.warning(header_msg)
        if review_tags:
            conflict_detail = (
                alert.get("conflict_detail")
                or alert.get("review_reason")
                or "模型结构化事实与病历原文证据存在潜在冲突或边界否定抑制，请临床医师人工核实原文！"
            )
            st.warning(f"🔎 **人工复核提示**：{conflict_detail}")
        evidence = "；".join(str(x) for x in alert.get("matched_evidence") or [])
        st.markdown(f"- **命中证据**：{evidence or '见结构化事实'}")
        st.markdown(f"- **建议处置动作**：**{action_zh}**")
        st.markdown(f"- **临床警示**：{alert.get('clinical_alert', '')}")
        if alert.get("contraindicated_actions"):
            contra = "；".join(str(x) for x in alert["contraindicated_actions"])
            st.markdown(f"- **绝对禁忌操作**：{contra}")
        if alert.get("suggested_handling"):
            st.markdown(f"- **处置建议**：{alert['suggested_handling']}")
        if alert.get("risk_mechanism"):
            st.markdown(f"- **风险机制**：{alert['risk_mechanism']}")
        refs = "；".join(str(x) for x in alert.get("reference") or [])
        st.caption(f"📚 教材 / 指南依据：{refs or '未注明'}")


def _render_top_safety_banner(state: Dict[str, Any]) -> None:
    unified_alerts = state.get("unified_alerts") or []
    risk_report = state.get("risk_report") or {}
    suppressed = risk_report.get("suppressed_matches") or []

    if unified_alerts:
        has_high_or_critical = any(
            str(a.get("severity", "high")) in ("critical", "high") for a in unified_alerts
        )
        capsules = summarize_contraindicated_capsules(unified_alerts)
        capsule_line = "  ".join(f"`{tag}`" for tag in capsules)
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
        if capsule_line:
            st.markdown(f"**⚡ 红线禁忌速览**：{capsule_line}")

        if len(unified_alerts) > 2:
            critical_alerts = [a for a in unified_alerts if str(a.get("severity", "")) == "critical"]
            other_alerts = [a for a in unified_alerts if str(a.get("severity", "")) != "critical"]
            for alert in critical_alerts:
                _render_single_alert_card(alert)
            if other_alerts:
                with st.expander(
                    f"📋 其余 {len(other_alerts)} 项高危/提示级安全约束（点击展开明细）",
                    expanded=False,
                ):
                    for alert in other_alerts:
                        _render_single_alert_card(alert)
        else:
            for alert in unified_alerts:
                _render_single_alert_card(alert)
    else:
        st.success("✅ **安全扫描通过**：确定性双轨扫描与量表阈值未命中任何红旗征或高危用药冲突。")

    with st.expander(
        f"🛡️ 误报拦截审计记录（否定词与洗脱期过滤：共 {len(suppressed)} 条）",
        expanded=False,
    ):
        if suppressed:
            rows = [
                {
                    "规则ID": s.get("rule_id", ""),
                    "类型": "红旗征" if s.get("rule_type") == "red_flag" else "用药冲突",
                    "命中词": s.get("term", ""),
                    "拦截原因": s.get("reason", ""),
                    "来源": SOURCE_LABEL_ZH.get(str(s.get("source", "")), str(s.get("source", ""))),
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
    st.markdown("#### 🕒 跨诊次时间线演变")
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
        st.markdown("**💊 当前在用 / 未满洗脱期药物**")
        active_meds = snapshot.get("active_medications") or []
        if active_meds:
            st.dataframe(
                [
                    {
                        "药物": m.get("name", ""),
                        "状态": MED_STATUS_ZH.get(str(m.get("status", "")), str(m.get("status", ""))),
                        "停药天数": format_active_medication_stop_display(m),
                    }
                    for m in active_meds
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("当前无高风险在用或洗脱期内药物。")

    with snap_c2:
        st.markdown("**🛑 已停用且超洗脱期药物**")
        stopped_meds = snapshot.get("stopped_medications") or []
        if stopped_meds:
            st.dataframe(
                [
                    {
                        "药物": m.get("name", ""),
                        "状态": MED_STATUS_ZH.get(str(m.get("status", "stopped")), "已停药"),
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
        st.markdown("**🦴 长期手术 / 内固定 / 病史约束**")
        for sh in surg_history:
            st.markdown(f"- `{sh}`")

    superseded_list = snapshot.get("superseded_conditions") or []
    if superseded_list:
        st.markdown("**🔄 已被最新诊次推翻 / 已过保护期的旧状态**")
        st.dataframe(
            [
                {
                    "过期旧状态": item.get("old_state", ""),
                    "最新状态": item.get("new_state", ""),
                    "解除规则提示": item.get("rule_hint") or "—",
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
        st.markdown("#### 结构化症状极性")
        syms = facts.get("symptoms") or []
        if syms:
            st.dataframe(
                [
                    {
                        "症状/体征": s.get("name", ""),
                        "判定状态": POLARITY_ZH.get(str(s.get("polarity", "")), str(s.get("polarity", ""))),
                    }
                    for s in syms
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("未识别到结构化红旗相关症状。")
    with c_med:
        st.markdown("#### 结构化用药状态")
        meds = facts.get("medications") or []
        if meds:
            st.dataframe(
                [
                    {
                        "药物/制剂": m.get("name", ""),
                        "当前状态": MED_STATUS_ZH.get(str(m.get("status", "")), str(m.get("status", ""))),
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


def _render_single_scale_item_widget(
    spec: Any,
    item: Any,
    raw_val: Any,
    item_key: str,
) -> Optional[float]:
    score_options = parse_discrete_score_options(spec, item)
    if score_options is not None:
        sorted_vals = sorted(score_options.keys())
        options: List[Optional[float]] = [None] + sorted_vals
        current = (
            float(raw_val)
            if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool)
            else None
        )
        default_opt_idx = options.index(current) if current in options else 0
        val = st.selectbox(
            f"{item.label}",
            options=options,
            index=default_opt_idx,
            format_func=lambda v, m=score_options: "⚠️ 未填写（请补录）" if v is None else m.get(v, f"{v:g} 分"),
            key=item_key,
        )
        return float(val) if val is not None else None

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
    fmt = "%d" if step == 1.0 and min_v.is_integer() and max_v.is_integer() else "%.1f"
    val = st.number_input(
        f"{item.label}（{min_v:g}~{max_v:g} 分）",
        min_value=min_v,
        max_value=max_v,
        value=default_num,
        step=step,
        format=fmt,
        placeholder="留空表示未填写（严禁默认补0）",
        key=item_key,
    )
    return float(val) if val is not None else None


def _render_scale_manual_inputs(
    spec: Any,
    existing_params: Dict[str, Any],
    key_prefix: str,
) -> Dict[str, Any]:
    """Render dynamic manual input widgets grouped by subscales without defaulting missing to 0."""
    collected: Dict[str, Any] = {}
    subscales = list(getattr(spec, "subscales", None) or [])

    if subscales:
        items_by_sub: Dict[str, List[Any]] = {sub.subscale_id: [] for sub in subscales}
        for item in spec.items:
            if item.subscale_id in items_by_sub:
                items_by_sub[item.subscale_id].append(item)

        def _is_filled(it: Any) -> bool:
            rv = existing_params.get(it.item_id)
            return isinstance(rv, (int, float)) and not isinstance(rv, bool)

        # Prioritize incomplete subscales at the top
        ordered_subs = sorted(
            subscales,
            key=lambda sub: sum(1 for it in items_by_sub[sub.subscale_id] if not _is_filled(it)) == 0,
        )

        for sub in ordered_subs:
            sub_items = items_by_sub[sub.subscale_id]
            filled_cnt = sum(1 for it in sub_items if _is_filled(it))
            missing_cnt = len(sub_items) - filled_cnt
            if missing_cnt > 0:
                header_label = f"⚠️ {sub.label} — 待补录 {missing_cnt}/{len(sub_items)} 项"
                expanded = True
            else:
                header_label = f"✅ {sub.label} — 已填满 {filled_cnt}/{len(sub_items)} 项（点击展开核对）"
                expanded = False

            with st.expander(header_label, expanded=expanded):
                cols = st.columns(2)
                for idx, item in enumerate(sub_items):
                    with cols[idx % 2]:
                        val = _render_single_scale_item_widget(
                            spec,
                            item,
                            existing_params.get(item.item_id),
                            f"{key_prefix}_{item.item_id}",
                        )
                        if val is not None:
                            collected[item.item_id] = val
    else:
        cols = st.columns(2)
        for idx, item in enumerate(spec.items):
            with cols[idx % 2]:
                val = _render_single_scale_item_widget(
                    spec,
                    item,
                    existing_params.get(item.item_id),
                    f"{key_prefix}_{item.item_id}",
                )
                if val is not None:
                    collected[item.item_id] = val

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
            f"可选：术前基线总分（用于计算改善率，{spec.min_total:g}~{spec.max_total:g} 分）",
            min_value=float(spec.min_total),
            max_value=float(spec.max_total),
            value=default_pre,
            step=1.0,
            format="%d",
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
                zh_err, tech_err = format_llm_error_zh(str(exc), model=chosen_model)
                st.error(f"刷新文书草稿失败：\n\n{zh_err}")
                with st.expander("🔍 查看技术细节（已脱敏）", expanded=False):
                    st.code(tech_err, language="text")

    if not scale_results:
        st.info("病历文本中未检测到明确量表条目分值，可在下方“手动新增量表评估”选择量表计算。")

    for idx, res in enumerate(scale_results):
        scale_id = res.get("scale_id", "")
        spec = spec_map.get(scale_id)
        direction_text = describe_score_direction(res.get("score_direction"))

        with st.container(border=True):
            hdr_col, rm_col = st.columns([5, 1])
            with rm_col:
                if st.button("🗑️ 移除", key=f"remove_scale_{run_id}_{idx}_{scale_id}", width="stretch"):
                    st.session_state["copilot_state"] = remove_scale_from_state(state, idx)
                    st.rerun()

            if res.get("status") == "ok":
                with hdr_col:
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
                        b_sync, b_only = st.columns(2)
                        with b_sync:
                            if st.button(
                                "🔄 重算并同步刷新文书",
                                type="primary",
                                key=f"recalc_sync_ok_{run_id}_{idx}_{scale_id}",
                                width="stretch",
                            ):
                                try:
                                    new_state = recalculate_scale_in_state(
                                        state,
                                        scale_id,
                                        updated_params,
                                        scale_index=idx,
                                        refresh_draft=True,
                                        model=chosen_model,
                                        mock_mode=mock_mode,
                                    )
                                    st.session_state["copilot_state"] = new_state
                                    st.rerun()
                                except Exception as exc:
                                    zh_err, tech_err = format_llm_error_zh(str(exc), model=chosen_model)
                                    st.error(f"重算并刷新文书失败：\n\n{zh_err}")
                                    with st.expander("🔍 查看技术细节（已脱敏）", expanded=False):
                                        st.code(tech_err, language="text")
                        with b_only:
                            if st.button(
                                "仅重算量表",
                                key=f"recalc_ok_{run_id}_{idx}_{scale_id}",
                                width="stretch",
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
            else:
                with hdr_col:
                    st.subheader(f"⚠️ {res.get('scale_name', scale_id)} (`{scale_id}`) — 待补录或修正")
                st.caption(f"🧭 **风险方向说明**：{direction_text}")
                missing = res.get("missing_fields") or []
                invalid = res.get("invalid_fields") or []
                warn_lines = [
                    f"**{res.get('scale_name', scale_id)} (`{scale_id}`) 未完成确定性算分：{res.get('message', '')}**"
                ]
                if missing and spec is not None:
                    label_map = {it.item_id: it.label for it in spec.items}
                    missing_desc = "、".join(f"{label_map.get(fid, fid)}" for fid in missing[:8])
                    if len(missing) > 8:
                        missing_desc += f" 等共 {len(missing)} 项"
                    warn_lines.append(f"- **缺失必填字段 ({len(missing)} 项)**：{missing_desc}")
                if invalid:
                    invalid_desc = "；".join(f"{f.get('field')}: {f.get('reason')}" for f in invalid)
                    warn_lines.append(f"- **越界或非法字段 ({len(invalid)} 项)**：{invalid_desc}")
                st.warning("\n".join(warn_lines))

                if spec is not None:
                    st.markdown("**请手动补录以下条目分值（未填写条目将保持缺失校验，绝不默认填 0）：**")
                    updated_params = _render_scale_manual_inputs(
                        spec,
                        res.get("input_params") or {},
                        key_prefix=f"fix_scale_{run_id}_{idx}_{scale_id}",
                    )
                    b_only, b_sync = st.columns(2)
                    with b_only:
                        if st.button(
                            "仅重算量表",
                            type="primary",
                            key=f"recalc_err_{run_id}_{idx}_{scale_id}",
                            width="stretch",
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
                    with b_sync:
                        if st.button(
                            "🔄 重算并同步刷新文书",
                            key=f"recalc_sync_err_{run_id}_{idx}_{scale_id}",
                            width="stretch",
                        ):
                            try:
                                new_state = recalculate_scale_in_state(
                                    state,
                                    scale_id,
                                    updated_params,
                                    scale_index=idx,
                                    refresh_draft=True,
                                    model=chosen_model,
                                    mock_mode=mock_mode,
                                )
                                st.session_state["copilot_state"] = new_state
                                st.rerun()
                            except Exception as exc:
                                zh_err, tech_err = format_llm_error_zh(str(exc), model=chosen_model)
                                st.error(f"重算并刷新文书失败：\n\n{zh_err}")
                                with st.expander("🔍 查看技术细节（已脱敏）", expanded=False):
                                    st.code(tech_err, language="text")

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
        existing_ids = {r.get("scale_id") for r in scale_results}
        already_exists = chosen_add_id in existing_ids
        if already_exists:
            st.warning(
                f"⚠️ 上方已存在 **{scale_labels.get(chosen_add_id, chosen_add_id)}** 的评估卡片，"
                "请直接在对应卡片中修改并重算（或先点击卡片右上角「🗑️ 移除」后再重新添加）。"
            )
        add_spec = spec_map.get(chosen_add_id)
        if add_spec is not None and not already_exists:
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
            zh_err, tech_err = format_llm_error_zh(str(exc), model=chosen_model)
            st.error(f"文书草稿生成失败：\n\n{zh_err}")
            with st.expander("🔍 查看技术细节（已脱敏）", expanded=False):
                st.code(tech_err, language="text")

    if state.get("draft_stale"):
        st.warning("⚠️ 量表结果已更新，当前草稿仍基于更新前的数据，请点击 Tab 2 同步按钮或上方重新生成。")

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

    plain_text_draft = markdown_to_plain_text(draft_content)
    handover_text = build_clinical_handover_text(state)

    st.markdown("#### 📋 一键复制与文件导出")
    copy_mode = st.radio(
        "选择复制格式",
        options=["plain", "markdown"],
        format_func=lambda x: (
            "📋 纯文本版（去 Markdown 符号，适合直接粘进医院病历 / 微信 / 钉钉）"
            if x == "plain"
            else "📝 Markdown 原文版"
        ),
        horizontal=True,
        key=f"draft_copy_mode_{run_id}",
    )
    if copy_mode == "plain":
        st.code(plain_text_draft, language=None)
    else:
        st.code(draft_content, language="markdown")

    dl_c1, dl_c2, dl_c3 = st.columns(3)
    with dl_c1:
        st.download_button(
            "⬇️ 导出纯文本 (.txt)",
            data=plain_text_draft.encode("utf-8"),
            file_name=f"ortho_draft_{current_mode}.txt",
            mime="text/plain",
            width="stretch",
            key=f"dl_txt_{run_id}",
        )
    with dl_c2:
        st.download_button(
            "⬇️ 导出 Markdown (.md)",
            data=draft_content.encode("utf-8"),
            file_name=f"ortho_draft_{current_mode}.md",
            mime="text/markdown",
            width="stretch",
            key=f"dl_md_{run_id}",
        )
    with dl_c3:
        st.download_button(
            "⬇️ 导出合并临床交班单 (.txt)",
            data=handover_text.encode("utf-8"),
            file_name="ortho_clinical_handover.txt",
            mime="text/plain",
            width="stretch",
            key=f"dl_handover_{run_id}",
        )


def _render_tab_trace(state: Dict[str, Any]) -> None:
    trace = state.get("trace") or {}
    st.markdown("#### 模型与运行元信息（工程调试审计）")
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
            "fast_scan_only": bool(state.get("fast_scan_only")),
        }
    )

    with st.expander("1️⃣ 原始结构化提取 JSON (`extracted_note`)", expanded=False):
        st.json(state.get("extracted_note") or {})

    with st.expander("2️⃣ 双轨规则扫描完整 JSON (`risk_report`)", expanded=True):
        st.json(state.get("risk_report") or {})

    with st.expander("3️⃣ 确定性量表计算结果 JSON (`scale_results`)", expanded=False):
        st.json(state.get("scale_results") or [])


def _render_tab_kb_reference() -> None:
    """Render read-only Chinese knowledge base quick reference for clinicians."""
    rf_catalog = load_red_flags_catalog()
    dc_catalog = load_drug_conflicts_catalog()
    scales_catalog = load_scales_catalog()

    st.caption("以下规则与量表均实时读取自 `knowledge_base/*.yaml`（只读速查，所有临床决策须由执业医师复核）。")
    sub_rf, sub_dc, sub_sc = st.tabs(
        [
            f"🚨 红旗规则库（{len(rf_catalog.rules)} 条）",
            f"💊 用药冲突与洗脱窗（{len(dc_catalog.conflicts)} 组）",
            f"📏 骨伤康复量表（{len(scales_catalog.scales)} 套）",
        ]
    )

    with sub_rf:
        rf_rows = [
            {
                "规则编号": r.rule_id,
                "红旗名称": r.name,
                "分类": r.category,
                "严重度": SEVERITY_BADGE_MAP.get(r.severity, r.severity),
                "触发核心词": "、".join(r.trigger_symptoms.required_any[:6]),
                "绝对禁忌操作": "；".join(r.contraindicated_actions),
                "建议动作": CLINICAL_ACTION_ZH.get(r.recommended_action, r.recommended_action),
                "指南依据": "；".join(r.references),
            }
            for r in rf_catalog.rules
        ]
        st.dataframe(rf_rows, width="stretch", hide_index=True)

    with sub_dc:
        dc_rows = [
            {
                "冲突编号": c.conflict_id,
                "冲突组合": c.name,
                "严重度": SEVERITY_BADGE_MAP.get(c.severity, c.severity),
                "A组药物": f"{c.group_a.label}（{'、'.join(c.group_a.keywords[:5])}）",
                "A组洗脱窗": (
                    f"{c.group_a.washout_window_days} 天"
                    if c.group_a.washout_window_days is not None
                    else "—"
                ),
                "B组药物/状态": f"{c.group_b.label}（{'、'.join(c.group_b.keywords[:5])}）",
                "B组洗脱窗": (
                    f"{c.group_b.washout_window_days} 天"
                    if c.group_b.washout_window_days is not None
                    else "—"
                ),
                "临床处置": c.suggested_handling,
                "参考依据": "；".join(c.references),
            }
            for c in dc_catalog.conflicts
        ]
        st.dataframe(dc_rows, width="stretch", hide_index=True)

    with sub_sc:
        sc_rows = [
            {
                "量表ID": s.scale_id,
                "量表名称": s.name,
                "条目数": len(s.items),
                "分值范围": f"{s.min_total:g} ~ {s.max_total:g}",
                "分值方向": describe_score_direction(
                    "higher_is_worse" if s.direction == "higher_is_worse" else "lower_is_worse"
                ),
                "说明": s.version_note,
                "参考文献": "；".join(s.references),
            }
            for s in scales_catalog.scales
        ]
        st.dataframe(sc_rows, width="stretch", hide_index=True)


def _handle_sanitize_input() -> None:
    raw = st.session_state.get("raw_note_input", "")
    cleaned, counts = sanitize_clinical_pii(raw)
    if counts["total"] > 0:
        st.session_state["pre_sanitize_backup"] = raw
        st.session_state["raw_note_input"] = cleaned
        st.session_state["last_sanitize_msg"] = (
            f"已完成本地脱敏：姓名 {counts['person_name']} 处、手机号 {counts['phone']} 处、"
            f"身份证号 {counts['id_card']} 处、住院/病历号 {counts['record_id']} 处。"
        )
    else:
        st.session_state["last_sanitize_msg"] = "未检测到需要替换的显式隐私字段（未改动医学词汇）。"


def _handle_undo_sanitize() -> None:
    backup = st.session_state.get("pre_sanitize_backup")
    if backup is not None:
        st.session_state["raw_note_input"] = backup
        st.session_state["pre_sanitize_backup"] = None
        st.session_state["last_sanitize_msg"] = "已撤销上次本地脱敏。"


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
    st.session_state.setdefault("pre_sanitize_backup", None)
    st.session_state.setdefault("last_sanitize_msg", "")
    st.session_state.setdefault("analysis_history", [])
    st.session_state.setdefault("last_uploaded_file_sig", None)

    chosen_model, mock_mode = _render_sidebar(default_config)

    st.title("🩺 OrthoRehab 骨伤 / 康复桌面干活副驾")
    st.caption(
        "脱敏病历结构化 → 双轨确定性红旗/用药冲突扫描 → 严校验量表算分 → 高约束查房/康复宣教文书"
    )

    uploaded_file = st.file_uploader(
        "📂 上传脱敏病历文件（支持 .txt / .md / .docx，读取后自动填入下方文本框）",
        type=["txt", "md", "docx"],
        key="clinical_note_uploader",
    )
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        file_sig = f"{uploaded_file.name}:{len(raw_bytes)}"
        if st.session_state.get("last_uploaded_file_sig") != file_sig:
            try:
                extracted_txt = extract_text_from_uploaded_file(uploaded_file.name, raw_bytes)
                st.session_state["raw_note_input"] = extracted_txt
                st.session_state["last_uploaded_file_sig"] = file_sig
                st.session_state["copilot_state"] = None
                st.success(f"✅ 已导入文件 `{uploaded_file.name}`（共 {len(extracted_txt)} 字），请先核查脱敏情况。")
            except Exception as exc:
                st.error(f"文件导入失败：{exc}")

    st.text_area(
        "脱敏病历 / 跨诊次记录原文",
        key="raw_note_input",
        height=210,
        placeholder="请粘贴已脱敏的骨伤/康复病历文本，或点击左侧边栏演示病例按钮一键填充...",
    )

    pii_stats = detect_potential_pii(st.session_state.get("raw_note_input", ""))
    if pii_stats["total"] > 0:
        st.warning(
            f"🔒 **隐私保护提醒**：检测到文本中可能包含 {pii_stats['total']} 处敏感标识"
            f"（显式姓名 {pii_stats['person_name']} / 手机号 {pii_stats['phone']} / "
            f"身份证号 {pii_stats['id_card']} / 住院或病历号 {pii_stats['record_id']}），"
            "强烈建议先点击下方「🧹 一键本地脱敏」后再分析！"
        )

    pii_c1, pii_c2, pii_c3 = st.columns([1, 1, 3])
    with pii_c1:
        st.button("🧹 一键本地脱敏", key="sanitize_pii_btn", width="stretch", on_click=_handle_sanitize_input)
    with pii_c2:
        st.button(
            "↩️ 撤销脱敏",
            key="undo_sanitize_btn",
            width="stretch",
            disabled=st.session_state.get("pre_sanitize_backup") is None,
            on_click=_handle_undo_sanitize,
        )
    with pii_c3:
        if st.session_state.get("last_sanitize_msg"):
            st.caption(st.session_state["last_sanitize_msg"])

    col_mode, col_run, col_fast, col_clear = st.columns([1.8, 1.1, 1.3, 0.8])
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
    with col_fast:
        st.write("")
        fast_clicked = st.button(
            "⚡ 极速安全筛查",
            width="stretch",
            key="fast_scan_btn",
            help="仅运行确定性红旗与用药冲突规则引擎，零大模型等待，秒出安全红线。",
        )
    with col_clear:
        st.write("")
        st.button(
            "🧹 清空",
            width="stretch",
            key="clear_btn",
            on_click=_clear_workspace,
        )

    if fast_clicked:
        raw_input = st.session_state.get("raw_note_input", "")
        if not raw_input.strip():
            st.warning("请先输入脱敏病历文本，或从左侧边栏选择一键填充演示病例。")
        else:
            try:
                fast_state = run_fast_safety_scan(raw_input)
                st.session_state["copilot_state"] = fast_state
                st.session_state["analysis_run_id"] = st.session_state.get("analysis_run_id", 0) + 1
                _record_analysis_history(fast_state, label_prefix="⚡ ")
            except Exception as exc:
                st.error(f"极速筛查失败：{sanitize_error_message(exc)}")

    if run_clicked:
        raw_input = st.session_state.get("raw_note_input", "")
        if not raw_input.strip():
            st.warning("请先输入脱敏病历文本，或从左侧边栏选择一键填充演示病例。")
        else:
            try:
                t_start = time.perf_counter()
                with st.status("正在执行 4 步骨伤/康复辅助分析...", expanded=True) as status_box:
                    st.write(f"1/4 正在调用 `{chosen_model}`（{'Mock' if mock_mode else '在线'}）执行结构化与时间线抽取...")
                    active_client = LLMClient()
                    extraction_messages = build_extraction_messages(raw_input.strip())
                    extract_res = active_client.extract_structured_json(
                        extraction_messages,
                        schema_model=CopilotExtractedNote,
                        model=chosen_model,
                        mock_mode=mock_mode,
                    )
                    if extract_res.get("status") != "ok" or not isinstance(extract_res.get("data"), dict):
                        raise RuntimeError(extract_res.get("message") or "病历结构化抽取失败。")
                    extracted_note = extract_res["data"]

                    st.write("2/4 正在运行双轨确定性红旗、洗脱期与时间线冲突扫描...")
                    merged_facts = merge_snapshot_into_structured_facts(
                        extracted_note.get("current_state_snapshot"),
                        fallback_facts=extracted_note.get("structured_facts"),
                    )
                    risk_report = scan_clinical_risks(
                        raw_text=raw_input.strip(),
                        structured_facts=merged_facts,
                    )

                    st.write("3/4 正在执行确定性量表校验与算分...")
                    scale_results = evaluate_scale_entries(extracted_note.get("scale_params", []))

                    st.write("4/4 正在生成高约束临床文书并执行安全复检...")
                    draft_result = generate_draft_document(
                        selected_mode,
                        extracted_note,
                        risk_report,
                        scale_results,
                        model=chosen_model,
                        mock_mode=mock_mode,
                        client=active_client,
                    )
                    unified_alerts = collect_unified_alerts(risk_report, scale_results)
                    elapsed_s = time.perf_counter() - t_start
                    status_box.update(
                        label=f"✅ 分析完成（耗时 {elapsed_s:.2f}s | 检出 {len(unified_alerts)} 项警示）",
                        state="complete",
                        expanded=False,
                    )

                result_state = {
                    "raw_text": raw_input.strip(),
                    "extracted_note": extracted_note,
                    "risk_report": risk_report,
                    "scale_results": scale_results,
                    "unified_alerts": unified_alerts,
                    "draft": draft_result,
                    "draft_mode": selected_mode,
                    "draft_stale": False,
                    "trace": {
                        "extraction_model": extract_res.get("model"),
                        "extraction_mock": extract_res.get("mock"),
                        "extraction_raw_content": extract_res.get("raw_content"),
                        "draft_model": draft_result.get("model"),
                        "draft_mock": draft_result.get("mock"),
                        "draft_usage": draft_result.get("usage", {}),
                        "elapsed_seconds": round(elapsed_s, 2),
                    },
                }
                st.session_state["copilot_state"] = result_state
                st.session_state["analysis_run_id"] = st.session_state.get("analysis_run_id", 0) + 1
                _record_analysis_history(result_state)
            except Exception as exc:
                zh_err, tech_err = format_llm_error_zh(str(exc), model=chosen_model)
                st.error(f"分析执行失败：\n\n{zh_err}")
                with st.expander("🔍 查看技术细节（已脱敏）", expanded=False):
                    st.code(tech_err, language="text")

    state = st.session_state.get("copilot_state")
    if not state:
        st.info("👆 请点击上方 **「🚀 开始分析」** 或 **「⚡ 极速安全筛查」** 按钮（可先在左侧边栏选择任意演示病例一键填充）。")
        with st.expander("📚 知识库速查（只读：红旗规则 / 用药冲突洗脱窗 / 临床量表）", expanded=False):
            _render_tab_kb_reference()
        return

    st.divider()
    _render_top_safety_banner(state)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Tab 1：结构化病历与时间线",
            "Tab 2：确定性量表计算",
            "Tab 3：文书草稿",
            "Tab 4：执行轨迹 (Trace)",
            "Tab 5：📚 知识库速查",
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
    with tab5:
        _render_tab_kb_reference()


if __name__ == "__main__":
    main()
