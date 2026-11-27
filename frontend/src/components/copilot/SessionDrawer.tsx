import React, { useState } from "react";
import {
  BarChart3,
  Check,
  ChevronDown,
  ChevronRight,
  Clock,
  Inbox,
  PanelRightClose,
  PanelRightOpen,
  ShieldAlert,
  X,
} from "lucide-react";
import type {
  CardDiffPayload,
  ExtractedNote,
  ScaleResult,
  ScaleSpec,
  UnifiedAlert,
} from "../../api/types";
import { Modal } from "../Modal";
import { ScaleEditor } from "../ScaleEditor";
import { SEVERITY_LABELS } from "../../utils/labels";

export type DrawerTab = "alerts" | "timeline" | "scales" | "diffs";

interface SessionDrawerProps {
  open: boolean;
  activeTab: DrawerTab;
  hasCriticalAlert: boolean;
  alerts: UnifiedAlert[];
  extractedNote: ExtractedNote;
  scaleResults: ScaleResult[];
  scalesCatalog: ScaleSpec[];
  pendingDiffs: CardDiffPayload[];
  busyAction?: boolean;
  onToggleOpen: (open: boolean) => void;
  onChangeTab: (tab: DrawerTab) => void;
  onResolveRedFlag: (ruleId: string, reason: string) => Promise<void>;
  onRecalculateScale: (
    scaleId: string,
    params: Record<string, number | null>
  ) => Promise<void>;
  onCommitDiff: (diff: CardDiffPayload) => Promise<void>;
  onDismissDiff: (diffId: string) => Promise<void>;
}

export const SessionDrawer: React.FC<SessionDrawerProps> = ({
  open,
  activeTab,
  hasCriticalAlert,
  alerts,
  extractedNote,
  scaleResults,
  scalesCatalog,
  pendingDiffs,
  busyAction = false,
  onToggleOpen,
  onChangeTab,
  onResolveRedFlag,
  onRecalculateScale,
  onCommitDiff,
  onDismissDiff,
}) => {
  const [expandedAlertIdx, setExpandedAlertIdx] = useState<Record<number, boolean>>({});
  const [resolveRuleId, setResolveRuleId] = useState<string | null>(null);
  const [resolveReason, setResolveReason] = useState("");
  const [editingScaleId, setEditingScaleId] = useState<string | null>(null);
  const [addingScaleId, setAddingScaleId] = useState<string>("");

  const timelineEvents = extractedNote?.timeline_events || [];
  const scaleSpecMap = new Map(scalesCatalog.map((s) => [s.scale_id, s]));

  if (!open) {
    return (
      <div className="flex shrink-0 flex-col items-center">
        <button
          type="button"
          onClick={() => onToggleOpen(true)}
          data-testid="drawer-expand-btn"
          title="展开本次诊疗抽屉"
          className={`glass-surface relative flex flex-col items-center gap-2 rounded-3xl px-2.5 py-4 text-xs font-semibold transition hover:bg-white/90 ${
            hasCriticalAlert ? "border-rose-300 text-rose-800" : "text-slate-700"
          }`}
        >
          <PanelRightOpen className="h-4 w-4" />
          <span className="[writing-mode:vertical-rl]">本次诊疗</span>
          {alerts.length > 0 && (
            <span className="rounded-full bg-rose-700 px-1.5 py-0.5 text-[10px] font-bold text-white">
              {alerts.length}
            </span>
          )}
          {pendingDiffs.length > 0 && (
            <span className="h-2 w-2 rounded-full bg-amber-500" title="有待提交变更" />
          )}
        </button>
      </div>
    );
  }

  return (
    <aside
      data-testid="session-drawer"
      className={`glass-surface flex min-h-0 w-full shrink-0 flex-col rounded-3xl md:h-full md:w-[320px] xl:w-[368px] ${
        hasCriticalAlert ? "border-rose-300/90" : ""
      }`}
    >
      {/* Header: Collapse ONLY (D10: critical drawer may collapse, NEVER close / suppress) */}
      <div className="flex items-center justify-between border-b border-slate-200/70 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-slate-900">本次诊疗上下文</span>
          {hasCriticalAlert && (
            <span className="rounded-full bg-rose-700 px-2 py-0.5 text-[10px] font-bold text-white">
              阻断告警
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={() => onToggleOpen(false)}
          data-testid="drawer-collapse-btn"
          title="收起抽屉"
          className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white/80 px-2 py-1 text-[11px] font-medium text-slate-600 hover:bg-white"
        >
          <PanelRightClose className="h-3.5 w-3.5" />
          <span>收起</span>
        </button>
      </div>

      {/* 4 Tabs: Alerts / Timeline / Scales / Pending Diffs */}
      <div className="grid grid-cols-4 gap-1 border-b border-slate-200/60 bg-slate-900/5 p-1.5 text-[11px]">
        <button
          type="button"
          onClick={() => onChangeTab("alerts")}
          className={`flex items-center justify-center gap-1 rounded-xl py-1.5 font-semibold transition ${
            activeTab === "alerts"
              ? "bg-white text-rose-800 shadow-sm"
              : "text-slate-600 hover:text-slate-900"
          }`}
        >
          <ShieldAlert className="h-3 w-3" />
          <span>告警({alerts.length})</span>
        </button>

        <button
          type="button"
          onClick={() => onChangeTab("timeline")}
          className={`flex items-center justify-center gap-1 rounded-xl py-1.5 font-semibold transition ${
            activeTab === "timeline"
              ? "bg-white text-cyan-800 shadow-sm"
              : "text-slate-600 hover:text-slate-900"
          }`}
        >
          <Clock className="h-3 w-3" />
          <span>时间线</span>
        </button>

        <button
          type="button"
          onClick={() => onChangeTab("scales")}
          className={`flex items-center justify-center gap-1 rounded-xl py-1.5 font-semibold transition ${
            activeTab === "scales"
              ? "bg-white text-teal-800 shadow-sm"
              : "text-slate-600 hover:text-slate-900"
          }`}
        >
          <BarChart3 className="h-3 w-3" />
          <span>量表({scaleResults.length})</span>
        </button>

        <button
          type="button"
          onClick={() => onChangeTab("diffs")}
          className={`relative flex items-center justify-center gap-1 rounded-xl py-1.5 font-semibold transition ${
            activeTab === "diffs"
              ? "bg-white text-amber-800 shadow-sm"
              : "text-slate-600 hover:text-slate-900"
          }`}
        >
          <Inbox className="h-3 w-3" />
          <span>待提交</span>
          {pendingDiffs.length > 0 && (
            <span
              data-testid="pending-diff-dot"
              className="h-2 w-2 rounded-full bg-rose-500"
            />
          )}
        </button>
      </div>

      {/* Tab Body */}
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3.5 text-xs">
        {activeTab === "alerts" && (
          <>
            {alerts.length === 0 ? (
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50/70 p-4 text-center text-emerald-900">
                当前诊疗回合红旗征与用药冲突均阴性。
              </div>
            ) : (
              alerts.map((al, idx) => {
                const sev = SEVERITY_LABELS[al.severity] || SEVERITY_LABELS.high;
                const exp = Boolean(expandedAlertIdx[idx]);
                return (
                  <div
                    key={`${al.rule_id}-${idx}`}
                    className={`rounded-2xl border p-3 ${
                      al.severity === "critical"
                        ? "border-rose-300 bg-rose-50/85"
                        : "border-amber-200 bg-amber-50/80"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <span
                          className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                            al.severity === "critical"
                              ? "bg-rose-700 text-white"
                              : "bg-amber-200 text-amber-900"
                          }`}
                        >
                          {sev.label}
                        </span>
                        <div className="mt-1 font-bold text-slate-900">
                          {al.rule_id} · {al.name}
                        </div>
                      </div>
                      {Boolean(al.inherited_from_card) && (
                        <button
                          type="button"
                          onClick={() => {
                            setResolveRuleId(al.rule_id);
                            setResolveReason("");
                          }}
                          className="shrink-0 rounded-lg border border-rose-300 bg-white px-2 py-0.5 text-[10px] font-semibold text-rose-800 hover:bg-rose-100"
                        >
                          解除红旗
                        </button>
                      )}
                    </div>

                    {al.clinical_alert && (
                      <p className="mt-1.5 leading-relaxed text-slate-800">{al.clinical_alert}</p>
                    )}

                    <button
                      type="button"
                      onClick={() =>
                        setExpandedAlertIdx((prev) => ({ ...prev, [idx]: !prev[idx] }))
                      }
                      className="mt-2 inline-flex items-center gap-1 text-[11px] font-semibold text-slate-700"
                    >
                      {exp ? (
                        <ChevronDown className="h-3 w-3" />
                      ) : (
                        <ChevronRight className="h-3 w-3" />
                      )}
                      展开 L2 依据
                    </button>

                    {exp && (
                      <div className="mt-1.5 space-y-1 rounded-xl bg-white/80 p-2 text-[11px] text-slate-700">
                        {al.matched_evidence && al.matched_evidence.length > 0 && (
                          <div>命中依据：{al.matched_evidence.join("；")}</div>
                        )}
                        {al.contraindicated_actions && al.contraindicated_actions.length > 0 && (
                          <div className="text-rose-800">
                            绝对禁忌：{al.contraindicated_actions.map((x) => `严禁${x}`).join("；")}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </>
        )}

        {activeTab === "timeline" && (
          <>
            {extractedNote?.current_state_snapshot?.diagnosis_override_note && (
              <div className="rounded-2xl border border-cyan-200 bg-cyan-50/80 p-3 text-cyan-950">
                <div className="font-bold">状态时间线消解提示</div>
                <p className="mt-1 leading-relaxed">
                  {extractedNote.current_state_snapshot.diagnosis_override_note}
                </p>
              </div>
            )}

            {timelineEvents.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 p-4 text-center text-slate-400">
                暂无抽取的跨诊次时间线节点
              </div>
            ) : (
              <div className="space-y-2">
                {timelineEvents.map((ev, idx) => (
                  <div
                    key={idx}
                    className="rounded-2xl border border-slate-200/80 bg-white/90 p-3"
                  >
                    <div className="flex items-center justify-between gap-2 text-[11px]">
                      <span className="font-bold text-cyan-800">{ev.date}</span>
                      <span className="rounded bg-slate-100 px-2 py-0.5 text-slate-600">
                        {ev.department}
                      </span>
                    </div>
                    <div className="mt-1 leading-relaxed text-slate-800">{ev.event}</div>
                    {ev.key_change && (
                      <div className="mt-1 text-[11px] font-semibold text-amber-800">
                        状态变化：{ev.key_change}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {activeTab === "scales" && (
          <div className="space-y-3">
            {scalesCatalog.length > 0 && (
              <div className="flex gap-1.5">
                <select
                  value={addingScaleId}
                  onChange={(e) => setAddingScaleId(e.target.value)}
                  className="clinical-input flex-1 py-1.5 text-xs"
                >
                  <option value="">选择骨伤量表手动补录...</option>
                  {scalesCatalog.map((sc) => (
                    <option key={sc.scale_id} value={sc.scale_id}>
                      {sc.name} ({sc.scale_id})
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  disabled={!addingScaleId}
                  onClick={() => {
                    if (addingScaleId) setEditingScaleId(addingScaleId);
                  }}
                  className="rounded-xl bg-cyan-700 px-3 py-1.5 font-semibold text-white disabled:opacity-40"
                >
                  补录
                </button>
              </div>
            )}

            {scaleResults.length === 0 && !editingScaleId && (
              <div className="rounded-2xl border border-dashed border-slate-200 p-4 text-center text-slate-400">
                本会话暂未检测到量表评分，可从上方选择量表手动录入（严禁缺项补 0）。
              </div>
            )}

            {scaleResults.map((sc, idx) => {
              const spec = scaleSpecMap.get(sc.scale_id);
              const isOk = sc.status === "ok";
              const isEditing = editingScaleId === sc.scale_id;

              return (
                <div
                  key={`${sc.scale_id}-${idx}`}
                  className={`rounded-2xl border p-3 ${
                    isOk
                      ? "border-slate-200 bg-white/95"
                      : "border-amber-300 bg-amber-50/80"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="font-mono text-[11px] font-bold text-cyan-800">
                        {sc.scale_id}
                      </div>
                      <div className="font-bold text-slate-900">
                        {sc.scale_name || sc.scale_id}
                      </div>
                    </div>
                    {spec && (
                      <button
                        type="button"
                        onClick={() =>
                          setEditingScaleId(isEditing ? null : sc.scale_id)
                        }
                        className="rounded-lg border border-slate-200 bg-white px-2 py-0.5 text-[11px] font-semibold text-slate-700 hover:bg-slate-50"
                      >
                        {isEditing ? "收起" : isOk ? "重算" : "补录缺项"}
                      </button>
                    )}
                  </div>

                  {isOk ? (
                    <div className="mt-2 space-y-1 text-slate-700">
                      <div>
                        总分：<strong className="text-sm text-slate-900">{sc.total_score}</strong> 分
                        （{sc.severity_band}）
                      </div>
                      {sc.red_flag_triggered && (
                        <div className="rounded-lg bg-rose-100 px-2 py-1 text-[11px] font-semibold text-rose-900">
                          {sc.red_flag_reason || "触发量表红旗联动"}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="mt-1.5 text-[11px] text-amber-900">
                      {sc.message || "存在缺失必填项，已阻止补 0 算分。"}
                    </div>
                  )}

                  {isEditing && spec && (
                    <div className="mt-3 border-t border-slate-200 pt-3">
                      <ScaleEditor
                        spec={spec}
                        initialParams={sc.input_params || {}}
                        busy={busyAction}
                        submitPrimaryLabel="确定性核算并保存"
                        onSubmitPrimary={async (params) => {
                          await onRecalculateScale(sc.scale_id, params);
                          setEditingScaleId(null);
                        }}
                      />
                    </div>
                  )}
                </div>
              );
            })}

            {/* Manual new scale not yet in scaleResults */}
            {editingScaleId &&
              !scaleResults.some((s) => s.scale_id === editingScaleId) &&
              scaleSpecMap.get(editingScaleId) && (
                <div className="rounded-2xl border border-cyan-200 bg-white/95 p-3">
                  <ScaleEditor
                    spec={scaleSpecMap.get(editingScaleId)!}
                    initialParams={{}}
                    busy={busyAction}
                    submitPrimaryLabel="确定性核算并保存"
                    onSubmitPrimary={async (params) => {
                      await onRecalculateScale(editingScaleId, params);
                      setEditingScaleId(null);
                      setAddingScaleId("");
                    }}
                  />
                </div>
              )}
          </div>
        )}

        {activeTab === "diffs" && (
          <div className="space-y-2.5">
            {pendingDiffs.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 p-4 text-center text-slate-400">
                暂无待提交到 PATIENT.md 的档案增量变更
              </div>
            ) : (
              pendingDiffs.map((diff, idx) => (
                <div
                  key={diff.diff_id || idx}
                  className="space-y-2 rounded-2xl border border-amber-200 bg-amber-50/75 p-3"
                >
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-bold text-amber-950">档案变更提案</span>
                    <span className="font-mono text-slate-500">{diff.turn_id || ""}</span>
                  </div>

                  {diff.auto_red_flags && diff.auto_red_flags.length > 0 && (
                    <div className="text-[11px] text-rose-800">
                      红旗（已自动生效）：
                      {diff.auto_red_flags.map((r) => `${r.id} ${r.name}`).join("、")}
                    </div>
                  )}
                  {diff.add_medications && diff.add_medications.length > 0 && (
                    <div>
                      <div className="font-semibold text-slate-800">拟写入用药史：</div>
                      <ul className="ml-3 list-disc text-[11px] text-slate-700">
                        {diff.add_medications.map((m, i) => (
                          <li key={i}>
                            {m.name}（{m.status}）
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {diff.add_surgical_history && diff.add_surgical_history.length > 0 && (
                    <div className="text-[11px] text-slate-700">
                      拟写入手术史：{diff.add_surgical_history.join("、")}
                    </div>
                  )}
                  {diff.add_scale_history && diff.add_scale_history.length > 0 && (
                    <div className="text-[11px] text-slate-700">
                      拟写入量表历史：
                      {diff.add_scale_history
                        .map((s) => `${s.scale_id}=${s.total_score}`)
                        .join("；")}
                    </div>
                  )}

                  <div className="flex items-center justify-end gap-1.5 pt-1">
                    {diff.diff_id && (
                      <button
                        type="button"
                        disabled={busyAction}
                        onClick={() => onDismissDiff(diff.diff_id!)}
                        className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-600 hover:bg-slate-50"
                      >
                        <X className="h-3 w-3" />
                        忽略
                      </button>
                    )}
                    <button
                      type="button"
                      disabled={busyAction}
                      onClick={() => onCommitDiff(diff)}
                      data-testid="commit-diff-btn"
                      className="inline-flex items-center gap-1 rounded-xl bg-cyan-700 px-3 py-1 text-[11px] font-semibold text-white hover:bg-cyan-800 disabled:opacity-50"
                    >
                      <Check className="h-3 w-3" />
                      确认提交至档案卡
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Resolve active inherited red flag modal */}
      <Modal
        open={Boolean(resolveRuleId)}
        onClose={() => !busyAction && setResolveRuleId(null)}
        title={`手动解除红旗 ${resolveRuleId || ""}`}
        maxWidthClass="max-w-md"
      >
        <div className="space-y-3">
          <p className="text-xs leading-relaxed text-slate-600">
            请填写临床解除理由（例如专科查体或 MRI 明确排除），提交后将同步更新 <code>PATIENT.md</code> 与本会话审计轨迹。
          </p>
          <textarea
            rows={3}
            value={resolveReason}
            onChange={(e) => setResolveReason(e.target.value)}
            placeholder="必填：解除理由..."
            className="clinical-input text-xs"
          />
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setResolveRuleId(null)}
              className="rounded-xl border border-slate-200 px-3 py-1.5 text-xs text-slate-600"
            >
              取消
            </button>
            <button
              type="button"
              disabled={!resolveReason.trim() || busyAction}
              onClick={async () => {
                if (!resolveRuleId) return;
                await onResolveRedFlag(resolveRuleId, resolveReason.trim());
                setResolveRuleId(null);
              }}
              className="rounded-xl bg-rose-700 px-4 py-1.5 text-xs font-semibold text-white hover:bg-rose-800 disabled:opacity-50"
            >
              确认解除
            </button>
          </div>
        </div>
      </Modal>
    </aside>
  );
};
