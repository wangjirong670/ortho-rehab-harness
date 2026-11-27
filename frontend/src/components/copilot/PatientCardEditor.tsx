import React, { useEffect, useState } from "react";
import {
  CheckCircle2,
  EyeOff,
  History,
  MessageSquarePlus,
  Plus,
  RotateCcw,
  Save,
  ShieldAlert,
  Trash2,
  X,
} from "lucide-react";
import { api } from "../../api/client";
import type {
  PatientCard,
  PatientDetailResponse,
  PatientHistoryItem,
  PatientMedicationItem,
} from "../../api/types";
import { Modal } from "../Modal";

interface PatientCardEditorProps {
  patient: PatientDetailResponse;
  onUpdated: (next: PatientDetailResponse) => void;
  onStartNewSession: (patientId: string) => void;
}

const SOURCE_BADGE: Record<string, string> = {
  clinician: "bg-cyan-100 text-cyan-800",
  extracted: "bg-slate-100 text-slate-700",
  system: "bg-emerald-100 text-emerald-800",
};

const SOURCE_TEXT: Record<string, string> = {
  clinician: "医生手录",
  extracted: "模型抽取",
  system: "系统联动",
};

export const PatientCardEditor: React.FC<PatientCardEditorProps> = ({
  patient,
  onUpdated,
  onStartNewSession,
}) => {
  const [card, setCard] = useState<PatientCard>(patient.card);
  const [body, setBody] = useState(patient.body);
  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState<{ type: "ok" | "err"; text: string } | null>(null);

  const [newRfId, setNewRfId] = useState("");
  const [newRfName, setNewRfName] = useState("");
  const [resolveTargetId, setResolveTargetId] = useState<string | null>(null);
  const [resolveReason, setResolveReason] = useState("");
  const [resolving, setResolving] = useState(false);

  const [newMedName, setNewMedName] = useState("");
  const [newMedStatus, setNewMedStatus] = useState<"active" | "stopped" | "unknown">("active");
  const [newMedStopDate, setNewMedStopDate] = useState("");

  const [newAllergy, setNewAllergy] = useState("");
  const [newSurgery, setNewSurgery] = useState("");

  const [historyOpen, setHistoryOpen] = useState(false);
  const [revisions, setRevisions] = useState<PatientHistoryItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  useEffect(() => {
    setCard(patient.card);
    setBody(patient.body);
    setFeedback(null);
  }, [patient]);

  const handleSave = async () => {
    setSaving(true);
    setFeedback(null);
    try {
      const updated = await api.updatePatient(patient.patient_id, { card, body });
      onUpdated(updated);
      setFeedback({ type: "ok", text: `已保存 PATIENT.md（修订版本 r${updated.revision}）` });
    } catch (err) {
      setFeedback({
        type: "err",
        text: err instanceof Error ? err.message : "保存 PATIENT.md 失败",
      });
    } finally {
      setSaving(false);
    }
  };

  const handleAddRedFlag = () => {
    const rid = newRfId.trim().toUpperCase();
    if (!rid) return;
    if (card.active_red_flags.some((rf) => rf.id === rid && rf.status === "active")) {
      setFeedback({ type: "err", text: `红旗 ${rid} 已在有效列表中。` });
      return;
    }
    const today = new Date().toISOString().slice(0, 10);
    setCard((prev) => ({
      ...prev,
      active_red_flags: [
        ...prev.active_red_flags,
        {
          id: rid,
          name: newRfName.trim() || rid,
          since: today,
          status: "active",
          source: "clinician",
        },
      ],
    }));
    setNewRfId("");
    setNewRfName("");
  };

  const handleConfirmResolveRedFlag = async () => {
    if (!resolveTargetId) return;
    const reason = resolveReason.trim();
    if (!reason) return;
    setResolving(true);
    setFeedback(null);
    try {
      const updated = await api.commitPatientChanges(patient.patient_id, {
        changes: {
          resolve_red_flags: [{ id: resolveTargetId, resolved_reason: reason }],
        },
      });
      onUpdated(updated);
      setResolveTargetId(null);
      setResolveReason("");
      setFeedback({ type: "ok", text: `已解除红旗 ${resolveTargetId} 并记入审计修订。` });
    } catch (err) {
      setFeedback({
        type: "err",
        text: err instanceof Error ? err.message : "解除红旗失败",
      });
    } finally {
      setResolving(false);
    }
  };

  const handleAddMed = () => {
    const name = newMedName.trim();
    if (!name) return;
    const nextMed: PatientMedicationItem = {
      name,
      status: newMedStatus,
      stop_date: newMedStopDate.trim() || null,
      source: "clinician",
    };
    setCard((prev) => ({
      ...prev,
      medications: [...prev.medications.filter((m) => m.name !== name), nextMed],
    }));
    setNewMedName("");
    setNewMedStopDate("");
  };

  const openHistory = async () => {
    setHistoryOpen(true);
    setLoadingHistory(true);
    try {
      const res = await api.listPatientHistory(patient.patient_id);
      setRevisions(res.revisions || []);
    } catch {
      setRevisions([]);
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleRestore = async (rev: number) => {
    setSaving(true);
    try {
      const updated = await api.restorePatientRevision(patient.patient_id, rev);
      onUpdated(updated);
      setHistoryOpen(false);
      setFeedback({ type: "ok", text: `已从 r${rev} 恢复并生成新版本 r${updated.revision}` });
    } catch (err) {
      setFeedback({
        type: "err",
        text: err instanceof Error ? err.message : "回滚版本失败",
      });
    } finally {
      setSaving(false);
    }
  };

  const activeFlags = card.active_red_flags.filter((rf) => rf.status === "active");
  const resolvedFlags = card.active_red_flags.filter((rf) => rf.status === "resolved");

  return (
    <div className="flex h-full min-h-0 flex-col space-y-4 overflow-y-auto pr-1">
      {/* Header bar */}
      <div className="glass-surface flex flex-wrap items-center justify-between gap-3 rounded-3xl p-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded-lg bg-cyan-100 px-2 py-0.5 font-mono text-[11px] font-bold text-cyan-900">
              PATIENT.md
            </span>
            <span className="font-mono text-xs text-slate-500">
              {patient.patient_id} · r{card.revision}
            </span>
          </div>
          <h2 className="mt-1 text-base font-bold text-slate-900">
            {card.display_name || patient.patient_id} 长期临床档案卡
          </h2>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => onStartNewSession(patient.patient_id)}
            className="inline-flex items-center gap-1.5 rounded-xl border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs font-semibold text-cyan-900 hover:bg-cyan-100"
          >
            <MessageSquarePlus className="h-3.5 w-3.5" />
            发起新诊疗会话
          </button>
          <button
            type="button"
            onClick={openHistory}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
          >
            <History className="h-3.5 w-3.5" />
            修订历史
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={saving}
            className="inline-flex items-center gap-1.5 rounded-xl bg-cyan-700 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-cyan-800 disabled:opacity-50"
          >
            <Save className="h-3.5 w-3.5" />
            {saving ? "保存中..." : "保存档案卡"}
          </button>
        </div>
      </div>

      {/* D15 De-identification guarantee banner */}
      <div className="flex items-center gap-2 rounded-2xl border border-cyan-200/70 bg-cyan-50/75 px-4 py-2.5 text-xs text-cyan-950">
        <EyeOff className="h-4 w-4 shrink-0 text-cyan-700" />
        <span>
          <strong>隐私隔离说明：</strong>
          下方「展示标识」仅保存在本地 <code>case_archives/</code> 供界面识别；每次调用模型前，
          <code>build_prompt_context()</code> 会由确定性代码强制剔除该名称，不外发任何可识别身份信息。
        </span>
      </div>

      {feedback && (
        <div
          className={`flex items-center gap-2 rounded-2xl border px-4 py-2.5 text-xs font-medium ${
            feedback.type === "ok"
              ? "border-emerald-200 bg-emerald-50 text-emerald-900"
              : "border-rose-200 bg-rose-50 text-rose-900"
          }`}
        >
          <CheckCircle2 className="h-4 w-4 shrink-0" />
          <span>{feedback.text}</span>
        </div>
      )}

      {/* Basic identity & Red flags */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <section className="clinical-paper space-y-3 rounded-3xl p-5">
          <div className="flex items-center justify-between">
            <h3 className="flex items-center gap-1.5 text-sm font-bold text-slate-900">
              <ShieldAlert className="h-4 w-4 text-rose-600" />
              持续生效红旗（只增不静默删）
            </h3>
            <span className="rounded-full bg-rose-100 px-2.5 py-0.5 text-[11px] font-bold text-rose-800">
              有效 {activeFlags.length} 项
            </span>
          </div>

          <p className="text-[11px] text-slate-500">
            每个新诊疗会话都会自动继承以下有效红旗，并参与输出安全复检；仅医生填写解除理由后才可解除。
          </p>

          <div className="space-y-2">
            {activeFlags.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50/60 p-3 text-xs text-slate-400">
                当前无持续生效红旗
              </div>
            ) : (
              activeFlags.map((rf) => (
                <div
                  key={rf.id}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-rose-200 bg-rose-50/70 px-3.5 py-2.5"
                >
                  <div>
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-xs font-bold text-rose-900">{rf.id}</span>
                      <span
                        className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                          SOURCE_BADGE[rf.source] || SOURCE_BADGE.extracted
                        }`}
                      >
                        {SOURCE_TEXT[rf.source] || rf.source}
                      </span>
                    </div>
                    <div className="text-xs font-medium text-rose-950">{rf.name || rf.id}</div>
                    <div className="text-[11px] text-rose-700/80">触发日期：{rf.since}</div>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setResolveTargetId(rf.id);
                      setResolveReason("");
                    }}
                    className="rounded-xl border border-rose-300 bg-white px-2.5 py-1 text-[11px] font-semibold text-rose-800 hover:bg-rose-100"
                  >
                    手动解除
                  </button>
                </div>
              ))
            )}
          </div>

          {/* Add clinician red flag */}
          <div className="flex flex-wrap gap-1.5 border-t border-slate-100 pt-2">
            <input
              type="text"
              value={newRfId}
              onChange={(e) => setNewRfId(e.target.value)}
              placeholder="规则ID（如 RF_001）"
              className="clinical-input w-32 py-1.5 text-xs"
            />
            <input
              type="text"
              value={newRfName}
              onChange={(e) => setNewRfName(e.target.value)}
              placeholder="红旗说明（医生手录）"
              className="clinical-input min-w-[140px] flex-1 py-1.5 text-xs"
            />
            <button
              type="button"
              onClick={handleAddRedFlag}
              className="inline-flex items-center gap-1 rounded-xl bg-slate-800 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-900"
            >
              <Plus className="h-3.5 w-3.5" />
              手录红旗
            </button>
          </div>

          {resolvedFlags.length > 0 && (
            <div className="space-y-1 border-t border-slate-100 pt-2">
              <div className="text-[11px] font-semibold text-slate-400">
                已审计解除 ({resolvedFlags.length})
              </div>
              {resolvedFlags.map((rf, idx) => (
                <div
                  key={`${rf.id}-${idx}`}
                  className="rounded-xl bg-slate-50 px-3 py-1.5 text-[11px] text-slate-500"
                >
                  <span className="font-mono font-semibold">{rf.id}</span> · 理由：
                  {rf.resolved_reason}
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Display name, Allergies, Surgeries */}
        <section className="clinical-paper space-y-3 rounded-3xl p-5">
          <div>
            <label className="mb-1 block text-xs font-bold text-slate-800">
              展示标识（仅本地界面显示，不入提示词）
            </label>
            <input
              type="text"
              value={card.display_name}
              onChange={(e) => setCard((prev) => ({ ...prev, display_name: e.target.value }))}
              className="clinical-input text-xs"
            />
          </div>

          <div>
            <div className="mb-1 text-xs font-bold text-slate-800">既往手术与内固定史</div>
            <div className="mb-2 flex flex-wrap gap-1.5">
              {card.surgical_history.map((s) => (
                <span
                  key={s}
                  className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-xs text-slate-700"
                >
                  {s}
                  <button
                    type="button"
                    onClick={() =>
                      setCard((prev) => ({
                        ...prev,
                        surgical_history: prev.surgical_history.filter((x) => x !== s),
                      }))
                    }
                    className="text-slate-400 hover:text-rose-600"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))}
            </div>
            <div className="flex gap-1.5">
              <input
                type="text"
                value={newSurgery}
                onChange={(e) => setNewSurgery(e.target.value)}
                placeholder="新增手术史（如：THA全髋置换 / ACDF）"
                className="clinical-input flex-1 py-1.5 text-xs"
              />
              <button
                type="button"
                onClick={() => {
                  const val = newSurgery.trim();
                  if (!val) return;
                  setCard((prev) => ({
                    ...prev,
                    surgical_history: prev.surgical_history.includes(val)
                      ? prev.surgical_history
                      : [...prev.surgical_history, val],
                  }));
                  setNewSurgery("");
                }}
                className="rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50"
              >
                添加
              </button>
            </div>
          </div>

          <div>
            <div className="mb-1 text-xs font-bold text-slate-800">药物过敏史</div>
            <div className="mb-2 flex flex-wrap gap-1.5">
              {card.allergies.map((al) => (
                <span
                  key={al}
                  className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-xs text-amber-900"
                >
                  {al}
                  <button
                    type="button"
                    onClick={() =>
                      setCard((prev) => ({
                        ...prev,
                        allergies: prev.allergies.filter((x) => x !== al),
                      }))
                    }
                    className="text-amber-500 hover:text-rose-600"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))}
            </div>
            <div className="flex gap-1.5">
              <input
                type="text"
                value={newAllergy}
                onChange={(e) => setNewAllergy(e.target.value)}
                placeholder="新增过敏原（如：磺胺 / NSAIDs）"
                className="clinical-input flex-1 py-1.5 text-xs"
              />
              <button
                type="button"
                onClick={() => {
                  const val = newAllergy.trim();
                  if (!val) return;
                  setCard((prev) => ({
                    ...prev,
                    allergies: prev.allergies.includes(val)
                      ? prev.allergies
                      : [...prev.allergies, val],
                  }));
                  setNewAllergy("");
                }}
                className="rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50"
              >
                添加
              </button>
            </div>
          </div>
        </section>
      </div>

      {/* Medications & Washout tracking */}
      <section className="clinical-paper space-y-3 rounded-3xl p-5">
        <h3 className="text-sm font-bold text-slate-900">患者长期用药与停药洗脱记录</h3>
        <div className="space-y-1.5">
          {card.medications.length === 0 ? (
            <div className="text-xs text-slate-400">暂无长期用药记录</div>
          ) : (
            card.medications.map((med) => (
              <div
                key={med.name}
                className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200/80 bg-slate-50/60 px-3 py-2 text-xs"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-slate-900">{med.name}</span>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                      med.status === "active"
                        ? "bg-amber-100 text-amber-800"
                        : "bg-emerald-100 text-emerald-800"
                    }`}
                  >
                    {med.status === "active"
                      ? "在用"
                      : med.status === "stopped"
                        ? "已停药"
                        : "状态待明确"}
                  </span>
                  {med.stop_date && (
                    <span className="text-slate-500">停药日：{med.stop_date}</span>
                  )}
                  {typeof med.stopped_days_ago === "number" && (
                    <span className="text-slate-500">已停 {med.stopped_days_ago} 天</span>
                  )}
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] ${
                      SOURCE_BADGE[med.source] || SOURCE_BADGE.extracted
                    }`}
                  >
                    {SOURCE_TEXT[med.source] || med.source}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() =>
                    setCard((prev) => ({
                      ...prev,
                      medications: prev.medications.filter((m) => m.name !== med.name),
                    }))
                  }
                  className="rounded-lg p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-700"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="flex flex-wrap gap-1.5 border-t border-slate-100 pt-2">
          <input
            type="text"
            value={newMedName}
            onChange={(e) => setNewMedName(e.target.value)}
            placeholder="药名（如：华法林 / 氯吡格雷）"
            className="clinical-input min-w-[160px] flex-1 py-1.5 text-xs"
          />
          <select
            value={newMedStatus}
            onChange={(e) => setNewMedStatus(e.target.value as "active" | "stopped" | "unknown")}
            className="clinical-input w-28 py-1.5 text-xs"
          >
            <option value="active">在用</option>
            <option value="stopped">已停药</option>
            <option value="unknown">待核实</option>
          </select>
          <input
            type="text"
            value={newMedStopDate}
            onChange={(e) => setNewMedStopDate(e.target.value)}
            placeholder="停药日期 YYYY-MM-DD"
            className="clinical-input w-40 py-1.5 text-xs"
          />
          <button
            type="button"
            onClick={handleAddMed}
            className="rounded-xl bg-slate-800 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-900"
          >
            添加用药
          </button>
        </div>
      </section>

      {/* Freeform clinical body (also fed into rule_scanner) */}
      <section className="clinical-paper space-y-2 rounded-3xl p-5">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-900">患者档案自由正文（PATIENT.md Body）</h3>
          <span className="text-[11px] text-slate-400">正文同步送入确定性红旗扫描引擎</span>
        </div>
        <textarea
          rows={5}
          value={body}
          onChange={(e) => setBody(e.target.value)}
          placeholder="记录跨诊次关键临床脉络、既往重要检查与医生备忘..."
          className="clinical-input font-mono text-xs leading-relaxed"
        />
      </section>

      {/* Manual red-flag resolution modal */}
      <Modal
        open={Boolean(resolveTargetId)}
        onClose={() => !resolving && setResolveTargetId(null)}
        title={`手动解除红旗 ${resolveTargetId || ""}`}
        maxWidthClass="max-w-md"
      >
        <div className="space-y-3">
          <p className="text-xs leading-relaxed text-slate-600">
            高危红旗禁止静默删除。请填写明确的临床解除理由（例如专科 MRI 排除依据），系统将写入
            <code>PATIENT.md</code> 修订历史与审计轨迹。
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
              onClick={() => setResolveTargetId(null)}
              className="rounded-xl border border-slate-200 px-3 py-1.5 text-xs text-slate-600"
            >
              取消
            </button>
            <button
              type="button"
              disabled={!resolveReason.trim() || resolving}
              onClick={handleConfirmResolveRedFlag}
              className="rounded-xl bg-rose-700 px-4 py-1.5 text-xs font-semibold text-white hover:bg-rose-800 disabled:opacity-50"
            >
              {resolving ? "提交中..." : "确认解除并记入审计"}
            </button>
          </div>
        </div>
      </Modal>

      {/* Revision history modal */}
      <Modal
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        title="PATIENT.md 修订版本历史"
        maxWidthClass="max-w-lg"
      >
        <div className="space-y-2">
          {loadingHistory ? (
            <div className="py-6 text-center text-xs text-slate-400">正在读取修订历史...</div>
          ) : revisions.length === 0 ? (
            <div className="py-6 text-center text-xs text-slate-400">暂无修订快照</div>
          ) : (
            revisions.map((r) => (
              <div
                key={r.revision}
                className="flex items-center justify-between gap-2 rounded-2xl border border-slate-200 bg-slate-50/70 px-3.5 py-2.5 text-xs"
              >
                <div>
                  <div className="font-mono font-bold text-slate-900">
                    r{String(r.revision).padStart(3, "0")}
                    {r.revision === card.revision && (
                      <span className="ml-2 rounded-full bg-cyan-100 px-2 py-0.5 text-[10px] text-cyan-800">
                        当前
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    {r.updated_at || "-"} · 有效红旗 {r.active_red_flag_count} · 用药{" "}
                    {r.medication_count}
                  </div>
                </div>
                {r.revision !== card.revision && (
                  <button
                    type="button"
                    onClick={() => handleRestore(r.revision)}
                    className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-semibold text-slate-700 hover:bg-slate-100"
                  >
                    <RotateCcw className="h-3 w-3" />
                    回滚到此版本
                  </button>
                )}
              </div>
            ))
          )}
        </div>
      </Modal>
    </div>
  );
};
