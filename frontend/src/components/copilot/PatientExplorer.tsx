import React, { useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  FolderOpen,
  MessageSquarePlus,
  Plus,
  Search,
  ShieldAlert,
  Trash2,
  UserPlus,
} from "lucide-react";
import type { PatientSummary, SessionSummary } from "../../api/types";
import { Modal } from "../Modal";

interface PatientExplorerProps {
  patients: PatientSummary[];
  selectedPatientId: string | null;
  activeView: "card" | "session";
  sessionsByPatient: Record<string, SessionSummary[]>;
  selectedSessionId: string | null;
  loading?: boolean;
  onSelectPatientCard: (patientId: string) => void;
  onSelectSession: (patientId: string, sessionId: string) => void;
  onCreatePatient: (payload: { display_name: string; body: string }) => Promise<void>;
  onCreateSession: (patientId: string, title?: string) => Promise<void>;
  onDeletePatient?: (patientId: string) => Promise<void>;
  onExpandPatient?: (patientId: string) => void;
}

export const PatientExplorer: React.FC<PatientExplorerProps> = ({
  patients,
  selectedPatientId,
  activeView,
  sessionsByPatient,
  selectedSessionId,
  loading = false,
  onSelectPatientCard,
  onSelectSession,
  onCreatePatient,
  onCreateSession,
  onDeletePatient,
  onExpandPatient,
}) => {
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [createOpen, setCreateOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newBody, setNewBody] = useState("");
  const [creating, setCreating] = useState(false);
  const [createErr, setCreateErr] = useState<string | null>(null);

  const filteredPatients = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return patients;
    return patients.filter(
      (p) =>
        p.display_name.toLowerCase().includes(needle) ||
        p.patient_id.toLowerCase().includes(needle) ||
        p.active_red_flag_ids.some((id) => id.toLowerCase().includes(needle))
    );
  }, [patients, query]);

  const toggleExpand = (patientId: string) => {
    const next = !isPatientExpanded(patientId);
    setExpanded((prev) => ({ ...prev, [patientId]: next }));
    if (next) {
      onExpandPatient?.(patientId);
    }
  };

  const isPatientExpanded = (patientId: string): boolean => {
    if (patientId in expanded) return expanded[patientId];
    return patientId === selectedPatientId;
  };

  const handleCreateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleanedName = newName.trim();
    if (!cleanedName) {
      setCreateErr("请输入患者展示标识（如：302床 张某）。");
      return;
    }
    setCreating(true);
    setCreateErr(null);
    try {
      await onCreatePatient({ display_name: cleanedName, body: newBody.trim() });
      setExpanded((prev) => ({ ...prev }));
      setNewName("");
      setNewBody("");
      setCreateOpen(false);
    } catch (err) {
      setCreateErr(err instanceof Error ? err.message : "创建患者失败");
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="mb-2 flex items-center justify-between gap-2 px-1">
        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
          患者工作区 ({patients.length})
        </span>
        <button
          type="button"
          onClick={() => setCreateOpen(true)}
          data-testid="new-patient-btn"
          className="inline-flex items-center gap-1 rounded-xl bg-cyan-700 px-2.5 py-1 text-[11px] font-semibold text-white shadow-sm transition hover:bg-cyan-800"
        >
          <Plus className="h-3 w-3" />
          新建患者
        </button>
      </div>

      <div className="relative mb-2.5">
        <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-slate-400" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="检索患者标识 / 红旗ID..."
          className="w-full rounded-xl border border-slate-200/90 bg-white/85 py-1.5 pl-8 pr-2.5 text-xs text-slate-800 placeholder-slate-400 focus:border-cyan-600/60 focus:outline-none focus:ring-2 focus:ring-cyan-500/20"
        />
      </div>

      <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto pr-0.5">
        {loading && patients.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate-200/80 bg-white/45 p-4 text-center text-xs text-slate-500">
            正在加载患者档案...
          </div>
        ) : filteredPatients.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate-200/80 bg-white/45 p-4 text-center text-xs text-slate-500">
            {patients.length === 0 ? "暂无患者档案，点击右上角「新建患者」开始接诊。" : "无匹配患者"}
          </div>
        ) : (
          filteredPatients.map((pt) => {
            const isExpanded = isPatientExpanded(pt.patient_id);
            const isCardActive =
              selectedPatientId === pt.patient_id && activeView === "card";
            const sessions = sessionsByPatient[pt.patient_id] || [];

            return (
              <div
                key={pt.patient_id}
                className={`rounded-2xl border transition-all ${
                  selectedPatientId === pt.patient_id
                    ? "border-cyan-200/90 bg-white/85 shadow-sm"
                    : "border-white/70 bg-white/55 hover:bg-white/75"
                }`}
              >
                {/* Patient Folder Header: click name opens PATIENT.md card */}
                <div className="flex items-center gap-1 px-2 py-1.5">
                  <button
                    type="button"
                    onClick={() => toggleExpand(pt.patient_id)}
                    aria-label={isExpanded ? "收起会话列表" : "展开会话列表"}
                    className="rounded-lg p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                  >
                    {isExpanded ? (
                      <ChevronDown className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5" />
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      setExpanded((prev) => ({ ...prev, [pt.patient_id]: true }));
                      onSelectPatientCard(pt.patient_id);
                    }}
                    className={`flex min-w-0 flex-1 items-center justify-between gap-1.5 rounded-xl px-2 py-1 text-left transition ${
                      isCardActive
                        ? "bg-cyan-50 text-cyan-950 ring-1 ring-cyan-300/70"
                        : "text-slate-800 hover:bg-slate-100/70"
                    }`}
                    title={`点击查看/编辑 ${pt.display_name || pt.patient_id} 档案卡 (PATIENT.md)`}
                  >
                    <span className="flex min-w-0 items-center gap-1.5">
                      <FolderOpen className="h-3.5 w-3.5 shrink-0 text-cyan-700" />
                      <span className="truncate text-xs font-semibold">
                        {pt.display_name || pt.patient_id}
                      </span>
                    </span>

                    <span className="flex shrink-0 items-center gap-1">
                      {pt.active_red_flag_count > 0 && (
                        <span
                          title={`有效红旗：${pt.active_red_flag_ids.join(", ")}`}
                          className="inline-flex items-center gap-0.5 rounded-full bg-rose-100 px-1.5 py-0.5 text-[10px] font-bold text-rose-800"
                        >
                          <ShieldAlert className="h-2.5 w-2.5" />
                          ⚠{pt.active_red_flag_count}
                        </span>
                      )}
                      <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
                        {pt.session_count}
                      </span>
                    </span>
                  </button>
                </div>

                {/* Sessions list inside expanded patient folder */}
                {isExpanded && (
                  <div className="border-t border-slate-100/90 px-2.5 pb-2 pt-1.5">
                    <div className="mb-1 flex items-center justify-between">
                      <button
                        type="button"
                        onClick={() => onCreateSession(pt.patient_id)}
                        className="inline-flex w-full items-center justify-center gap-1 rounded-xl border border-dashed border-cyan-300/80 bg-cyan-50/50 px-2 py-1 text-[11px] font-semibold text-cyan-800 transition hover:bg-cyan-50"
                      >
                        <MessageSquarePlus className="h-3 w-3" />
                        新建诊疗会话
                      </button>
                      {onDeletePatient && (
                        <button
                          type="button"
                          onClick={() => onDeletePatient(pt.patient_id)}
                          title="删除该患者文件夹"
                          className="ml-1 rounded-lg p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-700"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      )}
                    </div>

                    {sessions.length === 0 ? (
                      <div className="px-2 py-1.5 text-[11px] text-slate-400">
                        暂无诊疗会话
                      </div>
                    ) : (
                      <div className="space-y-1">
                        {sessions.map((sess) => {
                          const isSessActive =
                            selectedPatientId === pt.patient_id &&
                            activeView === "session" &&
                            selectedSessionId === sess.session_id;
                          return (
                            <button
                              key={sess.session_id}
                              type="button"
                              onClick={() => onSelectSession(pt.patient_id, sess.session_id)}
                              className={`flex w-full items-center justify-between gap-2 rounded-xl px-2.5 py-1.5 text-left text-[11px] transition ${
                                isSessActive
                                  ? "bg-gradient-to-r from-cyan-700 to-cyan-600 font-semibold text-white shadow-sm"
                                  : "text-slate-700 hover:bg-slate-100/80"
                              }`}
                            >
                              <span className="truncate">{sess.title || sess.session_id}</span>
                              <span className="flex shrink-0 items-center gap-1">
                                {sess.has_critical && (
                                  <span
                                    className={`h-1.5 w-1.5 rounded-full ${
                                      isSessActive ? "bg-rose-200" : "bg-rose-500"
                                    }`}
                                  />
                                )}
                                {sess.pending_diff_count > 0 && (
                                  <span
                                    title="有待提交的档案变更"
                                    className={`rounded-full px-1 text-[9px] font-bold ${
                                      isSessActive
                                        ? "bg-white/25 text-white"
                                        : "bg-amber-100 text-amber-800"
                                    }`}
                                  >
                                    +{sess.pending_diff_count}
                                  </span>
                                )}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      <Modal
        open={createOpen}
        onClose={() => !creating && setCreateOpen(false)}
        title="新建患者档案卡（PATIENT.md）"
        maxWidthClass="max-w-md"
      >
        <form onSubmit={handleCreateSubmit} className="space-y-3">
          <div className="rounded-xl border border-cyan-200/70 bg-cyan-50/70 p-2.5 text-[11px] text-cyan-950">
            展示标识（如「302床 张某」）仅用于左侧栏识别；向模型拼装上下文时，系统会通过确定性代码强制剔除该字段。
          </div>

          <div>
            <label className="mb-1 block text-xs font-semibold text-slate-700">
              患者展示标识 <span className="text-rose-600">*</span>
            </label>
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="例如：302床 张某"
              className="clinical-input text-xs"
              autoFocus
            />
          </div>

          <div>
            <label className="mb-1 block text-xs font-semibold text-slate-700">
              基础病历摘要 / 长期背景（可选）
            </label>
            <textarea
              rows={3}
              value={newBody}
              onChange={(e) => setNewBody(e.target.value)}
              placeholder="可填写基础疾病、既往内固定手术史或长期用药史..."
              className="clinical-input text-xs"
            />
          </div>

          {createErr && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-800">
              {createErr}
            </div>
          )}

          <div className="flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={() => setCreateOpen(false)}
              className="rounded-xl border border-slate-200 px-3.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
            >
              取消
            </button>
            <button
              type="submit"
              disabled={creating}
              className="inline-flex items-center gap-1.5 rounded-xl bg-cyan-700 px-4 py-1.5 text-xs font-semibold text-white hover:bg-cyan-800 disabled:opacity-50"
            >
              <UserPlus className="h-3.5 w-3.5" />
              {creating ? "创建中..." : "创建并接诊"}
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
};
