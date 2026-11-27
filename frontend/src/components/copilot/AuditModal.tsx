import React, { useEffect, useState } from "react";
import { Download, FileJson2, Filter } from "lucide-react";
import { api } from "../../api/client";
import type { TraceEvent } from "../../api/types";
import { Modal } from "../Modal";

interface AuditModalProps {
  open: boolean;
  onClose: () => void;
  patientId: string | null;
  sessionId: string | null;
  focusTurnId?: string | null;
}

const EVENT_TYPE_LABELS: Record<string, { label: string; color: string }> = {
  session_meta: { label: "会话创建", color: "bg-slate-100 text-slate-700" },
  user_input: { label: "临床输入", color: "bg-cyan-100 text-cyan-900" },
  pii_check: { label: "PII脱敏检测", color: "bg-indigo-100 text-indigo-800" },
  extraction: { label: "结构化抽取", color: "bg-sky-100 text-sky-900" },
  scan: { label: "双轨扫描", color: "bg-rose-100 text-rose-800" },
  scale: { label: "量表计算", color: "bg-teal-100 text-teal-900" },
  web_fetch: { label: "联网检索", color: "bg-violet-100 text-violet-900" },
  generation: { label: "模型生成", color: "bg-emerald-100 text-emerald-900" },
  verification: { label: "输出复检", color: "bg-amber-100 text-amber-900" },
  card_diff: { label: "档案变更提案", color: "bg-orange-100 text-orange-900" },
  human_action: { label: "医师操作", color: "bg-purple-100 text-purple-900" },
  education: { label: "患者宣教", color: "bg-lime-100 text-lime-900" },
};

export const AuditModal: React.FC<AuditModalProps> = ({
  open,
  onClose,
  patientId,
  sessionId,
  focusTurnId,
}) => {
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [onlyFocusTurn, setOnlyFocusTurn] = useState(false);

  useEffect(() => {
    setOnlyFocusTurn(Boolean(focusTurnId));
  }, [focusTurnId, open]);

  useEffect(() => {
    if (!open || !patientId || !sessionId) return;
    let cancelled = false;
    setLoading(true);
    setErr(null);
    api
      .getSessionTrace(patientId, sessionId)
      .then((res) => {
        if (!cancelled) setEvents(res.events || []);
      })
      .catch((e) => {
        if (!cancelled) setErr(e instanceof Error ? e.message : "加载审计轨迹失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, patientId, sessionId]);

  const visibleEvents =
    onlyFocusTurn && focusTurnId
      ? events.filter((ev) => ev.turn_id === focusTurnId)
      : events;

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={`诊疗审计轨迹（${sessionId || "-"}）`}
      maxWidthClass="max-w-3xl"
    >
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-slate-200/80 bg-slate-50/70 px-3.5 py-2 text-xs">
          <div className="flex items-center gap-2 text-slate-600">
            <FileJson2 className="h-4 w-4 text-cyan-700" />
            <span>
              数据源：<code>sessions/{sessionId || ""}.trace.jsonl</code>（共 {events.length} 条）
            </span>
          </div>

          <div className="flex items-center gap-2">
            {focusTurnId && (
              <button
                type="button"
                onClick={() => setOnlyFocusTurn((v) => !v)}
                className={`inline-flex items-center gap-1 rounded-xl border px-2.5 py-1 text-[11px] font-semibold transition ${
                  onlyFocusTurn
                    ? "border-cyan-600 bg-cyan-700 text-white"
                    : "border-slate-200 bg-white text-slate-700"
                }`}
              >
                <Filter className="h-3 w-3" />
                {onlyFocusTurn ? `已筛选回合 ${focusTurnId}` : "查看全部回合"}
              </button>
            )}

            {patientId && sessionId && (
              <button
                type="button"
                onClick={() => api.downloadSessionTrace(patientId, sessionId)}
                className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-semibold text-slate-700 hover:bg-slate-100"
              >
                <Download className="h-3 w-3" />
                导出 .trace.jsonl
              </button>
            )}
          </div>
        </div>

        {loading ? (
          <div className="py-10 text-center text-xs text-slate-400">正在读取审计事件流...</div>
        ) : err ? (
          <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800">
            {err}
          </div>
        ) : visibleEvents.length === 0 ? (
          <div className="py-10 text-center text-xs text-slate-400">暂无匹配的审计事件</div>
        ) : (
          <div className="max-h-[58vh] space-y-2 overflow-y-auto pr-1">
            {visibleEvents.map((ev) => {
              const meta = EVENT_TYPE_LABELS[ev.type] || {
                label: ev.type,
                color: "bg-slate-100 text-slate-700",
              };
              const isHighlighted = Boolean(focusTurnId && ev.turn_id === focusTurnId);
              return (
                <div
                  key={ev.seq}
                  className={`rounded-2xl border p-3 text-xs ${
                    isHighlighted
                      ? "border-cyan-300 bg-cyan-50/45"
                      : "border-slate-200/80 bg-white"
                  }`}
                >
                  <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[11px] text-slate-400">#{ev.seq}</span>
                      <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${meta.color}`}>
                        {meta.label}
                      </span>
                      {ev.turn_id && (
                        <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-500">
                          {ev.turn_id}
                        </span>
                      )}
                    </div>
                    <span className="font-mono text-[11px] text-slate-400">{ev.ts}</span>
                  </div>
                  <pre className="max-h-40 overflow-auto rounded-xl bg-slate-900/95 p-2.5 font-mono text-[11px] leading-relaxed text-slate-100">
                    {JSON.stringify(ev.payload, null, 2)}
                  </pre>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Modal>
  );
};
