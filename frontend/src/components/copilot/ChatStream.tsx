import React, { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  BookOpenText,
  Brain,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ClipboardCheck,
  Copy,
  FileSearch,
  Info,
  Loader2,
  ShieldAlert,
  ShieldX,
  Sparkles,
} from "lucide-react";
import type {
  ChatStepEvent,
  PatientCard,
  ReasoningVerification,
  SessionEducationItem,
  SessionMessage,
  UnifiedAlert,
} from "../../api/types";
import { MarkdownView } from "../MarkdownView";
import { getClinicalActionLabel, SEVERITY_LABELS } from "../../utils/labels";

interface ChatStreamProps {
  patientCard: PatientCard | null;
  messages: SessionMessage[];
  educations: SessionEducationItem[];
  streaming: boolean;
  liveSteps: ChatStepEvent[];
  liveAlerts: UnifiedAlert[];
  liveAlertsTurnId?: string | null;
  generatingEdu: boolean;
  defaultExpandSteps?: boolean;
  onGenerateEducation: (turnId?: string | null) => void;
  onOpenAudit: (turnId?: string | null) => void;
  onBuildHandover: () => void;
}

const AlertCardInline: React.FC<{ alert: UnifiedAlert }> = ({ alert }) => {
  const [expanded, setExpanded] = useState(false);
  const sev = SEVERITY_LABELS[alert.severity] || SEVERITY_LABELS.high;
  const isCritical = alert.severity === "critical";

  return (
    <div
      className={`rounded-2xl border p-3 text-xs transition ${
        isCritical
          ? "border-rose-300/90 bg-rose-50/90 text-rose-950"
          : alert.severity === "medium"
            ? "border-amber-200 bg-amber-50/85 text-amber-950"
            : "border-orange-200 bg-orange-50/85 text-orange-950"
      }`}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
              isCritical
                ? "bg-rose-700 text-white"
                : alert.severity === "medium"
                  ? "bg-amber-200 text-amber-900"
                  : "bg-orange-200 text-orange-900"
            }`}
          >
            {sev.label}
          </span>
          <span className="font-mono text-[11px] font-bold">{alert.rule_id}</span>
          <span className="font-bold">{alert.name}</span>
          {Boolean(alert.inherited_from_card) && (
            <span className="rounded-full bg-cyan-100 px-2 py-0.5 text-[10px] font-semibold text-cyan-900">
              档案卡继承
            </span>
          )}
        </div>

        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="inline-flex items-center gap-1 rounded-lg bg-white/80 px-2 py-0.5 text-[11px] font-medium text-slate-700 hover:bg-white"
        >
          {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          L2 规则依据
        </button>
      </div>

      {alert.clinical_alert && (
        <p className="mt-1.5 leading-relaxed">{alert.clinical_alert}</p>
      )}

      {expanded && (
        <div className="mt-2 space-y-1.5 rounded-xl border border-white/80 bg-white/80 p-2.5 text-[11px] text-slate-700">
          <div>
            <span className="font-semibold text-slate-900">触发来源：</span>
            {alert.source_label || alert.source || "双轨规则扫描"} ·{" "}
            {getClinicalActionLabel(alert.clinical_action)}
          </div>
          {alert.matched_evidence && alert.matched_evidence.length > 0 && (
            <div>
              <span className="font-semibold text-slate-900">命中词 / 依据：</span>
              {alert.matched_evidence.join("；")}
            </div>
          )}
          {alert.contraindicated_actions && alert.contraindicated_actions.length > 0 && (
            <div className="text-rose-800">
              <span className="font-semibold">绝对禁忌：</span>
              {alert.contraindicated_actions.map((c) => `严禁${c}`).join("；")}
            </div>
          )}
          {alert.reference && alert.reference.length > 0 && (
            <div className="text-slate-500">
              <span className="font-semibold">参考依据：</span>
              {alert.reference.join("；")}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const ReasoningBlockL3: React.FC<{
  reasoning?: string;
  verification?: ReasoningVerification;
}> = ({ reasoning, verification }) => {
  const [open, setOpen] = useState(false);
  const cleaned = (reasoning || "").trim();

  if (!cleaned) {
    return (
      <div className="mb-2 flex items-center gap-1.5 rounded-xl bg-slate-50 px-2.5 py-1 text-[11px] text-slate-400">
        <Brain className="h-3 w-3 shrink-0" />
        <span>该模型未提供推理过程</span>
      </div>
    );
  }

  const hasConflict = Boolean(verification?.has_conflict);

  return (
    <div
      data-testid="l3-reasoning-box"
      className={`mb-2.5 rounded-2xl border px-3 py-2 text-xs ${
        hasConflict
          ? "border-rose-200 bg-rose-50/60"
          : "border-slate-200/80 bg-slate-50/75"
      }`}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full flex-wrap items-center justify-between gap-2 text-left"
      >
        <span className="flex flex-wrap items-center gap-1.5 font-semibold text-slate-700">
          {open ? (
            <ChevronDown className="h-3.5 w-3.5 text-slate-500" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-slate-500" />
          )}
          <Brain className="h-3.5 w-3.5 text-cyan-700" />
          <span>L3 模型原始推理</span>
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-900">
            未经复检
          </span>
          {hasConflict && (
            <span className="rounded-full bg-rose-700 px-2 py-0.5 text-[10px] font-bold text-white">
              ⚠ 推理含禁忌冲突标注（未拦截正文）
            </span>
          )}
        </span>
        <span className="text-[10px] text-slate-400">{open ? "收起" : "展开详情"}</span>
      </button>

      {open && (
        <div className="mt-2 space-y-2 border-t border-slate-200/70 pt-2">
          {hasConflict && verification?.message && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-2.5 py-1.5 text-[11px] text-rose-900">
              {verification.message}
            </div>
          )}
          <pre className="max-h-60 overflow-auto whitespace-pre-wrap rounded-xl bg-white/90 p-2.5 font-mono text-[11px] leading-relaxed text-slate-700">
            {cleaned}
          </pre>
        </div>
      )}
    </div>
  );
};

export const ChatStream: React.FC<ChatStreamProps> = ({
  patientCard,
  messages,
  educations,
  streaming,
  liveSteps,
  liveAlerts,
  generatingEdu,
  defaultExpandSteps = false,
  onGenerateEducation,
  onOpenAudit,
  onBuildHandover,
}) => {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [stepsExpanded, setStepsExpanded] = useState<boolean>(defaultExpandSteps);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, streaming, liveAlerts.length, educations.length]);

  const activeInheritedFlags = (patientCard?.active_red_flags || []).filter(
    (rf) => rf.status === "active"
  );

  const handleCopy = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId((prev) => (prev === id ? null : prev)), 1800);
    } catch {
      // ignore clipboard errors
    }
  };

  return (
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-1 py-2">
      {/* Patient inherited red flags banner */}
      {activeInheritedFlags.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-rose-200/80 bg-rose-50/75 px-3.5 py-2 text-xs text-rose-950">
          <ShieldAlert className="h-4 w-4 shrink-0 text-rose-600" />
          <span className="font-semibold">已继承患者档案卡有效红旗：</span>
          {activeInheritedFlags.map((rf) => (
            <span
              key={rf.id}
              className="rounded-full bg-rose-700 px-2 py-0.5 font-mono text-[10px] font-bold text-white"
            >
              {rf.id} {rf.name ? `· ${rf.name}` : ""}
            </span>
          ))}
        </div>
      )}

      {messages.length === 0 && !streaming && (
        <div className="clinical-paper mx-auto my-8 max-w-xl rounded-3xl p-6 text-center">
          <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-2xl bg-cyan-50 text-cyan-700">
            <Sparkles className="h-5 w-5" />
          </div>
          <h3 className="text-sm font-bold text-slate-900">开始本次诊疗会话</h3>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-500">
            首条消息请输入本次诊疗病历（支持左侧 <code>+</code> 导入 .txt/.md/.docx）；后续消息可继续追问方案、补录量表或生成独立患者宣教要点。
          </p>
        </div>
      )}

      {messages.map((msg, idx) => {
        if (msg.role === "system") {
          return (
            <div
              key={`sys-${idx}`}
              className="mx-auto flex max-w-2xl items-center justify-center gap-2 rounded-2xl border border-amber-200/90 bg-amber-50/80 px-4 py-2 text-xs font-medium text-amber-900"
            >
              <Info className="h-3.5 w-3.5 shrink-0 text-amber-600" />
              <span>{msg.content}</span>
            </div>
          );
        }

        if (msg.role === "user") {
          return (
            <div key={`u-${idx}`} className="flex justify-end">
              <div className="max-w-[85%] rounded-3xl rounded-tr-md bg-gradient-to-br from-cyan-700 to-teal-700 px-4 py-3 text-xs leading-relaxed text-white shadow-md shadow-cyan-900/10">
                <div className="mb-1 flex items-center justify-between gap-3 text-[10px] text-cyan-100/85">
                  <span>医师输入</span>
                  <span>
                    {msg.output_mode === "screen" ? "简要筛查" : "完整分析"}
                    {msg.model ? ` · ${msg.model}` : ""}
                  </span>
                </div>
                <div className="whitespace-pre-wrap">{msg.content}</div>
              </div>
            </div>
          );
        }

        // Assistant clinical response (D8 & D10: Alerts rendered ABOVE body content)
        const alerts = msg.alerts || [];
        const msgId = msg.turn_id || `ast-${idx}`;
        const blocked = Boolean(msg.blocked);

        return (
          <div key={msgId} className="space-y-2.5">
            {alerts.length > 0 && (
              <div className="space-y-2" data-testid="inline-alert-stack">
                {alerts.map((al, aIdx) => (
                  <AlertCardInline key={`${al.rule_id}-${aIdx}`} alert={al} />
                ))}
              </div>
            )}

            <article
              className={`rounded-3xl p-4 shadow-sm ${
                blocked
                  ? "glass-danger border border-rose-300"
                  : "clinical-paper border border-slate-200/90"
              }`}
            >
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-2 text-[11px]">
                <div className="flex flex-wrap items-center gap-2">
                  {blocked ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-rose-700 px-2.5 py-0.5 font-bold text-white">
                      <ShieldX className="h-3 w-3" />
                      确定性复检已拦截
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2.5 py-0.5 font-semibold text-emerald-900">
                      <CheckCircle2 className="h-3 w-3" />
                      复检通过
                    </span>
                  )}
                  {msg.model && (
                    <span className="rounded-md bg-slate-100 px-2 py-0.5 font-mono text-slate-600">
                      {msg.model}
                    </span>
                  )}
                  {msg.stale && (
                    <span className="rounded-full bg-amber-100 px-2 py-0.5 font-semibold text-amber-800">
                      量表已变更，建议基于最新分值重新提问
                    </span>
                  )}
                </div>
                <span className="font-mono text-slate-400">{msg.ts || ""}</span>
              </div>

              <ReasoningBlockL3
                reasoning={msg.reasoning}
                verification={msg.reasoning_verification}
              />

              <div className="text-sm leading-relaxed text-slate-800">
                <MarkdownView content={msg.content} />
              </div>

              {/* Assistant reply action bar */}
              <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-slate-100 pt-2.5 text-[11px]">
                <button
                  type="button"
                  onClick={() => handleCopy(msgId, msg.content)}
                  className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1 font-medium text-slate-700 hover:bg-slate-50"
                >
                  {copiedId === msgId ? (
                    <>
                      <Check className="h-3 w-3 text-emerald-600" />
                      已复制
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      复制
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={onBuildHandover}
                  className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1 font-medium text-slate-700 hover:bg-slate-50"
                >
                  <ClipboardCheck className="h-3 w-3" />
                  交班摘要
                </button>

                <button
                  type="button"
                  disabled={generatingEdu}
                  onClick={() => onGenerateEducation(msg.turn_id)}
                  className="inline-flex items-center gap-1 rounded-xl border border-cyan-200 bg-cyan-50/70 px-2.5 py-1 font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
                >
                  <BookOpenText className="h-3 w-3" />
                  {generatingEdu ? "生成宣教中..." : "生成患者宣教"}
                </button>

                <button
                  type="button"
                  onClick={() => onOpenAudit(msg.turn_id)}
                  className="ml-auto inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1 font-medium text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                  title="打开审计轨迹弹窗并定位该回复"
                >
                  <FileSearch className="h-3 w-3" />
                  🔍 审计
                </button>
              </div>
            </article>
          </div>
        );
      })}

      {/* Live SSE pipeline steps & early-pushed alerts BEFORE content arrives */}
      {(streaming || liveSteps.length > 0) && (
        <div className="space-y-2.5">
          {liveSteps.length > 0 && (
            <div className="glass-surface rounded-2xl px-3.5 py-2.5 text-xs">
              <button
                type="button"
                onClick={() => setStepsExpanded((v) => !v)}
                className="flex w-full items-center justify-between font-semibold text-slate-700"
              >
                <span className="flex items-center gap-2">
                  {streaming ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-cyan-700" />
                  ) : (
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                  )}
                  L1 护栏流水线{streaming ? "执行中..." : "（已完成）"}
                </span>
                {stepsExpanded ? (
                  <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5 text-slate-400" />
                )}
              </button>

              {(streaming || stepsExpanded) && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {liveSteps.map((st, idx) => (
                    <span
                      key={`${st.step}-${idx}`}
                      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-medium ${
                        st.status === "done"
                          ? "bg-emerald-100 text-emerald-900"
                          : "bg-cyan-100 text-cyan-900"
                      }`}
                    >
                      {st.status === "done" ? "✓" : "⋯"} {st.label}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {streaming && liveAlerts.length > 0 && (
            <div className="space-y-2">
              {liveAlerts.map((al, i) => (
                <AlertCardInline key={`live-${al.rule_id}-${i}`} alert={al} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Standalone patient education documents (D7) */}
      {educations.length > 0 && (
        <div className="space-y-2 pt-2">
          <div className="flex items-center gap-1.5 px-1 text-xs font-bold text-slate-600">
            <BookOpenText className="h-3.5 w-3.5 text-cyan-700" />
            本会话独立患者宣教文书（均已过确定性输出复检）
          </div>
          {educations.map((edu, idx) => (
            <div
              key={`edu-${idx}`}
              className={`rounded-3xl border p-4 ${
                edu.blocked
                  ? "border-rose-300 bg-rose-50/90"
                  : "border-teal-200/90 bg-teal-50/45"
              }`}
            >
              <div className="mb-2 flex items-center justify-between text-[11px]">
                <span className="inline-flex items-center gap-1 font-bold text-teal-900">
                  {edu.blocked ? (
                    <AlertTriangle className="h-3.5 w-3.5 text-rose-600" />
                  ) : (
                    <CheckCircle2 className="h-3.5 w-3.5 text-teal-700" />
                  )}
                  分阶段康复宣教要点 #{idx + 1}
                </span>
                <span className="font-mono text-slate-400">
                  {edu.model || ""} · {edu.ts}
                </span>
              </div>
              <div className="text-sm text-slate-800">
                <MarkdownView content={edu.content} />
              </div>
            </div>
          ))}
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
};
