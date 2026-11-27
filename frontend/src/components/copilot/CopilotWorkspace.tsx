import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  FileSearch,
  FolderHeart,
  MessageSquarePlus,
  UserPlus,
} from "lucide-react";
import { api } from "../../api/client";
import type {
  CardDiffPayload,
  ChatStepEvent,
  OutputMode,
  PatientDetailResponse,
  PatientSummary,
  ScaleSpec,
  SessionState,
  SessionSummary,
  ThinkingLevel,
  UnifiedAlert,
} from "../../api/types";
import { Modal } from "../Modal";
import { AuditModal } from "./AuditModal";
import { ChatStream } from "./ChatStream";
import { Composer } from "./Composer";
import { PatientCardEditor } from "./PatientCardEditor";
import { PatientExplorer } from "./PatientExplorer";
import { DrawerTab, SessionDrawer } from "./SessionDrawer";

interface CopilotWorkspaceProps {
  availableModels: string[];
  defaultModel: string;
  mockMode: boolean;
  scalesCatalog: ScaleSpec[];
  defaultExpandSteps?: boolean;
  onRenderSidebarSlot?: (node: React.ReactNode) => void;
  onRenderHeaderActions?: (node: React.ReactNode) => void;
}

export const CopilotWorkspace: React.FC<CopilotWorkspaceProps> = ({
  availableModels,
  defaultModel,
  mockMode,
  scalesCatalog,
  defaultExpandSteps = false,
  onRenderSidebarSlot,
  onRenderHeaderActions,
}) => {
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [loadingPatients, setLoadingPatients] = useState(false);
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [patientDetail, setPatientDetail] = useState<PatientDetailResponse | null>(null);
  const [activeView, setActiveView] = useState<"card" | "session">("session");

  const [sessionsByPatient, setSessionsByPatient] = useState<Record<string, SessionSummary[]>>({});
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [sessionState, setSessionState] = useState<SessionState | null>(null);

  const [streaming, setStreaming] = useState(false);
  const [liveSteps, setLiveSteps] = useState<ChatStepEvent[]>([]);
  const [liveAlerts, setLiveAlerts] = useState<UnifiedAlert[]>([]);
  const [liveAlertsTurnId, setLiveAlertsTurnId] = useState<string | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [generatingEdu, setGeneratingEdu] = useState(false);
  const [busyAction, setBusyAction] = useState(false);

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState<DrawerTab>("alerts");

  const [auditOpen, setAuditOpen] = useState(false);
  const [auditTurnId, setAuditTurnId] = useState<string | null>(null);

  const [handoverText, setHandoverText] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const refreshPatients = useCallback(async () => {
    setLoadingPatients(true);
    try {
      const res = await api.listPatients();
      setPatients(res.patients || []);
      return res.patients || [];
    } catch {
      return [];
    } finally {
      setLoadingPatients(false);
    }
  }, []);

  const refreshSessionsForPatient = useCallback(async (patientId: string) => {
    try {
      const res = await api.listSessions(patientId);
      setSessionsByPatient((prev) => ({
        ...prev,
        [patientId]: res.sessions || [],
      }));
      return res.sessions || [];
    } catch {
      return [];
    }
  }, []);

  const loadPatientAndSessions = useCallback(
    async (patientId: string) => {
      try {
        const [detail, sessions] = await Promise.all([
          api.getPatient(patientId),
          refreshSessionsForPatient(patientId),
        ]);
        setPatientDetail(detail);
        return { detail, sessions };
      } catch {
        return null;
      }
    },
    [refreshSessionsForPatient]
  );

  // Initial load
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const list = await refreshPatients();
      if (cancelled || list.length === 0) return;
      const firstId = list[0].patient_id;
      setSelectedPatientId(firstId);
      const loaded = await loadPatientAndSessions(firstId);
      if (cancelled || !loaded) return;
      if (loaded.sessions.length > 0) {
        const sid = loaded.sessions[0].session_id;
        setSelectedSessionId(sid);
        setActiveView("session");
        try {
          const st = await api.getSession(firstId, sid);
          if (!cancelled) setSessionState(st);
        } catch {
          // ignore
        }
      } else {
        setActiveView("card");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loadPatientAndSessions, refreshPatients]);

  const handleSelectPatientCard = useCallback(
    async (patientId: string) => {
      setSelectedPatientId(patientId);
      setActiveView("card");
      setStreamError(null);
      await loadPatientAndSessions(patientId);
    },
    [loadPatientAndSessions]
  );

  const handleSelectSession = useCallback(
    async (patientId: string, sessionId: string) => {
      setSelectedPatientId(patientId);
      setSelectedSessionId(sessionId);
      setActiveView("session");
      setStreamError(null);
      setLiveSteps([]);
      setLiveAlerts([]);
      try {
        const [detail, state] = await Promise.all([
          api.getPatient(patientId),
          api.getSession(patientId, sessionId),
        ]);
        setPatientDetail(detail);
        setSessionState(state);

        // Auto-open critical alerts or default timeline/diffs
        const hasCrit = (state.latest_alerts || []).some((a) => a.severity === "critical");
        const hasMissingScale = (state.latest_scale_results || []).some(
          (s) => s.status !== "ok"
        );
        if (hasCrit) {
          setDrawerOpen(true);
          setDrawerTab("alerts");
        } else if (hasMissingScale) {
          setDrawerTab("scales");
        } else {
          setDrawerTab("timeline");
        }
      } catch (err) {
        setStreamError(err instanceof Error ? err.message : "加载会话失败");
      }
    },
    []
  );

  const handleCreatePatient = useCallback(
    async (payload: { display_name: string; body: string }) => {
      const detail = await api.createPatient(payload);
      await refreshPatients();
      setSelectedPatientId(detail.patient_id);
      setPatientDetail(detail);
      const createdSess = await api.createSession(detail.patient_id, "初诊诊疗");
      await refreshSessionsForPatient(detail.patient_id);
      setSelectedSessionId(createdSess.session_id);
      setSessionState(createdSess);
      setActiveView("session");
    },
    [refreshPatients, refreshSessionsForPatient]
  );

  const handleCreateSession = useCallback(
    async (patientId: string, title?: string) => {
      const created = await api.createSession(patientId, title || "");
      await Promise.all([refreshPatients(), refreshSessionsForPatient(patientId)]);
      const detail = await api.getPatient(patientId);
      setSelectedPatientId(patientId);
      setPatientDetail(detail);
      setSelectedSessionId(created.session_id);
      setSessionState(created);
      setActiveView("session");
      setLiveSteps([]);
      setLiveAlerts([]);
      setStreamError(null);
    },
    [refreshPatients, refreshSessionsForPatient]
  );

  const handleDeletePatient = useCallback(
    async (patientId: string) => {
      await api.deletePatient(patientId);
      const next = await refreshPatients();
      if (selectedPatientId === patientId) {
        if (next.length > 0) {
          await handleSelectPatientCard(next[0].patient_id);
        } else {
          setSelectedPatientId(null);
          setPatientDetail(null);
          setSelectedSessionId(null);
          setSessionState(null);
        }
      }
    },
    [handleSelectPatientCard, refreshPatients, selectedPatientId]
  );

  const handleSendChat = async (payload: {
    content: string;
    model: string;
    thinking: ThinkingLevel;
    output_mode: OutputMode;
    pii_action: "none" | "sanitized" | "override_raw";
  }) => {
    if (!selectedPatientId || !selectedSessionId) return;
    setStreamError(null);
    setStreaming(true);
    setLiveSteps([]);
    setLiveAlerts([]);
    setLiveAlertsTurnId(null);

    // Optimistic append of user message while streaming
    setSessionState((prev) =>
      prev
        ? {
            ...prev,
            messages: [
              ...prev.messages,
              {
                role: "user",
                content: payload.content,
                model: payload.model,
                output_mode: payload.output_mode,
                thinking: payload.thinking,
              },
            ],
          }
        : prev
    );

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await api.streamSessionChat(
        selectedPatientId,
        selectedSessionId,
        {
          content: payload.content,
          model: payload.model,
          thinking: payload.thinking,
          output_mode: payload.output_mode,
          mock_mode: mockMode,
          pii_action: payload.pii_action,
        },
        {
          onStep: (ev) => {
            setLiveSteps((prev) => {
              const idx = prev.findIndex((s) => s.step === ev.step);
              if (idx >= 0) {
                const copy = [...prev];
                copy[idx] = ev;
                return copy;
              }
              return [...prev, ev];
            });
          },
          onAlert: (ev) => {
            const alerts = ev.unified_alerts || [];
            setLiveAlerts(alerts);
            setLiveAlertsTurnId(ev.turn_id || null);
            const hasCritical = alerts.some((a) => a.severity === "critical");
            if (hasCritical) {
              // D10: Critical red flag forces drawer open to alerts tab
              setDrawerOpen(true);
              setDrawerTab("alerts");
            }
          },
          onError: (msg) => {
            setStreamError(msg);
          },
          onDone: () => {
            // Finalize by replaying authoritative trace from backend
          },
        },
        controller.signal
      );

      const [replayed, ptDetail] = await Promise.all([
        api.getSession(selectedPatientId, selectedSessionId),
        api.getPatient(selectedPatientId),
      ]);
      setSessionState(replayed);
      setPatientDetail(ptDetail);
      await Promise.all([
        refreshPatients(),
        refreshSessionsForPatient(selectedPatientId),
      ]);

      // D13 auto-tab focus after turn completes
      const hasCritical = (replayed.latest_alerts || []).some(
        (a) => a.severity === "critical"
      );
      const hasIncompleteScale = (replayed.latest_scale_results || []).some(
        (s) => s.status !== "ok"
      );
      if (hasCritical) {
        setDrawerOpen(true);
        setDrawerTab("alerts");
      } else if (hasIncompleteScale) {
        setDrawerOpen(true);
        setDrawerTab("scales");
      } else if ((replayed.pending_card_diffs || []).length > 0 && !drawerOpen) {
        setDrawerTab("timeline");
      }
    } catch (err) {
      setStreamError(err instanceof Error ? err.message : "发送诊疗分析失败");
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const handleGenerateEducation = async (turnId?: string | null) => {
    if (!selectedPatientId || !selectedSessionId) return;
    setGeneratingEdu(true);
    setStreamError(null);
    try {
      await api.generateSessionEducation(selectedPatientId, selectedSessionId, {
        model: defaultModel,
        mock_mode: mockMode,
        turn_id: turnId || undefined,
      });
      const updated = await api.getSession(selectedPatientId, selectedSessionId);
      setSessionState(updated);
    } catch (err) {
      setStreamError(err instanceof Error ? err.message : "生成患者宣教失败");
    } finally {
      setGeneratingEdu(false);
    }
  };

  const handleRecalculateScale = async (
    scaleId: string,
    params: Record<string, number | null>
  ) => {
    if (!selectedPatientId || !selectedSessionId) return;
    setBusyAction(true);
    try {
      const res = await api.recalculateSessionScale(selectedPatientId, selectedSessionId, {
        scale_id: scaleId,
        updated_params: params,
      });
      setSessionState(res.session);
      const pt = await api.getPatient(selectedPatientId);
      setPatientDetail(pt);
      await refreshPatients();
      if ((res.session.latest_alerts || []).some((a) => a.severity === "critical")) {
        setDrawerOpen(true);
        setDrawerTab("alerts");
      }
    } catch (err) {
      setStreamError(err instanceof Error ? err.message : "量表重算失败");
    } finally {
      setBusyAction(false);
    }
  };

  const handleResolveRedFlag = async (ruleId: string, reason: string) => {
    if (!selectedPatientId || !selectedSessionId) return;
    setBusyAction(true);
    try {
      const pt = await api.commitPatientChanges(selectedPatientId, {
        session_id: selectedSessionId,
        changes: {
          resolve_red_flags: [{ id: ruleId, resolved_reason: reason }],
        },
      });
      setPatientDetail(pt);
      const sess = await api.getSession(selectedPatientId, selectedSessionId);
      setSessionState(sess);
      await refreshPatients();
    } catch (err) {
      setStreamError(err instanceof Error ? err.message : "解除红旗失败");
    } finally {
      setBusyAction(false);
    }
  };

  const handleCommitDiff = async (diff: CardDiffPayload) => {
    if (!selectedPatientId || !selectedSessionId) return;
    setBusyAction(true);
    try {
      const changes: Record<string, unknown> = {};
      if (diff.add_medications?.length) changes.add_medications = diff.add_medications;
      if (diff.add_surgical_history?.length) {
        changes.add_surgical_history = diff.add_surgical_history;
      }
      if (diff.add_scale_history?.length) {
        changes.add_scale_history = diff.add_scale_history;
      }
      const pt = await api.commitPatientChanges(selectedPatientId, {
        session_id: selectedSessionId,
        diff_id: diff.diff_id,
        changes,
      });
      setPatientDetail(pt);
      const sess = await api.getSession(selectedPatientId, selectedSessionId);
      setSessionState(sess);
      await Promise.all([
        refreshPatients(),
        refreshSessionsForPatient(selectedPatientId),
      ]);
    } catch (err) {
      setStreamError(err instanceof Error ? err.message : "提交档案变更失败");
    } finally {
      setBusyAction(false);
    }
  };

  const handleDismissDiff = async (diffId: string) => {
    if (!selectedPatientId || !selectedSessionId) return;
    setBusyAction(true);
    try {
      const res = await api.dismissSessionDiff(
        selectedPatientId,
        selectedSessionId,
        diffId
      );
      setSessionState(res.session);
      await refreshSessionsForPatient(selectedPatientId);
    } finally {
      setBusyAction(false);
    }
  };

  const handleBuildHandover = async () => {
    if (!sessionState) return;
    const latestAssistant = [...(sessionState.messages || [])]
      .reverse()
      .find((m) => m.role === "assistant");
    const firstUser = (sessionState.messages || []).find((m) => m.role === "user");
    const stateProxy = {
      raw_text: firstUser?.content || "",
      extracted_note: sessionState.latest_extracted_note || {},
      unified_alerts: sessionState.latest_alerts || [],
      scale_results: sessionState.latest_scale_results || [],
      draft: { content: latestAssistant?.content || "" },
    };
    try {
      const res = await api.exportHandover(stateProxy);
      setHandoverText(res.handover_text);
    } catch {
      // ignore
    }
  };

  // Inject PatientExplorer into GlassShell's left middle slot
  useEffect(() => {
    onRenderSidebarSlot?.(
      <PatientExplorer
        patients={patients}
        selectedPatientId={selectedPatientId}
        activeView={activeView}
        sessionsByPatient={sessionsByPatient}
        selectedSessionId={selectedSessionId}
        loading={loadingPatients}
        onSelectPatientCard={handleSelectPatientCard}
        onSelectSession={handleSelectSession}
        onCreatePatient={handleCreatePatient}
        onCreateSession={handleCreateSession}
        onDeletePatient={handleDeletePatient}
        onExpandPatient={(pid) => {
          void refreshSessionsForPatient(pid);
        }}
      />
    );
    return () => onRenderSidebarSlot?.(null);
  }, [
    activeView,
    handleCreatePatient,
    handleCreateSession,
    handleDeletePatient,
    handleSelectPatientCard,
    handleSelectSession,
    loadingPatients,
    onRenderSidebarSlot,
    patients,
    refreshSessionsForPatient,
    selectedPatientId,
    selectedSessionId,
    sessionsByPatient,
  ]);

  // Inject top-right Audit button into GlassShell header
  useEffect(() => {
    if (selectedPatientId && selectedSessionId && activeView === "session") {
      onRenderHeaderActions?.(
        <button
          type="button"
          onClick={() => {
            setAuditTurnId(null);
            setAuditOpen(true);
          }}
          data-testid="open-audit-btn"
          className="inline-flex items-center gap-1.5 rounded-full border border-slate-200/90 bg-white/85 px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-sm hover:bg-white"
        >
          <FileSearch className="h-3.5 w-3.5 text-cyan-700" />
          <span>审计轨迹</span>
        </button>
      );
    } else {
      onRenderHeaderActions?.(null);
    }
    return () => onRenderHeaderActions?.(null);
  }, [activeView, onRenderHeaderActions, selectedPatientId, selectedSessionId]);

  const activeAlerts =
    streaming && liveAlerts.length > 0 ? liveAlerts : sessionState?.latest_alerts || [];
  const hasCriticalAlert = activeAlerts.some((a) => a.severity === "critical");

  if (!selectedPatientId || !patientDetail) {
    return (
      <div className="glass-surface flex min-h-[520px] flex-col items-center justify-center rounded-3xl p-8 text-center">
        <div className="mb-3 flex h-14 w-14 items-center justify-center rounded-3xl bg-cyan-50 text-cyan-700">
          <FolderHeart className="h-7 w-7" />
        </div>
        <h2 className="text-base font-bold text-slate-900">建立第一位骨伤康复患者档案</h2>
        <p className="mt-1.5 max-w-md text-xs leading-relaxed text-slate-500">
          v0.4 采用「患者档案卡 PATIENT.md + 多诊次会话追溯」机制。每个患者文件夹跨会话自动继承有效红旗、长期用药洗脱期与手术内固定边界。
        </p>
        <button
          type="button"
          onClick={() => {
            const btn = document.querySelector<HTMLButtonElement>('[data-testid="new-patient-btn"]');
            btn?.click();
          }}
          className="mt-5 inline-flex items-center gap-1.5 rounded-2xl bg-gradient-to-r from-cyan-700 to-teal-600 px-5 py-2.5 text-xs font-semibold text-white shadow-md"
        >
          <UserPlus className="h-4 w-4" />
          立即新建患者
        </button>
      </div>
    );
  }

  return (
    <div className="flex min-h-[540px] flex-col gap-3 md:h-[calc(100vh-92px)] md:flex-row">
      {/* Middle Pane: PatientCardEditor OR Active Session ChatStream + Composer */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {activeView === "card" ? (
          <PatientCardEditor
            patient={patientDetail}
            onUpdated={(next) => {
              setPatientDetail(next);
              void refreshPatients();
            }}
            onStartNewSession={(pid) => {
              void handleCreateSession(pid);
            }}
          />
        ) : !sessionState ? (
          <div className="glass-surface flex flex-1 flex-col items-center justify-center rounded-3xl p-6 text-center">
            <p className="text-xs text-slate-500">该患者尚无诊疗会话，点击下方创建初诊或复诊会话。</p>
            <button
              type="button"
              onClick={() => handleCreateSession(selectedPatientId)}
              className="mt-3 inline-flex items-center gap-1.5 rounded-xl bg-cyan-700 px-4 py-2 text-xs font-semibold text-white"
            >
              <MessageSquarePlus className="h-3.5 w-3.5" />
              新建诊疗会话
            </button>
          </div>
        ) : (
          <>
            {streamError && (
              <div className="mb-2 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-2 text-xs text-rose-800">
                {streamError}
              </div>
            )}

            <ChatStream
              patientCard={patientDetail.card}
              messages={sessionState.messages || []}
              educations={sessionState.educations || []}
              streaming={streaming}
              liveSteps={liveSteps}
              liveAlerts={liveAlerts}
              liveAlertsTurnId={liveAlertsTurnId}
              generatingEdu={generatingEdu}
              defaultExpandSteps={defaultExpandSteps}
              onGenerateEducation={handleGenerateEducation}
              onOpenAudit={(turnId) => {
                setAuditTurnId(turnId || null);
                setAuditOpen(true);
              }}
              onBuildHandover={handleBuildHandover}
            />

            <Composer
              availableModels={availableModels}
              defaultModel={defaultModel}
              mockMode={mockMode}
              streaming={streaming}
              onSend={handleSendChat}
            />
          </>
        )}
      </div>

      {/* Right Pane: Current Encounter SessionDrawer */}
      {activeView === "session" && sessionState && (
        <SessionDrawer
          open={drawerOpen}
          activeTab={drawerTab}
          hasCriticalAlert={hasCriticalAlert}
          alerts={activeAlerts}
          extractedNote={sessionState.latest_extracted_note || {}}
          scaleResults={sessionState.latest_scale_results || []}
          scalesCatalog={scalesCatalog}
          pendingDiffs={sessionState.pending_card_diffs || []}
          busyAction={busyAction}
          onToggleOpen={setDrawerOpen}
          onChangeTab={setDrawerTab}
          onResolveRedFlag={handleResolveRedFlag}
          onRecalculateScale={handleRecalculateScale}
          onCommitDiff={handleCommitDiff}
          onDismissDiff={handleDismissDiff}
        />
      )}

      {/* Audit Modal */}
      <AuditModal
        open={auditOpen}
        onClose={() => setAuditOpen(false)}
        patientId={selectedPatientId}
        sessionId={selectedSessionId}
        focusTurnId={auditTurnId}
      />

      {/* Handover Summary Modal */}
      <Modal
        open={Boolean(handoverText)}
        onClose={() => setHandoverText(null)}
        title="临床交班摘要（纯文本可直接粘贴至病程记录）"
        maxWidthClass="max-w-2xl"
      >
        <div className="space-y-3">
          <pre className="max-h-[55vh] overflow-auto whitespace-pre-wrap rounded-2xl border border-slate-200 bg-slate-50 p-4 font-sans text-xs leading-relaxed text-slate-800">
            {handoverText || ""}
          </pre>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={async () => {
                if (handoverText) {
                  await navigator.clipboard.writeText(handoverText);
                }
                setHandoverText(null);
              }}
              className="rounded-xl bg-cyan-700 px-4 py-1.5 text-xs font-semibold text-white"
            >
              一键复制并关闭
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
};
