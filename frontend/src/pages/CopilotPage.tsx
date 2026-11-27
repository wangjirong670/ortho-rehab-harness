import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Calculator,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Copy,
  Download,
  FileText,
  FolderOpen,
  FolderPlus,
  History,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Trash2,
  Undo2,
  Upload,
  X,
  Zap,
} from "lucide-react";
import { api } from "../api/client";
import type {
  CaseWorkspaceRecord,
  CaseWorkspaceSummary,
  CopilotAnalysisResponse,
  DemoCaseItem,
  DraftMode,
  PiiFinding,
  ScaleResult,
  ScaleSpec,
  UnifiedAlert,
} from "../api/types";
import { MarkdownView } from "../components/MarkdownView";
import { ScaleEditor } from "../components/ScaleEditor";
import {
  DRAFT_MODE_LABELS,
  formatActiveMedStopDisplay,
  getClinicalActionLabel,
  getScoreDirectionLabel,
  MED_STATUS_ZH,
  POLARITY_ZH,
  SEVERITY_LABELS,
  SOURCE_LABEL_ZH,
  TCM_FIELD_LABELS,
} from "../utils/labels";

interface CopilotPageProps {
  activeModel: string;
  mockMode: boolean;
  scalesCatalog: ScaleSpec[];
  actionLabels?: Record<string, string>;
}

type RightTab = "timeline" | "scales" | "draft" | "trace";

function nowClock(): string {
  return new Date().toTimeString().slice(0, 8);
}

function formatIsoClock(iso?: string): string {
  if (!iso) return nowClock();
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) {
    return iso.slice(11, 19) || nowClock();
  }
  return d.toTimeString().slice(0, 8);
}

function formatShortDateTime(iso?: string): string {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(0, 16).replace("T", " ");
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${mm}-${dd} ${hh}:${mi}`;
}

function deriveFallbackCapsules(alerts: UnifiedAlert[]): string[] {
  const tags: string[] = [];
  const add = (tag: string) => {
    if (!tags.includes(tag)) tags.push(tag);
  };
  for (const a of alerts) {
    const blob = `${a.name} ${a.clinical_alert || ""} ${(a.contraindicated_actions || []).join(" ")}`;
    if (/旋转扳法|斜扳|掰脖子|暴力正骨|踩跷/.test(blob)) add("禁用旋转扳法/暴力正骨");
    if (/艾灸|重灸|热敷|烤灯|热疗/.test(blob)) add("禁用艾灸/热敷热疗");
    if (/三七|活血|跌打丸|NSAIDs|同服|叠服|药酒/.test(blob)) add("严禁活血药/止痛药叠加");
    if (/推拿|揉捏|按摩/.test(blob)) add("禁止患处推拿揉按");
    if (/牵引/.test(blob)) add("禁止盲目脊柱牵引");
    if (/弃拐|全负重|跑跳|开链/.test(blob)) add("禁止早期负重/跑跳");
    if (/屈髋|矮凳|翘二郎腿|内旋/.test(blob)) add("禁止屈髋>90°/盘腿内旋");
    if (/小针刀|封闭/.test(blob)) add("禁用有创小针刀/封闭");
    if (/急诊|马尾|深静脉血栓/.test(blob)) add("严禁拖延保守观察");
  }
  return tags.slice(0, 8);
}

export const CopilotPage: React.FC<CopilotPageProps> = ({
  activeModel,
  mockMode,
  scalesCatalog,
  actionLabels,
}) => {
  const [demoCases, setDemoCases] = useState<DemoCaseItem[]>([]);
  const [rawText, setRawText] = useState("");
  const [activeDemoId, setActiveDemoId] = useState<string | null>(null);
  const [draftMode, setDraftMode] = useState<DraftMode>("ward_round");

  const [piiFindings, setPiiFindings] = useState<PiiFinding[]>([]);
  const [hasPii, setHasPii] = useState(false);
  const [preSanitizeBackup, setPreSanitizeBackup] = useState<string | null>(null);
  const [piiFeedback, setPiiFeedback] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<CopilotAnalysisResponse | null>(null);

  // Local Case Workspace State
  const [activeCase, setActiveCase] = useState<CaseWorkspaceRecord | null>(null);
  const [savedTimeLabel, setSavedTimeLabel] = useState<string | null>(null);
  const [savingCase, setSavingCase] = useState(false);
  const [caseList, setCaseList] = useState<CaseWorkspaceSummary[]>([]);
  const [archivesDir, setArchivesDir] = useState<string>("");
  const [caseSearch, setCaseSearch] = useState("");
  const [highRiskOnly, setHighRiskOnly] = useState(false);
  const [loadingCases, setLoadingCases] = useState(false);
  const [editingCaseId, setEditingCaseId] = useState<string | null>(null);
  const [editingCaseTitle, setEditingCaseTitle] = useState("");
  const [confirmDeleteCaseId, setConfirmDeleteCaseId] = useState<string | null>(null);

  // Modals: Follow-up & Revisions
  const [followupModalOpen, setFollowupModalOpen] = useState(false);
  const [followupDate, setFollowupDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [followupText, setFollowupText] = useState("");
  const [revisionsModalOpen, setRevisionsModalOpen] = useState(false);

  const [activeTab, setActiveTab] = useState<RightTab>("timeline");
  const [alertsExpanded, setAlertsExpanded] = useState(false);
  const [suppressedExpanded, setSuppressedExpanded] = useState(false);
  const [expandedScaleEditorIdx, setExpandedScaleEditorIdx] = useState<number | null>(null);
  const [addScaleId, setAddScaleId] = useState<string>("");

  const [loadingAction, setLoadingAction] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [copyToast, setCopyToast] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const importCaseInputRef = useRef<HTMLInputElement | null>(null);

  const scaleSpecMap = useMemo(() => {
    const map = new Map<string, ScaleSpec>();
    for (const s of scalesCatalog) map.set(s.scale_id, s);
    return map;
  }, [scalesCatalog]);

  const refreshCaseList = useCallback(
    async (qVal = caseSearch, riskVal = highRiskOnly) => {
      setLoadingCases(true);
      try {
        const res = await api.listCaseWorkspaces({
          q: qVal,
          high_risk_only: riskVal,
        });
        setCaseList(res.cases || []);
        setArchivesDir(res.archives_dir || "");
      } catch {
        // non-fatal if backend offline initially
      } finally {
        setLoadingCases(false);
      }
    },
    [caseSearch, highRiskOnly]
  );

  useEffect(() => {
    let active = true;
    api
      .getDemoCases()
      .then((res) => {
        if (active) setDemoCases(res.cases || []);
      })
      .catch(() => {
        // non-fatal if backend offline initially
      });
    return () => {
      active = false;
    };
  }, []);

  // Load & filter Case Workspace list (debounced on search input)
  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refreshCaseList(caseSearch, highRiskOnly);
    }, 220);
    return () => window.clearTimeout(timer);
  }, [caseSearch, highRiskOnly, refreshCaseList]);

  // Debounced PII detection on rawText changes
  useEffect(() => {
    const trimmed = rawText.trim();
    if (!trimmed) {
      setHasPii(false);
      setPiiFindings([]);
      return;
    }
    const timer = window.setTimeout(async () => {
      try {
        const res = await api.detectPii(trimmed);
        setHasPii(Boolean(res.has_pii));
        setPiiFindings(res.findings || []);
      } catch {
        // ignore transient errors during typing
      }
    }, 450);
    return () => window.clearTimeout(timer);
  }, [rawText]);

  const showToast = (msg: string) => {
    setCopyToast(msg);
    window.setTimeout(() => setCopyToast(null), 2600);
  };

  const applyLoadedCase = (record: CaseWorkspaceRecord) => {
    setActiveCase(record);
    setRawText(record.raw_text || "");
    const nextMode = ((record.draft_mode ||
      record.analysis_state?.draft_mode ||
      "ward_round") as DraftMode);
    setDraftMode(nextMode);
    setAnalysis(record.analysis_state || null);
    setAlertsExpanded(false);
    setExpandedScaleEditorIdx(null);
    setActiveDemoId(null);
    setPreSanitizeBackup(null);
    setPiiFeedback(null);
    setErrorMsg(null);
    setSavedTimeLabel(formatIsoClock(record.updated_at));
  };

  const persistCaseWorkspace = async (
    nextRawText: string,
    nextDraftMode: DraftMode,
    nextAnalysis: CopilotAnalysisResponse,
    opts?: {
      recordRevision?: boolean;
      revisionNote?: string;
      caseIdOverride?: string | null;
    }
  ): Promise<CaseWorkspaceRecord | null> => {
    if (!nextRawText.trim()) return null;
    const targetCaseId =
      opts?.caseIdOverride !== undefined ? opts.caseIdOverride : activeCase?.case_id;
    setSavingCase(true);
    try {
      const saved = await api.saveCaseWorkspace({
        case_id: targetCaseId || undefined,
        title: targetCaseId ? activeCase?.title : undefined,
        raw_text: nextRawText,
        draft_mode: nextDraftMode,
        analysis_state: nextAnalysis,
        record_revision: opts?.recordRevision ?? false,
        revision_note: opts?.revisionNote,
      });
      setActiveCase(saved);
      setSavedTimeLabel(formatIsoClock(saved.updated_at));
      void refreshCaseList();
      return saved;
    } catch {
      // Auto-archive failure should not block clinical workflow
      return null;
    } finally {
      setSavingCase(false);
    }
  };

  const handleNewBlankCase = () => {
    setActiveCase(null);
    setRawText("");
    setAnalysis(null);
    setActiveDemoId(null);
    setPreSanitizeBackup(null);
    setPiiFeedback(null);
    setSavedTimeLabel(null);
    setErrorMsg(null);
    setAlertsExpanded(false);
    setExpandedScaleEditorIdx(null);
  };

  const handleSelectDemo = (c: DemoCaseItem) => {
    setActiveCase(null);
    setSavedTimeLabel(null);
    setRawText(c.raw_text);
    setActiveDemoId(c.id);
    setDraftMode(c.default_draft_mode || "ward_round");
    setPreSanitizeBackup(null);
    setPiiFeedback(null);
    setErrorMsg(null);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoadingAction("upload");
    setErrorMsg(null);
    try {
      const res = await api.uploadClinicalFile(file);
      setActiveCase(null);
      setSavedTimeLabel(null);
      setRawText(res.text);
      setActiveDemoId(null);
      setPreSanitizeBackup(null);
      setPiiFeedback(`已导入 ${res.filename}（${res.char_count} 字）`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "病历文件解析失败");
    } finally {
      setLoadingAction(null);
      e.target.value = "";
    }
  };

  const handleImportCaseJson = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoadingAction("import-case");
    setErrorMsg(null);
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      const imported = await api.importCaseWorkspace(parsed);
      applyLoadedCase(imported);
      void refreshCaseList();
      showToast(`已导入病例档案「${imported.title}」`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "病例档案 JSON 导入失败");
    } finally {
      setLoadingAction(null);
      e.target.value = "";
    }
  };

  const handleOpenCase = async (caseId: string) => {
    setLoadingAction(`open-${caseId}`);
    setErrorMsg(null);
    try {
      const record = await api.getCaseWorkspace(caseId);
      applyLoadedCase(record);
      showToast(`已载入档案「${record.title}」（零模型秒级还原）`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "加载本地病例档案失败");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleRenameCase = async (caseId: string) => {
    const nextTitle = editingCaseTitle.trim();
    if (!nextTitle) {
      setEditingCaseId(null);
      return;
    }
    try {
      const updated = await api.updateCaseWorkspaceMeta(caseId, { title: nextTitle });
      if (activeCase?.case_id === caseId) {
        setActiveCase(updated);
      }
      setEditingCaseId(null);
      void refreshCaseList();
      showToast("档案名称已更新");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "重命名失败");
    }
  };

  const handleExportCase = async (caseId: string) => {
    try {
      const archive = await api.exportCaseWorkspace(caseId);
      const blob = new Blob([JSON.stringify(archive, null, 2)], {
        type: "application/json;charset=utf-8",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${archive.case_id}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      showToast(`已导出 ${archive.case_id}.json`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "导出病例档案失败");
    }
  };

  const handleDeleteCase = async (caseId: string) => {
    try {
      await api.deleteCaseWorkspace(caseId);
      if (activeCase?.case_id === caseId) {
        setActiveCase(null);
        setSavedTimeLabel(null);
      }
      setConfirmDeleteCaseId(null);
      void refreshCaseList();
      showToast("已删除本地病例档案");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "删除病例档案失败");
    }
  };

  const handleSanitize = async () => {
    if (!rawText.trim()) return;
    setLoadingAction("sanitize");
    setErrorMsg(null);
    try {
      const backup = rawText;
      const res = await api.sanitizePii(rawText);
      if (res.changed) {
        setPreSanitizeBackup(backup);
        setRawText(res.sanitized_text);
        setPiiFeedback(res.summary || "已完成隐私脱敏，可随时撤销恢复");
      } else {
        setPiiFeedback("未发现需替换的个人敏感字段");
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "脱敏处理失败");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleUndoSanitize = () => {
    if (preSanitizeBackup !== null) {
      setRawText(preSanitizeBackup);
      setPreSanitizeBackup(null);
      setPiiFeedback("已恢复脱敏前原始文本");
    }
  };

  const handleFastScan = async () => {
    if (!rawText.trim()) {
      setErrorMsg("请先输入或选择一份病历文本。");
      return;
    }
    setLoadingAction("fast-scan");
    setErrorMsg(null);
    try {
      const res = await api.fastScan(rawText);
      setAnalysis(res);
      setAlertsExpanded((res.unified_alerts || []).length > 0);
      setActiveTab("timeline");
      void persistCaseWorkspace(rawText, draftMode, res, {
        recordRevision: true,
        revisionNote: "极速安全筛查",
      });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "极速筛查失败");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleAnalyze = async (overrideText?: string, overrideCaseId?: string | null) => {
    const targetText = overrideText ?? rawText;
    if (!targetText.trim()) {
      setErrorMsg("请先输入或选择一份病历文本。");
      return;
    }
    setLoadingAction("analyze");
    setErrorMsg(null);
    try {
      const res = await api.analyzeCopilot({
        raw_text: targetText,
        draft_mode: draftMode,
        model: activeModel || undefined,
        mock_mode: mockMode,
      });
      setAnalysis(res);
      setAlertsExpanded(false);
      setExpandedScaleEditorIdx(null);
      setActiveTab("timeline");
      void persistCaseWorkspace(targetText, draftMode, res, {
        recordRevision: true,
        revisionNote: "完整临床分析",
        caseIdOverride: overrideCaseId,
      });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "临床分析执行失败");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleRecalculateScale = async (
    scaleId: string,
    updatedParams: Record<string, number | null>,
    scaleIndex: number | undefined,
    refreshDraft: boolean
  ) => {
    if (!analysis) return;
    setLoadingAction("scale");
    setErrorMsg(null);
    try {
      const activeMode = (analysis.draft_mode as DraftMode) || draftMode;
      const next = await api.recalculateScale({
        analysis_state: analysis,
        scale_id: scaleId,
        updated_params: updatedParams,
        scale_index: scaleIndex,
        refresh_draft: refreshDraft,
        draft_mode: activeMode,
        model: activeModel || undefined,
        mock_mode: mockMode,
      });
      setAnalysis(next);
      setExpandedScaleEditorIdx(null);
      if (scaleIndex === undefined) {
        setAddScaleId("");
      }
      showToast(refreshDraft ? "量表已重算并刷新当前文书" : "量表已重算（文书待刷新）");
      void persistCaseWorkspace(next.raw_text || rawText, activeMode, next, {
        recordRevision: false,
        revisionNote: refreshDraft ? "重算量表并刷新文书" : "更新量表评估",
      });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "量表重算失败");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleRemoveScale = async (scaleIndex: number) => {
    if (!analysis) return;
    setLoadingAction("scale");
    setErrorMsg(null);
    try {
      const next = await api.removeScale({
        analysis_state: analysis,
        scale_index: scaleIndex,
      });
      setAnalysis(next);
      setExpandedScaleEditorIdx(null);
      const activeMode = (next.draft_mode as DraftMode) || draftMode;
      void persistCaseWorkspace(next.raw_text || rawText, activeMode, next, {
        recordRevision: false,
        revisionNote: "移除量表评估",
      });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "移除量表失败");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleSwitchDraftMode = async (targetMode: DraftMode) => {
    setDraftMode(targetMode);
    if (!analysis) return;

    const cachedDraft = analysis.drafts_by_mode?.[targetMode];
    if (cachedDraft) {
      const isStale = Boolean(analysis.draft_stale_by_mode?.[targetMode]);
      const nextState: CopilotAnalysisResponse = {
        ...analysis,
        draft: cachedDraft,
        draft_mode: targetMode,
        draft_stale: isStale,
        draft_plain_text: cachedDraft.plain_text ?? analysis.draft_plain_text,
        draft_cache_hit: true,
      };
      setAnalysis(nextState);
      showToast(`已秒切「${DRAFT_MODE_LABELS[targetMode] || targetMode}」（本地缓存命中）`);
      if (activeCase?.case_id) {
        void persistCaseWorkspace(nextState.raw_text || rawText, targetMode, nextState, {
          recordRevision: false,
          revisionNote: `切换文书模式`,
        });
      }
      return;
    }

    setLoadingAction("draft");
    setErrorMsg(null);
    try {
      const next = await api.regenerateDraft({
        analysis_state: analysis,
        draft_mode: targetMode,
        model: activeModel || undefined,
        mock_mode: mockMode,
        force: false,
      });
      setAnalysis(next);
      showToast(
        next.draft_cache_hit
          ? `已切换「${DRAFT_MODE_LABELS[targetMode] || targetMode}」（缓存命中）`
          : `已生成「${DRAFT_MODE_LABELS[targetMode] || targetMode}」`
      );
      void persistCaseWorkspace(next.raw_text || rawText, targetMode, next, {
        recordRevision: false,
        revisionNote: `生成${DRAFT_MODE_LABELS[targetMode] || targetMode}`,
      });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "文书切换生成失败");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleForceRegenerateDraft = async (targetMode: DraftMode) => {
    if (!analysis) return;
    setLoadingAction("draft");
    setErrorMsg(null);
    setDraftMode(targetMode);
    try {
      const next = await api.regenerateDraft({
        analysis_state: analysis,
        draft_mode: targetMode,
        model: activeModel || undefined,
        mock_mode: mockMode,
        force: true,
      });
      setAnalysis(next);
      showToast(`已重新生成「${DRAFT_MODE_LABELS[targetMode] || targetMode}」`);
      void persistCaseWorkspace(next.raw_text || rawText, targetMode, next, {
        recordRevision: false,
        revisionNote: `重新生成${DRAFT_MODE_LABELS[targetMode] || targetMode}`,
      });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "文书重新生成失败");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleSubmitFollowup = async (analyzeAfter: boolean) => {
    if (!activeCase || !followupText.trim()) return;
    setLoadingAction("followup");
    setErrorMsg(null);
    try {
      const updated = await api.appendCaseFollowup(activeCase.case_id, {
        visit_date: followupDate.trim() || undefined,
        followup_text: followupText.trim(),
      });
      setActiveCase(updated);
      setRawText(updated.raw_text);
      setSavedTimeLabel(formatIsoClock(updated.updated_at));
      setFollowupText("");
      setFollowupModalOpen(false);
      void refreshCaseList();
      if (analyzeAfter) {
        setLoadingAction(null);
        await handleAnalyze(updated.raw_text, updated.case_id);
      } else {
        showToast("已追加新诊次记录，可点击「开始完整分析」更新评估");
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "追加复诊记录失败");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleRestoreRevision = async (rev: number) => {
    if (!activeCase) return;
    setLoadingAction(`restore-${rev}`);
    setErrorMsg(null);
    try {
      const restored = await api.restoreCaseRevision(activeCase.case_id, rev);
      applyLoadedCase(restored);
      setRevisionsModalOpen(false);
      void refreshCaseList();
      showToast(`已恢复至历史版本 v${rev}`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "恢复历史版本失败");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleCopyPlainOrHandover = async (kind: "plain" | "markdown" | "handover") => {
    if (!analysis) return;
    try {
      if (kind === "markdown") {
        await navigator.clipboard.writeText(analysis.draft?.content || "");
        showToast("已复制 Markdown 原文");
        return;
      }
      const res = await api.buildHandover(analysis);
      const text = kind === "handover" ? res.handover_text : res.draft_plain_text;
      await navigator.clipboard.writeText(text);
      showToast(kind === "handover" ? "已复制临床交班摘要" : "已复制纯文本草稿");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "生成复制文本失败");
    }
  };

  const handleDownloadDraft = async (kind: "plain" | "markdown" | "handover") => {
    if (!analysis) return;
    try {
      let content = analysis.draft?.content || "";
      let filename = `ortho_draft_${analysis.draft_mode || draftMode}.md`;
      if (kind !== "markdown") {
        const res = await api.buildHandover(analysis);
        content = kind === "handover" ? res.handover_text : res.draft_plain_text;
        filename =
          kind === "handover"
            ? "ortho_clinical_handover.txt"
            : `ortho_draft_${analysis.draft_mode || draftMode}.txt`;
      }
      const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "导出文件失败");
    }
  };

  const alerts = analysis?.unified_alerts || [];
  const hasCriticalOrHigh = alerts.some(
    (a) => a.severity === "critical" || a.severity === "high"
  );
  const capsules =
    analysis?.contraindicated_capsules && analysis.contraindicated_capsules.length > 0
      ? analysis.contraindicated_capsules
      : deriveFallbackCapsules(alerts);
  const suppressedMatches = analysis?.risk_report?.suppressed_matches || [];

  const existingScaleIds = new Set((analysis?.scale_results || []).map((s) => s.scale_id));
  const addableScales = scalesCatalog.filter((s) => !existingScaleIds.has(s.scale_id));
  const effectiveAddScaleId =
    addScaleId && !existingScaleIds.has(addScaleId)
      ? addScaleId
      : addableScales[0]?.scale_id || "";
  const selectedAddSpec = scaleSpecMap.get(effectiveAddScaleId) || null;

  const activeDraftMode = ((analysis?.draft_mode || draftMode) as DraftMode);
  const currentDraftStale = Boolean(
    analysis?.draft_stale_by_mode?.[activeDraftMode] ?? analysis?.draft_stale
  );

  return (
    <div className="space-y-4">
      {copyToast && (
        <div className="fixed bottom-5 right-5 z-40 rounded-full bg-slate-900/90 px-4 py-2 text-xs font-medium text-white shadow-lg backdrop-blur-md">
          {copyToast}
        </div>
      )}

      {errorMsg && (
        <div className="glass-danger flex items-center justify-between gap-3 rounded-2xl px-4 py-3 text-xs text-rose-900">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0 text-rose-700" />
            <span>{errorMsg}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorMsg(null)}
            className="rounded-lg px-2 py-1 font-medium text-rose-700 hover:bg-rose-100"
          >
            关闭
          </button>
        </div>
      )}

      {/* Top Case Workspace Status Strip */}
      <div className="glass-surface flex flex-wrap items-center justify-between gap-3 rounded-2xl px-4 py-2.5 text-xs">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 font-semibold text-slate-800">
            <FolderOpen className="h-4 w-4 text-cyan-700" />
            <span className="max-w-[260px] truncate">
              {activeCase ? activeCase.title : "未归档新病历（分析后自动建档）"}
            </span>
          </span>
          {activeCase && (
            <span className="rounded-md bg-slate-100/90 px-2 py-0.5 font-mono text-[10px] text-slate-500">
              {activeCase.case_id}
            </span>
          )}
          {savingCase ? (
            <span className="inline-flex items-center gap-1 rounded-full bg-cyan-50 px-2.5 py-0.5 text-[11px] font-medium text-cyan-800">
              <RefreshCw className="h-3 w-3 animate-spin" />
              存档同步中...
            </span>
          ) : savedTimeLabel ? (
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-0.5 text-[11px] font-medium text-emerald-800">
              <CheckCircle2 className="h-3 w-3" />
              已自动存档 {savedTimeLabel}
            </span>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {activeCase && (
            <button
              type="button"
              onClick={() => setRevisionsModalOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200/90 bg-white/90 px-3 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-white"
            >
              <History className="h-3.5 w-3.5 text-cyan-700" />
              历史版本 ({activeCase.revisions?.length ?? activeCase.revision_count ?? 1})
            </button>
          )}
          <button
            type="button"
            onClick={handleNewBlankCase}
            className="inline-flex items-center gap-1.5 rounded-xl border border-cyan-200 bg-cyan-50/85 px-3 py-1.5 text-xs font-semibold text-cyan-900 transition-colors hover:bg-cyan-100"
          >
            <FolderPlus className="h-3.5 w-3.5" />
            新建空白病例
          </button>
          <input
            ref={importCaseInputRef}
            type="file"
            accept=".json"
            className="hidden"
            onChange={handleImportCaseJson}
          />
          <button
            type="button"
            disabled={loadingAction === "import-case"}
            onClick={() => importCaseInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200/90 bg-white/90 px-3 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-white"
          >
            <Upload className="h-3.5 w-3.5 text-cyan-700" />
            导入 JSON
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12 lg:items-start">
        {/* Left 36% (~4-5 cols on 12-col grid; using 4.5 feel via lg:col-span-4 or custom flex) */}
        <section className="glass-surface space-y-4 rounded-3xl p-4 lg:col-span-4">
          {/* Demo Cases Capsules */}
          <div>
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-700">典型骨伤/康复演示病例</span>
              <span className="text-[11px] text-slate-400">点击一键载入</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {demoCases.length === 0 ? (
                <span className="text-[11px] text-slate-400">连接后端后自动加载演示病例</span>
              ) : (
                demoCases.map((c) => {
                  const selected = activeDemoId === c.id;
                  return (
                    <button
                      key={c.id}
                      type="button"
                      onClick={() => handleSelectDemo(c)}
                      className={`rounded-full border px-2.5 py-1 text-[11px] font-medium transition-all ${
                        selected
                          ? "border-cyan-700 bg-cyan-700 text-white shadow-sm"
                          : "border-white/90 bg-white/75 text-slate-700 hover:border-cyan-300 hover:bg-white"
                      }`}
                    >
                      {c.title}
                    </button>
                  );
                })
              )}
            </div>
          </div>

          {/* Upload, Follow-up & Clear Toolbar */}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.md,.docx"
                className="hidden"
                onChange={handleFileUpload}
              />
              <button
                type="button"
                disabled={loadingAction === "upload"}
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200/90 bg-white/90 px-3 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-white"
              >
                <Upload className="h-3.5 w-3.5 text-cyan-700" />
                导入病历 (.txt/.md/.docx)
              </button>
              {activeCase && (
                <button
                  type="button"
                  onClick={() => {
                    setFollowupDate(new Date().toISOString().slice(0, 10));
                    setFollowupModalOpen(true);
                  }}
                  className="inline-flex items-center gap-1 rounded-xl border border-cyan-200 bg-cyan-50/90 px-2.5 py-1.5 text-xs font-semibold text-cyan-800 transition-colors hover:bg-cyan-100"
                >
                  <Plus className="h-3.5 w-3.5" />
                  追加新诊次
                </button>
              )}
            </div>
            {rawText && (
              <button
                type="button"
                onClick={() => {
                  setRawText("");
                  setActiveDemoId(null);
                  setPreSanitizeBackup(null);
                  setPiiFeedback(null);
                }}
                className="text-xs text-slate-400 transition-colors hover:text-rose-600"
              >
                清空文本
              </button>
            )}
          </div>

          {/* Clinical Editor (Matte High-Readability Layer) */}
          <div className="clinical-paper overflow-hidden rounded-2xl p-3">
            <textarea
              rows={10}
              value={rawText}
              onChange={(e) => {
                setRawText(e.target.value);
                setPiiFeedback(null);
              }}
              placeholder="请粘贴门诊/住院病历、多诊次随访记录或主诉现病史（建议先使用下方脱敏工具去除姓名、身份证及手机号）..."
              className="w-full resize-y bg-transparent text-[13px] leading-relaxed text-slate-900 placeholder-slate-400 focus:outline-none"
            />
            <div className="mt-2 flex items-center justify-between border-t border-slate-100 pt-2 text-[11px] text-slate-400">
              <span>当前字数：{rawText.trim().length} 字</span>
              <span>严禁向公网大模型发送未脱敏身份信息</span>
            </div>
          </div>

          {/* PII Detection & Sanitization Bar */}
          {(hasPii || preSanitizeBackup !== null || piiFeedback) && (
            <div
              className={`rounded-2xl border p-3 text-xs ${
                hasPii
                  ? "border-amber-300/80 bg-amber-50/85 text-amber-900"
                  : "border-emerald-200/80 bg-emerald-50/80 text-emerald-900"
              }`}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="space-y-0.5">
                  <div className="font-semibold">
                    {hasPii
                      ? "检测到潜在个人隐私信息（建议脱敏后分析）"
                      : piiFeedback || "隐私校验状态"}
                  </div>
                  {hasPii && piiFindings.length > 0 && (
                    <div className="text-[11px] text-amber-800">
                      {piiFindings.map((f) => `${f.label} ×${f.count}`).join(" · ")}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-1.5">
                  {hasPii && (
                    <button
                      type="button"
                      disabled={loadingAction === "sanitize"}
                      onClick={handleSanitize}
                      className="rounded-lg bg-amber-600 px-2.5 py-1 text-[11px] font-semibold text-white transition-colors hover:bg-amber-700"
                    >
                      一键脱敏
                    </button>
                  )}
                  {preSanitizeBackup !== null && (
                    <button
                      type="button"
                      onClick={handleUndoSanitize}
                      className="inline-flex items-center gap-1 rounded-lg border border-slate-300 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-50"
                    >
                      <Undo2 className="h-3 w-3" />
                      撤销脱敏
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Draft Mode Selector with Cached/Stale Status Badges */}
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-slate-700">目标临床文书模式</label>
            <div className="grid grid-cols-2 gap-2">
              {(["ward_round", "rehab_education"] as DraftMode[]).map((mode) => {
                const active = draftMode === mode;
                const hasCached = Boolean(analysis?.drafts_by_mode?.[mode]);
                const isStale = Boolean(analysis?.draft_stale_by_mode?.[mode]);
                return (
                  <button
                    key={mode}
                    type="button"
                    disabled={loadingAction === "draft"}
                    onClick={() => {
                      if (analysis && analysis.raw_text === rawText) {
                        void handleSwitchDraftMode(mode);
                      } else {
                        setDraftMode(mode);
                      }
                    }}
                    className={`flex items-center justify-between gap-1.5 rounded-xl border px-3 py-2 text-left text-xs font-medium transition-all ${
                      active
                        ? "border-cyan-700 bg-cyan-50/90 text-cyan-900 shadow-sm"
                        : "border-white/80 bg-white/70 text-slate-600 hover:bg-white"
                    }`}
                  >
                    <span className="truncate">{DRAFT_MODE_LABELS[mode]}</span>
                    {hasCached && (
                      <span
                        className={`shrink-0 rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${
                          isStale
                            ? "bg-amber-100 text-amber-800"
                            : "bg-emerald-100 text-emerald-800"
                        }`}
                      >
                        {isStale ? "待刷新" : "已生成"}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Primary Actions */}
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <button
              type="button"
              disabled={Boolean(loadingAction)}
              onClick={() => void handleAnalyze()}
              className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-cyan-700 to-teal-600 px-4 py-2.5 text-xs font-semibold text-white shadow-md shadow-cyan-900/15 transition-all hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-55"
            >
              <Sparkles className="h-4 w-4" />
              {loadingAction === "analyze" ? "完整分析中..." : "开始完整分析"}
            </button>
            <button
              type="button"
              disabled={Boolean(loadingAction)}
              onClick={handleFastScan}
              className="inline-flex items-center justify-center gap-1.5 rounded-2xl border border-cyan-200 bg-white/85 px-3.5 py-2.5 text-xs font-semibold text-cyan-800 transition-all hover:bg-white disabled:cursor-not-allowed disabled:opacity-55"
            >
              <Zap className="h-3.5 w-3.5 text-amber-600" />
              {loadingAction === "fast-scan" ? "筛查中..." : "极速安全筛查"}
            </button>
          </div>

          {/* Local Case Workspace Cabinet */}
          <div className="space-y-2.5 border-t border-white/70 pt-3">
            <div className="flex items-center justify-between text-xs">
              <span
                className="inline-flex items-center gap-1.5 font-semibold text-slate-700"
                title={archivesDir ? `本地存储目录：${archivesDir}` : undefined}
              >
                <FolderOpen className="h-3.5 w-3.5 text-cyan-700" />
                🗂️ 本地病例档案柜 ({caseList.length})
              </span>
              <button
                type="button"
                onClick={() => setHighRiskOnly((v) => !v)}
                className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold transition-colors ${
                  highRiskOnly
                    ? "border-rose-300 bg-rose-100 text-rose-800"
                    : "border-slate-200 bg-white/80 text-slate-600 hover:bg-white"
                }`}
              >
                {highRiskOnly ? "仅高危红旗 ✓" : "仅看高危红旗"}
              </button>
            </div>

            <div className="relative">
              <Search className="pointer-events-none absolute left-2.5 top-2 h-3.5 w-3.5 text-slate-400" />
              <input
                type="text"
                value={caseSearch}
                onChange={(e) => setCaseSearch(e.target.value)}
                placeholder="搜索病例标题、红旗标签或病历关键词..."
                className="w-full rounded-xl border border-slate-200/90 bg-white/85 py-1.5 pl-8 pr-7 text-xs text-slate-800 placeholder-slate-400 focus:border-cyan-600 focus:outline-none"
              />
              {caseSearch && (
                <button
                  type="button"
                  onClick={() => setCaseSearch("")}
                  className="absolute right-2 top-1.5 text-slate-400 hover:text-slate-600"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>

            {loadingCases && caseList.length === 0 ? (
              <p className="text-[11px] text-slate-400">正在加载本地档案柜...</p>
            ) : caseList.length === 0 ? (
              <p className="text-[11px] text-slate-400">
                {caseSearch || highRiskOnly
                  ? "未匹配到符合条件的本地病例档案。"
                  : "暂无本地病例档案，分析完成后将自动持久化到本地档案柜。"}
              </p>
            ) : (
              <div className="max-h-72 space-y-1.5 overflow-y-auto pr-1">
                {caseList.map((item) => {
                  const isCurrent = activeCase?.case_id === item.case_id;
                  const isEditingName = editingCaseId === item.case_id;
                  const isConfirmDel = confirmDeleteCaseId === item.case_id;
                  return (
                    <div
                      key={item.case_id}
                      className={`rounded-xl border p-2.5 text-xs transition-all ${
                        isCurrent
                          ? "border-cyan-600 bg-cyan-50/85 shadow-sm"
                          : "border-white/85 bg-white/70 hover:bg-white"
                      }`}
                    >
                      {isEditingName ? (
                        <div className="flex items-center gap-1.5">
                          <input
                            type="text"
                            value={editingCaseTitle}
                            onChange={(e) => setEditingCaseTitle(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") void handleRenameCase(item.case_id);
                              if (e.key === "Escape") setEditingCaseId(null);
                            }}
                            autoFocus
                            className="min-w-0 flex-1 rounded-lg border border-cyan-400 bg-white px-2 py-1 text-xs text-slate-900 focus:outline-none"
                          />
                          <button
                            type="button"
                            onClick={() => void handleRenameCase(item.case_id)}
                            className="rounded-lg bg-cyan-700 px-2 py-1 text-[10px] font-semibold text-white"
                          >
                            保存
                          </button>
                          <button
                            type="button"
                            onClick={() => setEditingCaseId(null)}
                            className="rounded-lg px-1.5 py-1 text-[10px] text-slate-500 hover:bg-slate-100"
                          >
                            取消
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-start justify-between gap-2">
                          <button
                            type="button"
                            onClick={() => void handleOpenCase(item.case_id)}
                            className="min-w-0 flex-1 text-left"
                          >
                            <div className="truncate font-semibold text-slate-800">{item.title}</div>
                            <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[10px] text-slate-400">
                              <span>{formatShortDateTime(item.updated_at)}</span>
                              <span>· v{item.revision_count}</span>
                              {item.scale_count > 0 && <span>· {item.scale_count} 量表</span>}
                              {item.draft_modes_ready.length > 0 && (
                                <span className="rounded bg-cyan-100/80 px-1 text-cyan-800">
                                  {item.draft_modes_ready.length === 2
                                    ? "双文书就绪"
                                    : item.draft_modes_ready[0] === "ward_round"
                                      ? "查房稿"
                                      : "宣教稿"}
                                </span>
                              )}
                            </div>
                          </button>

                          <div className="flex shrink-0 flex-col items-end gap-1">
                            <span
                              className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                                item.critical_count > 0
                                  ? "bg-rose-100 text-rose-800"
                                  : item.alert_count > 0
                                    ? "bg-amber-100 text-amber-800"
                                    : "bg-emerald-100 text-emerald-800"
                              }`}
                            >
                              {item.alert_count} 项警示
                            </span>
                            <div className="flex items-center gap-1">
                              <button
                                type="button"
                                title="重命名"
                                onClick={() => {
                                  setEditingCaseId(item.case_id);
                                  setEditingCaseTitle(item.title);
                                  setConfirmDeleteCaseId(null);
                                }}
                                className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                              >
                                <Pencil className="h-3 w-3" />
                              </button>
                              <button
                                type="button"
                                title="导出 .json"
                                onClick={() => void handleExportCase(item.case_id)}
                                className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-cyan-700"
                              >
                                <Download className="h-3 w-3" />
                              </button>
                              {isConfirmDel ? (
                                <div className="flex items-center gap-1">
                                  <button
                                    type="button"
                                    onClick={() => void handleDeleteCase(item.case_id)}
                                    className="rounded bg-rose-600 px-1.5 py-0.5 text-[10px] font-semibold text-white"
                                  >
                                    确认
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => setConfirmDeleteCaseId(null)}
                                    className="rounded px-1 py-0.5 text-[10px] text-slate-500 hover:bg-slate-100"
                                  >
                                    取消
                                  </button>
                                </div>
                              ) : (
                                <button
                                  type="button"
                                  title="删除档案"
                                  onClick={() => setConfirmDeleteCaseId(item.case_id)}
                                  className="rounded p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600"
                                >
                                  <Trash2 className="h-3 w-3" />
                                </button>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </section>

        {/* Right 64% (8 cols on 12-col grid) */}
        <section className="space-y-4 lg:col-span-8">
          {!analysis ? (
            <div className="glass-surface flex min-h-[520px] flex-col items-center justify-center rounded-3xl p-8 text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-cyan-50 text-cyan-700 shadow-inner">
                <ShieldCheck className="h-7 w-7" />
              </div>
              <h2 className="mt-4 text-base font-bold text-slate-900">
                骨伤与康复双轨安全工作台已就绪
              </h2>
              <p className="mt-2 max-w-md text-xs leading-relaxed text-slate-500">
                在左侧选择一个演示病例或粘贴脱敏病历，点击「极速安全筛查」可零等待命中红旗与用药冲突；点击「开始完整分析」将同步生成跨诊次快照、确定性量表核算与受护栏约束的临床文书。
              </p>
            </div>
          ) : (
            <>
              {/* Compact Collapsible Liquid Glass Safety Redline Strip */}
              <div
                className={`rounded-3xl p-4 transition-all duration-200 ${
                  alerts.length === 0
                    ? "glass-safe"
                    : hasCriticalOrHigh
                      ? "glass-danger"
                      : "glass-warning"
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
                    {alerts.length === 0 ? (
                      <>
                        <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-700" />
                        <span className="text-sm font-semibold text-emerald-900">
                          安全扫描通过：未检出红旗征或高危中西医用药冲突
                        </span>
                      </>
                    ) : (
                      <>
                        <ShieldAlert
                          className={`h-5 w-5 shrink-0 ${
                            hasCriticalOrHigh ? "text-rose-700" : "text-amber-700"
                          }`}
                        />
                        <span
                          className={`text-sm font-bold ${
                            hasCriticalOrHigh ? "text-rose-900" : "text-amber-900"
                          }`}
                        >
                          触发 {alerts.length} 项临床安全警示
                        </span>
                        <div className="flex flex-wrap gap-1.5">
                          {capsules.map((cap) => (
                            <span
                              key={cap}
                              className="rounded-full border border-rose-200/90 bg-white/85 px-2.5 py-0.5 text-[11px] font-semibold text-rose-800 shadow-sm"
                            >
                              {cap}
                            </span>
                          ))}
                        </div>
                      </>
                    )}
                  </div>

                  {alerts.length > 0 && (
                    <button
                      type="button"
                      onClick={() => setAlertsExpanded((v) => !v)}
                      className="inline-flex items-center gap-1 rounded-full bg-white/85 px-3 py-1 text-xs font-semibold text-slate-700 shadow-sm transition-colors hover:bg-white"
                    >
                      {alertsExpanded ? (
                        <>
                          收起详情 <ChevronUp className="h-3.5 w-3.5" />
                        </>
                      ) : (
                        <>
                          展开证据与教材依据 <ChevronDown className="h-3.5 w-3.5" />
                        </>
                      )}
                    </button>
                  )}
                </div>

                {alertsExpanded && alerts.length > 0 && (
                  <div className="mt-4 space-y-3 border-t border-rose-200/50 pt-3">
                    {alerts.map((alert, idx) => {
                      const sev = SEVERITY_LABELS[alert.severity] || SEVERITY_LABELS.high;
                      const refs = alert.reference || alert.references || [];
                      return (
                        <div
                          key={`${alert.rule_id}-${idx}`}
                          className="clinical-paper rounded-2xl p-4 text-xs"
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div className="flex flex-wrap items-center gap-2">
                              <span
                                className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold ${
                                  sev.tone === "critical"
                                    ? "bg-rose-700 text-white"
                                    : sev.tone === "high"
                                      ? "bg-orange-100 text-orange-800"
                                      : "bg-amber-100 text-amber-800"
                                }`}
                              >
                                {sev.label}
                              </span>
                              <h4 className="text-sm font-bold text-slate-900">{alert.name}</h4>
                              {alert.category && (
                                <span className="rounded-md bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600">
                                  {alert.category}
                                </span>
                              )}
                            </div>
                            <span className="rounded-full bg-cyan-50 px-3 py-1 text-[11px] font-semibold text-cyan-900">
                              建议动作：{getClinicalActionLabel(alert.clinical_action, actionLabels)}
                            </span>
                          </div>

                          {alert.clinical_alert && (
                            <p className="mt-2.5 leading-relaxed text-slate-800">
                              {alert.clinical_alert}
                            </p>
                          )}

                          {alert.matched_evidence && alert.matched_evidence.length > 0 && (
                            <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                              <span className="font-semibold text-slate-500">命中证据：</span>
                              {alert.matched_evidence.map((ev, eIdx) => (
                                <span
                                  key={eIdx}
                                  className="rounded-md border border-rose-200 bg-rose-50 px-2 py-0.5 text-[11px] text-rose-900"
                                >
                                  {ev}
                                </span>
                              ))}
                            </div>
                          )}

                          {alert.contraindicated_actions &&
                            alert.contraindicated_actions.length > 0 && (
                              <div className="mt-2 text-rose-800">
                                <span className="font-semibold">绝对禁忌操作：</span>
                                {alert.contraindicated_actions.join("；")}
                              </div>
                            )}

                          {refs.length > 0 && (
                            <div className="mt-2 border-t border-slate-100 pt-2 text-[11px] text-slate-500">
                              <span className="font-semibold text-slate-600">指南/教材依据：</span>
                              {refs.join("；")}
                              <span className="ml-2 font-mono text-[10px] text-slate-400">
                                [{alert.rule_id}]
                              </span>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* 4 Clinical Tabs */}
              <div className="glass-surface rounded-3xl p-4 sm:p-5">
                <div className="mb-4 flex flex-wrap gap-1.5 rounded-2xl bg-slate-900/5 p-1">
                  {(
                    [
                      { key: "timeline", label: "时间线与快照", icon: Clock },
                      {
                        key: "scales",
                        label: `量表算分与补录 (${(analysis.scale_results || []).length})`,
                        icon: Calculator,
                      },
                      { key: "draft", label: "文书草稿与复检", icon: FileText },
                      { key: "trace", label: "审计轨迹", icon: ShieldCheck },
                    ] as const
                  ).map((t) => {
                    const Icon = t.icon;
                    const active = activeTab === t.key;
                    return (
                      <button
                        key={t.key}
                        type="button"
                        onClick={() => setActiveTab(t.key)}
                        className={`inline-flex flex-1 items-center justify-center gap-1.5 rounded-xl px-3 py-2 text-xs font-semibold transition-all ${
                          active
                            ? "bg-white text-cyan-800 shadow-sm"
                            : "text-slate-600 hover:text-slate-900"
                        }`}
                      >
                        <Icon className="h-3.5 w-3.5" />
                        <span className="whitespace-nowrap">{t.label}</span>
                      </button>
                    );
                  })}
                </div>

                {/* Tab 1: Timeline & Snapshot */}
                {activeTab === "timeline" && (
                  <TabTimelineSnapshot note={analysis.extracted_note || {}} />
                )}

                {/* Tab 2: Scales & Manual Backfill */}
                {activeTab === "scales" && (
                  <div className="space-y-4">
                    {currentDraftStale && (
                      <div className="glass-warning flex flex-wrap items-center justify-between gap-2 rounded-2xl px-4 py-3 text-xs text-amber-900">
                        <span>量表评估结果已更新，当前文书草稿尚未同步刷新。</span>
                        <button
                          type="button"
                          disabled={Boolean(loadingAction)}
                          onClick={() => void handleForceRegenerateDraft(activeDraftMode)}
                          className="inline-flex items-center gap-1.5 rounded-xl bg-amber-600 px-3 py-1.5 font-semibold text-white hover:bg-amber-700"
                        >
                          <RefreshCw className="h-3.5 w-3.5" />
                          依据最新量表刷新文书
                        </button>
                      </div>
                    )}

                    {(analysis.scale_results || []).length === 0 && (
                      <div className="clinical-paper rounded-2xl p-5 text-center text-xs text-slate-500">
                        病历中未提取到明确量表条目分值，可在下方「手动新增量表评估」添加 4 套骨伤/康复量表。
                      </div>
                    )}

                    {(analysis.scale_results || []).map((res, idx) => (
                      <ScaleResultCard
                        key={`${res.scale_id}-${idx}`}
                        result={res}
                        spec={scaleSpecMap.get(res.scale_id)}
                        isEditing={expandedScaleEditorIdx === idx || res.status !== "ok"}
                        busy={loadingAction === "scale"}
                        onToggleEdit={() =>
                          setExpandedScaleEditorIdx((cur) => (cur === idx ? null : idx))
                        }
                        onRemove={() => handleRemoveScale(idx)}
                        onRecalculate={(params, refreshDraft) =>
                          handleRecalculateScale(res.scale_id, params, idx, refreshDraft)
                        }
                      />
                    ))}

                    {/* Add New Scale */}
                    <div className="clinical-paper rounded-2xl p-4">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                          <Plus className="h-4 w-4 text-cyan-700" />
                          手动新增量表评估
                        </div>
                        {addableScales.length > 0 && (
                          <select
                            value={effectiveAddScaleId}
                            onChange={(e) => setAddScaleId(e.target.value)}
                            className="rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-800 focus:border-cyan-600 focus:outline-none"
                          >
                            {addableScales.map((s) => (
                              <option key={s.scale_id} value={s.scale_id}>
                                {s.name}
                              </option>
                            ))}
                          </select>
                        )}
                      </div>

                      {addableScales.length === 0 ? (
                        <p className="mt-2 text-xs text-slate-500">
                          全部 4 套标准骨伤/康复量表均已加入当前评估列表。
                        </p>
                      ) : selectedAddSpec ? (
                        <div className="mt-3">
                          <p className="mb-3 text-[11px] text-slate-500">
                            {selectedAddSpec.version_note}
                          </p>
                          <ScaleEditor
                            key={selectedAddSpec.scale_id}
                            spec={selectedAddSpec}
                            initialParams={{}}
                            busy={loadingAction === "scale"}
                            submitPrimaryLabel="计算并加入评估"
                            onSubmitPrimary={(params) =>
                              handleRecalculateScale(
                                selectedAddSpec.scale_id,
                                params,
                                undefined,
                                false
                              )
                            }
                          />
                        </div>
                      ) : null}
                    </div>
                  </div>
                )}

                {/* Tab 3: Draft & Verification */}
                {activeTab === "draft" && (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex flex-wrap items-center gap-2">
                        {(["ward_round", "rehab_education"] as DraftMode[]).map((mode) => {
                          const active = activeDraftMode === mode;
                          const hasCached = Boolean(analysis.drafts_by_mode?.[mode]);
                          const isStale = Boolean(analysis.draft_stale_by_mode?.[mode]);
                          return (
                            <button
                              key={mode}
                              type="button"
                              disabled={Boolean(loadingAction)}
                              onClick={() => void handleSwitchDraftMode(mode)}
                              className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-semibold transition-all ${
                                active
                                  ? "border-cyan-700 bg-cyan-700 text-white shadow-sm"
                                  : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                              }`}
                            >
                              <span>{DRAFT_MODE_LABELS[mode]}</span>
                              {hasCached && (
                                <span
                                  className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${
                                    isStale
                                      ? "bg-amber-100 text-amber-800"
                                      : active
                                        ? "bg-white/20 text-white"
                                        : "bg-emerald-100 text-emerald-800"
                                  }`}
                                >
                                  {isStale ? "待刷新" : "已生成"}
                                </span>
                              )}
                            </button>
                          );
                        })}
                        <button
                          type="button"
                          disabled={Boolean(loadingAction)}
                          onClick={() => void handleForceRegenerateDraft(activeDraftMode)}
                          className="inline-flex items-center gap-1.5 rounded-xl border border-cyan-200 bg-cyan-50/80 px-3 py-1.5 text-xs font-semibold text-cyan-800 hover:bg-cyan-100/80"
                        >
                          <RefreshCw
                            className={`h-3.5 w-3.5 ${
                              loadingAction === "draft" ? "animate-spin" : ""
                            }`}
                          />
                          🔄 重新生成当前文书
                        </button>
                      </div>

                      <div className="flex flex-wrap items-center gap-2">
                        {analysis.draft_cache_hit && (
                          <span className="rounded-full bg-cyan-50 px-2.5 py-0.5 text-[11px] font-medium text-cyan-800">
                            ⚡ 缓存秒切
                          </span>
                        )}
                        {/* Verification Badge */}
                        {analysis.draft?.blocked ? (
                          <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-100 px-3 py-1 text-xs font-bold text-rose-800">
                            <ShieldAlert className="h-3.5 w-3.5" />
                            复检拦截：违背安全禁忌
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-800">
                            <ShieldCheck className="h-3.5 w-3.5" />
                            确定性输出复检通过
                          </span>
                        )}
                      </div>
                    </div>

                    {currentDraftStale && (
                      <div className="glass-warning flex flex-wrap items-center justify-between gap-2 rounded-2xl px-4 py-3 text-xs text-amber-900">
                        <span>
                          量表评估结果已更新，当前模式「{DRAFT_MODE_LABELS[activeDraftMode] || activeDraftMode}」文书草稿尚未同步刷新。
                        </span>
                        <button
                          type="button"
                          disabled={Boolean(loadingAction)}
                          onClick={() => void handleForceRegenerateDraft(activeDraftMode)}
                          className="inline-flex items-center gap-1.5 rounded-xl bg-amber-600 px-3 py-1.5 font-semibold text-white hover:bg-amber-700"
                        >
                          <RefreshCw className="h-3.5 w-3.5" />
                          一键重新生成当前文书
                        </button>
                      </div>
                    )}

                    {analysis.draft?.blocked ? (
                      <div className="glass-danger space-y-2 rounded-2xl p-4 text-xs text-rose-900">
                        <div className="font-bold">
                          安全拦截：生成的文书草稿涉嫌违背当前激活的红旗/禁忌约束，已阻止渲染！
                        </div>
                        {(analysis.draft.safety_check?.violations || []).map((v, i) => (
                          <div
                            key={i}
                            className="rounded-xl border border-rose-200 bg-white/80 p-2.5 text-rose-900"
                          >
                            <div className="font-semibold">
                              违禁动作：{v.matched_action}（关联规则：{v.rule_id}）
                            </div>
                            {v.snippet && <div className="mt-1 text-slate-600">上下文：{v.snippet}</div>}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <>
                        <div className="clinical-paper rounded-2xl p-5">
                          <MarkdownView content={analysis.draft?.content || "暂无文书内容。"} />
                        </div>

                        <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-white/80 bg-white/75 p-3">
                          <div className="flex flex-wrap gap-2">
                            <button
                              type="button"
                              onClick={() => handleCopyPlainOrHandover("plain")}
                              className="inline-flex items-center gap-1.5 rounded-xl bg-cyan-700 px-3.5 py-2 text-xs font-semibold text-white hover:bg-cyan-800"
                            >
                              <Copy className="h-3.5 w-3.5" />
                              复制纯文本（贴入病历/微信）
                            </button>
                            <button
                              type="button"
                              onClick={() => handleCopyPlainOrHandover("handover")}
                              className="inline-flex items-center gap-1.5 rounded-xl border border-cyan-200 bg-cyan-50 px-3.5 py-2 text-xs font-semibold text-cyan-900 hover:bg-cyan-100"
                            >
                              <Copy className="h-3.5 w-3.5" />
                              复制完整交班摘要
                            </button>
                            <button
                              type="button"
                              onClick={() => handleCopyPlainOrHandover("markdown")}
                              className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50"
                            >
                              复制 Markdown
                            </button>
                          </div>

                          <div className="flex flex-wrap gap-1.5">
                            <button
                              type="button"
                              onClick={() => handleDownloadDraft("plain")}
                              className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-medium text-slate-600 hover:bg-slate-50"
                            >
                              <Download className="h-3 w-3" /> .txt
                            </button>
                            <button
                              type="button"
                              onClick={() => handleDownloadDraft("markdown")}
                              className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-medium text-slate-600 hover:bg-slate-50"
                            >
                              <Download className="h-3 w-3" /> .md
                            </button>
                            <button
                              type="button"
                              onClick={() => handleDownloadDraft("handover")}
                              className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-medium text-slate-600 hover:bg-slate-50"
                            >
                              <Download className="h-3 w-3" /> 交班单
                            </button>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                )}

                {/* Tab 4: Audit Trace */}
                {activeTab === "trace" && (
                  <div className="space-y-4 text-xs">
                    <div className="clinical-paper grid grid-cols-2 gap-3 rounded-2xl p-4 sm:grid-cols-4">
                      <div>
                        <div className="text-slate-400">结构化抽取模型</div>
                        <div className="mt-0.5 font-mono font-semibold text-slate-800">
                          {String(analysis.trace?.extraction_model || "-")}
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-400">文书生成模型</div>
                        <div className="mt-0.5 font-mono font-semibold text-slate-800">
                          {String(analysis.trace?.draft_model || "-")}
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-400">运行模式</div>
                        <div className="mt-0.5 font-semibold text-slate-800">
                          {analysis.fast_scan_only
                            ? "极速规则扫描"
                            : analysis.trace?.draft_mock
                              ? "Mock 确定性仿真"
                              : "Live 真实模型"}
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-400">抑制误报数</div>
                        <div className="mt-0.5 font-semibold text-emerald-700">
                          {suppressedMatches.length} 条（否定词/洗脱期过滤）
                        </div>
                      </div>
                    </div>

                    {/* Suppressed matches table */}
                    <div className="clinical-paper overflow-hidden rounded-2xl">
                      <button
                        type="button"
                        onClick={() => setSuppressedExpanded((v) => !v)}
                        className="flex w-full items-center justify-between px-4 py-3 text-left font-semibold text-slate-800 hover:bg-slate-50"
                      >
                        <span>
                          否定词与药物洗脱期防误报审计记录（{suppressedMatches.length} 条）
                        </span>
                        {suppressedExpanded ? (
                          <ChevronUp className="h-4 w-4 text-slate-400" />
                        ) : (
                          <ChevronDown className="h-4 w-4 text-slate-400" />
                        )}
                      </button>
                      {suppressedExpanded && (
                        <div className="border-t border-slate-100 p-4">
                          {suppressedMatches.length === 0 ? (
                            <p className="text-slate-400">本次未触发否定词或洗脱期抑制条目。</p>
                          ) : (
                            <div className="overflow-x-auto">
                              <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                                <thead className="bg-slate-50">
                                  <tr>
                                    <th className="px-3 py-2 font-semibold text-slate-600">规则 ID</th>
                                    <th className="px-3 py-2 font-semibold text-slate-600">命中词</th>
                                    <th className="px-3 py-2 font-semibold text-slate-600">拦截原因</th>
                                    <th className="px-3 py-2 font-semibold text-slate-600">来源轨道</th>
                                    <th className="px-3 py-2 font-semibold text-slate-600">详情</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                  {suppressedMatches.map((m, i) => (
                                    <tr key={i}>
                                      <td className="px-3 py-2 font-mono text-slate-700">
                                        {m.rule_id || "-"}
                                      </td>
                                      <td className="px-3 py-2 font-medium text-slate-900">
                                        {m.term || "-"}
                                      </td>
                                      <td className="px-3 py-2 text-emerald-700">
                                        {m.reason || "-"}
                                      </td>
                                      <td className="px-3 py-2 text-slate-600">
                                        {SOURCE_LABEL_ZH[String(m.source || "")] || m.source || "-"}
                                      </td>
                                      <td className="px-3 py-2 text-slate-500">{m.detail || "-"}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Raw JSON Inspection */}
                    <details className="clinical-paper rounded-2xl p-4">
                      <summary className="cursor-pointer font-semibold text-slate-800">
                        结构化提取详情（extracted_note JSON）
                      </summary>
                      <pre className="mt-3 max-h-72 overflow-auto rounded-xl bg-slate-950 p-3 font-mono text-[11px] text-slate-100">
                        {JSON.stringify(analysis.extracted_note || {}, null, 2)}
                      </pre>
                    </details>

                    <details className="clinical-paper rounded-2xl p-4">
                      <summary className="cursor-pointer font-semibold text-slate-800">
                        双轨安全扫描详情（risk_report JSON）
                      </summary>
                      <pre className="mt-3 max-h-72 overflow-auto rounded-xl bg-slate-950 p-3 font-mono text-[11px] text-slate-100">
                        {JSON.stringify(analysis.risk_report || {}, null, 2)}
                      </pre>
                    </details>
                  </div>
                )}
              </div>
            </>
          )}
        </section>
      </div>

      {/* Follow-up Visit Modal */}
      {followupModalOpen && activeCase && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/35 p-4 backdrop-blur-sm">
          <div className="glass-surface w-full max-w-lg space-y-4 rounded-3xl p-5 shadow-2xl">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-900">➕ 追加跨诊次复诊记录</h3>
                <p className="mt-0.5 text-[11px] text-slate-500">
                  当前档案：{activeCase.title}（{activeCase.case_id}）
                </p>
              </div>
              <button
                type="button"
                onClick={() => setFollowupModalOpen(false)}
                className="rounded-xl p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="mb-1 block font-semibold text-slate-700">复诊日期</label>
                <input
                  type="text"
                  value={followupDate}
                  onChange={(e) => setFollowupDate(e.target.value)}
                  placeholder="例如：2026-11-27"
                  className="clinical-input text-xs"
                />
              </div>
              <div>
                <label className="mb-1 block font-semibold text-slate-700">
                  本次复诊情况 / 用药变更 / 专科查体（仅用脱敏描述）
                </label>
                <textarea
                  rows={5}
                  value={followupText}
                  onChange={(e) => setFollowupText(e.target.value)}
                  placeholder="例如：华法林已停药 10 天；复查行走步态较前好转，但仍偶有双手麻木..."
                  className="clinical-input resize-y text-xs leading-relaxed"
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-end gap-2 pt-1">
              <button
                type="button"
                onClick={() => setFollowupModalOpen(false)}
                className="rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-medium text-slate-600 hover:bg-slate-50"
              >
                取消
              </button>
              <button
                type="button"
                disabled={!followupText.trim() || Boolean(loadingAction)}
                onClick={() => void handleSubmitFollowup(false)}
                className="rounded-xl border border-cyan-200 bg-cyan-50 px-3.5 py-2 text-xs font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
              >
                {loadingAction === "followup" ? "保存中..." : "仅追加保存"}
              </button>
              <button
                type="button"
                disabled={!followupText.trim() || Boolean(loadingAction)}
                onClick={() => void handleSubmitFollowup(true)}
                className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-cyan-700 to-teal-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:brightness-105 disabled:opacity-50"
              >
                <Sparkles className="h-3.5 w-3.5" />
                追加并立即重新分析
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Case Revisions History Modal */}
      {revisionsModalOpen && activeCase && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/35 p-4 backdrop-blur-sm">
          <div className="glass-surface flex max-h-[80vh] w-full max-w-xl flex-col space-y-4 rounded-3xl p-5 shadow-2xl">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-900">
                  📜 档案历史版本 ({activeCase.revisions?.length || 0})
                </h3>
                <p className="mt-0.5 text-[11px] text-slate-500">
                  {activeCase.title}（{activeCase.case_id}）· 支持一键恢复任意历史快照
                </p>
              </div>
              <button
                type="button"
                onClick={() => setRevisionsModalOpen(false)}
                className="rounded-xl p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="space-y-2 overflow-y-auto pr-1 text-xs">
              {(activeCase.revisions || []).length === 0 ? (
                <p className="py-6 text-center text-slate-400">暂无历史版本记录。</p>
              ) : (
                [...(activeCase.revisions || [])]
                  .slice()
                  .reverse()
                  .map((revItem) => (
                    <div
                      key={revItem.rev}
                      className="clinical-paper flex flex-wrap items-center justify-between gap-3 rounded-2xl p-3"
                    >
                      <div className="min-w-0 flex-1 space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="rounded-md bg-cyan-100 px-2 py-0.5 font-mono text-[11px] font-bold text-cyan-900">
                            v{revItem.rev}
                          </span>
                          <span className="font-semibold text-slate-800">
                            {revItem.note || "临床分析快照"}
                          </span>
                          <span className="text-[11px] text-slate-400">
                            {formatShortDateTime(revItem.timestamp)}
                          </span>
                        </div>
                        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                          <span>警示数：{revItem.alert_count ?? 0}</span>
                          {revItem.draft_modes_ready && revItem.draft_modes_ready.length > 0 && (
                            <span>
                              · 文书：{revItem.draft_modes_ready.map((m) => DRAFT_MODE_LABELS[m as DraftMode] || m).join(" / ")}
                            </span>
                          )}
                        </div>
                      </div>
                      <button
                        type="button"
                        disabled={Boolean(loadingAction)}
                        onClick={() => void handleRestoreRevision(revItem.rev)}
                        className="shrink-0 rounded-xl border border-cyan-200 bg-cyan-50 px-3 py-1.5 text-xs font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
                      >
                        {loadingAction === `restore-${revItem.rev}` ? "恢复中..." : "恢复此版本"}
                      </button>
                    </div>
                  ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ---------------- Sub-components ----------------

const TabTimelineSnapshot: React.FC<{ note: CopilotAnalysisResponse["extracted_note"] }> = ({
  note,
}) => {
  const tcm = note.tcm_four_diagnostics || {};
  const tcmEntries = Object.entries(tcm).filter(([, v]) => v && String(v).trim());
  const timeline = note.timeline_events || [];
  const snapshot = note.current_state_snapshot || {};
  const facts = note.structured_facts || {};

  return (
    <div className="space-y-4 text-xs">
      {/* Core Summary & TCM */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="clinical-paper space-y-2.5 rounded-2xl p-4">
          <h4 className="text-sm font-bold text-slate-900">患者画像与主诉摘要</h4>
          <div>
            <span className="font-semibold text-slate-500">脱敏画像：</span>
            <span className="text-slate-800">{note.patient_summary || "未提及"}</span>
          </div>
          <div>
            <span className="font-semibold text-slate-500">主诉：</span>
            <span className="font-medium text-slate-900">{note.chief_complaint || "未提及"}</span>
          </div>
          <div>
            <span className="font-semibold text-slate-500">现病史：</span>
            <span className="leading-relaxed text-slate-700">
              {note.history_of_present_illness || "未提及"}
            </span>
          </div>
          <div className="rounded-xl bg-cyan-50/75 p-2.5 text-cyan-950">
            <span className="font-semibold">影像学 / 检验摘要：</span>
            {note.imaging_summary || "未提及"}
          </div>
        </div>

        <div className="clinical-paper space-y-2.5 rounded-2xl p-4">
          <h4 className="text-sm font-bold text-slate-900">中医四诊与专科查体</h4>
          {tcmEntries.length > 0 ? (
            <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
              {tcmEntries.map(([k, v]) => (
                <div key={k} className="rounded-xl bg-slate-50/90 px-2.5 py-1.5">
                  <span className="font-semibold text-cyan-800">
                    {TCM_FIELD_LABELS[k] || k}：
                  </span>
                  <span className="text-slate-700">{String(v)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-slate-400">未提取到明确中医四诊信息。</p>
          )}

          <div className="pt-1">
            <div className="mb-1 font-semibold text-slate-500">专科查体要点</div>
            {(note.physical_exam || []).length > 0 ? (
              <ul className="list-disc space-y-1 pl-4 text-slate-700">
                {(note.physical_exam || []).map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-400">暂无专科查体记录。</p>
            )}
          </div>
        </div>
      </div>

      {/* Longitudinal Timeline */}
      <div className="clinical-paper rounded-2xl p-4">
        <h4 className="mb-3 text-sm font-bold text-slate-900">跨诊次纵向演变时间线</h4>
        {timeline.length === 0 ? (
          <p className="text-slate-400">暂无跨诊次时间线事件。</p>
        ) : (
          <div className="space-y-3 border-l-2 border-cyan-200 pl-4">
            {timeline.map((ev, idx) => (
              <div key={idx} className="relative rounded-xl border border-slate-100 bg-slate-50/70 p-3">
                <span className="absolute -left-[22px] top-4 h-2.5 w-2.5 rounded-full border-2 border-white bg-cyan-600 shadow" />
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-md bg-cyan-100 px-2 py-0.5 font-mono text-[11px] font-semibold text-cyan-900">
                    {ev.date || "未标明时间"}
                  </span>
                  <span className="font-semibold text-slate-700">
                    {ev.department || "未标明科室"}
                  </span>
                </div>
                <p className="mt-1.5 text-slate-800">{ev.event}</p>
                {ev.key_change && (
                  <p className="mt-1 text-[11px] font-medium text-amber-800">
                    关键转折：{ev.key_change}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Current State Snapshot */}
      <div className="clinical-paper space-y-3 rounded-2xl p-4">
        <h4 className="text-sm font-bold text-slate-900">当前状态快照（时间线冲突消解）</h4>
        {snapshot.diagnosis_override_note && (
          <div className="rounded-xl border border-cyan-200 bg-cyan-50/80 px-3 py-2 text-cyan-950">
            <span className="font-semibold">最新状态修正提示：</span>
            {snapshot.diagnosis_override_note}
          </div>
        )}

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <div className="rounded-xl border border-slate-200/80 p-3">
            <div className="mb-2 font-semibold text-slate-800">当前在用 / 洗脱期内药物</div>
            {(snapshot.active_medications || []).length === 0 ? (
              <p className="text-slate-400">当前无高风险在用或洗脱期内药物。</p>
            ) : (
              <div className="space-y-1.5">
                {(snapshot.active_medications || []).map((m, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between rounded-lg bg-rose-50/70 px-2.5 py-1.5"
                  >
                    <span className="font-medium text-slate-900">{m.name}</span>
                    <span className="text-[11px] text-rose-800">
                      {formatActiveMedStopDisplay(m)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-xl border border-slate-200/80 p-3">
            <div className="mb-2 font-semibold text-slate-800">已停用且过洗脱期药物</div>
            {(snapshot.stopped_medications || []).length === 0 ? (
              <p className="text-slate-400">暂无已过洗脱期的停药记录。</p>
            ) : (
              <div className="space-y-1.5">
                {(snapshot.stopped_medications || []).map((m, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between rounded-lg bg-emerald-50/70 px-2.5 py-1.5"
                  >
                    <span className="font-medium text-slate-900">{m.name}</span>
                    <span className="text-[11px] text-emerald-800">
                      {m.stopped_days_ago != null ? `已停药 ${m.stopped_days_ago} 天` : "已停用"}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {(snapshot.active_surgical_history || []).length > 0 && (
          <div>
            <div className="mb-1.5 font-semibold text-slate-700">长期手术 / 内固定约束</div>
            <div className="flex flex-wrap gap-1.5">
              {(snapshot.active_surgical_history || []).map((sh, i) => (
                <span
                  key={i}
                  className="rounded-full border border-slate-200 bg-slate-100 px-2.5 py-1 text-[11px] font-medium text-slate-800"
                >
                  {sh}
                </span>
              ))}
            </div>
          </div>
        )}

        {(snapshot.superseded_conditions || []).length > 0 && (
          <div>
            <div className="mb-1.5 font-semibold text-slate-700">已被新诊次覆盖的旧状态</div>
            <div className="space-y-1">
              {(snapshot.superseded_conditions || []).map((s, i) => (
                <div key={i} className="rounded-lg bg-slate-50 px-3 py-1.5 text-slate-700">
                  <span className="line-through text-slate-400">{s.old_state}</span>
                  <span className="mx-2 text-cyan-700">→</span>
                  <span className="font-medium text-slate-900">{s.new_state}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Structured symptoms & meds summary */}
        {((facts.symptoms || []).length > 0 || (facts.medications || []).length > 0) && (
          <div className="grid grid-cols-1 gap-3 border-t border-slate-100 pt-3 md:grid-cols-2">
            <div>
              <div className="mb-1.5 font-semibold text-slate-600">结构化症状极性</div>
              <div className="flex flex-wrap gap-1.5">
                {(facts.symptoms || []).map((s, i) => (
                  <span
                    key={i}
                    className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium ${
                      s.polarity === "present"
                        ? "bg-rose-50 text-rose-800"
                        : s.polarity === "absent"
                          ? "bg-emerald-50 text-emerald-800"
                          : "bg-slate-100 text-slate-700"
                    }`}
                  >
                    {s.name} · {POLARITY_ZH[s.polarity] || s.polarity}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-1.5 font-semibold text-slate-600">结构化用药清单</div>
              <div className="flex flex-wrap gap-1.5">
                {(facts.medications || []).map((m, i) => (
                  <span
                    key={i}
                    className="rounded-full bg-slate-100 px-2.5 py-0.5 text-[11px] font-medium text-slate-700"
                  >
                    {m.name} · {MED_STATUS_ZH[m.status] || m.status}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const ScaleResultCard: React.FC<{
  result: ScaleResult;
  spec?: ScaleSpec;
  isEditing: boolean;
  busy: boolean;
  onToggleEdit: () => void;
  onRemove: () => void;
  onRecalculate: (params: Record<string, number | null>, refreshDraft: boolean) => void;
}> = ({ result, spec, isEditing, busy, onToggleEdit, onRemove, onRecalculate }) => {
  const isOk = result.status === "ok";
  const itemLabelMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const it of spec?.items || []) map.set(it.item_id, it.label);
    return map;
  }, [spec]);

  return (
    <div className="clinical-paper space-y-3 rounded-2xl p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="text-sm font-bold text-slate-900">
              {result.scale_name || result.scale_id}
            </h4>
            <span className="rounded-md bg-slate-100 px-2 py-0.5 font-mono text-[10px] text-slate-500">
              {result.scale_id}
            </span>
            {!isOk && (
              <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-[11px] font-semibold text-amber-800">
                待补录或修正
              </span>
            )}
          </div>
          <p className="mt-1 text-[11px] text-slate-500">
            {getScoreDirectionLabel(result.score_direction)}
          </p>
        </div>

        <div className="flex items-center gap-1.5">
          {isOk && spec && (
            <button
              type="button"
              onClick={onToggleEdit}
              className="rounded-xl border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
            >
              {isEditing ? "收起编辑" : "核对/修改条目"}
            </button>
          )}
          <button
            type="button"
            disabled={busy}
            onClick={onRemove}
            className="rounded-xl p-1.5 text-slate-400 transition-colors hover:bg-rose-50 hover:text-rose-600"
            title="移除该量表"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      {isOk ? (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <div className="rounded-xl bg-slate-50/90 p-3">
              <div className="text-[11px] text-slate-500">确定性总分</div>
              <div className="mt-0.5 font-mono text-lg font-bold text-slate-900">
                {result.total_score} <span className="text-xs font-normal text-slate-400">/ {result.max_total}</span>
              </div>
            </div>
            <div className="rounded-xl bg-slate-50/90 p-3">
              <div className="text-[11px] text-slate-500">严重度分级</div>
              <div className="mt-0.5 text-sm font-bold text-cyan-800">{result.severity_band}</div>
            </div>
            <div className="rounded-xl bg-slate-50/90 p-3">
              <div className="text-[11px] text-slate-500">
                {result.normalized_score_pct !== undefined ? "0-100 标准化分" : "理论分值区间"}
              </div>
              <div className="mt-0.5 font-mono text-sm font-bold text-slate-800">
                {result.normalized_score_pct !== undefined
                  ? `${result.normalized_score_pct}%`
                  : `${result.min_total} ~ ${result.max_total}`}
              </div>
            </div>
          </div>

          {result.subscale_scores && Object.keys(result.subscale_scores).length > 0 && (
            <div className="overflow-x-auto rounded-xl border border-slate-100">
              <table className="min-w-full divide-y divide-slate-100 text-xs">
                <thead className="bg-slate-50/80">
                  <tr>
                    <th className="px-3 py-2 text-left font-semibold text-slate-600">子维度</th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-600">得分 / 满分</th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-600">标准化</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {Object.entries(result.subscale_scores).map(([sid, sub]) => (
                    <tr key={sid}>
                      <td className="px-3 py-1.5 text-slate-800">{sub.label}</td>
                      <td className="px-3 py-1.5 font-mono text-slate-900">
                        {sub.score} / {sub.max_score}
                      </td>
                      <td className="px-3 py-1.5 font-mono text-slate-600">
                        {sub.normalized_pct !== undefined ? `${sub.normalized_pct}%` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {result.improvement_rate && (
            <div
              className={`rounded-xl px-3 py-2 text-xs ${
                result.improvement_rate.status === "ok"
                  ? "bg-emerald-50 text-emerald-900"
                  : "bg-amber-50 text-amber-900"
              }`}
            >
              {result.improvement_rate.status === "ok" ? (
                <span>
                  平林法 JOA 改善率：<strong>{result.improvement_rate.improvement_rate_pct}%</strong>（
                  {result.improvement_rate.efficacy_grade}） · 公式{" "}
                  <code className="font-mono">{result.improvement_rate.formula}</code>
                </span>
              ) : (
                <span>改善率提示：{result.improvement_rate.message}</span>
              )}
            </div>
          )}

          {result.red_flag_triggered && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-medium text-rose-900">
              量表阈值联动红旗：{result.red_flag_reason}
            </div>
          )}
        </>
      ) : (
        <div className="rounded-xl border border-amber-200 bg-amber-50/80 p-3 text-xs text-amber-900">
          <div className="font-semibold">{result.message}</div>
          {(result.missing_fields || []).length > 0 && (
            <div className="mt-1 text-amber-800">
              缺失必填项（{(result.missing_fields || []).length} 项）：
              {(result.missing_fields || [])
                .slice(0, 6)
                .map((id) => itemLabelMap.get(id) || id)
                .join("、")}
              {(result.missing_fields || []).length > 6 && " 等"}
            </div>
          )}
        </div>
      )}

      {isEditing && spec && (
        <div className="border-t border-slate-100 pt-3">
          <ScaleEditor
            key={`${result.scale_id}-${JSON.stringify(result.input_params || result.item_scores || {})}`}
            spec={spec}
            initialParams={result.input_params || result.item_scores || {}}
            busy={busy}
            submitPrimaryLabel="仅重算量表"
            submitSecondaryLabel="重算并同步刷新文书"
            onSubmitPrimary={(params) => onRecalculate(params, false)}
            onSubmitSecondary={(params) => onRecalculate(params, true)}
          />
        </div>
      )}
    </div>
  );
};
