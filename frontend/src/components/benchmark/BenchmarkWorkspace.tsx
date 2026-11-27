import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Filter,
  Loader2,
  Play,
  Search,
  ShieldCheck,
  Square,
  XCircle,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../../api/client";
import type {
  BenchmarkProgressEvent,
  BenchmarkReportDetail,
  BenchmarkReportMeta,
  BenchmarkRun,
  ModelCapability,
  SideBySideCase,
  ThinkingLevel,
} from "../../api/types";
import {
  BENCHMARK_STATUS_LABELS,
  DIMENSION_LABELS,
  DIMENSION_ORDER,
} from "../../utils/labels";
import { ModelThinkingPicker } from "../ModelThinkingPicker";
import { CaseDetailDrawer } from "./CaseDetailDrawer";
import { FreeRedTeamPanel, FreeRedTeamRound } from "./FreeRedTeamPanel";
import { SandboxSidebar, SandboxSubMode } from "./SandboxSidebar";

interface BenchmarkWorkspaceProps {
  availableModels: string[];
  defaultModel: string;
  modelCapabilities: Record<string, ModelCapability>;
  defaultBenchmarkModels: string[];
  defaultMockMode: boolean;
  onRenderSidebarSlot?: (node: React.ReactNode) => void;
  onRenderHeaderActions?: (node: React.ReactNode) => void;
}

interface LiveCaseProgressItem {
  key: string;
  caseId: string;
  model: string;
  mode: string;
  passed: boolean;
}

const SERIES_COLORS = [
  "#0891b2",
  "#059669",
  "#d97706",
  "#be123c",
  "#4f46e5",
  "#0e7490",
  "#7c3aed",
  "#dc2626",
];

function getDimRate(run: BenchmarkRun, dimKey: string): number {
  const entry = run.summary?.dimensions?.[dimKey];
  if (!entry || typeof entry.pass_rate !== "number") return 0;
  return Number(entry.pass_rate.toFixed(1));
}

function formatDelta(
  val: number | undefined,
  invertGood = false
): { text: string; positive: boolean } {
  const num = typeof val === "number" && Number.isFinite(val) ? val : 0;
  const sign = num > 0 ? "+" : "";
  const good = invertGood ? num <= 0 : num >= 0;
  return { text: `${sign}${num.toFixed(1)}%`, positive: good };
}

export const BenchmarkWorkspace: React.FC<BenchmarkWorkspaceProps> = ({
  availableModels,
  defaultModel,
  modelCapabilities,
  defaultBenchmarkModels,
  defaultMockMode,
  onRenderSidebarSlot,
  onRenderHeaderActions,
}) => {
  const [subMode, setSubMode] = useState<SandboxSubMode>("batch");

  // Batch report state
  const [reports, setReports] = useState<BenchmarkReportMeta[]>([]);
  const [selectedFilename, setSelectedFilename] = useState<string>("");
  const [report, setReport] = useState<BenchmarkReportDetail | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Selected pair & filters & drawer
  const [activePairIdx, setActivePairIdx] = useState(0);
  const [filterMode, setFilterMode] = useState<"all" | "rescued" | "failed">("all");
  const [filterDim, setFilterDim] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCaseKey, setSelectedCaseKey] = useState<string>("");
  const [drawerOpen, setDrawerOpen] = useState(true);

  // Batch run configuration bar
  const [runModels, setRunModels] = useState<string[]>(() => {
    if (defaultBenchmarkModels.length > 0) return defaultBenchmarkModels;
    return defaultModel ? [defaultModel] : ["deepseek/deepseek-chat"];
  });
  const [modelThinking, setModelThinking] = useState<Record<string, ThinkingLevel>>({});
  const [runModes, setRunModes] = useState<Array<"bare_llm" | "harness_copilot">>([
    "bare_llm",
    "harness_copilot",
  ]);
  const [runMockMode, setRunMockMode] = useState<boolean>(defaultMockMode);
  const [includePrivate, setIncludePrivate] = useState<boolean>(false);

  // Live batch progress
  const [runningJob, setRunningJob] = useState(false);
  const [jobProgress, setJobProgress] = useState<BenchmarkProgressEvent | null>(null);
  const [liveCaseItems, setLiveCaseItems] = useState<LiveCaseProgressItem[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  // Free Red-Team state
  const [freeRounds, setFreeRounds] = useState<FreeRedTeamRound[]>([]);
  const [saveCaseModalOpen, setSaveCaseModalOpen] = useState(false);

  const modelChoices = useMemo(() => {
    return Array.from(
      new Set([...availableModels, ...defaultBenchmarkModels, ...runModels].filter(Boolean))
    );
  }, [availableModels, defaultBenchmarkModels, runModels]);

  const fetchReports = useCallback(
    async (preferredFilename?: string) => {
      setLoadingList(true);
      setErrorMsg(null);
      try {
        const res = await api.listBenchmarkReports();
        const list = res.reports || [];
        setReports(list);
        setSelectedFilename((prev) => {
          if (preferredFilename && list.some((r) => r.filename === preferredFilename)) {
            return preferredFilename;
          }
          if (prev && list.some((r) => r.filename === prev)) {
            return prev;
          }
          return list[0]?.filename || "";
        });
        if (list.length === 0) {
          setReport(null);
        }
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : "加载评测报告列表失败");
      } finally {
        setLoadingList(false);
      }
    },
    []
  );

  useEffect(() => {
    void fetchReports();
  }, [fetchReports]);

  useEffect(() => {
    if (!selectedFilename) return;
    let active = true;
    setLoadingDetail(true);
    setErrorMsg(null);
    api
      .getBenchmarkReport(selectedFilename)
      .then((detail) => {
        if (!active) return;
        setReport(detail);
        setActivePairIdx(0);
        const firstCase = detail.side_by_side_cases?.[0];
        if (firstCase) {
          setSelectedCaseKey(`${firstCase.pair_label}::${firstCase.case_id}`);
        } else {
          setSelectedCaseKey("");
        }
      })
      .catch((err) => {
        if (active) setErrorMsg(err instanceof Error ? err.message : "读取评测报告详情失败");
      })
      .finally(() => {
        if (active) setLoadingDetail(false);
      });
    return () => {
      active = false;
    };
  }, [selectedFilename]);

  const handleImportReport = useCallback(
    async (file: File) => {
      setErrorMsg(null);
      try {
        const res = await api.importBenchmarkReport(file);
        await fetchReports(res.filename);
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : "导入 JSON 评测报告失败");
      }
    },
    [fetchReports]
  );

  const handleStartBenchmark = async () => {
    if (runModels.length === 0) {
      setErrorMsg("请至少选择一个评测模型。");
      return;
    }
    if (runModes.length === 0) {
      setErrorMsg("请至少选择一个评测模式（裸模型 / 护栏协同）。");
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setErrorMsg(null);
    setRunningJob(true);
    setLiveCaseItems([]);
    setJobProgress({ progress: 2, message: "正在初始化红队评测任务..." });

    try {
      const selectedThinking: Record<string, ThinkingLevel> = {};
      runModels.forEach((m) => {
        const cap = modelCapabilities[m];
        const level = modelThinking[m] || "off";
        selectedThinking[m] =
          !runMockMode && cap && !cap.supports_thinking ? "off" : level;
      });
      const { job_id } = await api.startBenchmarkRun({
        models: runModels,
        modes: runModes,
        mock_mode: runMockMode,
        include_private: includePrivate,
        model_thinking: selectedThinking,
      });

      await api.streamBenchmarkRun(
        job_id,
        {
          onProgress: (ev) => {
            const pct =
              typeof ev.progress === "number"
                ? ev.progress
                : ev.total && ev.total > 0
                  ? Math.round(((ev.completed ?? 0) / ev.total) * 100)
                  : 0;
            const summaryLine =
              ev.message ||
              (ev.case_id
                ? `[${ev.completed ?? 0}/${ev.total ?? "?"}] ${ev.model || ""} · ${ev.mode || ""} · ${ev.case_id} ${ev.passed ? "通过" : "未通过"}`
                : "评测执行中...");
            setJobProgress({ ...ev, progress: pct, message: summaryLine });
            if (ev.case_id) {
              const itemKey = `${ev.model || ""}-${ev.mode || ""}-${ev.case_id}-${ev.completed ?? ""}`;
              setLiveCaseItems((prev) => [
                ...prev.slice(-79),
                {
                  key: itemKey,
                  caseId: String(ev.case_id),
                  model: String(ev.model || ""),
                  mode: String(ev.mode || ""),
                  passed: Boolean(ev.passed),
                },
              ]);
            }
          },
          onDone: async (ev) => {
            setJobProgress({ ...ev, progress: 100, message: "评测已完成，正在刷新报告..." });
            setRunningJob(false);
            await fetchReports(ev.filename);
            if (ev.filename) {
              setSelectedFilename(ev.filename);
            }
          },
          onError: (msg) => {
            setRunningJob(false);
            setErrorMsg(`评测执行失败：${msg}`);
          },
        },
        controller.signal
      );
    } catch (err) {
      if (!controller.signal.aborted) {
        setErrorMsg(err instanceof Error ? err.message : "启动或流式读取评测失败");
      }
      setRunningJob(false);
    }
  };

  const handleToggleRunMode = (mode: "bare_llm" | "harness_copilot") => {
    setRunModes((prev) =>
      prev.includes(mode) ? prev.filter((m) => m !== mode) : [...prev, mode]
    );
  };

  const freeBreachCount = useMemo(
    () => freeRounds.filter((r) => Boolean(r.result.bare?.breached)).length,
    [freeRounds]
  );

  // Inject left sidebar slot into GlassShell
  useEffect(() => {
    if (!onRenderSidebarSlot) return;
    onRenderSidebarSlot(
      <SandboxSidebar
        subMode={subMode}
        onChangeSubMode={setSubMode}
        reports={reports}
        selectedFilename={selectedFilename}
        loadingReports={loadingList}
        onSelectReport={(fn) => {
          setSelectedFilename(fn);
          setSubMode("batch");
        }}
        onRefreshReports={() => {
          void fetchReports();
        }}
        onImportReport={(file) => {
          void handleImportReport(file);
        }}
        freeTurnCount={freeRounds.length}
        freeBreachCount={freeBreachCount}
        onOpenSaveCaseModal={() => {
          setSubMode("free");
          setSaveCaseModalOpen(true);
        }}
      />
    );
    return () => onRenderSidebarSlot(null);
  }, [
    onRenderSidebarSlot,
    subMode,
    reports,
    selectedFilename,
    loadingList,
    fetchReports,
    handleImportReport,
    freeRounds.length,
    freeBreachCount,
  ]);

  // Inject top-right export actions when viewing a batch report
  useEffect(() => {
    if (!onRenderHeaderActions) return;
    if (subMode === "batch" && report) {
      onRenderHeaderActions(
        <div className="flex flex-wrap items-center gap-1.5">
          {(["csv", "json", "markdown"] as const).map((fmt) => (
            <button
              key={fmt}
              type="button"
              onClick={() => void api.exportBenchmarkReport(report.filename, fmt)}
              className="inline-flex items-center gap-1 rounded-xl border border-slate-200/90 bg-white/90 px-2.5 py-1.5 text-[11px] font-semibold text-slate-700 hover:bg-white"
            >
              <Download className="h-3 w-3 text-cyan-700" />
              导出 {fmt === "markdown" ? "MD" : fmt.toUpperCase()}
            </button>
          ))}
        </div>
      );
    } else {
      onRenderHeaderActions(null);
    }
    return () => onRenderHeaderActions(null);
  }, [onRenderHeaderActions, subMode, report]);

  const runs = report?.runs || [];
  const pairs = report?.pairs || [];
  const activePair = pairs[activePairIdx] || pairs[0] || null;

  const radarData = useMemo(() => {
    return DIMENSION_ORDER.map((dimKey) => {
      const point: Record<string, string | number> = {
        dimension: DIMENSION_LABELS[dimKey] || dimKey,
      };
      runs.forEach((run) => {
        point[run.run_label] = getDimRate(run, dimKey);
      });
      return point;
    });
  }, [runs]);

  const filteredSideBySide = useMemo(() => {
    const list = report?.side_by_side_cases || [];
    const needle = searchQuery.trim().toLowerCase();
    return list.filter((item) => {
      if (filterMode === "rescued" && !item.rescued) return false;
      if (filterMode === "failed" && !item.any_failed) return false;
      if (filterDim !== "all" && item.dimension !== filterDim) return false;
      if (needle) {
        const hay = [
          item.case_id,
          item.category,
          item.dimension_label || DIMENSION_LABELS[item.dimension] || item.dimension,
          item.expected_action,
          ...(item.bare_result?.failure_reasons || []),
          ...(item.harness_result?.failure_reasons || []),
          ...(item.bare_result?.turns || []).map((t) => `${t.user_msg} ${t.assistant_msg}`),
          ...(item.harness_result?.turns || []).map((t) => `${t.user_msg} ${t.assistant_msg}`),
        ]
          .join(" ")
          .toLowerCase();
        if (!hay.includes(needle)) return false;
      }
      return true;
    });
  }, [report?.side_by_side_cases, filterMode, filterDim, searchQuery]);

  const selectedCase: SideBySideCase | null = useMemo(() => {
    const list = report?.side_by_side_cases || [];
    if (list.length === 0) return null;
    return (
      list.find((item) => `${item.pair_label}::${item.case_id}` === selectedCaseKey) ||
      filteredSideBySide[0] ||
      list[0]
    );
  }, [report?.side_by_side_cases, selectedCaseKey, filteredSideBySide]);

  const rescuedTotal = (report?.side_by_side_cases || []).filter((c) => c.rescued).length;
  const failedTotal = (report?.side_by_side_cases || []).filter((c) => c.any_failed).length;

  return (
    <div className="flex min-h-[540px] flex-col gap-3 md:h-[calc(100vh-92px)] md:flex-row">
      {/* Center Column */}
      <div className="relative z-20 flex min-h-0 min-w-0 flex-1 flex-col gap-3">
        {errorMsg && (
          <div className="glass-danger flex shrink-0 items-center justify-between gap-2 rounded-2xl px-4 py-2.5 text-xs text-rose-900">
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

        {subMode === "free" ? (
          <div className="min-h-0 flex-1">
            <FreeRedTeamPanel
              availableModels={modelChoices}
              defaultModel={defaultModel || modelChoices[0] || "deepseek/deepseek-chat"}
              modelCapabilities={modelCapabilities}
              defaultMockMode={defaultMockMode}
              rounds={freeRounds}
              saveModalOpen={saveCaseModalOpen}
              onCloseSaveModal={() => setSaveCaseModalOpen(false)}
              onOpenSaveModal={() => setSaveCaseModalOpen(true)}
              onAppendRound={(round) => setFreeRounds((prev) => [...prev, round])}
              onClearRounds={() => setFreeRounds([])}
            />
          </div>
        ) : (
          <>
            {/* Scrollable Batch Report / Live Progress Area */}
            <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
              {runningJob ? (
                <div className="glass-surface space-y-3 rounded-3xl p-4 sm:p-5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin text-cyan-700" />
                      <h3 className="text-sm font-bold text-slate-900">
                        批量红队评测执行中
                      </h3>
                    </div>
                    <span className="rounded-full bg-cyan-50 px-2.5 py-0.5 text-xs font-semibold text-cyan-800">
                      {jobProgress?.progress ?? 0}%
                    </span>
                  </div>

                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200/80">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-cyan-600 to-teal-500 transition-all duration-300"
                      style={{ width: `${Math.min(100, Math.max(2, jobProgress?.progress ?? 2))}%` }}
                    />
                  </div>
                  <p className="text-xs text-slate-600">{jobProgress?.message || "准备执行..."}</p>

                  {liveCaseItems.length > 0 && (
                    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-4">
                      {liveCaseItems.map((item) => (
                        <div
                          key={item.key}
                          className={`flex items-center justify-between gap-1.5 rounded-xl border px-2.5 py-2 text-[11px] ${
                            item.passed
                              ? "border-emerald-200 bg-emerald-50/75 text-emerald-900"
                              : "border-rose-200 bg-rose-50/75 text-rose-900"
                          }`}
                        >
                          <div className="min-w-0">
                            <div className="truncate font-mono font-bold">{item.caseId}</div>
                            <div className="truncate text-[10px] opacity-75">
                              {item.mode === "bare_llm" ? "裸模型" : "护栏"} · {item.model.split("/").pop()}
                            </div>
                          </div>
                          {item.passed ? (
                            <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-600" />
                          ) : (
                            <XCircle className="h-3.5 w-3.5 shrink-0 text-rose-600" />
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : loadingDetail ? (
                <div className="glass-surface flex min-h-[300px] items-center justify-center rounded-3xl text-xs text-slate-500">
                  正在加载评测报告详情...
                </div>
              ) : !report ? (
                <div className="glass-surface flex min-h-[320px] flex-col items-center justify-center rounded-3xl p-8 text-center">
                  <ShieldCheck className="h-10 w-10 text-cyan-700" />
                  <h3 className="mt-3 text-base font-bold text-slate-900">
                    暂无选中的批量评测报告
                  </h3>
                  <p className="mt-1.5 max-w-md text-xs text-slate-500">
                    在左侧选择或导入历史 JSON 报告，或直接在下方配置条勾选模型与模式后点击「发起批量评测」。
                  </p>
                </div>
              ) : (
                <>
                  {/* Compact Report Meta Strip */}
                  <div className="glass-surface flex flex-wrap items-center gap-2 rounded-2xl px-3.5 py-2.5 text-xs text-slate-600">
                    <span className="font-mono font-semibold text-slate-800">{report.filename}</span>
                    <span>·</span>
                    <span>{report.meta?.created_at || "-"}</span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                        report.meta?.mock_mode
                          ? "bg-amber-100 text-amber-800"
                          : "bg-emerald-100 text-emerald-800"
                      }`}
                    >
                      {report.meta?.mock_mode ? "Mock 仿真" : "真实模型"}
                    </span>
                    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-700">
                      {report.meta?.case_count ?? 0} 题
                    </span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
                        report.meta?.include_private
                          ? "bg-rose-100 text-rose-800"
                          : "bg-cyan-50 text-cyan-800"
                      }`}
                    >
                      {report.meta?.include_private ? "含私有红队集" : "仅公开集"}
                    </span>
                  </div>

                  {report.infra_errors && report.infra_errors.length > 0 && (
                    <div className="glass-warning rounded-2xl px-4 py-2.5 text-xs text-amber-900">
                      <div className="font-semibold">存在模型调用或结构化抽取基础设施告警：</div>
                      <ul className="mt-1 list-disc pl-4">
                        {report.infra_errors.map((errItem, idx) => {
                          const text = Array.isArray(errItem)
                            ? `${errItem[0]}: ${errItem[1]}`
                            : `${errItem.model || "未知模型"}: ${errItem.error || ""}`;
                          return <li key={idx}>{text}</li>;
                        })}
                      </ul>
                    </div>
                  )}

                  {/* KPI Delta Strip */}
                  {activePair && (
                    <div className="space-y-2">
                      {pairs.length > 1 && (
                        <div className="flex flex-wrap items-center gap-1.5 text-xs">
                          <span className="font-semibold text-slate-600">对比模型组：</span>
                          {pairs.map((p, idx) => (
                            <button
                              key={p.pair_label}
                              type="button"
                              onClick={() => setActivePairIdx(idx)}
                              className={`rounded-full px-2.5 py-1 text-[11px] font-semibold transition-all ${
                                idx === activePairIdx
                                  ? "bg-cyan-700 text-white"
                                  : "bg-white/80 text-slate-700 hover:bg-white"
                              }`}
                            >
                              {p.pair_label}
                            </button>
                          ))}
                        </div>
                      )}

                      <div className="grid grid-cols-2 gap-2.5 xl:grid-cols-4">
                        <CompactKpiCard
                          title="总体通过率"
                          value={`${(activePair.harness_summary?.overall_pass_rate ?? 0).toFixed(1)}%`}
                          bareValue={`裸模型 ${(activePair.bare_summary?.overall_pass_rate ?? 0).toFixed(1)}%`}
                          delta={formatDelta(activePair.deltas?.overall_pass_rate)}
                        />
                        <CompactKpiCard
                          title="安全红线否决率"
                          value={`${(activePair.harness_summary?.safety_veto_rate ?? 0).toFixed(1)}%`}
                          bareValue={`裸模型 ${(activePair.bare_summary?.safety_veto_rate ?? 0).toFixed(1)}%`}
                          delta={formatDelta(activePair.deltas?.safety_veto_rate, true)}
                        />
                        <CompactKpiCard
                          title="红旗安全拦截"
                          value={`${(
                            activePair.harness_summary?.dimensions?.red_flag_interception?.pass_rate ?? 0
                          ).toFixed(1)}%`}
                          bareValue={`裸模型 ${(
                            activePair.bare_summary?.dimensions?.red_flag_interception?.pass_rate ?? 0
                          ).toFixed(1)}%`}
                          delta={formatDelta(activePair.deltas?.red_flag_interception)}
                        />
                        <CompactKpiCard
                          title="防过度拒答校准"
                          value={`${(
                            activePair.harness_summary?.dimensions?.over_refusal_calibration?.pass_rate ?? 0
                          ).toFixed(1)}%`}
                          bareValue={`裸模型 ${(
                            activePair.bare_summary?.dimensions?.over_refusal_calibration?.pass_rate ?? 0
                          ).toFixed(1)}%`}
                          delta={formatDelta(activePair.deltas?.over_refusal_calibration)}
                        />
                      </div>
                    </div>
                  )}

                  {/* Charts Row */}
                  {runs.length > 0 && (
                    <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                      <div className="glass-surface rounded-3xl p-4">
                        <h3 className="mb-1 text-xs font-bold text-slate-900">四维能力通过率雷达图</h3>
                        <p className="mb-2 text-[11px] text-slate-500">
                          虚线：Bare LLM ｜ 实线：Harness Copilot
                        </p>
                        <div className="h-56 w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <RadarChart data={radarData} outerRadius="70%">
                              <PolarGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                              <PolarAngleAxis
                                dataKey="dimension"
                                tick={{ fill: "#334155", fontSize: 10, fontWeight: 600 }}
                              />
                              <PolarRadiusAxis
                                angle={90}
                                domain={[0, 100]}
                                tick={{ fill: "#64748b", fontSize: 9 }}
                              />
                              {runs.map((run, idx) => {
                                const color = SERIES_COLORS[idx % SERIES_COLORS.length];
                                const isBare = run.mode === "bare_llm";
                                return (
                                  <Radar
                                    key={run.run_label}
                                    name={run.run_label}
                                    dataKey={run.run_label}
                                    stroke={color}
                                    strokeWidth={isBare ? 2 : 2.5}
                                    strokeDasharray={isBare ? "5 4" : undefined}
                                    fill={color}
                                    fillOpacity={isBare ? 0.08 : 0.22}
                                  />
                                );
                              })}
                              <Tooltip
                                formatter={(value: number) => `${Number(value).toFixed(1)}%`}
                                contentStyle={{
                                  borderRadius: "12px",
                                  border: "1px solid rgba(255,255,255,0.9)",
                                  background: "rgba(255,255,255,0.95)",
                                  fontSize: "11px",
                                }}
                              />
                              <Legend wrapperStyle={{ fontSize: "10px" }} />
                            </RadarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      <div className="glass-surface rounded-3xl p-4">
                        <h3 className="mb-1 text-xs font-bold text-slate-900">四维能力分组柱状图</h3>
                        <p className="mb-2 text-[11px] text-slate-500">
                          对比各组在红旗拦截、纵向记忆、辨证工具与防过度拒答的通过率
                        </p>
                        <div className="h-56 w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={radarData} margin={{ top: 8, right: 8, left: -16, bottom: 4 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                              <XAxis
                                dataKey="dimension"
                                tick={{ fill: "#334155", fontSize: 10, fontWeight: 500 }}
                              />
                              <YAxis
                                domain={[0, 100]}
                                unit="%"
                                tick={{ fill: "#64748b", fontSize: 10 }}
                              />
                              <Tooltip
                                formatter={(value: number) => `${Number(value).toFixed(1)}%`}
                                contentStyle={{
                                  borderRadius: "12px",
                                  border: "1px solid rgba(255,255,255,0.9)",
                                  background: "rgba(255,255,255,0.95)",
                                  fontSize: "11px",
                                }}
                              />
                              <Legend wrapperStyle={{ fontSize: "10px" }} />
                              {runs.map((run, idx) => {
                                const color = SERIES_COLORS[idx % SERIES_COLORS.length];
                                const isBare = run.mode === "bare_llm";
                                return (
                                  <Bar
                                    key={run.run_label}
                                    dataKey={run.run_label}
                                    fill={color}
                                    fillOpacity={isBare ? 0.45 : 0.92}
                                    radius={[6, 6, 0, 0]}
                                  />
                                );
                              })}
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Case Table (click to inspect in right CaseDetailDrawer) */}
                  <div className="glass-surface rounded-3xl p-4">
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <h3 className="text-xs font-bold text-slate-900">
                          逐题对照列表（点击任一行在右侧回放）
                        </h3>
                        <p className="text-[11px] text-slate-500">
                          共 {report.side_by_side_cases?.length || 0} 条，当前筛选{" "}
                          {filteredSideBySide.length} 条
                        </p>
                      </div>

                      <div className="flex flex-wrap items-center gap-1.5">
                        {(
                          [
                            { id: "all", label: `全部 (${report.side_by_side_cases?.length || 0})` },
                            { id: "rescued", label: `护栏挽回 (${rescuedTotal})` },
                            { id: "failed", label: `仍有失误 (${failedTotal})` },
                          ] as const
                        ).map((tab) => (
                          <button
                            key={tab.id}
                            type="button"
                            onClick={() => setFilterMode(tab.id)}
                            className={`rounded-xl px-2.5 py-1 text-[11px] font-semibold transition-all ${
                              filterMode === tab.id
                                ? "bg-cyan-700 text-white"
                                : "border border-slate-200 bg-white/85 text-slate-600 hover:bg-white"
                            }`}
                          >
                            {tab.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="mb-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                      <div className="relative">
                        <Filter className="pointer-events-none absolute left-2.5 top-2 h-3.5 w-3.5 text-slate-400" />
                        <select
                          value={filterDim}
                          onChange={(e) => setFilterDim(e.target.value)}
                          className="w-full rounded-xl border border-slate-200 bg-white/95 py-1.5 pl-8 pr-3 text-xs text-slate-700 focus:border-cyan-600 focus:outline-none"
                        >
                          <option value="all">全部评估维度</option>
                          {DIMENSION_ORDER.map((d) => (
                            <option key={d} value={d}>
                              {DIMENSION_LABELS[d]}
                            </option>
                          ))}
                        </select>
                      </div>

                      <div className="relative">
                        <Search className="pointer-events-none absolute left-2.5 top-2 h-3.5 w-3.5 text-slate-400" />
                        <input
                          type="text"
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          placeholder="搜索用例编号、类别或对话关键词..."
                          className="w-full rounded-xl border border-slate-200 bg-white/95 py-1.5 pl-8 pr-3 text-xs text-slate-800 placeholder:text-slate-400 focus:border-cyan-600 focus:outline-none"
                        />
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      {filteredSideBySide.length === 0 ? (
                        <div className="rounded-2xl border border-dashed border-slate-200 py-8 text-center text-xs text-slate-500">
                          没有符合筛选条件的红队用例
                        </div>
                      ) : (
                        filteredSideBySide.map((item) => {
                          const rowKey = `${item.pair_label}::${item.case_id}`;
                          const active = selectedCase && `${selectedCase.pair_label}::${selectedCase.case_id}` === rowKey;
                          const bareOk = item.bare_result?.passed;
                          const harnessOk = item.harness_result?.passed;
                          const bareStatus = item.bare_result?.status || "-";
                          const harnessStatus = item.harness_result?.status || "-";
                          return (
                            <button
                              key={rowKey}
                              type="button"
                              onClick={() => {
                                setSelectedCaseKey(rowKey);
                                setDrawerOpen(true);
                              }}
                              className={`flex w-full flex-wrap items-center justify-between gap-2 rounded-2xl border px-3.5 py-2.5 text-left transition-all ${
                                active
                                  ? "border-cyan-400 bg-cyan-50/75 ring-2 ring-cyan-500/30"
                                  : "border-slate-200/80 bg-white/80 hover:bg-white"
                              }`}
                            >
                              <div className="flex min-w-0 flex-wrap items-center gap-2">
                                <span className="font-mono text-xs font-bold text-slate-900">
                                  {item.case_id}
                                </span>
                                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-700">
                                  {item.dimension_label || DIMENSION_LABELS[item.dimension] || item.dimension}
                                </span>
                                {item.rescued && (
                                  <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-bold text-emerald-800">
                                    护栏挽回
                                  </span>
                                )}
                                {item.regressed && (
                                  <span className="rounded-full bg-rose-100 px-2 py-0.5 text-[10px] font-bold text-rose-800">
                                    护栏退化
                                  </span>
                                )}
                                <span className="truncate text-[11px] text-slate-500">
                                  {item.pair_label}
                                </span>
                              </div>

                              <div className="flex items-center gap-1.5 text-[11px]">
                                <span
                                  className={`rounded-full px-2 py-0.5 font-medium ${
                                    bareOk
                                      ? "bg-emerald-50 text-emerald-700"
                                      : "bg-rose-50 text-rose-700"
                                  }`}
                                >
                                  裸：{BENCHMARK_STATUS_LABELS[bareStatus]?.label || bareStatus}
                                </span>
                                <span
                                  className={`rounded-full px-2 py-0.5 font-semibold ${
                                    harnessOk
                                      ? "bg-emerald-100 text-emerald-800"
                                      : "bg-rose-100 text-rose-800"
                                  }`}
                                >
                                  护：{BENCHMARK_STATUS_LABELS[harnessStatus]?.label || harnessStatus}
                                </span>
                              </div>
                            </button>
                          );
                        })
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Bottom Sticky Run Configuration Bar */}
            <div className="glass-surface relative z-30 shrink-0 rounded-2xl p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex min-w-0 items-center gap-2">
                  <span className="shrink-0 text-[11px] font-semibold text-slate-600">
                    评测模型：
                  </span>
                  <ModelThinkingPicker
                    mode="multi"
                    direction="up"
                    testId="batch-model-selector"
                    availableModels={modelChoices}
                    selectedModels={runModels}
                    modelCapabilities={modelCapabilities}
                    modelThinking={modelThinking}
                    mockMode={runMockMode}
                    disabled={runningJob}
                    onChangeModels={setRunModels}
                    onChangeThinking={(m, level) =>
                      setModelThinking((prev) => ({ ...prev, [m]: level }))
                    }
                  />
                </div>

                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <label className="inline-flex cursor-pointer items-center gap-1 text-slate-700">
                    <input
                      type="checkbox"
                      disabled={runningJob}
                      checked={runModes.includes("bare_llm")}
                      onChange={() => handleToggleRunMode("bare_llm")}
                      className="h-3.5 w-3.5 rounded text-cyan-700"
                    />
                    裸模型
                  </label>
                  <label className="inline-flex cursor-pointer items-center gap-1 text-slate-700">
                    <input
                      type="checkbox"
                      disabled={runningJob}
                      checked={runModes.includes("harness_copilot")}
                      onChange={() => handleToggleRunMode("harness_copilot")}
                      className="h-3.5 w-3.5 rounded text-cyan-700"
                    />
                    护栏协同
                  </label>
                  <label className="inline-flex cursor-pointer items-center gap-1 text-slate-700">
                    <input
                      type="checkbox"
                      disabled={runningJob}
                      checked={includePrivate}
                      onChange={(e) => setIncludePrivate(e.target.checked)}
                      className="h-3.5 w-3.5 rounded text-cyan-700"
                    />
                    含私有库
                  </label>
                  <label className="inline-flex cursor-pointer items-center gap-1 text-slate-700">
                    <input
                      type="checkbox"
                      disabled={runningJob}
                      checked={runMockMode}
                      onChange={(e) => setRunMockMode(e.target.checked)}
                      className="h-3.5 w-3.5 rounded text-cyan-700"
                    />
                    Mock 仿真
                  </label>

                  {runningJob ? (
                    <button
                      type="button"
                      onClick={() => {
                        abortRef.current?.abort();
                        setRunningJob(false);
                      }}
                      className="inline-flex items-center gap-1.5 rounded-xl bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-rose-700"
                    >
                      <Square className="h-3.5 w-3.5" />
                      停止监控
                    </button>
                  ) : (
                    <button
                      type="button"
                      data-testid="start-batch-benchmark-btn"
                      onClick={() => void handleStartBenchmark()}
                      className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-cyan-700 to-teal-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow-sm hover:brightness-105"
                    >
                      <Play className="h-3.5 w-3.5" />
                      发起批量评测
                    </button>
                  )}
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Right Column: Batch Case Detail Drawer */}
      {subMode === "batch" && (
        <CaseDetailDrawer
          open={drawerOpen}
          selectedCase={selectedCase}
          onToggleOpen={setDrawerOpen}
        />
      )}
    </div>
  );
};

interface CompactKpiCardProps {
  title: string;
  value: string;
  bareValue: string;
  delta: { text: string; positive: boolean };
}

const CompactKpiCard: React.FC<CompactKpiCardProps> = ({
  title,
  value,
  bareValue,
  delta,
}) => (
  <div className="glass-surface rounded-2xl px-3.5 py-2.5">
    <div className="text-[11px] font-medium text-slate-500">{title}</div>
    <div className="mt-1 flex items-baseline justify-between gap-2">
      <span className="text-xl font-bold tracking-tight text-slate-900">{value}</span>
      <span
        className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
          delta.positive ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"
        }`}
      >
        {delta.text}
      </span>
    </div>
    <div className="mt-0.5 text-[10px] text-slate-500">{bareValue}</div>
  </div>
);
