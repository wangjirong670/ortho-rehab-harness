import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Download,
  Filter,
  Play,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Upload,
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
import { api } from "../api/client";
import type {
  BenchmarkCaseResult,
  BenchmarkProgressEvent,
  BenchmarkReportDetail,
  BenchmarkReportMeta,
  BenchmarkRun,
  HighlightSpan,
  SideBySideCase,
} from "../api/types";
import { Modal } from "../components/Modal";
import { SafeHighlightText } from "../components/SafeHighlightText";
import {
  BENCHMARK_STATUS_LABELS,
  DIMENSION_LABELS,
  DIMENSION_ORDER,
  getClinicalActionLabel,
} from "../utils/labels";

interface BenchmarkPageProps {
  presetModels: string[];
  ollamaModels: string[];
  defaultBenchmarkModels: string[];
  defaultMockMode: boolean;
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

function formatDelta(val: number | undefined, invertGood = false): {
  text: string;
  positive: boolean;
} {
  const num = typeof val === "number" && Number.isFinite(val) ? val : 0;
  const sign = num > 0 ? "+" : "";
  const good = invertGood ? num <= 0 : num >= 0;
  return { text: `${sign}${num.toFixed(1)}%`, positive: good };
}

function resolveTurnHighlights(
  highlightsMap: Record<string, HighlightSpan[]> | undefined,
  turnIndex: number,
  turnNumber?: number
): HighlightSpan[] {
  if (!highlightsMap) return [];
  if (highlightsMap[String(turnIndex)]) return highlightsMap[String(turnIndex)];
  if (turnNumber !== undefined && highlightsMap[String(turnNumber)]) {
    return highlightsMap[String(turnNumber)];
  }
  return [];
}

export const BenchmarkPage: React.FC<BenchmarkPageProps> = ({
  presetModels,
  ollamaModels,
  defaultBenchmarkModels,
  defaultMockMode,
}) => {
  const [reports, setReports] = useState<BenchmarkReportMeta[]>([]);
  const [selectedFilename, setSelectedFilename] = useState<string>("");
  const [report, setReport] = useState<BenchmarkReportDetail | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Pair selection for KPI cards
  const [activePairIdx, setActivePairIdx] = useState(0);

  // Side-by-side filters
  const [filterMode, setFilterMode] = useState<"all" | "rescued" | "failed">("all");
  const [filterDim, setFilterDim] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");

  // Run Benchmark Modal & SSE state
  const [runModalOpen, setRunModalOpen] = useState(false);
  const [runModels, setRunModels] = useState<string[]>(
    defaultBenchmarkModels.length > 0 ? defaultBenchmarkModels : ["deepseek/deepseek-chat"]
  );
  const [customModelInput, setCustomModelInput] = useState("");
  const [runModes, setRunModes] = useState<Array<"bare_llm" | "harness_copilot">>([
    "bare_llm",
    "harness_copilot",
  ]);
  const [runMockMode, setRunMockMode] = useState<boolean>(defaultMockMode);
  const [includePrivate, setIncludePrivate] = useState<boolean>(false);

  const [runningJob, setRunningJob] = useState(false);
  const [jobProgress, setJobProgress] = useState<BenchmarkProgressEvent | null>(null);
  const [jobLogs, setJobLogs] = useState<string[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const importInputRef = useRef<HTMLInputElement | null>(null);

  const availableModels = useMemo(() => {
    const set = new Set<string>([
      ...presetModels,
      ...ollamaModels,
      ...defaultBenchmarkModels,
      ...runModels,
    ]);
    return Array.from(set).filter(Boolean);
  }, [presetModels, ollamaModels, defaultBenchmarkModels, runModels]);

  const fetchReports = async (preferredFilename?: string) => {
    setLoadingList(true);
    setErrorMsg(null);
    try {
      const res = await api.listBenchmarkReports();
      const list = res.reports || [];
      setReports(list);
      const target =
        (preferredFilename && list.find((r) => r.filename === preferredFilename)?.filename) ||
        (selectedFilename && list.find((r) => r.filename === selectedFilename)?.filename) ||
        list[0]?.filename ||
        "";
      if (target) {
        setSelectedFilename(target);
      } else {
        setReport(null);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "加载评测报告列表失败");
    } finally {
      setLoadingList(false);
    }
  };

  useEffect(() => {
    fetchReports();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selectedFilename) return;
    let active = true;
    setLoadingDetail(true);
    setErrorMsg(null);
    api
      .getBenchmarkReport(selectedFilename)
      .then((detail) => {
        if (active) {
          setReport(detail);
          setActivePairIdx(0);
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

  const handleImportJson = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg(null);
    try {
      const res = await api.importBenchmarkReport(file);
      await fetchReports(res.filename);
      setSelectedFilename(res.filename);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "导入评测报告失败");
    } finally {
      e.target.value = "";
    }
  };

  const handleToggleRunModel = (model: string) => {
    setRunModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    );
  };

  const handleAddCustomModel = () => {
    const trimmed = customModelInput.trim();
    if (!trimmed) return;
    if (!runModels.includes(trimmed)) {
      setRunModels((prev) => [...prev, trimmed]);
    }
    setCustomModelInput("");
  };

  const handleStartBenchmark = async () => {
    if (runModels.length === 0 || runModes.length === 0) {
      setErrorMsg("请至少选择一个评测模型和一种运行模式。");
      return;
    }
    setRunningJob(true);
    setJobProgress({ progress: 0, message: "正在创建评测任务..." });
    setJobLogs(["正在提交评测任务到后端调度器..."]);
    setErrorMsg(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const { job_id } = await api.startBenchmarkRun({
        models: runModels,
        modes: runModes,
        mock_mode: runMockMode,
        include_private: includePrivate,
      });

      setJobLogs((prev) => [...prev, `任务已启动 (${job_id})，正在连接实时进度流...`]);

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
            setJobLogs((prev) => [...prev.slice(-30), summaryLine]);
          },
          onDone: async (ev) => {
            setJobProgress({ ...ev, progress: 100, message: "评测已完成，正在刷新报告..." });
            setJobLogs((prev) => [
              ...prev.slice(-30),
              ev.message || `评测执行完成：${ev.filename || ""}`,
            ]);
            setRunningJob(false);
            await fetchReports(ev.filename);
            if (ev.filename) {
              setSelectedFilename(ev.filename);
            }
            window.setTimeout(() => setRunModalOpen(false), 600);
          },
          onError: (msg) => {
            setRunningJob(false);
            setErrorMsg(`评测执行失败：${msg}`);
            setJobLogs((prev) => [...prev, `错误：${msg}`]);
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

  const runs = report?.runs || [];
  const pairs = report?.pairs || [];
  const activePair = pairs[activePairIdx] || pairs[0] || null;

  // Chart datasets
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

  const rescuedTotal = (report?.side_by_side_cases || []).filter((c) => c.rescued).length;
  const failedTotal = (report?.side_by_side_cases || []).filter((c) => c.any_failed).length;

  return (
    <div className="space-y-4">
      {errorMsg && (
        <div className="glass-danger flex items-center justify-between gap-2 rounded-2xl px-4 py-3 text-xs text-rose-900">
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

      {/* Top Controls Bar */}
      <div className="glass-surface flex flex-wrap items-center justify-between gap-3 rounded-3xl p-4">
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2.5">
          <select
            value={selectedFilename}
            disabled={loadingList || reports.length === 0}
            onChange={(e) => setSelectedFilename(e.target.value)}
            className="min-w-[260px] max-w-full flex-1 rounded-2xl border border-slate-200/90 bg-white/95 px-3.5 py-2 text-xs font-medium text-slate-800 focus:border-cyan-600 focus:outline-none"
          >
            {reports.length === 0 ? (
              <option value="">暂无历史评测报告</option>
            ) : (
              reports.map((r) => (
                <option key={r.filename} value={r.filename}>
                  {r.label || `${r.filename} (${r.case_count}题)`}
                </option>
              ))
            )}
          </select>

          <button
            type="button"
            onClick={() => fetchReports()}
            className="rounded-2xl border border-slate-200 bg-white/85 p-2 text-slate-600 hover:bg-white"
            title="刷新报告列表"
          >
            <RefreshCw className={`h-4 w-4 ${loadingList ? "animate-spin" : ""}`} />
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <input
            ref={importInputRef}
            type="file"
            accept=".json"
            className="hidden"
            onChange={handleImportJson}
          />
          <button
            type="button"
            onClick={() => importInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 rounded-2xl border border-slate-200 bg-white/90 px-3.5 py-2 text-xs font-semibold text-slate-700 hover:bg-white"
          >
            <Upload className="h-3.5 w-3.5 text-cyan-700" />
            导入报告 JSON
          </button>

          <button
            type="button"
            onClick={() => setRunModalOpen(true)}
            className="inline-flex items-center gap-1.5 rounded-2xl bg-gradient-to-r from-cyan-700 to-teal-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-cyan-900/15 hover:brightness-105"
          >
            <Play className="h-3.5 w-3.5" />
            运行新评测
          </button>
        </div>
      </div>

      {loadingDetail ? (
        <div className="glass-surface flex min-h-[320px] items-center justify-center rounded-3xl text-xs text-slate-500">
          正在加载评测报告详情...
        </div>
      ) : !report ? (
        <div className="glass-surface flex min-h-[380px] flex-col items-center justify-center rounded-3xl p-8 text-center">
          <ShieldCheck className="h-10 w-10 text-cyan-700" />
          <h3 className="mt-3 text-base font-bold text-slate-900">暂无可展示的红队评测报告</h3>
          <p className="mt-1.5 max-w-md text-xs text-slate-500">
            点击右上角「运行新评测」发起 Mock 快速仿真或真实模型评测，或导入外部 JSON 评测结果文件。
          </p>
        </div>
      ) : (
        <>
          {/* Meta & Export Strip */}
          <div className="glass-surface flex flex-wrap items-center justify-between gap-3 rounded-2xl px-4 py-3 text-xs">
            <div className="flex flex-wrap items-center gap-2 text-slate-600">
              <span className="font-mono font-semibold text-slate-800">{report.filename}</span>
              <span>·</span>
              <span>时间：{report.meta?.created_at || "-"}</span>
              <span>·</span>
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
                {report.meta?.include_private ? "含私有红队集" : "仅公开集（隐私隔离）"}
              </span>
            </div>

            <div className="flex items-center gap-1.5">
              {(["csv", "json", "markdown"] as const).map((fmt) => (
                <button
                  key={fmt}
                  type="button"
                  onClick={() => api.exportBenchmarkReport(report.filename, fmt)}
                  className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white/90 px-2.5 py-1 text-[11px] font-semibold text-slate-700 hover:bg-white"
                >
                  <Download className="h-3 w-3 text-cyan-700" />
                  导出 {fmt === "markdown" ? "MD" : fmt.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Infra errors if any */}
          {report.infra_errors && report.infra_errors.length > 0 && (
            <div className="glass-warning rounded-2xl px-4 py-3 text-xs text-amber-900">
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

          {/* KPI Delta Cards */}
          {activePair && (
            <div className="space-y-2">
              {pairs.length > 1 && (
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="font-semibold text-slate-600">对比模型组：</span>
                  {pairs.map((p, idx) => (
                    <button
                      key={p.pair_label}
                      type="button"
                      onClick={() => setActivePairIdx(idx)}
                      className={`rounded-full px-3 py-1 font-medium transition-all ${
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

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <KpiDeltaCard
                  title="总体通过率"
                  value={`${(activePair.harness_summary?.overall_pass_rate ?? 0).toFixed(1)}%`}
                  bareValue={`裸模型 ${(activePair.bare_summary?.overall_pass_rate ?? 0).toFixed(1)}%`}
                  delta={formatDelta(activePair.deltas?.overall_pass_rate)}
                />
                <KpiDeltaCard
                  title="安全红线否决率（越低越安全）"
                  value={`${(activePair.harness_summary?.safety_veto_rate ?? 0).toFixed(1)}%`}
                  bareValue={`裸模型 ${(activePair.bare_summary?.safety_veto_rate ?? 0).toFixed(1)}%`}
                  delta={formatDelta(activePair.deltas?.safety_veto_rate, true)}
                />
                <KpiDeltaCard
                  title="红旗安全拦截能力"
                  value={`${(
                    activePair.harness_summary?.dimensions?.red_flag_interception?.pass_rate ?? 0
                  ).toFixed(1)}%`}
                  bareValue={`裸模型 ${(
                    activePair.bare_summary?.dimensions?.red_flag_interception?.pass_rate ?? 0
                  ).toFixed(1)}%`}
                  delta={formatDelta(activePair.deltas?.red_flag_interception)}
                />
                <KpiDeltaCard
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

          {/* Charts: Radar + Grouped Bar */}
          {runs.length > 0 && (
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              <div className="glass-surface rounded-3xl p-4 sm:p-5">
                <h3 className="mb-1 text-sm font-bold text-slate-900">四维能力通过率雷达图</h3>
                <p className="mb-3 text-[11px] text-slate-500">
                  虚线代表裸模型（Bare LLM），实线代表护栏协同（Harness Copilot）
                </p>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData} outerRadius="72%">
                      <PolarGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                      <PolarAngleAxis
                        dataKey="dimension"
                        tick={{ fill: "#334155", fontSize: 11, fontWeight: 600 }}
                      />
                      <PolarRadiusAxis
                        angle={90}
                        domain={[0, 100]}
                        tick={{ fill: "#64748b", fontSize: 10 }}
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
                            strokeWidth={isBare ? 2 : 2.6}
                            strokeDasharray={isBare ? "5 4" : undefined}
                            fill={color}
                            fillOpacity={isBare ? 0.08 : 0.22}
                          />
                        );
                      })}
                      <Tooltip
                        formatter={(value: number) => `${Number(value).toFixed(1)}%`}
                        contentStyle={{
                          borderRadius: "14px",
                          border: "1px solid rgba(255,255,255,0.9)",
                          background: "rgba(255,255,255,0.94)",
                          fontSize: "12px",
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: "11px" }} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="glass-surface rounded-3xl p-4 sm:p-5">
                <h3 className="mb-1 text-sm font-bold text-slate-900">四维能力分组柱状图</h3>
                <p className="mb-3 text-[11px] text-slate-500">
                  对比各运行配置在红旗拦截、纵向记忆、辨证工具及防过度拒答的通过率
                </p>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={radarData} margin={{ top: 10, right: 12, left: -12, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                      <XAxis
                        dataKey="dimension"
                        tick={{ fill: "#334155", fontSize: 11, fontWeight: 500 }}
                      />
                      <YAxis
                        domain={[0, 100]}
                        unit="%"
                        tick={{ fill: "#64748b", fontSize: 11 }}
                      />
                      <Tooltip
                        formatter={(value: number) => `${Number(value).toFixed(1)}%`}
                        contentStyle={{
                          borderRadius: "14px",
                          border: "1px solid rgba(255,255,255,0.9)",
                          background: "rgba(255,255,255,0.94)",
                          fontSize: "12px",
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: "11px" }} />
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

          {/* Dimension Comparison Table */}
          {report.dimension_rows && report.dimension_rows.length > 0 && (
            <div className="glass-surface overflow-hidden rounded-3xl p-4 sm:p-5">
              <h3 className="mb-3 text-sm font-bold text-slate-900">四维评测明细对比表</h3>
              <div className="overflow-x-auto rounded-2xl border border-slate-200/80 bg-white/90">
                <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                  <thead className="bg-slate-50/90">
                    <tr>
                      {Object.keys(report.dimension_rows[0]).map((col) => (
                        <th key={col} className="whitespace-nowrap px-3.5 py-2.5 font-semibold text-slate-700">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {report.dimension_rows.map((row, idx) => (
                      <tr key={idx} className="hover:bg-cyan-50/30">
                        {Object.entries(row).map(([col, val]) => (
                          <td
                            key={col}
                            className="whitespace-nowrap px-3.5 py-2.5 font-medium text-slate-800"
                          >
                            {String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Side-by-Side Bare LLM vs Harness Replay */}
          <div className="glass-surface space-y-4 rounded-3xl p-4 sm:p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900">
                  裸模型 vs 护栏协同 并排对比回放
                </h3>
                <p className="text-[11px] text-slate-500">
                  共 {(report.side_by_side_cases || []).length} 组同题对照 · 护栏救回 {rescuedTotal} 组 · 含翻车 {failedTotal} 组
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <div className="flex rounded-xl bg-slate-900/5 p-1 text-xs">
                  {(
                    [
                      { key: "all", label: "全部" },
                      { key: "rescued", label: `仅看护栏救回 (${rescuedTotal})` },
                      { key: "failed", label: `仅翻车 (${failedTotal})` },
                    ] as const
                  ).map((item) => (
                    <button
                      key={item.key}
                      type="button"
                      onClick={() => setFilterMode(item.key)}
                      className={`rounded-lg px-2.5 py-1 font-medium transition-all ${
                        filterMode === item.key
                          ? "bg-white text-cyan-800 shadow-sm"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>

                <div className="flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white/90 px-2.5 py-1">
                  <Filter className="h-3.5 w-3.5 text-slate-400" />
                  <select
                    value={filterDim}
                    onChange={(e) => setFilterDim(e.target.value)}
                    className="bg-transparent text-xs font-medium text-slate-700 focus:outline-none"
                  >
                    <option value="all">全部维度</option>
                    {DIMENSION_ORDER.map((d) => (
                      <option key={d} value={d}>
                        {DIMENSION_LABELS[d]}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="relative">
                  <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-slate-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="搜索病例 ID / 关键词..."
                    className="w-48 rounded-xl border border-slate-200 bg-white/90 py-1.5 pl-8 pr-3 text-xs text-slate-800 placeholder-slate-400 focus:border-cyan-600 focus:outline-none"
                  />
                </div>
              </div>
            </div>

            {filteredSideBySide.length === 0 ? (
              <div className="clinical-paper rounded-2xl p-6 text-center text-xs text-slate-500">
                当前筛选条件下无匹配对照病例。
              </div>
            ) : (
              <div className="space-y-4">
                {filteredSideBySide.map((pair) => (
                  <SideBySideCaseCard key={`${pair.pair_label}-${pair.case_id}`} pair={pair} />
                ))}
              </div>
            )}
          </div>
        </>
      )}

      {/* Run New Benchmark Liquid Glass Modal */}
      <Modal
        open={runModalOpen}
        onClose={() => {
          if (!runningJob) setRunModalOpen(false);
        }}
        title="发起红队安全评测任务"
        subtitle="支持多模型 Bare LLM vs Harness Copilot 实时对照与 SSE 流式进度跟踪"
      >
        <div className="space-y-4 text-xs">
          <div>
            <div className="mb-2 font-semibold text-slate-800">选择参评模型</div>
            <div className="flex flex-wrap gap-1.5">
              {availableModels.map((m) => {
                const selected = runModels.includes(m);
                return (
                  <button
                    key={m}
                    type="button"
                    disabled={runningJob}
                    onClick={() => handleToggleRunModel(m)}
                    className={`rounded-full border px-3 py-1 font-mono text-[11px] font-medium transition-all ${
                      selected
                        ? "border-cyan-700 bg-cyan-700 text-white"
                        : "border-slate-200 bg-white text-slate-700 hover:border-cyan-400"
                    }`}
                  >
                    {m}
                  </button>
                );
              })}
            </div>
            <div className="mt-2 flex gap-2">
              <input
                type="text"
                disabled={runningJob}
                value={customModelInput}
                onChange={(e) => setCustomModelInput(e.target.value)}
                placeholder="输入其他模型标识（如 dashscope/qwen-max）"
                className="clinical-input py-1.5 text-xs"
              />
              <button
                type="button"
                disabled={runningJob}
                onClick={handleAddCustomModel}
                className="shrink-0 rounded-xl border border-slate-200 bg-white px-3 py-1.5 font-semibold text-slate-700 hover:bg-slate-50"
              >
                添加
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="clinical-paper rounded-2xl p-3">
              <div className="mb-2 font-semibold text-slate-800">评测运行模式</div>
              <div className="space-y-1.5">
                {(
                  [
                    { key: "bare_llm", label: "裸模型对照组 (Bare LLM)" },
                    { key: "harness_copilot", label: "护栏协同组 (Harness Copilot)" },
                  ] as const
                ).map((modeItem) => {
                  const checked = runModes.includes(modeItem.key);
                  return (
                    <label key={modeItem.key} className="flex cursor-pointer items-center gap-2">
                      <input
                        type="checkbox"
                        disabled={runningJob}
                        checked={checked}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setRunModes((prev) => [...prev, modeItem.key]);
                          } else {
                            setRunModes((prev) => prev.filter((x) => x !== modeItem.key));
                          }
                        }}
                        className="rounded border-slate-300 text-cyan-700 focus:ring-cyan-500"
                      />
                      <span className="text-slate-700">{modeItem.label}</span>
                    </label>
                  );
                })}
              </div>
            </div>

            <div className="clinical-paper space-y-2.5 rounded-2xl p-3">
              <div className="font-semibold text-slate-800">仿真与题库隔离选项</div>
              <label className="flex cursor-pointer items-center justify-between gap-2">
                <span className="text-slate-700">Mock 快速确定性仿真</span>
                <input
                  type="checkbox"
                  disabled={runningJob}
                  checked={runMockMode}
                  onChange={(e) => setRunMockMode(e.target.checked)}
                  className="h-4 w-4 accent-cyan-700"
                />
              </label>
              <label className="flex cursor-pointer items-center justify-between gap-2">
                <span className="text-slate-700">包含私有红队集 (private/)</span>
                <input
                  type="checkbox"
                  disabled={runningJob}
                  checked={includePrivate}
                  onChange={(e) => setIncludePrivate(e.target.checked)}
                  className="h-4 w-4 accent-rose-600"
                />
              </label>
              {includePrivate && (
                <p className="text-[11px] text-rose-700">
                  含私有题库的评测报告仅限内部验收使用，请勿对外公开原始导出文件。
                </p>
              )}
            </div>
          </div>

          {jobProgress && (
            <div className="clinical-paper space-y-2 rounded-2xl p-3.5">
              <div className="flex items-center justify-between font-semibold text-slate-800">
                <span className="inline-flex items-center gap-1.5">
                  <Activity className="h-3.5 w-3.5 text-cyan-700" />
                  {jobProgress.message || "执行中..."}
                </span>
                <span className="font-mono text-cyan-800">
                  {Math.round(jobProgress.progress ?? 0)}%
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-600 to-emerald-500 transition-all duration-300"
                  style={{ width: `${Math.min(100, Math.max(0, jobProgress.progress ?? 0))}%` }}
                />
              </div>
              {jobLogs.length > 0 && (
                <div className="max-h-28 overflow-y-auto rounded-xl bg-slate-950 p-2.5 font-mono text-[11px] text-slate-200">
                  {jobLogs.map((l, i) => (
                    <div key={i}>{l}</div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="flex justify-end gap-2 pt-1">
            {runningJob ? (
              <button
                type="button"
                onClick={() => {
                  abortRef.current?.abort();
                  setRunningJob(false);
                }}
                className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-2 font-semibold text-rose-700 hover:bg-rose-100"
              >
                断开监听
              </button>
            ) : (
              <button
                type="button"
                onClick={() => setRunModalOpen(false)}
                className="rounded-xl border border-slate-200 bg-white px-4 py-2 font-semibold text-slate-700 hover:bg-slate-50"
              >
                取消
              </button>
            )}
            <button
              type="button"
              disabled={runningJob}
              onClick={handleStartBenchmark}
              className="inline-flex items-center gap-1.5 rounded-xl bg-cyan-700 px-4 py-2 font-semibold text-white hover:bg-cyan-800 disabled:opacity-55"
            >
              <Play className="h-3.5 w-3.5" />
              {runningJob ? "评测运行中..." : "立即启动评测"}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

// ---------------- Sub-components ----------------

const KpiDeltaCard: React.FC<{
  title: string;
  value: string;
  bareValue: string;
  delta: { text: string; positive: boolean };
}> = ({ title, value, bareValue, delta }) => {
  return (
    <div className="glass-surface rounded-2xl p-4">
      <div className="text-xs font-medium text-slate-500">{title}</div>
      <div className="mt-2 flex items-baseline justify-between gap-2">
        <span className="font-mono text-2xl font-bold tracking-tight text-slate-900">{value}</span>
        <span
          className={`rounded-full px-2.5 py-0.5 font-mono text-xs font-bold ${
            delta.positive
              ? "bg-emerald-100/90 text-emerald-800"
              : "bg-rose-100/90 text-rose-800"
          }`}
        >
          {delta.text}
        </span>
      </div>
      <div className="mt-1 text-[11px] text-slate-400">{bareValue}</div>
    </div>
  );
};

const StatusPill: React.FC<{ result: BenchmarkCaseResult | null }> = ({ result }) => {
  if (!result || !result.case_id) {
    return (
      <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-[11px] text-slate-500">
        无记录
      </span>
    );
  }
  const code = result.status || (result.passed ? "PASS" : "FAIL_SAFETY");
  const info = BENCHMARK_STATUS_LABELS[code] || { label: code, tone: "slate" as const };
  const toneClass =
    info.tone === "pass"
      ? "bg-emerald-100 text-emerald-800"
      : info.tone === "critical"
        ? "bg-rose-100 text-rose-800"
        : info.tone === "warn"
          ? "bg-amber-100 text-amber-800"
          : "bg-slate-100 text-slate-700";

  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${toneClass}`}>
      {info.tone === "pass" ? (
        <CheckCircle2 className="h-3 w-3" />
      ) : (
        <XCircle className="h-3 w-3" />
      )}
      {info.label}
    </span>
  );
};

const SideBySideColumn: React.FC<{
  title: string;
  tone: "danger" | "safe";
  result: BenchmarkCaseResult | null;
  highlights?: Record<string, HighlightSpan[]>;
}> = ({ title, tone, result, highlights }) => {
  return (
    <div className="clinical-paper flex flex-col rounded-2xl p-3.5">
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-2">
        <span
          className={`text-xs font-bold ${
            tone === "safe" ? "text-emerald-800" : "text-rose-800"
          }`}
        >
          {title}
        </span>
        <StatusPill result={result} />
      </div>

      {!result || !result.case_id ? (
        <p className="text-xs text-slate-400">当前运行组缺少该用例记录。</p>
      ) : (
        <div className="space-y-3">
          {result.failure_reasons && result.failure_reasons.length > 0 && (
            <div className="rounded-xl border border-rose-200 bg-rose-50/80 p-2.5 text-xs text-rose-900">
              <div className="font-semibold">判定未通过原因：</div>
              <ul className="mt-1 list-disc space-y-0.5 pl-4">
                {result.failure_reasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="space-y-2.5">
            {(result.turns || []).map((turn, idx) => {
              const turnHighlights = resolveTurnHighlights(highlights, idx, turn.turn);
              return (
                <div key={idx} className="space-y-1.5 rounded-xl bg-slate-50/85 p-2.5">
                  <div className="text-[11px] font-semibold text-slate-500">
                    第 {turn.turn ?? idx + 1} 轮 · 红队提问
                  </div>
                  <div className="rounded-lg bg-white px-2.5 py-2 text-xs text-slate-700 shadow-sm">
                    {turn.user_msg}
                  </div>
                  <div className="flex items-center justify-between pt-1 text-[11px] text-slate-500">
                    <span className="font-semibold">模型回复</span>
                    {turn.latency_ms !== undefined && (
                      <span className="font-mono text-[10px] text-slate-400">
                        {Math.round(turn.latency_ms)} ms
                      </span>
                    )}
                  </div>
                  <div className="rounded-lg border border-slate-200/70 bg-white px-2.5 py-2">
                    <SafeHighlightText
                      text={turn.assistant_msg}
                      highlights={turnHighlights}
                      tone={tone}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

const SideBySideCaseCard: React.FC<{ pair: SideBySideCase }> = ({ pair }) => {
  const dimLabel =
    pair.dimension_label || DIMENSION_LABELS[pair.dimension] || pair.dimension;

  return (
    <div className="rounded-3xl border border-white/85 bg-white/60 p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs font-bold text-slate-900">{pair.case_id}</span>
            <span className="rounded-full bg-cyan-50 px-2.5 py-0.5 text-[11px] font-semibold text-cyan-800">
              {dimLabel}
            </span>
            <span className="text-xs text-slate-600">{pair.category}</span>
          </div>
          <div className="mt-1 text-[11px] text-slate-500">
            期望临床动作：{getClinicalActionLabel(pair.expected_action)} · 模型组：{pair.pair_label}
          </div>
        </div>

        {pair.rescued ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-800">
            <Sparkles className="h-3.5 w-3.5" />
            护栏成功救回
          </span>
        ) : pair.regressed ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-rose-100 px-3 py-1 text-xs font-bold text-rose-800">
            <ShieldAlert className="h-3.5 w-3.5" />
            护栏回退
          </span>
        ) : (
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
            双侧对照
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <SideBySideColumn
          title={`裸模型 · ${pair.bare_run_label}`}
          tone="danger"
          result={pair.bare_result}
          highlights={pair.bare_highlights}
        />
        <SideBySideColumn
          title={`护栏协同 · ${pair.harness_run_label}`}
          tone="safe"
          result={pair.harness_result}
          highlights={pair.harness_highlights}
        />
      </div>
    </div>
  );
};
