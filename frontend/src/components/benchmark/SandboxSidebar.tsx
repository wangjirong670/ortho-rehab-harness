import React, { useMemo, useRef, useState } from "react";
import {
  FileSpreadsheet,
  Flame,
  PlusCircle,
  RefreshCw,
  Search,
  ShieldCheck,
  Upload,
} from "lucide-react";
import type { BenchmarkReportMeta } from "../../api/types";

export type SandboxSubMode = "batch" | "free";

interface SandboxSidebarProps {
  subMode: SandboxSubMode;
  onChangeSubMode: (mode: SandboxSubMode) => void;
  reports: BenchmarkReportMeta[];
  selectedFilename: string;
  loadingReports: boolean;
  onSelectReport: (filename: string) => void;
  onRefreshReports: () => void;
  onImportReport: (file: File) => void;
  freeTurnCount: number;
  freeBreachCount: number;
  onOpenSaveCaseModal: () => void;
}

export const SandboxSidebar: React.FC<SandboxSidebarProps> = ({
  subMode,
  onChangeSubMode,
  reports,
  selectedFilename,
  loadingReports,
  onSelectReport,
  onRefreshReports,
  onImportReport,
  freeTurnCount,
  freeBreachCount,
  onOpenSaveCaseModal,
}) => {
  const [query, setQuery] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const filteredReports = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return reports;
    return reports.filter(
      (r) =>
        r.label.toLowerCase().includes(needle) ||
        r.filename.toLowerCase().includes(needle) ||
        (r.models || []).some((m) => m.toLowerCase().includes(needle))
    );
  }, [reports, query]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Sub-mode Toggle: Batch Benchmark vs Free Red-Team */}
      <div className="mb-2.5 grid grid-cols-2 gap-1 rounded-2xl border border-white/80 bg-slate-900/5 p-1 text-[11px]">
        <button
          type="button"
          data-testid="sandbox-tab-batch"
          onClick={() => onChangeSubMode("batch")}
          className={`flex items-center justify-center gap-1 rounded-xl py-1.5 font-semibold transition ${
            subMode === "batch"
              ? "bg-white text-cyan-900 shadow-sm"
              : "text-slate-600 hover:text-slate-900"
          }`}
        >
          <ShieldCheck className="h-3.5 w-3.5" />
          <span>批量评测</span>
        </button>

        <button
          type="button"
          data-testid="sandbox-tab-free"
          onClick={() => onChangeSubMode("free")}
          className={`flex items-center justify-center gap-1 rounded-xl py-1.5 font-semibold transition ${
            subMode === "free"
              ? "bg-white text-rose-800 shadow-sm"
              : "text-slate-600 hover:text-slate-900"
          }`}
        >
          <Flame className="h-3.5 w-3.5" />
          <span>自由红队</span>
        </button>
      </div>

      {subMode === "batch" ? (
        <>
          <div className="mb-2 flex items-center justify-between gap-1 px-1">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              评测报告 ({reports.length})
            </span>
            <div className="flex items-center gap-1">
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  e.target.value = "";
                  if (file) onImportReport(file);
                }}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                title="导入外部报告 JSON"
                className="rounded-lg border border-slate-200 bg-white/80 p-1 text-slate-600 hover:bg-white"
              >
                <Upload className="h-3.5 w-3.5" />
              </button>
              <button
                type="button"
                onClick={onRefreshReports}
                title="刷新评测记录"
                className="rounded-lg border border-slate-200 bg-white/80 p-1 text-slate-600 hover:bg-white"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${loadingReports ? "animate-spin" : ""}`} />
              </button>
            </div>
          </div>

          <div className="relative mb-2">
            <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-slate-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="检索报告 / 模型..."
              className="w-full rounded-xl border border-slate-200/90 bg-white/85 py-1.5 pl-8 pr-2.5 text-xs text-slate-800 placeholder-slate-400 focus:border-cyan-600/60 focus:outline-none"
            />
          </div>

          <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto pr-0.5">
            {filteredReports.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200/80 bg-white/50 p-4 text-center text-xs text-slate-400">
                暂无历史评测报告，可在底部配置模型后发起新一轮评测。
              </div>
            ) : (
              filteredReports.map((r) => {
                const active = selectedFilename === r.filename;
                return (
                  <button
                    key={r.filename}
                    type="button"
                    onClick={() => onSelectReport(r.filename)}
                    className={`w-full rounded-2xl border px-3 py-2 text-left transition ${
                      active
                        ? "border-cyan-300 bg-gradient-to-r from-cyan-700 to-cyan-600 text-white shadow-sm"
                        : "border-white/80 bg-white/65 text-slate-800 hover:bg-white"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1.5">
                      <span className="flex min-w-0 items-center gap-1.5">
                        <FileSpreadsheet className="h-3.5 w-3.5 shrink-0 opacity-80" />
                        <span className="truncate text-xs font-semibold">
                          {r.label || r.filename}
                        </span>
                      </span>
                      <span
                        className={`shrink-0 rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${
                          active
                            ? "bg-white/20 text-white"
                            : r.mock_mode
                              ? "bg-amber-100 text-amber-800"
                              : "bg-emerald-100 text-emerald-800"
                        }`}
                      >
                        {r.mock_mode ? "Mock" : "Live"}
                      </span>
                    </div>
                    <div
                      className={`mt-1 truncate text-[10px] ${
                        active ? "text-cyan-100" : "text-slate-500"
                      }`}
                    >
                      {r.case_count} 题 · {(r.models || []).join(", ") || "-"}
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </>
      ) : (
        <div className="space-y-3 rounded-2xl border border-white/80 bg-white/60 p-3 text-xs">
          <div className="font-bold text-slate-900">自由红队攻防对照（C 模式）</div>
          <p className="leading-relaxed text-slate-600">
            同一诱导话术同时并发打给左栏 <strong>裸模型</strong> 与右栏{" "}
            <strong>统一 run_turn() 护栏</strong>。若右栏仍未守住，立即点击下方沉淀为私有用例补丁。
          </p>

          <div className="grid grid-cols-2 gap-2 text-center">
            <div className="rounded-xl bg-white/90 p-2">
              <div className="text-[10px] text-slate-400">测试轮数</div>
              <div className="mt-0.5 font-mono text-base font-bold text-slate-900">
                {freeTurnCount}
              </div>
            </div>
            <div className="rounded-xl bg-white/90 p-2">
              <div className="text-[10px] text-slate-400">左栏失守轮</div>
              <div className="mt-0.5 font-mono text-base font-bold text-rose-700">
                {freeBreachCount}
              </div>
            </div>
          </div>

          <button
            type="button"
            disabled={freeTurnCount === 0}
            onClick={onOpenSaveCaseModal}
            data-testid="sidebar-save-redteam-case-btn"
            className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl bg-rose-700 px-3 py-2 text-xs font-semibold text-white shadow-sm hover:bg-rose-800 disabled:cursor-not-allowed disabled:opacity-45"
          >
            <PlusCircle className="h-3.5 w-3.5" />
            沉淀当前话术为私有用例
          </button>
        </div>
      )}
    </div>
  );
};
