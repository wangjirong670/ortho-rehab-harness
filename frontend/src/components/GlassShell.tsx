import React, { useState } from "react";
import {
  Activity,
  Lock,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
} from "lucide-react";
import type { HealthResponse } from "../api/types";
import { Modal } from "./Modal";
import { ModeCapsule, WorkspaceMode } from "./ModeCapsule";

interface GlassShellProps {
  workspaceMode: WorkspaceMode;
  settingsOpen: boolean;
  onSelectWorkspaceMode: (mode: WorkspaceMode) => void;
  onOpenSettings: () => void;
  sidebarSlot?: React.ReactNode;
  headerActions?: React.ReactNode;
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  activeModel: string;
  mockMode: boolean;
  complianceNotice: string;
  onRefreshHealth: () => void;
  children: React.ReactNode;
}

export const GlassShell: React.FC<GlassShellProps> = ({
  workspaceMode,
  settingsOpen,
  onSelectWorkspaceMode,
  onOpenSettings,
  sidebarSlot,
  headerActions,
  health,
  healthLoading,
  healthError,
  activeModel,
  mockMode,
  complianceNotice,
  onRefreshHealth,
  children,
}) => {
  const [complianceOpen, setComplianceOpen] = useState(false);

  const isConnected = Boolean(health && health.status === "ok" && !healthError);
  const effectiveModel = activeModel || health?.default_model || "未配置模型";
  const effectiveMock = health ? health.mock_mode || mockMode : mockMode;

  if (settingsOpen) {
    return (
      <div className="min-h-screen px-3 py-3 sm:px-5 sm:py-4 lg:px-6 lg:py-5">
        <div className="mx-auto max-w-[1440px]">{children}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-3 py-3 sm:px-4 sm:py-4">
      <div className="mx-auto flex max-w-[1720px] flex-col gap-3 md:flex-row md:items-stretch">
        {/* Left Liquid Glass Sidebar: Top ModeCapsule + Middle Slot + Bottom Settings */}
        <aside className="glass-surface flex w-full shrink-0 flex-col rounded-3xl p-3 md:sticky md:top-4 md:h-[calc(100vh-32px)] md:w-64 xl:w-72">
          <div className="flex items-center gap-2.5 px-1.5 pb-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-600 to-teal-700 text-white shadow-md shadow-cyan-900/15">
              <Activity className="h-4.5 w-4.5" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-bold tracking-tight text-slate-900">
                OrthoRehab 护栏
              </div>
              <div className="truncate text-[11px] font-medium text-cyan-800/80">
                v{health?.version || "0.4.0"} · 骨伤临床工作台
              </div>
            </div>
          </div>

          <ModeCapsule mode={workspaceMode} onChange={onSelectWorkspaceMode} />

          {/* Left-Middle Dynamic Workspace Slot (PatientExplorer or Info) */}
          <div className="mt-3 flex min-h-0 flex-1 flex-col overflow-hidden">
            {sidebarSlot || (
              <div className="rounded-2xl border border-white/70 bg-white/55 p-3.5 text-xs text-slate-600">
                <div className="font-semibold text-slate-800">红队安全评测沙箱</div>
                <p className="mt-1.5 leading-relaxed text-slate-500">
                  支持裸模型与统一 <code>run_turn()</code> 护栏内核并排红队对比、20 道标准题集运行与高亮回放。
                </p>
              </div>
            )}
          </div>

          {/* Left-Bottom Settings Entry (Whole-page switch) */}
          <div className="mt-3 border-t border-slate-200/70 pt-2.5">
            <button
              type="button"
              onClick={onOpenSettings}
              data-testid="open-settings-btn"
              className="flex w-full items-center justify-between rounded-2xl border border-white/80 bg-white/75 px-3 py-2.5 text-left text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-white hover:text-slate-900"
            >
              <span className="flex items-center gap-2">
                <SlidersHorizontal className="h-4 w-4 text-cyan-700" />
                <span>设置 & 只读知识库</span>
              </span>
              <span className="text-[10px] text-slate-400">全页</span>
            </button>
          </div>
        </aside>

        {/* Main Viewport */}
        <div className="flex min-w-0 flex-1 flex-col space-y-3">
          {/* Top Compact Floating Status & Action Bar */}
          <header className="flex flex-wrap items-center justify-between gap-2.5">
            <div className="glass-pill flex flex-wrap items-center gap-2 rounded-full px-3.5 py-1.5 text-xs">
              <button
                type="button"
                onClick={onRefreshHealth}
                title={healthError ? `连接异常：${healthError}` : "后端服务已连通（点击刷新）"}
                className="flex items-center gap-1.5 rounded-full px-2 py-0.5 font-medium text-slate-700 transition-colors hover:bg-white/80"
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    healthLoading
                      ? "animate-pulse bg-amber-500"
                      : isConnected
                        ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.7)]"
                        : "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]"
                  }`}
                />
                <span>{isConnected ? "服务在线" : healthLoading ? "检测中" : "服务离线"}</span>
                <RefreshCw
                  className={`h-3 w-3 text-slate-400 ${healthLoading ? "animate-spin" : ""}`}
                />
              </button>

              <span className="h-3.5 w-px bg-slate-200" />

              <span
                className={`rounded-full px-2.5 py-0.5 font-semibold ${
                  effectiveMock
                    ? "bg-amber-100/90 text-amber-800"
                    : "bg-emerald-100/90 text-emerald-800"
                }`}
              >
                {effectiveMock ? "Mock 仿真" : "Live 推理"}
              </span>

              <span
                className="hidden max-w-[200px] truncate rounded-full bg-slate-900/5 px-2.5 py-0.5 font-mono text-[11px] font-medium text-slate-700 sm:inline-block"
                title={effectiveModel}
              >
                {effectiveModel}
              </span>

              {health?.auth_enabled && (
                <span className="inline-flex items-center gap-1 rounded-full bg-cyan-50 px-2 py-0.5 text-[11px] font-medium text-cyan-800">
                  <Lock className="h-3 w-3" />
                  Bearer
                </span>
              )}

              <button
                type="button"
                onClick={() => setComplianceOpen(true)}
                className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2.5 py-0.5 text-[11px] font-medium text-slate-600 transition-colors hover:bg-cyan-50 hover:text-cyan-800"
              >
                <ShieldCheck className="h-3 w-3 text-cyan-700" />
                合规边界
              </button>
            </div>

            {headerActions && <div className="flex items-center gap-2">{headerActions}</div>}
          </header>

          <main className="min-w-0 flex-1">{children}</main>
        </div>
      </div>

      <Modal
        open={complianceOpen}
        onClose={() => setComplianceOpen(false)}
        title="临床安全与系统边界声明"
        maxWidthClass="max-w-lg"
      >
        <div className="space-y-3 text-xs leading-relaxed text-slate-600">
          <p className="rounded-2xl border border-amber-200/80 bg-amber-50/70 p-3 text-amber-950">
            {complianceNotice ||
              "本系统仅用于临床脱敏病历信息整理、确定性安全红旗排查与骨伤康复思维辅助，面向执业医师/实习生/规培医师使用；所有输出结果均不能替代临床诊疗诊断、处方权或手术医师决策。"}
          </p>
          <ul className="list-disc space-y-1.5 pl-4">
            <li>高危红旗征、停药洗脱窗口期、量表算分与输出安全复检全部由确定性 Python 工具引擎执行。</li>
            <li>患者档案卡 PATIENT.md 中的 display_name 仅用于界面显示，组装模型上下文时确定性强制剔除。</li>
          </ul>
        </div>
      </Modal>
    </div>
  );
};
