import React, { useState } from "react";
import {
  Activity,
  BookOpenCheck,
  Lock,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Stethoscope,
} from "lucide-react";
import type { HealthResponse, PageKey } from "../api/types";
import { Modal } from "./Modal";

interface GlassShellProps {
  activePage: PageKey;
  onSelectPage: (page: PageKey) => void;
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  activeModel: string;
  mockMode: boolean;
  complianceNotice: string;
  onRefreshHealth: () => void;
  children: React.ReactNode;
}

const NAV_ITEMS: Array<{
  key: PageKey;
  label: string;
  desc: string;
  icon: React.FC<{ className?: string }>;
}> = [
  {
    key: "copilot",
    label: "临床副驾",
    desc: "双轨红旗扫描 · 量表 · 文书",
    icon: Stethoscope,
  },
  {
    key: "benchmark",
    label: "红队评测沙箱",
    desc: "裸模型 vs 护栏协同对照",
    icon: ShieldCheck,
  },
  {
    key: "kb",
    label: "医学知识库",
    desc: "红旗征 · 洗脱期 · 量表规范",
    icon: BookOpenCheck,
  },
  {
    key: "settings",
    label: "模型与设置",
    desc: "多模型路由 · 服务地址鉴权",
    icon: SlidersHorizontal,
  },
];

export const GlassShell: React.FC<GlassShellProps> = ({
  activePage,
  onSelectPage,
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

  return (
    <div className="min-h-screen px-3 py-3 sm:px-5 sm:py-4 lg:px-6 lg:py-5">
      <div className="mx-auto flex max-w-[1640px] flex-col gap-4 lg:flex-row lg:items-start">
        {/* Left Liquid Glass Sidebar */}
        <aside className="glass-surface w-full shrink-0 rounded-3xl p-4 lg:sticky lg:top-5 lg:w-64">
          <div className="flex items-center gap-3 px-2 py-1.5">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-600 to-teal-700 text-white shadow-md shadow-cyan-900/15">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <div className="text-sm font-bold tracking-tight text-slate-900">
                OrthoRehab 护栏
              </div>
              <div className="text-[11px] font-medium text-cyan-800/80">
                骨伤康复确定性安全系统
              </div>
            </div>
          </div>

          <nav className="mt-5 grid grid-cols-2 gap-1.5 lg:grid-cols-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = activePage === item.key;
              return (
                <button
                  key={item.key}
                  type="button"
                  onClick={() => onSelectPage(item.key)}
                  className={`group flex items-center gap-3 rounded-2xl px-3.5 py-2.5 text-left transition-all duration-200 ${
                    active
                      ? "bg-gradient-to-r from-cyan-700 to-cyan-600 text-white shadow-md shadow-cyan-900/15"
                      : "text-slate-700 hover:bg-white/75 hover:text-slate-900"
                  }`}
                >
                  <Icon
                    className={`h-4 w-4 shrink-0 transition-transform duration-200 group-hover:scale-105 ${
                      active ? "text-cyan-100" : "text-cyan-700"
                    }`}
                  />
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold">{item.label}</div>
                    <div
                      className={`hidden truncate text-[11px] lg:block ${
                        active ? "text-cyan-100/90" : "text-slate-500"
                      }`}
                    >
                      {item.desc}
                    </div>
                  </div>
                </button>
              );
            })}
          </nav>

          <div className="mt-5 hidden rounded-2xl border border-white/70 bg-white/55 p-3.5 text-xs text-slate-600 lg:block">
            <div className="flex items-center gap-1.5 font-semibold text-slate-800">
              <ShieldAlert className="h-3.5 w-3.5 text-cyan-700" />
              临床安全原则
            </div>
            <p className="mt-1.5 leading-relaxed text-slate-600">
              红旗禁忌、中西药洗脱期与量表算分均由确定性规则引擎兜底，不依赖大模型自由发挥。所有文书仅作脱敏辅助，需执业医师复核签字。
            </p>
            {health?.version && (
              <div className="mt-2.5 flex items-center justify-between border-t border-slate-200/70 pt-2 text-[11px] text-slate-500">
                <span>引擎版本</span>
                <span className="font-mono font-medium text-slate-700">v{health.version}</span>
              </div>
            )}
          </div>
        </aside>

        {/* Main Viewport */}
        <div className="min-w-0 flex-1 space-y-4">
          {/* Top Floating Status Pill */}
          <header className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-lg font-bold tracking-tight text-slate-900 sm:text-xl">
                {NAV_ITEMS.find((n) => n.key === activePage)?.label}
              </h1>
              <p className="text-xs text-slate-500">
                {NAV_ITEMS.find((n) => n.key === activePage)?.desc}
              </p>
            </div>

            <div className="glass-pill flex flex-wrap items-center gap-2 rounded-full px-3.5 py-1.5 text-xs">
              {/* Backend Connectivity Dot */}
              <button
                type="button"
                onClick={onRefreshHealth}
                title={healthError ? `连接异常：${healthError}（点击重试）` : "后端服务已连通（点击刷新）"}
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
                <RefreshCw className={`h-3 w-3 text-slate-400 ${healthLoading ? "animate-spin" : ""}`} />
              </button>

              <span className="h-3.5 w-px bg-slate-200" />

              {/* Current Model */}
              <div className="flex items-center gap-1.5 px-1 text-slate-700">
                <span className="text-slate-400">模型</span>
                <span className="max-w-[180px] truncate font-mono text-[11px] font-semibold text-cyan-800">
                  {effectiveModel}
                </span>
              </div>

              <span className="h-3.5 w-px bg-slate-200" />

              {/* Mock / Live Badge */}
              <span
                className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                  effectiveMock
                    ? "bg-amber-100/90 text-amber-800"
                    : "bg-emerald-100/90 text-emerald-800"
                }`}
              >
                {effectiveMock ? "Mock 仿真" : "Live 真实推理"}
              </span>

              {health?.auth_enabled && (
                <>
                  <span className="h-3.5 w-px bg-slate-200" />
                  <span className="inline-flex items-center gap-1 rounded-full bg-cyan-50 px-2 py-0.5 text-[11px] font-medium text-cyan-800">
                    <Lock className="h-3 w-3" />
                    令牌鉴权
                  </span>
                </>
              )}

              <span className="h-3.5 w-px bg-slate-200" />

              <button
                type="button"
                onClick={() => setComplianceOpen(true)}
                className="rounded-full bg-slate-900/5 px-2.5 py-0.5 font-medium text-slate-700 transition-colors hover:bg-cyan-700 hover:text-white"
              >
                合规声明
              </button>
            </div>
          </header>

          {healthError && (
            <div className="glass-danger flex flex-wrap items-center justify-between gap-2 rounded-2xl px-4 py-2.5 text-xs text-rose-900">
              <span>
                <strong>后端服务未连通：</strong>
                {healthError}（若使用自定义远程后端或 Bearer Token，请前往「模型与设置」配置）
              </span>
              <button
                type="button"
                onClick={() => onSelectPage("settings")}
                className="rounded-xl bg-rose-700 px-3 py-1 font-medium text-white transition-colors hover:bg-rose-800"
              >
                前往设置
              </button>
            </div>
          )}

          <main>{children}</main>
        </div>
      </div>

      <Modal
        open={complianceOpen}
        onClose={() => setComplianceOpen(false)}
        title="临床合规与数据隐私声明"
        subtitle="OrthoRehab 中医骨伤与康复科安全护栏工作台"
      >
        <div className="space-y-4 text-sm leading-relaxed text-slate-700">
          <div className="clinical-paper rounded-2xl p-4">
            <p className="whitespace-pre-wrap">
              {complianceNotice ||
                "1. 本系统仅提供临床信息结构化整理、确定性安全红旗筛查及量表数值核算辅助，不构成独立医疗诊断或处方指令。\n2. 严禁将未脱敏的患者姓名、身份证号、手机号、住院号等个人隐私信息发送至公网大模型；请优先使用工作台「一键脱敏」或本地 Ollama 模型。\n3. 所有查房提纲、康复宣教及红旗拦截提示均须经执业医师结合患者实时查体与影像学复核确认。"}
            </p>
          </div>
          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => setComplianceOpen(false)}
              className="rounded-xl bg-cyan-700 px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-cyan-800"
            >
              我已了解
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
};
