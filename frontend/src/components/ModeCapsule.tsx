import React from "react";
import { ShieldCheck, Stethoscope } from "lucide-react";

export type WorkspaceMode = "copilot" | "benchmark";

interface ModeCapsuleProps {
  mode: WorkspaceMode;
  onChange: (mode: WorkspaceMode) => void;
}

export const ModeCapsule: React.FC<ModeCapsuleProps> = ({ mode, onChange }) => {
  return (
    <div
      className="flex rounded-2xl border border-white/80 bg-slate-900/5 p-1 shadow-inner"
      role="tablist"
      aria-label="工作台主模式切换"
    >
      <button
        type="button"
        role="tab"
        aria-selected={mode === "copilot"}
        onClick={() => onChange("copilot")}
        className={`flex flex-1 items-center justify-center gap-1.5 rounded-xl px-2.5 py-2 text-xs font-semibold transition-all duration-150 ${
          mode === "copilot"
            ? "bg-gradient-to-r from-cyan-700 to-cyan-600 text-white shadow-sm"
            : "text-slate-600 hover:bg-white/70 hover:text-slate-900"
        }`}
      >
        <Stethoscope className="h-3.5 w-3.5 shrink-0" />
        <span>临床副驾</span>
      </button>

      <button
        type="button"
        role="tab"
        aria-selected={mode === "benchmark"}
        onClick={() => onChange("benchmark")}
        className={`flex flex-1 items-center justify-center gap-1.5 rounded-xl px-2.5 py-2 text-xs font-semibold transition-all duration-150 ${
          mode === "benchmark"
            ? "bg-gradient-to-r from-cyan-700 to-cyan-600 text-white shadow-sm"
            : "text-slate-600 hover:bg-white/70 hover:text-slate-900"
        }`}
      >
        <ShieldCheck className="h-3.5 w-3.5 shrink-0" />
        <span>红队沙箱</span>
      </button>
    </div>
  );
};
