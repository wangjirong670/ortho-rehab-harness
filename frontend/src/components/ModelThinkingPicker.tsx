import React, { useEffect, useMemo, useRef, useState } from "react";
import { Brain, Check, ChevronDown, ChevronRight, Plus } from "lucide-react";
import { api } from "../api/client";
import type { ModelCapability, ThinkingLevel } from "../api/types";

const THINKING_OPTIONS: Array<{ key: ThinkingLevel; label: string }> = [
  { key: "off", label: "关" },
  { key: "low", label: "低" },
  { key: "medium", label: "中" },
  { key: "high", label: "高" },
];

const THINKING_LABEL_MAP: Record<ThinkingLevel, string> = {
  off: "关",
  low: "低",
  medium: "中",
  high: "高",
};

interface ModelThinkingPickerProps {
  mode?: "single";
  direction?: "up" | "down";
  availableModels: string[];
  selectedModels: string[];
  modelCapabilities?: Record<string, ModelCapability>;
  modelThinking: Record<string, ThinkingLevel>;
  mockMode: boolean;
  disabled?: boolean;
  disabledTooltip?: string;
  testId?: string;
  onChangeModels: (nextModels: string[]) => void;
  onChangeThinking: (model: string, level: ThinkingLevel) => void;
}

export const ModelThinkingPicker: React.FC<ModelThinkingPickerProps> = ({
  direction = "up",
  availableModels,
  selectedModels,
  modelCapabilities = {},
  modelThinking,
  mockMode,
  disabled = false,
  disabledTooltip,
  testId = "model-thinking-picker",
  onChangeModels,
  onChangeThinking,
}) => {
  const [open, setOpen] = useState(false);
  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  const [customModelInput, setCustomModelInput] = useState("");
  const [customModels, setCustomModels] = useState<string[]>([]);
  const [dynamicCaps, setDynamicCaps] = useState<Record<string, ModelCapability>>({});
  const containerRef = useRef<HTMLDivElement | null>(null);

  const modelChoices = useMemo(() => {
    return Array.from(
      new Set([...availableModels, ...selectedModels, ...customModels].filter(Boolean))
    );
  }, [availableModels, selectedModels, customModels]);

  // Close popover on outside click or Escape
  useEffect(() => {
    if (!open) return;
    const onMouseDown = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  // Auto-probe capabilities for models not included in initial settings payload
  useEffect(() => {
    const missing = modelChoices.filter(
      (m) => !modelCapabilities[m] && !dynamicCaps[m]
    );
    if (missing.length === 0) return;
    missing.forEach((m) => {
      api
        .getModelCapabilities(m)
        .then((cap) => {
          setDynamicCaps((prev) => (prev[m] ? prev : { ...prev, [m]: cap }));
        })
        .catch(() => {
          // non-fatal
        });
    });
  }, [dynamicCaps, modelCapabilities, modelChoices]);

  const resolveCap = (model: string): ModelCapability | undefined =>
    modelCapabilities[model] || dynamicCaps[model];

  const getEffectiveLevel = (model: string): ThinkingLevel => {
    const cap = resolveCap(model);
    const chosen = modelThinking[model] || "off";
    if (mockMode) return chosen;
    if (!cap || !cap.supports_thinking) return "off";
    if (!cap.supported_levels.includes(chosen)) {
      return cap.default_level || cap.supported_levels[0] || "off";
    }
    return chosen;
  };

  const handleToggleSelectModel = (model: string) => {
    onChangeModels([model]);
    setExpandedModel(model);
  };

  const handleAddCustomModel = () => {
    const trimmed = customModelInput.trim();
    if (!trimmed) return;
    setCustomModels((prev) => (prev.includes(trimmed) ? prev : [...prev, trimmed]));
    onChangeModels([trimmed]);
    setExpandedModel(trimmed);
    setCustomModelInput("");
  };

  const primaryModel = selectedModels[0] || "";
  const primaryLevel = primaryModel ? getEffectiveLevel(primaryModel) : "off";

  const triggerText = primaryModel || "选择模型";

  return (
    <div ref={containerRef} className="relative inline-block">
      <button
        type="button"
        data-testid={testId}
        disabled={disabled}
        title={
          disabled && disabledTooltip
            ? disabledTooltip
            : primaryModel
              ? `${primaryModel}（思考：${THINKING_LABEL_MAP[primaryLevel]}）`
              : "选择模型并展开配置思考强度"
        }
        onClick={() => !disabled && setOpen((prev) => !prev)}
        className="inline-flex max-w-[280px] items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-2.5 py-1.5 font-mono text-[11px] font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
      >
        <span className="truncate">{triggerText}</span>
        {primaryModel && (
          <span
            data-testid={`${testId}-thinking-pill`}
            className={`inline-flex shrink-0 items-center gap-0.5 rounded-md px-1.5 py-0.5 font-sans text-[10px] font-semibold ${
              primaryLevel !== "off"
                ? "bg-cyan-50 text-cyan-800"
                : "bg-slate-100 text-slate-500"
            }`}
          >
            <Brain className="h-2.5 w-2.5" />
            {THINKING_LABEL_MAP[primaryLevel]}
          </span>
        )}
        <ChevronDown className="h-3.5 w-3.5 shrink-0 text-slate-400" />
      </button>

      {open && !disabled && (
        <div
          className={`absolute left-0 z-40 w-[340px] max-w-[92vw] rounded-2xl border border-slate-200 bg-white p-2.5 shadow-2xl ${
            direction === "up" ? "bottom-full mb-2" : "top-full mt-2"
          }`}
        >
          <div className="mb-1.5 flex items-center justify-between px-1 text-[11px]">
            <span className="font-semibold text-slate-700">
              选择模型（右侧可展开配置思考等级）
            </span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="shrink-0 rounded-lg px-2 py-0.5 font-semibold text-cyan-700 hover:bg-cyan-50"
            >
              完成
            </button>
          </div>

          <div className="max-h-64 space-y-1 overflow-y-auto pr-0.5">
            {modelChoices.map((m) => {
              const isSelected = selectedModels.includes(m);
              const cap = resolveCap(m);
              const nativeSupported = Boolean(cap?.supports_thinking);
              const canAdjustThinking = mockMode || nativeSupported;
              const isExpanded = isSelected && expandedModel === m;
              const level = getEffectiveLevel(m);
              const allowed = mockMode
                ? (["off", "low", "medium", "high"] as ThinkingLevel[])
                : cap?.supported_levels || ["off"];

              return (
                <div
                  key={m}
                  className={`rounded-xl border transition ${
                    isSelected
                      ? "border-cyan-200 bg-cyan-50/70"
                      : "border-transparent hover:bg-slate-50"
                  }`}
                >
                  <div className="flex items-center justify-between gap-1.5 px-2 py-1.5">
                    <button
                      type="button"
                      onClick={() => handleToggleSelectModel(m)}
                      className="flex min-w-0 flex-1 items-center gap-2 text-left"
                    >
                      <span
                        className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full border ${
                          isSelected
                            ? "border-cyan-600 bg-cyan-600 text-white"
                            : "border-slate-300 bg-white"
                        }`}
                      >
                        {isSelected && <Check className="h-2.5 w-2.5" />}
                      </span>
                      <span
                        className={`truncate font-mono text-[11px] ${
                          isSelected ? "font-semibold text-cyan-950" : "text-slate-700"
                        }`}
                      >
                        {m}
                      </span>
                      {nativeSupported && (
                        <span className="shrink-0 rounded bg-slate-100 px-1 py-0.5 text-[9px] font-medium text-slate-600">
                          思考
                        </span>
                      )}
                    </button>

                    {isSelected && (
                      <button
                        type="button"
                        data-testid={`thinking-expand-${m}`}
                        disabled={!canAdjustThinking}
                        title={
                          canAdjustThinking
                            ? "展开选择思考等级"
                            : cap?.note || "真实 API 下该模型不支持思考调节"
                        }
                        onClick={() =>
                          canAdjustThinking &&
                          setExpandedModel((prev) => (prev === m ? null : m))
                        }
                        className={`inline-flex shrink-0 items-center gap-1 rounded-lg border px-1.5 py-0.5 text-[10px] font-semibold transition ${
                          !canAdjustThinking
                            ? "cursor-not-allowed border-slate-200 bg-slate-100 text-slate-400"
                            : isExpanded
                              ? "border-cyan-500 bg-cyan-700 text-white"
                              : level !== "off"
                                ? "border-cyan-300 bg-white text-cyan-800 hover:bg-cyan-50"
                                : "border-slate-200 bg-white text-slate-600 hover:bg-slate-100"
                        }`}
                      >
                        <span>
                          {!canAdjustThinking
                            ? "不支持"
                            : `思考:${THINKING_LABEL_MAP[level]}`}
                        </span>
                        {canAdjustThinking &&
                          (isExpanded ? (
                            <ChevronDown className="h-3 w-3" />
                          ) : (
                            <ChevronRight className="h-3 w-3" />
                          ))}
                      </button>
                    )}
                  </div>

                  {isExpanded && (
                    <div
                      data-testid="thinking-selector"
                      className="border-t border-cyan-100 bg-white/95 px-2.5 py-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[10px] font-semibold text-slate-600">
                          思考强度
                        </span>
                        <div className="inline-flex items-center gap-0.5 rounded-lg border border-slate-200 bg-slate-100/80 p-0.5 text-[10px]">
                          {THINKING_OPTIONS.map((opt) => {
                            const levelEnabled =
                              canAdjustThinking && allowed.includes(opt.key);
                            const active = level === opt.key;
                            return (
                              <button
                                key={opt.key}
                                type="button"
                                disabled={!levelEnabled}
                                data-testid={`thinking-level-${opt.key}`}
                                onClick={() =>
                                  levelEnabled && onChangeThinking(m, opt.key)
                                }
                                className={`rounded-md px-2 py-0.5 font-semibold transition ${
                                  active
                                    ? "bg-white text-cyan-800 shadow-sm"
                                    : levelEnabled
                                      ? "text-slate-600 hover:text-slate-900"
                                      : "cursor-not-allowed text-slate-400"
                                }`}
                              >
                                {opt.label}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                      <div className="mt-1 text-[10px] leading-snug text-slate-500">
                        {mockMode && !nativeSupported
                          ? "Mock 仿真模式：可展开体验各思考等级轨迹（真实 API 下该模型不支持思考调节）。"
                          : cap?.note || "选择本模型推理思考强度档位。"}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="mt-2 flex items-center gap-1.5 border-t border-slate-100 pt-2">
            <input
              type="text"
              value={customModelInput}
              onChange={(e) => setCustomModelInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleAddCustomModel();
                }
              }}
              placeholder="添加自定义模型 ID..."
              className="min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1 font-mono text-[11px] text-slate-800 placeholder:font-sans placeholder:text-slate-400 focus:border-cyan-600 focus:outline-none"
            />
            <button
              type="button"
              onClick={handleAddCustomModel}
              className="rounded-xl border border-slate-200 bg-white p-1.5 text-slate-600 hover:bg-slate-50"
              title="添加模型"
            >
              <Plus className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
