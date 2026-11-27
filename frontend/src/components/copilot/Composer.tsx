import React, { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  EyeOff,
  Globe,
  Loader2,
  Paperclip,
  Send,
  ShieldAlert,
  ShieldCheck,
  Undo2,
} from "lucide-react";
import { api } from "../../api/client";
import type {
  ModelCapability,
  OutputMode,
  PiiFinding,
  ThinkingLevel,
} from "../../api/types";
import { Modal } from "../Modal";
import { ModelThinkingPicker } from "../ModelThinkingPicker";

interface ComposerProps {
  availableModels: string[];
  defaultModel: string;
  modelCapabilities?: Record<string, ModelCapability>;
  mockMode: boolean;
  offlineMode?: boolean;
  disabled?: boolean;
  streaming: boolean;
  onSend: (payload: {
    content: string;
    model: string;
    thinking: ThinkingLevel;
    output_mode: OutputMode;
    pii_action: "none" | "sanitized" | "override_raw";
    web_search: boolean;
  }) => Promise<void>;
}

function isLocalOrMockModel(model: string, mockMode: boolean): boolean {
  if (mockMode) return true;
  const lower = (model || "").toLowerCase();
  return lower.includes("ollama");
}

export const Composer: React.FC<ComposerProps> = ({
  availableModels,
  defaultModel,
  modelCapabilities = {},
  mockMode,
  offlineMode = false,
  disabled = false,
  streaming,
  onSend,
}) => {
  const [text, setText] = useState("");
  const [selectedModel, setSelectedModel] = useState(defaultModel || availableModels[0] || "");
  const [modelThinking, setModelThinking] = useState<Record<string, ThinkingLevel>>({});
  const [outputMode, setOutputMode] = useState<OutputMode>("full");
  const [webSearch, setWebSearch] = useState(false);

  const [piiFindings, setPiiFindings] = useState<PiiFinding[]>([]);
  const [piiDismissed, setPiiDismissed] = useState(false);
  const [undoText, setUndoText] = useState<string | null>(null);
  const [sanitizedApplied, setSanitizedApplied] = useState(false);
  const [rawConfirmOpen, setRawConfirmOpen] = useState(false);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!selectedModel && defaultModel) {
      setSelectedModel(defaultModel);
    }
  }, [defaultModel, selectedModel]);

  // Debounced live PII check
  useEffect(() => {
    const trimmed = text.trim();
    if (!trimmed) {
      setPiiFindings([]);
      setPiiDismissed(false);
      return;
    }
    const timer = setTimeout(() => {
      api
        .detectPii(text)
        .then((res) => {
          setPiiFindings(res.has_pii ? res.findings || [] : []);
          if (!res.has_pii) {
            setPiiDismissed(false);
          }
        })
        .catch(() => {
          // non-fatal
        });
    }, 320);
    return () => clearTimeout(timer);
  }, [text]);

  const isLocal = isLocalOrMockModel(selectedModel, mockMode);
  const hasActivePiiWarning = piiFindings.length > 0 && !piiDismissed;
  const cloudPiiGate = hasActivePiiWarning && !isLocal && outputMode === "full";
  const controlsGreyed = outputMode === "screen";

  const handleSanitize = async (): Promise<string> => {
    try {
      const res = await api.sanitizePii(text);
      if (res.changed) {
        setUndoText(text);
        setText(res.sanitized_text);
        setSanitizedApplied(true);
        setPiiFindings([]);
        return res.sanitized_text;
      }
    } catch {
      // ignore
    }
    return text;
  };

  const handleUndoSanitize = () => {
    if (undoText !== null) {
      setText(undoText);
      setUndoText(null);
      setSanitizedApplied(false);
      setPiiDismissed(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setUploading(true);
    setUploadErr(null);
    try {
      const res = await api.uploadClinicalFile(file);
      setText((prev) => (prev.trim() ? `${prev.trim()}\n\n${res.text}` : res.text));
      setUndoText(null);
      setSanitizedApplied(false);
      setPiiDismissed(false);
    } catch (err) {
      setUploadErr(err instanceof Error ? err.message : "导入文件失败");
    } finally {
      setUploading(false);
    }
  };

  const dispatchSend = async (
    payloadText: string,
    piiAction: "none" | "sanitized" | "override_raw"
  ) => {
    const cleaned = payloadText.trim();
    if (!cleaned || disabled || streaming) return;
    const chosenModel = selectedModel || defaultModel;
    const cap = modelCapabilities[chosenModel];
    const effectiveThinking: ThinkingLevel =
      !mockMode && cap && !cap.supports_thinking
        ? "off"
        : modelThinking[chosenModel] || "off";
    setText("");
    setUndoText(null);
    setSanitizedApplied(false);
    setPiiFindings([]);
    setPiiDismissed(false);
    await onSend({
      content: cleaned,
      model: chosenModel,
      thinking: effectiveThinking,
      output_mode: outputMode,
      pii_action: piiAction,
      web_search: !offlineMode && !controlsGreyed && webSearch,
    });
  };

  const handlePrimarySend = async () => {
    if (!text.trim() || disabled || streaming) return;
    if (cloudPiiGate) {
      const sanitized = await handleSanitize();
      await dispatchSend(sanitized, "sanitized");
      return;
    }
    await dispatchSend(text, sanitizedApplied ? "sanitized" : "none");
  };

  const modelChoices = Array.from(
    new Set([selectedModel, defaultModel, ...availableModels].filter(Boolean))
  );

  return (
    <div className="space-y-2 pt-1">
      {/* Inline PII banner (D15) */}
      {(hasActivePiiWarning || undoText !== null) && (
        <div
          data-testid="pii-inline-banner"
          className={`flex flex-wrap items-center justify-between gap-2 rounded-2xl border px-3.5 py-2 text-xs ${
            cloudPiiGate
              ? "border-amber-300 bg-amber-50/90 text-amber-950"
              : "border-cyan-200 bg-cyan-50/80 text-cyan-950"
          }`}
        >
          <div className="flex flex-wrap items-center gap-1.5">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-amber-600" />
            {piiFindings.length > 0 ? (
              <span>
                检测到疑似敏感隐私（
                {piiFindings.map((f) => `${f.label}×${f.count}`).join("、")}）：
                {cloudPiiGate
                  ? "当前为云端模型，点击主按钮将自动脱敏后发送。"
                  : "当前为本地/离线仿真环境，可按需脱敏。"}
              </span>
            ) : (
              <span>已应用一键 PII 脱敏。</span>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {piiFindings.length > 0 && (
              <>
                <button
                  type="button"
                  onClick={() => handleSanitize()}
                  className="inline-flex items-center gap-1 rounded-xl bg-amber-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-amber-700"
                >
                  <EyeOff className="h-3 w-3" />
                  一键脱敏
                </button>
                <button
                  type="button"
                  onClick={() => setPiiDismissed(true)}
                  className="rounded-xl border border-amber-300/80 bg-white px-2.5 py-1 text-[11px] font-medium text-amber-900 hover:bg-amber-100/70"
                >
                  忽略
                </button>
              </>
            )}
            {undoText !== null && (
              <button
                type="button"
                onClick={handleUndoSanitize}
                className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-50"
              >
                <Undo2 className="h-3 w-3" />
                撤销脱敏
              </button>
            )}
          </div>
        </div>
      )}

      {uploadErr && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-1.5 text-xs text-rose-800">
          {uploadErr}
        </div>
      )}

      <div className="glass-surface rounded-3xl p-3">
        <textarea
          rows={3}
          value={text}
          disabled={disabled || streaming}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
              e.preventDefault();
              void handlePrimarySend();
            }
          }}
          placeholder="输入本次诊疗病历、随访变化或临床追问（Ctrl+Enter 发送）..."
          className="w-full resize-none rounded-2xl border border-slate-200/80 bg-white/95 px-3.5 py-2.5 text-xs leading-relaxed text-slate-900 placeholder-slate-400 focus:border-cyan-600/60 focus:outline-none focus:ring-2 focus:ring-cyan-500/20"
        />

        {/* Controls Bar */}
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,.md,.docx"
              className="hidden"
              onChange={handleFileChange}
            />
            <button
              type="button"
              disabled={disabled || streaming || uploading}
              onClick={() => fileInputRef.current?.click()}
              title="导入 .txt / .md / .docx 病历文本"
              className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              <Paperclip className="h-3.5 w-3.5" />
              <span>{uploading ? "解析中" : "+ 导入"}</span>
            </button>

            {/* Output mode: full vs screen */}
            <div className="inline-flex rounded-xl border border-slate-200 bg-slate-100/80 p-0.5 text-[11px]">
              <button
                type="button"
                onClick={() => setOutputMode("full")}
                className={`rounded-lg px-2 py-1 font-semibold transition ${
                  outputMode === "full"
                    ? "bg-white text-cyan-800 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                完整分析
              </button>
              <button
                type="button"
                onClick={() => setOutputMode("screen")}
                className={`rounded-lg px-2 py-1 font-semibold transition ${
                  outputMode === "screen"
                    ? "bg-white text-amber-800 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                简要筛查
              </button>
            </div>

            {/* Unified Model + Expandable Thinking Picker */}
            <ModelThinkingPicker
              mode="single"
              direction="up"
              testId="copilot-model-picker"
              availableModels={modelChoices}
              selectedModels={selectedModel ? [selectedModel] : []}
              modelCapabilities={modelCapabilities}
              modelThinking={modelThinking}
              mockMode={mockMode}
              disabled={controlsGreyed}
              disabledTooltip="简要筛查为纯确定性引擎，不调用大模型"
              onChangeModels={(next) => {
                if (next[0]) setSelectedModel(next[0]);
              }}
              onChangeThinking={(m, level) =>
                setModelThinking((prev) => ({ ...prev, [m]: level }))
              }
            />

            {/* D9 Global Offline-Safe Autonomous Web Search Toggle (default OFF) */}
            <button
              type="button"
              data-testid="web-search-toggle"
              disabled={offlineMode || controlsGreyed}
              onClick={() => !offlineMode && !controlsGreyed && setWebSearch((v) => !v)}
              title={
                offlineMode
                  ? "全局离线模式已开启，联网检索已禁用（可在设置中关闭离线模式）"
                  : controlsGreyed
                    ? "简要筛查仅运行本地确定性规则，不执行外部检索"
                    : "开启后允许模型在本轮自主检索权威医学指南/药品白名单（查询词强制脱敏，仅作引用）"
              }
              className={`inline-flex items-center gap-1 rounded-xl border px-2.5 py-1.5 text-[11px] font-semibold transition ${
                offlineMode || controlsGreyed
                  ? "cursor-not-allowed border-slate-200 bg-slate-100 text-slate-400"
                  : webSearch
                    ? "border-cyan-600 bg-cyan-700 text-white shadow-sm"
                    : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
              }`}
            >
              <Globe className="h-3.5 w-3.5" />
              <span>{offlineMode ? "离线禁用" : webSearch ? "联网:开" : "联网"}</span>
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            {cloudPiiGate && (
              <button
                type="button"
                disabled={disabled || streaming}
                onClick={() => setRawConfirmOpen(true)}
                className="inline-flex items-center gap-1 rounded-xl border border-rose-200 bg-white px-2.5 py-1.5 text-[11px] font-semibold text-rose-700 hover:bg-rose-50"
              >
                <ShieldAlert className="h-3.5 w-3.5" />
                原文发送
              </button>
            )}

            <button
              type="button"
              disabled={!text.trim() || disabled || streaming}
              onClick={handlePrimarySend}
              data-testid="send-chat-btn"
              className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-cyan-700 to-teal-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow-sm hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {streaming ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  分析中...
                </>
              ) : cloudPiiGate ? (
                <>
                  <ShieldCheck className="h-3.5 w-3.5" />
                  脱敏后发送
                </>
              ) : (
                <>
                  <Send className="h-3.5 w-3.5" />
                  {outputMode === "screen" ? "极速筛查" : "发送"}
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Confirm raw PII send to cloud model */}
      <Modal
        open={rawConfirmOpen}
        onClose={() => setRawConfirmOpen(false)}
        title="确认向云端模型发送未脱敏原文？"
        maxWidthClass="max-w-md"
      >
        <div className="space-y-3 text-xs leading-relaxed text-slate-600">
          <p>
            当前检测到疑似患者隐私信息，且所选模型为云端接口。若坚持发送未脱敏原文，本次越权放行记录将写入{" "}
            <code>pii_check</code> 审计轨迹。
          </p>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setRawConfirmOpen(false)}
              className="rounded-xl border border-slate-200 px-3 py-1.5 font-medium text-slate-600"
            >
              取消（改用脱敏发送）
            </button>
            <button
              type="button"
              onClick={() => {
                setRawConfirmOpen(false);
                void dispatchSend(text, "override_raw");
              }}
              className="rounded-xl bg-rose-700 px-3.5 py-1.5 font-semibold text-white hover:bg-rose-800"
            >
              确认原文发送并记入审计
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
};
