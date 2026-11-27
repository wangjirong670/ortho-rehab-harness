import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowLeft,
  BookOpenCheck,
  CheckCircle2,
  Cpu,
  Globe,
  KeyRound,
  RefreshCw,
  Save,
  Server,
  ShieldCheck,
  Sliders,
  XCircle,
} from "lucide-react";
import {
  api,
  getApiBaseUrl,
  getAuthToken,
  setApiBaseUrlOverride,
  setAuthToken,
  STORAGE_KEYS,
} from "../api/client";
import type {
  HealthResponse,
  KbOverviewResponse,
  ProviderConfigItem,
  SettingsLlmResponse,
  TestConnectionResponse,
} from "../api/types";
import { KnowledgeBasePanel } from "../components/KnowledgeBasePanel";

export type SettingsTabKey = "models" | "privacy" | "preferences" | "kb" | "about";

interface SettingsPageProps {
  settings: SettingsLlmResponse | null;
  kb: KbOverviewResponse | null;
  kbLoading: boolean;
  health: HealthResponse | null;
  defaultExpandSteps: boolean;
  onChangeDefaultExpandSteps: (v: boolean) => void;
  onSettingsUpdated: (next: SettingsLlmResponse) => void;
  onRefreshHealth: () => void;
  onBack: () => void;
}

interface ProviderCardSpec {
  id: string;
  label: string;
  envKey: string;
  baseEnvKey?: string;
  defaultBase?: string;
  sampleModel: string;
  isLocal?: boolean;
}

const FALLBACK_PROVIDERS: ProviderCardSpec[] = [
  {
    id: "deepseek",
    label: "DeepSeek 深度求索",
    envKey: "DEEPSEEK_API_KEY",
    baseEnvKey: "DEEPSEEK_API_BASE",
    defaultBase: "https://api.deepseek.com",
    sampleModel: "deepseek/deepseek-chat",
  },
  {
    id: "dashscope",
    label: "阿里云百炼 / 通义千问",
    envKey: "DASHSCOPE_API_KEY",
    baseEnvKey: "DASHSCOPE_API_BASE",
    defaultBase: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    sampleModel: "dashscope/qwen-plus",
  },
  {
    id: "openai",
    label: "OpenAI 兼容 / 企业中转",
    envKey: "OPENAI_API_KEY",
    baseEnvKey: "OPENAI_API_BASE",
    defaultBase: "https://api.openai.com/v1",
    sampleModel: "openai/gpt-4o-mini",
  },
  {
    id: "ollama",
    label: "本地 Ollama 私有部署",
    envKey: "",
    baseEnvKey: "OLLAMA_API_BASE",
    defaultBase: "http://localhost:11434",
    sampleModel: "ollama_chat/qwen2.5:7b",
    isLocal: true,
  },
];

function normalizeProviders(
  raw: SettingsLlmResponse["providers"] | undefined
): ProviderConfigItem[] {
  if (!raw) return [];
  if (Array.isArray(raw)) {
    return raw.map((item) => ({ ...item, id: item.id || item.key || "" }));
  }
  return Object.entries(raw).map(([id, item]) => ({
    ...item,
    id: item.id || item.key || id,
  }));
}

function formatBenchmarkModels(raw: string | string[] | undefined): string {
  if (!raw) return "";
  return Array.isArray(raw) ? raw.join(", ") : raw;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({
  settings,
  kb,
  kbLoading,
  health,
  defaultExpandSteps,
  onChangeDefaultExpandSteps,
  onSettingsUpdated,
  onRefreshHealth,
  onBack,
}) => {
  const [activeTab, setActiveTab] = useState<SettingsTabKey>("models");

  const [copilotModel, setCopilotModel] = useState(
    settings?.copilot_default_model || "deepseek/deepseek-chat"
  );
  const [benchmarkModelsText, setBenchmarkModelsText] = useState(
    formatBenchmarkModels(settings?.benchmark_models)
  );
  const [mockMode, setMockMode] = useState<boolean>(settings?.llm_mock_mode ?? false);

  const [apiKeysInput, setApiKeysInput] = useState<Record<string, string>>({});
  const [apiBasesInput, setApiBasesInput] = useState<Record<string, string>>({});
  const [testResults, setTestResults] = useState<Record<string, TestConnectionResponse>>({});
  const [testingProvider, setTestingProvider] = useState<string | null>(null);

  const [ollamaModels, setOllamaModels] = useState<string[]>(settings?.ollama_models || []);
  const [ollamaReachable, setOllamaReachable] = useState<boolean | null>(null);
  const [probingOllama, setProbingOllama] = useState(false);

  const [backendUrl, setBackendUrl] = useState<string>(() => getApiBaseUrl());
  const [bearerToken, setBearerToken] = useState<string>(() => getAuthToken());

  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState<{ type: "ok" | "err"; text: string } | null>(null);

  useEffect(() => {
    if (!settings) return;
    setCopilotModel(settings.copilot_default_model);
    setBenchmarkModelsText(formatBenchmarkModels(settings.benchmark_models));
    setMockMode(Boolean(settings.llm_mock_mode));
    setOllamaModels(settings.ollama_models || []);

    const list = normalizeProviders(settings.providers);
    const bases: Record<string, string> = {};
    for (const p of list) {
      const pid = p.id || p.key || "";
      if (pid && p.api_base) bases[pid] = p.api_base;
    }
    setApiBasesInput((prev) => ({ ...bases, ...prev }));
  }, [settings]);

  const providerCards = useMemo(() => {
    const serverProviders = normalizeProviders(settings?.providers);
    const serverMap = new Map<string, ProviderConfigItem>();
    for (const p of serverProviders) {
      const pid = p.id || p.key || "";
      if (pid) serverMap.set(pid, p);
    }

    const merged: Array<ProviderCardSpec & { server?: ProviderConfigItem }> =
      FALLBACK_PROVIDERS.map((fp) => {
        const sp = serverMap.get(fp.id);
        return {
          ...fp,
          envKey: sp?.api_key_env || sp?.env_key || fp.envKey,
          baseEnvKey: sp?.api_base_env || sp?.base_env_key || fp.baseEnvKey,
          sampleModel: sp?.default_model || fp.sampleModel,
          server: sp,
        };
      });

    for (const sp of serverProviders) {
      const pid = sp.id || sp.key || "";
      if (pid && !merged.some((m) => m.id === pid)) {
        merged.push({
          id: pid,
          label: sp.label || pid,
          envKey: sp.api_key_env || sp.env_key || "",
          baseEnvKey: sp.api_base_env || sp.base_env_key,
          defaultBase: sp.api_base || "",
          sampleModel: sp.default_model || sp.model_suggestions?.[0] || sp.models?.[0] || "",
          isLocal: pid.includes("ollama"),
          server: sp,
        });
      }
    }
    return merged;
  }, [settings?.providers]);

  const allModelChoices = useMemo(() => {
    const set = new Set<string>([
      ...(settings?.preset_models || []),
      ...ollamaModels,
      copilotModel,
    ]);
    return Array.from(set).filter(Boolean);
  }, [settings?.preset_models, ollamaModels, copilotModel]);

  const handleSaveLlmSettings = async () => {
    setSaving(true);
    setFeedback(null);
    try {
      const envUpdates: Record<string, string> = {};
      for (const card of providerCards) {
        const keyVal = apiKeysInput[card.id]?.trim();
        if (card.envKey && keyVal) {
          envUpdates[card.envKey] = keyVal;
        }
        const baseVal = apiBasesInput[card.id]?.trim();
        if (card.baseEnvKey && baseVal !== undefined && baseVal !== "") {
          envUpdates[card.baseEnvKey] = baseVal;
        }
      }

      const parsedBenchmarkModels = benchmarkModelsText
        .split(/[,，\n]+/)
        .map((s) => s.trim())
        .filter(Boolean)
        .join(",");

      const updated = await api.updateLlmSettings({
        copilot_default_model: copilotModel,
        benchmark_models: parsedBenchmarkModels,
        llm_mock_mode: mockMode,
        env_updates: Object.keys(envUpdates).length > 0 ? envUpdates : undefined,
      });

      onSettingsUpdated(updated);
      setApiKeysInput({});
      setFeedback({ type: "ok", text: "模型与运行配置已保存并生效。" });
      onRefreshHealth();
    } catch (err) {
      setFeedback({
        type: "err",
        text: err instanceof Error ? err.message : "保存模型设置失败",
      });
    } finally {
      setSaving(false);
    }
  };

  const handleProbeOllama = async () => {
    setProbingOllama(true);
    try {
      const res = await api.probeOllamaModels();
      setOllamaModels(res.models || []);
      setOllamaReachable(res.reachable);
    } catch (err) {
      setOllamaReachable(false);
      setFeedback({
        type: "err",
        text: err instanceof Error ? err.message : "探测 Ollama 服务失败",
      });
    } finally {
      setProbingOllama(false);
    }
  };

  const handleTestConnection = async (card: ProviderCardSpec & { server?: ProviderConfigItem }) => {
    setTestingProvider(card.id);
    try {
      const res = await api.testConnection({
        provider: card.id,
        api_key: apiKeysInput[card.id]?.trim() || undefined,
        api_base: apiBasesInput[card.id]?.trim() || card.server?.api_base || card.defaultBase,
        model: card.server?.default_model || card.sampleModel,
      });
      setTestResults((prev) => ({ ...prev, [card.id]: res }));
    } catch (err) {
      setTestResults((prev) => ({
        ...prev,
        [card.id]: {
          ok: false,
          message: err instanceof Error ? err.message : "连通性测试失败",
        },
      }));
    } finally {
      setTestingProvider(null);
    }
  };

  const handleSaveEndpointSettings = () => {
    setApiBaseUrlOverride(backendUrl);
    setAuthToken(bearerToken);
    setFeedback({ type: "ok", text: "后端接入地址与鉴权令牌已保存到本地浏览器。" });
    onRefreshHealth();
  };

  const TABS: Array<{ key: SettingsTabKey; label: string; icon: React.FC<{ className?: string }> }> = [
    { key: "models", label: "模型与接口", icon: Cpu },
    { key: "privacy", label: "隐私与安全", icon: ShieldCheck },
    { key: "preferences", label: "显示偏好", icon: Sliders },
    { key: "kb", label: "医学知识库（只读）", icon: BookOpenCheck },
    { key: "about", label: "关于与自检", icon: Activity },
  ];

  const rfCount = Array.isArray(kb?.red_flags)
    ? kb.red_flags.length
    : kb?.red_flags?.rules?.length || 0;
  const dcCount = Array.isArray(kb?.drug_conflicts)
    ? kb.drug_conflicts.length
    : kb?.drug_conflicts?.conflicts?.length || 0;
  const scCount = Array.isArray(kb?.scales) ? kb.scales.length : kb?.scales?.scales?.length || 0;

  return (
    <div className="space-y-4">
      {/* Top Bar of Full-Page Settings */}
      <div className="glass-surface flex flex-wrap items-center justify-between gap-3 rounded-3xl p-4">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            data-testid="settings-back-btn"
            className="inline-flex items-center gap-1.5 rounded-2xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-sm hover:bg-slate-50"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            返回工作台
          </button>
          <div>
            <h1 className="text-base font-bold text-slate-900">系统设置与医学知识库</h1>
            <p className="text-[11px] text-slate-500">
              管理多模型接口、隐私策略、界面偏好及只读确定性规则清单
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-1 rounded-2xl bg-slate-900/5 p-1">
          {TABS.map((t) => {
            const Icon = t.icon;
            const active = activeTab === t.key;
            return (
              <button
                key={t.key}
                type="button"
                onClick={() => setActiveTab(t.key)}
                className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold transition ${
                  active
                    ? "bg-white text-cyan-900 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span>{t.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {feedback && (
        <div
          className={`flex items-center justify-between rounded-2xl px-4 py-3 text-xs font-medium ${
            feedback.type === "ok" ? "glass-safe text-emerald-900" : "glass-danger text-rose-900"
          }`}
        >
          <span>{feedback.text}</span>
          <button
            type="button"
            onClick={() => setFeedback(null)}
            className="rounded-lg px-2 py-0.5 hover:bg-white/60"
          >
            关闭
          </button>
        </div>
      )}

      {/* Tab 1: Models & Endpoints */}
      {activeTab === "models" && (
        <div className="space-y-4">
          <section className="glass-surface space-y-4 rounded-3xl p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-cyan-700" />
                <h2 className="text-sm font-bold text-slate-900">默认推理模型与运行模式</h2>
              </div>

              <button
                type="button"
                disabled={saving}
                onClick={handleSaveLlmSettings}
                className="inline-flex items-center gap-1.5 rounded-2xl bg-gradient-to-r from-cyan-700 to-teal-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-cyan-900/15 hover:brightness-105 disabled:opacity-55"
              >
                <Save className="h-3.5 w-3.5" />
                {saving ? "保存中..." : "保存配置"}
              </button>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div className="clinical-paper rounded-2xl p-4">
                <label className="block text-xs font-semibold text-slate-800">
                  临床副驾默认模型
                </label>
                <select
                  value={copilotModel}
                  onChange={(e) => setCopilotModel(e.target.value)}
                  className="clinical-input mt-2 font-mono text-xs"
                >
                  {allModelChoices.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
                <p className="mt-1.5 text-[11px] text-slate-500">
                  各消息仍可在输入框下方独立选模型，此项为新会话默认值
                </p>
              </div>

              <div className="clinical-paper rounded-2xl p-4">
                <label className="block text-xs font-semibold text-slate-800">
                  红队评测默认对比模型（逗号分隔）
                </label>
                <input
                  type="text"
                  value={benchmarkModelsText}
                  onChange={(e) => setBenchmarkModelsText(e.target.value)}
                  placeholder="deepseek/deepseek-chat, dashscope/qwen-plus"
                  className="clinical-input mt-2 font-mono text-xs"
                />
                <p className="mt-1.5 text-[11px] text-slate-500">
                  新建红队评测时默认预选的模型列表
                </p>
              </div>

              <div className="clinical-paper flex flex-col justify-between rounded-2xl p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-xs font-semibold text-slate-800">Mock 确定性仿真模式</div>
                    <div className="mt-0.5 text-[11px] text-slate-500">
                      开启后全程离线运行，不消耗外部 API 额度
                    </div>
                  </div>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={mockMode}
                    onClick={() => setMockMode((v) => !v)}
                    className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ${
                      mockMode ? "bg-amber-500" : "bg-emerald-600"
                    }`}
                  >
                    <span
                      className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition duration-200 ${
                        mockMode ? "translate-x-5" : "translate-x-0"
                      }`}
                    />
                  </button>
                </div>
                <div className="mt-2 text-[11px] font-semibold">
                  当前状态：
                  <span className={mockMode ? "text-amber-700" : "text-emerald-700"}>
                    {mockMode ? "Mock 仿真（离线调试）" : "Live 真实模型推理"}
                  </span>
                </div>
              </div>
            </div>
          </section>

          <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {providerCards.map((card) => {
              const server = card.server;
              const testRes = testResults[card.id];
              const isTesting = testingProvider === card.id;

              return (
                <div key={card.id} className="glass-surface space-y-3 rounded-3xl p-5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-bold text-slate-900">{card.label}</h3>
                      <p className="font-mono text-[11px] text-slate-500">
                        示例模型：{card.sampleModel || "-"}
                      </p>
                    </div>

                    {!card.isLocal && (
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                          server?.has_key || server?.api_key_set
                            ? "bg-emerald-100 text-emerald-800"
                            : "bg-slate-100 text-slate-600"
                        }`}
                      >
                        {server?.has_key || server?.api_key_set
                          ? `已配置 (${server?.masked_key || server?.api_key_masked || "sk-***"})`
                          : "未配置 Key"}
                      </span>
                    )}
                  </div>

                  <div className="clinical-paper space-y-3 rounded-2xl p-4 text-xs">
                    {card.envKey && (
                      <div>
                        <label className="mb-1 flex items-center gap-1 font-semibold text-slate-700">
                          <KeyRound className="h-3.5 w-3.5 text-cyan-700" />
                          API Key（留空则保持服务端现有密钥不变）
                        </label>
                        <input
                          type="password"
                          autoComplete="new-password"
                          value={apiKeysInput[card.id] || ""}
                          onChange={(e) =>
                            setApiKeysInput((prev) => ({ ...prev, [card.id]: e.target.value }))
                          }
                          placeholder={
                            server?.masked_key || server?.api_key_masked
                              ? `当前已保存：${server?.masked_key || server?.api_key_masked}`
                              : `请输入 ${card.envKey}`
                          }
                          className="clinical-input font-mono text-xs"
                        />
                      </div>
                    )}

                    {card.baseEnvKey && (
                      <div>
                        <label className="mb-1 flex items-center gap-1 font-semibold text-slate-700">
                          <Globe className="h-3.5 w-3.5 text-cyan-700" />
                          Base URL 接口地址
                        </label>
                        <input
                          type="text"
                          value={
                            apiBasesInput[card.id] ?? server?.api_base ?? card.defaultBase ?? ""
                          }
                          onChange={(e) =>
                            setApiBasesInput((prev) => ({ ...prev, [card.id]: e.target.value }))
                          }
                          placeholder={card.defaultBase}
                          className="clinical-input font-mono text-xs"
                        />
                      </div>
                    )}

                    {card.isLocal && (
                      <div className="space-y-2 rounded-xl bg-slate-50 p-3">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="font-semibold text-slate-700">
                            已发现本地 Ollama 模型（{ollamaModels.length}）
                          </span>
                          <button
                            type="button"
                            disabled={probingOllama}
                            onClick={handleProbeOllama}
                            className="inline-flex items-center gap-1 rounded-xl border border-cyan-200 bg-white px-2.5 py-1 text-[11px] font-semibold text-cyan-800 hover:bg-cyan-50"
                          >
                            <RefreshCw
                              className={`h-3 w-3 ${probingOllama ? "animate-spin" : ""}`}
                            />
                            探测 Ollama 模型
                          </button>
                        </div>
                        {ollamaReachable === false && (
                          <p className="text-[11px] text-rose-700">
                            未探测到运行中的 Ollama 服务，请确认本地 11434 端口已启动。
                          </p>
                        )}
                        {ollamaModels.length > 0 && (
                          <div className="flex flex-wrap gap-1.5">
                            {ollamaModels.map((m) => (
                              <button
                                key={m}
                                type="button"
                                onClick={() => setCopilotModel(m)}
                                className={`rounded-full px-2.5 py-0.5 font-mono text-[11px] ${
                                  copilotModel === m
                                    ? "bg-cyan-700 text-white"
                                    : "bg-white text-slate-700 shadow-sm hover:bg-cyan-50"
                                }`}
                              >
                                {m}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                      <button
                        type="button"
                        disabled={isTesting}
                        onClick={() => handleTestConnection(card)}
                        className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-1.5 font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-55"
                      >
                        <ShieldCheck className="h-3.5 w-3.5 text-cyan-700" />
                        {isTesting ? "测试中..." : "测试连通性"}
                      </button>

                      {testRes && (
                        <span
                          className={`inline-flex items-center gap-1 text-[11px] font-medium ${
                            testRes.ok ? "text-emerald-700" : "text-rose-700"
                          }`}
                        >
                          {testRes.ok ? (
                            <CheckCircle2 className="h-3.5 w-3.5" />
                          ) : (
                            <XCircle className="h-3.5 w-3.5" />
                          )}
                          {testRes.message}
                          {testRes.latency_ms != null && ` (${Math.round(testRes.latency_ms)} ms)`}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </section>

          <section className="glass-surface space-y-3 rounded-3xl p-5">
            <div className="flex items-center gap-2">
              <Server className="h-4 w-4 text-cyan-700" />
              <h2 className="text-sm font-bold text-slate-900">内网 / 云端后端服务地址与鉴权设置</h2>
            </div>
            <div className="clinical-paper grid grid-cols-1 gap-4 rounded-2xl p-4 md:grid-cols-2">
              <div>
                <label className="mb-1 block text-xs font-semibold text-slate-700">
                  远程后端地址覆盖（ORTHO_API_BASE_URL）
                </label>
                <input
                  type="text"
                  value={backendUrl}
                  onChange={(e) => setBackendUrl(e.target.value)}
                  placeholder="留空使用默认同源/代理"
                  className="clinical-input font-mono text-xs"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs font-semibold text-slate-700">
                  访问令牌（Bearer Token）
                </label>
                <input
                  type="password"
                  autoComplete="off"
                  value={bearerToken}
                  onChange={(e) => setBearerToken(e.target.value)}
                  placeholder="服务端未开启鉴权时可留空"
                  className="clinical-input font-mono text-xs"
                />
              </div>
            </div>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  setBackendUrl("");
                  setBearerToken("");
                  setApiBaseUrlOverride("");
                  setAuthToken("");
                  onRefreshHealth();
                }}
                className="rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-medium text-slate-700"
              >
                恢复默认同源配置
              </button>
              <button
                type="button"
                onClick={handleSaveEndpointSettings}
                className="rounded-xl bg-cyan-700 px-4 py-2 text-xs font-semibold text-white"
              >
                应用连接设置并重检
              </button>
            </div>
          </section>

          {/* Model Thinking Capabilities Matrix (D19) */}
          {settings?.model_capabilities &&
            Object.keys(settings.model_capabilities).length > 0 && (
              <section className="glass-surface space-y-3 rounded-3xl p-5">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="text-sm font-bold text-slate-900">
                    模型思考强度能力表（只读探测映射）
                  </h2>
                  <span className="text-[11px] text-slate-500">
                    自动将「关 / 低 / 中 / 高」映射为各厂商原生推理参数
                  </span>
                </div>
                <div className="clinical-paper overflow-x-auto rounded-2xl p-3">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 text-[11px] text-slate-500">
                        <th className="px-2 py-1.5">模型标识</th>
                        <th className="px-2 py-1.5">支持思考</th>
                        <th className="px-2 py-1.5">协议机制</th>
                        <th className="px-2 py-1.5">可用档位</th>
                        <th className="px-2 py-1.5">说明</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {Object.values(settings.model_capabilities).map((cap) => (
                        <tr key={cap.model}>
                          <td className="px-2 py-2 font-mono text-[11px] font-semibold text-slate-800">
                            {cap.model}
                          </td>
                          <td className="px-2 py-2">
                            <span
                              className={`inline-block whitespace-nowrap rounded-full px-2 py-0.5 text-[10px] font-bold ${
                                cap.supports_thinking
                                  ? "bg-emerald-100 text-emerald-800"
                                  : "bg-slate-100 text-slate-500"
                              }`}
                            >
                              {cap.supports_thinking ? "支持" : "不支持"}
                            </span>
                          </td>
                          <td className="px-2 py-2 font-mono text-[11px] text-slate-600">
                            {cap.mechanism}
                          </td>
                          <td className="px-2 py-2 text-[11px] text-slate-700">
                            {cap.supported_levels.join(" / ")}
                          </td>
                          <td className="px-2 py-2 text-[11px] text-slate-500">{cap.note}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}
        </div>
      )}

      {/* Tab 2: Privacy & Security */}
      {activeTab === "privacy" && (
        <section className="glass-surface space-y-3 rounded-3xl p-5 text-xs leading-relaxed text-slate-700">
          <h2 className="text-sm font-bold text-slate-900">患者隐私隔离与 PII 分级护栏</h2>
          <div className="clinical-paper space-y-2 rounded-2xl p-4">
            <p>
              <strong>1. 档案展示名确定性剔除：</strong>
              <code>PATIENT.md</code> 中的 <code>display_name</code>（例如「302床 张某」）仅在本地界面显示；组装任意模型提示词前，确定性函数{" "}
              <code>build_prompt_context()</code> 会强行剥离该名称，绝不向模型外发。
            </p>
            <p>
              <strong>2. 云端模型与本地模型分级脱敏拦截：</strong>
              输入病历触发姓名/电话/身份证/住院号检测时，若目标为云端模型，发送按钮自动变为「脱敏后发送」；坚持发送原文须二次确认并记录至会话{" "}
              <code>.trace.jsonl</code> 的 <code>pii_check</code> 审计事件。本地 Ollama 模型仅作温和提醒不强阻断。
            </p>
            <p>
              <strong>3. 本地档案库安全边界：</strong>
              所有患者文件均落盘在本地 <code>case_archives/</code>，已加入 <code>.gitignore</code> 与打包安全审计黑名单，永不进入发行包。
            </p>
          </div>
        </section>
      )}

      {/* Tab 3: Display Preferences */}
      {activeTab === "preferences" && (
        <section className="glass-surface space-y-3 rounded-3xl p-5 text-xs">
          <h2 className="text-sm font-bold text-slate-900">交互与显示偏好</h2>
          <div className="clinical-paper flex items-center justify-between rounded-2xl p-4">
            <div>
              <div className="font-semibold text-slate-900">流水线步骤条完成后保持展开</div>
              <div className="mt-0.5 text-[11px] text-slate-500">
                开启后，L1 确定性流水线步骤条在单轮推理结束时仍默认展开展示各阶段耗时标签。
              </div>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={defaultExpandSteps}
              onClick={() => {
                const next = !defaultExpandSteps;
                onChangeDefaultExpandSteps(next);
                try {
                  localStorage.setItem(
                    STORAGE_KEYS.PREFERENCES,
                    JSON.stringify({ defaultExpandSteps: next })
                  );
                } catch {
                  // ignore
                }
              }}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${
                defaultExpandSteps ? "bg-cyan-700" : "bg-slate-300"
              }`}
            >
              <span
                className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition ${
                  defaultExpandSteps ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </div>
        </section>
      )}

      {/* Tab 4: Knowledge Base (Read-Only) */}
      {activeTab === "kb" && <KnowledgeBasePanel kb={kb} loading={kbLoading} />}

      {/* Tab 5: About & Self Check */}
      {activeTab === "about" && (
        <section className="glass-surface space-y-3 rounded-3xl p-5 text-xs">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-900">系统状态与确定性内核自检</h2>
            <button
              type="button"
              onClick={onRefreshHealth}
              className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-3 py-1.5 font-semibold text-slate-700 hover:bg-slate-50"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              重新自检
            </button>
          </div>

          <div className="clinical-paper grid grid-cols-1 gap-3 rounded-2xl p-4 sm:grid-cols-3">
            <div>
              <div className="text-[11px] text-slate-500">服务端版本</div>
              <div className="mt-1 font-mono text-sm font-bold text-slate-900">
                v{health?.version || "0.4.0"}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-500">运行模式</div>
              <div className="mt-1 font-mono text-sm font-bold text-slate-900">
                {health?.mode || "-"} ({health?.mock_mode ? "Mock" : "Live"})
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-500">已加载医学规则资产</div>
              <div className="mt-1 font-mono text-sm font-bold text-cyan-800">
                {rfCount} 红旗 · {dcCount} 冲突 · {scCount} 量表
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};
