import React, { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import type {
  HealthResponse,
  KbOverviewResponse,
  PageKey,
  SettingsLlmResponse,
} from "./api/types";
import { GlassShell } from "./components/GlassShell";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import { CopilotPage } from "./pages/CopilotPage";
import { KnowledgeBasePage } from "./pages/KnowledgeBasePage";
import { SettingsPage } from "./pages/SettingsPage";

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<PageKey>("copilot");

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [settings, setSettings] = useState<SettingsLlmResponse | null>(null);
  const [kb, setKb] = useState<KbOverviewResponse | null>(null);
  const [kbLoading, setKbLoading] = useState(false);

  const loadBootstrap = useCallback(async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      const h = await api.health();
      setHealth(h);
    } catch (err) {
      setHealthError(err instanceof Error ? err.message : "无法连接到后端 API 服务");
    } finally {
      setHealthLoading(false);
    }

    try {
      const llm = await api.getLlmSettings();
      setSettings(llm);
    } catch {
      // May fail if auth token is required and not yet entered; user can configure in Settings
    }

    setKbLoading(true);
    try {
      const overview = await api.getKbOverview();
      setKb(overview);
    } catch {
      // non-fatal
    } finally {
      setKbLoading(false);
    }
  }, []);

  useEffect(() => {
    loadBootstrap();
  }, [loadBootstrap]);

  const activeModel = settings?.copilot_default_model || health?.default_model || "";
  const mockMode = settings?.llm_mock_mode ?? health?.mock_mode ?? false;
  const scalesList = Array.isArray(kb?.scales) ? kb.scales : kb?.scales?.scales || [];
  const defaultBenchmarkList = Array.isArray(settings?.benchmark_models)
    ? settings.benchmark_models
    : (settings?.benchmark_models || "")
        .split(/[,，\n]+/)
        .map((s) => s.trim())
        .filter(Boolean);

  return (
    <GlassShell
      activePage={activePage}
      onSelectPage={setActivePage}
      health={health}
      healthLoading={healthLoading}
      healthError={healthError}
      activeModel={activeModel}
      mockMode={mockMode}
      complianceNotice={settings?.compliance_notice || ""}
      onRefreshHealth={loadBootstrap}
    >
      {activePage === "copilot" && (
        <CopilotPage
          activeModel={activeModel}
          mockMode={mockMode}
          scalesCatalog={scalesList}
          actionLabels={kb?.labels?.clinical_actions}
        />
      )}

      {activePage === "benchmark" && (
        <BenchmarkPage
          presetModels={settings?.preset_models || []}
          ollamaModels={settings?.ollama_models || []}
          defaultBenchmarkModels={defaultBenchmarkList}
          defaultMockMode={mockMode}
        />
      )}

      {activePage === "kb" && <KnowledgeBasePage kb={kb} loading={kbLoading} />}

      {activePage === "settings" && (
        <SettingsPage
          settings={settings}
          onSettingsUpdated={(next) => setSettings(next)}
          onRefreshHealth={loadBootstrap}
        />
      )}
    </GlassShell>
  );
};

export default App;
