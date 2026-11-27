import React, { useCallback, useEffect, useState } from "react";
import { api, STORAGE_KEYS } from "./api/client";
import type {
  HealthResponse,
  KbOverviewResponse,
  SettingsLlmResponse,
} from "./api/types";
import { GlassShell } from "./components/GlassShell";
import { WorkspaceMode } from "./components/ModeCapsule";
import { CopilotWorkspace } from "./components/copilot/CopilotWorkspace";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import { SettingsPage } from "./pages/SettingsPage";

function loadDefaultExpandSteps(): boolean {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.PREFERENCES);
    if (!raw) return false;
    const parsed = JSON.parse(raw) as { defaultExpandSteps?: boolean };
    return Boolean(parsed?.defaultExpandSteps);
  } catch {
    return false;
  }
}

export const App: React.FC = () => {
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("copilot");
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [settings, setSettings] = useState<SettingsLlmResponse | null>(null);
  const [kb, setKb] = useState<KbOverviewResponse | null>(null);
  const [kbLoading, setKbLoading] = useState(false);

  const [sidebarSlot, setSidebarSlot] = useState<React.ReactNode>(null);
  const [headerActions, setHeaderActions] = useState<React.ReactNode>(null);
  const [defaultExpandSteps, setDefaultExpandSteps] = useState<boolean>(
    loadDefaultExpandSteps
  );

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
      // May fail before auth token is configured
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
    void loadBootstrap();
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
  const availableModels = Array.from(
    new Set([
      ...(settings?.preset_models || []),
      ...(settings?.ollama_models || []),
      activeModel,
    ].filter(Boolean))
  );

  return (
    <GlassShell
      workspaceMode={workspaceMode}
      settingsOpen={settingsOpen}
      onSelectWorkspaceMode={(mode) => {
        setSettingsOpen(false);
        setWorkspaceMode(mode);
        if (mode !== "copilot") {
          setSidebarSlot(null);
          setHeaderActions(null);
        }
      }}
      onOpenSettings={() => setSettingsOpen(true)}
      sidebarSlot={workspaceMode === "copilot" ? sidebarSlot : null}
      headerActions={workspaceMode === "copilot" ? headerActions : null}
      health={health}
      healthLoading={healthLoading}
      healthError={healthError}
      activeModel={activeModel}
      mockMode={mockMode}
      complianceNotice={settings?.compliance_notice || ""}
      onRefreshHealth={loadBootstrap}
    >
      {settingsOpen ? (
        <SettingsPage
          settings={settings}
          kb={kb}
          kbLoading={kbLoading}
          health={health}
          defaultExpandSteps={defaultExpandSteps}
          onChangeDefaultExpandSteps={setDefaultExpandSteps}
          onSettingsUpdated={(next) => setSettings(next)}
          onRefreshHealth={loadBootstrap}
          onBack={() => setSettingsOpen(false)}
        />
      ) : workspaceMode === "copilot" ? (
        <CopilotWorkspace
          availableModels={availableModels}
          defaultModel={activeModel}
          mockMode={mockMode}
          scalesCatalog={scalesList}
          defaultExpandSteps={defaultExpandSteps}
          onRenderSidebarSlot={setSidebarSlot}
          onRenderHeaderActions={setHeaderActions}
        />
      ) : (
        <BenchmarkPage
          presetModels={settings?.preset_models || []}
          ollamaModels={settings?.ollama_models || []}
          defaultBenchmarkModels={defaultBenchmarkList}
          defaultMockMode={mockMode}
        />
      )}
    </GlassShell>
  );
};

export default App;
