import type {
  BenchmarkProgressEvent,
  BenchmarkReportDetail,
  BenchmarkReportsResponse,
  CaseListResponse,
  CaseWorkspaceRecord,
  CopilotAnalysisResponse,
  DemoCasesResponse,
  DraftMode,
  HandoverResponse,
  HealthResponse,
  KbOverviewResponse,
  OllamaModelsResponse,
  PiiDetectResponse,
  PiiSanitizeResponse,
  SaveCaseWorkspaceRequest,
  SettingsLlmResponse,
  StartBenchmarkRunRequest,
  TestConnectionResponse,
  UpdateSettingsLlmRequest,
  UploadResponse,
} from "./types";

export const STORAGE_KEYS = {
  AUTH_TOKEN: "ortho_auth_token",
  API_BASE_URL: "ortho_api_base_url",
  COPILOT_HISTORY: "ortho_copilot_history",
} as const;

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(message: string, status = 0, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function getApiBaseUrl(): string {
  try {
    const override = localStorage.getItem(STORAGE_KEYS.API_BASE_URL)?.trim();
    if (override) {
      return override.replace(/\/+$/, "");
    }
  } catch {
    // Ignore storage access errors
  }
  const envBase = (import.meta.env.VITE_API_BASE_URL || "").trim();
  return envBase.replace(/\/+$/, "");
}

export function setApiBaseUrlOverride(url: string): void {
  const cleaned = url.trim().replace(/\/+$/, "");
  if (cleaned) {
    localStorage.setItem(STORAGE_KEYS.API_BASE_URL, cleaned);
  } else {
    localStorage.removeItem(STORAGE_KEYS.API_BASE_URL);
  }
}

export function getAuthToken(): string {
  try {
    return localStorage.getItem(STORAGE_KEYS.AUTH_TOKEN)?.trim() || "";
  } catch {
    return "";
  }
}

export function setAuthToken(token: string): void {
  const cleaned = token.trim();
  if (cleaned) {
    localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, cleaned);
  } else {
    localStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
  }
}

function buildUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const base = getApiBaseUrl();
  return `${base}${normalizedPath}`;
}

function buildHeaders(extra?: HeadersInit, isFormData = false): Headers {
  const headers = new Headers(extra);
  const token = getAuthToken();
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  if (!isFormData && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  return headers;
}

async function parseErrorMessage(res: Response): Promise<{ message: string; detail: unknown }> {
  let detail: unknown = null;
  try {
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const data = await res.json();
      detail = data;
      const msg =
        (typeof data?.detail === "string" && data.detail) ||
        (typeof data?.message === "string" && data.message) ||
        (typeof data?.error === "string" && data.error);
      if (msg) {
        return { message: msg, detail };
      }
    } else {
      const txt = await res.text();
      if (txt) {
        return { message: txt.slice(0, 300), detail: txt };
      }
    }
  } catch {
    // ignore parse failure
  }
  if (res.status === 401 || res.status === 403) {
    return {
      message: "鉴权失败（401/403），请在「模型与设置」中核对访问令牌 (Bearer Token)。",
      detail,
    };
  }
  return { message: `请求失败 (${res.status} ${res.statusText || ""})`.trim(), detail };
}

async function apiRequest<T>(path: string, init: RequestInit = {}, isFormData = false): Promise<T> {
  const res = await fetch(buildUrl(path), {
    ...init,
    headers: buildHeaders(init.headers, isFormData),
  });
  if (!res.ok) {
    const { message, detail } = await parseErrorMessage(res);
    throw new ApiError(message, res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health(): Promise<HealthResponse> {
    return apiRequest<HealthResponse>("/api/v1/health");
  },

  // ---------------- Copilot ----------------
  getDemoCases(): Promise<DemoCasesResponse> {
    return apiRequest<DemoCasesResponse>("/api/v1/copilot/demo-cases");
  },

  uploadClinicalFile(file: File): Promise<UploadResponse> {
    const form = new FormData();
    form.append("file", file);
    return apiRequest<UploadResponse>(
      "/api/v1/copilot/upload",
      { method: "POST", body: form },
      true
    );
  },

  detectPii(text: string): Promise<PiiDetectResponse> {
    return apiRequest<PiiDetectResponse>("/api/v1/copilot/pii/detect", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
  },

  sanitizePii(text: string): Promise<PiiSanitizeResponse> {
    return apiRequest<PiiSanitizeResponse>("/api/v1/copilot/pii/sanitize", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
  },

  fastScan(rawText: string): Promise<CopilotAnalysisResponse> {
    return apiRequest<CopilotAnalysisResponse>("/api/v1/copilot/fast-scan", {
      method: "POST",
      body: JSON.stringify({ raw_text: rawText }),
    });
  },

  analyzeCopilot(payload: {
    raw_text: string;
    draft_mode: DraftMode;
    model?: string;
    mock_mode?: boolean;
  }): Promise<CopilotAnalysisResponse> {
    return apiRequest<CopilotAnalysisResponse>("/api/v1/copilot/analyze", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  recalculateScale(payload: {
    analysis_state: CopilotAnalysisResponse;
    scale_id: string;
    updated_params: Record<string, number | null>;
    scale_index?: number;
    refresh_draft: boolean;
    draft_mode?: DraftMode;
    model?: string;
    mock_mode?: boolean;
  }): Promise<CopilotAnalysisResponse> {
    return apiRequest<CopilotAnalysisResponse>("/api/v1/copilot/scales/recalculate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  removeScale(payload: {
    analysis_state: CopilotAnalysisResponse;
    scale_index: number;
  }): Promise<CopilotAnalysisResponse> {
    return apiRequest<CopilotAnalysisResponse>("/api/v1/copilot/scales/remove", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  regenerateDraft(payload: {
    analysis_state: CopilotAnalysisResponse;
    draft_mode: DraftMode;
    model?: string;
    mock_mode?: boolean;
    force?: boolean;
  }): Promise<CopilotAnalysisResponse> {
    return apiRequest<CopilotAnalysisResponse>("/api/v1/copilot/draft/regenerate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  buildHandover(analysis_state: CopilotAnalysisResponse): Promise<HandoverResponse> {
    return apiRequest<HandoverResponse>("/api/v1/copilot/handover", {
      method: "POST",
      body: JSON.stringify({ analysis_state }),
    });
  },

  // ---------------- Case Workspace ----------------
  listCaseWorkspaces(params?: { q?: string; high_risk_only?: boolean }): Promise<CaseListResponse> {
    const query = new URLSearchParams();
    if (params?.q?.trim()) {
      query.set("q", params.q.trim());
    }
    if (params?.high_risk_only) {
      query.set("high_risk_only", "true");
    }
    const qs = query.toString();
    return apiRequest<CaseListResponse>(`/api/v1/cases${qs ? `?${qs}` : ""}`);
  },

  getCaseWorkspace(caseId: string): Promise<CaseWorkspaceRecord> {
    return apiRequest<CaseWorkspaceRecord>(`/api/v1/cases/${encodeURIComponent(caseId)}`);
  },

  saveCaseWorkspace(payload: SaveCaseWorkspaceRequest): Promise<CaseWorkspaceRecord> {
    return apiRequest<CaseWorkspaceRecord>("/api/v1/cases", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  updateCaseWorkspaceMeta(
    caseId: string,
    payload: { title?: string; tags?: string[] }
  ): Promise<CaseWorkspaceRecord> {
    return apiRequest<CaseWorkspaceRecord>(`/api/v1/cases/${encodeURIComponent(caseId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
  },

  appendCaseFollowup(
    caseId: string,
    payload: { visit_date?: string; followup_text: string }
  ): Promise<CaseWorkspaceRecord> {
    return apiRequest<CaseWorkspaceRecord>(
      `/api/v1/cases/${encodeURIComponent(caseId)}/followup`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    );
  },

  restoreCaseRevision(caseId: string, rev: number): Promise<CaseWorkspaceRecord> {
    return apiRequest<CaseWorkspaceRecord>(
      `/api/v1/cases/${encodeURIComponent(caseId)}/restore/${rev}`,
      { method: "POST" }
    );
  },

  deleteCaseWorkspace(caseId: string): Promise<{ deleted?: boolean; case_id?: string }> {
    return apiRequest<{ deleted?: boolean; case_id?: string }>(
      `/api/v1/cases/${encodeURIComponent(caseId)}`,
      { method: "DELETE" }
    );
  },

  exportCaseWorkspace(caseId: string): Promise<CaseWorkspaceRecord> {
    return apiRequest<CaseWorkspaceRecord>(
      `/api/v1/cases/${encodeURIComponent(caseId)}/export`
    );
  },

  importCaseWorkspace(payload: unknown): Promise<CaseWorkspaceRecord> {
    return apiRequest<CaseWorkspaceRecord>("/api/v1/cases/import", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  // ---------------- Benchmark ----------------
  listBenchmarkReports(): Promise<BenchmarkReportsResponse> {
    return apiRequest<BenchmarkReportsResponse>("/api/v1/benchmark/reports");
  },

  getBenchmarkReport(filename: string): Promise<BenchmarkReportDetail> {
    return apiRequest<BenchmarkReportDetail>(
      `/api/v1/benchmark/reports/${encodeURIComponent(filename)}`
    );
  },

  importBenchmarkReport(file: File): Promise<{ filename: string }> {
    const form = new FormData();
    form.append("file", file);
    return apiRequest<{ filename: string }>(
      "/api/v1/benchmark/reports/import",
      { method: "POST", body: form },
      true
    );
  },

  async exportBenchmarkReport(
    filename: string,
    format: "csv" | "json" | "markdown"
  ): Promise<void> {
    const url = buildUrl(
      `/api/v1/benchmark/reports/${encodeURIComponent(filename)}/export?format=${format}`
    );
    const res = await fetch(url, {
      method: "GET",
      headers: buildHeaders(undefined, true),
    });
    if (!res.ok) {
      const { message, detail } = await parseErrorMessage(res);
      throw new ApiError(message, res.status, detail);
    }
    const blob = await res.blob();
    const ext = format === "markdown" ? "md" : format;
    const stem = filename.replace(/\.json$/i, "");
    const dlName = `${stem}.${ext}`;

    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = dlName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(objectUrl);
  },

  startBenchmarkRun(payload: StartBenchmarkRunRequest): Promise<{ job_id: string }> {
    return apiRequest<{ job_id: string }>("/api/v1/benchmark/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async streamBenchmarkRun(
    jobId: string,
    handlers: {
      onProgress?: (event: BenchmarkProgressEvent) => void;
      onDone?: (event: BenchmarkProgressEvent) => void;
      onError?: (message: string, event?: BenchmarkProgressEvent) => void;
    },
    signal?: AbortSignal
  ): Promise<void> {
    const url = buildUrl(`/api/v1/benchmark/runs/${encodeURIComponent(jobId)}/stream`);
    const headers = buildHeaders({ Accept: "text/event-stream" }, true);
    const res = await fetch(url, { method: "GET", headers, signal });
    if (!res.ok) {
      const { message } = await parseErrorMessage(res);
      throw new ApiError(message, res.status);
    }
    if (!res.body) {
      throw new ApiError("浏览器不支持流式响应读取。", 0);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let sawTerminalEvent = false;

    const dispatchBlock = (rawBlock: string) => {
      const lines = rawBlock.split(/\r?\n/);
      let eventName = "message";
      const dataLines: string[] = [];

      for (const line of lines) {
        if (!line || line.startsWith(":")) continue;
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trimStart());
        }
      }
      if (dataLines.length === 0) return;

      const rawData = dataLines.join("\n");
      let parsed: BenchmarkProgressEvent = {};
      try {
        parsed = JSON.parse(rawData) as BenchmarkProgressEvent;
      } catch {
        parsed = { message: rawData };
      }

      if (eventName === "done") {
        sawTerminalEvent = true;
        handlers.onDone?.(parsed);
      } else if (eventName === "error") {
        sawTerminalEvent = true;
        handlers.onError?.(parsed.error || parsed.message || "评测执行出错", parsed);
      } else {
        handlers.onProgress?.(parsed);
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || "";
      for (const blk of blocks) {
        if (blk.trim()) dispatchBlock(blk);
      }
    }

    if (buffer.trim()) {
      dispatchBlock(buffer);
    }

    if (!sawTerminalEvent && !signal?.aborted) {
      handlers.onDone?.({ message: "评测流已结束" });
    }
  },

  // ---------------- Knowledge Base ----------------
  getKbOverview(): Promise<KbOverviewResponse> {
    return apiRequest<KbOverviewResponse>("/api/v1/kb/overview");
  },

  // ---------------- Settings ----------------
  getLlmSettings(): Promise<SettingsLlmResponse> {
    return apiRequest<SettingsLlmResponse>("/api/v1/settings/llm");
  },

  updateLlmSettings(payload: UpdateSettingsLlmRequest): Promise<SettingsLlmResponse> {
    return apiRequest<SettingsLlmResponse>("/api/v1/settings/llm", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  },

  testConnection(payload: {
    provider: string;
    api_key?: string;
    api_base?: string;
    model?: string;
  }): Promise<TestConnectionResponse> {
    return apiRequest<TestConnectionResponse>("/api/v1/settings/test-connection", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  probeOllamaModels(): Promise<OllamaModelsResponse> {
    return apiRequest<OllamaModelsResponse>("/api/v1/settings/ollama-models");
  },
};
