import type {
  BenchmarkProgressEvent,
  BenchmarkReportDetail,
  BenchmarkReportsResponse,
  CardDiffPayload,
  ChatAlertEvent,
  ChatContentEvent,
  ChatDoneEvent,
  ChatStepEvent,
  HandoverResponse,
  HealthResponse,
  KbOverviewResponse,
  OllamaModelsResponse,
  OutputMode,
  PatientCard,
  PatientDetailResponse,
  PatientHistoryItem,
  PatientSummary,
  PiiDetectResponse,
  PiiSanitizeResponse,
  SessionEducationResponse,
  SessionScaleResponse,
  SessionState,
  SessionSummary,
  SettingsLlmResponse,
  StartBenchmarkRunRequest,
  TestConnectionResponse,
  ThinkingLevel,
  TraceEvent,
  UpdateSettingsLlmRequest,
  UploadResponse,
  VerificationResult,
} from "./types";

export const STORAGE_KEYS = {
  AUTH_TOKEN: "ortho_auth_token",
  API_BASE_URL: "ortho_api_base_url",
  PREFERENCES: "ortho_ui_preferences",
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
      message: "鉴权失败（401/403），请在「设置」中核对访问令牌 (Bearer Token)。",
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

function triggerDownloadBlob(blob: Blob, filename: string): void {
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objectUrl);
}

export interface ChatStreamHandlers {
  onStep?: (ev: ChatStepEvent) => void;
  onAlert?: (ev: ChatAlertEvent) => void;
  onReasoning?: (ev: { turn_id?: string; reasoning: string }) => void;
  onContent?: (ev: ChatContentEvent) => void;
  onVerification?: (ev: VerificationResult & { turn_id?: string }) => void;
  onCardDiff?: (ev: CardDiffPayload) => void;
  onDone?: (ev: ChatDoneEvent) => void;
  onError?: (message: string) => void;
}

export const api = {
  health(): Promise<HealthResponse> {
    return apiRequest<HealthResponse>("/api/v1/health");
  },

  // ---------------- Copilot Helpers (Upload / PII / Handover) ----------------
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

  exportHandover(analysis_state: Record<string, unknown>): Promise<HandoverResponse> {
    return apiRequest<HandoverResponse>("/api/v1/copilot/handover", {
      method: "POST",
      body: JSON.stringify({ analysis_state }),
    });
  },

  // ---------------- Patients Workspace (v0.4 PATIENT.md) ----------------
  listPatients(): Promise<{ patients: PatientSummary[] }> {
    return apiRequest<{ patients: PatientSummary[] }>("/api/v1/patients");
  },

  createPatient(payload: {
    display_name: string;
    patient_id?: string;
    body?: string;
    card?: Partial<PatientCard>;
  }): Promise<PatientDetailResponse> {
    return apiRequest<PatientDetailResponse>("/api/v1/patients", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  getPatient(patientId: string): Promise<PatientDetailResponse> {
    return apiRequest<PatientDetailResponse>(`/api/v1/patients/${encodeURIComponent(patientId)}`);
  },

  updatePatient(
    patientId: string,
    payload: { card: PatientCard; body: string }
  ): Promise<PatientDetailResponse> {
    return apiRequest<PatientDetailResponse>(`/api/v1/patients/${encodeURIComponent(patientId)}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  },

  commitPatientChanges(
    patientId: string,
    payload: {
      changes: Record<string, unknown>;
      session_id?: string;
      diff_id?: string;
    }
  ): Promise<PatientDetailResponse> {
    return apiRequest<PatientDetailResponse>(
      `/api/v1/patients/${encodeURIComponent(patientId)}/commit`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    );
  },

  listPatientHistory(
    patientId: string
  ): Promise<{ patient_id: string; revisions: PatientHistoryItem[] }> {
    return apiRequest<{ patient_id: string; revisions: PatientHistoryItem[] }>(
      `/api/v1/patients/${encodeURIComponent(patientId)}/history`
    );
  },

  restorePatientRevision(patientId: string, rev: number): Promise<PatientDetailResponse> {
    return apiRequest<PatientDetailResponse>(
      `/api/v1/patients/${encodeURIComponent(patientId)}/restore/${rev}`,
      { method: "POST" }
    );
  },

  deletePatient(patientId: string): Promise<{ deleted: boolean; patient_id: string }> {
    return apiRequest<{ deleted: boolean; patient_id: string }>(
      `/api/v1/patients/${encodeURIComponent(patientId)}`,
      { method: "DELETE" }
    );
  },

  // ---------------- Clinical Sessions & Event Trace (v0.4) ----------------
  listSessions(
    patientId: string
  ): Promise<{ patient_id: string; sessions: SessionSummary[] }> {
    return apiRequest<{ patient_id: string; sessions: SessionSummary[] }>(
      `/api/v1/patients/${encodeURIComponent(patientId)}/sessions`
    );
  },

  createSession(patientId: string, title = ""): Promise<SessionState> {
    return apiRequest<SessionState>(
      `/api/v1/patients/${encodeURIComponent(patientId)}/sessions`,
      {
        method: "POST",
        body: JSON.stringify({ title }),
      }
    );
  },

  getSession(patientId: string, sessionId: string): Promise<SessionState> {
    return apiRequest<SessionState>(
      `/api/v1/patients/${encodeURIComponent(patientId)}/sessions/${encodeURIComponent(sessionId)}`
    );
  },

  getSessionTrace(
    patientId: string,
    sessionId: string,
    turnId?: string
  ): Promise<{
    patient_id: string;
    session_id: string;
    turn_id?: string | null;
    events: TraceEvent[];
  }> {
    const qs = turnId ? `?turn_id=${encodeURIComponent(turnId)}` : "";
    return apiRequest(
      `/api/v1/patients/${encodeURIComponent(patientId)}/sessions/${encodeURIComponent(sessionId)}/trace${qs}`
    );
  },

  async downloadSessionTrace(patientId: string, sessionId: string): Promise<void> {
    const url = buildUrl(
      `/api/v1/patients/${encodeURIComponent(patientId)}/sessions/${encodeURIComponent(sessionId)}/trace?download=true`
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
    triggerDownloadBlob(blob, `${sessionId}.trace.jsonl`);
  },

  async streamSessionChat(
    patientId: string,
    sessionId: string,
    payload: {
      content: string;
      model?: string;
      thinking?: ThinkingLevel;
      output_mode?: OutputMode;
      mock_mode?: boolean;
      pii_action?: "none" | "sanitized" | "override_raw";
    },
    handlers: ChatStreamHandlers,
    signal?: AbortSignal
  ): Promise<void> {
    const url = buildUrl(
      `/api/v1/patients/${encodeURIComponent(patientId)}/sessions/${encodeURIComponent(sessionId)}/chat`
    );
    const res = await fetch(url, {
      method: "POST",
      headers: buildHeaders({ Accept: "text/event-stream" }),
      body: JSON.stringify(payload),
      signal,
    });
    if (!res.ok) {
      const { message, detail } = await parseErrorMessage(res);
      throw new ApiError(message, res.status, detail);
    }
    if (!res.body) {
      throw new ApiError("浏览器不支持流式响应读取。", 0);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let sawTerminal = false;

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
      let parsed: Record<string, unknown> = {};
      try {
        parsed = JSON.parse(rawData) as Record<string, unknown>;
      } catch {
        parsed = { message: rawData };
      }

      if (eventName === "step") {
        handlers.onStep?.(parsed as unknown as ChatStepEvent);
      } else if (eventName === "alert") {
        handlers.onAlert?.(parsed as unknown as ChatAlertEvent);
      } else if (eventName === "reasoning") {
        handlers.onReasoning?.(parsed as unknown as { turn_id?: string; reasoning: string });
      } else if (eventName === "content") {
        handlers.onContent?.(parsed as unknown as ChatContentEvent);
      } else if (eventName === "verification") {
        handlers.onVerification?.(parsed as unknown as VerificationResult & { turn_id?: string });
      } else if (eventName === "card_diff") {
        handlers.onCardDiff?.(parsed as unknown as CardDiffPayload);
      } else if (eventName === "done") {
        sawTerminal = true;
        handlers.onDone?.(parsed as unknown as ChatDoneEvent);
      } else if (eventName === "error") {
        sawTerminal = true;
        handlers.onError?.(String(parsed.message || "对话流执行出错"));
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

    if (!sawTerminal && !signal?.aborted) {
      handlers.onDone?.({ patient_id: patientId, session_id: sessionId });
    }
  },

  generateSessionEducation(
    patientId: string,
    sessionId: string,
    payload: { model?: string; mock_mode?: boolean; turn_id?: string } = {}
  ): Promise<SessionEducationResponse> {
    return apiRequest<SessionEducationResponse>(
      `/api/v1/patients/${encodeURIComponent(patientId)}/sessions/${encodeURIComponent(sessionId)}/education`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    );
  },

  recalculateSessionScale(
    patientId: string,
    sessionId: string,
    payload: { scale_id: string; updated_params: Record<string, number | null> }
  ): Promise<SessionScaleResponse> {
    return apiRequest<SessionScaleResponse>(
      `/api/v1/patients/${encodeURIComponent(patientId)}/sessions/${encodeURIComponent(sessionId)}/scales`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    );
  },

  dismissSessionDiff(
    patientId: string,
    sessionId: string,
    diffId: string
  ): Promise<{ session: SessionState }> {
    return apiRequest<{ session: SessionState }>(
      `/api/v1/patients/${encodeURIComponent(patientId)}/sessions/${encodeURIComponent(sessionId)}/diffs/${encodeURIComponent(diffId)}/dismiss`,
      { method: "POST" }
    );
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
    triggerDownloadBlob(blob, `${stem}.${ext}`);
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
