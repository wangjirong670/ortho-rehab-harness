export type PageKey = "copilot" | "benchmark" | "settings";

export type OutputMode = "full" | "screen";
export type ThinkingLevel = "off" | "low" | "medium" | "high";
export type EntrySource = "clinician" | "extracted" | "system";
export type DraftMode = "ward_round" | "rehab_education";

export interface HealthResponse {
  status: string;
  version: string;
  mode: "local_desktop" | "server" | string;
  auth_enabled: boolean;
  mock_mode: boolean;
  default_model: string;
  frontend_ready: boolean;
}

export interface UploadResponse {
  filename: string;
  text: string;
  char_count: number;
}

export interface PiiFinding {
  type: string;
  label: string;
  count: number;
  samples: string[];
}

export interface PiiDetectResponse {
  has_pii: boolean;
  findings: PiiFinding[];
}

export interface PiiSanitizeResponse {
  sanitized_text: string;
  changed: boolean;
  summary: string;
}

export interface TimelineEvent {
  date: string;
  department: string;
  event: string;
  key_change: string;
}

export interface SymptomFact {
  name: string;
  polarity: "present" | "absent" | "uncertain" | string;
}

export interface MedicationFact {
  name: string;
  status: "active" | "stopped" | "unknown" | string;
  stopped_days_ago?: number | null;
}

export interface SupersededCondition {
  old_state: string;
  new_state: string;
  rule_hint?: string | null;
}

export interface CurrentStateSnapshot {
  active_medications?: MedicationFact[];
  stopped_medications?: MedicationFact[];
  current_symptoms?: SymptomFact[];
  active_surgical_history?: string[];
  superseded_conditions?: SupersededCondition[];
  diagnosis_override_note?: string | null;
}

export interface ScaleParamEntry {
  scale_id: string;
  params: Record<string, number | null>;
}

export interface ExtractedNote {
  patient_summary?: string;
  chief_complaint?: string;
  history_of_present_illness?: string;
  tcm_four_diagnostics?: Record<string, string>;
  physical_exam?: string[];
  imaging_summary?: string;
  timeline_events?: TimelineEvent[];
  structured_facts?: {
    symptoms?: SymptomFact[];
    medications?: MedicationFact[];
  };
  current_state_snapshot?: CurrentStateSnapshot;
  scale_params?: ScaleParamEntry[];
}

export interface UnifiedAlert {
  rule_id: string;
  rule_type?: string;
  name: string;
  category?: string;
  severity: "critical" | "high" | "medium" | "low" | string;
  source?: string;
  source_label?: string;
  matched_evidence?: string[];
  clinical_action?: string;
  clinical_alert?: string;
  clinical_rationale?: string;
  contraindicated_actions?: string[];
  reference?: string[];
  references?: string[];
  [key: string]: unknown;
}

export interface SuppressedMatch {
  rule_id?: string;
  rule_type?: string;
  term?: string;
  reason?: string;
  source?: string;
  detail?: string;
  [key: string]: unknown;
}

export interface RiskReport {
  risk_level?: string;
  recommended_action?: string;
  alerts?: UnifiedAlert[];
  suppressed_matches?: SuppressedMatch[];
  summary?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface SubscaleScore {
  label: string;
  score: number;
  min_score: number;
  max_score: number;
  normalized_pct?: number;
}

export interface ImprovementRateResult {
  status: "ok" | "error" | string;
  pre_score?: number;
  post_score?: number;
  max_score?: number;
  improvement_rate_pct?: number | null;
  efficacy_grade?: string;
  formula?: string;
  message?: string;
}

export interface ScaleResult {
  status: "ok" | "error" | string;
  scale_id: string;
  scale_name?: string;
  version_note?: string;
  score_direction?: "higher_is_worse" | "lower_is_worse" | string | null;
  total_score?: number;
  min_total?: number;
  max_total?: number;
  severity_band?: string;
  normalized_score_pct?: number;
  item_scores?: Record<string, number>;
  input_params?: Record<string, number | null>;
  subscale_scores?: Record<string, SubscaleScore>;
  improvement_rate?: ImprovementRateResult;
  red_flag_triggered?: boolean;
  red_flag_reason?: string | null;
  caution_triggered?: boolean;
  caution_reason?: string | null;
  linked_rule_ids?: string[];
  references?: string[];
  error_code?: string;
  message?: string;
  missing_fields?: string[];
  invalid_fields?: Array<{ field: string; reason: string }>;
}

export interface DraftSafetyViolation {
  rule_id: string;
  rule_name?: string;
  matched_action: string;
  snippet?: string;
  reason?: string;
}

export interface DraftDocument {
  mode: DraftMode | string;
  mode_label?: string;
  content: string;
  plain_text?: string;
  generated_at?: string;
  blocked: boolean;
  safety_check?: {
    passed: boolean;
    blocked: boolean;
    violations?: DraftSafetyViolation[];
    message?: string;
  };
  model?: string;
  mock?: boolean;
  usage?: Record<string, unknown>;
}

export type DraftResult = DraftDocument;

export interface CopilotAnalysisResponse {
  raw_text: string;
  fast_scan_only?: boolean;
  extracted_note: ExtractedNote;
  risk_report: RiskReport;
  scale_results: ScaleResult[];
  unified_alerts: UnifiedAlert[];
  contraindicated_capsules?: string[];
  draft: DraftDocument;
  draft_mode: DraftMode | string;
  draft_stale?: boolean;
  draft_plain_text?: string;
  drafts_by_mode?: Partial<Record<DraftMode, DraftResult>>;
  draft_stale_by_mode?: Partial<Record<DraftMode, boolean>>;
  draft_cache_hit?: boolean;
  trace?: Record<string, unknown>;
}

export interface HandoverResponse {
  handover_text: string;
  draft_plain_text: string;
}

// ---------------- Patient Workspace & Session Trace Types (v0.4) ----------------

export interface PatientRedFlagItem {
  id: string;
  name?: string | null;
  since: string;
  status: "active" | "resolved";
  source: EntrySource;
  resolved_reason?: string | null;
  resolved_at?: string | null;
}

export interface PatientMedicationItem {
  name: string;
  status: "active" | "stopped" | "unknown";
  start_date?: string | null;
  stop_date?: string | null;
  stopped_days_ago?: number | null;
  source: EntrySource;
}

export interface PatientScaleRecord {
  scale_id: string;
  date: string;
  total_score?: number | null;
  severity_band?: string | null;
  status: string;
  source: EntrySource;
  params?: Record<string, unknown>;
}

export interface PatientCard {
  patient_id: string;
  display_name: string;
  active_red_flags: PatientRedFlagItem[];
  medications: PatientMedicationItem[];
  allergies: string[];
  surgical_history: string[];
  scale_history: PatientScaleRecord[];
  created_at?: string | null;
  updated_at?: string | null;
  revision: number;
}

export interface PatientSummary {
  patient_id: string;
  display_name: string;
  active_red_flag_count: number;
  active_red_flag_ids: string[];
  session_count: number;
  revision: number;
  created_at: string;
  updated_at: string;
}

export interface PatientDetailResponse {
  patient_id: string;
  display_name: string;
  revision: number;
  card: PatientCard;
  body: string;
  raw_markdown: string;
}

export interface PatientHistoryItem {
  revision: number;
  updated_at: string;
  active_red_flag_count: number;
  active_red_flag_ids: string[];
  medication_count: number;
}

export interface SessionSummary {
  session_id: string;
  patient_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turn_count: number;
  alert_count: number;
  has_critical: boolean;
  pending_diff_count: number;
}

export interface VerificationResult {
  passed: boolean;
  blocked: boolean;
  violations?: DraftSafetyViolation[];
  message?: string;
}

export interface ReasoningVerification {
  checked: boolean;
  has_conflict: boolean;
  violations?: DraftSafetyViolation[];
  message?: string;
}

export interface ModelCapability {
  model: string;
  supports_thinking: boolean;
  mechanism: string;
  supported_levels: ThinkingLevel[];
  default_level: ThinkingLevel;
  returns_reasoning_text: boolean;
  note: string;
}

export interface SessionMessage {
  role: "user" | "assistant" | "system";
  content: string;
  raw_content?: string;
  reasoning?: string;
  reasoning_verification?: ReasoningVerification;
  turn_id?: string | null;
  ts?: string;
  model?: string | null;
  mock?: boolean;
  output_mode?: OutputMode;
  thinking?: ThinkingLevel;
  blocked?: boolean;
  verification?: VerificationResult;
  alerts?: UnifiedAlert[];
  scale_results?: ScaleResult[];
  stale?: boolean;
}

export interface CardDiffPayload {
  diff_id?: string;
  turn_id?: string | null;
  ts?: string;
  auto_red_flags?: Array<{
    id: string;
    name: string;
    severity: string;
    source: string;
  }>;
  add_medications?: PatientMedicationItem[];
  add_surgical_history?: string[];
  add_scale_history?: PatientScaleRecord[];
}

export interface SessionEducationItem {
  ts: string;
  turn_id?: string | null;
  content: string;
  blocked: boolean;
  model?: string | null;
  verification?: VerificationResult;
}

export interface SessionState {
  session_id: string;
  patient_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: SessionMessage[];
  history_for_llm: Array<{ role: string; content: string }>;
  latest_extracted_note: ExtractedNote;
  latest_risk_report: RiskReport;
  latest_alerts: UnifiedAlert[];
  latest_scale_results: ScaleResult[];
  pending_card_diffs: CardDiffPayload[];
  educations: SessionEducationItem[];
  event_count: number;
}

export interface TraceEvent {
  seq: number;
  ts: string;
  turn_id?: string | null;
  type: string;
  payload: Record<string, unknown>;
}

export interface ChatStepEvent {
  turn_id?: string;
  step: string;
  status: "running" | "done" | string;
  label: string;
}

export interface ChatAlertEvent {
  turn_id?: string;
  unified_alerts: UnifiedAlert[];
  risk_report?: RiskReport;
  scale_results?: ScaleResult[];
  inherited_red_flags?: string[];
  auto_saved_red_flags?: string[];
}

export interface ChatContentEvent {
  turn_id?: string;
  content: string;
  model?: string;
  mock?: boolean;
  output_mode?: OutputMode;
  blocked?: boolean;
  latency_ms?: number;
}

export interface ChatDoneEvent {
  turn_id?: string;
  patient_id: string;
  session_id: string;
  blocked?: boolean;
  alert_count?: number;
}

export interface SessionEducationResponse {
  patient_id: string;
  session_id: string;
  turn_id: string;
  mode: string;
  mode_label: string;
  content: string;
  raw_content?: string;
  blocked: boolean;
  safety_check: VerificationResult;
  inherited_red_flags?: string[];
  model?: string;
  mock?: boolean;
}

export interface SessionScaleResponse {
  scale_result: ScaleResult;
  auto_saved_red_flags: string[];
  session: SessionState;
}

// ---------------- Benchmark Types ----------------

export interface BenchmarkReportMeta {
  filename: string;
  label: string;
  created_at: string;
  mock_mode: boolean;
  case_count: number;
  include_private: boolean;
  models: string[];
}

export interface BenchmarkReportsResponse {
  reports: BenchmarkReportMeta[];
}

export interface BenchmarkRunSummary {
  total_cases?: number;
  passed_cases?: number;
  overall_pass_rate?: number;
  safety_veto_count?: number;
  safety_veto_rate?: number;
  over_refusal_count?: number;
  over_refusal_applicable?: number;
  over_refusal_rate?: number;
  dimensions?: Record<
    string,
    {
      total: number;
      passed: number;
      pass_rate: number;
    }
  >;
  [key: string]: unknown;
}

export interface BenchmarkRun {
  run_label: string;
  model: string;
  mode: "bare_llm" | "harness_copilot" | string;
  total_latency_ms?: number;
  summary: BenchmarkRunSummary;
  results?: BenchmarkCaseResult[];
}

export interface BenchmarkPair {
  pair_label: string;
  bare_run_label: string;
  harness_run_label: string;
  bare_summary: BenchmarkRunSummary;
  harness_summary: BenchmarkRunSummary;
  deltas: {
    overall_pass_rate: number;
    safety_veto_rate: number;
    red_flag_interception: number;
    longitudinal_memory: number;
    tcm_syndrome_and_tools: number;
    over_refusal_calibration: number;
    [key: string]: number;
  };
}

export interface HighlightSpan {
  start: number;
  end: number;
  text: string;
  reason: string;
}

export interface BenchmarkTurn {
  turn: number;
  user_msg: string;
  assistant_msg: string;
  latency_ms?: number;
}

export interface ForbiddenViolation {
  turn?: number;
  matched_text?: string;
  pattern?: string;
  rationale?: string;
  snippet?: string;
}

export interface SuppressedForbiddenHit {
  turn?: number;
  matched_text?: string;
  negation_token?: string;
  reason?: string;
}

export interface BenchmarkCaseResult {
  case_id: string;
  status?: string;
  passed?: boolean;
  category?: string;
  dimension?: string;
  risk_level?: string;
  expected_action?: string;
  visibility?: string;
  failure_reasons?: string[];
  extraction_failed?: boolean;
  extraction_error?: string;
  infra_error?: string;
  rule_engine_check?: {
    checked?: boolean;
    matched?: boolean;
    expected_rule_ids?: string[];
    triggered_rule_ids?: string[];
  };
  forbidden_violations?: ForbiddenViolation[];
  suppressed_forbidden_hits?: SuppressedForbiddenHit[];
  must_mention_matched?: string[];
  must_mention_missing?: string[];
  numeric_target_checks?: Record<string, unknown>;
  turns?: BenchmarkTurn[];
}

export interface SideBySideCase {
  pair_label: string;
  case_id: string;
  category: string;
  dimension: string;
  dimension_label?: string;
  risk_level: string;
  expected_action: string;
  visibility: string;
  bare_run_label: string;
  harness_run_label: string;
  rescued: boolean;
  regressed: boolean;
  any_failed: boolean;
  bare_result: BenchmarkCaseResult | null;
  harness_result: BenchmarkCaseResult | null;
  bare_highlights?: Record<string, HighlightSpan[]>;
  harness_highlights?: Record<string, HighlightSpan[]>;
}

export interface BenchmarkReportDetail {
  filename: string;
  meta: {
    created_at: string;
    mock_mode: boolean;
    case_count: number;
    include_private: boolean;
    report_id?: string;
  };
  runs: BenchmarkRun[];
  pairs: BenchmarkPair[];
  dimension_rows: Array<Record<string, string | number>>;
  side_by_side_cases: SideBySideCase[];
  infra_errors?: Array<{ model?: string; error?: string } | [string, string]>;
}

export interface StartBenchmarkRunRequest {
  models: string[];
  modes: Array<"bare_llm" | "harness_copilot">;
  mock_mode: boolean;
  include_private: boolean;
  model_thinking?: Record<string, ThinkingLevel>;
}

export interface BenchmarkProgressEvent {
  progress?: number;
  completed?: number;
  total?: number;
  message?: string;
  model?: string;
  mode?: string;
  case_id?: string;
  passed?: boolean;
  status?: string;
  current_model?: string;
  current_mode?: string;
  current_case?: string;
  filename?: string;
  report_id?: string;
  error?: string;
  [key: string]: unknown;
}

// ---------------- Knowledge Base Types ----------------

export interface RedFlagRule {
  rule_id: string;
  name: string;
  category: string;
  severity: string;
  review_status?: string;
  trigger_symptoms: {
    required_any: string[];
    supporting_any?: string[];
  };
  negation_guards?: string[];
  contraindicated_actions: string[];
  recommended_action: string;
  clinical_alert: string;
  clinical_rationale: string;
  references: string[];
}

export interface DrugGroup {
  label: string;
  keywords: string[];
  washout_window_days?: number | null;
}

export interface DrugConflictRule {
  conflict_id: string;
  name: string;
  severity: string;
  review_status?: string;
  group_a: DrugGroup;
  group_b: DrugGroup;
  risk_mechanism: string;
  clinical_alert: string;
  suggested_handling: string;
  references: string[];
}

export interface ScaleSubscale {
  subscale_id: string;
  label: string;
  min_score: number;
  max_score: number;
  item_count: number;
}

export interface ScaleItemSpec {
  item_id: string;
  label: string;
  subscale_id?: string | null;
  min_value: number;
  max_value: number;
  step?: number | null;
  allowed_values?: number[] | null;
  required?: boolean;
}

export interface SeverityBand {
  label: string;
  min_score: number;
  max_score: number;
}

export interface ScaleSpec {
  scale_id: string;
  name: string;
  version_note: string;
  review_status?: string;
  direction: "higher_is_worse" | "higher_is_better" | string;
  min_total: number;
  max_total: number;
  formula_type: string;
  subscales?: ScaleSubscale[];
  items: ScaleItemSpec[];
  severity_bands: SeverityBand[];
  improvement_rate_formula?: string | null;
  references: string[];
}

export interface KbOverviewResponse {
  red_flags: RedFlagRule[] | { rules: RedFlagRule[] };
  drug_conflicts: DrugConflictRule[] | { conflicts: DrugConflictRule[] };
  scales: ScaleSpec[] | { scales: ScaleSpec[] };
  labels: {
    clinical_actions?: Record<string, string>;
    tcm_fields?: Record<string, string>;
    dimensions?: Record<string, string>;
    severity?: Record<string, string>;
    score_direction?: Record<string, string>;
    womac_scores?: Record<string, string>;
  };
}

// ---------------- Settings Types ----------------

export interface ProviderConfigItem {
  key?: string;
  id?: string;
  label: string;
  api_key_env?: string;
  env_key?: string;
  api_base_env?: string;
  base_env_key?: string;
  has_key?: boolean;
  api_key_set?: boolean;
  masked_key?: string;
  api_key_masked?: string;
  api_base?: string;
  default_model?: string;
  model_suggestions?: string[];
  models?: string[];
  needs_key?: boolean;
  description?: string;
  [key: string]: unknown;
}

export interface SettingsLlmResponse {
  copilot_default_model: string;
  benchmark_models: string | string[];
  llm_mock_mode: boolean;
  preset_models: string[];
  model_capabilities?: Record<string, ModelCapability>;
  providers: ProviderConfigItem[] | Record<string, ProviderConfigItem>;
  ollama_models: string[];
  compliance_notice: string;
}

// ---------------- Free Red-Team Types (v0.4 P2) ----------------

export interface FreeRedTeamTurnRequest {
  user_msg: string;
  bare_history: Array<{ role: string; content: string }>;
  harness_history: Array<{ role: string; content: string }>;
  model?: string;
  thinking: ThinkingLevel;
  mock_mode?: boolean;
}

export interface FreeRedTeamTurnResponse {
  model: string;
  mock: boolean;
  thinking: ThinkingLevel;
  bare: {
    content: string;
    reasoning?: string;
    latency_ms: number;
    breached: boolean;
    verification: VerificationResult;
    error?: string | null;
  };
  harness: {
    content: string;
    raw_content: string;
    reasoning?: string;
    reasoning_verification?: ReasoningVerification;
    latency_ms: number;
    blocked: boolean;
    breached: boolean;
    verification: VerificationResult;
    unified_alerts: UnifiedAlert[];
    scale_results: ScaleResult[];
  };
  guardrail_gap: boolean;
}

export interface SaveRedTeamCaseRequest {
  case_id: string;
  category: string;
  dimension: "red_flag_interception" | "longitudinal_memory" | "tcm_syndrome_and_tools" | "over_refusal_calibration";
  risk_level: "critical" | "high" | "medium" | "low";
  patient_profile: string;
  expected_rule_ids: string[];
  expected_action: string;
  must_mention_any: string[];
  forbidden_patterns: string[];
  turns: Array<{ turn: number; user_msg: string }>;
}

export interface SaveRedTeamCaseResponse {
  saved: boolean;
  case_id: string;
  path: string;
}

export interface UpdateSettingsLlmRequest {
  copilot_default_model?: string;
  benchmark_models?: string;
  llm_mock_mode?: boolean;
  env_updates?: Record<string, string>;
}

export interface TestConnectionResponse {
  ok: boolean;
  message: string;
  latency_ms?: number | null;
  models?: string[];
}

export interface OllamaModelsResponse {
  models: string[];
  reachable: boolean;
}
