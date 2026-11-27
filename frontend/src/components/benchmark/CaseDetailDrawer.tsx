import React from "react";
import {
  CheckCircle2,
  PanelRightClose,
  PanelRightOpen,
  ShieldAlert,
  XCircle,
} from "lucide-react";
import type { HighlightSpan, SideBySideCase } from "../../api/types";
import { BENCHMARK_STATUS_LABELS, DIMENSION_LABELS } from "../../utils/labels";
import { SafeHighlightText } from "../SafeHighlightText";

interface CaseDetailDrawerProps {
  open: boolean;
  selectedCase: SideBySideCase | null;
  onToggleOpen: (open: boolean) => void;
}

function resolveTurnHighlights(
  highlightsMap: Record<string, HighlightSpan[]> | undefined,
  turnIndex: number,
  turnNumber?: number
): HighlightSpan[] {
  if (!highlightsMap) return [];
  if (highlightsMap[String(turnIndex)]) return highlightsMap[String(turnIndex)];
  if (turnNumber !== undefined && highlightsMap[String(turnNumber)]) {
    return highlightsMap[String(turnNumber)];
  }
  return [];
}

export const CaseDetailDrawer: React.FC<CaseDetailDrawerProps> = ({
  open,
  selectedCase,
  onToggleOpen,
}) => {
  if (!open) {
    return (
      <div className="flex shrink-0 flex-col items-center">
        <button
          type="button"
          onClick={() => onToggleOpen(true)}
          className="glass-surface flex flex-col items-center gap-2 rounded-3xl px-2.5 py-4 text-xs font-semibold text-slate-700 hover:bg-white/90"
        >
          <PanelRightOpen className="h-4 w-4" />
          <span className="[writing-mode:vertical-rl]">单题详情</span>
        </button>
      </div>
    );
  }

  const bareStatus =
    BENCHMARK_STATUS_LABELS[selectedCase?.bare_result?.status || ""] || {
      label: selectedCase?.bare_result?.status || "未运行",
      tone: "slate",
    };
  const harnessStatus =
    BENCHMARK_STATUS_LABELS[selectedCase?.harness_result?.status || ""] || {
      label: selectedCase?.harness_result?.status || "未运行",
      tone: "slate",
    };

  const turns =
    selectedCase?.bare_result?.turns || selectedCase?.harness_result?.turns || [];

  return (
    <aside
      data-testid="case-detail-drawer"
      className="glass-surface flex min-h-0 w-full shrink-0 flex-col rounded-3xl md:h-full md:w-[340px] xl:w-[390px]"
    >
      <div className="flex items-center justify-between border-b border-slate-200/70 px-4 py-3">
        <span className="text-xs font-bold text-slate-900">单题攻防详情与判定依据</span>
        <button
          type="button"
          onClick={() => onToggleOpen(false)}
          className="inline-flex items-center gap-1 rounded-xl border border-slate-200 bg-white/80 px-2 py-1 text-[11px] font-medium text-slate-600 hover:bg-white"
        >
          <PanelRightClose className="h-3.5 w-3.5" />
          <span>收起</span>
        </button>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3.5 text-xs">
        {!selectedCase ? (
          <div className="rounded-2xl border border-dashed border-slate-200 p-6 text-center text-slate-400">
            在中间题集列表中点击任意一题，查看多轮诱导话术、裸模型 vs 护栏原始回复及违规词高亮。
          </div>
        ) : (
          <>
            <div className="rounded-2xl border border-slate-200/80 bg-white/90 p-3">
              <div className="flex flex-wrap items-center justify-between gap-1.5">
                <span className="font-mono text-xs font-bold text-slate-900">
                  {selectedCase.case_id}
                </span>
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-700">
                  {DIMENSION_LABELS[selectedCase.dimension] || selectedCase.dimension}
                </span>
              </div>
              <div className="mt-1 font-semibold text-slate-700">{selectedCase.category}</div>

              <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                <div className="rounded-xl border border-slate-200 bg-slate-50 px-2.5 py-1.5">
                  <div className="text-slate-400">裸模型</div>
                  <div className="mt-0.5 font-bold text-slate-800">{bareStatus.label}</div>
                </div>
                <div className="rounded-xl border border-cyan-200 bg-cyan-50/70 px-2.5 py-1.5">
                  <div className="text-cyan-700">Harness 护栏</div>
                  <div className="mt-0.5 font-bold text-cyan-950">{harnessStatus.label}</div>
                </div>
              </div>
            </div>

            {/* Deterministic Rule Engine Audit */}
            {selectedCase.harness_result?.rule_engine_check && (
              <div className="rounded-2xl border border-slate-200/80 bg-white/90 p-3 text-[11px]">
                <div className="flex items-center gap-1 font-bold text-slate-800">
                  <ShieldAlert className="h-3.5 w-3.5 text-cyan-700" />
                  确定性规则引擎对照
                </div>
                <div className="mt-1.5 space-y-1 text-slate-600">
                  <div>
                    预期规则：
                    {(selectedCase.harness_result.rule_engine_check.expected_rule_ids || []).join(
                      "、"
                    ) || "无"}
                  </div>
                  <div>
                    实测触发：
                    {(selectedCase.harness_result.rule_engine_check.triggered_rule_ids || []).join(
                      "、"
                    ) || "无"}
                  </div>
                </div>
              </div>
            )}

            {/* Failure reasons if any */}
            {((selectedCase.bare_result?.failure_reasons || []).length > 0 ||
              (selectedCase.harness_result?.failure_reasons || []).length > 0) && (
              <div className="space-y-1.5 rounded-2xl border border-rose-200 bg-rose-50/75 p-3 text-[11px]">
                <div className="font-bold text-rose-900">判定明细</div>
                {(selectedCase.bare_result?.failure_reasons || []).map((r, i) => (
                  <div key={`bf-${i}`} className="flex items-start gap-1 text-rose-900">
                    <XCircle className="mt-0.5 h-3 w-3 shrink-0" />
                    <span>[裸模型] {r}</span>
                  </div>
                ))}
                {(selectedCase.harness_result?.failure_reasons || []).map((r, i) => (
                  <div key={`hf-${i}`} className="flex items-start gap-1 text-rose-900">
                    <XCircle className="mt-0.5 h-3 w-3 shrink-0" />
                    <span>[Harness] {r}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Turn-by-turn Replay with SafeHighlightText */}
            <div className="space-y-2.5">
              {turns.map((turn, idx) => {
                const bareTurn = selectedCase.bare_result?.turns?.[idx];
                const harnessTurn = selectedCase.harness_result?.turns?.[idx];
                const bareSpans = resolveTurnHighlights(
                  selectedCase.bare_highlights,
                  idx,
                  turn.turn
                );
                const harnessSpans = resolveTurnHighlights(
                  selectedCase.harness_highlights,
                  idx,
                  turn.turn
                );

                return (
                  <div
                    key={idx}
                    className="space-y-2 rounded-2xl border border-slate-200/90 bg-white/95 p-3"
                  >
                    <div className="rounded-xl bg-slate-100/90 px-2.5 py-2">
                      <div className="text-[10px] font-bold text-slate-500">
                        Turn {turn.turn || idx + 1} · 诱导话术
                      </div>
                      <div className="mt-0.5 whitespace-pre-wrap text-slate-800">
                        {turn.user_msg}
                      </div>
                    </div>

                    {bareTurn && (
                      <div className="rounded-xl border border-rose-200/80 bg-rose-50/40 p-2.5">
                        <div className="mb-1 flex items-center justify-between text-[10px] font-bold text-rose-900">
                          <span>裸模型回复</span>
                          {bareSpans.length > 0 && (
                            <span className="rounded bg-rose-700 px-1.5 py-0.5 text-white">
                              高亮 {bareSpans.length} 处
                            </span>
                          )}
                        </div>
                        <SafeHighlightText
                          text={bareTurn.assistant_msg}
                          highlights={bareSpans}
                          tone="danger"
                        />
                      </div>
                    )}

                    {harnessTurn && (
                      <div className="rounded-xl border border-emerald-200/80 bg-emerald-50/40 p-2.5">
                        <div className="mb-1 flex items-center justify-between text-[10px] font-bold text-emerald-900">
                          <span className="flex items-center gap-1">
                            <CheckCircle2 className="h-3 w-3" />
                            Harness 护栏回复
                          </span>
                        </div>
                        <SafeHighlightText
                          text={harnessTurn.assistant_msg}
                          highlights={harnessSpans}
                          tone="safe"
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </aside>
  );
};
