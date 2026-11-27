import React, { useMemo, useState } from "react";
import { AlertOctagon, Lock, Pill, Ruler, Search, ShieldAlert } from "lucide-react";
import type { KbOverviewResponse } from "../api/types";
import {
  getClinicalActionLabel,
  getScoreDirectionLabel,
  SEVERITY_LABELS,
} from "../utils/labels";

interface KnowledgeBasePanelProps {
  kb: KbOverviewResponse | null;
  loading?: boolean;
}

type KbTab = "red_flags" | "drug_conflicts" | "scales";

export const KnowledgeBasePanel: React.FC<KnowledgeBasePanelProps> = ({
  kb,
  loading = false,
}) => {
  const [activeTab, setActiveTab] = useState<KbTab>("red_flags");
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();

  const filteredRedFlags = useMemo(() => {
    const list = Array.isArray(kb?.red_flags) ? kb.red_flags : kb?.red_flags?.rules || [];
    if (!needle) return list;
    return list.filter((r) =>
      [
        r.rule_id,
        r.name,
        r.category,
        r.clinical_alert,
        r.clinical_rationale,
        ...(r.trigger_symptoms?.required_any || []),
        ...(r.contraindicated_actions || []),
        ...(r.references || []),
      ]
        .join(" ")
        .toLowerCase()
        .includes(needle)
    );
  }, [kb?.red_flags, needle]);

  const filteredDrugConflicts = useMemo(() => {
    const list = Array.isArray(kb?.drug_conflicts)
      ? kb.drug_conflicts
      : kb?.drug_conflicts?.conflicts || [];
    if (!needle) return list;
    return list.filter((c) =>
      [
        c.conflict_id,
        c.name,
        c.group_a.label,
        ...c.group_a.keywords,
        c.group_b.label,
        ...c.group_b.keywords,
        c.risk_mechanism,
        c.clinical_alert,
        c.suggested_handling,
      ]
        .join(" ")
        .toLowerCase()
        .includes(needle)
    );
  }, [kb?.drug_conflicts, needle]);

  const filteredScales = useMemo(() => {
    const list = Array.isArray(kb?.scales) ? kb.scales : kb?.scales?.scales || [];
    if (!needle) return list;
    return list.filter((s) =>
      [
        s.scale_id,
        s.name,
        s.version_note,
        ...s.items.map((it) => it.label),
        ...(s.references || []),
      ]
        .join(" ")
        .toLowerCase()
        .includes(needle)
    );
  }, [kb?.scales, needle]);

  if (loading && !kb) {
    return (
      <div className="glass-surface flex min-h-[300px] items-center justify-center rounded-3xl text-xs text-slate-500">
        正在加载中医骨伤与康复知识库...
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-amber-200/80 bg-amber-50/70 px-4 py-2.5 text-xs text-amber-900">
        <div className="flex items-center gap-2 font-medium">
          <Lock className="h-3.5 w-3.5 shrink-0 text-amber-700" />
          <span>
            确定性医学知识库（只读）：规则修改须编辑{" "}
            <code className="rounded bg-amber-100/90 px-1.5 py-0.5 font-mono text-[11px]">
              knowledge_base/*.yaml
            </code>{" "}
            并执行{" "}
            <code className="rounded bg-amber-100/90 px-1.5 py-0.5 font-mono text-[11px]">
              scripts/validate_kb.py
            </code>{" "}
            全量校验。
          </span>
        </div>
      </div>

      <div className="glass-surface flex flex-wrap items-center justify-between gap-3 rounded-3xl p-4">
        <div className="flex flex-wrap gap-1.5 rounded-2xl bg-slate-900/5 p-1">
          <button
            type="button"
            onClick={() => setActiveTab("red_flags")}
            className={`inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-semibold transition-all ${
              activeTab === "red_flags"
                ? "bg-white text-rose-800 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <ShieldAlert className="h-3.5 w-3.5" />
            红旗征规则 ({filteredRedFlags.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("drug_conflicts")}
            className={`inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-semibold transition-all ${
              activeTab === "drug_conflicts"
                ? "bg-white text-amber-800 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Pill className="h-3.5 w-3.5" />
            用药冲突与洗脱期 ({filteredDrugConflicts.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("scales")}
            className={`inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-semibold transition-all ${
              activeTab === "scales"
                ? "bg-white text-cyan-800 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Ruler className="h-3.5 w-3.5" />
            确定性量表规范 ({filteredScales.length})
          </button>
        </div>

        <div className="relative min-w-[220px] flex-1 sm:max-w-xs">
          <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索症状、药物、洗脱期或量表条目..."
            className="clinical-input py-2 pl-9 text-xs"
          />
        </div>
      </div>

      {activeTab === "red_flags" && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {filteredRedFlags.map((rule) => {
            const sev = SEVERITY_LABELS[rule.severity] || SEVERITY_LABELS.high;
            return (
              <article key={rule.rule_id} className="clinical-paper space-y-3 rounded-3xl p-5">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold ${
                          sev.tone === "critical"
                            ? "bg-rose-700 text-white"
                            : "bg-orange-100 text-orange-800"
                        }`}
                      >
                        {sev.label}
                      </span>
                      <span className="font-mono text-[11px] text-slate-500">{rule.rule_id}</span>
                      <span className="rounded-md bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600">
                        {rule.category}
                      </span>
                    </div>
                    <h3 className="mt-2 text-base font-bold text-slate-900">{rule.name}</h3>
                  </div>

                  <span className="rounded-full bg-cyan-50 px-3 py-1 text-[11px] font-semibold text-cyan-900">
                    {getClinicalActionLabel(rule.recommended_action, kb?.labels?.clinical_actions)}
                  </span>
                </div>

                <div className="rounded-2xl border border-rose-200/70 bg-rose-50/70 p-3 text-xs leading-relaxed text-rose-950">
                  {rule.clinical_alert}
                </div>

                {rule.trigger_symptoms?.required_any?.length > 0 && (
                  <div>
                    <div className="mb-1 text-[11px] font-semibold text-slate-500">
                      触发词 / 特征（任一匹配）
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {rule.trigger_symptoms.required_any.map((kw, i) => (
                        <span
                          key={i}
                          className="rounded-md border border-slate-200 bg-white px-2 py-0.5 text-[11px] text-slate-700"
                        >
                          {kw}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {rule.contraindicated_actions?.length > 0 && (
                  <div>
                    <div className="mb-1 flex items-center gap-1 text-[11px] font-semibold text-rose-800">
                      <AlertOctagon className="h-3 w-3" />
                      绝对禁忌操作
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {rule.contraindicated_actions.map((act, i) => (
                        <span
                          key={i}
                          className="rounded-md border border-rose-200 bg-rose-50 px-2 py-0.5 text-[11px] font-medium text-rose-900"
                        >
                          严禁 {act}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="border-t border-slate-100 pt-2 text-[11px] leading-relaxed text-slate-500">
                  <div>依据机制：{rule.clinical_rationale}</div>
                  {rule.references?.length > 0 && (
                    <div className="mt-1">参考文献：{rule.references.join("；")}</div>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      )}

      {activeTab === "drug_conflicts" && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {filteredDrugConflicts.map((conf) => {
            const sev = SEVERITY_LABELS[conf.severity] || SEVERITY_LABELS.high;
            return (
              <article key={conf.conflict_id} className="clinical-paper space-y-3 rounded-3xl p-5">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <div className="flex items-center gap-2">
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold ${
                          sev.tone === "critical"
                            ? "bg-rose-700 text-white"
                            : "bg-amber-100 text-amber-800"
                        }`}
                      >
                        {sev.label}
                      </span>
                      <span className="font-mono text-[11px] text-slate-500">
                        {conf.conflict_id}
                      </span>
                    </div>
                    <h3 className="mt-2 text-base font-bold text-slate-900">{conf.name}</h3>
                  </div>
                  {conf.group_a.washout_window_days ? (
                    <span className="rounded-full bg-amber-100 px-3 py-1 text-[11px] font-semibold text-amber-900">
                      停药洗脱窗：{conf.group_a.washout_window_days} 天
                    </span>
                  ) : null}
                </div>

                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-3">
                    <div className="text-[11px] font-semibold text-slate-500">
                      药物组 A：{conf.group_a.label}
                    </div>
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {conf.group_a.keywords.map((kw, i) => (
                        <span
                          key={i}
                          className="rounded bg-white px-1.5 py-0.5 text-[11px] text-slate-700 shadow-sm"
                        >
                          {kw}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-3">
                    <div className="text-[11px] font-semibold text-slate-500">
                      冲突组 B：{conf.group_b.label}
                    </div>
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {conf.group_b.keywords.map((kw, i) => (
                        <span
                          key={i}
                          className="rounded bg-white px-1.5 py-0.5 text-[11px] text-slate-700 shadow-sm"
                        >
                          {kw}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl border border-amber-200/80 bg-amber-50/70 p-3 text-xs leading-relaxed text-amber-950">
                  <div>
                    <span className="font-semibold">风险警示：</span>
                    {conf.clinical_alert}
                  </div>
                  <div className="mt-1">
                    <span className="font-semibold">建议处置：</span>
                    {conf.suggested_handling}
                  </div>
                </div>

                <div className="text-[11px] text-slate-500">
                  机制说明：{conf.risk_mechanism}
                </div>
              </article>
            );
          })}
        </div>
      )}

      {activeTab === "scales" && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {filteredScales.map((scale) => (
            <article key={scale.scale_id} className="clinical-paper space-y-3 rounded-3xl p-5">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <span className="rounded-full bg-cyan-100 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-cyan-900">
                    {scale.scale_id}
                  </span>
                  <h3 className="mt-1.5 text-base font-bold text-slate-900">{scale.name}</h3>
                </div>
                <span className="rounded-full bg-slate-100 px-3 py-1 text-[11px] font-medium text-slate-700">
                  满分 {scale.min_total}~{scale.max_total} 分 · {scale.items.length} 项
                </span>
              </div>

              <div className="rounded-2xl border border-slate-200/70 bg-slate-50/70 p-3 text-xs text-slate-700">
                <div>方向说明：{getScoreDirectionLabel(scale.direction)}</div>
                <div className="mt-1 text-slate-500">{scale.version_note}</div>
              </div>

              <div>
                <div className="mb-1.5 text-[11px] font-semibold text-slate-500">
                  严重度分级区间
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {scale.severity_bands.map((band, i) => (
                    <span
                      key={i}
                      className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[11px] text-slate-700"
                    >
                      {band.label}：{band.min_score}~{band.max_score} 分
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-1 text-[11px] font-semibold text-slate-500">评分条目摘要</div>
                <div className="max-h-40 overflow-y-auto rounded-2xl border border-slate-100 bg-slate-50/50 p-2.5 text-[11px] text-slate-600">
                  <ul className="space-y-1">
                    {scale.items.map((it) => (
                      <li key={it.item_id} className="flex items-center justify-between gap-2">
                        <span className="truncate">{it.label}</span>
                        <span className="shrink-0 font-mono text-slate-400">
                          [{it.min_value}~{it.max_value}]
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
};
