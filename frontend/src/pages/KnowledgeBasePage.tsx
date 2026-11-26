import React, { useMemo, useState } from "react";
import { AlertOctagon, Pill, Ruler, Search, ShieldAlert } from "lucide-react";
import type { KbOverviewResponse } from "../api/types";
import {
  getClinicalActionLabel,
  getScoreDirectionLabel,
  SEVERITY_LABELS,
} from "../utils/labels";

interface KnowledgeBasePageProps {
  kb: KbOverviewResponse | null;
  loading: boolean;
}

type KbTab = "red_flags" | "drug_conflicts" | "scales";

export const KnowledgeBasePage: React.FC<KnowledgeBasePageProps> = ({ kb, loading }) => {
  const [activeTab, setActiveTab] = useState<KbTab>("red_flags");
  const [query, setQuery] = useState("");

  const needle = query.trim().toLowerCase();

  const filteredRedFlags = useMemo(() => {
    const list = Array.isArray(kb?.red_flags)
      ? kb.red_flags
      : kb?.red_flags?.rules || [];
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
    const list = Array.isArray(kb?.scales)
      ? kb.scales
      : kb?.scales?.scales || [];
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
      <div className="glass-surface flex min-h-[360px] items-center justify-center rounded-3xl text-xs text-slate-500">
        正在加载中医骨伤与康复知识库...
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Top Tabs & Search */}
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

        <div className="relative min-w-[240px] flex-1 sm:max-w-xs">
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

      {/* Tab 1: Red Flags */}
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

                <p className="text-xs leading-relaxed text-slate-800">{rule.clinical_alert}</p>

                <div className="rounded-2xl bg-slate-50/90 p-3 text-xs text-slate-600">
                  <span className="font-semibold text-slate-800">临床机理：</span>
                  {rule.clinical_rationale}
                </div>

                <div className="space-y-1.5 text-xs">
                  <div className="font-semibold text-slate-700">核心触发征象</div>
                  <div className="flex flex-wrap gap-1.5">
                    {(rule.trigger_symptoms?.required_any || [])
                      .filter((t) => !t.startsWith("regex:"))
                      .slice(0, 12)
                      .map((kw, i) => (
                        <span
                          key={i}
                          className="rounded-lg border border-rose-200/80 bg-rose-50/70 px-2 py-0.5 text-[11px] text-rose-900"
                        >
                          {kw}
                        </span>
                      ))}
                  </div>
                </div>

                <div className="text-xs text-rose-800">
                  <span className="font-semibold">绝对禁忌操作：</span>
                  {rule.contraindicated_actions.join("；")}
                </div>

                <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-2.5 text-[11px] text-slate-500">
                  <span>依据：{rule.references.join("；")}</span>
                  <span className="font-mono text-[10px] text-slate-400">{rule.rule_id}</span>
                </div>
              </article>
            );
          })}
        </div>
      )}

      {/* Tab 2: Drug Conflicts & Washout Windows */}
      {activeTab === "drug_conflicts" && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {filteredDrugConflicts.map((c) => {
            const sev = SEVERITY_LABELS[c.severity] || SEVERITY_LABELS.high;
            return (
              <article key={c.conflict_id} className="clinical-paper space-y-3 rounded-3xl p-5">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold ${
                          sev.tone === "critical"
                            ? "bg-rose-700 text-white"
                            : "bg-amber-100 text-amber-800"
                        }`}
                      >
                        {sev.label}
                      </span>
                      {c.group_a.washout_window_days != null && (
                        <span className="inline-flex items-center gap-1 rounded-full border border-rose-200 bg-rose-50 px-2.5 py-0.5 text-[11px] font-bold text-rose-800">
                          <AlertOctagon className="h-3 w-3" />
                          A 组洗脱期 {c.group_a.washout_window_days} 天
                        </span>
                      )}
                      {c.group_b.washout_window_days != null && (
                        <span className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-[11px] font-bold text-amber-800">
                          B 组洗脱期 {c.group_b.washout_window_days} 天
                        </span>
                      )}
                    </div>
                    <h3 className="mt-2 text-base font-bold text-slate-900">{c.name}</h3>
                  </div>
                  <span className="font-mono text-[11px] text-slate-400">{c.conflict_id}</span>
                </div>

                <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                  <div className="rounded-2xl border border-slate-200/80 bg-slate-50/75 p-3 text-xs">
                    <div className="flex items-center justify-between font-semibold text-slate-800">
                      <span>{c.group_a.label}</span>
                      {c.group_a.washout_window_days != null && (
                        <span className="text-[11px] font-bold text-rose-700">
                          洗脱 {c.group_a.washout_window_days} 天
                        </span>
                      )}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {c.group_a.keywords.slice(0, 10).map((kw, i) => (
                        <span
                          key={i}
                          className="rounded-md bg-white px-2 py-0.5 text-[11px] text-slate-700 shadow-sm"
                        >
                          {kw}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-200/80 bg-slate-50/75 p-3 text-xs">
                    <div className="flex items-center justify-between font-semibold text-slate-800">
                      <span>{c.group_b.label}</span>
                      {c.group_b.washout_window_days != null && (
                        <span className="text-[11px] font-bold text-amber-700">
                          洗脱 {c.group_b.washout_window_days} 天
                        </span>
                      )}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {c.group_b.keywords.slice(0, 10).map((kw, i) => (
                        <span
                          key={i}
                          className="rounded-md bg-white px-2 py-0.5 text-[11px] text-slate-700 shadow-sm"
                        >
                          {kw}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <p className="text-xs leading-relaxed text-slate-800">{c.clinical_alert}</p>
                <div className="rounded-xl bg-cyan-50/75 px-3 py-2 text-xs text-cyan-950">
                  <span className="font-semibold">建议临床处置：</span>
                  {c.suggested_handling}
                </div>
                <div className="border-t border-slate-100 pt-2 text-[11px] text-slate-500">
                  参考依据：{c.references.join("；")}
                </div>
              </article>
            );
          })}
        </div>
      )}

      {/* Tab 3: Scales Specifications */}
      {activeTab === "scales" && (
        <div className="space-y-4">
          {filteredScales.map((scale) => (
            <article key={scale.scale_id} className="clinical-paper space-y-4 rounded-3xl p-5">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-base font-bold text-slate-900">{scale.name}</h3>
                    <span className="rounded-md bg-slate-100 px-2 py-0.5 font-mono text-[11px] text-slate-500">
                      {scale.scale_id}
                    </span>
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-slate-600">{scale.version_note}</p>
                </div>

                <div className="flex flex-wrap gap-2 text-xs">
                  <span className="rounded-full bg-cyan-50 px-3 py-1 font-semibold text-cyan-900">
                    分值：{scale.min_total} ~ {scale.max_total} 分
                  </span>
                  <span className="rounded-full bg-slate-100 px-3 py-1 font-medium text-slate-700">
                    {getScoreDirectionLabel(scale.direction)}
                  </span>
                </div>
              </div>

              {/* Severity Bands */}
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {scale.severity_bands.map((b, i) => (
                  <div key={i} className="rounded-2xl border border-slate-200/80 bg-slate-50/80 p-2.5 text-xs">
                    <div className="font-semibold text-slate-900">{b.label}</div>
                    <div className="mt-0.5 font-mono text-[11px] text-cyan-800">
                      {b.min_score} ~ {b.max_score} 分
                    </div>
                  </div>
                ))}
              </div>

              {/* Subscales if present */}
              {scale.subscales && scale.subscales.length > 0 && (
                <div className="flex flex-wrap gap-2 text-xs">
                  {scale.subscales.map((sub) => (
                    <span
                      key={sub.subscale_id}
                      className="rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-slate-700"
                    >
                      <strong>{sub.label}</strong> · 区间 {sub.min_score}~{sub.max_score}
                    </span>
                  ))}
                </div>
              )}

              <details className="rounded-2xl border border-slate-200/80 bg-slate-50/60 p-3 text-xs">
                <summary className="cursor-pointer font-semibold text-slate-800">
                  查看全部评分条目（{scale.items.length} 项）
                </summary>
                <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
                  {scale.items.map((it) => (
                    <div
                      key={it.item_id}
                      className="flex items-center justify-between rounded-xl bg-white px-3 py-2"
                    >
                      <span className="text-slate-800">{it.label}</span>
                      <span className="ml-2 shrink-0 font-mono text-[11px] text-slate-500">
                        {it.allowed_values
                          ? it.allowed_values.join("/")
                          : `${it.min_value}~${it.max_value}`}
                      </span>
                    </div>
                  ))}
                </div>
              </details>

              {scale.improvement_rate_formula && (
                <div className="rounded-xl bg-emerald-50/80 px-3 py-2 text-xs text-emerald-900">
                  改善率计算公式：<code className="font-mono">{scale.improvement_rate_formula}</code>
                </div>
              )}

              <div className="border-t border-slate-100 pt-2 text-[11px] text-slate-500">
                参考文献：{scale.references.join("；")}
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
};
