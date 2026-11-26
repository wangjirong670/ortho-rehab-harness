import React, { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, Info } from "lucide-react";
import type { ScaleItemSpec, ScaleSpec } from "../api/types";
import { getDiscreteScaleOptions, getScoreDirectionLabel } from "../utils/labels";

interface ScaleEditorProps {
  spec: ScaleSpec;
  initialParams: Record<string, number | null | undefined>;
  busy?: boolean;
  submitPrimaryLabel: string;
  submitSecondaryLabel?: string;
  onSubmitPrimary: (params: Record<string, number | null>) => void;
  onSubmitSecondary?: (params: Record<string, number | null>) => void;
}

function normalizeInitial(
  spec: ScaleSpec,
  raw: Record<string, number | null | undefined>
): Record<string, number | null> {
  const out: Record<string, number | null> = {};
  for (const item of spec.items) {
    const val = raw[item.item_id];
    out[item.item_id] = typeof val === "number" && Number.isFinite(val) ? val : null;
  }
  if (spec.improvement_rate_formula) {
    const pre = raw.pre_score;
    out.pre_score = typeof pre === "number" && Number.isFinite(pre) ? pre : null;
  }
  return out;
}

const SingleScaleItemField: React.FC<{
  spec: ScaleSpec;
  item: ScaleItemSpec;
  value: number | null;
  onChange: (val: number | null) => void;
}> = ({ spec, item, value, onChange }) => {
  const discreteOptions = useMemo(() => getDiscreteScaleOptions(spec, item), [spec, item]);
  const step = item.step ?? 1;
  const isUnset = value === null || value === undefined;

  if (discreteOptions) {
    return (
      <div
        className={`rounded-xl border p-3 transition-colors ${
          isUnset
            ? "border-amber-300/80 bg-amber-50/60"
            : "border-slate-200/90 bg-white/95"
        }`}
      >
        <div className="mb-2 flex items-start justify-between gap-2">
          <label className="text-xs font-medium leading-snug text-slate-800">{item.label}</label>
          {isUnset && (
            <span className="shrink-0 rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-800">
              待补录
            </span>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {discreteOptions.map((opt) => {
            const selected = value === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => onChange(selected ? null : opt.value)}
                className={`rounded-lg border px-2.5 py-1 text-xs font-medium transition-all ${
                  selected
                    ? "border-cyan-700 bg-cyan-700 text-white shadow-sm"
                    : "border-slate-200 bg-white text-slate-700 hover:border-cyan-400 hover:bg-cyan-50/60"
                }`}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div
      className={`rounded-xl border p-3 transition-colors ${
        isUnset
          ? "border-amber-300/80 bg-amber-50/60"
          : "border-slate-200/90 bg-white/95"
      }`}
    >
      <div className="mb-2 flex items-start justify-between gap-2">
        <label className="text-xs font-medium leading-snug text-slate-800">
          {item.label}
          <span className="ml-1 text-slate-400">
            （{item.min_value} ~ {item.max_value} 分）
          </span>
        </label>
        {isUnset && (
          <span className="shrink-0 rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-800">
            待补录
          </span>
        )}
      </div>

      <div className="flex items-center gap-3">
        <input
          type="range"
          min={item.min_value}
          max={item.max_value}
          step={step}
          value={isUnset ? item.min_value : value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="h-1.5 flex-1 cursor-pointer accent-cyan-700"
        />
        <input
          type="number"
          min={item.min_value}
          max={item.max_value}
          step={step}
          value={isUnset ? "" : value}
          placeholder="未填"
          onChange={(e) => {
            const raw = e.target.value.trim();
            if (raw === "") {
              onChange(null);
            } else {
              const parsed = Number(raw);
              if (Number.isFinite(parsed)) onChange(parsed);
            }
          }}
          className="w-20 rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-center font-mono text-xs text-slate-900 focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-500/20"
        />
        {!isUnset && (
          <button
            type="button"
            onClick={() => onChange(null)}
            className="text-[11px] text-slate-400 hover:text-slate-700"
          >
            清空
          </button>
        )}
      </div>
    </div>
  );
};

export const ScaleEditor: React.FC<ScaleEditorProps> = ({
  spec,
  initialParams,
  busy = false,
  submitPrimaryLabel,
  submitSecondaryLabel,
  onSubmitPrimary,
  onSubmitSecondary,
}) => {
  const [params, setParams] = useState<Record<string, number | null>>(() =>
    normalizeInitial(spec, initialParams)
  );
  const [collapsedSubs, setCollapsedSubs] = useState<Record<string, boolean>>({});

  const subscales = spec.subscales || [];
  const hasSubscales = subscales.length > 0;

  const updateItem = (itemId: string, val: number | null) => {
    setParams((prev) => ({ ...prev, [itemId]: val }));
  };

  const filledCount = spec.items.filter(
    (it) => typeof params[it.item_id] === "number" && Number.isFinite(params[it.item_id])
  ).length;

  const cleanPayload = (): Record<string, number | null> => {
    const out: Record<string, number | null> = {};
    for (const [k, v] of Object.entries(params)) {
      if (typeof v === "number" && Number.isFinite(v)) {
        out[k] = v;
      }
    }
    return out;
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl bg-slate-50/90 px-3 py-2 text-xs text-slate-600">
        <div className="flex items-center gap-1.5">
          <Info className="h-3.5 w-3.5 text-cyan-700" />
          <span>{getScoreDirectionLabel(spec.direction)}</span>
        </div>
        <span className="font-medium text-slate-700">
          已填 {filledCount} / {spec.items.length} 项（未填写条目绝不默认补 0）
        </span>
      </div>

      {hasSubscales ? (
        <div className="space-y-2.5">
          {subscales.map((sub) => {
            const subItems = spec.items.filter((it) => it.subscale_id === sub.subscale_id);
            const subFilled = subItems.filter(
              (it) => typeof params[it.item_id] === "number"
            ).length;
            const missing = subItems.length - subFilled;
            const defaultExpanded = missing > 0;
            const isExpanded =
              sub.subscale_id in collapsedSubs
                ? !collapsedSubs[sub.subscale_id]
                : defaultExpanded;

            return (
              <div
                key={sub.subscale_id}
                className="overflow-hidden rounded-2xl border border-slate-200/80 bg-white/75"
              >
                <button
                  type="button"
                  onClick={() =>
                    setCollapsedSubs((prev) => ({
                      ...prev,
                      [sub.subscale_id]: isExpanded,
                    }))
                  }
                  className="flex w-full items-center justify-between px-3.5 py-2.5 text-left text-xs font-semibold text-slate-800 hover:bg-slate-50/80"
                >
                  <div className="flex items-center gap-2">
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4 text-slate-400" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-slate-400" />
                    )}
                    <span>{sub.label}</span>
                  </div>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                      missing > 0
                        ? "bg-amber-100 text-amber-800"
                        : "bg-emerald-100 text-emerald-800"
                    }`}
                  >
                    {missing > 0
                      ? `待补录 ${missing}/${subItems.length} 项`
                      : `已填满 ${subFilled}/${subItems.length} 项`}
                  </span>
                </button>
                {isExpanded && (
                  <div className="grid grid-cols-1 gap-2.5 border-t border-slate-100 p-3 md:grid-cols-2">
                    {subItems.map((item) => (
                      <SingleScaleItemField
                        key={item.item_id}
                        spec={spec}
                        item={item}
                        value={params[item.item_id] ?? null}
                        onChange={(v) => updateItem(item.item_id, v)}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-2.5 md:grid-cols-2">
          {spec.items.map((item) => (
            <SingleScaleItemField
              key={item.item_id}
              spec={spec}
              item={item}
              value={params[item.item_id] ?? null}
              onChange={(v) => updateItem(item.item_id, v)}
            />
          ))}
        </div>
      )}

      {spec.improvement_rate_formula && (
        <div className="rounded-xl border border-slate-200/90 bg-white/90 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="text-xs font-medium text-slate-800">
                可选：术前基线总分（用于计算平林法 JOA 改善率，{spec.min_total} ~ {spec.max_total} 分）
              </div>
              <div className="text-[11px] text-slate-500">仅需计算术前术后改善率时填写</div>
            </div>
            <input
              type="number"
              min={spec.min_total}
              max={spec.max_total}
              step={1}
              value={params.pre_score ?? ""}
              placeholder="选填"
              onChange={(e) => {
                const raw = e.target.value.trim();
                updateItem("pre_score", raw === "" ? null : Number(raw));
              }}
              className="w-28 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-center font-mono text-xs text-slate-900 focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-500/20"
            />
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 pt-1">
        <button
          type="button"
          disabled={busy}
          onClick={() => onSubmitPrimary(cleanPayload())}
          className="rounded-xl bg-cyan-700 px-4 py-2 text-xs font-semibold text-white shadow-sm transition-all hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? "计算中..." : submitPrimaryLabel}
        </button>
        {submitSecondaryLabel && onSubmitSecondary && (
          <button
            type="button"
            disabled={busy}
            onClick={() => onSubmitSecondary(cleanPayload())}
            className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-700 transition-all hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitSecondaryLabel}
          </button>
        )}
      </div>
    </div>
  );
};
