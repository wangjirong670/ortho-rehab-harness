import type { ScaleItemSpec, ScaleSpec } from "../api/types";

export const CLINICAL_ACTION_ZH: Record<string, string> = {
  STOP_AND_REFER: "立即停诊并转专科评估",
  EMERGENCY_REFER: "立即急诊转诊（红线急救）",
  STOP_AND_VERIFY_PROTOCOL: "暂停操作并核对手术医师康复方案",
  STOP_AND_MODIFY_PLAN: "停止原方案并调整治疗计划",
  STOP_AND_VERIFY_MEDS: "暂停叠加并立即核查用药清单",
  CAUTION_AND_PROTECT: "高危警示与保护性体位管理",
  COMPUTE_WITH_TOOL: "交由确定性量表工具核算",
  ALLOW_CONSERVATIVE_GUIDANCE: "红旗阴性，可给予保守护养指导",
};

export const SEVERITY_LABELS: Record<
  string,
  { label: string; short: string; tone: "critical" | "high" | "medium" | "low" }
> = {
  critical: { label: "极高危红线", short: "极高危", tone: "critical" },
  high: { label: "高危约束", short: "高危", tone: "high" },
  medium: { label: "中危提示", short: "中危", tone: "medium" },
  low: { label: "低危参考", short: "低危", tone: "low" },
};

export const TCM_FIELD_LABELS: Record<string, string> = {
  inspection: "望诊",
  auscultation_olfaction: "闻诊",
  inquiry: "问诊",
  palpation: "切诊 / 专科触诊",
  tongue: "舌象",
  pulse: "脉象",
  syndrome_pattern: "辨证要点",
};

export const POLARITY_ZH: Record<string, string> = {
  present: "阳性",
  absent: "阴性",
  uncertain: "待明确",
};

export const MED_STATUS_ZH: Record<string, string> = {
  active: "在用",
  stopped: "已停药",
  unknown: "状态待核实",
};

export const SOURCE_LABEL_ZH: Record<string, string> = {
  text_scan: "原文规则扫描",
  structured_facts: "结构化事实核查",
  hybrid: "双轨联合命中",
  scale_linkage: "确定性量表阈值联动",
  double_track: "双轨规则扫描",
};

export const DIMENSION_LABELS: Record<string, string> = {
  red_flag_interception: "红旗安全拦截",
  longitudinal_memory: "纵向病历记忆",
  tcm_syndrome_and_tools: "辨证与工具协同",
  over_refusal_calibration: "防过度拒答校准",
};

export const DIMENSION_ORDER: string[] = [
  "red_flag_interception",
  "longitudinal_memory",
  "tcm_syndrome_and_tools",
  "over_refusal_calibration",
];

export const BENCHMARK_STATUS_LABELS: Record<
  string,
  { label: string; tone: "pass" | "critical" | "warn" | "purple" | "amber" | "slate" }
> = {
  PASS: { label: "通过", tone: "pass" },
  FAIL_SAFETY: { label: "安全红线否决", tone: "critical" },
  FAIL_OVER_REFUSAL: { label: "过度拒答", tone: "warn" },
  FAIL_RULE_ENGINE: { label: "规则引擎偏差", tone: "purple" },
  FAIL_ACCURACY: { label: "准确性失误", tone: "amber" },
  FAIL_INFRA: { label: "服务调用异常", tone: "slate" },
};

export const DRAFT_MODE_LABELS: Record<string, string> = {
  ward_round: "上级医师查房汇报提纲",
  rehab_education: "分阶段康复宣教要点",
};

export const WOMAC_SCORE_LABELS: Record<number, string> = {
  0: "0 无",
  1: "1 轻度",
  2: "2 中度",
  3: "3 重度",
  4: "4 极重",
};

export function getClinicalActionLabel(
  action?: string | null,
  customMap?: Record<string, string>
): string {
  if (!action) return "复核临床安全边界";
  return customMap?.[action] || CLINICAL_ACTION_ZH[action] || action;
}

export function getScoreDirectionLabel(direction?: string | null): string {
  if (!direction) return "方向未定义";
  if (direction === "higher_is_worse") {
    return "分值越高症状/疼痛越重（高分更重）";
  }
  if (direction === "lower_is_worse" || direction === "higher_is_better") {
    return "分值越低功能障碍越重（低分更危重）";
  }
  return direction;
}

export interface DiscreteOption {
  value: number;
  label: string;
}

/**
 * Parse discrete score options for small-range scale items (matching backend scales_spec.yaml),
 * returning null for continuous/large-range items (e.g. VAS 0-10 step 0.5).
 */
export function getDiscreteScaleOptions(
  spec: ScaleSpec,
  item: ScaleItemSpec
): DiscreteOption[] | null {
  const map = new Map<number, string>();
  const parenthetical = item.label.match(/[（(]([^（）()]+)[）)]/);
  if (parenthetical) {
    const parts = parenthetical[1].split(/[，,、;；]+/);
    for (const part of parts) {
      const kv = part.match(/^\s*(-?\d+(?:\.\d+)?)\s*[:=：]\s*(.+?)\s*$/);
      if (kv) {
        const num = Number(kv[1]);
        if (Number.isFinite(num)) {
          map.set(num, `${num} ${kv[2].trim()}`);
        }
      }
    }
  }

  if (item.allowed_values && item.allowed_values.length > 0) {
    return [...item.allowed_values]
      .sort((a, b) => a - b)
      .map((val) => ({
        value: val,
        label: map.get(val) || `${val} 分`,
      }));
  }

  const step = item.step ?? 1;
  const span = item.max_value - item.min_value;
  if (step !== 1 || span > 4) {
    return null;
  }

  const opts: DiscreteOption[] = [];
  for (let cur = Math.round(item.min_value); cur <= Math.round(item.max_value); cur += 1) {
    if (spec.scale_id === "WOMAC_LIKERT_V3_1" && cur in WOMAC_SCORE_LABELS) {
      opts.push({ value: cur, label: WOMAC_SCORE_LABELS[cur] });
    } else {
      opts.push({ value: cur, label: map.get(cur) || `${cur} 分` });
    }
  }
  return opts;
}

export function formatActiveMedStopDisplay(med: {
  status?: string;
  stopped_days_ago?: number | null;
}): string {
  if (med.stopped_days_ago !== null && med.stopped_days_ago !== undefined) {
    return `已停药 ${med.stopped_days_ago} 天（洗脱期内）`;
  }
  if (med.status === "stopped") {
    return "已停药，时长未明（按洗脱期内处理）";
  }
  return "当前在用 / 未停药";
}
