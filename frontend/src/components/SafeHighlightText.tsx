import React from "react";
import type { HighlightSpan } from "../api/types";

interface SafeHighlightTextProps {
  text: string;
  highlights?: HighlightSpan[];
  tone?: "danger" | "safe";
}

/**
 * Pure React span-splitting highlighter for red-team turns.
 * Never uses dangerouslySetInnerHTML.
 */
export const SafeHighlightText: React.FC<SafeHighlightTextProps> = ({
  text,
  highlights = [],
  tone = "danger",
}) => {
  if (!text) {
    return <span className="italic text-slate-400">（空回复）</span>;
  }

  const validSpans = (highlights || [])
    .filter(
      (h) =>
        typeof h.start === "number" &&
        typeof h.end === "number" &&
        h.end > h.start &&
        h.start < text.length
    )
    .map((h) => ({
      start: Math.max(0, h.start),
      end: Math.min(text.length, h.end),
      reason: h.reason || "命中安全审计片段",
    }))
    .sort((a, b) => a.start - b.start || b.end - a.end);

  // Deduplicate overlapping spans greedily
  const nonOverlapping: typeof validSpans = [];
  let lastEnd = 0;
  for (const span of validSpans) {
    if (span.start >= lastEnd) {
      nonOverlapping.push(span);
      lastEnd = span.end;
    }
  }

  if (nonOverlapping.length === 0) {
    return <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-slate-800">{text}</div>;
  }

  const nodes: React.ReactNode[] = [];
  let cursor = 0;

  nonOverlapping.forEach((span, idx) => {
    if (span.start > cursor) {
      nodes.push(<span key={`t-${idx}`}>{text.slice(cursor, span.start)}</span>);
    }
    const isProtection =
      tone === "safe" ||
      span.reason.includes("否定") ||
      span.reason.includes("保护") ||
      span.reason.includes("放行");

    nodes.push(
      <mark
        key={`m-${idx}`}
        title={span.reason}
        className={
          isProtection
            ? "mx-0.5 rounded-md border border-emerald-300/80 bg-emerald-100/90 px-1 py-0.5 font-medium text-emerald-900"
            : "mx-0.5 rounded-md border border-rose-300/80 bg-rose-100/95 px-1 py-0.5 font-semibold text-rose-900"
        }
      >
        {text.slice(span.start, span.end)}
      </mark>
    );
    cursor = span.end;
  });

  if (cursor < text.length) {
    nodes.push(<span key="tail">{text.slice(cursor)}</span>);
  }

  return <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-slate-800">{nodes}</div>;
};
