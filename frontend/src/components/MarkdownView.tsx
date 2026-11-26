import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownViewProps {
  content: string;
  className?: string;
}

export const MarkdownView: React.FC<MarkdownViewProps> = ({ content, className = "" }) => {
  return (
    <div className={`text-sm leading-7 text-slate-800 ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="mb-3 mt-1 border-b border-slate-200/80 pb-2 text-lg font-bold text-slate-900">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="mb-2 mt-4 flex items-center gap-2 text-base font-semibold text-cyan-900">
              <span className="h-4 w-1 rounded-full bg-cyan-600" />
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="mb-1.5 mt-3 text-sm font-semibold text-slate-900">{children}</h3>
          ),
          p: ({ children }) => <p className="mb-2.5 last:mb-0">{children}</p>,
          ul: ({ children }) => <ul className="mb-3 list-disc space-y-1 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1 pl-5">{children}</ol>,
          li: ({ children }) => <li className="text-slate-700">{children}</li>,
          blockquote: ({ children }) => (
            <blockquote className="my-3 rounded-r-xl border-l-4 border-cyan-600/70 bg-cyan-50/60 px-3.5 py-2 text-slate-700">
              {children}
            </blockquote>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-slate-900">{children}</strong>
          ),
          code: ({ children }) => (
            <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs text-cyan-800">
              {children}
            </code>
          ),
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto rounded-xl border border-slate-200">
              <table className="min-w-full divide-y divide-slate-200 text-xs">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-slate-50/90">{children}</thead>,
          th: ({ children }) => (
            <th className="px-3 py-2 text-left font-semibold text-slate-700">{children}</th>
          ),
          td: ({ children }) => (
            <td className="border-t border-slate-100 px-3 py-2 text-slate-700">{children}</td>
          ),
          hr: () => <hr className="my-4 border-slate-200/80" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};
