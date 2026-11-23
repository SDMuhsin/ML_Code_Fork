import { useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Loader2, Play, SquareArrowOutUpRight } from "lucide-react";
import { CodeBlock } from "./CodeBlock";
import type { RunResult } from "@/lib/kata";
import { cn } from "@/lib/utils";

export interface DemoHandlers {
  run: (index: number) => Promise<RunResult>;
  openInPlayground: (code: string) => void;
}

function textOf(node: any): string {
  if (!node) return "";
  if (node.type === "text") return node.value;
  return (node.children || []).map(textOf).join("");
}

export function DemoCell({ index, code, handlers }: { index: number; code: string; handlers: DemoHandlers }) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);

  const run = async () => {
    setRunning(true);
    try {
      setResult(await handlers.run(index));
    } catch (e: any) {
      setResult({ status: "error", error: { phase: "demo", type: "NetworkError", message: e.message, traceback: e.message }, checks: [], console: [] });
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="my-4 overflow-hidden rounded-lg border border-white/[0.07] bg-[#0f1117]">
      <div className="flex items-center justify-between border-b border-white/[0.05] bg-white/[0.02] px-2 py-1">
        <span className="pl-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70">example</span>
        <div className="flex items-center gap-1">
          <button
            onClick={() => handlers.openInPlayground(code)}
            className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-muted-foreground hover:bg-white/5 hover:text-foreground"
            title="Copy into the playground to edit and poke at it"
          >
            <SquareArrowOutUpRight className="h-3 w-3" /> Playground
          </button>
          <button
            onClick={run}
            disabled={running}
            className="flex items-center gap-1 rounded bg-emerald-500/10 px-2 py-1 text-[11px] font-medium text-emerald-400 hover:bg-emerald-500/20 disabled:opacity-60"
            title="Run this example (it sees the names earlier examples defined)"
          >
            {running ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3 fill-current" />} Run
          </button>
        </div>
      </div>
      <CodeBlock code={code} className="rounded-none border-0" copyable={false} />
      {result && (
        <div className="border-t border-white/[0.05] bg-black/30 px-4 py-2.5">
          {result.output ? (
            <pre className="whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-foreground/85">{result.output}</pre>
          ) : !result.error ? (
            <span className="font-mono text-[11px] italic text-muted-foreground">(no output)</span>
          ) : null}
          {result.error && (
            <pre className="mt-1 whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-rose-300">
              {result.error.traceback || `${result.error.type}: ${result.error.message}`}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export function KataMarkdown({
  children,
  demos,
  handlers,
  className,
  compact,
}: {
  children: string;
  demos?: string[];
  handlers?: DemoHandlers;
  className?: string;
  compact?: boolean;
}) {
  return (
    <div className={cn("kata-md text-[14px] leading-[1.7] text-foreground/85", compact && "text-[13px]", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ node, ...p }) => <h2 className="mb-2 mt-6 text-lg font-semibold text-foreground" {...p} />,
          h2: ({ node, ...p }) => <h3 className="mb-2 mt-6 text-[15px] font-semibold text-foreground first:mt-0" {...p} />,
          h3: ({ node, ...p }) => <h4 className="mb-1.5 mt-4 text-sm font-semibold text-foreground" {...p} />,
          p: ({ node, ...p }) => <p className="my-2.5" {...p} />,
          ul: ({ node, ...p }) => <ul className="my-2.5 list-disc space-y-1 pl-5 marker:text-muted-foreground/60" {...p} />,
          ol: ({ node, ...p }) => <ol className="my-2.5 list-decimal space-y-1 pl-5 marker:text-muted-foreground/60" {...p} />,
          strong: ({ node, ...p }) => <strong className="font-semibold text-foreground" {...p} />,
          a: ({ node, ...p }) => <a className="text-primary underline-offset-2 hover:underline" {...p} />,
          blockquote: ({ node, ...p }) => (
            <blockquote className="my-3 border-l-2 border-primary/50 bg-primary/[0.04] py-1 pl-4 pr-2 italic text-foreground/80" {...p} />
          ),
          table: ({ node, ...p }) => (
            <div className="my-3 overflow-x-auto rounded-lg border border-white/[0.07]">
              <table className="w-full text-[12.5px]" {...p} />
            </div>
          ),
          thead: ({ node, ...p }) => <thead className="bg-white/[0.03] text-left text-muted-foreground" {...p} />,
          th: ({ node, ...p }) => <th className="px-3 py-1.5 font-medium" {...p} />,
          td: ({ node, ...p }) => <td className="border-t border-white/[0.05] px-3 py-1.5 align-top" {...p} />,
          code: ({ node, className: cls, children: c, ...p }) => (
            <code className="rounded-[5px] border border-white/[0.06] bg-white/[0.05] px-1.5 py-[1px] font-mono text-[0.86em] text-[#e6c07b]" {...p}>
              {c}
            </code>
          ),
          pre: ({ node }) => {
            const codeEl: any = (node as any)?.children?.[0];
            const cls: string = (codeEl?.properties?.className || []).join(" ");
            const text = textOf(codeEl).replace(/\n$/, "");
            const demo = cls.match(/language-pydemo-(\d+)/);
            if (demo && handlers) {
              const i = Number(demo[1]);
              return <DemoCell index={i} code={demos?.[i] ?? text} handlers={handlers} />;
            }
            const lang = cls.match(/language-(\w+)/)?.[1] || "text";
            return <CodeBlock code={text} language={lang === "pydemo" ? "python" : lang} className="my-3" />;
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

export function Section({ title, icon, children, right }: { title: string; icon?: ReactNode; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="mt-6">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          {icon}
          {title}
        </h3>
        {right}
      </div>
      {children}
    </section>
  );
}
