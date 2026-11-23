import { useRef } from "react";
import { CheckCircle2, XCircle } from "lucide-react";
import { InlineCode } from "./CodeBlock";
import type { KataProblem, RunResult } from "@/lib/kata";
import { cn } from "@/lib/utils";

/** The "predict the shape" form: one input per expression, checked all at once. */
export function PredictForm({
  items, answers, setAnswers, result, onSubmit,
}: {
  items: KataProblem["items"];
  answers: Record<string, string>;
  setAnswers: (a: Record<string, string>) => void;
  result: RunResult | null;
  onSubmit: () => void;
}) {
  const refs = useRef<(HTMLInputElement | null)[]>([]);
  const verdict = (key: string) => result?.checks.find((c) => c.label.startsWith(`${key}  `));
  const answered = items.filter((it) => (answers[it.key] || "").trim()).length;

  return (
    <div className="h-full overflow-y-auto bg-[#0f1117] p-5">
      <div className="mb-4 flex items-baseline justify-between">
        <p className="text-sm text-muted-foreground">
          {items[0]?.kind === "raises"
            ? "For each expression, will it run or raise?"
            : <>Write each shape as a tuple: <code className="font-mono text-foreground/80">(2, 3)</code>, <code className="font-mono text-foreground/80">(8,)</code>, or <code className="font-mono text-foreground/80">()</code> for rank 0. Reason it out; do not run it first.</>}
        </p>
        <span className="shrink-0 pl-4 font-mono text-xs text-muted-foreground">{answered}/{items.length} answered</span>
      </div>
      <div className="space-y-2">
        {items.map((it, i) => {
          const v = verdict(it.key);
          const value = answers[it.key] || "";
          return (
            <div
              key={it.key}
              className={cn(
                "rounded-lg border px-3 py-2.5 transition-colors",
                !v ? "border-white/[0.06] bg-white/[0.015]" : v.passed ? "border-emerald-500/25 bg-emerald-500/[0.04]" : "border-rose-500/25 bg-rose-500/[0.04]",
              )}
            >
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-white/[0.05] font-mono text-xs text-muted-foreground">{it.key}</span>
                <div className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap"><InlineCode code={it.expr} /></div>
                {it.kind === "raises" ? (
                  <div className="flex shrink-0 rounded-md border border-white/[0.08] p-0.5">
                    {["ok", "raise"].map((opt) => (
                      <button
                        key={opt}
                        onClick={() => setAnswers({ ...answers, [it.key]: opt })}
                        className={cn(
                          "rounded px-3 py-1 font-mono text-xs",
                          value === opt ? (opt === "ok" ? "bg-emerald-500/20 text-emerald-300" : "bg-rose-500/20 text-rose-300") : "text-muted-foreground hover:text-foreground",
                        )}
                      >
                        {opt}
                      </button>
                    ))}
                  </div>
                ) : (
                  <input
                    ref={(el) => (refs.current[i] = el)}
                    value={value}
                    placeholder="( , )"
                    spellCheck={false}
                    onChange={(e) => setAnswers({ ...answers, [it.key]: e.target.value })}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !(e.ctrlKey || e.metaKey)) {
                        e.preventDefault();
                        const nxt = refs.current.slice(i + 1).find(Boolean);
                        if (nxt) nxt.focus();
                        else onSubmit();
                      }
                    }}
                    className="w-36 shrink-0 rounded-md border border-white/[0.08] bg-black/30 px-2.5 py-1 font-mono text-[13px] text-foreground placeholder:text-white/15 focus:border-primary/50 focus:outline-none"
                  />
                )}
                <span className="w-4 shrink-0">
                  {v && (v.passed ? <CheckCircle2 className="h-4 w-4 text-emerald-400" /> : <XCircle className="h-4 w-4 text-rose-400" />)}
                </span>
              </div>
              {v && v.details.length > 0 && (
                <div className={cn("mt-1.5 pl-9 font-mono text-[11.5px]", v.passed ? "text-muted-foreground" : "text-rose-200/90")}>
                  {v.details.join(" · ")}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <p className="mt-4 text-xs text-muted-foreground/70">Enter moves to the next box; Enter on the last one checks them all.</p>
    </div>
  );
}
