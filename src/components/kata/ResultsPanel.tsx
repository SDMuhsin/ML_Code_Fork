import { useEffect, useRef } from "react";
import Editor, { type OnMount } from "@monaco-editor/react";
import { motion } from "framer-motion";
import {
  AlertTriangle, ArrowRight, CheckCircle2, ChevronRight, CornerDownLeft, FlaskConical, Loader2,
  Play, Sparkles, Terminal, Variable, XCircle,
} from "lucide-react";
import type { CheckResult, InspectEntry, RunResult } from "@/lib/kata";
import { cn } from "@/lib/utils";

export type ResultsTab = "checks" | "variables" | "console" | "playground";

const isMac = typeof navigator !== "undefined" && /Mac/.test(navigator.platform);
export const MOD = isMac ? "⌘" : "Ctrl";

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded border border-white/10 bg-white/[0.04] px-1.5 py-[1px] font-mono text-[10.5px] text-muted-foreground">
      {children}
    </kbd>
  );
}

function CheckRow({ c, i }: { c: CheckResult; i: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 3 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(i * 0.02, 0.4), duration: 0.15 }}
      className={cn(
        "rounded-md border px-3 py-2",
        c.passed ? "border-transparent bg-transparent" : "border-rose-500/15 bg-rose-500/[0.04]",
      )}
    >
      <div className="flex items-start gap-2.5">
        {c.passed ? (
          <CheckCircle2 className="mt-[2px] h-4 w-4 shrink-0 text-emerald-400" />
        ) : (
          <XCircle className="mt-[2px] h-4 w-4 shrink-0 text-rose-400" />
        )}
        <div className="min-w-0 flex-1">
          <div className={cn("font-mono text-[12.5px]", c.passed ? "text-foreground/75" : "text-foreground")}>{c.label}</div>
          {c.details.length > 0 && (
            <div className={cn("mt-1 space-y-0.5 font-mono text-[12px] leading-relaxed", c.passed ? "text-muted-foreground" : "text-rose-200/90")}>
              {c.details.map((d, k) => (
                <div key={k} className="whitespace-pre-wrap break-words">{d}</div>
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

function ChecksView({
  result, running, runningOn, onJumpToLine, onNext, nextLabel, kind, accepted,
}: {
  result: RunResult | null;
  running: boolean;
  runningOn: string;
  onJumpToLine: (line: number) => void;
  onNext?: () => void;
  nextLabel?: string;
  kind: "code" | "predict";
  accepted?: React.ReactNode;
}) {
  if (running && !result) {
    return (
      <div className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" /> Running on {runningOn}…
      </div>
    );
  }
  if (!result) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-sm text-muted-foreground">
        <FlaskConical className="h-7 w-7 text-white/15" />
        <div>
          {kind === "predict" ? "Write every shape first, then check them all." : "Run your code to see every check."}
          <div className="mt-2 flex items-center justify-center gap-1.5 text-xs">
            <Kbd>{MOD}</Kbd> + <Kbd>Enter</Kbd>
          </div>
        </div>
      </div>
    );
  }
  const n = result.checks.length;
  const p = result.checks.filter((c) => c.passed).length;
  const err = result.error;
  const passed = result.status === "passed";
  const t = result.timing;

  return (
    <div className={cn("space-y-3 p-3 transition-opacity", running && "opacity-50")}>
      <div className="flex flex-wrap items-center gap-3">
        {passed ? (
          <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-emerald-400" />
            <span className="text-lg font-semibold text-emerald-400">Accepted</span>
            <span className="text-sm text-muted-foreground">all {n} checks passed</span>
          </motion.div>
        ) : err ? (
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-rose-400" />
            <span className="text-lg font-semibold text-rose-400">{err.type}</span>
            <span className="text-sm text-muted-foreground">
              {err.phase === "user" ? "in your code" : err.phase === "given" ? "in the given setup" : `during ${err.phase}`}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <XCircle className="h-5 w-5 text-amber-400" />
            <span className="text-lg font-semibold text-amber-400">
              {p} / {n}
            </span>
            <span className="text-sm text-muted-foreground">checks passed</span>
          </div>
        )}
        <div className="ml-auto flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
          {result.device && <span className="rounded bg-white/[0.04] px-1.5 py-0.5 uppercase">{result.device}</span>}
          {t?.user_ms != null && <span title="time spent in your code">your code {t.user_ms < 1 ? t.user_ms.toFixed(2) : t.user_ms.toFixed(1)} ms</span>}
          {result.wall_ms != null && <span title="round trip including the checks">· {result.wall_ms} ms total</span>}
        </div>
        {passed && onNext && (
          <button
            onClick={onNext}
            className="flex items-center gap-1.5 rounded-md bg-emerald-500/15 px-3 py-1.5 text-xs font-semibold text-emerald-300 hover:bg-emerald-500/25"
          >
            {nextLabel || "Next problem"} <ArrowRight className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {passed && accepted}

      {!passed && result.todo_lines && result.todo_lines.length > 0 && !err && (
        <div className="flex items-center gap-2 rounded-md border border-amber-400/15 bg-amber-400/[0.05] px-3 py-2 text-xs text-amber-200/90">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
          Your code still has <code className="font-mono">TODO</code> on line{result.todo_lines.length > 1 ? "s" : ""}{" "}
          {result.todo_lines.map((l, i) => (
            <button key={l} className="font-mono underline decoration-dotted underline-offset-2 hover:text-amber-100" onClick={() => onJumpToLine(l)}>
              {l}{i < result.todo_lines!.length - 1 ? "," : ""}
            </button>
          ))}
        </div>
      )}

      {err && (
        <div className="rounded-md border border-rose-500/20 bg-rose-500/[0.05]">
          <div className="flex items-center justify-between px-3 pt-2.5">
            <div className="font-mono text-[12.5px] text-rose-200">{err.message}</div>
            {err.line && err.phase === "user" && (
              <button
                onClick={() => onJumpToLine(err.line!)}
                className="flex shrink-0 items-center gap-1 rounded px-2 py-0.5 text-[11px] text-rose-200 hover:bg-rose-500/10"
              >
                line {err.line} <CornerDownLeft className="h-3 w-3" />
              </button>
            )}
          </div>
          {err.traceback && (
            <pre className="overflow-x-auto whitespace-pre-wrap px-3 pb-3 pt-1.5 font-mono text-[11.5px] leading-relaxed text-rose-100/60">{err.traceback}</pre>
          )}
        </div>
      )}

      {result.uses && result.uses.some((u) => u.note) && (
        <div className="space-y-1 text-xs text-amber-200/80">
          {result.uses.filter((u) => u.note).map((u) => (
            <div key={u.id}>
              {u.id}: {u.note}
            </div>
          ))}
        </div>
      )}

      {n > 0 && (
        <div className="space-y-1">
          {result.checks.map((c, i) => (
            <CheckRow key={i} c={c} i={i} />
          ))}
        </div>
      )}
    </div>
  );
}

function VarCard({ v }: { v: InspectEntry }) {
  const isT = v.kind === "tensor";
  return (
    <div className="rounded-md border border-white/[0.06] bg-white/[0.015]">
      <div className="flex flex-wrap items-center gap-2 border-b border-white/[0.04] px-3 py-1.5">
        <span className="font-mono text-[13px] font-semibold text-sky-300">{v.name}</span>
        {isT ? (
          <>
            <span className="rounded bg-primary/15 px-1.5 py-[1px] font-mono text-[11.5px] text-primary">({v.shape!.join(", ")}{v.shape!.length === 1 ? "," : ""})</span>
            <span className="font-mono text-[11.5px] text-muted-foreground">{v.dtype}</span>
            <span className={cn("font-mono text-[11.5px]", v.device?.startsWith("cuda") ? "text-emerald-300" : "text-muted-foreground")}>{v.device}</span>
            <span className="font-mono text-[11px] text-muted-foreground/70" title="stride: elements to step per axis">
              stride ({v.stride!.join(", ")}{v.stride!.length === 1 ? "," : ""})
            </span>
            {!v.contiguous && <span className="rounded bg-amber-400/10 px-1.5 text-[10.5px] text-amber-300">non-contiguous</span>}
            {v.stride?.includes(0) && v.numel! > 1 && (
              <span className="rounded bg-violet-400/10 px-1.5 text-[10.5px] text-violet-300" title="an expanded view: positions along the stride-0 axis alias one element">
                stride-0 view
              </span>
            )}
          </>
        ) : (
          <span className={cn("font-mono text-[11.5px]", v.kind === "todo" || v.kind === "missing" ? "text-amber-300" : "text-muted-foreground")}>
            {v.kind === "todo" ? "TODO" : v.kind}
          </span>
        )}
      </div>
      <pre className="max-h-48 overflow-auto whitespace-pre px-3 py-2 font-mono text-[11.5px] leading-relaxed text-foreground/75">
        {isT ? v.preview : v.summary}
      </pre>
    </div>
  );
}

export function ResultsPanel(props: {
  tab: ResultsTab;
  setTab: (t: ResultsTab) => void;
  result: RunResult | null;
  running: boolean;
  runningOn: string;
  kind: "code" | "predict";
  onJumpToLine: (line: number) => void;
  onNext?: () => void;
  nextLabel?: string;
  /** Shown under "Accepted": the grade and the review buttons. */
  accepted?: React.ReactNode;
  playground: { code: string; setCode: (s: string) => void; run: () => void; running: boolean; result: RunResult | null };
}) {
  const { tab, setTab, result, playground } = props;
  const consoleText = result?.console?.filter((c) => c.text.trim()) ?? [];
  const nVars = result?.inspect?.length ?? 0;
  const tabs: { id: ResultsTab; label: string; icon: JSX.Element; badge?: string | number; hidden?: boolean }[] = [
    {
      id: "checks", label: "Checks", icon: <FlaskConical className="h-3.5 w-3.5" />,
      badge: result ? `${result.checks.filter((c) => c.passed).length}/${result.checks.length}` : undefined,
    },
    { id: "variables", label: "Variables", icon: <Variable className="h-3.5 w-3.5" />, badge: nVars || undefined, hidden: props.kind === "predict" },
    { id: "console", label: "Console", icon: <Terminal className="h-3.5 w-3.5" />, badge: consoleText.length ? "•" : undefined },
    { id: "playground", label: "Playground", icon: <Play className="h-3.5 w-3.5" /> },
  ];

  const pgRun = useRef(playground.run);
  pgRun.current = playground.run;
  const onPgMount: OnMount = (editor, monaco) => {
    editor.addCommand(monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => pgRun.current());
  };

  return (
    <div className="flex h-full flex-col bg-[#0c0e13]">
      <div className="flex h-9 shrink-0 items-center gap-0.5 border-b border-white/[0.06] px-2">
        {tabs.filter((t) => !t.hidden).map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
              tab === t.id ? "bg-white/[0.07] text-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {t.icon}
            {t.label}
            {t.badge !== undefined && (
              <span
                className={cn(
                  "rounded px-1 font-mono text-[10px]",
                  t.id === "checks" && result?.status === "passed" ? "bg-emerald-400/15 text-emerald-300" :
                    t.id === "checks" ? "bg-amber-400/10 text-amber-300" : "text-sky-300",
                )}
              >
                {t.badge}
              </span>
            )}
          </button>
        ))}
        {props.running && <Loader2 className="ml-2 h-3.5 w-3.5 animate-spin text-muted-foreground" />}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {tab === "checks" && <ChecksView {...props} />}

        {tab === "variables" && (
          <div className="space-y-2 p-3">
            {!result ? (
              <p className="p-6 text-center text-sm text-muted-foreground">After a run, every name your code defines shows up here with its shape, dtype, device and stride.</p>
            ) : nVars === 0 ? (
              <p className="p-6 text-center text-sm text-muted-foreground">{result.error ? "Your code raised before the variables could be inspected." : "Nothing to inspect."}</p>
            ) : (
              result.inspect!.map((v) => <VarCard key={v.name} v={v} />)
            )}
          </div>
        )}

        {tab === "console" && (
          <div className="p-3">
            {consoleText.length === 0 ? (
              <p className="p-6 text-center text-sm text-muted-foreground">
                Anything your code <code className="font-mono">print</code>s appears here. <code className="font-mono">print(S(t))</code> is a good habit.
              </p>
            ) : (
              consoleText.map((c, i) => (
                <div key={i} className="mb-3">
                  <div className="mb-1 flex items-center gap-1 text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground/70">
                    <ChevronRight className="h-3 w-3" /> {c.phase === "user" ? "your code" : c.phase}
                  </div>
                  <pre className="whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-foreground/85">{c.text}</pre>
                </div>
              ))
            )}
          </div>
        )}

        {tab === "playground" && (
          <div className="flex h-full flex-col">
            <div className="flex items-center justify-between gap-3 border-b border-white/[0.04] px-3 py-1.5 text-[11.5px] text-muted-foreground">
              <span>
                Runs after the Given and {props.kind === "predict" ? "nothing else" : "your code"}, so every name is in scope. A trailing expression is printed.
              </span>
              <button
                onClick={playground.run}
                disabled={playground.running}
                className="flex shrink-0 items-center gap-1.5 rounded-md bg-white/[0.06] px-2.5 py-1 text-xs font-medium text-foreground hover:bg-white/10 disabled:opacity-50"
              >
                {playground.running ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3 fill-current" />}
                Run <span className="text-muted-foreground">⇧ Enter</span>
              </button>
            </div>
            <div className="grid min-h-0 flex-1 grid-cols-2">
              <div className="min-h-0 border-r border-white/[0.04]">
                <Editor
                  height="100%"
                  language="python"
                  theme="kata-dark"
                  value={playground.code}
                  onChange={(v) => playground.setCode(v ?? "")}
                  onMount={onPgMount}
                  options={{
                    minimap: { enabled: false }, fontSize: 13, lineNumbers: "off", glyphMargin: false, folding: false,
                    scrollBeyondLastLine: false, padding: { top: 8 }, quickSuggestions: false, suggestOnTriggerCharacters: false,
                    wordBasedSuggestions: "off", parameterHints: { enabled: false }, renderLineHighlight: "none",
                    fontFamily: "'JetBrains Mono', monospace", automaticLayout: true, tabSize: 4,
                  }}
                />
              </div>
              <div className="min-h-0 overflow-auto p-3">
                {!playground.result ? (
                  <p className="text-xs text-muted-foreground">Output appears here. Try <code className="font-mono">S(some_tensor)</code>.</p>
                ) : (
                  <>
                    {playground.result.output && (
                      <pre className="whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-foreground/85">{playground.result.output}</pre>
                    )}
                    {playground.result.error && (
                      <pre className="whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-rose-300">
                        {playground.result.error.phase === "user" ? "Your solution raised before the playground ran:\n" : ""}
                        {playground.result.error.traceback || playground.result.error.message}
                      </pre>
                    )}
                    {!playground.result.output && !playground.result.error && (
                      <span className="font-mono text-xs italic text-muted-foreground">(no output)</span>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
