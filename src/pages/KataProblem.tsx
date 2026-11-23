import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import Editor, { type OnMount } from "@monaco-editor/react";
import type { editor as MEditor } from "monaco-editor/esm/vs/editor/editor.api.js";
import { toast } from "sonner";
import {
  BookOpen, Check, ChevronLeft, ChevronRight, Clock, Cpu, Eye, FileCode2, History, KeyRound,
  Lightbulb, List, Loader2, Lock, Play, RotateCcw, Sparkles, Target,
} from "lucide-react";
import "@/lib/monaco";
import {
  kataApi, label, timeAgo, type Device, type Grade, type GradeInfo, type HistoryEntry, type KataProblem as TProblem, type RunResult,
} from "@/lib/kata";
import { describeNext, invalidateStudy, useGoNext, useStudyNext } from "@/lib/useStudy";
import { AttemptStats, GradeBar, NextButton, ReviewBanner } from "@/components/kata/StudyBits";
import { KataMarkdown, Section, type DemoHandlers } from "@/components/kata/KataMarkdown";
import { CodeBlock } from "@/components/kata/CodeBlock";
import { DeviceToggle, DifficultyPill, GpuChip, StatusIcon } from "@/components/kata/bits";
import { MOD, ResultsPanel, type ResultsTab } from "@/components/kata/ResultsPanel";
import { PredictForm } from "@/components/kata/PredictForm";
import { ProblemList } from "@/components/kata/ProblemList";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription,
  AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { cn } from "@/lib/utils";

type LeftTab = "description" | "solution" | "history";

function readLS(key: string, fallback: string) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function writeLS(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch { /* storage unavailable */ }
}

const EDITOR_OPTIONS: MEditor.IStandaloneEditorConstructionOptions = {
  minimap: { enabled: false },
  fontSize: 14,
  lineHeight: 22,
  fontFamily: "'JetBrains Mono', monospace",
  fontLigatures: false,
  tabSize: 4,
  insertSpaces: true,
  scrollBeyondLastLine: false,
  padding: { top: 14, bottom: 14 },
  renderLineHighlight: "line",
  cursorBlinking: "smooth",
  smoothScrolling: true,
  automaticLayout: true,
  bracketPairColorization: { enabled: true },
  stickyScroll: { enabled: false },
  // The goal is writing this with no autocomplete, so there is none.
  quickSuggestions: false,
  suggestOnTriggerCharacters: false,
  wordBasedSuggestions: "off",
  parameterHints: { enabled: false },
  acceptSuggestionOnEnter: "off",
  tabCompletion: "off",
  snippetSuggestions: "none",
};

export default function KataProblem() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();

  const { data: p, error: loadError } = useQuery({
    queryKey: ["kata-problem", id],
    queryFn: () => kataApi.problem(id),
    gcTime: 0,
    refetchOnWindowFocus: false,
  });
  const { data: env } = useQuery({
    queryKey: ["kata-env"],
    queryFn: kataApi.env,
    refetchInterval: (q) => (q.state.data?.env ? 15000 : 1500),
  });

  // ---- per-problem state -------------------------------------------------
  const [code, setCode] = useState("");
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [result, setResult] = useState<RunResult | null>(null);
  const [running, setRunning] = useState(false);
  const [resultsTab, setResultsTab] = useState<ResultsTab>("checks");
  const [leftTab, setLeftTab] = useState<LeftTab>("description");
  const [hints, setHints] = useState(0);
  const [solution, setSolution] = useState<{ solution: string; items: { key: string; answer: string }[] } | null>(null);
  const [progress, setProgress] = useState<TProblem["progress"] | null>(null);
  const [saveState, setSaveState] = useState<"saved" | "saving" | "idle">("idle");
  const [listOpen, setListOpen] = useState(false);
  const [pgCode, setPgCode] = useState("");
  const [pgResult, setPgResult] = useState<RunResult | null>(null);
  const [pgRunning, setPgRunning] = useState(false);
  const [openHistory, setOpenHistory] = useState<number | null>(null);
  const [graded, setGraded] = useState<GradeInfo | null>(null);
  const [grading, setGrading] = useState<Grade | null>(null);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [lessonOpen, setLessonOpen] = useState(false);
  const { data: next } = useStudyNext(p?.studyTrack);
  const { go: goNext, busy: nextBusy } = useGoNext();
  const [device, setDeviceState] = useState<Device>(() => (readLS("kata.device", "cpu") === "cuda" ? "cuda" : "cpu"));
  const setDevice = (d: Device) => {
    setDeviceState(d);
    writeLS("kata.device", d);
  };

  const initFor = useRef<string | null>(null);
  const editorRef = useRef<MEditor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<typeof import("monaco-editor/esm/vs/editor/editor.api.js") | null>(null);
  const todoDecos = useRef<MEditor.IEditorDecorationsCollection | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pending = useRef<{ id: string; body: { code?: string; answers?: Record<string, string> } } | null>(null);

  const flushSave = useCallback((keepalive = false) => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = null;
    const job = pending.current;
    pending.current = null;
    if (!job) return;
    kataApi.saveDraft(job.id, job.body, keepalive).then(() => setSaveState("saved")).catch(() => setSaveState("idle"));
  }, []);

  // Reset when the route changes, flushing any unsaved draft for the old problem.
  useEffect(() => {
    setResult(null);
    setResultsTab("checks");
    setLeftTab("description");
    setHints(0);
    setSolution(null);
    setOpenHistory(null);
    setGraded(null);
    setPgResult(null);
    setPgCode(readLS(`kata.pg.${id}`, "# Poke at anything here. Every name from the Given and your code is in scope.\n"));
    return () => flushSave(true);
  }, [id, flushSave]);

  // Initialise editor contents once the problem arrives, and again when a review starts or is
  // cancelled (the server swaps the draft for the starter, or puts your code back).
  useEffect(() => {
    if (!p) return;
    const key = `${p.id}:${p.progress.open?.kind === "review" ? p.progress.open.startedAt : ""}`;
    if (initFor.current === key) return;
    const reinit = initFor.current?.startsWith(`${p.id}:`);
    initFor.current = key;
    setCode(p.progress.draft ?? p.starter);
    setAnswers(p.progress.answers ?? {});
    setProgress(p.progress);
    setSaveState("idle");
    setHints(p.progress.open?.hints ?? 0);
    setLessonOpen(false);
    if (reinit) {
      setResult(null);
      setGraded(null);
    }
    setSolution(null);
    if (p.progress.revealed) kataApi.solution(p.id).then(setSolution).catch(() => {});
  }, [p]);

  const scheduleSave = (body: { code?: string; answers?: Record<string, string> }) => {
    pending.current = { id, body };
    setSaveState("saving");
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => flushSave(), 700);
  };

  const onCodeChange = (v: string | undefined) => {
    const next = v ?? "";
    setCode(next);
    scheduleSave({ code: next });
    const m = monacoRef.current;
    const model = editorRef.current?.getModel();
    if (m && model) m.editor.setModelMarkers(model, "kata", []);
  };

  const onAnswersChange = (a: Record<string, string>) => {
    setAnswers(a);
    scheduleSave({ answers: a });
  };

  // Amber highlight on every remaining TODO.
  const paintTodos = useCallback(() => {
    const ed = editorRef.current;
    const model = ed?.getModel();
    if (!ed || !model) return;
    const ranges = model.findMatches("\\bTODO\\b", false, true, true, null, false);
    const decos = ranges.map((r) => ({ range: r.range, options: { inlineClassName: "kata-todo", overviewRuler: { color: "#f59e0b", position: 4 } } }));
    if (!todoDecos.current) todoDecos.current = ed.createDecorationsCollection(decos);
    else todoDecos.current.set(decos);
  }, []);
  useEffect(paintTodos, [code, paintTodos]);

  const todoCount = useMemo(() => (code.match(/\bTODO\b/g) || []).length, [code]);
  const allowed: Device[] = p?.devices ?? ["cpu"];
  const effDevice: Device = device === "cuda" && allowed.includes("cuda") && env?.env?.cuda_available ? "cuda" : "cpu";

  // ---- running -----------------------------------------------------------
  const run = useCallback(async () => {
    if (!p || running) return;
    setRunning(true);
    setResultsTab("checks");
    if (saveTimer.current) clearTimeout(saveTimer.current);
    pending.current = null;
    const wasSolved = progress?.status === "solved";
    try {
      const body = p.kind === "predict" ? { answers, device: effDevice } : { code, device: effDevice };
      const { result: r, progress: pr, graded: g, next: nx } = await kataApi.run(p.id, body);
      setResult(r);
      setSaveState("saved");
      setProgress(pr);
      setGraded(g);
      qc.setQueryData(["kata-study-next", p.studyTrack], nx);
      invalidateStudy(qc);

      const m = monacoRef.current;
      const model = editorRef.current?.getModel();
      if (m && model && r.error?.phase === "user" && r.error.line) {
        const ln = Math.min(r.error.line, model.getLineCount());
        m.editor.setModelMarkers(model, "kata", [{
          startLineNumber: ln, endLineNumber: ln, startColumn: 1, endColumn: model.getLineMaxColumn(ln),
          message: `${r.error.type}: ${r.error.message}`, severity: m.MarkerSeverity.Error,
        }]);
        editorRef.current?.revealLineInCenterIfOutsideViewport(ln);
      }
      if (r.status === "passed" && (!wasSolved || g?.kind === "review")) {
        const d = describeNext(nx);
        toast.success(g?.kind === "review" ? `Review done · ${label(p.id)}` : `Solved ${label(p.id)} · ${p.title}`, {
          description: g?.kind !== "review" && p.takeaway ? "The takeaway is unlocked in the description." : undefined,
          action: nx.kind !== "done" ? { label: `${d.verb} ${d.id} →`, onClick: () => goNext(nx) } : undefined,
        });
      }
    } catch (e: any) {
      setResult({
        status: "error", checks: [], console: [],
        error: { phase: "network", type: "Cannot reach the server", message: e.message, traceback: "Is the API server running? Start everything with:  npm run dev:all" },
      });
    } finally {
      setRunning(false);
    }
  }, [p, running, answers, code, effDevice, progress?.status, qc, goNext]);

  const runRef = useRef(run);
  runRef.current = run;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !e.defaultPrevented) {
        e.preventDefault();
        runRef.current();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const onMount: OnMount = (ed, monaco) => {
    editorRef.current = ed;
    monacoRef.current = monaco as any;
    ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => runRef.current());
    paintTodos();
    ed.focus();
  };

  const jumpToLine = (line: number) => {
    const ed = editorRef.current;
    if (!ed) return;
    ed.revealLineInCenter(line);
    ed.setPosition({ lineNumber: line, column: ed.getModel()?.getLineMaxColumn(line) ?? 1 });
    ed.focus();
  };

  const runPlayground = useCallback(async () => {
    if (!p) return;
    setPgRunning(true);
    writeLS(`kata.pg.${p.id}`, pgCode);
    try {
      const { result: r } = await kataApi.scratch(p.id, code, pgCode, effDevice);
      setPgResult(r);
    } catch (e: any) {
      setPgResult({ status: "error", checks: [], console: [], error: { phase: "network", type: "NetworkError", message: e.message, traceback: e.message } });
    } finally {
      setPgRunning(false);
    }
  }, [p, code, pgCode, effDevice]);

  const demoHandlers: DemoHandlers = useMemo(() => ({
    run: async (index: number) => (await kataApi.demo(id, index, effDevice)).result,
    openInPlayground: (c: string) => {
      setPgCode(c);
      setPgResult(null);
      setResultsTab("playground");
    },
  }), [id, effDevice]);

  const revealSolution = async () => {
    if (!p) return;
    const s = await kataApi.solution(p.id);
    setSolution(s);
    setProgress((old) => (old ? { ...old, revealed: true } : old));
  };

  const changeReview = async (start: boolean) => {
    if (!p) return;
    flushSave();
    setReviewBusy(true);
    try {
      const r = start ? await kataApi.startReview(p.id) : await kataApi.cancelReview(p.id);
      qc.setQueryData(["kata-problem", p.id], r.problem);
      invalidateStudy(qc);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setReviewBusy(false);
    }
  };

  const regrade = async (g: Grade) => {
    if (!p) return;
    setGrading(g);
    try {
      const r = await kataApi.grade(p.id, g);
      setGraded(r.lastGrade);
      setProgress(r.progress);
      invalidateStudy(qc);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setGrading(null);
    }
  };

  const showHint = () => {
    if (!p) return;
    const n = hints + 1;
    setHints(n);
    kataApi.hint(p.id, n).catch(() => {});
    setProgress((old) => (old?.open ? { ...old, open: { ...old.open, hints: Math.max(old.open.hints, n) } } : old));
  };

  const resetCode = () => {
    if (!p) return;
    if (p.kind === "predict") onAnswersChange({});
    else onCodeChange(p.starter);
    setResult(null);
  };

  // ---- render ------------------------------------------------------------
  if (loadError) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-3 bg-[#0b0d12] text-muted-foreground">
        <div>{(loadError as Error).message}</div>
        <Link to="/kata" className="text-primary">Back to the kata</Link>
      </div>
    );
  }

  const status = progress?.status ?? "unsolved";
  const solved = status === "solved";
  const firstInModule = p?.module.problemIds[0] === p?.id;
  const reviewing = progress?.open?.kind === "review";
  // No Next button when "next" is the problem already open.
  const showNext = Boolean(next && next.kind !== "done" && !(next.id === p?.id && (next.kind === "new" || reviewing)));
  const nextText = describeNext(next);
  const runningOn = effDevice === "cuda" ? (env?.env?.device_name?.replace("NVIDIA GeForce ", "") ?? "CUDA") : "CPU";

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#0b0d12] text-foreground">
      {/* ── Top bar ─────────────────────────────────────────────── */}
      <header className="flex h-12 shrink-0 items-center gap-2 border-b border-white/[0.06] bg-[#10131a] px-3">
        <div className="flex min-w-0 flex-1 items-center gap-1.5">
          <Link to="/kata" className="flex items-center gap-2 rounded-md px-2 py-1 text-sm font-semibold hover:bg-white/5">
            <span className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-orange-500 to-amber-400 text-black">
              <Cpu className="h-3.5 w-3.5" />
            </span>
            <span className="hidden sm:inline">Kata</span>
          </Link>
          <div className="mx-1 h-5 w-px bg-white/10" />
          <Sheet open={listOpen} onOpenChange={setListOpen}>
            <SheetTrigger asChild>
              <button className="flex items-center gap-1.5 rounded-md px-2 py-1 text-sm text-muted-foreground hover:bg-white/5 hover:text-foreground">
                <List className="h-4 w-4" /> <span className="hidden md:inline">Problems</span>
              </button>
            </SheetTrigger>
            <SheetContent side="left" className="w-[420px] overflow-y-auto border-white/10 bg-[#10131a] sm:max-w-[420px]">
              <SheetHeader className="mb-4">
                <SheetTitle>All problems, in path order</SheetTitle>
              </SheetHeader>
              <ProblemList currentId={id} onPick={() => setListOpen(false)} />
            </SheetContent>
          </Sheet>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                disabled={!p?.position.prev}
                onClick={() => p?.position.prev && navigate(`/kata/${p.position.prev}`)}
                className="rounded-md p-1.5 text-muted-foreground hover:bg-white/5 hover:text-foreground disabled:opacity-30"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Previous{p?.position.prev ? `: ${label(p.position.prev)}` : ""}</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                disabled={!p?.position.next}
                onClick={() => p?.position.next && navigate(`/kata/${p.position.next}`)}
                className="rounded-md p-1.5 text-muted-foreground hover:bg-white/5 hover:text-foreground disabled:opacity-30"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Next{p?.position.next ? `: ${label(p.position.next)}` : ""}</TooltipContent>
          </Tooltip>
          {p && (
            <div className="ml-1 hidden min-w-0 items-center gap-1.5 truncate text-sm text-muted-foreground lg:flex">
              <span className="font-mono text-primary">{p.module.id}</span>
              <span className="truncate">{p.module.title}</span>
              <span className="font-mono text-xs text-muted-foreground/60">· {p.position.index + 1}/{p.position.total}</span>
            </div>
          )}
        </div>

        <button
          onClick={run}
          disabled={!p || running}
          className="flex items-center gap-2 rounded-md bg-emerald-600 px-4 py-1.5 text-sm font-semibold text-white shadow-[0_0_0_1px_rgba(16,185,129,0.4),0_4px_14px_-4px_rgba(16,185,129,0.5)] transition-colors hover:bg-emerald-500 disabled:opacity-60"
        >
          {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-3.5 w-3.5 fill-current" />}
          {p?.kind === "predict" ? "Check" : "Run"}
          <span className="hidden font-mono text-[11px] font-normal text-white/70 md:inline">{MOD}+↵</span>
        </button>

        <div className="flex flex-1 items-center justify-end gap-2">
          {showNext && (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="hidden md:block">
                  <NextButton next={next} onGo={() => goNext(next, p?.studyTrack)} busy={nextBusy} compact
                    className={next?.kind === "review" ? "shadow-none" : "bg-white/[0.06] text-foreground shadow-none hover:bg-white/10"} />
                </div>
              </TooltipTrigger>
              <TooltipContent>{nextText.title ? `${nextText.title}: ` : ""}{nextText.sub}</TooltipContent>
            </Tooltip>
          )}
          <DeviceToggle device={effDevice} onChange={setDevice} env={env} allowed={allowed} />
          <GpuChip env={env} />
        </div>
      </header>

      {!p ? (
        <div className="flex flex-1 items-center justify-center text-muted-foreground">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading {label(id)}…
        </div>
      ) : (
        <ResizablePanelGroup direction="horizontal" autoSaveId="kata-h" className="min-h-0 flex-1">
          {/* ── Left: description / solution / history ──────────────── */}
          <ResizablePanel defaultSize={42} minSize={25} className="flex flex-col bg-[#0e1016]">
            <div className="flex h-10 shrink-0 items-center gap-0.5 border-b border-white/[0.06] px-2">
              {([
                ["description", "Description", <BookOpen key="i" className="h-3.5 w-3.5" />],
                ["solution", "Solution", <KeyRound key="i" className="h-3.5 w-3.5" />],
                ["history", `History${progress?.history.length ? ` · ${progress.history.length}` : ""}`, <History key="i" className="h-3.5 w-3.5" />],
              ] as [LeftTab, string, JSX.Element][]).map(([t, name, icon]) => (
                <button
                  key={t}
                  onClick={() => setLeftTab(t)}
                  className={cn(
                    "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium",
                    leftTab === t ? "bg-white/[0.07] text-foreground" : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {icon}
                  {name}
                </button>
              ))}
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto">
              {leftTab === "description" && (
                <div className="px-6 pb-16 pt-5">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="font-mono">{label(p.id)}</span>
                    <span>·</span>
                    <span>{p.module.title}</span>
                    {p.module.minutes && (
                      <span className="flex items-center gap-1"><Clock className="h-3 w-3" /> module ~{p.module.minutes} min</span>
                    )}
                  </div>
                  <h1 className="mt-1.5 text-[22px] font-semibold leading-tight tracking-tight">{p.title}</h1>
                  <div className="mt-2.5 flex flex-wrap items-center gap-2">
                    <DifficultyPill d={p.difficulty} />
                    {p.kind === "predict" && (
                      <span className="inline-flex items-center gap-1 rounded-full border border-sky-400/20 bg-sky-400/10 px-2 py-[1px] text-[11px] font-medium text-sky-300">
                        <Eye className="h-3 w-3" /> Predict
                      </span>
                    )}
                    <span className={cn(
                      "inline-flex items-center gap-1 rounded-full px-2 py-[1px] text-[11px] font-medium",
                      solved ? "bg-emerald-400/10 text-emerald-300" : status === "attempted" ? "bg-amber-400/10 text-amber-300" : "bg-white/[0.04] text-muted-foreground",
                    )}>
                      <StatusIcon status={status} className="h-3 w-3" />
                      {solved ? "Solved" : status === "attempted" ? "Attempted" : "New"}
                    </span>
                    {(progress?.runs ?? 0) > 0 && (
                      <span className="text-[11px] text-muted-foreground">
                        {progress!.runs} run{progress!.runs > 1 ? "s" : ""}
                        {progress!.stats.wrong ? `, ${progress!.stats.wrong} wrong` : ""}
                      </span>
                    )}
                  </div>

                  {progress && <ReviewBanner progress={progress} onStart={() => changeReview(true)} onCancel={() => changeReview(false)} busy={reviewBusy} />}

                  {firstInModule && p.module.intro && (
                    <div className="mt-4 rounded-lg border border-white/[0.05] bg-white/[0.02] px-4 py-3 text-[13px] text-muted-foreground">
                      <KataMarkdown compact>{p.module.intro}</KataMarkdown>
                    </div>
                  )}

                  {p.uses.length > 0 && (
                    <div className="mt-4 rounded-lg border border-white/[0.05] bg-white/[0.02] px-4 py-3">
                      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Builds on</div>
                      <div className="flex flex-wrap gap-1.5">
                        {p.uses.map((u) => (
                          <Link
                            key={u.id}
                            to={`/kata/${u.id}`}
                            className={cn(
                              "flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs",
                              u.source === "yours" ? "border-emerald-500/25 bg-emerald-500/[0.06] text-emerald-200" : "border-white/[0.08] text-muted-foreground",
                            )}
                            title={u.source === "yours" ? "Your last passing code is used" : "Not solved yet, so the reference version stands in"}
                          >
                            {u.source === "yours" ? <Check className="h-3 w-3" /> : <FileCode2 className="h-3 w-3" />}
                            <span className="font-mono">{label(u.id)}</span> {u.title}
                            <span className="text-[10px] opacity-70">{u.source === "yours" ? "yours" : "reference"}</span>
                          </Link>
                        ))}
                      </div>
                    </div>
                  )}

                  {p.lesson && (
                    <Section title="Lesson" icon={<BookOpen className="h-3 w-3" />}>
                      {reviewing && !lessonOpen ? (
                        <button
                          onClick={() => setLessonOpen(true)}
                          className="flex w-full items-center gap-2 rounded-lg border border-dashed border-white/10 px-4 py-2.5 text-left text-xs text-muted-foreground hover:border-white/25 hover:text-foreground"
                        >
                          <BookOpen className="h-3.5 w-3.5" /> Hidden during the review. Show the lesson
                        </button>
                      ) : (
                        <KataMarkdown demos={p.demos} handlers={demoHandlers}>{p.lesson}</KataMarkdown>
                      )}
                    </Section>
                  )}

                  <Section title="Your task" icon={<Target className="h-3 w-3" />}>
                    <div className="rounded-lg border border-primary/25 bg-primary/[0.035] px-4 py-1">
                      <KataMarkdown>{p.task || "_No task text._"}</KataMarkdown>
                    </div>
                  </Section>

                  {(p.given || p.module.shared) && (
                    <Section title="Given" right={<span className="text-[11px] text-muted-foreground/70">runs before your code</span>}>
                      {p.given ? <CodeBlock code={p.given} /> : <p className="text-xs text-muted-foreground">Nothing beyond the module setup.</p>}
                      {p.module.shared && (
                        <details className="group mt-2">
                          <summary className="cursor-pointer select-none text-xs text-muted-foreground hover:text-foreground">
                            Module setup, shared by every {p.module.id} problem ({p.module.shared.split("\n").length} lines)
                          </summary>
                          <CodeBlock code={p.module.shared} className="mt-2" maxHeight={420} />
                        </details>
                      )}
                    </Section>
                  )}

                  {p.hints.length > 0 && (
                    <Section title="Hints" icon={<Lightbulb className="h-3 w-3" />}>
                      <div className="space-y-2">
                        {p.hints.slice(0, hints).map((h, i) => (
                          <div key={i} className="rounded-lg border border-amber-400/15 bg-amber-400/[0.04] px-4 py-1 text-[13px]">
                            <KataMarkdown compact>{`**Hint ${i + 1}.** ${h}`}</KataMarkdown>
                          </div>
                        ))}
                        {hints < p.hints.length && (
                          <button
                            onClick={showHint}
                            className="flex w-full items-center gap-2 rounded-lg border border-dashed border-white/10 px-4 py-2.5 text-left text-xs text-muted-foreground hover:border-amber-400/30 hover:text-amber-200"
                          >
                            <Lightbulb className="h-3.5 w-3.5" />
                            {hints === 0 ? "Stuck? Show a hint" : `Show hint ${hints + 1} of ${p.hints.length}`}
                          </button>
                        )}
                      </div>
                    </Section>
                  )}

                  {p.takeaway && (
                    <Section title="Takeaway" icon={<Sparkles className="h-3 w-3" />}>
                      {solved && !reviewing ? (
                        <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/[0.04] px-4 py-1">
                          <KataMarkdown>{p.takeaway}</KataMarkdown>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 rounded-lg border border-white/[0.06] px-4 py-3 text-xs text-muted-foreground">
                          <Lock className="h-3.5 w-3.5" />
                          {reviewing ? "Hidden during the review: it can give the answer away." : "Shown once the problem is solved."}
                        </div>
                      )}
                    </Section>
                  )}

                  {solved && !reviewing && showNext && (
                    <button
                      onClick={() => goNext(next, p?.studyTrack)}
                      className="mt-8 flex w-full items-center justify-between rounded-lg border border-white/[0.07] bg-white/[0.02] px-4 py-3 text-sm hover:border-primary/40 hover:bg-primary/[0.04]"
                    >
                      <span className="text-muted-foreground">{nextText.sub}</span>
                      <span className="flex items-center gap-2 font-medium">
                        {nextText.verb} <span className="font-mono text-primary">{nextText.id}</span> <ChevronRight className="h-4 w-4" />
                      </span>
                    </button>
                  )}
                </div>
              )}

              {leftTab === "solution" && (
                <div className="px-6 pb-16 pt-6">
                  {!p.hasSolution ? (
                    <p className="text-sm text-muted-foreground">This problem has no reference solution.</p>
                  ) : solution ? (
                    <>
                      <div className="mb-4 rounded-lg border border-white/[0.06] bg-white/[0.02] px-4 py-3 text-[13px] text-muted-foreground">
                        Close this tab and retype the answer from memory.
                      </div>
                      {p.kind === "predict" ? (
                        <div className="overflow-hidden rounded-lg border border-white/[0.07]">
                          {p.items.map((it) => (
                            <div key={it.key} className="flex items-center gap-3 border-b border-white/[0.04] px-3 py-2 last:border-0">
                              <span className="font-mono text-xs text-muted-foreground">{it.key}</span>
                              <code className="flex-1 truncate font-mono text-[12.5px]">{it.expr}</code>
                              <code className="font-mono text-[12.5px] text-emerald-300">{solution.items.find((s) => s.key === it.key)?.answer}</code>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <CodeBlock code={solution.solution} />
                      )}
                    </>
                  ) : (
                    <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-6 text-center">
                      <KeyRound className="mx-auto h-8 w-8 text-white/20" />
                      <p className="mx-auto mt-3 max-w-sm text-[13px] leading-relaxed text-muted-foreground">
                        Read a solution only after your own attempt has failed.
                        {(progress?.attempts ?? 0) === 0 && " You have not run anything yet."}
                      </p>
                      <button
                        onClick={revealSolution}
                        className={cn(
                          "mt-4 rounded-md px-4 py-2 text-xs font-medium",
                          (progress?.attempts ?? 0) > 0 ? "bg-white/10 text-foreground hover:bg-white/15" : "border border-white/10 text-muted-foreground hover:text-foreground",
                        )}
                      >
                        {(progress?.attempts ?? 0) > 0 ? "Show the solution" : "Show it anyway"}
                      </button>
                    </div>
                  )}
                </div>
              )}

              {leftTab === "history" && (
                <div className="px-4 pb-16 pt-4">
                  {progress && <AttemptStats progress={progress} />}
                  {!progress?.history.length ? (
                    <p className="p-6 text-center text-sm text-muted-foreground">Every run is kept here, with the code you ran.</p>
                  ) : (
                    <div className="space-y-1">
                      {progress.history.map((h: HistoryEntry, i) => (
                        <div key={h.at + i} className="rounded-md border border-white/[0.05]">
                          <button onClick={() => setOpenHistory(openHistory === i ? null : i)} className="flex w-full items-center gap-3 px-3 py-2 text-left text-xs hover:bg-white/[0.03]">
                            <span className={cn("w-20 font-semibold", h.status === "passed" ? "text-emerald-400" : h.status === "error" ? "text-rose-400" : "text-amber-400")}>
                              {h.status === "passed" ? "Accepted" : h.status === "error" ? "Error" : `${h.passed}/${h.total}`}
                            </span>
                            <span className="flex-1 truncate text-muted-foreground">{h.error || (h.status === "passed" ? `all ${h.total} checks` : `${h.total - h.passed} failing`)}</span>
                            {h.attempt === "review" && <span className="rounded bg-sky-500/15 px-1 text-[10px] text-sky-200">review</span>}
                            {h.kind === "todo" && h.status !== "passed" && <span className="rounded bg-amber-400/10 px-1 text-[10px] text-amber-200/80" title="Had TODOs left, so it did not count as a wrong run">TODO</span>}
                            <span className="font-mono uppercase text-muted-foreground/70">{h.device}</span>
                            <span className="w-20 text-right text-muted-foreground/70">{timeAgo(h.at)}</span>
                          </button>
                          {openHistory === i && (
                            <div className="border-t border-white/[0.05] p-3">
                              {h.code !== undefined ? (
                                <>
                                  <CodeBlock code={h.code} maxHeight={360} />
                                  <button
                                    onClick={() => onCodeChange(h.code!)}
                                    className="mt-2 rounded-md bg-white/[0.06] px-3 py-1.5 text-xs hover:bg-white/10"
                                  >
                                    Load this into the editor
                                  </button>
                                </>
                              ) : h.answers ? (
                                <pre className="font-mono text-xs text-muted-foreground">{Object.entries(h.answers).map(([k, v]) => `${k}: ${v}`).join("\n")}</pre>
                              ) : null}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </ResizablePanel>

          <ResizableHandle className="bg-white/[0.06] hover:bg-primary/50" />

          {/* ── Right: editor + results ─────────────────────────────── */}
          <ResizablePanel defaultSize={58} minSize={30}>
            <ResizablePanelGroup direction="vertical" autoSaveId="kata-v">
              <ResizablePanel defaultSize={60} minSize={20} className="flex flex-col">
                <div className="flex h-10 shrink-0 items-center gap-3 border-b border-white/[0.06] bg-[#0f1117] px-3">
                  <div className="flex items-center gap-1.5 text-xs text-foreground/80">
                    <FileCode2 className="h-3.5 w-3.5 text-sky-400" />
                    <span className="font-mono">{p.kind === "predict" ? "answers" : "solution.py"}</span>
                  </div>
                  {p.kind === "code" && todoCount > 0 && (
                    <span className="rounded bg-amber-400/10 px-1.5 py-[1px] font-mono text-[10.5px] text-amber-300">
                      {todoCount} TODO{todoCount > 1 ? "s" : ""}
                    </span>
                  )}
                  <span className="text-[11px] text-muted-foreground/60">
                    {saveState === "saving" ? "saving…" : saveState === "saved" ? "saved" : ""}
                  </span>
                  <div className="ml-auto flex items-center gap-1">
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <button className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-white/5 hover:text-foreground">
                          <RotateCcw className="h-3.5 w-3.5" /> Reset
                        </button>
                      </AlertDialogTrigger>
                      <AlertDialogContent className="border-white/10 bg-[#141720]">
                        <AlertDialogHeader>
                          <AlertDialogTitle>Start over?</AlertDialogTitle>
                          <AlertDialogDescription>
                            This puts the starter back in the editor. Your earlier runs stay in History.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction onClick={resetCode}>Reset</AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  </div>
                </div>
                <div className="min-h-0 flex-1">
                  {p.kind === "predict" ? (
                    <PredictForm items={p.items} answers={answers} setAnswers={onAnswersChange} result={result} onSubmit={run} />
                  ) : (
                    <Editor
                      key={p.id}
                      height="100%"
                      language="python"
                      theme="kata-dark"
                      value={code}
                      onChange={onCodeChange}
                      onMount={onMount}
                      options={EDITOR_OPTIONS}
                      loading={<Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />}
                    />
                  )}
                </div>
              </ResizablePanel>
              <ResizableHandle className="bg-white/[0.06] hover:bg-primary/50" />
              <ResizablePanel defaultSize={40} minSize={12}>
                <ResultsPanel
                  tab={resultsTab}
                  setTab={setResultsTab}
                  result={result}
                  running={running}
                  runningOn={runningOn}
                  kind={p.kind}
                  onJumpToLine={jumpToLine}
                  onNext={!graded && showNext ? () => goNext(next, p?.studyTrack) : undefined}
                  nextLabel={`${nextText.verb} ${nextText.id}`.trim()}
                  accepted={graded ? (
                    <GradeBar info={graded} onGrade={regrade} grading={grading} next={showNext ? next : null} onNext={() => goNext(next, p?.studyTrack)} nextBusy={nextBusy} />
                  ) : undefined}
                  playground={{ code: pgCode, setCode: setPgCode, run: runPlayground, running: pgRunning, result: pgResult }}
                />
              </ResizablePanel>
            </ResizablePanelGroup>
          </ResizablePanel>
        </ResizablePanelGroup>
      )}
    </div>
  );
}
