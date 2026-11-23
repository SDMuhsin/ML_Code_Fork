import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  AlertTriangle, Brain, ChevronDown, Clock, Cpu, LayoutGrid, Loader2, Network, RotateCcw, TrendingUp, Zap,
} from "lucide-react";
import "@/lib/monaco";
import { fmtBytes, fmtDue, kataApi, label, type Card, type Catalog, type ModuleSummary, type StudyOverview, type TrackSummary } from "@/lib/kata";
import { useGoNext } from "@/lib/useStudy";
import { DifficultyPill, KindIcon, StatusIcon } from "@/components/kata/bits";
import { KataMarkdown } from "@/components/kata/KataMarkdown";
import { Roadmap, RoadmapLegend } from "@/components/kata/Roadmap";
import { NextButton } from "@/components/kata/StudyBits";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

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

const isDue = (c: Card | null) => Boolean(c && new Date(c.due).getTime() <= Date.now());

/** "due" chip for a problem row. With `long`, a not-yet-due card shows when it is due. */
function DueChip({ card, reviewing, long }: { card: Card | null; reviewing?: boolean; long?: boolean }) {
  if (reviewing) return <span className="rounded-full bg-sky-500/20 px-1.5 py-[1px] text-[10px] font-semibold text-sky-200">reviewing</span>;
  if (isDue(card)) return <span className="rounded-full bg-sky-500/20 px-1.5 py-[1px] text-[10px] font-semibold text-sky-200">due</span>;
  if (long && card) return <span className="text-[10.5px] text-muted-foreground">review {fmtDue(card.due)}</span>;
  return null;
}

function EnvCard() {
  const { data: env } = useQuery({
    queryKey: ["kata-env"],
    queryFn: kataApi.env,
    refetchInterval: (q) => (q.state.data?.env ? 15000 : 1500),
  });
  const e = env?.env;
  return (
    <div className="rounded-xl border border-white/[0.07] bg-gradient-to-br from-white/[0.04] to-transparent p-4">
      <div className="mb-3 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        <Cpu className="h-3.5 w-3.5" /> Runtime
      </div>
      {!env ? (
        <div className="text-sm text-rose-300">Cannot reach the API server on port 5001. Start it with <code className="font-mono">npm run dev:all</code>.</div>
      ) : !e ? (
        env.lastError ? (
          <pre className="whitespace-pre-wrap font-mono text-xs text-rose-300">{env.lastError}</pre>
        ) : (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Importing torch…
          </div>
        )
      ) : (
        <div className="space-y-2">
          <div className="flex items-center gap-2.5">
            <span className={cn("flex h-8 w-8 items-center justify-center rounded-lg", e.cuda_available ? "bg-emerald-500/15 text-emerald-300" : "bg-white/5 text-muted-foreground")}>
              <Zap className="h-4 w-4" />
            </span>
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold">{e.cuda_available ? e.device_name : "No CUDA device"}</div>
              <div className="font-mono text-[11px] text-muted-foreground">
                {e.cuda_available ? `sm_${e.capability?.replace(".", "")} · ${fmtBytes(e.total_memory)} · ${e.sm_count} SMs · CUDA ${e.cuda_version}` : "torch.cuda.is_available() is False"}
              </div>
            </div>
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px] text-muted-foreground">
            <span>torch {e.torch}</span>
            <span>python {e.python}</span>
            <span className="text-emerald-300/80">{env.ready} warm worker{env.ready === 1 ? "" : "s"}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function ModuleCard({ m, i }: { m: ModuleSummary; i: number }) {
  const solved = m.problems.filter((p) => p.status === "solved").length;
  const done = solved === m.problems.length;
  const pct = (solved / Math.max(1, m.problems.length)) * 100;
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(i * 0.03, 0.3) }}
      className={cn("rounded-xl border bg-[#12151c]", done ? "border-emerald-500/20" : "border-white/[0.06]")}
    >
      <div className="flex items-start gap-3 px-4 pt-4">
        <div className={cn(
          "flex h-9 w-9 shrink-0 items-center justify-center rounded-lg font-mono text-sm font-bold",
          done ? "bg-emerald-500/15 text-emerald-300" : "bg-primary/10 text-primary",
        )}>
          {m.id}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-3">
            <h3 className="truncate text-[15px] font-semibold">{m.title}</h3>
            <span className="shrink-0 font-mono text-[11px] text-muted-foreground">{solved}/{m.problems.length}</span>
          </div>
          <div className="mt-0.5 line-clamp-2 text-[12.5px] leading-snug text-muted-foreground">
            {m.intro.replace(/[*`]/g, "")}
          </div>
        </div>
      </div>
      <div className="mx-4 mt-3 h-1 overflow-hidden rounded-full bg-white/[0.05]">
        <div className="h-full rounded-full bg-emerald-400/80 transition-all" style={{ width: `${pct}%` }} />
      </div>
      <div className="p-2">
        {m.problems.map((p) => (
          <Link
            key={p.id}
            to={`/kata/${p.id}`}
            className="group flex items-center gap-2.5 rounded-md px-2 py-1.5 text-[13px] text-foreground/85 hover:bg-white/[0.04]"
          >
            <StatusIcon status={p.status} className="h-3.5 w-3.5" />
            <span className="w-9 font-mono text-[11.5px] text-muted-foreground">{label(p.id)}</span>
            <span className="flex-1 truncate group-hover:text-foreground">{p.title}</span>
            <DueChip card={p.card} reviewing={p.reviewing} />
            <KindIcon kind={p.kind} />
            <DifficultyPill d={p.difficulty} className="text-[10px]" />
          </Link>
        ))}
      </div>
      {m.minutes && (
        <div className="flex items-center gap-1 border-t border-white/[0.04] px-4 py-2 text-[11px] text-muted-foreground/70">
          <Clock className="h-3 w-3" /> about {m.minutes >= 90 ? `${Math.round(m.minutes / 60)} h` : `${m.minutes} min`}
        </div>
      )}
    </motion.div>
  );
}

function Collapsible({ title, icon, children, defaultOpen = false }: { title: string; icon: JSX.Element; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-xl border border-white/[0.07] bg-[#12151c]">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-semibold">
        {icon}
        <span className="flex-1">{title}</span>
        <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition-transform", open && "rotate-180")} />
      </button>
      {open && <div className="border-t border-white/[0.05] px-4 pb-3 pt-1">{children}</div>}
    </div>
  );
}

function QueueStat({ n, text, tone }: { n: number | string; text: string; tone?: string }) {
  return (
    <div className="min-w-[60px]">
      <div className={cn("text-xl font-black tabular-nums", tone)}>{n}</div>
      <div className="text-[11px] leading-tight text-muted-foreground">{text}</div>
    </div>
  );
}

/** Segmented control over the tracks of progression.md, each with its due count. Hidden with one track. */
function TrackSwitcher({ tracks, current, onPick }: { tracks: TrackSummary[]; current: string; onPick: (id: string) => void }) {
  if (tracks.length < 2) return null;
  return (
    <div className="mt-6 flex flex-wrap items-center gap-1 rounded-lg border border-white/[0.06] bg-white/[0.03] p-1">
      {tracks.map((t) => (
        <button
          key={t.id}
          onClick={() => onPick(t.id)}
          className={cn(
            "flex items-center gap-2 rounded-md px-3 py-1.5 text-[13px] font-medium",
            current === t.id ? "bg-white/10 text-foreground" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {t.title}
          <span className="text-[11px] tabular-nums text-muted-foreground">{t.solved}/{t.total}</span>
          {t.due > 0 && <span className="rounded-full bg-sky-500/20 px-1.5 py-[1px] text-[10px] font-semibold text-sky-200">{t.due} due</span>}
        </button>
      ))}
    </div>
  );
}

function StudyCard({ data, study, track }: { data: Catalog; study?: StudyOverview; track: string }) {
  const { go, busy } = useGoNext();
  const sum = (study?.tracks ?? data.tracks ?? []).find((t) => t.id === track);
  const solved = sum?.solved ?? data.solved;
  const total = sum?.total ?? data.total;
  const pct = Math.round((solved / Math.max(1, total)) * 100);
  const c = study?.counts;
  const next = study?.next ?? (track === "main" ? data.study : undefined);
  return (
    <div className="mt-6 rounded-xl border border-white/[0.07] bg-[#12151c] p-4">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-4">
        <div className="min-w-[160px] flex-1">
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-black">{solved}</span>
            <span className="text-sm text-muted-foreground">of {total} solved</span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
            <div className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-emerald-300 transition-all" style={{ width: `${pct}%` }} />
          </div>
        </div>
        {c && c.cards > 0 && (
          <div className="flex gap-4">
            <QueueStat n={c.due} text="due now" tone={c.due ? "text-sky-300" : "text-muted-foreground"} />
            <QueueStat n={c.dueTomorrow} text="tomorrow" tone="text-foreground/70" />
            <QueueStat n={c.dueWeek} text="this week" tone="text-foreground/70" />
          </div>
        )}
        <NextButton next={next} onGo={() => go(next, track)} busy={busy} />
      </div>
      {c?.laterToday && (
        <div className="mt-3 text-[12px] text-muted-foreground">
          A problem you needed the solution for comes back {fmtDue(c.laterToday)}, to retype from memory.
        </div>
      )}
    </div>
  );
}

const pctText = (x: number | null) => (x === null ? "—" : `${Math.round(x * 100)}%`);

function StatsCard({ study }: { study: StudyOverview }) {
  const t = study.totals;
  if (!t.runs) return null;
  return (
    <div className="rounded-xl border border-white/[0.07] bg-[#12151c] p-4">
      <div className="mb-3 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        <TrendingUp className="h-3.5 w-3.5" /> Your record
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-3 text-[12px]">
        <div>
          <div className="text-lg font-bold tabular-nums">{pctText(t.retention)}</div>
          <div className="text-muted-foreground">reviews remembered{t.reviews ? ` (${t.recalled}/${t.reviews})` : ""}</div>
        </div>
        <div>
          <div className="text-lg font-bold tabular-nums">{pctText(t.firstTryRate)}</div>
          <div className="text-muted-foreground">first solves with no wrong run or help</div>
        </div>
        <div>
          <div className="text-lg font-bold tabular-nums">{t.runs}</div>
          <div className="text-muted-foreground">runs, {t.wrong} wrong</div>
        </div>
        <div>
          <div className="text-lg font-bold tabular-nums">{study.counts.mature}</div>
          <div className="text-muted-foreground">problems held 3+ weeks</div>
        </div>
      </div>
      <div className="mt-3 border-t border-white/[0.05] pt-2 text-[11.5px] text-muted-foreground">
        Today: {study.today.learned} new solved · {study.today.reviews} reviewed · {study.today.runs} runs
      </div>
      {study.trouble.length > 0 && (
        <div className="mt-3 border-t border-white/[0.05] pt-2">
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Trouble spots</div>
          {study.trouble.map((p) => (
            <Link key={p.id} to={`/kata/${p.id}`} className="flex items-center gap-2 rounded px-1 py-1 text-[12.5px] hover:bg-white/[0.04]">
              <span className="w-9 font-mono text-[11px] text-muted-foreground">{label(p.id)}</span>
              <span className="flex-1 truncate">{p.title}</span>
              <span className="text-[10.5px] text-muted-foreground">{p.lapses ? `forgot ${p.lapses}× · ` : ""}{p.wrong} wrong</span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function ModuleDrawer({ m, study, onClose }: { m: ModuleSummary | null; study?: StudyOverview; onClose: () => void }) {
  const navigate = useNavigate();
  const node = study?.tree.nodes.find((n) => n.id === m?.id);
  const first = m?.problems.find((p) => p.status !== "solved");
  return (
    <Sheet open={Boolean(m)} onOpenChange={(o) => !o && onClose()}>
      <SheetContent side="right" className="w-full overflow-y-auto border-white/10 bg-[#10131a] sm:max-w-[440px]">
        {m && (
          <>
            <SheetHeader className="mb-3 text-left">
              <div className="font-mono text-xs text-primary">{m.id}{node ? ` · ${node.lessonTitle}` : ""}</div>
              <SheetTitle className="text-xl">{m.title}</SheetTitle>
            </SheetHeader>
            {m.intro && <div className="text-[13px] text-muted-foreground"><KataMarkdown compact>{m.intro}</KataMarkdown></div>}
            {m.after.length > 0 && (
              <div className="mt-3 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                Builds on
                {m.after.map((a) => {
                  const an = study?.tree.nodes.find((n) => n.id === a);
                  return (
                    <span key={a} className={cn("rounded-md border px-1.5 py-0.5 font-mono", an?.status === "done" ? "border-emerald-500/30 text-emerald-300" : "border-white/10")}>
                      {a}{an ? ` ${an.solved}/${an.total}` : ""}
                    </span>
                  );
                })}
              </div>
            )}
            {node?.status === "locked" && (
              <div className="mt-3 rounded-md border border-amber-400/15 bg-amber-400/[0.05] px-3 py-2 text-xs text-amber-200/90">
                Finish the modules it builds on first. You can still open its problems, but Next will not send you here yet.
              </div>
            )}
            <div className="mt-4 overflow-hidden rounded-lg border border-white/[0.06]">
              {m.problems.map((p) => (
                <Link key={p.id} to={`/kata/${p.id}`} className="flex items-center gap-2.5 border-b border-white/[0.04] px-3 py-2 text-[13px] last:border-0 hover:bg-white/[0.04]">
                  <StatusIcon status={p.status} className="h-3.5 w-3.5" />
                  <span className="w-9 font-mono text-[11.5px] text-muted-foreground">{label(p.id)}</span>
                  <span className="flex-1 truncate">{p.title}</span>
                  <DueChip card={p.card} reviewing={p.reviewing} long />
                  <KindIcon kind={p.kind} />
                  <DifficultyPill d={p.difficulty} className="text-[10px]" />
                </Link>
              ))}
            </div>
            {m.minutes && (
              <div className="mt-2 flex items-center gap-1 text-[11px] text-muted-foreground/70">
                <Clock className="h-3 w-3" /> about {m.minutes >= 90 ? `${Math.round(m.minutes / 60)} h` : `${m.minutes} min`}
              </div>
            )}
            {first && (
              <button
                onClick={() => navigate(`/kata/${first.id}`)}
                className="mt-4 w-full rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90"
              >
                {first.status === "attempted" ? "Continue" : "Start"} <span className="font-mono">{label(first.id)}</span> {first.title}
              </button>
            )}
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}

export default function KataHome() {
  const { data, error, isLoading } = useQuery({ queryKey: ["kata-catalog"], queryFn: kataApi.catalog });
  const [trackPick, setTrackPick] = useState(() => readLS("kata.track", "main"));
  const pickTrack = (t: string) => {
    setTrackPick(t);
    writeLS("kata.track", t);
  };
  const { data: study } = useQuery({
    queryKey: ["kata-study", trackPick],
    queryFn: () => kataApi.study(trackPick),
    refetchInterval: 60_000,
    placeholderData: (prev) => prev,
  });
  // The server answers for main when the stored track is unknown, parked or empty.
  const track = study?.track ?? trackPick;
  const tracks = study?.tracks ?? data?.tracks ?? [];
  const [view, setViewState] = useState<"tree" | "list">(() => (readLS("kata.homeView", "tree") === "list" ? "list" : "tree"));
  const setView = (v: "tree" | "list") => {
    setViewState(v);
    writeLS("kata.homeView", v);
  };
  const [picked, setPicked] = useState<string | null>(null);
  const modules = data?.lessons.flatMap((l) => l.modules).filter((m) => (m.track ?? "main") === track) ?? [];
  const trackLessons = data?.lessons.filter((l) => l.modules.some((m) => (m.track ?? "main") === track)) ?? [];
  const lesson = trackLessons[0] ?? data?.lessons[0];
  const pickedModule = modules.find((m) => m.id === picked) ?? null;
  const next = study?.next ?? (track === "main" ? data?.study : undefined);
  const nextModule = next && next.kind !== "done" ? next.module : null;
  const issues = study?.issues ?? data?.issues ?? [];

  return (
    <div className="min-h-screen bg-[#0e1015] text-foreground">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/3 h-[420px] w-[820px] rounded-full bg-primary/[0.06] blur-3xl" />
        <div className="absolute bottom-0 right-0 h-[400px] w-[500px] rounded-full bg-emerald-500/[0.03] blur-3xl" />
      </div>
      <div className="relative mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          <div className="min-w-0">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-orange-500 to-amber-400 text-black shadow-lg shadow-primary/20">
                <Brain className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-2xl font-black tracking-tight">Kernel Kata</h1>
                <p className="text-sm text-muted-foreground">
                  {data && data.lessons.length > 1
                    ? `${data.lessons.length} lessons${tracks.length > 1 ? ` in ${tracks.length} tracks` : ""}.`
                    : `${lesson?.title ?? "Lesson 1 · PyTorch"}: from tensor shapes to one decode step of BioMistral-7B.`}
                </p>
              </div>
            </div>

            {isLoading && <div className="mt-10 flex items-center gap-2 text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" /> Loading the curriculum…</div>}
            {error && (
              <div className="mt-8 rounded-xl border border-rose-500/20 bg-rose-500/5 p-4 text-sm text-rose-200">
                Cannot load the kata: {(error as Error).message}. Is the API server running? (<code className="font-mono">npm run dev:all</code>)
              </div>
            )}

            {issues.length > 0 && (
              <div className="mt-6 rounded-xl border border-amber-400/25 bg-amber-400/[0.05] p-4 text-[13px] text-amber-100/90">
                <div className="mb-1 flex items-center gap-2 font-semibold"><AlertTriangle className="h-4 w-4" /> The progression index needs fixing</div>
                <ul className="list-disc space-y-0.5 pl-5 font-mono text-[12px]">{issues.map((m) => <li key={m}>{m}</li>)}</ul>
                <div className="mt-1 text-[12px] text-amber-100/70">Edit <code className="font-mono">server/kata/content/progression.md</code>, then reload.</div>
              </div>
            )}

            {data && (
              <>
                <TrackSwitcher tracks={tracks} current={track} onPick={pickTrack} />
                <StudyCard data={data} study={study} track={track} />

                {lesson?.intro && trackLessons.length === 1 && (
                  <div className="mt-4 rounded-xl border border-white/[0.05] px-5 py-1 text-muted-foreground">
                    <KataMarkdown compact>{lesson.intro}</KataMarkdown>
                  </div>
                )}

                <div className="mt-6 flex items-center justify-between gap-3">
                  <h2 className="text-sm font-semibold text-muted-foreground">
                    {view === "tree" ? "The path. Click a module to see its problems." : "Modules, in path order"}
                  </h2>
                  <div className="flex shrink-0 items-center rounded-md border border-white/[0.06] bg-white/[0.03] p-0.5">
                    {([["tree", "Roadmap", <Network key="i" className="h-3.5 w-3.5" />], ["list", "List", <LayoutGrid key="i" className="h-3.5 w-3.5" />]] as const).map(([v, name, icon]) => (
                      <button
                        key={v}
                        onClick={() => setView(v)}
                        className={cn("flex items-center gap-1.5 rounded-[5px] px-2.5 py-1 text-xs font-medium", view === v ? "bg-white/10 text-foreground" : "text-muted-foreground hover:text-foreground")}
                      >
                        {icon} {name}
                      </button>
                    ))}
                  </div>
                </div>

                {view === "tree" ? (
                  study ? (
                    <div className="mt-3 rounded-xl border border-white/[0.06] bg-[#0f1218]/80 px-2 py-5 sm:px-4">
                      <Roadmap nodes={study.tree.nodes} nextModule={nextModule} selected={picked} onPick={setPicked} />
                      <div className="mt-4"><RoadmapLegend /></div>
                    </div>
                  ) : (
                    <div className="mt-10 flex items-center justify-center gap-2 text-sm text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" /> Drawing the path…</div>
                  )
                ) : (
                  <div className="mt-3 grid gap-4 md:grid-cols-2">
                    {modules.map((m, i) => <ModuleCard key={m.id} m={m} i={i} />)}
                  </div>
                )}
              </>
            )}
          </div>

          <aside className="space-y-4 lg:sticky lg:top-6 lg:self-start">
            <EnvCard />
            {study && <StatsCard study={study} />}
            <div className="rounded-xl border border-white/[0.07] bg-[#12151c] p-4 text-[12.5px] leading-relaxed text-muted-foreground">
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider">How it works</div>
              <ul className="list-disc space-y-1 pl-4">
                <li>Read the lesson, run its examples, then fill the <code className="font-mono text-amber-300">TODO</code>s.</li>
                <li><kbd className="font-mono">Ctrl+Enter</kbd> runs every check on your machine, on CPU or your GPU.</li>
                <li>The Variables tab shows each tensor's shape, dtype, device and stride.</li>
                <li>There is no autocomplete. The goal is to write the code without help.</li>
                <li>
                  <RotateCcw className="mr-1 inline h-3 w-3 text-sky-300" />
                  Solved problems come back for review, Anki-style: tomorrow, then a few days, then weeks.
                  Each attempt is graded from your runs and any hints or solution you used.{" "}
                  <b className="text-foreground/80">Next</b> picks a due review first, then the next new problem on the path.
                </li>
              </ul>
            </div>
            {lesson?.traps && (
              <Collapsible title="The traps" icon={<AlertTriangle className="h-4 w-4 text-amber-400" />}>
                <KataMarkdown compact>{lesson.traps}</KataMarkdown>
              </Collapsible>
            )}
            {lesson?.recall && (
              <Collapsible title="Recall list" icon={<Brain className="h-4 w-4 text-sky-400" />}>
                <KataMarkdown compact>{lesson.recall}</KataMarkdown>
              </Collapsible>
            )}
          </aside>
        </div>
      </div>
      <ModuleDrawer m={pickedModule} study={study} onClose={() => setPicked(null)} />
    </div>
  );
}
