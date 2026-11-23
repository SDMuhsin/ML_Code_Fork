import { motion } from "framer-motion";
import { ArrowRight, Brain, CalendarClock, Loader2, RotateCcw, X } from "lucide-react";
import {
  fmtDue, fmtInterval, GRADE_NAMES, timeAgo,
  type AttemptLog, type Grade, type GradeInfo, type ProblemProgress, type StudyNext,
} from "@/lib/kata";
import { describeNext } from "@/lib/useStudy";
import { cn } from "@/lib/utils";

const gradeStyle: Record<Grade, { on: string; text: string }> = {
  1: { on: "border-rose-400/50 bg-rose-500/20 text-rose-100", text: "text-rose-300" },
  2: { on: "border-amber-400/50 bg-amber-500/20 text-amber-100", text: "text-amber-300" },
  3: { on: "border-emerald-400/50 bg-emerald-500/20 text-emerald-100", text: "text-emerald-300" },
  4: { on: "border-sky-400/50 bg-sky-500/20 text-sky-100", text: "text-sky-300" },
};

const gradeHelp: Record<Grade, string> = {
  1: "I could not have written it without the solution. Retype it in 10 minutes.",
  2: "I got there, but it was a struggle.",
  3: "I wrote it with normal effort.",
  4: "Trivial now. Push it far out.",
};

export function NextButton({ next, onGo, busy, compact, className }: {
  next?: StudyNext | null; onGo: () => void; busy?: boolean; compact?: boolean; className?: string;
}) {
  const d = describeNext(next);
  const done = next?.kind === "done";
  const review = next?.kind === "review";
  return (
    <button
      onClick={onGo}
      disabled={busy || !next || done}
      title={d.sub}
      className={cn(
        "group flex items-center gap-2 rounded-lg font-semibold transition-colors disabled:cursor-default",
        compact ? "px-3 py-1.5 text-xs" : "px-4 py-2.5 text-sm shadow-lg",
        done
          ? "bg-white/[0.05] text-muted-foreground"
          : review
            ? "bg-sky-500 text-white shadow-sky-500/20 hover:bg-sky-400"
            : "bg-primary text-primary-foreground shadow-primary/20 hover:bg-primary/90",
        className,
      )}
    >
      {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : review ? <RotateCcw className="h-3.5 w-3.5" /> : null}
      <span className="flex min-w-0 flex-col items-start text-left leading-tight">
        <span className="truncate">
          {d.verb}
          {d.id && <span className="ml-1.5 font-mono opacity-80">{d.id}</span>}
          {!compact && d.title && <span className="ml-2 hidden font-normal opacity-80 sm:inline">{d.title}</span>}
        </span>
        {!compact && d.sub && <span className="mt-0.5 text-[11px] font-normal opacity-75">{d.sub}</span>}
      </span>
      {!done && <ArrowRight className="h-4 w-4 shrink-0 transition-transform group-hover:translate-x-0.5" />}
    </button>
  );
}

/** Under "Accepted": how the attempt was graded, Anki's four buttons to change it, and Next. */
export function GradeBar({ info, onGrade, grading, next, onNext, nextBusy }: {
  info: GradeInfo; onGrade: (g: Grade) => void; grading: Grade | null;
  next?: StudyNext | null; onNext: () => void; nextBusy?: boolean;
}) {
  const g = info.grade;
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg border border-white/[0.07] bg-white/[0.02] p-3"
    >
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-[13px]">
        <Brain className="h-3.5 w-3.5 translate-y-0.5 text-sky-300" />
        <span className="text-muted-foreground">{info.kind === "review" ? "Review" : "First solve"} graded</span>
        <span className={cn("font-semibold", gradeStyle[g].text)}>{GRADE_NAMES[g]}</span>
        <span className="text-muted-foreground">
          {info.auto ? `· ${info.reason}` : `· you chose it (automatic was ${GRADE_NAMES[info.autoGrade]}: ${info.reason})`}
        </span>
        <span className="text-muted-foreground">· next review <span className="text-foreground">{fmtDue(info.due)}</span></span>
      </div>
      <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[11px] text-muted-foreground">How did it feel?</span>
        {([1, 2, 3, 4] as Grade[]).map((k) => (
          <button
            key={k}
            onClick={() => k !== g && onGrade(k)}
            disabled={grading !== null}
            title={gradeHelp[k]}
            className={cn(
              "flex min-w-[64px] flex-col items-center rounded-md border px-2.5 py-1 text-xs transition-colors",
              k === g ? gradeStyle[k].on : "border-white/[0.08] text-muted-foreground hover:border-white/20 hover:text-foreground",
            )}
          >
            <span className="font-semibold">{grading === k ? <Loader2 className="h-3 w-3 animate-spin" /> : GRADE_NAMES[k]}</span>
            <span className="font-mono text-[10px] opacity-75">{fmtInterval(info.options[k])}</span>
          </button>
        ))}
        {next && (
          <div className="ml-auto">
            <NextButton next={next} onGo={onNext} busy={nextBusy} compact />
          </div>
        )}
      </div>
    </motion.div>
  );
}

/** Top of the description: what kind of session this is. */
export function ReviewBanner({ progress, onStart, onCancel, busy }: {
  progress: ProblemProgress; onStart: () => void; onCancel: () => void; busy: boolean;
}) {
  const card = progress.card;
  const open = progress.open;
  if (open?.kind === "review") {
    return (
      <div className="mt-4 flex items-start gap-3 rounded-lg border border-sky-400/25 bg-sky-400/[0.06] px-4 py-3 text-[13px]">
        <RotateCcw className="mt-0.5 h-4 w-4 shrink-0 text-sky-300" />
        <div className="min-w-0 flex-1">
          <div className="font-semibold text-sky-100">Review: write it from memory</div>
          <div className="mt-0.5 leading-relaxed text-sky-100/70">
            {card?.last ? `Last done ${timeAgo(card.last)}. ` : ""}
            The editor starts from the starter; your earlier code is in History.
            Hints or the solution will lower the grade.
          </div>
        </div>
        <button
          onClick={onCancel}
          disabled={busy}
          title="Stop this review without grading it and put your previous code back"
          className="flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-xs text-sky-200/70 hover:bg-white/5 hover:text-sky-100"
        >
          <X className="h-3.5 w-3.5" /> Cancel review
        </button>
      </div>
    );
  }
  if (!card) return null;
  const due = new Date(card.due).getTime() <= Date.now();
  if (due) {
    return (
      <div className="mt-4 flex items-center gap-3 rounded-lg border border-sky-400/25 bg-sky-400/[0.06] px-4 py-3 text-[13px]">
        <CalendarClock className="h-4 w-4 shrink-0 text-sky-300" />
        <div className="flex-1 text-sky-100/80">
          <span className="font-semibold text-sky-100">Due for review.</span> Write it again from the starter, without looking.
        </div>
        <button onClick={onStart} disabled={busy} className="flex items-center gap-1.5 rounded-md bg-sky-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-sky-400">
          {busy && <Loader2 className="h-3 w-3 animate-spin" />} Start review
        </button>
      </div>
    );
  }
  return (
    <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
      <CalendarClock className="h-3.5 w-3.5" />
      <span>Next review <span className="text-foreground/80">{fmtDue(card.due)}</span></span>
      <span className="text-muted-foreground/50">·</span>
      <button onClick={onStart} disabled={busy} className="underline decoration-dotted underline-offset-2 hover:text-foreground" title="Review now from the starter. It counts as an early review.">
        practise it now
      </button>
    </div>
  );
}

function Tile({ k, v, sub, tone }: { k: string; v: React.ReactNode; sub?: string; tone?: string }) {
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
      <div className="text-[10.5px] uppercase tracking-wider text-muted-foreground">{k}</div>
      <div className={cn("mt-0.5 text-lg font-semibold tabular-nums", tone)}>{v}</div>
      {sub && <div className="text-[11px] text-muted-foreground">{sub}</div>}
    </div>
  );
}

/** History tab header: the numbers, then one row per attempt. */
export function AttemptStats({ progress }: { progress: ProblemProgress }) {
  const s = progress.stats;
  const fs = s.firstSolve;
  if (!s.runs && !progress.log.length) return null;
  return (
    <div className="mb-4 space-y-3">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Tile k="Runs" v={s.runs} sub={s.todoRuns ? `${s.todoRuns} with TODOs left` : undefined} />
        <Tile k="Wrong runs" v={s.wrong} tone={s.wrong ? "text-amber-300" : undefined} sub="finished code that failed" />
        <Tile
          k="First solve"
          v={fs ? `run ${fs.runs}` : progress.open?.kind === "learn" ? "…" : "—"}
          sub={fs ? [fs.hints ? `${fs.hints} hint${fs.hints > 1 ? "s" : ""}` : "", fs.revealed ? "saw solution" : ""].filter(Boolean).join(", ") || "no help" : progress.open ? `${progress.open.runs} runs so far` : undefined}
        />
        <Tile
          k="Reviews"
          v={s.reviews}
          sub={progress.card ? `${s.lapses ? `${s.lapses} forgotten · ` : ""}next ${fmtDue(progress.card.due)}` : "after the first solve"}
        />
      </div>
      {progress.log.length > 0 && (
        <div>
          <div className="mb-1 px-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Attempts</div>
          <div className="overflow-hidden rounded-md border border-white/[0.05]">
            {progress.log.map((a: AttemptLog, i) => (
              <div key={a.endedAt + i} className="flex items-center gap-3 border-b border-white/[0.04] px-3 py-1.5 text-xs last:border-0">
                <span className="w-20 text-muted-foreground">{a.kind === "review" ? "Review" : "First solve"}</span>
                <span className={cn("w-12 font-semibold", gradeStyle[a.grade].text)}>{GRADE_NAMES[a.grade]}</span>
                <span className="flex-1 truncate text-muted-foreground">
                  {a.migrated ? a.reason : `${a.runs} run${a.runs === 1 ? "" : "s"}, ${a.wrong} wrong${a.hints ? `, ${a.hints} hint${a.hints > 1 ? "s" : ""}` : ""}${a.revealed ? ", saw solution" : ""}`}
                </span>
                <span className="text-muted-foreground/70">{timeAgo(a.endedAt)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {progress.open && (
        <div className="px-1 text-[11px] text-muted-foreground">
          In progress: {progress.open.kind === "review" ? "a review" : "first solve"}, {progress.open.runs} run{progress.open.runs === 1 ? "" : "s"} so far
          {progress.open.wrong ? `, ${progress.open.wrong} wrong` : ""}.
        </div>
      )}
    </div>
  );
}

