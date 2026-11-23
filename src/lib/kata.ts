import { API_URL } from "@/lib/api";

export type Difficulty = "Easy" | "Medium" | "Hard";
export type KataStatus = "solved" | "attempted" | "unsolved";
export type Device = "cpu" | "cuda";

export interface KataEnv {
  python: string;
  ready: number;
  starting: number;
  lastError: string | null;
  env: null | {
    python: string;
    torch: string;
    cuda_available: boolean;
    cuda_version: string | null;
    device_name?: string;
    capability?: string;
    total_memory?: number;
    sm_count?: number;
  };
}

export interface ProblemSummary {
  id: string;
  n: number;
  title: string;
  difficulty: Difficulty;
  kind: "code" | "predict";
  status: KataStatus;
  runs: number;
  attempts: number;
  solvedAt?: string | null;
  lastAt?: string | null;
  card: Card | null;
  reviewing: boolean;
}

// ---- spaced repetition ------------------------------------------------------

export type Grade = 1 | 2 | 3 | 4;
export const GRADE_NAMES: Record<Grade, string> = { 1: "Again", 2: "Hard", 3: "Good", 4: "Easy" };

export interface Card {
  state: "new" | "learning" | "review" | "relearning";
  due: string;
  interval: number;
  ease: number;
  reps: number;
  lapses: number;
  last: string | null;
}

export interface StudyNext {
  kind: "review" | "new" | "done";
  id?: string;
  title?: string;
  module?: string;
  moduleTitle?: string;
  problemKind?: "code" | "predict";
  resumed?: boolean;
  card?: Card;
  lastSeen?: string | null;
  dueCount?: number;
  started?: boolean;
  skippedAhead?: boolean;
  nextDue?: string | null;
  track?: string;
}

/** One track of progression.md: its own path, Next queue and due count. */
export interface TrackSummary {
  id: string;
  title: string;
  modules: number;
  total: number;
  solved: number;
  due: number;
}

export interface GradeInfo {
  kind: "learn" | "review";
  grade: Grade;
  auto: boolean;
  autoGrade: Grade;
  reason: string;
  endedAt: string;
  runs: number;
  wrong: number;
  hints: number;
  revealed: boolean;
  options: Record<Grade, { due: string; interval: number; state: Card["state"] }>;
  due: string;
  interval: number;
  state: Card["state"];
}

export interface AttemptLog {
  kind: "learn" | "review";
  startedAt: string;
  endedAt: string;
  runs: number;
  wrong: number;
  todo: number;
  hints: number;
  revealed: boolean;
  grade: Grade;
  autoGrade: Grade;
  auto: boolean;
  reason: string;
  migrated?: boolean;
}

export interface TreeNode {
  id: string;
  index: number;
  title: string;
  minutes: number | null;
  lesson: string;
  lessonTitle: string;
  depth: number;
  after: string[];
  total: number;
  solved: number;
  due: number;
  status: "done" | "active" | "ready" | "locked";
}

export interface StudyOverview {
  now: string;
  track: string;
  tracks: TrackSummary[];
  next: StudyNext;
  due: { id: string; title: string; module: string; due: string; interval: number; state: Card["state"]; reviewing: boolean }[];
  counts: { total: number; solved: number; cards: number; mature: number; due: number; dueTomorrow: number; dueWeek: number; laterToday: string | null };
  today: { reviews: number; learned: number; runs: number };
  totals: { runs: number; wrong: number; passes: number; reviews: number; recalled: number; firstTry: number; learned: number; retention: number | null; firstTryRate: number | null };
  trouble: { id: string; title: string; module: string; wrong: number; lapses: number }[];
  tree: { nodes: TreeNode[]; lessons: string[] };
  issues: string[];
}

export interface ModuleSummary {
  id: string;
  title: string;
  minutes: number | null;
  intro: string;
  after: string[];
  track: string;
  problems: ProblemSummary[];
}

export interface Catalog {
  lessons: { id: string; title: string; intro: string; recall: string; traps: string; modules: ModuleSummary[] }[];
  total: number;
  solved: number;
  next: string | null;
  study: StudyNext;
  tracks: TrackSummary[];
  issues: string[];
}

export interface HistoryEntry {
  at: string;
  status: "passed" | "failed" | "error";
  passed: number;
  total: number;
  device: Device;
  ms: number | null;
  error: string | null;
  kind?: "passed" | "wrong" | "todo" | "sys";
  attempt?: "learn" | "review" | "practice";
  code?: string;
  answers?: Record<string, string>;
}

export interface KataProblem {
  id: string;
  n: number;
  title: string;
  difficulty: Difficulty;
  kind: "code" | "predict";
  devices: Device[];
  lesson: string;
  demos: string[];
  task: string;
  given: string;
  starter: string;
  hints: string[];
  takeaway: string;
  hasSolution: boolean;
  items: { key: string; expr: string; kind: "shape" | "raises" }[];
  uses: { id: string; title: string; source: "yours" | "reference" | "missing" }[];
  module: { id: string; title: string; minutes: number | null; intro: string; shared: string; problemIds: string[] };
  lesson_info: { id: string; title: string };
  /** The track in progression.md ("off" = parked). */
  track: string;
  /** The track Next and due counts use for this problem ("off" falls back to main). */
  studyTrack: string;
  position: { index: number; total: number; prev: string | null; next: string | null };
  progress: ProblemProgress;
}

export interface ProblemProgress {
  status: KataStatus;
  runs: number;
  attempts: number;
  draft?: string;
  answers?: Record<string, string>;
  solvedAt: string | null;
  revealed: boolean;
  history: HistoryEntry[];
  stats: {
    runs: number; passes: number; wrong: number; todoRuns: number; reviews: number; lapses: number;
    firstSolve: { runs: number; wrong: number; hints: number; revealed: boolean; at: string } | null;
  };
  open: { kind: "learn" | "review"; startedAt: string; runs: number; wrong: number; hints: number; revealed: boolean } | null;
  log: AttemptLog[];
  card: Card | null;
  lastGrade: GradeInfo | null;
}

export interface CheckResult {
  label: string;
  passed: boolean;
  details: string[];
}

export interface RunError {
  phase: string;
  type: string;
  message: string;
  line?: number | null;
  col?: number | null;
  traceback: string;
}

export interface InspectEntry {
  name: string;
  kind: string;
  summary?: string;
  shape?: number[];
  dtype?: string;
  device?: string;
  stride?: number[];
  contiguous?: boolean;
  numel?: number;
  preview?: string;
}

export interface RunResult {
  status: "passed" | "failed" | "error";
  error: RunError | null;
  checks: CheckResult[];
  console: { phase: string; text: string }[];
  inspect?: InspectEntry[];
  uses?: { id: string; title: string; source: string; note?: string }[];
  todo_lines?: number[];
  timing?: { user_ms?: number; checks_ms?: number; total_ms?: number };
  device?: Device;
  output?: string;
  wall_ms?: number;
}

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      msg = body.error || msg;
    } catch (_) { /* not json */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

const base = `${API_URL}/api/kata`;
const json = (method: string, body: unknown, keepalive = false): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
  keepalive,
});

export const kataApi = {
  env: () => fetch(`${base}/env`).then(j<KataEnv>),
  catalog: () => fetch(`${base}/catalog`).then(j<Catalog>),
  problem: (id: string) => fetch(`${base}/problems/${id}`).then(j<KataProblem>),
  solution: (id: string) =>
    fetch(`${base}/problems/${id}/solution`).then(j<{ solution: string; items: { key: string; answer: string }[] }>),
  saveDraft: (id: string, body: { code?: string; answers?: Record<string, string> }, keepalive = false) =>
    fetch(`${base}/problems/${id}/draft`, json("PUT", body, keepalive)).then(j<{ ok: boolean }>),
  run: (id: string, body: { code?: string; answers?: Record<string, string>; device: Device }) =>
    fetch(`${base}/problems/${id}/run`, json("POST", body)).then(
      j<{ result: RunResult; graded: GradeInfo | null; progress: ProblemProgress; next: StudyNext }>,
    ),
  study: (track = "main") => fetch(`${base}/study?track=${encodeURIComponent(track)}`).then(j<StudyOverview>),
  studyNext: (track = "main") => fetch(`${base}/study/next?track=${encodeURIComponent(track)}`).then(j<StudyNext>),
  startReview: (id: string) =>
    fetch(`${base}/problems/${id}/review`, json("POST", {})).then(j<{ resumed: boolean; problem: KataProblem }>),
  cancelReview: (id: string) =>
    fetch(`${base}/problems/${id}/review`, { method: "DELETE" }).then(j<{ ok: boolean; problem: KataProblem }>),
  grade: (id: string, grade: Grade) =>
    fetch(`${base}/problems/${id}/grade`, json("POST", { grade })).then(j<{ lastGrade: GradeInfo; card: Card; progress: ProblemProgress }>),
  hint: (id: string, shown: number) =>
    fetch(`${base}/problems/${id}/hint`, json("POST", { shown })).then(j<{ hints: number }>),
  demo: (id: string, index: number, device: Device) =>
    fetch(`${base}/problems/${id}/demo`, json("POST", { index, device })).then(j<{ result: RunResult }>),
  scratch: (id: string, code: string, scratch: string, device: Device) =>
    fetch(`${base}/problems/${id}/scratch`, json("POST", { code, scratch, device })).then(j<{ result: RunResult }>),
};

/** "1c-2" -> "1c.2" for display. */
export const label = (id: string) => id.replace("-", ".");

export function fmtBytes(n?: number) {
  if (!n) return "";
  return `${(n / 1024 ** 3).toFixed(1)} GB`;
}

export function timeAgo(iso: string) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric" });
}

/** "in 10 min", "tomorrow", "in 3 days", "in 2 wk", "in 3 mo"; "now" when due. */
export function fmtDue(iso: string | null | undefined, now = Date.now()) {
  if (!iso) return "";
  const ms = new Date(iso).getTime() - now;
  if (ms <= 0) return "now";
  const min = Math.round(ms / 60000);
  if (min < 60) return `in ${Math.max(1, min)} min`;
  // Calendar days, so a card due at 4 am tomorrow reads "tomorrow", not "in 9 h".
  const d = new Date(iso);
  const n = new Date(now);
  const days = Math.round((new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
    - new Date(n.getFullYear(), n.getMonth(), n.getDate()).getTime()) / 86400000);
  if (days <= 0) return `in ${Math.round(min / 60)} h`;
  if (days === 1) return "tomorrow";
  if (days < 14) return `in ${days} days`;
  if (days < 60) return `in ${Math.round(days / 7)} wk`;
  return `in ${Math.round(days / 30)} mo`;
}

/** Short interval label for a grade button: "10m", "1d", "6d", "3w", "2mo". */
export function fmtInterval(o: { interval: number; state: Card["state"] }) {
  if (o.state === "learning" || o.state === "relearning") return "10m";
  const d = o.interval;
  if (d < 14) return `${d}d`;
  if (d < 60) return `${Math.round(d / 7)}w`;
  return `${Math.round(d / 30)}mo`;
}
