// Your kata progress: one JSON file, no database and no login needed.
// Lives at server/data/kata-progress.json (gitignored). Back it up if you want to keep it.
//
// Per problem it keeps:
//   counters   runs, passes, wrong, todoRuns, sysRuns (lifetime; never trimmed)
//   history    the last 30 runs, with the code
//   open       the attempt in progress: a "learn" attempt (until first solved) or a "review"
//   log        finished attempts: how many runs, wrong runs, hints, solution opened, grade
//   card       the spaced-repetition card (srs.js), created when the problem is first solved
//
// A *run* is one press of Run. An *attempt* is all the runs from starting a
// problem (or a review of it) until it passes.

const fs = require('fs');
const path = require('path');
const srs = require('./srs');

const DATA_DIR = process.env.KATA_DATA_DIR || path.join(__dirname, '..', 'data');
const FILE = path.join(DATA_DIR, 'kata-progress.json');
const MAX_HISTORY = 30;
const MAX_LOG = 60;
const VERSION = 2;

let state = null;
let saveTimer = null;

function migrate(id, e) {
  if (e.runs === undefined) e.runs = e.attempts || 0; // v1 called runs "attempts"
  delete e.attempts;
  for (const k of ['passes', 'wrong', 'todoRuns', 'sysRuns']) e[k] = e[k] || 0;
  e.history = e.history || [];
  e.log = e.log || [];
  if (e.status === 'solved' && !e.card) {
    const t = Date.parse(e.solvedAt || e.lastAt) || Date.now();
    e.card = srs.schedule(null, 3, t, id);
    e.log.push({ kind: 'learn', startedAt: e.solvedAt, endedAt: new Date(t).toISOString(), runs: e.runs, wrong: 0,
      todo: 0, hints: 0, revealed: Boolean(e.revealedAt), grade: 3, auto: true, reason: 'solved before attempts were tracked', migrated: true });
  }
}

function load() {
  if (state) return state;
  try {
    state = JSON.parse(fs.readFileSync(FILE, 'utf8'));
  } catch (_) {
    state = {};
  }
  state.problems = state.problems || {};
  state.settings = state.settings || {};
  if (state.v !== VERSION) {
    for (const [id, e] of Object.entries(state.problems)) migrate(id, e);
    state.v = VERSION;
  }
  return state;
}

function flush() {
  if (!state) return;
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const tmp = FILE + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(state, null, 1));
  fs.renameSync(tmp, FILE);
}

function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flush, 150);
}

function entry(id) {
  const s = load();
  if (!s.problems[id]) {
    s.problems[id] = { status: 'unsolved', runs: 0, passes: 0, wrong: 0, todoRuns: 0, sysRuns: 0, history: [], log: [] };
  }
  return s.problems[id];
}

function get(id) {
  return load().problems[id] || null;
}

function all() {
  return load().problems;
}

const blankAttempt = (kind, now) => ({ kind, startedAt: now, runs: 0, wrong: 0, todo: 0, hints: 0, revealed: false });

/** The attempt runs and hints count toward; opens a learn attempt for an unsolved problem. */
function current(e, now) {
  if (!e.open && e.status !== 'solved') e.open = blankAttempt('learn', now);
  return e.open || null;
}

function cardSummary(e) {
  const c = e && e.card;
  if (!c) return null;
  return { state: c.state, due: c.due, interval: c.interval, ease: c.ease, reps: c.reps, lapses: c.lapses, last: c.last };
}

function summary(id) {
  const e = get(id);
  if (!e) return { status: 'unsolved', runs: 0, attempts: 0, card: null, reviewing: false };
  return {
    status: e.status,
    runs: e.runs,
    attempts: e.runs, // older clients
    solvedAt: e.solvedAt || null,
    lastAt: e.lastAt || null,
    card: cardSummary(e),
    reviewing: Boolean(e.open && e.open.kind === 'review'),
  };
}

/** Everything the workspace shows about attempts on one problem. */
function details(id) {
  const e = get(id) || {};
  const log = e.log || [];
  const first = log.find((a) => a.kind === 'learn');
  return {
    status: e.status || 'unsolved',
    runs: e.runs || 0,
    attempts: e.runs || 0,
    draft: e.draft,
    answers: e.answers,
    solvedAt: e.solvedAt || null,
    // In a review the solution is gated again; seeing it then counts against the review.
    revealed: e.open && e.open.kind === 'review' ? e.open.revealed : Boolean(e.revealedAt),
    history: (e.history || []).slice(0, MAX_HISTORY),
    stats: {
      runs: e.runs || 0,
      passes: e.passes || 0,
      wrong: e.wrong || 0,
      todoRuns: e.todoRuns || 0,
      firstSolve: first ? { runs: first.runs, wrong: first.wrong, hints: first.hints, revealed: first.revealed, at: first.endedAt } : null,
      reviews: log.filter((a) => a.kind === 'review').length,
      lapses: e.card ? e.card.lapses : 0,
    },
    open: e.open ? { kind: e.open.kind, startedAt: e.open.startedAt, runs: e.open.runs, wrong: e.open.wrong, hints: e.open.hints, revealed: e.open.revealed } : null,
    log: log.slice(-20).reverse().map(({ before, ...a }) => a),
    card: cardSummary(e),
    lastGrade: lastGradeInfo(id, e),
  };
}

/** The most recent graded attempt, with what the other grades would have done (for the override buttons). */
function lastGradeInfo(id, e) {
  const a = e.log && e.log[e.log.length - 1];
  if (!a || !('before' in a) || !e.card || e.card.last !== a.endedAt) return null;
  return {
    kind: a.kind, grade: a.grade, auto: a.auto, autoGrade: a.autoGrade, reason: a.reason, endedAt: a.endedAt,
    runs: a.runs, wrong: a.wrong, hints: a.hints, revealed: a.revealed,
    options: srs.preview(a.before, Date.parse(a.endedAt), id),
    due: e.card.due, interval: e.card.interval, state: e.card.state,
  };
}

function saveDraft(id, { code, answers }) {
  const e = entry(id);
  if (typeof code === 'string') e.draft = code;
  if (answers && typeof answers === 'object') e.answers = answers;
  e.draftAt = new Date().toISOString();
  save();
}

/** passed | todo (unfinished code, not a wrong answer) | sys (not your fault) | wrong */
function classify({ code, answers, result }) {
  if (result.status === 'passed') return 'passed';
  const phase = result.error && result.error.phase;
  if (phase === 'worker' || phase === 'harness' || phase === 'setup') return 'sys';
  if (typeof code === 'string' && /\bTODO\b/.test(code)) return 'todo';
  if (answers && typeof answers === 'object' && Object.values(answers).some((v) => !String(v || '').trim())) return 'todo';
  if (!answers && typeof code !== 'string') return 'todo';
  return 'wrong';
}

function recordRun(id, { code, answers, device, result }) {
  const e = entry(id);
  const now = new Date().toISOString();
  const kind = classify({ code, answers, result });
  const passed = kind === 'passed';
  const checks = result.checks || [];
  e.runs += 1;
  e.lastAt = now;
  if (passed) e.passes += 1;
  else if (kind === 'wrong') e.wrong += 1;
  else if (kind === 'todo') e.todoRuns += 1;
  else e.sysRuns += 1;
  if (!e.firstRunAt) e.firstRunAt = now;
  if (typeof code === 'string') e.draft = code;
  if (answers) e.answers = answers;

  const a = kind === 'sys' ? e.open : current(e, now);
  if (a && kind !== 'sys') {
    a.runs += 1;
    if (kind === 'wrong') a.wrong += 1;
    if (kind === 'todo') a.todo += 1;
  }

  let graded = null;
  if (passed) {
    if (e.status !== 'solved') e.solvedAt = now;
    e.status = 'solved';
    if (typeof code === 'string') e.lastPassingCode = code;
    if (answers) e.lastPassingAnswers = answers;
    if (a) graded = finishAttempt(id, e, a, now);
  } else if (e.status !== 'solved') {
    e.status = 'attempted';
  }

  e.history.unshift({
    at: now,
    status: result.status,
    kind,
    attempt: a ? a.kind : 'practice',
    passed: checks.filter((c) => c.passed).length,
    total: checks.length,
    device,
    ms: result.timing ? result.timing.user_ms : null,
    error: result.error ? `${result.error.type}: ${result.error.message}`.slice(0, 300) : null,
    code: typeof code === 'string' ? code : undefined,
    answers: answers || undefined,
  });
  e.history = e.history.slice(0, MAX_HISTORY);
  save();
  return { entry: e, graded };
}

function finishAttempt(id, e, a, now) {
  const auto = srs.autoGrade(a);
  const before = e.card ? { ...e.card } : null;
  e.card = srs.schedule(before, auto.grade, Date.parse(now), id);
  e.card.last = now; // exact match marks "this attempt set the card" (for regrading)
  const { draftBefore, answersBefore, ...rest } = a;
  e.log.push({ ...rest, endedAt: now, grade: auto.grade, autoGrade: auto.grade, auto: true, reason: auto.reason, before });
  e.log = e.log.slice(-MAX_LOG);
  e.open = null;
  return lastGradeInfo(id, e);
}

/** Override the grade of the attempt that just finished (Anki's Again/Hard/Good/Easy). */
function regrade(id, grade) {
  const e = get(id);
  const a = e && e.log && e.log[e.log.length - 1];
  if (!a || !e.card || e.card.last !== a.endedAt) return { error: 'There is no just-finished attempt to grade.' };
  const g = Number(grade);
  if (![1, 2, 3, 4].includes(g)) return { error: 'grade must be 1, 2, 3 or 4' };
  e.card = srs.schedule(a.before, g, Date.parse(a.endedAt), id);
  e.card.last = a.endedAt;
  a.grade = g;
  a.auto = g === a.autoGrade;
  save();
  return { lastGrade: lastGradeInfo(id, e), card: cardSummary(e) };
}

function noteHint(id, shown) {
  const e = entry(id);
  const a = current(e, new Date().toISOString());
  if (a) a.hints = Math.max(a.hints, Math.min(20, Number(shown) || 0));
  save();
  return a ? a.hints : 0;
}

function markRevealed(id) {
  const e = entry(id);
  const now = new Date().toISOString();
  if (!e.revealedAt) e.revealedAt = now;
  const a = current(e, now);
  if (a) a.revealed = true;
  save();
}

/** Begin (or resume) a review: the editor goes back to the starter; your last passing code stays in History. */
function startReview(id) {
  const e = get(id);
  if (!e || !e.card) return { error: 'Solve this problem once before reviewing it.' };
  if (e.open && e.open.kind === 'review') return { resumed: true };
  e.open = { ...blankAttempt('review', new Date().toISOString()), draftBefore: e.draft, answersBefore: e.answers };
  delete e.draft;
  delete e.answers;
  save();
  return { resumed: false };
}

/** Abandon a review without grading it, and put your previous code back. */
function cancelReview(id) {
  const e = get(id);
  if (!e || !e.open || e.open.kind !== 'review') return { error: 'No review is in progress.' };
  if (e.open.draftBefore !== undefined) e.draft = e.open.draftBefore;
  if (e.open.answersBefore !== undefined) e.answers = e.open.answersBefore;
  e.open = null;
  save();
  return { ok: true };
}

function settings() {
  return load().settings;
}

function updateSettings(patch) {
  Object.assign(load().settings, patch);
  save();
  return load().settings;
}

function resetAll() {
  state = { v: VERSION, problems: {}, settings: load().settings };
  flush();
}

process.on('exit', () => { try { flush(); } catch (_) { /* ignore */ } });

module.exports = {
  get, all, summary, details, saveDraft, recordRun, regrade, noteHint, markRevealed, startReview, cancelReview,
  settings, updateSettings, resetAll, flush, FILE, _classify: classify,
};
