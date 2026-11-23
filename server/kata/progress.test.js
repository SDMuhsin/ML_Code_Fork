// node kata/progress.test.js  (also run by `npm run kata:test`)
// Drives progress.js + study.js with fake run results; needs no Python.
// Uses a throwaway data folder, never server/data/kata-progress.json.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const dir = fs.mkdtempSync(path.join(process.env.KATA_TEST_TMP || path.join(__dirname, '..', 'data'), '.test-kata-progress-'));
process.env.KATA_DATA_DIR = dir;
const progress = require('./progress');
const study = require('./study');
const { getCatalog } = require('./content');

const cat = getCatalog();
const [p1, p2, p3] = cat.order;
const pass = { status: 'passed', checks: [{ passed: true }] };
const fail = { status: 'failed', checks: [{ passed: false }] };
const crash = { status: 'error', checks: [], error: { phase: 'worker', type: 'WorkerCrashed', message: 'x' } };
const run = (id, result, code = 'x = 1') => progress.recordRun(id, { code, device: 'cpu', result });
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok ', name); };

t('fresh: Next is the first problem on the path', () => {
  const nx = study.next(cat);
  assert.equal(nx.kind, 'new');
  assert.equal(nx.id, p1);
});

t('TODO runs and worker crashes are not wrong runs', () => {
  run(p1, fail, 'x = TODO');
  run(p1, crash);
  const d = progress.details(p1);
  assert.equal(d.stats.runs, 2);
  assert.equal(d.stats.wrong, 0);
  assert.equal(d.open.kind, 'learn');
  assert.equal(d.open.runs, 1);
  assert.equal(d.status, 'attempted');
});

t('solving with 3 wrong runs grades Hard, due tomorrow', () => {
  run(p1, fail); run(p1, fail); run(p1, fail);
  const { graded } = run(p1, pass);
  assert.equal(graded.grade, 2);
  assert.equal(graded.reason, '3 wrong runs');
  assert.equal(graded.interval, 1);
  const d = progress.details(p1);
  assert.equal(d.status, 'solved');
  assert.deepEqual([d.stats.firstSolve.runs, d.stats.firstSolve.wrong], [5, 3]);
  assert.equal(d.open, null);
  assert.equal(study.next(cat).id, p2); // not due yet, so the next new problem
});

t('overriding the grade to Easy pushes the review out to 4 days', () => {
  const r = progress.regrade(p1, 4);
  assert.equal(r.lastGrade.grade, 4);
  assert.equal(r.card.interval, 4);
  assert.equal(r.lastGrade.auto, false);
  assert.equal(progress.regrade(p1, 3).card.interval, 1); // and back
});

t('a run after solving (not a review) does not touch the card', () => {
  const before = progress.details(p1).card.due;
  const { graded } = run(p1, pass);
  assert.equal(graded, null);
  assert.equal(progress.details(p1).card.due, before);
});

t('opening the solution then passing grades Again: back in 10 minutes', () => {
  progress.noteHint(p2, 1);
  progress.markRevealed(p2);
  const { graded } = run(p2, pass);
  assert.equal(graded.grade, 1);
  assert.equal(graded.state, 'learning');
  assert.ok(Date.parse(graded.due) - Date.now() < 11 * 60 * 1000);
});

t('a due card comes before new problems, and a review resets the editor', () => {
  progress.get(p1).card.due = new Date(Date.now() - 1000).toISOString();
  const nx = study.next(cat);
  assert.equal(nx.kind, 'review');
  assert.equal(nx.id, p1);
  progress.saveDraft(p1, { code: 'my passing code' });
  assert.equal(progress.startReview(p1).resumed, false);
  assert.equal(progress.details(p1).draft, undefined);
  assert.equal(progress.startReview(p1).resumed, true); // idempotent
  assert.equal(study.next(cat).resumed, true);
});

t('cancelling a review restores the code and leaves it due', () => {
  progress.cancelReview(p1);
  assert.equal(progress.details(p1).draft, 'my passing code');
  assert.equal(study.next(cat).id, p1);
});

t('a clean review grades Good and grows the interval', () => {
  progress.startReview(p1);
  // pretend it was last seen 2 days ago and due yesterday
  const c = progress.get(p1).card;
  c.last = new Date(Date.now() - 2 * 864e5).toISOString();
  progress.get(p1).log.at(-1).endedAt = c.last;
  const { graded } = run(p1, pass);
  assert.equal(graded.kind, 'review');
  assert.equal(graded.grade, 3);
  assert.ok(graded.interval >= 2, graded.interval);
  const o = study.overview(cat);
  assert.equal(o.totals.reviews, 1);
  assert.equal(o.totals.retention, 1);
  assert.equal(o.today.reviews, 1);
});

t('in a review, the solution is gated again', () => {
  progress.get(p2).card.due = new Date(Date.now() - 1000).toISOString();
  progress.startReview(p2);
  assert.equal(progress.details(p2).revealed, false);
  progress.markRevealed(p2);
  assert.equal(progress.details(p2).revealed, true);
  progress.cancelReview(p2);
});

t('tree: module states and due counts', () => {
  const o = study.overview(cat);
  const first = o.tree.nodes[0];
  assert.equal(first.id, cat.progression[0].id);
  assert.ok(['active', 'done'].includes(first.status));
  assert.equal(o.tree.nodes.at(-1).status, 'locked');
  assert.ok(o.tree.nodes.some((x) => x.due > 0));
});

t('a v1 progress file migrates: attempts -> runs, solved gets a card', () => {
  const s = progress.all();
  s[p3] = { status: 'solved', attempts: 7, solvedAt: new Date(Date.now() - 5 * 864e5).toISOString(), history: [] };
  progress.flush();
  const raw = JSON.parse(fs.readFileSync(progress.FILE, 'utf8'));
  raw.v = 1;
  fs.writeFileSync(progress.FILE, JSON.stringify(raw));
  delete require.cache[require.resolve('./progress')];
  const fresh = require('./progress');
  const d = fresh.details(p3);
  assert.equal(d.runs, 7);
  assert.equal(d.card.state, 'review');
  assert.ok(Date.parse(d.card.due) < Date.now()); // was due 4 days ago
});

// progress.js flushes on exit; this handler is registered after it, so it runs last.
process.on('exit', () => fs.rmSync(dir, { recursive: true, force: true }));
console.log(`\n${n} progress/study tests passed`);
process.exit(0);
