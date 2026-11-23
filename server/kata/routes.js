// /api/kata: the kernel-practice problems as LeetCode-style exercises.

const express = require('express');
const { getCatalog } = require('./content');
const { getPool } = require('./pool');
const progress = require('./progress');
const study = require('./study');

const router = express.Router();
const pool = getPool();

function find(id) {
  return getCatalog().byId.get(id) || null;
}

function pickDevice(p, requested) {
  const env = pool.status().env;
  const want = requested === 'cuda' ? 'cuda' : 'cpu';
  if (want === 'cuda' && env && !env.cuda_available) return 'cpu';
  if (!p.devices.includes(want)) return p.devices[0] || 'cpu';
  return want;
}

/** Earlier problems this one builds on: your last passing code, else the reference. */
function resolveUses(p) {
  return (p.uses || []).map((uid) => {
    const hit = find(uid);
    if (!hit) return { id: uid, title: uid, source: 'missing', reference: '' };
    const e = progress.get(uid);
    const mine = e && e.lastPassingCode;
    return {
      id: uid,
      title: hit.problem.title,
      source: mine ? 'yours' : 'reference',
      code: mine || '',
      reference: hit.problem.solution,
    };
  });
}

function publicProblem(hit) {
  const { problem: p, module: m, lesson: l } = hit;
  const cat = getCatalog();
  // Position (prev/next) is within the problem's own track.
  const t = (cat.tracks || []).find((x) => x.id === hit.track);
  const order = t ? t.order : cat.order;
  const i = order.indexOf(p.id);
  return {
    id: p.id,
    n: p.n,
    title: p.title,
    difficulty: p.difficulty,
    kind: p.kind,
    devices: p.devices,
    lesson: p.lesson,
    demos: p.demos,
    task: p.task,
    given: p.given,
    starter: p.starter,
    hints: p.hints,
    takeaway: p.takeaway,
    hasSolution: Boolean(p.solution || p.items.length),
    items: p.items.map(({ key, expr, kind }) => ({ key, expr, kind: kind || 'shape' })),
    uses: resolveUses(p).map(({ id, title, source }) => ({ id, title, source })),
    module: { id: m.id, title: m.title, minutes: m.minutes, intro: m.intro, shared: m.shared,
      problemIds: m.problems.map((x) => x.id) },
    lesson_info: { id: l.id, title: l.title },
    track: hit.track,
    studyTrack: study.trackId(cat, hit.track), // the track Next and due counts use ("off" -> main)
    position: { index: i, total: order.length, prev: order[i - 1] || null, next: order[i + 1] || null },
    progress: progress.details(p.id),
  };
}

router.get('/env', (req, res) => {
  res.json(pool.status());
});

router.get('/catalog', (req, res) => {
  const cat = getCatalog();
  const rank = new Map(cat.progression.map((s, i) => [s.id, i]));
  const stepOf = (id) => cat.progression.find((s) => s.id === id) || { after: [], track: 'main' };
  const lessons = cat.lessons.map((l) => ({ ...l, modules: [...l.modules].sort((a, b) => rank.get(a.id) - rank.get(b.id)) }));
  let total = 0;
  let solved = 0;
  let next = null;
  const out = lessons.map((l) => ({
    id: l.id, title: l.title, intro: l.intro, recall: l.recall, traps: l.traps,
    modules: l.modules.map((m) => ({
      id: m.id, title: m.title, minutes: m.minutes, intro: m.intro,
      after: stepOf(m.id).after,
      track: stepOf(m.id).track,
      problems: m.problems.map((p) => {
        const s = progress.summary(p.id);
        total++;
        if (s.status === 'solved') solved++;
        else if (!next) next = p.id;
        return { id: p.id, n: p.n, title: p.title, difficulty: p.difficulty, kind: p.kind, ...s };
      }),
    })),
  }));
  res.json({ lessons: out, total, solved, next, study: study.next(cat), tracks: study.tracks(cat), issues: cat.issues });
});

router.get('/problems/:id', (req, res) => {
  const hit = find(req.params.id);
  if (!hit) return res.status(404).json({ error: `No kata problem ${req.params.id}` });
  res.json(publicProblem(hit));
});

router.get('/problems/:id/solution', (req, res) => {
  const hit = find(req.params.id);
  if (!hit) return res.status(404).json({ error: 'not found' });
  progress.markRevealed(hit.problem.id);
  const p = hit.problem;
  res.json({ solution: p.solution, items: p.items.map(({ key, answer }) => ({ key, answer })) });
});

router.put('/problems/:id/draft', (req, res) => {
  const hit = find(req.params.id);
  if (!hit) return res.status(404).json({ error: 'not found' });
  progress.saveDraft(hit.problem.id, req.body || {});
  res.json({ ok: true });
});

router.post('/problems/:id/run', async (req, res) => {
  const hit = find(req.params.id);
  if (!hit) return res.status(404).json({ error: 'not found' });
  const { problem: p, module: m } = hit;
  const device = pickDevice(p, req.body.device);
  const code = typeof req.body.code === 'string' ? req.body.code : '';
  let answers = req.body.answers || null;
  if (p.kind === 'predict') {
    // Every item present, so a blank one counts as "not filled in" rather than wrong.
    const a = answers && typeof answers === 'object' ? answers : {};
    answers = Object.fromEntries(p.items.map((it) => [it.key, typeof a[it.key] === 'string' ? a[it.key] : '']));
  }

  const job = p.kind === 'predict'
    ? { mode: 'predict', device, shared: m.shared, given: p.given, items: p.items, answers: answers || {} }
    : { mode: 'run', device, shared: m.shared, uses: resolveUses(p), given: p.given, user: code,
      checks: p.checks, starter: p.starter, inspect: p.inspect };

  const result = await pool.run(job, { timeoutMs: p.timeout * 1000 });
  const { graded } = progress.recordRun(p.id, { code: p.kind === 'predict' ? undefined : code, answers, device, result });
  res.json({ result, graded, progress: progress.details(p.id), next: study.next(getCatalog(), Date.now(), hit.track) });
});

router.post('/problems/:id/demo', async (req, res) => {
  const hit = find(req.params.id);
  if (!hit) return res.status(404).json({ error: 'not found' });
  const { problem: p, module: m } = hit;
  const idx = Number(req.body.index);
  if (!(idx >= 0 && idx < p.demos.length)) return res.status(400).json({ error: 'bad demo index' });
  const demo = typeof req.body.code === 'string' ? req.body.code : p.demos[idx];
  const result = await pool.run({
    mode: 'demo', device: pickDevice(p, req.body.device), shared: m.shared,
    prior_demos: p.demos.slice(0, idx), demo,
  }, { timeoutMs: 20000 });
  res.json({ result });
});

router.post('/problems/:id/scratch', async (req, res) => {
  const hit = find(req.params.id);
  if (!hit) return res.status(404).json({ error: 'not found' });
  const { problem: p, module: m } = hit;
  const result = await pool.run({
    mode: 'scratch', device: pickDevice(p, req.body.device), shared: m.shared, uses: resolveUses(p),
    given: p.given, user: p.kind === 'predict' ? '' : (req.body.code || ''), scratch: req.body.scratch || '',
  }, { timeoutMs: 20000 });
  res.json({ result });
});

// ---- spaced repetition and the study queue --------------------------------

router.get('/study', (req, res) => res.json(study.overview(getCatalog(), Date.now(), req.query.track)));
router.get('/study/next', (req, res) => res.json(study.next(getCatalog(), Date.now(), req.query.track)));

/** Start (or resume) a review: the editor goes back to the starter. */
router.post('/problems/:id/review', (req, res) => {
  const hit = find(req.params.id);
  if (!hit) return res.status(404).json({ error: 'not found' });
  const r = progress.startReview(hit.problem.id);
  if (r.error) return res.status(400).json(r);
  res.json({ ...r, problem: publicProblem(hit) });
});

router.delete('/problems/:id/review', (req, res) => {
  const hit = find(req.params.id);
  if (!hit) return res.status(404).json({ error: 'not found' });
  const r = progress.cancelReview(hit.problem.id);
  if (r.error) return res.status(400).json(r);
  res.json({ ok: true, problem: publicProblem(hit) });
});

/** Override the automatic grade of the attempt that just passed. */
router.post('/problems/:id/grade', (req, res) => {
  const hit = find(req.params.id);
  if (!hit) return res.status(404).json({ error: 'not found' });
  const r = progress.regrade(hit.problem.id, req.body && req.body.grade);
  if (r.error) return res.status(400).json(r);
  res.json({ ...r, progress: progress.details(hit.problem.id) });
});

router.post('/problems/:id/hint', (req, res) => {
  const hit = find(req.params.id);
  if (!hit) return res.status(404).json({ error: 'not found' });
  res.json({ hints: progress.noteHint(hit.problem.id, req.body && req.body.shown) });
});

router.get('/settings', (req, res) => res.json(progress.settings()));
router.put('/settings', (req, res) => res.json(progress.updateSettings(req.body || {})));

module.exports = router;
