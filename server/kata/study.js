// "What should I do next?": the study queue, stats, and the roadmap tree's data.
//
// Next, in priority order:
//   1. a review you already started            (kind "review", resumed)
//   2. a due card: 10-minute retypes first, then the most overdue
//   3. the next unsolved problem on the path whose module's prerequisites are done
//   4. any unsolved problem (you skipped ahead)
//   5. nothing: everything is solved and nothing is due ("done", with when the next card is due)

const srs = require('./srs');
const progress = require('./progress');
const { MAIN, OFF } = require('./content');

// ---- tracks ------------------------------------------------------------------
// Each track (progression.md "# Track: x") is its own path, tree, Next queue and
// due count. study.next/overview see one track at a time through a view: the
// catalog with order/progression cut down to that track. "off" is parked: it is
// never offered by Next and its cards are never counted as due.

/** The track to study: a listed, non-parked track, else main. */
function trackId(catalog, track) {
  const t = String(track || MAIN).toLowerCase();
  return t !== OFF && (catalog.tracks || []).some((x) => x.id === t) ? t : MAIN;
}

function view(catalog, track) {
  if (!catalog.tracks) return catalog; // a hand-made catalog without tracks
  const id = trackId(catalog, track);
  const t = catalog.tracks.find((x) => x.id === id) || { id, title: id, order: [] };
  return {
    ...catalog,
    track: { id: t.id, title: t.title },
    order: t.order,
    progression: catalog.progression.filter((s) => s.track === id),
  };
}

/** The home page's switcher: every track that has modules (main always), except "off". */
function tracks(catalog, now = Date.now()) {
  return (catalog.tracks || []).filter((t) => t.id !== OFF && (t.id === MAIN || t.modules.length)).map((t) => {
    let solved = 0;
    let due = 0;
    for (const id of t.order) {
      const s = progress.summary(id);
      if (s.status === 'solved') solved++;
      if (s.card && (s.reviewing || srs.isDue(s.card, now))) due++;
    }
    return { id: t.id, title: t.title, modules: t.modules.length, total: t.order.length, solved, due };
  });
}

function moduleState(catalog) {
  const out = new Map();
  for (const step of catalog.progression) {
    const ids = catalog.order.filter((id) => catalog.byId.get(id).module.id === step.id);
    const solved = ids.filter((id) => progress.summary(id).status === 'solved').length;
    out.set(step.id, { ids, solved, done: ids.length > 0 && solved === ids.length });
  }
  return out;
}

function dueList(catalog, now) {
  const due = [];
  for (const id of catalog.order) {
    const s = progress.summary(id);
    if (!s.card) continue;
    if (s.reviewing || srs.isDue(s.card, now)) {
      const t = Date.parse(s.card.due);
      const short = s.card.state === 'learning' || s.card.state === 'relearning';
      // Overdue relative to the interval, so a 2-day card a day late beats a 30-day card a day late.
      const overdue = (now - t) / (Math.max(1, s.card.interval) * srs.DAY);
      due.push({ id, reviewing: s.reviewing, short, overdue, due: s.card.due, interval: s.card.interval, state: s.card.state });
    }
  }
  due.sort((a, b) => (b.reviewing - a.reviewing) || (b.short - a.short) || (b.overdue - a.overdue)
    || (catalog.order.indexOf(a.id) - catalog.order.indexOf(b.id)));
  return due;
}

function nextNew(catalog, mods) {
  const ready = (moduleId) => {
    const step = catalog.progression.find((s) => s.id === moduleId);
    return step.after.every((a) => mods.get(a).done);
  };
  let fallback = null;
  for (const id of catalog.order) {
    if (progress.summary(id).status === 'solved') continue;
    if (ready(catalog.byId.get(id).module.id)) return { id, skippedAhead: false };
    if (!fallback) fallback = id;
  }
  return fallback ? { id: fallback, skippedAhead: true } : null;
}

function describe(catalog, id) {
  const hit = catalog.byId.get(id);
  return { id, title: hit.problem.title, module: hit.module.id, moduleTitle: hit.module.title, problemKind: hit.problem.kind };
}

function next(catalog, now = Date.now(), track = MAIN) {
  catalog = view(catalog, track);
  const mods = moduleState(catalog);
  const due = dueList(catalog, now);
  if (due.length) {
    const d = due[0];
    const s = progress.summary(d.id);
    return { kind: 'review', track: catalog.track && catalog.track.id, ...describe(catalog, d.id), resumed: d.reviewing, card: s.card, lastSeen: s.card.last, dueCount: due.length };
  }
  const n = nextNew(catalog, mods);
  if (n) {
    const s = progress.summary(n.id);
    return { kind: 'new', track: catalog.track && catalog.track.id, ...describe(catalog, n.id), started: s.status === 'attempted', skippedAhead: n.skippedAhead };
  }
  const upcoming = catalog.order.map((id) => progress.summary(id).card).filter(Boolean).map((c) => c.due).sort()[0] || null;
  return { kind: 'done', nextDue: upcoming, track: catalog.track && catalog.track.id };
}

function overview(catalog, now = Date.now(), track = MAIN) {
  const all = catalog;
  catalog = view(catalog, track);
  const mods = moduleState(catalog);
  const due = dueList(catalog, now);
  const today = srs.dayStart(now);
  const tomorrow = srs.addDays(now, 1);
  const week = srs.addDays(now, 7);

  let solved = 0;
  let cards = 0;
  let mature = 0; // interval of 3 weeks or more: Anki's "mature"
  let dueTomorrow = 0;
  let dueWeek = 0;
  let laterToday = null; // next short relearning step not yet due
  const totals = { runs: 0, wrong: 0, passes: 0, reviews: 0, recalled: 0, firstTry: 0, learned: 0 };
  const todayStats = { reviews: 0, learned: 0, runs: 0 };
  const trouble = [];

  for (const id of catalog.order) {
    const e = progress.get(id);
    if (!e) continue;
    if (e.status === 'solved') solved++;
    totals.runs += e.runs || 0;
    totals.wrong += e.wrong || 0;
    totals.passes += e.passes || 0;
    for (const h of e.history || []) if (Date.parse(h.at) >= today) todayStats.runs++;
    for (const a of e.log || []) {
      const ended = Date.parse(a.endedAt);
      if (a.kind === 'review') {
        totals.reviews++;
        if (a.grade > 1) totals.recalled++;
        if (ended >= today) todayStats.reviews++;
      } else if (!a.migrated) {
        totals.learned++;
        if (a.wrong === 0 && !a.hints && !a.revealed) totals.firstTry++;
        if (ended >= today) todayStats.learned++;
      }
    }
    const c = e.card;
    if (c) {
      cards++;
      if (c.interval >= 21) mature++;
      const t = Date.parse(c.due);
      if (t > now && t < tomorrow) {
        if (c.state === 'learning' || c.state === 'relearning') laterToday = laterToday && laterToday < c.due ? laterToday : c.due;
      } else if (t >= tomorrow && t < srs.addDays(now, 2)) dueTomorrow++;
      if (t > now && t < week) dueWeek++;
    }
    // Trouble spots: problems that took many wrong runs or were forgotten.
    const score = (c ? c.lapses * 3 : 0) + (e.wrong || 0) * 0.5 + (e.log || []).filter((a) => a.revealed).length * 2;
    if (score >= 2) trouble.push({ ...describe(catalog, id), wrong: e.wrong || 0, lapses: c ? c.lapses : 0, score });
  }
  trouble.sort((a, b) => b.score - a.score);

  return {
    now: new Date(now).toISOString(),
    track: catalog.track ? catalog.track.id : MAIN,
    tracks: tracks(all, now),
    next: next(all, now, track),
    due: due.map((d) => ({ ...d, ...describe(catalog, d.id) })),
    counts: { total: catalog.order.length, solved, cards, mature, due: due.length, dueTomorrow, dueWeek, laterToday },
    today: todayStats,
    totals: {
      ...totals,
      retention: totals.reviews ? totals.recalled / totals.reviews : null,
      firstTryRate: totals.learned ? totals.firstTry / totals.learned : null,
    },
    trouble: trouble.slice(0, 6),
    tree: tree(catalog, mods, now),
    issues: catalog.issues,
  };
}

/** Nodes and edges of the roadmap. Layout is the client's job; depth is given. */
function tree(catalog, mods, now) {
  const nodes = catalog.progression.map((step, index) => {
    const ms = mods.get(step.id);
    const hit = catalog.byId.get(ms.ids[0]);
    const m = hit ? hit.module : { title: step.id, minutes: null };
    let dueHere = 0;
    let attempted = 0;
    for (const id of ms.ids) {
      const s = progress.summary(id);
      if (s.card && (srs.isDue(s.card, now) || s.reviewing)) dueHere++;
      if (s.status === 'attempted') attempted++;
    }
    const ready = step.after.every((a) => mods.get(a).done);
    const status = ms.done ? 'done' : ms.solved || attempted ? 'active' : ready ? 'ready' : 'locked';
    return {
      id: step.id, index, title: m.title, minutes: m.minutes, lesson: step.lesson,
      lessonTitle: hit ? hit.lesson.title : step.lesson, depth: step.depth, after: step.after,
      total: ms.ids.length, solved: ms.solved, due: dueHere, status,
    };
  });
  return { nodes, lessons: [...new Set(nodes.map((n) => n.lesson))] };
}

module.exports = { next, overview, tracks, trackId };
