// Spaced repetition for kata problems: an SM-2 / Anki-style scheduler.
// Pure functions, no I/O. Times are epoch ms. Card fields are JSON-safe.
//
// A card exists once a problem has been solved. Grades:
//   1 Again  you needed the solution         -> back in 10 minutes ("retype it from memory")
//   2 Hard   solved, but with hints or several wrong runs
//   3 Good   solved cleanly
//   4 Easy   only when you pick it
// Days roll over at 4 am local time, so "due tomorrow" means tomorrow morning
// rather than exactly 24 hours later.

const MIN = 60 * 1000;
const DAY = 24 * 60 * MIN;
const ROLLOVER_HOUR = 4;
const RELEARN_MIN = 10;
const MAX_DAYS = 365;
const EASE0 = 2.5;
const EASE_MIN = 1.3;
const EASE_MAX = 3.5;
const GRADES = { 1: 'Again', 2: 'Hard', 3: 'Good', 4: 'Easy' };

/** Start of the study day (4 am) containing t. */
function dayStart(t) {
  const d = new Date(t);
  d.setHours(ROLLOVER_HOUR, 0, 0, 0);
  if (d.getTime() > t) d.setDate(d.getDate() - 1);
  return d.getTime();
}

/** 4 am, n study days after the day containing t (DST-safe: uses calendar days). */
function addDays(t, n) {
  const d = new Date(dayStart(t));
  d.setDate(d.getDate() + n);
  return d.getTime();
}

/** Whole study days between two instants. */
function daysBetween(a, b) {
  return Math.round((dayStart(b) - dayStart(a)) / DAY);
}

function newCard() {
  return { state: 'new', ease: EASE0, interval: 0, reps: 0, lapses: 0, due: null, last: null };
}

// Deterministic jitter (about ±5%) so problems solved in one sitting do not all fall due on the same day.
function fuzz(days, seed) {
  if (days < 3) return days;
  let h = 2166136261;
  for (const ch of seed) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  const spread = Math.max(1, Math.round(days * 0.05));
  return days + ((h >>> 0) % (2 * spread + 1)) - spread;
}

const clampEase = (e) => Math.min(EASE_MAX, Math.max(EASE_MIN, Math.round(e * 100) / 100));

/**
 * The card after grading an attempt finished at `now`. Never mutates `card`.
 * `id` only seeds the fuzz.
 */
function schedule(card, grade, now, id = '') {
  const c = { ...newCard(), ...(card || {}) };
  const g = Math.min(4, Math.max(1, Math.round(grade)));
  c.reps += 1;
  let days = null; // null = a minutes-long relearning step
  let floor = 1; // the jitter may not push the interval below this
  let exact = false; // no jitter at all

  if (c.state === 'new' || c.state === 'learning') {
    if (g === 1) c.state = 'learning';
    else {
      c.state = 'review';
      days = g === 4 ? 4 : 1;
      exact = true; // graduating intervals are exact: the buttons promise 1d / 4d
      if (g === 2) c.ease = clampEase(c.ease - 0.15);
      if (g === 4) c.ease = clampEase(c.ease + 0.15);
    }
  } else if (c.state === 'relearning') {
    if (g !== 1) {
      c.state = 'review';
      days = Math.max(1, c.interval);
    }
  } else {
    // review
    const ivl = Math.max(1, c.interval);
    const elapsed = c.last ? daysBetween(Date.parse(c.last), now) : ivl;
    if (g === 1) {
      c.lapses += 1;
      c.ease = clampEase(c.ease - 0.2);
      c.interval = Math.max(1, Math.round(ivl * 0.5)); // used when the relearning step passes
      c.state = 'relearning';
    } else if (elapsed < ivl) {
      // Reviewed early (practice): the interval should not jump ahead of the time actually waited.
      const base = Math.max(1, elapsed);
      const hard = Math.max(1, Math.round(Math.min(ivl, base * 1.2)));
      const good = Math.max(ivl, Math.round(base * c.ease));
      const easy = Math.max(good + 1, Math.round(base * c.ease * 1.3));
      days = g === 2 ? hard : g === 3 ? good : easy;
      if (g >= 3) floor = ivl;
    } else {
      const late = elapsed - ivl;
      const hard = Math.max(ivl + 1, Math.round(ivl * 1.2));
      const good = Math.max(hard + 1, Math.round((ivl + late / 2) * c.ease));
      const easy = Math.max(good + 1, Math.round((ivl + late) * c.ease * 1.3));
      days = g === 2 ? hard : g === 3 ? good : easy;
      floor = ivl + 1;
    }
    if (g === 2) c.ease = clampEase(c.ease - 0.15);
    if (g === 4) c.ease = clampEase(c.ease + 0.15);
  }

  if (days === null) {
    c.due = new Date(now + RELEARN_MIN * MIN).toISOString();
  } else {
    days = Math.min(MAX_DAYS, exact || days === 1 ? days : Math.max(floor, fuzz(days, `${id}:${c.reps}`)));
    c.interval = days;
    c.due = new Date(addDays(now, days)).toISOString();
  }
  c.last = new Date(now).toISOString();
  return c;
}

/** What each grade would do, for the Again/Hard/Good/Easy buttons. */
function preview(card, now, id = '') {
  const out = {};
  for (const g of [1, 2, 3, 4]) {
    const c = schedule(card, g, now, id);
    out[g] = { due: c.due, interval: c.interval, state: c.state };
  }
  return out;
}

/**
 * The grade an attempt earned, from what happened during it.
 * attempt: { revealed, hints, wrong }  (wrong = failed or erroring runs with no TODO left in the code)
 */
function autoGrade(a) {
  const wrong = a.wrong || 0;
  const hints = a.hints || 0;
  const bits = [];
  if (wrong) bits.push(`${wrong} wrong run${wrong > 1 ? 's' : ''}`);
  if (hints) bits.push(`${hints} hint${hints > 1 ? 's' : ''}`);
  if (a.revealed) return { grade: 1, reason: 'you opened the solution' };
  if (hints >= 1 || wrong >= 3) return { grade: 2, reason: bits.join(', ') };
  return { grade: 3, reason: bits.length ? bits.join(', ') : 'first try, no hints' };
}

function isDue(card, now) {
  return Boolean(card && card.due && Date.parse(card.due) <= now);
}

module.exports = { schedule, preview, autoGrade, isDue, newCard, dayStart, addDays, daysBetween, GRADES, DAY, MIN };
