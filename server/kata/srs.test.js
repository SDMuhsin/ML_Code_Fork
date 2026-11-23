// node kata/srs.test.js  (also run by `npm run kata:test`)
const assert = require('assert');
const srs = require('./srs');

const T0 = new Date(2026, 10, 3, 21, 0).getTime(); // Nov 3, 9 pm local
const at = (days, hour = 21) => new Date(2026, 10, 3 + days, hour, 0).getTime();
const day = (iso) => srs.daysBetween(T0, Date.parse(iso));
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok ', name); };

t('first clean solve is due tomorrow at 4 am', () => {
  const c = srs.schedule(null, 3, T0, 'x');
  assert.equal(c.state, 'review');
  assert.equal(c.interval, 1);
  assert.equal(new Date(c.due).getHours(), 4);
  assert.equal(day(c.due), 1);
});

t('11:30 pm and 1 am the same night are the same study day', () => {
  const late = new Date(2026, 10, 3, 23, 30).getTime();
  const after = new Date(2026, 10, 4, 1, 0).getTime();
  assert.equal(srs.dayStart(late), srs.dayStart(after));
});

t('Again on first solve comes back in 10 minutes, then graduates to 1 day', () => {
  let c = srs.schedule(null, 1, T0, 'x');
  assert.equal(c.state, 'learning');
  assert.equal(Date.parse(c.due) - T0, 10 * srs.MIN);
  assert.ok(!srs.isDue(c, T0 + 5 * srs.MIN) && srs.isDue(c, T0 + 11 * srs.MIN));
  c = srs.schedule(c, 3, T0 + 15 * srs.MIN, 'x');
  assert.equal(c.state, 'review');
  assert.equal(c.interval, 1);
});

t('Good, on time, grows roughly 1 -> 3 -> 7 -> 18 days', () => {
  let c = srs.schedule(null, 3, T0, 'grow');
  const ivls = [c.interval];
  let now = T0;
  for (let i = 0; i < 3; i++) {
    now = Date.parse(c.due) + 10 * 3600 * 1000; // reviewed at 2 pm on the due day
    c = srs.schedule(c, 3, now, 'grow');
    ivls.push(c.interval);
  }
  assert.ok(ivls[1] >= 2 && ivls[1] <= 4, ivls);
  assert.ok(ivls[2] >= 6 && ivls[2] <= 9, ivls);
  assert.ok(ivls[3] >= 15 && ivls[3] <= 24, ivls);
});

t('grade order: Again < Hard < Good < Easy for a review card', () => {
  const c = { ...srs.newCard(), state: 'review', interval: 6, last: new Date(T0).toISOString(), due: new Date(at(6, 4)).toISOString() };
  const p = srs.preview(c, at(6, 12), 'p');
  assert.equal(p[1].state, 'relearning');
  assert.ok(p[2].interval < p[3].interval && p[3].interval < p[4].interval, JSON.stringify(p));
  assert.ok(p[2].interval > 6);
});

t('a lapse halves the interval, lowers ease, and relearns in 10 min', () => {
  const c0 = { ...srs.newCard(), state: 'review', interval: 20, last: new Date(T0).toISOString() };
  const c = srs.schedule(c0, 1, at(20), 'l');
  assert.equal(c.lapses, 1);
  assert.equal(c.ease, 2.3);
  assert.equal(c.state, 'relearning');
  const back = srs.schedule(c, 3, at(20) + 20 * srs.MIN, 'l');
  assert.equal(back.state, 'review');
  assert.equal(back.interval, 10);
});

t('practising early never shrinks the interval on Good', () => {
  const c0 = { ...srs.newCard(), state: 'review', interval: 10, last: new Date(T0).toISOString() };
  const c = srs.schedule(c0, 3, at(0, 22), 'e'); // same evening
  assert.ok(c.interval >= 10, c.interval);
});

t('ease never drops below 1.3', () => {
  let c = { ...srs.newCard(), state: 'review', interval: 2, ease: 1.35, last: new Date(T0).toISOString() };
  c = srs.schedule(c, 1, at(2), 'm');
  assert.equal(c.ease, 1.3);
});

t('fuzz spreads problems solved together', () => {
  const ivls = new Set();
  for (const id of ['1a-1', '1a-2', '1b-1', '1b-2', '1c-1', '1c-2']) {
    const c = { ...srs.newCard(), state: 'review', interval: 20, last: new Date(T0).toISOString() };
    ivls.add(srs.schedule(c, 3, at(20), id).interval);
  }
  assert.ok(ivls.size > 1, [...ivls]);
});

t('Easy on a first solve is exactly 4 days for every problem id', () => {
  for (const id of ['1a-1', '1a-2', '1b-1', '1c-3', '1n-9']) assert.equal(srs.schedule(null, 4, T0, id).interval, 4, id);
});

t('autoGrade', () => {
  assert.equal(srs.autoGrade({ revealed: true, wrong: 0 }).grade, 1);
  assert.equal(srs.autoGrade({ hints: 1 }).grade, 2);
  assert.equal(srs.autoGrade({ wrong: 3 }).grade, 2);
  assert.equal(srs.autoGrade({ wrong: 2 }).grade, 3);
  assert.equal(srs.autoGrade({}).reason, 'first try, no hints');
  assert.equal(srs.autoGrade({ wrong: 1, hints: 1 }).reason, '1 wrong run, 1 hint');
});

console.log(`\n${n} scheduler tests passed`);
