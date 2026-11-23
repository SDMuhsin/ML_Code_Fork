// node kata/tracks.test.js  (also run by `npm run kata:test`)
// Tracks in progression.md: parsing, per-track Next and due counts, the parked "off" track.
// Uses a throwaway content folder and data folder; never the real content or progress.
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const base = process.env.KATA_TEST_TMP || path.join(__dirname, '..', 'data');
const root = fs.mkdtempSync(path.join(base, '.test-kata-tracks-'));
const contentDir = path.join(root, 'content');
process.env.KATA_CONTENT_DIR = contentDir;
process.env.KATA_DATA_DIR = path.join(root, 'data');

const write = (rel, text) => {
  const f = path.join(contentDir, rel);
  fs.mkdirSync(path.dirname(f), { recursive: true });
  fs.writeFileSync(f, text);
};
const lessonOf = { xa: '01_a', xb: '01_a', ya: '02_b', yb: '02_b', za: '03_c' };
for (const [mod, lesson] of Object.entries(lessonOf)) {
  write(`${lesson}/_lesson.md`, `---\ntitle: Lesson ${lesson}\n---\n`);
  write(`${lesson}/${mod}_m/_module.md`, `---\nid: ${mod}\ntitle: Module ${mod}\nminutes: 5\n---\n`);
  for (const n of [1, 2]) write(`${lesson}/${mod}_m/0${n}_p.md`, `---\ntitle: ${mod} problem ${n}\n---\n# Task\nx\n`);
}
write('progression.md', `# Progression

- a prose bullet above the first heading is ignored

## 01_a
- xa  first     after: none
- xb  second

# Track: side  (title: A side track)

## 02_b
- ya  side one  after: none
- yb  side two  after: ya, xb

# Track: off
- za  parked

# Track: empty  (title: Nothing yet)
`);

const progress = require('./progress');
const study = require('./study');
const { getCatalog, _internals } = require('./content');
const cat = getCatalog();
const pass = { status: 'passed', checks: [{ passed: true }] };
const solve = (id) => progress.recordRun(id, { code: 'x = 1', device: 'cpu', result: pass });
const makeDue = (id) => { progress.get(id).card.due = new Date(Date.now() - 1000).toISOString(); };
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok ', name); };

t('parse: track headings, titles, and modules per track', () => {
  assert.deepEqual(cat.tracks.map((x) => [x.id, x.title, x.modules]), [
    ['main', 'Main path', ['xa', 'xb']],
    ['side', 'A side track', ['ya', 'yb']],
    ['off', 'Parked', ['za']],
    ['empty', 'Nothing yet', []],
  ]);
  assert.deepEqual(cat.tracks[1].order, ['ya-1', 'ya-2', 'yb-1', 'yb-2']);
  assert.equal(cat.byId.get('ya-1').track, 'side');
  // depth is per track: ya is a root of its own tree
  assert.equal(cat.progression.find((s) => s.id === 'ya').depth, 0);
  assert.equal(cat.progression.find((s) => s.id === 'yb').depth, 1);
});

t('an old file without track headings is all main', () => {
  const { steps, tracks } = _internals.parseProgression('# Progression\n\n- prose\n\n## 01_a\n- xa after: none\n- xb\n');
  assert.deepEqual(tracks.map((x) => x.id), ['main']);
  assert.deepEqual(steps.map((s) => [s.id, s.track]), [['xa', 'main'], ['xb', 'main']]);
});

t('an after: that crosses tracks is an issue and is dropped', () => {
  assert.ok(cat.issues.some((m) => m.includes('yb') && m.includes('xb') && m.includes('track')), cat.issues.join('\n'));
  assert.deepEqual(cat.progression.find((s) => s.id === 'yb').after, ['ya']);
  assert.equal(cat.issues.length, 1, cat.issues.join('\n')); // the empty track is not an issue
});

t('next() per track never returns another track\'s problem', () => {
  assert.equal(study.next(cat, Date.now(), 'main').id, 'xa-1');
  assert.equal(study.next(cat, Date.now(), 'side').id, 'ya-1');
  assert.equal(study.next(cat).id, 'xa-1'); // default = main
  for (const id of ['xa-1', 'xa-2', 'xb-1', 'xb-2']) solve(id);
  const nx = study.next(cat, Date.now(), 'main');
  assert.equal(nx.kind, 'done'); // main is finished; it does not spill into side or off
  assert.equal(study.next(cat, Date.now(), 'side').id, 'ya-1');
});

t('due counts are per track', () => {
  solve('ya-1');
  makeDue('xa-1');
  makeDue('xb-1');
  makeDue('ya-1');
  const main = study.overview(cat, Date.now(), 'main');
  const side = study.overview(cat, Date.now(), 'side');
  assert.equal(main.counts.due, 2);
  assert.equal(side.counts.due, 1);
  assert.equal(main.counts.total, 4);
  assert.deepEqual(side.tree.nodes.map((x) => x.id), ['ya', 'yb']);
  assert.equal(study.next(cat, Date.now(), 'side').id, 'ya-1');
  assert.equal(study.next(cat, Date.now(), 'side').kind, 'review');
  const list = Object.fromEntries(study.tracks(cat).map((x) => [x.id, [x.solved, x.total, x.due]]));
  assert.deepEqual(list, { main: [4, 4, 2], side: [1, 4, 1] });
});

t('the off track is hidden, its cards are never due, and asking for it gives main', () => {
  solve('za-1');
  makeDue('za-1');
  assert.ok(!study.tracks(cat).some((x) => x.id === 'off' || x.id === 'empty'));
  assert.equal(study.trackId(cat, 'off'), 'main');
  assert.equal(study.trackId(cat, 'nope'), 'main');
  assert.notEqual(study.next(cat, Date.now(), 'off').id, 'za-1');
  for (const tr of ['main', 'side', 'off']) {
    const o = study.overview(cat, Date.now(), tr);
    assert.ok(!o.due.some((d) => d.id.startsWith('za')), tr);
    assert.ok(!o.tree.nodes.some((x) => x.id === 'za'), tr);
  }
  assert.ok(cat.byId.has('za-1')); // still openable by URL
  assert.equal(progress.details('za-1').status, 'solved'); // and its progress is kept
});

process.on('exit', () => fs.rmSync(root, { recursive: true, force: true }));
console.log(`\n${n} track tests passed`);
process.exit(0);
