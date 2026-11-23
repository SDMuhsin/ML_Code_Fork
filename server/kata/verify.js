#!/usr/bin/env node
// Regression test for kata content. For every problem:
//   - the reference solution passes on CPU, and on CUDA when the problem allows it
//   - the starter runs, and fails cleanly (never "passed", never a crash in your-code phase)
//   - every lesson demo runs without error
//   - predict problems: the stated answers pass; blank answers fail
//
//   node kata/verify.js            everything
//   node kata/verify.js 1c 1d-2    only these modules / problems
//   node kata/verify.js --cpu      skip the CUDA runs
//   node kata/verify.js -v         print every check of every failing run

const { getCatalog } = require('./content');
const { WarmPool } = require('./pool');

const args = process.argv.slice(2);
const cpuOnly = args.includes('--cpu');
const verbose = args.includes('-v');
const filters = args.filter((a) => !a.startsWith('-'));

const POOL_SIZE = Number(process.env.KATA_VERIFY_POOL || 3);
const pool = new WarmPool({ size: POOL_SIZE }).start();

// Feed jobs to the pool no faster than it has workers. Queueing everything at
// once makes late jobs hit the pool's "waiting for a worker" timeout.
let active = 0;
const queue = [];
function limited(job, opts) {
  return new Promise((resolve) => {
    const go = () => {
      active++;
      pool.run(job, opts).then((r) => {
        active--;
        if (queue.length) queue.shift()();
        resolve(r);
      });
    };
    if (active < POOL_SIZE) go();
    else queue.push(go);
  });
}

function selected(p) {
  if (!filters.length) return true;
  return filters.some((f) => p.id === f || p.module === f);
}

function refUses(p, byId) {
  return (p.uses || []).map((uid) => {
    const hit = byId.get(uid);
    return { id: uid, title: uid, source: 'reference', reference: hit ? hit.problem.solution : '' };
  });
}

function summarize(r) {
  const bad = (r.checks || []).filter((c) => !c.passed);
  const lines = [];
  if (r.error) lines.push(`${r.error.phase || ''} ${r.error.type}: ${r.error.message}`.trim() + (r.error.line ? ` (line ${r.error.line})` : ''));
  for (const c of bad.slice(0, verbose ? 99 : 4)) lines.push(`x ${c.label}: ${(c.details || []).join(' | ')}`);
  if (!verbose && bad.length > 4) lines.push(`... and ${bad.length - 4} more failing checks`);
  if (verbose && r.error && r.error.traceback) lines.push(r.error.traceback);
  return lines;
}

async function main() {
  const { byId, order, issues } = getCatalog();
  // The progression index is checked first: it needs no Python and is the
  // usual mistake when a new lesson is added.
  if (issues.length) {
    console.log('FAIL  content/progression.md');
    for (const m of issues) console.log(`        ${m}`);
    console.log('\nFix the progression index first (see content/progression.md).');
    process.exit(1);
  }
  console.log(`ok    progression: ${order.length} problems on the path`);
  const ids = order.filter((id) => selected(byId.get(id).problem));
  if (!ids.length) {
    console.log('no problems match', filters);
    process.exit(1);
  }
  // Wait for one worker so we know whether CUDA is here.
  const probe = await pool.run({ mode: 'demo', shared: '', demo: '1', prior_demos: [] });
  const cuda = !cpuOnly && pool.env && pool.env.cuda_available;
  if (probe.status !== 'passed') {
    console.error('worker probe failed:', JSON.stringify(probe.error));
    process.exit(2);
  }
  console.log(`python ${pool.env.python} | torch ${pool.env.torch} | cuda ${cuda ? pool.env.device_name : 'skipped'}\n`);

  const problems = [];
  let failures = 0;

  const jobsFor = (p, m) => {
    const out = [];
    const devices = ['cpu', ...(cuda && p.devices.includes('cuda') ? ['cuda'] : [])];
    if (p.kind === 'predict') {
      const answers = Object.fromEntries(p.items.map((it) => [it.key, it.answer]));
      for (const d of devices) {
        out.push({ name: `answers/${d}`, expect: 'pass', job: { mode: 'predict', device: d, shared: m.shared, given: p.given, items: p.items, answers } });
      }
      out.push({ name: 'blank', expect: 'fail', job: { mode: 'predict', device: 'cpu', shared: m.shared, given: p.given, items: p.items, answers: {} } });
    } else {
      const base = { mode: 'run', shared: m.shared, uses: refUses(p, byId), given: p.given, checks: p.checks, starter: p.starter, inspect: p.inspect };
      for (const d of devices) out.push({ name: `solution/${d}`, expect: 'pass', job: { ...base, device: d, user: p.solution } });
      out.push({ name: 'starter', expect: 'fail', job: { ...base, device: 'cpu', user: p.starter } });
    }
    p.demos.forEach((demo, i) => {
      out.push({ name: `demo ${i}`, expect: 'pass', job: { mode: 'demo', device: 'cpu', shared: m.shared, prior_demos: p.demos.slice(0, i), demo } });
    });
    return out;
  };

  await Promise.all(ids.map(async (id) => {
    const { problem: p, module: m } = byId.get(id);
    const runs = jobsFor(p, m);
    const results = await Promise.all(runs.map((r) => limited(r.job, { timeoutMs: (p.timeout || 20) * 1000 + 10000 })));
    const notes = [];
    let ok = true;
    runs.forEach((r, i) => {
      const res = results[i];
      if (r.expect === 'pass' && res.status !== 'passed') {
        ok = false;
        notes.push(`  ${r.name}: expected PASS, got ${res.status}`, ...summarize(res).map((s) => '      ' + s));
      }
      if (r.expect === 'fail') {
        if (res.status === 'passed') {
          ok = false;
          notes.push(`  ${r.name}: expected the starter to FAIL, but it passed`);
        } else if (res.status === 'error') {
          ok = false;
          notes.push(`  ${r.name}: starter must fail cleanly, but it errored`, ...summarize(res).map((s) => '      ' + s));
        } else if (!(res.checks || []).length) {
          ok = false;
          notes.push(`  ${r.name}: starter produced no checks`);
        }
      }
      if (r.name.startsWith('solution/cpu') || r.name.startsWith('answers/cpu')) {
        p._nchecks = (res.checks || []).length;
      }
    });
    if (!ok) failures++;
    problems.push({ id, ok, notes, n: p._nchecks, title: p.title });
  }));

  problems.sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
  for (const r of problems) {
    console.log(`${r.ok ? 'ok  ' : 'FAIL'}  ${r.id.padEnd(6)} ${String(r.n ?? '?').padStart(3)} checks  ${r.title}`);
    for (const n of r.notes) console.log(n);
  }
  console.log(`\n${problems.length - failures}/${problems.length} problems verified${cuda ? ' (cpu + cuda)' : ' (cpu)'}`);
  pool.shutdown();
  process.exit(failures ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
