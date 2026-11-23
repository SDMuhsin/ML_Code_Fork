// Loads kata problems from plain Markdown files. See content/README.md for the format.
//
//   content/<lesson>/_lesson.md              lesson title, recall list, traps
//   content/<lesson>/<module>/_module.md     module id, title, intro, shared setup
//   content/<lesson>/<module>/NN_slug.md     one problem; id = <module id>-<NN>
//
// Files are re-read when they change, so editing a problem shows up on reload.

const fs = require('fs');
const path = require('path');

const CONTENT_DIR = process.env.KATA_CONTENT_DIR || path.join(__dirname, 'content');
const SECTIONS = new Set(['Intro', 'Shared', 'Lesson', 'Task', 'Given', 'Starter', 'Solution',
  'Checks', 'Hints', 'Takeaway', 'Items', 'Recall', 'Traps']);

function parseFrontmatter(text) {
  const meta = {};
  const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!m) return { meta, body: text };
  for (const raw of m[1].split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const i = line.indexOf(':');
    if (i < 0) continue;
    const key = line.slice(0, i).trim();
    let val = line.slice(i + 1).replace(/\s+#.*$/, '').trim();
    if (val.startsWith('[') && val.endsWith(']')) {
      val = val.slice(1, -1).split(',').map((s) => s.trim().replace(/^["']|["']$/g, '')).filter(Boolean);
    } else if (/^-?\d+(\.\d+)?$/.test(val)) {
      val = Number(val);
    } else if (val === 'true' || val === 'false') {
      val = val === 'true';
    } else {
      val = val.replace(/^["']|["']$/g, '');
    }
    meta[key] = val;
  }
  return { meta, body: text.slice(m[0].length) };
}

/** Split a body into `# Section` blocks, ignoring headings inside code fences. */
function splitSections(body) {
  const out = {};
  let cur = null;
  let fence = null;
  for (const line of body.split(/\r?\n/)) {
    const f = line.match(/^\s*(```+|~~~+)/);
    if (f) {
      if (!fence) fence = f[1];
      else if (line.trim().startsWith(fence) && line.trim().replace(/[`~]/g, '') === '') fence = null;
    }
    const h = !fence && !f && line.match(/^# (\w+)\s*$/);
    if (h && SECTIONS.has(h[1])) {
      cur = h[1];
      out[cur] = [];
      continue;
    }
    if (cur) out[cur].push(line);
  }
  for (const k of Object.keys(out)) out[k] = out[k].join('\n').replace(/^\n+|\s+$/g, '');
  return out;
}

/** The contents of the first fenced block, or the whole text if there is none. */
function codeOf(text) {
  if (!text) return '';
  const m = text.match(/^(```+|~~~+)[^\n]*\n([\s\S]*?)\n\1\s*$/m);
  return (m ? m[2] : text).replace(/\s+$/, '');
}

/** Pull ```python demo blocks out of lesson markdown and number them. */
function extractDemos(md) {
  const demos = [];
  if (!md) return { md: '', demos };
  const lines = md.split('\n');
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    if (/^```python\s+demo\s*$/.test(lines[i])) {
      const code = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) code.push(lines[i++]);
      out.push('```pydemo-' + demos.length, ...code, '```');
      demos.push(code.join('\n'));
    } else {
      out.push(lines[i]);
    }
  }
  return { md: out.join('\n'), demos };
}

function parseHints(text) {
  if (!text) return [];
  const hints = [];
  for (const line of text.split('\n')) {
    const m = line.match(/^(?:\d+\.|-)\s+(.*)$/);
    if (m) hints.push(m[1]);
    else if (hints.length && line.trim()) hints[hints.length - 1] += '\n' + line.replace(/^ {2,3}/, '');
  }
  return hints;
}

function loadModule(dir, lessonId) {
  const modFile = path.join(dir, '_module.md');
  if (!fs.existsSync(modFile)) return null;
  const { meta, body } = parseFrontmatter(fs.readFileSync(modFile, 'utf8'));
  const sec = splitSections(body);
  const module = {
    id: String(meta.id || path.basename(dir)),
    lesson: lessonId,
    title: meta.title || path.basename(dir),
    minutes: meta.minutes || null,
    intro: sec.Intro || '',
    shared: codeOf(sec.Shared || ''),
    problems: [],
  };
  const files = fs.readdirSync(dir).filter((f) => /^\d+_.*\.md$/.test(f)).sort();
  for (const f of files) {
    const n = parseInt(f, 10);
    const { meta: pm, body: pb } = parseFrontmatter(fs.readFileSync(path.join(dir, f), 'utf8'));
    const ps = splitSections(pb);
    const { md: lesson, demos } = extractDemos(ps.Lesson || '');
    let items = [];
    if (ps.Items) {
      try {
        items = JSON.parse(codeOf(ps.Items));
      } catch (e) {
        items = [];
        console.error(`[kata] ${f}: could not parse # Items JSON: ${e.message}`);
      }
    }
    module.problems.push({
      id: `${module.id}-${n}`,
      n,
      file: path.relative(CONTENT_DIR, path.join(dir, f)).replace(/\\/g, '/'),
      module: module.id,
      title: pm.title || f,
      difficulty: pm.difficulty || 'Easy',
      kind: pm.kind || (items.length ? 'predict' : 'code'),
      devices: pm.devices || ['cpu', 'cuda'],
      uses: pm.uses || [],
      inspect: pm.inspect || null,
      timeout: pm.timeout || 20,
      lesson,
      demos,
      task: ps.Task || '',
      given: codeOf(ps.Given || ''),
      starter: codeOf(ps.Starter || ''),
      solution: codeOf(ps.Solution || ''),
      checks: codeOf(ps.Checks || ''),
      hints: parseHints(ps.Hints),
      takeaway: ps.Takeaway || '',
      items,
    });
  }
  return module;
}

function loadAll() {
  const lessons = [];
  if (!fs.existsSync(CONTENT_DIR)) return { lessons, byId: new Map(), order: [], progression: [], issues: ['no content folder'], tracks: [{ id: MAIN, title: TRACK_TITLES.main, modules: [], order: [] }] };
  for (const ld of fs.readdirSync(CONTENT_DIR).sort()) {
    const dir = path.join(CONTENT_DIR, ld);
    if (!fs.statSync(dir).isDirectory()) continue;
    const lf = path.join(dir, '_lesson.md');
    const { meta, body } = fs.existsSync(lf) ? parseFrontmatter(fs.readFileSync(lf, 'utf8')) : { meta: {}, body: '' };
    const sec = splitSections(body);
    const lesson = {
      id: meta.id || ld, title: meta.title || ld, intro: sec.Intro || '',
      recall: sec.Recall || '', traps: sec.Traps || '', modules: [],
    };
    for (const md of fs.readdirSync(dir).sort()) {
      const mdir = path.join(dir, md);
      if (!fs.statSync(mdir).isDirectory()) continue;
      const mod = loadModule(mdir, lesson.id);
      if (mod) lesson.modules.push(mod);
    }
    lessons.push(lesson);
  }
  const { path: progression, issues, tracks } = buildProgression(lessons);
  const modById = new Map();
  for (const l of lessons) for (const m of l.modules) modById.set(m.id, { module: m, lesson: l });

  const byId = new Map();
  const order = [];
  const trackById = new Map(tracks.map((t) => [t.id, t]));
  for (const step of progression) {
    const { module: m, lesson: l } = modById.get(step.id);
    const t = trackById.get(step.track);
    t.modules.push(step.id);
    for (const p of m.problems) {
      byId.set(p.id, { problem: p, module: m, lesson: l, track: step.track });
      order.push(p.id);
      t.order.push(p.id);
    }
  }
  // A problem should only build on problems that come before it on the path.
  for (const id of order) {
    for (const u of byId.get(id).problem.uses) {
      if (!byId.has(u)) issues.push(`${id} uses ${u}, which does not exist`);
      else if (order.indexOf(u) > order.indexOf(id)) issues.push(`${id} uses ${u}, which comes later in progression.md`);
    }
  }
  for (const msg of issues) console.warn(`[kata] progression: ${msg}`);
  return { lessons, byId, order, progression, issues, tracks };
}

const MAIN = 'main';
const OFF = 'off'; // parked: hidden from home and Next, still openable by URL
const TRACK_TITLES = { main: 'Main path', off: 'Parked' };

/**
 * `- 1a  any title text   after: none` -> { id, after: [], track } (see content/progression.md).
 * `# Track: pinns  (title: Physics-informed networks)` starts a track; anything before the
 * first track heading is in "main". Returns { steps, tracks: [{ id, title }] } in file order.
 */
function parseProgression(text) {
  const steps = [];
  const tracks = [{ id: MAIN, title: TRACK_TITLES.main }];
  let track = MAIN;
  let inList = false; // bullets above the first "## " (or "# Track:") heading are prose
  for (const raw of text.split(/\r?\n/)) {
    const th = raw.match(/^#\s+Track:\s*([\w-]+)\s*(?:\(\s*title:\s*([^)]*)\))?/i);
    if (th) {
      track = th[1].toLowerCase();
      inList = true;
      let t = tracks.find((x) => x.id === track);
      if (!t) tracks.push(t = { id: track, title: TRACK_TITLES[track] || track });
      if (th[2] && th[2].trim()) t.title = th[2].trim();
      continue;
    }
    if (/^##\s/.test(raw)) inList = true;
    const m = inList && raw.match(/^\s*[-*]\s+(\S+)(.*)$/);
    if (!m) continue;
    const a = m[2].match(/\bafter:\s*([^#]*)$/);
    let after = null; // null = "the module listed just above (in the same track)"
    if (a) {
      const v = a[1].trim();
      after = /^(none|-|)$/i.test(v) ? [] : v.split(/[\s,]+/).filter(Boolean);
    }
    steps.push({ id: m[1].replace(/[`*]/g, ''), after, track });
  }
  return { steps, tracks };
}

/**
 * The learning path: every module once, in progression.md order, each with the
 * modules it builds on. Problems are never dropped: an unlisted module is
 * appended at the end and reported in `issues`.
 */
function buildProgression(lessons) {
  const issues = [];
  const known = new Map();
  for (const l of lessons) for (const m of l.modules) known.set(m.id, l);
  const file = path.join(CONTENT_DIR, 'progression.md');
  let steps = [];
  let trackList = [{ id: MAIN, title: TRACK_TITLES.main }];
  if (fs.existsSync(file)) ({ steps, tracks: trackList } = parseProgression(fs.readFileSync(file, 'utf8')));
  else issues.push('content/progression.md is missing, so modules are in folder order');

  const out = [];
  const seen = new Map(); // module id -> track
  const lastIn = (track) => { for (let i = out.length - 1; i >= 0; i--) if (out[i].track === track) return out[i].id; return null; };
  const listed = new Set(steps.map((s) => s.id));
  for (const s of steps) {
    if (!known.has(s.id)) { issues.push(`progression.md lists ${s.id}, but no module has that id`); continue; }
    if (seen.has(s.id)) { issues.push(`progression.md lists ${s.id} twice; the first one counts`); continue; }
    let after = s.after;
    if (after === null) { const prev = lastIn(s.track); after = prev ? [prev] : []; }
    after = after.filter((a) => {
      if (seen.has(a)) {
        if (seen.get(a) === s.track) return true;
        issues.push(`${s.id} (track ${s.track}) is after: ${a}, which is in track ${seen.get(a)}; each track is its own tree`);
        return false;
      }
      issues.push(known.has(a) && listed.has(a)
        ? `${s.id} is after: ${a}, but ${a} is listed below it in progression.md`
        : `${s.id} is after: ${a}, but no module has that id`);
      return false;
    });
    seen.set(s.id, s.track);
    out.push({ id: s.id, after, track: s.track });
  }
  for (const [id] of known) {
    if (seen.has(id)) continue;
    if (fs.existsSync(file)) issues.push(`module ${id} is not in progression.md (shown at the end of the main path for now)`);
    const prev = lastIn(MAIN);
    out.push({ id, after: prev ? [prev] : [], track: MAIN });
    seen.set(id, MAIN);
  }
  // Depth in the tree = longest chain of prerequisites above it.
  const depth = new Map();
  for (const s of out) {
    s.lesson = known.get(s.id).id;
    s.depth = s.after.length ? 1 + Math.max(...s.after.map((a) => depth.get(a))) : 0;
    depth.set(s.id, s.depth);
  }
  const tracks = trackList.map((t) => ({ ...t, modules: [], order: [] }));
  return { path: out, issues, tracks };
}

// Cheap change detection: the newest mtime under content/.
function newestMtime(dir) {
  let t = 0;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    t = Math.max(t, e.isDirectory() ? newestMtime(p) : fs.statSync(p).mtimeMs);
  }
  return t;
}

let cache = null;
let cacheStamp = -1;
function getCatalog() {
  const stamp = fs.existsSync(CONTENT_DIR) ? newestMtime(CONTENT_DIR) : 0;
  if (!cache || stamp !== cacheStamp) {
    cache = loadAll();
    cacheStamp = stamp;
  }
  return cache;
}

module.exports = { getCatalog, CONTENT_DIR, MAIN, OFF, _internals: { loadAll, parseProgression, buildProgression, parseFrontmatter, splitSections, codeOf, extractDemos, parseHints } };
