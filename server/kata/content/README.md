# Kata content format

Every problem is one Markdown file. The server re-reads these on change, so
editing a file and reloading the page is the whole authoring loop.

```
content/
  01_pytorch/                 one lesson
    _lesson.md                lesson title, intro, recall list, traps
    1a_shapes/                one module (one notebook of the original course)
      _module.md              id, title, minutes, intro, shared setup
      01_three_tensors.md     problem 1a-1
      02_ranks.md             problem 1a-2
```

The problem id is `<module id>-<file number>`: `1a-1`, `1a-2`, ...
Progress is keyed by that id, so do not renumber files once used.

## Adding a lesson: the progression index

`content/progression.md` is the learning path. It sets the order "Next" walks
through new problems, and its `after:` edges draw the roadmap tree on /kata.
Every module must be listed there exactly once. To add a lesson:

1. Add the lesson folder (`02_cuda/_lesson.md`, its modules and problems).
2. Append its modules to `progression.md` under a `## 02_cuda` heading, in
   learning order, e.g. `- 2a  Threads and blocks   after: 1n`.
   Leave `after:` out to mean "after the module just above".
3. Run `npm run kata:verify`. It fails at once (before starting Python) on a
   module missing from the index, an unknown or duplicate id, an `after:`
   that points below itself, or a `uses:` that points later in the path.
   The /kata home page shows the same problems in an amber box.

Nothing else is needed: the roadmap lays out any number of lessons by itself
(one band per lesson, one row per depth of `after:`).

### Tracks: main path, a detached track, or parked

`progression.md` is split into tracks by `# Track: <id>  (title: Shown title)`
headings. Each track has its own roadmap tree, its own Next queue and its own
due count; the /kata home page shows a switcher when more than one track has
modules. Modules above the first track heading belong to `main`.

- **In the main path:** list the modules under `# Track: main`, with `after:`
  pointing at main modules (e.g. `after: 1n`).
- **In its own track:** list them under their own heading, e.g.
  `# Track: pinns  (title: Physics-informed networks)` then `## 02_pinns`,
  the first module with `after: none`. An `after:` may not cross tracks
  (verify reports it). A track with no modules yet is hidden.
- **Parked:** list them under `# Track: off`. They are hidden from home and
  Next, still open by URL, and their cards are kept but never counted as due.

Moving a module between tracks keeps its progress (it is keyed by problem id).

Run `npm run kata:verify` (in `server/`) after any change. It runs every
reference solution on CPU and CUDA, runs every starter (which must fail
cleanly, not crash), and runs every demo.

## `_module.md`

````markdown
---
id: 1a
title: Shape, dtype, rank
minutes: 20
---
# Intro
One or two sentences: what you can do by the end.

# Shared
```python
# Optional. Runs before every problem and demo in this module.
# Put helpers here that several problems need (make_blob, an oracle, ...).
```
````

## A problem file

````markdown
---
title: Make three tensors
difficulty: Easy            # Easy | Medium | Hard
kind: code                  # code (default) | predict
devices: [cpu, cuda]        # optional; drop cuda if a check is CPU-specific
uses: [1n-1, 1n-2]          # optional; earlier problems whose code this one calls
inspect: [a, b, c]          # optional; default = names assigned in Starter
timeout: 20                 # optional, seconds
---
# Lesson
Teaching text shown above the task: the rule, then a runnable example.
Use `##` and below for headings inside it.

```python demo
x = torch.randn(2, 3, 4)
print("shape", S(x))
```

A `python demo` block gets a Run button. Demos run in order, each seeing the
names the earlier demos defined, like notebook cells. A trailing expression is
echoed, as in a notebook. Plain ```python blocks are shown but not runnable.

# Task
What to produce, stated precisely. Name every variable or function the checks read.

# Given
```python
# Runs before the learner's code. Shown read-only. Defines the inputs.
T, V = 3, 10
logits = torch.randn(T, V)
```

# Starter
```python
best = TODO
```

# Solution
```python
best = logits.argmax(-1)
```

# Checks
```python
check("best", best, shape=(T,))
expect("peak is a python float", isinstance(peak, float), f"peak is {type(peak).__name__}")
```

# Hints
1. First, gentle hint. A nudge toward the right tool.
2. Second hint, more specific. Never the full answer.

# Takeaway
Shown once the problem is solved: why it mattered, the trap it avoids.
````

## The namespace

Every run starts from: `torch`, `F` (`torch.nn.functional`), `math`, and

| name | |
|---|---|
| `TODO` | the placeholder. Any arithmetic, indexing or attribute access on it raises "a TODO placeholder is still in your code". |
| `S(t)` | `tuple(t.shape)` |
| `check(label, got, want=None, shape=None, dtype=None, atol=1e-5, rtol=1e-4)` | records pass/fail; **never raises** |
| `same(label, a, b)` | two shapes/tuples equal |
| `expect(label, cond, why="")` | a plain boolean condition |
| `is_blank(x)` | still TODO? |
| `DEVICE` | `"cpu"` or `"cuda"` for this run |
| `gen(seed)` | `torch.Generator(device=DEVICE).manual_seed(seed)` |

`torch.manual_seed(0)` is called at the start of each run. On a CUDA run
`torch.set_default_device("cuda")` is in effect, so factory calls
(`torch.randn`, `torch.zeros`, `torch.arange`, ...) make CUDA tensors.
Therefore:

- Seeded generators must come from `gen(seed)`, never `torch.Generator()`, or
  the CUDA run fails ("Expected a 'cuda' device type for generator").
- Checks compare on CPU in float64, so device never matters for `want=`.
- Pure-Python loops over tensor elements are slow on CUDA (each `float(t[i])`
  syncs). Convert with `.tolist()` first in oracles.

## How checks run

The order is: module Shared → uses → Given → Starter-or-learner code → Checks.

Checks run one top-level statement at a time. If a statement raises, that
statement becomes one failed check (labelled with its `check("label", ...)`
string if it is a single call) and the rest still run. So:

- Prefer many small `check(...)` lines over one big block.
- Do not use `assert`. Use `expect(...)`.
- A `for` loop of checks is fine; if it raises part-way the earlier results stay.
- Checks may define helper variables, e.g. `out = repeat_kv(x, 2)` then checks on `out`.
  Such *setup* lines (no `check/same/expect` call in them) never show up as a
  check of their own while the learner's code is unfinished. The checks that
  read `out` say why instead ("not filled in yet: `repeat_kv` still has a TODO
  (line 3)"). A real error in setup is shown once, as "setting up the checks: …".
  A `print(...)` in Checks that needs an unfinished value is silently skipped.
- A bare `TODO` line inside a function is turned into a call that raises, so
  the learner gets "still has a TODO (line N)". A bare `TODO` at top level
  is a no-op. If a check depends on it (1k-2's prefill), test for the untouched
  state and `expect(...)` a plain-English message, rather than letting `check`
  dump two tensors.

The Starter must run without raising in the learner phase (assigning
`TODO` is fine; `TODO + 1` is not), so that a fresh starter shows a list of
"not filled in yet" checks, not a crash. Put TODO inside function bodies when
the task is a function.

## Predict problems

`kind: predict` shows a form instead of an editor. The learner types one
shape per expression (or picks ok/raise). Put the expressions in `# Items` as
JSON. The truth is computed live by evaluating `expr` after Given; `answer` is
only what "Show solution" displays, and `verify` checks it matches.

````markdown
# Given
```python
w = torch.arange(32).reshape(4, 8)
```

# Items
```json
[
  {"key": "a", "expr": "w[2]", "answer": "(8,)"},
  {"key": "b", "expr": "torch.zeros(4, 2).expand(4, 7)", "kind": "raises", "answer": "raise"}
]
```
````

`kind` is `shape` (default) or `raises` (answer `ok` or `raise`).

## Style, carried over from kernel-practice

- Rule first, a short runnable example whose output *is* the lesson, then the task.
- Exact, plain sentences. No filler.
- Checks should say *what* is wrong (`shape is (3,), want (3, 1)`), so the
  learner can localise the mistake without reading the solution.
- Every trap the original notebook mentions belongs in the Lesson or Takeaway.
