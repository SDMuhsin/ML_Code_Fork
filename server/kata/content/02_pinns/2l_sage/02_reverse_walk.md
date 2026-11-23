---
title: Walk the tape backwards
difficulty: Hard
uses: [2l-1]
inspect: [vjp_terms, reverse_walk]
---
# Lesson
## The backward pass, on names

The tape lists the operations in the order they ran. Walk it from the last
entry to the first, as a backward pass does, and apply the rule of `2k-2` to
each entry. This walk runs on names, not numbers, so each contribution is
written down as a piece of code: the adjoint of `_t0` is called `adj__t0`,
and through `("mul", u, u_x, _t0)` the input `u` gets the string
`"adj__t0 * u_x"`.

The walk keeps a dict from each variable's name to the list of its
contribution strings. It starts with the outputs. The adjoint given for an
output is its **seed**: for a per-point loss `l` with `L = mean(l)` it is
`1/N` at every point. For the Burgers tape of `2l-1`, whose last entry is
`_t4`, the dict starts as `{"_t4": ["seed"]}`, and the
generated code will receive `seed = 1/N`. Then, for each entry from the last:

- if its output has no list, nothing downstream reads it: skip the entry;
- else append one string per input, by the rule of its op.

The demo does this for `q = u * u + u_x`, with the tape written out and only
the two rules this tape needs:

```python demo
class V:
    def __init__(self, name):
        self.name = name
u, u_x, t0, t1 = V("u"), V("u_x"), V("_t0"), V("_t1")
tape = [("mul", u, u, t0), ("add", t0, u_x, t1)]
adj = {"_t1": ["seed"]}
for e in reversed(tape):
    op, a, b, out = e
    g = f"adj_{out.name}"
    if out.name not in adj:
        continue
    if op == "add":
        terms = [(a, g), (b, g)]
    else:                                   # mul
        terms = [(a, f"{g} * {b.name}"), (b, f"{g} * {a.name}")]
    for var, expr in terms:
        adj.setdefault(var.name, []).append(expr)
print(adj)
```

`u` has two strings: the product reads it twice, and in `2l-3` they are
added. The five rules as strings, with `g` the adjoint name of the entry's
output:

| entry | contributions |
|---|---|
| `("add", a, b, out)` | `a`: `g`; `b`: `g` |
| `("sub", a, b, out)` | `a`: `g`; `b`: the string `"-" + g` |
| `("mul", a, b, out)` | `a`: `g * b`; `b`: `g * a` |
| `("smul", k, a, out)` | `a`: `k * g` (the number written with `repr(k)`) |
| `("cadd", a, k, out)` | `a`: `g` |

# Task
- `vjp_terms(entry, g)`: the list of `(TracedVar, string)` pairs for one tape
  entry, `g` being the adjoint name of its output, by the table.
- `reverse_walk(tape, output_vars, seed_names)`: the dict described above,
  starting from `{v.name: [s] for v, s in zip(output_vars, seed_names)}`.

`TracedVar` is your `2l-1` class.

# Starter
```python
def vjp_terms(entry, g):
    op = entry[0]
    if op == "add":
        a, b = entry[1], entry[2]
        return [(a, g), (b, g)]
    TODO

def reverse_walk(tape, output_vars, seed_names):
    adj = {v.name: [s] for v, s in zip(output_vars, seed_names)}
    TODO
    return adj
```

# Solution
```python
def vjp_terms(entry, g):
    op = entry[0]
    if op == "add":
        a, b = entry[1], entry[2]
        return [(a, g), (b, g)]
    if op == "sub":
        return [(entry[1], g), (entry[2], f"-{g}")]
    if op == "mul":
        a, b = entry[1], entry[2]
        return [(a, f"{g} * {b.name}"), (b, f"{g} * {a.name}")]
    if op == "smul":
        return [(entry[2], f"{entry[1]!r} * {g}")]
    if op == "cadd":
        return [(entry[1], g)]
    raise ValueError(f"unknown op {op}")

def reverse_walk(tape, output_vars, seed_names):
    adj = {v.name: [s] for v, s in zip(output_vars, seed_names)}
    for entry in reversed(tape):
        out = entry[-1]
        if out.name not in adj:
            continue
        for var, expr in vjp_terms(entry, f"adj_{out.name}"):
            adj.setdefault(var.name, []).append(expr)
    return adj
```

# Checks
```python
def trace(make_loss, names, *consts):
    tape = []
    vs = {n: TracedVar(n, tape) for n in names}
    return tape, make_loss(*consts)(vs)
def evaluate(tape, adjd, stack, names, seed):
    bad = [x for v in adjd.values() for x in v if not isinstance(x, str) or " object at " in x]
    if bad:
        raise ValueError(f"the contribution {bad[0]!r} is not code: inside the f-string use the variable's .name")
    # Runs the tape on numbers, then sums each list of strings in reverse tape order.
    env = dict(as_dict(stack, names), seed=seed)
    for e in tape:
        op, out = e[0], e[-1].name
        val = lambda x: env[x.name]
        env[out] = {"add": lambda: val(e[1]) + val(e[2]), "sub": lambda: val(e[1]) - val(e[2]),
                    "mul": lambda: val(e[1]) * val(e[2]), "smul": lambda: e[1] * val(e[2]),
                    "cadd": lambda: val(e[1]) + e[2]}[op]()
    for e in reversed(tape):
        n = e[-1].name
        if n in adjd:
            env["adj_" + n] = sum(eval(x, {}, env) for x in adjd[n])
    return torch.cat([sum(eval(x, {}, env) for x in adjd[n]) if n in adjd else torch.zeros_like(stack[:, :1]) for n in names], 1)
tape, l = trace(make_burgers_loss, BURGERS_NAMES, 0.05)
d = reverse_walk(tape, [l], ["seed"])
expect("the walk reaches the fields", len(d) > 1, f"the dict is {d}: walk reversed(tape); the seed sits on the last entry, so a forward walk skips everything")
expect("u has one contribution, from the product", d.get("u") == ["adj__t0 * u_x"], f"d['u'] = {d.get('u')}")
expect("smul writes the constant first: u_xx gets '0.05 * adj__t1'", d.get("u_xx") == ["0.05 * adj__t1"], f"d['u_xx'] = {d.get('u_xx')}. The table writes the constant first, as the real sage.py does; the order matters only to this check.")
expect("_t3 gets two contributions from l = _t3 * _t3", len(d.get("_t3", [])) == 2, f"d['_t3'] = {d.get('_t3')}")
torch.manual_seed(6)
bst = torch.randn(30, 4, dtype=torch.float64)
b_got = evaluate(tape, d, bst, BURGERS_NAMES, 1 / 30)
b_want = autograd_adjoint(make_burgers_loss(0.05), bst, BURGERS_NAMES)
for i, n in enumerate(BURGERS_NAMES):
    check(f"Burgers: adjoint of {n}, evaluated, = autograd", b_got[:, i:i + 1], want=b_want[:, i:i + 1], atol=1e-12)
tape, l = trace(make_ns_loss, NS_NAMES, 0.05)
d = reverse_walk(tape, [l], ["seed"])
nst = torch.randn(30, 13, dtype=torch.float64)
n_got = evaluate(tape, d, nst, NS_NAMES, 1 / 30)
n_want = autograd_adjoint(make_ns_loss(0.05), nst, NS_NAMES)
for i, n in enumerate(NS_NAMES):
    check(f"Navier-Stokes: adjoint of {n}, evaluated, = autograd", n_got[:, i:i + 1], want=n_want[:, i:i + 1], atol=1e-12)
expect("p is never read, so it has no list", "p" not in d, "the NS loss reads p_x and p_y, not p")
tape = []
a, b = TracedVar("a", tape), TracedVar("b", tape)
unused = a * b
used = a + b
d = reverse_walk(tape, [used], ["seed"])
expect("an entry that does not reach the output is skipped", "_t0" not in d and d.get("a") == ["adj__t1"], f"got {d}; the product a * b is not read by the output")
```

# Hints
1. `vjp_terms`: one `if` per op, each returning a list of `(variable, string)` pairs. `f"{k!r}"` writes a number as Python reads it back.
2. `reverse_walk`: `for entry in reversed(tape)`, skip when `entry[-1].name not in adj`, else extend the lists with `setdefault(name, []).append(...)`.

# Takeaway
The walk is the backward pass, done once on names instead of on tensors at
every step. Its result is the text of the computation. The checks
evaluated that text and matched autograd on the Navier-Stokes loss to
1e-12. The real `sage.py` has the same walk inside `emit_backward`, plus a
sixth rule, for `sqrt`.
