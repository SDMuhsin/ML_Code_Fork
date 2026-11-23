---
title: Emit, exec, cache
difficulty: Hard
uses: [2l-1, 2l-2]
inspect: [emit_backward, build_sage]
timeout: 30
---
# Lesson
## From strings to a function

The walk gave, for each variable, a list of contribution strings. Turn them
into the source of one function, `fn(stack, seed)`, that returns the adjoint
stack:

1. the header `def loss_adjoint(stack, seed):`;
2. one line per field, `u_x = stack[:, 3:4]`, in the order of `input_names`;
3. one forward line per tape entry, from `forward_line` in Shared
   (`_t0 = u * u_x`). The mul rule needs these values;
4. the adjoint lines: for each entry **in reverse tape order** whose output
   has a list, `adj__t3 = <first>`, then `adj__t3 = adj__t3 + <next>` for each
   further string. Reverse order makes `adj__t3` complete before any later
   line reads it. In any other order a line can read an adjoint before it is
   written, and the call fails with `UnboundLocalError: cannot access local
   variable 'adj__t2' where it is not associated with a value`;
5. the same for each field in `input_names`. A field with no list, such as
   `p` in Navier-Stokes, gets `adj_p = torch.zeros_like(p)`;
6. `return torch.cat([adj_u, adj_v, ...], 1)`, in the order of `input_names`.

Code like this, with no loops or branches, is called **straight-line code**.
`exec(source, ns)` runs a string as Python, with the dict `ns` as its only
names. Running a `def` defines the function, so afterwards
`ns["loss_adjoint"]` is it. The source sees no other names, so `torch` must
be in `ns` for `torch.cat` and `torch.zeros_like` to work. Every body line
needs 4 spaces of indentation, as in a file:

```python demo
lines = ["def double_it(stack, seed):"]       # header: no indentation
lines.append("    a = stack[:, 0:1]")         # body lines: 4 spaces
lines.append("    return torch.cat([2.0 * a, seed * a], 1)")
src = "\n".join(lines)                        # one string, one line per entry
print(src)
ns = {"torch": torch}
exec(src, ns)                                 # defines double_it inside ns
print(ns["double_it"](torch.ones(2, 1), 0.5))
bare = {}
exec(src, bare)                               # defining works without torch ...
try:
    bare["double_it"](torch.ones(2, 1), 0.5)  # ... calling does not
except NameError as e:
    print("without torch in the dict:", e)
```

## Trace once, call every step

Tracing, walking and `exec` take milliseconds. Do them once and keep the
function in a dict, a **cache**. The constants of the loss, such as `nu`, are
written into the source as numbers (`0.05 * adj__t1`), so the cache key must
contain them. A key of the loss name alone returns the function for the old
`nu`, with no error. Here `make_ns_loss(nu)` returns the loss for one `nu`,
and the key is `(make_loss, tuple(input_names), *consts)`. A dict key can be
any value that cannot change, such as a function or a tuple; a list cannot,
hence `tuple(input_names)`.

# Task
- `emit_backward(tape, output_vars, seed_names, input_names, func_name)`:
  return `(source, fn)`, the source string and the function from `exec`, by
  steps 1-6. Use your `reverse_walk` from `2l-2` and `forward_line` from
  Shared. The function's arguments are `stack` and then one per seed name.
- `build_sage(make_loss, input_names, *consts)`: the cache lookup is given.
  Inside it, make one shared `tape = []`. `make_loss(*consts)` returns the
  per-point loss function, which takes a dict `{name: variable}` (read
  `make_ns_loss` in Shared); call it on `{n: TracedVar(n, tape) for n in
  input_names}`. Then call `emit_backward` with that loss as the one output,
  seed name `"seed"` and `func_name` `"loss_adjoint"`, and store the function
  in `SAGE_CACHE[key]`.

# Starter
```python
SAGE_CACHE = {}

def emit_backward(tape, output_vars, seed_names, input_names, func_name):
    adj = reverse_walk(tape, output_vars, seed_names)
    lines = [f"def {func_name}(stack, {', '.join(seed_names)}):"]
    lines += [f"    {n} = stack[:, {i}:{i + 1}]" for i, n in enumerate(input_names)]
    TODO
    src = "\n".join(lines)
    ns = {"torch": torch}
    exec(src, ns)
    return src, ns[func_name]

def build_sage(make_loss, input_names, *consts):
    key = (make_loss, tuple(input_names), *consts)
    if key not in SAGE_CACHE:
        TODO
    return SAGE_CACHE[key]
```

# Solution
```python
SAGE_CACHE = {}

def emit_backward(tape, output_vars, seed_names, input_names, func_name):
    adj = reverse_walk(tape, output_vars, seed_names)
    lines = [f"def {func_name}(stack, {', '.join(seed_names)}):"]
    lines += [f"    {n} = stack[:, {i}:{i + 1}]" for i, n in enumerate(input_names)]
    lines += [f"    {forward_line(e)}" for e in tape]

    def accumulate(name):
        exprs = adj[name]
        lines.append(f"    adj_{name} = {exprs[0]}")
        lines.extend(f"    adj_{name} = adj_{name} + {x}" for x in exprs[1:])

    for e in reversed(tape):
        if e[-1].name in adj:
            accumulate(e[-1].name)
    for n in input_names:
        if n in adj:
            accumulate(n)
        else:
            lines.append(f"    adj_{n} = torch.zeros_like({n})")
    lines.append(f"    return torch.cat([{', '.join('adj_' + n for n in input_names)}], 1)")
    src = "\n".join(lines)
    ns = {"torch": torch}
    exec(src, ns)
    return src, ns[func_name]

def build_sage(make_loss, input_names, *consts):
    key = (make_loss, tuple(input_names), *consts)
    if key not in SAGE_CACHE:
        tape = []
        loss = make_loss(*consts)({n: TracedVar(n, tape) for n in input_names})
        SAGE_CACHE[key] = emit_backward(tape, [loss], ["seed"], input_names, "loss_adjoint")[1]
    return SAGE_CACHE[key]
```

# Checks
```python
tape = []
bl = make_burgers_loss(0.05)({n: TracedVar(n, tape) for n in BURGERS_NAMES})
bsrc, bfn = emit_backward(tape, [bl], ["seed"], BURGERS_NAMES, "burgers_adjoint")
expect("emit_backward returns (source string, function)", isinstance(bsrc, str) and callable(bfn), f"got {type(bsrc).__name__}, {type(bfn).__name__}")
expect("the source defines burgers_adjoint(stack, seed)", bsrc.startswith("def burgers_adjoint(stack, seed):"), f"first line: {bsrc.splitlines()[0]}")
expect("no autograd in the source", "backward" not in bsrc and "grad" not in bsrc, "the emitted code must be plain arithmetic")
torch.manual_seed(7)
bst = torch.randn(30, 4, dtype=torch.float64)
check("Burgers: emitted adjoint = autograd", bfn(bst, 1 / 30), want=autograd_adjoint(make_burgers_loss(0.05), bst, BURGERS_NAMES), shape=(30, 4), atol=1e-12)
ns_fn = build_sage(make_ns_loss, NS_NAMES, 0.05)
nst = torch.randn(30, 13, dtype=torch.float64)
got = ns_fn(nst, 1 / 30)
check("Navier-Stokes: emitted adjoint = autograd", got, want=autograd_adjoint(make_ns_loss(0.05), nst, NS_NAMES), shape=(30, 13), atol=1e-12)
check("adjoint of p (not read) is 0", got[:, 2:3], want=torch.zeros(30, 1), atol=0)
expect("second call returns the cached function", build_sage(make_ns_loss, NS_NAMES, 0.05) is ns_fn, "build_sage traced again: look the key up in SAGE_CACHE first")
fn2 = build_sage(make_ns_loss, NS_NAMES, 0.025)
expect("a new nu gives a new function", fn2 is not ns_fn, "the key must contain the constants")
check("nu = 0.025: emitted adjoint = autograd", fn2(nst, 1 / 30), want=autograd_adjoint(make_ns_loss(0.025), nst, NS_NAMES), atol=1e-12)
def make_deep_loss():
    def loss_point(Fd):
        a = Fd["u"] * Fd["v"]
        b = a + Fd["u"]
        c = b * a                          # a is read at two depths
        return c * c
    return loss_point
dst = torch.randn(30, 2, dtype=torch.float64)
check("a value read at two depths: emitted adjoint = autograd", build_sage(make_deep_loss, ["u", "v"])(dst, 1 / 30), want=autograd_adjoint(make_deep_loss(), dst, ["u", "v"]), atol=1e-12)
tape = []
br = make_burgers_loss(0.05)
bf = {n: TracedVar(n, tape) for n in BURGERS_NAMES}
rr = bf["u"] * bf["u_x"] - 0.05 * bf["u_xx"] - bf["f"]
sq = rr * rr
two_src, two_fn = emit_backward(tape, [rr, sq], ["g_r", "g_l"], BURGERS_NAMES, "two_seeds")
g_r = torch.randn(30, 1, dtype=torch.float64)
ts = bst.clone().requires_grad_()
cols = as_dict(ts, BURGERS_NAMES)
r_t = cols["u"] * cols["u_x"] - 0.05 * cols["u_xx"] - cols["f"]
((g_r * r_t).sum() + (r_t * r_t).sum() / 30).backward()
check("two outputs with two seeds (g_r on r, 1/N on r*r) = autograd", two_fn(bst, g_r, 1 / 30), want=ts.grad, atol=1e-12)
with torch.no_grad():
    nograd = ns_fn(nst.float(), 1 / 30)
check("runs under torch.no_grad() in float32", nograd, want=got.float(), atol=1e-4)
```

# Hints
1. Step 3 is one list comprehension with `forward_line(e)`. For steps 4-5 write a small helper `accumulate(name)` that appends the `adj_` lines for one name.
2. If you get `UnboundLocalError: cannot access local variable 'adj__t2'` (or a `NameError` naming `adj__t...`), a line reads `adj__t2` before it is written: loop `for e in reversed(tape)`. Print `src` to see it.
3. `build_sage`: `make_loss(*consts)` is the per-point loss; call it on `{n: TracedVar(n, tape) for n in input_names}`.

# Takeaway
The program wrote the Navier-Stokes adjoint, 75 lines of straight-line code
with 40 adjoint lines, and it matched autograd to 1e-12. The same program
writes the 331 adjoint lines of the residual in `pinn-solver-accel`, 171
operations over 79 fields, with friction off. With friction the residual also
uses `sqrt`, whose rule `2m-2` gives. Measured against autograd in float64,
the difference was 0.0. The real `sage.py` (in the PR's `usask_accel/`
package) has
`emit_backward(tape, output_vars, seed_names, input_names, func_name)` with
the same steps; its source uses `jax.numpy` instead of `torch`.
