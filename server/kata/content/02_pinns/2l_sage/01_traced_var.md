---
title: Record instead of compute
difficulty: Medium
inspect: [TracedVar]
---
# Lesson
## What does `a * b` run?

For objects of your own class, Python turns each operator into a method
call. `a * b` runs `a.__mul__(b)`, `a + b` runs `a.__add__(b)`, `a - b` runs
`a.__sub__(b)`, and `-a` runs `a.__neg__()`. Defining these methods is
called **operator overloading**.

`2 * x` first tries `(2).__mul__(x)`. A number does not know your class, so
that returns the special value `NotImplemented`, and Python then runs
`x.__rmul__(2)`, the "right-hand" version. The same holds for `__radd__` and
`__rsub__`. The demo uses a class whose methods print instead of computing:

```python demo
class Loud:
    def __init__(self, name):
        self.name = name
    def __mul__(self, other):
        print("__mul__ ", self.name, "*", getattr(other, "name", other))
        return Loud("prod")
    def __rmul__(self, other):
        print("__rmul__", other, "*", self.name)
        return Loud("prod")
a, b = Loud("a"), Loud("b")
c = a * b
d = 2 * a
e = a * 2.5
```

## A tape

A **tape** is a list that records each operation as an entry
`(op, inputs..., out)`. **Tracing** means running a function once on
recording objects instead of tensors, to fill the tape. The residual
functions of this lesson are written once, as plain arithmetic, and work on
both.

`TracedVar(name, tape)` is such an object. Its methods create a new
`TracedVar` for the result, named `_t0`, `_t1`, ... by its position on the
tape, and append an entry. The given `_record` uses `*` twice; the demo shows
both uses on plain strings:

```python demo
def record(tape, op, *operands):        # *operands: any number of arguments, as one tuple
    out = f"_t{len(tape)}"
    tape.append((op, *operands, out))   # * inside a tuple: spread that tuple's items into it
    return out
tape = []
record(tape, "mul", "u", "u_x")
record(tape, "smul", 0.05, "u_xx")
print(tape)
```

Five ops are enough for a PDE residual:

| code | entry |
|---|---|
| `a + b`, `a - b`, `a * b` (both TracedVar) | `("add", a, b, out)`, `("sub", ...)`, `("mul", ...)` |
| `k * a`, `a * k` (`k` a number) | `("smul", k, a, out)` |
| `a + k`, `k + a` | `("cadd", a, k, out)` |
| `a - k` | `("cadd", a, -k, out)` |
| `k - a` | `-a`, then `+ k` (your own `__neg__` and `__add__` record both) |
| `-a` | `("smul", -1.0, a, out)` |

# Task
Complete `TracedVar`. `__init__`, `_record` and `__mul__` are given. Write
`__rmul__`, `__add__`, `__radd__`, `__sub__`, `__rsub__` and `__neg__` so that
each records the entry in the table. Use `isinstance(other, TracedVar)` to
tell a variable from a number.

# Starter
```python
class TracedVar:
    def __init__(self, name, tape):
        self.name, self.tape = name, tape

    def _record(self, op, *operands):
        out = TracedVar(f"_t{len(self.tape)}", self.tape)
        self.tape.append((op, *operands, out))
        return out

    def __mul__(self, other):
        if isinstance(other, TracedVar):
            return self._record("mul", self, other)
        return self._record("smul", other, self)

    def __rmul__(self, other):
        TODO

    def __add__(self, other):
        TODO

    def __radd__(self, other):
        TODO

    def __sub__(self, other):
        TODO

    def __rsub__(self, other):
        TODO

    def __neg__(self):
        TODO
```

# Solution
```python
class TracedVar:
    def __init__(self, name, tape):
        self.name, self.tape = name, tape

    def _record(self, op, *operands):
        out = TracedVar(f"_t{len(self.tape)}", self.tape)
        self.tape.append((op, *operands, out))
        return out

    def __mul__(self, other):
        if isinstance(other, TracedVar):
            return self._record("mul", self, other)
        return self._record("smul", other, self)

    def __rmul__(self, other):
        return self._record("smul", other, self)

    def __add__(self, other):
        if isinstance(other, TracedVar):
            return self._record("add", self, other)
        return self._record("cadd", self, other)

    def __radd__(self, other):
        return self._record("cadd", self, other)

    def __sub__(self, other):
        if isinstance(other, TracedVar):
            return self._record("sub", self, other)
        return self._record("cadd", self, -other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return self._record("smul", -1.0, self)
```

# Checks
```python
def fresh(*names):
    tape = []
    return tape, [TracedVar(n, tape) for n in names]
def show(tape):
    return [tuple(x.name if isinstance(x, TracedVar) else x for x in e) for e in tape]
tape, (u, v) = fresh("u", "v")
r = 2 * u
expect("2 * u records smul", show(tape) == [("smul", 2, "u", "_t0")], f"tape is {show(tape)}; want [('smul', 2, 'u', '_t0')]")
tape, (u, v) = fresh("u", "v")
r = u + v
expect("u + v records add", show(tape) == [("add", "u", "v", "_t0")], f"tape is {show(tape)}")
tape, (u, v) = fresh("u", "v")
r = u + 1.5
r = 1.5 + u
expect("u + 1.5 and 1.5 + u record cadd", show(tape) == [("cadd", "u", 1.5, "_t0"), ("cadd", "u", 1.5, "_t1")], f"tape is {show(tape)}")
tape, (u, v) = fresh("u", "v")
r = u - v
r = u - 1.0
expect("u - v records sub, u - 1.0 records cadd -1.0", show(tape) == [("sub", "u", "v", "_t0"), ("cadd", "u", -1.0, "_t1")], f"tape is {show(tape)}")
tape, (u, v) = fresh("u", "v")
r = -u
expect("-u records smul -1.0", show(tape) == [("smul", -1.0, "u", "_t0")], f"tape is {show(tape)}")
tape, (u, v) = fresh("u", "v")
r = 1.0 - u
expect("1.0 - u records smul -1.0, then cadd 1.0", show(tape) == [("smul", -1.0, "u", "_t0"), ("cadd", "_t0", 1.0, "_t1")], f"tape is {show(tape)}")
expect("the result is the last entry's out", r is tape[-1][-1], "return the TracedVar that _record returns")
tape, vs = fresh(*BURGERS_NAMES)
l = make_burgers_loss(0.05)(dict(zip(BURGERS_NAMES, vs)))
expect("Burgers loss: mul, smul, sub, sub, mul", [e[0] for e in tape] == ["mul", "smul", "sub", "sub", "mul"], f"ops are {[e[0] for e in tape]}")
tape, vs = fresh(*NS_NAMES)
l = make_ns_loss(0.05)(dict(zip(NS_NAMES, vs)))
expect("Navier-Stokes loss: 20 entries", len(tape) == 20, f"the tape has {len(tape)} entries")
```

# Hints
1. `__rmul__` and `__radd__` are only called when the left operand is a number.
2. `__rsub__` computes `other - self`: build it from `__neg__` and `+`.

# Takeaway
Running the Navier-Stokes loss on 13 `TracedVar`s filled a tape of 20
entries, from the same function that computes the loss on tensors. A trap:
a constant must be a Python float. `np.float64(0.05) * u` happens to record
`0.05`, but `u * np.float64(0.05)` records `np.float64(0.05)`, and a torch
number records `tensor(0.0500)`. In `2l-3` both are written into the
generated source as that text, which then fails or rounds the value. The
real code
writes `float(c1e)` for that reason. It also names variables with a counter
kept on the class instead of the tape length.
