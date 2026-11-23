---
title: Where the time and memory went
difficulty: Medium
devices: [cpu]
uses: [2j-3, 2l-1, 2l-2, 2l-3]
inspect: [n_saved, residual_A, sage_A]
timeout: 60
---
# Lesson
## Three versions of one step

The Navier-Stokes step of `2m-1` on `demo_net(din=2, dout=3)` (width 32,
depth 3) at 1024 points, with your `jet2d`. A = `1024 * 32 * 4` bytes, as in
`2f`. The checks print these rows with your times.

| step | FLOPs | saved |
|---|---|---|
| nested autograd, then `loss.backward()` | 34.7F | 90.2 A |
| jet, autograd on the residual, `loss.backward()` | 14.6F | 42.6 A |
| jet, emitted adjoint, `stack.backward(gradient=)` | 14.6F | ? |

The jet removed the graph of a graph: 2.4x fewer FLOPs, 2.1x less memory,
and 1.8-2.5x less time on one CPU thread (our machine). The emitted adjoint
removes only the residual's own graph. By the rules of `2f-1`, each `mul`
saves its two operands; `add`, `sub` and constant times save nothing. One
`(N, 1)` field is `1/32` A at width 32. The fields here are column slices of
one `(N, 13)` stack. A slice shares its parent's memory (`2f-2` counted
memory by storage block), so saving one slice keeps all 13 columns alive.
`torch.cat` and slicing save nothing themselves (their backward needs only
shapes, like `sum` in `2f-1`).

## Time, and a compiler

PyTorch as used here runs each operation as soon as Python reaches it
(**eager mode**). Each operation runs as one **kernel**, a separate small
program on the CPU or GPU. The emitted code has about as many operations as
autograd's backward of the residual. On this 20-operation residual the
difference is within run-to-run noise: on our machine, medians of 5
interleaved repeats (as the checks print) ranged from 6% faster to 4% slower,
and single repeats from 27% faster to 16% slower. A compiler
that sees the whole step (`jax.jit` in the real code, `torch.compile` in
PyTorch) can merge many small elementwise operations into one kernel
(**fusion**). This lesson describes fusion; it did not measure it.

## On the real residual

The residual in `pinn-solver-accel` has 79 fields, 9 residuals and 171
operations. Timed alone at 4096 points in eager mode, with the seeds ready
beforehand, the emitted adjoint took 0.56 (CPU, one thread) and 0.70 (GPU,
RTX 3050 Ti) of autograd's time. But the real code passes one seed per
residual, such as `2 r / N` for `mean(r**2)` (`2k-1`). Those seeds need the
residual values, so the residuals are computed first and again inside the
emitted code. Timed that way on the GPU, it took 1.10 of autograd's time. A
compiler can merge the two computations.

# Task
- `n_saved`: how many distinct `(N, 1)` tensors autograd saves for the loss
  of `make_ns_loss` (read it in Shared) when every field is a separate
  tensor: the operands of each `mul`, each tensor counted once.
- `residual_A`: the memory the residual graph keeps in this step, in A, where
  the fields are slices of the stack: the stack's storage plus the saved
  tensors that are not fields.
- `sage_A`: your estimate of the `?` cell, from the jet row and `residual_A`.

The checks measure the three steps and print them.

# Given
```python
torch.set_num_threads(1)
```

# Starter
```python
n_saved = TODO
residual_A = TODO
sage_A = TODO
```

# Solution
```python
n_saved = 9            # u, v, u_x, u_y, v_x, v_y, and r1, r2, r3 (each squared as r * r)
residual_A = (13 + 3) / 32   # the whole (N, 13) stack, plus r1, r2, r3
sage_A = 42.6 - residual_A
```

# Checks
```python
torch.manual_seed(0)
N = 1024
A = N * 32 * 4
mnet = demo_net(din=2, dout=3)
mxy = torch.rand(N, 2) * 2 - 1
lp = make_ns_loss(0.05)
fields = {n: torch.randn(N, 1) for n in NS_NAMES}
n_res, b_res = count_saved(lambda: lp({n: t.clone().requires_grad_() for n, t in fields.items()}).mean())
m_fields = round(b_res / (N * 4))
expect(f"n_saved = {m_fields}", not is_blank(n_saved) and n_saved == m_fields, f"you said {n_saved}; autograd saved {n_res} times, {m_fields} distinct (N, 1) tensors. The four products u * u_x, ... save 6 fields; the three squares r * r save r1, r2, r3.")
sage = build_sage(make_ns_loss, NS_NAMES, 0.05)
F1 = flops(lambda: run_net(mnet, mxy))[0]
def step_nested():
    lp(as_dict(nested_stack(mnet, mxy), NS_NAMES)).mean().backward()
def step_jet():
    lp(as_dict(ns_stack(jet2d(mnet, mxy)), NS_NAMES)).mean().backward()
def step_sage():
    st = ns_stack(jet2d(mnet, mxy))
    with torch.no_grad():
        adj = sage(st, 1 / N)
    st.backward(gradient=adj)
saved = {"nested": lambda: lp(as_dict(nested_stack(mnet, mxy), NS_NAMES)).mean(),
         "jet": lambda: lp(as_dict(ns_stack(jet2d(mnet, mxy)), NS_NAMES)).mean(),
         "sage": lambda: ns_stack(jet2d(mnet, mxy))}
import time, statistics
def ms(fn, n=5):
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1e3
for fn in (step_nested, step_jet, step_sage):
    for _ in range(3): fn()
times = {"nested": [], "jet": [], "sage": []}
for i in range(5):    # interleaved repeats, alternating the order of jet and sage
    order = [("jet", step_jet), ("sage", step_sage)]
    for name, fn in (order if i % 2 == 0 else order[::-1]) + [("nested", step_nested)]:
        times[name].append(ms(fn))
rows = {}
for name, fn in [("nested", step_nested), ("jet", step_jet), ("sage", step_sage)]:
    fl = flops(fn)[0]
    sv = count_saved(saved[name])[1] / A
    rows[name] = (fl / F1, sv, statistics.median(times[name]))
    print(f"{name:7s} {fl / F1:5.1f}F  saved {sv:5.1f} A  median {rows[name][2]:6.1f} ms")
print(f"sage / jet time, median of 5 interleaved repeats: {statistics.median([s / j for s, j in zip(times['sage'], times['jet'])]):.2f}")
m_sage = rows["sage"][1]
m_diff = rows["jet"][1] - m_sage
expect(f"residual_A within 10% of the measured {m_diff:.2f} A", not is_blank(residual_A) and abs(residual_A - m_diff) <= 0.1 * m_diff, f"you said {residual_A}; separate tensors would give {b_res / A:.2f} A. The six saved fields are slices of the (N, 13) stack, so they keep all 13 columns (13/32 A); r1, r2, r3 are separate tensors.")
expect(f"sage_A within 2% of {m_sage:.1f} A", not is_blank(sage_A) and abs(sage_A - m_sage) <= 0.02 * m_sage, f"you said {sage_A}; the jet row saves {rows['jet'][1]:.1f} A, and the emitted adjoint removes the residual's graph")
print(f"residual graph with separate field tensors: {b_res / A:.2f} A; with slices of the stack (measured jet - sage): {m_diff:.2f} A")
```

# Hints
1. In `r1`: `u * u_x` and `v * u_y`. `r2` has two more products, and `l = r1 * r1 + r2 * r2 + r3 * r3` three squares.
2. A product of two different tensors saves both; `r * r` saves `r` once.

# Takeaway
What each part of the accelerated step did, measured on this step:

- The jet removed the graph of a graph: 34.7F to 14.6F, 90.2 A to 42.6 A,
  and about half the time.
- The emitted adjoint (SAGE) removed the residual's graph: 0.5 A (42.6 to
  42.1), the stack's 13 columns plus `r1`, `r2`, `r3`. With separate field
  tensors it would be 9 of them, 0.28 A. The time did not change measurably.
- A fused activation with a hand-written backward saves only its input jet.
  On a width-64 jet kept as one tensor it cut the jet's saved memory by 38%, against
  about 1% for SAGE here. Its backward needs the third derivative of the
  activation (for tanh, `-2 (1 - t**2) (1 - 3 t**2)`). It is described here,
  not built.

## Reading the real code

In `usask_accel/` of `pinn-solver-accel`:

- `sage.py`: your `TracedVar`, walk and emitter.
  - It adds `sqrt`, with the rule `adj / (2 * out)`. A helper `sqrt(x)`
    records when `x` is a `TracedVar` and calls `jnp.sqrt` otherwise.
  - A new op takes three pieces: a method that calls `_record`, a forward
    line, and a VJP rule. For `sin`: `out = torch.sin(x)` and `adj * torch.cos(x)`.
  - The emitted function is `fn(pred_det, g, *seeds)`. `pred_det` is the
    field stack. The batch is padded with unused rows to a fixed size, and
    `g["N_all"]` is the number of rows to read.
- `model.py`: `residual_forward` (9 residuals) and `build_sage_backward`,
  cached on the constants as yours is.
- `finetune_backend.py` lines 136-160:
  - `jax.vjp(f, x)` returns `f(x)` and a function that maps an adjoint to the
    gradients of the inputs: the role of `stack.backward(gradient=adj)`.
  - One seed per residual, `(2 w / count) * c * m * r`. This is your `2 r / N`
    with a loss weight `w`, a scale `c` from `out_scale`, a mask `m` that is 0
    on padded rows, and `count` real points.
- `jet.py`:
  - 3D input, 10 channels (mixed derivatives), SiLU (the activation `x * sigmoid(x)`) instead of tanh.
  - A first layer of sines and cosines of the input (`_encode_derivs`).
  - The outputs are decoded into physical fields by a pointwise function,
    differentiated with nested `jax.jvp` (`jet_fields`). A JVP is the
    forward-mode counterpart of a VJP: a derivative times a direction, which
    is what your jet carries.
  - The inputs are scaled to about `[-1, 1]` before the network;
    `inv_range` converts the derivatives back to the original units.
  - The fused activation is a hand-written GPU kernel, and values are stored
    in lower-precision formats.

The whole step there runs under `jax.jit`. Its published 450 ms to 14 ms per
step mixes all of these; it is not a number for your code.

To check that you could write this from a blank file, rewrite the given
pieces yourself: `forward_line` (Shared in `2l`), `_record` and `__mul__`
(given in `2l-1`), and the header and field lines of `emit_backward` (given
in `2l-3`).
