---
title: The compare function
difficulty: Easy
---
# Lesson
## What bf16 is

bf16 has the same 8 exponent bits as fp32, so the same range, and 7 mantissa
bits, so about two decimal digits.

```python demo
f = torch.finfo(torch.bfloat16)
print(f)
print("eps =", f.eps, "= 2 ** -7, so 7 mantissa bits are stored")
```

Read it off `eps`: the smallest step above 1.0 is `2**-7`, so seven mantissa
bits are stored. (Some libraries report 8, counting the implicit leading
bit; `torch.finfo` does not report it at all.)

```python demo
for v in [1.0, 1.0 + 1/256, 1.0 + 1/128, 300.0, 300.5]:
    print(f"{v:>12} -> {float(torch.tensor(v).bfloat16()):>12}")
```

## bf16 addition is not associative

```python demo
t = torch.randn(4096, dtype=torch.bfloat16)

fwd = torch.tensor(0.0, dtype=torch.bfloat16)
for z in t:
    fwd = fwd + z

rev = torch.tensor(0.0, dtype=torch.bfloat16)
for z in reversed(t):
    rev = rev + z

print(f"forward {float(fwd):.5f}   reverse {float(rev):.5f}   fp32 {float(t.float().sum()):.5f}")
```

The same numbers summed in a different order give a different answer, and
neither result is wrong.

So "my kernel matches the reference" can never mean bit-equality. The real
project uses three separate gates:

1. argmax agreement ≥ 98.5% against a torch oracle built from the same
   packed bytes.
2. identical greedy tokens and max |Δlogit| ≤ 0.5 on the tensor-core
   build. Measured 0.09–0.14.
3. batch 4 against four batch-1 runs, which catches any reduction whose
   order depends on batch size. On the row-loop build that gate is literal
   bit-exactness; on the tensor-core build BioMistral uses, the
   fp32 slice sums make token order non-deterministic, so there it is the
   same "identical tokens, max |Δlogit| ≤ 0.5" bar as gate 2.

The 98.5% comes from HuggingFace's own eager and sdpa attention, which agree
on only 98.61% of prefill positions for `gemma-3-4b`. Two mathematically
identical bf16 paths do no better than that.

# Task
Write `compare(got, ref)`, returning a dict with `max_abs` and `argmax_agree`
(both Python floats) for two logit tensors of shape `(T, V)`:

- `max_abs`: the largest absolute difference anywhere
- `argmax_agree`: the fraction of rows whose argmax over `V` is the same

Compute in fp64 so the metric is not itself losing precision.

There is no relative error, because logits straddle zero and relative error
is meaningless near zero.

# Given
```python
V = 10
ref = torch.randn(5, V)
got = ref.clone()
got[0, 0] += 1.5
got[1] = got[1].flip(0)
```

# Starter
```python
def compare(got, ref):
    g, r = got.double(), ref.double()
    return dict(
        max_abs=TODO,
        argmax_agree=TODO,
    )
```

# Solution
```python
def compare(got, ref):
    g, r = got.double(), ref.double()
    return dict(
        max_abs=float((g - r).abs().max()),
        argmax_agree=float((g.argmax(-1) == r.argmax(-1)).double().mean()),
    )
```

# Checks
```python
m = compare(got, ref)
print(m)
check("max_abs", m["max_abs"], want=(got.double() - ref.double()).abs().max())
check("argmax_agree", m["argmax_agree"], want=(got.argmax(-1) == ref.argmax(-1)).double().mean())
check("identical tensors agree perfectly", compare(ref, ref)["argmax_agree"], want=1.0)
```

# Hints
1. `max_abs` is a full reduction of `|g - r|`; `argmax_agree` compares `argmax(-1)` of both, row by row.
2. A boolean tensor has no `mean`; convert it with `.double()` first. `float(...)` turns the rank-0 result into a Python float.
