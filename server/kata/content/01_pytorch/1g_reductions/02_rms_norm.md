---
title: RMSNorm
difficulty: Medium
---
# Task
Write `rms_norm(x, w, eps=1e-5)`.

BioMistral's version, in words:

> Take `x`. In fp32, compute the reciprocal square root of the mean of its
> squares over the last axis, with `eps` added to that mean. Scale `x` by
> it. Round back to `x`'s dtype. *Then* multiply by `w`.

Three details that change the answer:

1. `eps` goes inside the reciprocal square root, added to the mean.
2. The cast back happens before the weight multiply.
3. The squaring happens in fp32, not in the input's dtype.

Also use the single-op reciprocal square root rather than `1 / sqrt(...)`.
The kernel uses it. At bf16 the two give identical results, so this is for
consistency with the kernel; no check here can tell them apart.

The operation names are not given. Recall them.

Three of the checks each target one of the three details.

# Given
```python
D = 8
xf = torch.randn(2, 3, D)
wf = torch.rand(D) + 0.5
```

# Starter
```python
def rms_norm(x, w, eps=1e-5):
    h = TODO            # fp32 copy
    h = TODO            # normalised
    return TODO         # cast back, then scale
```

# Solution
```python
def rms_norm(x, w, eps=1e-5):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)
```

# Checks
```python
got = rms_norm(xf, wf)
ref = wf * (xf / xf.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt())

check("shape", got, shape=(2, 3, D))
check("matches an independent formulation", got, want=ref)
check("eps is inside, so zeros stay zero",
      rms_norm(torch.zeros(1, D), torch.ones(D)), want=torch.zeros(1, D))
check("bf16 in, bf16 out", rms_norm(xf.bfloat16(), wf.bfloat16()),
      dtype=torch.bfloat16, shape=(2, 3, D))

# 1. the fp32 upcast. In fp16, 300 squared overflows; in fp32 it does not.
x16 = torch.full((1, D), 300.0, dtype=torch.float16)
check("upcasts before squaring", rms_norm(x16, torch.ones(D, dtype=torch.float16)),
      want=torch.ones(1, D, dtype=torch.float16))

# 2. eps inside the reciprocal square root, not added after it.
xs = torch.full((1, D), 1e-4).bfloat16()
wb = (torch.rand(D, generator=gen(1)) + 0.5).bfloat16()
check("eps goes inside", rms_norm(xs, wb).float(),
      want=(wb * (xs.float() * torch.rsqrt(
          xs.float().pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16()).float())

# 3. the cast back happens before the weight multiply.
xb = (torch.randn(1, D, generator=gen(0)) * 3).bfloat16()
rr = torch.rsqrt(xb.float().pow(2).mean(-1, keepdim=True) + 1e-5)
check("rounds before the weight multiply", rms_norm(xb, wb).float(),
      want=(wb * (xb.float() * rr).bfloat16()).float(), atol=0.0, rtol=0.0)
```

# Hints
1. You need four operations: an fp32 cast, a mean of squares over the last axis (keeping it), a reciprocal square root, and a cast back with `.to(...)`.
2. The single op is `torch.rsqrt`, and `eps` is added to the mean before it, not to its result. At the end, `.to(x.dtype)` comes before `* w`.

# Takeaway
Detail 2 does nothing in fp32. In bf16 it changes the rounding, and that is
the kind of difference that shows up later as a logit mismatch. Module 1l
explains why.
