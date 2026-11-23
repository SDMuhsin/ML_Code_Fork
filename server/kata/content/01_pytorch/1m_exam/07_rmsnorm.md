---
title: RMSNorm
difficulty: Medium
---
# Task
Write `rms_norm(x, w, eps=1e-5)` in the llama convention. The formula and the
operation names are not given.

# Given
```python
D = 8
```

# Starter
```python
def rms_norm(x, w, eps=1e-5):
    return TODO
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
xx, ww = torch.randn(2, 3, D), torch.rand(D) + 0.5
check("fp32", rms_norm(xx, ww),
      want=ww * (xx / xx.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()))
check("bf16 stays bf16", rms_norm(xx.bfloat16(), ww.bfloat16()),
      dtype=torch.bfloat16, shape=(2, 3, D))
check("eps is inside", rms_norm(torch.zeros(1, D), torch.ones(D)), want=torch.zeros(1, D))
# fp16 input: 300 squared overflows fp16 but not fp32, so this separates an
# implementation that upcasts from one that does not.
check("squares in fp32, not the input dtype",
      rms_norm(torch.full((1, D), 300.0, dtype=torch.float16),
               torch.ones(D, dtype=torch.float16)),
      want=torch.ones(1, D, dtype=torch.float16))
# and the cast back must precede the weight multiply
_xb = (torch.randn(1, D, generator=gen(0)) * 3).bfloat16()
_wb = (torch.rand(D, generator=gen(1)) + 0.5).bfloat16()
_rr = torch.rsqrt(_xb.float().pow(2).mean(-1, keepdim=True) + 1e-5)
check("rounds before the weight multiply", rms_norm(_xb, _wb).float(),
      want=(_wb * (_xb.float() * _rr).bfloat16()).float(),
      atol=0.0, rtol=0.0)
```

# Hints
1. 1g Drill 2. Three details change the answer: where `eps` goes, which dtype the squaring happens in, and when the cast back happens.
2. Work on an fp32 copy; scale by the reciprocal square root (one op) of the last-axis mean of squares, `eps` added to that mean. Cast back to `x.dtype` before multiplying by `w`.

# Takeaway
In fp16, 300 squared overflows; in fp32 it does not, so the upcast is visible.
The cast-back order does nothing in fp32, but in bf16 it changes the rounding,
and that is the kind of difference that shows up later as a logit mismatch (1l).
