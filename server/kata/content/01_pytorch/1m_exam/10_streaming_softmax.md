---
title: Streaming softmax, three chunks
difficulty: Hard
---
# Task
Each chunk is summarised as `(m, l, o)`: max, `sum(exp(s-m))`, and
`sum(exp(s-m) * v)`. Then `s` and `vv` are deleted.

```text
m = max(m1, m2);  c1 = exp(m1-m);  c2 = exp(m2-m)
l = l1*c1 + l2*c2
o = o1*c1 + o2*c2
```

- `merge(s1, s2)` returns the merged `(m, l, o)` of two states.
- `merged`: fold `merge` over the three entries of `states`, one `(m, l, o)` tuple.
- `got`: the softmax-weighted sum of `vv`, shape `(DH,)`, from `merged`. Normalise once.

# Given
```python
Skv, DH = 9, 4
s = torch.randn(Skv) * 10
vv = torch.randn(Skv, DH)
want = F.softmax(s, -1) @ vv

states = []
for cs, cv in zip(s.split(3), vv.split(3)):
    m = cs.max()
    e = (cs - m).exp()
    states.append((m, e.sum(), e @ cv))
del s, vv
```

# Starter
```python
def merge(s1, s2):
    m1, l1, o1 = s1
    m2, l2, o2 = s2
    return TODO, TODO, TODO


merged = TODO       # fold merge over states
got = TODO          # normalise, once
```

# Solution
```python
def merge(s1, s2):
    m1, l1, o1 = s1
    m2, l2, o2 = s2
    m = torch.maximum(m1, m2)
    c1, c2 = (m1 - m).exp(), (m2 - m).exp()
    return m, l1 * c1 + l2 * c2, o1 * c1 + o2 * c2


merged = merge(merge(states[0], states[1]), states[2])
_, l_all, o_all = merged
got = o_all / l_all
```

# Checks
```python
check("shape", got, shape=(DH,))
check("equals the full softmax", got, want=want, atol=1e-5)
check("order does not matter",
      (lambda a: a[2] / a[1])(merge(states[2], merge(states[1], states[0]))),
      want=want, atol=1e-5)
```

# Hints
1. 1i Drill 2 merges two halves with exactly these correction factors; here `o` takes the place of the `e` vectors.
2. `torch.maximum` for `m`, and both states get a correction factor. Merge the first two states, then merge that result with the third; divide `o` by `l` only at the end.

# Takeaway
Both states get a correction factor, and you divide by `l` exactly once, at
the end. Nothing after the split touched `s`: the three summaries are enough,
in any order.
