---
title: repeat_kv
difficulty: Medium
---
# Lesson
## expand, repeat, repeat_interleave

All three make an axis longer. They differ in cost and in the order of the
result.

```python demo
kv = torch.tensor([10.0, 20.0])

print("kv                     ", kv.tolist())
print("kv.repeat(2)           ", kv.repeat(2).tolist(), "  a b a b")
print("kv.repeat_interleave(2)", kv.repeat_interleave(2).tolist(), "  a a b b")
```

`expand` works only on a length-1 axis and gives a view with stride 0: the
same memory read twice. It costs nothing, and it is aliased: every position
along that axis is the same element.

```python demo
e = kv[:, None].expand(2, 2)

print("shape", S(e), " stride", e.stride(), " shares memory:", e.data_ptr() == kv.data_ptr())

# A single-element write succeeds and changes every position along the
# stride-0 axis, with no error.
e2 = kv.clone()[:, None].expand(2, 2)
src = e2.clone()
e2[0, 0] = 99.0
print("after e2[0, 0] = 99 the whole row reads", e2[0].tolist())

# In-place ops do refuse, because they would write the same address twice.
try:
    e.mul_(2.0)
except RuntimeError as err:
    print("but mul_ raises:", str(err).split(":")[-1].strip()[:52])
```

# Task
BioMistral has 8 key/value heads serving 32 query heads. Key/value head `j`
serves query heads `4j` to `4j+3`.

Write `repeat_kv(x, n_rep)` so that `x` of shape `(B, HKV, T, DH)` becomes
`(B, HKV*n_rep, T, DH)` with that mapping, copying the data at most once.
It must also work when `n_rep == 1`.

It takes two operations, and the first one copies nothing.

# Given
```python
HKV, G, T, DH = 2, 2, 3, 2
xk = torch.randn(1, HKV, T, DH)
```

# Starter
```python
def repeat_kv(x, n_rep):
    b, hkv, t, dh = x.shape
    if n_rep == 1:
        return x
    return TODO
```

# Solution
```python
def repeat_kv(x, n_rep):
    b, hkv, t, dh = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None].expand(b, hkv, n_rep, t, dh).reshape(b, hkv * n_rep, t, dh)
```

# Checks
```python
rk = repeat_kv(xk, G)
check("shape", rk, shape=(1, HKV * G, T, DH))
check("kv head 0 -> q head 0", rk[:, 0], want=xk[:, 0])
check("kv head 0 -> q head 1", rk[:, 1], want=xk[:, 0])
check("kv head 1 -> q head 2", rk[:, 2], want=xk[:, 1])
check("kv head 1 -> q head 3", rk[:, 3], want=xk[:, 1])
check("n_rep=1 is the identity", repeat_kv(xk, 1), want=xk)
```

# Hints
1. Of `expand`, `repeat` and `repeat_interleave`, only `expand` is free, and it only grows a length-1 axis. So you need a length-1 axis somewhere first.
2. Insert a blank axis after the head axis, `expand` it to `n_rep`, then collapse the two head axes together.

# Takeaway
A `repeat`-based answer has the right shape and the wrong contents. It gives
`kv0, kv1, kv0, kv1` where you need `kv0, kv0, kv1, kv1`.

Nothing crashes, and the model produces slightly wrong text. This is a common
bug in GQA code.
