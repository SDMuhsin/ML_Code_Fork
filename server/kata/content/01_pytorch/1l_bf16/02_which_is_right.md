---
title: Which one is right?
difficulty: Easy
---
# Lesson
## Find the bug

Below are three implementations of a small attention step. One is correct,
and each of the other two has one silent bug from an earlier module.

Find the correct one from the numbers first, then read the code.

The module setup defines them, together with the inputs `setup()` and the
helper `expand_kv`:

```python
H, HKV, DH = 4, 2, 2
G = H // HKV


def setup():
    torch.manual_seed(5)
    Tq, Skv = 3, 6
    return (torch.randn(1, H, Tq, DH), torch.randn(1, HKV, Skv, DH),
            torch.randn(1, HKV, Skv, DH), torch.arange(Skv - Tq, Skv), torch.arange(Skv))


def expand_kv(x, n_rep):
    b, hkv, skv, dh = x.shape
    return x[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(b, hkv * n_rep, skv, dh)


def attn_A(q, kk, vv, q_pos, kv_pos, n_rep, window):
    k, v = expand_kv(kk, n_rep), expand_kv(vv, n_rep)
    allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - window)
    s = (q @ k.transpose(-1, -2)) * DH ** -0.5
    p = F.softmax(s.masked_fill(~allowed[None, None], float("-inf")), -1, dtype=torch.float32)
    return (p @ v).transpose(1, 2).reshape(1, q.shape[2], -1)


def attn_B(q, kk, vv, q_pos, kv_pos, n_rep, window):
    k, v = kk.repeat(1, n_rep, 1, 1), vv.repeat(1, n_rep, 1, 1)
    allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - window)
    s = (q @ k.transpose(-1, -2)) * DH ** -0.5
    p = F.softmax(s.masked_fill(~allowed[None, None], float("-inf")), -1, dtype=torch.float32)
    return (p @ v).transpose(1, 2).reshape(1, q.shape[2], -1)


def attn_C(q, kk, vv, q_pos, kv_pos, n_rep, window):
    k, v = expand_kv(kk, n_rep), expand_kv(vv, n_rep)
    allowed = kv_pos[None, :] > q_pos[:, None] - window
    s = (q @ k.transpose(-1, -2)) * DH ** -0.5
    p = F.softmax(s.masked_fill(~allowed[None, None], float("-inf")), -1, dtype=torch.float32)
    return (p @ v).transpose(1, 2).reshape(1, q.shape[2], -1)
```

```python demo
args = setup()
for name, fn in [("A", attn_A), ("B", attn_B), ("C", attn_C)]:
    print(f"attn_{name} -> {S(fn(*args, G, 4))}")
```

All three run and give the same output shape.

## The oracle

The oracle below loops over heads and calls PyTorch's own attention. Its
structure differs from all three, so agreeing with it means something. It is in the module setup too:

```python
def oracle(q, kk, vv, q_pos, kv_pos, n_rep, window):
    B, H_, Tq, DH_ = q.shape
    allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - window)
    out = torch.empty(B, H_, Tq, DH_)
    for h in range(H_):
        out[:, h] = F.scaled_dot_product_attention(
            q[:, h], kk[:, h // n_rep], vv[:, h // n_rep],
            attn_mask=allowed, scale=DH_ ** -0.5)
    return out.transpose(1, 2).reshape(B, Tq, H_ * DH_)
```

The demo below uses `compare` from the previous problem, in its reference form.

```python demo
def compare(got, ref):
    g, r = got.double(), ref.double()
    return dict(
        max_abs=float((g - r).abs().max()),
        argmax_agree=float((g.argmax(-1) == r.argmax(-1)).double().mean()),
    )


args = setup()
ref = oracle(*args, G, 4)
for name, fn in [("A", attn_A), ("B", attn_B), ("C", attn_C)]:
    print(f"  attn_{name}: max_abs = {compare(fn(*args, G, 4), ref)['max_abs']:.4e}")
```

# Task
Which one is right? Set `correct` to `"A"`, `"B"` or `"C"`.

# Given
```python
def compare(got, ref):
    g, r = got.double(), ref.double()
    return dict(
        max_abs=float((g - r).abs().max()),
        argmax_agree=float((g.argmax(-1) == r.argmax(-1)).double().mean()),
    )


args = setup()
ref = oracle(*args, G, 4)
```

# Starter
```python
correct = TODO      # "A", "B" or "C"
```

# Solution
```python
correct = "A"
```

# Checks
```python
expect("the correct implementation", correct == "A", f"you said {correct!r}; look at the numbers again")
```

# Hints
1. Run the second demo. A correct implementation agrees with the oracle to float32 rounding, around `1e-7`; a buggy one is off by far more.
2. Exactly one of the three has a `max_abs` near zero. The other two are off by a visible amount.
