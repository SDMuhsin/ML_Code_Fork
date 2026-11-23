---
id: 1l
title: bfloat16, and finding the bug
minutes: 25
---
# Intro
From here on, most of the work is finding where two results diverge, for
example which step causes a 0.3 difference in the logits.

# Shared
```python
# The "find the bug" suspects: three attention steps and an oracle.
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
