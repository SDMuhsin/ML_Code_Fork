---
id: 1n
title: "Capstone: one decode step"
minutes: 150
---
# Intro
You build a working decoder layer with BioMistral's exact architecture, in
nine small checked parts. Give it its own sitting of two to three hours.

Each part calls the earlier parts it builds on: your last passing version of
each, or the reference version if you have not solved that part yet (the page
says which). So a part you cannot solve yet does not block the ones after
it.

# Shared
```python
import math

CFG = dict(D=8, H=4, HKV=2, DH=2, I=16, V=10, n_layers=2,
           window=4, eps=1e-5, theta=10000.0)
CFG["G"] = CFG["H"] // CFG["HKV"]
CFG["scale"] = CFG["DH"] ** -0.5
assert CFG["H"] * CFG["DH"] == CFG["D"]


def make_weights(cfg, seed=0):
    g = gen(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.3
    D, H, HKV, DH, I, V = (cfg["D"], cfg["H"], cfg["HKV"], cfg["DH"], cfg["I"], cfg["V"])
    return dict(
        embed=r(V, D), lm_head=r(V, D),
        final_norm=torch.rand(D, generator=g) + 0.5,
        layers=[dict(
            ln1=torch.rand(D, generator=g) + 0.5,
            wq=r(H * DH, D), wk=r(HKV * DH, D), wv=r(HKV * DH, D), wo=r(D, H * DH),
            ln2=torch.rand(D, generator=g) + 0.5,
            wg=r(I, D), wu=r(I, D), wd=r(D, I),
        ) for _ in range(cfg["n_layers"])])


def new_cache(cfg, max_ctx, dtype=torch.float32):
    sh = (cfg["n_layers"], 1, cfg["HKV"], max_ctx, cfg["DH"])
    return torch.zeros(sh, dtype=dtype), torch.zeros(sh, dtype=dtype)


# ---- the oracle: explicit Python loops in fp64 ----

def _rms(x, w, eps):
    r = 1.0 / math.sqrt(sum(float(z) ** 2 for z in x) / len(x) + eps)
    return [float(x[i]) * r * float(w[i]) for i in range(len(x))]


def _mv(W_, x):
    return [sum(float(W_[o][i]) * x[i] for i in range(len(x))) for o in range(len(W_))]


def _rope(vec, p, dh, theta):
    """The same rotation as rotate_half, index by index, written differently
    so that agreement is an independent check."""
    half = dh // 2
    out = [0.0] * dh
    for i in range(dh):
        ang = p / (theta ** (2.0 * (i % half) / dh))
        partner = -vec[i + half] if i < half else vec[i - half]
        out[i] = vec[i] * math.cos(ang) + partner * math.sin(ang)
    return out


def naive_step(tok, pos, W, kc, vc, cfg):
    D, H, HKV, DH, I = cfg["D"], cfg["H"], cfg["HKV"], cfg["DH"], cfg["I"]
    G, win, eps, th, sc = cfg["G"], cfg["window"], cfg["eps"], cfg["theta"], cfg["scale"]
    # Weights as plain Python lists, read once: element-by-element tensor reads are slow on CUDA.
    embed, lm_head, final_norm = W["embed"].tolist(), W["lm_head"].tolist(), W["final_norm"].tolist()
    layers = [{n: t.tolist() for n, t in lw.items()} for lw in W["layers"]]
    h = [float(z) for z in embed[tok]]
    for L, lw in enumerate(layers):
        x = _rms(h, lw["ln1"], eps)
        qf, kf, vf = _mv(lw["wq"], x), _mv(lw["wk"], x), _mv(lw["wv"], x)
        for kvh in range(HKV):
            kc[L][kvh][pos] = _rope(kf[kvh*DH:(kvh+1)*DH], pos, DH, th)
            vc[L][kvh][pos] = vf[kvh*DH:(kvh+1)*DH]
        lo = max(0, pos - win + 1)
        attn = [0.0] * (H * DH)
        for hh in range(H):
            kvh = hh // G
            qh = _rope(qf[hh*DH:(hh+1)*DH], pos, DH, th)
            s = [sum(qh[d] * kc[L][kvh][j][d] for d in range(DH)) * sc
                 for j in range(lo, pos + 1)]
            mx = max(s)
            e = [math.exp(z - mx) for z in s]
            tot = sum(e)
            for d in range(DH):
                attn[hh*DH + d] = sum(e[n] * vc[L][kvh][lo+n][d] for n in range(len(e))) / tot
        o = _mv(lw["wo"], attn)
        h2 = [h[i] + o[i] for i in range(D)]
        y = _rms(h2, lw["ln2"], eps)
        gate, upv = _mv(lw["wg"], y), _mv(lw["wu"], y)
        act = [(gate[i] / (1.0 + math.exp(-gate[i]))) * upv[i] for i in range(I)]
        dn = _mv(lw["wd"], act)
        h = [h2[i] + dn[i] for i in range(D)]
    return torch.tensor(_mv(lm_head, _rms(h, final_norm, eps)), dtype=torch.float64)


def naive_cache(cfg, max_ctx):
    mk = lambda: [[[[0.0]*cfg["DH"] for _ in range(max_ctx)]
                   for _ in range(cfg["HKV"])] for _ in range(cfg["n_layers"])]
    return mk(), mk()


# ---- helpers from 1m ----

def rope_tables(positions, head_dim, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def repeat_kv(x, n_rep):
    b, hkv, t, dh = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None].expand(b, hkv, n_rep, t, dh).reshape(b, hkv * n_rep, t, dh)


def rms_norm(x, w, eps=1e-5):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


def gqa_attention(q, keys, vals, allowed, scale, n_rep):
    b, h, tq, dh = q.shape
    k, v = repeat_kv(keys, n_rep), repeat_kv(vals, n_rep)
    scores = ((q @ k.transpose(-1, -2)) * scale).masked_fill(
        ~allowed[None, None], float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return (probs @ v).transpose(1, 2).reshape(b, tq, h * dh)


W = make_weights(CFG)
```
