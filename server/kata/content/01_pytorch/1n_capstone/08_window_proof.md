---
title: "H. Prove the window slides"
difficulty: Medium
uses: [1n-1, 1n-2, 1n-3, 1n-4, 1n-5, 1n-6]
---
# Lesson
## Poisoning the cache

With `window=4`, a query at position 7 must not see position 1 and must see
position 6.

To test that, poison the cache: corrupt one slot and see whether the logits
move. The same method localises kernel bugs without a debugger.

# Task
Given caches `snap_k`, `snap_v` holding positions `0..6`, run one step of
token `5` at position `7` three times, each on its own copy of the snapshot:

- `clean`: on an uncorrupted copy
- `kd, vd`: a copy with slot 1 corrupted (add `100.0` to the keys there), and
  `dead`, the logits on it
- `kl, vl`: a copy with slot 6 corrupted the same way, and `live`, the logits on it
- `window_holds`: `True` iff corrupting slot 1 changed nothing (`atol=1e-6`)
- `window_reaches`: `True` iff corrupting slot 6 did change things

# Given
```python
kc, vc = new_cache(CFG, 20)
for p, t in enumerate([4, 1, 7, 2, 9, 0, 3]):
    forward_step(t, p, W, kc, vc, CFG)
snap_k, snap_v = kc.clone(), vc.clone()
```

# Starter
```python
clean = TODO                # position 7 on an uncorrupted copy
kd, vd = TODO, TODO         # a copy with slot 1 corrupted
dead = TODO
kl, vl = TODO, TODO         # a copy with slot 6 corrupted
live = TODO
window_holds = TODO         # True iff corrupting slot 1 changed nothing
window_reaches = TODO       # True iff corrupting slot 6 did change things
```

# Solution
```python
clean = forward_step(5, 7, W, snap_k.clone(), snap_v.clone(), CFG)
kd, vd = snap_k.clone(), snap_v.clone()
kd[:, :, :, 1, :] += 100.0
dead = forward_step(5, 7, W, kd, vd, CFG)
kl, vl = snap_k.clone(), snap_v.clone()
kl[:, :, :, 6, :] += 100.0
live = forward_step(5, 7, W, kl, vl, CFG)
window_holds = torch.allclose(dead, clean, atol=1e-6)
window_reaches = not torch.allclose(live, clean, atol=1e-6)
```

# Checks
```python
check("clean is position 7 on the snapshot", clean,
      want=forward_step(5, 7, W, snap_k.clone(), snap_v.clone(), CFG))
expect("slot 1 of the kd/vd copy is corrupted",
       not (torch.equal(kd[:, :, :, 1], snap_k[:, :, :, 1]) and torch.equal(vd[:, :, :, 1], snap_v[:, :, :, 1])),
       "kd and vd match the snapshot at slot 1")
expect("slot 6 of the kl/vl copy is corrupted",
       not (torch.equal(kl[:, :, :, 6], snap_k[:, :, :, 6]) and torch.equal(vl[:, :, :, 6], snap_v[:, :, :, 6])),
       "kl and vl match the snapshot at slot 6")
check("dead", dead, shape=(1, 1, CFG["V"]))
check("live", live, shape=(1, 1, CFG["V"]))
expect("window_holds is what dead vs clean shows", window_holds == torch.allclose(dead, clean, atol=1e-6))
expect("window_reaches is what live vs clean shows", window_reaches == (not torch.allclose(live, clean, atol=1e-6)))
check("out-of-window keys are ignored", window_holds, want=True)
check("in-window keys matter", window_reaches, want=True)
```

# Hints
1. `forward_step` writes position 7 into the cache it is given, so every experiment needs its own `.clone()` of the snapshot.
2. The cache layout is `(layer, batch, kv_head, position, head_dim)`, so slot 1 of every layer and head is `[:, :, :, 1, :]`; add to it in place. Compare logits with `torch.allclose(..., atol=1e-6)`.

# Takeaway
At `pos=7` with `window=4` the live range is `[4, 7]`. If the first check
failed, your `lo` is too small. If the second failed, your window read is not
reaching the recent keys.
