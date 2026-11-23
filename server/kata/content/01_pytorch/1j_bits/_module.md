---
id: 1j
title: Bits and the weight format
minutes: 30
---
# Intro
BioMistral fits in 2.9 GB, about 3.2 bits per weight. The codes are 4-bit
and half the weights are pruned, but the survivor bitmap costs a bit per
column and the group scales are stored densely, so the real packed cost is
above the 2 bits those two numbers suggest. This module reads the format.

# Shared
```python
def make_blob(K, N, group=8, seed=0):
    """A miniature packed matrix, as raw bytes, the way a file looks."""
    g = gen(seed)
    nblk, blk = max(1, N // 32), min(32, N)
    keep = torch.zeros(K, N, dtype=torch.bool)
    for r in range(K):
        for t in range(nblk):                       # exactly half of every block
            sel = torch.randperm(blk, generator=g)[:blk // 2]
            keep[r, t * blk + sel] = True
    codes = torch.randint(0, 16, (K, N), generator=g)
    codes[~keep] = 0
    mask = ((keep.int().reshape(K, N // 8, 8) << torch.arange(8)).sum(-1)).to(torch.uint8)
    surv = codes[keep].reshape(K, N // 2)
    nib = (surv[:, 0::2] | (surv[:, 1::2] << 4)).to(torch.uint8)
    scale = (torch.rand(K, N // group, generator=g) + 0.5).half()
    zero = torch.randint(0, 16, (K, N // group), generator=g).to(torch.uint8)
    blob = torch.cat([mask.reshape(-1), nib.reshape(-1),
                      scale.reshape(-1).view(torch.uint8), zero.reshape(-1)])
    return blob, keep, codes, scale, zero
```
