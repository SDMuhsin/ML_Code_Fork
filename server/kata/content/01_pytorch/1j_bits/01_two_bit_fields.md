---
title: Two-bit fields
difficulty: Easy
---
# Lesson
## Picking bits out of bytes

| | |
|---|---|
| `x >> n`, `x << n` | shift |
| `x & m` | keep only these bits |
| `x & 0xF` | the low 4 bits, one nibble |
| `(x >> 4) & 0xF` | the high nibble |

```python demo
b = torch.tensor([0xAB, 0x3C], dtype=torch.uint8)

print("bytes      ", [f"0x{v:02X}" for v in b.tolist()])
print("low nibble ", (b & 0xF).tolist())
print("high nibble", ((b >> 4) & 0xF).tolist())
```

A byte holds two 4-bit codes. Promote to a wider integer type before you
shift left.

```python demo
small = torch.tensor([0xFF], dtype=torch.uint8)

print("uint8 0xFF << 4 =", (small << 4).tolist(), " wrapped")
print("int64 0xFF << 4 =", (small.long() << 4).tolist(), " correct")
```

## Unpacking a bitmap without a loop

To turn `N/8` mask bytes into `N` booleans, lowest bit first:

```python demo
mask_bytes = torch.tensor([0b00001101, 0b10110000], dtype=torch.uint8)

bits = (mask_bytes[:, None] >> torch.arange(8, dtype=torch.uint8)) & 1
keep = bits.reshape(-1).bool()

print("bytes        ", [f"{v:08b}" for v in mask_bytes.tolist()])
print("keep         ", keep.int().tolist())
print("columns kept ", torch.nonzero(keep).flatten().tolist())
```

`bytes[:, None]` is `(Nb, 1)`, `arange(8)` is `(8,)`, so broadcasting gives
`(Nb, 8)` where entry `[i, j]` is bit `j` of byte `i`. Then `reshape`
flattens last-axis-fastest, which is byte 0's bits 0..7, then byte 1's.

This works because the bit order is lowest-first and `reshape` walks
row-major. The packer chose lowest-first for that reason.

# Task
Extract the 2-bit field at position `e` from the 32-bit word `w`, for every
`e` in `0..7`, as a Python list `fields`. Fields sit low to high, field `e` at
bits `[2e, 2e+2)`.

# Given
```python
w = 0x1BE4
```

# Starter
```python
fields = [TODO for e in range(8)]
```

# Solution
```python
fields = [(w >> (2 * e)) & 0x3 for e in range(8)]
```

# Checks
```python
check("fields", torch.tensor(fields), want=torch.tensor([0, 1, 2, 3, 3, 2, 1, 0]))
```

# Hints
1. Shift the field you want down to bit 0, then throw away everything above it.
2. Field `e` starts at bit `2 * e`. A field two bits wide is kept by a mask with exactly the two lowest bits set.
