---
id: 01_pytorch
title: Lesson 1 · PyTorch
---
# Intro
Everything needed to write a reference implementation of one BioMistral-7B
decode step. Lesson 3's CUDA kernels are checked against that reference.

Fourteen modules, one topic each, in order. `1a`–`1f` is a comfortable first
sitting. `1m` is the exam (no teaching), `1n` the capstone: one decode step,
built in parts that call your earlier parts.

Predict each result before you run it, and print `S(t)` often. When you stall,
go back a module before you open a solution. Do not fix a failing check by changing the check.

# Recall
A day after finishing, and again a week later, write each of these from memory
in a playground, with nothing else open.

1. split `(B, T, H*DH)` into `(B, H, T, DH)`, and invert it
2. expand 8 kv heads to 32 query heads so kv head `j` serves `4j..4j+3`
3. RMSNorm, llama convention, with the cast in the right place
4. the RoPE tables for a list of positions
5. `rotate_half` and `apply_rope` for `(B, H, T, DH)`
6. a causal plus sliding-window mask from `q_pos` and `kv_pos`
7. a masked softmax accumulating in fp32, returning bf16
8. the streaming-softmax merge of two `(m, l, o)` states
9. unpack `N/8` mask bytes into `N` booleans, no loop
10. scatter survivor codes into a dense matrix
11. the group-wise dequant affine
12. write one position into a preallocated cache and read the live window
13. SwiGLU, with `silu` on the right one of gate and up
14. `max_abs` and `argmax_agree` between two logit tensors
15. the flat offset of `t[i, j, k]` from `t.stride()`, and the inverse
16. a tiled matvec: row groups of `R`, column tiles of `C`

If you can type all sixteen cold, go on to lesson 2.

# Traps
Each of these gives a wrong answer with no error message. They recur in every later lesson.

1. k and v are a quarter of q's width. BioMistral has 32 query heads and 8 key/value heads, so `HKV*DH` is not `H*DH`.
2. GQA expansion is a block repeat: `expand` then `reshape`, or `repeat_interleave`. Never `repeat`.
3. Mistral's `post_attention_layernorm` is the pre-MLP norm.
4. `qkv` is row-concatenated; `gateup` is row-interleaved.
5. The per-input-column scale is folded into the activation, not the weight, so a fused matrix needs one column-scaled copy of the input per constituent.
6. Pruned weights must dequantize to exactly 0.0. Apply the keep mask last.
7. A transposed attention mask is invisible at decode (`T == 1`) and only breaks in prefill.
8. Promote to a wider integer before shifting bytes into a word.
9. Omitting `keepdim` is silent on a square tensor and an error on a rectangular one.
10. An `expand`ed view is writable and aliased: writing one element writes every position along the stride-0 axis.
11. The bar is 98.5% argmax agreement, not bit-equality.
