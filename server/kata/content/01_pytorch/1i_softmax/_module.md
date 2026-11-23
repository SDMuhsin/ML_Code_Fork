---
id: 1i
title: Masking and softmax
minutes: 25
---
# Intro
By the end you can compute a softmax from pieces, without ever holding the
whole row. That is what lets 160 thread blocks each own a slice of the keys.
