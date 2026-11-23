# Progression

The learning path. Every module must be listed here exactly once. A module
that is missing, listed twice, or unknown is reported by `npm run kata:verify`
and shown as a warning on the /kata home page.

- The order of this list is the order "Next" walks through new problems.
- `after:` names the modules this one builds on. These are the edges of the
  roadmap tree on /kata. Leave `after:` out to mean "after the module listed
  just above", which is right for a plain linear lesson. Write `after: none`
  for a starting point. An `after:` module must be listed *above* this one.
- Anything after the module id other than `after:` is ignored, so the
  title can sit there as a reminder.

**Tracks.** A `# Track: <id>  (title: Some title)` heading starts a track.
Each track is a separate path: its own roadmap tree, its own Next queue and its
own due count, chosen with the switcher on /kata. `after:` may only name
modules in the same track. Modules above the first track heading are in
`main`. `# Track: off` parks modules: hidden from home and Next, still
openable by URL, their cards kept but never counted as due. Moving a module to
another track keeps its progress. A track with no modules yet is not shown.

To add a lesson, add its folder under `content/`, then append its modules
under the track it belongs to (or a new `# Track:` heading). Point the first
one's `after:` at the module(s) it needs in that track, or `none`.

# Track: main  (title: Main path)

## 01_pytorch

- 1a  Shape, dtype, rank                      after: none
- 1b  Indexing and slicing                    after: 1a
- 1c  reshape, view, transpose                after: 1a
- 1d  Strides and tiles                       after: 1c
- 1e  Blank dimensions                        after: 1b, 1c
- 1f  Broadcasting                            after: 1e
- 1g  Reductions and keepdim                  after: 1f
- 1h  matmul and F.linear                     after: 1c, 1f
- 1i  Masking and softmax                     after: 1g
- 1j  Bits and the weight format              after: 1b
- 1k  The key/value cache                     after: 1b, 1c
- 1l  bfloat16, and finding the bug           after: 1g
- 1m  Exam                                    after: 1d, 1h, 1i, 1j, 1k, 1l
- 1n  Capstone: one decode step               after: 1m

# Track: pinns  (title: Physics-informed networks)

## 02_pinns

- 2a  Gradients of one number             after: none
- 2b  Layers and modules                  after: 2a
- 2c  The training loop                   after: 2b
- 2d  Derivatives with respect to inputs  after: 2c
- 2e  A first PINN                        after: 2d
- 2h  When a PINN fails                   after: 2e
- 2f  What autograd saves                 after: 2e
- 2g  The cost of a PINN step             after: 2f
- 2i  The heat equation                   after: 2g, 2h
