# The positional "and" vector

Can a third expert built position by position, rather than subtracted across the whole sequence,
supply what the word "and" adds to a joint prompt? Rendered 2026-11-04 on mscluster108, outside
Slurm, from the walk at
[the plan it was walked from](../../plans/completed/today-the-positional-and-vector.md). These
numbers are read in [the finding "can a text-space third expert restore composition"](../../../report/can-a-text-space-third-expert-restore-composition.md).

**The answer: no on identity, yes on plurality.** `x_pos` draws two separate animals where plain
product-of-experts draws one fused one, on 7 of 8 seeds, which is the strongest count anything in this
series has reached. It draws one whole cat beside one whole dog on **0 of the 7 judgeable seeds**,
against a pass mark of 4 written before the first tile rendered.

## What x_pos is

Start from the joint prompt's own 77-position embedding. At the positions holding a word that one of
the single prompts also holds, subtract that single prompt's state at its own position for that word.
Leave everything else exactly as the joint prompt has it: the start token, the word "and", the end
token, and all 72 padding positions.

For "dog and cat" against "dog" and "cat", both of SDXL's tokenizers give the same split, so `x_pos`
is the joint prompt's sequence with position 1 (dog) and position 3 (cat) changed and nothing else.
The pooled vector stays the joint prompt's, because it carries no token positions to align.

## The figures

| File | What it shows |
|---|---|
| [token-norms-x-pos.png](token-norms-x-pos.png) | Per-position L2 norm, one bar group per token position 0 to 76, four series (x_raw, x_fix, x_pos, the joint prompt), top panel the CLIP ViT-L slice (channels 0 to 767), bottom panel the OpenCLIP bigG slice (channels 768 to 2047). Numbers in [token-norms-x-pos.json](token-norms-x-pos.json). |
| [how-the-third-expert-changed.png](how-the-third-expert-changed.png) | Every third expert tried, in the order tried, three seeds down and eight columns across, same starting noise along each row. **At 300 px a tile this is for the trajectory, not for calls**: identity cannot be judged at that size. |
| [all-seeds-target-poe-xpos.png](all-seeds-target-poe-xpos.png) and per-seed/ (on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/per-seed`) | Per seed, three whole tiles: the joint prompt's target, plain PoE, and the x_pos column. Whole tiles rather than crops, so no choice of mine can flatter or damn one. |
| positional-sheet.png | The eight-column sheet: the four references, then x_fix alone and in the product, then x_pos alone and in the product. |
| [calls.json](calls.json) | The call on every tile with what is in it, the judgeable seeds, the pass mark and the tally. |
| supervisor-sheet-1-before-x-pos.png | The six-column supervisor sheet, which is what `1-arithmetic-vector-x.png` holds. An eight-column version carrying x_pos was installed there briefly on 2026-11-04 and taken back out: that sheet is the x_fix comparison and x_pos does not belong in it. x_pos has its own sheet, `8-x-pos-subtracted-in-place.png`, which is this folder's `positional-sheet.png`. |
| [premise-word-recovery.json](premise-word-recovery.json) | The premise under every x variant, tested directly. For 200 random pairs from a 276-noun dictionary, the rank of B when the dictionary is sorted by closeness to E("A and B") − E("A"). Written by `scripts/and_vector_premise.py`, text encoders only, no UNet. |

## The premise holds: text arithmetic does recover the held-out word

Every variant above assumes E("A and B") − E("A") carries B, the way king − man + woman carries
queen. Measured on 200 random pairs against a 276-noun dictionary, where chance is a median rank of
138:

| readout | median rank of B | top-1 | top-10 |
|---|---|---|---|
| content mean, subtracted | **1** | 98.5% | 100% |
| content mean, no subtraction | 2 | 21.0% | 100% |
| content mean, a decoy word | 144 | 0.0% | 2.5% |
| pooled vector, subtracted | **1** | 75.0% | 92.5% |
| pooled vector, no subtraction | 2 | 39.0% | 97.0% |
| pooled vector, a decoy word | 126 | 0.5% | 4.5% |

The decoy rows sit at chance, which is what makes the top rows readable. Subtraction lifts top-1
from 21% to 98.5% on the content mean and from 39% to 75% on the pooled vector, so it is doing real
work rather than the joint prompt merely naming B.

**Read the content row with care.** That readout averages the content positions, so for "A and B"
it is roughly (A + and + B)/3, and subtracting A's mean is partly bound by arithmetic to land near
B. The pooled row is the one that is not: it is a learned projection rather than an average, and it
still reaches 75% top-1 with its decoy at chance.

**What this rules out.** The x variants did not fail because differences in CLIP's text space are
meaningless. They carry the word. So the failure lives either in the format handed to the UNet, which
the norm figure shows is not a well-formed prompt for x_raw and x_fix, or in the product itself,
which is where x_pos points.

**The nine other files in this folder, so nothing here is unexplained.** `dog-and-cat-norms.png` and
its JSON are byte-identical to `token-norms-x-pos.png` and its JSON: the first pair is what
`and_vector_norms.py` names its output, the second is the same file under the name the work asked
for. `supervisor-1-arithmetic-vector-x`, `supervisor-3-real-phrases` and
`supervisor-4-x-at-rising-weight`, PNG and JSON each, are the sheets as built here before being
installed into `supervisor-meeting/1-inference-only/`, and each is byte-identical to its installed
copy. `positional-sheet.json` is the sidecar of `positional-sheet.png`, listing every tile that went
into it. None of the nine is separate evidence; they are the same six things twice, kept because
`artifacts/results/` is gitignored and a staging copy is the only record of what was installed.

## What the numbers mean

**The build check.** `x_pos` differs from the joint prompt at exactly two of the 77 positions, 1 and 3.
The largest absolute difference across the start token, the end token and all 72 pads is **0**, against
an allowed 0 (`SCAFFOLD_MAX_DIFF` in `scripts/and_vector_norms.py`). This proves the code does what the
section above says. It is a build check and says nothing about composition.

**Position 1 is exactly zero in both encoders**, and that is the text encoder's causal mask rather than
a bug: position 1 of "dog and cat" and position 1 of "dog" both read `<start> dog` and produce the same
state, so the subtraction cancels. A position-wise subtraction can only carry information from the
second content word onward.

**Pad median, as a fraction of the joint prompt's own.** x_fix sits at 0.572 (CLIP ViT-L) and 0.555
(bigG) against a 0.1 mark it was meant to meet. x_pos sits at exactly 1.000 in both, because those
positions are the joint prompt's numbers unchanged. The sequence SDXL is handed is now one a real
prompt could produce.

**The counts, over the 8 rendered seeds.** Tiles showing two separate animals: x_raw 0, x_fix 1,
the best phrase naming both animals 5, **x_pos 7**. Tiles that collapsed to wallpaper, texture or flat
grey: x_raw 8, x_fix 3, **x_pos 0**. Clear cat-and-dog pairs on the 7 judgeable seeds: x_raw 0, x_fix 1,
every phrase 0, **x_pos 0**.

**Why seed 8 is not judgeable.** The joint prompt draws two dogs on seed 8, so that seed has no
reachable cat-and-dog answer and cannot count either way. The seed set is 1 to 8, the set every other
sheet in `supervisor-meeting/1-inference-only` uses, which is what makes these counts comparable with
the earlier ones.

## What it separates that was confounded before

The earlier failures mixed two questions. Whether the third expert's conditioning sequence was one
SDXL could read at all, and whether a third additive term can stop the two animal experts fusing.
`x_pos` settles the first: the sequence is in distribution and every tile is a photograph. The second
still fails, and it now fails on its own.

The split it leaves is sharper than "hybrid". **Where and how many** is recoverable from a fixed
text-space condition. **What each animal is** is not, because each expert's cross-attention has already
fused the two animals inside its own forward pass, which a term added afterwards cannot reach into.

## x_and: only what the joint adds, and it does nothing at all

x_and is the empty prompt's scaffolding (start token, end token, all 72 pads) with only what the
joint prompt adds written into the word slots: zero at "dog", where the subtraction cancels exactly,
the joint's "and" whole, and the small remainder at "cat". The pooled summary stays the empty
prompt's, so this tests the word slots alone.

**Result: 0 of 8 tiles show two separate animals, and 0 of 7 judgeable seeds show a clear pair.**
Every tile is one fused cat-dog, which is where plain product-of-experts already was. Seeds 4 and 5
left photography for flat graphics. The sheet is
[poe-x-and-all-seeds.png](poe-x-and-all-seeds.png).

Mean absolute pixel difference from the joint prompt's render, 0 to 255, averaged over 8 seeds:

| | distance from the joint prompt's image |
|---|---|
| x_and used alone | 37.82 |
| PoE(dog, cat, x_and) | **48.49** |
| plain PoE(dog, cat) | **47.05** |

**The third expert made no difference, and that is the structural finding.** A condition enters the
guided sum as w·(ε_x − ε_∅), its distance from the empty prompt. x_and was built to be nearly the
empty prompt plus a small remainder, so that term is nearly zero and the render is plain PoE's. The
more faithfully a third expert is built as a *difference*, the less it can do.

## What this cannot claim

One pair and seven judgeable seeds, so a count of 2 or 3 would have been inconclusive and 0 is only
firm because it is far from the mark. One weight: `x_pos` enters at the run's guidance scale and no
weight sweep was run for it, so a null here does not rule out some other weight. Nothing about whether
a correction learned inside the denoiser composes, which is a different route entirely.

<!-- cluster:start -->

## On the cluster

The other 11 files of this result (6.2 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 3 | 0.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/` |
| `per-seed` | 8 | 6.2 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/per-seed/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/<run>/ <run>/
```

<!-- cluster:end -->
