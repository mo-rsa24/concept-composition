# 🎯 Today: does the "and" vector work when it is built position by position?

<!-- template: exempt. Reason: a one-sitting walk off an idea map's claim 1, not a scope plan -->

> [!NOTE]
> **You will hold:** a third expert `x_pos` built from the joint prompt's own token sequence with each
> single prompt's word subtracted only at the position holding that same word; a per-position norm
> table proving it matches the joint prompt at the start token, the end token and the padding; the
> cat-by-dog sheet with `x_pos` as one more column; and its clear-pair count against the pass mark of
> 4 of the judgeable seeds. **Time:** ~1.5 h, 4 minutes of it on a GPU · **Stop point:** step 5's
> README row

Walk it with:

```text
/drip-execute-plan plans/10-what-poe-loses-when-it-composes/today-the-positional-and-vector.md
```

## Contents

- [Why this exists](#why-this-exists)
- [How x_pos is built](#how-x_pos-is-built)
- [The test, written before any render](#the-test-written-before-any-render)
- [Where things land](#where-things-land)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What it cannot claim](#what-it-cannot-claim)

## Why this exists

Navigation: 📋 [Contents](#contents) | [Next](#how-x_pos-is-built) ➡️

Six ways of building a third expert beside "dog" and "cat" have been tried and every one returned at
most 1 clear pair of the 7 judgeable seeds. The best of them, `x_fix`, cancels the start-of-text token
by adding the empty prompt's embedding to the whole 77-position sequence. That cancellation is global,
so it lands everywhere: the
[measured norms](../../artifacts/ideas/the-and-vector-as-a-third-expert/IDEA_MAP.md#1--x--em--ed--ec-is-a-usable-conditioning)
show the padding still holding 0.77 of the joint prompt's pad norm in CLIP ViT-L and 0.49 in OpenCLIP
bigG against a pass mark of 0.1, and the word positions sitting above a real prompt's. SDXL is handed a
sequence no prompt would produce.

The positional version has not been run. Subtracting only where the words are leaves every position
that carries no word exactly as a real prompt has it, which removes the one objection the norm figure
raises. Whether that is enough to stop the two animal experts fusing is the open question, and the
answer either way is cheap: the tiles for every other column already exist.

## How x_pos is built

Navigation: ⬅️ [Previous](#why-this-exists) | 📋 [Contents](#contents) | [Next](#the-test-written-before-any-render) ➡️

Both of SDXL's tokenizers split the sheet's three prompts the same way:

| Prompt | Positions 0 to 4 |
|---|---|
| joint, "dog and cat" | `<start>` `dog` `and` `cat` `<end>` |
| a, "dog" | `<start>` `dog` `<end>` |
| b, "cat" | `<start>` `cat` `<end>` |

So `x_pos` is the joint prompt's own sequence with two subtractions and nothing else:

| Position | What x_pos holds |
|---|---|
| 0 | the joint's `<start>`, untouched |
| 1 | the joint's `dog` minus `E("dog")`'s `dog` |
| 2 | the joint's `and`, untouched: no single prompt has that word to subtract |
| 3 | the joint's `cat` minus `E("cat")`'s `cat` |
| 4 to 76 | the joint's `<end>` and every pad, untouched |

Words are matched on the token itself, left to right, each joint position used once, so a prompt pair
with a repeated word ("a cat" and "a dog" inside "a cat and a dog") pairs the second prompt's `a` with
the joint's second `a`. The pooled vector stays the joint prompt's, because it has no positions to
align. Everything runs in fp32 and is cast to the model dtype after, as the other variants do.

**What the norm table can and cannot show.** Positions 0, the end token and the pads are copied from
the joint prompt, so they match it to the last bit by construction. The table proves the code does what
this section says. It is a build check and it says nothing about whether `x_pos` composes.

## The test, written before any render

Navigation: ⬅️ [Previous](#how-x_pos-is-built) | 📋 [Contents](#contents) | [Next](#where-things-land) ➡️

**The seeds.** Seeds 1 to 8, the set every sheet in
[supervisor-meeting/1-inference-only](../../supervisor-meeting/1-inference-only) renders, from the same
cached starting noise per seed. The joint prompt draws two dogs on seed 8, so **seeds 1 to 7 are
judgeable** and seed 8 is reported but not counted. That is the same 7 every earlier third expert was
measured on, which is what makes the counts comparable.

**The calls on every tile:** `2` one cat and one dog, each whole; `2h` two animals, at least one a
hybrid; `h` one hybrid animal; `1` a single plain animal; `x` no animal. Called from full-resolution
tiles, never from the downscaled sheet, because calling a downscaled sheet has already overcalled clear
pairs here once.

**The pass mark, unchanged from the sheets this is compared against:**

| Result | Reading |
|---|---|
| `PoE(dog, cat, x_pos)` is `2` on at least 4 of the 7 judgeable seeds | **supported**: building the vector position by position supplies what "and" adds |
| `2` on 0 or 1 judgeable seeds | **null**: the positional build is no better than the global one, and a fixed text-space vector is not the missing piece |
| `2` on 2 or 3 | **inconclusive** at 7 seeds |

**What would surprise me.** The prediction from the earlier runs is `h` on most seeds, the same hybrid
plain PoE draws, because the two animal experts fuse in their own forward passes and a third additive
term cannot pull them apart. A clear pair on 4 or more seeds would say the fusion was an artefact of
handing SDXL an off-distribution sequence rather than a property of the product.

## Where things land

Navigation: ⬅️ [Previous](#the-test-written-before-any-render) | 📋 [Contents](#contents) | [Next](#tasks) ➡️

```
scripts/and_vector_sheet.py                      x_pos, and the column sets that carry it
scripts/and_vector_norms.py                      x_pos in the norm table and the figure
scripts/and_vector_x_sheet.py                    the six-column supervisor layout, plus x_pos
artifacts/results/the-positional-and-vector/     the figure and its card
  token-norms-x-pos.png / .json                  per-position norms, both encoders, four sequences
  README.md                                      the card: what the figure shows and what it means
supervisor-meeting/1-inference-only/             sheets 1, 3 and 4 rebuilt with the x_pos column
supervisor-meeting/README.md                     section 1's table, the x_pos row

on the cluster, nothing synced through git (another session holds uncommitted tracked files):
/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/and_vector/_scripts/   the scripts, copied
/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/and_vector/dog__x__cat/seedNN/
  x_pos.png  poe_x_pos.png                       the 16 new tiles; every other column already there
```

## Tasks

For Claude to execute. Ask Claude to do these.

### 1. 🔧 Build x_pos and show the norm table (no GPU)

- [x] ✅ **1.1** Add the word-position alignment and `x_pos` to `scripts/and_vector_sheet.py`, with the two
  new columns `x_pos` and `poe_x_pos`, and `poe_x_pos` appended to the phrase and weight-sweep sets. ✓ verified 2026-11-04: `x_pos` and `poe_x_pos` in `COLUMNS`, appended to `PHRASES` and `SWEEP` through the new `BY_KEY` lookup, and a `positional` set added; `python3 -m py_compile` exit 0
- [x] ✅ **1.2** Add `x_pos` to `scripts/and_vector_norms.py`'s sequences, its figure, and a check that
  reports the largest absolute difference between `x_pos` and the joint prompt over position 0, the end
  token and the pads. That difference must be 0. ✓ verified 2026-11-04: `word_pairs` and `positional_x` added, the bar lives in the source as `SCAFFOLD_MAX_DIFF = 0.0`, and the figure carries four series
- [x] ✅ **1.3** Run the norm table on the sheet's prompts and print it. ✓ verified 2026-11-04: largest absolute difference from the joint prompt at the start token, the end token and all 72 pads is 0 against an allowed 0; the only positions differing at all are 1 and 3. Position 1 is exactly 0 in both encoders, because a [causal mask](../../../../goal-setting/learning/deep-learning/diffusion-models/text-embedding-arithmetic-in-sdxl/plans/02-position-zero-depends-on-itself.md) makes position 1 of "dog and cat" and of "dog" the same state. Files: `artifacts/results/the-positional-and-vector/dog-and-cat-norms.json` and `.png`
  - Command: copy both scripts to the cluster scratch folder, then
    `POE_REPO=<cluster repo> <co3 python> <scratch>/and_vector_norms.py --joint "dog and cat" --a dog --b cat --out artifacts/results/the-positional-and-vector`
  - ✅ `x_pos` equals the joint prompt at position 0, at the end token and at every pad, difference 0.
  - ❌ Any non-zero difference means the alignment is wrong. Stop; do not render.
  <a href="../../artifacts/results/the-positional-and-vector/dog-and-cat-norms.png"><img src="../../artifacts/results/the-positional-and-vector/dog-and-cat-norms.png" width="200" alt="Two bar charts of embedding norm at each of the 77 token positions of 'dog and cat', one for CLIP ViT-L and one for OpenCLIP bigG, with bars for x_raw, x_fix, x_pos and the joint prompt"></a>
  *Norm at each token position, four sequences, both encoders.*

▶ **Next: [instruction 4](#4--say-whether-to-spend-the-gpu)** (read the table and say yes or no).

### 2. 🔧 Render the two new columns (GPU, ~4 min)

- [ ] **2.1** Render `x_pos` alone and `PoE(dog, cat, x_pos)` on seeds 1 to 8, asserting CUDA is
  available before the first pass, on a `biggpu` node.
  - 16 tiles. Every other column is already on disk and is skipped.
- [ ] **2.2** Check all 16 landed and are photographs, not grey or black squares.

◀ **Needs: [instruction 4](#4--say-whether-to-spend-the-gpu)**, so the build is known good.
▶ **Next: [task 3](#3--the-sheets-the-figure-and-the-card)**.

### 3. 🔧 The sheets, the figure and the card

- [ ] **3.1** Build the cat-by-dog `x_pos` sheet and call every tile from the full-resolution files,
  writing the calls per seed with the counted seeds marked.
- [ ] **3.2** Rebuild sheets 1, 3 and 4 under `supervisor-meeting/1-inference-only/` with the `x_pos`
  column added, every earlier column kept.
- [ ] **3.3** Write `artifacts/results/the-positional-and-vector/token-norms-x-pos.png` and the card
  beside it, naming the axes, the four sequences and what the numbers mean.
- [ ] **3.4** Add the `x_pos` row to `supervisor-meeting/README.md` section 1 with its clear-pair count.

▶ **Next: [instruction 5](#5--veto-the-calls)**.

## Instructions

For you to follow manually. Do these yourself.

### 4. 👀 Say whether to spend the GPU

◀ **Needs: [tasks 1.1 to 1.3](#1--build-x_pos-and-show-the-norm-table-no-gpu)**, so the table exists.

- [ ] **Read the norm table** in the walk's message. Check three things: `x_pos` at position 0 equals
  the joint prompt's, the end token and pad rows read as a difference of 0, and the word positions 1
  and 3 are the only rows where `x_pos` differs from the joint.
- [ ] **Say `go`** to spend the 4 minutes of GPU, or `stop` if the table is wrong.

▶ **Next: [task 2.1](#2--render-the-two-new-columns-gpu-4-min)**.

### 5. 👀 Veto the calls

◀ **Needs: [task 3.1](#3--the-sheets-the-figure-and-the-card)**, so the calls exist.

- [ ] **Open the rebuilt sheet** and read the `PoE(dog, cat, x_pos)` column against the joint column.
- [ ] **Veto or accept each call.** A call changed here changes the count and the row.

▶ **Next:** the count is the plan's answer; the walk closes on it.

## What it cannot claim

Navigation: ⬅️ [Previous](#instructions) | 📋 [Contents](#contents)

**One pair, seven seeds.** Cat by dog only, and 7 judgeable seeds. A count of 2 or 3 is inconclusive
and needs a second pair and more seeds before it means anything.

**One weight.** `x_pos` enters at the run's guidance scale, like every other third expert. The weight
sweep that `x_fix` got has not been run for `x_pos`, so a null here does not rule out some other weight.

**Nothing about the LoRA.** This is an inference-only test. It says nothing about whether a learned
correction inside the denoiser composes, which is the route the idea map leaves open.

**The pooled vector is the joint prompt's.** `x_pos`'s pooled input is a real prompt's rather than a
difference, which is deliberate, and it means the global channel is not being tested here at all.
