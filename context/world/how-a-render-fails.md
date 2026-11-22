# 🌍 How a render fails

A render that goes wrong here goes wrong in a small number of recognisable ways, and each look
points at a different cause. This file names the looks, says what causes each and how sure that
is, and points at a real tile showing it, so a failed render can be read for its cause instead of
only being called bad. The single failure the project is built around, one fused animal, has its
own entry in [chimera.md](chimera.md).

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [The seed sets the look first](#the-seed-sets-the-look-first)
- [The looks](#the-looks)
- [The gallery](#the-gallery)
- [Where this came from](#where-this-came-from)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-seed-sets-the-look-first) ➡️

- **Register**: whether a render reads as a photograph or as a drawing (pencil sketch, engraving,
  flat colour illustration, clipart). Two renders can compose equally well and sit in different
  registers.
- **The comparison sheets**: three images, one per held-out seed, each rendering every trained
  adapter on the same cat and dog cell. Filed in
  [which adapter composes cat and dog](../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/README.md).
  A tile is named below by its seed and the label printed above it on the sheet.

## The seed sets the look first

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-looks) ➡️

**Whether a render is a photograph or a drawing is decided mostly by the seed's starting noise,
and nearly every adapter keeps its seed's register.** ✅

With no adapter at all, seed 12 draws a photograph and seed 11 draws a pencil sketch. Counted tile
by tile on the comparison sheets:

```text
seed 12 (photograph with no adapter)   21 of 23 adapters draw a photograph
                                         1 draws an illustration ("empty branch from cache")
                                         1 draws mush ("54-cell + plural prompt")
seed 11 (sketch with no adapter)       19 of 23 draw a sketch or flat illustration
                                         2 draw a photograph, both trained against filtered renders
                                         2 draw mush
```

So a drawing is usually the seed's doing rather than the adapter's, and an adapter's fidelity read
on a single seed mostly measures that seed. The two adapters that override a drawing seed were both
trained against filtered renders ("target is a picture" and "picture + cached empty"). The first of
them draws the same scene on every seed, which is its own failure below.

**The empty branch.** Each step also runs the model on the empty prompt, and that prediction is the reference guidance measures the prompt's pull against `stated` (`environment/known-failures.md` calls it the unconditional, empty prompt branch; see the [classifier-free guidance card](../../../../Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Classifier-Free%20Guidance.md#where-to-learn-this)). "From cache" means a stored copy of that prediction is used instead of the adapter's changed one `stated` (the design plan for the correction loss, `freeze_null`).

## The looks

Navigation: ⬅️ [The seed sets the look first](#the-seed-sets-the-look-first) | 📋 [TOC](#table-of-contents) | [Next](#the-gallery) ➡️

✅ seen on all three seeds or settled by a comparison that moved one thing · 🟡 seen, cause
not settled, so the entry names a candidate · ❓ seen once, cause unknown

| What it looks like | What causes it | How sure | Where to see it |
|---|---|---|---|
| one fused animal | no correction, or one too weak to carry to this pair | ✅ | every sheet, "no adapter" |
| flat clipart or line art of one animal, often in a dark circle | only the empty branch was free to move: it can restyle the whole picture and cannot place a second named animal | ✅ | seeds 9 and 11, "only the empty branch"; `v57w10-04-nullonly` at step 3,000 |
| brown or grey mush with no animal in it | trained far past its peak | ✅ | every sheet, "original, rank 8, 450k" |
| the same mush, from a changed branch prompt | branch prompts rewritten to "two animals" | ✅ | every sheet, "54-cell + plural prompt" |
| grey pencil sketch or engraving, often in a round frame | the seed's own register; on a photographic seed, the larger pools (54, 72, 182 cells) also fail to hold the photograph | 🟡 | seed 11, "no adapter"; seed 9, "54 cells" and "72 cells" |
| flat colour illustration of two animals | the seed's register again, rendered in colour by the v57 and v54 pools | 🟡 | seed 11, "43 cells, all 50 steps" |
| invented people | the plain loss trained past its peak on a broad pool | 🟡 | `v57w10-01-base` at step 10,000, seeds 9 and 10; seed 11, "43 cells, first 10, on the picture" |
| oil painting or impressionist scene | unknown; seen early in one run that froze the empty branch | ❓ | `v57w10-03-nulloff` at steps 1,000 and 3,000 |
| three or more animals on the right backdrop | plurality learned without identity | 🟡 | seed 9, "43 cells, all 50 steps" and "43 cells, first 10, on the picture" |
| two animals of the wrong species | identity lost while plurality holds | 🟡 | seed 11, "picture + cached empty" (a horse); seed 9, "judged on the picture" (two cats) |
| the same scene on every seed | trained against a filtered set of renders and learned one scene from it | 🟡 | every sheet, "target is a picture": a dog and a cat in a green mountain field each time |

The `v57w10-*` renders sit under
`/datasets/mmolefe/poe_repair_min/outputs/showcase/v57w10/<run>/samples/per_epoch/` on the cluster,
one folder per checkpoint.

## The gallery

Navigation: ⬅️ [The looks](#the-looks) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

`how-a-render-fails-gallery.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/how-a-render-fails-gallery.png`

Six of the looks above, one panel each, every panel a render already on disk. A red box marks the
patch, and the same patch is enlarged in the corner of its own panel, so the fused face, the
invented person, the third animal and the horse standing in for a dog can be seen rather than taken
on trust. Built by `scripts/failure_gallery.py`; the panels, their source files and their crop boxes
are listed in the sidecar beside the image.

Five looks still have no panel: the mush from a rewritten branch prompt, the pencil sketch, the flat
colour illustration, the oil painting, and the same scene on every seed. The last one needs several
seeds side by side rather than one crop.

## Where this came from

Navigation: ⬅️ [The gallery](#the-gallery) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

Read by eye from the three comparison sheets (seeds 9, 11 and 12, twenty-three adapters and the
no-adapter control, rendered 2026-09-21 by `scripts/across_adapters.py`) and from the per-checkpoint
renders of the `v57w10` cohort. The clipart row is also backed by a comparison that moved one thing:
the adapter with only the empty branch free against the plain adapter, same step, same seed, where
the plain one draws several animals and this one draws one. No automated scorer was used; the
[instance counter](compose-rate.md#what-people-get-wrong) cannot tell a cat from a second dog, and the [edge measure](../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-noise-sharpen-the-corrected-render.md#what-would-have-counted) prefers a sketch to a
photograph.

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

The page names looks by the label printed above a tile on the comparison sheets. The sheet below puts all 72 of those tiles on one page, and the code shows what each label stands for.

### How it fits together

*Picture held back.* `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/every-adapter-cat-and-dog-by-seed.png` has a tile with a shape that could not be identified at full size, so it is named here and not shown.

Rows are seeds 9, 11 and 12 and the 24 columns are the no-adapter control and 23 adapters, grouped by what the run changed (the original recipe's rank and regularisation, a different loss, which cells it trains on, what the target is, extra loss terms). The title on the sheet says the seed sets the look and only two adapters trained on filtered pictures override it `verified` (opened; the title is legible at full size). In the no-adapter column seed 11 is a pencil sketch and seed 12 a photograph `verified`. The two outlined columns, printed here as "target is a filtered picture" and "filtered picture + cached empty branch" and on the per-seed sheets as "target is a picture" and "picture + cached empty", show a green-field photograph on all three seeds `verified`; the card describes them the same way `stated` (the folder's card). The page's tally was counted from the three per-seed sheets, which hold the same 72 tiles, and the tally itself was not recounted here.

### The code behind it

Each label above a tile on the three per-seed sheets is a row of one table in the script that rendered them. The by-seed sheet above prints its own labels, from the `GROUPS` table in `_archive/scripts/across_adapters_by_family.py`. Excerpt from `scripts/across_adapters.py` (lines 32 to 42):

```python
# (label, run directory, checkpoint step, rank). The step is the last save unless an earlier one
# is the one this project actually reads, which is noted where it applies.
ROWS = [
    ("no adapter",                 None,                                            None,   32),
    ("original, rank 8, 450k",     "showcase/phase1_r8_450k",                       450000,  8),
    ("original, rank 16",          "showcase/phase1_r16_100k",                      30000,  16),
    ("original, rank 32",          "showcase/phase1_r32_100k",                      30050,  32),
    ("original + weight decay",    "showcase/phase1_r32_wd0.1_100k",                30000,  32),
    ("original + weight average",  "showcase/phase1_r32_wd0.1_ema0.999_40k",        30000,  32),
    ("judged on the picture",      "showcase/phase1_r32_x0loss_40k",                25000,  32),
    ("charged for its size",       "showcase/phase1_r32_energy0.05_from30050_40k",  40050,  32),
```

- The label is what the page's "Where to see it" column quotes, so a tile can be matched to a run directory and a step from this table.
- The page's "original, rank 8, 450k" mush tile is the rank 8 run read at step 450,000, the longest-trained row shown here, which is consistent with the page's cause "trained far past its peak" `verified` (read in the code; the cause itself is the page's).
- The table continues past line 42; the remaining rows were not excerpted.

The gallery on this page is built by `scripts/failure_gallery.py`, whose first panel entry is (lines 33 to 36):

```python
    ("across_adapters_seed09/00.png", (330, 330, 720, 720),
     "One fused animal",
     "no correction at all: plain product-of-experts composition",
     "cat and dog, seed 9, no adapter"),
```

- The tuple is a render file, the crop box on the 1024 pixel render, the look, its cause and which render it is.

### Further illustrations

The sheet above shows the "same scene on every seed" look that the page's gallery section lists as having no panel yet. The first outlined column carries it, with a dog and a cat in the same green mountain field on all three seeds `verified`. That is the adapter the page names for this look, and no cropped panel for it exists.

### Go deeper

| What | Link |
|---|---|
| The sheets' card, with the per-seed sheets and how they were made | [which adapter composes cat and dog](../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/README.md) |
| The one fused animal in detail | [chimera](chimera.md) |
| The scorer that cannot tell these looks apart | [compose rate](compose-rate.md) |
| The adapter that these tiles come from | [LoRA corrector](lora-corrector.md) |

### Still open

- The by-seed sheet's card says it was made by `scripts/across_adapters_by_family.py`. That file is not in `scripts/` now; a copy is at `_archive/scripts/across_adapters_by_family.py`, and `RENAMES.md` was not found to list it.
- The five looks the page says have no panel (rewritten-prompt mush, pencil sketch, flat colour illustration, oil painting, same scene on every seed) still have no cropped panel, so the gap line on the page stands.
<!-- related:end -->

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
