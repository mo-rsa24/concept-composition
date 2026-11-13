# 🧪 Does a model with no CLIP anywhere still hybridise?

This plan asks whether the same summed-guidance composition that fuses a cat and a dog into one
animal on SDXL also fuses them on PixArt-alpha, whose only text encoder is Flan-T5-XXL. A fusion
there removes CLIP's text embeddings from the list of possible causes.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/14-does-a-clip-free-model-hybridise.md — <CLIP exonerated or still on the list>
```

## Recommended skill

▶ `/pair-figure` ✅ the PixArt sheet means nothing on its own; it is read against the SDXL counts
   from the owning finding, and deciding how those two sit beside each other is that skill's job.
   alt: `/analyze-figure` on the finished sheet first, if the by-eye calls turn out to be close.

## Position in the plan tree

**Step 77 of 77.** Waits on nothing. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 6 | [hypothesis-03: when-in-the-run-it-matters](../../../../artifacts/plans/completed/03-does-the-correction-cause-composition/plans/hypothesis/05-when-in-the-run-it-matters.md) ◑ | established that the correction is what composes on SDXL, which is the result this plan asks the cause of |
| 27 | [hypothesis-04: does-the-corrector-compose-in-the-same-window](04-does-the-corrector-compose-in-the-same-window.md) ✅ | owns the SDXL counts this plan quotes: joint prompt 8 of 8, plain product 0 of 8 |
| **77 (current)** | **hypothesis-14: does-a-clip-free-model-hybridise** ⚠️ | **runs the same product on a model with no CLIP in it, and asks whether the fusion survives the swap** |

Design only. Verdicts and run state live in
[the paired review file](../../review/14-does-a-clip-free-model-hybridise.md).

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks) — things for Claude to execute
- [Instructions](#instructions) — things for you to do manually
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## What this asks, in one line

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

Ask PixArt-alpha for a cat and a dog by the same arithmetic SDXL gets asked, and count how many of
eight renders hold two clear animals.

## Quick context: where you are

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**What the project has already shown.** Summing two prompts' guidance directions on SDXL and
sampling the sum gives one fused animal, not two. Over the eight held-out seeds of cat × dog the
plain product composes 0 of 8 while the joint prompt composes 8 of 8, from
[the corrector finding](../../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md).

**What has never been checked.** Every render in that result came from SDXL, which conditions on
two CLIP text encoders. Nothing in the project has yet asked whether the fusion is a property of
the arithmetic or a property of what CLIP hands the model.

**Why PixArt-alpha and not Stable Diffusion 3.** PixArt-alpha's only text encoder is Flan-T5-XXL
and it carries no CLIP in any role. Stable Diffusion 3 carries CLIP-L, CLIP-G and T5 together, so
a result there cannot separate them, and the project's existing
[SD 3.5 renderer](../../../../../5-text-embedding-third-expert/scripts/sd35_and_vector_render.py) already works on that mixed
conditioning rather than against it.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**The two models differ on more than CLIP.**

Swapping SDXL for PixArt-alpha changes the text encoder, the denoiser architecture, the sampler,
the step count and the guidance scale all at once. This is a mixed comparison by
[the one-axis rule](../../../../../../../.claude/EXPERIMENT_CONVENTIONS.md), and the plan does not
pretend otherwise. It is worth running anyway because of what each outcome licenses: a fusion on
PixArt-alpha removes CLIP as a *necessary* cause, which is a one-directional claim a mixed
comparison can still support. Two clear animals on PixArt-alpha licenses nothing, because any of
the other four differences could be responsible.

**The settings are PixArt-alpha's own, not SDXL's.**

20 steps at guidance 4.5 on DPM-Solver++ order 2, read from the installed pipeline's signature
rather than carried over from SDXL's 50 steps at 7.5. Matching SDXL's settings instead would make
the comparison no cleaner (the sampler and architecture still differ) and would run the model
outside the configuration it was released in.

**The by-eye calls and the SDXL counts were made by different instruments.**

The SDXL counts come from the repo's GroundingDINO instance count, which
[cannot tell a cat from a dog](../../../../artifacts/results/can-we-trust-the-compose-score/README.md)
and so counts animals rather than naming them. The PixArt calls are made by eye, in three
categories, because a fused animal and two animals of the same species are both things the count
scorer reads identically. The card states which number came from which instrument.

**A call at contact-sheet scale is not a call.**

Two renders that read as a clear cat and a clear dog in a downscaled sheet read as hybrids at full
resolution. Every call in this plan is made from the full-resolution tile, and the sheet exists for
the reader rather than for the counting.

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Written before the runs.** Pre-registered in
[the review file](../../review/14-does-a-clip-free-model-hybridise.md).

The product of "a cat" and "a dog" on PixArt-alpha produces one animal, not two, on most of eight
seeds, and CLIP is therefore not necessary for the fusion.

- **Support:** 2 or fewer of 8 product renders hold two clear animals, while the joint-prompt
  column holds two clear animals on 6 or more of 8. CLIP is removed as a necessary cause.
- **Null:** 6 or more of 8 product renders hold two clear animals. The arithmetic composes on a
  CLIP-free model, and the SDXL failure is not explained by the arithmetic alone. This outcome
  names no cause, because four other things changed with the model.
- **Inconclusive:** anything between 3 and 5 of 8, or a joint-prompt column that itself holds two
  clear animals on fewer than 6 of 8, which would mean the model cannot draw the pair at all and
  the product column says nothing about composition.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

A reviewer reading the paper's central failure asks what in the model causes it, and the [text
encoder](../../../../../../../goal-setting/learning/deep-learning/diffusion-models/text-embedding-arithmetic-in-sdxl/MASTER_PLAN.md) is the first suspect anyone names: two CLIP encoders trained on short captions, pooled into
one vector, asked to carry two subjects. If that suspicion is right, the paper's correction is a
patch on a text-embedding problem and the right fix is a better encoder. If the fusion survives on
a model with no CLIP anywhere, that reading is closed off and the failure sits in the arithmetic or
in the sampler instead, which is where the rest of this scope already looks.

The cost is one download and about ten minutes of GPU time, against a question that would otherwise
be answered in the rebuttal with an argument rather than a figure.

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

**One script**, `1-lora-correction/scripts/pixart_multi_poe.py`, standalone beside the SDXL and SD 3.5 versions and
touching neither. It encodes each prompt and the empty prompt through Flan-T5-XXL, runs the
PixArt-alpha denoiser once per condition per step, and combines them as

```
eps = eps_empty + gs * sum_i (eps_i - eps_empty)
```

which is the pipeline's own classifier-free guidance generalised from one condition to many. Four
columns per seed: "a cat" alone, "a dog" alone, the product of the two, and the joint prompt "a cat
and a dog".

**One check before any product render.** The script run with a single condition must reproduce what
`PixArtAlphaPipeline` produces for the same prompt, seed, dtype and settings. The plan stops there
and shows the comparison, because a composition script that cannot reproduce the plain output would
produce hybrids that mean nothing.

**One sheet, one calls file, one card** under
`1-lora-correction/artifacts/results/does-a-clip-free-model-hybridise/`, built with the existing
`and_vector_sheet.sheet` so the columns are labelled the way every other sheet in the repo is.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Goal.** Close off, or fail to close off, the reading that CLIP's text embeddings cause the
fusion.

**Definition of done.** The card at
`1-lora-correction/artifacts/results/does-a-clip-free-model-hybridise/README.md` states how many of eight PixArt-alpha
product renders hold two clear animals, in three categories, with the SDXL counts quoted from the
owning finding beside them, and says what the comparison does and does not settle.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- `sentencepiece` was absent from `co3` and the T5 tokenizer cannot load without it. Installed
  2026-11-04, version 0.2.2.
- The weights live under `/datasets/mmolefe/poe_repair_min/models/PixArt-XL-2-1024-MS`, 21.8 GB:
  the T5 encoder in two shards at 19.05 GB, the denoiser at 2.45 GB, the decoder at 0.33 GB.
  `/datasets` is the only filesystem large files go to, per
  [environment/storage.md](../../../../../environment/storage.md), and both the fetch and the render
  guard that mount rather than an assumed one.
- Renders and the sheet write under `1-lora-correction/artifacts/results/does-a-clip-free-model-hybridise/`, which is
  small enough for the repo side; nothing in this plan writes a checkpoint.
- The models run in fp16, including the identity check's reference, because an fp32 against fp16
  comparison of the same loop differs by a few grey levels on its own and would be read as a
  plumbing fault.
- The session node reached by `scripts/cluster.sh` is `mscluster84`, an RTX 3090 with 24 GB. The T5
  encoder is 9.5 GB in fp16 and is freed after the four prompts are encoded, so the render loop
  holds only the 1.2 GB denoiser and the decoder.
- `sinfo` never shows GPU memory here, so node capability comes from
  [environment/hpc/nodes.md](../../../../../environment/hpc/nodes.md), not from Slurm.
- Never run a broad `find` across the cluster filesystem; it hangs on the mounts and the job is
  killed.

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/14-does-a-clip-free-model-hybridise.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against `context/`, `environment/`, `runbook/`,
      `report/`, and the learning journeys that name this project.
  - Paste: `/xref-pillar @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/14-does-a-clip-free-model-hybridise.md`
  - Done when: the scan comes back with no candidates, or its proposed links have been applied.

▶ **Next: [task 1.1](#1--the-composition-script)**.

### 1. 🔨 The composition script

- [ ] **1.1** Write `1-lora-correction/scripts/pixart_multi_poe.py`: encode each condition and the empty prompt,
      batch one denoiser pass per condition per step, combine as the empty prediction plus the
      guidance scale times the summed differences, step the scheduler once, decode.
  - Mirror [the SD 3.5 renderer](../../../../../5-text-embedding-third-expert/scripts/sd35_and_vector_render.py)'s shape, which runs
    the same sum on a non-SDXL model, rather than the SDXL script's hand-rolled DDIM.
  - Three things the pipeline does that the script must do too: keep only the first four of the
    denoiser's eight output channels, pass the prompt attention mask alongside the embeddings, and
    build the resolution and aspect-ratio conditioning the 1024 model expects.
  - Defaults are PixArt-alpha's own, 20 steps at guidance 4.5, and are printed into every sidecar.
  - A disk guard reads the filesystem the output path resolves to, not `/datasets` by assumption.
  - **Done when:** the file compiles and the three formulas (SDXL, SD 3.5, PixArt) read as the same
    arithmetic side by side.
- [ ] **1.2** Fetch the weights to `/datasets/mmolefe/poe_repair_min/models/PixArt-XL-2-1024-MS`
      with a disk guard on that mount, and load the tokenizer to prove `sentencepiece` took.
  - **Done when:** the four subfolders are on disk at 21.8 GB and `T5Tokenizer` loads from the
    local path with no network access.

▶ **Next: [task 2.1](#2--the-single-prompt-identity-check)**.

### 2. 🚀 The single-prompt identity check

◀ **Needs: [task 1.2](#1--the-composition-script)**, the weights on disk.

- [x] **2.1** Render "a cat" at seed 1 twice: once through `PixArtAlphaPipeline` as released, once
      through the new script with one condition, same dtype, same scheduler, same 20 steps at 4.5.
  - Report the largest absolute difference between the two images in 0-to-255 pixel values, and the
    fraction of pixels that differ at all.
  - **Done when:** the two numbers are in the review file. A largest difference of 0 means the two
    paths are the same computation. Anything above about 2 grey levels is a plumbing fault and task
    3 does not run.
  - ✅ Largest absolute difference 0 grey levels, 0.0 of pixels differing, and the two PNGs are
    byte-identical (md5 `d9c08c28048b8da98124be0246099b07`). Ran 2026-11-04 on mscluster84, an RTX
    3090, from `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-clip-free` at commit
    `2506ea21`; text encoder `T5EncoderModel` at 4.8B parameters, scheduler
    `DPMSolverMultistepScheduler`. Images and record in
    `/datasets/mmolefe/poe_repair_min/outputs/clip_free_hybridise/`.

▶ **Next: [instruction 3.1](#3--pass-the-identity-check)**.

### 4. 🚀 The four columns on eight seeds

◀ **Needs: [instruction 3.1](#3--pass-the-identity-check)**, your yes on the identity check.

- [x] **4.1** Render all four columns on seeds 1 to 8, 32 tiles at 1024², keeping the
      full-resolution PNGs and a sidecar each.
  - **Done when:** 32 tiles exist and every sidecar names the same 20 steps and guidance 4.5.
  - ✅ 32 tiles under `/datasets/mmolefe/poe_repair_min/outputs/clip_free_hybridise/a_cat__x__a_dog/`,
    all 32 sidecars reading 20 steps, guidance 4.5, 1024².
- [x] **4.2** Build the four-column sheet at
      `1-lora-correction/artifacts/results/does-a-clip-free-model-hybridise/pixart-cat-dog-sheet.png` through
      `and_vector_sheet.sheet`, rows the eight seeds, columns cat alone, dog alone, the product,
      the joint prompt.
  - **Done when:** the sheet is on disk and its own JSON record names every tile it drew.
  - ✅ `1-lora-correction/artifacts/results/does-a-clip-free-model-hybridise/pixart-cat-dog-sheet.png` with its JSON
    record beside it.

▶ **Next: [task 5.1](#5--the-calls-and-the-card)**.

### 5. 🖼️ The calls and the card

◀ **Needs: [task 4.2](#4--the-four-columns-on-eight-seeds)**, the tiles and the sheet.

- [x] **5.1** Open each of the eight product tiles at full resolution and place it in one of three
      categories: two clear animals, one hybrid animal, or one single animal. Write one record per
      tile to `1-lora-correction/artifacts/results/does-a-clip-free-model-hybridise/calls.json` with its seed, its
      category and one sentence saying what is in the picture.
  - A hybrid is its own category, not a failed version of either other one. Do not call a clear pair
    from the sheet.
  - **Done when:** eight records exist and the three counts sum to eight.
  - ✅ 0 two clear animals, 6 one hybrid animal, 2 one single animal, in `calls.json`, with the
    joint column read the same way at 7 of 8.
- [x] **5.2** Write the card at
      `1-lora-correction/artifacts/results/does-a-clip-free-model-hybridise/README.md`: the model, its text encoder,
      the guidance scale and step count used and that they are PixArt-alpha's own defaults rather
      than SDXL's, the three counts out of eight, the SDXL counts quoted from
      [the corrector finding](../../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md),
      and what the comparison does and does not settle.
  - **Done when:** the card answers the definition of done above, and names the two instruments.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

### Close out. 🔄 Record what this plan taught

- [ ] **6.1** Answer the review file against the bar written in it, and write the finding under
      `1-lora-correction/report/is-the-gap-the-samplers-or-the-models/` if the verdict is support or null.
- [ ] **6.2** File the sheet, the calls and the card under `artifacts/results/` with the card entry
      and one reference line in this plan's Figure Catalog.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 3. 👁️ Pass the identity check

◀ **Needs: [task 2.1](#2--the-single-prompt-identity-check)**, the two numbers.

- [x] **3.1** Look at the two "a cat" renders side by side and at the two numbers beneath them.
  - Expected result: one picture, printed twice, and a largest difference of 0 grey levels.
  - ❌ If the pictures differ anywhere a person can see, the composition path is not the pipeline's
    path and the product renders would measure the difference rather than the composition. Say stop
    rather than accepting a close-enough match.
  - ✅ Say `done` with what you saw, which is what releases task 4.
  - ✅ Passed 2026-11-04: the pair read as one picture, and the two files are byte-identical.

▶ **Next: [task 4.1](#4--the-four-columns-on-eight-seeds)**.

### 7. 👁️ Read the sheet and judge the calls

◀ **Needs: [task 5.2](#5--the-calls-and-the-card)**, the sheet and the card.

- [ ] **7.1** Open the four-column sheet and read the product column against the joint-prompt
      column.
  - Expected result: the joint-prompt column holds two animals on most rows. Whatever the product
    column holds is the result.
- [ ] **7.2** Disagree with any call in `calls.json` you think is wrong, by seed.
  - The counts in the card are rewritten from your reading, not argued with.

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
# the tokenizer loads, which is what sentencepiece was missing for
$PY -c "import sentencepiece, diffusers; from diffusers import PixArtAlphaPipeline; print(diffusers.__version__)"
# expect: 0.29.2

# the weights are all there
du -sh /datasets/mmolefe/poe_repair_min/models/PixArt-XL-2-1024-MS
# expect: about 22G

# the mount the renders go to has room
df -h /datasets
# expect: under 90% used
```

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

### Generated during execution

| File | What it shows |
|---|---|
| `1-lora-correction/artifacts/results/does-a-clip-free-model-hybridise/pixart-cat-dog-sheet.png` | Rows are seeds 1 to 8, columns are "a cat" alone, "a dog" alone, the product of the two, and the joint prompt "a cat and a dog", all on PixArt-alpha at 20 steps and guidance 4.5. What to notice: whether the third column holds one animal or two, and whether the fourth column holds two. |

## Code references

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- `1-lora-correction/scripts/pixart_multi_poe.py`, this plan's only new code.
- [the SDXL composition](../../../../scripts/concept_multi_poe.py), the same sum on SDXL, untouched
  by this plan.
- [the SD 3.5 renderer](../../../../../5-text-embedding-third-expert/scripts/sd35_and_vector_render.py), the same sum on a non-SDXL
  model, which this script's shape follows.
- `5-text-embedding-third-expert/scripts/and_vector_sheet.py`, whose `sheet` builds the labelled column sheet.

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

Nothing waits on this plan. A null here (the product composing on PixArt-alpha) would be the one
outcome that opens new work, because it would mean the arithmetic is not sufficient to cause the
fusion and the next question is which of the other four differences carries it.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From project catalog

- `poe-disk-001`: a guard that reads a different mount from the one written to. Both the fetch and
  the render resolve their output path first and guard that.
- `poe-render-001`: see [environment/known-failures.md](../../../../../environment/known-failures.md).

### New to this plan

#### 🟡 the tokenizer loads but produces different tokens than the pipeline

PixArt-alpha's pipeline cleans captions before tokenising, which needs `ftfy` and
`beautifulsoup4`. Both are present in `co3`. If either were missing the pipeline would warn and
skip the cleaning while the script did it, so the two paths would tokenise different text and the
identity check would fail for a reason that has nothing to do with the arithmetic. The check at
task 2.1 catches this, and the fix is to match the pipeline's `clean_caption` setting rather than
to loosen the comparison.

#### 🟡 the scheduler's state is carried between columns

DPM-Solver++ at order 2 keeps the previous step's prediction, so a scheduler reused across columns
without being reset would make a column depend on the one rendered before it. Every column
re-runs `set_timesteps` before its loop, and a column rendered alone must match the same column
rendered in a batch.
