# 🧪 Does SDXL still fuse when it is sampled the way PixArt-alpha is?

This plan asks whether the cat-and-dog fusion on SDXL survives at PixArt-alpha's sampler, step
count and guidance scale. It narrows the mixed comparison
[hypothesis-14](14-does-a-clip-free-model-hybridise.md) left behind, where five things changed at
once.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/16-does-sdxl-compose-at-pixarts-settings.md — <how many of eight composed at 20 steps and 4.5>
```

## Recommended skill

▶ `/pair-figure` ✅ the two sheets, SDXL at its own settings and SDXL at PixArt's, are read against
   each other row by row, and deciding how they sit side by side is that skill's job.
   alt: `/analyze-figure` on the finished sheet first, if the by-eye calls turn out to be close.

## Position in the plan tree

**Step 79 of 79.** Waits on nothing; the tiles it needs are rendered by its own script. The one
order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 27 | [hypothesis-04: does-the-corrector-compose-in-the-same-window](04-does-the-corrector-compose-in-the-same-window.md) ✅ | owns the SDXL counts at SDXL's own settings: joint prompt 8 of 8, plain product 0 of 8 |
| 77 | [hypothesis-14: does-a-clip-free-model-hybridise](14-does-a-clip-free-model-hybridise.md) ✅ | fused on PixArt-alpha 8 of 8, which removed CLIP but left five differences uncut |
| **79 (current)** | **hypothesis-16: does-sdxl-compose-at-pixarts-settings** ⚠️ | **cuts four of those five by sampling SDXL the way PixArt is sampled** |

Design only. Verdicts and run state live in
[the paired review file](../../review/16-does-sdxl-compose-at-pixarts-settings.md).

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

Ask SDXL for a cat and a dog by the same arithmetic, at 20 steps of DPM-Solver++ and guidance 4.5
instead of 50 steps of DDIM at 7.5, and count how many of eight renders hold two clear animals.

## Quick context: where you are

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**What the project has already shown.** On SDXL at 50 steps of DDIM and guidance 7.5, the product
of "a cat" and "a dog" composes 0 of 8 held-out seeds while the joint prompt composes 8 of 8, from
[the corrector finding](../../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md).
On PixArt-alpha at 20 steps of DPM-Solver++ and guidance 4.5 the product composes 0 of 8 while the
joint prompt holds a clear pair on 7 of 8, from
[does a clip-free model hybridise](../../../../report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md).

**What those two cannot say together.** The second run swapped five things at once: the text
encoder, the denoiser, the sampler, the step count and the guidance scale. The fusion surviving the
swap removed CLIP as a necessary cause, and named no cause.

**What this run changes.** One axis, on one model. Same SDXL, same CLIP, same cached starting
noise, same eight held-out seeds; only the sampler, the step count and the guidance scale move to
PixArt's values. Those three travel together because they are one choice, "sample it the way PixArt
is sampled", and separating them is a later question nobody has asked yet.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Three settings move together, so the comparison is one axis with three parts.**

A fusion at the new settings kills all three at once as the carrier. Composition at the new
settings says one of the three matters without saying which, and that is a follow-up, not a defect
of this run.

**The control is this script at the old settings, not the published number.**

The published 0 of 8 came from a different script. Running the new script at DDIM, 50 steps and 7.5
and getting the same renders is what proves the new code path is not the thing that changed. This
is the gate, and the product renders at the new settings do not run until it passes.

**Lower guidance weakens every prompt's push, so the joint column is the control again.**

At 4.5 instead of 7.5 a fusion could be a weak-guidance artefact. The joint prompt at the same 4.5
drawing a clean pair is what rules that out, exactly as it did on PixArt.

**The calls are made by eye, at full resolution.**

The repo's instance-count scorer
[cannot tell a cat from a dog](../../../../artifacts/results/can-we-trust-the-compose-score/README.md),
so it reads a hybrid and two cats identically. A call at contact-sheet scale is not a call.

**The existing SDXL code paths are not modified.**

`scripts/concept_multi_poe.py` and `scripts/and_vector_sheet.py` pin DDIM, 50 steps and 7.5, and
every published number came out of them. The new settings live in a new script beside them.

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Written before the runs.** Pre-registered in
[the review file](../../review/16-does-sdxl-compose-at-pixarts-settings.md).

The product of "a cat" and "a dog" on SDXL still produces one animal when SDXL is sampled at
PixArt-alpha's settings, so the sampler, the step count and the guidance scale are not what carries
the fusion.

- **Support:** 2 or fewer of 8 product renders hold two clear animals, while the joint-prompt
  column at the same settings holds two on 6 or more of 8. The three sampling settings are removed
  as the cause, leaving the text encoder and the denoiser.
- **Null:** 6 or more of 8 product renders hold two clear animals. One of the three sampling
  settings carries the failure, the published 0 of 8 is a property of how SDXL was sampled rather
  than of the arithmetic, and the next question is which of the three.
- **Inconclusive:** anything between 3 and 5 of 8, or a joint-prompt column below 6 of 8, which
  would mean SDXL cannot draw the pair at these settings at all.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

The paper argues that summed-guidance composition drops a term the joint prompt carries. A reviewer
can answer that in one line: you ran it at 50 DDIM steps and 7.5, and the failure is an artefact of
that configuration. Nothing in the project currently answers them, because every SDXL number comes
from exactly those settings.

It costs no download and about half an hour of one GPU, and it either closes the objection or
reveals that a cheap settings change does what the trained correction does, which would be the more
important of the two outcomes.

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

**One script**, `scripts/sdxl_matched_multi_poe.py`, beside the pinned ones and touching neither. It
takes the sampler, the step count and the guidance scale as arguments, encodes each prompt and the
empty prompt through SDXL's two CLIP encoders, runs one denoiser pass per condition per step, and
combines them as

```
eps = eps_empty + gs * sum_i (eps_i - eps_empty)
```

Four columns per seed: "a cat" alone, "a dog" alone, the product, the joint prompt. The starting
latent is the cached x_T every other SDXL result in this repo starts from.

**One control run before the matched run.** The same script at DDIM, 50 steps and guidance 7.5 on
the same seeds, whose product column must read 0 of 8, matching the published number.

**One sheet, one calls file, one card** under
`artifacts/results/does-sdxl-compose-at-pixarts-settings/`, built with `and_vector_sheet.sheet`.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Goal.** Remove the sampler, the step count and the guidance scale from the list of things that
could be causing the fusion, or discover that they are it.

**Definition of done.** The card at
`artifacts/results/does-sdxl-compose-at-pixarts-settings/README.md` states how many of eight SDXL
product renders at 20 steps of DPM-Solver++ and guidance 4.5 hold two clear animals, in three
categories, beside the same count from the control run at SDXL's own settings and the PixArt counts
from the owning finding.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- SDXL's weights and the cached starting latents are already on the cluster; this plan downloads
  nothing.
- Renders write under `artifacts/results/does-sdxl-compose-at-pixarts-settings/`, and the script's
  disk guard reads the filesystem that path resolves to rather than an assumed one.
- The session node reached by `scripts/cluster.sh` is `mscluster84`, an RTX 3090 with 24 GB, which
  holds SDXL at fp16 with room for four conditions in one batch.
- `scripts/cluster.sh run` refuses to sync when the branch is checked out in another worktree on
  the cluster, which happens whenever a parallel session has it. Give this plan's branch its own
  cluster worktree and launch with `CLUSTER_DIR` pointed at it.
- Never run a broad `find` across the cluster filesystem; it hangs on the mounts and the job is
  killed.

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/16-does-sdxl-compose-at-pixarts-settings.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against `context/`, `environment/`, `runbook/`,
      `report/`, and the learning journeys that name this project.
  - Paste: `/xref-pillar @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/16-does-sdxl-compose-at-pixarts-settings.md`
  - Done when: the scan comes back with no candidates, or its proposed links have been applied.

▶ **Next: [task 1.1](#1--the-settings-free-composition-script)**.

### 1. 🔨 The settings-free composition script

- [x] **1.1** Write `scripts/sdxl_matched_multi_poe.py`: the same summed guidance as
      `concept_multi_poe.py`, with `--sampler`, `--steps` and `--guidance` as arguments and all four
      columns per seed.
  - DDIM keeps the repo's hand-rolled step so the control run matches the published renders;
    DPM-Solver++ uses the scheduler's own `step`, and its timesteps are reset per tile because an
    order-2 solver keeps state.
  - Every sidecar names the sampler, the step count and the guidance scale.
  - A disk guard reads the filesystem the output path resolves to.
  - **Done when:** the file compiles and its guidance line reads as the same arithmetic as
    `concept_multi_poe.py:66`.

▶ **Next: [task 2.1](#2--the-control-run-at-sdxls-own-settings)**.

### 2. 🚀 The control run at SDXL's own settings

◀ **Needs: [task 1.1](#1--the-settings-free-composition-script)**, the script.

- [ ] **2.1** Run the script at `--sampler ddim --steps 50 --guidance 7.5` on seeds 9 to 16, all
      four columns.
  - **Done when:** the product column reads 0 of 8 and the joint column 8 of 8, matching the
    published counts. Anything else means the new code path differs from the one those numbers came
    from, and the matched run does not start.

▶ **Next: [instruction 3.1](#3--pass-the-control-run)**.

### 4. 🚀 The matched run at PixArt's settings

◀ **Needs: [instruction 3.1](#3--pass-the-control-run)**, your yes on the control.

- [ ] **4.1** Run the script at `--sampler dpmpp --steps 20 --guidance 4.5` on the same seeds 9 to
      16, all four columns.
  - **Done when:** 32 tiles exist and every sidecar names DPM-Solver++, 20 steps and guidance 4.5.
- [ ] **4.2** Build the sheet at
      `artifacts/results/does-sdxl-compose-at-pixarts-settings/sdxl-matched-sheet.png` through
      `and_vector_sheet.sheet`, rows the eight seeds, columns cat alone, dog alone, the product,
      the joint prompt.
  - The tile root needs the `and_vector/<slug>/seedNN/` layout that helper reads.
  - **Done when:** the sheet is on disk with its JSON record beside it.

▶ **Next: [task 5.1](#5--the-calls-and-the-card)**.

### 5. 🖼️ The calls and the card

◀ **Needs: [task 4.2](#4--the-matched-run-at-pixarts-settings)**, the tiles and the sheet.

- [ ] **5.1** Open each of the eight product tiles at full resolution and place it in one of three
      categories: two clear animals, one hybrid animal, or one single animal. Write one record per
      tile to `artifacts/results/does-sdxl-compose-at-pixarts-settings/calls.json` with its seed,
      its category and one sentence saying what is in the picture. Read the joint column the same
      way.
  - **Done when:** eight records exist and the three counts sum to eight.
- [ ] **5.2** Write the card at
      `artifacts/results/does-sdxl-compose-at-pixarts-settings/README.md`: the settings of both
      runs, the three counts out of eight for each, the PixArt counts quoted from
      [its finding](../../../../report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md),
      and what the one-axis comparison settles.
  - **Done when:** the card answers the definition of done above.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

### Close out. 🔄 Record what this plan taught

- [ ] **6.1** Answer the review file against the bar written in it, and write the finding under
      `report/is-the-gap-the-samplers-or-the-models/`.
- [ ] **6.2** File the sheet, the calls and the card under `artifacts/results/` with the card entry
      and one reference line in this plan's Figure Catalog.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 3. 👁️ Pass the control run

◀ **Needs: [task 2.1](#2--the-control-run-at-sdxls-own-settings)**, the control counts.

- [ ] **3.1** Look at the control run's product column beside the published renders of the same
      seeds.
  - Expected result: one animal on all eight, the same eight pictures as before.
  - ❌ If they differ, the new script is not the old code path and the matched run would measure
    the script rather than the settings. Say stop.
  - ✅ Say `done` with what you saw, which is what releases task 4.

▶ **Next: [task 4.1](#4--the-matched-run-at-pixarts-settings)**.

### 7. 👁️ Read the sheet and judge the calls

◀ **Needs: [task 5.2](#5--the-calls-and-the-card)**, the sheet and the card.

- [ ] **7.1** Open the sheet and read the product column against the joint-prompt column.
- [ ] **7.2** Disagree with any call in `calls.json` you think is wrong, by seed.
  - The counts in the card are rewritten from your reading, not argued with.

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
# the script's two samplers both construct
$PY -c "from diffusers import DDIMScheduler, DPMSolverMultistepScheduler; print('schedulers ok')"
# expect: schedulers ok

# the cached starting latents for the seeds this plan uses are on disk
ls /datasets/mmolefe/poe_repair_min/outputs/interaction_term | head
# expect: the interaction-term tree, from which cell_from_slug resolves seeds 9 to 16

# the mount the renders go to has room
df -h /datasets
# expect: under 90% used
```

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

### Generated during execution

| File | What it shows |
|---|---|
| `artifacts/results/does-sdxl-compose-at-pixarts-settings/sdxl-matched-sheet.png` | Rows are held-out seeds 9 to 16, columns are "a cat" alone, "a dog" alone, the product of the two, and the joint prompt, all on SDXL at 20 steps of DPM-Solver++ and guidance 4.5. What to notice: whether the third column holds one animal or two, against the same column at SDXL's own 50 steps and 7.5, which holds one on all eight. |

## Code references

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- `scripts/sdxl_matched_multi_poe.py`, this plan's only new code.
- [the SDXL composition](../../../../scripts/concept_multi_poe.py), pinned to DDIM at 50 steps and
  7.5, untouched by this plan and the source of every published SDXL number.
- [the PixArt composition](../../../../scripts/pixart_multi_poe.py), the same sum on the CLIP-free
  model, whose settings this plan borrows.
- `scripts/and_vector_sheet.py`, whose `sheet` builds the labelled column sheet.

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

A null here (SDXL composing at PixArt's settings) opens the larger work: which of the three
settings carries it, and whether the trained correction is reproducing a sampler change. Support
leaves the text encoder and the denoiser as the two live candidates, and the next cut on that axis
is a second CLIP-free model with a different encoder family.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From project catalog

- `poe-disk-001`: a guard that reads a different mount from the one written to. The script resolves
  its output path first and guards that.
- `poe-launch-002`: a job that silently runs on the CPU. The script exits when no CUDA device is
  visible.

### New to this plan

#### 🟡 the two samplers do not start from the same latent scaling

The repo's DDIM path divides the cached x_T by the scheduler's sigma before stepping. DPM-Solver++
expects the same scaling, and a mismatch shows as a washed-out or over-saturated render rather than
as an error. The control run is what catches it: if the DDIM column reproduces the published
pictures, the scaling is the one those numbers used.

#### 🟡 the branch is checked out in another worktree on the cluster

`scripts/cluster.sh run` stops with `already used by worktree` whenever a parallel session holds
the branch. Add a cluster worktree for this plan's branch and pass `CLUSTER_DIR`, rather than
switching the shared checkout's branch underneath whoever is using it.
