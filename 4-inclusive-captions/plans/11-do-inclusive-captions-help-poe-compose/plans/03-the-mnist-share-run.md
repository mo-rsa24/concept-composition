# 🧪 The MNIST share run: with the share of two-digit pictures set directly, does the product of 7 and 8 draw two digits?

## Recommended prompt

Walk this plan with `/drip-execute-plan 4-inclusive-captions/plans/11-do-inclusive-captions-help-poe-compose/plans/03-the-mnist-share-run.md`.

## Position in the plan tree

Plan 03 of scope 11. It reruns [the MNIST stage](02-the-mnist-stage.md) with the setting named by what it
does to the training data. Plan 02's models, samples and NULL verdict stay as the record of the dial run.
Its 0% models are reused here, because 0% share is the same training distribution as dial 0.

## Table of contents

- [Quick context](#quick-context)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Success and failure outcomes](#success-and-failure-outcomes)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Outputs](#outputs)
- [Figure Catalog](#figure-catalog)
- [Error Matrix](#error-matrix)
- [Orchestration](#orchestration)
- [References](#references)
- [Next step](#next-step)

## Quick context

Navigation: 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

Plan 02 set a dial: the weight of the single labels against the joint label on two-digit canvases. The
share of "7" pictures that also hold an 8 followed from it (0%, 4.2%, 8.9%, 12.4%, 14.7%), could not
pass about 21%, and the joint label "7 and 8" was trained less as the dial rose. This run sets the share
directly and holds everything else fixed.

**The label is picked first, from one fixed mix used at every share:** blank 10%, the 40 seen joint
labels 9% ("7 and 8" 4.5%, the other 39 split evenly), single digits 81% at MNIST's own digit
frequencies. **Then the canvas:** a draw labelled with digit d holds d beside a second digit with
probability s and d alone otherwise; the second digit is 8 for "7", 7 for "8", and uniform over d's seen
partners for every other digit. A joint label "a and b" always shows a and b; a blank draw shows a
canvas from plan 02's dial-0 mix. The five held-out pairs never share a canvas.

**The shares:** s = 0, 10, 20, 40 and 80%, three trainings each. At s = 0 this is plan 02's dial-0
distribution, so its three models are reused; 12 new models train. Everything else is plan 02's:
768,000 canvases per model built fresh (6,000 steps of 128), the width-48 U-Net, guidance 3, the
corrector, 256 seeds, the eight rows, the judge pinned by its sha256.

The question this answers: once each single label has seen the two digits together on s% of its
pictures, and the joint label is trained exactly as before, does the product of "7" and "8" draw two
separate right digits?

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#success-and-failure-outcomes) ➡️

- Trains and draws on the cluster's `bigbatch` partition (RTX 3090, 24 GB), one job per model, at most six
  running at once (QOS `mss_bigbatch`, `MaxJobsPU=6`), with the `co3` python at
  `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`. Idle `bigbatch` nodes often carry a dead GPU:
  probe with `nvidia-smi` and pin a healthy node with `--nodelist` ([nodes](../../../../environment/hpc/nodes.md)).
- MNIST's raw files are copied to `/datasets/mmolefe/raw/mnist/` and read through the
  `MNIST_RAW` environment variable; the laptop keeps its default path. Not yet recorded in `environment/`:
  task 1.3 records it.
- Checkpoints go to `/datasets/mmolefe/poe_repair_min/outputs/mnist-share/checkpoints/`, never `/home-mscluster`
  ([storage](../../../../environment/storage.md)); samples to the same folder's `samples/`, and come back to
  the laptop with `scripts/cluster.sh pull`-style rsync for scoring.
- Scoring runs on the laptop, where the pinned judge lives (`data/mnist/judge`).
- Measured on the laptop's 4070 in plan 02: 12.4 to 15.5 min of training and 1,974 to 2,319 s of drawing per
  model. The 3090's times are measured on the first job.

## Success and failure outcomes

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

The bar is written as named constants in `poe_repair/experiments/inclusive_captions/mnist/share_bar.py`
before any share model trains. All counts are of 256 seeds, read by the pinned judge.

| Verdict | What it takes |
|---|---|
| a share counts | "7" alone and "8" alone each draw their lone digit on at least 95% of what training implies, round(0.95 × 256 × (1 − s)): 243 at 0%, 219 at 10%, 195 at 20%, 146 at 40%, 49 at 80%, in all three trainings |
| supported | at a counting share of 10, 20, 40 or 80%, the corrected 7 × 8 product draws two separate right digits on at least 128 of 256 in all three trainings, while 0% stays at 12 or fewer in all three |
| null | at no counting share does the corrected product's three-training mean reach 39 of 256 |
| inconclusive | anything else |

The judge's reading holds at a share only if, on one 64-tile sheet of training 1's corrected 7 × 8
product, it agrees with blind eye calls on at least 58 of 64. Where it does not, that share is read by eye
on the first 64 seeds of each training, with the same thresholds scaled to 64: 32 to support, a mean of
9.75 for null.

Reported beside the verdict, not deciding it: the plain product; "7 and 8" (trained the same at every
share, so it should stay near plan 02's 151 to 160 of 256); the corrected 4 × 7 product, a pair never
seen together; and, at 80%, how often "7" alone itself draws a 7 beside an 8, since a two-digit product
there can come from the single labels drawing pairs on their own.

What would surprise: the plain product reaching 128 at 10%, or "7 and 8" moving far from 151.

## Tasks

*For Claude to execute.*

Navigation: ⬅️ [Previous](#success-and-failure-outcomes) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 1. 🔧 The share canvas maker and the bar

- [ ] 1.1 `share_bar.py` holds the table above as constants, committed before any share model trains.
- [ ] 1.2 `canvases.py` gains `ShareCanvasMaker(share, ...)`, leaving `CanvasMaker` unchanged, and reads
  `MNIST_RAW` from the environment.
  Check: `python -I poe_repair/experiments/inclusive_captions/mnist/canvases.py --share` prints, at each
  share, blank 0.10, joint 0.09, single 0.81, the share of "7" draws holding an 8, of "8" draws holding a
  7, and of "4" draws holding a second digit, each within 0.005 of s, and asserts no held-out pair shares a
  canvas.
- [ ] 1.3 MNIST's four raw files copied to `/datasets/mmolefe/raw/mnist/`, checked by sha256
  against the laptop's, and the fact added to `environment/storage.md`.

▶ **Next: [task 2](#2--the-trainer-and-the-job)**.

### 2. 🔧 The trainer and the job

- [ ] 2.1 `train.py --share s --only-seed t --out <dir>` trains one share model and saves
  `share-<s>-seed-<t>.pt`; `sample.py --ckpt <path> --out <dir>` draws its eight rows unchanged.
- [ ] 2.2 `4-inclusive-captions/scripts/mnist_share.sbatch` trains then draws one model, with the disk guard on `/datasets`, the
  `co3` path check and a working-GPU check.
  Check: one job at share 10%, seed 1, finishes and prints its training and drawing times.

▶ **Next: [task 3](#3--the-twelve-models)**.

### 3. 🚀 The twelve models

- [ ] 3.1 Submit the remaining 11 jobs, at most six running, on probed healthy nodes.
  Check: 12 checkpoints and 12 sample files under `/datasets/mmolefe/poe_repair_min/outputs/mnist-share/`.

▶ **Next: [task 4](#4--scores-and-verdict)**.

### 4. 📊 Scores and verdict

- [ ] 4.1 Pull the 12 sample files to `data/mnist/share-samples/`, then
  `python -I poe_repair/experiments/inclusive_captions/mnist/score_share.py` calls every tile, writes
  `scores.csv` and `counts.json`, and prints SUPPORTED, NULL or INCONCLUSIVE with every share's counts.
- [ ] 4.2 The 64-tile judge-against-eye sheets, one per share (training 1, corrected 7 × 8), with blind eye
  calls; the agreement count is written into `counts.json`.

▶ **Next: [instruction 5](#5--tick-one-verdict-line)**.

## Instructions

*For you to do by hand.*

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

### 5. ✅ Tick one verdict line

◀ **Needs: [task 4](#4--scores-and-verdict)** done.

- [ ] **Do:** open the results figure and `score_share.py`'s printout side by side, overrule any eye call
  you disagree with, and tick supported, null or inconclusive in the table above.
- **Done when:** one line is ticked with the deciding counts beside it.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

| Output | Where |
|---|---|
| code | `poe_repair/experiments/inclusive_captions/mnist/` (`share_bar.py`, `canvases.py`, `train.py`, `score_share.py`), `4-inclusive-captions/scripts/mnist_share.sbatch` |
| checkpoints, cluster samples | `/datasets/mmolefe/poe_repair_min/outputs/mnist-share/` |
| samples for scoring | `data/mnist/share-samples/` (ignored by git) |
| scores, counts, card | `4-inclusive-captions/artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/` |

## Figure Catalog

Navigation: ⬅️ [Previous](#outputs) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

### Pending

| Figure | Lane | Save to |
|---|---|---|
| one row per share: the share, what training saw, eight product samples, the count of 256 (the layout of the stage's r6) | subject | `4-inclusive-captions/artifacts/scenes/how-the-inclusive-labels-mnist-stage-works/figures/` |
| the product's count of 256 against the share, three trainings each, plain and corrected, with the 128 line | subject | same |

### Generated during plan execution

None yet.

## Error Matrix

Navigation: ⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#orchestration) ➡️

| Pattern | What it looks like here | Fix |
|---|---|---|
| a dead `bigbatch` GPU | the job's GPU check aborts it in seconds | probe idle nodes with `nvidia-smi`, pin a healthy one |
| `MaxJobsPU` | the seventh job waits with reason `QOSMaxJobsPerUserLimit` | expected; it starts when one finishes |
| output on `/home-mscluster` | checkpoints fill the home filesystem | write to `/datasets` only; the disk guard checks the filesystem written to |

## Orchestration

Navigation: ⬅️ [Previous](#error-matrix) | 📋 [TOC](#table-of-contents) | [Next](#references) ➡️

| Command | Triggered by | Outcome |
|---|---|---|
| `scripts/cluster.sh run "sbatch --nodelist=<node> 4-inclusive-captions/scripts/mnist_share.sbatch ..."` | Claude, task 2.2 and 3.1 | one model trained and drawn per job |
| `/sync-plan-tree 4-inclusive-captions/plans/11-do-inclusive-captions-help-poe-compose/` | the close | statuses aggregated |

## References

Navigation: ⬅️ [Previous](#orchestration) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [The MNIST stage](02-the-mnist-stage.md), its results card and figures under
  `4-inclusive-captions/artifacts/scenes/how-the-inclusive-labels-mnist-stage-works/figures/`.
- [The toy relabelling experiment, explained](../../../artifacts/scenes/how-the-toy-relabelling-experiment-works/the-toy-relabelling-experiment-explained.md), whose replacing design this run follows.

## Next step

Navigation: ⬅️ [Previous](#references) | 📋 [TOC](#table-of-contents)

Task 1.1, the bar.
