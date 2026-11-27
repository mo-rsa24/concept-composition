# 🧪 Paper 1 setup: the code, every choice behind it, and the commands that run the first result

Loop 16 wrote the setup code for [Paper 1's setup plan](../../../../plans/paper-1-clevr-touching/plans/01-setup.md) on the laptop,
task by task, and Loop 18 ([the supervisor's page](supervisor.md)) reviewed and accepted each one. Nothing has run on the cluster,
nothing is committed and nothing is pushed. The code serves [Paper 1's first-result plan](../../../../plans/paper-1-clevr-touching/plans/02-first-result.md)
and the guardrails the owner adopted from [the guard note](../guarding-the-first-results/paper-1.md).

## Contents

- [Where it stands](#where-it-stands)
- [The files](#the-files)
- [Each task, its check and its accept](#each-task-its-check-and-its-accept)
- [Choices made where the plans were silent](#choices-made-where-the-plans-were-silent)
- [What the first runs are the first real test of](#what-the-first-runs-are-the-first-real-test-of)
- [Still open](#still-open)
- [The commands that run the first result](#the-commands-that-run-the-first-result)

## Where it stands

Tasks 0 to 7 of the setup plan are accepted by Loop 18 and ticked. Task 10's script, `paper1_model_check.py`, is written and
accepted, and its box stays open because the box covers running it. The empty-scene model was found (board question 121):
`/datasets/mmolefe/poe_repair_min/loop-6/m7/full/models/empty/network-snapshot-0002000-0.100.pkl`, 248,538,254 bytes, sha256
`5e7905efd3b723d79507dde0af9d5d63d959738a2e08ab234646101ece8fb272`.

The launch mode is the owner's: both papers' first results run on one held Blackwell node (mscluster110, one 96 GB card, `co3_bw`)
as two processes, one per paper, started as plain `bash` commands from the cluster checkout. The owner allocates the node and
pushes the branch.

Tests: 41 Paper 1 tests pass with a CPU build of torch kept outside the repo (`tests/test_paper1_*.py`, warnings as errors), and the
19 that need no torch pass on the laptop's own python.

## The files

| File | What it does |
|---|---|
| `2-classifier-guidance/scripts/paper1/paper1_sample.py` | the colour-expert run's Heun sampler ported unchanged; lone samples, mean empty scene, plain sum, paste and lower-on-top paste, the touching split per seed set, the region sampler, the push and the random push, contact kept, the weight grid, and the check sheets |
| `2-classifier-guidance/scripts/paper1/paper1_disc.py` | the head on the frozen experts' features: the block choice by cross-validation inside the 64 training scenes, training with shifts, a bounded logit, reweighted negatives, the overfitting gauge and the optional alignment loss; the check with the direction check and the depth-order preference |
| `2-classifier-guidance/scripts/paper1/paper1_sheets.py` | the blind sheets: select the seeds, build under neutral names, check nothing leaks, settle two reviewers with a third |
| `2-classifier-guidance/scripts/paper1/paper1_first_bars.py` | the first result's 13 bars as named constants; the reader that gives yes, no or inconclusive; `choose-w` for the weight and window from the owner's tuning calls |
| `2-classifier-guidance/scripts/paper1/paper1_pack.py` | the joint renders squashed to 64 pixels with `clevr_data.py`'s own `to_64`, with masks, touching flags, composition, the leakage check and the depth-order count |
| `2-classifier-guidance/scripts/paper1/paper1_model_check.py` | lists the three model files and checks their hashes; writes `models.txt` |
| `2-classifier-guidance/scripts/paper1/paper1_gpu.sbatch`, `paper1_train.sbatch`, `paper1_render.sbatch` | the job scripts, run as plain `bash` on the held node; each opens with the directory and disk guard, the python path by host and the GPU guard, and ends by copying the run's small files and any blind sheet to `outputs/paper1/<run>/` |
| `2-classifier-guidance/scripts/paper1/bigbatch-exclude.txt` | the `bigbatch` exclude list, for a later stage that goes back to the queue |
| `2-classifier-guidance/scripts/method_search/clevr_render.py` | gains a joint mode (`--joint R,B` or `--joint cycle`, `--masks 64`); 227 lines added, none changed or removed |
| `tests/test_paper1_sample.py`, `test_paper1_disc.py`, `test_paper1_sheets.py`, `test_paper1_pack.py`, `test_paper1_first_bars.py`, `test_paper1_model_check.py` | the tests |

## Each task, its check and its accept

| Task | Files | Check output | Loop 18 |
|---|---|---|---|
| 0. Check the plan | none written | verify-plan, run report-only, found 8 structure fixes; xref-pillar proposed 3 links; listed below under Still open | accepted as report-only |
| 1. Port the sampler, write the job scripts | `paper1_sample.py`, the three job scripts, `bigbatch-exclude.txt` | py_compile ok; `grep -n /workspace` prints nothing; `bash -n` ok on all three, each with the `co3` path, the disk guard on its run folder, the GPU guard and the copy | accepted after 2 fixes: the split keeps each seed set apart, and the docstring no longer holds the string `/workspace` |
| 2. The region sampler | `paper1_sample.py` | py_compile ok; with both masks all ones the region sampler equals the plain sum bit for bit (`torch.equal`) | accepted; the formula's order accepted (below) |
| 3. The head and its training | `paper1_disc.py`, `tests/test_paper1_disc.py` | every training setting of the full plan's table is a default argument at its value; 4-fold block choice with no held-out scene; tests of the folds, shifts, bound, reweighting and the alignment loss pass | accepted after 2 rounds: the held-out split was replaced by cross-validation inside the 64, then the guard note's additions 1 to 6, 13 and guardrail 7 went in |
| 4. The push and the random push | `paper1_sample.py`, `tests/test_paper1_sample.py` | window 15 to 22 by default (15 evaluations, σ 2.90 to 0.283), 10 to 22 as the other setting (25, σ 10.52 to 0.283); ours adds exactly w σ² ∇ logit with the sign check at +1; w = 0 equals the region sampler bit for bit; the random push matches the real push's length and is zero where it is zero; the weight-grid rule tested | accepted after the guard note's items 7, 8 and 9 and guardrails 4a and 9 |
| 5. The blind sheets | `paper1_sheets.py`, the lower-on-top paste in `paper1_sample.py` | a source PNG's text chunk does not travel; a planted row name and a planted seed both fail the check; settle field by field; the tuning total gives 11, 11 and 10 | accepted |
| 6. The renderer's joint mode | `clevr_render.py`, `paper1_pack.py`, `tests/test_paper1_pack.py` | the diff only adds lines; no f-string or walrus for Blender's Python 3.5; footprints touch at a gap of 0 and never overlap; the pack writes the agreed layout and exits 1 on a picture shared between the 64 and the 448 | accepted |
| 7. The bars and the reader | `paper1_first_bars.py`, the direction check and depth preference in `paper1_disc.py`, `tests/test_paper1_first_bars.py` | the 13 constants match the table; the verdict file's p values reproduce (20 against 5 gives 0.002; 5 against 0 gives 0.031); the worked example reads gain 15, p 0.002, yes; the alignment loss equals the hand-computed 1 minus cosine and trains the head | accepted |
| 10. The model-file check (written, not run) | `paper1_model_check.py`, `tests/test_paper1_model_check.py` | matching files pass; a wrong hash exits 1 with MISMATCH | script accepted; the box stays open until it runs |

After task 7 was accepted, writing this hand-off's commands found seven gaps that would have stopped or confused the first
result. They are fixed and tested, and listed for Loop 18 in the message that delivered this file:

1. The job scripts accept `--run <run>/<part>`, so the negatives' lone samples and the tuning run's second window sit inside their run's folder.
2. The job scripts also copy a run's `sheet/` folder to `outputs/paper1/<run>/`, so reviewers can pull a blind sheet; `key.json` stays beside it, outside `sheet/`.
3. The joint mode writes `settings-<set>.txt` at the run's top: every generator setting for the joint renders beside the experts' (`--use-gpu 0`, `--tile 32`), and the Blender version (first-result task 5's `settings.txt`).
4. The pack's sheet is at 4 times size with the asked counts under each, so the smoke render's 24 scenes give first-result task 5's `sheet.png` (as `sheet-first-64.png` with `--k 24`).
5. `split` records each count's object pixel sizes (median, 10th and 90th percentiles), so the negatives' composition sits beside the pack's (first-result task 8).
6. `paper1_first_bars.py choose-w` chooses the weight and window from the owner's tuning calls (first-result task 13), by the rule under the choices below.
7. The reader's picture strip is built from the blind sheet's own pictures, so it builds on the laptop where the calls are read.

## Choices made where the plans were silent

Each was told to Loop 18 in its task's message; Loop 18's decision is beside it.

| Choice | Loop 18 |
|---|---|
| A run writes `PAPER1_ROOT/<run>/count-<k>/<seed:06d>.png` (k is each colour's count, 1 to 3) and its small outputs at the run folder's top level; `PAPER1_ROOT = /datasets/mmolefe/poe_repair_min/paper1` is the one output constant | accepted |
| One job runs several commands separated by a lone `+`; each must carry `--run <run>` (or `<run>/<part>`), checked by the job | accepted |
| The port sheet reads the archive as a folder holding `c11/`, `c22/`, `c33/`, the layout of m10's `rows/plain/` | accepted |
| Heun tells the denoiser which of a step's two evaluations it is; the arithmetic is m10's | accepted |
| The push acts on the first evaluation of every window step and the correction of every step but the last, so every pushed σ is one of the window's own; **this differs from m10**, which pushed both evaluations of every step in its window | accepted, and stated as the rule for whichever window is set |
| The region sampler is written M_red D_red + M_blue D_blue + (1 − M_red − M_blue) D_empty, the same sum as the plan's D_empty + M_red (D_red − D_empty) + M_blue (D_blue − D_empty), because with both masks all ones it performs exactly the plain sum's floating-point operations; the plan's order drifts by up to 3.8e-6 per evaluation | accepted after Loop 18 measured both orders |
| The split is stored per seed set (evaluation, tuning, check, negatives, heldout-negatives) | Loop 18's fix |
| The head: nearest-neighbour upsampling of the 8 by 8 features; one conditioning network Linear(4, 128), SiLU, Linear(128, 1,024) for both blocks; scale applied as (1 + s) | accepted |
| Block candidates: 8x8_block1 or 8x8_block3 crossed with 16x16_block1 or 16x16_block3; 4-fold cross-validation inside the 64 training scenes against the region sampler at the tuning seeds, 250 updates per fold, so the choice costs one full training | Loop 18 decided cross-validation for the owner; 250 per fold accepted |
| Shifts up to 4 whole pixels with edges repeated, both classes; logit bound 4 tanh(raw / 4); negatives reweighted to the positives' touching share per count; overfitting gauge every 250 updates against 256 held-out negatives | from the guard note, accepted |
| The alignment loss is off by default; the aligned rerun uses `ALIGN_WEIGHT_ON = 1.0`, the full plan's aligned variant | Loop 18 decided for the owner |
| `OTHER_GPU_HOURS = 2.38`: the cost table's other steps at their upper ends (2.05) plus the guardrails' about 20 GPU-minutes | accepted |
| The training noise stays σ log-uniform 0.28 to 10.5, covering both windows | Loop 18 decided |
| The overlap band is the intersection of the two dilated object masks, about 2 pixels wide at a contact | accepted |
| `FIRST_PUSH_MIN_GREY` reads `median_change_grey`, each picture's mean absolute change over all pixels, median over pictures; the in-band and off-band changes print beside it | Loop 18 decided the stricter reading |
| Contact kept: a foreground pixel is red where R − B > 0.1 and blue where B − R > 0.1, so a purple blend is neither | accepted |
| The random push: the seed's own Gaussian noise times the real push's per-pixel size, scaled to the real push's length at each evaluation | from the guard note, accepted |
| The weight grid: the push smoke steps w over 0.25, 0.5, 1, 2, 4, 8 and 16; w1 is the weight whose median in-band change is nearest `FIRST_W_ANCHOR_GREY` (5) on a log scale; the grid is w1 × 1, 2, 4, 8 | accepted |
| Blind sheet names are 8 letters with no digit; three call templates (a, b, c); settle exits 1 on any unresolved field | accepted |
| Lower-on-top paste: an object's base is its lowest mask row; a tie keeps blue on top, as mask paste does | accepted |
| Joint mode: red against blue refuses only overlapping ground footprints (exact disc and turned-square tests), so a gap of 0 is allowed; `--joint cycle` gives scene i (i mod 3) + 1 of each colour, the pool's sizes in turn | accepted |
| The smoke render uses seed base 8,090,000, clear of the training pool (8,000,000 to 8,014,999) and the held-out base (8,100,000) | not yet reviewed; recorded here |
| Fewer than `FIRST_REAL_RIGHT_SHARE_MIN` (0.85) of the real touching scenes on the sheet called right, counted as ceil(0.85 x n) (65 of the 76 this sheet holds, 82 of 96), makes the answer "inconclusive", reason "could not measure it: the rubric at 64 pixels", whatever the gain, p or random push; checked before the no, yes and inconclusive rule; `answer.json` carries `real_scenes_right`, `real_scenes_called`, `real_scenes_needed` and `rubric_readable`. `FIRST_REAL_RIGHT_MIN = 82` stays in the file, unused, for the record | decided for the owner on board question 123; read as a share by the owner on 2026-11-27, because the sheet held 76 real scenes, and written by Loop 18 while Loop 16 was offline |
| `choose-w`: among pushed settings whose contact kept is at most 0.10 below the region sampler's (a rise is fine), the most right on the 32 tuning seeds; ties to the smaller w, then the window 15 to 22; none allowed exits 3 for the owner | not yet reviewed; recorded here |
| `FREE_MIN_MIB` 12,000 for sampling and 16,000 for training on the shared card; a busy card is allowed, a missing or faulted one refused | accepted |
| The CPU job script keeps a GPU guard that refuses only a missing or faulted card | accepted |

## What the first runs are the first real test of

- **The joint mode's masks in Blender.** This laptop has no copy of the CLEVR generator or Blender. `write_masks` relies on the
  released `render_images.py`'s `render_shadeless(blender_objects, path)` drawing three `random.random()` values per object, in
  object order, for its flat colour, and on Blender 2.78c bundling numpy. The first result's smoke render (its task 5) is the
  first real run. If either assumption fails, the step stops with "mask render shows N flat colours for M objects" (or an import
  error) rather than writing wrong masks.
- **The rates on the Blackwell card.** The cost lines were written for a `bigbatch` RTX 3090. The first GPU job re-measures
  seconds per picture on the Blackwell card (`timing.jsonl` in each run), and the head's projection at update 200 uses its own
  measured rate.
- **The port.** The port sheet (first-result task 2) compares the plain sum with the archived pictures by eye; pixel identity is
  not expected across cards.

## Still open

- **First-result task 1's seed check has no script.** It lists the first result's ranges beside the full plan's and every
  earlier job's and runs on the laptop; no setup task wrote it.
- **The archive's layout inside `m10-comparison.tar` is unknown.** The port sheet needs the folder holding `c11/`, `c22/`, `c33/`;
  the commands below find it after unpacking.
- **Task 0's 8 plan fixes and 3 links**, for the owner or the next plan sync: task groups use bare checkboxes; groups 1 to 6 and
  the close out lack Next or Needs lines and instruction 11 lacks Next; the Error Matrix lacks its catalog subsections and entry
  count; Outputs lacks its Pending, Generated and Organization subsections; the Orchestration table has no Outcome column; there
  is no `review/01-setup.md`; instruction 9.2 names no environment page; the root `MASTER_PLAN.md` never mentions
  paper-1-clevr. Links proposed: line 47 "the colour-expert run" to `environment/clevr-and-edm2.md#the-colour-experts-model-files`;
  line 228 "W&B project" to `environment/storage.md#access-paths`; line 331 "Pull it" to
  `runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md#3-harvest-what-is-running`.
- **The commit.** Loop 18 lists the files for the owner; nothing here is committed.

## The commands that run the first result

Run in order on the held node, from the cluster checkout, after the owner pushes the branch and allocates mscluster110. Every job
script picks `co3_bw` on mscluster110 to 112 and `co3` elsewhere, pins `CUDA_VISIBLE_DEVICES=0` for GPU work, and runs its guards
before any work. Each step's "read" line says what to look at before the next. The step numbers are the first-result plan's.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/concept-composition
P=/datasets/mmolefe/poe_repair_min/paper1
S=2-classifier-guidance/scripts/paper1
L=$P/logs
mkdir -p "$L"            # the paper1 root and its log folder; every job's directory guard needs the root to exist
run() { local kind=$1 name=$2; shift 2; bash "$S/paper1_$kind.sbatch" "$name" "$@" 2>&1 | tee -a "$L/$name.log"; }
LONE="--red-dir $P/paper1-first-lone-red --blue-dir $P/paper1-first-lone-blue --mean-empty $P/paper1-first-lone-red/mean-empty.pt"
SPLIT="--split $P/paper1-first-lone-red/split.json"
```

**Setup task 10, the model-file check** (CPU, a few minutes). Read `outputs/paper1/paper1-model-check/models.txt`: three files,
all "match", "passes: True".

```bash
run render paper1-model-check python $S/paper1_model_check.py --run paper1-model-check
```

**Step 2, the port and the lone samples** (about 0.2 GPU-hours on a 3090). Read the port sheet (purple blends where the archived
ones are), `timing.jsonl` (seconds per picture on this card), `split.json`'s touching shares and `lone-sheet.png`.

```bash
mkdir -p $P/m10-archive && tar -xf /datasets/mmolefe/poe_repair_min/loop-6/m10/comparison/m10-comparison.tar -C $P/m10-archive
ARCHIVE=$(dirname "$(find $P/m10-archive -type d -name c11 -path '*plain*' | head -1)"); echo "archive plain sum: $ARCHIVE"
run gpu paper1-first-sampler-port paper1_sample.py plainsum --run paper1-first-sampler-port --counts 1,2,3 --seeds 0-7 \
  + paper1_sample.py port-sheet --run paper1-first-sampler-port --archive "$ARCHIVE" --seeds 0-7
run gpu paper1-first-lone-red paper1_sample.py lone --run paper1-first-lone-red --expert red --seeds 20000-20255,10000-10255,30000-30127 \
  + paper1_sample.py lone --run paper1-first-lone-red --expert empty --seeds 0-1023 \
  + paper1_sample.py mean-empty --run paper1-first-lone-red --empty-dir $P/paper1-first-lone-red/empty
run gpu paper1-first-lone-blue paper1_sample.py lone --run paper1-first-lone-blue --expert blue --seeds 20000-20255,10000-10255,30000-30127
run render paper1-first-lone-red python $S/paper1_sample.py split --run paper1-first-lone-red --set evaluation $LONE --seeds 20000-20255 \
  + python $S/paper1_sample.py split --run paper1-first-lone-red --set tuning $LONE --seeds 10000-10255 \
  + python $S/paper1_sample.py split --run paper1-first-lone-red --set check $LONE --seeds 30000-30127 \
  + python $S/paper1_sample.py pair-sheet --run paper1-first-lone-red --left $P/paper1-first-lone-red --right $P/paper1-first-lone-blue \
      --left-label red --right-label blue --seeds 20000-20015 --name lone-sheet.png
```

**Step 3, the region sampler's check pictures** (about 0.1 GPU-hours). Read `identity.txt` ("passes True"; its pictures at counts 1
and 3 are plain sums), `band-distance.json`, `region-beside-paste.png` and `mask-sheet.png`. Then the owner's instruction 4.

```bash
run gpu paper1-first-region-check paper1_sample.py region --run paper1-first-region-check --identity \
      --mean-empty $P/paper1-first-lone-red/mean-empty.pt --counts 1,3 --seeds 30000-30007 \
  + paper1_sample.py region --run paper1-first-region-check $LONE --counts 2 $SPLIT --set check --side touching --limit 64 \
  + paper1_sample.py paste --run paper1-first-region-check/paste $LONE --counts 2 --seeds 30000-30127 \
  + paper1_sample.py pair-sheet --run paper1-first-region-check --left $P/paper1-first-region-check \
      --right $P/paper1-first-region-check/paste --left-label region --right-label paste --counts 2 $SPLIT --set check \
      --side touching --limit 64 --name region-beside-paste.png \
  + paper1_sample.py mask-sheet --run paper1-first-region-check $LONE --counts 2 --seeds 30000-30007
```

**Step 5, the joint smoke render** (CPU, about 0.2 node-hours). This is the masks' first real run. Read `sheet-first-64.png` (24
scenes at 4 times size, counts under each), `settings-smoke.txt` (every row "same yes") and `touching_share.json`. Then the
owner's instruction 6, which fixes the rubric as `with-shadows` or `without-shadows`.

```bash
run render paper1-first-render-smoke render smoke cycle cycle 8090000 0 24 \
  + python $S/paper1_pack.py --run paper1-first-render-smoke --renders $P/paper1-first-render-smoke/renders/smoke --count 24 --k 24
RUBRIC=without-shadows    # or with-shadows, by the owner's answer at instruction 6
```

**Step 7, the 512 joint scenes and the pack** (CPU, about 1.1 node-hours). Read `touching_share.json` (pooled between 0.1 and
0.9), `leakage.txt` (0 shared; the pack exits 1 otherwise), `composition.txt`, `depth-order.json` and `sheet-first-64.png`.

```bash
run render paper1-first-render-joint render joint cycle cycle 8000000 0 512
run render paper1-first-pack python $S/paper1_pack.py --run paper1-first-pack --renders $P/paper1-first-render-joint/renders/joint --count 512 --k 64
```

**Step 8, the negatives** (about 0.65 GPU-hours). Read `split.json` (negatives' touching share and object sizes beside the pack's
`composition.txt`).

```bash
NLONE="--red-dir $P/paper1-first-negatives/lone-red --blue-dir $P/paper1-first-negatives/lone-blue --mean-empty $P/paper1-first-lone-red/mean-empty.pt"
run gpu paper1-first-negatives paper1_sample.py lone --run paper1-first-negatives/lone-red --expert red --seeds 200000-201023,202048-202303 \
  + paper1_sample.py lone --run paper1-first-negatives/lone-blue --expert blue --seeds 200000-201023,202048-202303 \
  + paper1_sample.py split --run paper1-first-negatives --set negatives $NLONE --seeds 200000-201023 \
  + paper1_sample.py split --run paper1-first-negatives --set heldout-negatives $NLONE --seeds 202048-202303 \
  + paper1_sample.py region --run paper1-first-negatives/train $NLONE --seeds 200000-201023 \
  + paper1_sample.py region --run paper1-first-negatives/heldout $NLONE --seeds 202048-202303
```

**Step 9, one head at k = 64** (about 0.5 to 1 GPU-hours, plus the block choice). It draws the region sampler at all 256 tuning
seeds per count (step 11's region row reuses them), chooses the blocks, then trains. Read `blocks.json` (the chosen blocks and the
probes), `projection.jsonl` (it stops with exit 3 past 4 GPU-hours), `overfit.json`, `train.json` (the loss falls over the first
1,000 updates) and the W&B run.

```bash
run train paper1-first-disc-k64 paper1_sample.py region --run paper1-first-disc-k64/tuning-region $LONE --seeds 10000-10255 \
  + paper1_disc.py choose --run paper1-first-disc-k64 --positives $P/paper1-first-pack/k64-s1 --negatives $P/paper1-first-disc-k64/tuning-region \
  + paper1_disc.py train --run paper1-first-disc-k64 --positives $P/paper1-first-pack/k64-s1 --negatives $P/paper1-first-negatives/train \
      --negatives-split $P/paper1-first-negatives/split.json --gauge-negatives $P/paper1-first-negatives/heldout \
      --blocks-file $P/paper1-first-disc-k64/blocks.json --seed 0
HEAD=$P/paper1-first-disc-k64/head.pt
```

**Step 10, the head check and the direction check.** Read `accuracy.json`, `depth_preference.json` and `direction.json`. The check
exits 4 when the direction check fails (the training scenes' median cosine under 0.10 in both of the two lowest bands).

```bash
CHECK="--positives $P/paper1-first-pack/k64-s1 --negatives $P/paper1-first-negatives/train --heldout-positives $P/paper1-first-pack/heldout \
  --heldout-negatives $P/paper1-first-negatives/heldout --heldout-negatives-split $P/paper1-first-negatives/split.json \
  --tuning-split $P/paper1-first-lone-red/split.json --red-lone $P/paper1-first-lone-red --blue-lone $P/paper1-first-lone-blue \
  --mean-empty $P/paper1-first-lone-red/mean-empty.pt"
run gpu paper1-first-disc-check paper1_disc.py check --run paper1-first-disc-check --head $HEAD $CHECK
```

**Only if that check exited 4: the aligned head, and the check again.** The rerun trains with the alignment loss at weight 1.0
into a new run folder, counting the first head's hours under the 7 GPU-hour stop. A second exit 4 stops the first result for the
owner.

```bash
H=$(python3 -c "import json; b=json.load(open('$P/paper1-first-disc-k64/blocks.json')); t=json.load(open('$P/paper1-first-disc-k64/train.json')); print(round((b['seconds'] + t['training']['seconds']) / 3600, 3))")
run train paper1-first-disc-k64-aligned paper1_disc.py train --run paper1-first-disc-k64-aligned --positives $P/paper1-first-pack/k64-s1 \
  --negatives $P/paper1-first-negatives/train --negatives-split $P/paper1-first-negatives/split.json \
  --gauge-negatives $P/paper1-first-negatives/heldout --blocks-file $P/paper1-first-disc-k64/blocks.json --seed 0 \
  --align-weight 1.0 --already-gpu-hours $H
HEAD=$P/paper1-first-disc-k64-aligned/head.pt
run gpu paper1-first-disc-check-aligned paper1_disc.py check --run paper1-first-disc-check-aligned --head $HEAD $CHECK
# exit 4 again: stop here; the owner diagnoses from the gradient maps and the accuracies, or closes the route
```

**Step 10, the push smoke and the weight grid** (about 0.2 GPU-hours with the ladder). Read `push.json`: at w = 0 the largest
change is 0 (the identity); at w = 1 `median_change_grey` is at least 1; the sign check reads +1 in every band and the share of
non-finite gradient entries is 0. Read `grad-maps.png` and `weight-grid.json`, written before step 11.

```bash
run gpu paper1-first-push-smoke paper1_sample.py region --run paper1-first-push-smoke/region $LONE --counts 1,2,3 $SPLIT --set check --side touching --limit 16 \
  + paper1_sample.py push --run paper1-first-push-smoke --head $HEAD --weights ladder $LONE $SPLIT --set check --side touching --limit 16 \
      --compare-to paper1-first-push-smoke/region --grad-maps 8 \
  + paper1_sample.py weight-grid --run paper1-first-push-smoke --smoke $P/paper1-first-push-smoke/push.json
```

**Step 11, the tuning weights and windows** (about 0.3 to 0.5 GPU-hours). Four weights at steps 15 to 22, w = 1 and 2 at steps 10
to 22, and the region sampler; contact kept for each; then the 224-picture blind sheet for the owner. Read `contact.json` and
`check.json` (0 hits).

```bash
run gpu paper1-first-tune-weight paper1_sample.py push --run paper1-first-tune-weight --head $HEAD --weights-file $P/paper1-first-push-smoke/weight-grid.json \
      --window 15-22 $LONE $SPLIT --set tuning --side touching \
  + paper1_sample.py push --run paper1-first-tune-weight/window-10-22 --head $HEAD --weights 1,2 --window 10-22 $LONE $SPLIT --set tuning --side touching
F=$(ls -d $P/paper1-first-tune-weight/w-* $P/paper1-first-tune-weight/window-10-22/w-* | paste -sd,)
ROWS="region=$P/paper1-first-disc-k64/tuning-region"
for d in $P/paper1-first-tune-weight/w-*; do ROWS="$ROWS,win15-22-w${d##*/w-}=$d"; done
for d in $P/paper1-first-tune-weight/window-10-22/w-*; do ROWS="$ROWS,win10-22-w${d##*/w-}=$d"; done
run render paper1-first-tune-weight python $S/paper1_sample.py contact --run paper1-first-tune-weight --folders "$F,$P/paper1-first-disc-k64/tuning-region" \
      --mean-empty $P/paper1-first-lone-red/mean-empty.pt $SPLIT --set tuning --side touching \
  + python $S/paper1_sheets.py select --run paper1-first-tune-weight $SPLIT --set tuning --touch-total 32 --rows "$ROWS" \
  + python $S/paper1_sheets.py build --run paper1-first-tune-weight --items $P/paper1-first-tune-weight/items.json --rubric $RUBRIC \
      --shuffle-seed 11 --reviewers owner \
  + python $S/paper1_sheets.py check --run paper1-first-tune-weight
```

**Instruction 12 and step 13's choice, on the laptop.** The owner pulls the sheet, fills `sheet/calls-owner.csv` without opening
`key.json`, and `choose-w` picks the setting (exit 3 means no setting keeps contact: stop for the owner).

```bash
scripts/cluster.sh pull outputs/paper1/paper1-first-tune-weight
python3 2-classifier-guidance/scripts/paper1/paper1_first_bars.py choose-w --root outputs/paper1 --run paper1-first-tune-weight \
  --sheet paper1-first-tune-weight --contact outputs/paper1/paper1-first-tune-weight/contact.json
# then write the chosen w and window into the verdict file's Runs table; they are frozen from here
```

**Step 13, the six rows** (about 0.4 GPU-hours and 0.05 CPU node-hours). Set `W` and `WIN` from `chosen-setting.json`. Read each
row's `push.json` (the repeat check's largest difference) and `contact.json`.

```bash
W=1; WIN=15-22           # from chosen-setting.json
run gpu paper1-first-row-plainsum paper1_sample.py plainsum --run paper1-first-row-plainsum --counts 1,2,3 --seeds 20000-20255
run gpu paper1-first-row-region paper1_sample.py region --run paper1-first-row-region $LONE --seeds 20000-20255
run gpu paper1-first-row-ours-k64 paper1_sample.py push --run paper1-first-row-ours-k64 --head $HEAD --weights $W --window $WIN $LONE \
  --seeds 20000-20255 --compare-to paper1-first-row-region --repeat-check 8
run gpu paper1-first-row-randompush paper1_sample.py push --random-push --run paper1-first-row-randompush --head $HEAD --weights $W \
  --window $WIN $LONE --seeds 20000-20255 --compare-to paper1-first-row-region
run render paper1-first-row-paste python $S/paper1_sample.py paste --run paper1-first-row-paste $LONE --seeds 20000-20255
run render paper1-first-row-paste-lower python $S/paper1_sample.py paste --order lower-on-top --run paper1-first-row-paste-lower $LONE --seeds 20000-20255
R=""; for r in plainsum paste region ours-k64 randompush paste-lower; do R="$R,$P/paper1-first-row-$r"; done
run render paper1-first-row-region python $S/paper1_sample.py contact --run paper1-first-row-region --folders "${R#,}" \
  --mean-empty $P/paper1-first-lone-red/mean-empty.pt $SPLIT --set evaluation --side touching
```

**Step 14, the blind sheet** (CPU). Six rows at the 96 touching seeds, ours and paste at the 96 apart seeds, and the 96 real
touching scenes. Read `items.json`'s summary and `check.json` (0 hits).

```bash
run render paper1-first-eye-sheet python $S/paper1_sheets.py select --run paper1-first-eye-sheet $SPLIT --set evaluation \
    --rows plainsum=$P/paper1-first-row-plainsum,paste=$P/paper1-first-row-paste,region=$P/paper1-first-row-region,ours=$P/paper1-first-row-ours-k64,randompush=$P/paper1-first-row-randompush,paste-lower-on-top=$P/paper1-first-row-paste-lower \
    --apart-rows ours,paste --real $P/paper1-first-pack/heldout \
  + python $S/paper1_sheets.py build --run paper1-first-eye-sheet --items $P/paper1-first-eye-sheet/items.json --rubric $RUBRIC --shuffle-seed 14 \
  + python $S/paper1_sheets.py check --run paper1-first-eye-sheet
```

**Instruction 15 and step 16, on the laptop.** Two reviewers fill `sheet/calls-a.csv` and `sheet/calls-b.csv`, a third fills
`sheet/calls-c.csv` on the fields they disagree on, and the reader gives the answer, the 2 by 2 table and the strip.

```bash
scripts/cluster.sh pull outputs/paper1/paper1-first-eye-sheet
python3 2-classifier-guidance/scripts/paper1/paper1_first_bars.py read --root outputs/paper1 --run paper1-first-read \
  --sheet paper1-first-eye-sheet --calls calls-a.csv calls-b.csv --third calls-c.csv
```

<!-- built-to: loop-16 hand-off -->
