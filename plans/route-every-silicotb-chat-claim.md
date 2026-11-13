# 🎯 Every claim in the silicotuberculosis chat, read and backed by evidence
<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** a ledger of every claim in the ChatGPT conversation, each marked tested,
> cited, already answered here, or out of reach, with a figure beside every tested one, and one
> shared page that links them · **Time:** ~9 h to finish rung 1 and rung 2, ~16 h budget at 2×
> for learning; later rungs re-routed · **Stop point:** rung 1 closes when every claim has a
> mark; rung 2 closes when the fibrosis classifier test prints its verdict line

Built to: route-file-template v2

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [What we know so far](#what-we-know-so-far)
- [What you will achieve](#what-you-will-achieve)
- [The route at a glance](#the-route-at-a-glance)
- [Tasks](#tasks) (the AI runs these)
- [Instructions](#instructions) (you do these by hand)
- [Quizzes and feedback](#quizzes-and-feedback)
- [Agents and hooks](#agents-and-hooks)
- [Later](#later)
- [Why this chain](#why-this-chain)

## Quick context: where you are

| | |
|---|---|
| The source | [the conversation, saved as markdown](../artifacts/drips/silicotb-conversation/source/conversation.md): 8 turns, 19,004 words, fetched 2026-11-03 from the share link, plus [three screenshots](../artifacts/drips/silicotb-conversation/source/screenshots/) |
| Nothing is under way | no walk map exists for this conversation yet |
| Already routed | turn 7 (multi-label LoRA on cats and dogs) is the [inclusive-captions pilot route](../4-inclusive-captions/plans/route-inclusive-captions-pilot.md), resumed there, not here |
| Cadence | you steer: no fixed windows, so the schedule is a block order, not a calendar |

[↑ Back to top](#table-of-contents)

## What we know so far

Each fact is settled, with its source. The steps it touches are named.

| Fact | Source | Touches |
|---|---|---|
| NIH ChestX-ray14 has 11 films labelled both pneumonia and fibrosis (8 PA), 1,420 pneumonia without fibrosis | `/datasets/mmolefe/data/nih/Data_Entry_2017.csv`, counted 2026-11-03 | steps 3, 4 |
| NIH has 36 fibrosis plus emphysema films (28 PA), 23 pneumonia plus emphysema | same file | step 1 turn 4, later |
| NIH cardiomegaly plus effusion: 1,063 both; effusion plus atelectasis: 3,275 both | same file | step 2 |
| VinDr-CXR (Kaggle train, 15,000 films) carries no pneumonia or TB label; consolidation is the nearest local finding | `/datasets/mmolefe/vinbigdata/train.csv` | steps 3, 4 |
| VinDr: 1,617 fibrosis films; fibrosis plus consolidation 187; fibrosis plus lung opacity 666; fibrosis plus ILD 211 | same file, image-level union over radiologists | steps 3, 4 |
| Box labels exist for VinDr (`train.csv` x/y columns) and ChestX-Det (`/datasets/mmolefe/chestxdet/*.json`); lung masks in `/datasets/mmolefe/chexmask/` | the folders | step 2 |
| Paper3 Mission B already ran single-disease classifiers on six comorbid pairs; cardiomegaly plus effusion is its control; pneumonia plus emphysema is the one pair that drops, with 0 both-films to train on | [Paper3 sweep matrix](/home-mscluster/mmolefe/Playground/PhD/Paper3/plans/does-comorbidity-break-detectors/plans/does-occlusion-explain-the-drop/plans/run-the-pairs/plans/00-sweep-matrix.md) | steps 1, 4 |
| Paper3's trained pair classifiers sit at `/datasets/mmolefe/comorbid-classifier-audit/outputs/pair_experts/` (read-only from here) | the folder | step 4 |
| A TB plus pneumonia dataset search prompt is already written | [Paper3 dataset search](/home-mscluster/mmolefe/Playground/PhD/Paper3/plans/does-comorbidity-break-detectors/source/tb-pneumonia-dataset-search.md) | step 1 turns 2 to 4 |
| The interaction term `r_t`, unseen-pair transfer and the joint prompt's failures are already findings in this repo | [this repo's report](../report/00-INDEX.md) | step 1 turns 5, 6, 8 |
| Unknown: whether a compute node can download model weights (TorchXRayVision, MedSAM) from the internet | not in `environment/` | step 2; ask before it runs |

[↑ Back to top](#table-of-contents)

## What you will achieve

| Stage | Tier | You will have |
|---|---|---|
| rung 1: read and mark | own for the medical turns, gloss for the diffusion turns | the conversation walked turn by turn, `CLAIMS.md` with a mark on every claim, a co-occurrence matrix, an annotated sample sheet, reference images beside each imaging claim |
| rung 2: one claim tested | own | `HYPOTHESIS.md` written before the run, then sensitivity, AUC and calibration error (ECE) for a consolidation classifier on fibrotic against non-fibrotic lungs, counts per group, the verdict line |
| rung 3 onward | settled at rung 2's close | one rung per claim family worth testing, re-routed from rung 2's retrospective |
| **outcome** | | the claims page: every claim, its mark, its figure or citation, one link to share |

[↑ Back to top](#table-of-contents)

## The route at a glance

```
▶ Recommended  you: read (1) → criterion (3) → test (4) → figure (5) → retro (7)
               ‖ AI, background: claim ledger (0) · matrix + annotated samples (2) · page (6)
               (spliced from flow 20 and the paper-reading recipe; drip-read on a chat
               transcript is a proposed use, since the skill is written for papers)
  alt          /drip-idea on the conversation    better if you want verdicts per claim and
                                                 do not care about reading it in order
```

Time: rung 1 reading ~4.5 h (5 medical turns × 30 min × 1.5 for tangents, plus 3 diffusion turns
at a gloss, `estimate`), rung 2 ~4.5 h (`--request` 5 pieces × 30 min × 1.5 + the criterion walk at
its measured 20 min median over 13 runs). Budget at 2×: ~16 h. AI ~3 h, about 2 h of it beside
your reading.

[↑ Back to top](#table-of-contents)

## Tasks

For Claude to execute. Ask Claude to do these.

### 0. 🤖 Write the claim ledger (AI, background, ~15 min, start first)

```text
Dispatch a general-purpose agent in the background. Read
artifacts/drips/silicotb-conversation/source/conversation.md end to end, never sampled. Write
artifacts/drips/silicotb-conversation/CLAIMS.md: one row per checkable claim with its turn and
heading, its kind (imaging fact, epidemiology, AI behaviour, dataset fact, diffusion theory,
experiment design), the route that would settle it (cite a source, count in our data, run a
classifier, already a finding in report/ or in Paper3's report/, out of reach and why), and the
dataset and counts it would need from the facts table in plans/route-every-silicotb-chat-claim.md.
Mark claims ChatGPT stated without a source. Do not test anything. Report the row count per kind.
```

leaves `CLAIMS.md`, which steps 1, 3 and 6 read.

### 2. 🤖 Co-occurrence matrix and annotated samples (AI, background, beside step 1, ~60 min)

Started from step 1's `run` at turn 3, or pasted directly:

```text
/data-distributions co-occurrence of findings in /datasets/mmolefe/data/nih/Data_Entry_2017.csv
(14 labels, PA and all views separately) and /datasets/mmolefe/vinbigdata/train.csv (image-level
union of the 3 radiologists, 14 findings). Output: two count matrices with true counts in every
cell and a log colour scale disclosed, saved with a README card to
artifacts/results/which-lung-findings-occur-together/.
Mark the pairs the conversation names (pneumonia+fibrosis, fibrosis+emphysema,
cardiomegaly+effusion, effusion+atelectasis).
```

```text
/visualize-data-samples VinDr films from the four named pairs, 6 per pair plus 6 single-finding
films each: radiologist boxes from /datasets/mmolefe/vinbigdata/train.csv, lung masks from
/datasets/mmolefe/chexmask/, and per-finding scores from a pretrained TorchXRayVision DenseNet
(all-datasets weights) printed on each film. Save to
artifacts/results/which-lung-findings-occur-together/samples/. Stop and ask first if the weights
cannot be downloaded from the node.
```

serves the ledger's dataset-fact rows and the outcome page's first panel.

### 4. 🧪 The classifier test (you steer, AI builds; ~4 h, budget 8 h; Opus, high)

Runs after step 3's `HYPOTHESIS.md` exists. A fork of the existing
`drip-execute-plan --request` shape:

```text
/drip-execute-plan --request Test whether a classifier that finds consolidation (the pneumonia
stand-in, since VinDr has no pneumonia label) in non-fibrotic lungs becomes less sensitive or
overconfident when the lungs are fibrotic.
Starts from: /datasets/mmolefe/vinbigdata/train.csv and train/ (DICOM); HYPOTHESIS.md from step 3.
Settled facts: 187 films have fibrosis+consolidation, 1,617 fibrosis, 15,000 total (counted
2026-11-03); NIH has only 11 pneumonia+fibrosis films, shown as a look, never scored.
Design: train on consolidation-positive and negative films with no fibrosis; evaluate on four
groups (consolidation yes/no × fibrosis yes/no), matched on view and age where the CSV allows.
Only fibrosis differs between the two consolidation-positive groups; print the counts per group.
Outputs: sensitivity at a threshold fixed on the non-fibrotic validation set, AUC, ECE with
reliability bins, bootstrap 95% CIs, in artifacts/results/does-fibrosis-hide-pneumonia/.
Target: the support, null and inconclusive lines in HYPOTHESIS.md, unchanged after the run.
Proof: the verdict line printed against them. Limits: large files to /datasets only; one GPU job.
Controls: piece 1 draw record <one VinDr image_id> (the film's CSV rows to its boxes);
piece 2 plumb TorchXRayVision DenseNet, part mode (what its features give before training);
piece 3 example (the four group counts); piece 4 prove (the threshold is fixed before eval);
piece 5 test.
Side routes: piece 5 /pair-figure sensitivity per group beside the 6 films the model missed most.
```

### 6. 🌐 The shared claims page (AI, after step 5, ~30 min)

```text
/results-scene the claims page from artifacts/drips/silicotb-conversation/CLAIMS.md: one row per
claim with its mark, its figure from artifacts/results/ or its citation, and the reference images
fetched in step 1. Publish as a private artifact; then write its scene-library section and run
the register refresh so the Linked from column is not "none".
```

[↑ Back to top](#table-of-contents)

## Instructions

For you to follow manually. Do these yourself.

### 1. 🌊 Read the conversation, one turn at a time (you, ~4.5 h, budget 9 h; Sonnet, medium)

```text
/drip-read --lean artifacts/drips/silicotb-conversation/source/conversation.md
Purpose: give every claim in CLAIMS.md a mark (tested, cited, already answered, out of reach).
This is a ChatGPT transcript, not a paper: treat each turn as a section and each claim as a chunk.
Depth: turns 1 to 4 own (the medicine and the AI limits); turns 5, 6 and 8 gloss, linking the
findings in report/00-INDEX.md; turn 7 links 4-inclusive-captions/plans/route-inclusive-captions-pilot.md and stops.
Bar: every row of CLAIMS.md marked. Box: 9 h over as many sittings as you like.
Controls:
  turn 1 sketch (the three overlap regimes), prove (each imaging claim, with a fetched reference
         image from Wikimedia Commons or an open-access paper, licence stated), scout paper
         (South African silicotuberculosis studies)
  turn 2 why (mimicry, masking, interaction), route to step 3 on the interaction claim
  turn 3 run (step 2's matrix and samples), draw record <VinDr image_id>, example (pair counts)
  turn 4 example (NIH 11 against VinDr 187), plumb TorchXRayVision DenseNet, prove
         (fibrosis+consolidation count)
  each turn's close: test
Side routes:
  turn 1 /picture-speak the three regimes of overlap, with real CT and X-ray images per regime
  turn 2 /drip-idea --hypothesis "a classifier that finds pneumonia in normal lungs fails in
         fibrotic lungs"
  turn 3 /pair-figure each co-occurrence count beside 6 films of that pair
  turn 5 /pressure-test the claim that exact PoE loses the interaction R
```

gate: before step 3, `CLAIMS.md` has a mark on turns 1 to 4. ◀ Needs: task 0.

path, calibration (from the lookup):
1. ⬜ [Conditional probability from a table of counts](/home-mscluster/mmolefe/goal-setting/learning/ai-engineering/agent-evaluation/selective-evaluation/plans/01-conditional-probability-from-a-table-of-counts.md#1--drip-it-on-the-toy) · [video, 9:35](https://www.youtube.com/watch?v=sqDVrXq_eh0&t=575s)
2. ⬜ [Calibration: reliability bins and ECE](/home-mscluster/mmolefe/goal-setting/learning/ai-engineering/agent-evaluation/selective-evaluation/plans/09-calibration-reliability-bins-and-expected-calibration-error.md#1--drip-it-on-the-toy)

path, sensitivity and AUC: no journey yet, expect a `gap` line.

path, product of experts (turn 5, gloss): rung 8 is [Derive the product-of-experts rule from Bayes and the fork](/home-mscluster/mmolefe/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/17-poe-rule-from-the-fork.md#1--drip-it-on-the-toy); rungs 1 to 7 are listed by the lookup and stay `later` at gloss depth.

### 3. ✍️ Write the falsification criterion before any run (you, ~20 min, measured median of 13 runs; Opus)

```text
/drip-idea --hypothesis A consolidation classifier trained on non-fibrotic VinDr films loses
sensitivity, or becomes overconfident, on films that also show fibrosis. Comparison: the 187
fibrosis+consolidation films against consolidation without fibrosis, matched on view. Write the
support line (a sensitivity drop of at least <n> with a CI that excludes 0), the null band, the
inconclusive band, the counts needed for that CI width, and what would surprise. Compile to
artifacts/drips/silicotb-conversation/HYPOTHESIS.md.
```

▶ Next: task 4.

### 5. 🖼️ Look at the result, then the figure (you, ~30 min)

Open `artifacts/results/does-fibrosis-hide-pneumonia/` in VS Code. Open the 6 missed films the pair
figure shows and say, for each, whether you would have called the consolidation yourself. Then
tick the HYPOTHESIS.md line the verdict matches.

### 7. 🔁 Explain it back (you, ~40 min)

```text
/drip-retro artifacts/drips/silicotb-conversation/
```

```text
/pressure-test the claim in artifacts/drips/silicotb-conversation/HYPOTHESIS.md, with the
verdict from artifacts/results/does-fibrosis-hide-pneumonia/
```

[↑ Back to top](#table-of-contents)

## Quizzes and feedback

After turns 1 to 4 close in step 1, the walk's `test` builds one quiz page (quiz-cadence, in the
background), prepared first by `quiz-prep-pulse` against the radiology terms the turns use
(perilymphatic nodules, progressive massive fibrosis, tree-in-bud, CPFE). Bar: every question the
classifier test needs right (sensitivity, ECE, the matched comparison), 80% of the rest
(`estimate`). Misses become the first rungs of step 3.

[↑ Back to top](#table-of-contents)

## Agents and hooks

**Agents.** A general-purpose agent writes the ledger (task 0). `xref-cadence` runs at each
turn's close over `CLAIMS.md` and the walk map, linking to Paper3's sweep and this repo's report.
`video-cadence` runs once over the walk map for the radiology terms. `paper-cadence` files any
paper the turn 1 `scout` fetches. No `hld-cadence`: nothing here is a running system, so `draw`
is used only as `draw record`.

**Hooks.** The existing session-start plan pulse and the cadence queue already cover this walk.
One optional addition: a stop hook that prints the `CLAIMS.md` tally (marked against unmarked),
set up with `/update-config`. It is offered, not installed.

[↑ Back to top](#table-of-contents)

## Later

- Fibrosis plus emphysema (CPFE): 36 NIH films, too few for a classifier test; a look and a count.
- TB plus silicosis itself: no public set on disk; the Paper3 dataset search prompt is the start.
- HIV plus TB: no imaging label for HIV in any set on disk; out of reach.
- Effusion plus atelectasis masking: 3,275 NIH films, testable; a candidate for rung 3.
- Turn 9 (inclusive training across seeds and pairs): after the inclusive-captions pilot closes.

[↑ Back to top](#table-of-contents)

## Why this chain

<details>
<summary>Why a reading walk with a ledger, and why VinDr for the test</summary>

You asked for coverage of the whole conversation, so the spine is a walk in the conversation's
order, and the ledger keeps that coverage countable. Tests sit at the turn that raises them, so
data work starts when the reading reaches turn 3 and not before. The example claim cannot be
scored on NIH with 11 films; VinDr's 187 fibrosis plus consolidation films can carry a CI, at the
cost that consolidation is a stand-in for pneumonia, which the verdict must say.

</details>

[↑ Back to top](#table-of-contents)
