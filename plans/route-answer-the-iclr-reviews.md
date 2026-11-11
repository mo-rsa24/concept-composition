# 🧪 Answer the ICLR 2027 reviews of submission 56173, Reviewer PrzE first
<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** a revised paper in a shared Overleaf with new scores, baselines, pair lists and a
> three-concept result; a second Overleaf project with drafted replies; comment text ready to post ·
> **Time:** about 14 h of your time to finish (estimate), plus about 1 h of posting · **Stop point:** everything posted before Wednesday 18 November, inside a 15 h box ·
> **Outcome board:** [the paper review board](../report/paper-review/board.md) ·
> **Background:** [the references file](route-answer-the-iclr-reviews-references.md)

Built to: route-file-template v3

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [What we know so far](#what-we-know-so-far)
- [What you will achieve](#what-you-will-achieve)
- [The route at a glance](#the-route-at-a-glance)
- [Schedule](#schedule)
- [Tasks](#tasks) (the AI runs these)
- [Instructions](#instructions) (you do these by hand)
- [Quizzes and feedback](#quizzes-and-feedback)
- [Next step](#next-step)

## Quick context: where you are

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-we-know-so-far) ➡️

**Goal:** Reviewer PrzE (rating 1, confidence 4) sees each claim backed by numbers, and a respectful reply goes up on OpenReview.
**Objective (the whole chain):** scores for the adapter against PoE, CO3 and more corrected samplers on the same held-out pairs and seeds; the [48 training pairs](../report/paper-review/held-out-set.md#the-48-training-pairs) listed; a three-concept result; an answer to UHz6's early-steps question.
**Outcome (the whole chain):** a revised paper in a shared Overleaf, a second Overleaf project for the replies, and comment text for you to post.
**Outcome board:** [the paper review board](../report/paper-review/board.md), one row per outcome, updated as each step closes.
**Background:** what was found on disk, the plan weight and the rationale are in [the references file](route-answer-the-iclr-reviews-references.md).

## What we know so far

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#what-you-will-achieve) ➡️

| Fact | Status | Shown by | Changes |
|---|---|---|---|
| The training pool file lists 48 training pairs and 4 held-out pairs (cat×dog, elephant×penguin, piano×candle, skateboard×helmet) | found | `artifacts/_shared/cross_pair_pool_configs/pair_pool_v58.yaml`, 2026-11-10 | PRZE-3a, PRZE-3b |
| Earlier pool files held out only cat×dog and elephant×penguin | found | `pair_pool_v54.yaml` to `pair_pool_v57.yaml`, 2026-11-10 | PRZE-3a |
| Figure 5 shows tent×snowy mountain and lighthouse×stormy sea, which sit in no held-out list | found | the submitted PDF page 7 against the pool files, 2026-11-10 | PRZE-3a |
| The `cross_pair_pooled__r32__lr1e-04__ep600__…` folders are September runs on the v54 pool (24 training pairs, 600 epochs) and sit behind no figure; the paper's 1,000 epochs is what the figure runs used | verified | each folder's `config.json` and `dataset_meta.json`, 2026-11-10 | PRZE-3a, PRZE-11 |
| The adapter of record is v58-03-self at step 50,000 (48 training pairs, 1,000 epochs); Figure 5's "ours" column also uses v57-11b-x0-self-decay at step 50,000 (cat and dog) and v58-05-self-contrast at step 15,000 (elephant and penguin) | verified | each tile's record under `/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/` and each run's `config.json`, 2026-11-10; [the pair lists page](../report/paper-review/prze-3a-pair-lists.md#the-adapter-behind-the-paper) | PRZE-3a, PRZE-5, PRZE-8 |
| The v58 runs held out five pairs including bison×horse; `pair_pool_v58.yaml` dropped bison×horse from its held-out list on 2026-10-21 (commit 08d27d9b), after training; the 48 training pairs are identical | verified | the run's `pair_pool.json` against the yaml, 2026-11-10 | PRZE-3a, PRZE-3b |
| The paper's "ten steps" are step indices 0 to 9 of 50 DDIM steps, timesteps 981 to 801, the noisiest ten; so the paper's adapter is already early-trained | verified | `train_step_range [0, 10]` in each run's `config.json`, `train_pooled.py` line 532, the cache's `meta.json`, 2026-11-10 | PRZE-11 |
| The corrector's update matches the paper's unadjusted Langevin step; its order, step-size schedule and constant differ; adjusted MALA and HMC need an energy that SDXL does not output | read | `poe_repair/methods/_poe_langevin.py` and the code-against-paper audit, 2026-11-10 | PRZE-2 |
| The corrector's update lands on its own exact stationary mean and variance at δ = 0.005, 0.02, 0.05 and 0.1 on N(−1, 0.5) × N(1, 0.8), within 3 Monte Carlo standard errors | verified | `python3 scripts/check_langevin_two_gaussians.py`, 2026-11-11 | PRZE-2 |
| On the toy along SDXL's 50 levels, the paper's step-first order lands nearer the product than our corrector-first order (KL 0.0017 against 0.0031 at c = 3, k = 20), and either order fixes plain PoE's collapse (KL 2.25) | verified | `python3 scripts/check_langevin_order.py`, 2026-11-11 | PRZE-2 |
| The joint-prompt Langevin and unadjusted HMC drivers are byte-identical to the plain samplers at k = 0 on SDXL | verified | Slurm 84583, `paper_review/logs/prze2-identity-84583.out` | PRZE-2 |
| At seed 10, PoE + Langevin removes PoE's fused hybrid but keeps one of the two things on cat × dog, elephant × penguin and tent × snowy mountain, where joint + Langevin and CO3 show both | seen, one seed, by eye | [R6](../artifacts/results/does-a-langevin-corrector-fix-the-product/five_columns.png), 2026-11-11 | PRZE-2 |
| [Attend-and-Excite](../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/06-attend-and-excite-bringing-a-missing-subject-back.md)'s diffusers pipeline targets Stable Diffusion 1.x and 2.x; [CONFORM](../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/07-conform-pulling-look-alike-subjects-apart.md)'s repository loads Stable Diffusion v1.5; neither has an official SDXL port | searched | the diffusers documentation and the gemlab-vt/conform repository, 2026-11-10 | PRZE-1 |
| mscluster112 has one healthy RTX PRO 6000 Blackwell of 97,887 MiB; `/datasets` has 212 TB free | verified | `nvidia-smi` and a torch matmul under `co3_bw` through `scripts/cluster.sh`, 2026-11-10 | PRZE-10 |
| mscluster107 is your interactive job 83423 with two Quadro RTX 8000 of 49,152 MiB each | verified | `nvidia-smi` through `scripts/cluster.sh`, 2026-11-10 | PRZE-10 |
| mscluster108 (two Quadro RTX 8000, `co3`) and mscluster111 (one RTX PRO 6000 Blackwell, `co3_bw`) are idle and pass a torch matmul on every card; mscluster110 passes too but carries another user's job 83315 until about 2026-11-12 11:30 | verified | `nvidia-smi -L`, a 1000 by 1000 matmul, `sinfo -N` and `squeue -w` through `scripts/cluster.sh sh`, 2026-11-10; [the node table](../report/paper-review/prze-10-compute.md#the-nodes-probed-2026-11-10) | PRZE-10 |
| The adapter takes two concept prompts and the null prompt, so a third concept has no input slot | read | submitted PDF Appendix B.1, equation 13 | PRZE-4 |
| The live Overleaf project, exported as `_ICLR2027__LoRA_Plurality.zip`, is the submitted source and the one to edit; neither local copy is: the clone at `paper/overleaf-iclr/` was last pulled before the PDF was built (2026-09-26 13:41 SAST) and holds uncommitted edits, and `paper/iclr/` is an older draft | verified | headings equal, a two-way sentence match and exact `grep` of every flagged sentence, `pdfinfo`, `git log` and `git status` in the clone, 2026-11-10; [the ground check](../artifacts/drips/answer-iclr-reviews/ground-check.md#3-the-local-tex-against-the-submitted-pdf) | PRZE-5 |
| The PDF says the adapter sits on cross-attention (page 1, page 7) and trains at every denoising step (page 1); the adapter of record trained on self- and cross-attention, on steps 0 to 9 only | verified | the PDF text against v58-03-self's `config.json`, 2026-11-10 | PRZE-5, PRZE-11 |
| The CO3 paper (arXiv 2509.25940) scores three prompt sets from Chefer et al. (animals, animals-objects, objects) with [BLIP-VQA](../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/03-blip-vqa-asking-a-question-per-phrase.md) and [ImageReward](../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/02-imagereward-the-preference-number.md), 50-step DDIM, guidance 5.0, on SD1.5, SDXL and baselines run from their public code; it does not state prompts per set, seeds per prompt, resolution or how the two metrics were computed | read | the paper's HTML version, 2026-11-10, and your screenshots of its Tables 1, 5, 6 and 7 | PRZE-1, step 2 (protocol), step 7 (scores) |
| CO3's qualitative figures put one prompt per row and one method per column, with SD1.5 methods and SDXL methods grouped | read | the same HTML version, Figures 3, 4 and 6 | PRZE-2, PRZE-9 |
| The paper's Figure 5 ("ours" column) is four hand-picked cells from three different adapters, one seed per row: the cross+self adapter for cat and dog, v58-05 self-contrast at step 15,000 (raw weights) for elephant and penguin, the v58 self adapter for the tent and lighthouse rows; CO3 draws its own noise | found | `artifacts/results/four-pairs-four-methods/README.md` and `scripts/paper_main_grid.py` lines 21 to 30, 2026-11-10 | PRZE-3a, PRZE-8, PRZE-5, step 4 |
| OpenReview closes Wednesday 18 November; supervisors want the draft Thursday 12 or Friday 13 November | stated by you; confirm | your message, 2026-11-10 | the schedule |
| The export builds with Tectonic on mscluster but not page for page: Tectonic's XeTeX engine ignores the `times` package and sets the body in Latin Modern, so the build runs 14 pages against the submitted 13; the PDF inside the zip is an older 16-page build from 2026-09-25 | verified | Tectonic build and `pdffonts` against the submitted PDF, 2026-11-10; [the step 2 walk map](../artifacts/drips/answer-iclr-reviews/EXECUTE_WALK-step-2.md) | PRZE-5, PRZE-9 |
| Figure 5's elephant and penguin "ours" tile is `ours_v58-05-self-contrast_s015000_rawA_w50.png`, pixel for pixel; the reward fine-tunes `refl-kl-K0` to `K3` (W&B `t4wiwy5p`, `u73uvqyg`, `yj7l40h5`, `ybjd4trt`) start from v58-05 step 15,000 and trained on cat and dog and on elephant and penguin, seeds 1 to 8 | verified | mean absolute pixel difference 0.00 against 6.32 to 14.10 for the fine-tunes; each run's W&B config, 2026-11-10 | PRZE-3a, PRZE-8 |
| The comparison protocol is agreed: one adapter, picked from 9 fixed candidates on cat and dog and elephant and penguin at one shared seed, which then leave every held-out rate; [ToMe](../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/08-tome-fusing-an-object-with-its-attributes.md) joins the SDXL table, five binding methods the SD 1.5 table, FK steering the CO3-protocol table | agreed | your yes, 2026-11-10; [the protocol](../report/paper-review/protocol.md) | PRZE-8, steps 4 to 12 |
| Thirteen slots (R1 to R13) have a size, a placeholder in `paper/iclr-56173/` and a result file; `scripts/refresh_paper_slots.py --check` passes; R1's eight score columns at 8 pt run about 3.4 in wider than the text | verified | Tectonic build, 22 pages, exit 0, 2026-11-10; [the layout sheet](../report/paper-review/layout-sheet.md) | PRZE-9, step 9 |
| Wave 1 is rendering on three cards with none of the three trainings stopped: jobs 1 to 3 on mscluster112 (pid 36785), R1 joint on mscluster108 card 0 (pid 59533), R1 PoE on mscluster108 card 1 (pid 59822), and R1 ours queued on mscluster112 behind jobs 1 to 3 (waiting loop pid 37731); outputs under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/` | running | each launch log in `/datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/`, 2026-11-11; [the jobs table](../report/paper-review/prze-10-compute.md#the-jobs) | PRZE-10, PRZE-1, PRZE-11 |
| One SDXL image in one long process takes about 6.7 s for ours on the RTX PRO 6000 Blackwell, 17 s for the joint prompt and 22 s for PoE on an RTX 8000, at about 8.2 to 8.9 GB of card memory; a CO3 image takes about 50 s on an RTX 8000 | measured | tile timestamps of the first wave 1 tiles and of `paper_review/co3_selection/`, `nvidia-smi`, 2026-11-11 | PRZE-10, steps 6 and 7 |
| The R1 joint prompt is drawn as written: 240 prompts join their concepts with "and", 36 with "with"; the first sidecar reads "a horse with a glasses" | verified | `scripts/review_benchmark_pairs.txt` and `r1_joint/a_horse__x__a_glasses/seed09/mono.json`, 2026-11-11 | PRZE-1 |

## What you will achieve

Navigation: ⬅️ [Previous](#what-we-know-so-far) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

| Stage | Tier | You will have | You will draw or read | Check |
|---|---|---|---|---|
| 1 to 2 Ground and protocol | own | the node table, the real training recipe, an agreed protocol | the pipeline page (step 6) | you can say which checkpoint and why it is frozen |
| 3 and 7 The eleven words | gloss | one entry per word and a one-page scorer table | the two failure-first diagrams | the table has a blind spot for each metric, shown on your images |
| 4 to 5 Held-out set and compute | own | a committed list, a priced job table, wave 1 running | the census contact sheets | the list is committed before any adapter result exists |
| 8 Corrector | own | the Gaussian check passing and the six-column figure | the figure | the printed PASS for four step sizes |
| 9 to 11 Scores, SD 1.5, three concepts | gloss | the three tables, the rates with intervals, the verdicts | the grids | SUPPORTS, NULL or INCONCLUSIVE against the bar written first |
| 12 Gated trainings | own | OPEN or CLOSED with the number | the config beside the paper's recipe | each launch preceded by your yes |
| 13 to 14 Paper and replies | own | the revised paper, the reply files | the compiled PDF | the paper compiles inside the page limit; every number in a reply matches |
| 15 Explain it back | own | the retro walked | none | you explain the five things aloud |
| **Outcome** | | **a revised paper in a shared Overleaf, a replies project, comment text ready to post** | | **posted before Wednesday 18 November** |

## The route at a glance

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#schedule) ➡️

▶ **Proposed (no curated flow fits; steps 2 to 13 settle one per round):** `1 check the ground → 2 write the protocol and the layout sheet → 3 submerge the eleven gap words (the diagram gate now, the journey builds in the background) → 4 choose the held-out set → 5 the compute plan and the first GPU jobs ‖ 6 draw the evaluation pipeline → 7 read the benchmark papers together → 8 corrector proof and six-column figure → 9 scores → 10 SD 1.5 set and its ceiling test → 11 three-concept test → 12 gated trainings → 13 revise the paper and fill the slots → 14 draft the replies → 15 explain it back` (steps 6 and 7 are walks, so they run one after the other while the GPU jobs from step 5 work)

## Schedule

Navigation: ⬅️ [Previous](#the-route-at-a-glance) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Windows are the ones proposed in round 19 (you gave no others). Take the cuts listed in the references file if the total shrinks.

| Block | Steps | Checkpoint at the end of the block |
|---|---|---|
| Tue 10 Nov, evening, 3 h | 1, 2, start 4 | you agreed the protocol; the held-out list is drafted and the census is running |
| Wed 11 Nov, morning, 2 h | finish 4 (freeze by commit), 5 (compute plan, launch wave 1) | the frozen list is committed; the first sampling job runs on mscluster112 and the SD 1.5 set on 107 |
| Wed 11 Nov, afternoon, 3 h | 6, 3, 7 while the GPU works | the pipeline page is published; the one-page scorer table exists |
| Wed 11 Nov, evening, 2 h | 8 | the Gaussian check printed PASS; the six-column figure is built |
| Thu 12 Nov, morning, 2.5 h | 9, 10, 11 read at checkpoints | the three tables and the SUPPORTS, NULL or INCONCLUSIVE line |
| Thu 12 Nov, afternoon, 3 h | 12, 13 | the revised paper compiles inside the page limit and goes to your supervisors |
| Fri 13 to Mon 16 Nov, as you can | 14, 15; the gated trainings run on 112 | the replies drafted with every number matched; any open training evaluated |
| Tue 17 Nov, 1 to 2 h | refresh slots, apply your supervisors' edits, push, upload the revised PDF, post | everything posted before Wednesday 18 Nov |

If Thursday slips, the draft goes to your supervisors on Friday midday with the missing slots as placeholders. Mscluster107's allocation (job 83423) ends about Thursday afternoon, so nothing long runs there after that.

## Tasks

*For the AI to run. Paste each prompt into a session.*

Navigation: ⬅️ [Previous](#schedule) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 1. 🧪 Check the ground: the GPU nodes, what the adapter was trained on, which source to edit

You ~45 min (estimate) · AI ~40 min (estimate) · Sonnet, medium · moves: PRZE-3a, PRZE-5, PRZE-10, PRZE-11

**Objective:** three questions answered with evidence: which nodes work, which pool and checkpoint and noise steps the paper's adapter used, and how the submitted paper differs from the local tex.
**Outcome:** `artifacts/drips/answer-iclr-reviews/ground-check.md`, the node table in the compute page, findings on the pair-list and early-training pages.
**Serves:** needed by steps 2 to 4 (the protocol, the held-out set and the first GPU jobs all name these facts).
**Sees:** none · **Ticket:** none · **Source page:** the submitted paper, Section 4 and Appendix A

```text
/drip-execute-plan --request "Check the ground before any run: the GPU nodes, what the paper's
adapter was trained on, and which paper source to edit"
Model: Sonnet, medium effort. Box: 90 minutes of your time. Filing: tied.
Controls: piece 1 draw (laptop, login node and nodes are named), piece 2 example (one training
state's fields), piece 3 prove (a claimed difference between two sources).
Side routes: none; no installed mode fits better than the walk itself.

Background
The paper "Restoring plurality in compositional diffusion by training correction adapters" is under
review at ICLR 2027 as submission 56173. Three reviewers asked for scores, pair lists and baselines.
The repo is /home/molef/PhD/poe_repair_min on the laptop. Runs happen on mscluster and go through
scripts/cluster.sh. This step checks three facts that every later run depends on.

Objectives
1. Probe mscluster108, 111 and 110 with nvidia-smi -L and a torch CUDA test under co3_bw.
2. Name the pair pool, the checkpoint and the noise steps behind the paper's adapter.
3. List every difference between the submitted paper and the local tex in the sections to edit.

What to read
- report/paper-review/board.md and pages prze-3a, prze-5, prze-10, prze-11: the outcome lines to
  tick.
- plans/route-answer-the-iclr-reviews.md: the table "What we know so far".
- environment/hpc/nodes.md and environment/overview.md: partitions, the co3 and co3_bw interpreters.
- artifacts/_shared/cross_pair_pool_configs/: pool files and the checkpoint folders under
  all_groups.
- poe_repair/paths.py lines 137 to 154: where the training cache and caches are kept.
- 56173_Restoring_plurality_in_c.pdf, Section 4 and Appendix A: the training recipe as stated.
- paper/iclr/iclr2027_conference.tex: the local source.

What is already known
- mscluster112 is healthy: one RTX PRO 6000 Blackwell, 97,887 MiB, a torch matmul passes (verified
  2026-11-10). mscluster107 is the user's job 83423 with two Quadro RTX 8000 of 49,152 MiB.
- /datasets has 212 TB free. The shell on the laptop reaches the cluster through scripts/cluster.sh.
- pair_pool_v58.yaml lists 48 training pairs and 4 held-out pairs. Earlier pools held out two.
- Figure 5 shows tent and lighthouse rows that are in no held-out list.
- A checkpoint folder is named ...r32...ep600..., and the paper says 1,000 epochs.
- The paper says it trains on "ten steps" per trajectory and does not say which.
- The local tex section titles differ from the submitted PDF.

When something is missing
- The Overleaf clone needs a token. Ask the user for the token or for a zip exported from Overleaf.
  Meanwhile compare the local tex with the PDF text only, and mark the result as partial.
- If the training cache or the checkpoint's config file is not where paths.py says, search only
  under /datasets/mmolefe, never with find on /. Ask the user for the path if it is not there.
- If a node does not answer, retry once through scripts/cluster.sh and report it as unreachable.

Rules
- Read only on the cluster. Submit no job, write no file there, and do not clone Overleaf.
- Write only under report/paper-review/ and artifacts/drips/answer-iclr-reviews/.
- Show the node table and the pool and checkpoint finding before ticking any outcome line.

What to hand back
- report/paper-review/prze-10-compute.md updated with the node table.
- report/paper-review/prze-3a-pair-lists.md and prze-11-early-training.md with the findings.
- artifacts/drips/answer-iclr-reviews/ground-check.md: the source differences, with page numbers.
- New rows in "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: python3 ~/.claude/skills/skill-route/scripts/board.py report/paper-review/board.md
  --step 1 prints the counts, and each ticked line cites the command output behind it.

What you will see
Along the way: the node table, then the pool and checkpoint finding, then the source differences,
each followed by one question. At the end: what was asked, what was done with paths, PASS or MISS,
what is open, and at most one question. PASS means all three objectives answered with evidence.
You judge from the ground-check file.
```

- `added:` the checkpoint-identity check (from PRZE-3a) · `added:` which ten steps the training states come from (from PRZE-11) · `dropped:` the paper-reading, diagram and corrector asks (why: later steps)
- [x] **Gate:** you answer the Overleaf question (token, or a zip exported from Overleaf) before step 2 edits the paper.
- **Leaves:** `artifacts/drips/answer-iclr-reviews/ground-check.md`
- **Learning path:** gloss only, [LoRA](/home/molef/Obsidian/Artificial%20Intelligence/Large%20Language%20Models/LoRA.md#where-to-learn-this); no rung is required before this step.

### 2. 🧪 Write the comparison protocol and the figure and table layout sheet

You ~60 min (estimate) · AI ~50 min (estimate) · Opus, medium · moves: PRZE-8, PRZE-9

**Objective:** a protocol page you have agreed, a layout sheet with one row per new figure and table, and placeholder slots in the paper that refill from result files, with the paper still compiling.
**Outcome:** `report/paper-review/protocol.md`, `report/paper-review/layout-sheet.md`, new rows in `paper/iclr-56173/figures.md`, `scripts/refresh_paper_slots.py`.
**Serves:** needed by steps 4 to 12 (no evaluation job starts before you agree the protocol).
**Sees:** none · **Ticket:** none · **Source page:** [the CO3 paper](https://arxiv.org/abs/2509.25940), Table 1 and Figure 3

```text
/drip-execute-plan --request "Write the comparison protocol and the figure and table layout sheet
for the ICLR review response, and set up placeholder slots that refill from result files"
Model: Opus, medium effort (the protocol is design). Box: 60 minutes of your time. Filing: tied.
Controls: piece 1 example (one worked row of the protocol on a real pair), piece 2 draw (the table
and grid layouts), piece 3 debug (the refill script's first run).
Side routes: piece 2 /plan-figures paper/iclr-56173/figures.md (plans the figure set against the
register); piece 3 none.

Background
Reviewers of submission 56173 asked for scores against standard benchmarks, more baselines and
aggregate rates over held-out pairs. Before any run, the comparison rules and the shape of every
new figure and table are fixed, so no result can be accused of an unfair comparison and every
figure fits the ICLR layout cleanly. The repo is /home/molef/PhD/poe_repair_min.

Objectives
1. Write report/paper-review/protocol.md with the rules every result must obey.
2. Write report/paper-review/layout-sheet.md with one row per new figure and table.
3. Add a row per slot to paper/iclr-56173/figures.md, a placeholder at final size in the paper source, and
   scripts/refresh_paper_slots.py that rebuilds a slot from its result file and compiles the paper.

What to read
- report/paper-review/board.md and pages prze-1, prze-2, prze-8, prze-9, prze-11: the outcome lines.
- plans/route-answer-the-iclr-reviews.md: the table "What we know so far".
- paper/iclr-56173/figures.md: the figure register, with statuses reserved, fillable, built and broken.
- paper/iclr-56173/iclr2027_conference.tex lines 25 to 70: the placeholder macros already in the source.
- artifacts/drips/answer-iclr-reviews/ground-check.md: step 1's findings, once it exists.
- 56173_Restoring_plurality_in_c.pdf and Reviews.pdf: the paper and the three reviews.
- environment/paper.md: how the paper builds here, since no system LaTeX exists.

What is already known
- The CO3 paper (arXiv 2509.25940) scores the Chefer et al. prompt sets animals, animals-objects and
  objects with BLIP-VQA and ImageReward at 50-step DDIM and guidance 5.0, and does not state prompts
  per set, seeds, resolution or metric code. We run every method ourselves and never copy numbers.
- Some prompts in those sets may match the 48 training pairs. The main table excludes them.
- Rules the user set: same starting noise per pair and seed, tuning on a set that excludes every
  evaluation pair, the shown images chosen by a fixed rule, network evaluations per image as a
  column, comparisons that differ in more than one thing marked mixed, and every number paired with
  a qualitative grid from the same run.
- Attend-and-Excite and CONFORM run on SD 1.5 with their official code, unmodified.
- The Langevin corrector costs about 20 times a PoE image at 20 steps (estimate), so correctors run
  on a labelled subset and PoE and the adapter are reported on that subset too.
- The paper builds with Tectonic, not pdfLaTeX, per environment/paper.md.

When something is missing
- If the paper does not compile before any edit, stop and report the error. Ask the user whether to
  fix the build or work on the layout sheet only.
- If the placeholder macros cannot hold a slot at the required size, write the sheet with that slot
  marked unresolved and ask the user how wide it should be.
- Overleaf access is not needed here. Everything is edited in the local source.

Rules
- Do not run any job on the cluster. Do not change any result file.
- Do not edit the body text of the paper; only slots, placeholders and the register.
- Show protocol.md to the user and wait for a yes before piece 2.

What to hand back
- report/paper-review/protocol.md and layout-sheet.md.
- New rows in paper/iclr-56173/figures.md, status reserved, each with its claim, layout and result file.
- scripts/refresh_paper_slots.py, and the compile log showing the paper builds with every slot as
  a placeholder.
- Proof: python3 scripts/refresh_paper_slots.py --check prints PASS and the slot count, and
  the board redraws with --step 2 ticking PRZE-8 and PRZE-9.

What you will see
Along the way: the protocol page, then the layout sheet with a drawn sketch of each table and grid,
then the first refill of one slot, each followed by one question. At the end: what was asked, what
was done with paths, PASS or MISS, what is open, at most one question. PASS means the paper
compiles with every slot as a placeholder and one slot refills from a result file. You judge by
opening the compiled PDF.
```

- `added:` the CO3-style table shape, the overlap rule and the multi-method grid (from your CO3 screenshots) · `added:` the Tectonic build fact (from `environment/paper.md`) · `dropped:` Overleaf push (why: needs your token, and the refill works locally first)
- [x] **Gate:** you read `protocol.md` and say yes before any evaluation job starts.
- **Leaves:** `report/paper-review/protocol.md` and `layout-sheet.md`
- **Learning path:** none needed; the step is design.

### 3. 📚 Submerge the eleven gap words

You ~30 min (about 9 of it measured on 4 runs, about 20 at the diagram gate an estimate) · AI ~10 min measured for the open, then ~85 min of background build (borrowed estimate from `/immerse`, 3 runs) · Opus, medium · moves: PRZE-1 (the plain note on each score)

**Objective:** the eleven words each have an entry saying what they measure or change and where they sit in our table, inside a registered journey.
**Outcome:** the approved diagram set and the journey folder, registered in `~/goal-setting/learning/JOURNEYS.md`.
**Serves:** needed by step 6 (reading the benchmark papers) and step 8 (scores) for the plain notes.
**Sees:** none · **Ticket:** none · **Source page:** [the CO3 paper](https://arxiv.org/abs/2509.25940), Table 1

```text
/submerge I want to understand the eleven evaluation tools and rival methods that the ICLR reviewers
named, well enough to use each one and place it in our results: CLIP score, BLIP-VQA, ImageReward,
T2I-CompBench, GenEval, Attend-and-Excite, CONFORM, SynGen, Divide and Bind, InitNO and ToMe
Model: Opus, medium effort. Box: 30 minutes of your time for the open and the gate. Tier: gloss for
every word. Filing: tied (the journey's own folders hold what it makes).
Controls: at the gate options (other ways to group the eleven words) and why (why a node exists).
Side routes: once the journey exists, /submerge --names <journey folder> (the names pass over all
eleven words).

Background
Three reviewers of submission 56173 at ICLR 2027 asked for scores on standard benchmarks and
comparisons with other methods. Reviewer PrzE named T2I-CompBench, GenEval, CLIP and ImageReward.
Reviewer UHz6 named Attend-and-Excite and CONFORM. The CO3 paper (arXiv 2509.25940) scores the
Chefer et al. prompt sets with BLIP-VQA and ImageReward against Attend-and-Excite, SynGen, Divide
and Bind, InitNO and ToMe. We must run these and place them in a table with our adapter, in about
three working days. The repo is /home/molef/PhD/poe_repair_min.

Objectives
1. Check whether an existing journey under ~/goal-setting/learning already covers any of the eleven
   words, and say which, before proposing a new one.
2. Settle with the user the premise and purpose, then draw one diagram of the scoring region and one
   of the rival-methods region, each node naming the failure it exists to fix.
3. Price every node in hours, and let the user trim before any plan is written.
4. Mint the journey, register it in ~/goal-setting/learning/JOURNEYS.md, and record the three roots.

What to read
- Reviews.pdf: the exact wording of the three reviewers' requests.
- plans/route-answer-the-iclr-reviews.md: the table "What we know so far", and step 2's protocol.
- composition/focus/eval_metrics.py: CLIP, EVA-CLIP and ImageReward scorers already written here.
- paper/iclr-56173/figures.md and report/paper-review/prze-1-scores.md: where each score lands.
- report/paper-review/protocol.md and report/paper-review/layout-sheet.md: the agreed comparison rules
  and the table each word lands in (R1 on SDXL in the main text, R5 on SD 1.5 in the appendix).
- The CO3 paper, arXiv 2509.25940, Table 1 and Figure 3: how the same words are used in a table.
- T2I-CompBench++, arXiv 2307.06350, from our own reference list.
- The CONFORM repository, github.com/gemlab-vt/conform, and the Attend-and-Excite pipeline page in
  the diffusers documentation.
- ~/goal-setting/learning/JOURNEYS.md and the term index: what already teaches any of these words.

What is already known
- The purpose is live: step 8 of the route runs these scorers and baselines, and step 12 puts them
  in the paper.
- Attend-and-Excite's pipeline in diffusers targets Stable Diffusion 1.x and 2.x. CONFORM's
  repository loads Stable Diffusion v1.5. Neither has an official SDXL port.
- CO3's authors did not state prompts per set, seeds per prompt, resolution or how BLIP-VQA and
  ImageReward were computed. We run every method ourselves.
- The user is a visual, hands-on learner with little time. Gloss tier means use it and place it.
- The learning tree has no note or plan for any of the eleven words today.
- The protocol is agreed: ToMe runs on SDXL in R1; Attend-and-Excite, SynGen, Divide and Bind,
  InitNO and CONFORM run on SD 1.5 in R5, each from its official code, unmodified. That ToMe's
  official code loads SDXL and that SynGen, Divide and Bind and InitNO load SD 1.x is recalled, not
  checked: open each repository and report the model it loads, since a different answer moves the
  method to the other table.
- R1's and R5's columns are compose rate, CLIP score, ImageReward, BLIP-VQA, T2I-CompBench++,
  GenEval and network evaluations per image.

When something is missing
- If a rival method's official repository is not named above, search for it, show the user the
  candidate, and stage it only after the user confirms.
- If an existing journey covers a word, offer --over that journey rather than a new one.
- If the user cannot render the storyboard images in the gate window, stop at the ASCII diagram and
  leave the cold-start resume line, so the gate continues later.

Rules
- Plan nothing until the integration picture matches what the user imagined.
- Create no journey until the user trims the hour estimates.
- Touch no file outside the learning tree, the playground and Obsidian roots, except the route's
  "What we know so far" table.

What to hand back
- The approved diagram set and the journey folder path, registered in JOURNEYS.md.
- A one-line entry per word, each saying what it measures or changes and where it sits in our table.
- The three roots (learning, playground, Obsidian) written into the journey's gate file.
- New rows in "What we know so far" in plans/route-answer-the-iclr-reviews.md.

What you will see
Along the way: the existing-journey check, then the two diagrams with failures marked, then the
priced node list, each followed by one question. At the end: what was asked, what was made with
paths, what is still open, and at most one question. The step is done when the journey is
registered and every one of the eleven words has an entry. You judge from the journey folder.
```

- `added:` the grouping into two regions and the gloss tier (from your time limit) · `added:` the three roots and registration (from the global rule on journeys) · `dropped:` nothing from your message
- [ ] **Gate:** you trim the hour estimates before any plan is written.
- **Leaves:** the journey folder, registered in `~/goal-setting/learning/JOURNEYS.md`
- **Learning path:** this step is the learning path for step 8; no rung comes before it.

### 4. 🧪 Choose and freeze the evaluation set, the tuning set and the one checkpoint

You ~40 min (estimate) · AI ~30 min (estimate) plus a cluster census of about 1.6 h on the Blackwell (estimate: 480 images at about 12 s) and about 0.5 h for the 144 showcase renders (estimate, same rate) · Sonnet, medium · moves: PRZE-3b, PRZE-3a, PRZE-8

**Objective:** at least 8 pairs in each of three groups (hard, easy but unseen, not competing), the protocol's four tuning pairs, seeds 9 to 16, and the one adapter named by the protocol's showcase rule, all committed before any adapter result on the evaluation pairs exists.
**Outcome:** `report/paper-review/held-out-set.md` with its commit hash, census contact sheets under `artifacts/results/`, the showcase calls that name the adapter, and the result files for slots R7 and R8.
**Serves:** needed by steps 5 to 12: every sampling job, score and figure reads this list.
**Sees:** `artifacts/results/four-pairs-four-methods/` (the three-adapter figure this step corrects) · **Ticket:** none · **Source page:** the submitted paper, Section 4.1 and Figure 5

```text
/drip-execute-plan --request "Choose and freeze the evaluation pairs, the tuning pairs and the one
adapter checkpoint for the ICLR review response, before any adapter result on them is read"
Model: Sonnet, medium effort. Box: 60 minutes of your time. Filing: on, because the census makes
contact sheets that picture-cadence and record-cadence should file.
Controls: piece 1 example (one candidate pair with its prompts), piece 2 prove (a pair's
instance count against its tiles), piece 4 test (that the frozen file carries every field).
Side routes: none; no installed mode fits better than the walk itself.

Background
Reviewers of submission 56173 asked for success rates over held-out pairs and seeds. The paper's
adapter was tested on four pairs, and its main figure used three different adapters. Before any
evaluation, this step fixes which pairs are scored, which are used for tuning, and which one
checkpoint is the paper's adapter. The repo is /home/molef/PhD/poe_repair_min and runs go through
scripts/cluster.sh.

Objectives
1. List the candidate pairs from the pair files and the Attend-and-Excite prompt sets, minus the 48
   training pairs.
2. Run the joint prompt and plain product at seeds 9 to 16 on the cluster and sort the candidates
   into hard, easy but unseen, and not competing.
3. Run the protocol's showcase rule (report/paper-review/protocol.md, rule 1): its 9 candidate
   checkpoints render cat and dog and elephant and penguin at seeds 9 to 16, 144 renders. Call each
   full-resolution tile, show the user every call, and name the winner and the shared seed in rule 1.
4. Write the frozen file and commit it: the three groups, the showcase pairs, the protocol's four
   tuning pairs, seeds 9 to 16 and the named adapter. Then write the result files for slots R7 (the
   pair list) and R8 (the showcase calls) and run python3 scripts/refresh_paper_slots.py --slot R7
   and --slot R8 with --compile.

What to read
- report/paper-review/protocol.md (agreed 2026-11-10): rule 1's showcase rule and candidate table,
  rule 4's pair sets. artifacts/drips/answer-iclr-reviews/ground-check.md: the adapter's training.
- report/paper-review/layout-sheet.md: slots R7 and R8, their sizes and result files.
- artifacts/_shared/cross_pair_pool_configs/: pair_pool_v58.yaml (48 training, 4 held-out) and
  pair_prompts*.yaml (candidate prompts).
- artifacts/results/four-pairs-four-methods/README.md and scripts/paper_main_grid.py lines 21 to 30:
  the three adapters behind Figure 5.
- scripts/composing_unseen_pairs.py, scripts/concept_pair_sheet.py, scripts/figure_candidates.py
  and scripts/census/: sampling and counting scripts that already exist; reuse, do not rewrite.
- environment/hpc/nodes.md and environment/hpc/throughput.md: nodes and measured sampling times.

What is already known
- pair_pool_v58.yaml holds 48 training pairs and 4 held-out pairs. Earlier pools held out two.
- The protocol makes cat and dog and elephant and penguin showcase pairs: used to pick the adapter
  and shown in Figure 5, never counted in a held-out rate. Held out at training and evaluated:
  bison and horse, piano and candle, skateboard and helmet. The tuning set is lion and meerkat,
  camel and pelican, donkey and duck, chicken and rabbit (training pairs) at seeds 9 to 16.
- The reward fine-tunes refl-kl-K0 to K3 trained on both showcase pairs and are not candidates.
- Figure 5 uses three adapters: cross+self (cat and dog), v58-05 self-contrast step 15,000 raw
  (elephant and penguin), v58 self (tent and lighthouse). Source: the README above.
- The compose count is a GroundingDINO instance count, validated on labelled images. It counts
  animals but cannot tell a cat from a dog (earlier notes, 64 of 64 false dog calls).
- The cat and dog joint prompt is wrong on 4 of 8 held-out seeds (earlier notes), so that pair is
  judged per seed.
- Seeds 9 to 16 were never trained on (earlier notes).
- mscluster112 is healthy (one Blackwell, 97,887 MiB). mscluster107 is the user's job with two
  RTX 8000. 108 and 111 were idle and passed a torch check on 2026-11-10; 110 is held by another
  user until about 2026-11-12 11:30. Run under co3_bw on Blackwell nodes and co3 on the others.
- The Attend-and-Excite prompt sets must be fetched from its repository and confirmed by the user;
  their contents are not yet on disk.

When something is missing
- If the Attend-and-Excite prompt lists cannot be fetched, use the pair files only, say so in the
  frozen file, and ask the user to supply the lists.
- If the census job cannot start on mscluster112 (card fault or node busy), run it on mscluster107
  with the same script and note the node. Ask the user before using any other node.
- If fewer than 8 pairs fall in a group, report the count and ask the user whether to widen the
  candidate pool or accept a smaller group.

Rules
- Run the adapter only on the two showcase pairs, never on an evaluation candidate.
- Commit the frozen file before any adapter result on the evaluation pairs is generated, and say the
  commit hash in the closing report.
- Start the cluster job with scripts/cluster.sh after a commit, and show the user the command
  first. Write outputs under /datasets only.

What to hand back
- report/paper-review/held-out-set.md: pairs by group, the showcase pairs, the tuning pairs, seeds 9
  to 16, the named adapter and shared seed, and the commit hash.
- Protocol rule 1 updated with the winner, its calls and the shared seed.
- The R7 and R8 result files, and the compile showing both slots filled.
- Contact sheets of the census under artifacts/results/ with a card entry.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: git log -1 -- report/paper-review/held-out-set.md prints the commit, and the board
  redraws with --step 4 ticking the PRZE-3b, PRZE-3a and PRZE-8 lines this step owns.

What you will see
Along the way: the candidate list, then the census contact sheets with counts, then the proposed
groups, then the showcase calls and the named adapter, each followed by one question. At the end: what was asked, what was
done with paths, PASS or MISS, what is open, and at most one question. PASS means the frozen file
is committed with at least 8 pairs in each group (or the smaller counts accepted by you), the
adapter is named by the showcase rule, and no adapter result exists on the evaluation pairs. You judge from the frozen file and the contact sheets.
```

- `added:` the single-checkpoint rule and the tuning set (from the Figure 5 finding) · `added:` the commit as proof of order (from your fairness rule) · `dropped:` nothing from your message
- [x] **Gate:** you agree the three groups and the showcase calls before the frozen file is committed.
- **Leaves:** `report/paper-review/held-out-set.md`
- **Learning path:** none needed.

### 5. 🧪 The compute plan and the first GPU jobs

You ~30 min (estimate) · AI ~40 min (estimate), plus the jobs · Sonnet, medium · moves: PRZE-10 (and feeds PRZE-1, PRZE-2, PRZE-4, PRZE-11)

**Objective:** every GPU job has a node, a priority and a priced estimate, and the wave 1 jobs are running or finished with output counts matching the plan.
**Outcome:** the compute plan table in `report/paper-review/prze-10-compute.md`, run folders under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/`, and logs.
**Serves:** needed by steps 8 to 12 (every score and figure reads these outputs).
**Sees:** none · **Ticket:** none · **Source page:** `environment/hpc/execution-protocol.md`

| Job | Where | Hours (estimate) | Priority | Waits on |
|---|---|---|---|---|
| Joint, PoE, adapter, CO3 on every cell (960 images, CO3 may cost double) | 112 | 3.2 to 5 | 1 | step 4's frozen file |
| Adapter over steps 0 to 10 and 0 to 25 (480 images) | 112 | 1.6 | 2 | job 1's adapter run |
| Langevin and unadjusted HMC, all 50 steps, k=20, 24 cells each | 112 | 3.4 | 3 | the two-Gaussian proof |
| The same samplers over steps 0 to 10, 48 cells (a labelled variant) | 112 | 1.6 | 4 | the proof |
| SD 1.5 set: joint, PoE, measured correction, A&E, CONFORM | 107, two cards | 2 to 4 | 3, in parallel | step 4 |
| Three-concept test, about 160 images | 107 | 0.5 | 5 | the bar written first |
| Scores for every image | 112 or 107 | about 1 | 6 | jobs 1 to 5 |
| Gated: early-only adapter training | 112 | 7 to 10 | opens only if job 2 shows signal | job 2 |
| Gated: SD 1.5 adapter training | 112 | 7 to 10 | opens only if the ceiling test passes | the SD 1.5 set |

```text
/drip-execute-plan --request "Write the compute plan for the ICLR review response and launch the
first wave of GPU jobs on the cluster, after the user says yes to each launch command"
Model: Sonnet, medium effort. Box: 45 minutes of your time. Filing: on, because job logs and
contact sheets are worth keeping as the record of what ran.
Controls: piece 1 example (one job's row with its measured basis), piece 3 debug (a launch that
does not start), piece 4 test (that output counts match the plan).
Side routes: piece 3 /calibrate --predict <the launch command> (records an 80% interval for runtime
and memory before each job); piece 3 /run-experiment on the node it picks for a job.

Background
Step 4 left a frozen list of evaluation pairs, tuning pairs, seeds 9 to 16 and one adapter
checkpoint. Steps 2 and 4 left the protocol and that list. This step places every GPU job on a node
with a priority and an estimate, then launches the ones that no gate blocks. The repo is
/home/molef/PhD/poe_repair_min and cluster actions go through scripts/cluster.sh.

Objectives
1. Write the compute plan as a table in report/paper-review/prze-10-compute.md: job, node, memory,
   hours with its basis (measured or estimate), priority, and what it waits on.
2. Record an 80% runtime and memory prediction for each wave 1 job before it starts.
3. Launch wave 1 on mscluster112 and mscluster107 by the shared-device path, each after the user
   says yes to the exact command.
4. Poll each job, count its outputs against the plan, and report any job that is stuck.

What to read
- report/paper-review/protocol.md, held-out-set.md and ground-check.md, once the steps leave them.
- environment/hpc/execution-protocol.md: the shared-device path and its safety rules, all mandatory.
- environment/hpc/throughput.md and environment/hpc/nodes.md: measured times and node facts.
- environment/storage.md: the filesystem rule and the disk guard.
- scripts/cluster.sh and the existing launchers in scripts/ (figure_candidates.sbatch,
  composing_unseen_pairs.py, joint_poe_adapter_grid.py): reuse, do not rewrite.
- report/paper-review/prze-1, prze-2, prze-4, prze-11: the outcome lines each job feeds.

What is already known
- The user holds one biggpu slot (interactive job 83423 on mscluster107, two RTX 8000 of 49,152 MiB,
  22 hours 39 minutes elapsed at 2026-11-10, limit 3 days). biggpu allows one job per user, so other
  jobs use the shared-device path (environment/hpc/execution-protocol.md).
- mscluster112 has one healthy RTX PRO 6000 Blackwell of 97,887 MiB; a torch matmul passes under
  co3_bw; /datasets has 212 TB free (verified 2026-11-10).
- One SDXL cell takes about 11 to 16 seconds on the Blackwell and 51 on an RTX 8000 (throughput.md).
- The Langevin corrector costs about 21 times a PoE image at 20 steps per level, over all 50 levels.
- Rank-32 training needs about 25 GB and ran 0.236 seconds per step for rank 8 on the Blackwell.
- Outputs go under /datasets/mmolefe/poe_repair_min/ only, never /home-mscluster.
- Dates: draft to supervisors Thursday 12 or Friday 13 November; OpenReview closes Wednesday 18.

When something is missing
- If mscluster112 is taken by another user's job before launch, run the Blackwell jobs on 110 or 111
  only after probing them with nvidia-smi -L and a torch check, and ask the user first.
- If a job's launcher does not exist, write the smallest one beside the existing launchers and show
  it before running it.
- If the allocation on 107 is near its limit, move the job to 112 and say so.

Rules
- Show each launch command and wait for a yes. Submit nothing, kill nothing and pkill nothing by
  pattern. Never run find on /. Follow every safety rule in the execution protocol.
- Do not launch a gated training in this step.
- Commit before scripts/cluster.sh run, which refuses uncommitted tracked changes.

What to hand back
- The compute plan table with one prediction per wave 1 job.
- Run folders under /datasets/mmolefe/poe_repair_min/outputs/paper_review/<job>/ and logs beside
  them.
- New rows in "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: for each wave 1 job, pgrep on the node shows it running, nvidia-smi shows its memory, and
  the output count matches the plan's expected count by the time it finishes.

What you will see
Along the way: the plan table, then each job's prediction and launch command, then each job's first
outputs, each followed by one question. At the end: what was asked, what was launched with paths,
PASS or MISS, what is open, and at most one question. PASS means every wave 1 job is running or
finished with its output count matching the plan. You judge from the table and the logs.
```

- `added:` the allocation end time of job 83423 (from your screenshot) · `added:` the one-slot rule (from `execution-protocol.md`) · `dropped:` the gated trainings (why: they open on a test result, not on a date)
- [ ] **Gate:** you say yes to each launch command, and to starting any gated training later.
- **Leaves:** the compute plan table and the run folders
- **Learning path:** none needed.

### 6. 🏗️ Draw the evaluation pipeline

You ~40 min (estimate) · AI ~30 min (estimate; no measured median, the mode graph has no entry yet) · Sonnet, medium · moves: PRZE-9

**Objective:** one published page showing seven pieces (methods, inputs, images, scorers, tables, figure slots, machines), each marked where an unfair comparison could enter.
**Outcome:** the page URL, the saved SVG and spec under `artifacts/scenes/`, and its card; a line linking it in `report/paper-review/prze-9-slots.md`.
**Serves:** the picture the layout sheet's slots hang on, and the evaluation paragraph of the paper.
**Sees:** `hld-diagram/references/registry.json` (no page of this system today) · **Ticket:** none · **Source page:** none

```text
/drip-hld-diagram --from plans/route-answer-the-iclr-reviews.md the evaluation pipeline from a
method to an image to a score to a table to a slot in the paper
Model: Sonnet, medium effort. Box: 45 minutes of your time. Filing: tied.
Controls: round 1 options (other ways to cut the pipeline into pieces), piece 4 why (why each scorer
sits where it does), piece 7 plumb (the shared-device launch path).
Side routes: none; hld-diagram's own modes cover the hardware level.

Background
The paper under review at ICLR 2027 (submission 56173) needs a results section with scores,
baselines and held-out rates. The pipeline that makes them has many parts and several unfair
comparisons would be easy to introduce by accident. This step draws the pipeline once so every
later step and the paper's evaluation paragraph agree on it. The repo is
/home/molef/PhD/poe_repair_min.

Objectives
1. Check hld-diagram/references/registry.json for an existing page of this system, and say what is
   found.
2. Cut the pipeline into 4 to 8 named pieces and settle and draw one per round.
3. Add the hardware level for the machines and the filesystems.
4. Mark on each piece where an unfair comparison could enter, and what the protocol says about it.
5. Publish the page, save the SVG and its spec under artifacts/, and write the card.

What to read
- plans/route-answer-the-iclr-reviews.md: the chain, the table "What we know so far", every step.
- report/paper-review/protocol.md, layout-sheet.md, held-out-set.md and prze-10-compute.md, once
  steps 2, 4 and 5 have left them.
- report/paper-review/board.md and the pages for the rows each piece serves.
- environment/overview.md, environment/hpc/nodes.md and environment/storage.md: machines and disks.
- composition/focus/eval_metrics.py: the scorers that already exist here.
- hld-diagram/references/registry.json: existing pages, to avoid a second page of the same system.

What is already known
- No page of this pipeline exists in the registry (18 pages, checked 2026-11-10).
- Methods on SDXL: joint prompt, PoE, PoE + Langevin, CO3, the frozen adapter,
  Feynman-Kac steering. On SD 1.5: Attend-and-Excite and CONFORM with official code, unmodified.
- Adjusted MALA and HMC cannot run on SDXL, which outputs a noise prediction and no energy.
- The scorers: GroundingDINO instance count, CLIP score, ImageReward, BLIP-VQA, GenEval and
  T2I-CompBench prompts, following the CO3 paper's three prompt sets.
- Machines: the laptop writes code and the paper; mscluster107 is the user's job with two RTX 8000;
  mscluster112 has one Blackwell; outputs go to /datasets; Overleaf holds the shared paper.
- Fairness rules from the protocol: same starting noise per pair and seed, tuning pairs disjoint
  from evaluation pairs, one frozen checkpoint, every number beside a qualitative grid.

When something is missing
- If the protocol or the frozen list does not exist yet, draw the pieces that do not depend on
  them, mark the others as planned, and say which are planned.
- If the hardware level needs a fact not in environment/, ask the user, who can run the command.

Rules
- One piece per round. Wait for the user's yes before drawing the next.
- Draw only what exists or is planned in the route. Mark each piece built, planned or open.
- Do not publish until the user has seen the whole picture.

What to hand back
- The published page URL, the saved SVG and spec under artifacts/scenes/, and the card entry.
- A line added to report/paper-review/prze-9-slots.md linking the page.
- Proof: the registry lists the new page, and the page opens with every piece present.

What you will see
Along the way: the registry check, then the piece list, then each piece drawn with its fairness
mark, each followed by one question. At the end: what was asked, what was made with paths and the
URL, PASS or MISS, what is open, and at most one question. PASS means the page is published and
shows all seven pieces with their marks. You judge by opening the page.
```

- `added:` the fairness mark on every piece · `added:` the hardware level · `dropped:` nothing from your message
- [ ] **Gate:** you have seen the whole picture before it is published.
- **Leaves:** the published page and its card
- **Learning path:** none needed.

### 7. 📄 Read the scorers together

You ~60 min (estimate; one measured `--lean` run took 38 min for one paper, too few runs to trust) · AI ~50 min (estimate) · Sonnet, medium · moves: PRZE-1 (the one-page note per score)

**Objective:** every metric has a row (what it measures, scale, blind spot, which of our rows it feeds), with each blind spot shown on one of your own images.
**Outcome:** `artifacts/drips/answer-iclr-reviews/read-the-scorers/COMBINE_MAP.md` and the one-page table, linked from `report/paper-review/prze-1-scores.md`.
**Serves:** needed by step 9 (scores) and the evaluation paragraph of the paper.
**Sees:** the step 6 pipeline page · **Ticket:** none · **Source page:** [the CO3 paper](https://arxiv.org/abs/2509.25940), Table 1

```text
/drip-read --combine --lean "how a benchmark decides that two named objects are both in the picture,
and what its number cannot tell you" 2509.25940 2307.06350 <ImageReward paper> <GenEval paper>
Model: Sonnet, medium effort. Box: 60 minutes of your time. Filing: tied.
Controls: each metric example (its number on the two small-object images), each metric break (an
image that fools it), the last round prove (a claim about where a score sits in our table).
Side routes: after the read, /submerge --names <journey folder from step 3> (the names pass over the
eleven words, once step 3 has made the journey).

Background
Reviewers of submission 56173 asked for scores on standard benchmarks. Before running them we need
to know what each one measures, its scale, and its blind spots, so each number lands in the right
place in the results and is not over-claimed. The repo is /home/molef/PhD/poe_repair_min. The
reader is a visual, hands-on learner with little time who needs to use each metric and place it,
not derive it.

Objectives
1. Resolve and show the arXiv ids of the ImageReward and GenEval papers before reading anything.
2. Read the one idea across the four sources, one chunk per round, on two small-object images.
3. Before each scorer answers, ask the user to predict which image it ranks higher.
4. Leave a one-page table: metric, what it measures, scale, blind spot, which rows of our results
   it feeds.

What to read
- arXiv 2509.25940 (CO3), Table 1 and the evaluation section.
- arXiv 2307.06350 (T2I-CompBench++), the BLIP-VQA description.
- The ImageReward and GenEval papers, once their ids are confirmed.
- composition/focus/eval_metrics.py: CLIP, EVA-CLIP and ImageReward scorers already written here.
- report/paper-review/protocol.md and held-out-set.md, once steps 2 and 4 have left them.
- The step 4 census contact sheets under artifacts/results/: the source of the two small-object
  images.

What is already known
- CO3 scores the Chefer et al. prompt sets animals, animals-objects and objects with BLIP-VQA and
  ImageReward, and does not state prompts per set, seeds, resolution or how the metrics were run.
- Our own count is GroundingDINO instances, validated on labelled images. It counts animals but
  cannot
  tell a cat from a dog (earlier notes, 64 of 64 false dog calls).
- CLIP separates pairs from fused renders when ranking (earlier notes); that is a ranking, not a
  verdict on whether two objects are present.
- The user wants to use each metric and place it, and has about three working days.

When something is missing
- If an arXiv id cannot be resolved, show the candidate titles and ask the user to confirm one.
- If no census images exist yet, use two images from artifacts/results/four-pairs-four-methods/ and
  say so.
- If a scorer cannot run on this machine, say which and run it on the cluster after the user says
  yes.

Rules
- One chunk per round. Predict before the answer, every time.
- Do not run any scorer on the evaluation set; only on the two small-object images.
- Do not copy a published number into our results; cite it as context only.

What to hand back
- artifacts/drips/answer-iclr-reviews/read-the-scorers/COMBINE_MAP.md and the one-page table.
- A vault note written by the walk's own close, in the Obsidian vault.
- A line added to report/paper-review/prze-1-scores.md linking the table.
- Proof: the table has a row for every metric, and each blind spot cites the image that showed it.

What you will see
Along the way: the ids, then each metric's chunk with your prediction first and its answer second,
each followed by one question. At the end: what was asked, what was made with paths, PASS or MISS,
what is open, and at most one question. PASS means every metric has a row with its blind spot shown
on one of your own images. You judge from the table.
```

- `added:` the predict-before-answer rule · `added:` the two small-object images from our own census · `dropped:` the rival-methods reading (why: step 3 covers them, and it is a separate idea)
- [ ] **Gate:** the one-page table exists before step 9 runs the scorers.
- **Leaves:** `artifacts/drips/answer-iclr-reviews/read-the-scorers/COMBINE_MAP.md`
- **Learning path:** gloss only; the eleven words are linked in step 3's journey.

### 8. 🧪 Prove the Langevin corrector and render the six-column figure

You ~120 min at `--depth objective` (estimate) · AI ~150 min (estimate) plus GPU job 3 of step 5 (about 3.4 h on the Blackwell; the Gaussian proof runs on the laptop in seconds) · Opus, medium · moves: PRZE-2, and feeds PRZE-1's baselines

**Objective:** our update passes a check against the exact stationary variance at four step sizes; the two new drivers match the plain sampler at zero steps; c and k are chosen on the tuning pairs by a rule written first; the six-column figure is built from the frozen settings.
**Outcome:** `scripts/check_langevin_two_gaussians.py` and its table, the two drivers, tuning contact sheets, and the figure in its slot with its sidecar file.
**Serves:** the PRZE-2 baseline rows and the figure the reviewer asked to see.
**Sees:** the step 6 pipeline page · **Ticket:** none · **Source page:** [Reduce, Reuse, Recycle](https://arxiv.org/abs/2302.11552), Section 2 and Appendix B

```text
/drip-execute-plan --request --depth objective "Prove our Langevin corrector against the Reduce,
Reuse, Recycle paper on two Gaussians, add the joint-prompt and unadjusted HMC drivers, choose the
settings on the tuning pairs, and render the six-column comparison figure"
Model: Opus, medium effort. Box: 120 minutes of your time. Filing: on, because the tuning contact
sheets and the figure are what picture-cadence and record-cadence should file.
Controls: piece 1 prove (the exact stationary variance against the chain), piece 2 debug (the
driver at zero corrector steps), piece 3 example (one tuning cell with its image), piece 5 draw
(the figure's layout before the render).
Side routes: piece 1 /drip-execute-plan --reproduce 2302.11552 only if the Gaussian check misses
and the paper's own toy is needed as a second reference.

Background
Reviewer PrzE asked for an MCMC corrector baseline. Our Langevin corrector
(poe_repair/methods/_poe_langevin.py) uses the paper's unadjusted Langevin update with a different
step-size rule and a different order of corrector and step. This step proves the update on a
toy with a known answer, builds the joint-prompt and unadjusted HMC drivers, and renders the
figure. The repo is /home/molef/PhD/poe_repair_min.

Objectives
1. Write a numpy check of the update on two Gaussian experts and print the chain's mean and
   variance against the exact stationary values, for four step sizes.
2. Measure the order difference, corrector first against step first, on the same toy.
3. Build the joint-prompt Langevin driver and the unadjusted HMC driver, each byte-identical to the
   plain sampler at zero corrector steps.
4. Choose c and k on the tuning pairs by a numeric rule written before the run, saving an image per
   tuning cell.
5. Render the six-column figure on the fixed seeds from one starting noise per row, CO3 included.

What to read
- poe_repair/methods/_poe_langevin.py: the corrector and its precision rules.
- composition/reduce_reuse_recycle/anneal_samplers.py: the authors' samplers, which differ from
  their paper in 28 places (the audit below); implement from the paper's equations.
- The code-against-paper audit at
  /home/molef/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/drips/read-reduce-reuse-recycle/routes/code-vs-paper.md
- plans/06-is-the-gap-the-samplers-or-the-models/plans/tools/02-the-corrector-and-the-step-size-it-runs-at.md
  and
  report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md
- report/paper-review/protocol.md, held-out-set.md, layout-sheet.md: rules, the tuning pairs and the
  figure's slot.
- scripts/corrector_residual_curve.py and scripts/figure_candidates.py: existing drivers and the
  grid.

What is already known
- Our update matches the paper's unadjusted Langevin step; the order, the step-size schedule and
  the constant differ. Adjusted MALA and HMC need an energy that SDXL does not output.
- On a Gaussian with variance V, the update with step delta has stationary variance
  V / (1 - delta / (2 V)); the product of N(-1, 0.5) and N(1, 0.8) is N(-0.2308, 0.3077) with
  variance as the second argument (audit numbers).
- Single-seed residual ratios scatter 15 to 70 percent; a latent-norm guard is blind to destroyed
  samples; c = 3 with k = 20 was the working point; long chains drift from photo to illustration.
- The corrector costs about 21 times a PoE image at k = 20 over all 50 levels; the 0 to 10 variant
  is a labelled deviation, never the headline row.
- cat and dog and elephant and penguin stay out of every tuning set. CO3 must start from the same
  noise as the other columns.

When something is missing
- If the Gaussian check misses the exact values, stop, show the numbers, and ask the user before
  changing the update.
- If CO3 cannot take an external starting noise, say so, mark its column not matched, and ask the
  user how to proceed.
- If the tuning pairs are not frozen yet, stop and say step 4 is owed.

Rules
- Show the check's pass or miss line before building the drivers.
- The tuning rule is written to the protocol before any tuning cell runs.
- Launch GPU work through the step 5 plan and its commands only, after the user's yes.

What to hand back
- scripts/check_langevin_two_gaussians.py and its printed table.
- The two drivers, the tuning contact sheets and the chosen c and k with the rule that chose them.
- The six-column figure in its slot, with its sidecar file listing seeds, noise and settings.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: python3 scripts/check_langevin_two_gaussians.py prints PASS for all four step sizes, and
  the board redraws with --step 8 ticking the PRZE-2 lines.

What you will see
Along the way: the check's table, then the order measurement, then a tuning contact sheet, then
the figure draft, each followed by one question. At the end: what was asked, what was done with
paths, PASS or MISS, what is open, and at most one question. PASS means the Gaussian check passes
at all four step sizes, the drivers match the plain sampler at zero steps, and the figure is built
from the frozen settings. You judge from the table and the figure.
```

- `added:` the exact stationary variance as the bar · `added:` the rule to tune on tuning pairs · `dropped:` the adjusted MALA and HMC samplers (why: SDXL has no energy to evaluate)
- [ ] **Gate:** you see the Gaussian check's pass or miss line before the drivers are built.
- **Leaves:** the six-column figure and the chosen c and k
- **Learning path:** gloss only, [Langevin](/home/molef/Obsidian/Mathematics/Statistics/Markov%20Chain%20Monte%20Carlo/Langevin%20Dynamics.md#where-to-learn-this).

### 9. 🧪 Score every image and build the tables

You ~60 min at three checkpoints (estimate) · AI ~90 min (estimate) plus about 1 h of scoring on a GPU (estimate; job 7 of step 5) · Sonnet, medium · moves: PRZE-1, PRZE-3b

**Objective:** every image from the evaluation, early-steps, baseline and SD 1.5 jobs is scored with [CLIP score](../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/01-clip-score-and-what-it-cannot-tell-apart.md), ImageReward, BLIP-VQA, [GenEval](../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/05-geneval-a-detector-checks-the-objects.md) and [T2I-CompBench](../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/04-t2i-compbench-a-fixed-prompt-set-per-category.md) (any scorer that cannot be installed in its time box is reported as not run, with the reason), and the verdict against the bar written first is SUPPORTS, NULL or INCONCLUSIVE.
**Outcome:** one long results file with the axis columns on every row, the CO3-style table, the rate table with intervals, and the paired differences, in their slots.
**Serves:** PRZE-1 (scores) and PRZE-3b (rates over the held-out set).
**Sees:** the step 6 pipeline page · **Ticket:** none · **Source page:** [the CO3 paper](https://arxiv.org/abs/2509.25940), Table 1

```text
/drip-execute-plan --request --depth objective "Score every image from the evaluation, early-steps,
baseline and SD 1.5 jobs with CLIP score, ImageReward, BLIP-VQA, GenEval and T2I-CompBench, and
build
the CO3-style table, the rate table and the paired differences, with the bar written first"
Model: Sonnet, medium effort. Box: 60 minutes of your time, at three checkpoints. Filing: on.
Controls: piece 1 prove (each scorer on the two small-object images), piece 3 example (one cell's
row traced from the table to its image), piece 4 break (an image a scorer should fail on).
Side routes: piece 5 /analyze-run <the W&B run, if any> only for a training run, none for sampling.

Background
Reviewer PrzE found no quantitative metric in the paper and asked for T2I-CompBench, GenEval, CLIP
score and ImageReward against standard baselines. The images exist from steps 5 and 8 on the frozen
evaluation set. This step scores them under the protocol, builds the tables the paper needs and
fills
the paper's table slots. The repo is /home/molef/PhD/poe_repair_min.

Objectives
1. Install and verify each scorer on the two small-object images from step 7, and write each one's
   version and settings into the protocol.
2. Write the supports, null, inconclusive and reported-as-is bar into the protocol before any score.
3. Score every image and write one long results file with the axis columns on every row.
4. Build the CO3-style table, the rate table with intervals, and the paired differences.
5. Fill the table slots through scripts/refresh_paper_slots.py and check the paper still compiles.

What to read
- artifacts/drips/answer-iclr-reviews/read-the-scorers/COMBINE_MAP.md: what each metric measures.
- report/paper-review/protocol.md, held-out-set.md, layout-sheet.md and prze-10-compute.md.
- composition/focus/eval_metrics.py: the CLIP and ImageReward scorers already written here.
- The run folders under /datasets/mmolefe/poe_repair_min/outputs/paper_review/.
- environment/hpc/nodes.md: the co3 and co3_bw interpreters and the PyPI speed on the nodes.
- The CO3 paper, arXiv 2509.25940, Table 1, for the table's shape.

What is already known
- The images come from the frozen evaluation set at seeds 9 to 16, one starting noise per pair and
  seed, and one frozen adapter checkpoint (steps 4 and 5).
- Our compose count is a GroundingDINO instance count; it cannot tell a cat from a dog.
- The cat and dog joint prompt is wrong on 4 of 8 held-out seeds, so those seeds are unjudgeable
  for any score that uses the joint render.
- GenEval and T2I-CompBench have no code in this repo today.
- CO3 does not state prompts per set, seeds, resolution or its metric code; do not copy its numbers.
- Seeds within a pair are not independent; intervals resample pairs, then seeds.

When something is missing
- If a scorer will not install in its time box (30 minutes each), skip it, record the reason in
  the protocol, and ask the user whether to try a second approach.
- If an image is missing for a cell, list the cell, do not impute it, and report the count.
- If the frozen evaluation set or checkpoint is not there, stop and say steps 4 and 5 are owed.

Rules
- The bar is in the protocol before the first score is computed.
- Never substitute a different metric under a scorer's name.
- Show the user each checkpoint's result before the next stage: the scorer checks, the CO3-style
  table, then the paired differences.
- Launch scoring jobs through the step 5 plan, after the user's yes.

What to hand back
- The results file under /datasets/mmolefe/poe_repair_min/outputs/paper_review/scores/ and a copy
  of the small tables under report/paper-review/.
- The three tables and the paired differences, in their slots, with sidecar files.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: python3 scripts/refresh_paper_slots.py --check prints PASS, the paper compiles, and the
  board redraws with --step 9 ticking the PRZE-1 and PRZE-3b lines.

What you will see
Along the way: each scorer's result on the two small-object images, then the CO3-style table, then
the paired differences, each followed by one question. At the end: what was asked, what was done
with paths, SUPPORTS, NULL or INCONCLUSIVE against the written bar, what is open, and at most one
question. You judge from the tables and the grids.
```

- `added:` the written bar and the reported-as-is rule · `added:` the unjudgeable-seeds rule for cat and dog · `dropped:` T2I-CompBench's attribute-binding categories (why: the CO3 prompt sets only need its BLIP-VQA metric)
- [ ] **Gate:** the bar is in the protocol before the first score; you read each checkpoint's result before the next.
- **Leaves:** the results file and the three tables
- **Learning path:** gloss only; step 7's table is the rung.

### 10. 🧪 The SD 1.5 comparison set and the ceiling test

You ~45 min (estimate) · AI ~90 min (estimate) plus GPU job 5 of step 5 (2 to 4 h on 107's two cards) and the ceiling test (about 0.5 h) · Sonnet, medium · moves: PRZE-1

**Objective:** joint, plain PoE, the measured correction, Attend-and-Excite and CONFORM run on SD 1.5 over the evaluation pairs from one starting noise per pair and seed (CONFORM marked not matched if it cannot take external latents), and the ceiling test prints PASS or MISS against a bar written first.
**Outcome:** three new scripts, a saved token-index file, the SD 1.5 images and contact sheet, and the decision on the SD 1.5 adapter.
**Serves:** PRZE-1 (the other methods, on the same weights as their baseline) and the gate on the SD 1.5 adapter training.
**Sees:** the step 6 pipeline page · **Ticket:** none · **Source page:** [CONFORM](https://github.com/gemlab-vt/conform) and `composition/aae/README.md`

```text
/drip-execute-plan --request --depth objective "Build the SD 1.5 comparison set with the official
Attend-and-Excite and CONFORM code unmodified, joint and plain PoE, and the measured correction,
and run the ceiling test that decides whether an SD 1.5 adapter is trained"
Model: Sonnet, medium effort. Box: 45 minutes of your time. Filing: on.
Controls: piece 1 example (one pair's token indices), piece 2 prove (the measured correction on a
cell where PoE fails), piece 4 test (that every column used the same starting noise).
Side routes: none; no installed mode fits better than the walk itself.

Background
Reviewer UHz6 asked about Attend-and-Excite and CONFORM. Both are written for Stable Diffusion 1.x
and 2.x, and our adapter is on SDXL, so a fair comparison puts them on the same weights as a PoE
baseline and the measured correction, not on SDXL. This step builds that set and runs a test that
decides whether training an SD 1.5 adapter is worth the hours. The repo is
/home/molef/PhD/poe_repair_min.

Objectives
1. Add SD 1.5 to scripts/same_rule_four_models.py, or a copy of it, for joint and plain PoE.
2. Write the measured-correction sampler for SD 1.x: joint minus PoE added back at full strength.
3. Run Attend-and-Excite from composition/aae/ and CONFORM from its official repository, both
   unmodified, on the same pairs, seeds and starting noise.
4. Write the ceiling bar into the protocol, run the test, and report PASS or MISS.

What to read
- composition/aae/: the official Attend-and-Excite code (README, run.py, the pipeline).
- scripts/same_rule_four_models.py: joint prompt and PoE on four Stable Diffusion models.
- The CONFORM repository, github.com/gemlab-vt/conform, and arXiv 2312.06059.
- report/paper-review/protocol.md, held-out-set.md and prze-1-scores.md.
- poe_repair/methods/_sampling.py: the repo's own PoE rule, to copy the rule exactly.
- environment/hpc/nodes.md: co3 and co3_bw interpreters; mscluster107 has two RTX 8000 of 49 GB.

What is already known
- Attend-and-Excite's official code is already in composition/aae/ and takes a seeded generator.
- The four-models script draws SD 1.x noise from a CUDA generator seeded with the seed, so one seed
  is one tensor within a column.
- CONFORM's official code loads Stable Diffusion v1.5 and runs from a notebook (searched
  2026-11-10); whether it accepts external latents is not yet known.
- The SD 1.5 numbers are a separate table, marked not our adapter and not on SDXL.
- The SD 1.5 adapter training is gated on this test and would take 7 to 10 hours on the Blackwell.
- The tuning pairs and evaluation pairs are frozen by step 4; do not tune on the evaluation pairs.

When something is missing
- If SD 1.5 weights are not on the cluster, search the Hugging Face hub for the maintained mirror,
  show the user the repository, and download only after the user says yes.
- If CONFORM cannot take external latents, run it on its own seeds and mark the column not matched.
- If the evaluation pairs are not frozen, stop and say step 4 is owed.

Rules
- Do not edit Attend-and-Excite or CONFORM code. Call them as released, from a thin script.
- Token indices are computed by one function and saved, never hand-picked.
- Launch GPU work through the step 5 plan after the user's yes.
- The ceiling bar is in the protocol before the test runs.

What to hand back
- The three new scripts and the saved token-index file.
- The SD 1.5 images under /datasets/mmolefe/poe_repair_min/outputs/paper_review/sd15/ and a contact
  sheet with a card entry.
- The ceiling test's PASS or MISS line, and the decision it implies for the SD 1.5 adapter.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: the test script prints PASS or MISS with the rate, and the board redraws with --step 10.

What you will see
Along the way: one pair's token indices, then a contact sheet of the five SD 1.5 columns, then the
ceiling test's rate, each followed by one question. At the end: what was asked, what was done with
paths, PASS or MISS against the written bar, what is open, and at most one question. You judge from
the contact sheet and the rate.
```

- `added:` the thin-script rule so the baselines stay unmodified · `added:` the saved token-index file · `dropped:` the SD 2.x choice (why: CONFORM's official code is built on v1.5)
- [ ] **Gate:** the ceiling bar is in the protocol before the test runs; the SD 1.5 adapter training opens only on a PASS and your yes.
- **Leaves:** the ceiling test's PASS or MISS line
- **Learning path:** none beyond step 3's journey notes on the two methods.

### 11. 🧪 The three-concept test

You ~30 min (estimate) · AI ~45 min (estimate) plus about 0.8 h of GPU (240 images at about 12 s on the Blackwell, plus a few tuning images) · Sonnet, medium · moves: PRZE-4

**Objective:** joint, plain three-concept product and the frozen adapter inside a three-concept product, run on about 10 triples at seeds 9 to 16 from one starting noise per triple and seed, with the triples and the bar committed before any image exists, and the verdict SIGNAL or NO SIGNAL.
**Outcome:** `report/paper-review/three-concept-set.md`, the images and contact sheet, the rates with intervals, and the figure in its slot.
**Serves:** PRZE-4 (PrzE's question 2).
**Sees:** the step 6 pipeline page · **Ticket:** none · **Source page:** the submitted paper, Appendix B.1, equation 13

```text
/drip-execute-plan --request --depth objective "Test the frozen adapter inside a three-concept
product against plain three-concept PoE and the joint prompt, with the triples and the bar written
before any image exists"
Model: Sonnet, medium effort. Box: 30 minutes of your time. Filing: on.
Controls: piece 1 example (one triple's four branches), piece 3 test (that the adapter was active in
all four branches), piece 4 draw (the grid's layout before the render).
Side routes: none; no installed mode fits better than the walk itself.

Background
Reviewer PrzE's second question asks how the method performs when composing more than two concepts.
The adapter was trained on pairs, takes two concept prompts and the null prompt, and has no input
slot for a third. This step tests it zero-shot inside a three-concept product, with a bar written
before the run. The repo is /home/molef/PhD/poe_repair_min.

Objectives
1. Write the triple-selection rule and the list of about 10 triples and 3 tuning triples, and commit
   them before any image exists.
2. Write the bar into the protocol: the signal and no-signal intervals.
3. Tune the guidance weight for plain 3-PoE and the adapter equally on the tuning triples only.
4. Run joint, plain 3-PoE and adapter 3-PoE on 10 triples at seeds 9 to 16 from one starting noise
   per triple and seed, and report the rates and the interval.
5. Fill the figure slot with the grid, failures included.

What to read
- report/paper-review/protocol.md, held-out-set.md and prze-4-three-concepts.md.
- poe_repair/methods/_sampling.py: the repo's PoE rule and where the adapter is attached.
- The submitted paper, Appendix B.1: the adapter's three branches (equation 13).
- artifacts/_shared/cross_pair_pool_configs/pair_pool_v58.yaml: the 48 training pairs, whose
  concepts the triples must avoid.
- scripts/joint_poe_adapter_grid.py and scripts/figure_candidates.py: existing grid builders.

What is already known
- The adapter is a rank-32 module on the cross-attention layers, with no input slot for a third
  concept. It is trained on pairs only.
- The paper sets both guidance weights to 7.5; a three-term product has no published weight.
- Our instance count cannot tell species; it counts instances. A contact sheet is read by label.
- SDXL's joint prompt fails more often as concepts are added, so triple targets would be weak.
- Training a three-concept adapter is out of scope here; it opens only on the user's yes.

When something is missing
- If fewer than 8 triples avoid the training concepts, report the count and ask the user whether to
  accept fewer or allow a triple that reuses one training concept, labelled.
- If the frozen checkpoint cannot be attached to a four-branch call, stop and show the error.
- If the frozen file from step 4 is missing, stop and say step 4 is owed.

Rules
- The triples, rule and bar are committed before the first image is generated.
- Tune on tuning triples only. Never look at the 10 triples to choose a weight.
- Launch GPU work through the step 5 plan after the user's yes.

What to hand back
- report/paper-review/three-concept-set.md with the rule, the triples and the commit hash.
- Images under /datasets/mmolefe/poe_repair_min/outputs/paper_review/three_concepts/, a contact
  sheet with a card entry, and the rates with intervals.
- The figure in its slot, and rows added to "What we know so far" in the route file.
- Proof: the script prints SIGNAL or NO SIGNAL with the interval, and the board redraws with
  --step 11 ticking the PRZE-4 lines.

What you will see
Along the way: the triple list and the bar, then the tuning result, then the contact sheet and the
rates, each followed by one question. At the end: what was asked, what was done with paths, SIGNAL
or
NO SIGNAL against the written bar, what is open, and at most one question. You judge from the
contact sheet and the interval.
```

- `added:` the tuning triples for the guidance weight · `added:` the concepts-must-avoid-training rule · `dropped:` nothing from your message
- [ ] **Gate:** the triples, rule and bar are committed before the first image is generated.
- **Leaves:** `report/paper-review/three-concept-set.md` and the SIGNAL or NO SIGNAL line
- **Learning path:** none needed.

### 12. 🧪 Decide and launch the two gated trainings

You ~30 min (estimate) · AI ~60 min (estimate) plus 7 to 10 GPU hours per open training (your check) and about 0.8 h to evaluate each · Opus, medium · moves: PRZE-11 and, if the ceiling passes, PRZE-1

**Objective:** each training is OPEN or CLOSED against its written bar with the number shown, and each open one is launched only after your yes, matched to the paper's recipe in everything except the axis under test.
**Outcome:** for each training, the verdict, and for each open one the matched config, the prediction, the launch command and the W&B run id.
**Serves:** PRZE-11 (the early-steps question) and the SD 1.5 adapter row of PRZE-1.
**Sees:** the step 6 pipeline page · **Ticket:** none · **Source page:** the submitted paper, Appendix A

| Training | Gate that must pass first | Cost |
|---|---|---|
| Early-only adapter, states from steps 0 to 10 only | the early-steps inference test (step 5, job 2) moves the failure-type counts by the margin the protocol sets first | 7 to 10 GPU hours on the Blackwell, plus about 0.8 h to evaluate |
| SD 1.5 adapter | the ceiling test (step 10) passes | 7 to 10 GPU hours, plus a new pair filter and caches, plus evaluation |

```text
/drip-execute-plan --request --depth objective "Decide from the gate results whether to train the
early-only adapter and the SD 1.5 adapter, and launch only the ones whose gate passed, each matched
to the paper's recipe in everything except the axis under test"
Model: Opus, medium effort. Box: 30 minutes of your time. Filing: on.
Controls: piece 1 example (the matched config beside the paper's recipe, line by line), piece 3
debug
(a launch that does not start), piece 4 prove (that the new run differs from the old in one axis).
Side routes: piece 3 /calibrate --predict <the launch command> (80% intervals for runtime, memory
and
an outcome range); piece 4 /analyze-run <the W&B run> once the training is running.

Background
Reviewer UHz6 asked whether restricting the adapter to the early steps changes the failure types.
Both reviewers wanted a fair comparison with other methods on the same weights. Each training
answers one of these, and each opens only if a cheap test shows it can pay off. The repo is
/home/molef/PhD/poe_repair_min and cluster actions go through scripts/cluster.sh.

Objectives
1. Read the early-steps inference test (step 5, job 2) and the ceiling test (step 10) against their
   written bars and say, for each training, open or closed.
2. For each open training, write the matched config next to the paper's recipe, line by line, and
   show
   the user every difference.
3. Record a prediction for each open training, then launch it after the user's yes.
4. Set up the evaluation: the same frozen checkpoint-choice rule on the tuning set, then sampling on
   the evaluation set from the same starting noise.

What to read
- report/paper-review/protocol.md: the bars for both gates and the checkpoint-choice rule.
- report/paper-review/prze-11-early-training.md and prze-1-scores.md: the outcome lines.
- artifacts/drips/answer-iclr-reviews/ground-check.md: the paper's real recipe, pool and noise
  steps.
- environment/hpc/throughput.md: step times and the memory a rank-32 training needs.
- The training package under poe_repair/experiments/cross_pair_lora_pooling/ and its launchers
  (scripts/launch_sdlora.sh and the pooled-training launchers).
- The submitted paper, Appendix A: the stated training recipe.

What is already known
- Rank-32 training needs about 25 GB and takes about 0.236 seconds per step for rank 8 on the
  Blackwell
  (measured 2026-09-01); the user's check puts the full runs at 7 to 10 hours.
- The folder name of one checkpoint says 600 epochs while the paper says 1,000 (to be resolved by
  ground-check.md).
- The paper's pair filter keeps a pair if the joint prompt or CO3 composes; on SD 1.5 only the joint
  prompt can be used, which makes that comparison mixed.
- biggpu allows one job per user; training runs on mscluster112 by the shared-device path.
- Draft to supervisors is Thursday 12 or Friday 13 November; OpenReview closes Wednesday 18.

When something is missing
- If a gate's result is not there yet, say which step is owed and do nothing else.
- If the matched config cannot reproduce a line of the paper's recipe, show the line and ask the
  user.
- If the Blackwell is busy, ask before moving a training to a slower card.

Rules
- Launch nothing without both a passed gate and the user's yes to the exact command.
- Change exactly one axis per training and list every difference from the paper's recipe.
- Do not choose a checkpoint by looking at the evaluation set.
- Follow the execution protocol's safety rules for the shared-device path.

What to hand back
- For each training, open or closed with the number behind it, and for each open one the matched
  config, the prediction, the launch command and the W&B run id.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: the board redraws with --step 12, and each launched run shows in pgrep and nvidia-smi on
  its node.

What you will see
Along the way: each gate's result against its bar, then the config beside the paper's recipe, then
the
prediction and launch command, each followed by one question. At the end: what was asked, what was
done with paths, OPEN or CLOSED for each training with its number, what is open, and at most one
question. You judge from the gate numbers and the config diff.
```

- `added:` the single-axis rule and the same checkpoint-choice rule · `added:` the mixed label for the SD 1.5 pair filter · `dropped:` the three-concept training (why: closed by your decision, gated on the zero-shot test)
- [ ] **Gate:** each training launches only after its gate passed and you say yes to the exact command.
- **Leaves:** the verdicts and the W&B run ids
- **Learning path:** none needed.

### 13. ✍️ Revise the paper

You ~120 min (estimate) · AI ~90 min (estimate) · Opus, medium · moves: PRZE-5 (and fills every slot that has a result)

**Objective:** the working copy compiles under the ICLR template inside the page limit with the mechanical fixes, appendix tables, a compact evaluation section (one table, one figure in the main text), and a limitations paragraph including the closed gates with their numbers.
**Outcome:** the revised tex, in `paper/iclr-56173/`, which you upload to Overleaf, the compiled PDF, and `paper/iclr-56173/CHANGES_FOR_SUPERVISORS.md`.
**Serves:** PRZE-5 (the paper revised and shared) and the paper-side lines of PRZE-1 to PRZE-4.
**Sees:** the step 6 pipeline page · **Ticket:** none · **Source page:** `paper/iclr-56173/DRAFT_MAP.md`

```text
/drip-write the evaluation section and the limitations paragraph of the revision for the ICLR
reviews, after the AI applies the mechanical fixes and appendix tables as one diff for one yes
Model: Opus, medium effort. Box: 120 minutes of your time. Filing: tied.
Controls: each sentence example (the table row it describes), the evaluation section prove (every
number in the prose matches its table), the limitations paragraph why (why each limit is there).
Side routes: after the draft, /drip-proofread <the revised sections> (a full proofread pass over the
edited text).

Background
Submission 56173 got three reviews at ICLR 2027. Reviewer PrzE asked for scores on standard
benchmarks, more baselines, the pair lists and more than two concepts. The earlier steps made the
protocol, the held-out set, the scores, the corrector figure, the SD 1.5 set and the three-concept
test. This step puts them in the paper. paper/iclr-56173/ is the working copy and the user uploads it to Overleaf.
The repo is /home/molef/PhD/poe_repair_min and the manuscript map is paper/iclr-56173/DRAFT_MAP.md.

Objectives
1. Confirm the page limit and the working copy (paper/iclr-56173/) from the call, the template and
   the step 1 comparison.
2. Apply the mechanical fixes and appendix tables as one diff, and show it for one yes.
3. Draft the evaluation section, one sentence at a time with the user steering: one compact table
   and one figure in the main text, the rest in the appendix.
4. Draft the limitations paragraph, including the closed gates with their numbers.
5. Compile, count pages, run the refresh script's check, and write a plain list of changes per
   reviewer point.

What to read
- paper/iclr-56173/DRAFT_MAP.md and the working copy's tex: where each section stands.
- report/paper-review/board.md, protocol.md, layout-sheet.md, held-out-set.md and
  three-concept-set.md.
- The results files and tables from steps 8 to 11, in their slots.
- Reviews.pdf: the exact wording of every point being answered.
- The manuscript voice and number rules in the memory notes, and environment/paper.md for the build.
- 56173_Restoring_plurality_in_c.pdf: the submitted text, Sections 3 to 5 and Appendices A and B.

What is already known
- The submitted paper's Section 4 says the figure on the training diagram is "Figure 10" (it is
  Figure 8), and Section 5 has a broken sentence (reviewers UHz6 and k4nd).
- The training loss is only in an appendix diagram and must be an equation in Section 4 (k4nd).
- Figure 5's caption says CO3 used different starting noise while Section 4.1 says the same (k4nd).
- Figure 1 (final_figures/molefe_fig1.png) has its text baked into the pixels, printing at 3 to
  4 pt against the 8 pt rule, in two faces (serif in panel a, sans in panel b), with panel letters
  "a)" and "b)" where the redrawn figures use "(a)"; panel b's pictures print about 0.3 in wide.
  The fix is the re-lettering recipe in runbook/making-the-papers-figures/.
- Figure 7 (final_figures/molefe_fig2.png) has its text baked in at 1.3 to 1.8 pt in a sans face;
  its note "Adapter trained for 1,000 epochs" is step 50,000 by Appendix A's 50 steps per epoch,
  so its renders come from a later checkpoint than the adapter of record (v58-03-self, step 30,000,
  protocol rule 1) and are rendered again from it; the red zoom boxes in panel b are coloured frames, and the top-right box runs off the
  picture's edge; it prints at 0.9 of the text width, leaving white space at the sides.
- The paper builds with Tectonic, not pdfLaTeX (environment/paper.md).
- Closed gates are reported with the number that closed them. Mixed comparisons are marked mixed.
- Draft to supervisors Thursday 12 or Friday 13 November; OpenReview closes Wednesday 18.

When something is missing
- If paper/iclr-56173/ is missing or older than the Overleaf project, ask the user for a fresh
  export zip from Overleaf and unpack it there.
- If a result is missing for a slot, leave the placeholder and list the slot.
- If the paper exceeds the page limit, move the least central material to the appendix and show the
  user the list before cutting.

Rules
- Never ask for, print or store an Overleaf token. The user clones and pushes.
- Prose changes go through the user one sentence at a time; mechanical changes through one diff.
- Do not change a number except by refilling its slot from its result file.
- Do not claim more than the written bar allows.

What to hand back
- The revised tex in the working copy, committed locally, with the compiled PDF.
- paper/iclr-56173/CHANGES_FOR_SUPERVISORS.md: one line per edit, grouped by reviewer point.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: the compile log shows no errors and the page count; the refresh script's check prints PASS;
  the board redraws with --step 13 ticking the PRZE-5 lines this step owns.

What you will see
Along the way: the page limit, then the mechanical diff, then the evaluation section a sentence at
a time, then the limitations paragraph, each followed by one question. At the end: what was asked,
what was done with paths, PASS or MISS against compiling inside the page limit, what is open, and at
most one question. You judge by opening the compiled PDF.
```

- `added:` the no-token rule and the user-pushes rule · `added:` the closed-gates sentence · `dropped:` the fixes for reviewers UHz6's and k4nd's other points (why: you asked to focus on PrzE; the shared fixes cost nothing)
- [ ] **Gate:** you read the page limit and say yes to the mechanical diff before prose starts.
- **Leaves:** the revised tex, the PDF and `paper/iclr-56173/CHANGES_FOR_SUPERVISORS.md`
- **Learning path:** none needed.

### 14. ✍️ Draft the replies

You ~90 min (estimate) · AI ~45 min (estimate) · Opus, medium · moves: PRZE-6 and the text line of PRZE-7

**Objective:** a reply to Reviewer PrzE, point by point, in a markdown source that builds a LaTeX file for the Overleaf replies project and a plain-text file for OpenReview, with every number matched to the revised paper by a script.
**Outcome:** `replies/prze.md`, `replies/common.md`, the built zip, one plain-text file per comment with its character count, and the number check's output.
**Serves:** PRZE-6 (replies drafted) and PRZE-7 (the comment text).
**Sees:** none · **Ticket:** none · **Source page:** `Reviews.pdf`, Reviewer PrzE

```text
/drip-write the reply to Reviewer PrzE for submission 56173, point by point, after the AI builds the
reply files and the number check
Model: Opus, medium effort. Box: 90 minutes of your time. Filing: tied.
Controls: each part example (the table row or figure it cites), the opening why (why the reviewer's
strengths are named first), the closing prove (every number matches the revised paper).
Side routes: after the draft, /drip-proofread <the reply files> (a proofread pass over the replies).

Background
Reviewer PrzE rated submission 56173 a 1 with confidence 4 and listed two strengths, two weaknesses
and two questions. The paper has been revised (step 13). This step writes the reply, point by
point, in a markdown source that builds a LaTeX file for the Overleaf replies project and a plain
text file for OpenReview. The repo is /home/molef/PhD/poe_repair_min.

Objectives
1. Read the OpenReview forum page for the comment length limit and format, and say what it is.
2. Build replies/prze.md, scripts/build_reply_project.py and scripts/check_reply_numbers.py.
3. Draft the opening, weakness 1 with question 1, weakness 2, question 2 and the closing, one part
   at
   a time with the user steering.
4. Draft the short common comment to all reviewers: what changed, with the section and figure list.
5. Run the number check, build the Overleaf zip and the plain-text files, and print character
   counts.

What to read
- Reviews.pdf: Reviewer PrzE's exact words, and the other two reviews for overlapping points.
- paper/iclr-56173/CHANGES_FOR_SUPERVISORS.md and the revised paper's tables: the facts the reply cites.
- report/paper-review/board.md, protocol.md and the results tables from steps 8 to 11.
- The OpenReview forum page for submission 56173 (the user's browser, or a pasted copy).

What is already known
- Reviewer PrzE's strengths: the early-stage ablation (Figures 3 and 4) and a computable quantity
  between the joint prompt and PoE. Weaknesses: no quantitative benchmarks, narrow baselines, the 48
  pairs not listed. Questions: evaluation scores, and results for more than two concepts.
- Figure 5 used three adapters in the submission and uses one frozen checkpoint in the revision.
- Adjusted MALA and HMC cannot run on SDXL, which outputs no energy; the unadjusted versions can.
- Any baseline that beats the adapter in a table is reported as it came out.
- The user posts on OpenReview and creates the Overleaf replies project; Claude builds the files.

When something is missing
- If a number in a draft has no match in the paper or results file, remove it or ask the user for
  the source. Never estimate.
- If the character limit is unknown, assume 5,000 and ask the user to confirm on the forum page.
- If a result the reply needs does not exist yet, write the sentence as a placeholder marked TODO.

Rules
- Name the reviewer's own words first, in one clause, before answering each point.
- Say what changed and where, with the section, table or figure number.
- Claim nothing beyond the written bar. No defensive phrasing.
- Plain text for OpenReview: no LaTeX, ordinary punctuation, no dashes between clauses.
- Never ask for or store any Overleaf or OpenReview credential.

What to hand back
- replies/prze.md and replies/common.md, the built LaTeX project zip, and one plain-text file per
  comment with its character count.
- The number check's output with every number matched.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: python3 scripts/check_reply_numbers.py prints PASS with the count of numbers matched, and
  the board redraws with --step 14 ticking the PRZE-6 lines.

What you will see
Along the way: the forum's limit, then each part of the reply drafted with the table it cites, each
followed by one question. At the end: what was asked, what was done with paths, PASS or MISS on the
number check, what is open, and at most one question. You judge by reading the plain-text files.
```

- `added:` the number check and the two disclosures · `added:` the common comment · `dropped:` replies to UHz6 and k4nd (why: you asked to focus on PrzE first)
- [ ] **Gate:** every number in the reply matches the revised paper before you post.
- **Leaves:** the plain-text comment files and the replies zip
- **Learning path:** none needed.

### 15. 📖 Explain it back

You ~40 min (estimate: about 10 for the merge, about 30 for the retro) · AI ~20 min (estimate) · Sonnet, medium · moves: PRZE-5's explanation line

**Objective:** you explain, without notes, the five things a reviewer or supervisor would press on, and check each against its evidence file.
**Outcome:** `artifacts/notes/session-merges/2026-11-iclr-reviews-prze-retro.md` updated with each prediction and its outcome, and the list of things you could not yet explain.
**Serves:** the retention step of the chain; no deliverable waits on it.
**Sees:** the board and the row pages · **Ticket:** none · **Source page:** none

The five things: why one frozen checkpoint and what the three-adapter Figure 5 was; what the corrector proof showed and did not show; what each score measures and its blind spot on our own images; what the [held-out groups](../report/paper-review/held-out-set.md#the-held-out-groups) are and why the tuning set is separate; where a baseline beats the adapter if one does, and what the three-concept and early-steps tests said.

```text
/merge-sessions --retro <the session ids of steps 1 to 14, one per line, listed by the user>
Model: Sonnet, medium effort. Box: 10 minutes of your time. Filing: tied.
Controls: none; the merge runs and leaves one file.
Side routes: none.

Background
This is the retrospective of the work that answered Reviewer PrzE's review of submission 56173: the
ground check, protocol, held-out set, compute plan, corrector proof, scores, SD 1.5 set,
three-concept
test, gated trainings, paper revision and replies. The retro file feeds /drip-retro. The repo is
/home/molef/PhD/poe_repair_min.

Objectives
1. Merge the listed sessions into one retro file in dependency order, checked against the
   transcripts
   and the disk.
2. For each chunk, state the question it answered, the bar written first, and the result.

What to read
- The session transcripts named by the user, and artifacts/drips/answer-iclr-reviews/ROUTE_WALK.md.
- plans/route-answer-the-iclr-reviews.md: the "What we know so far" table, as the spine.
- report/paper-review/board.md and every row page's outcome lines.

What is already known
- Five things must be explainable afterward: the one frozen checkpoint, the corrector proof, what
  each score measures and its blind spot, the held-out groups and tuning set, and where a baseline
  beats the adapter if one does.

When something is missing
- If a session id has no transcript, list it and carry on with the rest.
- If a chunk has no recorded bar, say so in the file instead of inventing one.

Rules
- Check every claim against the disk. Write nothing the transcripts and files do not show.

What to hand back
- artifacts/notes/session-merges/2026-11-iclr-reviews-prze-retro.md.

What you will see
The retro file's path and its chunk list, and at most one question.
```

```text
/drip-retro artifacts/notes/session-merges/2026-11-iclr-reviews-prze-retro.md
Model: Sonnet, medium effort. Box: 30 minutes of your time. Filing: tied.
Controls: each chunk example (the table or image behind it); the last chunk prove (explain the five
things aloud, then check each against its file).
Side routes: after the retro, /quiz-me <the retro file> (a short live quiz on the five things, with
the misses recorded for next time).

Background
The retro file merges the sessions that answered Reviewer PrzE. The reader will meet reviewers and
supervisors who press on exactly the five things listed in the retro. Retention matters more than
speed here: the reader must be able to defend the work without notes.

Objectives
1. Walk each chunk, asking the reader to predict the result before it opens.
2. Record each prediction and whether it held.
3. End by having the reader explain the five things, and check each against its evidence file.

What to read
- artifacts/notes/session-merges/2026-11-iclr-reviews-prze-retro.md, and the board and tables it
  cites.

What is already known
- The reader has seen every result already; the aim now is explaining, not discovering.

When something is missing
- If a result file named in a chunk is missing, say so and skip that prediction.

Rules
- One chunk per round. Prediction first, result second.

What to hand back
- The retro file updated with each prediction and its outcome, and the list of things the reader
  could not yet explain.

What you will see
Along the way: each chunk's question, then the result after your prediction. At the end: which of
the five things you explained correctly, which missed, and at most one question.
```

- `added:` the quiz side route · `dropped:` `/pressure-test` (why: it vets ideas before any experiment, and these have run)
- [ ] **Gate:** you can explain all five things; any you cannot goes to the next sitting's first rung.
- **Leaves:** the updated retro file
- **Learning path:** none needed.

## Instructions

*For you to do by hand. Nothing here is delegated.*

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

### A. 🖱️ Share the paper's Overleaf project with your supervisors

- [ ] **Do:** open the paper's project in Overleaf, click Share (top right), add each supervisor's email, choose "Can edit", click Share.
- **You should see:** both names listed under the project's Share dialog.
- **Done when:** a supervisor confirms they can open the project.

### B. 🖱️ Download a fresh export of the paper before step 13

- [ ] **Do:** in the browser, open the paper's Overleaf project, click **Menu** (top left), then under **Download** click **Source**. Save the zip, then tell Claude its path so it is unpacked over `paper/iclr-56173/` (the slots, `figures.md` and the `\reviewslot` macro are kept).
- **You should see:** a zip of about 119 MB holding `iclr2027_conference.tex` and `figures/`.
- **Done when:** `python3 scripts/refresh_paper_slots.py --check` prints PASS after the unpack.

### C. 🖱️ Upload the revised paper to Overleaf after step 13

- [ ] **Do:** in the Overleaf project's file tree, click the **Upload** icon and drop in each file step 13 changed (`iclr2027_conference.tex`, every file under `slots/`, any new file under `figures/`), choosing **Overwrite** when asked. Nothing is pushed with git.
- **You should see:** **Recompile** produces the revised PDF with the filled slots.
- **Done when:** the Overleaf PDF matches `paper/iclr-56173/build/iclr2027_conference.pdf` page for page, or differs only by the font change the build fact in "What we know so far" names.

### D. 🖱️ Create the replies project in Overleaf (after step 14)

- [ ] **Do:** Overleaf, New Project, Upload Project, choose the zip step 14 builds, name it "ICLR 2027 56173 replies", then Share with your supervisors as in A.
- **You should see:** `main.tex` compiling to one PDF with PrzE's reply first.
- **Done when:** the PDF shows every point of PrzE's review answered.

### E. 🖱️ Render the storyboard images at step 3's diagram gate

- [ ] **Do:** when the walk gives you image prompts, paste each into ChatGPT, then paste the images back into the walk.
- **You should see:** one image per node region, matching the ASCII diagram.
- **Done when:** the integration picture matches what you imagined.

### F. 🖱️ Post on OpenReview (after step 14, by Tuesday 17 November)

- [ ] **Do:** log in to OpenReview, open the forum page for submission 56173, find Reviewer PrzE's review, click "Official Comment", paste the plain-text file from step 14, check the character count against the limit shown, submit. Then upload the revised PDF through the submission's Edit menu (the button names are the venue's; confirm on the page).
- **You should see:** your comment under PrzE's review and a new entry in the submission's Revisions list.
- **Done when:** the forum page shows both, and you save the link.

### G. ✍️ Say yes to each cluster launch and each gated training

- [ ] **Do:** read the exact command the walk shows and reply yes, or ask for a change.
- **You should see:** `pgrep` and `nvidia-smi` on the node showing the job, per step 5.
- **Done when:** the job's output count matches the plan.

## Quizzes and feedback

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Step 15 is the quiz: `/drip-retro` with a prediction before each chunk, then `/quiz-me` on the five things. A slip goes to the next sitting's first rung. A gap goes to the journey from step 3, learned at the gloss tier, then retaken.

## Next step

Navigation: ⬅️ [Previous](#quizzes-and-feedback) | 📋 [TOC](#table-of-contents)

**Start with:** step 1, check the ground, tonight. Say `re-route <what changed>` for a different chain, `calendar` to put the schedule on Google Calendar, or `todoist` to file the runbook as ordered subtasks. The background is in [the references file](route-answer-the-iclr-reviews-references.md).
