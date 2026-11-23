# Reproducing ToMe on T2I-CompBench and RareBench

## What this asks

Does the authors' released Token Merging (ToMe) code reproduce the scores printed for ToMe on SDXL? ToMe merges each subject's
words into one composite token and updates that token during the early sampling steps.

## The run

**Group.** Reproduction (group 4). Each number freezes as a baseline (group 5) when it lands.

**Varied axis.** None. ToMe runs on the prompts and seeds of the RTD and plain SDXL renders.

**Buys.** Four rows of `table.csv` that can get a verdict, and the Color render folder for side-by-side comparison.

| Row | Benchmark | Metric | Published | Allowed prompts |
|---|---|---|---|---|
| `ToMe-T1-054` | T2I-CompBench Shape | BLIP-VQA | 0.6051 | 293 of 300 |
| `ToMe-T1-053` | T2I-CompBench Texture | BLIP-VQA | 0.6894 | 300 of 300 |
| `RTD-T2-021` | T2I-CompBench Complex | ImageReward | 0.2891 | 988 of 1,000 |
| `RTD-T2-024` | RareBench Concat | ImageReward | -0.2041 | 33 of 40 |
| none | T2I-CompBench Color | BLIP-VQA | not comparable, by the owner's answer to question 13 | 288 of 300 |

**Cost.** ToMe took a median of 82 seconds per picture over the 1,106 renders of the review response. One seed on all 1,902
allowed prompts is about 43 GPU-hours, which is past the 24 GPU-hour limit. Shape, Texture and Concat on one seed are 626
pictures, about 14 GPU-hours, and run first. Complex on one seed is 988 pictures, about 23
GPU-hours, and Color is 288 pictures, about 7 GPU-hours.

## What the release leaves open

The owner's answers to board questions 17 and 18 settle the run: the `tome` environment, `RunConfig2`, one written merge rule, seed 0
only, with Shape, Texture and Concat first and then Complex and Color, inside a budget of about 44 GPU-hours.

| Setting | Paper | Released code |
|---|---|---|
| Steps optimised | the first 20%, so 10 of 50 | 7 in `RunConfig1` (object binding), 5 in `RunConfig2` (attribute binding) |
| Prompt parsing | SpaCy | `en_core_web_trf` in `run_demo.py` |
| Merged prompt and its length | not stated | set by hand for one demo prompt per config |
| Benchmark script | not released | none |
| Guidance, steps | not stated | 7.5, 50 |
| Negative prompt | not stated | "low res, ugly, blurry, artifact, unreal" |
| Sampler | not stated | SDXL's default scheduler, left in place by the pipeline |
| Images per prompt and seeds | not stated | demo seeds only |

**The merge rule.** The merged prompt is each noun chunk's head noun with its article, joined by "and", as `merged_prompt` in the
hub's `scripts/reproductions/tome_common.py` writes it. `prompt_length` is the token count of the original prompt. Under `RunConfig2`
the merged prompt is encoded and never used, because the end-of-text replacement waits for step 60 and a render has 50 steps.

**The environment.** ToMe pins torch 2.0.1, diffusers 0.28.0 and transformers 4.30.2. The cluster's `tome` environment holds those
versions and rendered the review response. The owner approved it for this run on board question 17.

**Seeds.** Seed 0 only, by the owner's answer to board question 18, with each number labelled by its seed count.

## What ToMe's parser does with the benchmark prompts

`tome_parse_check.py` ran the authors' `PromptParser` and `filter_text` on all 1,902 allowed prompts, on CPU, with no render. Its output
is `/datasets/mmolefe/poe_repair_min/loop-5/data/tome-parse.jsonl`.

| Benchmark | Allowed prompts | Parsed | Crash in `filter_text` | Nothing to merge |
|---|---|---|---|---|
| Color | 288 | 288 | 0 | 0 |
| Shape | 293 | 291 | 2 | 0 |
| Texture | 300 | 300 | 0 | 0 |
| Complex | 988 | 843 | 145 | 0 |
| Concat | 33 | 29 | 0 | 4 |

**The crash.** `_split_prompt` drops phrases such as "the right", while the token groups keep their positions, so `filter_text` looks
up an anchor past the end of its list. **Nothing to merge.** On prompts such as "a cactus made of steel and two sunflowers made of glass",
spaCy finds no noun with an attribute. By the owner's answer to board question 20, the alignment is repaired by pairing each attribute
group with the phrase holding its noun, and nothing else in ToMe changes. The 4 prompts with nothing to merge render as plain SDXL. Each
row prints ToMe's score with and without the repaired prompts and labels both prompt counts. The GPU budget stays at 44 hours.

**The repair, checked on all 1,902 prompts without rendering.** None of the 1,751 prompts that already parsed changed. All 147 that
crashed now parse (145 Complex, 2 Shape). Their 310 attribute groups each land on a phrase containing their noun. Two groups, one each
in `t2i_compbench-0746` and `t2i_compbench-0980`, sit in no kept phrase and are left out of the merge. The repaired parse is
`/datasets/mmolefe/poe_repair_min/loop-5/data/tome-parse-repaired.jsonl`.

## The test, fixed before the first run

Both metrics are scored per picture. The standard error comes from resampling prompts with replacement 10,000 times. ToMe prints
no standard error, so the published one is taken as equal to ours. On the allowed prompts, a row counts as `reproduced` when
|ours − published| ≤ 1.96 × √(SE_ours² + SE_published²), and otherwise as `not reproduced`. Every number carries its prompt and
seed counts. The rule is in `scripts/reproductions/verdict.py` in the replications hub.

BLIP-VQA is scored with `BLIPvqa_eval/BLIP_vqa.py` from T2I-CompBench at commit `4aa4042`, and plain SDXL on Shape and Texture
(0.5408 and 0.5637 in ToMe's Table 1) checks that scorer before any ToMe number is read.

## The ladder

Each rung saves its config, metrics, image sheet and commit under `/datasets/mmolefe/poe_repair_min/loop-5/renders/ToMe/<rung>/`.

1. **Smoke test.** Two prompts, with the time per picture recorded.
2. **Overfit.** Does not apply, because nothing trains.
3. **1% of the data.** Three prompts each from Shape and Texture and 10 from Complex, with sheets opened at full size.
4. **Controls.** With token updates and attention refinement switched off, ToMe's loop must match the same loop's plain SDXL to
   under 1e-5 (largest absolute latent difference), in one process on one card.
5. **Full protocol.** The prompts and seeds the owner's answers allow.
