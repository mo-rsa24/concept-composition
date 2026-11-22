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
pictures, about 14 GPU-hours, and are the first run if the owner approves ToMe. Complex on one seed is 988 pictures, about 23
GPU-hours, and Color is 288 pictures, about 7 GPU-hours.

## What the release leaves open

Nothing runs until the owner answers board question 17, so these choices are made before any ToMe score exists.

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

**Proposed if the owner approves.** `RunConfig2`, the config the authors label for attribute binding, unchanged. The merged prompt
is built by one rule: the nouns SpaCy finds, each with "a" or "an", joined by "and". `prompt_length` is the token count of the
original prompt, as in the demo.

**The environment.** ToMe pins torch 2.0.1, diffusers 0.28.0 and transformers 4.30.2. The cluster's `tome` environment holds exactly
those versions and rendered the review response, but `environment/` does not describe it and the brief names `co3` and `co3_bw`,
so using it needs the owner's yes.

**Seeds.** The Complex and Concat rows state three seeds per prompt. Board question 18 asks whether ToMe runs three seeds or one.
The default is seed 0, with each number labelled by its seed count.

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
