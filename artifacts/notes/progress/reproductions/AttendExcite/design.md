# Reproducing Attend-and-Excite on T2I-CompBench Color

## What this asks

Attend-and-Excite nudges the image during sampling until each chosen word has a strong patch in the model's cross-attention maps. The question is
whether the official code reaches the 0.6400 printed for it on T2I-CompBench Color. That score is BLIP-VQA: a question-answering model is asked whether
each coloured object is present, and the answers' probabilities are multiplied, so 1 is best.

## Status

The row is `not comparable`, and nothing has run. Board question 25 asks the owner whether to use a community copy of the base model.

## The row

| Row | Benchmark | Published | Where the number comes from |
|---|---|---|---|
| `CoMat-T1-013` | T2I-CompBench Color, BLIP-VQA | 0.6400 | CoMat Table 1 quotes it; T2I-CompBench++ Table VIII, p9, prints it as "Attn-Exct + SD v2" |

Its Shape and Texture rows are `not comparable` by the owner's answer to board question 13.

## What the benchmark paper settles

| Setting | Value | Source |
|---|---|---|
| Base model | Stable Diffusion 2 base | Table VIII, and `stabilityai/stable-diffusion-2-base` in `GORS_finetune/inference_eval.py` |
| Images | 10 per prompt, seeds 42 to 51 | the paper's section on evaluation metrics, and the same script |
| Plain SD2 sampler | DDIM, 30 steps, default guidance 7.5, fp16 | the same script |
| Scorer | BLIP-VQA, `BLIPvqa_eval/BLIP_vqa.py` at `4aa4042` | the benchmark's repository |

## What nothing settles

**The reimplementation is unreleased.** The paper says the authors "re-implement" Attend-and-Excite on SD2, and the benchmark's repository holds no copy.

**The base model has no official source now.** `stabilityai/stable-diffusion-2-base` no longer resolves on Hugging Face. Community copies remain, such as
`sd2-community/stable-diffusion-2-base`, and none is cached on the cluster.

**The boosted words are unstated.** The official `run.py` asks a person to type token indices. Of the 288 Color prompts the subject guard allows, 224 fit
"a {colour} {noun} and a {colour} {noun}", and 64 need a rule.

## The run, if the owner approves it

**Group.** Reproduction (group 4), then frozen as a baseline (group 5).

**Varied axis.** None. Plain SD2 runs first as the check on the mirror and scorer.

**Check first.** Plain SD2 must reproduce the printed 0.5065 on the same 288 prompts. If it fails, the rival does not run and this row stays
`not comparable`, because the failure would be in the base model copy or the scorer.

**The rule for boosted words, fixed before any run.** Boost the last word of each noun phrase found by spaCy, the parser in ToMe's `prompt_utils.py`.
A prompt with fewer than two noun phrases renders through plain SD2, and its count goes in the row's note.

**Sampler.** The official pipeline at `163efdf` with its defaults: 50 steps, guidance 7.5, 25 steps of attention updates, scale factor 20.

**Cost.** 2,880 images each. Plain SD2 is about 1.5 GPU-hours and Attend-and-Excite about 15, both unmeasured until a smoke test.

**Buys.** `CoMat-T1-013`, the Attend-and-Excite number on the Color target.

**The test.** The score is per image, so the standard error comes from resampling prompts with replacement 10,000 times. The row counts as `reproduced`
when |ours − 0.6400| ≤ 1.96 × √(SE_ours² + SE_ours²), using our standard error for the published one because the paper prints none. The row's note says
it is scored on 288 of 300 prompts and that it tests a rebuild of an unreleased implementation.
