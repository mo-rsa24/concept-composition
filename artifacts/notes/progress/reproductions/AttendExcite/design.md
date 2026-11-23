# Reproducing Attend-and-Excite on T2I-CompBench Color

## What this asks

Attend-and-Excite nudges the image during sampling until each chosen word has a strong patch in the model's cross-attention maps. The question is
whether the official code reaches the 0.6400 printed for it on T2I-CompBench Color. That score is BLIP-VQA: a question-answering model is asked whether
each coloured object is present, and the answers' probabilities are multiplied, so 1 is best.

## Status

The row is `not comparable` by the owner's answer to board question 25, because it was run on Stable Diffusion 2 and cannot change an SDXL
target. Nothing is downloaded and no GPU time is spent on it.

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
