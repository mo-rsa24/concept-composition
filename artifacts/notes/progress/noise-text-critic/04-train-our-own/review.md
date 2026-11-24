# Review: does our own noise-aware text and image model read unseen digit pairs? (step W4)

**State: null.** The encoder learns the digit pairs it was trained on: 0.758 balanced accuracy on held-back seen canvases in the SDXL-matched window, steps 14 to 25. It does not carry that to digits 5 to 9, whose names never reached its loss. It reads 0.590 on unseen canvases, 0.168 below seen, and 0.502 on digits apart against wrong digits, which is chance. All three of the design's checks fail. The [design](design.md) holds the bars, and W&B run `en52x031` holds the numbers.

## Runs

| Job | Stage | Code | Outcome |
|---|---|---|---|
| 94162 | ladder | `8cbdaf49` | Cancelled before it started: its snapshot left out `2-classifier-guidance/scripts/mnist/`, which holds `canvas_unet.py` |
| 94163 | ladder | `8cbdaf49` | Cancelled before it started, during the `bigbatch` outage, when the ladder was moved to the laptop |
| none | ladder, on the laptop | `8cbdaf49` | PyTorch through `uv` on WSL printed nothing and wrote no file, so the ladder went back to the cluster |
| 94208 | ladder: smoke, overfit, 1% | `8cbdaf49` | 0.128 seconds per update. The overfit test placed 16 of 16 right (15 needed). The 1% run read seen canvases at 0.660 in the matched window |
| 94211 | final: 8,000 updates at batch 256 | `8cbdaf49` | 158 seconds of training on data hash `f70b0fb3…`. Verdict null |

## Result

Balanced accuracy on reading the true caption against the wrong ones. Thresholds are chosen on half of the seen test and applied once to the unseen test. 0.5 is chance.

| Split | Steps 14 to 25, judged | Digits apart against wrong digits | Steps 0 to 15, read second |
|---|---|---|---|
| Seen pairs, digits 0 to 4 (1,000 canvases) | **0.758** | 0.637 | 0.536 |
| Unseen pairs, digits 5 to 9 (2,000 canvases) | **0.590** | 0.502 | 0.504 |
| Bar | at least 0.75 unseen, within 0.05 of seen | at least 0.75 | reported only |

| Step | 14 | 16 | 18 | 20 | 22 | 24 | 25 |
|---|---|---|---|---|---|---|---|
| Seen | 0.64 | 0.67 | 0.72 | 0.77 | 0.83 | 0.88 | 0.89 |
| Unseen | 0.52 | 0.55 | 0.59 | 0.58 | 0.67 | 0.64 | 0.63 |

**What it means.** On seen digits, the model reads more and more of the picture as noise falls, reaching 0.89 by step 25. On unseen digits it gains only a little, and it never tells "a 7 beside a 9" from the wrong digits. Its image encoder never saw digits 5 to 9, and nothing in training ties their shapes to CLIP's words for them. So the frozen text encoder alone gives the image side no route to unseen concepts. This matches W2, where pretrained models that saw everything still could not read the early steps. Here the early steps are learnable, but only for concepts the model trained on (inferred from the two splits).

**Steps 0 to 15.** Both splits stay between 0.49 and 0.60 at every one of these steps, as Loop 10 found for every input it tried there (`score-space/05-signal/review.md`).

## The cost of an SDXL version

These are estimates, from no run:
- **Data.** Training NoisyCLIP's way means SDXL latents saved along each sampling path. 20,000 pictures at about 10 seconds each on an RTX 3090 is about 55 GPU-hours to render (inferred from SDXL's 50-step render time on that card).
- **Training.** Fine-tuning a ViT-L/14 image encoder on 20,000 × 10 noisy latents is about 10 to 20 GPU-hours (inferred).
- **Together,** about 70 GPU-hours, against this loop's 8.7 GPU-hour budget.

On the digits, what the model learns does not reach concepts outside its training captions. So an SDXL version would read the pairs it trained on and is not expected to generalise to new pairs. That is the property both papers need. This loop does not recommend it.

## Handed to Loop 10

The trained encoder is at `/datasets/mmolefe/poe_repair_min/loop-9/w4/full/encoder.pt`, with data hash `f70b0fb3…`, for Loop 10's step X6 baseline.
