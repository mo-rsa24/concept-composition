# Design: can a sentence score read binding? (step W2d)

W2's binding test used recoloured cat and dog pictures, and its labels failed the eye check (18 of 32 right), so binding was never tested. The owner chose on 2026-11-24 to test binding in two forms:
- **Part binding:** each subject keeps only its own body parts. This is the failure CO3 still makes.
- **Colour binding:** a red cat and a blue dog against the swap, with the colour labels rebuilt.

This file fixes the label rule, the sentences and the bars before any label or score exists.

## Part binding on CO3's pictures

**Pictures.** CO3's own row of Loop 6's comparison: 12 tuning pairs at seeds 6001 to 6008, 96 pictures, at `/datasets/mmolefe/poe_repair_min/loop-6/m1/paper-2/full/renders/co3/<pair>/seed-<seed>.png`. Loop 6 rendered them, so this loop's no-scoring rule on its own pictures does not apply. Giraffe × tortoise is labelled and scored but stays off every sheet. If either label has fewer than 30 pictures, the push and random rows (192 more) are labelled under the same rule before any score.

**The label rule, by eye at full size.** Each picture gets one call:

| Call | Meaning | Label |
|---|---|---|
| `own_parts` | Two separate subjects, A and B, and every visible part belongs to the subject it is on | 1 |
| `borrowed` | Two separate subjects, and at least one carries a clear part of the other: a face, head, beak, wings, legs, paws, tail, markings, fur or feathers in the other's pattern, or a sail on the lighthouse | 0 |
| `not_two` | Not two separate subjects (one subject, a fused hybrid, the same subject twice, a wrong subject, or extras) | left out |
| `unsure` | Too small, hidden or stylised to call | left out |

Drawn style does not change the call. A part touching or overlapping the other subject is not a borrowed part.

**How the labels are made.** Two labellers work independently and blind to each other. Each sees the 96 pictures under neutral names, in a shuffled order, with only the pair's two subject names. Where they disagree, the call is made by a third look at full size, and the agreement rate is reported beside the result. The calls are saved to `02d-binding/part-binding-calls.csv`, with each labeller's call and the final call.

**The latents.** The pictures were saved without latents, so each is encoded back through SDXL's VAE at 1024 pixels, scaled by 0.13025. That round trip is a difference from W2, where the latents were the sampler's own. The review reports it.

**Sentences**, fixed per pair with subjects A and B as in the joint prompt:

| Row | Right | Wrong |
|---|---|---|
| Pair-specific | "a photo of a {A} and a {B}, each with its own body" | "a photo of a {A} with parts of a {B}" and "a photo of a {B} with parts of a {A}" |
| Pair-free | "a photo of two separate subjects, each complete" | "a photo of two subjects with mixed-up body parts" |

All the sentences name the same subjects, so the score difference rests on binding and not on which subjects are present. Each sentence is scored plain and with W2c's five wordings.

**Protocol.** W2's: four noise draws per step at steps 0 to 15, 20 to 29, 30, 40 and 49, plus the clean latent; the same split of seeds into halves, within each pair; the same threshold rule, balanced accuracy and bootstrap; and NoisyCLIP, frozen CLIP and Latent CLIP.

## Colour binding, with whole-animal regions

W2's recolouring used the word maps, which landed on the cat's head. The rebuilt rule is written in its own section here once W2c lands, before any recoloured picture is made. It uses the detector's box for each animal (GroundingDINO, as in `2-classifier-guidance/scripts/rung2/label_with_detector.py`) to bound the region, and it gets a new eye check on 32 pictures against the same 90% bar.

## Bars

| Result | What it takes |
|---|---|
| **Finished-picture check** | At least 0.75 on the reading half for the clean latent, before any early step counts |
| **Reads early** | A mean of at least 0.75 over steps 0 to 15, as in W2 |
| **Enough pictures** | At least 8 of each label in each half, as in W2. Fewer, and the result is inconclusive |
| **Labels hold** | The two labellers agree on at least 80% of the pictures they both call `own_parts` or `borrowed`. Less, and the rule is too loose to score against |

## Run

- **Group:** `loop9-w2d`. **Varied axis:** the reader, with pictures, labels, draws and halves fixed.
- **Cost:** labelling by eye with no API tokens; then one `bigbatch` job of about 0.3 GPU-hours (estimate) for the VAE encoding and the three readers.
- **Buys:** whether any sentence score sees binding at all, first on the finished picture and then early, on the failure CO3 actually makes.
- **Code:** `2-classifier-guidance/scripts/noise_text_critic/w2d_binding.py`. **Output:** `/datasets/mmolefe/poe_repair_min/loop-9/w2d/`. **Review:** `02d-binding/review.md`.
