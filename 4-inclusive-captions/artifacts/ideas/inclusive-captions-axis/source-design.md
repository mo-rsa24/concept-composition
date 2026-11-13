# Source design: inclusive captions for the cat and dog experts

The design as proposed in a ChatGPT conversation (share link
https://chatgpt.com/share/6ae9602c-0378-83ea-a868-6df6c06a6c91), copied here so the idea walk
and the route can cite it. It is the input to `/drip-idea`, not a settled design: every claim in
it is checked there, and the route at `plans/route-inclusive-captions-pilot.md` lists the
problems already found.

## The question

Does training the single-animal conditions on images where both animals appear make the cat and
dog scores more compatible under product-of-experts composition? In densities: is
p(x | cat present) learned, instead of p(x | cat only)?

## Three LoRA training schemes

Pretrained checkpoint, images, LoRA rank, optimiser and step budget are held fixed. Only the
caption each image is trained under changes.

| LoRA | Captions | Purpose |
|---|---|---|
| A | one caption per image: cat-only "a photo of a cat", dog-only "a photo of a dog", both "a photo of a cat and a dog" | baseline |
| B | inclusive with dropout: a both-animal image is trained under cat 25%, dog 25%, joint 40%, unconditional 10%, sampled fresh each time it is drawn | the intervention |
| C | inclusive "without dropping individual conditions" (not defined further in the source) | ablation |

The source's sampler also gives single-animal images a 10% "a photo of animals" caption, and it
recommends the empty caption instead for an exact unconditional branch.

## Data

MS COCO 2017 instance annotations, selected by category id. The source proposes 300 cat-only,
300 dog-only and 200 both, plus 100 to 200 images with neither as a background control, and says
to audit the both-animal images by hand because annotations can miss animals. Bounding-box
overlap of the two animals is recorded so spatial composition can be read against it.

## Evaluation

Same seeds and sampler settings for every model, 200 to 500 images per condition proposed.

| Strategy | Conditioning |
|---|---|
| individual | "a photo of a cat" / "a photo of a dog" |
| joint prompt | "a photo of a cat and a dog" |
| PoE raw | s_C + s_D |
| PoE corrected | s_C + s_D − s_0 |

Measures: both animals present without a fused animal, one of each, distinct bodies and
locations, caption adherence, diversity. An object detector first, then manual review.

The unmodified SDXL checkpoint is evaluated too, before any fine-tuning.

## Two phases

**Phase A:** LoRAs A, B and C on the same data.

**Phase B:** a LoRA trained on cat-only and dog-only images only, so the both-animal images are
removed. SDXL was pretrained on web data holding both animals, so this measures what
fine-tuning shifts, not zero-shot composition.

## Proposed configuration

SDXL 1.0, U-Net LoRA with text encoders frozen, rank 8, 512 × 512, about 800 to 1,000 training
images, checkpoints at 250, 500, 1,000 and 2,000 steps.

## The follow-up question the source names

If inclusive captions help PoE, did they widen the overlap of the learned cat and dog
conditionals, correct what the captions mean, or only change the score geometry in a way the
approximate sampler likes? The source says these three are distinguishable by experiment.
