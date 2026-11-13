# Do Inclusive Captions Help PoE Compose

## Where this scope sits in the order

The root [running order](../../MASTER_PLAN.md) owns the one order across every scope. This scope
is new and holds one plan, which is not yet in that order.

It stands apart from the scopes either side of it. Scopes 01 to 09 all repair a trained model's
composition at inference time or with an adapter learned from cached noise. This one changes the
training captions and nothing else, so it is the only scope here that touches the data rather
than the model or the sampler.

## Mission

Find out whether a single-animal condition composes better when the training photos holding both
animals were captioned under each animal's name as well as jointly.

Product-of-experts composition on SDXL fails because the cat condition has learned cats without
dogs: fine-tuning on joint-captioned photos pushes the dog out of it. If each both-animal photo
is sometimes trained under "a photo of a cat" and sometimes under "a photo of a dog", both
conditions keep the other animal inside them, and the product of two such conditions should hold
more pictures containing both animals.

## Run kind

Experiment. Two LoRAs trained on identical photographs, differing in one cell of a caption
table, scored on the same 128 seeds against a criterion written before either ran.

## Depends on

**The photo pool, which exists.** 200 cat-only, 200 dog-only and 91 both-animal COCO photos, the 91 chosen by looking at all 131 that clear the size filter,
measured rather than assumed, carded at
[the photo pool](../../artifacts/results/the-inclusive-captions-photo-pool/README.md).

**The caption sampler, which exists and is checked.**
[`captions.py`](../../poe_repair/experiments/inclusive_captions/captions.py) decides which caption
each photo is trained under, and its `--check` mode measures the realised shares off 200,000
draws against three bars held as constants in its source.

**The instrument, which exists.** GroundingDINO counts the animals in a render and
[the crop classifier](../../scripts/census/crop_classifier_cat_vs_dog.py) names them, carded at
[does the cat prompt already draw dogs](../../artifacts/results/does-the-cat-prompt-already-draw-dogs/README.md).

**An image-and-caption denoising trainer, which does not exist.** Every trainer in this repo
learns from cached noise tensors toward a correction target, with no photographs, captions or
text encoder in the loop. Building one is the bulk of the work and the first plan's piece 3.

## What this scope cannot answer

Whether the product's failure is about co-occurrence at all. That is the share sweep's question,
and this scope only anchors one point of it. Whether the crop classifier holds up when two
animals share a frame with overlapping boxes, which is untested and which the three-category
score depends on. And whether the result transfers past cat and dog, since one pair is trained.

## Objectives

**Fix what counts as working before anything runs.** The [compose rate](../../context/world/compose-rate.md) run B has to reach against
run A, the band inside which the answer is "could not tell", and the result that kills the idea.

**Build the trainer and watch it train.** An image-and-caption LoRA trainer against this repo's
own SDXL loader, reporting to Weights and Biases with a four-column sample sheet at fixed
checkpoints: "a photo of a dog", "a photo of a cat", the joint prompt, and the product of the cat
and dog conditions on the same eight seeds throughout.

**Train the two runs and read them.** 2,000 steps each on the same 491 photos, then 128 seeds of
product renders each, scored in three categories and read against the criterion.

## Goals

| Goal | What it produces | How it is judged |
|---|---|---|
| The criterion, pre-registered | the review file's question section, filled before any training | every one of its six parts carries a number |
| The data build finished | the manifest carrying COCO captions, and the both-animal photos audited by eye | every photo in the manifest opens; the audit names how many were rejected and why |
| The trainer | an SDXL LoRA trained on photographs and captions, 20-step smoke run green | the smoke run writes a checkpoint and a sheet, and the loss is finite |
| The watching sheet | one four-column image per checkpoint in W&B, eight fixed seeds | the product column is present at step 0, before any training |
| The two runs | run A and run B, 2,000 steps, one W&B run each | both reach step 2,000 and save a final checkpoint |
| The answer | the compose rate for A and B on 128 shared seeds, as a figure with its sidecar | the figure's number is read against the criterion's three lines, and one of them is ticked |

## Expected Outcome

Run B's product compose rate is above run A's by more than the criterion's band, and the
held-out denoising loss on unseen both-animal photos is lower under B. The honest expectation is
weaker than that: the effect the design predicts and the smallest move 128 seeds can resolve are
the same size, so the inconclusive band is the most likely single outcome.

## Definition of Done

- [ ] The review file's question section is filled, with a number on each of its six parts, and it
      was committed before the first training step of run A or run B.
- [ ] The manifest carries a caption for every photo, and every path in it opens.
- [x] ✅ The 131 both-animal photos have been looked at, and the count rejected is written down with
      the reason.
- [ ] The trainer runs 2,000 steps on a 3090 without running out of memory, and its 20-step smoke
      run is green.
- [ ] W&B holds one run per arm with a four-column sheet at steps 0, 250, 500, 1,000 and 2,000.
- [ ] 128 product renders exist per arm at 1,024 square, scored in three categories.
- [ ] The compose-rate figure exists with its sidecar, and exactly one of the criterion's three
      lines is ticked against it.
- [ ] The finding is written to `report/`, and the figure is filed with its card entry.

## Sub-Scopes

None.

## Plans

| Plan | What it does | Status |
|---|---|---|
| [the pilot, end to end](plans/01-the-inclusive-captions-pilot.md) | all six pieces, from the criterion to the answer | ◯ not started |

## Review files

| Review | The question it pre-registers |
|---|---|
| [does captioning widen the cat condition](review/01-does-captioning-widen-the-cat-condition.md) | written in the plan's piece 1, before any training |

## Environment Context

This project's environment facts live in [environment/](../../environment/00-INDEX.md). Read
[the overview](../../environment/overview.md) before anything here that touches compute. The
facts this scope leans on are the `co3` absolute python path, the rule that large artifacts go to
`/datasets` only, the one-job-per-user limit on `biggpu` that puts long runs outside Slurm, and
the fp16 upcast rule.

## Glossary

**The cat condition.** What the model has learned to draw when the prompt is "a photo of a cat".

**The product.** `eps_uncond + g * (eps_a + eps_b - 2 * eps_uncond)`, the product of the cat and
dog conditions as this repo computes it at
[`training_cache.py:11`](../../poe_repair/training_cache.py).

**Run A and run B.** Two LoRAs trained on the same 491 photos. A captions every both-animal photo
jointly. B captions it "a photo of a cat" 40% of the time, "a photo of a dog" 40%, jointly 10%,
resampled on every draw. Both leave the caption blank 10% of the time, on every photo.

**The blank-caption rate.** How often a photo is trained with no caption at all. It is what
classifier-free guidance subtracts at sampling time and the product subtracts twice, so two runs
with different blank rates are not comparable. Every run here shares one 10% rate.

**Compose rate.** The fraction of product renders holding exactly one cat and one dog with no
hybrid, judged per class off full-resolution tiles.
