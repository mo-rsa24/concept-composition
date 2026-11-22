# Does a model with no CLIP in it still fuse a cat and a dog?

Summed-guidance composition on **PixArt-alpha** (`PixArt-XL-2-1024-MS`), whose only text encoder is
**Flan-T5-XXL** (`T5EncoderModel`, 4.8B parameters). No CLIP in any role, which is the reason this
model and not Stable Diffusion 3, whose three encoders include CLIP-L and CLIP-G.

**Settings: 20 steps, guidance 4.5, DPM-Solver++ order 2, 1024², fp16.** These are PixArt-alpha's
own pipeline defaults, read off the released code, not SDXL's 50 steps at 7.5.

## What came back

| Column | Two clear animals | One hybrid animal | One single animal |
|---|---|---|---|
| The product, "a cat" and "a dog" summed | **0 of 8** | 6 of 8 | 2 of 8 (both cats) |
| The joint prompt "a cat and a dog" | **7 of 8** | 0 of 8 | 0 of 8 |

The eighth joint tile, seed 2, holds two cats rather than a cat and a dog, so it counts as neither.

Every product call was made by eye from the full-resolution 1024² tile, one at a time. The repo's
instance-count scorer counts animals without naming them, so it reads a hybrid and two cats
identically and cannot make these calls. Per-tile records with one sentence each are in
[the calls](calls.json); the sheet is [pixart-cat-dog-sheet.png](pixart-cat-dog-sheet.png), rows the
eight seeds, columns cat alone, dog alone, the product, the joint prompt.

## The SDXL side, quoted not re-run

From [does a corrector alone produce two animals](../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md):
joint prompt 8 of 8, plain product 0 of 8, held-out seeds 9 to 16, counted by the validated instance
count rather than by eye.

## What this settles

**CLIP is not necessary for the fusion.** The same arithmetic on a model that has never seen a CLIP
embedding produces the same single fused animal on every seed, so an account of the failure that
rests on CLIP's pooled caption representation cannot be the whole story.

**The weak-guidance escape is closed.** PixArt runs at 4.5 where SDXL runs at 7.5, so a fusion here
could have been a scale artefact. The joint column at the same 4.5 draws a clean pair on 7 of 8
seeds, which shows the scale is high enough to render two animals when the model is asked for them
in one sentence.

## What it does not settle

Swapping SDXL for PixArt changes five things at once: the text encoder, the denoiser (a UNet for a
diffusion transformer), the sampler, the step count and the guidance scale. Only the fusion
direction licenses a claim, because no other difference can manufacture a fusion that CLIP was
required for. It says nothing about which of the five carries the failure, and nothing about whether
CLIP is *sufficient* for it.

It also does not say the product is useless: the two single-animal tiles (seeds 2 and 3) are a
different failure from the six hybrids, and the plan keeps them apart because dropping a prompt and
averaging two prompts are different mechanisms.

## How these were made

The composition script is `scripts/pixart_multi_poe.py`, which predicts with each prompt separately
and with the empty prompt, then combines as the empty prediction plus the guidance scale times the
sum of each prompt's difference from it. Before any product render it was checked against the
released pipeline on one prompt: **largest absolute difference 0 of 255 grey levels, 0.0 of pixels
differing, the two PNGs byte-identical** (md5 `d9c08c28048b8da98124be0246099b07`), so the hybrids
above are the arithmetic rather than the plumbing.

Tiles and sidecars live on the cluster at
`/datasets/mmolefe/poe_repair_min/outputs/clip_free_hybridise/`, each sidecar naming the model,
encoder, scheduler, seed and settings. Rendered 2026-11-04 on mscluster84, an RTX 3090, from the
cluster worktree `poe_repair_min-clip-free` at commit `2506ea21`.

<!-- cluster:start -->

## On the cluster

The other 1 files of this result (0.0 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 1 | 0.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/<run>/ <run>/
```

<!-- cluster:end -->
