# Does SDXL still fuse when it is sampled the way PixArt-alpha is?

Summed-guidance composition on **SDXL**, run twice on the same pair, the same cached starting
latents and the same held-out seeds 9 to 16, with only the sampling configuration changed.

| | Sampler | Steps | Guidance |
|---|---|---|---|
| The control | DDIM | 50 | 7.5 |
| The matched run | DPM-Solver++ order 2 | 20 | 4.5 |

The matched row is PixArt-alpha's own released defaults, borrowed so that the fusion seen on that
model can be compared against SDXL on one axis instead of five.

## What came back

| Column | Two clear animals | One hybrid animal | Other |
|---|---|---|---|
| The product, control (DDIM 50 at 7.5) | **0 of 8** | 8 of 8 | |
| The product, matched (DPM-Solver++ 20 at 4.5) | **0 of 8** | 7 of 8 | 1 of 8 holds two animals whose species are unclear (seed 16) |
| The joint prompt, control | 8 of 8 hold two separate animals | | 5 of 8 hold a clear cat beside a clear dog |
| The joint prompt, matched | 8 of 8 hold two separate animals | | 5 of 8 hold a clear cat beside a clear dog |

Sheets: [the control](control/sdxl-control-sheet.png) and [the matched run](sdxl-matched-sheet.png),
rows the eight seeds, columns cat alone, dog alone, the product, the joint prompt. Per-tile records
with one sentence each are in [the calls](calls.json).

## What this settles

**The sampler, the step count and the guidance scale are not what carries the fusion.** Changing all
three to PixArt's values leaves the product column exactly where it was.

Together with [does a clip-free model hybridise](../../../report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md),
which removed CLIP, the five differences between the two models are down to two: the denoiser
architecture and the text encoder, with the encoder already shown not to be necessary.

## What it does not settle

The three settings moved together, so this names the group and not a member of it. Nobody has asked
whether 20 steps alone, or guidance 4.5 alone, does anything.

Seed 16's matched tile holds two bodies and is called as neither category, because the smaller
animal carries cat ears on a dog's snout. It is recorded rather than rounded into the hybrid count,
and the call can be disagreed with from the tile.

The joint-prompt column on these seeds draws two animals of the same species on 3 of 8 (control)
and 3 of 8 (matched), so it is read as "two separate animals", which is the instrument every
published SDXL count in this project used. That shows the configuration renders two subjects; it
does not show the model separates these two species reliably on these seeds.

## How these were made

`scripts/sdxl_matched_multi_poe.py`, which takes the sampler, the step count and the guidance scale
as arguments and leaves `scripts/concept_multi_poe.py` and `scripts/and_vector_sheet.py`, the two
paths every published SDXL number came from, untouched. The control run exists to prove that: at
DDIM, 50 steps and 7.5 the new script reproduces the published 0 of 8 and 8 of 8.

Run 2026-11-04 on mscluster84, an RTX 3090. Tiles and sidecars at
`/datasets/mmolefe/poe_repair_min/outputs/sdxl_matched/`.

<!-- cluster:start -->

## On the cluster

The other 2 files of this result (0.0 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-sdxl-compose-at-pixarts-settings/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 1 | 0.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-sdxl-compose-at-pixarts-settings/` |
| `control` | 1 | 0.0 MB | [sdxl-control-sheet.png](control/sdxl-control-sheet.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-sdxl-compose-at-pixarts-settings/control/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-sdxl-compose-at-pixarts-settings/<run>/ <run>/
```

<!-- cluster:end -->
