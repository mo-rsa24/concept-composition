# Which fidelity fix wins

The adapter restores plurality and costs picture quality. These sheets put every fix that needs no
retraining on the same eight held-out cells, from the same starting noise and the same checkpoint,
so a difference along a row is the treatment and nothing else. The verdict is in
[can any render-time fix recover the picture quality the adapter costs](../../../report/is-the-gap-the-samplers-or-the-models/does-any-render-time-fix-recover-picture-quality.md);
each sheet is read one at a time in [the figure explainer](figure-explainer.md).

## What is here

| File | What it shows |
|---|---|
| `a_cat__x__a_dog__seed09.png` | the pair and seed where the adapter looks most like its target |
| `a_cat__x__a_dog__seed10.png` | the same pair on a seed whose joint prompt draws two dogs, so the target is wrong |
| `a_tiger__x__a_dog__seed09.png` | two animals of different size, every fix leaving the layout alone |
| `a_tiger__x__a_dog__seed10.png` | guidance 10 adding a third animal and guidance 12 replacing the dog with a tiger |
| `an_elephant__x__a_penguin__seed09.png` | the pair that fails most often, composing with the sizes inverted |
| `an_elephant__x__a_penguin__seed10.png` | the cleanest cell in the set, at the right relative sizes |
| `a_chess_board__x__an_hourglass__seed09.png` | the object pair, where re-weighting invents a carved wolf |
| `a_chess_board__x__an_hourglass__seed10.png` | a cell the plain product already composes unaided |
| `sharpness.json` | median Laplacian variance per column and per cell, with the ratio to the adapter |
| `sheets.json` | which columns each sheet was built from and how many tiles were found |

Eleven columns per sheet, left to right: the joint prompt, the plain product, the adapter as it
ships, guidance 10, guidance 12, the experts weighted 3.0 and 3.0, a repaint at strength 0.25 and at
0.40 with the adapter off, the adapter for the first 20 steps only, and the first of eight nearby
starting noises.

## How it was made

`scripts/fidelity_harness.py submit` on the cluster against the v58-05-self-contrast checkpoint at
step 15,000, which put up ten Slurm jobs, 59549 to 59558, on the `batch` partition on 2026-09-24.
Each job rendered one column over four pairs (cat and dog, elephant and penguin, tiger and dog,
chess board and hourglass) at seeds 9 and 10, all held out of training, and took 11 to 19 minutes.
The tiles landed in `/datasets/mmolefe/poe_repair_min/outputs/fidelity_harness/` and the sheets were
built from them with `fidelity_harness.py sheets`. The best-of-8 job was stopped at 23 minutes with
two of its eight cells done, when every run was stopped for a re-assessment, which is why that
column is blank on six sheets.

`sharpness.json` was written by `scripts/local/tile_sharpness.py` over the 87 full-resolution tiles
on the laptop, not on the cluster. It is Laplacian variance on the greyscale render, the same
measure [the noise-search finding](../../../report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-noise-sharpen-the-corrected-render.md) used.

## What it says

The one clear harm is applying the correction's implied expert weights explicitly in the sampler at
a constant 3.0 and 3.0. Both subjects dissolve into a smeared wash on all eight cells, it scores
below even the plain product, and on one cell it adds a carved wolf that neither prompt names.

Raising the guidance scale scores highest of anything here, 1.45 at scale 10 and 1.80 at 12 against
the adapter, and on three of the eight cells it changes what is in the picture: a toy giraffe, a
third animal, a rebuilt hourglass. Higher contrast raises this measure whether or not detail was
drawn, so the score and the pictures disagree, and the pictures decide.

The two repaint passes and the 20-step handover draw the adapter's own picture again. They score
1.14, 1.24 and 1.09 against a 1.10 bar borrowed from another finding after these images existed.
That is where the eye cannot separate them and the number is too close to the bar to settle it.

## What it cannot say

Whether any column still draws two things by count: the validated instance-count scorer was never
run over these tiles, so every composition statement here is an eye read. Whether the fidelity cost
behaves this way on another checkpoint: everything is one checkpoint at one step. Whether a falling
damping schedule, about 3 early and about 1 late, behaves like the constant tested here. And on the
two seed-10 animal cells it cannot say anything about nearness to the joint prompt, because that
target does not show the pair.

<!-- cluster:start -->

## On the cluster

The other 4 files of this result (4.5 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/which-fidelity-fix-wins/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 4 | 4.5 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/which-fidelity-fix-wins/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/which-fidelity-fix-wins/<run>/ <run>/
```

<!-- cluster:end -->
