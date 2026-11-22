# Does classifier guidance work on the released weights

Dhariwal and Nichol's classifier guidance ([arXiv 2105.05233](https://arxiv.org/pdf/2105.05233#page=8)), run on their own released 256 px unconditional ImageNet model and 256 px noisy classifier. It does nothing at scale 0. At scale 10 it draws the asked-for class: 46 of 48 renders across the three classes are called that class by the released classifier, against 9 of 48 at scale 1. All four targets passed the bars written in [route step 4](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md) before any run, at level 1 (the authors' code and weights, unretrained).

## Scale 0 changes nothing

`sheet-identity-scale-0-against-plain.png` is not shown because it failed the subject check

On 4 of 4 seeds the render at scale 0 has the same sha256 as plain sampling (no `cond_fn`), on both the uint8 image and the float32 sample. Plain sampling run twice on seed 0 also matches itself, so the GPU is deterministic here and the match is not luck. The checksums are in [identity_checksums.csv](identity/identity_checksums.csv). `verified`

## The corgi figure, and the two classes this project needs

Each sheet shows scale 1 on the left and scale 10 on the right, with seed i starting from the same noise in both. A tile's frame is green where the released classifier's top-1 call (at t = 0, on the saved render) is the asked-for class.

| Class | Scale 1 | Scale 10 | By eye at scale 10, sheet read | By eye, tile by tile, scale 1 and 10 | Sheet |
|---|---|---|---|---|---|
| Pembroke Welsh corgi (263), the paper's [Figure 3](https://arxiv.org/pdf/2105.05233#page=8) | 2 of 16, mean p 0.067 | 16 of 16, mean p 0.854 | 13 clear corgis; seeds 2, 6, 11 corgi-coloured with a fox or dingo face | 0 and 9 of 16 | `sheet-pembroke-welsh-corgi-scale-1-and-10.png` is not shown because it failed the subject check |
| tabby cat (281) | 2 of 16, mean p 0.066 | 15 of 16, mean p 0.737 | 12 clear tabbies; seed 8 is an ornate object called pickelhaube | 0 and 11 of 16 | `sheet-tabby-cat-scale-1-and-10.png` is not shown because it failed the subject check |
| golden retriever (207) | 5 of 16, mean p 0.195 | 15 of 16, mean p 0.907 | 14 clear goldens; seed 4 is called Labrador | 3 and 10 of 16 | `sheet-golden-retriever-scale-1-and-10.png` is not shown because it failed the subject check |

Every per-sample call is in [classifier_calls.csv](classifier_calls.csv), 96 rows, one per render, each with its tile path, the classifier's `hit`, and the tile-by-tile eye call in `your_call` (1 or 0) with `called_by`. The classifier's columns are `verified`. Both eye columns are readings by Claude and `inferred`: the sheet read counted corgi-coloured and Labrador-like dogs, and the tile-by-tile read, made for route instruction 13 at 256 px, says yes only to a clear tile of the asked-for breed, so a fox face, a beagle, a sheltie, a Labrador or a garbled face is a no.

**The classifier grades its own push.** The same released classifier guides the sample and calls it, so its 46 of 48 is partly self-fulfilling. The by-eye column and instruction 13's count are the independent check.

## Where the push enters their sampler

In `guided_diffusion/gaussian_diffusion.py` at upstream commit `22e0df8`:

- **Ancestral sampling (what ran here):** `condition_mean`, line 356, adds `variance × gradient` to the step's mean, where `gradient = scale × ∇ log p(y | x_t)` from `cond_fn`. This is the paper's Algorithm 1 ([page 7](https://arxiv.org/pdf/2105.05233#page=7)).
- **DDIM:** `condition_score`, line 371, subtracts `√(1 − ᾱ_t) × gradient` from the predicted noise, then rebuilds the predicted clean image and the mean from it. This is the paper's Algorithm 2 ([page 7](https://arxiv.org/pdf/2105.05233#page=7)), and it is the form the product sampler ports.

**One step of it, seed 0, corgi.** The size of the push (the step's mean after the push minus before, as a vector norm) is largest early and goes to 0 at the end, because the variance it is multiplied by goes to 0. At scale 10 it is 0.18 at the first step and 0.32 at step 50 of 250, against a mean of norm about 429, so under 0.1% of the step. At scale 10 the classifier's raw gradient falls from 0.52 at step 50 to 0.11 at step 200, as the sample becomes a corgi. At scale 1 it rises from 0.54 to 1.85, because the sample never does. Per-step numbers: [push_per_step_c263_s10_seed0.json](guide/push_per_step_c263_s10_seed0.json) and push_per_step_c263_s1_seed0.json (on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/guide/push_per_step_c263_s1_seed0.json`). `verified`

**Each line of it, beside the debugger.** [What each line of classifier guidance computes](../what-each-line-of-classifier-guidance-computes/README.md) walks these two files section by section, with each equation taken apart on an exact toy and these per-step numbers on the steps they come from.

<details>
<summary>How it was made</summary>

**The run.** [guided_diffusion_reproduce.py](../../../scripts/guided_diffusion_reproduce.py) imports the upstream `script_util` and runs `classifier_sample.py`'s own `cond_fn` and `p_sample_loop`, with the README's unconditional 256 px `MODEL_FLAGS` and its `--timestep_respacing 250`. It loads weights with `torch.load` instead of `dist_util`, so co3 needed neither mpi4py nor blobfile. Each seed is its own batch of 1, with `torch.manual_seed(seed)` set right before sampling, and cuDNN is in deterministic mode. The per-step log wraps `diffusion.condition_mean` on the instance; the upstream file is untouched.

**Where.** T1 and T2 ran as Slurm jobs 89232 and 89235 on mscluster108, Quadro RTX 8000. T3 and T4 ran with nohup on mscluster107's card 1, because mscluster108 was taken by another user. A render took 22 s plain and 37 s guided. The sheets come from [guided_diffusion_sheets.py](../../../scripts/guided_diffusion_sheets.py).

**Paths.** Weights: `/datasets/mmolefe/poe_repair_min/weights/guided-diffusion/` (`256x256_diffusion_uncond.pt` 2,211,383,297 bytes, `256x256_classifier.pt` 216,496,432, `64x64_classifier.pt` 261,889,658, and `64x64_diffusion.pt` for step 8). Arrays: `/datasets/mmolefe/poe_repair_min/outputs/guided_diffusion_reproduce/`. Clones: `/home/molef/PhD/replications/guided-diffusion` and `/home-mscluster/mmolefe/Playground/PhD/replications/guided-diffusion`.

**Not done.** The paper's FID numbers (33.0 and 12.0 for Figure 3) need 50,000 samples each and were not attempted. The 16 samples per cell carry no confidence interval. Only ancestral sampling ran, not DDIM.

</details>

## Where the eye and the classifier disagree

`eye-against-classifier-disagreements.png` is not shown because it failed the subject check

**What it shows.** The 22 tiles where the tile-by-tile eye call and the classifier's call differ, out of 96. Every one is the classifier saying yes and the eye saying no; at scale 10 the classifier says 46 of 48 and the eye 30 of 48.

**How it was made.** Built from `classifier_calls.csv` on 2026-11-17 by pasting each disagreeing tile at its native 256 px under its class, scale and seed. `inferred`, since the eye side is one reading.

<!-- cluster:start -->

## On the cluster

The other 115 files of this result (22.2 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 5 | 9.9 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/` |
| `guide` | 102 | 11.6 MB | [c281_s1_seed12.png](guide/c281_s1_seed12.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/guide/` |
| `identity` | 8 | 0.8 MB | [identity_seed2_plain.png](identity/identity_seed2_plain.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/identity/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/<run>/ <run>/
```

<!-- cluster:end -->
