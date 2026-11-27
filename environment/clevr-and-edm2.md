# 🧊 Environment: CLEVR rendering and EDM2 training on the cluster

Navigation: 📋 [Index](00-INDEX.md) | [Overview](overview.md#compute--runtime)

What is true about the CLEVR renderer and the EDM2 trainer this project's CLEVR experiments run on. The thing a plan gets wrong
most often here is assuming Blender can use the GPU, or reusing a render seed base an earlier job already used.

## Table of contents

- [Blender and the CLEVR generator](#blender-and-the-clevr-generator)
- [Render seed bases already used](#render-seed-bases-already-used)
- [EDM2](#edm2)
- [The colour experts' model files](#the-colour-experts-model-files)
- [Still open](#still-open)

## Blender and the CLEVR generator

Navigation: 📋 [TOC](#table-of-contents) | [Next](#render-seed-bases-already-used) ➡️

**Blender 2.78c renders on the CPU only.** The GPU test (job 94448, 8 pictures on one RTX 3090) stopped because Blender 2.78c asks
for NVIDIA's compiler to build its GPU code and the cluster has none (the colour-expert run's
[review](../artifacts/notes/progress/method-search/05-two-experts-by-colour/review.md), job table). A render job therefore takes a
28-thread `bigbatch` CPU node and no card.

| Thing | Path | Source |
|---|---|---|
| Blender 2.78c | `/datasets/mmolefe/poe_repair_min/loop-5/third_party/blender-2.78c/blender-2.78c-linux-glibc219-x86_64/blender` | line 15 of `2-classifier-guidance/scripts/method_search/m7_render.sbatch` |
| The CLEVR generator (`render_images.py`) | `/datasets/mmolefe/poe_repair_min/loop-5/third_party/clevr-dataset-gen/image_generation`, exported as `CLEVR_GEN` | line 16 of the same job script |
| The project's render driver | `2-classifier-guidance/scripts/method_search/clevr_render.py`, run inside Blender | line 18 of the same job script |
| The 64-pixel packer | `2-classifier-guidance/scripts/method_search/clevr_data.py`; packing two 15,000-picture sets took 12 minutes 35 seconds (job 95420) | the colour-expert review, job table |

**Measured render rates on one 28-thread CPU node:** 5.46 seconds per single-object image and 4.50 per empty scene (the
[CLEVR scaling plan](../artifacts/notes/progress/targets/clevr-scaling-plan.md#what-it-costs)); 4.71 to 5.77 seconds per picture
for scenes of up to 6 objects (jobs 94548 to 94552).

## Render seed bases already used

Navigation: ⬅️ [Blender](#blender-and-the-clevr-generator) | 📋 [TOC](#table-of-contents) | [Next](#edm2) ➡️

`clevr_render.py` seeds scene number `i` with `random.seed(seed_base + i)`, so two jobs with overlapping ranges render the same
scenes. The bases `m7_render.sbatch` assigns:

| Set | Seed base | Range used |
|---|---|---|
| `p1`, `p2`, `p3` (fixed positions) | 1,000,000; 2,000,000; 3,000,000 | from each base upward |
| `empty` | 4,000,000 | from the base upward |
| `s0` to `s7` | 5,000,000 + M × 100,000 | 5,000,000 to about 5,800,000 |
| `r1` to `r3`, `b1` to `b3` (the colour experts' renders, 5,000 each) | 6,000,000 + (length of the colour word) × 100,000 + K × 10,000 | 6,310,000 to about 6,435,000 |
| `c00` to `c33` (the colour-expert scorer's renders, 250 each, jobs 94765 to 94780) | 7,000,000 + X × 100,000 + Y × 10,000 | 7,000,000 to 7,330,249 |

| Paper 1's joint scenes (`paper1-first-render-joint`, 512 scenes) | 8,000,000 | 8,000,000 to 8,000,511 |
| Paper 1's second joint set (`paper1-render-joint2-p0` to `p5`, 500 each, jobs 97042 to 97047) | 8,100,000 | 8,100,000 to 8,102,999 |

A new render set starts at 8,200,000 or above.

## EDM2

Navigation: ⬅️ [Seed bases](#render-seed-bases-already-used) | 📋 [TOC](#table-of-contents) | [Next](#the-colour-experts-model-files) ➡️

| Fact | Value | Source |
|---|---|---|
| Checkout | `/home-mscluster/mmolefe/Playground/PhD/projective-composition/reference/edm2` (`train_edm2.py`) | line 15 of `2-classifier-guidance/scripts/method_search/m7_train.sbatch` |
| Memory | edm2-img64-xs at 256 pictures per pass ran out of a 24 GB card; 128 per pass with batch 256 fits | the colour-expert review, overfit row; `m7_train.sbatch` sets `BATCH_GPU=128` below 40 GB |
| Training rate | 6.82 seconds per thousand images seen on one RTX 3090 at 128 per pass (M7); 4.28 on an RTX 4090 (the colour-expert overfit); ratio 1.59 | the CLEVR scaling plan; the colour-expert review |
| Blackwell nodes | `mscluster110` to `112` need `co3_bw`; under `co3` a job runs on the CPU with no error. `m7_train.sbatch` picks the environment by host name | the [node table](hpc/nodes.md#partitions) |
| Short runs | keep the learning-rate ramp at 0.93% of training at batch 256 | ledger entry M7 |

## The colour experts' model files

Navigation: ⬅️ [EDM2](#edm2) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

The red and blue experts' final snapshots (0.050 and 0.100 EMA), stats and logs are in
`/datasets/mmolefe/poe_repair_min/loop-6/m10/full/models/r-full/` and `b-full/`, each snapshot's sha256 in `pod.sha256` beside it.
The empty-scene model is the M7 run's final 0.100 snapshot,
`/datasets/mmolefe/poe_repair_min/loop-6/m7/full/models/empty/network-snapshot-0002000-0.100.pkl` (248,538,254 bytes, sha256
`5e7905efd3b723d79507dde0af9d5d63d959738a2e08ab234646101ece8fb272`; verified by listing on 2026-11-26). On the RunPod pod the
same file sat at `models/empty/network-snapshot-0002000-0.100.pkl` (line 39 of `m10_pod/m10_compose.py`).
Every picture of the colour-expert comparison is in `/datasets/mmolefe/poe_repair_min/loop-6/m10/comparison/m10-comparison.tar`
(209 MB).

## Still open

Navigation: ⬅️ [Model files](#the-colour-experts-model-files) | 📋 [TOC](#table-of-contents)

- **Whether EDM2 training-state files were copied back.** The colour-expert review lists snapshots, stats, logs and samples, not
  `training-state-*.pt`, so resuming training from a colour expert may need a loader that starts from the snapshot's weights.
