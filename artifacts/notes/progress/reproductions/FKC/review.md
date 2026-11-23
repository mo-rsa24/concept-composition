# Review: reproducing FKC on GenEval

This file holds what the runs in [the FKC design](design.md) produced. FKC is not rerun, by the owner's answer to board question 41, so its printed 0.78 stays as context. Plain SDXL on the same prompts,
the number a Paper 2 win must beat, scores 0.7399 of 396 images correct, against the 0.80 printed (pass band 0.6468 to 0.8330, reproduced).

## Runs

Outputs are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/FKC/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. A commit pair is the repository commit, then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Timing of one 8-particle batch | 93248 | `ac85826a-47e659d` | `timing/fkc-timing.json` | done |
| Plain SDXL, 99 two-object prompts at 4 images | 93251 | `ac85826a-47e659d` | `full/sdxl/` | done |
| GenEval scoring of plain SDXL | 93311 | `ac85826a-47e659d` | `full/sdxl-scores.jsonl`, `full/sdxl-result.json` | done, W&B `qcdz35cj` |

## Questions

**What does one FKC image cost?** One batch of 8 particles takes 205.6 seconds on an RTX 3090, against 12.8 seconds for a plain render, so 16.1 times as
long. The authors' sampler calls the model twice per step, which doubles the 8 particles. The batch peaks at 7.9 GB with 4 particles per U-Net call.

| Particles per image | Images | Measured or scaled cost |
|---|---|---|
| 8 | 396 | 22.6 GPU-hours, from the measured batch |
| 4 | 396 | about 11 GPU-hours, scaled from the 8-particle batch and not measured |

**Do the particles collapse?** Yes, on this one prompt. All 8 particles at the last step descend from one ancestor, as the paper's note on weight
degeneracy (p7) describes.

**Does FKC reproduce its published score?** Not tested, by the owner's choice. Its printed 0.78 sits below plain SDXL's printed 0.80 on the same prompts, and a Paper 2 win must also
beat plain SDXL, so the rerun cannot raise the bar on this row.
