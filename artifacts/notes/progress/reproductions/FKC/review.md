# Review: reproducing FKC on GenEval

This file holds what the runs in [the FKC design](design.md) produced. FKC has no verdict and has not rendered a scored image. By the owner's answer to
board question 40, plain SDXL runs first and one FKC batch is timed. Board question 41 brings the measured cost back to the owner.

## Runs

Outputs are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/FKC/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. A commit pair is the repository commit, then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Timing of one 8-particle batch | 93248 | `ac85826a-47e659d` | `timing/fkc-timing.json` | done |
| Plain SDXL, 99 two-object prompts at 4 images | 93251 | `ac85826a-47e659d` | `full/sdxl/` | running |
| GenEval scoring of plain SDXL | queued | `ac85826a-47e659d` | `full/sdxl-scores.jsonl` | waits on the renders |

## Questions

**What does one FKC image cost?** One batch of 8 particles takes 205.6 seconds on an RTX 3090, against 12.8 seconds for a plain render, so 16.1 times as
long. The authors' sampler calls the model twice per step, which doubles the 8 particles. The batch peaks at 7.9 GB with 4 particles per U-Net call.

| Particles per image | Images | Measured or scaled cost |
|---|---|---|
| 8 | 396 | 22.6 GPU-hours, from the measured batch |
| 4 | 396 | about 11 GPU-hours, scaled from the 8-particle batch and not measured |

**Do the particles collapse?** Yes, on this one prompt. All 8 particles at the last step descend from one ancestor, as the paper's note on weight
degeneracy (p7) describes.

**Does FKC reproduce its published score?** Not run. Its printed 0.78 sits below plain SDXL's printed 0.80 on the same prompts, and a Paper 2 win must also
beat plain SDXL, so the rerun cannot raise the bar on this row.
