# Review: reproducing CO3

This file holds what the runs in [the CO3 design](design.md) produced. CO3 has no verdict yet.

## Runs

Pictures and scores are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/CO3/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. A commit pair is the repository commit, then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Smoke test | 92910 (retry of 92908) | `67b0bafa-b91076f` | `smoke/` | passed: 2 pictures at 49 and 38 seconds on an RTX 3090 |
| Zero-correction check | 92911 (retry of 92909) | `67b0bafa-b91076f` | `controls/zero_check.json` | passed |
| 1% of the data | 92924, 92925 | `67b0bafa-b91076f` | `one-percent/` | running |

Jobs 92908 and 92909 failed on import, because the release does not ship `composers/utils/gaussian_smoothing.py`. The stub in the
hub's `scripts/reproductions/co3_common.py` stands in for that unused import.

## Questions

**Does a zero correction reproduce CO3's plain loop?** ✅ Yes. With the corrector replaced by the identity on all 6 corrected steps,
the largest absolute difference between final latents was 0.0, against a bar of 1e-5, in one process on an RTX 3090 at float16.

**Does the real corrector change the render?** ✅ Yes. The largest absolute difference in the final latent was 4.16, against a floor
of 1e-3.

**Does CO3 reproduce its published ImageReward on the allowed prompts?** Unknown until the full run finishes.
