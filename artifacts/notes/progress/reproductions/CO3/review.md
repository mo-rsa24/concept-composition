# Review: reproducing CO3

This file holds what the runs in [the CO3 design](design.md) produced. CO3 has no verdict yet.

## Runs

Pictures and scores are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/CO3/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. A commit pair is the repository commit, then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Smoke test | 92910 (retry of 92908) | `67b0bafa-b91076f` | `smoke/` | passed: 2 pictures at 49 and 38 seconds on an RTX 3090 |
| Zero-correction check | 92911 (retry of 92909) | `67b0bafa-b91076f` | `controls/zero_check.json` | passed |
| 1% of the data | 92924, 92925, scored by 92927 | `67b0bafa-b91076f` | `one-percent/` | passed |
| Full protocol | 4 Complex shards, 1 Concat job and scoring job 93326 | `67b0bafa-b91076f` | `full/`, `full/co3-scores.jsonl` | done |

Jobs 92908 and 92909 failed on import, because the release does not ship `composers/utils/gaussian_smoothing.py`. The stub in the
hub's `scripts/reproductions/co3_common.py` stands in for that unused import.

## Questions

**Does a zero correction reproduce CO3's plain loop?** ✅ Yes. With the corrector replaced by the identity on all 6 corrected steps,
the largest absolute difference between final latents was 0.0, against a bar of 1e-5, in one process on an RTX 3090 at float16.

**Does the real corrector change the render?** ✅ Yes. The largest absolute difference in the final latent was 4.16, against a floor
of 1e-3.

**Do the 1% pictures show the prompts' subjects, with scores in ImageReward's usual range?** ✅ Yes. The seed 0 comparison sheets for
all 14 prompts were read at 420 pixels per tile beside plain SDXL, and per-picture scores ran from -1.51 to 1.84. The sheets are in
`one-percent/sheets/`. The median render took 38.4 seconds over 30 Complex pictures, so the full 3,063 pictures cost about 33 GPU-hours.
These 42 pictures are too few to compare with a published score, so no test was applied.

| Benchmark | Prompts | Pictures | Mean ImageReward | Bootstrap standard error | W&B run |
|---|---|---|---|---|---|
| T2I-CompBench Complex | 10 of 1,000 | 30 | 0.188 | 0.219 | `6ej2ebnk` |
| RareBench Concat | 4 of 40 | 12 | 0.107 | 0.387 | `qci2qjgy` |

**Does CO3 reproduce its published ImageReward on the allowed prompts?** On Complex no, and on Concat yes on a weak test.

| Benchmark | Prompts and seeds | CO3, ours | Pass band | Printed | Verdict | Plain SDXL, same pictures | RTD, same pictures | W&B run |
|---|---|---|---|---|---|---|---|---|
| T2I-CompBench Complex | 988 at 3 seeds | 0.3326 | 0.2584 to 0.4069 | 0.4406 | not reproduced | 0.2637 | 0.2423 | `02fl0ia0` |
| RareBench Concat | 33 at 3 seeds | 0.4374 | -0.0678 to 0.9425 | 0.1610 | reproduced, weak test | 0.3035 | 0.4166 | `i11r81ox` |

The Rectify Then Diffuse paper prints RTD above CO3 on Complex, at 0.4661 against 0.4406. In our runs on the same pictures CO3 scores above RTD, at 0.3326 against 0.2423. Concat's
band is too wide to separate its printed score from ours, even though ours is 0.2764 higher.

**How do CO3 and RTD score on Complex's two-subject prompts?** The fixed list in `two-subject-prompts.json` holds 463 of the 988 allowed prompts. These are descriptive numbers on saved pictures, with no new render and no change to any verdict. The gain is
the method minus plain SDXL on the same pictures, with a standard error from 10,000 resamples of prompts.

| Method | Prompts and seeds | Score | Plain SDXL | Gain | Standard error |
|---|---|---|---|---|---|
| CO3 | 463 at 3 seeds | 0.4628 | 0.3678 | 0.0950 | 0.0303 |
| RTD | 463 at 3 seeds | 0.3689 | 0.3678 | 0.0011 | 0.0239 |
