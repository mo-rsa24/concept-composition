# Review: reproducing Rectify Then Diffuse

This file holds what the runs in [the RTD design](design.md) produced. RTD has no verdict yet, because the full runs on the
allowed prompts are in progress.

## Runs

Pictures and scores are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/RTD/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. A commit pair is the repository commit, then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Prepare prompts | 92793 | `1c86bf0a-5c50228` | `data/rtd/processed/` | done: 1,000 Complex, 40 Concat |
| Smoke test | 92814, 92815, 92817, 92818 | `1c86bf0a-5c50228` | `smoke/` | passed |
| Zero-update check | 92816, then 92823 with the guard | `1c86bf0a-5c50228`, `e39a31ca-0739e7e` | `controls/` | passed |
| 1% of the data | 92819, 92821, 92822, 92834 (retry of 92820), scored by 92836 and 92837 | `e39a31ca-0739e7e` | `one-percent/` | passed |
| Full protocol, Concat | 92840, 92843 | `71c7a091-2d34fee` | `full/` | running |
| Full protocol, Complex | 92859 to 92862, continuing the pictures of 92838, 92839, 92841 and 92842 | `045e027f-94b2919`, and `71c7a091-2d34fee` for the pictures already made | `full/` | running |

Jobs 92820 and 92842 failed when two jobs created one prompt's output folder at the same moment on the shared disk. Jobs
92838, 92839 and 92841 were stopped because they carried the same risk. Commit `94b2919` in the hub splits jobs by prompt, so
no two jobs write into one folder. Both commits render with the same pipeline and settings, and finished pictures were kept.

## Questions

**Does a zero update reproduce plain SDXL?** ✅ Yes. The largest absolute difference between final latents was 0.0, against a
bar of 1e-5, in one process on an RTX 3090 at float16.

**Does the real update change the render?** ✅ Yes. The largest absolute difference in the final latent was 3.25, against a
floor of 1e-3.

**Do the 1% pictures show the prompts' subjects, with scores in ImageReward's usual range?** ✅ Yes. The seed 0 comparison
sheets for all 14 prompts were read at 420 pixels per tile, and per-picture scores ran from -1.71 to 1.96. The sheets are in
`one-percent/sheets/`. These 42 pictures per method are too few to compare with a published score, so no test was applied.

| Method | Benchmark | Prompts | Pictures | Mean ImageReward | Bootstrap standard error | W&B run |
|---|---|---|---|---|---|---|
| RTD | T2I-CompBench Complex | 10 of 1,000 | 30 | 0.111 | 0.184 | `z9j1mj2x` |
| RTD | RareBench Concat | 4 of 40 | 12 | -0.387 | 0.457 | `nay9gs6u` |
| plain SDXL | T2I-CompBench Complex | 10 of 1,000 | 30 | 0.155 | 0.166 | `t0xu62a0` |
| plain SDXL | RareBench Concat | 4 of 40 | 12 | -0.131 | 0.405 | `207g8vl0` |

**Does RTD reproduce its published ImageReward on the allowed prompts?** Unknown until the full runs finish.
