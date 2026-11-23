# Review: reproducing Rectify Then Diffuse

This file holds what the runs in [the RTD design](design.md) produced. RTD does not reproduce its printed T2I-CompBench
Complex score on the 988 allowed prompts. It passes on RareBench Concat, where 33 prompts make the test weak. By the owner's
answer to board question 26, the printed 0.466 stays the bar, and this rerun of 0.242 is reported beside it.

## Runs

Pictures and scores are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/RTD/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. A commit pair is the repository commit, then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Prepare prompts | 92793 | `1c86bf0a-5c50228` | `data/rtd/processed/` | done: 1,000 Complex, 40 Concat |
| Smoke test | 92814, 92815, 92817, 92818 | `1c86bf0a-5c50228` | `smoke/` | passed |
| Zero-update check | 92816, then 92823 with the guard | `1c86bf0a-5c50228`, `e39a31ca-0739e7e` | `controls/` | passed |
| 1% of the data | 92819, 92821, 92822, 92834 (retry of 92820), scored by 92836 and 92837 | `e39a31ca-0739e7e` | `one-percent/` | passed |
| Full protocol, Concat | 92840, 92843 | `71c7a091-2d34fee` | `full/` | done: 99 pictures per method |
| Full protocol, Complex | 92859 to 92862, continuing the pictures of 92838, 92839, 92841 and 92842 | `045e027f-94b2919`, and `71c7a091-2d34fee` for the pictures already made | `full/` | done: 2,964 pictures per method |
| Full scoring | 93064 (RTD), 93068 (plain SDXL) | `045e027f-94b2919` | `full/rtd-scores.jsonl`, `full/sdxl-scores.jsonl` | done: 3,063 scores per method |

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

**Does plain SDXL reproduce its printed ImageReward, which checks our prompts and scorer?** ✅ Yes, on both benchmarks.

**Does RTD reproduce its published ImageReward on the allowed prompts?** ❌ Not on Complex. ✅ On Concat, with a weak test.

ImageReward is a learned score of how well a picture matches its prompt, and higher is better. Each row averages 3 seeds per
prompt. The pass band is ours ± 1.96 × √2 × the bootstrap standard error, because neither paper prints a standard error.

| Method | Benchmark | Prompts | Ours | Pass band | Printed | Verdict | W&B run |
|---|---|---|---|---|---|---|---|
| RTD | T2I-CompBench Complex | 988 of 1,000 | 0.2423 | 0.1751 to 0.3095 | 0.4661 | not reproduced | `5i9dt7by` |
| RTD | RareBench Concat | 33 of 40 | 0.4166 | -0.0268 to 0.8600 | 0.4464 | reproduced | `tc75fx0i` |
| plain SDXL | T2I-CompBench Complex | 988 of 1,000 | 0.2637 | 0.2019 to 0.3256 | 0.3083 | reproduced | `8802binw` |
| plain SDXL | RareBench Concat | 33 of 40 | 0.3035 | -0.1269 to 0.7338 | 0.1018 | reproduced | `1gh0vhil` |

**Does RTD beat plain SDXL on the same prompts and seeds?** 🟡 Not shown on either benchmark. RTD minus plain SDXL, matched
picture by picture, is -0.0214 on Complex (standard error 0.0153 over prompts), where the printed scores differ by +0.1578. On
Concat it is +0.1131 (standard error 0.0983).

**Why does the Complex score miss?** 🟡 Unknown. These checks rule out the causes that could be tested here:

- **The pictures are sound.** [Eight Complex prompts drawn at random](plain-sdxl-against-rtd-eight-random-complex-prompts-seed0.png)
  (generator seed 0, picture seed 0) show clean renders from both methods. RTD moves the layout and sometimes loses an object:
  on the cube, rectangle and sphere prompt it drops the cube.
- **The scorer does not always follow the composition.** On the lemon and lime prompt, RTD's picture shows separate lemons and
  limes as a flat pattern and scores -0.90, against -0.02 for a plain SDXL picture of limes with lemon-coloured flesh. Eight
  prompts are too few to say how often this happens.
- **The update ran.** None of the 2,964 Complex scores is identical between the two methods, and the zero-update check above passes.
- **The code is the authors'.** Both methods call their `RTDPipeline` at `241eb37`, whose `RTDConfig.validate` fixes the printed
  settings, with the library versions they pin.

Three causes remain open: the authors' H200 against our RTX 3090s, an ImageReward package or preprocessing difference the paper
does not state, and the printed number itself. Nothing run here separates them.
