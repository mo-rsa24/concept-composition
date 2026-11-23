# Review: reproducing CoMat

This file holds what the runs in [the CoMat design](design.md) produced. CoMat has no verdict yet. Its control, smoke test and 1% pictures pass. By the owner's answer to board
question 43, CoMat runs on Color only, because its printed Color score is the best on that benchmark. Jobs 93312, 93313, 93315 and 93316 render the 288 allowed
Color prompts at 10 seeds. CoMat's Shape and Texture numbers stay as printed context, and 35 unscored Shape pictures stay on the cluster.

## Runs

Pictures are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/CoMat/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. A commit pair is the repository commit, then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Controls | 93224 | `a5ef6e9e-3173925` | `controls/zero.json` | passed |
| Smoke test, 2 Shape prompts on seed 42 | 93230 | `a5ef6e9e-3173925` | `smoke/` | passed: 15.1 and 15.2 seconds per picture on a Quadro RTX 8000 |
| 1% of the data, 9 prompts on 10 seeds | 93231 | `a5ef6e9e-3173925` | `one-percent/` | passed: 90 pictures, median 15.8 seconds each on a Quadro RTX 8000 |

## Questions

**Did every LoRA layer load?** ✅ Yes. All 560 up and down pairs merged into their layers. On a random input, each merged layer matches the LoRA's own
forward rule to within 4.3e-6, against a bar of 1e-4.

**At LoRA scale 0, does the pipeline match the fine-tuned U-Net alone?** ✅ Yes. The largest absolute difference between final latents is 0.0, against a
bar of 1e-5, in one process on a Quadro RTX 8000 at float16.

**Does the LoRA change the render?** ✅ Yes. At scale 1 the largest absolute difference in the final latent is 4.59, against a floor of 1e-3.

**Do the 1% pictures show their prompts' subjects?** ✅ Yes, as renders. The [sheet of 9 prompts at seed 42](comat-one-percent-seed42.png), read at 400 pixels per
tile, shows clean images with most colours and materials on the right objects. Some prompts still drift: "a pentagonal stop sign" renders as a small
diamond sign beside a traffic light. No score was computed at this rung.

**Does CoMat reproduce its published Color score?** Unknown until the Color renders finish and are scored.
