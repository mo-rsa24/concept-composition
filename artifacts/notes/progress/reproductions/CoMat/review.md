# Review: reproducing CoMat

This file holds what the runs in [the CoMat design](design.md) produced. CoMat has no verdict yet. Its control, smoke test and 1% pictures pass. By the owner's answer to board
question 43, CoMat runs on Color only, because its printed Color score is the best on that benchmark. Jobs 93312, 93313, 93316 and 93407 render the 288 allowed
Color prompts at 10 seeds. Job 93407 replaced 93315, whose RTX 3090 on mscluster61 was held at 315 MHz by software thermal slowdown and took 63 seconds per picture against 13 to 16 elsewhere. CoMat's Shape and Texture numbers stay as printed context, and 35 unscored Shape pictures stay on the cluster.

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

**Does CoMat reproduce its published Color score?** ✅ Yes. CoMat scores 0.7696 on the 288 allowed prompts at 10 seeds (2,880 pictures), with a pass band of 0.7331 to 0.8060, and the paper prints 0.7827 (scoring job 93588,
commits `8e7cea34-4bfd680`, W&B `dk8g2tpx`). The plain SDXL check on the same prompts (job 93584) was at 202 of 288 pictures at 14:36 on 2026-11-23. From 15:07 the cluster's login node accepted connections and then hung, so its later progress has not been read.
