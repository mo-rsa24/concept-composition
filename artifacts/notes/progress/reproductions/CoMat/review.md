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
commits `8e7cea34-4bfd680`, W&B `dk8g2tpx`). 

**Does plain SDXL through CoMat's sampler match the printed plain score?** ✅ Yes. On the 288 allowed Color prompts at seed 42 it scores 0.6172, with a pass band of 0.5582 to 0.6761 (render job 93584, scoring job 93661, commits `8e7cea34-4bfd680`, W&B
`gk8ylo47`). CoMat prints 0.5879 for plain SDXL and ToMe prints 0.6369, and both sit inside that band.

**How does CoMat score on the prompts each comparison can use?** These are descriptive numbers from `scripts/reproductions/named_subsets.py` and change no verdict. CoMat's score uses its 10 seeds. Plain SDXL has seed 42 only, so the gain is CoMat minus plain SDXL on
the seed 42 pictures, with a standard error from 10,000 resamples of prompts. CoMat prints a gain of 0.1948 over plain SDXL on all prompts.

| Subset | Prompts | CoMat, 10 seeds | CoMat, seed 42 | Plain SDXL, seed 42 | Gain | Standard error |
|---|---|---|---|---|---|---|
| all allowed | 288 | 0.7696 | 0.7816 | 0.6172 | 0.1644 | 0.0183 |
| two subjects (`two-subject-prompts.json`) | 233 | 0.8018 | 0.8122 | 0.6367 | 0.1755 | 0.0212 |
| outside CoMat's training list | 256 | 0.7802 | 0.7889 | 0.6128 | 0.1761 | 0.0198 |
