# Review: reproducing CoMat

This file holds what the runs in [the CoMat design](design.md) produced. CoMat has no verdict yet. Its control and smoke test pass, and the 1% rung
is running.

## Runs

Pictures are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/CoMat/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. A commit pair is the repository commit, then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Controls | 93224 | `a5ef6e9e-3173925` | `controls/zero.json` | passed |
| Smoke test, 2 Shape prompts on seed 42 | 93230 | `a5ef6e9e-3173925` | `smoke/` | passed: 15.1 and 15.2 seconds per picture on a Quadro RTX 8000 |

## Questions

**Did every LoRA layer load?** ✅ Yes. All 560 up and down pairs merged into their layers. On a random input, each merged layer matches the LoRA's own
forward rule to within 4.3e-6, against a bar of 1e-4.

**At LoRA scale 0, does the pipeline match the fine-tuned U-Net alone?** ✅ Yes. The largest absolute difference between final latents is 0.0, against a
bar of 1e-5, in one process on a Quadro RTX 8000 at float16.

**Does the LoRA change the render?** ✅ Yes. At scale 1 the largest absolute difference in the final latent is 4.59, against a floor of 1e-3.

**Does CoMat reproduce its published scores?** Unknown until the full runs finish.
