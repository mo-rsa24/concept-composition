# Review: reproducing ToMe

This file holds what the runs in [the ToMe design](design.md) produced. ToMe has no verdict yet. Its parser repair, smoke test,
zero-update control and 1% pictures pass. By the owner's answer to board question 43, ToMe is scored on Shape and Texture only, where its printed scores are the
best on each benchmark. Those renders are complete on seed 0. Its Complex and Color runs stopped, with 179 of 988 Complex and 72 of 288 Color pictures unscored on the cluster.

## Runs

Pictures are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/ToMe/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. A commit pair is the repository commit, then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Parser repair check, CPU | 93144 | `23171540-ac83a43` | `data/tome-parse-repaired.jsonl` | passed |
| Smoke test, 2 prompts | 93146 | `3ceb63cb-bdc0e0a` | `smoke/` | passed: 60.6 and 53.2 seconds per picture |
| Attention repeatability checks | 93148, 93149, 93155, 93166, 93196, 93199 | from `3ceb63cb-c5c5cf1` to `d37f3100-4e4c6fd` | `controls/` | found the cause |
| Zero-update control | 93202 | `d37f3100-b36238e` | `controls/zero-final.json` | passed |
| 1% of the data, 16 prompts on seed 0 | 93204 (Complex), 93207 (Shape and Texture, retry of 93203) | `d37f3100-b36238e`, `9efe30d1-b988f7a` | `one-percent/` | passed: median 57 seconds per picture |

Jobs 93145 and 93167 failed the control because of the repeatability problem below. Jobs 93197, 93198, 93200 and 93201 ran out of memory
on two attempted fixes. Job 93151 failed on a mistake in the trace script. Job 93203 failed when two jobs created one output folder at
the same moment on the shared disk, and folder creation now retries.

## Questions

**Does the repaired parser leave every working prompt unchanged?** ✅ Yes. None of the 1,751 prompts that already parsed changed. All 147
that crashed now parse, and their 310 attribute groups each land on a phrase containing their noun. Two groups, one each in
`t2i_compbench-0746` and `t2i_compbench-0980`, sit in no kept phrase and are left out of the merge.

**With its updates off, does ToMe's loop match plain SDXL?** ✅ Yes. The largest absolute difference between final latents is 0.0, against
a bar of 1e-5, on the math attention backend in one process on an RTX 3090 at float16.

**Does the real update change the render?** ✅ Yes. The largest absolute difference in the final latent is 3.56, against a floor of 1e-3.

**Is a ToMe render repeatable?** ❌ No. Torch 2.0.1's default attention backends return slightly different outputs for identical inputs.

| Attention backend, stock SDXL with no ToMe code | One U-Net call, largest difference over 4 repeats | Full 50-step render, over 3 repeats | Seconds per render | Peak memory |
|---|---|---|---|---|
| Default | 0.00195 | 1.476 | 13.5 | 6.9 GB |
| Flash off | 0.00195 | 1.458 | 14.4 | 6.9 GB |
| Flash and memory-efficient off (math) | 0.0 | 0.0 | 17.6 | 8.0 GB |

The math backend runs out of a 24 GB card during ToMe's gradient steps, so renders use the default backends and each ToMe score carries
this run-to-run noise. The same failure appeared on three nodes (mscluster54, mscluster61 and mscluster87), so it is not one faulty card.

**Do the 1% pictures show their prompts' subjects?** ✅ Yes, as renders. The [sheet of all 16 pictures](tome-one-percent-seed0.png) (3 Shape,
3 Texture, 10 Complex, seed 0, read at 340 pixels per tile) shows clean images of the named subjects, including the three repaired prompts
in it. It does not show that every attribute landed on the right object: on "a pentagonal stop sign and a spherical traffic light" the two
objects fuse into one. No score was computed at this rung.

**Does our BLIP-VQA scorer match the benchmark's?** ✅ Yes. Run unchanged at the benchmark's package versions, it returns 0.7110, 0.9217 and
0.8554 on the three example pictures, the scores the repository ships, with a largest gap of 0.0 against a bar of 0.01 (job 93213,
commits `fc040954-8207676`, output `renders/blip-canary/result.json`).

**Does plain SDXL reproduce ToMe's printed scores, the check on our scorer?** 🟡 On Texture yes, on Shape no. Plain SDXL ran through ToMe's pipeline
on seed 0 (jobs 93214, 93228 and 93229, commits `5f16f1f6-1abb2a8`). The pass band is ours ± 1.96 × √2 × the bootstrap standard error.

| Benchmark | Prompts | Ours | Pass band | ToMe and RPG print | Verdict | CoMat prints | W&B run |
|---|---|---|---|---|---|---|---|
| Shape | 293 of 300 | 0.4825 | 0.4301 to 0.5348 | 0.5408 | not reproduced | 0.4687 | `92suk4nc` |
| Texture | 300 of 300 | 0.5233 | 0.4630 to 0.5835 | 0.5637 | reproduced | 0.5299 | `acw7fxeb` |

Two papers print different plain SDXL scores on this benchmark, and both of ours fall inside the pass band around CoMat's numbers. By the owner's answer
to board question 39, ToMe's Shape row still gets its verdict, with this missed check printed beside it.

**Does ToMe reproduce its published Shape score?** ❌ No, by a small margin. ToMe scores 0.5502 on 293 prompts at seed 0, with a pass band of 0.4976 to 0.6028, and
the paper prints 0.6051 (job 93303, commits `9efe30d1-b988f7a`, W&B `8cowoyea`). On the 291 prompts that needed no parse repair it scores 0.5495, with a band of
0.4968 to 0.6022 (W&B `c8c153mf`). Plain SDXL missed its printed Shape score on the same pipeline, so part of this gap may come from the setup. ToMe's gain over plain
SDXL is 0.0677 in our run and 0.0643 in the paper's printed numbers.

**Does ToMe reproduce its published Texture score?** ✅ Yes. ToMe scores 0.6503 on all 300 prompts at seed 0, with a pass band of 0.5970 to 0.7036, and the paper
prints 0.6894 (job 93317, commits `9efe30d1-b988f7a`, W&B `ljypsxnd`). No Texture prompt needed a parse repair. ToMe's gain over plain SDXL on the same prompts is
0.1271 in our run and 0.1257 in the paper's printed numbers.

**How does ToMe score on the prompts each comparison can use?** The fairness audit names three subsets. The two-subject prompts are the ones Paper 2's two-output checker can address, and the third leaves out prompts
that appear in CoMat's training list (`CoMat/prompts-in-comat-training.json`). These are descriptive numbers from `scripts/reproductions/named_subsets.py` and change no verdict. The gain is ToMe minus plain SDXL on the same pictures, with a
standard error from 10,000 resamples of prompts.

| Benchmark | Subset | Prompts | ToMe | Plain SDXL | Gain | Standard error |
|---|---|---|---|---|---|---|
| Shape | all | 293 | 0.5502 | 0.4825 | 0.0677 | 0.0157 |
| Shape | two subjects | 261 | 0.5950 | 0.5189 | 0.0761 | 0.0169 |
| Shape | outside CoMat's training list | 289 | 0.5462 | 0.4781 | 0.0681 | 0.0159 |
| Texture | all | 300 | 0.6503 | 0.5233 | 0.1271 | 0.0194 |
| Texture | two subjects | 240 | 0.7553 | 0.6184 | 0.1369 | 0.0231 |

No Texture prompt is in CoMat's training list.
