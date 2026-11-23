# Review: reproducing ToMe

This file holds what the runs in [the ToMe design](design.md) produced. ToMe has no verdict yet. Its parser repair, smoke test and
zero-update control pass, and the 1% rung comes next.

## Runs

Pictures are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-5/renders/ToMe/`, and logs are in
`/datasets/mmolefe/poe_repair_min/loop-5/logs/`. A commit pair is the repository commit, then the replications hub commit.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| Parser repair check, CPU | 93144 | `23171540-ac83a43` | `data/tome-parse-repaired.jsonl` | passed |
| Smoke test, 2 prompts | 93146 | `3ceb63cb-bdc0e0a` | `smoke/` | passed: 60.6 and 53.2 seconds per picture |
| Attention repeatability checks | 93148, 93149, 93155, 93166, 93196, 93199 | from `3ceb63cb-c5c5cf1` to `d37f3100-4e4c6fd` | `controls/` | found the cause |
| Zero-update control | 93202 | `d37f3100-b36238e` | `controls/zero-final.json` | passed |

Jobs 93145 and 93167 failed the control because of the repeatability problem below. Jobs 93197, 93198, 93200 and 93201 ran out of memory
on two attempted fixes. Job 93151 failed on a mistake in the trace script.

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

**Does ToMe reproduce its published scores?** Unknown until the full runs finish.
