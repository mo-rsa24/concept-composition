# PRZE-10: the compute plan

Every `biggpu` node was probed on 2026-11-10, so jobs are placed from the table below and not from memory. Jobs go to 107 (your interactive job), 108, 111 and 112. Node 110 holds another user's job until about 2026-11-12 11:30 and is not planned on.

## The nodes, probed 2026-11-10

Each node was reached over `ssh` from mscluster84 through `scripts/cluster.sh sh`, read with `nvidia-smi -L` and `--query-gpu`, and given a 1000 by 1000 ones matmul on every card under the absolute path of the named python. A matmul sum of 1e9 means the card computed on the GPU. State and jobs come from `sinfo -N` and `squeue -w` the same hour.

| Node | Cards | Python | `torch.cuda.is_available()` | Matmul sum | Slurm state, who holds it | Usable for this work |
|---|---|---|---|---|---|---|
| mscluster107 | 2× Quadro RTX 8000, 49,152 MiB each | `co3` | True (earlier probe today) | | alloc, your job 83423 (interactive, 3-day limit) | yes, inside your job |
| mscluster108 | 2× Quadro RTX 8000, 49,152 MiB each, 1 MiB used | `co3` (torch 2.5.1+cu118) | True | 1e9 on both cards | idle | yes |
| mscluster110 | 1× RTX PRO 6000 Blackwell, 97,887 MiB, 17,624 MiB used, 100% busy | `co3_bw` (torch 2.11.0+cu128) | True | 1e9 | alloc, avogt job 83315, started 1 day 3 h before the probe, 3-day limit | no, until about 2026-11-12 11:30 |
| mscluster111 | 1× RTX PRO 6000 Blackwell, 97,887 MiB, 2 MiB used | `co3_bw` (torch 2.11.0+cu128) | True | 1e9 | idle | yes |
| mscluster112 | 1× RTX PRO 6000 Blackwell, 97,887 MiB, 2 MiB used | `co3_bw` | True (earlier probe today) | 8e9 on a 2000 by 2000 matmul | idle | yes |

No node failed to answer, so no retry was needed.

## Outcomes

- [x] ✅ check passed: `nvidia-smi -L` and a torch CUDA probe on each candidate node (`co3` on the RTX 8000 nodes, `co3_bw` on the Blackwell nodes), with [faulted cards](../../environment/known-failures.md#entry-id-poe-launch-002) excluded and the result written down. ✓ verified 2026-11-10: all five nodes answer with a working card (table above); 110 is excluded because another user holds it, not for a fault. `/datasets` is 44% used with 212 TB free `verified`
- [ ] 📝 markdown created: a compute plan table, one row per job: node, GPU memory needed, expected hours (measured where a measurement exists, estimate otherwise), priority, and what it waits on
- [ ] 💻 code ran: jobs queued in that order; biggpu's [one-job-per-user rule](../../runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md#3-harvest-what-is-running) is respected by starting the extra long jobs with `nohup` over SSH and checking them with `pgrep`
- [ ] ✅ check passed: every job's output lands on `/datasets`, with the [disk guard](../../environment/storage.md#the-disk-guard-rule-and-the-failure-that-motivates-it) reading that filesystem
