# 🧪 PRZE-1: CO3 as a baseline on R1 and R3

**Goal:** reviewer PrzE sees the adapter beside CO3, run at its authors' published settings and from our starting noise.
**Objective:** CO3 renders one tuning cell, its published default is written into [the protocol](../report/paper-review/protocol.md), and it is launched at that default on R1 (276 prompts, seeds 9 to 12) and R3 (53 held-out pairs, seeds 9 to 16).
**Definition of done:** every CO3 evaluation job is running or finished, with the output count heading to 1,104 for R1 and 424 for R3.
**Route:** [answer the ICLR reviews](route-answer-the-iclr-reviews.md), between steps 8 and 9; outcome PRZE-1, compute jobs 7 and 8 of [the jobs table](../report/paper-review/prze-10-compute.md#the-jobs).

## Environment Facts This Plan Depends On

- [The shared-device path and its safety rules](../environment/hpc/execution-protocol.md): a guard inside each launcher refuses a card holding over 1 GB, every path is absolute, the log header names node, card and PID.
- [Storage](../environment/storage.md): outputs under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/<method>_default/` only, and the disk guard reads that filesystem.
- CO3 needs the `co3` environment (it has xformers); `co3_bw` on the Blackwell nodes has none, so CO3 runs only on RTX 8000 or RTX 3090 cards.
- `scripts/cluster.sh run` refuses to sync while another session's three trainer files are uncommitted, so launchers run from copies under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers/`, each under a new file name, and name those copies by absolute path because `sbatch` runs a launcher from the Slurm spool.
- Seeds 9 to 16 have no cache entry, so their starting latent is the seeded fp16 CPU draw `load_shared_latents` falls back to, saved per seed under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/latents/`.

## Decisions

| Decision | Chosen | Other option |
|---|---|---|
| CO3's starting noise | a wrapper outside `composition/debottam_co3/` calls its `sample_loop` with our seed latent; the vendored code is not edited | its own noise from the seed number, marked mixed |
| ToMe | dropped from the SDXL table: its official code (commit `f256a49a`) did not import within the 30-minute limit, because its requirements leave `huggingface_hub` unpinned and diffusers 0.28 fails against 0.36; the paper states the limit | one more attempt with the hub library pinned |
| SuperDiff | out of every table and figure: it does not claim to fix hybrids, so it is no composition baseline (`ef5494c9`); its tiles under `superdiff_default/` stay unused | a mixed column with its own noise and sampler |
| Tuning | none: CO3 runs at its published default only | a grid of six settings on the tuning pairs |

## Tasks

For Claude to execute.

### 1. 🧪 One tuning cell per method

- [x] **CO3 and SuperDiff each render lion and meerkat at seed 9 on SDXL, 1024 by 1024, at their published defaults, with a sidecar carrying protocol rule 9's columns and whether the starting latent was ours**
  - ✓ verified 2026-11-11 (Slurm 84574, mscluster48 RTX 3090): `co3_default/a_lion__x__a_meerkat/seed09/co3.json` (our seed 9 latent, 79.7 s, nfe 260) and `superdiff_default/.../superdiff.json` (own noise, 200 steps, 165.6 s, nfe 600), under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/`
  - ▶ Next: 2

### 2. 📝 The published defaults in the protocol

- [x] **Write each method's published default and the script that renders it into rule 3 of `report/paper-review/protocol.md`, with a dated line in `report/paper-review/history.md`**
  - ✓ verified 2026-11-11: rule 3's table in [the protocol](../report/paper-review/protocol.md) and the two 2026-11-11 rule 3 rows in [the history](../report/paper-review/history.md), committed before any evaluation image
  - ◀ Needs: 1
  - ▶ Next: 3

### 3. 🚀 Launch the evaluations

- [x] **For each method at its published default: an 80% interval for runtime and memory, the exact command, your yes, the launch, the first tiles**
  - ✓ verified 2026-11-11 14:10: CO3 complete, R1 1,104 of 1,104 and R3 424 of 424 tiles with 1,528 sidecars under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/co3_default/`, counted from mscluster107 (mscluster84's view undercounted by 35); Slurm 84590, 84591, 84757, 84759, 84760, 84761 and two mscluster108 helpers (`cdd58a18`)
  - Launcher: `scripts/review_baselines_eval.sh`, every Nth cell of the 1,528 per job; each job renders its R1 share first and its R3 share last. CO3 runs as two `NSHARDS=2` jobs and four `NSHARDS=6` jobs (shards 2 to 5); a tile that exists is skipped, so they meet without repeating work.
  - ◀ Needs: 2
  - ▶ Next: 4

### 4. ✅ Count, file and record

- [x] **Count each job's outputs on its launch node against 1,104 and 424; fill rows 7 and 8 of the jobs table with measured rates and cards; add the route step and its rows to "What we know so far"**
  - ✓ verified 2026-11-11: 1,104 and 424 counted from mscluster107; rows 7 and 8 in [the jobs table](../report/paper-review/prze-10-compute.md#the-jobs); route step 8b and five rows in [the route](route-answer-the-iclr-reviews.md#what-we-know-so-far)
  - ◀ Needs: 3

## Instructions

For you to follow.

### 1. 👀 Look at the one-cell renders

- [ ] **Open each method's lion and meerkat seed 9 tile and say whether it shows two animals, a hybrid or one animal**
  - ◀ Needs: task 1

### 2. 👀 Judge PASS or MISS

- [ ] **Read the count table task 4 prints, and say PASS or MISS against the definition of done**
  - ◀ Needs: task 4
