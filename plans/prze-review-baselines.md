# 🧪 PRZE-1: CO3 and SuperDiff as tuned baselines on R1 and R3

**Goal:** reviewer PrzE sees the adapter beside two standard composition baselines, each run fairly: its published default and one tuned setting, both from our starting noise wherever its code accepts it.
**Objective:** each method renders one tuning cell, gets a grid and a winning rule written into [the protocol](../report/paper-review/protocol.md) before any tuning image, has its winner frozen from a contact sheet, and is launched at its default and frozen settings on R1 (276 prompts, seeds 9 to 12) and R3 (53 held-out pairs, seeds 9 to 16).
**Definition of done:** every method has a frozen setting chosen by the written rule, and every evaluation job at both settings is running or finished, with its output count heading to 1,104 for R1 and 424 for R3.
**Route:** [answer the ICLR reviews](route-answer-the-iclr-reviews.md), between steps 8 and 9; outcome PRZE-1, compute jobs 7 and 8 of [the jobs table](../report/paper-review/prze-10-compute.md#the-jobs).

## Environment Facts This Plan Depends On

- [The shared-device path and its safety rules](../environment/hpc/execution-protocol.md): a guard inside each launcher refuses a card holding over 1 GB, every path is absolute, the log header names node, card and PID.
- [Storage](../environment/storage.md): outputs under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/<method>_<setting>/` only, and the disk guard reads that filesystem.
- CO3 needs the `co3` environment (it has xformers); `co3_bw` on the Blackwell nodes has none, so CO3 runs only on RTX 8000 or RTX 3090 cards.
- `scripts/cluster.sh run` refuses to sync while another session's three trainer files are uncommitted, so launchers run from copies under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers/`, each under a new file name.

## Decisions

| Decision | Chosen | Other option |
|---|---|---|
| CO3's starting noise | a wrapper outside `composition/debottam_co3/` calls its `sample_loop` with our cached latent; the vendored code is not edited | its own noise from the seed number, marked mixed |
| ToMe | dropped from the SDXL table: its official code (commit `f256a49a`) did not import within the 30-minute limit, because its requirements leave `huggingface_hub` unpinned and diffusers 0.28 fails against 0.36; the paper states the limit | one more attempt with the hub library pinned |
| SuperDiff's starting noise | its own, from the seed number; its stochastic sampler differs on two axes anyway, so its column is mixed | |

## Tasks

For Claude to execute.

### 1. 🧪 One tuning cell per method

- [x] **CO3 and SuperDiff each render lion and meerkat at seed 9 on SDXL, 1024 by 1024, at their published defaults, with a sidecar carrying protocol rule 9's columns and whether the starting latent was ours**
  - ✓ verified 2026-11-11 (Slurm 84574, mscluster48 RTX 3090): `co3_default/a_lion__x__a_meerkat/seed09/co3.json` (our seed 9 latent, 79.7 s, nfe 260) and `superdiff_default/.../superdiff.json` (own noise, 200 steps, 165.6 s, nfe 600), under `/datasets/mmolefe/poe_repair_min/outputs/paper_review/`
  - ▶ Next: 2

### 2. 📝 Grids and the winning rule

- [ ] **Write each method's grid of at most six settings over its main knob, and the numeric rule that picks the winner, into `report/paper-review/protocol.md`, with a dated line in `report/paper-review/history.md`**
  - ◀ Needs: 1
  - ▶ Next: 3

### 3. 🚀 Run the grids and freeze the winners

- [ ] **Run each grid on lion and meerkat, camel and pelican, donkey and duck, chicken and rabbit at seeds 9 to 16, score with the instance count, build one contact sheet per method, and freeze the winner by the written rule**
  - ◀ Needs: 2, and your yes to each launch command
  - ▶ Next: 4

### 4. 🚀 Launch the evaluations

- [ ] **For each method at its default and its frozen setting: an 80% interval for runtime and memory from `/calibrate --predict`, the exact command, your yes, the launch, the first tiles**
  - ◀ Needs: 3
  - ▶ Next: 5

### 5. ✅ Count, file and record

- [ ] **Count each job's outputs on its launch node against 1,104 and 424; fill rows 7 and 8 of the jobs table with measured rates and cards; file the contact sheets with card entries; add the route step and its rows to "What we know so far"**
  - ◀ Needs: 4

## Instructions

For you to follow.

### 1. 👀 Look at the three one-cell renders

- [ ] **Open each method's lion and meerkat seed 9 tile and say whether it shows two animals, a hybrid or one animal**
  - ◀ Needs: task 1

### 2. 👀 Approve the grids and the rule

- [ ] **Read the three grids and the rule in the protocol and say yes or change them, before any tuning image exists**
  - ◀ Needs: task 2

### 3. 👀 Check species on each contact sheet

- [ ] **Open each contact sheet and check by eye that the winner's two animals are the named species; the instance count cannot tell species apart**
  - ◀ Needs: task 3

### 4. 👀 Judge PASS or MISS

- [ ] **Read the count table task 5 prints, and say PASS or MISS against the definition of done**
  - ◀ Needs: task 5
