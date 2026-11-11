# 🧪 PRZE-2: prove the Langevin corrector and render the five-column figure

**Goal:** reviewer PrzE gets the MCMC corrector baseline they asked for, built on an update shown correct on a toy with a known answer.
**Objective:** the corrector's update passes a check against the exact stationary mean and variance at four step sizes; the joint-prompt Langevin and unadjusted HMC drivers match the plain sampler at zero corrector steps; the Langevin setting (c = 3, k = 20) is written into the protocol before any review image; appendix figure R6 is built at that setting.
**Definition of done:** `python3 scripts/check_langevin_two_gaussians.py` prints `PASS: all four step sizes`; both drivers produce byte-identical latents to the plain sampler at k = 0; R6 sits at `artifacts/results/does-a-langevin-corrector-fix-the-product/five_columns.png` with a sidecar listing seeds, noise and settings.
**Route:** [answer the ICLR reviews](route-answer-the-iclr-reviews.md), step 8, outcome PRZE-2.

## Environment Facts This Plan Depends On

- [The shared-device path and its safety rules](../environment/hpc/execution-protocol.md): every GPU job goes through job 3 of [the compute plan](../report/paper-review/prze-10-compute.md) after a yes to its exact command.
- [Storage](../environment/storage.md): renders go under `/datasets/mmolefe/poe_repair_min/` only.
- The corrector keeps its chain latent in float32 and the DDIM step in float16 (`poe_repair/methods/_poe_langevin.py`, docstring).
- CO3's sampler draws its own noise from the integer seed and takes no latent (`composition/debottam_co3/composers/Co3.py:373`), so its column is marked mixed under rule 7.

## Tasks

For Claude to execute.

### 1. 🧮 The two-Gaussian check

- [x] **Write `scripts/check_langevin_two_gaussians.py`: 200,000 chains of the corrector's update on N(-1, 0.5) × N(1, 0.8) at δ = 0.005, 0.02, 0.05, 0.1; pass when mean and variance sit within 3 Monte Carlo standard errors of −0.2308 and V / (1 − δ/(2V)); print the gap to 0.3077 per δ**
  - ▶ Next: instruction 1

### 2. 🔁 The order difference

- [x] **Anneal the same toy over noise levels with the corrector before each step and after it, and print each order's final mean and variance against N(−0.2308, 0.3077)** ✓ verified 2026-11-11: `python3 scripts/check_langevin_order.py`, 13 s, exit 0; at c = 3, k = 20 the KL divergence to the product is 0.00309 corrector first and 0.00171 step first, against 2.25 for plain PoE
  - ◀ Needs: instruction 1
  - ▶ Next: instruction 2

### 3. 💻 The two drivers

- [x] **A joint-prompt Langevin driver and an unadjusted HMC driver beside `run_poe_langevin`, each byte-identical to the plain sampler at k = 0 on one SDXL cell** ✓ verified 2026-11-11: Slurm 84583 on mscluster48, `4 passed`, both SDXL lines `max |diff| = 0.000e+00 PASS`, `PASS: both drivers at k = 0`
  - Code: `poe_repair/methods/_corrector_drivers.py`, composers `poe_repair/composers/joint_langevin.py` and `poe_uhmc.py`; HMC's k counts iterations of L = 5 leapfrog steps, so a level costs 5k + 2 UNet calls.
  - Check: `pytest tests/test_corrector_drivers_identity.py` (stand-in UNet, CPU), then `python scripts/check_corrector_identity.py --pair a_cat__x__a_dog --seed 10` on a GPU, which prints `PASS: both drivers at k = 0`.
  - ◀ Needs: instruction 2
  - ▶ Next: instruction 3

### 4. 🎛️ The setting

- [x] **Write the Langevin setting into protocol rule 3: c = 3, k = 20 over all 50 levels, no tuning grid; joint + Langevin at the same setting; unadjusted HMC not a baseline** ✓ verified 2026-11-11: rule 3 and its history row, written before any review corrector image
  - ◀ Needs: 3
  - ▶ Next: 5

### 5. 🖼️ The five-column figure

- [x] **Render R6 at c = 3, k = 20 and seed 10: joint, joint + Langevin, PoE and PoE + Langevin from the rule 2 starting latent, CO3 from its own noise; write its sidecar; add the rows to the route's "What we know so far" and redraw the board with `--step 8`** ✓ verified 2026-11-11: mscluster107 card 0, log `paper_review/logs/prze2-r6-107.out` ends `render exit 0  co3 exit 0  sheet exit 0  tiles 20`; every row's starting latent matches `latents/seed10.pt`
  - ◀ Needs: 4
  - Command: `sbatch --export=ALL,REPO=<checkout> <checkout>/scripts/review_r6_renders.sbatch` (16 tiles from `scripts/review_r6_renders.py --render`, 4 CO3 tiles from `scripts/review_baseline_cell.py --method co3 --group r6`, then `--sheet`); the log ends `sheet exit 0  tiles 20`.
  - ▶ Next: instruction 4

## Instructions

For you to follow.

### 1. ✅ Run the check

- [x] **In the VS Code terminal at the repo root, run `python3 scripts/check_langevin_two_gaussians.py`** ✓ confirmed 2026-11-11: you read the printed table in chat (four PASS rows; bias 0.0272 at δ = 0.05, 0.0597 at δ = 0.1) and said "done"
  - What you should see: a table of four rows, each ending in PASS, and the last line `PASS: all four step sizes`.
  - What failure looks like: a row ending in MISS, with its var column away from its exact column.
  - ◀ Needs: task 1

### 2. ✅ Read the order measurement

- [x] **Run the order measurement and say which order lands nearer the product** ✓ confirmed 2026-11-11: you read the table in chat (step first 0.00171 against corrector first 0.00309 at c = 3, k = 20) and said "done", keeping our order
  - ◀ Needs: task 2

### 3. ✅ Read the zero-step identity

- [x] **Read the identity check's printed maximum difference for each driver; you want 0** ✓ confirmed 2026-11-11: you read the Slurm 84583 log in chat (both lines `max |diff| = 0.000e+00 PASS`) and said "done"
  - ◀ Needs: task 3

### 4. 👀 Judge the figure

- [ ] **Open R6 and its sidecar, and say PASS or MISS against the definition of done**
  - ◀ Needs: task 5
