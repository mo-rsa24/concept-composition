# 🔬 Why the second animal takes on the first animal's looks

The adapter of record (v58-03-self, step 50,000) now draws two animals, but the second often wears
the first one's species (lion and meerkat at seed 1 gives a lion and a lion cub). This plan tests
two causes, cheapest first, on 4 training pairs at unseen seeds 9 to 16, and proposes a retrain
only if neither fix reaches the bar.

## Quick context

**Cause 1: the adapter runs where it never trained.** It trained on DDIM steps 0 to 9 only
(`config.json` `train_step_range [0, 10]`), and every earlier render left it on for 25 or 50 steps.
The fix is switching it off after step 9 (`--window 10`).

**Cause 2: self-attention spreads one animal's look onto the other's body.** v58-03-self adapts
self- and cross-attention; v58-01-plain adapts cross-attention only, with the same recipe. The fix
is the cross-attention-only adapter, compared at step 47,500 where both have a checkpoint.

**The ladder, per tile.** 0 one animal · 1 a hybrid of both · 2 two bodies, one with the other's
species look · 3 two animals, each its own species · destroyed for a broken image. Labelled by eye
from full-resolution tiles.

## The bar

PASS: a setting reaches level 3 on at least 6 more of the 32 tiles than v58-03-self at step 50,000
with the adapter on all 50 steps, with no more than 2 tiles dropping below level 2 against that
same baseline. Judged from `labels.csv` and the sheets. Time box: 2 sittings, about 3 hours.

## Environment Facts This Plan Depends On

- mscluster112, one RTX PRO 6000 Blackwell, `co3_bw` python at
  `/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python` (prze-10-compute.md, 2026-11-10).
- Renders go to `/datasets/mmolefe/poe_repair_min/outputs/species-fidelity/` only.
- The cluster checkout is at the laptop's commit and `figure_candidates.py` matches by checksum.
- The renderer reads starting noise from `data/pilot/seed_N/a_cat__x__a_dog`, which holds no seed
  from 9 to 16, so each seed's noise is built from the seed number, shared by every setting.

## Tasks

For Claude to execute.

### 1. Render

- [x] ✅ **1.1 lion × meerkat, v58-03-self step 50,000, windows 10, 25, 50, seeds 9 to 16 (24 tiles)** ✓ verified 2026-11-10: `=== exit 0`, 24 tiles, the renderer's adapter-step count check passed on every tile; 29 s (w10), 46 s (w25), about 46 s (w50) per tile on an RTX 8000, 4.5 min model load
  - Launcher: `1-lora-correction/artifacts/results/why-the-second-animal-loses-its-species/render.sbatch`, tag `self50k`
  - Expect: log shows `n_matched` from 420 layers, `exit 0`, 24 files `ours_self50k_w{10,25,50}.png`
  - Time the job; give the total for the rest before going on
- [x] ✅ **1.2 the other three pairs, same settings (72 tiles)** ✓ verified 2026-11-10: `=== exit 0` on both processes, mscluster107 cards 0 and 1
- [x] ✅ **1.3 the plain product for all 4 pairs, in its own process (32 tiles, `poe.png`)** ✓ verified 2026-11-10: `=== exit 0`, 128 tiles on disk after 1.1 to 1.3
- [ ] **1.4 windows 15 and 35 for all 4 pairs (64 tiles), to place where the lion is lost and where the second animal turns into the first**
  - Expect: 64 files `ours_self50k_w15.png`, `ours_self50k_w35.png`; `=== exit 0` in `logs/sf-w1535-card{0,1}-107.out`
- [x] ✅ **1.5 Count the adapter layers in each checkpoint** ✓ verified 2026-11-10: v58-03-self 420 self-attention and 420 cross-attention weight tensors at both 47,500 and 50,000; v58-01-plain 0 and 420

▶ Next: 2.1

### 2. Label and compare

- [ ] **2.1 Label the 96 window tiles plus 32 plain tiles; pick the best window by level-3 count**
- [ ] **2.2 v58-01-plain and v58-03-self at step 47,500, best window (64 tiles), tags `plain47k`, `self47k`**
- [ ] **2.3 Label those 64**

▶ Next: 3.1

### 3. Write up

- [ ] **3.1 `labels.csv`, contact sheets, counts per setting, the count command, README with the verdict**
- [ ] **3.2 New rows in "What we know so far" in `plans/route-answer-the-iclr-reviews.md`**
- [ ] **3.3 If MISS: one retrain proposal, the one flag that changes and its cost in hours; stop for a yes**

## Instructions

For you.

### A. Look at the first pair's row

- [ ] **A.1 Open the lion × meerkat sheet and say whether the tiles look rendered as intended** ◀ Needs: 1.1

### B. Check the label table

- [ ] **B.1 Read the label table against the sheets and overrule any label you disagree with** ◀ Needs: 2.1

### C. Approve or decline the retrain proposal, if one is written

- [ ] **C.1 Yes or no on the proposal** ◀ Needs: 3.3
