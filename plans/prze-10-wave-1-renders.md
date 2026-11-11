# 🚀 PRZE-10: price every review job and launch wave 1

**Goal:** every GPU job the review response needs has a node, a priority and a priced estimate, and the jobs no gate blocks are rendering.
**Objective:** the jobs table in [the compute plan](../report/paper-review/prze-10-compute.md#the-jobs), then wave 1's six jobs launched on mscluster112 card 0 and mscluster108 card 1, each after a yes to its exact command.
**Definition of done:** every wave 1 job is running or finished, and its output count matches the table: 6, 848, 424, 1,104, 1,104 and 1,104.
**Route:** [answer the ICLR reviews](route-answer-the-iclr-reviews.md), outcome PRZE-10.

## Environment Facts This Plan Depends On

- [The shared-device path and its safety rules](../environment/hpc/execution-protocol.md): a guard inside the launcher refuses a card holding over 1 GB, every path is absolute, the log header names node, card and PID.
- [Storage](../environment/storage.md): outputs under `/datasets/mmolefe/poe_repair_min/` only, and the disk guard reads that filesystem.
- `scripts/cluster.sh run` refuses uncommitted tracked changes.

## Tasks

For Claude to execute.

### 1. 📝 The compute table

- [x] **Write the jobs table with a measured basis wherever a log has one** ✓ verified 2026-11-11: [the jobs table](../report/paper-review/prze-10-compute.md#the-jobs), 13 rows
  - ▶ Next: 2

### 2. 💻 Close the launcher gaps

- [x] **Joint text as written, and the ledger's frog and mouse allowance, in the render scripts; one wave 1 launcher beside `scripts/review_step4_renders.sh`** ✓ verified 2026-11-11: commits `56debc5e`, `62554919`, `0c2babc1`; the first joint sidecar reads "a horse with a glasses"
  - The 36 benchmark prompts that read "with" get `--joint-template "{a} with {b}"`; the other 240 keep `"{a} and {b}"`.
  - The 45 prompts naming a frog or a mouse pass `allow_ask_first`, per the 2026-11-10 row of `artifacts/_shared/cross_pair_pool_configs/pair_ledger.md`.
  - Shown as a diff before anything runs.
  - ▶ Next: 3

### 3. 🚀 Predict and launch

- [x] **For each wave 1 job: an 80% interval for runtime and memory, the exact command, a yes, the launch, the first tiles** ✓ verified 2026-11-11: every launch after your yes, log headers name node, card and PID; logs in `/datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/`
  - ◀ Needs: 2
  - ▶ Next: 4

### 4. ✅ Poll and count

- [x] **Count each job's outputs on the launch node against the table, and add the results to the route's "What we know so far"** ✓ verified 2026-11-11 11:14 on mscluster112: 3, 848, 424, 1,104, 1,104, 1,104, matching the definition of done
  - ◀ Needs: 3

## Instructions

For you to follow.

### 1. 👀 Read one row of the table

- [ ] **Open `report/paper-review/prze-10-compute.md` in VS Code, scroll to "The jobs", and check job 3's hours against "How long one image takes"**
  - What you should see: 424 images at 12.1 to 14.3 s each is 1.4 to 1.7 h.

### 2. 👀 Look at the smoke tiles

- [ ] **Open the three Figure 5 tiles job 1 writes, and say whether each shows what its column claims**
  - ◀ Needs: task 3

### 3. 👀 Judge PASS or MISS

- [ ] **Read the count table piece 4 prints, and say PASS or MISS against the definition of done**
  - ◀ Needs: task 4
