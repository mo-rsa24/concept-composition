# 🗂️ Regroup poe_repair_min into project folders

`poe_repair_min` holds five research questions in one tree, so a result from one question sits
beside results from four others and nobody can find it. This plan moves every item into the
folder of the question it answers, copying the medical repo's shape
(`/home/molef/PhD/medical-imaging/compositional_diffusion`), so time can be allotted per project
and each project's board says where it stands. Where this repo sits among the rest of the PhD is
in `/home/molef/PhD/ORGANOGRAM.md`.

## Quick context

**The shape it ends in**

```
poe_repair_min/                      (renamed sdxl-composition when it moves under portfolios/)
├── README.md                        one row per project: its question, its status, its board
├── context/  environment/  runbook/                 shared, stay at the root
├── poe_repair/  pyproject.toml  tests/  scripts/local/   shared code, stays at the root
├── artifacts/                       only what several projects use (26 items) and _shared/, caches/
├── 1-lora-correction/               195 items: scopes 01 to 09, the ICLR paper, the adapter results
├── 2-classifier-guidance/           folded back from /home/molef/PhD/poe-classifier-guidance
├── 3-understanding-compositionality/   81 items: scope 10, the toy, dog and camel in the rain
├── 4-inclusive-captions/            12 items: scope 11
├── 5-text-embedding-third-expert/   32 items: the AND vector, PoE(cat, dog, x)
└── _archive/                        143 items cited by nothing, waiting for your yes
     each project: README.md, plans/, report/ (its board), artifacts/{results,drips,ideas,notes,scenes}/, src/
```

**Where every item goes** is [the inventory](regroup-into-project-folders-inventory.md): 506
items from all three checkouts, each with its project and the reason it was placed there. 15 items
leave for the medical portfolio. The move reads the inventory, so a correction there is a
correction to the move.

**What could go wrong**

- **Results are gitignored**, so each checkout keeps its own copy on disk, and at least 30 results
  folders exist in only one of the three. The move gathers them from all three first.
- **Scripts find the repo root by counting folders up** from their own file. A script that moves
  one folder deeper finds the wrong root, so every moved script is rechecked.
- **Other sessions write into these folders.** Session `41c11448` was still writing the toy files
  on 2026-11-13. Live sessions are told the new paths or finish first.
- **`poe_repair/paths.py` hard-codes** `artifacts/_shared/cross_pair_pool_configs/`,
  `artifacts/caches/` and the three `artifacts/rung*` folders. Those stay where they are.

## Instructions

*For you, before any task runs.*

### 1. ⚖️ Check the inventory

- [ ] **1.1 Read the counts table** at the top of [the inventory](regroup-into-project-folders-inventory.md).
- [ ] **1.2 Settle the one undecided row:** `scripts/xhat0_readback.py`, cited equally by
      compositionality (6) and the text embedding (6).
- [ ] **1.3 Scan the rows marked `judged by name`** (51 items nothing cites, placed from their
      names) and change any that are wrong.
- [ ] **1.4 Scan the archive list** (141 scripts and 2 walks cited by nothing). Anything you still
      use gets a project instead.
- [ ] **1.5 Decide the dog-in-the-rain and camel-in-the-rain results.** They are placed in
      compositionality; the text embedding is the other candidate.

✅ Done when the inventory has no `?` rows and you have said the archive list may move.

## Tasks

*For Claude, after instruction 1.*

### 2. 🔀 Bring the three checkouts onto one branch

- [ ] **2.1 Reach a quiet window.** Run `python3 scripts/quiet_window_check.py --me <your
      session id>`; it prints `CLEAR` only when no other session wrote in the last hour and no
      file is uncommitted in any checkout. On 2026-11-13 at 07:05 it printed `WAIT: 3 live
      session(s), 334 uncommitted file(s)`. What stands in the way:
  - [ ] **The live sessions finish or pause:** `17925940` (main checkout, the ICLR review
        answers), `41c11448` (the toy and the caption model), `96716791` (a walk in
        `how-much-co-occurrence-the-product-needs`).
  - [ ] **The main checkout's ICLR review work is committed by its sessions:** `a40d70e8` (14
        files under `paper/iclr-56173`, `poe_repair/experiments`, `report/paper-review`),
        `1a9eecdc` (4), and five sessions with 1 or 2 files each.
  - [ ] **The main checkout's 204 walk files that no recent session names** under
        `artifacts/drips/` are committed or put on the archive list.
  - [ ] **`.playwright-mcp/` in the main checkout** (376 MB of browser-tool screenshots, 365
        folders, not in `.gitignore`) goes into `.gitignore` and onto the archive list.
  - [ ] **This worktree's 74 files are committed**, 22 of them from the classifier move.
- [ ] **2.2 Merge the open branches** (`exec/how-much-co-occurrence-the-product-needs`,
      `paper/answer-k4nd-uhz6`, `exec/correction-loss-01`, and the non-classifier files of
      `exec/can-a-classifier-supply-the-missing-term`) into one branch, with `/reconcile-machines`
      for the cluster side.
- [ ] **2.3 Gather the gitignored folders** (`artifacts/results/` and anything else ignored) from
      all three checkouts into the main checkout. A folder in two checkouts is compared file by
      file, and the newer copy is kept.
  - Check: every results folder in the inventory exists once, in the main checkout.

### 3. 📁 Create the project folders and move the items

- [ ] **3.1 Create the five project folders**, each with `README.md`, `plans/`, `report/`,
      `artifacts/` and `src/`.
- [ ] **3.2 Move every row of the inventory to its project**, with `git mv` for tracked items and
      `mv` for ignored ones. Scope 10's plan `today-a-lora-on-the-text-encoders.md` goes to the
      text embedding, not with the rest of scope 10.
- [ ] **3.3 Move the archive list into `_archive/`**, keeping the same sub-paths.
  - Check: a script lists every path in the inventory and finds each one at its new home and
    nowhere else.

### 4. 🔧 Repair what the move broke

- [ ] **4.1 Rewrite every moved script's repo-root lookup** and every import of moved experiment
      code, then compile every Python file and check every import resolves.
- [ ] **4.2 Repoint every markdown link** with a link checker over the whole repo, until it
      reports zero broken links.
- [ ] **4.3 Write the moves into `RENAMES.md`** and give each project's `README.md` its question,
      its status and its board.
  - Check: the link checker reports 0 broken links, the compile reports 0 failures, and the import
    check reports 0 unresolved imports.

### 5. 🧩 Fold classifier guidance back in, and send the medical items out

- [ ] **5.1 Move `/home/molef/PhD/poe-classifier-guidance` in as `2-classifier-guidance/`**, with
      its commit noted in `RENAMES.md`.
- [ ] **5.2 Move the 15 medical items** into `/home/molef/PhD/medical-imaging/compositional_diffusion`.
- [ ] **5.3 Update `/home/molef/PhD/ORGANOGRAM.md`** and the memory entry that points at it.

### 6. 🖥️ Rebuild the cluster checkout

- [ ] **6.1 Fast-forward the cluster checkout** through `scripts/cluster.sh` and re-run one
      canary per project there, where torch exists, so each project's imports are proven live.

## Next step

Instruction 1.1: read the counts table in the inventory.
