# 🧭 Regroup the core repo into project folders

`poe_repair_min` becomes the SDXL-composition portfolio of `/home/molef/PhD/ORGANOGRAM.md`: one
repo, one folder per project, each with its own `plans/`, `report/` with its outcome board, and
`artifacts/`, sharing `context/`, `environment/`, `runbook/` and the `poe_repair` package at the
root. The medical repo, `medical-imaging/compositional_diffusion`, already has this shape and is
the model to copy. The core `RENAMES.md` lists every old path.

| Project | Its folder | What goes into it | Sessions that worked on it | Status |
|---|---|---|---|---|
| LoRA correction | not created | scopes 01 to 09 and the ICLR paper | most of the 61 core sessions | open |
| Classifier guidance | `/home/molef/PhD/poe-classifier-guidance`, a separate repo for now | plans 15, 17, 18, the route, the board, the walk maps, the presence and dog heads, 21 scripts | `cb5e6071` (the step 1 read), and `6dac739e` where the idea started | moved out 2026-11-13; folds back in as a folder |
| Understanding compositionality | not created | scope 10 ([what PoE loses when it composes](10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md)), the simple-distributions walk and its toy (`toy/`, the [toy card](../artifacts/results/how-much-co-occurrence-the-product-needs/README.md), `how-much-co-occurrence-the-product-needs`), the [dog-in-the-rain](../artifacts/results/does-a-dog-in-the-rain-come-out-wet/README.md) and [camel-in-the-rain results](../artifacts/results/does-a-camel-in-the-rain-come-out-wet/README.md), and the work defining compositional terms so they can be learned | `41c11448` (live 2026-11-13), `553b6a70`, part of `e7a90753` | open |
| Inclusive captions | not created | scope 11, its route, the pilot walk, the photo pool and training set, `poe_repair/experiments/inclusive_captions/` | not yet counted | open |
| Text embedding as a third expert, PoE(cat, dog, x) | not created | the AND-vector walks, results and `and_vector_*` scripts, the [text-space finding](../report/can-a-text-space-third-expert-restore-composition.md), plan 10's [text-encoder LoRA plan](10-what-poe-loses-when-it-composes/today-a-lora-on-the-text-encoders.md) | `6dac739e` | open |
| Medical x-rays | leaves for the medical portfolio | the x-ray, silicotb and two-disease routes, their walks and idea folders, `overlap-regimes-on-real-films` | `b4e1ea0a`, most of `e7a90753` | open |

## Before the next move

- [x] **Write the regrouping as its own plan**, listing every scope, walk, idea and results folder
      against the project it goes to: [the regrouping plan](regroup-into-project-folders.md) and
      its inventory of 506 items.
- [ ] **Decide which results go to compositionality and which to the text embedding.** The
      dog-in-the-rain and camel-in-the-rain results and the `a-dog-and-rain` walk sit between the
      two; this table puts them in compositionality.
- [ ] **Wait for `41c11448` to close, or give it the new paths**, before compositionality moves,
      because it writes the toy card and scope 10.
- [ ] **Bring the rest of branch `exec/can-a-classifier-supply-the-missing-term` into the core
      repo:** the [regional composer](10-what-poe-loses-when-it-composes/is-the-layout-the-missing-piece.md#3--the-regional-composer) for plan 10's layout question, the silicotb route files, an
      `environment/hpc/nodes.md` edit and two walk maps.
- [ ] **Create the GitHub remote and the cluster checkout** for whatever the classifier work
      becomes, after a yes. The real import check runs there, because the laptop's Python has no
      torch.
