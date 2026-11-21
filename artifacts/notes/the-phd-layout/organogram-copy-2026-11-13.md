# 🗂️ The PhD, sorted by kind of work

Every folder under `/home/molef/PhD`, what kind of work it is, and where it belongs. Read this to
decide where time goes and where a new piece of work lives. Folders stay where they are until each
is moved on purpose, one at a time, because a move breaks cluster checkouts, worktrees and absolute
paths in plans and memory.

<a href="artifacts/notes/the-phd-layout/the-phd-layout.png"><img src="concept-composition/artifacts/notes/the-phd-layout/the-phd-layout.png" width="800" alt="The PhD folder sorted by kind: two research portfolios with their projects, writing, replications with the published-methods hub, cluster bring-up, and the archive"></a>

## The shape

```
/home/molef/PhD
├── portfolios/      one repo per family of questions; context/, environment/, runbook/ shared at its root
│   ├── medical-imaging/      chest x-rays: projects 0, A1, A2, B1, C1, silicotb, overlap regimes
│   └── concept-composition/  composing two concepts (cat and dog, cat and rain), on SDXL so far:
│                             LoRA correction, classifier guidance, understanding compositionality,
│                             inclusive captions, text embedding as a third expert
│        └── <project>/       one question: its own plans/, report/ with its board, artifacts/
├── papers/          one repo per paper being written: the LaTeX and the figures it prints
├── replications/    other people's code, run or reproduced
├── admin/           the proposal, cluster bring-up
└── _archive/        kept until a deletion pass you approve
```

## Rules

**A project is one question with a bar and a board.** It lives as a folder inside its portfolio,
with its own `plans/`, `report/` and `artifacts/`. Context, environment and runbook are shared at
the portfolio root and never copied into a project.

**An idea is not a project until it has a question and a bar.** Until then it is an idea map in
its portfolio's `artifacts/ideas/`, and at each review it is promoted to a project, parked, or
dropped.

**A session works in one project.** Its working folder is that project's folder, so overlaps show
up as sessions in the wrong place rather than as files in the wrong place.

**Results are found by reading down, never by browsing.** A project's `artifacts/results/` is
grouped by the question each result answers, two levels at most (`results/<group>/<subgroup>/
<result>/`), with a `README.md` at every level listing each child by its card's title, its first
sentence and a thumbnail. In `concept-composition`, `scripts/regroup/results_index.py` writes those
indexes from `scripts/regroup/result-groups.txt`; a project's own `README.md` links its results
index.

**Every result folder is cited by a finding or a card in its project.** In `concept-composition`,
`scripts/regroup/uncited_results.py` lists the ones that are not. One that is not goes on
that project's archive list. Listed folders move to `_archive/` after you have looked at the list,
and `_archive/` is deleted only in a later pass you approve.

## Every folder

Sessions are Claude Code sessions whose working folder is that folder, counted on 2026-11-13.

### Portfolios

| Folder | What it is | Becomes | Sessions | Cluster path |
|---|---|---|---|---|
| `concept-composition` | composing two concepts in one image, on SDXL so far, in five project folders; its data folder on the cluster is still `/datasets/mmolefe/poe_repair_min` and its Python package is `poe_repair` | `portfolios/concept-composition` | 61 (29 since 2026-11-06), last 2026-11-13 | `Playground/PhD/concept-composition`, with the symlink `Playground/PhD/poe_repair_min` at its old name; data in `/datasets/mmolefe/poe_repair_min`, which also holds two worktrees where the review jobs ran (`outputs/species-fidelity/code-idea2`, `worktrees/prze2`); to retire: `Playground/PhD/poe_repair_min-classifier-guided`, `Playground/PhD/poe_repair_min-clip-free`, `Playground/PhD/poe_repair_min-dog-and-rain`, `Playground/PhD/poe_repair_min-inclusive-captions`, `Playground/PhD/poe_repair_min-r11`; also `Playground/PhD/simple-distributions-walk` (the understanding project's rebuild walk), `Playground/PhD/joint-vs-poe-grid` (joint prompt against PoE grids on CLEVR, SD and SD3.5) and `/datasets/mmolefe/sdxl_shapes` (SDXL shape renders with their code) |
| `concept-composition/2-classifier-guidance` | classifier guidance as a stand-in for the LoRA; the separate repo is archived | a project folder in `concept-composition` | 0 | inside the checkout |
| `medical-imaging/compositional_diffusion` | chest x-rays: two findings in one film, a shortcut through a device, a detector weakened by a second condition | `portfolios/medical-imaging`; gains the silicotb, overlap-regime and two-disease work now in `concept-composition` | 11, last 2026-10-20 | `Playground/PhD/medical-imaging` (holds the checkout); data in `/datasets/mmolefe/compositional_diffusion` and `/datasets/mmolefe/shortcut_companion`; also `Playground/PhD/xray_shortcut_companion` (project B's chest-tube annotations); the SepVAE line, which stays important: `Playground/PhD/baselineSepVAE` (1,002 GB on `/home-mscluster`), `Playground/PhD/miniSepVAE`, `Playground/PhD/sepVAE`, `Playground/PhD/reference_sepvae`, to be gathered as branches of one SepVAE repository that record its evolution; also `Playground/PhD/igm_cxr` (chest x-ray generation code) and `/datasets/mmolefe/comorbid-classifier-audit` (a comorbid classifier audit's outputs) |

### Papers being written

| Folder | What it is | Becomes | Cluster path |
|---|---|---|---|
| `Paper3` | "PoE Composition of Correlated Chest Diseases", 4.2 GB, the medical portfolio's earlier paper; kept 2026-11-13 | `papers/` | `Playground/PhD/Paper3`; a second clone at `/datasets/mmolefe/paper3-hippocampus`; data in `/datasets/mmolefe/data` and `/datasets/mmolefe/checkpoints` (per `/datasets/mmolefe/README.md`) |
| `concept-composition/paper` | the ICLR paper (scope 07 of the LoRA correction project) | stays in `concept-composition` until it is split into `papers/` | inside the checkout; its figure and review scratch sits loose in the cluster home folder (below) |

`eccv2026`, `Paper2` and `neurips2026` are archived (below); every commit of each is on GitHub.

### Replications

Other people's code, cloned or reproduced, in `/home/molef/PhD/replications/`. Symlinks at the old top-level names keep older paths working.

| Folder | What it is | Cluster path |
|---|---|---|
| `replications/published-methods-on-our-pairs` | the hub for running fourteen published composition and steering methods on cat × dog, lion × meerkat and typewriter × cactus, beside the plain product and our adapter; each paper's code is cloned beside it; see its README | not on the cluster yet |
| `replications/reduce-reuse-recycle` | Du et al.'s code, branch `walk`; holds this project's two walks in `artifacts/drips/`; pushed to the fork github.com/mo-rsa24/reduce_reuse_recycle, branch `walk`; `origin` stays the authors' | `~/PhD` (holds the checkout); environments in `/datasets/mmolefe/reduce-reuse-recycle` |
| `replications/original-superdiff` | the superposition (SuperDiff) paper's code | `Playground/PhD/original-superdiff` |
| `replications/latent-diffusion` | CompVis latent diffusion, branch `grayscale` | `Playground/PhD/compvis` (same remote, mo-rsa24/compvis) |
| `replications/projective-composition` | projective composition, with its own environment and runbook | `Playground/PhD/projective-composition`; outputs in `/datasets/mmolefe/projective-composition` |
| `medical-imaging/compositional_diffusion/b-shortcut-through-a-device/b1-drain-shortcut-in-pneumothorax/calibration` | github.com/Amineki6/calibration, cloned for B1; ignored by the medical repo | inside the medical checkout |

### Admin

| Folder | What it is | Cluster path |
|---|---|---|
| `Proposal` | the PhD proposal, branch `dev_comments_v1`; holds the chaptered third version (last changed 2026-04-28) found in eccv2026; the cluster's newer files (stage 4 and the 2026-04 to 2026-05 edits) are on branch `from-cluster-2026-11-13`, and one file still waits on a merge by hand | none of its own; its cluster copy is `Playground/PhD/baselineSepVAE/proposal/` |
| `cluster-bringup` | cluster bring-up, experiment re-entry, and the route that sorts the cluster to match this page | `~/regroup-tools` (the route's scripts and job logs), `/datasets/mmolefe/README.md` (the storage rules), `/datasets/mmolefe/offload` (a 2026-09-21 survey of what to move off `/home-mscluster`) |

### Archived on 2026-11-13

In `/home/molef/PhD/_archive/`, listed in its `README.md`: `eccv2026`, `Paper2`, `neurips2026`,
`Paper1`, `Test`, `eccv`, `xray`, a git repair backup, three loose root files,
the separate classifier repo, `.playwright-mcp/` and three large downloads from `concept-composition`,
and the retired worktree's scratch folder.

### On the cluster only

Everything on mscluster that has no row above, from the census of 2026-11-13. `Playground/PhD/`
means `/home-mscluster/mmolefe/Playground/PhD/` and `~/` means `/home-mscluster/mmolefe/`. Every
entry with its size, git remote and evidence is in
[the cluster folder listing](cluster-bringup/report/cluster-organogram/cluster-folders.md), and its
check script prints any cluster entry this page does not name. The names below are the old paths,
and each still works: since 2026-11-13 the `/datasets/mmolefe` entries live in `raw/`,
`projects/<project>/` and `_archive/` there, and the per-idea copies in `Playground/PhD/_archive/`,
each with a symlink at its old name.

**Archive candidates.** The cluster twins of the laptop's archived repos: `Playground/PhD/eccv2026`
and `/datasets/mmolefe/eccv2026`, `Playground/PhD/neurips2026` and `/datasets/mmolefe/neurips2026`,
and `Playground/PhD/phase2` (the same repo as `Paper2`). The 2025 SuperDiff work:
`Playground/PhD/superdiff-ldm` (414 GB on `/home-mscluster`), `Playground/PhD/superdiff-cond`,
`Playground/PhD/superdiff-single-conditional-ldm`, `Playground/PhD/super-diffusion`,
`Playground/PhD/Super-Position Of Medical Imaging Diffusion Models For Disease Discovery`. Early
experiments: `Playground/PhD/composable_diffusion_models`,
`Playground/PhD/composable_diffusion_models_2`, `Playground/PhD/simple-diffusion`,
`Playground/PhD/assumptions`, `/datasets/mmolefe/mini_sd3_project`. Empty or backup folders:
`Playground/PhD/todoist`, `~/demo`, `~/ckpts`, `~/results`, `~/Obsidian.pre-git-backup-2026-10-13`,
`~/poe_repair_min-prerewrite-20260916T064537Z.bundle`, `/datasets/mmolefe/backup`,
`/datasets/mmolefe/experiments`, `/datasets/mmolefe/mini_sd3_data`.

**Raw datasets**, one folder each, shared by every project: `/datasets/mmolefe/cat-and-dog-captions`,
`/datasets/mmolefe/celeba`, `/datasets/mmolefe/chess`, `/datasets/mmolefe/chestxdet`,
`/datasets/mmolefe/chexmask`, `/datasets/mmolefe/cleaned`, `/datasets/mmolefe/cleaned_2`,
`/datasets/mmolefe/coco2017`, `/datasets/mmolefe/metadata`, `/datasets/mmolefe/relaion2b-en`,
`/datasets/mmolefe/siim_acr`, `/datasets/mmolefe/taix_ray`, `/datasets/mmolefe/vinbigdata`,
`/datasets/mmolefe/vinbigdata-chest-xray-abnormalities-detection.zip`, and
`Playground/PhD/datasets` (cleaned chest x-rays, the same name as `/datasets/mmolefe/cleaned`).

**The ICLR paper's scratch, loose in the home folder**, belonging to `concept-composition`:
`~/tmp-fig5`, `~/tmp-fig5b`, `~/tmp-fig8`, `~/tmp-r11`, `~/tmp-r11b`, `~/tmp-eq`, `~/after_b.sh`,
`~/run_b_after_a.sh`, `~/list_runs.py`, `~/fetch_pixart.py`, `~/rejects_page.py`, `~/mask_left.png`,
`~/mask_right.png`, `~/scratch`, and `~/liver_cect_download.sh` and `~/liver_cect_gallery.py`
(loose copies of the tracked `scripts/download_liver_cect.sh` and `scripts/liver_cect_gallery.py`,
whose data sits in `/datasets/mmolefe/poe_repair_min/data/liver-cect`).

**Environments and tools**, which stay where they are: `~/miniconda3`, `~/miniforge3`, `~/envs-sd35`,
`~/libs`, `~/wandb`, `~/logs`, `/datasets/mmolefe/envs`, `/datasets/mmolefe/conda-pkgs`,
`/datasets/mmolefe/repos`. `~/Playground` is the root of the cluster code.

**Learning**, the cluster clones of the learning side: `~/goal-setting`, `~/Obsidian`,
`~/playground`, `~/Tutorials`, `~/Papers`, and `Playground/PhD/toy_ldm_solutions` (toy latent-diffusion
notebooks and tests).

### Learning material

Learning work lives in the learning tree, not here. `SDXL` (the module-by-module SDXL build) is
`/home/molef/goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/sources/sdxl-module-build/`,
and the velocity-field and normalizing-flow notebooks are
`/home/molef/goal-setting/learning/deep-learning/diffusion-models/03-score-sde/sources/velocity-fields-and-normalizing-flows/`.

## What to do, in order

- [x] **Move classifier guidance out of the core tree** (2026-11-13, to `poe-classifier-guidance`).
- [x] **Regroup `concept-composition` into project folders**, per
      `concept-composition/plans/split-into-idea-repos.md`, then fold `poe-classifier-guidance` back
      in as one of them.
- [x] **Move the medical routes and their walks** from `concept-composition` into the medical portfolio.
- [x] **Confirm the archive candidates** above and move the confirmed ones to `_archive/`.
- [x] **List each project's uncited results**, look at the lists, and archive what you approve (2026-11-13: one archived, four linked from findings).
