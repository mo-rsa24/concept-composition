# Organizing the PhD folder: what stands, what waits

Hand-off from session 804ea312 (2026-11-13). Read this first when picking up portfolio-level work.
The index of every folder is [the PhD organogram](/home/molef/PhD/ORGANOGRAM.md).

## What stands

**The layout.** `/home/molef/PhD` holds two portfolios (`concept-composition`, which was
`poe_repair_min`, and `medical-imaging/compositional_diffusion`), `Paper3`, `Proposal`,
`cluster-bringup`, `replications/` and `_archive/`. Old names are symlinks to new ones:
`poe_repair_min`, and the four reproductions at the top level. The cluster checkout is
`Playground/PhD/concept-composition`, with a symlink at its old name.

**What keeps an old name on purpose.** `/datasets/mmolefe/poe_repair_min` and the Python package
`poe_repair`. Nothing should rename them.

**concept-composition.** Five project folders. Results are grouped by question, two levels at most,
with an index at every level (`scripts/regroup/results_index.py`). The root holds only the
tool-read markdown files, code and the repo's folders. Every result is cited somewhere
(`scripts/regroup/uncited_results.py` reports 0 of 63).

**medical-imaging.** Its root and B1 results are grouped into question folders, with its own copy
of the move tool in `scripts/indexes/`.

**replications/.** `latent-diffusion`, `original-superdiff`, `projective-composition`,
`reduce-reuse-recycle`, and the hub `published-methods-on-our-pairs`, whose README holds the
question and the fourteen papers.

**Proposal.** `dev_comments_v1` holds the cluster's work and the 23 stage-4 files the import
missed. Stage 4 builds with Tectonic (44 pages, matching the May build).

**The transfer number.** The compose-rate scorer no longer counts the trainer's comparison strips.
The pooled adapter composes 0.906 of held-out renders at step 60000, level with 0.909 on its
training pairs. F8a, F8b and the F6 subspace demo are rebuilt on the clean rates. See step 10's
review in `1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/review/`.

## What waits

| Where | What | Who |
|---|---|---|
| `replications/published-methods-on-our-pairs` | the route for the fourteen papers is running in its own session | that session |
| concept-composition, step 11 | the fifteen pooled runs need anchor pictures for 17 of 19 pairs first | a cluster session |
| concept-composition, step 10 | read the two direction measures for cat × dog and frog × toad | any session |
| `paper/iclr/DRAFT_MAP.md` | "the last scored checkpoint" should name step 60000 | you |
| Proposal stages 3, 3 v1, 3 v2 | may miss files the same way stage 4 did; compare with the cluster's `baselineSepVAE/proposal` and build each | any session |
| the timing page | parts 2 and 3 of the walkthrough, at `1-lora-correction/artifacts/scenes/when-the-correction-must-arrive/index.html` | you |
| `MASTER_PLAN.md`, `environment/known-failures.md` | uncommitted edits from sessions 96716791 and b3cb8672 | those sessions |

## One lesson for moving folders

A move tool that rewrites a folder's bare name also rewrites that word in ordinary sentences, so
"an exact toy check" became a path. Rewrite a path only where a slash follows the name, or through
a link that resolves into the moved folder. Rewrite links only in markdown and HTML, since Python
code contains `](` too.
