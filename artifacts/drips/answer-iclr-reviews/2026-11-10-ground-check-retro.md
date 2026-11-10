# Retrospective: route step 1, check the ground

Walk: [the walk map](EXECUTE_WALK.md). Result file: [the ground check](ground-check.md). Board after the step: [board picture](../../../report/paper-review/board-2026-11-10-step-1.png).

## 1. The nodes

**The request.** Probe mscluster108, 111 and 110 with `nvidia-smi -L` and a torch CUDA test under the right python.

**The problem.** The environment notes recorded 111 as unusable from torch (2026-09-05) and 112 as faulted (2026-10-13), and Slurm cannot show GPU health.

**What was done.** `scripts/cluster.sh sh` from mscluster84, `ssh` to each node, `nvidia-smi -L` and `--query-gpu`, a 1000 by 1000 ones matmul on every card under `co3` (108) or `co3_bw` (110, 111); `sinfo -N` and `squeue -w` for state and owner.

**Where the results live.** [The node table](../../../report/paper-review/prze-10-compute.md#the-nodes-probed-2026-11-10).

**Before and after.** Before: 2 of 5 `biggpu` nodes probed. After: 5 of 5; 108 and 111 idle and computing (matmul sum 1e9 per card); 110 computing but held by avogt job 83315 until about 2026-11-12 11:30.

**Depends on.** Nothing earlier; piece 2 used the same cluster path.

## 2. What the paper's adapter was trained on

**The request.** Name the pair pool, the checkpoint and the noise steps behind the paper's adapter.

**The problem.** A checkpoint folder named `ep600` against the paper's 1,000 epochs, an unstated choice of "ten steps", and Figure 5 rows outside every held-out list.

**What was done.** Read each Figure 5 "ours" tile's record under `/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/`, then each run's `config.json`, `pair_pool.json`, `dataset_meta.json`, `verdict.json`; the trainer's step filter at `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py` line 532; the cache's `meta.json` and one `step_000.pt`.

**Where the results live.** [The pair lists page](../../../report/paper-review/prze-3a-pair-lists.md#the-adapter-behind-the-paper), [the early-steps page](../../../report/paper-review/prze-11-early-training.md#which-steps-the-papers-adapter-trained-on), section 2 of [the ground check](ground-check.md).

**Before and after.** Before: one candidate checkpoint (`ep600`, v54 pool, 24 pairs). After: three runs behind Figure 5, the adapter of record v58-03-self step 50,000 (48 training pairs, 1,000 epochs, step indices 0 to 9, timesteps 981 to 801); the run held out five pairs, the pool file now four; the `ep600` folders sit behind no figure.

**Depends on.** Piece 1's cluster path.

## 3. The local tex against the submitted PDF

**The request.** List every difference between the submitted paper and the local tex, with page numbers.

**The problem.** Two local sources, no Overleaf access at first.

**What was done.** `pdfinfo`, `pdftotext`, `git log` and `git status` in the clone; heading lists; a sentence matcher ([the script](compare_pdf_to_tex.py)) run against both local copies, then against the Overleaf export `_ICLR2027__LoRA_Plurality.zip` once you supplied it; exact `grep` of every flagged sentence.

**Where the results live.** Section 3 of [the ground check](ground-check.md); [the sentences that differ from the clone](pdf-vs-overleaf-clone-sentences.tsv); [the PRZE-5 page](../../../report/paper-review/prze-5-paper-revised.md).

**Before and after.** Before: "section titles differ" with no source settled. After: the export's 14 headings equal the PDF's and every flagged sentence is in it; the clone (last pulled 2026-09-26 08:17 UTC) and `paper/iclr/` (2026-09-21) are both older. The PDF describes the adapter as cross-attention only (pages 1 and 7) and as trained at every denoising step (page 1); the run used self- and cross-attention on steps 0 to 9.

**Depends on.** Piece 2's adapter config, which exposed the page 1 and 7 misstatements.

## The stop point

Bar: all three objectives answered with evidence, judged from the ground check. Time box: 90 minutes. Used: 15:06 to 15:32, 26 minutes. The bar came first: PASS. Way on: ship.

## Decisions

| Decision | Chosen | Options | Who chose | Why |
|---|---|---|---|---|
| Overleaf access | carry on without it, ask at piece 3 | ask at the open | you, `defaults` | no token in hand |
| Node plan | 107, 108, 111, 112; 110 busy until 2026-11-12 | your own order | you, `yes` | 110 carries another user's job |
| Adapter of record | v58-03-self step 50,000, all three Figure 5 runs listed | one run only | you | one recipe for scores, the figure's mix kept visible |
| PRZE-11 training test | all-50-steps adapter against the paper's steps-0-to-9 adapter | keep as written | you, `all 50` | the paper's adapter already is early-only |
| Source to edit | the live Overleaf project via the export | the clone, `paper/iclr/` | evidence | headings and sentences match only the export |

## Dead ends and not done

- `environment/hpc/nodes.md` still says mscluster111 cannot run torch; the probe says it can. Not written: this walk could write only under `report/paper-review/` and the walk folder.
- The clone at `paper/overleaf-iclr/` holds uncommitted edits made before the final Overleaf restructure; reconcile before using it.
- The PDF was not recompiled from the export (no system LaTeX), so no byte or page-count match is claimed.
- Nothing was committed: the brief did not ask for commits, and parallel sessions edit this branch.
- `_ICLR2027__LoRA_Plurality.zip` (118 MB) sits untracked at the repository root.

## The family table

| What | Id | State |
|---|---|---|
| this walk's session | 9b0c49df-d0f0-42b3-8a7c-79583537f555 | closed |
| runs, threads, handoffs | none | |

---

Validation: every repository path above was listed on disk on 2026-11-10; numbers were copied from the command outputs of this walk.
