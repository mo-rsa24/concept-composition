# Ground check before any review run

Three facts every later run for submission 56173 depends on: which GPU nodes work, what the paper's adapter was trained on, and how the local manuscript source differs from the submitted PDF.

## 1. The nodes

Answered. All five `biggpu` nodes were probed on 2026-11-10 with `nvidia-smi -L` and a torch matmul under the node's python; 107, 108, 111 and 112 are usable and 110 is held by another user's job until about 2026-11-12 11:30. The table and the commands are in [the compute plan page](../../../../report/paper-review/prze-10-compute.md#the-nodes-probed-2026-11-10).

## 2. What the paper's adapter was trained on

Answered. The adapter of record is v58-03-self at step 50,000: 48 training pairs from `pair_pool_v58.yaml`, 182 cells over 23 seeds (none from 9 to 16), 1,000 epochs of 50 optimiser steps, trained only on the ten noisiest of 50 DDIM steps (timesteps 981 to 801). Figure 5's "ours" column mixes it with two other runs, and the run held out a fifth pair (bison×horse) that the pool file has since dropped. Table, paths and checks in [the pair lists page](../../../../report/paper-review/prze-3a-pair-lists.md#the-adapter-behind-the-paper) and [the early-steps page](../../../../report/paper-review/prze-11-early-training.md#which-steps-the-papers-adapter-trained-on).

**One training state, field by field** (`/datasets/mmolefe/poe_repair_min/outputs/training_cache/train/a_badger__x__a_magpie/seed_2/residuals/step_000.pt`, read 2026-11-10):

| Field | Shape, dtype | Norm | What it is |
|---|---|---|---|
| `step_index` | 0 | | position in the 50-step schedule; 0 is the first and noisiest |
| `timestep` | 981 | | the model's noise level at that position |
| `x_t` | 1×4×128×128, fp16 | 255.2 | the latent at this step, the same for every pair on seed 2 |
| `eps_a_raw` | 1×4×128×128, fp16 | 255.9 | SDXL's noise prediction for "a badger" |
| `eps_b_raw` | 1×4×128×128, fp16 | 255.9 | the prediction for "a magpie" |
| `eps_j_raw` | 1×4×128×128, fp16 | 255.9 | the prediction for the joint prompt "a badger and a magpie" |
| `eps_uncond` | 1×4×128×128, fp16 | 255.9 | the prediction for the empty prompt |

The cell holds 50 such files; with `train_step_range [0, 10)` the trainer reads `step_000.pt` to `step_009.pt`. The target the loss compares against is built from the joint and per-concept predictions; that build lives in the trainer and is not restated here.

## 3. The local tex against the submitted PDF

Answered. **The source to edit is the live Overleaf project**, exported on 2026-11-10 as `_ICLR2027__LoRA_Plurality.zip` at the repository root (62 files; `iclr2027_conference.tex` 89,856 bytes). It is the submitted source:

- its 14 `\section` and `\subsection` titles are the PDF's, in the PDF's order (Introduction; Background and Related Work with 2.1 and 2.2; The Measured Plurality Term; Learning the Plurality Term with Results; Discussion; AI Declaration; appendix A Implementation details and B Training and inference derivation with B.1 and B.2);
- on pages 1 to 9 its sentence match sits at the matcher's noise floor (27 of 66, 27 of 60 and 23 of 83 PDF sentences below 0.9), and every PDF sentence flagged as absent on pages 7 to 9, 12 and 13 was found in it by exact `grep` (figure captions, which the matcher strips, and sentences split by inline maths); pages 10 and 11 are the reference list, which the tex takes from the `.bib`;
- in the other direction, ten zip sentences the reverse matcher flagged were each found in the PDF text by exact phrase, so no edit made after submission was found.

The PDF was not recompiled from the zip (no system LaTeX here), so a byte-level or page-count match is not claimed.

`56173_Restoring_plurality_in_c.pdf` is 13 pages, built 2026-09-26 13:41 SAST (`pdfinfo`). **Neither local copy is that source**, so neither is edited:

| Source | What it is | Last change | Matches the PDF? |
|---|---|---|---|
| `_ICLR2027__LoRA_Plurality.zip`, `iclr2027_conference.tex` | export of the live Overleaf project | exported 2026-11-10 13:28 | yes, as above |
| `paper/overleaf-iclr/iclr2027_conference.tex` | a git clone of the Overleaf project (`origin` git.overleaf.com/6a86dc92ca15fe47d75b86f6), with uncommitted local edits to the tex, the PDF and four figures | last commit 2026-09-26 08:17 UTC, working file 2026-09-26 12:09 SAST | no: headings and Section 4 differ |
| `paper/iclr/iclr2027_conference.tex` | the repository's own draft, with the appendix all commented-out template placeholders | last commit 2026-09-21 (5832a806) | no: 101 of 101 compared sentences differ |

The PDF was built about an hour and a half after the clone's working file was last saved, so its final edits were made on Overleaf and never pulled. The clone's uncommitted edits have to be reconciled with the Overleaf version before the clone is used again.

**Headings, PDF page by page, against the clone:**

| PDF page | Submitted PDF | Clone (`paper/overleaf-iclr`) |
|---|---|---|
| 1 | 1 Introduction | Introduction |
| 3 | 2 Background and Related Work, 2.1 Latent Diffusion and Classifier-free Guidance, 2.2 Compositional Latent Diffusion | the same |
| 4 | 3 The Measured Plurality Term | 3 Measuring and Restoring Plurality in Product-of-Experts Composition, with subsections The Measured Plurality Term, Restoring Plurality with the Measured Correction, Learning the Plurality Term |
| 7 | 4 Learning the Plurality Term, 4.1 Results | 4 Results, with Composition on held-out pairs, Sensitivity and limits of the correction |
| 9 | 5 Discussion | no Discussion section |
| 10 | 6 AI Declaration | none; the clone ends its main text with Conclusion |
| 12 | A Implementation Details, B Training and Inference Derivation, B.1, B.2 | the same names |

**Sentence by sentence.** Each PDF sentence of eight words or more was matched to its closest sentence in the clone's tex after stripping LaTeX ([the comparison script](compare_pdf_to_tex.py); [every differing sentence with its page](pdf-vs-overleaf-clone-sentences.tsv)). A sentence counts as differing below a 0.9 match ratio and as absent below 0.6.

| PDF pages | Sentences | Differing | Absent from the clone |
|---|---|---|---|
| 1 to 3 (taken as unchanged, the noise floor) | 66 | 27 | 19 |
| 7 to 9 and 12 (Section 4, its results, Appendix A) | 101 | 81 | 67 |
| 10 to 13 | 85 | 71 | 63 |

The matcher's noise floor is high (41% on pages that should barely differ), so the counts show where text moved rather than counting edits one by one. Section 4 and the appendix sit at 80%, well above that floor.

**Two places where the PDF misstates the adapter of record** (v58-03-self, `config.json`):

| PDF page | The PDF says | The run did |
|---|---|---|
| 1, abstract | "we train a low-rank adapter on the cross-attention layers of Stable Diffusion XL to predict the residual at every denoising step" | LoRA on `attn1` and `attn2` (self- and cross-attention) q/k/v; trained on step indices 0 to 9 only |
| 7, Section 4 | "the query, key, and value projections of every cross-attention of layer of Stable Diffusion XL" | the same `attn1` and `attn2` targets |
