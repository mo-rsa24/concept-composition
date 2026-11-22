# Rivals: every paper the two papers are measured against

This table lists the 22 rival papers, which of our two papers each one rivals, and what it evaluates on. Paper 1
composes separately trained models at inference. Paper 2 restores plurality in one conditional model. The
[two-paper route](../../../../plans/route-two-papers.md) states both claims.

**Read state** says how far the numbers have been taken from the PDF. `listed` means the paper's identity and datasets
were checked on its pages, and no number has been copied yet. `numbers read` means its tables are in `numbers.csv`.
`figures read` means its qualitative figures are in `failures.md`. `could not read` means neither the PDF nor the arXiv
HTML page loaded.

**Datasets** come from the page named beside each one. The arXiv version is the one printed on the PDF's first page.

## Paper 1: experts that never met

| Rival | arXiv | PDF | Pages | Datasets and the page that states them | Read state |
|---|---|---|---|---|---|
| Liu | 2206.01714v6 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/compositional-visual-generation-with-composable-diffusion-models.pdf) | 30 | CLEVR, 30,000 images at 128 px (p8). Relational CLEVR, 50,000 images at 128 px (p9). FFHQ, 51,067 labelled images (p9). 5,000 generated images per test setting (p9) | `figures read` |
| Du | 2302.11552v6 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/compositional-generation-with-energy-based-diffusion-models-and-mcmc.pdf) | 23 | 2D densities (Table 1, p6). CLEVR cube positions (Table 2, p6). ImageNet 128 px (p19) | `figures read` |
| SuperDiff | 2412.17762v2 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/superposition-of-diffusion-models-using-the-ito-density-estimator.pdf) | 54 | CIFAR-10 split into two disjoint halves (Table 1, p8). Stable Diffusion concept pairs (Table 2, p8). Molecules and proteins (p9). Also rivals Paper 2 | `figures read` |
| FKC | 2503.02819v2 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/feynman-kac-correctors-in-diffusion.pdf) | 44 | ImageNet-512 with EDM2-XS, 10,000 images per score (Table 2, p7). LJ-13 (p8). Docking on 100 protein target pairs (Table 4, p9). SDXL on GenEval (Table A4, p38) | `figures read` |
| Bradley-Projective | 2502.04549v3 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/mechanisms-of-projective-composition-of-diffusion-models.pdf) | 40 | CLEVR with 1 to 5 objects, 100 samples per method counted by hand (p15, p17). SDXL examples (p1, p10) | `figures read` |
| Bradley-Local | 2509.16447v3 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/local-mechanisms-of-compositional-generalization-in-conditional-diffusion.pdf) | 36 | CLEVR by location and by colour, 1,024 samples per configuration (p3, p23). SDXL feature analysis, 24 prompts and 10 seeds (p10) | `figures read` |
| CoInD | 2503.01145v1 | [downloaded](pdfs/2503.01145.pdf) | 33 | Colored MNIST, Shapes3D and CelebA (p7). FID on CelebA (p7, p9) | `figures read` |
| MultiDiffusion | 2302.08113v1 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/region-and-binding-failures/multidiffusion-fusing-diffusion-paths-for-controlled-image-generation.pdf) | 16 | Panoramas, 2,000 samples for each of 8 text prompts (p6). COCO validation subset with region IoU (Table 2, p7) | `figures read` |
| PiCo | 2505.03203v1 | [downloaded](pdfs/2505.03203.pdf) | 16 | Attend-and-Excite and T2I-CompBench, 500 prompts and 10 seeds (p4). MS-COCO validation, 5,000 pairs for FID (p4, p9) | `figures read` |
| FreeFuse | 2510.23515v3 | [downloaded](pdfs/2510.23515.pdf) | 26 | The authors' own benchmark of 15 LoRAs, 470 images per method, on FLUX.1-dev (p9, p10) | `figures read` |
| DomainExpansion | 2605.23275v1 | [downloaded](pdfs/2605.23275.pdf) | 17 | Slakh2100 music, FAD on 128 samples (Table 1, p3). CLEVR, accuracy on 256 samples (Table 2, p3). Satellite images from a new Google Maps set, FID (Table 3, p4) | `figures read` |
| Sculpting | 2309.16115v1 | [downloaded](pdfs/2309.16115.pdf) | 36 | GFlowNets on a 32 by 32 grid (Figure 3, p12) and on molecules (p2). Colored MNIST with diffusion models (Figure 6, p14) | `figures read` |
| Catastrophic | 2606.23920v1 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/catastrophic-compositional-generation.pdf) | 41 | 2D Gaussian mixtures, sliced Wasserstein distance on 5,000 samples over 30 training runs (Tables 1 and 2, p10, p11). Semi-synthetic objects in a room (p11, p31) | `figures read` |
| TestTime | 2605.07078v1 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/test-time-compositional-generalization-via-concept-discovery.pdf) | 18 | Color-MNIST with 40 held-out slots of 160, and compositional CelebA (p7, p8) | `figures read` |
| LiftScores | 2505.13740v2 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/scorers/compositional-generation-with-lift-scores.pdf) | 28 | CLEVR position task (Tables 5 to 7, p21, p22). Attend-and-Excite prompts (Tables 3 and 8, p8, p23). The shelf PDF is the ICML 2025 version. Also rivals Paper 2 | `listed` |
| LogicalGuidance | 2602.05549v2 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/logical-guidance-for-exact-composition-of-diffusion-models.pdf) | 32 | Colored MNIST and Shapes3D (Tables 2 and 4, p5, p7). CelebA (Table 3, p6) | `listed` |

## Paper 2: restoring plurality in one model

| Rival | arXiv | PDF | Pages | Datasets and the page that states them | Read state |
|---|---|---|---|---|---|
| CO3 | 2509.25940v2 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/region-and-binding-failures/steer-away-from-mode-collisions.pdf) | 25 | Attend-and-Excite prompts (p6, p7). T2I-CompBench Complex and RareBench Concat, Relations and Complex (p6, Table 2, p7) | `figures read` |
| ToMe | 2411.07132v1 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/region-and-binding-failures/token-merging-for-training-free-semantic-binding.pdf) | 27 | T2I-CompBench (p7, p8). The authors' GPT-4o object-binding set of 50 prompts (p7) | `figures read` |
| AttendExcite | 2301.13826v2 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/region-and-binding-failures/attend-and-excite-attention-based-semantic-guidance-for-text-to-image-diffusion.pdf) | 24 | The authors' 276 prompts: 66 animal-animal, 66 object-object and 144 animal-object, with 64 seeds each (p6) | `figures read` |
| FlipConcept | 2502.15203v2 | [downloaded](pdfs/2502.15203.pdf) | 7 | Custom Concept 101 plus images generated by Stable Diffusion, five evaluation sets (p4). Scores in the table on p6 | `figures read` |
| CFGRevisited | 2608.16786v1 | [downloaded](pdfs/2608.16786.pdf) | 13 | GenEval, DPG-Bench and selected OneIG families, on SD3.5 Medium and FLUX.2 klein (p2) | `figures read` |
| CountLoop | 2508.16644v5 | [on the shelf](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/region-and-binding-failures/countloop-high-instance-image-generation.pdf) | 17 | COCO-Count, T2I-CompBench Count, and the authors' CountLoop-S and CountLoop-M sets (p8, Table 1, p10) | `listed` |

SuperDiff and LiftScores rival both papers and are counted once each, so the total is 22.

## For the shelf to take in

The seven papers marked `downloaded` are not on the [paper shelf](../../../../../../goal-setting/learning/papers/Papers.md).
Their PDFs sit in `artifacts/notes/progress/targets/pdfs/`, named by arXiv id: CoInD, PiCo, FreeFuse, DomainExpansion,
Sculpting, FlipConcept and CFGRevisited.

## Published methods checked and left off

The replications hub runs 12 published methods outside this list. A method joins the list when its own tables score it on a
dataset that can be a headline. LiftScores, LogicalGuidance and CountLoop met that. Board question 3 asks the owner whether
this is the right cut.

| Method | Why it is off the list |
|---|---|
| Conditional Object Composition (2611.00804) | Its tables score success rate on MS COCO. Attend-and-Excite appears there as a method |
| Conflict-Aware Guidance | Its CelebA-HQ table is an image-editing task, a different task from composing at generation |
| MCMC Correction (2307.14012) | Its tables use CIFAR-100, and the candidate benchmark is the disjoint CIFAR-10 split |
| When Do Diffusion Models Learn Multiple Objects | No table scores it on a candidate dataset |
| The other 5 hub methods | Their text names none of the candidate datasets |
