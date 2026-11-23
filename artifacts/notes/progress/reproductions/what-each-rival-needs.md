# What each rival needs before it can be reproduced

Step D2 reproduces 11 rivals across 28 rows of `table.csv`. This page says, per rival, which code and weights exist, so
the runs start with the rivals that can run today and the expensive ones reach the owner as a costed question.

Upstream repositories sit in `/home/molef/PhD/replications/<folder>/`, cloned at the commit shown. They are read here and
have not been run.

## The rivals by what is missing

| Rival | Rows | Folder and commit | Code | Weights | What is missing |
|---|---|---|---|---|---|
| RTD | 2 | `rectify-then-diffuse` `241eb37` | official, with prompt manifests for Complex (1,000) and RareBench (120) | plain SDXL | nothing known |
| CO3 | 2 | `co3` `35ea972` | official sampler; its README lists benchmark evaluation as coming soon | plain SDXL | our own batch loop over the prompts |
| ToMe | 5 | `tome` `f256a49` | official demo only | plain SDXL | our own batch loop; its Color row is not comparable |
| CoMat | 3 | `comat` `595cf4a` | training code; README's "release the checkpoints" box is unticked | `CaraJ/CoMat_sdxl_ft_unet` on Hugging Face holds an SDXL LoRA and a 100-step fine-tuned U-Net, and no SD1.5 weights were found | the weights check out as the authors'; each row is judged on CoMat's own Table 1 at 1024 pixels (board question 19) |
| Attend-and-Excite | 3 | `attend-and-excite` `163efdf` | official | its printed Color run used Stable Diffusion 2 base | all three rows are not comparable, so nothing runs (board questions 13 and 25) |
| Liu | 6 | `composable-diffusion` `77af82b` | official | its printed Color run used Stable Diffusion 2 base; the CoInD rows use CoInD's own checkpoints | its CMNIST, Color, Shape and Texture rows are not comparable (board questions 13 and 25) |
| FKC | 1 | `fkc-diffusion` `aa6f5ed` | no SDXL GenEval script in the authors' repository; the hub holds an uncommitted port | plain SDXL | port the authors' EDM2 sampler to SDXL and test it against `FKC-TA4-038`, 0.78 on the 553 released prompts (board question 23) |
| CoInD | 2 | `coind` `99056cc` | official, with CS and FID scripts | CelebA on SD1.5: `sd_celeba.zip` (25.3 GB, Google Drive) holds CoInD and Composed GLIDE, and the authors' sheet prints the same 49.15 and 14.19. Shapes3D: the sheet's tab is empty | context by board question 35: CelebA stopped after its classifier and smoke test, and Shapes3D is skipped |
| SuperDiff | 1 | `super-diffusion` `e859107` | official CIFAR trainer and evaluator | SD and SDXL only on Hugging Face, and no CIFAR weights found | CIFAR-10 is context with no row (board question 30); its CMNIST row is not comparable |
| Bradley-Projective | 1 | `projective-composition` `b57343a` | the authors released none; this is our replication on EDM2 | our `clevr-none-20k-64-xs-8Mi` and `clevr-color-20k-64-xs-8Mi` (46 GB) on `/datasets` | its row stays blocked (board question 21), and its rule runs on our own single-object CLEVR models (board question 30); these checkpoints are colour-conditioned at 64 pixels |
| LogicalGuidance | 1 | `logdiff` `94ef35b` | official, linked from its ICML 2026 slides, with training and evaluation scripts for CMNIST | none released | context by board question 35, so nothing runs |

## Scorers and prompt sets

| Benchmark | Where the prompts are | Scorer | Status |
|---|---|---|---|
| T2I-CompBench Color, Shape, Texture | `t2i-compbench/examples/dataset/{color,shape,texture}_val.txt`, 300 prompts each | BLIP-VQA, `BLIPvqa_eval/BLIP_vqa.py`, run unchanged at the benchmark's package versions | canary passed: the 3 scores shipped in `examples/reward/vqa_result.json` reproduce exactly |
| T2I-CompBench Complex | RTD's manifest, 1,000 prompts | ImageReward | smoke test passed |
| RareBench Concat | RTD's manifest, 40 of the 120 RareBench prompts | ImageReward | not yet run |
| GenEval | `geneval/prompts/evaluation_metadata.jsonl`, 553 prompts of which 99 are two-object | Mask2Former through `mmdet` 2.28.2, in `loop-5/envs/geneval` at the authors' pins | environment built and weights checked, not yet run |

FKC's printed 1,000-prompt list was never released, so by the owner's answer to board question 23 its row is tested on the released
553 prompts against `FKC-TA4-038`.

## Prompts the subject guard keeps out

`scripts/shared/disallowed_subjects.py` applies to every render and score here, and no pair-ledger row covers this work.
By the owner's answer to board question 14, rivals and our method are scored on the same allowed prompts, each row is judged on them, and every number carries its prompt count. The prompts needing a decision stay out.

| Benchmark | Prompts | Blocked permanently | Need a ledger decision | Allowed |
|---|---|---|---|---|
| T2I-CompBench Complex | 1,000 | 6 | 6 | 988 |
| RareBench Concat | 40 | 1 | 6 | 33 |
| T2I-CompBench Color | 300 | 2 | 10 | 288 |
| T2I-CompBench Shape | 300 | 2 | 5 | 293 |
| T2I-CompBench Texture | 300 | 0 | 0 | 300 |
| GenEval | 553 | 0 | 0 | 553 |

## The order the runs take

The 7 targets are T2I-CompBench Color, Shape, Texture and Complex, GenEval, and CLEVR for each paper.

1. **CO3 and ToMe** finish their T2I-CompBench renders and are scored.
2. **CoMat** on Color, Shape and Texture.
3. **FKC** on GenEval's 553 prompts.
4. **CLEVR**, under its own design from `targets/clevr-scaling-plan.md`, starting with installing Blender and timing one batch.
5. **The checkers** retrain on the data the owner set on board question 37, before step D3 compares candidates.
