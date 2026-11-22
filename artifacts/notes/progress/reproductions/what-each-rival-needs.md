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
| ToMe | 5 | `tome` `f256a49` | official demo only | plain SDXL | our own batch loop; Color waits on question 13 |
| CoMat | 3 | `comat` `595cf4a` | training code; README's "release the checkpoints" box is unticked | `CaraJ/CoMat_sdxl_ft_unet` on Hugging Face holds an SDXL LoRA and a 100-step fine-tuned U-Net, and no SD1.5 weights were found | confirm the LoRA is the CoMat model; the Shape and Texture rows pick SD1.5, so they wait on question 13 |
| Attend-and-Excite | 3 | `attend-and-excite` `163efdf` | official | SD1.4 | Shape and Texture wait on question 13 |
| Liu | 6 | `composable-diffusion` `77af82b` | official | SD pipelines for T2I-CompBench; the CoInD rows use CoInD's own checkpoints | four of six rows wait on question 13 |
| FKC | 1 | `fkc-diffusion` `aa6f5ed` | no SDXL GenEval script in the authors' repository; the hub holds an uncommitted port | plain SDXL | commit the port by named paths, check it against the plain product |
| CoInD | 2 | `coind` `99056cc` | official, with CS and FID scripts | a spreadsheet linked from its README | fetch the sheet and confirm Shapes3D and CelebA checkpoints exist |
| SuperDiff | 2 | `super-diffusion` `e859107` | official CIFAR trainer and evaluator | SD and SDXL only on Hugging Face, and no CIFAR weights found | two CIFAR models trained at least 3 times each; the CMNIST row waits on question 13 |
| Bradley-Projective | 1 | `projective-composition` `b57343a` | the authors released none; this is our replication on EDM2 | our `clevr-none-20k-64-xs-8Mi` and `clevr-color-20k-64-xs-8Mi` (46 GB) on `/datasets` | check these match the printed protocol, which is 128 by 128 pixels |
| LogicalGuidance | 1 | none | no repository named in its PDF | none | look for code, then file a question |

## Scorers and prompt sets

| Benchmark | Where the prompts are | Scorer | Status |
|---|---|---|---|
| T2I-CompBench Color, Shape, Texture | `t2i-compbench/examples/dataset/{color,shape,texture}_val.txt`, 300 prompts each | BLIP-VQA, `BLIPvqa_eval/BLIP_vqa.py` | not yet run |
| T2I-CompBench Complex | RTD's manifest, 1,000 prompts | ImageReward | smoke test passed |
| RareBench Concat | RTD's manifest, 40 of the 120 RareBench prompts | ImageReward | not yet run |
| GenEval | `geneval/prompts/evaluation_metadata.jsonl`, 553 prompts of which 99 are two-object | Mask2Former through `mmdet` 2.x | not yet run |

FKC's row names "GenEval (1000 prompts)", while the released metadata holds 553. FKC's design has to settle which set the
printed number used.

## Prompts the subject guard keeps out

`scripts/shared/disallowed_subjects.py` applies to every render and score here, and no pair-ledger row covers this work.
Board question 14 asks the owner how to treat the rows it affects.

| Benchmark | Prompts | Blocked permanently | Need a ledger decision | Allowed |
|---|---|---|---|---|
| T2I-CompBench Complex | 1,000 | 6 | 6 | 988 |
| RareBench Concat | 40 | 1 | 6 | 33 |
| T2I-CompBench Color | 300 | 2 | 10 | 288 |
| T2I-CompBench Shape | 300 | 2 | 5 | 293 |
| T2I-CompBench Texture | 300 | 0 | 0 | 300 |
| GenEval | 553 | 0 | 0 | 553 |

## The order the runs take

1. **RTD** first, because its repository holds the full protocol for two rows and it uses only plain SDXL.
2. **CO3 and ToMe** on the same prompts and seeds as RTD.
3. **CoMat, Attend-and-Excite, Liu and FKC** on their T2I-CompBench and GenEval rows.
4. **CoInD and Liu** on Shapes3D and CelebA, once the checkpoint sheet is read.
5. **Bradley-Projective, SuperDiff and LogicalGuidance**, each as a costed question where its run would pass 24 GPU-hours or
   where no code exists.
