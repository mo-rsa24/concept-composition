# Reproducing CoMat on T2I-CompBench Color, Shape and Texture

## What this asks

Do CoMat's released SDXL weights reproduce the T2I-CompBench scores printed for CoMat? CoMat is a LoRA, a small trained add-on
to SDXL's U-Net, that teaches the model to attend to every concept in a prompt.

## The run

**Group.** Reproduction (group 4). Each number freezes as a baseline (group 5) when it lands.

**Varied axis.** None. CoMat runs on the T2I-CompBench prompts the subject guard allows.

**Buys.** Three rows of `table.csv`: CoMat on Color (288 of 300 prompts), Shape (293 of 300) and Texture (300 of 300).

**Cost.** T2I-CompBench's own instructions generate 10 images per prompt, so 881 allowed prompts are 8,810 pictures. At 1024 pixels,
the size the owner chose on board question 19, that is about 34 GPU-hours at the 14 seconds per picture measured for SDXL on an
RTX 3090.

## Are the released weights CoMat's?

`CaraJ/CoMat_sdxl_ft_unet` on Hugging Face holds `sdxl/lora/pytorch_lora_weights.safetensors` and `sdxl/unet_ft100/`.

| Check | What was found |
|---|---|
| Who published it | the account's full name is Dongzhi Jiang, CoMat's first author |
| LoRA rank | 128 on all 560 `down` matrices, the rank in `scripts/sdxl.sh` and in the paper's Table 8 |
| Was it trained | sampled `up` matrices have no zero entries and a root mean square of about 0.0008; a LoRA's `up` matrix starts at zero |
| The fine-tuned U-Net | the folder name matches the 100-step fine-tune on 512-pixel images that the paper and README describe |
| What is missing | the repository has no model card, CoMat's GitHub README does not link it, and its "release the checkpoints" box is unticked |

These files are the authors' trained CoMat SDXL LoRA and fine-tuned U-Net. Nothing states that this is the exact checkpoint behind
Table 1, so a miss could come from a different checkpoint.

## Which published number

By the owner's answer to board question 19, each row is judged against CoMat's own Table 1 (arXiv v3, p7). ToMe's quotes of the
same model differ, so they go in each row's note.

| Benchmark | CoMat's own Table 1, the number judged | ToMe's quote, in the note |
|---|---|---|
| Color | 0.7827 (`CoMat-T1-055`, the target) | 0.7774 |
| Shape | 0.5329 (`CoMat-T1-056`) | 0.5262 (`ToMe-T1-051`) |
| Texture | 0.6468 (`CoMat-T1-057`) | 0.6591 (`ToMe-T1-050`) |

**The plain SDXL check on Color.** Plain SDXL runs through CoMat's sampler on the 288 allowed Color prompts at seed 42, then the same BLIP-VQA scorer. CoMat prints 0.5879 for plain SDXL on
Color (`CoMat-T1-049`) and ToMe prints 0.6369 (`ToMe-T1-001`), so this shows where our setup sits and lets CoMat's gain over plain be read on matched pictures. Cost: 288 pictures at about 14 seconds, about 1.1
GPU-hours. Buys: the check on the Color row, the one benchmark where plain SDXL has not run through our scorer.

## Loading the weights

The files are at commit `c7a18451` of the Hugging Face repository, in `/datasets/mmolefe/poe_repair_min/loop-5/third_party/`, and both large
files match their published SHA-256. The LoRA keys carry a doubled `unet.unet.` prefix, which a standard loader can skip without an error. The
hub's `comat_generate.py` therefore merges all 560 up and down pairs into their layers itself and refuses to run unless all of them land. CoMat's
validation pipeline is the standard SDXL loop with the watermark off, so the stock pipeline runs with the watermark off.

## What the release leaves open

| Setting | Paper | Released code | Used here |
|---|---|---|---|
| Evaluation resolution | not stated; training is at 512 | the validation call leaves the size unset, so SDXL's default of 1024 applies | 1024, by the owner's answer to board question 19 |
| Sampler and steps | DDPM, 50 steps, for training | validation uses `--scheduler` DDPM and `--total_step` 50 | DDPM, 50 steps |
| Guidance | 7.5 for training | `--cfg_scale` default 7.5 | 7.5 |
| Images per prompt | follows the benchmark's defaults | T2I-CompBench's `inference_eval.py` makes 10 | 10 |
| Seeds | not stated | T2I-CompBench uses 42 to 51 | 42 to 51 |
| Scorer | "latest" T2I-CompBench code, scorer unnamed | `BLIPvqa_eval/BLIP_vqa.py` at commit `4aa4042` | BLIP-VQA at that commit |

CoMat's paper says the benchmark's evaluation code has changed over time, so the scorer's commit is recorded with every number.

## The test, fixed before the first run

The standard error comes from resampling prompts with replacement 10,000 times, keeping each prompt's images together. CoMat prints
no standard error, so the published one is taken as equal to ours. On the allowed prompts, a row counts as `reproduced` when
|ours − published| ≤ 1.96 × √(SE_ours² + SE_published²), and otherwise as `not reproduced`. Every number carries its prompt and image
counts. The rule is in `scripts/reproductions/verdict.py` in the replications hub.

## The ladder

Each rung saves its config, metrics, image sheet and commit under `/datasets/mmolefe/poe_repair_min/loop-5/renders/CoMat/<rung>/`.

1. **Smoke test.** Two prompts, recording the time per picture.
2. **Overfit.** Does not apply, because nothing trains.
3. **1% of the data.** Three prompts from each subset, with the sheet opened at full size beside SDXL.
4. **Controls.** With the LoRA's scale at 0, the loaded pipeline must match the same pipeline without the LoRA to under 1e-5
   (largest absolute latent difference), in one process on one card. At scale 1 the latent must differ by more than 1e-3, and the merged weights
   must match the LoRA's own forward rule on every layer to under 1e-4.
5. **Full protocol.** All allowed prompts at 10 images each.

## Cross-references

- The mention of **CoMat**, a paper of reading sitting 4, in [the route to two papers, under The steps](../../../../../plans/route-two-papers.md#the-steps).
