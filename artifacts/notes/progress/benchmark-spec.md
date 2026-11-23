# Benchmark spec: what each rival ran

This page says what a run has to copy before its number can sit beside a rival's. It lists the 10 benchmarks the
owner accepted as targets, then 166 dataset settings from 24 of the 25 rival papers, each with its
split, metrics and sample count. Rectify Then Diffuse appears with its two target settings only, and RPG has no section
because it sets no target. The numbers come from the [rivals' numbers table](targets/numbers.csv), and the
rules for judging a win are in the [comparison protocol](targets/comparison-protocol.md).

**Split** means which data trained the model and which data was evaluated. **Count** means what stands behind one
reported number: generated images, prompts, images per prompt, seeds or runs. A **setting** is one rival paper on one
dataset or task, as the numbers table names it.

## Table of contents

- [The 10 benchmarks to beat](#the-10-benchmarks-to-beat)
- [What the rivals leave out](#what-the-rivals-leave-out)
- [Every rival's datasets](#every-rivals-datasets)
- [How this was checked](#how-this-was-checked)

## The 10 benchmarks to beat

| Paper | Benchmark | Metric | Rival's number | Split | Count behind the number | Where |
|---|---|---|---|---|---|---|
| Paper 1 | CLEVR location-conditioned, trained on one object | Kmax, the most objects that succeed, higher is better | Bradley-Local 5 | EDM2 xs local patch model trained on images holding one object, location-conditioned at 128x128, 500,000 training samples per object count. | 1,024 samples per configuration, judged by a ResNet-18 counting classifier with 99.74% test accuracy. | [Local Mechanisms Table 1, page 3](https://arxiv.org/pdf/2509.16447#page=3) |
| Paper 1 | Shapes3D Partial support, ∧ composition | CS (%), higher is better | CoInD 91.1 | Shapes3D trained on the orthogonal partial support composition split of Schott et al.; AND and NOT evaluations restricted to unseen compositions under orthogonal partial support. | not stated | [CoInD Table 4, page 20](https://arxiv.org/pdf/2503.01145#page=20) |
| Paper 1 | CMNIST, AND | Conformity score (%), higher is better | LogicalGuidance 83.6 | Diffusion model trained on CMNIST (hyperparameters in Table 9); no train/test split stated. Evaluation uses randomly generated AND queries (100 unique queries per task type). | 10,000 samples from 100 unique queries per task type with 100 samples each. | [LogicalGuidance Table 4, page 7](https://arxiv.org/pdf/2602.05549#page=7) |
| Paper 1 | CelebA Orthogonal support, ∧ composition | CS (%), higher is better | CoInD 49.15 | Stable Diffusion v1.5 fine-tuned on data constructed from CelebA with labels converted to text; AND composition prompts 'smiling' and 'male' evaluated against the joint prompt. | Not stated for this fine-tuned SDv1.5 table. The 10,000 samples on page 22 are for FID in the separate CelebA U-Net experiment. | [CoInD Table 6, page 27](https://arxiv.org/pdf/2503.01145#page=27) |
| Paper 2 | T2I-CompBench Color | T2I-CompBench result, higher is better | CoMat 0.7827 | Trained on the T2I-CompBench training set, all of HRS-Bench and 5,000 prompts chosen at random from ABC-6K (about 20,000 prompts); evaluation follows T2I-CompBench default settings, the evaluated split is not stated | Not stated. The paper describes the benchmark as 6,000 prompts over 6 sub-categories and does not say how many were evaluated, how many images per prompt or which seeds. | [CoMat Table 1, page 7](https://arxiv.org/pdf/2404.03653#page=7) |
| Paper 2 | T2I-CompBench Shape | BLIP-VQA, higher is better | ToMe 0.6051 | Training-free method on SDXL; evaluated on the 300 validation prompts of this T2I-CompBench subset. | 300 validation prompts; images per prompt and seeds not stated | [ToMe Table 1, page 8](https://arxiv.org/pdf/2411.07132#page=8) |
| Paper 2 | T2I-CompBench Texture | BLIP-VQA, higher is better | ToMe 0.6894 | Training-free method on SDXL; evaluated on the 300 validation prompts of this T2I-CompBench subset. | 300 validation prompts; images per prompt and seeds not stated | [ToMe Table 1, page 8](https://arxiv.org/pdf/2411.07132#page=8) |
| Paper 2 | T2I-CompBench Complex | ImageReward (IR), higher is better | RTD 0.4661 | No training (Rectify Then Diffuse on pretrained SDXL). Evaluated on the 1,000 prompts of the T2I-CompBench complex compositional split. | 1,000 prompts, three fixed seeds per prompt (3,000 images). | [Rectify Then Diffuse Table 2, page 6](https://arxiv.org/pdf/2608.03135#page=6) |
| Paper 2 | RareBench Concat | ImageReward (IR), higher is better | RTD 0.4464 | No training (Rectify Then Diffuse on pretrained SDXL). Evaluated on the Concat category of RareBench. | RareBench holds 120 prompts across Concat, Relations and Complex, and the Concat share is not stated. Three fixed seeds per prompt. | [Rectify Then Diffuse Table 2, page 6](https://arxiv.org/pdf/2608.03135#page=6) |
| Paper 2 | GenEval (553 prompts) | Two object, higher is better | FKC 0.78 | SDXL with classifier-free guidance plus FKC at inference, guidance weight 7.5; no training stated. Evaluated on GenEval's released 553 prompts, of which 99 are two-object. | 553 prompts; images per prompt and seeds not stated. | [FKC Table A4, page 39](https://arxiv.org/pdf/2503.02819#page=39) |

## What the rivals leave out

**Only 3 of the 10 targets come with a stated image count:** Bradley's local patch model (1,024), Logical Guidance
(10,000) and Rectify Then Diffuse on T2I-CompBench Complex (1,000 prompts at three seeds, so 3,000). Its RareBench Concat row
states three seeds and leaves out how many of the 120 prompts are Concat. ToMe states 300 prompts on its two benchmarks and FKC
states 553, and both leave out images per prompt and seeds. The other 3 targets have no count for that number.

A run on those 7 benchmarks records the count it used, and its row reads "partly matched", as the
[comparison protocol](targets/comparison-protocol.md#numbers) requires. Step D2 reruns one published number per rival,
which is where a missing setting can be tested.

**GenEval's target uses the 553 released prompts.** FKC released no list of its 1,000 prompts. On the 553, FKC prints 0.78 and
plain SDXL prints 0.80, so the target is 0.79 and a win must also beat plain SDXL's 0.80 on the same prompts and seeds.

**CIFAR-10 disjoint halves are context.** SuperDiff's task there is the OR setting, which samples from either model. Paper 1
claims AND, both things in one image, so the 4.11 FID sets no target.

**Paper 1's CLEVR target is the most objects that succeed for models trained on one object.** A count K succeeds when K objects
appear in at least 50% of images and at least K-2 appear in at least 90%. Bradley's local patch model reaches 5 and separately
composed models reach 4, so the target is 6. Diffusion Domain Expansion's 44.5 at 5 positions is a different benchmark, because it trains one base
model on all 30,000 multi-object CLEVR images. The [reasons for the four judged targets](targets/proposed-target-changes.md)
give the number behind each.

## Every rival's datasets

A ★ marks a setting that holds one of the 10 targets. A count can be partial, and its text then names what the paper
leaves out.

<details>
<summary><b>AttendExcite</b> (<a href="https://arxiv.org/abs/2301.13826">arXiv 2301.13826</a>): 6 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| Attend-and-Excite Animal-Animal | No training split; a new prompt benchmark built from 12 animals, 12 objects and 11 colors, evaluated on Stable Diffusion. This subset uses the template "a [animalA] and a [animalB]", 66 prompts. Side comparison with Prompt-to-Prompt uses 20 randomly selected prompts per subset; user study uses 10 randomly selected prompts per subset. | Average CLIP text-text similarity, Average Maximum Object CLIP image-text similarity, Relative difference from Attend-and-Excite in CLIP text-text similarity (%), Relative difference from Attend-and-Excite in Maximum Object Similarity (%), Share of responses preferring the method (%) | 66 prompts, 64 images per prompt from 64 random seeds applied across all methods. User study: 65 respondents, 10 prompts, the same 4 seeds for all methods. Prompt-to-Prompt comparison: 20 prompts per subset (images per prompt not stated). | 7, 8, 14 |
| Attend-and-Excite Animal-Object | No training split; a new prompt benchmark built from 12 animals, 12 objects and 11 colors, evaluated on Stable Diffusion. This subset uses the template "a [animal] and a [color][object]", 144 prompts. Side comparison with Prompt-to-Prompt uses 20 randomly selected prompts per subset; user study uses 10 randomly selected prompts per subset. | Average CLIP text-text similarity, Average Maximum Object CLIP image-text similarity, Relative difference from Attend-and-Excite in CLIP text-text similarity (%), Relative difference from Attend-and-Excite in Maximum Object Similarity (%), Share of responses preferring the method (%) | 144 prompts, 64 images per prompt from 64 random seeds applied across all methods. User study: 65 respondents, 10 prompts, the same 4 seeds for all methods. Prompt-to-Prompt comparison: 20 prompts per subset (images per prompt not stated). | 7, 8, 14 |
| Attend-and-Excite Object-Object | No training split; a new prompt benchmark built from 12 animals, 12 objects and 11 colors, evaluated on Stable Diffusion. This subset uses the template "a [colorA][objectA] and a [colorB][objectB]", 66 prompts. Side comparison with Prompt-to-Prompt uses 20 randomly selected prompts per subset; user study uses 10 randomly selected prompts per subset. | Average CLIP text-text similarity, Average Maximum Object CLIP image-text similarity, Relative difference from Attend-and-Excite in CLIP text-text similarity (%), Relative difference from Attend-and-Excite in Maximum Object Similarity (%), Share of responses preferring the method (%) | 66 prompts, 64 images per prompt from 64 random seeds applied across all methods. User study: 65 respondents, 10 prompts, the same 4 seeds for all methods. Prompt-to-Prompt comparison: 20 prompts per subset (images per prompt not stated). | 7, 8, 14 |
| Complex prompts from StructureDiffusion and Conceptual Captions | No training split; 40 prompts collected from the qualitative examples in the StructureDiffusion paper and the Conceptual Captions dataset, evaluated on Stable Diffusion against StructureDiffusion. | CLIP Similarity, Relative difference from Attend-and-Excite in CLIP Similarity (%) | 40 prompts; the same 64 random seeds for all methods (images per prompt not printed in this passage). | 14 |
| Runtime for one image | No dataset; runtime measurement only, evaluated on an A100 GPU. | Runtime (seconds per image) | Approximate seconds for a single image; number of timed images or repeats not stated. | 10 |
| User study prompts, lowest-performing prompt for Attend-and-Excite | User study prompts: 10 randomly selected prompts per subset, the same 4 randomly selected seeds for all methods. Which prompt was lowest-performing is not identified beyond its vote share. | Share of votes on that prompt (%) | 65 respondents; the quoted figure is one prompt's vote share (59.09% for Attend-and-Excite, 16% each for SD and StructureDiffusion). | 8 |

</details>
<details>
<summary><b>Bradley-Local</b> (<a href="https://arxiv.org/abs/2509.16447">arXiv 2509.16447</a>): 4 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| CLEVR color-conditioned | Color-conditioned CLEVR model trained on 1-5 objects, tested on up to 8 colors (more than 5 colors is out of distribution). | Kmax, Manual count of images with 0 correctly colored objects (of 64), Manual count of images with 1 correctly colored objects (of 64), Manual count of images with 10 correctly colored objects (of 64), Manual count of images with 2 correctly colored objects (of 64), Manual count of images with 3 correctly colored objects (of 64), Manual count of images with 4 correctly colored objects (of 64), Manual count of images with 5 correctly colored objects (of 64), Manual count of images with 6 correctly colored objects (of 64), Manual count of images with 7 correctly colored objects (of 64), Manual count of images with 8 correctly colored objects (of 64), Manual count of images with 9 correctly colored objects (of 64) | 64 samples per configuration, counted manually | 30 |
| CLEVR location-conditioned ★ | Location-conditioned models trained on CLEVR images with 1, 1-3 or 1-5 objects (500,000 samples per object count, 128x128) and tested on K conditioned locations beyond the training count. | Automated classifier count of images with 0 objects (of 1024), Automated classifier count of images with 1 objects (of 1024), Automated classifier count of images with 10 objects (of 1024), Automated classifier count of images with 2 objects (of 1024), Automated classifier count of images with 3 objects (of 1024), Automated classifier count of images with 4 objects (of 1024), Automated classifier count of images with 5 objects (of 1024), Automated classifier count of images with 6 objects (of 1024), Automated classifier count of images with 7 objects (of 1024), Automated classifier count of images with 8 objects (of 1024), Automated classifier count of images with 9 objects (of 1024), Kmax, Objects the model can draw, at any location (at least), Test accuracy (%) | 1024 samples per configuration, classified by a ResNet-18 counter | 3, 23, 28, 29 |
| MDM feature-space disentanglement | Matryoshka Diffusion Model trained on a subset of 50M Flickr images (training set known); hidden states of the innermost U-net recorded for 12 single-concept prompts in 6 categories. | Disentanglement ratio, high end of stated range, Disentanglement ratio, low end of stated range | 12 prompts with 10 seeds per prompt, single forward pass at t = 999 | 9 |
| SDXL feature-space disentanglement | Pre-trained SDXL base 1.0, no training; hidden states recorded for 24 prompts in 4 semantic categories. | Between-category cosine similarity, high end of stated range, Between-category cosine similarity, low end of stated range, Disentanglement ratio, Disentanglement ratio, high end of stated range, Disentanglement ratio, low end of stated range, Within-category cosine similarity, high end of stated range, Within-category cosine similarity, low end of stated range | 24 prompts with 10 seeds per prompt | 9 |

</details>
<details>
<summary><b>Bradley-Projective</b> (<a href="https://arxiv.org/abs/2502.04549">arXiv 2502.04549</a>): 2 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| CLEVR color-conditioned | Result of Bradley (2025) reported here: a color-conditioned CLEVR model trained on 1-5 objects; this paper composes the same model explicitly in Figure 16. | Length generalization (number of colors) | not stated | 37, 39 |
| CLEVR location-conditioned | Explicit composition of single-object location-conditioned CLEVR models (single-object dataset of 1,550,000 samples, with empty, unconditional or learned background) tested at N = 1 to 6 composed locations; learned-composition Kmax values are taken from Bradley (2025). | Images with 0 objects in correct locations (count of 100), Images with 1 objects in correct locations (count of 100), Images with 2 objects in correct locations (count of 100), Images with 3 objects in correct locations (count of 100), Images with 4 objects in correct locations (count of 100), Images with 5 objects in correct locations (count of 100), Images with 6 objects in correct locations (count of 100), Kmax | 100 samples per composition method, with objects in correct locations counted manually | 17, 39 |

</details>
<details>
<summary><b>Catastrophic</b> (<a href="https://arxiv.org/abs/2606.23920">arXiv 2606.23920</a>): 6 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| 2D Gaussian mixture, in-distribution (six components at {-3,0,3}x{-2.5,2.5}, sigma=0.6) | Mixture-of-Gaussians models trained as in the 2D Gaussian setting on a dataset of N points; in-distribution target; ground-truth samples from rejection sampling | MMD2, SW2 | 5000 samples from the target and 5000 empirical samples, averaged over 30 runs (Table 2 follows the conventions of Table 1) | 11, 39, 40 |
| 2D Gaussian mixture, out-of-distribution (P^a1 means (-6,1.5),(0,0); P^a2 means (0,-9),(-1.5,6); P^0 all four means; sigma=1) | Mixture-of-Gaussians models trained as in the 2D Gaussian setting on a dataset of N points; out-of-distribution target; ground-truth samples from importance sampling | MMD2, SW2 | 5000 samples from the target and 5000 empirical samples, averaged over 30 runs (Table 2 follows the conventions of Table 1) | 11, 39, 40, 41 |
| 2D Gaussian, factorized conditionals, in-distribution (P^a1=N(0,diag(10,1)), P^a2=N(0,diag(1,10)), P^0=N(0,10I), P^w=N(0,I)) | Conditional diffusion model trained on a dataset of N points (N = 100, 1000 or 10000) drawn from the source distributions P^0, P^a1, P^a2 (or analytic scores); evaluated against samples from the composed target P^w, in-distribution setting where the target is well-supported by the training data | MMD2, SW2 | Each value is the sliced Wasserstein-2 distance between 5000 samples from the target and 5000 empirical samples, averaged over 30 training runs | 10, 28, 34, 36, 37 |
| 2D Gaussian, factorized conditionals, out-of-distribution (P^a1=N(0,diag(10,1)), P^a2=N(0,diag(1,10)), P^0=N(0,I), P^w=N(0,10I)) | Conditional diffusion model trained on a dataset of N points (N = 100, 1000 or 10000) drawn from the source distributions P^0, P^a1, P^a2 (or analytic scores); evaluated against samples from the composed target P^w, out-of-distribution setting where the target lies in low-density regions of the source distributions | MMD2, SW2 | Each value is the sliced Wasserstein-2 distance between 5000 samples from the target and 5000 empirical samples, averaged over 30 training runs | 10, 28, 35, 38 |
| 2D Gaussian, non-factorized conditionals, in-distribution (P^a1=N(0,diag(10,1)), P^a2=N(0,diag(1,10)), P^0=N(0,20I), P^w=N(0,(20/21)I)) | Conditional diffusion model trained on a dataset of N points (N = 100, 1000 or 10000) drawn from the source distributions P^0, P^a1, P^a2 (or analytic scores); evaluated against samples from the composed target P^w, in-distribution setting where the target is well-supported by the training data | MMD2, SW2 | Each value is the sliced Wasserstein-2 distance between 5000 samples from the target and 5000 empirical samples, averaged over 30 training runs | 10, 28, 34, 35, 37 |
| 2D Gaussian, non-factorized conditionals, out-of-distribution (P^a1=N(0,diag(10,1)), P^a2=N(0,diag(1,10)), P^0=N(0,1.1I), P^w=N(0,(110/21)I)) | Conditional diffusion model trained on a dataset of N points (N = 100, 1000 or 10000) drawn from the source distributions P^0, P^a1, P^a2 (or analytic scores); evaluated against samples from the composed target P^w, out-of-distribution setting where the target lies in low-density regions of the source distributions | MMD2, SW2 | Each value is the sliced Wasserstein-2 distance between 5000 samples from the target and 5000 empirical samples, averaged over 30 training runs | 10, 28, 36, 38, 39 |

</details>
<details>
<summary><b>CFGRevisited</b> (<a href="https://arxiv.org/abs/2608.16786">arXiv 2608.16786</a>): 29 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| DPG-Bench | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. | DPG (mean x 100), DPG (mean x 100): 95% difference margin | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 3, 10 |
| DPG-Bench L1 Attr | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L1 score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 10 |
| DPG-Bench L1 Entity | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L1 score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 10 |
| DPG-Bench L1 Global | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L1 score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 10 |
| DPG-Bench L1 Other | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L1 score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 10 |
| DPG-Bench L1 Relation | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L1 score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 10 |
| DPG-Bench L2 attribute color | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 10 |
| DPG-Bench L2 attribute other | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 10 |
| DPG-Bench L2 attribute shape | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 10 |
| DPG-Bench L2 attribute size | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 10 |
| DPG-Bench L2 attribute texture | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 10 |
| DPG-Bench L2 non-attribute count | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 non-attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 11 |
| DPG-Bench L2 non-attribute global | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 non-attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 11 |
| DPG-Bench L2 non-attribute non-sp. | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 non-attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 11 |
| DPG-Bench L2 non-attribute part | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 non-attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 11 |
| DPG-Bench L2 non-attribute spatial | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 non-attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 11 |
| DPG-Bench L2 non-attribute state | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 non-attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 11 |
| DPG-Bench L2 non-attribute text | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 non-attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 11 |
| DPG-Bench L2 non-attribute whole | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. This is a DPG-Bench sub-category; the paper gives no separate split for it. | DPG-Bench L2 non-attribute score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 11 |
| GenEval | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. | GE, GE: 95% difference margin, GenEval Overall, GenEval Overall difference from CFG | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 3, 6, 9 |
| GenEval attr | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. Only the attr task of GenEval is read here; GenEval is resampled within tasks. | GenEval category score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 9 |
| GenEval color | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. Only the color task of GenEval is read here; GenEval is resampled within tasks. | GenEval category score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 9 |
| GenEval count | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. Only the count task of GenEval is read here; GenEval is resampled within tasks. | GenEval category score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 9 |
| GenEval pos | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. Only the pos task of GenEval is read here; GenEval is resampled within tasks. | GenEval category score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 9 |
| GenEval single | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. Only the single task of GenEval is read here; GenEval is resampled within tasks. | GenEval category score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 9 |
| GenEval two | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. Only the two task of GenEval is read here; GenEval is resampled within tasks. | GenEval category score | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 9 |
| OneIG-Bench General Object | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. | Obj, Obj: 95% difference margin | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 3 |
| OneIG-Bench Knowledge Reasoning | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. | Rsn, Rsn: 95% difference margin | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 3 |
| OneIG-Bench Text Rendering | Training-free guidance methods on frozen SD3.5 Medium and FLUX.2 [klein] 4B Base; no training data. The full prompt set of the benchmark is evaluated, and methods share prompts and seeds. | Txt, Txt: 95% difference margin | Full prompt set per model and method; number of prompts, seeds and images per prompt are not stated. Margins use 10,000 paired prompt-bootstrap replicates (each resamples the original number of prompts with replacement). | 3 |

</details>
<details>
<summary><b>CO3</b> (<a href="https://arxiv.org/abs/2509.25940">arXiv 2509.25940</a>): 16 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| Animals (Attend-Excite category; this table names no benchmark) | No training (CO3 on pretrained SDXL with and without CO3 components). Evaluated on the Attend-Excite prompts of Chefer et al. (2023), category Animals. | BLIP-VQA, ImageReward | not stated | 18, 20, 21 |
| Animals Objects (Attend-Excite category; this table names no benchmark) | No training (CO3 on pretrained SDXL with and without CO3 components). Evaluated on the Attend-Excite prompts of Chefer et al. (2023), category Animals Objects. | BLIP-VQA, ImageReward | not stated | 18, 20, 21 |
| Animals, Animals Objects and Objects (Attend-Excite category; this table names no benchmark) | No training (CO3 on pretrained SDXL, with and without CO3). Evaluated on the Attend-Excite prompts of Chefer et al. (2023), categories Animals, Animals Objects and Objects. | Avg | not stated | 18 |
| Attend-Excite all three categories | No training (ablation of CO3 components on pretrained SDXL). Evaluated on the Attend-Excite prompts of Chefer et al. (2023), all three categories (Animal-Animal, Animal-Object, Object-Object). | Avg | not stated | 8 |
| Attend-Excite Animal-Animal | No training (CO3 is a sampling-time corrector on pretrained SDXL, so there is no train or validation split). Evaluated on the Attend-Excite two-concept prompts of Chefer et al. (2023), category Animal-Animal. | BLIP-VQA, DINOv2, ImageReward | not stated | 7, 8, 9 |
| Attend-Excite Animal-Object | No training (CO3 is a sampling-time corrector on pretrained SDXL, so there is no train or validation split). Evaluated on the Attend-Excite two-concept prompts of Chefer et al. (2023), category Animal-Object. | BLIP-VQA, DINOv2, ImageReward | not stated | 7, 8, 9 |
| Attend-Excite Animals | No training. Human evaluation on the Attend-Excite prompts of Chefer et al. (2023), category Animals. | Human Evaluation | Human evaluation by 11 participants, ratings 0 to 5 normalized to 0 to 1; number of prompts and images per prompt not stated. | 17 |
| Attend-Excite Animals-Objects | No training. Human evaluation on the Attend-Excite prompts of Chefer et al. (2023), category Animals-Objects. | Human Evaluation | Human evaluation by 11 participants, ratings 0 to 5 normalized to 0 to 1; number of prompts and images per prompt not stated. | 17 |
| Attend-Excite Object-Object | No training (CO3 is a sampling-time corrector on pretrained SDXL, so there is no train or validation split). Evaluated on the Attend-Excite two-concept prompts of Chefer et al. (2023), category Object-Object. | BLIP-VQA, DINOv2, ImageReward | not stated | 7, 8, 9 |
| Attend-Excite Objects | No training. Human evaluation on the Attend-Excite prompts of Chefer et al. (2023), category Objects. | Human Evaluation | Human evaluation by 11 participants, ratings 0 to 5 normalized to 0 to 1; number of prompts and images per prompt not stated. | 17 |
| Objects (Attend-Excite category; this table names no benchmark) | No training (CO3 on pretrained SDXL with and without CO3 components). Evaluated on the Attend-Excite prompts of Chefer et al. (2023), category Objects. | BLIP-VQA, ImageReward | not stated | 18, 20, 21 |
| RareBench Complex | No training (CO3 on pretrained SDXL). Evaluated on the Complex category of RareBench (rare or novel concept prompts). | Human Evaluation (HE), ImageReward (IR) | Human evaluation by 11 participants; number of prompts, images per prompt and seeds not stated. | 7 |
| RareBench Concat | No training (CO3 on pretrained SDXL). Evaluated on the Concat category of RareBench (rare or novel concept prompts). | Human Evaluation (HE), ImageReward (IR) | Human evaluation by 11 participants; number of prompts, images per prompt and seeds not stated. | 7 |
| RareBench Relations | No training (CO3 on pretrained SDXL). Evaluated on the Relations category of RareBench (rare or novel concept prompts). | Human Evaluation (HE), ImageReward (IR) | Human evaluation by 11 participants; number of prompts, images per prompt and seeds not stated. | 7 |
| Runtime for one sample | No training. Runtime and peak memory of SDXL, ToMe and CO3 measured on an NVIDIA A100; which prompts are timed is not stated. | Memory (GB), Time (s) | Time and peak memory to generate one sample; number of repeated runs not stated. | 18 |
| T2ICompbench Complex | No training (CO3 on pretrained SDXL). Evaluated on the Complex category of T2I-CompBench (prompts with three or more concepts). | Human Evaluation (HE), ImageReward (IR) | Human evaluation by 11 participants; number of prompts, images per prompt and seeds not stated. | 7 |

</details>
<details>
<summary><b>CoInD</b> (<a href="https://arxiv.org/abs/2503.01145">arXiv 2503.01145</a>): 12 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| CelebA orthogonal partial support, smiling male held out | Generative model trained on CelebA with all combinations of smiling and male attributes except smiling=1, male=1 (orthogonal partial support); evaluated on the unseen smiling male compositions through joint sampling and AND composition, with FID computed on the smiling male subset. | CS (%), FID, JSD | 10000 generated samples used to evaluate FID | 9 |
| CelebA Orthogonal support | Stable Diffusion v1.5 fine-tuned on data constructed from CelebA with labels converted to text; evaluated with the joint prompt 'Photo of a smiling male celebrity' and AND / NOT prompt combinations. The paper does not restate which compositions are held out for this fine-tune beyond the CelebA orthogonal support used elsewhere. | CS (%), CS improvement ratio (x), FID, JSD | not stated | 27 |
| CelebA Orthogonal support, ∧ composition ★ | Stable Diffusion v1.5 fine-tuned on data constructed from CelebA with labels converted to text; AND composition prompts 'smiling' and 'male' evaluated against the joint prompt. | CS (%), FID | not stated | 27 |
| Colored MNIST Diagonal partial support | Diffusion model trained on Colored MNIST with diagonal partial support; AND and NOT evaluations restricted to unseen compositions. | CS (%), CS improvement ratio (x) | not stated | 30, 32 |
| Colored MNIST Non Uniform support | Diffusion model trained on Colored MNIST with non-uniform support (all compositions observed, a digit with its corresponding colors at probability 0.5); evaluated by sampling from the joint and from the product of marginals, on compositions seen during training with less frequency. | CS (%), JSD | not stated | 5 |
| Colored MNIST Non-uniform support | Diffusion model trained on Colored MNIST with non-uniform support; evaluated on compositions observed during training with less frequency. | CS (%), JSD, Shannon entropy H of the color attribute | not stated | 9, 19, 32 |
| Colored MNIST Partial support | Diffusion model trained on Colored MNIST with diagonal partial support (each digit assigned only one of two colors during training); evaluation of the composition tasks is restricted to unseen compositions. | CS (%), JSD | not stated | 5, 19 |
| Colored MNIST Uniform support | Diffusion model trained on Colored MNIST with uniform support (each training image pairs a randomly selected digit and color); evaluated by sampling from the joint and from the product of learned marginals. | CS (%), JSD | not stated | 5, 19, 32 |
| Shapes3D Orthogonal partial support | Shapes3D trained on the orthogonal partial support composition split of Schott et al.; AND and NOT evaluations restricted to unseen compositions under orthogonal partial support. | CS (%), CS improvement ratio (x), R2 | not stated | 30, 32 |
| Shapes3D Partial support | Shapes3D trained on the orthogonal partial support composition split of Schott et al.; AND and NOT evaluations restricted to unseen compositions under orthogonal partial support. | CS (%), JSD, R2 | not stated | 20 |
| Shapes3D Partial support, ∧ composition ★ | Shapes3D trained on the orthogonal partial support composition split of Schott et al.; AND and NOT evaluations restricted to unseen compositions under orthogonal partial support. | CS (%), R2 | not stated | 20 |
| Shapes3D Uniform support | Shapes3D trained with full (uniform) support, consisting of all samples of the dataset; evaluated by sampling from the joint and the product of marginals. | CS (%), JSD, R2 | not stated | 20, 32 |

</details>
<details>
<summary><b>CoMat</b> (<a href="https://arxiv.org/abs/2404.03653">arXiv 2404.03653</a>): 11 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| COCO validation set | FID computed on data from the COCO validation set; the training-side COCO use is the COCO training set for the mixed latent strategy | FID-10K | 10K data from the COCO validation set (FID-10K) | 8 |
| DPG-Bench | Not trained on dense prompts; DPG-Bench is used only for evaluation | DPG, DPG, printed gain over SD1.5, DPG, printed gain over SDXL | 1065 dense prompts; images per prompt and seeds not stated | 8 |
| Image-to-text sensitivity test set | Test set built from 100 images randomly sampled from the ARO benchmark plus 100 synthetic images generated by SD1.5 and SDXL from T2I-CompBench prompts; used to score image-to-text models, no training | Parameters (B), Parameters (M), Sensitivity Score | 200 images in the test data | 20 |
| ImageNet-1K classes | Attack method run on SD1.5 and CoMat-SD1.5 with short and long prompts built from the 1,000 ImageNet-1K class names; no training split | Success rate of prompt attack (%) | 20 samples per class over 1,000 ImageNet-1K classes | 10 |
| T2I-CompBench Color ★ | Trained on the T2I-CompBench training set, all of HRS-Bench and 5,000 prompts chosen at random from ABC-6K (about 20,000 prompts); evaluation follows T2I-CompBench default settings, the evaluated split is not stated | T2I-CompBench result, T2I-CompBench result, printed gain over SD1.5, T2I-CompBench result, printed gain over SDXL | Benchmark described as 6,000 compositional prompts over 6 sub-categories, default settings; evaluated prompts per sub-category, images per prompt and seeds not stated | 7, 9 |
| T2I-CompBench Complex | Trained on the T2I-CompBench training set, all of HRS-Bench and 5,000 prompts chosen at random from ABC-6K (about 20,000 prompts); evaluation follows T2I-CompBench default settings, the evaluated split is not stated | T2I-CompBench result, T2I-CompBench result, printed gain over SD1.5, T2I-CompBench result, printed gain over SDXL | Benchmark described as 6,000 compositional prompts over 6 sub-categories, default settings; evaluated prompts per sub-category, images per prompt and seeds not stated | 7, 9 |
| T2I-CompBench Non-Spatial | Trained on the T2I-CompBench training set, all of HRS-Bench and 5,000 prompts chosen at random from ABC-6K (about 20,000 prompts); evaluation follows T2I-CompBench default settings, the evaluated split is not stated | T2I-CompBench result, T2I-CompBench result, printed gain over SD1.5, T2I-CompBench result, printed gain over SDXL | Benchmark described as 6,000 compositional prompts over 6 sub-categories, default settings; evaluated prompts per sub-category, images per prompt and seeds not stated | 7, 9 |
| T2I-CompBench Shape | Trained on the T2I-CompBench training set, all of HRS-Bench and 5,000 prompts chosen at random from ABC-6K (about 20,000 prompts); evaluation follows T2I-CompBench default settings, the evaluated split is not stated | T2I-CompBench result, T2I-CompBench result, printed gain over SD1.5, T2I-CompBench result, printed gain over SDXL | Benchmark described as 6,000 compositional prompts over 6 sub-categories, default settings; evaluated prompts per sub-category, images per prompt and seeds not stated | 7, 9 |
| T2I-CompBench Spatial | Trained on the T2I-CompBench training set, all of HRS-Bench and 5,000 prompts chosen at random from ABC-6K (about 20,000 prompts); evaluation follows T2I-CompBench default settings, the evaluated split is not stated | T2I-CompBench result, T2I-CompBench result, printed gain over SD1.5, T2I-CompBench result, printed gain over SDXL | Benchmark described as 6,000 compositional prompts over 6 sub-categories, default settings; evaluated prompts per sub-category, images per prompt and seeds not stated | 7, 9 |
| T2I-CompBench Texture | Trained on the T2I-CompBench training set, all of HRS-Bench and 5,000 prompts chosen at random from ABC-6K (about 20,000 prompts); evaluation follows T2I-CompBench default settings, the evaluated split is not stated | T2I-CompBench result, T2I-CompBench result, printed gain over SD1.5, T2I-CompBench result, printed gain over SDXL | Benchmark described as 6,000 compositional prompts over 6 sub-categories, default settings; evaluated prompts per sub-category, images per prompt and seeds not stated | 7, 9 |
| TIFA | Same training prompts as above (T2I-CompBench training set, HRS-Bench, ABC-6K subset); TIFA is evaluated at its default settings, with no TIFA training stated | TIFA, TIFA, printed gain over SD1.5, TIFA, printed gain over SDXL | 4,000 text prompts and 25,000 questions across 12 categories; images per prompt and seeds not stated | 8 |

</details>
<details>
<summary><b>CountLoop</b> (<a href="https://arxiv.org/abs/2508.16644">arXiv 2508.16644</a>): 10 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| All four benchmarks (human study) | Human study across all four benchmarks (COCO-Count, T2I-CompBench Count, CountLoop-S, CountLoop-M), comparing CountLoop, FLUX, LMD, SLD and CountGen. | Aesthetics (user score, 0 to 5), Alignment (user score, 0 to 5), Count (user score, 0 to 5), Overall (user score, 0 to 5) | 30 participants, each rating 15 blinded sets of 5 images | 11 |
| COCO-Count | Training-free method; evaluated on the COCO-Count set, an MS-COCO subset. | Ex. (%), MAE, Sp. | not stated | 10 |
| COCO-Count, T2I-CompBench, CountLoop-S and CountLoop-M | Training-free method evaluated on four sets: COCO-Count, T2I-CompBench Count, CountLoop-S and CountLoop-M. | Counting error reduction, upper end (%) | CountLoop-S 200 prompts and CountLoop-M 200 prompts; prompt counts for COCO-Count and T2I-CompBench Count, and images per prompt, not stated | 2 |
| CountLoop-M | Training-free method; evaluated on CountLoop-M, a new benchmark of multi-category prompts with 30 to 200 instances. | Ex. (%), MAE, Sp. | 200 prompts (30 to 200 instances each); images per prompt and seeds not stated | 10 |
| CountLoop-S | Training-free method; evaluated on CountLoop-S, a new benchmark of single-category prompts with 30 to 200 instances. | Additional MAE reduction (%), Error ratio (times lower), Ex. (%), MAE, MAE (approximate), MAE change, MAE plateau (approximate), MAE reduction (%), Sp., Spatial, Time saved (%), Wall-clock time (s, approximate), vs Best (%) | 200 prompts (30 to 200 instances each); images per prompt and seeds not stated | 10, 11, 12, 13 |
| CountLoop-S, 100-prompt subsample | Commercial system Nano Banana Pro evaluated on a 100-prompt subsample of CountLoop-S. | MAE, Spatial | 100 prompts (subsample of CountLoop-S); images per prompt not stated | 10 |
| CountLoop-S, large objects | CountLoop-S results broken out by object size group (large, medium, small classes), at instance counts N of 30, 75, 125 and 200. | MAE, Relative MAE (%) | not stated | 13 |
| CountLoop-S, small objects | CountLoop-S results broken out by object size group (large, medium, small classes), at instance counts N of 30, 75, 125 and 200. | Delta MAE (approximate), MAE, Relative MAE (%) | not stated | 13 |
| High-instance scenes | Training-free method evaluated on the two new high-instance benchmarks, CountLoop-S (single category) and CountLoop-M (multi-category), 30 to 200 instances per prompt. | Relative MAE reduction, lower end (%), Relative MAE reduction, upper end (%) | 200 prompts in each of the two benchmarks; images per prompt not stated | 3 |
| T2I-CompBench Count | Training-free method; evaluated on the T2I-CompBench Count set, a subset of T2I-CompBench. | Ex. (%), MAE, Sp. | not stated | 10 |

</details>
<details>
<summary><b>DomainExpansion</b> (<a href="https://arxiv.org/abs/2605.23275">arXiv 2605.23275</a>): 4 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| CLEVR (object positions) | Base model trained on the CLEVR dataset (30,000 images, 128x128 downsampled to 64x64) with one conditioning; coordinator trained with up to 2 conditionings, evaluated with up to 5 conditionings on randomly generated position sets. | Accuracy | 256 random sets of positions per number of conditionings, one generated image per set (256 generated samples per accuracy). | 3 |
| CLEVR (object positions), 5 conditionings | Base model trained on the CLEVR dataset (30,000 images, 128x128 downsampled to 64x64) with one conditioning; coordinator trained with up to 2 conditionings, evaluated with up to 5 conditionings on randomly generated position sets. | Accuracy | 256 random sets of positions per number of conditionings, one generated image per set (256 generated samples per accuracy). | 3 |
| Google Maps satellite images | Dataset of 7,300 map and satellite image pairs collected with the Google Maps API; coordinator trained on N = 96, evaluated at N = 96 and out-of-distribution N = 128. A held-out train/test split of the 7,300 pairs is not stated; FID is against the whole dataset. | FID | 3,072 generated samples compared with the whole 7,300-image satellite dataset. | 3 |
| Slakh2100 | Base model and coordinators trained on the Slakh2100 training set (1,500 tracks); coordinator trained on 48 s tracks (Ltrain = 5), evaluated on 4l (48 s) and 10l (120 s) tracks (Ltest = 13); FAD is computed against middle parts cut from the training dataset tracks of the same length. | FAD, Size (M parameters) | 128 generated tracks per reported FAD value; the paper notes 128 vs 1,024 tracks changes FAD by a small margin. | 3 |

</details>
<details>
<summary><b>Du</b> (<a href="https://arxiv.org/abs/2302.11552">arXiv 2302.11552</a>): 5 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| 2D densities (mixture) | Diffusion models trained on each component distribution (two sets of 3 Gaussians); evaluation samples the mixture. No held-out split stated. | LL, Var, ln(MMD) | not stated | 6 |
| 2D densities (product) | Two separate diffusion models are trained, one per component distribution (8-Gaussian ring, uniform strip); evaluation samples the product of the two. No held-out split stated. | LL, RAISE, Var | not stated | 6 |
| CLEVR (cube positions) | Trained on a generated dataset of 200,000 64x64 CLEVR images with 1 to 5 cubes, conditioning on the location of one shape; evaluation composes the model with itself over several cube positions. No held-out split stated. | Accuracy | not stated | 6 |
| ImageNet | Models trained with each energy parameterization on ImageNet; evaluation is unconditional generation. | FID | FID on 1k samples. | 17 |
| ImageNet 128x128 | Unconditional diffusion model trained on ImageNet at 128x128 (1 million iterations), with a noise-conditioned classifier; evaluation samples class-conditioned images. No held-out split stated. | Accuracy, FID, Inception Score | Inception Score and FID calculated on 50000 generated samples; sample count behind classifier accuracy not stated. | 7 |

</details>
<details>
<summary><b>FKC</b> (<a href="https://arxiv.org/abs/2503.02819">arXiv 2503.02819</a>): 12 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| 100 protein target pairs (P1, P2) | Pretrained dual-pocket diffusion model with FKC at inference; evaluated on 100 protein pairs, with 5 molecule sizes sampled from the original training set of Guan et al. (2023) | (P1 * P2) docking score product, Better than ref. (fraction), Div., P1 docking score, P2 docking score, QED, SA, Val. & Uniq., max(P1, P2) docking score, mean docking score of the reference molecules | 100 protein pairs, 32 ligands for each of 5 molecule lengths per pair; values are averages over all generated molecules in each run | 9 |
| GenEval (1000 prompts) | SDXL with classifier-free guidance plus FKC at inference (no training stated); evaluated on 1000 GenEval prompts | Color attribution, Colors, Counting, Overall, Position, Single object, Two object | 1000 prompts; images per prompt and seeds not stated; mean values reported | 39 |
| GenEval (553 prompts) ★ | SDXL with classifier-free guidance plus FKC at inference (no training stated); evaluated on the GenEval original 553 prompts | Color attribution, Colors, Counting, Overall, Position, Single object, Two object | 553 prompts; images per prompt and seeds not stated; mean values reported | 39 |
| ImageNet-512 | EDM2-XS model trained on ImageNet-512 (pretrained, FKC resampling applied at inference); which classes or prompts are evaluated is not stated | CLIP, IR (ImageReward) | CLIP and ImageReward scores averaged over 10,000 images | 7, 38 |
| LDMol molecules, property pair GSK3b / DRD2 | Pretrained LDMol latent diffusion model (text-conditioned) with FKC at inference; prompted with the text 'This molecule inhibits {protein_name}' for the property pair; no training split stated | (P1, P2) top-1, P1, (P1, P2) top-1, P2, Div., P1 top-10, P2 top-10, Qual., Val. & Uniq. | Top-10 molecules (highest P1*P2) from a batch of 512, averaged over 5 runs; top-1 molecule reported overall | 42, 44 |
| LDMol molecules, property pair JNK3 / DRD2 | Pretrained LDMol latent diffusion model (text-conditioned) with FKC at inference; prompted with the text 'This molecule inhibits {protein_name}' for the property pair; no training split stated | (P1, P2) top-1, P1, (P1, P2) top-1, P2, Div., P1 top-10, P2 top-10, Qual., Val. & Uniq. | Top-10 molecules (highest P1*P2) from a batch of 512, averaged over 5 runs; top-1 molecule reported overall | 42, 44 |
| LDMol molecules, property pair JNK3 / GSK3b | Pretrained LDMol latent diffusion model (text-conditioned) with FKC at inference; prompted with the text 'This molecule inhibits {protein_name}' for the property pair; no training split stated | (P1, P2) top-1, P1, (P1, P2) top-1, P2, Div., P1 top-10, P2 top-10, Qual., Val. & Uniq. | Top-10 molecules (highest P1*P2) from a batch of 512, averaged over 5 runs; top-1 molecule reported overall | 42, 44 |
| LJ-13 (Lennard-Jones system of 13 particles) | DEM model trained at base temperature TL = 2, then annealed at inference to lower target temperatures; evaluated against reference MCMC samples (20k samples from 10 chains per temperature) | Distance-W2, Energy-W1, Energy-W2, distance-W2 | Mean and standard deviation over 3 seeds (Table 3); metrics computed over 3 runs, with DEM run for one seed only (Table A2) | 8, 36 |
| Mixture of 40 Gaussians | No training data: ground-truth scores of the mixture are computed analytically; evaluated by sampling from an annealed distribution with inverse temperature beta = 3 | Energy-W2, MMD, Total Var, W1, W2 | 5 runs with 10k samples each | 36 |
| One protein target pair (UniProt P23786/P05023) | Single protein pocket pair (UniProt P23786/P05023) used as the validation set for SDE-setting ablations; molecule sizes sampled from the original training distribution of Guan et al. (2023) | (P1 * P2) docking score product, Better than ref. (fraction), Div., P1 docking score, P2 docking score, Qual., Val. & Uniq., max(P1, P2) docking score, mean docking score of the reference molecules | 32 molecules for each of 5 molecule sizes for one protein pair per setting | 40 |
| Protein pair ATP1A1 (P1) and CPT2 (P2) | Pretrained LDMol with FKC; molecules generated from text prompts of binding to ATP1A1 (UniProt P05023) and CPT2 (UniProt P23786), evaluated by AutoDock Vina docking | (P1, P2) top-1, P1 docking score, (P1, P2) top-1, P2 docking score, (P1, P2) top-10, P1 docking score, (P1, P2) top-10, P2 docking score, Div., docking score against ATP1A1, docking score against CPT2 | 32 molecules generated (Tempered Noise SDE, beta = 1.5) | 43 |
| Protein target pairs shown as examples | Display only: best-docking molecules for the protein pair GRM5/RRM1 (UniProt P41594/P23921) drawn from the main dual-target experiment; training split not stated for this display | Spearman rank correlation between FKC weight and docking score | Main experiment setup: 5 molecule sizes with 32 molecules per size; the number shown per pair is the best-docking molecule at each size | 40 |

</details>
<details>
<summary><b>FlipConcept</b> (<a href="https://arxiv.org/abs/2502.15203">arXiv 2502.15203</a>): 1 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| Custom Concept 101 plus images generated by Stable Diffusion (five evaluation sets) | Method with no training stated; evaluated on the Custom Concept 101 dataset plus additional Stable Diffusion images, using five evaluation sets with ChatGPT-generated prompts. | Additional training (s), I_CLIP, I_DINO, Inference (s), T_CLIP, Total (s) | 50 images generated per method | 4, 5, 6 |

</details>
<details>
<summary><b>FreeFuse</b> (<a href="https://arxiv.org/abs/2510.23515">arXiv 2510.23515</a>): 6 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| All available LoRAs in the LoRAShop test set | No training. LoRA perturbation magnitude measured across FLUX double-stream blocks on all available LoRAs in the LoRAShop test set. | Effective LoRA weight perturbation, mid/deep vs early (ratio of //dW//_F/sqrt(/dW/)), Inference-time LoRA residual, mid/deep vs early (ratio of //dx//_F///x_base//_F) | not stated | 6 |
| FreeFuse benchmark of 15 LoRAs (10 character, 5 object) | No training (FreeFuse is training-free, using 15 existing subject LoRAs: 10 character, 5 object). Evaluation set: 20 multi-character scenes each using 4 concurrent LoRAs, plus character-object prompts. | Character Similarity Clip-I, Character Similarity DINOv2, Character Similarity DINOv3, Character Similarity Dreamsim, Clip-T, Face Similarity ArcFace, Face Similarity LVFace, HPSv2, HPSv3, Object Similarity Clip-I, Object Similarity DINOv2, Object Similarity DINOv3, Object Similarity Dreamsim | 470 images generated for each method. | 10 |
| FreeFuse latency measurement, 2 to 6 subject LoRAs | No training. Latency measured with 2 to 6 subject LoRAs on a single L20; which prompts are timed is not stated. | Inference latency (seconds), Speedup vs. LoRAShop (%), Speedup vs. LoRAShop (x) | not stated | 11, 26 |
| FreeFuse LoRA-count ablation, k in {2,3,4,5} subject LoRAs | No training. Controlled ablation with k in {2,3,4,5} subject LoRAs; the baseline and FreeFuse use the same LoRAs and prompts. | ArcFace ID, ArcFace ID delta (FreeFuse minus Baseline), HPSv2, HPSv2 delta (FreeFuse minus Baseline) | not stated | 24 |
| FreeFuse user study of 40 cases (20 character-character, 20 character-object) | No training. Blind ranking of anonymized, shuffled outputs on 40 cases (20 character-character, 20 character-object). | User Study Q1 (mean rank), User Study Q2 (mean rank) | 17 participants aged 18-40, each ranking methods per case; score is the average of participant rankings across 40 cases. | 10 |
| Public LoRAShop dual-subject protocol | No training. Evaluated on the public dual-subject protocol of LoRAShop, using its LoRA weights and prompting setup. | Aesthetic v2, ArcFace ID, CLIP-T bigG, HPSv2 | not stated | 25 |

</details>
<details>
<summary><b>LiftScores</b> (<a href="https://arxiv.org/abs/2505.13740">arXiv 2505.13740</a>): 7 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| 2D synthetic | Training: per component distribution, 8000 data points sampled randomly and one diffusion model trained per distribution (3-layer MLP). Evaluation: every method generates 8000 samples; CompLift accepts or rejects 8000 Composable Diffusion samples. | Accuracy (%), CD (Chamfer Distance), Ratio (acceptance ratio of samples using CompLift) | 8000 samples generated per method; CompLift filters 8000 Composable Diffusion samples (50 cached trials, or 50 or 1000 vanilla trials). | 7, 19 |
| Attend-and-Excite Animals | No training (pretrained Stable Diffusion 1.4, 2.1 and SDXL). Evaluated on the Attend-and-Excite composed prompts of the animal-animal class: 66 unique animal combinations. | CLIP, CLIP gain (%), IR (ImageReward), IR gain (%), TIFA, minCLIP | 5 initial samples per prompt over 66 unique animal combinations; cached CompLift uses 50 trials, vanilla CompLift 200 trials; scores averaged within the combination class. | 8, 23, 24, 27, 28 |
| Attend-and-Excite Object&Animal | No training (pretrained Stable Diffusion 1.4, 2.1 and SDXL). Evaluated on the Attend-and-Excite composed prompts of the animal-object class: 144 unique animal-object combinations. | CLIP, CLIP gain (%), IR (ImageReward), IR gain (%), TIFA, minCLIP | 5 initial samples per prompt over 144 unique animal-object combinations; cached CompLift uses 50 trials, vanilla CompLift 200 trials; scores averaged within the combination class. | 8, 23, 24, 27, 28 |
| Attend-and-Excite Objects | No training (pretrained Stable Diffusion 1.4, 2.1 and SDXL). Evaluated on the Attend-and-Excite composed prompts of the object-object class: 66 unique object combinations. | CLIP, CLIP gain (%), IR (ImageReward), IR gain (%), TIFA, minCLIP | 5 initial samples per prompt over 66 unique object combinations; cached CompLift uses 50 trials, vanilla CompLift 200 trials; scores averaged within the combination class. | 8, 23, 24, 27, 28 |
| Attend-and-Excite prompts | No training (same Stable Diffusion models). Evaluated on the Attend-and-Excite composed prompts (3 classes: animal-animal, animal-object, object-object). | Average gain (%) | not stated | 27 |
| CLEVR Position | CLEVR Position: pretrained diffusion model of Liu et al. (2022), no new training. Evaluated on 5000 combinations of 1 to 5 positions; FID is computed against the training set provided by Liu et al. | Accuracy (%), Accuracy (%), pretrained classifier, FID | 5000 combinations of positions with various numbers of constraints (paper does not say whether 5000 is per constraint number); each composition starts from 10 Composable Diffusion samples; CompLift uses 1 trial at t=928. | 21, 22 |
| Figure 10 samples, prompt "a black car and a white clock" | No training. 100 samples from Figure 10 (prompt a black car and a white clock) labelled by hand for presence of both a black car and a white clock; metrics are scored against these labels. | PR AUC, ROC AUC | 100 hand-labelled samples: 62 positive and 38 negative. | 28 |

</details>
<details>
<summary><b>Liu</b> (<a href="https://arxiv.org/abs/2206.01714">arXiv 2206.01714</a>): 4 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| CelebA-HQ | 30,000 CelebA-HQ image-label pairs used to train the scoring classifiers, split into 24,000 training and 6,000 validation pairs | Validation accuracy (%) | 6,000 validation image-label pairs for the classifiers; generated images evaluated per method are not counted separately for CelebA-HQ | 20 |
| CLEVR | Generator trained on the CLEVR training set of 30,000 images at 128x128; classifier for scoring trained on 24,000 of the 30,000 image-label pairs and validated on 6,000; evaluation on 1, 2 and 3 component compositions | Acc (%), Accuracy advantage over EBM (%), Accuracy advantage over the best baseline (%), FID, Validation accuracy (%) | 5,000 generated images per test setting per method; FID uses 5,000 images | 3, 11, 12, 20 |
| FFHQ | FFHQ with 51,067 images labelled with smile, gender and glasses; train and test split for the generator is not stated; scoring classifiers trained on CelebA-HQ pairs | Acc (%), FID | 5,000 generated images per test setting per method; FID uses 5,000 images | 20 |
| Relational CLEVR | Training dataset of 50,000 images at 128x128 with one relation label each; classifier trained on a split of 40,000 training and 10,000 validation images; evaluation on 1, 2 and 3 component compositions | Acc (%), FID, Validation accuracy (%) | 5,000 generated images per test setting per method; FID uses 5,000 images | 12, 20 |

</details>
<details>
<summary><b>LogicalGuidance</b> (<a href="https://arxiv.org/abs/2602.05549">arXiv 2602.05549</a>): 5 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| CelebA | Model trained on CelebA (Table 9); no train/test split stated. Evaluation restricted to two binary attributes, Blond/Non-blond and Male/Female. | Accuracy (%), CS (Conformity Score), FID | FID: 5,000 samples per task (100 images for each of 50 queries). Conformity score: Appendix E.2 gives 10,000 samples from 100 unique queries per task type. | 6, 23 |
| CMNIST | Diffusion model trained on CMNIST (hyperparameters in Table 9); no train/test split stated. Evaluation uses randomly generated logical queries (N=2 to 5 operators, NOT, OR-CI, OR-ME, AND), 100 unique queries per task type. | Accuracy (%), CS (Conformity Score), Conformity score (%) | 10,000 samples from 100 unique queries per task type with 100 samples each. | 5, 7, 23, 26 |
| CMNIST, AND ★ | Diffusion model trained on CMNIST (hyperparameters in Table 9); no train/test split stated. Evaluation uses randomly generated AND queries (100 unique queries per task type). | CS (Conformity Score), Conformity score (%) | 10,000 samples from 100 unique queries per task type with 100 samples each. | 5, 7, 26 |
| GRM5-RRM1 dual-target ligand design | Uses the experimental setup of Skreta et al. 2025a on the GRM5-RRM1 protein pair; no training split stated for this paper. | (A*B), A (docking score), B (docking score), Delta(A,B), Div., Qual., V. & U. | 32 ligands of size 23 generated over 8 experiments. | 8 |
| Shapes3D | Diffusion model trained on Shapes3D (hyperparameters in Table 9); no train/test split stated. Evaluation uses randomly generated logical queries, 100 unique queries per task type. | Accuracy (%), Conformity score (%) | 10,000 samples from 100 unique queries per task type with 100 samples each. | 5, 7, 23 |

</details>
<details>
<summary><b>MultiDiffusion</b> (<a href="https://arxiv.org/abs/2302.08113">arXiv 2302.08113</a>): 2 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| COCO validation subset | No training; evaluated on a subset of the COCO validation set with 2 to 4 foreground objects, excluding people and masks occupying less than 5% of the image. | IoU | 1K samples (one per COCO validation example in the subset); images per example not stated. | 7 |
| Panoramas from 8 text prompts | No training; MultiDiffusion applied to a pretrained Stable Diffusion; for each text prompt the reference set is N images of 512x512 from the base model, compared with 512x512 crops of generated panoramas, repeated for 8 text prompts. | CLIP-aesthetic, CLIP-score, FID | N = 2000 generated panoramas (one random 512x512 crop each) per prompt, repeated over 8 text prompts. | 5 |

</details>
<details>
<summary><b>PiCo</b> (<a href="https://arxiv.org/abs/2505.03203">arXiv 2505.03203</a>): 5 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| CC-500 and T2I-Compbench (500 prompts) | Evaluation prompts only: 100 random prompts from CC-500 plus 100 prompts each from T2I-CompBench color, shape, texture and complex; no training split stated | Alignment: Loss (%), Alignment: Tie (%), Alignment: Win (%), Quality: Loss (%), Quality: Tie (%), Quality: Win (%), Random: Both (%), Random: None (%), Selected: Both (%), Selected: Both change from Random (percentage points), Selected: None (%), Selected: None change from Random (percentage points) | 500 randomly selected prompts, 10 unique seeds per prompt, 5,000 images in total | 5 |
| CC-500 and T2I-Compbench (subset not restated for this table) | Evaluation prompts only: 100 random prompts from CC-500 plus 100 prompts each from T2I-CompBench color, shape, texture and complex; no training split stated | Both (%), Both/Single (%), DINO, None (%), Runtime per prompt (s), Single (%) | Human annotation on 500 randomly sampled generated images with two objects; 500 prompts with 10 seeds each (5,000 images) were generated | 5, 9, 10 |
| MS-COCO | CLIP-I and FID tested on the MSCOCO dataset; the split is not stated for the main comparison | CLIP-I, FID | not stated | 5 |
| MS-COCO val (5k image-text pairs) | FID assessed on image-text pairs from the MS-COCO val set | FID | 5k image-text pairs from MS-COCO val set | 10 |
| T2I-Compbench (400 prompts) | Objective metrics on 400 prompts from T2I-CompBench; no training split stated | CLIP-S, CLIPScore, Random: DINO, Selected: DINO, Selected: DINO change from Random, VQAScore | 4000 images from 400 T2I-CompBench prompts | 5, 7 |

</details>
<details>
<summary><b>RTD</b> (<a href="https://arxiv.org/abs/2608.03135">arXiv 2608.03135</a>): its 2 target settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| T2I-CompBench Complex ★ | No training (Rectify Then Diffuse on pretrained SDXL); the 1,000 prompts of the complex compositional split. | Human Evaluation (HE), ImageReward (IR), S-IoU5 | 1,000 prompts, three fixed seeds per prompt (3,000 images). Human evaluation by 11 participants. | 4, 6 |
| RareBench Concat ★ | No training (Rectify Then Diffuse on pretrained SDXL); the Concat category of RareBench. | Human Evaluation (HE), ImageReward (IR) | RareBench holds 120 prompts across Concat, Relations and Complex, and the Concat share is not stated. Three fixed seeds per prompt. Human evaluation by 11 participants. | 4, 6 |

</details>
<details>
<summary><b>Sculpting</b> (<a href="https://arxiv.org/abs/2309.16115">arXiv 2309.16115</a>): 3 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| 2D grid domain, 32 x 32 | Base GFlowNets trained on reward functions defined on the 32 x 32 grid; evaluated against the ground-truth composition computed from the known base probabilities | L1 distance to the ground-truth composition | not stated | 29 |
| MNIST image generation with diffusion models | Three diffusion models pre-trained on colored MNIST digits: p1 cyan digits less than 4, p2 cyan and beige digits less than 2, p3 cyan and beige even digits less than 4; evaluated by composing them | Batch size, Batch size (trajectories per base model), Data generation share of classifier training time (%), Data generation time, minutes part (min), Data generation time, seconds part (s), Elapsed real time, minutes part (min), Elapsed real time, seconds part (s), Time-steps per trajectory, Total elapsed real time, minutes part (min), Total elapsed real time, seconds part (s), Training steps | not stated | 35 |
| Molecule generation with SEH, SA and QED rewards | Base GFlowNets trained with reward functions SEH, SA and QED (no dataset split); evaluated by sampling from each base and composed model | Average pairwise Tanimoto similarity, Batch size, Batch size (trajectories per base model, all states used), Data generation share of classifier training time (%), Data generation time, hours part (h), Data generation time, minutes part (min), Data generation time, seconds part (s), Earth mover's distance, Elapsed real time, hours part (h), Elapsed real time, minutes part (min), Elapsed real time, seconds part (s), Generated molecules above the reward threshold (count), Share of samples in reward bin (%), Tanimoto-separated modes above the reward threshold (count), Total elapsed real time, hours part (h), Total elapsed real time, minutes part (min), Total elapsed real time, seconds part (s), Training steps | 5 000 molecules sampled from each composed distribution and from each base GFlowNet; top 1 000 molecules used for average pairwise similarity in the diversity statistics | 13, 35, 36 |

</details>
<details>
<summary><b>SuperDiff</b> (<a href="https://arxiv.org/abs/2412.17762">arXiv 2412.17762</a>): 6 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| CIFAR-10 disjoint halves | Two diffusion models trained on two disjoint halves of CIFAR-10 (first 5 labels and last 5 labels), then combined at generation time. | FID, ODE inference, FID, SDE inference, FLD, ODE inference, FLD, SDE inference, IS, ODE inference, IS, SDE inference | not stated | 8 |
| CIFAR-10 split into two random partitions A and B | Two conditional diffusion models trained on two random partitions of the CIFAR-10 training data (labeled A and B). | FID, ODE inference, FID, SDE inference, FLD, ODE inference, FLD, SDE inference, IS, ODE inference, IS, SDE inference | not stated | 30 |
| LDMol molecules, property pair GSK3b / JNK3 | No training; pre-trained LDMol text-conditioned molecule diffusion model, one prompt per property, combined at inference. | Div., Qual., Top-1 P1, Top-1 P2, Top-10 (P1*P2), Top-10 P1, Top-10 P2, Top-10 min(P1,P2), V&U | Metrics from 5 runs of batch-size 512 | 10 |
| LDMol molecules, property pair GSK3b / QED | No training; pre-trained LDMol text-conditioned molecule diffusion model, one prompt per property, combined at inference. | Div., Qual., Top-1 P1, Top-1 P2, Top-10 (P1*P2), Top-10 P1, Top-10 P2, Top-10 min(P1,P2), V&U | Metrics from 5 runs of batch-size 512 | 10 |
| Proteins generated de novo (Proteus and FrameDiff checkpoints) | No training; two pre-trained protein diffusion models (Proteus and FrameDiff) combined at generation time, unconditional generation at lengths 100 to 300. | < 0.3 scTM (fraction), < 0.5 scTM (fraction), < 2A scRMSD (fraction designable), Frac. beta, Max. cluster, Max. scTM, Pairwise scTM, scRMSD | 50 generated proteins at each of the lengths 100, 150, 200, 250, 300 | 9 |
| Stable Diffusion concept pairs (20 pairs) | No training; pre-trained Stable Diffusion v1-4 prompt-conditioned models combined at inference, evaluated on 20 concept pairs. | Density evaluation time (seconds per image), Max. CLIP, Max. ImageReward, Max. TIFA, Min. CLIP, Min. ImageReward, Min. TIFA, /Delta/ CLIP, /Delta/ ImageReward, /Delta/ TIFA | 20 images for each of 20 concept pairs | 8, 33 |

</details>
<details>
<summary><b>TestTime</b> (<a href="https://arxiv.org/abs/2605.07078">arXiv 2605.07078</a>): 2 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| CelebA 23 OOD compositions | Backbone trained on 51 seen CelebA classes (26,775 train images, 3,825 validation images); 23 held-out OOD classes (9,914 images) used only at test time; evaluated on those 23. | CLIP (%), F1 (%), FID, Precision (%), Recall (%) | 100 query images per OOD class, framework run independently on each query; reference sets of 100 images (the queries, or 100 other held-out images); 16 saved generations per query per teacher and seed 42 (hyperparameter table); FID computed at N=100; mean and standard error over 23 compositions. The number of generated images behind one FID is not separately stated. | 9 |
| ColorMNIST 40 OOD compositions | Backbone trained on the 120 seen (digit, digit color, background color) slots of ColorMNIST (1,000 images per slot); the 40 unseen slots are held-out OOD compositions used only at test time; evaluated on those 40. | CLIP (%), F1 (%), FID, Precision (%), Recall (%) | 100 query images per OOD class, framework run independently on each query; reference sets of 100 images (the queries, or 100 other held-out images); 16 saved generations per query per teacher and seed 42 (hyperparameter table); FID computed at N=100; mean and standard error over 40 compositions. The number of generated images behind one FID is not separately stated. | 8 |

</details>
<details>
<summary><b>ToMe</b> (<a href="https://arxiv.org/abs/2411.07132">arXiv 2411.07132</a>): 6 settings</summary>

| Dataset | Split | Metrics | Count | Pages |
|---|---|---|---|---|
| GPT-4o Benchmark | Training-free method on SDXL; evaluated on a new GPT-4o Benchmark of 50 prompts built from the template 'a [objectA] with a [itemA] and a [objectB] with a [itemB].' | GPT-4o | 50 prompts; images per prompt and seeds not stated | 8 |
| T2I-CompBench all three subsets | Training-free method on SDXL; BLIP-VQA scored on the 300 validation prompts of each T2I-CompBench subset (color, texture, shape). | Time Cost (s) | 300 validation prompts per subset; images per prompt and seeds not stated | 17 |
| T2I-CompBench Color | Training-free method on SDXL; evaluated on the 300 validation prompts of this T2I-CompBench subset. | BLIP-VQA, Human-preference (ImageReward) | 300 validation prompts; images per prompt and seeds not stated | 8, 9, 17, 21 |
| T2I-CompBench Shape ★ | Training-free method on SDXL; evaluated on the 300 validation prompts of this T2I-CompBench subset. | BLIP-VQA, Human-preference (ImageReward) | 300 validation prompts; images per prompt and seeds not stated | 8, 9, 17, 21 |
| T2I-CompBench Texture ★ | Training-free method on SDXL; evaluated on the 300 validation prompts of this T2I-CompBench subset. | BLIP-VQA, Human-preference (ImageReward) | 300 validation prompts; images per prompt and seeds not stated | 8, 9, 17, 21 |
| Token embeddings of queen, king, woman and man | No training or evaluation set; PCA of text token representations for the prompts of the additivity examples (template 'a photo of a {object}'). | Cosine similarity | not stated | 5 |

</details>

## How this was checked

<details>
<summary>How the split and count of each setting were read and verified</summary>

**Reading.** Four readers took 164 of the settings to the rival's own PDF, appendices included, and copied
one sentence as evidence for each split and count. Their records, with every quote and page, are in
[the checked reads](benchmark-spec-reads/checked.json).

**Quotes.** A script found all 290 quotes on the pages cited: 164 for splits and 126 for counts. The other
38 counts are `not stated`, so they have no quote.

**Target rows.** The count on the target rows for CoMat, CoInD's fine-tuned CelebA table, Projective Composition and Rectify
Then Diffuse describes the printed metric on that row, read from the page by hand. Rectify Then Diffuse's two settings were
read the same way.

**One disagreement inside a paper.** Logical Guidance's appendix E.2 (page 23) gives the conformity score 10,000 samples from
100 queries per task type, without naming a dataset. Its page 24 gives CelebA FID 5,000 samples per task from 50 queries of
100 images each. Both are kept in the row for CelebA, and neither was chosen.

**What this page does not establish.** A rival's own code may fix a setting its paper leaves out. Code was not read for
this page.

</details>

## Cross-references

- The mention of **benchmark spec**, the page listing what each rival ran, written in the field-learning step, in [the route to two papers, under The steps](../../../plans/route-two-papers.md#the-steps).
