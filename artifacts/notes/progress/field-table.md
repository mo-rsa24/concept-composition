# Field table: who else composes diffusion models

This page lists the 40 papers a reviewer would expect the two papers to cite or beat, each checked
against its own PDF. Paper 1 composes diffusion models trained apart. Paper 2 makes one text-to-image model
draw two separate objects where a product of experts fuses them.

Claude and GPT-6 Astra scouted separately from one [shared brief](field-scouts/brief.md). Each returned 30
papers, 21 of them the same, and each found 9 the other missed. Across the 60 scout
rows, 52 headline numbers are right, 2 are partly right, none is wrong and
6 rows claim no number.

**Needs joint data** means the method uses an image or a prompt showing both concepts together, at training or
at sampling. **Closeness** says how directly a paper competes: a direct rival attacks the same problem and
prints a number we could be compared on, and a partial rival overlaps on part of the setting.

## Table of contents

- [What the check changes](#what-the-check-changes)
- [Paper 1: models trained apart](#paper-1-models-trained-apart)
- [Paper 2: two objects from one model](#paper-2-two-objects-from-one-model)
- [Papers that bear on both](#papers-that-bear-on-both)
- [Close to the topic, outside both problems](#close-to-the-topic-outside-both-problems)
- [How each row was checked](#how-each-row-was-checked)

## What the check changes

**Projective Composition holds the strongest CLEVR count near Paper 1's setting.** Its model sees only images of
one object and is composed across locations at sampling. In
[its Table 1 on page 17](https://arxiv.org/pdf/2502.04549#page=17), 98 of 100 hand-counted images show all 5
objects in the right places on an empty background. Paper 1's accepted CLEVR target rests on a different number:
Diffusion Domain Expansion's accuracy of 44.5 out of 100 at 5 positions, judged by a classifier on 256 samples.
The images, training sets and judges differ, so the two numbers cannot be ranked. The paper calls its composed
parts separate models (page 3), while
[its training appendix on page 15](https://arxiv.org/pdf/2502.04549#page=15) describes a model conditioned on x
and y coordinates, so the locations may be one network rather than networks trained apart.

**Diffusion Domain Expansion does not work in Paper 1's setting.** It trains one base model on all 30,000 CLEVR
images, each holding several objects, then trains a small coordinator on that model's outputs
([appendix B.2 on page 11](https://arxiv.org/pdf/2605.23275#page=11)). The session brief lists it as a direct
rival for Paper 1, and that disagreement waits for the owner.

**Four printed numbers need their context when quoted.** Feynman-Kac Correctors' Table 2 prints two results for
one setting because they use batch sizes 32 and 8. CoInD's text says "> 4×" where its Table 2 gives 8.63 against
2.51, which is 3.44×. Rectify Then Diffuse's 45.8% gain is against CO3, and against ToMe, the best other method in
that table, the gain is 16.5%. GenEval 2's Table 1 prints different Gemini human scores with and without prompt
rewriting (93.1 and 96.7), so a deviation must be read from its own column.

## Paper 1: models trained apart

8 papers.

| Paper | Found by | Closeness | Needs joint data | What it does | Main number | Where |
|---|---|---|---|---|---|---|
| [Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts](https://arxiv.org/abs/2503.02819) | both | partial rival | no | Adds Feynman-Kac importance weights to a diffusion sampler and resamples particles so samples track an annealed, geometric-mean or product target built from pretrained score models. | EDM2 on ImageNet-512, gamma 40, N 32, 10,000 images: EDM2+CFG CLIP 28.75, IR -0.24; EDM2+FKC CLIP 29.00, IR 0.04 at batch size 32 and CLIP 29.14, IR 0.05 at batch size 8 | page 7, Table 2; batch sizes on page 38, section F.4 and Table A3 |
| [Composing and Steering Diffusion Models with Generalized Feynman-Kac Correctors](https://arxiv.org/abs/2610.40383) | Claude | partial rival | no | Generalizes Feynman-Kac correctors with weighted particles so that samples follow an arbitrary function of several pretrained diffusion models' densities, with optional reward steering. | No comparable number is printed: MNIST results are assessed qualitatively, and Gaussian KL and L2 error against particle count appear only as a plot. | Page 12 (qualitative statement); page 10, Figure 2 |
| [Composition and Control with Distilled Energy Diffusion Models and Sequential Monte Carlo](https://arxiv.org/abs/2502.12786) | Claude | partial rival | partly | Distills pretrained diffusion models into energy functions and uses their potentials in sequential Monte Carlo to compose separately trained models. | No composition number is printed: the CelebA male AND glasses result is shown as images only. | Page 8, Section 5.3 and Figure 7; page 19, Appendix E.2 |
| [Mechanisms of Projective Composition of Diffusion Models](https://arxiv.org/abs/2502.04549) | both | theory | no | Defines a composition as a distribution that matches each source model on that model's own projected features, and proves that a linear combination of scores reaches it when the sources have factorized conditionals. | CLEVR, composing N = 4 single-object location models, 100 hand-counted samples: 98 of 100 images had all 4 objects in the correct locations with an empty background, against 0 of 100 with the unconditional background (77 had no correct object, 23 had one). | Page 17, Table 1 (Appendix B.3) |
| [Catastrophic Compositional Generation: Why Vanilla Diffusion Models Fail to Extrapolate](https://arxiv.org/abs/2606.23920) | both | theory | no | Separates sampler approximation error from score estimation error and shows that when the composed target lies outside the training support, score error dominates and Feynman-Kac correction makes it worse. | Non-factorized out-of-distribution Gaussian, model trained on N = 10000 samples: sliced Wasserstein-2 distance to the target rises from 0.8518 with K = 1 particle to 2.8322 with K = 256, while with analytic scores it falls from 0.6375 to 0.0820. | Page 10, Table 1(d), Section 5.1 |
| [Compositional Sculpting of Iterative Generative Processes](https://arxiv.org/abs/2309.16115) | both | tool | no | Trains a classifier on samples from pre-trained models and uses classifier guidance to sample their harmonic-mean or contrast composition. | Molecule GFlowNets at beta 32, 5,000 samples per model: 49% of samples from the composition y={SEH,SA,QED} fall in the bin high on all three rewards, against 5% for p_SEH, 5% for p_SA and 13% for p_QED | page 13, Table 1, row (d) |
| [MCMC-Correction of Score-Based Diffusion Models for Model Composition](https://arxiv.org/abs/2307.14012) | Claude | tool | no | Adds a Metropolis-Hastings-like acceptance step to score-based diffusion by estimating energy differences through line integrals of the score, so MCMC correction works without an energy model. | CIFAR-100 classifier-full guidance, 50k samples, score model: FID 13.3 with HMC-3C against 31.8 with the reverse process. | Page 13, Table 3 |
| [Compositional Generative Modeling from Decentralized Data (DCFM)](https://arxiv.org/abs/2606.10153) | Loop 14 | partial rival | no raw joint data, but each expert trains against the other experts' frozen models | Trains flow-matching experts on separate data silos, exchanges the models, and adds a conditional-independence penalty across peers (DCFM-A) or distils them into one student on their own samples (DCFM-B), so attribute combinations no silo saw can be sampled. | Colored MNIST, 10 clients, coverage 1/2, non-IID: novel-combination FID 9.15 for DCFM-B and 9.29 for DCFM-A against 17.12 to 38.27 for the six baselines. Chest X-ray results exist only as points in Figure 9 | Page 7, Table 1; page 8, Figure 9 |

## Paper 2: two objects from one model

21 papers.

| Paper | Found by | Closeness | Needs joint data | What it does | Main number | Where |
|---|---|---|---|---|---|---|
| [Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC](https://arxiv.org/abs/2302.11552) | both | direct rival | partly | Composes diffusion model scores and samples the product with annealed MCMC, using an energy parameterization so Metropolis-corrected samplers can run. | CLEVR, 5 cube positions, fraction of samples with all objects correct: 72.7 for EBM with HMC against 57.4 for the score model with reverse sampling. | Page 6, Table 2; metric defined on page 7 |
| [Steer Away From Mode Collisions: Improving Composition In Diffusion Models](https://arxiv.org/abs/2509.25940) | both | direct rival | yes | During early sampling, CO3 boosts the joint prompt's prediction and subtracts the single-concept predictions in Tweedie-denoised space, so the sample moves away from modes dominated by one concept. | Attend-and-Excite Animal-Animal prompts, BLIP-VQA: 0.7441 for CO3 against 0.6950 for SDXL and 0.6980 for Attend-and-Excite. | Page 7, Table 1 |
| [Rectify Then Diffuse: Disentangling Concepts Before Denoising Trajectory Unfolds](https://arxiv.org/abs/2608.03135) | Claude | direct rival | yes | Runs one pilot pass to get each concept's attention map, then makes one norm-bounded update to the starting noise that reduces overlap between the maps, and samples normally. | AE-Bench object-object prompts on SDXL: BLIP-VQA 0.7503 for RTD against 0.5146 for CO3 (45.8% higher) and 0.6440 for ToMe, the best other method (16.5% higher) | page 6, Table 1; percentages on page 5 |
| [Training-free Conditional Object Composition for Multi-object Image Generation with Diffusion Models](https://arxiv.org/abs/2611.00804) | Astra | direct rival | yes | Generates all but one of the requested objects first, then inpaints the remaining object into free space with layout guidance, choosing which object to hold back from attention maps or a detector. | MS COCO object sets, N = 6, success rate over 5 seeds: 35.8 +/- 5.3% for condComp (objDetect) against 19.2 +/- 3.3% for SDXL and 0.0% for Compositional Generation. | Page 9, Table 1 |
| [Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models](https://arxiv.org/abs/2301.13826) | both | direct rival | yes | At each early denoising step, shifts the latent along a gradient that raises the cross-attention of the most neglected subject token. | Animal-Animal prompts: CLIP text-text similarity 0.806 for Attend-and-Excite against 0.767 for Stable Diffusion and 0.692 for Composable Diffusion | page 7, Table 1 |
| [PiCo: Enhancing Text-Image Alignment with Improved Noise Selection and Precise Mask Control in Diffusion Models](https://arxiv.org/abs/2505.03203) | both | direct rival | yes | Scores candidate starting noises with a 5-step DDIM pass, keeps a good one, and uses generated pixel masks to modulate cross-attention during early denoising. | Human annotation of attribute binding with selected noise, Both: 41.8 for PiCo against 39.5 for GORS, 38.0 for Attend-and-Excite and 34.7 for Stable Diffusion | page 5, Table I |
| [ISAC: Training-Free Instance-to-Semantic Attention Control for Multi-Instance Generation](https://arxiv.org/abs/2505.20935) | Claude | direct rival | yes | Guides early denoising to stabilize instance layouts in self-attention, then binds each class's cross-attention inside those instances, with no fine-tuning. | IntraCompBench multi-class subset, 2-item prompts: 98% for SD3.5-M + ISAC_LO against 62% for SD3.5-M. | Page 11, Table 2 |
| [Be Decisive: Noise-Induced Layouts for Multi-Subject Generation](https://arxiv.org/abs/2505.21488) | Claude | direct rival | yes | A small trained network predicts a subject layout from the initial noise and refines it during denoising so each subject stays in its own region. | T2I-CompBench multi-class: 0.723 for this method against 0.676 for SDXL; LMD+, which requires training, scores 0.898. | Page 12, Appendix Table 2; metric on page 7 |
| [Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis](https://arxiv.org/abs/2212.05032) | Astra | direct rival | yes | Parses the prompt into phrases and uses their text encodings to manipulate cross-attention so each object keeps its attributes. | ABC-6K human comparison against Stable Diffusion, text-image alignment: 42.2% win, 35.6% lose, 22.2% tie, after filtering the 20% most similar pairs. | Page 6, Table 1 |
| [Compositional Visual Generation with Composable Diffusion Models](https://arxiv.org/abs/2206.01714) | both | partial rival | partly | Sums the guidance terms of several concept conditions from a diffusion model at each sampling step, so one image is pushed to satisfy all of them. | CLEVR 128x128, 3 components: binary classification accuracy 31.36% and FID 10.51 for Ours, against 7.34% and 58.33 for EBM | page 11, Table 1 |
| [MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation](https://arxiv.org/abs/2302.08113) | both | partial rival | partly | Runs one pretrained diffusion model over several crops or masked regions and reconciles their denoising steps at each timestep with a least-squares solution. | COCO region-based generation, IoU: 0.26 ± 0.12 for MultiDiffusion against 0.17 ± 0.11 for BLD and 0.16 ± 0.10 for SI; real COCO images score 0.43 ± 0.09. | Page 7, Table 2 |
| [TweedieMix: Improving Multi-Concept Fusion for Diffusion-based Image/Video Generation](https://arxiv.org/abs/2410.05591) | Astra | partial rival | yes | Samples early steps from the base model with a prompt naming all objects, extracts a region mask per object, then blends the individually fine-tuned concept models region by region in Tweedie-denoised space. | Multi-concept image generation over ten concept combinations: CLIP image similarity 0.8311 and DINO image similarity 0.5950 for TweedieMix, against 0.8095 and 0.5352 for Concept Weaver, with images lacking a foreground object excluded from scoring. | Page 8, Table 1; exclusion on page 15, Appendix A |
| [FreeFuse: Multi-Subject LoRA Fusion via Adaptive Token-Level Routing at Test Time](https://arxiv.org/abs/2510.23515) | both | partial rival | yes | Finds each subject's region from early attention in a flow-matching model and lets each subject's LoRA act only inside its own region. | Multi-subject LoRA generation on FLUX, 470 images per method: ArcFace 0.4275 for FreeFuse against 0.3886 for LoRAShop | page 10, Table 1; protocol on page 10 |
| [Token Merging for Training-Free Semantic Binding in Text-to-Image Synthesis](https://arxiv.org/abs/2411.07132) | both | partial rival | yes | Merges each object's attribute and sub-object tokens into one composite token so they share a cross-attention map, then refines that token in early steps with an entropy loss and a semantic binding loss. | GPT-4o object binding benchmark, 50 two-object prompts: 0.9549 for ToMe against 0.4907 for SDXL. | Page 8, Table 1; benchmark defined on page 7 |
| [BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion](https://arxiv.org/abs/2307.10816) | Astra | partial rival | yes | Adds inner-box, outer-box and corner constraints on cross-attention during denoising so each object lands in its given box, with no training. | YOLO score AP on a newly collected layout set: 22.3 for Stable Diffusion + BoxDiff against 2.8 for Stable Diffusion | page 8, Table 3 |
| [Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation](https://arxiv.org/abs/2403.16990) | Claude | partial rival | yes | Bounds the cross-attention and self-attention of each subject during sampling so features do not leak between subjects, guided by a given bounding-box layout. | DrawBench: counting F1 0.82 and spatial accuracy 0.36 for Bounded Attention, against 0.73 and 0.19 for Stable Diffusion. | Page 9, Table 1 |
| [GLIGEN: Open-Set Grounded Text-to-Image Generation](https://arxiv.org/abs/2301.07093) | Astra | partial rival | yes | Freezes a text-to-image model and adds trainable gated layers that take entity descriptions and spatial inputs such as boxes. | COCO2014 val-set, GLIGEN trained with detection data: FID 5.61 and YOLO AP 24.0, against FID 5.91 and AP 0.6 for the fine-tuned LDM baseline | page 6, Table 1 |
| [Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment](https://arxiv.org/abs/2306.08877) | Astra | partial rival | yes | Parses the prompt into entities and their modifiers, then optimizes the latent during sampling so each entity's attention map overlaps its modifiers and stays apart from other entities. | ABC-6K human evaluation, concept separation as share of majority votes: 28.00 for SynGen against 11.17 for Attend-and-Excite, with 50.17 having no majority winner. | Page 7, Table 1 |
| [Local Mechanisms of Compositional Generalization in Conditional Diffusion](https://arxiv.org/abs/2509.16447) | both | theory | partly | Proves that conditional projective composition is equivalent to local conditional scores, then enforces locality in a CLEVR model that previously failed to generalize to more objects. | Location-conditioned CLEVR trained on 1-3 objects, largest object count reached (Kmax): 9 for local Exp.2L against 2 for Exp.2. | Page 3, Table 1 |
| [T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-image Generation](https://arxiv.org/abs/2307.06350) | both | benchmark | yes | Provides 8,000 compositional prompts across attribute binding, object relationships, numeracy and complex scenes, with a matched evaluation metric for each category. | 8,000 compositional prompts used to benchmark 11 text-to-image models; on color binding, BLIP-VQA correlates with human ranking at Spearman rho 0.7958 against 0.2773 for CLIP | page 1, abstract; page 11, Table XII |
| [When Do Diffusion Models learn to Generate Multiple Objects?](https://arxiv.org/abs/2605.00273) | Astra | benchmark | partly | Builds a controlled synthetic dataset framework (MOSAIC) to measure how data size, concept imbalance, scene complexity and held-out combinations affect multi-object generation. | Base concept-generalization setting: Attribution and Spatial Relations achieve over 90% accuracy across dataset sizes and distributions on both U-Net and DiT. | Page 6, Section 5 and Figure 9 |

## Papers that bear on both

3 papers.

| Paper | Found by | Closeness | Needs joint data | What it does | Main number | Where |
|---|---|---|---|---|---|---|
| [Improving Compositional Generation with Diffusion Models Using Lift Scores](https://arxiv.org/abs/2505.13740) | Claude | partial rival | partly | Generates samples, scores each one with a lift score per condition using the original model, and rejects those that fail the composed condition. | SDXL on Attend-and-Excite animal pairs, ImageReward: 1.216 with CompLift (T=200) against 1.025 without it. | Page 8, Table 3 |
| [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513) | both | benchmark | yes | Scores generated images with an object detector and a colour classifier on separate tasks for presence, two-object co-occurrence, count, position, colour and attribute binding. | GenEval agrees with human annotators on 83% of images, against 88% pairwise agreement between annotators and 80% for CLIPScore, over 6,000 annotations on 1,200 images. | Page 6, Section 4, Analysis |
| [GenEval 2: Addressing Benchmark Drift in Text-to-Image Evaluation](https://arxiv.org/abs/2512.16853) | both | benchmark | yes | Audits GenEval against a human study, shows its automatic judge has drifted, and introduces a harder 800-prompt benchmark scored by Soft-TIFA from per-primitive questions. | Gemini 2.5 Flash Image on GenEval without rewriting: reported score 75.4, human score 93.1, net deviation 17.7. | Page 3, Table 1; discussion in Section 3.3, page 5 |

## Close to the topic, outside both problems

8 papers. Each is near one of the problems but differs in its data, its model or its goal.

| Paper | Found by | Closeness | Needs joint data | What it does | Main number | Where |
|---|---|---|---|---|---|---|
| [The Superposition of Diffusion Models Using the Itô Density Estimator](https://arxiv.org/abs/2412.17762) | both | partial rival | no | Reweights pretrained diffusion models during sampling with an Itô density estimate, giving an OR mixture or an AND that seeks equal density under each model. | Stable Diffusion, 20 images for each of 20 concept pairs, minimum TIFA: 39.92 for SuperDiff (AND) against 32.48 for average of scores and 27.58 for joint prompting. | Page 8, Table 2 |
| [Product of Experts for Visual Generation](https://arxiv.org/abs/2506.08894) | both | partial rival | yes | Samples from the product of generative models, visual language models and simulators using annealed MCMC with SMC-style resampling. | Character insertion on the Magic Insert set, 80 scenes: ImageReward 1.711+/-0.308 for Ours against 1.640+/-0.376 for SDEdit and 1.169+/-0.701 for Magic Insert | page 6, Table 1, Character Rendering Input block |
| [CoInD: Enabling Logical Compositions in Diffusion Models](https://arxiv.org/abs/2503.01145) | both | partial rival | no | Adds a Fisher divergence training penalty that enforces conditional independence between attribute marginals, so logical compositions work on held-out attribute combinations. | CelebA held-out smiling male, joint sampling: CS 8.63 and FID 43.97 for CoInD against CS 2.51 and FID 61.21 for Composed GLIDE. The text says > 4× while the table ratio is 3.44×. | Page 9, Table 2 and the sentence below it; page 25 for 'nearly 4×' |
| [Test-Time Compositional Generalization in Diffusion Models via Concept Discovery](https://arxiv.org/abs/2605.07078) | both | partial rival | yes | Finds density modes around one query image of an unseen combination, composes them into an analytic product-of-experts teacher, and can distill that teacher into a low-rank adapter. | ColorMNIST, 40 held-out compositions, Faithfulness F1: 87.6 ± 2.1 with the adapter and 75.3 ± 3.1 for PoE, against 0.0 ± 0.0 for Query-only. | Page 8, Table 1 |
| [Diffusion Domain Expansion: Learning to Coordinate Pre-trained Diffusion Models](https://arxiv.org/abs/2605.23275) | both | partial rival | yes | Trains a ViT coordinator on the outputs of one frozen diffusion model so it can generate larger objects or satisfy more conditions than it saw in coordinator training. | CLEVR, 5 conditionings, Heun sampler, accuracy on 256 samples: 44.5 for DDE against 36.3 for MultiDiffusion and 33.6 for RRR. | Page 3, Table 2; training details on pages 4 and 11-12 |
| [From Global to Factor-Wise Expert Composition in Discrete Diffusion Models](https://arxiv.org/abs/2607.11758) | Astra | partial rival | partly | At each denoising step, routes every grid position to whichever discrete diffusion expert is most confident there. | 120-task ARC-AGI-1 subset, RE-ARC IID evaluation, colour + occupancy experts: 90.5% exact for per-pixel routing against 78.4% for per-sample FKC, the best per-sample method | page 8, Table 2 |
| [Logical Guidance for the Exact Composition of Diffusion Models](https://arxiv.org/abs/2602.05549) | Claude | theory | partly | Derives exact guidance for Boolean formulas over attributes by weighting atomic conditional scores with time-dependent posterior probabilities, when the formula's circuit combines conditionally independent or mutually exclusive parts. | CMNIST conformity score on mutually exclusive OR: 98.0% for LogDiff against 77.0% for constant guidance; on AND both score 80.4%. | Page 5, Table 2 |
| [Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/abs/2408.09000) | Astra | theory | no | Proves that classifier-free guidance does not sample the gamma-powered distribution and that in the SDE limit it equals a DDIM predictor plus a Langevin corrector. | none printed: the results are theorems, Gaussian counterexamples and qualitative SDXL samples | abstract on page 1; contributions on pages 2 and 3; SDXL figures on pages 10 and 11 |

## How each row was checked

<details>
<summary>What was read, what a script confirmed, and why each paper sits in its group</summary>

**Sources.** The two scout tables are [Claude's](field-scouts/claude-scout/table.json) and
[GPT-6 Astra's](field-scouts/astra/table.json). Neither scout saw the other's table. Codex was available, so no
stand-in was needed for the second scout.

**Quotes.** A script matched each scout's quote against the PDF text, and readers resolved what extraction broke.
59 of 60 quotes are in the PDF. The other, Claude's quote from Product of Experts for Visual
Generation, is in the [arXiv abstract](https://arxiv.org/abs/2506.08894) and absent from the version 2 PDF.

**Numbers.** Three readers found each headline number at the page it was cited from and read table values from
page images. Each reader's record, with what it read per paper, is in reader checks
[one](field-scouts/checks/result-1.json), [two](field-scouts/checks/result-2.json) and
[three](field-scouts/checks/result-3.json). They read each abstract, the method's core and every passage holding
a claim, so a paper's other sections were not read end to end. No paper was left as `could not read`.

**Venues.** Many venues the scouts named are not printed on the PDF's first page, so this table leaves them out.
What each first page shows is under `venue_seen` in the reader checks.

**Why each paper sits in its group.**

- **Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC:** problem 2: one conditional model is composed with itself on CLEVR, and one text-to-image model across prompts (pages 6-8). No experiment read composes two models trained on disjoint image sets..
- **ISAC: Training-Free Instance-to-Semantic Attention Control for Multi-Instance Generation:** problem 2: one text-to-image model, with merged instances and semantic mixing as the stated failures (pages 1, 12)..
- **Be Decisive: Noise-Induced Layouts for Multi-Subject Generation:** problem 2: one SDXL model, with subject leakage between multiple subjects as the target failure (pages 1, 7, 12)..
- **Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis:** problem 2: one Stable Diffusion model, targeting attribute leakage, swapped attributes and missing objects (page 2)..
- **Compositional Visual Generation with Composable Diffusion Models:** problem 2: the components are 'instances of the same model' and separately trained models had limited success (p13), so it does not solve problem 1.
- **Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts:** problem 1: method only. The product is shown on molecules, the image runs are single-condition CFG, and no run composes two image concepts from separate models.
- **The Superposition of Diffusion Models Using the Itô Density Estimator:** both in setting, neither in goal: the disjoint-data experiment on CIFAR-10 uses OR, which samples a mixture (pages 7-8), and the Stable Diffusion AND experiment aims at a blended concept rather than two separate objects (page 8)..
- **MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation:** problem 2: one pretrained Stable Diffusion v2 model is run over regions (page 5). It does not compose separately trained models..
- **FreeFuse: Multi-Subject LoRA Fusion via Adaptive Token-Level Routing at Test Time:** problem 2: one FLUX base model with subject LoRAs and a joint prompt, which rules out problem 1's no-joint-prompt setting.
- **Product of Experts for Visual Generation:** neither: object insertion and controlled video from a specified scene, with experts of different kinds.
- **CoInD: Enabling Logical Compositions in Diffusion Models:** neither as defined, nearest problem 1: no joint data for the target pair, but it trains one monolithic model (pages 2, 9) on attributes of a single face rather than composing two separately trained models into two objects..
- **Improving Compositional Generation with Diffusion Models Using Lift Scores:** both: independently trained 2D components (page 2), and one Stable Diffusion model on composed prompts (page 8)..
- **Composition and Control with Distilled Energy Diffusion Models and Sequential Monte Carlo:** problem 1 in setting: two separately trained models are composed with SMC (page 8), but the training subsets may overlap and the output is one face with two attributes..
- **Test-Time Compositional Generalization in Diffusion Models via Concept Discovery:** neither as defined, nearest problem 1: one pretrained model and no second model, and the query image already shows the target composition (pages 1, 7)..
- **From Global to Factor-Wise Expert Composition in Discrete Diffusion Models:** neither: discrete grid puzzles with experts trained on one shared corpus.
- **Local Mechanisms of Compositional Generalization in Conditional Diffusion:** problem 2: studies one conditional model composing location conditions on CLEVR, with an SDXL analysis (pages 1-3, 8)..
- **Classifier-Free Guidance is a Predictor-Corrector:** neither: theory of classifier-free guidance, with no composition of two concepts.
- **Compositional Sculpting of Iterative Generative Processes:** problem 1: setting only. It composes separately trained models without joint data, but its harmonic mean is undefined on disjoint supports and its image test uses overlapping colored MNIST subsets (p6, p14).
- **When Do Diffusion Models learn to Generate Multiple Objects?:** problem 2: studies when one trained diffusion model generates multiple objects. It proposes no composition method..

</details>
