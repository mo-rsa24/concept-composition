# Where each method stands and where it breaks

For each dataset and metric this page names the best published result, who holds it, the condition where the methods
stop working, and what matching the benchmark costs. The targets on the scoreboard are chosen from these rows.

**How to read a row.** A number is only comparable with another on the same table, subset, metric and protocol, so one
dataset can carry several rows. `Printed above it` names a method outside the 22 rivals that the same table shows ahead of
the best rival. `Disputed` means the paper contradicts itself about that number, and a disputed number cannot set a target
until the page settles it. Every value is copied from `numbers.csv`, which holds its page and both reads. The best
published row of every benchmark, including datasets this page does not discuss, is in `reads/limits-all.csv`.

## Paper 1

### CLEVR

| Benchmark | Metric | Best published | Holder | Where | Printed above it | Disputed |
|---|---|---|---|---|---|---|
| CLEVR, 1 Component | Acc (%) | 86.42 | Liu | Liu Table 1, p11 |  |  |
| CLEVR, 1 Component | FID | 29.29 | Liu | Liu Table 1, p11 |  |  |
| CLEVR, 2 Components | Acc (%) | 59.20 | Liu | Liu Table 1, p11 |  |  |
| CLEVR, 2 Components | FID | 15.94 | Liu | Liu Table 1, p11 |  |  |
| CLEVR, 3 Components | Acc (%) | 31.36 | Liu | Liu Table 1, p11 |  |  |
| CLEVR, 3 Components | FID | 10.51 | Liu | Liu Table 1, p11 |  |  |
| Relational CLEVR, 1 Component | Acc (%) | 60.40 | Liu | Liu Table 2, p12 | StyleGAN2-ADA 67.71 (Liu Table 2); EBM [28] 78.14 (Liu Table 2) |  |
| Relational CLEVR, 1 Component | FID | 29.06 | Liu | Liu Table 2, p12 | StyleGAN2-ADA 20.55 (Liu Table 2); StyleGAN2 22.29 (Liu Table 2); GLIDE 17.61 (Liu Table 2) |  |
| Relational CLEVR, 2 Components | Acc (%) | 21.84 | Liu | Liu Table 2, p12 | EBM [28] 24.16 (Liu Table 2) |  |
| Relational CLEVR, 2 Components | FID | 29.82 | Liu | Liu Table 2, p12 | GLIDE 28.56 (Liu Table 2) |  |
| Relational CLEVR, 3 Components | Acc (%) | 2.80 | Liu | Liu Table 2, p12 | EBM [28] 4.26 (Liu Table 2) |  |
| Relational CLEVR, 3 Components | FID | 26.11 | Liu | Liu Table 2, p12 |  |  |
| CLEVR (cube positions), Compose 1 cube positions | Accuracy | 91.6 | Du | Du Table 2, p6 |  |  |
| CLEVR (cube positions), Compose 2 cube positions | Accuracy | 82.9 | Du | Du Table 2, p6 |  |  |
| CLEVR (cube positions), Compose 3 cube positions | Accuracy | 80.1 | Du | Du Table 2, p6 |  |  |
| CLEVR (cube positions), Compose 4 cube positions | Accuracy | 76.5 | Du | Du Table 2, p6 |  |  |
| CLEVR (cube positions), Compose 5 cube positions | Accuracy | 72.7 | Du | Du Table 2, p6 |  |  |
| CLEVR (object positions), Coordinate 1 conditionings | Accuracy | 98.0 | Du | DomainExpansion Table 2, p3 |  | Page 4 says the coordinator outperforms the baselines significantly, while Table 2 (p3) prints the RRR baseline at 98.0 and the coordinator at 96.5 at one conditioning. |
| CLEVR (object positions), Coordinate 2 conditionings | Accuracy | 94.5 | Du | DomainExpansion Table 2, p3 |  | Page 4 says the coordinator outperforms the baselines significantly, while Table 2 (p3) prints the RRR baseline at 94.5 and the coordinator at 94.1 at two conditionings. |
| CLEVR (object positions), Coordinate 3 conditionings | Accuracy | 86.3 | DomainExpansion | DomainExpansion Table 2, p3 |  |  |
| CLEVR (object positions), Coordinate 4 conditionings | Accuracy | 66.8 | DomainExpansion | DomainExpansion Table 2, p3 |  |  |
| CLEVR (object positions), 5 conditionings, Coordinate 5 conditionings | Accuracy | 44.5 | DomainExpansion | DomainExpansion Table 2, p3 |  |  |
| CLEVR Position, 1 position constraints | Accuracy (%) | 100.0 | Liu | LiftScores Table 5, p21 |  |  |
| CLEVR Position, 2 position constraints | Accuracy (%) | 100.0 | Liu | LiftScores Table 5, p21 |  |  |
| CLEVR Position, 3 position constraints | Accuracy (%) | 99.9 | LiftScores | LiftScores Table 5, p21 |  |  |
| CLEVR Position, 4 position constraints | Accuracy (%) | 97.5 | LiftScores | LiftScores Table 5, p21 |  |  |
| CLEVR Position, 5 position constraints | Accuracy (%) | 90.3 | LiftScores | LiftScores Table 5, p21 |  |  |
| CLEVR Position, 1 position constraints | FID | 36.8 | LiftScores | LiftScores Table 6, p21 |  |  |
| CLEVR Position, 2 position constraints | FID | 36.2 | LiftScores | LiftScores Table 6, p21 |  |  |
| CLEVR Position, 3 position constraints | FID | 36.4 | LiftScores | LiftScores Table 6, p21 |  |  |
| CLEVR Position, 4 position constraints | FID | 48.8 | Liu | LiftScores Table 6, p21 |  |  |
| CLEVR Position, 5 position constraints | FID | 50.1 | Liu | LiftScores Table 6, p21 |  |  |
| CLEVR Position, 1 position constraints | Accuracy (%), pretrained classifier | 56.2 | LiftScores | LiftScores Table 7, p22 |  |  |
| CLEVR Position, 2 position constraints | Accuracy (%), pretrained classifier | 25.3 | LiftScores | LiftScores Table 7, p22 |  |  |
| CLEVR Position, 3 position constraints | Accuracy (%), pretrained classifier | 6.7 | LiftScores | LiftScores Table 7, p22 |  |  |
| CLEVR Position, 4 position constraints | Accuracy (%), pretrained classifier | 0.7 | LiftScores | LiftScores Table 7, p22 |  |  |
| CLEVR Position, 5 position constraints | Accuracy (%), pretrained classifier | 0.1 | LiftScores | LiftScores Table 7, p22 |  |  |
| CLEVR location-conditioned, trained on one object, Trained on 1 | Kmax | 4 | Bradley-Projective | Bradley-Projective Table 2, p39 |  |  |
| CLEVR location-conditioned, Trained on 1-2 | Kmax | 5 | Bradley-Local | Bradley-Projective Table 2, p39 |  |  |
| CLEVR location-conditioned, Trained on 1-3 | Kmax | 9 | Bradley-Local | Bradley-Projective Table 2, p39 |  | Table 2 (p39) and p37 credit Bradley-Local's learned composition trained on 1-3 objects with Kmax 9, while Bradley-Local prints 7 for that grid-conditioned model and 9 only for its local patch models (Table 1, p3). |
| CLEVR location-conditioned, Trained on 1-4 | Kmax | 10 | Bradley-Local | Bradley-Projective Table 2, p39 |  | Table 2 (p39) credits Bradley-Local's learned composition trained on 1-4 objects with Kmax 10, and Bradley-Local v3 prints no Kmax for a model trained on 1-4 objects. |
| CLEVR location-conditioned, Trained on 1-5 | Kmax | 10 | Bradley-Local | Bradley-Projective Table 2, p39 |  |  |
| CLEVR location-conditioned, Trained on 1-6 | Kmax | 11 | Bradley-Local | Bradley-Projective Table 2, p39 |  | Table 2 (p39) credits Bradley-Local's learned composition trained on 1-6 objects with Kmax 11, and Bradley-Local v3 prints no Kmax for a model trained on 1-6 objects and its count tables stop at 10 objects (Table 2, p28). |
| CLEVR location-conditioned, trained on one object, Trained on 1 object | Kmax | 5 | Bradley-Local | Bradley-Local Table 1, p3 |  |  |
| CLEVR location-conditioned, Trained on 1-3 objects | Kmax | 9 | Bradley-Local | Bradley-Local Table 1, p3 |  | Exp.3L's tied Kmax of 9 fails the 50% rule on p23, with 228 of 1024 images at 9 and 214 at 10 (Table 2, p28), and Exp.2L's 9 meets it only if K objects means at least K, with 351 at exactly 9 and 317 at 10. |
| CLEVR location-conditioned, Trained on 1-5 objects | Kmax | 10 | Bradley-Local | Bradley-Local Table 1, p3 |  | Exp.3's Kmax of 6 on this training set fails the 50% rule on p23, with 367 of 1024 images at 6 and 45 at 7 (Table 2, p28), and the best value of 10 comes from other experiments. |
| CLEVR color-conditioned, Color, trained 1 | Kmax | 1 | Bradley-Local | Bradley-Local Table 4, p30 |  |  |
| CLEVR color-conditioned, Color, trained 1-3 | Kmax | 4 | Bradley-Local | Bradley-Local Table 4, p30 |  |  |
| CLEVR color-conditioned, Color, trained 1-5 | Kmax | 7 | Bradley-Local | Bradley-Local Table 4, p30 |  |  |
| CLEVR location-conditioned, 5 objects composed, Compose N=5 distributions | Images with 5 objects in correct locations (count of 100) | 98 | Bradley-Projective | Bradley-Projective Table 1, p17 |  | The appendix describes a model conditioned on coordinates (p15), so the composed parts may be one network. |

**Where it breaks.** Liu's task places objects at given positions on the larger CLEVR images, a separate benchmark from Du's cube positions. The composed model's accuracy falls from 86.42 at one position to 59.20 at two and 31.36 at three. Seen in Liu Fig 1 p2; Liu Fig 4 p11; Liu Fig 14 p29. Relational CLEVR composes relations between objects, a different task from placing objects at positions. The composed model's accuracy falls from 60.40 at one relation to 21.84 at two and 2.80 at three. Seen in Liu Fig 1 p2; Liu Fig 5 p12. Du's task places cubes at given positions on smaller images from its own larger dataset, so its accuracies are a separate benchmark from Liu's. EBM with HMC falls from 91.6 at one cube to 72.7 at five, and the reverse sampler credited to Liu falls from 70.8 to 57.4. Seen in Du Fig 3 p6. This task places objects at positions kept apart inside a central region of downsampled images, with the coordinator trained on at most two conditionings. The best printed accuracy falls from 98.0 at one conditioning to 94.5, 86.3, 66.8 and 44.5 at five. Seen in DomainExpansion Fig 2 p5; DomainExpansion Fig B.2 p14. This task places objects at positions on Liu's pretrained model and scores them with a SAM2 mask verifier, and CompLift scores only the samples its filter accepts. Composable Diffusion and CompLift both print 100.0 through two constraints, then CompLift falls to 90.3 at five and Composable Diffusion to 78.7. Seen in LiftScores Fig 8 p7; LiftScores Fig 18 p21; LiftScores Fig 19 p22. Scored with Liu's pretrained classifier in place of the SAM2 verifier, CompLift's accuracy falls from 56.2 at one constraint to 5.7 at three and 0.1 at five. Seen in LiftScores Fig 18 p21. This task counts objects at conditioned locations, with Kmax the largest count that succeeds. Explicit composition with an empty background stops at Kmax 4 at every training count, and with the learned background it reaches at most 9. Trained on one or two objects, the learned-background composition ties the learned composition at 5, and the paper credits the learned composition with 11 when trained on up to six. Seen in Bradley-Projective Fig 3 p3; Bradley-Projective Fig 10 p18; Bradley-Projective Fig 16 p38. This task counts objects at conditioned locations with an automated classifier that counts at most ten, so the top value cannot be told from a higher one. Trained on one to three objects, Exp.1 reaches Kmax 7, Exp.2 stops at 2 and Exp.3 at 3, and the local patch models reach 9. Seen in Bradley-Local Fig 1 p2; Bradley-Local Fig 7 p22; Bradley-Local Fig 8 p23. This task conditions on object colours and is counted by hand under a lower success threshold, so its Kmax is not comparable with the location Kmax. Kmax is 1 trained on one object, 4 on one to three and 7 on one to five. Seen in Bradley-Local Fig 13 p30. With an empty background 98 images in each hundred hold every object in place, and with the unconditional background 66 hold none. Seen in Bradley-Projective Fig 3 p9.

**What matching it costs.** The training set is 30,000 images at 128 × 128. (Liu p8) Each method generates 5, 000 images per test setting. (Liu p9) Each model is trained on a single Tesla V100 32GB GPU. (Liu p22) The diffusion models train on CLEVR for seven days (750, 000 iterations). (Liu p23) The training set is 50, 000 images at 128 × 128. (Liu p9) 200,000 images at 64 × 64 with between 1 to 5 cubes; 20000 training iterations at batch size 1024, roughly 8 hours on 8 TPUv2 cores. (Du p18) Base model trained on 30,000 images of size 128 by 128 downsampled to 64 by 64, for 1,050 epochs with a batch size of 32. (DomainExpansion p11) All training runs on a single A100 GPU. (DomainExpansion p9) Each count is scored on 256 random sets of positions. (DomainExpansion p12) Uses the pretrained diffusion model from Liu et al. and tests all methods with 5000 combinations of positions; the paper states no training hardware or time. (LiftScores p8) Training takes around 16 hours on 32 A100 GPUs with a batch size of 2048 over 128 × 2^20 samples; images are rendered at 320 × 240 and down-sampled to 128 × 128. (Bradley-Projective p15) 500,000 training samples per object count at 128 × 128; the paper states no hardware or hours. (Bradley-Local p22) Batch size of 2048 over 128 × 2^20 samples; the paper states no hardware or hours. (Bradley-Local p23) Colour Kmax is counted by hand over 64 samples per configuration with a 25% success threshold. (Bradley-Local p29)

### Colored MNIST

| Benchmark | Metric | Best published | Holder | Where | Printed above it | Disputed |
|---|---|---|---|---|---|---|
| Colored MNIST Uniform support, ∧ composition | CS (%) | 99.99 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps. |
| Colored MNIST Uniform support, ¬ Color composition | CS (%) | 100.00 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps. |
| Colored MNIST Uniform support, ¬ Digit composition | CS (%) | 89.60 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps. |
| Colored MNIST Uniform support, Violation of conditional independence between attributes | JSD | 0.10 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps. |
| Colored MNIST Non-uniform support, ∧ composition | CS (%) | 99.95 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps. |
| Colored MNIST Non-uniform support, ¬ Color composition | CS (%) | 93.75 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps. |
| Colored MNIST Non-uniform support, ¬ Digit composition | CS (%) | 84.98 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps. |
| Colored MNIST Non-uniform support, Violation of conditional independence between attributes | JSD | 0.15 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps. |
| Colored MNIST Partial support, ∧ composition | CS (%) | 52.38 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps, and p32 states 52.38% CS for this task while the per-cell values in Figure 17 on that page do not average to it. |
| Colored MNIST Partial support, ¬ Color composition | CS (%) | 64.84 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps. |
| Colored MNIST Partial support, ¬ Digit composition | CS (%) | 53.12 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps. |
| Colored MNIST Partial support, Violation of conditional independence between attributes | JSD | 1.11 | CoInD | CoInD Table 3, p19 |  | Table 5 (p22) prints 150 DDIM sampling steps for Colored MNIST and the text on p22 says 100 steps. |
| CMNIST, AND, AND | Conformity score (%) | 80.4 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| CMNIST, NOT | Conformity score (%) | 96.5 | LogicalGuidance | LogicalGuidance Table 2, p5 | Constant 99.5 (LogicalGuidance Table 2) | LoGDiff prints CMNIST NOT as 96.5 in Table 2 (p5) and 97.38 in Table 11 (p26). |
| CMNIST, OR-ME | Conformity score (%) | 98.0 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| CMNIST, OR-CI | Conformity score (%) | 97.2 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| CMNIST, N=2 | Conformity score (%) | 93.8 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| CMNIST, N=3 | Conformity score (%) | 93.3 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| CMNIST, N=4 | Conformity score (%) | 94.2 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| CMNIST, N=5 | Conformity score (%) | 94.4 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| CMNIST, AND, AND | Conformity score (%) | 83.6 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| CMNIST, NOT | Conformity score (%) | 98.4 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| CMNIST, OR-CI | Conformity score (%) | 98.0 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| CMNIST, N=2 | Conformity score (%) | 94.7 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| CMNIST, N=3 | Conformity score (%) | 94.3 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| CMNIST, N=4 | Conformity score (%) | 95.1 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| CMNIST, N=5 | Conformity score (%) | 95.5 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| CMNIST, AND, AND | CS (Conformity Score) | 80.39 | LogicalGuidance | LogicalGuidance Table 11, p26 |  |  |
| CMNIST, NOT | CS (Conformity Score) | 99.41 | Liu | LogicalGuidance Table 11, p26 |  | The constant-weight NOT row prints 99.5 in Table 2 (p5) where Table 11 (p26) prints 99.41 for the plain negation rule, and LoGDiff prints 96.5 in Table 2 and 97.38 in Table 11. |
| CMNIST, OR-ME | CS (Conformity Score) | 98.01 | LogicalGuidance | LogicalGuidance Table 11, p26 |  |  |
| CMNIST, OR-CI | CS (Conformity Score) | 97.54 | SuperDiff | LogicalGuidance Table 11, p26 |  |  |
| ColorMNIST 40 OOD compositions, Generate a held-out composition from one query, scored against the 100 query images | FID | 62.6 ±2.5 | TestTime | TestTime Table 1, p8 |  |  |
| ColorMNIST 40 OOD compositions, Generate a held-out composition from one query, scored against 100 other held-out images | FID | 84.7 ±2.8 | TestTime | TestTime Table 1, p8 |  |  |
| ColorMNIST 40 OOD compositions, Generate a held-out composition from one query, scored against the 100 query images | CLIP (%) | 95.1 ±0.1 | TestTime | TestTime Table 1, p8 |  |  |
| ColorMNIST 40 OOD compositions, Generate a held-out composition from one query, scored against 100 other held-out images | CLIP (%) | 94.8 ±0.2 | TestTime | TestTime Table 1, p8 |  |  |
| ColorMNIST 40 OOD compositions, Generate a held-out composition from one query, scored against the 100 query images | Precision (%) | 89.5 ±2.0 | TestTime | TestTime Table 1, p8 |  |  |
| ColorMNIST 40 OOD compositions, Generate a held-out composition from one query, scored against 100 other held-out images | Precision (%) | 82.2 ±2.7 | TestTime | TestTime Table 1, p8 |  |  |
| ColorMNIST 40 OOD compositions, Generate a held-out composition from one query, scored against the 100 query images | Recall (%) | 88.0 ±2.8 | TestTime | TestTime Table 1, p8 |  |  |
| ColorMNIST 40 OOD compositions, Generate a held-out composition from one query, scored against 100 other held-out images | Recall (%) | 78.8 ±3.8 | TestTime | TestTime Table 1, p8 |  |  |
| ColorMNIST 40 OOD compositions, Generate a held-out composition from one query, scored against the 100 query images | F1 (%) | 87.6 ±2.1 | TestTime | TestTime Table 1, p8 |  |  |
| ColorMNIST 40 OOD compositions, Generate a held-out composition from one query, scored against 100 other held-out images | F1 (%) | 76.3 ±2.6 | TestTime | TestTime Table 1, p8 |  |  |
| Colored MNIST, harmonic mean and contrast of separately trained models | no image composition score printed | no rival number |  |  |  |  |

**Where it breaks.** Composition holds while every digit and colour pair is seen and falls when pairs are held out. CoInD's ∧ CS is 99.99 on uniform support, 99.95 on non-uniform and 52.38 on partial, and Composed GLIDE's is 98.15, 86.10 and 7.40. Seen in CoInD Fig 1 p1; CoInD Fig 5 p8; CoInD Fig 16 p32. AND is LoGDiff's lowest task at 80.4, while formulas of two to five operators stay between 93.3 and 94.4. Seen in LogicalGuidance Fig 2 p5; LogicalGuidance Fig 7 p27; LogicalGuidance Fig 10 p28. Repulsive guidance is an added step that holds LoGDiff's best number on these tasks. AND stays the lowest task at 83.6, and NOT reaches 98.4. Seen in LogicalGuidance Fig 2 p5; LogicalGuidance Fig 7 p27. Rules written for separately trained models are applied here to one conditional model. On AND, the half-and-half average credited to Liu scores 26.61 and SuperDiff's equal-probability update 26.20, and the form adding both guidance terms to the unconditional branch, credited to LoGDiff and Liu, scores 80.39. Seen in LogicalGuidance Fig 2 p5; LogicalGuidance Fig 9 p28. Scores drop from the query images to other held-out images of the same class: the LoRA variant's F1 falls from 87.6 to 76.3 and its FID rises from 62.6 to 84.7. The paper says its small-sample FID must not be compared with published FIDs. Seen in TestTime Fig 3 p8; TestTime Fig 5 p17; TestTime Fig 6 p18. Sculpting prints sample grids and no image composition score. Its grids show three-way conditions leaving many zeros open or broken, and the result changes with the order of chaining. Seen in Sculpting Fig 6 p14; Sculpting Fig F.5 p34.

**What matching it costs.** Table 5 lists 50000 training steps and 1000 noise steps at sample size 28x3x3 for CoInD and Composed GLIDE. (CoInD p22) 50,000 training steps with a DDPM sampler of 50 timesteps (Table 9). (LogicalGuidance p21) Backbone training uses a single NVIDIA A40 GPU, roughly 6 hours, and the full pipeline takes about 10-17 minutes per query. (TestTime p15) On a V100 GPU, MNIST base diffusion trains for 200 steps in 10m 6s and the classifier trains for 200 steps in 30m 12s. (Sculpting p35)

### CelebA

| Benchmark | Metric | Best published | Holder | Where | Printed above it | Disputed |
|---|---|---|---|---|---|---|
| CelebA orthogonal partial support, smiling male held out, "smiling" ∧ "male" composition | CS (%) | 10.55 | Liu | CoInD Table 2, p9 | LACE 24.20 (CoInD Table 2) | The Table 2 caption (p9) says CoInD outperforms the baselines on CS across tasks, and this row prints CoInD at 8.79 below Composed GLIDE at 10.55. |
| CelebA orthogonal partial support, smiling male held out, "smiling" ∧ "male" composition | FID | 43.76 | CoInD | CoInD Table 2, p9 |  |  |
| CelebA orthogonal partial support, smiling male held out, Violation of conditional independence between attributes | JSD | 1.82 | CoInD | CoInD Table 2, p9 |  |  |
| CelebA orthogonal partial support, smiling male held out, Joint prompt "smiling male" | CS (%) | 8.63 | CoInD | CoInD Table 2, p9 |  | Page 9 says CoInD beats the baseline by more than 4× in joint sampling and p25 says nearly 4×, while Table 2 prints 8.63 against 2.51. |
| CelebA orthogonal partial support, smiling male held out, Joint prompt "smiling male" | FID | 43.97 | CoInD | CoInD Table 2, p9 |  |  |
| CelebA Orthogonal support, Violation of conditional independence between attributes | JSD | 0.37 | CoInD | CoInD Table 6, p27 |  |  |
| CelebA Orthogonal support, Joint | CS (%) | 58.57 | CoInD | CoInD Table 6, p27 |  |  |
| CelebA Orthogonal support, Joint | FID | 58.19 | CoInD | CoInD Table 6, p27 |  |  |
| CelebA Orthogonal support, ∧ composition, ∧ composition | CS (%) | 49.15 | CoInD | CoInD Table 6, p27 |  |  |
| CelebA Orthogonal support, ∧ composition, ∧ composition | FID | 61.16 | CoInD | CoInD Table 6, p27 |  |  |
| CelebA Orthogonal support, ¬ composition | CS (%) | 18.80 | CoInD | CoInD Table 6, p27 |  | Page 27 states a 2× improvement in ¬ CS over Composed GLIDE, while Table 6 prints 18.80 against 11.02. |
| CelebA Orthogonal support, ¬ composition | FID | 86.31 | CoInD | CoInD Table 6, p27 |  |  |
| CelebA, AND | CS (Conformity Score) | 0.63 | LogicalGuidance | LogicalGuidance Table 3, p6 |  |  |
| CelebA, AND | FID | 19.02 | LogicalGuidance | LogicalGuidance Table 3, p6 |  |  |
| CelebA, NOT | CS (Conformity Score) | 0.80 | LogicalGuidance | LogicalGuidance Table 3, p6 |  |  |
| CelebA, NOT | FID | 23.61 | LogicalGuidance | LogicalGuidance Table 3, p6 | Uncond. 21.60 (LogicalGuidance Table 3) |  |
| CelebA, OR-CI | CS (Conformity Score) | 0.97 | LogicalGuidance | LogicalGuidance Table 3, p6 |  |  |
| CelebA, OR-CI | FID | 18.47 | LogicalGuidance | LogicalGuidance Table 3, p6 | Constant 17.11 (LogicalGuidance Table 3) |  |
| CelebA 23 OOD compositions, Generate a held-out composition from one query, scored against the 100 query images | FID | 120.9 ±3.1 | TestTime | TestTime Table 2, p9 | Query-only 107.5 (TestTime Table 2) |  |
| CelebA 23 OOD compositions, Generate a held-out composition from one query, scored against 100 other held-out images | FID | 140.3 ±3.9 | TestTime | TestTime Table 2, p9 |  |  |
| CelebA 23 OOD compositions, Generate a held-out composition from one query, scored against the 100 query images | CLIP (%) | 68.5 ±0.8 | TestTime | TestTime Table 2, p9 |  | Page 9 says Query-only attains the strongest Faithfulness scores, while Table 2 prints PoE at 68.5 above Query-only at 66.6 on Faithfulness CLIP. |
| CelebA 23 OOD compositions, Generate a held-out composition from one query, scored against 100 other held-out images | CLIP (%) | 68.4 ±0.9 | TestTime | TestTime Table 2, p9 |  |  |
| CelebA 23 OOD compositions, Generate a held-out composition from one query, scored against the 100 query images | Precision (%) | 75.8 ±1.4 | TestTime | TestTime Table 2, p9 | Query-only 80.6 (TestTime Table 2) |  |
| CelebA 23 OOD compositions, Generate a held-out composition from one query, scored against 100 other held-out images | Precision (%) | 74.5 ±1.8 | TestTime | TestTime Table 2, p9 |  |  |
| CelebA 23 OOD compositions, Generate a held-out composition from one query, scored against the 100 query images | Recall (%) | 19.9 ±1.4 | TestTime | TestTime Table 2, p9 | Query-only 53.6 (TestTime Table 2) |  |
| CelebA 23 OOD compositions, Generate a held-out composition from one query, scored against 100 other held-out images | Recall (%) | 16.8 ±1.2 | TestTime | TestTime Table 2, p9 |  |  |
| CelebA 23 OOD compositions, Generate a held-out composition from one query, scored against the 100 query images | F1 (%) | 26.9 ±1.4 | TestTime | TestTime Table 2, p9 | Query-only 64.0 (TestTime Table 2) |  |
| CelebA 23 OOD compositions, Generate a held-out composition from one query, scored against 100 other held-out images | F1 (%) | 24.7 ±1.6 | TestTime | TestTime Table 2, p9 |  |  |

**Where it breaks.** With smiling male held out of training, composition rarely produces it: ∧ CS is 10.55 for Composed GLIDE and 8.79 for CoInD, and the joint prompt reaches 8.63 for CoInD. Seen in CoInD Fig 7 p10; CoInD Fig 11 p26; CoInD Fig 16 p32. On fine-tuned Stable Diffusion, ∧ CS is 49.15 for CoInD and 14.19 for Composed GLIDE, and ¬ CS stays lower at 18.80 and 11.02. Seen in CoInD Fig 13 p27. AND is LoGDiff's lowest conformity at 0.63 against 0.97 on OR-CI, and its FID is worst on NOT at 23.61. Seen in LogicalGuidance Fig 18 p32. Coverage of held-out compositions stays low: the best recall on other held-out images is 16.8 and the best F1 is 24.7, and PoE's recall is 12.0 on the query images and 9.0 on other held-out images. The paper says its small-sample FID must not be compared with published FIDs. Seen in TestTime Fig 3 p8; TestTime Fig 5 p17.

**What matching it costs.** CelebA at 128 × 128, trained for 500,000 steps on one A6000 GPU. (CoInD p22) 500,000 training steps with a DDIM sampler of 1000 timesteps (Table 9). (LogicalGuidance p21) Backbone training uses four NVIDIA RTX 6000 GPUs, roughly 50 hours, and the full pipeline takes about 10-17 minutes per query. (TestTime p15)

### Shapes3D

| Benchmark | Metric | Best published | Holder | Where | Printed above it | Disputed |
|---|---|---|---|---|---|---|
| Shapes3D Uniform support, ∧ composition | R2 | 0.98 | CoInD | CoInD Table 4, p20 |  |  |
| Shapes3D Uniform support, ∧ composition | CS (%) | 95.31 | CoInD | CoInD Table 4, p20 |  |  |
| Shapes3D Uniform support, ¬ composition | R2 | 0.92 | CoInD | CoInD Table 4, p20 |  |  |
| Shapes3D Uniform support, ¬ composition | CS (%) | 64.06 | CoInD | CoInD Table 4, p20 |  |  |
| Shapes3D Uniform support, Violation of conditional independence between attributes | JSD | 0.215 | CoInD | CoInD Table 4, p20 |  |  |
| Shapes3D Partial support, ∧ composition, ∧ composition | R2 | 0.97 | CoInD | CoInD Table 4, p20 |  |  |
| Shapes3D Partial support, ∧ composition, ∧ composition | CS (%) | 91.10 | CoInD | CoInD Table 4, p20 |  |  |
| Shapes3D Partial support, ¬ composition | R2 | 0.92 | CoInD | CoInD Table 4, p20 |  |  |
| Shapes3D Partial support, ¬ composition | CS (%) | 53.90 | CoInD | CoInD Table 4, p20 |  |  |
| Shapes3D Partial support, Violation of conditional independence between attributes | JSD | 0.287 | CoInD | CoInD Table 4, p20 |  |  |
| Shapes3D, AND | Conformity score (%) | 82.4 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| Shapes3D, NOT | Conformity score (%) | 96.4 | LogicalGuidance | LogicalGuidance Table 2, p5 | Constant 98.7 (LogicalGuidance Table 2) |  |
| Shapes3D, OR-ME | Conformity score (%) | 94.3 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| Shapes3D, OR-CI | Conformity score (%) | 96.0 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| Shapes3D, N=2 | Conformity score (%) | 88.8 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| Shapes3D, N=3 | Conformity score (%) | 88.6 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| Shapes3D, N=4 | Conformity score (%) | 85.1 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| Shapes3D, N=5 | Conformity score (%) | 87.6 | LogicalGuidance | LogicalGuidance Table 2, p5 |  |  |
| Shapes3D, AND | Conformity score (%) | 86.4 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| Shapes3D, NOT | Conformity score (%) | 99.3 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| Shapes3D, OR-ME | Conformity score (%) | 95.6 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| Shapes3D, OR-CI | Conformity score (%) | 96.7 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| Shapes3D, N=2 | Conformity score (%) | 91.8 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| Shapes3D, N=3 | Conformity score (%) | 90.5 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| Shapes3D, N=4 | Conformity score (%) | 89.8 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |
| Shapes3D, N=5 | Conformity score (%) | 89.9 | LogicalGuidance | LogicalGuidance Table 4, p7 |  |  |

**Where it breaks.** Composition weakens from uniform to partial support: Composed GLIDE's ∧ CS falls from 83.75 to 51.56 and CoInD's from 95.31 to 91.10, and CoInD's ¬ CS is 55.46 and 53.90. Seen in CoInD Fig 6 p9; CoInD Fig 16 p32; CoInD Fig 18 p33. AND is LoGDiff's lowest task at 82.4, and formulas print 88.8 at two operators, 85.1 at four and 87.6 at five. Seen in LogicalGuidance Fig 2 p5; LogicalGuidance Fig 8 p27; LogicalGuidance Fig 14 p30. Repulsive guidance is an added step that holds LoGDiff's best number on these tasks. AND stays the lowest task at 86.4, and NOT reaches 99.3. Seen in LogicalGuidance Fig 2 p5; LogicalGuidance Fig 8 p27.

**What matching it costs.** Table 5 lists 100000 training steps and 1000 noise steps at sample size 64x3x3. (CoInD p22) 100,000 training steps with a DDPM sampler of 50 timesteps (Table 9). (LogicalGuidance p21)

### CIFAR-10 split

| Benchmark | Metric | Best published | Holder | Where | Printed above it | Disputed |
|---|---|---|---|---|---|---|
| CIFAR-10 disjoint halves, unconditional image generation from two models trained on disjoint data (logical OR) | FID, ODE inference | 4.11 | SuperDiff | SuperDiff Table 1, p8 |  |  |
| CIFAR-10 disjoint halves, unconditional image generation from two models trained on disjoint data (logical OR) | IS, ODE inference | 9.21 | SuperDiff | SuperDiff Table 1, p8 |  |  |
| CIFAR-10 disjoint halves, unconditional image generation from two models trained on disjoint data (logical OR) | FLD, ODE inference | 5.89 ± 0.11 | SuperDiff | SuperDiff Table 1, p8 |  |  |
| CIFAR-10 disjoint halves, unconditional image generation from two models trained on disjoint data (logical OR) | FID, SDE inference | 4.00 | SuperDiff | SuperDiff Table 1, p8 | model_{A union B} 3.50 (SuperDiff Table 1); model_{A OR B} 3.99 (SuperDiff Table 1) | Page 7 says SuperDiff performs even better than the model trained on the union of both parts, while Table 1 (p8) prints SDE FID 4.00 for SuperDiff and 3.50 for that model. |
| CIFAR-10 disjoint halves, unconditional image generation from two models trained on disjoint data (logical OR) | IS, SDE inference | 9.48 | SuperDiff | SuperDiff Table 1, p8 |  |  |
| CIFAR-10 disjoint halves, unconditional image generation from two models trained on disjoint data (logical OR) | FLD, SDE inference | 5.20 ± 0.11 | SuperDiff | SuperDiff Table 1, p8 |  |  |
| CIFAR-10 split into two random partitions A and B, class-conditional image generation from two models trained on random partitions (logical OR) | FID, ODE inference | 4.63 | SuperDiff | SuperDiff Table A1, p30 |  |  |
| CIFAR-10 split into two random partitions A and B, class-conditional image generation from two models trained on random partitions (logical OR) | IS, ODE inference | 9.00 | SuperDiff | SuperDiff Table A1, p30 | model_{A union B} 9.04 (SuperDiff Table A1) |  |
| CIFAR-10 split into two random partitions A and B, class-conditional image generation from two models trained on random partitions (logical OR) | FLD, ODE inference | 6.81 ± 0.19 | SuperDiff | SuperDiff Table A1, p30 |  |  |
| CIFAR-10 split into two random partitions A and B, class-conditional image generation from two models trained on random partitions (logical OR) | FID, SDE inference | 4.23 | SuperDiff | SuperDiff Table A1, p30 | model_{A union B} 2.83 (SuperDiff Table A1) |  |
| CIFAR-10 split into two random partitions A and B, class-conditional image generation from two models trained on random partitions (logical OR) | IS, SDE inference | 9.42 | SuperDiff | SuperDiff Table A1, p30 | model_B 9.45 (SuperDiff Table A1); model_{A union B} 9.44 (SuperDiff Table A1) |  |
| CIFAR-10 split into two random partitions A and B, class-conditional image generation from two models trained on random partitions (logical OR) | FLD, SDE inference | 6.22 ± 0.15 | SuperDiff | SuperDiff Table A1, p30 | model_B 6.20 (SuperDiff Table A1) |  |

**Where it breaks.** Each model trained on one disjoint half alone prints ODE FID 14.00 and 13.09, and SuperDiff over the two prints 4.11. The paper does not state how many images enter these scores. On random partitions the single models print ODE FID 4.75 and 4.78 and SuperDiff prints 4.63, so this split separates the methods far less than the disjoint split. The paper does not state how many images enter these scores.

### Chest X-ray

| Benchmark | Metric | Best published | Holder | Where | Printed above it | Disputed |
|---|---|---|---|---|---|---|
| Chest X-ray | no rival metric | no rival number |  |  |  |  |

**Where it breaks.** No rival prints a chest X-ray result, so no published break exists.

## Paper 2

### T2I-CompBench

| Benchmark | Metric | Best published | Holder | Where | Printed above it | Disputed |
|---|---|---|---|---|---|---|
| T2I-CompBench Color, attribute binding | BLIP-VQA | 0.7656 | ToMe | ToMe Table 1, p8 | CoMat 0.7774 (ToMe Table 1) |  |
| T2I-CompBench Texture, attribute binding | BLIP-VQA | 0.6894 | ToMe | ToMe Table 1, p8 |  |  |
| T2I-CompBench Shape, attribute binding | BLIP-VQA | 0.6051 | ToMe | ToMe Table 1, p8 |  |  |
| T2I-CompBench Color, attribute binding | Human-preference (ImageReward) | 1.074 | ToMe | ToMe Table 1, p8 |  |  |
| T2I-CompBench Texture, attribute binding | Human-preference (ImageReward) | 0.9281 | ToMe | ToMe Table 1, p8 |  |  |
| T2I-CompBench Shape, attribute binding | Human-preference (ImageReward) | 0.5916 | ToMe | ToMe Table 1, p8 |  |  |
| T2I-CompBench Color, attribute binding | BLIP-VQA | 0.7656 | ToMe | ToMe Table 4, p21 | CoMat 0.7774 (ToMe Table 4) |  |
| T2I-CompBench Texture, attribute binding | BLIP-VQA | 0.6894 | ToMe | ToMe Table 4, p21 |  |  |
| T2I-CompBench Shape, attribute binding | BLIP-VQA | 0.6051 | ToMe | ToMe Table 4, p21 |  |  |
| T2ICompbench Complex, three or more concepts | ImageReward (IR) | 0.4406 | CO3 | CO3 Table 2, p7 |  |  |
| T2ICompbench Complex, three or more concepts | Human Evaluation (HE) | 0.8033 | CO3 | CO3 Table 2, p7 |  |  |
| T2I-CompBench Count, Single Category column group | MAE | 0.87 | CountLoop | CountLoop Table 1, p10 |  |  |
| T2I-CompBench Count, Single Category column group | Ex. (%) | 70 | CountLoop | CountLoop Table 1, p10 |  | Page 12 says the critic is kept distinct from the evaluation detector (OWLv2) so it cannot optimise against the metric, yet exact-count match is scored by CountGD++ (p8) and this 70 comes from the row using CountGD++ as its critic (Table 1, p10), while the GroundingDINO-critic row prints 55. |
| T2I-CompBench Count, Single Category column group | Sp. | 0.95 | CountLoop | CountLoop Table 1, p10 |  |  |
| CC-500 and T2I-Compbench (500 prompts), attribute binding, random noise | Random: Both (%) | 36.0 | PiCo | PiCo Table I, p5 |  |  |
| CC-500 and T2I-Compbench (500 prompts), attribute binding, random noise | Random: None (%) | 23.5 | PiCo | PiCo Table I, p5 |  |  |
| T2I-Compbench (400 prompts), attribute binding, random noise | Random: DINO | 0.637 | PiCo | PiCo Table I, p5 |  |  |
| CC-500 and T2I-Compbench (500 prompts), attribute binding, selected noise | Selected: Both (%) | 41.8 | PiCo | PiCo Table I, p5 |  | Table I prints the default PiCo setting at Both 41.8 (p5), Table III and the r_s = 10 row of Table V print 36.2 (p5, p10), and the T_c = 25 and gamma = 15 rows of Table V print 37.8 (p10). |
| CC-500 and T2I-Compbench (500 prompts), attribute binding, selected noise | Selected: None (%) | 11.8 | PiCo | PiCo Table I, p5 |  | Table I prints the default PiCo setting at None 11.8 (p5), Table III and the r_s = 10 row of Table V print 18.4 (p5, p10), the T_c = 25 and gamma = 15 rows of Table V print 19.7 (p10), the text on p9 says 13.4, and the Ours rows of Table VI print 19.8 (p10). |
| T2I-Compbench (400 prompts), attribute binding, selected noise | Selected: DINO | 0.730 | PiCo | PiCo Table I, p5 |  | Table I prints PiCo at DINO 0.730 (p5), and the Ours rows of Table VI print 0.659 (p10), the value Table I gives Stable Diffusion. |
| T2I-Compbench (400 prompts), objective metric, selected noise | VQAScore | 0.50 | PiCo | PiCo Table I, p5 |  | Table I prints this score with no caveat (p5), and Table IV prints the same score under a caption saying these results are not reliable for text-image alignment (p7). |
| T2I-Compbench (400 prompts), objective metric, selected noise | CLIP-S | 16.35 | AttendExcite | PiCo Table I, p5 |  | Table I prints this score with no caveat (p5), and Table IV prints the same score under a caption saying these results are not reliable for text-image alignment (p7). |
| T2I-Compbench (400 prompts), automatic comparison | VQAScore | 0.50 | PiCo | PiCo Table IV, p7 |  | Table IV's caption says its results are not reliable for text-image alignment (p7). |
| T2I-Compbench (400 prompts), automatic comparison | CLIPScore | 16.35 | AttendExcite | PiCo Table IV, p7 |  | Table IV's caption says its results are not reliable for text-image alignment (p7). |
| T2I-CompBench Color, attribute binding | T2I-CompBench result | 0.7827 | CoMat | CoMat Table 1, p7 |  | ToMe quotes CoMat on SDXL at 0.7774 (ToMe Table 1, p8), and CoMat's arXiv v3 prints 0.7827 (Table 1, p7). |
| T2I-CompBench Shape, attribute binding | T2I-CompBench result | 0.5329 | CoMat | CoMat Table 1, p7 |  | ToMe quotes CoMat on SDXL at 0.5262 (ToMe Table 1, p8), and CoMat's arXiv v3 prints 0.5329 (Table 1, p7). |
| T2I-CompBench Texture, attribute binding | T2I-CompBench result | 0.6468 | CoMat | CoMat Table 1, p7 | PixArt-α 0.6477 (CoMat Table 1) | ToMe quotes CoMat on SDXL at 0.6591 (ToMe Table 1, p8), and CoMat's arXiv v3 prints 0.6468 (Table 1, p7). CoMat's table highlights its own score as best while printing PixArt-α higher. |
| T2I-CompBench Complex, complex compositional prompts | ImageReward (IR) | 0.4661 | RTD | RTD Table 2, p6 |  | RTD's baseline ImageReward cells match CO3's table on 19 of 20 values, so CO3 was probably not rerun. |

**Where it breaks.** BLIP-VQA falls from Color to Shape: SDXL scores 0.6369 on Color and 0.5408 on Shape, and ToMe scores 0.7656 and 0.6051. Seen in ToMe Fig 5 p8 (prompt set not stated). ImageReward falls from Color to Shape: SDXL scores 0.7798 on Color and 0.4029 on Shape, and ToMe scores 1.074 and 0.5916. Seen in ToMe Fig 5 p8 (prompt set not stated). On prompts with three or more concepts, ToMe falls below plain SDXL on ImageReward (0.2891 against 0.3083) and on human evaluation (0.5866 against 0.5983). Seen in CO3 Fig 4 p8; CO3 Fig 10 p24. No baseline passes an exact-count match of 55% and SDXL has a counting MAE of 2.72. CountLoop reaches 70% only with CountGD++, the exact-count scorer, as its critic. Seen in CountLoop Fig 6 p9 (authors' count prompts). Annotators mark both concepts present in 27.5% of Stable Diffusion images and 36.0% of PiCo images under random noise, and in 41.8% of PiCo images with its selected noise. Seen in PiCo Fig 2 p4; PiCo Fig 16 p14. Plain SDXL scores 0.5879 on this subset in CoMat's table and 0.6369 in ToMe's, so the two tables do not share a protocol. Seen in CoMat Fig 6 p9; CoMat Fig 7 p10. Plain SDXL scores 0.4687 on this subset in CoMat's table and 0.5408 in ToMe's, so the two tables do not share a protocol. Seen in CoMat Fig 6 p9; CoMat Fig 7 p10. Plain SDXL scores 0.5299 on this subset in CoMat's table and 0.5637 in ToMe's, so the two tables do not share a protocol. Seen in CoMat Fig 6 p9; CoMat Fig 7 p10. RTD scores 0.4661 where CO3 scores 0.4406 in the same table, and shape and action words still fail in its pictures. Seen in RTD Fig 3 p5; RTD Fig 4 p6.

**What matching it costs.** All experiments ran on an NVIDIA-A40 GPU. (ToMe p16) All experiments ran on a single NVIDIA A6000 GPU (48GB). (CO3 p6) CountLoop is training-free. (CountLoop p1) Noise selection takes 2.03 s per prompt at the default T_s = 5. (PiCo p5)

### RareBench

| Benchmark | Metric | Best published | Holder | Where | Printed above it | Disputed |
|---|---|---|---|---|---|---|
| RareBench Concat, rare or novel concepts | ImageReward (IR) | 0.1610 | CO3 | CO3 Table 2, p7 |  |  |
| RareBench Concat, rare or novel concepts | Human Evaluation (HE) | 0.6167 | CO3 | CO3 Table 2, p7 |  |  |
| RareBench Relations, rare or novel concepts | ImageReward (IR) | 0.2588 | CO3 | CO3 Table 2, p7 | SDXL 0.3607 (CO3 Table 2); SynGen(SDXL) 0.3177 (CO3 Table 2); R2F 0.3620 (CO3 Table 2) | Page 8 says CO3 outperforms or matches the best baselines across all categories and metrics and matches or exceeds R2F on RareBench, yet Table 2 (p7) prints CO3 at 0.2588 against R2F at 0.3620. |
| RareBench Relations, rare or novel concepts | Human Evaluation (HE) | 0.6200 | CO3 | CO3 Table 2, p7 |  |  |
| RareBench Complex, rare or novel concepts | ImageReward (IR) | 1.2211 | CO3 | CO3 Table 2, p7 |  |  |
| RareBench Complex, rare or novel concepts | Human Evaluation (HE) | 0.61166 | CO3 | CO3 Table 2, p7 | SDXL 0.6283 (CO3 Table 2) | Page 8 says CO3 consistently outperforms or matches the best-performing baselines across all categories and metrics, yet Table 2 (p7) prints CO3 at 0.61166 against SDXL at 0.6283. |
| RareBench Concat, long-tail rare-concept prompts | ImageReward (IR) | 0.4464 | RTD | RTD Table 2, p6 |  | RTD's baseline ImageReward cells match CO3's table on 19 of 20 values, so CO3 was probably not rerun. |

**Where it breaks.** Rare attributes are dropped or ignored. On Concat, SDXL scores ImageReward 0.1018 and CO3 0.1610, and on Relations CO3 at 0.2588 sits below SDXL at 0.3607 and R2F at 0.3620. Seen in CO3 Fig 4 p8; CO3 Fig 11 p25. RTD scores 0.4464 where CO3 scores 0.1610 in the same table, and rare shape and action words still fail in its pictures. Seen in RTD Fig D.5 p19; RTD Fig D.6 p20.

**What matching it costs.** All experiments ran on a single NVIDIA A6000 GPU (48GB). (CO3 p6)

### GenEval

| Benchmark | Metric | Best published | Holder | Where | Printed above it | Disputed |
|---|---|---|---|---|---|---|
| GenEval (553 prompts), text-to-image generation on GenEval | Overall | 0.58 | FKC | FKC Table A4, p39 |  |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Single object | 0.99 | FKC | FKC Table A4, p39 |  |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Two object | 0.77 | FKC | FKC Table A4, p39 | SDXL 0.80 (FKC Table A4) |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Counting | 0.49 | FKC | FKC Table A4, p39 |  |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Colors | 0.87 | FKC | FKC Table A4, p39 |  |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Position | 0.10 | FKC | FKC Table A4, p39 | SDXL 0.15 (FKC Table A4); SDXL 0.11 (FKC Table A4) |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Color attribution | 0.22 | FKC | FKC Table A4, p39 | SDXL 0.23 (FKC Table A4) |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Overall | 0.57 | FKC | FKC Table A4, p39 |  |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Single object | 0.99 | FKC | FKC Table A4, p39 |  |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Two object | 0.78 | FKC | FKC Table A4, p39 | SDXL 0.80 (FKC Table A4) |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Counting | 0.46 | FKC | FKC Table A4, p39 |  |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Colors | 0.83 | FKC | FKC Table A4, p39 | SDXL 0.85 (FKC Table A4); SDXL 0.86 (FKC Table A4) |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Position | 0.13 | FKC | FKC Table A4, p39 | SDXL 0.15 (FKC Table A4) |  |
| GenEval (553 prompts), text-to-image generation on GenEval | Color attribution | 0.23 | FKC | FKC Table A4, p39 |  |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Overall | 0.57 | FKC | FKC Table A4, p39 | SDXL 0.58 (FKC Table A4) |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Single object | 0.99 | FKC | FKC Table A4, p39 |  |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Two object | 0.79 | FKC | FKC Table A4, p39 |  |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Counting | 0.42 | FKC | FKC Table A4, p39 | SDXL 0.45 (FKC Table A4) |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Colors | 0.86 | FKC | FKC Table A4, p39 | SDXL 0.88 (FKC Table A4) |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Position | 0.13 | FKC | FKC Table A4, p39 |  |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Color attribution | 0.21 | FKC | FKC Table A4, p39 |  |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Overall | 0.57 | FKC | FKC Table A4, p39 | SDXL 0.58 (FKC Table A4) |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Single object | 0.99 | FKC | FKC Table A4, p39 |  |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Two object | 0.80 | FKC | FKC Table A4, p39 |  |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Counting | 0.45 | FKC | FKC Table A4, p39 |  |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Colors | 0.83 | FKC | FKC Table A4, p39 | SDXL 0.88 (FKC Table A4) |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Position | 0.13 | FKC | FKC Table A4, p39 |  |  |
| GenEval (1000 prompts), text-to-image generation on GenEval | Color attribution | 0.22 | FKC | FKC Table A4, p39 |  |  |
| GenEval (553 prompts), GenEval leaderboard entries quoted in FKC Table A4 | Overall | no rival number |  |  |  |  |
| GenEval (553 prompts), GenEval leaderboard entries quoted in FKC Table A4 | Single object | no rival number |  |  |  |  |
| GenEval (553 prompts), GenEval leaderboard entries quoted in FKC Table A4 | Two object | no rival number |  |  |  |  |
| GenEval (553 prompts), GenEval leaderboard entries quoted in FKC Table A4 | Counting | no rival number |  |  |  |  |
| GenEval (553 prompts), GenEval leaderboard entries quoted in FKC Table A4 | Colors | no rival number |  |  |  |  |
| GenEval (553 prompts), GenEval leaderboard entries quoted in FKC Table A4 | Position | no rival number |  |  |  |  |
| GenEval (553 prompts), GenEval leaderboard entries quoted in FKC Table A4 | Color attribution | no rival number |  |  |  |  |
| GenEval single, single, SD3.5 Medium | GenEval category score | no rival number |  |  |  |  |
| GenEval two, two, SD3.5 Medium | GenEval category score | no rival number |  |  |  |  |
| GenEval count, count, SD3.5 Medium | GenEval category score | no rival number |  |  |  |  |
| GenEval color, color, SD3.5 Medium | GenEval category score | no rival number |  |  |  |  |
| GenEval pos, pos, SD3.5 Medium | GenEval category score | no rival number |  |  |  |  |
| GenEval attr, attr, SD3.5 Medium | GenEval category score | no rival number |  |  |  |  |
| GenEval, Overall (macro average), SD3.5 Medium | GenEval Overall | no rival number |  |  |  |  |
| GenEval single, single, FLUX.2 [klein] 4B Base | GenEval category score | no rival number |  |  |  |  |
| GenEval two, two, FLUX.2 [klein] 4B Base | GenEval category score | no rival number |  |  |  |  |
| GenEval count, count, FLUX.2 [klein] 4B Base | GenEval category score | no rival number |  |  |  |  |
| GenEval color, color, FLUX.2 [klein] 4B Base | GenEval category score | no rival number |  |  |  |  |
| GenEval pos, pos, FLUX.2 [klein] 4B Base | GenEval category score | no rival number |  |  |  |  |
| GenEval attr, attr, FLUX.2 [klein] 4B Base | GenEval category score | no rival number |  |  |  |  |
| GenEval, Overall (macro average), FLUX.2 [klein] 4B Base | GenEval Overall | no rival number |  |  |  |  |

**Where it breaks.** Position and Color attribution stay far below Single object: at the lower of the two guidance weights FKC scores 0.10 and 0.22 against 0.99. Seen in FKC Fig A5 p39. Position and Color attribution stay far below Single object: at the higher of the two guidance weights FKC scores 0.13 and 0.23 against 0.99. Seen in FKC Fig A5 p39. Position and Color attribution stay far below Single object: at the lower of the two guidance weights FKC scores 0.13 and 0.21 against 0.99. Seen in FKC Fig A5 p39. Position and Color attribution stay far below Single object: at the higher of the two guidance weights FKC scores 0.13 and 0.22 against 0.99. Seen in FKC Fig A5 p39. In the leaderboard entries FKC quotes, SDXL scores 0.15 on Position and 0.23 on Color attribution against 0.98 on Single object. Seen in FKC Fig A5 p39. With CFG, SD3.5 Medium scores 0.226 on pos and 0.484 on count against 1.000 on single. Seen in CFGRevisited Fig 1 p4; CFGRevisited Fig 2 p5. With CFG, FLUX.2 [klein] scores 0.560 on pos and 0.630 on attr against 0.988 on single. Seen in CFGRevisited Fig 1 p4; CFGRevisited Fig 2 p5.

**What matching it costs.** Inference used V100 32GB GPUs. (CFGRevisited p2)

### Attend-and-Excite prompts

| Benchmark | Metric | Best published | Holder | Where | Printed above it | Disputed |
|---|---|---|---|---|---|---|
| Attend-and-Excite Animal-Animal, two subjects: a [animalA] and a [animalB] | Average CLIP text-text similarity | 0.806 | AttendExcite | AttendExcite Table 1, p7 |  |  |
| Attend-and-Excite Animal-Object, two subjects: a [animal] and a [color][object] | Average CLIP text-text similarity | 0.830 | AttendExcite | AttendExcite Table 1, p7 |  |  |
| Attend-and-Excite Object-Object, two subjects: a [colorA][objectA] and a [colorB][objectB] | Average CLIP text-text similarity | 0.811 | AttendExcite | AttendExcite Table 1, p7 |  |  |
| Attend-and-Excite Animal-Animal, two subjects: a [animalA] and a [animalB] | Share of responses preferring the method (%) | 90.70 | AttendExcite | AttendExcite Table 2, p8 |  |  |
| Attend-and-Excite Animal-Object, two subjects: a [animal] and a [color][object] | Share of responses preferring the method (%) | 77.64 | AttendExcite | AttendExcite Table 2, p8 |  |  |
| Attend-and-Excite Object-Object, two subjects: a [colorA][objectA] and a [colorB][objectB] | Share of responses preferring the method (%) | 77.16 | AttendExcite | AttendExcite Table 2, p8 |  |  |
| Attend-Excite Animal-Animal, two-concept prompts: animal and animal | BLIP-VQA | 0.7441 | CO3 | CO3 Table 1, p7 |  | CO3's 50-step score is 0.7441 in Tables 1, 3, 6 and 7 (p7, p8, p18) and 0.7440 in Table 8 (p18), and Table 8's "SDXL (50 steps)" row carries Table 7's 10-step SDXL score 0.6428 where the 50-step score is 0.6950 (p18). |
| Attend-Excite Animal-Object, two-concept prompts: animal and object | BLIP-VQA | 0.8878 | CO3 | CO3 Table 1, p7 |  | CO3's 50-step score is 0.8878 in Tables 1, 3, 6 and 7 (p7, p8, p18) and 0.8877 in Table 8 (p18), and Table 8's "SDXL (50 steps)" row carries Table 7's 10-step SDXL score 0.8378 where Table 7's 50-step score is 0.8658 and Tables 1 and 3 print that SDXL score as 0.8654 (p7, p8, p18). |
| Attend-Excite Object-Object, two-concept prompts: colored object and colored object | BLIP-VQA | 0.6440 | ToMe | CO3 Table 1, p7 |  | ToMe's 0.6440 is the same in Tables 1 and 8 (p7, p18), but in this column Table 8's "SDXL (50 steps)" row carries Table 7's 10-step SDXL score 0.4752 where the 50-step score is 0.4926 (p18). |
| Attend-Excite Animal-Animal, two-concept prompts: animal and animal | ImageReward | 1.2342 | CO3 | CO3 Table 1, p7 |  | CO3's 50-step score is 1.2342 in Tables 1 and 3 (p7, p8) and 1.2341 in Tables 6, 7 and 8 (p18), and Table 8's "SDXL (50 steps)" row carries Table 7's 10-step SDXL score 0.4906 where the 50-step score is 0.7820 (p18). |
| Attend-Excite Animal-Object, two-concept prompts: animal and object | ImageReward | 1.6744 | CO3 | CO3 Table 1, p7 |  | CO3's 50-step score is 1.6744 in Tables 1 and 3 (p7, p8) and 1.6743 in Tables 6, 7 and 8 (p18), the same SDXL run prints 1.5574 in Tables 1, 6 and 7 (p7, p18) and 1.5474 in Table 3 (p8), and Table 8's "SDXL (50 steps)" row carries Table 7's 10-step SDXL score 1.1503 (p18). |
| Attend-Excite Object-Object, two-concept prompts: colored object and colored object | ImageReward | 1.0158 | CO3 | CO3 Table 1, p7 |  | In this column Table 8's "SDXL (50 steps)" row carries Table 7's 10-step SDXL score 0.1131 where the 50-step score is 0.6789 (p18). |
| Attend-Excite Animal-Animal, two-concept prompts: animal and animal | DINOv2 | 0.2623 | CO3 | CO3 Table 1, p7 |  |  |
| Attend-Excite Animal-Object, two-concept prompts: animal and object | DINOv2 | 0.3259 | CO3 | CO3 Table 1, p7 |  |  |
| Attend-Excite Object-Object, two-concept prompts: colored object and colored object | DINOv2 | 0.3283 | CO3 | CO3 Table 1, p7 |  |  |
| Attend-Excite Animal-Animal, two-concept prompts: animal and animal | ImageReward | 1.380 | CO3 | CO3 Table 4, p9 |  |  |
| Attend-Excite Animal-Object, two-concept prompts: animal and object | ImageReward | 1.770 | CO3 | CO3 Table 4, p9 |  |  |
| Attend-Excite Object-Object, two-concept prompts: colored object and colored object | ImageReward | 1.081 | CO3 | CO3 Table 4, p9 |  |  |
| Attend-Excite Animal-Animal, two-concept prompts: animal and animal | BLIP-VQA | 0.746 | CO3 | CO3 Table 4, p9 |  |  |
| Attend-Excite Animal-Object, two-concept prompts: animal and object | BLIP-VQA | 0.864 | CO3 | CO3 Table 4, p9 |  | Table 4 highlights PixArt's 0.861 as the best Animal-Object BLIP-VQA score, while PixArt + CO3 prints 0.864 (p9). |
| Attend-Excite Object-Object, two-concept prompts: colored object and colored object | BLIP-VQA | 0.581 | CO3 | CO3 Table 4, p9 |  |  |
| Attend-Excite Animals, two-concept prompts | Human Evaluation | 0.8817 | CO3 | CO3 Table 5, p17 |  |  |
| Attend-Excite Animals-Objects, two-concept prompts | Human Evaluation | 0.7466 | CO3 | CO3 Table 5, p17 |  |  |
| Attend-Excite Objects, two-concept prompts | Human Evaluation | 0.8250 | CO3 | CO3 Table 5, p17 |  |  |
| Animals (Attend-Excite category; this table names no benchmark), two-concept prompts | BLIP-VQA | 0.7576 | CO3 | CO3 Table 10, p20 |  |  |
| Animals (Attend-Excite category; this table names no benchmark), two-concept prompts | ImageReward | 1.3149 | CO3 | CO3 Table 10, p20 |  |  |
| Animals Objects (Attend-Excite category; this table names no benchmark), two-concept prompts | ImageReward | 1.7183 | CO3 | CO3 Table 12, p21 |  |  |
| Attend-and-Excite Animals, 2 animals | CLIP | 0.342 | LiftScores | LiftScores Table 3, p8 |  |  |
| Attend-and-Excite Animals, 2 animals | IR (ImageReward) | 1.244 | LiftScores | LiftScores Table 3, p8 |  |  |
| Attend-and-Excite Object&Animal, 1 animal and 1 object | CLIP | 0.364 | LiftScores | LiftScores Table 3, p8 |  |  |
| Attend-and-Excite Object&Animal, 1 animal and 1 object | IR (ImageReward) | 1.706 | LiftScores | LiftScores Table 3, p8 |  |  |
| Attend-and-Excite Objects, 2 objects | CLIP | 0.367 | LiftScores | LiftScores Table 3, p8 |  |  |
| Attend-and-Excite Objects, 2 objects | IR (ImageReward) | 0.896 | LiftScores | LiftScores Table 3, p8 |  |  |
| Attend-and-Excite Animals, 2 animals | CLIP | 0.346 | LiftScores | LiftScores Table 8, p23 |  |  |
| Attend-and-Excite Animals, 2 animals | IR (ImageReward) | 1.337 | LiftScores | LiftScores Table 8, p23 |  |  |
| Attend-and-Excite Object&Animal, 1 animal and 1 object | CLIP | 0.365 | LiftScores | LiftScores Table 8, p23 |  |  |
| Attend-and-Excite Object&Animal, 1 animal and 1 object | IR (ImageReward) | 1.516 | LiftScores | LiftScores Table 8, p23 |  |  |
| Attend-and-Excite Objects, 2 objects | CLIP | 0.371 | LiftScores | LiftScores Table 8, p23 |  |  |
| Attend-and-Excite Objects, 2 objects | IR (ImageReward) | 1.246 | LiftScores | LiftScores Table 8, p23 |  |  |
| Attend-and-Excite Animals, 2 animals | TIFA | 0.949 | LiftScores | LiftScores Table 11, p27 |  |  |
| Attend-and-Excite Object&Animal, 1 animal and 1 object | TIFA | 0.974 | LiftScores | LiftScores Table 11, p27 |  |  |
| Attend-and-Excite Objects, 2 objects | TIFA | 0.790 | LiftScores | LiftScores Table 11, p27 |  |  |
| Attend-and-Excite Animals, 2 animals | minCLIP | 0.250 | LiftScores | LiftScores Table 12, p28 |  |  |
| Attend-and-Excite Object&Animal, 1 animal and 1 object | minCLIP | 0.271 | LiftScores | LiftScores Table 12, p28 |  |  |
| Attend-and-Excite Objects, 2 objects | minCLIP | 0.271 | LiftScores | LiftScores Table 12, p28 |  |  |

**Where it breaks.** Stable Diffusion drops or fuses a subject on these prompts, with CLIP text-text similarity 0.765 on Object-Object against 0.811 for Attend-and-Excite. Seen in AttendExcite Fig 5 p5. In the user study, Stable Diffusion gets 2.32% of responses on Animal-Animal and Attend-and-Excite gets 90.70%. Seen in AttendExcite Fig 5 p5. Object-Object is the lowest category: SDXL scores BLIP-VQA 0.4926 there against 0.8654 on Animal-Object, and CO3 scores 0.5146 against 0.8878. Seen in CO3 Fig 3 p8; CO3 Fig 9 p23. On PixArt-Sigma, Object-Object is the lowest BLIP-VQA category: 0.559 for PixArt and 0.581 with CO3. Seen in CO3 Fig 6 p9; CO3 Fig 8 p22. Human evaluators score plain SDXL lowest on Animals, at 0.5266, where CO3 scores 0.8817. Seen in CO3 Fig 3 p8; CO3 Fig 9 p23. ImageReward is lowest on Objects, 0.662 for SDXL against 1.621 on Object&Animal. CompLift scores only the samples it accepts. Seen in LiftScores Fig 10 p9; LiftScores Fig 20 p23. Without the CompLift filter, Attend-and-Excite scores ImageReward 1.219 on Objects against 1.471 on Object&Animal. CompLift scores only the samples it accepts. Seen in LiftScores Fig 10 p9; LiftScores Fig 20 p23. TIFA is lowest on Objects, 0.755 for SDXL against 0.964 on Object&Animal. CompLift scores only the samples it accepts. Seen in LiftScores Fig 10 p9; LiftScores Fig 20 p23. minCLIP for SDXL is lowest on Animals at 0.243, against 0.269 on Object&Animal. CompLift scores only the samples it accepts. Seen in LiftScores Fig 10 p9; LiftScores Fig 20 p23.

**What matching it costs.** About 15.4 seconds per image on an A100 GPU when iterative refinement is used. (AttendExcite p10) All experiments ran on a single NVIDIA A6000 GPU (48GB). (CO3 p6)
