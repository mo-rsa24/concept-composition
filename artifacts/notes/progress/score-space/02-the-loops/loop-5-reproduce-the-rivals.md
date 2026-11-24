# Loop 5: rerun each rival's own code and check its printed number

Loop 5 reruns the rivals' released code on our cluster to see whether each one gives the number its paper prints, so that the targets of both papers rest on numbers that hold on our setup. It serves both papers, the separate-experts paper and the CO3 paper. Its owner code on the board is `s5`, and its one board step, D2 "Rivals reproduce one published number", is **done** (finished 2026-11-23, [the board](../../board.json)).

## The idea

Each rival's code is run as its authors shipped it, on the prompts, seeds and scorer its paper names, and our score is set beside the printed one. Plain SDXL runs on the same prompts, seeds and starting noise, which checks our prompts and scorer against the plain SDXL number the paper also prints.

One real case: Rectify Then Diffuse (RTD) on T2I-CompBench Complex. The authors' code at commit `241eb37` rendered the 988 of 1,000 prompts that the project's subject guard (`scripts/shared/disallowed_subjects.py`, which keeps some subjects out of every render) allows, at seeds 0, 1 and 2, at 1024 by 1024 pixels with 50 DDIM steps and guidance 5.0 ([the RTD design](../../reproductions/RTD/design.md)). ImageReward, a learned score of how well a picture matches its prompt where higher is better, averaged 0.2423 against the printed 0.4661 ([the RTD review](../../reproductions/RTD/review.md)). On the lemon and lime prompt, RTD's picture shows separate lemons and limes as a flat pattern and scores -0.90, while a plain SDXL picture of limes with lemon-coloured flesh scores -0.02 ([the RTD review](../../reproductions/RTD/review.md)).

## The principle

A printed number can serve as a bar only if it holds on the setup our own method will run on. The loop rests on two assumptions. The first is that the authors' code at the printed settings, scored by the same scorer, gives the printed number within sampling error. The second is that plain SDXL reproducing its own printed score shows the prompts and scorer are right, so a rival that then misses its number points to the rival or its settings ([the RTD design](../../reproductions/RTD/design.md), "Plain SDXL takes the same test against its printed scores").

Before any score is believed, each method's loop with its correction set to zero must reproduce the plain sampler to under 1e-5 (the largest absolute difference in the final latent), and the real correction must move the final latent by more than 1e-3 ([the session prompt](../../prompts/session-5-reproduce-and-choose.md); [the CO3 design](../../reproductions/CO3/design.md), "The ladder").

## The theory, with its equation

Loop 5's CO3 design states CO3's rule in words only: "CO3 corrects the early sampling steps of SDXL by moving away from what each single concept would draw on its own" ([the CO3 design](../../reproductions/CO3/design.md)). The equations below are typeset from the notes and paper pages that print them.

**The product of experts.** Composable Diffusion's conjunction, as [the Composable Diffusion note](../01-reading/composable-diffusion.md) prints it (Eq. 11, [p. 7](https://arxiv.org/pdf/2206.01714#page=7)):

$$
\hat\epsilon(x_t,t) = \epsilon_\theta(x_t,t) + \sum_{i=1}^{n} w_i\big(\epsilon_\theta(x_t,t\mid c_i) - \epsilon_\theta(x_t,t)\big)
$$

- $x_t$: the noisy latent at step $t$. Every concept reads the same $x_t$, so changing it changes all the predictions together.
- $\epsilon_\theta(x_t,t)$: the unconditional prediction. It is the base every concept's push is measured from.
- $\epsilon_\theta(x_t,t\mid c_i)$: the prediction under concept $c_i$ alone. The difference from the unconditional prediction is that concept's push.
- $w_i$: the weight on concept $c_i$. Raising it strengthens that concept's push; with all $w_i = 1$ the sum is $\epsilon_\varnothing + d_t(a) + d_t(b)$ for two concepts (inferred in the same note).
- $n$: the number of concepts. Adding a concept adds one more push and one more network call per step.

CO3's paper prints the same sum with per-concept guidance weights $\lambda_k$ (Eq. 10, [CO3 p. 3](https://arxiv.org/pdf/2509.25940v2#page=3)) and says it "is incorrect or doesn't correspond to the score of the diffusion forward distribution" at any $t > 0$ ([CO3 p. 4](https://arxiv.org/pdf/2509.25940v2#page=4)).

**CO3's corrector.** CO3 defines a target that favours the joint prompt and divides out each single concept (Eq. 12, [CO3 p. 4](https://arxiv.org/pdf/2509.25940v2#page=4)):

$$
\tilde p_t(x_t, C) \propto \frac{p_t(x_t\mid C)^{w_0}}{\prod_{k=1}^{K} p_t(x_t\mid c_k)^{w_k}}
$$

It composes in the space of denoised estimates (Eq. 13, [p. 4](https://arxiv.org/pdf/2509.25940v2#page=4)), which equals a guided noise prediction when the weights sum to 1 (Proposition 1a, [p. 5](https://arxiv.org/pdf/2509.25940v2#page=5)):

$$
\tilde x^{\text{tweedie}} = \sum_{k=0}^{K} w_k\, \hat x^{\text{tweedie}}\big[\epsilon^{\lambda,c_k}_t\big] = x_t - \sigma_t\,\tilde\epsilon^{\lambda,C}_t, \qquad \tilde\epsilon^{\lambda,C}_t = \epsilon^{\varnothing}_t + \lambda\Big(\sum_{k=0}^{K} w_k\,\epsilon^{c_k}_t - \epsilon^{\varnothing}_t\Big), \qquad \sum_{k=0}^{K} w_k = 1
$$

The concept weights come from how close the joint prediction is to each concept's (Eq. 14 and 15, [p. 5](https://arxiv.org/pdf/2509.25940v2#page=5) and [p. 6](https://arxiv.org/pdf/2509.25940v2#page=6)):

$$
d_k = \lVert \epsilon^{C} - \epsilon^{c_k} \rVert, \qquad a_k = \exp(-\beta\, d_k), \qquad w_k = -\frac{a_k}{\sum_{j=1}^{K} a_j}
$$

- $C$ and $c_0$: the joint prompt. Its weight $w_0$ is fixed, at 2.0 for the corrector ([p. 5](https://arxiv.org/pdf/2509.25940v2#page=5)), so the joint prompt pulls hardest.
- $c_k$ for $k \ge 1$: one single concept. Its weight $w_k$ is negative, so its prediction is subtracted and the sample is pushed away from that concept's own picture.
- $d_k$: the distance between the joint prediction and concept $k$'s prediction at the current step. A smaller $d_k$ means the sample already looks like concept $k$, and that concept then gets the larger negative weight.
- $\beta$: how sharply the weights follow the distances. The paper sets 0.8 in section 4.1 ([p. 6](https://arxiv.org/pdf/2509.25940v2#page=6)), and the released script uses 0.9 ([the CO3 design](../../reproductions/CO3/design.md)).
- $\lambda$: the guidance strength inside each prediction. The paper states 5.0 under DDIM in section 4.1, and the released script runs CFG++ at 0.8 ([the CO3 design](../../reproductions/CO3/design.md)).
- $\sigma_t$: the noise scale at step $t$, $\sqrt{1-\bar\alpha_t}$. It converts a noise prediction into a denoised estimate.
- The sum $\sum_k w_k$: at 1 the step corrects the current sample (the corrector), and at 0 it replaces the sample with weighted noise (the resampler, Proposition 1b, [p. 5](https://arxiv.org/pdf/2509.25940v2#page=5)). CO3 acts only during the first 20% of the 50 DDIM steps ($T_c = 0.8T$): the resampler on the first 3 steps, the corrector on the rest of that window, and plain sampling after it ([p. 6](https://arxiv.org/pdf/2509.25940v2#page=6)).

Eq. 12 divides by $p_t(x_t\mid c_k)^{w_k}$ while Eq. 13 states $w_k < 0$; read together, each single concept enters with a negative sign (inferred). The CO3 design records that the released corrector rescales by largest entries where Algorithm 3 prints a ratio of norms ([the CO3 design](../../reproductions/CO3/design.md)).

## The objective, with its bar

The measured quantity is each rival's score on its own benchmark metric, on the prompts the subject guard allows: ImageReward averaged over pictures (higher is better) on T2I-CompBench Complex, and BLIP-VQA (0 to 1, higher is better) on Color, Shape and Texture ([table.csv](../../reproductions/table.csv)).

The bar for a score per picture, as the CO3 design recorded it before the run ([the CO3 design](../../reproductions/CO3/design.md), "The test, fixed before the first run"):

> ImageReward is scored per picture. The standard error comes from resampling prompts with replacement 10,000 times. CO3 prints no standard error, so the published one is taken as equal to ours. On the allowed prompts, CO3 counts as `reproduced` on a row when |ours − published| ≤ 1.96 × √(SE_ours² + SE_published²), and otherwise as `not reproduced`.

The RTD design records the same rule ([the RTD design](../../reproductions/RTD/design.md)). For a rival that has to be retrained, the SuperDiff design records ([the SuperDiff design](../../reproductions/SuperDiff/design.md)):

> SuperDiff counts as `reproduced` when the published 4.11 lies within mean ± 4.303 × sd × √(1 + 1/3) of the three values, where 4.303 is the two-sided 95% t value for 2 degrees of freedom.

The step's proof, as the session prompt states it: "every pair of a `qualitative` row and a `targets` row with the same `paper` and `dataset` has a row in `table.csv` with a verdict, both commits, and a run id or a filed question" ([the session prompt](../../prompts/session-5-reproduce-and-choose.md)).

## Where it stands

**D2, done.** The board's note reads "table.csv: 20 rows, 2 reproduced, 3 not reproduced, 13 not comparable, 2 blocked; build_table.py --check prints every row meets the proof" ([the board](../../board.json)). [table.csv](../../reproductions/table.csv) holds the same 20 rows with the same counts. The two CLEVR rivals appear twice because both papers have a CLEVR target.

The five rows that got a verdict, each with the review that holds it. "Pass band" is the interval from the test above, built around our score.

| Rival and benchmark | Prompts and seeds | Ours | Pass band | Printed | Verdict | W&B run | Review |
|---|---|---|---|---|---|---|---|
| CoMat, T2I-CompBench Color, BLIP-VQA | 288 of 300, seeds 42 to 51 | 0.7696 | 0.7331 to 0.8060 | 0.7827 | reproduced | `dk8g2tpx` | [the CoMat review](../../reproductions/CoMat/review.md) |
| ToMe, T2I-CompBench Shape, BLIP-VQA | 293 of 300, seed 0 | 0.5502 | 0.4976 to 0.6028 | 0.6051 | not reproduced | `8cowoyea` | [the ToMe review](../../reproductions/ToMe/review.md) |
| ToMe, T2I-CompBench Texture, BLIP-VQA | 300 of 300, seed 0 | 0.6503 | 0.5970 to 0.7036 | 0.6894 | reproduced | `ljypsxnd` | [the ToMe review](../../reproductions/ToMe/review.md) |
| CO3, T2I-CompBench Complex, ImageReward | 988 of 1,000, 3 seeds | 0.3326 | 0.2584 to 0.4069 | 0.4406 | not reproduced | `02fl0ia0` | [the CO3 review](../../reproductions/CO3/review.md) |
| RTD, T2I-CompBench Complex, ImageReward | 988 of 1,000, 3 seeds | 0.2423 | 0.1751 to 0.3095 | 0.4661 | not reproduced | `5i9dt7by` | [the RTD review](../../reproductions/RTD/review.md) |

The plain SDXL checks on the same pipelines: Complex 0.2637 against 0.3083 printed, reproduced (W&B `8802binw`, [the RTD review](../../reproductions/RTD/review.md)); Color 0.6172 against 0.5879 and 0.6369 printed, both inside its band (W&B `gk8ylo47`, [the CoMat review](../../reproductions/CoMat/review.md)); Texture 0.5233 against 0.5637, reproduced; Shape 0.4825 against 0.5408, not reproduced (W&B `92suk4nc`, [the ToMe review](../../reproductions/ToMe/review.md)). By the owner's answer to board question 39, ToMe's Shape row keeps its verdict with that missed check printed beside it.

On matched pictures, the gain over plain SDXL reproduced for ToMe and failed for RTD. ToMe's gain is 0.1271 on Texture against 0.1257 in the paper's printed numbers, and 0.0677 on Shape against 0.0643 ([the ToMe review](../../reproductions/ToMe/review.md)). RTD minus plain SDXL on Complex is -0.0214 (standard error 0.0153), where the printed scores differ by +0.1578 ([the RTD review](../../reproductions/RTD/review.md)). On Complex's 463 two-subject prompts at 3 seeds, CO3's gain over plain SDXL is 0.0950 (standard error 0.0303) and RTD's is 0.0011 (standard error 0.0239), descriptive numbers that change no verdict ([the CO3 review](../../reproductions/CO3/review.md)).

RareBench Concat is a reported row with no target. CO3 scores 0.4374 against 0.1610 printed and RTD 0.4166 against 0.4464, both "reproduced" on a weak test of 33 prompts at 3 seeds ([the CO3 review](../../reproductions/CO3/review.md); [the RTD review](../../reproductions/RTD/review.md)).

**Rows with no result.** Fifteen rows have no score of ours ([table.csv](../../reproductions/table.csv)):

- 2 `blocked`: Bradley's local patch model on CLEVR (Kmax 5, the largest object count every smaller count also passes), once per paper. No code or weights were found, and by board question 49 it is not rebuilt.
- 13 `not comparable`. Five have no published number under the target's protocol (board question 13). Two Color rows were run on Stable Diffusion 2 (board question 25). Three printed numbers stand as context because only the rival with the best printed score per benchmark is rerun (board question 43: CoMat on Shape and Texture, ToMe on Complex). FKC on GenEval is not rerun, at a measured 22.6 GPU-hours, because its printed 0.78 sits below plain SDXL's printed 0.80 (board question 41). Projective Composition's Kmax 4, once per paper, is compared only on our own CLEVR models at equal training length (board question 36).

**Where the sources disagree.**

- The [CO3 review](../../reproductions/CO3/review.md), the [CoMat review](../../reproductions/CoMat/review.md) and the [ToMe review](../../reproductions/ToMe/review.md) each open with "has no verdict yet", while their later sections record the verdicts above, [table.csv](../../reproductions/table.csv) carries them, and [the board](../../board.json) marks D2 done.
- The [Liu design](../../reproductions/Liu/design.md) says Composable Diffusion's CelebA row `CoInD-T6-004` "runs with CoInD" and its Shapes3D row needs a costed question. [What each rival needs](../../reproductions/what-each-rival-needs.md) says CelebA "stopped after its classifier and smoke test" and Shapes3D "is skipped", and [table.csv](../../reproductions/table.csv) holds no CoInD row.

## The closest papers from the reading

- [Compositional Visual Generation with Composable Diffusion Models](../01-reading/composable-diffusion.md): its conjunction is the product rule that the Liu rows reproduce and that CO3 rewrites, and it lists fused hybrids as its own failure ([p. 7](https://arxiv.org/pdf/2206.01714#page=7), [p. 13](https://arxiv.org/pdf/2206.01714#page=13)).
- [Reduce, Reuse, Recycle](../01-reading/reduce-reuse-recycle.md): it shows the sum of noised scores equals the noised product's score only at $t = 0$, the reason CO3 also gives for moving away from score sums ([p. 4](https://arxiv.org/pdf/2302.11552#page=4)).
- [The Superposition of Diffusion Models](../01-reading/superdiff.md): SuperDiff is a Loop 5 rival whose CIFAR-10 row stayed context, and its AND mode draws one animal mixing dog and cat ([p. 40](https://arxiv.org/pdf/2412.17762#page=40)).
- [Diffusion Sampling Path Tells More](../01-reading/sampling-path-tells-more.md): it measures GenEval two-object on SDXL, the benchmark behind the FKC row, and its seed filter lowered SDXL counting from 0.4094 to 0.3250 ([p. 21](https://arxiv.org/pdf/2505.23343#page=21)).

## What it suggests for the separate-experts paper

Loop 5 reproduced no rival on this paper's datasets. Bradley's local patch model is blocked and Projective Composition is not comparable ([table.csv](../../reproductions/table.csv)), Liu's and SuperDiff's CMNIST rows have no number under the target's protocol ([the rows that took the owner's choice](../../reproductions/rows-with-no-matching-number.md)), and SuperDiff's CIFAR-10 row is context ([the SuperDiff design](../../reproductions/SuperDiff/design.md)). The paper's comparison therefore has to run each rival's composition rule on our own CLEVR models at equal training length, as the [comparison protocol](../../targets/comparison-protocol.md) already requires, with the printed Kmax values kept as context (inferred). A score-path representation could add a check on the product rule itself: whether the summed path of two separately trained experts lands where a model that saw both would go, read at low noise where Reduce, Reuse, Recycle says the sum and the noised product agree (inferred). Nothing in Loop 5 measured this.

## What it suggests for the CO3 paper

Keep CO3 as the base the checker's push is added to (inferred). On our pictures CO3 is the best finished rerun on Complex's two-subject prompts, at 0.4628 ([the comparison protocol](../../targets/comparison-protocol.md)), and its gain over plain SDXL there is 0.0950 against RTD's 0.0011 ([the CO3 review](../../reproductions/CO3/review.md)). Report every CO3 baseline with the settings that ran (CFG++ at 0.8, 6 corrected steps, β 0.9), because the void list holds the revision manuscript's DDIM-at-7.5 label for CO3 as possibly false ([the void list](../../void-list.md)). Report gains over plain SDXL on matched pictures beside absolute scores, since ToMe's gains reproduced while two papers print different plain SDXL scores on Shape (0.5408 and 0.4687, [the ToMe review](../../reproductions/ToMe/review.md)) (inferred). A score-path representation could replace the one-step distance $d_k = \lVert \epsilon^{C} - \epsilon^{c_k} \rVert$ that sets CO3's concept weights with a reading over steps 0 to 15, or rank CO3's seeds by summed path length as Diffusion Sampling Path Tells More ranks plain SDXL's (inferred).
