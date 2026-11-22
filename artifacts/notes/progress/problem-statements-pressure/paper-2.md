# Pressure test: Paper 2, restoring plurality in one model

## Part 1: The idea, distilled

The setting is one pretrained text-to-image model with the joint prompt allowed. A checker $C_t$ is trained on labelled renders to read, from a partly denoised image, whether two separate objects of the right concepts are present. Its gradient is added on steps 0 to 15:

$$
\hat\epsilon_t = \epsilon^{\text{base}}_t - \min\!\Big(1,\ \frac{0.4\,\|\epsilon^{\text{PoE}}_t\|}{\|\lambda\sigma_t g_t\|}\Big)\lambda\sigma_t g_t,
\qquad g_t=\nabla_{x_t}\log C_t(y=1\mid x_t, c_A, c_B),\quad \lambda=30
$$

The base is the product of experts or CO3. This form reads the cap as a ratio of norms. It also adds the concept inputs $c_A, c_B$, which the statement's formula omits and which any prompt outside the training pool requires.

## Part 2: The load-bearing claim

**Load-bearing**

On concepts the checker never trained on, its early-step gradient raises the rate of two distinct, correct objects above the best joint-prompt sampler. No valid run tests this. The push has never been added to CO3, and its cat and dog counts are on the void list.

**Supporting**

SDXL still fails on part of the two-object prompts, and a checker can be trained at scale.

**Unstated premises**

1. The state $x_t$ at steps 0 to 15 holds a readable signal about whether the final image will separate.
2. $C_t$ stays valid on the base it guides. It estimates $p(y\mid x_t)$ under the sampler that made its training states, and CO3 moves that distribution. The statement does not name that sampler.
3. The gradient's direction does the work. When the cap binds, $\lambda=30$ has no effect on the step size. The cap is also defined on the product's prediction when the base is CO3.

## Part 3: Where this sits in the literature

Gradient guidance from a detector or classifier for multi-object generation already exists. The five papers below were opened for this test and are absent from the [field table](../field-table.md).

| Paper | What it does | What still differs |
|---|---|---|
| [Correcting Diffusion Generation through Resampling](https://arxiv.org/pdf/2312.06038#page=6) | Its D-Guidance baseline adds the gradient of a detector and discriminator ratio to Stable Diffusion 2.1 to fix missing objects | Detector reads the predicted clean image, runs only at $\sigma<5$ and scores presence |
| [Counting Guidance](https://arxiv.org/pdf/2306.17567#page=6) | Adds one counter gradient per object type, separated with attention masks | Off-the-shelf counter on the predicted clean image |
| [Detector Guidance](https://arxiv.org/pdf/2306.02236#page=3) | Trains a detector on cross-attention maps to stop two objects merging into a hybrid | Acts by editing later attention maps |
| [Universal Guidance](https://arxiv.org/pdf/2302.07121#page=7) | Guides Stable Diffusion with an object detector's gradient | Needs bounding boxes as input |
| [Dynamic CFG via Online Feedback](https://arxiv.org/pdf/2509.16131#page=8) | Trains evaluators on noisy latents and tests gradient guidance from them | Targets general alignment and image fidelity |

Twelve web searches found no paper that trains a noise-conditioned checker on fused versus separate labels. That label, used at early steps, is the remaining contribution. A search cannot prove the gap is empty.

## Part 4: What is likely right

The failure is real on SDXL. CO3 scores 0.7500 on GenEval's two-object task in this project's run (fraction of 396 images judged correct, no interval), and [Feynman-Kac Correctors' Table A4](https://arxiv.org/pdf/2503.02819#page=39) prints 0.74 and 0.80 for plain SDXL on that task. Noise-aware training also targets a documented failure: the resampling paper limits its guidance to low noise because there "the object detector barely detects any object" at high noise ([page 14](https://arxiv.org/pdf/2312.06038#page=14)).

## Part 5: What is likely wrong

**The closest published test barely moved the outcome**

In the [resampling paper's Figure 4](https://arxiv.org/pdf/2312.06038#page=6), the y axis is FID and the x axis is object occurrence, the share of named objects a detector finds on a 0 to 1 scale. D-Guidance moves occurrence from about 0.55 to 0.56 on GPT-Synthetic and stays near 0.71 on MS-COCO (values read off the plot). Resampling with the same signal reaches about 0.74 and 0.85 with 15 images per caption. That run used a clean-image detector at low noise, so the early noise-aware version remains open. The burden of proof sits on it.

**The five benchmark rows do not score the claim**

Color, Shape and Texture use prompts like "a red book and a yellow vase" and ask one question per object and attribute phrase ([T2I-CompBench, page 6](https://arxiv.org/pdf/2307.06350#page=6)). The checker as specified reads no attribute. Complex prompts hold more than two objects or sub-categories ([page 2](https://arxiv.org/pdf/2307.06350#page=2)). RareBench Concat prompts look like "a hairy shark and two wigged octopuses" ([RareBench](https://arxiv.org/abs/2410.22376)). The Complex and Concat targets use ImageReward, a human-preference score that counts no objects. GenEval's two-object task, the number that motivates the paper, is missing from the five rows, and AE-Bench is unusable because the checker's pool was rendered from it.

**Two of the five targets are out of date**

The [benchmark spec](../benchmark-spec.md#the-10-benchmarks-to-beat) names CO3 as the number to beat on Complex (ImageReward 0.4406) and Concat (0.1610). [Rectify Then Diffuse's Table 2](https://arxiv.org/pdf/2608.03135#page=6) prints 0.4661 and 0.4464 on those rows. The statement lists this rival and leaves it out of its rerun list.

**The product is the wrong base for this setting**

With the joint prompt allowed, a user has no reason to start from the product. The informative bases are the joint prompt, CO3 and Rectify Then Diffuse, and the equation offers only the product and CO3.

**The motivating gap is unmeasured and may be SDXL-specific**

The statement gives no plain SDXL score from its own run, so 0.7500 cannot show what CO3 adds on this task. [CFG Revisited's Table 5](https://arxiv.org/pdf/2608.16786#page=9) prints plain guidance two-object scores of 0.833 on SD3.5 Medium and 0.889 on FLUX.2 [klein], with image counts unstated. A reviewer will ask why the work stops at SDXL.

## Part 6: Verdict

**Promising but needs rework.** The problem is real on SDXL and a noise-aware fused-versus-separate checker appears unpublished, but the method has no valid result, its closest published relative gained almost nothing from the gradient, and the chosen benchmarks do not measure two separate objects.

## Part 7: What to do next

1. **Move the evaluation onto the claim.** Rebuild the checker's pool from prompts and concepts outside GenEval and AE-Bench, so GenEval two-object and AE-Bench can be the main measures. Keep the five rows as secondary and add Rectify Then Diffuse to the reruns. GenEval's judge is still usable here, since [GenEval 2](https://arxiv.org/pdf/2512.16853#page=5) reports its deviation from human scores on SDXL within 3%.
2. **Specify the checker.** State its concept inputs, the sampler that produced its training states, and how the cap works on a CO3 base.
3. **Run the smallest decisive comparison.** On held-out concepts and matched seeds, compare CO3 alone, CO3 plus the checker gradient, CO3 plus a random direction of equal size, and CO3 with the same checker selecting among candidates at equal model evaluations. Score with GenEval's two-object judge and a human count of separate objects. Write the support, null and inconclusive rules before the first run.

A gradient that beats both controls supports the paper. A random direction that matches it means the checker's direction adds nothing. Selection that beats it repeats the resampling paper's result, and the contribution becomes a selector.
