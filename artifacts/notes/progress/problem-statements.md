# The two problem statements

Each statement says what the paper tries to solve, what it may use, what the evidence shows today and which published
work it has to beat. The field comes from the [field table](field-table.md), the rivals' protocols from the
[benchmark spec](benchmark-spec.md), and the state of our own results from [what we know](what-we-know.md).

## Table of contents

- [Paper 1: composing experts that never met](#paper-1-composing-experts-that-never-met)
- [Paper 2: restoring plurality in one model](#paper-2-restoring-plurality-in-one-model)

## Paper 1: composing experts that never met

**The problem**

Two diffusion models are trained on disjoint data, so neither has seen an image holding both concepts, and no joint
prompt or joint dataset exists. Each model's score pulls the whole image toward its own kind of picture, so their plain
product has no reason to place two separate objects. The question is whether the two trained models, plus a small
helper trained without joint images, can draw both concepts as separate objects.

**The setting**

Allowed: the two trained models, samples drawn from them, and a helper trained on those samples. Excluded: any real image
holding both concepts, any prompt naming both, and any model trained on the union of the two datasets.

**The proposed method**

Each model denoises only its own region, and a trained checker guides the combined sample:

$$
\hat\epsilon_t(x_t) = \mathrm{merge}\big(\epsilon^A_t(x_t|_{R_A}),\ \epsilon^B_t(x_t|_{R_B})\big) - \lambda_t\,\sigma_t\,\nabla_{x_t}\log C_t(\text{both concepts present}\mid x_t)
$$

$R_A$ and $R_B$ are the two regions, $\epsilon^A_t$ and $\epsilon^B_t$ are the two models' noise predictions at step $t$,
$C_t$ is the checker, and $\lambda_t$ is the strength of its push.

**The evidence today**

The strongest count is on the void list: 1,200 clean pairs of 1,280 by the digit-reading judge, on 20 MNIST digit pairs
kept out of tuning, with each model on a fixed half of a 32 by 64 pixel canvas. It stays out of any claim until the owner
decides it. Without fixed halves, the best run on one shared canvas draws 121 of 320 (38%) on five tuning pairs, against
a bar of 288. With each model free to draw on either side, two pushes draw 4 of 320. Every result so far is in pixel
space on MNIST or a toy. No shared latent space has been built.

**The closest published work**

Projective Composition composes location-conditioned models that saw only single-object CLEVR images, and 98 of 100
hand-counted images hold all 5 objects. MultiDiffusion reconciles one model across fixed regions. Diffusion Domain
Expansion trains a coordinator on one model's outputs. SuperDiff composes two models trained on disjoint CIFAR-10 halves
and samples their mixture. CoInD trains one model with a conditional-independence penalty. Compositional Sculpting trains a
classifier on samples from pretrained models and guides their composition, and its harmonic mean is undefined on disjoint
supports.

**What it is measured on**

The four Paper 1 rows in the [benchmark spec](benchmark-spec.md#the-10-benchmarks-to-beat): CLEVR by the most objects that succeed, Shapes3D,
CMNIST AND and CelebA. SuperDiff's disjoint CIFAR-10 halves are context, because that task samples from either model. The CLEVR target is 6 objects, one past the 5 that Bradley's local patch model reaches after training on single objects.

**The pressure-test verdict: promising but needs rework**

The problem is real, and neither the field table nor four web searches found a paper that solves it. With the checker off,
fixed regions are two independent samples pasted side by side, so the fixed-halves method makes a claim only where the
checker beats that paste. The form that matters is the one where the sampler chooses the placement, and it draws 121 of 320
on a shared canvas. None of the four benchmarks has both disjoint models and several separate objects. The
[full test](problem-statements-pressure/paper-1.md) names the next checks.

**What our records add to the verdict**

The test reports no paste baseline. The [final digit run](../../../2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md#step-8-the-final-test-on-unseen-pairs)
holds the nearest one: the same method without pushes draws 1,043 of 1,280, against 1,200 with them. Both come from the run on
the void list. That run still chooses its side with an early vote, so an exact sample-each-and-paste control has not been run.

**Papers the test adds, to verify before citing**

[DiffCollage](https://arxiv.org/abs/2303.17076) composes a score from factor and overlap scores, and
[MMDisCo](https://arxiv.org/abs/2405.17842) couples two pretrained models with a small discriminator trained on real pairs.

## Paper 2: restoring plurality in one model

**The problem**

One text-to-image model has seen both concepts, yet combining its predictions for two prompts as a product of experts
tends to fuse them into one hybrid object. CO3, the strongest sampler correction measured here, still misses part of
two-object prompts: in our run it scores 0.7500 on GenEval's two-object task, the fraction of 396 images judged correct,
with no interval. The question is whether a checker trained on what a separate pair looks like can raise the rate at which
the model draws two distinct objects, added to the product and to CO3.

**The setting**

Allowed: one pretrained text-to-image model, the joint prompt, and a checker trained on labelled renders. Doing without the
joint prompt is no contribution for this paper.

**The proposed method**

A checker trained at scale reads whether a partly denoised image holds two separate objects and whether each is the right
concept. Its gradient is added during the early steps:

$$
\hat\epsilon_t(x_t) = \epsilon^{\text{base}}_t(x_t) - \lambda_t\,\sigma_t\,\nabla_{x_t}\log C_t(\text{two separate, correct objects}\mid x_t)
$$

$\epsilon^{\text{base}}_t$ is the product's prediction or CO3's, $C_t$ is the checker, and $\lambda_t$ is its strength. The
kept recipe uses strength 30 on steps 0 to 15 with the push capped at 0.4 of the product's prediction.

**The evidence today**

On 128 cat and dog seeds, counted clean or near-clean by eye: plain product 0, kept push 54, joint prompt 69, CO3 101. The
scorer gives 0, 35, 57 and 89. The push's counts are on the void list, because each improvement round was kept or dropped
on these same seeds. No run has added the push to CO3, so the method this statement proposes has no result yet. The checker's
4,400-picture pool came from Attend-and-Excite renders, so it cannot be scored fairly on those prompts.

**The closest published work**

CO3 corrects the sampler with the joint and single-concept predictions. Attend-and-Excite, ToMe, Rectify Then Diffuse, ISAC
and Be Decisive work through attention or the starting noise. CoMat fine-tunes the model with a concept-matching loss.

**What it is measured on**

The six Paper 2 rows in the [benchmark spec](benchmark-spec.md#the-10-benchmarks-to-beat): T2I-CompBench Color, Shape, Texture
and Complex, GenEval two objects, and CLEVR by the most objects that succeed. RareBench Concat is a reported row. T2I-CompBench needs plain SDXL, ToMe and CoMat rerun on one setup before any win is claimed.

**The pressure-test verdict: promising but needs rework**

The problem is real on SDXL, and twelve web searches found no paper that trains a noise-aware checker on fused against separate
labels. The method has no valid result yet. In the closest published test, [Correcting Diffusion Generation through
Resampling](https://arxiv.org/pdf/2312.06038#page=6), a detector and discriminator gradient moves object occurrence (the share
of named objects found, 0 to 1) from about 0.55 to 0.56 on GPT-Synthetic, read off its Figure 4. Resampling with the same signal
reaches about 0.74. That test used a clean-image detector at low noise, so the early noise-aware version is still open. Of the
six benchmark rows, GenEval's two-object task and CLEVR count separate objects. The [full test](problem-statements-pressure/paper-2.md) names the next
checks.

**What our check adds to the verdict**

The Complex target and the reported RareBench Concat row come from [Rectify Then Diffuse's Table 2](https://arxiv.org/pdf/2608.03135#page=6),
which prints ImageReward 0.4661 and 0.4464 against CO3's 0.4406 and 0.1610. The paper's footnote about copied CO3 numbers is
marked on its Table 4, so it does not cover Table 2. Table 2's baseline ImageReward cells match CO3's own table on 19 of 20
values, so CO3 was probably not rerun, and the two methods should not be read as compared on one setup.

**Papers the test adds, to verify before citing**

[Counting Guidance](https://arxiv.org/abs/2306.17567), [Detector Guidance](https://arxiv.org/abs/2306.02236),
[Universal Guidance](https://arxiv.org/abs/2302.07121), [Dynamic CFG via Online Feedback](https://arxiv.org/abs/2509.16131) and
RareBench's source, [Rare-to-Frequent](https://arxiv.org/abs/2410.22376).
