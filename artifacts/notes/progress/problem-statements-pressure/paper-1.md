# Pressure test: Paper 1, composing experts that never met

## Part 1: The idea, distilled

Two diffusion models are trained on disjoint image sets. At sampling, model A predicts noise on region $R_A$, model B predicts noise on $R_B$, and a classifier $C_t$ trained on their samples pushes the merged image toward "both concepts present":

$$
\hat\epsilon_t(x_t) = \mathrm{merge}\big(\epsilon^A_t(x_t|_{R_A}),\ \epsilon^B_t(x_t|_{R_B})\big) - \lambda_t\,\sigma_t\,\nabla_{x_t}\log C_t(\text{both}\mid x_t)
$$

When the regions are a hard partition, the sampler update acts pixel by pixel:

$$
x_{t-1}|_{R_A} = a_t\,x_t|_{R_A} + b_t\,\epsilon^A_t(x_t|_{R_A}) + c_t\,z|_{R_A}
$$

No term from $R_B$ appears. At $\lambda_t = 0$ the two regions are independent draws, $p(x_0) = p_A(x_0|_{R_A})\,p_B(x_0|_{R_B})$. That is the distribution you get by sampling each model alone and pasting the two pictures side by side. Any coupling between the objects has to come through $\nabla\log C_t$.

## Part 2: The load-bearing claim

**Load-bearing:** a checker trained only on samples from A and B can steer the sampler to two separate objects when the regions are not handed to it. If this is false, the method is a paste with extra steps.

**Supporting:** restricting each model to its own region keeps it on inputs like its training data.

**Unstated premises:**

- $R_A$ and $R_B$ come from somewhere. The statement never says where, and this choice decides whether the problem is trivial.
- A positive class for "both present" can be built without joint images. The available build is a composite of an A sample and a B sample, which makes the paste distribution the checker's target.
- Pasting is an inadequate answer. The statement names no case where it fails.
- Two marginals determine how the objects should interact. They do not, because many joint distributions share the same two marginals.

## Part 3: Where this sits in the literature

**Confident (opened in the PDF or in the [field table](../field-table.md)):**

- [Projective Composition](https://arxiv.org/pdf/2502.04549#page=2) is the closest work. Its Theorem 5.3 proves the same decoupling for non-overlapping coordinates, and page 2 says the sampler "acts as if each model individually 'pastes' an object onto an empty background". [Table 1 on page 17](https://arxiv.org/pdf/2502.04549#page=17) gives 98 of 100 images with all 5 objects, using a third model of empty backgrounds trained on 50,000 images. With an unconditional background in its place, 0 of 100 images reach 5 objects. Its objects all come from one 1,550,000-image dataset, so disjoint concepts are untested.
- [Compositional Sculpting](https://arxiv.org/pdf/2309.16115#page=6) already trains a classifier on samples from pretrained models and guides their composition. Its harmonic mean is undefined on disjoint supports (page 6). The statement leaves this paper out of its closest work, and it is the nearest published version of the helper.
- [Composable Diffusion](https://arxiv.org/pdf/2206.01714#page=13) reports "limited success when composing diffusion models trained on different datasets" (page 13), which supports the problem being real.
- [Catastrophic Compositional Generation](https://arxiv.org/abs/2606.23920) shows score error dominating when the composed target lies outside training support. This is a likely cause of the shared-canvas failures.
- Region routing of separate experts appears in [FactorDiff](https://arxiv.org/abs/2607.11758) (one shared corpus, discrete grids), [TweedieMix](https://arxiv.org/abs/2410.05591) and [FreeFuse](https://arxiv.org/abs/2510.23515) (both need a joint prompt).

**Likely, verify before citing (absent from the field table):**

- [DiffCollage](https://arxiv.org/abs/2303.17076) (CVPR 2023) composes a score as factor scores minus overlap scores and says it works with pretrained models on different pieces. Its experiments mostly share one model. This was checked on a fetched HTML copy only.
- [MMDisCo](https://arxiv.org/abs/2405.17842) couples two independently pretrained audio and video models with a small discriminator as joint guidance. It needs real pairs, so it falls outside this setting. This was checked on the abstract only.

**Search limit:** four web searches on the setting and method found no paper composing disjoint-data models into two separate objects without joint data. Four searches cannot prove absence.

## Part 4: What is likely right

The problem is real and open in its hard form, where two disjoint-data models must place two objects without being told where. Keeping each model inside its own region is a sound way to avoid out-of-support score error.

## Part 5: What is likely wrong

**The fixed-region form reduces to pasting.** Part 1 shows this for $\lambda_t = 0$. The statement reports no paste baseline, so nothing shows the checker adds anything on fixed halves. The voided count of 1,200 of 1,280 would need this baseline even if the owner reinstates it.

**Where the method is more than a paste, it is failing.** The best shared-canvas run draws 121 of 320 against the statement's bar of 288, and the free-side run draws 4 of 320.

**The checker's target is circular.** If its positives are pasted composites, perfect guidance reproduces what a paste already gives for free.

**Interaction is unidentifiable from the setting.** Occlusion, shadow and relative scale appear in neither model's data. Projective Composition names shadows and occlusion as what breaks its factorization (page 7). Without a further assumption, independence is the only target this setting defines.

**Novelty rests on the combination.** Region routing and a checker trained on samples are both published. What remains is applying them to disjoint-data models, and that claim has content only when the sampler must choose the placement itself.

**None of the five benchmarks tests the setting.** Only CIFAR-10 has disjoint training data, only CLEVR has several separate objects as output, and no benchmark has both ([benchmark spec](../benchmark-spec.md#the-10-benchmarks-to-beat)).

| Benchmark (rival, number) | Disjoint models | Two separate objects | Why it misses |
|---|---|---|---|
| CLEVR, 5 positions (Domain Expansion, accuracy 44.5) | no | yes | Its base model trained on all 30,000 multi-object images, so this target breaks the setting's exclusions |
| Shapes3D AND (CoInD, CS 91.1%) | no | no | One model, attributes of one scene |
| CMNIST AND (Logical Guidance, 83.6%) | no | no | Digit and colour occupy the same pixels, so regions cannot separate them |
| CelebA AND (CoInD, CS 49.15%) | no | no | Smiling and male on one face, from one fine-tuned model |
| CIFAR-10 halves (SuperDiff, FID 4.11) | yes | no | An OR mixture with one class per image, so FID would count a two-object image as a defect |

## Part 6: Verdict

**Promising but needs rework:** the problem is real and no paper found solves it, but with the checker off the fixed-region method is exactly a paste of two independent samples, no baseline shows the checker improves on that paste, the free-placement version currently fails, and none of the five benchmarks measures the setting.

## Part 7: What to do next

1. **Measure the paste.** Report the $\lambda_t = 0$ fixed-halves run beside the guided run, on the same pairs, starting noises and judge. With a hard partition this run is the paste baseline and needs no new method code. If guidance does not beat it, fixed halves carry no claim about the checker.
2. **Move the contribution to placement.** Make $R_A$ and $R_B$ something the sampler decides, and state the target as the factorized distribution that Projective Composition defines. The check that this is on track is whether the free-side run draws more clean pairs with the checker on than the same sampler draws with it off, from the same starting noises.
3. **Build one benchmark that fits.** Split a multi-object dataset so the two models see disjoint object types and each image holds one object, then judge two-object images where shadows or overlap make a paste visibly wrong. Keep Projective Composition's hand count as the nearest reference and mark the five current rows as outside the setting until the owner decides them.
4. **Add Compositional Sculpting to the closest work** and state what the checker does that its classifier does not.
