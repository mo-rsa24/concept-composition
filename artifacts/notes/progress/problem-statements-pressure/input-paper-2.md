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

The five Paper 2 rows in the [benchmark spec](benchmark-spec.md#the-10-benchmarks-to-beat): T2I-CompBench Color, Shape, Texture
and Complex, and RareBench Concat. T2I-CompBench needs plain SDXL, ToMe and CoMat rerun on one setup before any win is claimed.
