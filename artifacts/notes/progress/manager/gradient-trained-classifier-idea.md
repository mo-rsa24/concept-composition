# A classifier trained so its gradient points the right way

The owner chose to test this as a fifth arm of the coloured-digit experiment (board step Z9, Loop 12),
after the reviewer reads it. It is an idea, not a result.

## Why

A classifier's push is its input gradient, and ordinary cross-entropy training leaves that gradient
unconstrained (Kawar, Ganz and Elad, arXiv 2208.08664, p. 5). On SDXL the kept checker's push pointed
against the term a product of experts drops, with a median cosine of -0.105 at step 0 on CO3
(`method-search/09-gradient-direction-check/review.md`). Ganz, Kawar and Elad (arXiv 2207.11378, pp. 4
and 5) train a classifier with cross-entropy plus a cosine penalty that pulls its input gradient
toward the diffusion model's class score minus its unconditional score. Loop 15 read both
(`directions/01-directions-that-generalise.md`, section 1).

## The arm on coloured digits

The same checker as arm 2 is used, trained with an added penalty: one minus the cosine between its input
gradient and a target direction at each noisy training picture. The target on the 40 seen pairs not
held back is the term the product drops: the prediction of the expert that saw the pair, asked with both
labels, minus the plain product's prediction. This is the same target as arm 4 (`leakage/09-dcfm-colored-mnist/design.md`).
The arm is compared with the plain checker (arm 2) and with the size-matched digit-and-colour push
(control a), on the same 5,000 pictures, gated first on the 10 held-back seen pairs.

## The owner's extension

If the penalty works, a classifier can carry a known direction for an abstract concept while staying a
plug-in push: its gradient can be scaled, which is a dial, and summed with others, and it adds to any
sampler, CO3 included. The open question is where each concept's target direction comes from:

| Concept | A candidate target direction | Its weakness |
|---|---|---|
| co-occurrence | the pooled adapter's correction, which drew two animals in 58 of 64 pictures of unseen pairs | the classifier then distils the adapter; it is only as general as the adapter |
| count | "two dogs" minus "a dog" from the same model | the joint prompt itself miscounts |
| each animal keeps its own parts, on CO3 | each animal's own prediction inside its box minus CO3's prediction (14 against CO3's 8 of 48, a near miss) | its boxes came from the finished picture |
| binding an attribute | "a red book" minus "a book" | no agreed measure of binding yet (`directions/03-dials.md`) |

The joint prompt's own direction may be the wrong target where the joint fails; the test of that is
running now (`method-search/10-is-the-joint-the-wrong-target/`).
