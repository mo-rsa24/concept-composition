# 🌍 Context: how each method the loops use works

Each folder here explains one method the loops use, from the whole system down to its parts: an overview page first, then one page per part, in the order of that method's zoom ladder under `2-classifier-guidance/artifacts/scenes/`. What the project's own words mean (product-of-experts composition, the interaction term, a compose rate) stays on the world pages listed in [the context index](../00-INDEX.md#the-themes). A method page links those for the meaning and adds what they do not carry: the ladder's pictures, a small worked example with real numbers, and the code.

## Table of contents

- [The methods, in reading order](#the-methods-in-reading-order)
- [How each folder reads](#how-each-folder-reads)

## The methods, in reading order

Navigation: 📋 [TOC](#table-of-contents) | [Next](#how-each-folder-reads) ➡️

**Start with the product of experts: folders 02 to 06 each add a push to the sampler it sets up.**

| Folder | What the method does | Which loop uses it |
|---|---|---|
| [01: the product of experts](01-the-product-of-experts/01-overview.md) | Adds the noise predictions of two models that each know one thing, takes the base's off once, and samples from the sum; the piece the sum leaves out is the interaction term | The sampler every push is added to: Loop 7's five-checker design and Loop 8's digit push (step V3) |
| 02: the readout checker, not written yet | A small trained head reads frozen SDXL's middle block and first two up blocks on a half-drawn picture, and the gradient of its score pushes the sampler | Loop 6, as Paper 2's kept checker |
| 03: the digit-map checkers, not written yet | Two small classifiers read maps of where each asked digit acts on a noisy canvas, one counting the digits and one naming them, and their gradient pushes the digit product | Loop 8 |
| 04: the sentence-score push, not written yet | A noise-aware CLIP scores a half-drawn latent against a right and a wrong sentence, and the gradient of the difference joins the noise prediction | Loop 9 |
| 05: Latent CLIP, not written yet | A CLIP model that reads SDXL latents directly, tried in place of the noise-aware CLIP for the same right-minus-wrong sentence score | Loop 9 (step W1b) |
| 06: the word-map checker, not written yet | A classifier on two words' cross-attention maps reads the spatial relation between two subjects, Learn-to-Steer's method applied to SDXL's product | Loop 7, which designed it as Paper 2's spatial-relation checker |
| 07: the score path, not written yet | The changes a frozen digit model's prediction makes when told each label, stacked over the noisy steps, read as a third space beside the caption and the canvas | Loop 10 |
| 08: the BLIP-VQA score, not written yet | Splits a prompt into one question per noun phrase, has BLIP-VQA answer each, and multiplies the yes-probabilities into one score | Loop 4, for the targets |

## How each folder reads

Navigation: ⬅️ [The methods, in reading order](#the-methods-in-reading-order) | 📋 [TOC](#table-of-contents)

**Every folder's pages run from the whole system to the code, and every page opens with the picture that opens its part.**

The numbered pages follow the method's zoom ladder: the overview (one step of the whole system), the network view, the miniature worked with real numbers, then the code. A ladder rung with nothing drawn is left out and the overview says so. Each name a page introduces is taken apart word by word on the picture, and each equation is shown by what moves when each symbol moves, per the teaching triad. Every claim carries a mark: ✅ verified this session, ✍️ stated by a named source and not rechecked, 🔍 inferred and confirmed by nobody. A number measured on a real run is kept apart from a number the toy computes, and each says which it is.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
