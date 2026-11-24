# 🌍 Context: how each method the loops use works

Each folder here explains one method the loops use, from the whole system down to its parts: an overview page first, then one page per part, in the order of that method's zoom ladder under `2-classifier-guidance/artifacts/scenes/`. What the project's own words mean (product-of-experts composition, the interaction term, a [compose rate](../world/compose-rate.md)) stays on the world pages listed in [the context index](../00-INDEX.md#the-themes). A method page links those for the meaning and adds what they do not carry: the ladder's pictures, a small worked example with real numbers, and the code.

The two project words have their own pages: product-of-experts composition is on [PoE composition](../world/poe-composition.md#what-poe-composition-is), and the interaction term is on [the interaction term page](../world/interaction-term.md#what-the-interaction-term-is).

🔤 **The name:** loop (as in Loop 8)
- **loop**: one of the background sessions that run this project's work in parallel, each owning one question and one lane of steps on [the progress board](../../artifacts/notes/progress/map.png), and each repeating the same cycle (pick the next step, do one piece of work, check it, wake again later). The number is the session's place in [the table of who owns what](../../artifacts/notes/progress/prompts/shared-rules.md#who-owns-what).
- **Consequence:** "Loop 8 trains the digit-map checkers" means the work, its review files and its pictures sit in Loop 8's folder, `artifacts/notes/progress/abstract-checkers/`, and its steps are the ones whose names start with V on the board. ✅ the owner table in the shared rules.

🔤 **The name:** zoom ladder
- **zoom**: moving a camera in so that one part fills the frame. Here each page shows one part of the page before it at a larger size: one sampler step of the whole system, then one network, then a miniature small enough to work by hand, then the lines of code.
- **ladder**: rungs climbed one at a time, in order. Each rung is one zoom level, from the whole system down to the code, and the method's scene folder numbers the rungs in that order, as the rungs table in [the NoisyCLIP ladder's README](../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/README.md#the-rungs) lists them.
- **Consequence:** a rung with nothing drawn gets no page, so a folder's page count says which rungs exist. Folder 06 drew its illustrations rung and holds five pages; folder 01 drew no rung 1 and holds four. ✅

## Table of contents

- [The methods, in reading order](#the-methods-in-reading-order)
- [How each folder reads](#how-each-folder-reads)

## The methods, in reading order

Navigation: 📋 [TOC](#table-of-contents) | [Next](#how-each-folder-reads) ➡️

**Start with the product of experts: folders 02 to 06 each add a push to the sampler it sets up.**

| Folder | What the method does | Which loop uses it |
|---|---|---|
| [01: the product of experts](01-the-product-of-experts/01-overview.md) | Adds the noise predictions of two models that each know one thing, takes the base's off once, and samples from the sum; the piece the sum leaves out is the interaction term | The sampler every push is added to: Loop 7's five-checker design and Loop 8's digit push (step V3) |
| [02: the readout checker](02-the-readout-checker/01-overview.md) | A small trained head reads frozen SDXL's middle block and first two up blocks on a half-drawn picture, and the gradient of its score pushes the sampler | Loop 6, as Paper 2's kept checker |
| [03: the digit-map checkers](03-the-digit-map-checkers/01-overview.md) | Two small classifiers read maps of where each asked digit acts on a noisy canvas, one counting the digits and one naming them, and their gradient pushes the digit product | Loop 8 |
| [04: the sentence-score push](04-the-sentence-score-push/01-overview.md) | A noise-aware CLIP scores a half-drawn latent against a right and a wrong sentence, and the gradient of the difference joins the noise prediction; and a small noise-aware encoder of our own, trained on digit canvases against CLIP's frozen text encoder, read on digits it never saw | Loop 9 (W2 scored NoisyCLIP, W4 trained our own encoder; the push, W3, never ran) |
| [05: Latent CLIP](05-latent-clip/01-overview.md) | A CLIP model that reads SDXL latents directly, tried in place of the noise-aware CLIP for the same right-minus-wrong sentence score; and a linear probe on its, NoisyCLIP's and frozen CLIP's picture vectors, asking whether they carry counting and prompt match at all | Loop 9 (step W1b read it, W2 scored it, W2b probed it; its push, W3, never ran) |
| [06: the word-map checker](06-the-word-map-checker/01-overview.md) | A classifier on two words' cross-attention maps reads the spatial relation between two subjects, Learn-to-Steer's method applied to SDXL's product; proposed only, no head trained or run on SDXL | Loop 7, which designed it as Paper 2's spatial-relation checker |
| [07: the score path](07-the-score-path/01-overview.md) | The changes a frozen digit model's prediction makes when told each label, stacked over the noisy steps, read as a third space beside the caption and the canvas | Loop 10 (step X5 read all ten labels' changes and missed its bar; the encoder and the shared space, X6 to X8, never ran) |
| [08: the BLIP-VQA score](08-the-blip-vqa-score/01-overview.md) | Splits a prompt into one question per noun phrase, has BLIP-VQA answer each, and multiplies the yes-probabilities into one score; a category's number is the mean over its pictures | Loop 4, for the Color, Shape and Texture targets, and Loop 5, which scored its rival reruns with the same code |

## How each folder reads

Navigation: ⬅️ [The methods, in reading order](#the-methods-in-reading-order) | 📋 [TOC](#table-of-contents)

**Every folder's pages run from the whole system to the code, and every page opens with the picture that opens its part.**

The numbered pages follow the method's zoom ladder: the overview (one step of the whole system), the network view, the miniature worked with real numbers, then the code. A ladder rung with nothing drawn is left out and the overview says so. Each name a page introduces is taken apart word by word on the picture, and each equation is shown by what moves when each symbol moves, per the teaching triad. Every claim carries a mark: ✅ verified this session, ✍️ stated by a named source and not rechecked, 🔍 inferred and confirmed by nobody. A number measured on a real run is kept apart from a number the toy computes, and each says which it is.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
