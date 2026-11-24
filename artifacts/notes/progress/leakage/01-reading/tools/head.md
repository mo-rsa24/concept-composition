# Mechanism table: where do papers say one subject's parts leak into the other?

This table holds one row for each of 13 papers read from the full PDF with appendices for step Z1 of Loop 12: the 5 named in [the loop's brief](../../prompts/session-12-where-co3-leaks.md) and 8 found by [the search for leakage papers](search-for-leakage-papers.md). Six more papers that Loops 4, 5 and 7 already read are cited from their notes. Each row links to its note, and every claim in a note carries its page.

<a href="../where-leakage-could-enter.png"><img src="../where-leakage-could-enter.png" width="620" alt="A prompt flows through the text encoder into SDXL's U-Net, where the layout, cross-attention and self-attention sit in one step, then into CO3's sum of predictions and out to a picture with two bodies and borrowed parts. Each of the five places carries the papers that blame it and the Z2 predictor that reads it."></a>

The picture places each paper by where it says leakage enters. Its source is [the diagram file](../where-leakage-could-enter.mmd).

## Table of contents

- [Words this uses](#words-this-uses)
- [What the reading changes for this loop](#what-the-reading-changes-for-this-loop)
- [The table](#the-table)
- [Papers read by other loops and cited here](#papers-read-by-other-loops-and-cited-here)
- [Closer papers these cite, not read](#closer-papers-these-cite-not-read)
- [Where sources disagree](#where-sources-disagree)
- [How the rows were checked](#how-the-rows-were-checked)

## Words this uses

Navigation: [Table of contents](#table-of-contents)

**Leakage, or borrowing.** One subject in the picture carries a part of the other: a face, a beak, fur, markings. In this project's hand labels (W2d), a picture with two separate subjects where one carries a clear part of the other is called `borrowed`. The papers use "semantic leakage", "subject mixing" and "attribute leakage" for overlapping things, and each note says which one its paper means.

**Cross-attention.** The U-Net layers where each position on the canvas looks at the prompt's words. Each word gets a map over the canvas.

**Self-attention.** The U-Net layers where each position on the canvas looks at every other position, so features can travel between the two subjects' regions.

**Sum of scores.** How noise predictions for different prompts are added or mixed. CO3 acts here.

## What the reading changes for this loop

Navigation: [Table of contents](#table-of-contents)

**One paper measures, across many pairs, what predicts a failed two-subject picture, and the answer is attention overlap.** [DAAM](daam.md) ran 1,000 prompts of the form "a(n) X and a(n) Y" on SD 2.0, half with both nouns from the same category (p. 7). Same-category pairs showed both objects in 52% of images against 61% for unrelated pairs, and their attention maps overlapped more, an IoU of 46.7 against 22.9 on a 0 to 100 scale. Within same-category pairs, images whose maps overlapped little were correct 71.7% of the time and images whose maps overlapped a lot 9.75% (Figure 6, read from the page image). So overlap sorted the pictures better than category did. Two limits: DAAM counts a missing object as the failure, so a borrowed part counts as correct, and its overlap is summed over all 30 steps of the same image it scores. No paper measures which pairs borrow parts and why. The search on 2026-11-24 found none, and none of the 13 notes reports one. The other measurements cover a handful of prompts or a different kind of leak:

- [Seeing Double](seeing-double.md) counts leaks on DALLE-2 for 10 prompt pairs. "An armadillo and an elephant" gave the elephant a shell in 12 of 12 images, while "a rhino and an elephant" leaked in 0 of 12 (p. 7). Across the pairs, 92.5% of test images leaked against 6.6% of controls (p. 4). The leak followed a distinctive part that could move to the other animal.
- [A Cat Is A Cat](a-cat-is-a-cat.md) swaps the order of the two animals in 400 prompts on SD1.4. The animal named first is drawn alone in 46.00% of images, and in 21.75% when it is named second (p. 4, Table 1). The paper traces this to CLIP's causal mask, which copies the first-named subject into the second subject's token (pp. 4 and 5).
- [Follow the Flow](follow-the-flow.md) patches the text encoder's output. One item's tokens drew a second item from the prompt in 1,699 of 15,950 item pairs (10.65%) on FLUX-schnell (p. 6). Its leaks are mostly a word read in the wrong sense, and it reports no such rate for SDXL's encoder.
- [The MMDiT similar-subject paper](mmdit-similar-subjects.md) uses an overlap ratio of the subjects' attention at step 5 to predict a failed picture on SD3, and restarts the sample when it fires (p. 13).

**Every place except the sum of scores has papers blaming it, and most of the evidence is examples.** Four papers blame the text encoder (Structured Diffusion Guidance blames cross-attention as well), four cross-attention, two cross-attention and self-attention together, and two the layout read with attention. Seeing Double leaves the place open. None blames the sum of scores, and CO3 is the only paper that acts there. Only five rows rest on a measurement over many prompts: DAAM's overlap on 1,000 prompts, A Cat Is A Cat's order swap, Follow the Flow's patching, [DeLeaker's](deleaker.md) ablation on its pair images, and the MMDiT paper's overlap ratio. The rest argue from attention maps of single prompts.

**This project's own data already disagrees with the commonest explanation.** [DeLeaker](deleaker.md) says leakage is "primarily driven by the visual similarity of the entities" (p. 14), and [A Cat Is A Cat](a-cat-is-a-cat.md) ties mixing to similar text embeddings. Neither tests it pair by pair. Under CO3, crow and raven borrow on 0 of 8 seeds while cheetah and cougar borrow on 7 of 8.

**For Z2, the predictor step (inferred):**

- Compute the text cosine twice: for the two names encoded alone, and for the two subject tokens inside the joint prompt, which is what A Cat Is A Cat's Table 5 measures (average 0.454).
- Define cross-attention overlap as DAAM does (each map upscaled, summed over heads and layers, thresholded at 0.4 of its maximum, then IoU), read at CO3's corrected steps instead of over all steps, so the number has a published counterpart and is read before the picture is decided. DAAM found the first 15 of 30 steps give 60.3 of its full 60.7 segmentation score (Figure 11, p. 11).
- Two optional columns the reading suggests: the overlap of the two subjects in the end-of-text and padding tokens' attention ([Magnet](magnet.md), [Bounded Attention](bounded-attention.md) p. 14), and a word-order swap that shows whether the borrowing follows the order of the names.
- Score borrowing by eye. DeLeaker found CLIP and BLIP crop scores uncorrelated with leakage labels (Spearman ρ between −0.027 and 0.027).

**For Z3, the one-fix step (inferred):** [Bounded Attention](bounded-attention.md) as published needs one box per subject from the user (p. 7). Its guidance runs from t = T to 0.7T with 5 forward and backward passes per step, about 75 extra passes per image, on SDXL layers 70 to 81 (pp. 14 and 15). [Be Decisive](be-decisive.md) supplies those boxes from the starting noise on SDXL at 50 steps, but trains a readout network and costs 77 seconds and 36 GB per image against 7 seconds and 8 GB (p. 12). Loop 6's M9 box reader failed 11 of 24 reads. Two fixes need no boxes and no extra U-Net passes: [A Cat Is A Cat's](a-cat-is-a-cat.md) embedding edit (9.14 against 8.21 seconds per image on SD1.4, p. 12) and [Follow the Flow's](follow-the-flow.md) token swap. CO3 already encodes each subject alone, so the swap's inputs exist.
