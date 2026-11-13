# Talking points: supervisor meeting, Thursday 2026-10-29, 13:00

## 1. Arithmetic on SDXL's text embeddings does not supply what "and" adds

**The question.** Is there a text-space "and" vector, built like king − man + woman, that turns
PoE of "dog" and "cat" into a picture of both animals when added as a third expert?

**The answer: not by arithmetic.** Three versions were tried on 8 seeds, every column from the same
starting noise:

| third expert | what it is | seeds with one cat and one dog, of the 7 where the joint prompt succeeds and PoE fails |
|---|---|---|
| x_raw | E("dog and cat") − E("dog") − E("cat") | 0: a flat grey tile on all 8 |
| x_fix | x_raw + E("") | 1 (seed 3); wallpaper on 3 seeds |
| x_aff | x_raw + 2E(""), the version an affine UNet would make exact | 0 |

The sheet: [dog-and-cat-sheet.png](../../results/what-the-and-vector-is-made-of/dog-and-cat-sheet.png).
The pass mark, written before rendering, was 4 of 7.

**Why it fails, measured before any picture.** CLIP's text encoder is causal, so the start token's
state is the same in every prompt and carries a norm of 1190 against about 15 for a word (CLIP ViT-L
half). x_raw is mostly that start token negated. Adding E("") cancels it, but the 70 padding positions
depend on the prompt before them, so they keep 0.57 of a prompt's padding norm ("dog and cat"; 0.77
for "a cat and a dog"), and the content positions come out larger than a real prompt's. The vector
SDXL is handed looks like no prompt it was trained on, and it draws texture.
[The norm chart](../../results/what-the-and-vector-is-made-of/a-cat-and-a-dog-norms.png).

## 2. A third expert restores how many animals, never which

Seven phrases and two lower weights of x_fix were tried as the third expert, same seeds and noise
([the phrase sheet](../../results/what-the-and-vector-is-made-of/dog-and-cat-phrases-sheet.png)).
The two strongest, read tile by tile:

| third expert beside "dog" and "cat" | a clear cat and a clear dog, of 7 | two animals, each a cat-dog hybrid, of 7 |
|---|---|---|
| "a dog next to a cat" | 0 | 5 (seeds 1, 3, 4, 6, 7; also 8) |
| "dog and cat", the joint prompt itself | 0 | 4 (seeds 1, 2, 3, 6; also 8) |

The other five phrases and x_fix at lower weight do no better on a clear pair.

**The point.** The joint prompt "dog and cat" on its own draws a clear dog and a clear cat on 7 of 8
seeds. Put the same prompt inside the product, beside the "dog" and "cat" experts, and it draws two
hybrids instead. So the fusion is done by the product of the two single-animal experts; adding the
missing "and", even in its most complete form, restores the count and not the identities.

**Turning x down does not help.** At weights 0.1 to 0.5, PoE with x returns its own hybrid on every
seed; the two-animal picture on seed 3 appears only at 0.75 and 1, the same weights that turn other
seeds to wallpaper or printed text
([the weight sweep](../../results/what-the-and-vector-is-made-of/dog-and-cat-weight-sweep-sheet.png)).

**How strong this is.** Eight seeds, one pair, calls made by Claude from the sheets, not yet checked
by a second reader, except the two rows in the table, which are the researcher's own calls.

## 3. What this points to

The missing piece is not a fixed offset in text space, because the UNet's response to its condition
is far from affine on these four prompts. Two routes stay open: learn the third condition (textual
inversion against the cached joint scores, or a new word that names neither animal), or learn
weights (the existing LoRA, which composes on 7 of 8 seeds in the remedies table of
[the write-up](../../../plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#91-the-remedies-by-where-they-act)).

**Where the record is.** The walk's map, with every claim and run:
[the "and" vector idea map](../../ideas/the-and-vector-as-a-third-expert/IDEA_MAP.md).
