# DALLE-2 is Seeing Double: Flaws in Word-to-Concept Mapping in Text2Image Models

arXiv 2210.10606 (v1, 19 Oct 2022), read from the full PDF (13 pages, appendix included). No venue is printed on the PDF.

## What it does

The paper builds small sets of hand-written prompts that make DALLE-2 use one word for two roles in the same picture, generates 12 images per prompt, and counts how often it happens (p. 3). Each leakage prompt is paired with a minimally changed control prompt that removes the cause, so the rate on the challenge prompt can be compared with the rate on the control (p. 3). It proposes no fix.

## Where it says leakage enters

Not located. The paper leaves open whether the behaviour comes from the text encoding, the generative model, or both (p. 2, p. 5). It cites the DALLE-2 authors' hypothesis that CLIP embeddings do not bind attributes to objects (p. 2), and from the "tall, long-legged, long-necked bird and a construction site" prompt (no word "crane", yet both kinds of crane appear) it suggests the model first maps entities into a concept space and only then renders (p. 4). Both are arguments. The measurement is behavioural only: counts of images showing the leaked property, with no access to the model's internals, since DALLE-2's code is not public (p. 5).

## What it changes there

Nothing. The paper is a diagnosis and calls for post-hoc mitigation methods as future work (p. 5).

## Does it need training, a layout or extra inputs

Nothing is trained and nothing is changed. The images come from the DALLE-2 product (p. 4, p. 5). The stimuli are written by hand (p. 5).

## Which model it runs on

DALLE-2 (p. 4). In preliminary runs DALLE-mini did not show the behaviours and Stable Diffusion (no version printed) showed them less often, which the authors attribute to those models following the prompt less reliably; no numbers are given (p. 4). Resolution and sampler are not printed. The text encoder is not named beyond the cited CLIP hypothesis (p. 2).

## What it measured on similar subjects

The entity-to-property set holds two-animal prompts, each with 12 images, counting images where one animal's property shows on the other (Table 3, p. 7; images on p. 11 and p. 12):

| Challenge prompt | Leaked images | Control prompt | Leaked images |
|---|---|---|---|
| an armadillo and an elephant | 12 of 12 | a rhino and an elephant | 0 of 12 |
| a fish and an elephant | 12 of 12 | a rhino and an elephant | 0 of 12 |
| A leopard and a piece of cloth | 11 of 12 | a leopard and a black towel | 5 of 12 |
| A seal is opening a letter | 10 of 12 | A large water mammal is opening a letter | 1 of 12 |
| A zebra and a street | 10 of 12 | A zebra and a gravel street | 0 of 12 |

In the example images the elephant carries the armadillo's banded shell (p. 11) and, with the fish, has a fish's body and tail fin (p. 12). In the modifier set, "A ladybug and a person" leaks in 11 of 12 against 6 of 12 for "A bug and a person" (for the second-order version of this prompt the paper explains the leak as the person drawn as a woman, from "lady", p. 13), and "An animal alongside a piggy bank" leaks in 12 of 12 against 0 of 12 beside a table (Table 2, p. 6).

Over the whole entity-to-property set (10 pairs, 120 images per side) the challenge prompts show the shared property in 92.5% of images and the controls in 6.6% (p. 4). Over the two-modifier set (6 pairs, 72 images per side) the rates are 97.2% and 15.2% (p. 4). The scale is the percentage of images showing the leaked property, higher meaning more leakage. Who judged the images and how is not printed.

## Which pairs leak and why

Argued from the stimuli, not measured. The paper says the entity that leaks onto is the underspecified one, and the leaked property has to be one that suits it: "fish is an underspecified noun and gold is an amicable property to fish" (p. 4). Zebra leaks onto street as crosswalk stripes, which the authors tie to the visual likeness of stripes and crosswalks (p. 4). Naming a specific kind removes it: "a fruit bat is flying over a stadium" shows only the animal (p. 2), and "a zebra and a gravel street" names a street type without crosswalks (p. 4).

The rhino rows bear on the brief. Rhino and elephant, two large grey animals of similar build, leak in 0 of 12 images, while armadillo and elephant and fish and elephant leak in 12 of 12 (p. 7). In these few prompts leakage rides on a distinctive transferable part (a shell, fins), and visual similarity of the pair does not produce it (inferred from three rows). Text-embedding similarity, attention overlap and token position are not measured.

## When in sampling leakage is set

Not measured.

## Numbers

| Metric (scale, direction) | Challenge prompts | Control prompts | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Images showing both senses of a homonym, % (higher is more leakage) | 80.3% | none | 17 homonym prompts, 12 images each | DALLE-2 | p. 3; Table 1, p. 6 |
| Images where one modifier lands on two entities, % | 97.2% | 15.2% | 6 pairs, 12 images each | DALLE-2 | p. 4; Table 2, p. 6 |
| Images where an entity's property lands on the other entity, % | 92.5% | 6.6% | 10 pairs, 12 images each | DALLE-2 | p. 4; Table 3, p. 7 |
| Same, armadillo and elephant, images out of 12 | 12 | 0 (rhino and elephant) | 1 pair | DALLE-2 | Table 3, p. 7 |
| Same, fish and elephant, images out of 12 | 12 | 0 (rhino and elephant) | 1 pair | DALLE-2 | Table 3, p. 7 |
| Second-order leakage (cause described, never named), images out of 12 | 12 for armadillo on a sea shore; 12 for the crane description | 0 and 0 | 4 pairs | DALLE-2 | Table 4, p. 7 |

## What it suggests for this loop

- For the predictor step (inferred): the paper offers a predictor this loop does not have yet: whether one subject has a distinctive transferable part (shell, fins, stripes, spots) that suits the other subject's body. The armadillo, fish and rhino rows suggest that a pair that looks alike overall (rhino and elephant, crow and raven) can leak less than a pair where one has a portable feature. The three planned predictors measure similarity, so a pair like crow and raven scoring high on all three while not borrowing would match this paper's rows.
- For the predictor step (inferred): the second-order rows (an armadillo on a sea shore puts shells on the beach) say leakage follows the concept the words evoke rather than the literal token, which favours measuring in $d_t$ map space or attention space over bare-name text cosine.
- For the one-fix step (inferred): naming a specific kind removed leakage in DALLE-2 (fruit bat, gravel street). A prompt-only control for CO3's leaking pairs is to add a distinguishing modifier to each subject and see whether borrowing drops, which separates a text-side cause from a layout or attention cause at no extra cost.
- For Paper 2's method (inferred): the challenge and control pairing (change one word, count images out of 12) is a clean evaluation design for leakage, and its two-animal prompts (armadillo and elephant, fish and elephant) can be rerun on SDXL with and without CO3 as an outside test set.

## Closer work it cites

- Hierarchical text-conditional image generation with CLIP latents (Ramesh et al. 2022), no arXiv id printed, p. 2: hypothesises that CLIP embeddings do not bind attributes to objects, which explains attribute mixing.
- A very preliminary analysis of DALL-E 2 (Marcus, Davis, Aaronson 2022), no arXiv id printed, p. 2: reports that a property given to one object often appears on the wrong object.
- Underspecification in scene description-to-depiction tasks (Hutchinson et al. 2022), arXiv 2210.05815, p. 2: underspecification, which this paper links to which entity receives leaked properties.
