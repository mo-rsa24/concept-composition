# Classifier-guided product of experts: illustrated map
<!-- student-read: 2026-11-17 job 20261117-065202-d21aa2 -->

Pictures of the route's claim: what the product of experts keeps and drops, where a classifier's push
enters the prediction, and the four stages that test it with two kinds of classifier. The pictures
serve the route's readers first and the paper later.

4 prompts · 4 rendered · 0 waiting

## Table of contents

- [Abstraction chain](#abstraction-chain)
- [Art direction](#art-direction)
- [Reference images](#reference-images)
- [Meaning palette](#meaning-palette)
- [Glyph vocabulary](#glyph-vocabulary)
- [Reading axes](#reading-axes)
- [Devices in play](#devices-in-play)
- [Subject lane](#subject-lane)
  - [Prompt 1 (Subject): What the product keeps and drops](#prompt-1-subject-what-the-product-keeps-and-drops)
  - [Prompt 2 (Subject): Where the push enters the prediction](#prompt-2-subject-where-the-push-enters-the-prediction)
  - [Subject capstone: The dropped part and the push that could return it](#subject-capstone-the-dropped-part-and-the-push-that-could-return-it)
- [Process lane](#process-lane)
  - [Process capstone: Four stages, two classifiers at each](#process-capstone-four-stages-two-classifiers-at-each)

## Abstraction chain

Navigation: 📋 [TOC](#table-of-contents) | [Next](#art-direction) ➡️

1. [Classifier-guided product of experts](diagram-prompts.md): subject "The dropped part and the push
   that could return it", process "Four stages, two classifiers at each"

## Art direction

Navigation: ⬅️ [Previous](#abstraction-chain) | 📋 [TOC](#table-of-contents) | [Next](#reference-images) ➡️

**Empty, chosen by the user.** These pictures are headed for the paper, so they take the flat, matte
look of a machine-learning paper figure (the ImageReward paper's figures are the model the user
named) rather than the glossy Vivid circuit look of the root map in `plans/diagram-prompts.md`. This
folder is its own project, so its map pins its own direction. Every prompt carries one Finish line
inside its Scene that asks for that flat look and rules out gloss.

**Zoom depth: one level.** Level 0 is the subject capstone. Prompt 1 opens the product, prompt 2
opens one denoising step. Nothing opens further.

The direction's style paragraph is embedded verbatim at the head of every prompt below:

> Style: no named house style is imposed, but the result is a clean, welcoming UI-grade picture and
> never a wireframe. Every component gets an icon, an emoji or a small illustration of what it
> actually is; a real product carries its official logo only where the component is that product;
> a physical thing (a RAM stick, a building, a server rack, a phone) is drawn as the real object,
> not a labelled box. Cards and panels are rounded, with a thin coloured border, a soft drop shadow
> and a lightly tinted background that tells groups apart; grey is kept for secondary text,
> dividers and things that are off or inactive. Light background, generous spacing, a clear type
> hierarchy. Any colour the prompt names a meaning for keeps that meaning alone. Beyond that, let
> the composition and visual treatment that best communicate this subject emerge from the context
> below, rather than from an imposed convention.

The equation for the guided prediction is not drawn in any picture here. Mathematics is graphed by
script, per the style guide, so a term-by-term figure of it belongs to `~/.claude/figkit/termfig.py`.

## Reference images

Navigation: ⬅️ [Previous](#art-direction) | 📋 [TOC](#table-of-contents) | [Next](#meaning-palette) ➡️

**Stage 1's corgi sheet.**
`../artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-pembroke-welsh-corgi-scale-1-and-10.png`,
the authors' released 256 px unconditional model pushed toward "Pembroke Welsh corgi" at classifier
strength 1 and 10, seeds 0 to 15, rendered in step 4 of the route. Attach it to the process
capstone. Its right half (strength 10, 16 of 16 called corgi) is what stage 1's photo tile looks
like. Copy the look of one photo, never the sheet's layout, frames or text.

## Meaning palette

Navigation: ⬅️ [Previous](#reference-images) | 📋 [TOC](#table-of-contents) | [Next](#glyph-vocabulary) ➡️

| Colour | Means |
|---|---|
| Blue | the product of experts: the cat and dog predictions combined, and what it draws |
| Amber | the dropped part, and the pair-aware classifier whose push could supply it |
| Teal | a classifier that checks one label at a time (cat present, dog present), whose push only repeats the product |
| Green | a stage that ran and passed its check |

Dashed outlines mean not yet shown. Grey means off, empty or not run. The toy's red circles and
blue squares are the toy's own picture content, drawn inside photo tiles only, and carry no meaning
from this key.

## Glyph vocabulary

Navigation: ⬅️ [Previous](#meaning-palette) | 📋 [TOC](#table-of-contents) | [Next](#reading-axes) ➡️

| Glyph | Tier | Stands for |
|---|---|---|
| Expert card | primary | the frozen diffusion model run with one prompt ("a cat", "a dog", "empty prompt"): a rounded card with a small padlock badge, the prompt in quotes, and a small photo of what that prompt alone draws |
| Product panel | primary | the blue panel where the expert predictions are combined into the product's prediction |
| Fused animal tile | primary | a photo of what the product draws: one animal body with a golden retriever's body and a grey tabby cat's face, pointed ears and stripes |
| Two-animal tile | primary | a photo of what was asked for: a grey tabby cat sitting beside a golden retriever, floor visible between them |
| Check chip | artifact | a small blue chip on a tile: "cat? yes", "dog? yes" |
| Dropped-part bracket | secondary | an amber bracket spanning the two tiles, marking what tells them apart |
| Noisy picture tile | primary | a grainy, half-formed picture partway through denoising |
| Plus junction | secondary | the round "+" where a push joins the product's prediction |
| Strength dial | secondary | a small dial on the push arrow: how hard the push is applied |
| Separate-label classifier | primary | a teal card with two checkboxes, one per label |
| Pair-aware classifier | primary | an amber card holding a tiny two-animal photo against a tiny fused-animal photo |
| Stage panel | primary | one stage of the route's ladder, numbered, with its model, its tile and its two classifier slots |
| Status chip | artifact | green "done" or grey "not run yet" on a stage panel |

No product logos. Every component is research code or a model; SDXL and ImageNet are named in text
only.

## Reading axes

Navigation: ⬅️ [Previous](#glyph-vocabulary) | 📋 [TOC](#table-of-contents) | [Next](#devices-in-play) ➡️

Left to right is the dominant axis everywhere. In the subject lane it is the direction of
computation (prompts to prediction to picture). In the process lane it is the stage order. The
vertical axis separates what the model does (top) from what a classifier adds (bottom).

## Devices in play

Navigation: ⬅️ [Previous](#reading-axes) | 📋 [TOC](#table-of-contents) | [Next](#subject-lane) ➡️

**Subject lane:** truth anchors (photo tiles of real-looking animals), dashed outlines for the
claim not yet shown, an in-image legend on the capstone. No sequencing devices.

**Process lane:** the same cast, plus numbered stage panels and status chips.

## Subject lane

Navigation: ⬅️ [Previous](#devices-in-play) | 📋 [TOC](#table-of-contents) | [Next](#prompt-1-subject-what-the-product-keeps-and-drops) ➡️

What the product is made of and where a classifier attaches to it. No time in these pictures.

### Prompt 1 (Subject): What the product keeps and drops

Navigation: ⬅️ [Previous](#subject-lane) | 📋 [TOC](#table-of-contents) | [Next](#prompt-2-subject-where-the-push-enters-the-prediction) ➡️

[built] 🖼️ rendered 2026-11-17: `diagrams/classifier-guided-poe-01-what-the-product-drops.png`
Save as: `diagrams/classifier-guided-poe-01-what-the-product-drops.png`

Visual thesis: each expert checks its own animal alone, so a fused animal passes both checks and the
product draws it; the part that tells one fused animal from two separate animals is what the product
drops.

```text
Style: no named house style is imposed, but the result is a clean, welcoming UI-grade picture and never a wireframe. Every component gets an icon, an emoji or a small illustration of what it actually is; a real product carries its official logo only where the component is that product; a physical thing (a RAM stick, a building, a server rack, a phone) is drawn as the real object, not a labelled box. Cards and panels are rounded, with a thin coloured border, a soft drop shadow and a lightly tinted background that tells groups apart; grey is kept for secondary text, dividers and things that are off or inactive. Light background, generous spacing, a clear type hierarchy. Any colour the prompt names a meaning for keeps that meaning alone. Beyond that, let the composition and visual treatment that best communicate this subject emerge from the context below, rather than from an imposed convention.

Scene: a figure explaining why combining a "cat" model and a "dog" model draws one fused animal instead of a cat and a dog. Read left to right. On the left, two expert cards stacked vertically. In the middle, one blue panel that combines them. On the right, two large square photo tiles side by side with an amber bracket spanning the gap between them. Finish: a flat, matte figure for a machine-learning paper, in the manner of the ImageReward paper's figures: a white page, real-looking photo tiles with thin coloured frames, short sans-serif labels, pale tinted panels. Nothing glossy: no gradients, no glow, no bevels, no 3D, no reflections. Landscape 16:9, at the largest size available.

Cast:
- Top expert card: a rounded card with a small padlock badge (the model is frozen), the prompt "a cat" in quotes, and a small photo of a grey tabby cat.
- Bottom expert card: the same card with the prompt "a dog" in quotes and a small photo of a golden retriever.
- The product panel: a pale blue panel with a blue border and a large multiplication sign, labelled "the product".
- Left photo tile, blue frame: what the product draws. Exactly one animal with one body: a golden retriever's body and legs with a grey tabby cat's face, pointed cat ears and tabby stripes running over the golden fur. One head, one body, one tail.
- Right photo tile, plain grey frame: what was asked for. A grey tabby cat sitting on a wooden floor beside a golden retriever lying down, a clear strip of floor between them. Two heads, two bodies.
- Under each tile, the same two small blue check chips side by side: "cat? yes" and "dog? yes". The two tiles carry identical chips.
- Above both tiles, one short blue caption spanning them: "both pass both checks".
- An amber bracket spanning the gap between the two tiles, labelled "the dropped part", with a short amber arrow from the bracket pointing at the right tile only, and a small amber caption under the bracket: "tells these apart".

Flows: one blue arrow from each expert card into the product panel, the two arrows meeting cleanly at the panel's left edge. One blue arrow from the product panel to the left tile only. No arrow reaches the right tile from the product panel.

Text in the image, spelled exactly: "a cat", "a dog", "the product", "what the product draws", "what was asked for", "cat? yes", "dog? yes", "both pass both checks", "the dropped part", "tells these apart".

Exclusions: no equation or mathematical symbols other than the multiplication sign, no classifier, no third animal, no animal with two heads, no extra limbs, no score numbers, no percentages. No duplicated components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon without a label, no perspective distortion applied to arrows, no visual metaphor that contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare connector line standing in for a real-world physical object the prompt names.
```

Faithfulness note: the left tile shows exactly one animal body mixing cat and dog features and the
right tile exactly two separate animals; both tiles carry the same two blue check chips; the
product panel's arrow reaches the left tile only; the amber bracket points at the right tile only.
The claim drawn is the route's Quick context (the product treats the two captions as independent
given the picture, so it drops the term that depends on the pair). The fused draw is what plain
product sampling gives on SDXL cat × dog in this project, which is why the piece is `[built]`.

### Prompt 2 (Subject): Where the push enters the prediction

Navigation: ⬅️ [Previous](#prompt-1-subject-what-the-product-keeps-and-drops) | 📋 [TOC](#table-of-contents) | [Next](#subject-capstone-the-dropped-part-and-the-push-that-could-return-it) ➡️

[built] 🖼️ rendered 2026-11-17: `diagrams/classifier-guided-poe-02-where-the-push-enters.png`
Save as: `diagrams/classifier-guided-poe-02-where-the-push-enters.png`

Visual thesis: a classifier reads the current noisy picture and adds a scaled push after the product prediction. The separate-label push repeats existing information; whether a pair-aware push improves separation on cat × dog is not yet shown.

```text
Style: no named house style is imposed, but the result is a clean, welcoming UI-grade picture and never a wireframe. Every component gets an icon, an emoji or a small illustration of what it actually is; a real product carries its official logo only where the component is that product; a physical thing (a RAM stick, a building, a server rack, a phone) is drawn as the real object, not a labelled box. Cards and panels are rounded, with a thin coloured border, a soft drop shadow and a lightly tinted background that tells groups apart; grey is kept for secondary text, dividers and things that are off or inactive. Light background, generous spacing, a clear type hierarchy. Any colour the prompt names a meaning for keeps that meaning alone. Beyond that, let the composition and visual treatment that best communicate this subject emerge from the context below, rather than from an imposed convention.

Scene: one denoising step, read left to right. Make the plus junction and the push entering it the focal point. Across the upper row place the noisy picture, three compact expert cards, the product prediction, the plus junction, the guided prediction and the next picture. Below the junction place two large, equally sized alternative classifier cards in a tray. Allocate enough space to read their labels at 800 px overall width. Keep the expert cards and output tiles secondary. Finish: flat, matte machine-learning paper figure, white background, thin borders, pale solid fills and sans-serif text. No gradients, glow, bevels, reflections, dimensional stacks or metallic controls. Landscape 16:9, largest available size.

Cast:
- Input tile labelled "noisy picture": mostly grain with an unresolved animal silhouette.
- Three compact cards under "frozen model", each with a small padlock: "a cat" with a tabby thumbnail, "a dog" with a retriever thumbnail, and "empty prompt" with a blank grey thumbnail.
- Compact blue panel labelled "product's prediction", containing a flat blue arrow pointing diagonally upward.
- A prominent round plus junction labelled "add push here".
- Compact neutral panel labelled "guided prediction". Do not draw a resultant direction, since no classifier or strength has been selected.
- Output tile labelled "next picture", with the caption "one step later". Show a slightly less noisy but still unresolved animal silhouette. Neither tile establishes whether there is one animal or two. Do not show a successful separated pair.
- Lower tray labelled "choose one classifier".
- Solid teal card: two checkbox outlines labelled "cat?" and "dog?", title "separate-label classifier", and a teal arrow parallel to the blue prediction arrow. Caption: "repeats the label checks". Do not depict one label checked and the other unchecked.
- Dashed amber card: a thumbnail of two separate animals beside a thumbnail of one fused animal, title "pair-aware classifier", and an amber arrow visibly nonparallel to the blue arrow. Caption: "could add the missing distinction". Put "not yet shown" directly beneath that caption in readable type. Arrow angles illustrate the conceptual distinction; they are not measured vectors.
- A small flat outlined dial on the upward push connection, labelled "push strength". No metallic texture.

Flows: a grey branch carries the noisy picture to all three expert cards. A separate grey branch from the same noisy picture enters the classifier tray, labelled "reads this noisy picture". Within the tray, route that input visibly to both alternative cards. Blue arrows carry the expert predictions into the product panel and from there into the plus junction. One neutral output from the tray passes through the strength dial and enters the plus junction from below. This shared output means one selected classifier, not both summed together. The junction feeds the guided prediction, which feeds the next picture. No classifier arrow enters the frozen model.

Colour key, one compact line: "blue: product", "teal: separate-label push", "amber: proposed pair-aware push", "dashed: not yet shown". Grey connections carry inputs or the selected push without assigning a classifier choice.

Text: use only the quoted labels above, plus the plus sign. All essential labels must remain legible at 800 px width. Break long classifier titles across two lines rather than shrinking them.

Exclusions: no equations, Greek letters, gradient symbols, numerical scores, training loop or finished successful pair. Only the amber classifier card has a dashed outline. No arrows ending in whitespace, crossing arrows, decorative icons, extra legends or unsupported result claims.
```

Faithfulness note: the push joins at the plus junction after the product panel and before the
guided prediction, and nowhere else; the model's three cards carry padlocks and receive no arrow
from the tray; the classifiers read the noisy picture, never the finished one; the teal arrow icon is
parallel to the blue one and the amber one is not; the amber card alone is dashed and carries
"not yet shown". This follows the route's Quick context, where the push is added to the product's
prediction scaled by a strength and the separate-label push equals what the cat and dog experts
already contribute. The sampler slot is `[built]` (board row R0: the product sampler pushes, capped,
and strength 0 matches plain sampling by checksum on cat × dog seed 9, job 89341); whether the
pair-aware push helps on cat × dog is not yet shown (plan 21's first discriminator rendered a guided
result sheet that has not been read yet, and its v2 retrain failed the held-out check, commit
`a51d6200`), which is why its card is dashed.

### Subject capstone: The dropped part and the push that could return it

Navigation: ⬅️ [Previous](#prompt-2-subject-where-the-push-enters-the-prediction) | 📋 [TOC](#table-of-contents) | [Next](#process-lane) ➡️

[built] 🖼️ rendered 2026-11-17: `diagrams/classifier-guided-poe-03-the-claim-in-one-picture.png`
Save as: `diagrams/classifier-guided-poe-03-the-claim-in-one-picture.png`

Visual thesis: the distinction missing from the product could inform a pair-aware classifier's push. The connection is available; the proposed benefit on cat × dog is not yet shown.

```text
Style: no named house style is imposed, but the result is a clean, welcoming UI-grade picture and never a wireframe. Every component gets an icon, an emoji or a small illustration of what it actually is; a real product carries its official logo only where the component is that product; a physical thing (a RAM stick, a building, a server rack, a phone) is drawn as the real object, not a labelled box. Cards and panels are rounded, with a thin coloured border, a soft drop shadow and a lightly tinted background that tells groups apart; grey is kept for secondary text, dividers and things that are off or inactive. Light background, generous spacing, a clear type hierarchy. Any colour the prompt names a meaning for keeps that meaning alone. Beyond that, let the composition and visual treatment that best communicate this subject emerge from the context below, rather than from an imposed convention.

Scene: two regions on a white page. Left title: "what the product drops". Right title: "where a push enters". The left occupies about two fifths of the width and the right the remainder. Use small region titles and no heavy enclosing frames. The focal connection is a clearly labelled dashed amber arrow linking the left comparison to the right pair-aware classifier. Give it a dedicated clear corridor above the bottom legend, with the label beside the arrow rather than far from either endpoint. Finish: flat matte paper figure, thin borders, solid pale fills, sans-serif labels, no gradients, glow, bevels, 3D stacks or metallic dials. Landscape 16:9 at the largest size available. Essential labels must read at 800 px width.

Left region:
- Compact frozen expert cards "a cat" and "a dog", each with a padlock and the same animal thumbnail used in the earlier figures.
- Blue arrows into a compact blue panel "the product", using the same flat multiplication glyph as picture 1.
- Below, two equally sized photo tiles. The left tile, labelled "product draw", has a blue frame and exactly one fused animal: one retriever body with a tabby face, pointed ears and stripes. The right tile, labelled "requested pair", has a neutral frame and exactly two separate animals, a tabby beside a retriever with visible floor between them.
- The product's output arrow reaches the fused tile only.
- Beneath each tile, identical blue chips "cat? yes" and "dog? yes". Above the tiles put "both pass both checks".
- An amber bracket spans the comparison, labelled "the dropped part" with the explanation "one fused body or two separate animals". Its short pointer reaches the requested-pair tile only.

Right region:
- Upper flow: "noisy picture", three compact expert cards under "frozen model", "product's prediction", a round plus junction, "guided prediction", "next picture". The expert cards have padlocks, labels "a cat", "a dog", "empty prompt", and matching thumbnails.
- The product panel contains the same flat blue diagonal arrow as picture 2. The guided-prediction panel has no resultant direction icon because no classifier or strength is selected.
- Both picture tiles remain grainy and unresolved. The next picture is only slightly cleaner. Neither reveals a successful separated pair.
- Below the plus junction, a tray labelled "choose one classifier" holds two alternatives. The solid teal card is titled "separate-label classifier", with two checkbox outlines labelled "cat?" and "dog?", a teal arrow parallel to the blue arrow, and "repeats the label checks".
- The dashed amber card is titled "pair-aware classifier", with two small comparison photos, one separate pair and one fused animal. Include "two, or fused?", a nonparallel amber arrow, and the readable caption "not yet shown". The arrow is conceptual, not a measured vector.
- One small flat outlined dial labelled "push strength" sits on the tray's output connection to the plus junction.

Flows: in the right region, neutral arrows carry the noisy picture into the three expert cards. A separate visible branch from that same tile enters the tray and reaches both alternative classifier cards, labelled "reads the noisy picture". Blue arrows carry expert predictions to the product panel and then to the plus junction. One neutral connection carries the selected classifier's push through the dial into the junction. The junction feeds the guided prediction and then the next picture. No classifier arrow enters an expert card.

Cross-region connection: exactly one dashed amber arrow begins at the dropped-part bracket and terminates at the pair-aware card. Label it "could supply this distinction". Treat it as a hypothesis link, not a computational input. No cross-region connection reaches the teal card.

Legend along the bottom, exactly four entries: "blue: the product", "amber: the dropped part", "teal: separate-label push", "dashed: not yet shown". Neutral grey provides structural connections and secondary labels only.

Exclusions: no equations, numerical scores, decorative layers, extra objects, model-training arrows or confirmed-success output. Do not shrink labels to fit: shorten routing and reduce panel padding first. Retain every teaching component from pictures 1 and 2, including the noisy-picture input to the classifiers, the check chips, the strength dial and the uncertainty label.
```

Faithfulness note: the only link between the halves is the dashed amber arrow from the dropped-part
bracket to the pair-aware card, and it is dashed because the claim is not yet shown on cat × dog; the teal card has no
link to the bracket; every object here appears in prompt 1 or prompt 2 and none of theirs is
missing; the four legend entries are the only colour meanings used.

## Process lane

Navigation: ⬅️ [Previous](#subject-capstone-the-dropped-part-and-the-push-that-could-return-it) | 📋 [TOC](#table-of-contents) | [Next](#process-capstone-four-stages-two-classifiers-at-each) ➡️

The same cast laid over the route's four stages. The lane has one picture, so that picture is its
capstone. History: `diagrams/process-versions/`. This lane is regenerated, never patched.

### Process capstone: Four stages, two classifiers at each

Navigation: ⬅️ [Previous](#process-lane) | 📋 [TOC](#table-of-contents)

🖼️ rendered 2026-11-17: `diagrams/classifier-guided-poe-04-four-stages-two-classifiers.png`
Save as: `diagrams/classifier-guided-poe-04-four-stages-two-classifiers.png`

Visual thesis: first check that classifier guidance works; then compare separate-label and pair-aware classifiers across the remaining test beds. The final transition changes the model, size and evaluator together.

```text
Style: no named house style is imposed, but the result is a clean, welcoming UI-grade picture and never a wireframe. Every component gets an icon, an emoji or a small illustration of what it actually is; a real product carries its official logo only where the component is that product; a physical thing (a RAM stick, a building, a server rack, a phone) is drawn as the real object, not a labelled box. Cards and panels are rounded, with a thin coloured border, a soft drop shadow and a lightly tinted background that tells groups apart; grey is kept for secondary text, dividers and things that are off or inactive. Light background, generous spacing, a clear type hierarchy. Any colour the prompt names a meaning for keeps that meaning alone. Beyond that, let the composition and visual treatment that best communicate this subject emerge from the context below, rather than from an imposed convention.

Scene: four numbered stage panels in a straight left-to-right row. Align their model areas, example tiles, classifier slots and status chips horizontally so the student can compare like with like. Use a modest title "four stages of testing" and subtitle "a miss stops the climb". Give the classifier comparison more space than the example photographs. Keep all essential labels readable at 800 px overall width. Finish: flat matte machine-learning paper figure, white background, thin outlines, pale solid fills and sans-serif text. No gradients, bevels, glow, dimensional icons or glossy finishes. Landscape 16:9 at the largest available size.

Shared layout: each stage has a title, model information, a compact example tile with its own explicit caption, two classifier slots side by side, and a status chip. Teal slots are the separate-label alternatives; amber slots are the pair-aware alternatives. In panels 3 and 4, amber outlines are dashed. Panel 2's amber slot is solid, because the toy test ran. Pictures illustrate the test cases; they are not new measured outputs.

Panel 1:
- Title "1: released weights".
- Model card with a padlock, "256 px, ImageNet".
- A compact corgi photo matching one example from the right half of the attached reference sheet. Copy only the photo's appearance, not the sheet layout or text. Caption "guided sample: corgi". This is the classifier-guided example, not a plain-sampling baseline.
- Solid teal slot with "authors' classifier" and "corgi".
- Empty grey dashed slot with "no pair here". Do not draw an amber classifier in it.
- Green status chip: "works: done".

Panel 2:
- Title "2: exact toy".
- Model card "known true answer", with a small target icon.
- Compact tile showing an overlapping red circle and blue square, caption "product: overlay". Shape colours belong only to this example.
- Teal slot "circle? square?", with separate circle and square icons.
- Solid amber slot "pair, or overlay?", with a small separated-shapes example beside an overlapping-shapes example.
- Green status chip: "passed: done".

Panel 3:
- Title "3: our photos".
- Two distinct model cards: "released model, frozen" with a padlock, and "our 64 px model". Keep both visible; these are two arms within this stage.
- Caption "550 photos, 150 both".
- Compact fused-animal tile: one retriever body with a tabby face and stripes, caption "product: fused".
- Teal slot "cat? dog?", with small separate animal icons.
- Dashed amber slot "two, or fused?", with the same separate-pair and fused-animal thumbnails used elsewhere in the set.
- Grey status "not run yet".

Panel 4:
- Title "4: SDXL cat and dog".
- Model card with a padlock, "SDXL, frozen".
- The same fused-animal example, caption "product: fused".
- Teal slot "species-only push", with cat and dog icons and the explanatory line "cat? dog?".
- Dashed amber slot "presence head", with a separate-pair thumbnail and the explanatory line "pair-aware". These words identify its role without inventing its training target.
- Grey status "under way: no verdict".

Transitions: reserve a clear horizontal band for the three connecting arrows rather than squeezing words between photographs. The first arrow reads "to a known answer"; the second reads "to real photos". The third reads "model, size, judge change" and carries a clearly readable grey chip "mixed step". Place the third label at the same reading size as the other transition labels. No headline may claim that every transition changes only one thing. No other arrows.

Legend, one compact line: "teal: one label each", "amber: pair-aware", "green: passed", "dashed: not yet shown". The empty grey slot in panel 1 remains explicitly labelled "no pair here", so it cannot be mistaken for a planned classifier.

Exclusions: no result numbers, success marks in panels 3 and 4, fifth stage, equations or product logos. SDXL and ImageNet appear as text only. No large decorative animal photos, unnecessary outer frames, unlabelled icons, crossing arrows or tiny transition text. Do not describe all four example tiles as plain-sampler outputs.
```

Faithfulness note: exactly four panels in route order, and no headline claims every step changes one
thing. Panel 1 alone has one filled classifier slot and an empty pair slot, and its corgi tile is
captioned as a guided sample, because step 4 ran only the authors' ImageNet classifier (board row
G0: 16, 15 and 15 of 16 called corgi, tabby and golden retriever at strength 10) and the reference
sheet's right half is that guidance at strength 10. Panel 2 has both slots solid and a green
"passed: done" chip, because board row A1 passed its bar (at pair weight 0.01, seed 7, the
pair-aware push gave 172 of 300 crisp pairs at w = 1 against at most 10 for the separate-label
push). Panels 3 and 4 show both slots with dashed amber cards. Panel 3's chip is grey "not run yet"
(rows A2r and A2c are todo); panel 4's is grey "under way: no verdict" (row A3q is marked issue:
plan 21's v2 discriminator failed its held-out check, commit `a51d6200`, and its first
discriminator's guided result sheet has not been read; row A3 is open). Panel 3 shows two model
cards, the two arms of step 8. The arrow into panel 4 alone carries "mixed step", because the ladder
says stage 3 to 4 changes the model, the size and the evaluator at once. "550 photos, 150 both" is
copied from the route's ladder, and stage 4's two classifiers (plan 15's presence head against the
species-only pushes of plans 17 and 18) from the same section.
