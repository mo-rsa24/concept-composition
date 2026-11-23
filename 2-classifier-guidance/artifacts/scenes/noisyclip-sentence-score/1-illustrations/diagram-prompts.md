# NoisyCLIP sentence score: illustrated map

Six pictures that tell the idea, drawn on the miniature's story and numbers. Every tile is an image model's teaching drawing, and none is an SDXL sample or a result. `6 prompts · 6 rendered · 0 waiting`

## Table of contents

- [Art direction](#art-direction)
- [Reference images](#reference-images)
- [Meaning palette](#meaning-palette)
- [Glyph vocabulary](#glyph-vocabulary)
- [Reading axes](#reading-axes)
- [Devices in play](#devices-in-play)
- [Subject lane](#subject-lane)
  - [Prompt 1: A cheap colour view of the latent](#prompt-1-subject-a-cheap-colour-view-of-the-latent)
  - [Prompt 2: One picture against two sentences](#prompt-2-subject-one-picture-against-two-sentences)
  - [Prompt 3: Training on half-drawn pictures](#prompt-3-subject-training-on-half-drawn-pictures)
  - [Prompt 4: The push bends the noise prediction](#prompt-4-subject-the-push-bends-the-noise-prediction)
  - [Prompt 5: Two places the signal is cut](#prompt-5-subject-two-places-the-signal-is-cut)
  - [Prompt 6: Trained later than we push](#prompt-6-subject-trained-later-than-we-push)
- [Process lane](#process-lane)

## Art direction

The approved clip-contrastive style, written out in the [style reference](../style-reference.md). The style paragraph opens every prompt.

## Reference images

Each render attaches the [approved style sample](../style-sample-clip-contrastive.png) and the three third-party layout references in `../reference/`, each downscaled to 1280 pixels on its longest side. They set layout devices only, and no picture, wording or logo from them is reproduced.

## Meaning palette

| Colour | Meaning |
|---|---|
| lilac `#b9a7e8` | text and sentences |
| green `#a8dcb5` | image encoder and image embeddings |
| pale blue `#bfe3f5` | a matching pair |
| orange `#c86a0a` | the push, and nothing else |
| grey `#6b7a88` | frozen or fixed |

## Glyph vocabulary

**Encoder.** A solid trapezoid or rounded block with its name inside.

**Embedding strip.** A thin row of cells shading light to dark.

**Noisy tile.** A square of grain, which never shows a finished composition.

## Reading axes

Left to right is the forward path. An orange lane is the backward signal.

## Devices in play

Fade for frozen parts, one return lane, a ⊗ where two embeddings meet, and a small grey tag saying whose numbers a picture shows.

## Subject lane

### Prompt 1 (Subject): A cheap colour view of the latent

[planned] 🖼️ rendered

Save as: `ni1-a-cheap-colour-view-of-the-latent.png`

**Visual thesis:** A half-drawn latent is not a picture, so a fixed sum of its four channels makes a rough colour view with no decoder.

```text
Style: an original, clean machine-learning explainer figure. White page, thin dark arrows, sans-serif labels, generous white space, no drop shadows. Colours carry meaning and nothing else: soft lilac #b9a7e8 for text and sentences, soft green #a8dcb5 for the image encoder and image embeddings, pale blue #bfe3f5 for a matching pair, orange #c86a0a for the push and nothing else, grey #6b7a88 for what is frozen or fixed. Encoders are solid trapezoids that narrow toward their output, or rounded pastel blocks, with the name inside. Embeddings are thin strips of small cells shading from light to dark. A frozen part keeps its shape, fades, and carries a small tag reading "frozen". The first attached image is this project's own style sample. The other attached images are third-party figures used only for their layout devices: copy no picture, wording, logo or exact arrangement from them. It is a finished, welcoming figure and never a bare wireframe.

Scene: Read left to right, in one row.

Cast:
- Far left: four small square grey grainy tiles stacked with a slight offset, labelled "4 latent channels".
- Next: a small flat grey rounded box labelled "fixed linear map" with a smaller line under it, "no VAE decode".
- Next: one larger square tile of fine multicoloured grain with no recognisable object in it, labelled "rough colour view".
- Right: one solid soft-green trapezoid narrowing to the right, labelled "Image Encoder", with a thin strip of six green cells shading light to dark at its output.
- Above the grey box: one faded, pale grey dashed outline of a large decoder block labelled "VAE decoder" with a small grey tag "skipped". No arrow touches it.

Flows: thin dark arrows from the channel tiles to the grey box, from the grey box to the colour tile, from the colour tile to the trapezoid, and from the trapezoid to the strip.

Text in the image, spelled exactly and no other words: "4 latent channels", "fixed linear map", "no VAE decode", "rough colour view", "Image Encoder", "VAE decoder", "skipped".

Exclusions: no title banner, no legend box, no paragraphs, no invented words, no logos, no watermark, no decorative circuitry, no duplicated components, no arrow ending in empty space, no arrow crossing a label or a picture, no orange on anything except the push.

Aspect ratio: landscape 3:2.
```

Faithfulness note: Four channel tiles; one grey map box; the colour tile shows grain and no object; the VAE block is faded with no arrow touching it.

### Prompt 2 (Subject): One picture against two sentences

[planned] 🖼️ rendered

Save as: `ni2-one-picture-against-two-sentences.png`

**Visual thesis:** The checker asks the same noisy picture about two sentences that use the same words in a different order, and keeps the difference.

```text
Style: an original, clean machine-learning explainer figure. White page, thin dark arrows, sans-serif labels, generous white space, no drop shadows. Colours carry meaning and nothing else: soft lilac #b9a7e8 for text and sentences, soft green #a8dcb5 for the image encoder and image embeddings, pale blue #bfe3f5 for a matching pair, orange #c86a0a for the push and nothing else, grey #6b7a88 for what is frozen or fixed. Encoders are solid trapezoids that narrow toward their output, or rounded pastel blocks, with the name inside. Embeddings are thin strips of small cells shading from light to dark. A frozen part keeps its shape, fades, and carries a small tag reading "frozen". The first attached image is this project's own style sample. The other attached images are third-party figures used only for their layout devices: copy no picture, wording, logo or exact arrangement from them. It is a finished, welcoming figure and never a bare wireframe.

Scene: A symmetric layout read upward. One wide flat grey bar across the top.

Cast:
- Bottom centre: one square tile of fine grain with no recognisable object, labelled "noisy picture", feeding a soft-green rounded block labelled "Image Encoder", feeding a thin strip of six green cells.
- Bottom left: a white card printing "a cat to the left of a dog", feeding a faded lilac rounded block labelled "Text Encoder" with a small tag "frozen", feeding a thin strip of six lilac cells.
- Bottom right: a white card printing "a dog to the left of a cat", feeding a second faded lilac rounded block labelled "Text Encoder" with a small tag "frozen", feeding a thin strip of six lilac cells.
- Between the left lilac strip and the green strip, a small white circle with a multiplication cross. Between the green strip and the right lilac strip, a second one.
- Above the left circle, a pale blue rounded chip printing "S right 0.9687". Above the right circle, a very pale grey rounded chip printing "S wrong 0.9995".
- The top bar prints "S right minus S wrong = −0.0308".
- A small grey tag at the bottom right: "toy numbers".

Flows: thin dark arrows upward through each column; each strip points sideways into its neighbouring circle; each circle points up to its chip; each chip points up to the bar.

Text in the image, spelled exactly and no other words: "noisy picture", "Image Encoder", "a cat to the left of a dog", "a dog to the left of a cat", "Text Encoder", "frozen", "S right 0.9687", "S wrong 0.9995", "S right minus S wrong = −0.0308", "toy numbers".

Exclusions: no title banner, no legend box, no paragraphs, no invented words, no logos, no watermark, no decorative circuitry, no duplicated components, no arrow ending in empty space, no arrow crossing a label or a picture, no orange on anything except the push.

Aspect ratio: landscape 3:2.
```

Faithfulness note: Two sentences spelled exactly; two crossed circles; chips print 0.9687 and 0.9995; the bar prints −0.0308; the noisy tile shows no object; the toy tag is present.

### Prompt 3 (Subject): Training on half-drawn pictures

[planned] 🖼️ rendered

Save as: `ni3-training-on-half-drawn-pictures.png`

**Visual thesis:** Only the image side learns: each half-drawn picture is pulled toward its own caption and away from the others.

```text
Style: an original, clean machine-learning explainer figure. White page, thin dark arrows, sans-serif labels, generous white space, no drop shadows. Colours carry meaning and nothing else: soft lilac #b9a7e8 for text and sentences, soft green #a8dcb5 for the image encoder and image embeddings, pale blue #bfe3f5 for a matching pair, orange #c86a0a for the push and nothing else, grey #6b7a88 for what is frozen or fixed. Encoders are solid trapezoids that narrow toward their output, or rounded pastel blocks, with the name inside. Embeddings are thin strips of small cells shading from light to dark. A frozen part keeps its shape, fades, and carries a small tag reading "frozen". The first attached image is this project's own style sample. The other attached images are third-party figures used only for their layout devices: copy no picture, wording, logo or exact arrangement from them. It is a finished, welcoming figure and never a bare wireframe.

Scene: Two lanes running left to right into a square matrix, text on top and pictures below.

Cast:
- Top lane: a small stack of three white caption cards, the front one printing "a red umbrella", feeding a faded lilac trapezoid labelled "Text Encoder" with a small tag "frozen".
- Lower lane: three small square tiles stacked vertically, each a grainy half-drawn picture: a red umbrella, a yellow armchair, a blue backpack, all still heavily covered in grain, feeding a saturated soft-green trapezoid labelled "Image Encoder" with a small tag "tuned".
- Centre right: a 3 by 3 similarity matrix of pale cells. The three diagonal cells are pale blue. Across its top sit three thin lilac strips labelled "T1", "T2", "T3". Down its left sit three thin green strips labelled "I1", "I2", "I3". Each cell prints its product, such as "I1·T2".
- Far right, in a rounded pale grey panel titled "attract and repel": one green dot "I1", a lilac dot "T1" joined to it by a short double-headed blue arrow labelled "attract", and two pale lilac dots "T2" and "T3" lower down. Two straight thin grey arrows both start at the edge of "I1" and point to "T2" and "T3", each stopping just before its dot. The word "repel" sits centred between "T2" and "T3" on their level, clear of both arrows.

Flows: thin elbow arrows from the text trapezoid drop to each T strip. Thin elbow arrows from the image trapezoid reach each I strip.

Text in the image, spelled exactly and no other words: "a red umbrella", "Text Encoder", "frozen", "Image Encoder", "tuned", "T1", "T2", "T3", "I1", "I2", "I3", "I1·T1", "I1·T2", "I1·T3", "I2·T1", "I2·T2", "I2·T3", "I3·T1", "I3·T2", "I3·T3", "attract and repel", "attract", "repel".

Exclusions: no title banner, no legend box, no paragraphs, no invented words, no logos, no watermark, no decorative circuitry, no duplicated components, no arrow ending in empty space, no arrow crossing a label or a picture, no orange on anything except the push.

Aspect ratio: landscape 3:2.
```

Faithfulness note: Nine cells with products; exactly the three diagonal cells pale blue; text trapezoid faded and tagged frozen; image trapezoid saturated and tagged tuned.

### Prompt 4 (Subject): The push bends the noise prediction

[planned] 🖼️ rendered

Save as: `ni4-the-push-bends-the-noise-prediction.png`

**Visual thesis:** The gradient is taken at the latent, but what changes is the noise prediction the sampler is about to use.

```text
Style: an original, clean machine-learning explainer figure. White page, thin dark arrows, sans-serif labels, generous white space, no drop shadows. Colours carry meaning and nothing else: soft lilac #b9a7e8 for text and sentences, soft green #a8dcb5 for the image encoder and image embeddings, pale blue #bfe3f5 for a matching pair, orange #c86a0a for the push and nothing else, grey #6b7a88 for what is frozen or fixed. Encoders are solid trapezoids that narrow toward their output, or rounded pastel blocks, with the name inside. Embeddings are thin strips of small cells shading from light to dark. A frozen part keeps its shape, fades, and carries a small tag reading "frozen". The first attached image is this project's own style sample. The other attached images are third-party figures used only for their layout devices: copy no picture, wording, logo or exact arrangement from them. It is a finished, welcoming figure and never a bare wireframe.

Scene: A simple vector picture on a pale grid, read left to right.

Cast:
- Left of centre: one small square grainy tile labelled "x_t", with a small orange dot on its edge and a short orange label "gradient taken here".
- From a black point to the right of the tile, a long grey arrow labelled "ε base".
- From the tip of the grey arrow, a shorter thick orange arrow, turning up and to the right, labelled "− λ σ g".
- From the black point to the tip of the orange arrow, a dark arrow labelled "ε̂".
- Right: a flat grey rounded box labelled "next sampler step".

Flows: one thin dark arrow from the tip region of the dark arrow to the grey box. A thin orange dashed curve runs from the orange dot on the tile to the start of the orange arrow. Nothing is drawn being added onto the tile itself.

Text in the image, spelled exactly and no other words: "x_t", "gradient taken here", "ε base", "− λ σ g", "ε̂", "next sampler step".

Exclusions: no title banner, no legend box, no paragraphs, no invented words, no logos, no watermark, no decorative circuitry, no duplicated components, no arrow ending in empty space, no arrow crossing a label or a picture, no orange on anything except the push.

Aspect ratio: landscape 3:2.
```

Faithfulness note: Tip-to-tail vector sum: grey arrow, orange arrow from its tip, dark resultant; the orange curve leaves the tile and no arrow points into the tile.

### Prompt 5 (Subject): Two places the signal is cut

[planned] 🖼️ rendered

Save as: `ni5-two-places-the-signal-is-cut.png`

**Visual thesis:** The released map blocks the gradient twice, at a clamped colour and at the conversion to bytes.

```text
Style: an original, clean machine-learning explainer figure. White page, thin dark arrows, sans-serif labels, generous white space, no drop shadows. Colours carry meaning and nothing else: soft lilac #b9a7e8 for text and sentences, soft green #a8dcb5 for the image encoder and image embeddings, pale blue #bfe3f5 for a matching pair, orange #c86a0a for the push and nothing else, grey #6b7a88 for what is frozen or fixed. Encoders are solid trapezoids that narrow toward their output, or rounded pastel blocks, with the name inside. Embeddings are thin strips of small cells shading from light to dark. A frozen part keeps its shape, fades, and carries a small tag reading "frozen". The first attached image is this project's own style sample. The other attached images are third-party figures used only for their layout devices: copy no picture, wording, logo or exact arrangement from them. It is a finished, welcoming figure and never a bare wireframe.

Scene: One horizontal pipeline of five small rounded boxes, with an orange return lane below it running right to left.

Cast:
- Boxes from left to right: "latent", "linear map", "clamp", ".byte()", "Image Encoder". The first four are flat grey, and "Image Encoder" is soft green.
- Below, a thick orange arrow enters from the right under "Image Encoder" and runs left under ".byte()", where it stops at a clear break with a small grey scissors icon and the label "no gradient".
- Further left, under "clamp", a separate short section of the orange lane fades to grey at a small grey square pixel printing "0", labelled "clamped colour".
- At the far left under "latent", a small faint orange dot with a question mark.
- A small grey tag at the bottom right: "the push uses a copy without .byte()".

Flows: thin dark arrows between the five boxes.

Text in the image, spelled exactly and no other words: "latent", "linear map", "clamp", ".byte()", "Image Encoder", "no gradient", "0", "clamped colour", "?", "the push uses a copy without .byte()".

Exclusions: no title banner, no legend box, no paragraphs, no invented words, no logos, no watermark, no decorative circuitry, no duplicated components, no arrow ending in empty space, no arrow crossing a label or a picture, no orange on anything except the push.

Aspect ratio: landscape 3:2.
```

Faithfulness note: Five boxes in order; the orange return lane visibly breaks at .byte() and fades at clamp.

### Prompt 6 (Subject): Trained later than we push

[planned] 🖼️ rendered

Save as: `ni6-trained-later-than-we-push.png`

**Visual thesis:** NoisyCLIP practised on the middle of the sampling run, and this loop would use it earlier, on noisier latents.

```text
Style: an original, clean machine-learning explainer figure. White page, thin dark arrows, sans-serif labels, generous white space, no drop shadows. Colours carry meaning and nothing else: soft lilac #b9a7e8 for text and sentences, soft green #a8dcb5 for the image encoder and image embeddings, pale blue #bfe3f5 for a matching pair, orange #c86a0a for the push and nothing else, grey #6b7a88 for what is frozen or fixed. Encoders are solid trapezoids that narrow toward their output, or rounded pastel blocks, with the name inside. Embeddings are thin strips of small cells shading from light to dark. A frozen part keeps its shape, fades, and carries a small tag reading "frozen". The first attached image is this project's own style sample. The other attached images are third-party figures used only for their layout devices: copy no picture, wording, logo or exact arrangement from them. It is a finished, welcoming figure and never a bare wireframe.

Scene: One long horizontal row of seven equal square tiles, evenly spaced, showing the same yellow armchair in a plain room emerging from noise. Tile 1 is pure grain with no shape. Tile 2 is grain with no shape. Tile 3 has only a faint blur where the chair will be. Tiles 4 and 5 show a soft, grainy chair. Tile 6 is nearly clear and tile 7 is clear.

Cast:
- Directly under each tile, centred on it, one label. In order: "step 0", "step 8", "step 15", "step 20", "step 29", "step 40", "step 49".
- Above tiles 1, 2 and 3, one orange bracket that spans exactly from the left edge of tile 1 to the right edge of tile 3, with a pale orange label centred over it: "this loop would push here".
- Above tiles 4 and 5, one dark grey bracket that spans exactly from the left edge of tile 4 to the right edge of tile 5, with a pale green label centred over it: "NoisyCLIP trained here".
- Nothing above tiles 6 and 7.

Flows: none.

Text in the image, spelled exactly and no other words: "step 0", "step 8", "step 15", "step 20", "step 29", "step 40", "step 49", "this loop would push here", "NoisyCLIP trained here".

Exclusions: no title banner, no legend box, no paragraphs, no invented words, no logos, no watermark, no decorative circuitry, no duplicated components, no arrow ending in empty space, no arrow crossing a label or a picture, no orange on anything except the push.

Aspect ratio: landscape 3:2.
```

Faithfulness note: Seven evenly spaced tiles labelled by step; the orange bracket covers tiles 1 to 3 (steps 0 to 15) and the green-labelled bracket covers tiles 4 and 5 (steps 20 and 29), with nothing over tiles 6 and 7. Tile spacing is even, so it is not a time axis.

## Process lane

This ladder has no process lane. Its build steps are in [the route](../route-zoom-ladder-noisyclip.md).
