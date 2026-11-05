# The caption model: illustrated map

How model 3 is trained, how it tells its captions apart, how PoE samples from it, and what
relabelling does, with one worked example per picture. Facts are from
`artifacts/notes/what-poe-loses-when-it-composes/toy_caption_dial_ldm.py`.

8 prompts · 5 rendered · 3 waiting

## Table of contents

- [Abstraction chain](#abstraction-chain)
- [Art direction](#art-direction)
- [Reference images](#reference-images)
- [Meaning palette](#meaning-palette)
- [Glyph vocabulary](#glyph-vocabulary)
- [Reading axes](#reading-axes)
- [Devices in play](#devices-in-play)
- [Subject lane](#subject-lane)
  - [Prompt 1 (Subject): One training step](#prompt-1-subject-one-training-step)
  - [Prompt 2 (Subject): The label picks a row](#prompt-2-subject-the-label-picks-a-row)
  - [Prompt 3 (Subject): Sampling with PoE](#prompt-3-subject-sampling-with-poe)
  - [Prompt 4 (Subject): Relabelling, toy and SDXL](#prompt-4-subject-relabelling-toy-and-sdxl)
  - [Prompt 5 (Subject): How the label table is built](#prompt-5-subject-how-the-label-table-is-built)
  - [Prompt 6 (Subject): One training step in numbers](#prompt-6-subject-one-training-step-in-numbers)
  - [Prompt 7 (Subject): One PoE step in numbers](#prompt-7-subject-one-poe-step-in-numbers)
  - [Subject capstone: The caption model end to end](#subject-capstone-the-caption-model-end-to-end)

## Abstraction chain

Navigation: 📋 [TOC](#table-of-contents) | [Next](#art-direction) ➡️

1. [The caption model](diagram-prompts.md): subject "The caption model end to end"

## Art direction

Navigation: ⬅️ [Previous](#abstraction-chain) | 📋 [TOC](#table-of-contents) | [Next](#reference-images) ➡️

Vendor tinted, chosen by Claude: a pipeline of unbranded parts (VAE, label table, MLP, sampler)
that must read apart at a glance. Zoom depth: one level; each prompt is one part of the model at
the resolution of a single worked example.

## Reference images

Navigation: ⬅️ [Previous](#art-direction) | 📋 [TOC](#table-of-contents) | [Next](#meaning-palette) ➡️

- `../../results/how-much-co-occurrence-the-product-needs/plain-figures/05-the-training-pictures.png`:
  the real 16 x 32 training pictures (red circle, blue square, both side by side on white). Attach
  it to every prompt that draws a training picture, so the small pictures look like these: a solid
  red disc and a solid blue square on a white strip, wider than tall.

## Meaning palette

Navigation: ⬅️ [Previous](#reference-images) | 📋 [TOC](#table-of-contents) | [Next](#glyph-vocabulary) ➡️

| Colour | Means |
|---|---|
| red | the "a red circle" caption and everything it selects |
| blue | the "a blue square" caption and everything it selects |
| purple | the joint caption "a red circle and a blue square" |
| grey | the empty caption, and noise |

## Glyph vocabulary

Navigation: ⬅️ [Previous](#meaning-palette) | 📋 [TOC](#table-of-contents) | [Next](#reading-axes) ➡️

- **Training picture** (artifact): a small white strip with a red disc, a blue square, or both.
- **Latent** (artifact): a row of 128 small squares, "128 numbers".
- **VAE** (primary): an encoder funnel narrowing into the latent, and a decoder widening back out.
- **Label table** (primary): a 4-row table, each row a strip of 64 cells, rows tinted red, blue,
  purple, grey, labelled 0 to 3.
- **MLP** (primary): a stack of three equal-width layers, "MLP: 3 × 1,024".
- **Noise** (artifact): a grey speckle tile.

## Reading axes

Navigation: ⬅️ [Previous](#glyph-vocabulary) | 📋 [TOC](#table-of-contents) | [Next](#devices-in-play) ➡️

Left to right is the order data flows. Top to bottom separates the alternatives being compared.

## Devices in play

Navigation: ⬅️ [Previous](#reading-axes) | 📋 [TOC](#table-of-contents) | [Next](#subject-lane) ➡️

Numbered step badges on prompts 1 and 3, a repeat bracket on prompt 3, a side-by-side split on
prompts 2 and 4.

## Subject lane

Navigation: ⬅️ [Previous](#devices-in-play) | 📋 [TOC](#table-of-contents) | [Next](#prompt-1-subject-one-training-step) ➡️

### Prompt 1 (Subject): One training step

Navigation: ⬅️ [Previous](#subject-lane) | 📋 [TOC](#table-of-contents) | [Next](#prompt-2-subject-the-label-picks-a-row) ➡️

[built] 🖼️ rendered 2026-11-05 `diagrams/caption-model-01-one-training-step.png`
Save as: `diagrams/caption-model-01-one-training-step.png`

Visual thesis: one training step shows the model a picture, its label and some noise, and the
model learns by guessing the clean picture back.

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a wide landscape diagram read left to right in one row, six numbered steps (badges 1 to 6), inside one container titled "One training step". Step 1 at far left: a small white strip picture holding one solid red disc on its left side, with a red tag beside it reading "label 0". Step 2: a VAE encoder drawn as a funnel narrowing from the picture into a row of 128 small squares (the latent). Step 3: a grey speckle noise tile and the latent meet at a mixing node; the result is a speckled latent row, with a chip "noise level 120 of 200". Step 4: three bars stacked into one input row: the noisy latent "128 numbers", a grey bar "noise level: 64", and a red bar "label row: 64", joined into "256 inputs". Above the red bar, a small 4-row table with row 0 tinted red and highlighted, an arrow from row 0 into the red bar. Step 5: an MLP drawn as three equal-width stacked layers. Step 6: its output, a clean latent row, beside the original clean latent; a light-blue measure tile between them reading "squared difference". A thin return arrow from the measure tile back to both the MLP and table row 0, labelled "update both". Beneath step 1, a small grey side chip: "10% of steps: empty label".

Cast: training picture (red disc on white strip); VAE encoder funnel; latent rows of small squares; grey noise tile; 4-row label table with row 0 red; MLP stack; light-blue measure tile.

Flows: solid arrows left to right in the colour of what they carry (red from the label, grey from the noise, dark for the latent). One dashed return arrow from the measure tile to the MLP and to table row 0, meaning "learning updates".

Text in the image: "One training step", "1", "2", "3", "4", "5", "6", "label 0", "VAE encoder", "128 numbers", "noise level 120 of 200", "noise level: 64", "label row: 64", "256 inputs", "MLP: 3 × 1,024", "guess clean picture", "true clean picture", "squared difference", "update both", "10% of steps: empty label".

Exclusions: no text encoder, no CLIP, no words fed to the network, no attention or cross-attention, no UNet, no logos, no extra steps beyond the six. no duplicated components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon without a label, no perspective distortion applied to arrows, no visual metaphor that contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare connector line standing in for a real-world physical object the prompt names.
```
Faithfulness note: the label enters only as table row 0 joined onto the input; there is no text
encoder and no attention anywhere; the MLP's output is a guess of the clean latent, compared with
the true clean latent; the return arrow reaches both the MLP and the table row.

### Prompt 2 (Subject): The label picks a row

Navigation: ⬅️ [Previous](#prompt-1-subject-one-training-step) | 📋 [TOC](#table-of-contents) | [Next](#prompt-3-subject-sampling-with-poe) ➡️

[built] 🖼️ rendered 2026-11-05 `diagrams/caption-model-02-the-label-picks-a-row.png`
Save as: `diagrams/caption-model-02-the-label-picks-a-row.png`

Visual thesis: the network tells captions apart only by which learned row it is handed, and the
same noisy input gives a different guess for each row.

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape diagram in two parts. Left part, container "The label table": a table of four rows, each row a strip of 64 small cells with varied shades, row 0 tinted red with caption text "a red circle", row 1 tinted blue "a blue square", row 2 tinted purple "a red circle and a blue square", row 3 grey "empty". Right part, container "Same input, different row": one speckled noisy latent row at the left, feeding three parallel horizontal lanes stacked top to bottom (red lane, blue lane, purple lane). Each lane: the noisy latent plus that lane's row (red, blue, purple) enters the same MLP stack (drawn three times, identical, each labelled "same MLP"), and exits to a small white strip picture: red lane ends in a picture with one red disc, blue lane ends in a picture with one blue square, purple lane ends in a picture with a red disc on the left and a blue square on the right. A light-blue rule tile under the lanes reads "rows learned from pictures".

Cast: the 4-row label table; noisy latent row; three identical MLP stacks; three result pictures.

Flows: from each table row an arrow in that row's colour into its lane. The noisy latent arrow is dark and splits into all three lanes. Lane arrows keep their lane colour to the result picture.

Text in the image: "The label table", "0", "1", "2", "3", "a red circle", "a blue square", "a red circle and a blue square", "empty", "Same input, different row", "noisy latent", "same MLP", "rows learned from pictures".

Exclusions: no text encoder, no CLIP, no letters or words entering the network, no attention, no tokens, no logos. no duplicated components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon without a label, no perspective distortion applied to arrows, no visual metaphor that contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare connector line standing in for a real-world physical object the prompt names.
```
Faithfulness note: the three MLP stacks are the same network with the same input except for the
table row; the caption words appear only as labels on the table, never as input to the network;
the cell shades are illustrative, not the trained values.

### Prompt 3 (Subject): Sampling with PoE

Navigation: ⬅️ [Previous](#prompt-2-subject-the-label-picks-a-row) | 📋 [TOC](#table-of-contents) | [Next](#prompt-4-subject-relabelling-toy-and-sdxl) ➡️

[built] 🖼️ rendered 2026-11-05 `diagrams/caption-model-03-sampling-with-poe.png`
Save as: `diagrams/caption-model-03-sampling-with-poe.png`

Visual thesis: PoE asks the same network three times per step, once per row, and adds the two
caption pushes on top of the empty one.

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape diagram read left to right, numbered badges 1 to 5. Badge 1 at far left: a fully grey speckle latent row, "pure noise". Then a large bracket container titled "Each of 200 steps", holding: badge 2, three parallel lanes stacked top to bottom: a grey lane "empty row", a red lane "row 0", a blue lane "row 1", each passing the current latent through the same MLP stack (drawn three times, labelled "same MLP"); badge 3, the three lane outputs meet at a light-blue combining tile showing the short formula "empty + 3 × (red push + blue push)"; badge 4, the result produces a slightly less noisy latent, with a loop arrow returning to the start of the bracket, labelled "repeat". After the bracket, badge 5: a VAE decoder funnel widening from the latent into a small white strip picture. Below that picture, two outcome strips side by side: one with a purple smudge labelled "0 relabelled: purple blob", one with a red disc left and a blue square right labelled "relabelled: both shapes".

Cast: noise latent row; three identical MLP stacks; combining tile; VAE decoder funnel; two outcome pictures.

Flows: dark latent arrows left to right; lane arrows in grey, red, blue; one dashed loop arrow for "repeat". No arrow from the joint row (purple) because PoE does not use it.

Text in the image: "1", "2", "3", "4", "5", "pure noise", "Each of 200 steps", "empty row", "row 0", "row 1", "same MLP", "empty + 3 × (red push + blue push)", "repeat", "VAE decoder", "0 relabelled: purple blob", "relabelled: both shapes".

Exclusions: no purple row in use, no text encoder, no CLIP, no attention, no logos, no extra networks. no duplicated components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon without a label, no perspective distortion applied to arrows, no visual metaphor that contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare connector line standing in for a real-world physical object the prompt names.
```
Faithfulness note: PoE uses rows 3 (empty), 0 and 1 only, never row 2; all three lanes are the
same network; the combination is the empty estimate plus 3 times the sum of the two caption pushes;
200 steps; the two outcomes are labelled by relabelling, not by sampler.

### Prompt 4 (Subject): Relabelling, toy and SDXL

Navigation: ⬅️ [Previous](#prompt-3-subject-sampling-with-poe) | 📋 [TOC](#table-of-contents) | [Next](#prompt-5-subject-how-the-label-table-is-built) ➡️

[built] 🖼️ rendered 2026-11-05 `diagrams/caption-model-04-relabelling-toy-and-sdxl.png`
Save as: `diagrams/caption-model-04-relabelling-toy-and-sdxl.png`

Visual thesis: relabelling in the toy is inclusive captioning in SDXL: the same both-things picture
is also filed under each single caption.

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape diagram split into two equal side-by-side containers. Left container titled "Toy model": one white strip picture with a red disc on the left and a blue square on the right, with three arrows fanning right into three labelled bins stacked top to bottom: a purple bin "label 2: joint", a red bin "label 0: red circle", a blue bin "label 1: blue square". The arrow to the purple bin is solid and marked "always"; the arrows to the red and blue bins are dashed and marked "if relabelled". Right container titled "SDXL": one flat illustration of a photograph frame showing a tabby cat sitting beside a golden retriever on a sofa, with three arrows fanning right into three caption cards: a purple card "a cat and a dog", a red card "a cat", a blue card "a dog". Same solid and dashed pattern and the same "always" and "if relabelled" marks. Under the right container a light-blue rule tile: "words read by CLIP". Under the left container a light-blue rule tile: "label numbers only".

Cast: toy both-shapes picture; three label bins; flat illustrated cat-and-dog photo (a striped tabby cat and a golden retriever, side by side, both fully visible); three caption cards; two rule tiles.

Flows: solid arrows mean the caption every both-things picture gets; dashed arrows mean the extra captions relabelling adds.

Text in the image: "Toy model", "SDXL", "label 2: joint", "label 0: red circle", "label 1: blue square", "a cat and a dog", "a cat", "a dog", "always", "if relabelled", "label numbers only", "words read by CLIP".

Exclusions: no merged cat-dog creature, no logos, no networks drawn, no extra bins. no duplicated components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon without a label, no perspective distortion applied to arrows, no visual metaphor that contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare connector line standing in for a real-world physical object the prompt names.
```
Faithfulness note: in both panels the joint caption is always given and the single captions are
the added extras; the cat and the dog are two separate, clearly distinct animals; the toy panel
names label numbers, the SDXL panel names words.

### Prompt 5 (Subject): How the label table is built

Navigation: ⬅️ [Previous](#prompt-4-subject-relabelling-toy-and-sdxl) | 📋 [TOC](#table-of-contents) | [Next](#prompt-6-subject-one-training-step-in-numbers) ➡️

[built] ⏳ not rendered
Save as: `diagrams/caption-model-06-how-the-label-table-is-built.png`

Visual thesis: the label table starts as four rows of random numbers and barely changes; the MLP
learns what each row means, and the rows only have to be different from each other.

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape diagram in three columns left to right. Left container "Start: random rows": a table of four rows, each a strip of 64 small cells in a red-white-blue value shading, row labels "0", "1", "2", "3" with caption tags "a red circle", "a blue square", "a red circle and a blue square", "empty" in red, blue, purple and grey; beneath row 0 a monospace value chip "-1.407  1.094  0.275  0.622 …". Middle: a wide arrow labelled "8,000 training steps" with a small light-blue rule tile under it reading "moves 0.018 on average". Right container "After training": the same four-row table looking nearly identical, with a value chip under row 0 reading "-1.375  1.091  0.272  0.603 …". Below both tables, a light-blue measure tile "rows nearly unrelated: cosine 0.05 to 0.21", and beside the right table an MLP stack with an arrow from the table into it, labelled "MLP learns meaning".

Cast: two copies of the 4-row label table; value chips; MLP stack; two light-blue rule tiles.

Flows: one wide grey arrow left to right between the tables; one dark arrow from the right table into the MLP.

Text in the image: "Start: random rows", "After training", "0", "1", "2", "3", "a red circle", "a blue square", "a red circle and a blue square", "empty", "-1.407  1.094  0.275  0.622 …", "-1.375  1.091  0.272  0.603 …", "8,000 training steps", "moves 0.018 on average", "rows nearly unrelated: cosine 0.05 to 0.21", "MLP learns meaning".

Exclusions: no words or letters entering the network, no text encoder, no CLIP, no attention, no logos. no duplicated components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon without a label, no perspective distortion applied to arrows, no visual metaphor that contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare connector line standing in for a real-world physical object the prompt names.
```
Faithfulness note: the two tables must look nearly the same; the value chips must read exactly as
quoted (real values from the model trained with 0 relabelled, seed 20261022, in
`artifacts/results/how-much-co-occurrence-the-product-needs/toy-caption-table-example.json`); the
caption tags are labels on the rows, not inputs.

### Prompt 6 (Subject): One training step in numbers

Navigation: ⬅️ [Previous](#prompt-5-subject-how-the-label-table-is-built) | 📋 [TOC](#table-of-contents) | [Next](#prompt-7-subject-one-poe-step-in-numbers) ➡️

[built] ⏳ not rendered
Save as: `diagrams/caption-model-07-one-training-step-in-numbers.png`

Visual thesis: one training step is arithmetic on short rows of numbers: mix noise into the clean
latent, attach the label's row, and the MLP's guess lands close to the clean latent.

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape diagram read left to right, numbered badges 1 to 5, each step a column showing a short horizontal vector as four cells with values and a trailing "…". Badge 1: a small white strip picture with a red disc on the left, tag "label 0", and under it the clean latent vector "-1.128  -1.288  -1.364  -1.296 …" labelled "clean latent (128)". Badge 2: a grey noise vector "0.310  1.216  0.774  1.670 …" labelled "random noise (128)", and a mixing tile reading "0.58 × clean + 0.81 × noise" with a chip "noise level 120 of 200". Badge 3: the result "-0.407  0.236  -0.168  0.600 …" labelled "noisy latent (128)". Badge 4: one long input bar split into three coloured segments: dark "noisy latent: 128", grey "noise level: 64", red "row 0: 64"; under the red segment its values "-1.375  1.091  0.272  0.603 …"; the whole bar labelled "256 inputs". Badge 5: an MLP stack, then its output vector "-1.024  -1.344  -1.356  -1.301 …" labelled "guess (128)", placed directly under the clean latent's values for comparison, with a light-blue measure tile "error 0.005".

Cast: training picture; five short vectors drawn as value cells; mixing tile; 256-input bar; MLP stack; measure tile.

Flows: dark arrows left to right for the latent, grey for noise, red from row 0 into the input bar.

Text in the image: "1", "2", "3", "4", "5", "label 0", "clean latent (128)", "-1.128  -1.288  -1.364  -1.296 …", "random noise (128)", "0.310  1.216  0.774  1.670 …", "0.58 × clean + 0.81 × noise", "noise level 120 of 200", "noisy latent (128)", "-0.407  0.236  -0.168  0.600 …", "noisy latent: 128", "noise level: 64", "row 0: 64", "-1.375  1.091  0.272  0.603 …", "256 inputs", "guess (128)", "-1.024  -1.344  -1.356  -1.301 …", "error 0.005".

Exclusions: no text encoder, no CLIP, no attention, no logos, no values other than those quoted. no duplicated components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon without a label, no perspective distortion applied to arrows, no visual metaphor that contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare connector line standing in for a real-world physical object the prompt names.
```
Faithfulness note: every number must read exactly as quoted (real values from
`toy-caption-table-example.json`); the input bar is 128 + 64 + 64 = 256; the guess is compared with
the clean latent, not with the noise.

### Prompt 7 (Subject): One PoE step in numbers

Navigation: ⬅️ [Previous](#prompt-6-subject-one-training-step-in-numbers) | 📋 [TOC](#table-of-contents) | [Next](#subject-capstone-the-caption-model-end-to-end) ➡️

[built] ⏳ not rendered
Save as: `diagrams/caption-model-08-one-poe-step-in-numbers.png`

Visual thesis: at one sampling step PoE asks the same MLP three times and combines the answers with
one line of arithmetic.

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape diagram read left to right. Left: one noisy latent vector labelled "current noisy latent", with a chip "noise level 150 of 200". It feeds three stacked lanes (grey, red, blue), each through an identical MLP stack labelled "same MLP", and each lane ends in a four-cell vector: grey lane "empty row" with "1.197  -0.616  1.522  -1.173 …", red lane "row 0" with "1.196  -0.627  1.511  -1.171 …", blue lane "row 1" with "1.261  -0.667  1.399  -1.204 …". The three vectors meet at a light-blue combining tile reading "empty + 3 × (red push + blue push)", whose output vector reads "1.385  -0.800  1.123  -1.261 …" labelled "PoE noise estimate". A final arrow labelled "remove a little noise" points to a slightly cleaner latent labelled "next step".

Cast: noisy latent; three identical MLP stacks; four value vectors; combining tile.

Flows: dark arrow from the latent into all three lanes; lane arrows grey, red, blue into the combining tile; dark arrow out.

Text in the image: "current noisy latent", "noise level 150 of 200", "same MLP", "empty row", "row 0", "row 1", "1.197  -0.616  1.522  -1.173 …", "1.196  -0.627  1.511  -1.171 …", "1.261  -0.667  1.399  -1.204 …", "empty + 3 × (red push + blue push)", "1.385  -0.800  1.123  -1.261 …", "PoE noise estimate", "remove a little noise", "next step".

Exclusions: no purple row in use, no text encoder, no CLIP, no attention, no logos, no values other than those quoted. no duplicated components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon without a label, no perspective distortion applied to arrows, no visual metaphor that contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare connector line standing in for a real-world physical object the prompt names.
```
Faithfulness note: every number must read exactly as quoted (real values from
`toy-caption-table-example.json`; for example the first number is 1.197 + 3 × ((1.196 − 1.197) +
(1.261 − 1.197)) = 1.385); PoE uses the empty row, row 0 and row 1 only.

### Subject capstone: The caption model end to end

Navigation: ⬅️ [Previous](#prompt-7-subject-one-poe-step-in-numbers) | 📋 [TOC](#table-of-contents)

[built] 🖼️ rendered 2026-11-05 `diagrams/caption-model-05-end-to-end.png`
Save as: `diagrams/caption-model-05-end-to-end.png`

Visual thesis: training files pictures under label rows, and PoE later asks two of those rows at
once, so what each row was trained on decides what PoE can draw.

```
Style: clean corporate cloud-architecture diagram in the style of official vendor documentation. White background; containers are left white, outlined, titled in bold at the top, and never tinted, so colour never sinks a member into its container. Official product logos and service icons at consistent size. Large rounded-outline containers grouping phases. Within a container, each named member (a phase, a lane, a tribe, a branch) carries its own signature colour on its own border, a pale tint of that colour as its fill, and on any arrow leaving it, so members read apart from each other at a glance; the same member's colour carries into every picture that opens it further. A component that is not a member's own thing keeps a shared category colour instead, one hue per kind rather than per member: a light-brown tile for a named branded product with no logo, an amber or apricot tile for an outside party, a light-blue tile for a rule, a measure, or an end user, grey dashed for anything planned or unconfirmed. Small labeled artifact icons (hexagonal data badges, document icons, key icons) beneath the flow for inputs and outputs. A human figure standing for an end user or a named role is a fully coloured flat-illustration figure with real skin tone, hair colour and clothing colour, never a bare line icon and never a photograph. Flat, precise, generous whitespace, clean sans-serif labels, no gradients, no decoration that carries no meaning.

Scene: a landscape overview in three containers left to right. Container 1 "Training pictures": three small stacks of white strip pictures, a red-disc stack, a blue-square stack, a both-shapes stack, each with its label bin (red "label 0", blue "label 1", purple "label 2"), and dashed arrows from the both-shapes stack into the red and blue bins marked "relabelled". Container 2 "The model": a VAE funnel into a 128-square latent, the 4-row label table (red, blue, purple, grey), and the MLP stack, with arrows from latent and table into the MLP. Container 3 "PoE sampling": a noise latent, a bracket "Each of 200 steps" holding three identical MLP lanes (grey, red, blue) and the combining tile, then a VAE decoder to a picture with a red disc and a blue square apart. One in-image legend at the bottom: red "a red circle", blue "a blue square", purple "joint caption", grey "empty or noise", dashed arrow "added by relabelling".

Cast: training picture stacks and label bins; VAE funnel and decoder; latent rows; label table; MLP stacks; combining tile; outcome picture; legend.

Flows: left to right across the three containers; label colours carried on arrows; dashed arrows only for relabelling.

Text in the image: "Training pictures", "label 0", "label 1", "label 2", "relabelled", "The model", "VAE", "128 numbers", "label table", "MLP: 3 × 1,024", "PoE sampling", "Each of 200 steps", "empty + 3 × (red push + blue push)", "a red circle", "a blue square", "joint caption", "empty or noise", "added by relabelling".

Exclusions: no text encoder, no CLIP, no attention, no logos, no components absent from prompts 1 to 4. no duplicated components, no orphan arrows, no arrow terminating in whitespace, no color used without a legend entry, no decorative circuitry or filigree, no invented or misapplied brand logos, no paragraphs of text, no illegible pseudo-code, no crossing arrows where routing could avoid it, no untitled containers, no icon without a label, no perspective distortion applied to arrows, no visual metaphor that contradicts the system's actual semantics, no watermark, no geometric pictogram, circuit symbol, or bare connector line standing in for a real-world physical object the prompt names.
```
Faithfulness note: the training container's relabelling arrows run from the both-shapes stack into
the single-caption bins only; PoE uses the grey, red and blue rows and not the purple; nothing
appears here that is absent from prompts 1 to 4.
