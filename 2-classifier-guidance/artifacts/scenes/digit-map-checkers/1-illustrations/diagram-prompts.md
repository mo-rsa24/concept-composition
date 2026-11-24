# The digit-map checkers: illustrated map

Three pictures that tell the idea of the two checkers, drawn on the miniature's story and numbers. They are written and not rendered. `3 prompts · 0 rendered · 3 waiting`

## Table of contents

- [Abstraction chain](#abstraction-chain)
- [Art direction](#art-direction)
- [Reference images](#reference-images)
- [Meaning palette](#meaning-palette)
- [Glyph vocabulary](#glyph-vocabulary)
- [Reading axes](#reading-axes)
- [Subject lane](#subject-lane)
  - [Prompt 1: The same eyes, two questions](#prompt-1-subject-the-same-eyes-two-questions)
  - [Prompt 2: Counting is easy, naming is not](#prompt-2-subject-counting-is-easy-naming-is-not)
  - [Prompt 3: Two backward passes](#prompt-3-subject-two-backward-passes)

## Abstraction chain

1. [The digit-map checkers](diagram-prompts.md): subject "Two backward passes". This ladder is standalone, so the chain has one entry.

## Art direction

Clean cheat-sheet, the owner's choice for journey 13 and the word-map checker ladder, carried over as the default because this ladder was built unattended. The style paragraph opens every prompt. These pictures tell the idea; the network view and the miniature carry the sizes and the working.

## Reference images

The same three look references as the word-map checker ladder, from journey 13's `artifacts/figures/network-view/reference/`: `cheatsheet-cnn-real-image-through-stages.png`, `cheatsheet-transfer-learning-frozen-faded-trained-glowing.png`, `cheatsheet-step-strip-with-arrows.png`. Also attach `../0.5-network-view/dc1-the-canvas-and-its-noise.png` so the digit canvases look like the real ones.

## Meaning palette

| Colour | Meaning |
|---|---|
| blue `#1f5fd0` | inputs: the noisy canvas and the timestep |
| green `#1d7a3e` | hidden values: the maps and the checker |
| red `#c0262d` | outputs: the logit and the noise guesses |
| orange `#c86a0a` | the backward signal and the push |
| purple `#7b3fa0` | the count branch |
| teal `#0f7f86` | the identity branch |

Grey `#6b7a88` is structural: frozen, fixed or off.

## Glyph vocabulary

**Digit canvas (primary).** A wide black tile, twice as wide as tall, with white handwritten digits, framed thin blue.

**Checker (primary).** A small rounded box glowing green with a tiny eye icon, the same box for both branches, with a purple or teal tab naming the branch.

**Label chip (secondary).** A rounded chip reading "1" or "0" in the branch's colour.

## Reading axes

Left to right is the forward path. A lane underneath, running right to left, is the backward signal. Stacked rows are a comparison.

## Subject lane

### Prompt 1 (Subject): The same eyes, two questions

[planned] ⏳ not rendered

Save as: `di1-the-same-eyes-two-questions.png`

**Visual thesis:** One network shape, trained twice on different labels, becomes two different checkers.

```text
Style: a clean machine-learning cheat-sheet figure in the manner of the attached reference pictures. White page, thin outlines, a representative few arrows, generous white space, small clear sans-serif labels, real-looking digit tiles with a thin coloured frame. The part the figure is about glows softly in its colour and everything else is faded pale grey. It is a finished, welcoming figure and never a bare wireframe. Copy the look of the references only, and none of their content or words.

Scene: Two stacked rows that share one column of canvases on the left. Colours carry meaning and nothing else: blue #1f5fd0 inputs, green #1d7a3e hidden values and the checker, red #c0262d outputs, purple #7b3fa0 the count branch, teal #0f7f86 the identity branch, grey #6b7a88 off or fixed.

Cast:
- Left column: two wide black digit tiles with thin blue frames, the top showing a handwritten "1" on the left half and "3" on the right half, labelled "apart"; the bottom showing "3" on the left half and "2" on the right half, labelled "wrong".
- Top row: a small glowing green rounded box with a purple tab reading "COUNT", then two purple chips, "1" beside the apart tile's arrow and "1" beside the wrong tile's arrow.
- Bottom row: the identical green box with a teal tab reading "IDENTITY", then two teal chips, "1" for apart and "0" for wrong.

Flows: from each tile one thin arrow into each row's box; from each box one arrow per tile to its chip.

Text in the image, spelled exactly and no other words: "apart", "wrong", "COUNT", "IDENTITY", "1", "0".

Exclusions: no title banner, no legend box, no paragraphs, no equations, no invented words, no logos, no watermark, no arrow ending in empty space, no arrow crossing a label or a picture.

Aspect ratio: landscape 3:2.
```

Faithfulness note: Two tiles only, showing 1 | 3 and 3 | 2; count chips read 1 and 1; identity chips read 1 and 0 (`numbers.json` key `examples.apart_1_3` and `examples.wrong_1_3`). The failure this answers is thinking the two checkers are different architectures.

### Prompt 2 (Subject): Counting is easy, naming is not

[planned] ⏳ not rendered

Save as: `di2-counting-is-easy-naming-is-not.png`

**Visual thesis:** Two blobs one per half can be seen through heavy noise; which digit each blob is cannot, on the same canvas.

```text
Style: a clean machine-learning cheat-sheet figure in the manner of the attached reference pictures. White page, thin outlines, a representative few arrows, generous white space, small clear sans-serif labels, real-looking digit tiles with a thin coloured frame. The part the figure is about glows softly in its colour and everything else is faded pale grey. It is a finished, welcoming figure and never a bare wireframe. Copy the look of the references only, and none of their content or words.

Scene: One heavily noised wide digit tile in the middle, grainy black and white, in which two faint brighter patches sit one in each half and no digit shape can be read. Colours carry meaning and nothing else: blue #1f5fd0 inputs, purple #7b3fa0 the count branch, teal #0f7f86 the identity branch.

Cast:
- Centre: the noisy tile with a thin blue frame, labelled "t = 480".
- Above it, purple: two soft purple ovals drawn over the two patches, one per half, and a purple caption "two blobs, one per half".
- Below it, teal: two teal question marks over the two patches, and a teal caption "which digits?".

Flows: none.

Text in the image, spelled exactly and no other words: "t = 480", "two blobs, one per half", "which digits?".

Exclusions: no readable digit inside the noisy tile, no title banner, no legend box, no equations, no invented words, no logos, no watermark.

Aspect ratio: landscape 3:2.
```

Faithfulness note: No digit may be readable in the tile; this matches `dc1`, where the digits cannot be seen by eye at t = 480. It is a teaching picture of the question and states no result.

### Prompt 3 (Subject): Two backward passes

[planned] ⏳ not rendered

Save as: `di3-two-backward-passes.png`

**Visual thesis:** Training's backward pass ends at the checker's weights; the push's backward pass runs on through the frozen model to the canvas and changes no weight.

```text
Style: a clean machine-learning cheat-sheet figure in the manner of the attached reference pictures. White page, thin outlines, a representative few arrows, generous white space, small clear sans-serif labels, real-looking digit tiles with a thin coloured frame. The part the figure is about glows softly in its colour and everything else is faded pale grey. It is a finished, welcoming figure and never a bare wireframe. Copy the look of the references only, and none of their content or words.

Scene: Two stacked rows, each a forward path left to right with an orange lane beneath running right to left. Colours carry meaning and nothing else: blue #1f5fd0 the noisy canvas, grey #6b7a88 the frozen digit model, green #1d7a3e the maps and the checker, red #c0262d the output, orange #c86a0a the backward signal.

Cast, the same in both rows: a noisy wide tile with a blue frame, a faded grey block labelled "frozen digit model", two small green squares labelled "maps", a glowing green box labelled "checker", a red chip labelled "logit".
- Top row, labelled "training": the orange lane starts at the red chip, runs left, and ends with small orange arrows pointing up into the checker box. A tiny orange wrench beside the checker. The lane stops before the maps.
- Bottom row, labelled "push": the orange lane starts at the red chip and runs all the way left under the checker, the maps and the grey block, and ends with an arrow up into the noisy tile.

Text in the image, spelled exactly and no other words: "training", "push", "frozen digit model", "maps", "checker", "logit".

Exclusions: no title banner, no legend box, no paragraphs, no equations, no invented words, no logos, no watermark, no arrow ending in empty space, no arrow crossing a label or a picture.

Aspect ratio: landscape 3:2.
```

Faithfulness note: In the top row the orange lane must end at the checker and never reach the maps or the grey block; in the bottom row it must reach the tile (`v2_learn.py` lines 106 and 126, and `03-design.md`'s push). The failure this answers is thinking the push trains the checker.
