#!/usr/bin/env python3
"""Write diagram-prompts.md for the ladder's six illustrations, with every number read from the miniature's key.

render-status.json holds each prompt's render mark, so rerunning this after a render keeps the file current.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = json.loads((HERE.parent / "2-miniature" / "miniature_answer_key.json").read_text())
STATUS_FILE = HERE / "render-status.json"
STATUS = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}


def n2(value):
    return f"{value:.2f}"


maps = KEY["maps"]
assert KEY["latent_start"]["cat"] == KEY["latent_start"]["dog"], "the joint prompt's dog map equals its cat map only on the fused start"
TOP_PRODUCT, LOW_PRODUCT = n2(maps["cat_layer1"][0]), n2(maps["cat_layer1"][2])
TOP_JOINT, LOW_JOINT = n2(maps["joint_prompt_cat_layer1"][0]), n2(maps["joint_prompt_cat_layer1"][2])
P_START = n2(KEY["forward_fused"]["p"][KEY["relations"].index("left of")])
P_CAPPED = n2(KEY["dose"]["one_capped_push_then_ddim"]["p_left"])
P_LOOP = n2(KEY["dose"]["lts_16_passes"]["p_left"])
PASSES = KEY["lts_loop"]["updates_made"]
assert (TOP_PRODUCT, TOP_JOINT, P_START, P_CAPPED, P_LOOP, PASSES) == ("0.98", "0.50", "0.20", "0.24", "0.99", 16)

STYLE = (
    "Style: a clean machine-learning cheat-sheet figure in the manner of the attached reference pictures. White page, thin "
    "outlines, a representative few arrows, generous white space, small clear sans-serif labels, real-looking photo tiles with "
    "a thin coloured frame. The part the figure is about glows softly in its colour and everything else is faded pale grey. "
    "It is a finished, welcoming figure and never a bare wireframe. Copy the look of the references only, and none of their "
    "content or words."
)
COLOURS = (
    "Colours carry meaning and nothing else: blue #1f5fd0 for inputs (the noisy picture), green #1d7a3e for hidden values (the "
    "word maps and the small judge), red #c0262d for outputs (the judge's answer), orange #c86a0a for the backward signal and "
    "the push, grey #6b7a88 for what is off, fixed or secondary. Photo content keeps its natural colours."
)
EXCLUDE = (
    "Exclusions: no title banner, no legend box, no paragraphs, no equations, no invented words, no logos, no watermark, no "
    "decorative circuitry, no duplicated components, no arrow ending in empty space, no arrow crossing a label or a picture."
)
ASPECT = "Aspect ratio: landscape 3:2."

PROMPTS = [
    dict(
        n=1, slug="wi1-where-each-word-landed", title="Where each word landed",
        thesis="The product's two prompts never compete, so both words light up the same fused body at full strength.",
        scene="Read left to right: one half-drawn picture, then two rows of word maps, one row for the product's two prompts and one for a single joint prompt.",
        cast=[
            'Left: one square photo tile with a thin blue frame, grainy and half-drawn, showing ONE fused animal across the top half of the frame: a single body with a cat\'s ear and eye on one side of the face and a dog\'s muzzle and floppy ear on the other. The bottom half is grain. Label under it: "half-drawn picture".',
            f'Top row, glowing green, in a rounded panel labelled "two prompts: the product": two 2 by 2 grids side by side, labelled "cat" and "dog". In each grid the two top cells are solid deep green #1d7a3e and print "{TOP_PRODUCT}" in white, and the two bottom cells are pale and print "{LOW_PRODUCT}".',
            f'Bottom row, in a rounded panel labelled "one joint prompt": the same two 2 by 2 grids, labelled "cat" and "dog". In each grid the two top cells are a medium tint of that same green and print "{TOP_JOINT}" in dark ink, and the two bottom cells are pale and print "{LOW_JOINT}".',
            'A small grey tag at the bottom right: "miniature\'s numbers, layer 1".',
        ],
        flows="One thin blue arrow from the picture to the top panel and one to the bottom panel, leaving the picture as one line that splits near the panels.",
        text=["half-drawn picture", "two prompts: the product", "one joint prompt", "cat", "dog", TOP_PRODUCT, LOW_PRODUCT, TOP_JOINT, LOW_JOINT, "miniature's numbers, layer 1"],
        note=f"Exactly one animal in the photo tile; four 2 by 2 grids; product grids print {TOP_PRODUCT} on top and {LOW_PRODUCT} below; joint grids print {TOP_JOINT} on top and {LOW_JOINT} below (key `maps.cat_layer1`, `maps.joint_prompt_cat_layer1`). The failure this answers is expecting the two words to divide the picture between them.",
    ),
    dict(
        n=2, slug="wi2-a-small-judge-reads-only-the-two-maps", title="A small judge reads only the two maps",
        thesis="The checker never looks at the picture: it reads where the two words landed and names their relation.",
        scene="Read left to right: a faded picture that is not connected to anything, two word maps, a small judge, four answers.",
        cast=[
            'Far left, faded pale grey and small: a photo tile of a cat on the left and a dog on the right, with a thin grey line struck diagonally across it. Label under it in grey: "never sees the picture".',
            'Centre left, glowing green: two 2 by 2 grids stacked, labelled "cat map" (the left column bright, the right column pale) and "dog map" (the right column bright, the left column pale).',
            'Centre: a small friendly judge character in a dark green robe, wearing a blindfold, holding a small gavel, sitting at a small bench, with a soft green glow. Label: "small judge".',
            'Right: four rounded answer chips in a column: "above", "below", "left of", "right of". Only "left of" is filled solid red #c0262d with white text; the other three are white with a pale grey outline and grey text.',
        ],
        flows='Two green arrows from the two maps into the judge, one red arrow from the judge to the "left of" chip. No arrow touches the faded picture.',
        text=["never sees the picture", "cat map", "dog map", "small judge", "above", "below", "left of", "right of"],
        note='The picture tile has no arrow attached; the judge is blindfolded; exactly four answer chips, and only "left of" is red. The failure this answers is thinking the checker judges the image itself.',
    ),
    dict(
        n=3, slug="wi3-it-learned-the-word-instead-of-the-picture", title="It learned the word instead of the picture",
        thesis="A judge trained only on true captions reads the caption's word out of the maps; one trained on true and wrong captions reads the layout.",
        scene="Read left to right: one photo with a wrong caption, then two judges in two rows, each giving its answer.",
        cast=[
            'Left: one photo tile with a thin blue frame, a brown dog clearly on the LEFT and a grey tabby cat clearly on the RIGHT, side by side on a plain floor. Above the tile a speech-style caption chip with a grey outline: "caption: a dog above a cat". Under the tile: "in the picture: dog left of cat".',
            'Between the photo and the judges: one small pair of green 2 by 2 grids labelled "word maps".',
            'Top row: a small judge character in a dark green robe, wearing a plain dark blindfold and no glasses, beside the label "trained on true captions". It answers with a red chip "says: above", with a small dark ink cross mark beside the chip.',
            'Bottom row: the same blindfolded judge character beside the label "trained on true and wrong captions". It answers with a red chip "says: left of", with a small dark ink tick mark beside the chip.',
            'A small grey tag at the bottom right: "Learn-to-Steer\'s example, paper p. 4".',
        ],
        flows="One blue arrow from the photo to the word maps. One green line leaves the word maps and splits near the two judges into two arrows, one to each. One red arrow from each judge to its answer chip.",
        text=["caption: a dog above a cat", "in the picture: dog left of cat", "word maps", "trained on true captions", "says: above", "trained on true and wrong captions", "says: left of", "Learn-to-Steer's example, paper p. 4"],
        note='The dog is on the left and the cat on the right; the caption says "above"; both judges are blindfolded and read only the word maps; the top judge says above with a cross and the bottom says left of with a tick. The top row is the paper's observation (p. 4). The bottom row is what the fix is meant to do, since the paper reports no direct leakage rate after it (p. 8). The failure this answers is trusting a high training accuracy.',
    ),
    dict(
        n=4, slug="wi4-the-judges-complaint-pushes-the-noise", title="The judge's complaint pushes the noise",
        thesis="The judge's error is sent back to the noisy picture, and the update moves the cat's map left and the dog's map right.",
        scene="Two lanes. The upper lane runs left to right: noisy picture, two word maps, small judge. The lower lane runs right to left in orange: the judge's error going back to the picture.",
        cast=[
            'Upper left: a square grainy photo tile with a thin blue frame, one fused cat-dog animal barely visible in the noise. Label: "noisy picture".',
            'Upper middle, glowing green: two crisp 2 by 2 grids side by side in deep green #1d7a3e, labelled "cat map" (left column dark, right column pale) and "dog map" (right column dark, left column pale). A small bold orange arrow sits directly above each grid: above the cat map it points LEFT, above the dog map it points RIGHT.',
            'Upper right: the small judge character with a white wig, dark green robe, green blindfold, gavel and bench, with a small empty orange speech mark. Label: "small judge".',
            'Lower lane: one thick orange arrow running from under the judge, right to left, with clear space above it, ending under the noisy picture and turning up into its bottom edge. On the arrow, the label "the error, sent back as a push".',
            'A small grey tag at the bottom right: "our proposal, not yet run on SDXL".',
        ],
        flows="Thin blue arrow from picture to the maps, thin green arrow from the maps to the judge, one thick orange return lane under them all. The orange lane touches no map and no label.",
        text=["noisy picture", "cat map", "dog map", "small judge", "the error, sent back as a push", "our proposal, not yet run on SDXL"],
        note="One orange return lane below the forward path, ending at the picture; the cat map's arrow points left and the dog map's right; the proposal tag is present. The failure this answers is thinking the judge's weights change during sampling.",
    ),
    dict(
        n=5, slug="wi5-one-capped-nudge-against-sixteen-big-steps", title="One capped nudge against sixteen big steps",
        thesis=f"On the miniature, one capped push moves the chance of the right relation from {P_START} to {P_CAPPED}, and {PASSES} full steps move it to {P_LOOP}.",
        scene="Two stacked rows, each ending in a horizontal meter of the chance that the judge says left of.",
        cast=[
            f'Top row, labelled "one capped push": one short orange arrow, then a horizontal meter from 0 to 1 with a thin outline, filled red up to about a quarter, a grey tick at {P_START} and the fill ending at {P_CAPPED}. Print "{P_START}" at the tick and "{P_CAPPED}" at the fill\'s end.',
            f'Bottom row, labelled "sixteen big steps": a strip of exactly {PASSES} evenly spaced orange arrows in a line, then the same meter filled red almost to the end, with a grey tick at {P_START} and the fill ending at {P_LOOP}. Print "{P_START}" at the tick and "{P_LOOP}" at the fill\'s end.',
            'Above the two meters, one small label: "chance of left of".',
            'A small grey tag at the bottom right: "miniature\'s numbers".',
        ],
        flows="No arrows between the rows.",
        text=["one capped push", "sixteen big steps", "chance of left of", P_START, P_CAPPED, P_LOOP, "miniature's numbers"],
        note=f"Exactly 1 orange arrow in the top row and exactly {PASSES} in the bottom row; meters print {P_START} to {P_CAPPED} and {P_START} to {P_LOOP} (key `dose`). The failure this answers is calling a checker useless after testing it at a far smaller dose.",
    ),
    dict(
        n=6, slug="wi6-the-map-can-move-without-a-second-animal", title="The map can move without a second animal",
        thesis="Separating the word maps is not the same as drawing two animals, and our own push on the maps drew two animals on 0 of 8 seeds.",
        scene="Read left to right: two separated word maps, an arrow with a question mark, one finished picture that still holds one animal.",
        cast=[
            'Left, glowing green: two 2 by 2 grids side by side, labelled "cat map" (left column bright) and "dog map" (right column bright).',
            'Middle: one grey arrow with a small grey question mark above it.',
            'Right: one larger, clean, finished photo tile with a thin grey frame, showing ONE single fused animal, with a cat\'s ear and a dog\'s muzzle on the same head, sitting in the middle of the frame. Label under it: "still one animal".',
            'Top left small grey tag: "the risk, drawn".',
            'Bottom, one small caption line in dark grey: "our SDXL push on the maps: two animals on 0 of 8 seeds".',
        ],
        flows="One grey arrow from the maps to the photo.",
        text=["cat map", "dog map", "still one animal", "the risk, drawn", "our SDXL push on the maps: two animals on 0 of 8 seeds"],
        note='Exactly one animal in the photo; the caption prints "0 of 8"; the "the risk, drawn" tag is present, because the run recorded the picture outcome and did not record that its maps separated. The failure this answers is reading a separated map as a separated picture.',
    ),
]


def prompt_block(p):
    cast = "\n".join(f"- {line}" for line in p["cast"])
    text = ", ".join(f'"{t}"' for t in dict.fromkeys(p["text"]))
    return f"{STYLE}\n\nScene: {p['scene']} {COLOURS}\n\nCast:\n{cast}\n\nFlows: {p['flows']}\n\nText in the image, spelled exactly and no other words: {text}.\n\n{EXCLUDE}\n\n{ASPECT}"


def main():
    rendered = sum(STATUS.get(p["slug"], "").startswith("🖼️") for p in PROMPTS)
    stuck = sum(STATUS.get(p["slug"], "").startswith("⚠️") for p in PROMPTS)
    waiting = len(PROMPTS) - rendered - stuck
    counts = f"{len(PROMPTS)} prompts · {rendered} rendered · {waiting} waiting" + (f" · {stuck} stuck" if stuck else "")
    toc = ["Abstraction chain", "Art direction", "Reference images", "Meaning palette", "Glyph vocabulary", "Reading axes", "Devices in play", "Subject lane", "Process lane"]

    def anchor(s):
        return s.lower().replace(" ", "-")

    out = [
        "# The word-map checker: illustrated map", "",
        f"Six pictures that tell the idea of Learn-to-Steer's spatial head, set on SDXL's product of experts, drawn on the miniature's story and numbers. Every photo tile is an image model's drawing for teaching, and none is an SDXL sample. `{counts}`", "",
        "## Table of contents", "",
    ]
    out += [f"- [{s}](#{anchor(s)})" for s in toc]
    for p in PROMPTS:
        out.append(f"  - [Prompt {p['n']}: {p['title']}](#prompt-{p['n']}-subject-{anchor(p['title']).replace(chr(39), '')})")
    out += [
        "", "## Abstraction chain", "",
        "1. [The word-map checker](diagram-prompts.md): subject \"The map can move without a second animal\". This ladder is standalone, so the chain has one entry.", "",
        "## Art direction", "",
        "Clean cheat-sheet, the owner's choice for journey 13 and for this ladder. The style paragraph opens every prompt. The zoom depth is one level: these pictures tell the idea, and the network view and the miniature carry the sizes and the working.", "",
        "## Reference images", "",
        "These three look references are attached to every render, each downscaled to about 1280 px. They set the drawing style only.", "",
        "- `cheatsheet-cnn-real-image-through-stages.png`: a real picture flowing through stages.",
        "- `cheatsheet-transfer-learning-frozen-faded-trained-glowing.png`: faded frozen parts beside glowing trained ones.",
        "- `cheatsheet-step-strip-with-arrows.png`: a strip of steps joined by arrows.", "",
        "All three live in journey 13's `artifacts/figures/network-view/reference/`.", "",
        "## Meaning palette", "",
        "| Colour | Meaning |", "|---|---|",
        "| blue `#1f5fd0` | inputs: the noisy picture |",
        "| green `#1d7a3e` | hidden values: word maps and the small judge |",
        "| red `#c0262d` | outputs: the judge's answer and the chance it gives |",
        "| orange `#c86a0a` | the backward signal and the push |", "",
        "Grey `#6b7a88` is structural: what is off, fixed or secondary.", "",
        "## Glyph vocabulary", "",
        "**Small judge (primary).** A small figure in a dark green robe, blindfolded because it reads only the maps. It stands for the trained head $C$.", "",
        "**Word map (primary).** A 2 by 2 grid of green cells, one per word, as in the miniature.", "",
        "**Noisy picture (primary).** A grainy photo tile with a blue frame, standing for the latent $x_t$.", "",
        "**Answer chip (secondary).** A rounded chip naming one relation, filled red when it is the judge's answer.", "",
        "## Reading axes", "",
        "Left to right is the forward path. A lane underneath, running right to left, is the backward signal. Stacked rows are a comparison.", "",
        "## Devices in play", "",
        "Glow against fade, one return lane, stacked comparison rows and a small grey tag saying whose numbers a picture shows.", "",
        "## Subject lane", "",
    ]
    for p in PROMPTS:
        status = STATUS.get(p["slug"], "⏳ not rendered")
        out += [
            f"### Prompt {p['n']} (Subject): {p['title']}", "",
            f"[planned] {status}", "",
            f"Save as: `{p['slug']}.png`", "",
            f"**Visual thesis:** {p['thesis']}", "",
            "```text", prompt_block(p), "```", "",
            f"Faithfulness note: {p['note']}", "",
        ]
    out += [
        "## Process lane", "",
        "This ladder has no plan tree, so it has no process lane. The order of the rungs is in the ladder's route file.", "",
    ]
    (HERE / "diagram-prompts.md").write_text("\n".join(out), encoding="utf-8")
    if "--dispatch-files" in sys.argv:
        drop = HERE / "drop" / "prompts"
        drop.mkdir(parents=True, exist_ok=True)
        for p in PROMPTS:
            (drop / f"{p['slug']}.txt").write_text(prompt_block(p) + "\n", encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
