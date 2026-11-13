# Diagram prompts: where everything in the PhD lives

One picture of the PhD folder as the [PhD organogram](/home/molef/PhD/ORGANOGRAM.md) sorts it, so a
reader sees in one glance which folder holds which kind of work. Source: the organogram as of
2026-11-13, and the folder listing of `/home/molef/PhD`.

## Art direction

**Schematic, drawn by script.** A plain tree on a white page: thin grey connectors, small text,
no cards, icons or shadows, so the picture shows only how the work splits. Drawn by
`make_the_phd_layout.py` beside this file, which writes `the-phd-layout.svg` and a PNG copy. An
earlier Codex render under the Empty direction came out too glossy and was replaced.

## Meaning palette

| Colour | Means |
|---|---|
| Blue | our own research portfolios and their projects |
| Green | writing: a paper or the proposal |
| Purple | other people's code, run or reproduced |
| Grey | admin and the archive |

## Glyph vocabulary

| Glyph | Tier | Stands for |
|---|---|---|
| Large folder card | primary | one group of the PhD folder by kind |
| Small project tile | secondary | one project, which is one question |
| Book | secondary | a paper or the proposal |
| Code bracket badge | secondary | a cloned code repository |
| Storage box | primary | the archive |
| Dashed grey arrow | artifact | a symlink from an old name |

## Devices in play

Colour coding with an in-image legend. No sequencing devices: this is a map of places, not a process.

## Reading axes

Top to bottom reads from our own research, through writing and other people's code, down to
admin and the archive. Left to right inside a card lists its members.

## Subject lane

### Prompt 1 (Subject): Where everything in the PhD lives
[built] 🖼️ rendered 2026-11-13 (`the-phd-layout.png`, drawn by `make_the_phd_layout.py`)
Save as: `the-phd-layout.png`

Thesis: every piece of PhD work has one obvious home, sorted by what kind of work it is.

```text
A clean, illustrated diagram of a researcher's PhD folder, sorted by kind of work, on a light
background, wide landscape format. Draw it as a set of large rounded folder cards, each with a
small icon and a coloured border, holding smaller tiles. Use colour to carry meaning: blue for
our own research portfolios and their projects, green for writing, purple for other people's
code. Every other colour choice is free; make it look like a well-designed product overview,
with soft shadows and small icons, not a plain wireframe.

At the top, a wide card titled "PhD" spans the image as the frame.

Inside it, the top row holds two blue cards side by side:
- A blue card titled "concept-composition" with the subtitle "two concepts, one image". Inside
  it, five small blue project tiles in a row, labelled "1 LoRA correction",
  "2 classifier guidance", "3 understanding composition", "4 inclusive captions" and
  "5 text third expert".
- A blue card titled "medical-imaging" with the subtitle "chest x-rays". Inside it, four small
  blue project tiles labelled "0 cat and dog", "A two findings", "B device shortcut" and
  "C weakened detector".

The middle row holds two cards:
- A green card titled "Writing", holding two book icons labelled "Paper3" and "Proposal".
- A purple card titled "replications", holding five tiles with a code bracket badge each,
  labelled "latent-diffusion", "original-superdiff", "projective-composition",
  "reduce-reuse-recycle", and, slightly larger and outlined, "published methods hub".

The bottom row holds two smaller cards:
- A neutral card titled "cluster-bringup" with a small server icon.
- A storage box titled "_archive", with the subtitle "kept, not deleted".

Symlinks: a small faded label "poe_repair_min" sits outside the concept-composition card, with a
dashed grey arrow pointing into that card. A small faded label "old names" sits outside the
replications card, with a dashed grey arrow pointing into it.

In the bottom right corner, a small legend reads: "blue: our research", "green: writing",
"purple: others' code", "dashed: old name".

Spell every label exactly as quoted. Do not add folders, projects, logos or labels that are not
named here. No watermark.
```

Faithfulness note: concept-composition holds exactly five projects and medical-imaging exactly
four; the published-methods hub sits inside replications and not inside concept-composition; the
dashed arrows start outside the cards and end inside them, so they read as old names pointing at
the new place, never the reverse.
