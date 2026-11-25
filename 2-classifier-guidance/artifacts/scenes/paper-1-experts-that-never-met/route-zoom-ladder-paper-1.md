# Route: Paper 1's zoom ladder

This route rebuilds or extends [Paper 1's ladder](README.md), one rung per step. It lives in the landing folder because this run
was limited to writing there, so no step has a row on the project's board. The question round was answered by the owner before the
run: rungs 0, 0.5, the maths page, 2, 3 and the close; level 1 skipped; the annotated explainer style for the overview and the clean
cheat-sheet style below it.

**To reopen it a rung at a time:**

```text
/drip-skill-route --from /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/paper-1-experts-that-never-met/route-zoom-ladder-paper-1.md
```

**Environment for every step.** `~/.claude/venvs/figures/bin/python` (numpy, matplotlib, Pillow). No step needs a GPU, PyTorch, the
cluster or a model call. Every command starts from the ladder folder:

```bash
cd /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/paper-1-experts-that-never-met
```

Every figure script runs the crossing check in `scripts/figure_kit.py` before it saves, and stops with a list of what crossed what.

## Step 1: the numbers

**Reads:** the cited design, review, report and code files (each quote is checked on its line), the PoE ladder's
`poe_toy.py` and `poe_interaction_numbers.json`. **Leaves:** `data/paper1_numbers.json`.

```bash
~/.claude/venvs/figures/bin/python scripts/make_paper1_numbers.py
```

It worked if it prints `pair checker weights 514497 separate 483402`, the product row `[811, 3893, 4000, 3999, 4000, 4000]`, the
merge row `[3859, 3971, 3955, 3938, 3893, 3758]` and `bayes check 4000`. A `cite failed` line means a quoted source line moved.

## Step 2: level 0, the overview

**Reads:** `data/paper1_numbers.json`, step M1's sheet. **Leaves:** `figures/level0-paper-1-overview.png` and `.svg`.

```bash
~/.claude/venvs/figures/bin/python scripts/make_overview.py
```

It worked if it prints `17 lines checked against 86 nodes and texts`.

## Step 3: level 0.5, the network view

**Reads:** the numbers, step M1's sheet, step M10's smoke sheet. **Leaves:** `figures/level05-the-region-merge`,
`level05-the-checker-push-lane` and `level05-regions-or-plain-product`, each as PNG and SVG. The page `01-the-network-view.md`
links seven figures from three other scenes beside them.

```bash
~/.claude/venvs/figures/bin/python scripts/make_network_view.py
```

It worked if it prints three `wrote` lines.

## Step 4: the maths, rendered

**Reads:** nothing but the equations in the script. **Leaves:** seven PNGs in `maths/`, embedded under the TeX on
`02-the-maths-step-by-step.md`, whose worked numbers come from `worked_state` in the numbers file.

```bash
~/.claude/venvs/figures/bin/python scripts/make_maths.py
```

## Step 5: level 2, the push on the toy

**Reads:** `toy_push` in the numbers file. **Leaves:** `figures/level2-the-push-on-the-toy.png` and `.svg`.

```bash
~/.claude/venvs/figures/bin/python scripts/make_toy_figure.py
```

It worked if it prints `4 lines checked against 19 nodes and texts`.

## Step 6: level 3 and the close

**Reads:** the code at commit `33cea79b`, the report pages named on `README.md`. **Leaves:** `04-the-code.md`, `README.md`,
`ladder.json`. No command: these are written by hand from the numbers file and the report pages, and re-read whenever a report page
they cite changes its verdict.

## What would extend it

**When the two-colour CLEVR experts have trained.** Add a CLEVR row to `04-the-code.md`, replace the overview's digit canvas with a
real two-colour sample if the owner wants the paper's main setting shown, and add the result to the table on `README.md` once its
report page exists.

**When the void-list run is decided.** Update the 1,200 of 1,280 row on `README.md` to whatever the owner decides.
