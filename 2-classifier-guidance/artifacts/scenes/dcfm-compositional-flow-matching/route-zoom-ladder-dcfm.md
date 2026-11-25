# Route: the DCFM zoom ladder

This route rebuilds or extends the [DCFM ladder](README.md), one rung per step. It lives in the landing folder, beside the ladder,
because the owner asked for it there; no step has a row on the project's board.

**To reopen it a rung at a time:**

```text
/drip-skill-route --from /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/dcfm-compositional-flow-matching/route-zoom-ladder-dcfm.md
```

**Environment for every step.** All steps run on the laptop's CPU with numpy and matplotlib from the figures virtual environment. No
step uses the cluster, a GPU, a downloaded model or Claude API tokens. The MNIST digits are read from the raw files already at
`/home/molef/Playground/data/MNIST/raw/`. Run every step from the landing folder, with numpy pinned to one thread (two numpy processes
on more threads slowed each other about forty times on this laptop):

```bash
cd /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/dcfm-compositional-flow-matching
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/.claude/venvs/figures/bin/python
```

## Step 1: the numbers

**Reads:** the DCFM PDF's printed values (typed into the script with their page) and nothing else. **Leaves:** `dcfm_numbers.json`,
`toy_arrays.npz`, `toy_run.log`.

```bash
$PY make_dcfm_numbers.py > toy_run.log 2>&1
```

It takes about ten minutes, almost all of it training the toy's small networks. It worked if `toy_run.log` ends with `wrote ...dcfm_numbers.json`,
`printed.colored_mnist.c_min_eq2` is 0.19, and `toy.ii_sum_rule_missing.four_cell_reference.shares["2,1"]` is above 0.99. A rerun must leave
every value already in the JSON unchanged (the seeds are fixed). `--printed-only` refreshes the printed and worked sections in a second.

## Step 2: level 0, the overview

**Reads:** `dcfm_numbers.json`, the MNIST raw files. **Leaves:** `figures/00-overview.png` and `.svg`.

```bash
$PY make_overview.py
```

It worked if it prints `wrote figures/00-overview.png and .svg`. The script refuses to write if any line crosses a box, a picture or a
label, or a label spills out of its box (`figcheck.py`).

## Step 3: level 0.5, the network view

**Reads:** `dcfm_numbers.json`, the MNIST raw files. **Leaves:** `figures/01a` to `figures/01d`, PNG and SVG.

```bash
$PY make_network_view.py
```

It worked if it prints four `wrote figures/01...` lines.

## Step 4: the maths, step by step

**Reads:** the PDF pages 2 to 3, 5 to 6 and 12 to 13. **Leaves:** `maths/*.png` (18 renders) and `02-the-maths-step-by-step.md`.

```bash
$PY render_maths.py
```

It worked if it prints 18 `wrote` lines. To check the page against the PDF, paste this into a session:

```text
Read /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/dcfm-compositional-flow-matching/02-the-maths-step-by-step.md and compare every equation with the DCFM PDF at /home/molef/PhD/concept-composition/artifacts/notes/progress/targets/pdfs/2606.10153.pdf (render pages 2, 3, 5, 6, 12, 13 with pdftoppm and read them). List every symbol that differs, and every worked number that does not match dcfm_numbers.json in the same folder. Change nothing; report only.
```

## Step 5: level 2, the toy

**Reads:** `dcfm_numbers.json`, `toy_arrays.npz`. **Leaves:** `figures/03a` to `figures/03e` and `03-the-toy.md`.

```bash
$PY make_toy_figures.py
```

It worked if it prints five `wrote figures/03...` lines.

## Step 6: level 3, the code (skipped)

DCFM's code is not public (checked 2026-11-25: no link in the paper, none on the arXiv record, no matching repository under the first
author's or the lab's GitHub account). If it appears, paste this:

```text
/zoom-ladder --auto DCFM, level 3 only: walk the authors' code for Eq. 9 to 17 line by line against /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/dcfm-compositional-flow-matching/02-the-maths-step-by-step.md, clone it read-only under ~/PhD/replications/, record the commit in ladder.json, and fill every "not stated" in dcfm_numbers.json that the code settles (network, solver, step count, lambda, pi, how r is drawn). Write only inside that folder.
```

## Step 7: level 1, the illustrations (skipped)

The owner skipped this rung for DCFM.

## Step 8: the closing page

**Reads:** every page above and `dcfm_numbers.json`. **Leaves:** `README.md`. It worked if every number on it appears in `dcfm_numbers.json`
and every link opens.
