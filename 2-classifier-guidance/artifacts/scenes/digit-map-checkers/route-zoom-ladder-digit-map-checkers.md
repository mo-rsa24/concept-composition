# Route: the digit-map checkers zoom ladder

This route rebuilds or extends the [digit-map checkers ladder](README.md), one rung per step. It lives in the landing folder because this run was limited to writing there. No step has a row on the project's board.

**To reopen it a rung at a time:**

```text
/drip-skill-route --from /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/digit-map-checkers/route-zoom-ladder-digit-map-checkers.md
```

**Environment for every step.** `~/.claude/venvs/figures/bin/python`, which has numpy and matplotlib. No step needs a GPU or PyTorch: the real outputs in `numbers/real/` come from cluster jobs that have already run. Every command below starts from the ladder folder:

```bash
cd /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/digit-map-checkers
```

## Step 1: the canvases

**Reads:** the four MNIST files. **Leaves:** `numbers/env/dataset.npz`, `numbers/env/manifest.json`, `numbers/env/canvas-kinds-sheet.png`.

```bash
~/.claude/venvs/figures/bin/python -I ../../../scripts/abstract_checkers/build_env.py --mnist /home/molef/PhD/replications/original-superdiff/data/MNIST/raw --out numbers/env --sheet numbers/env/canvas-kinds-sheet.png
```

It worked if it prints `data hash: f70b0fb362e13e255ad332ba14ca93dfb5cf051a5dfd3c86eb30482931d5a5f3`.

## Step 2: the numbers

**Reads:** `numbers/env/`, `numbers/real/count-final.json`, and `numbers/split-94085-log-lines.jsonl` if present. **Leaves:** `numbers/numbers.json`, with the final results under `results_final`, and `numbers/examples/`.

```bash
cd numbers && PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I make_numbers.py && cd ..
```

It worked if it prints `"weights": 481665`, `"count_train": 11504` and `"identity_train": 7004`.

## Step 3: level 0.5, the network view

**Reads:** `numbers/numbers.json`, `numbers/examples/`, `numbers/real/count-explained.npz`. **Leaves:** `dc0` to `dc7` in `0.5-network-view/`.

```bash
cd 0.5-network-view && PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I make_network_view.py && cd ..
```

It worked if every figure prints `0 crossings`.

## Step 4: level 3, the code walk

**Reads:** `v2_learn.py`, `pixel_checker.py`, `checker.py`, `canvas_unet.py`. **Leaves:** `3-code/walk-v2-learn.md`.

To refresh it after the code changes, paste this into a session:

```text
Re-check the code walk at /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/digit-map-checkers/3-code/walk-v2-learn.md against the current files it quotes. Every file and line range must still hold the quoted code; update the commit it names. Change only the walk, publish nothing and commit nothing. Report each citation you changed.
```

## Step 5: the rebuild companion

**Reads:** the code walk and the learning tree. **Leaves:** `rebuild-from-fundamentals.md`.

```text
Re-check every link in /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/digit-map-checkers/rebuild-from-fundamentals.md against the disk, and repair any that moved using the learning-tree search. For each rung marked Gap, search the learning tree once more and link a page if one now teaches it. Change only that file, publish nothing and commit nothing.
```

## Step 6: level 2, the miniature

**Reads:** nothing outside its folder. **Leaves:** `2-miniature/miniature_answer_key.json` and `mm1` to `mm3`.

```bash
cd 2-miniature
PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I miniature.py
PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I make_miniature_figures.py
cd ..
```

It worked if the first prints a mismatch under 1e-6 and raises nothing, and the second prints `0 crossings` three times.

## Step 7: level 1, the illustrations

**Reads:** the miniature's key and `numbers/numbers.json`. **Leaves:** `di1` to `di3` in `1-illustrations/`, with their status marks.

```text
/render-diagrams /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/digit-map-checkers/1-illustrations/diagram-prompts.md
```

## Step 8: the results figure

**Reads:** `numbers/numbers.json`'s `results_final` block. **Leaves:** `numbers/results-so-far.png` and `.svg`.

```bash
cd numbers && PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I make_results_figure.py && cd ..
```

It worked if it prints 0.791, 0.782, 0.555 and 0.556 for the four checker rows and 43 for the push. Every number in `results_final` is quoted from a stage review under `artifacts/notes/progress/abstract-checkers/` or read from `numbers/real/count-final.json`, so a changed review means editing `final_results()` in `numbers/make_numbers.py` and rerunning step 2 first.

## Step 9: the real outputs

**Reads:** `numbers/real/`. **Leaves:** nothing new; steps 2, 3 and 8 read it.

The four files came from the cluster: `count-final.json` and `count-explained.npz` from `/datasets/mmolefe/poe_repair_min/loop-8/v2/count-final/` (job 94108), `count-stages.npz` from job 94116 (`explain_stages.py`), and `trace/` from `/datasets/mmolefe/poe_repair_min/loop-8/v3/trace/` (job 94161, `trace_push.py`). The identity checker was never saved, so no real file exists for it.

## Step 10: the closing page

**Reads:** every rung. **Leaves:** `read-through-a-small-network.md`.

```bash
~/.claude/venvs/figures/bin/python -I numbers/check_pages.py
```

It worked if it prints `0 broken`.
