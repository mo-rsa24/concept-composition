# Route: the word-map checker zoom ladder

This route rebuilds or extends the [word-map checker ladder](README.md), one rung per step. It lives in the landing folder because this run was limited to writing there. No step has a row on the project's board.

**To reopen it a rung at a time:**

```text
/drip-skill-route --from /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/word-map-checker/route-zoom-ladder-word-map-checker.md
```

**Environment for every step.** The figures and numbers use a CPU-only Python environment at `$HOME/.cache/word-map-ladder-venv`. If that folder is gone, recreate it:

```bash
V=$HOME/.cache/word-map-ladder-venv
uv venv --python 3.11 $V
uv pip install --python $V/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python $V/bin/python numpy pyyaml cairosvg pillow
# the authors' code, needed only to regenerate numbers/numbers.json
git clone https://github.com/sapir1779/learn-to-steer numbers/sources/learn-to-steer
git -C numbers/sources/learn-to-steer checkout a2ae42746ec93c5936410964be01c85c0ca2266e
```

## Step 1: the numbers

**Reads:** the authors' checkout at `a2ae427`, the public U-Net configs in `numbers/sources/configs/`. **Leaves:** `numbers/numbers.json`.

```bash
cd /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/word-map-checker/numbers
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $HOME/.cache/word-map-ladder-venv/bin/python -I make_numbers.py
```

It worked if the output ends with `above 800: 10` and SDXL prints 1,300 maps per word.

## Step 2: level 0.5, the network view

**Reads:** `numbers/numbers.json`. **Leaves:** figures `wm0` to `wm6` in `0.5-network-view/`.

```bash
cd /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/word-map-checker/0.5-network-view
PYTHONDONTWRITEBYTECODE=1 $HOME/.cache/word-map-ladder-venv/bin/python make_network_view.py
```

It worked if every figure prints `0 crossings`.

## Step 3: level 3, the code walks

**Reads:** the authors' checkout, `poe_repair/methods/_classifier_guided.py` and `2-classifier-guidance/scripts/rung2/attention_push.py`. **Leaves:** `3-code/walk-learn-to-steer.md` and `3-code/walk-our-sampler.md`.

To refresh them after either codebase changes, paste this into a session:

```text
Re-check the two code walks in /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/word-map-checker/3-code/ against the current files they cite. Every file and line range must still hold the quoted code. Change only the walks. Do not run the authors' code, do not run SDXL, publish nothing and commit nothing. Report each citation you changed.
```

## Step 4: the rebuild companion

**Reads:** the two code walks and the learning tree. **Leaves:** `rebuild-from-fundamentals.md`.

To refresh it, paste this into a session:

```text
Re-check every link in /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/word-map-checker/rebuild-from-fundamentals.md against the disk, and repair any that moved using the learning-tree search. Change only that file, publish nothing and commit nothing.
```

## Step 5: level 2, the miniature

**Reads:** nothing outside its folder. **Leaves:** `2-miniature/miniature_answer_key.json` and figures `mm0` to `mm5`.

```bash
cd /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/word-map-checker/2-miniature
PYTHONDONTWRITEBYTECODE=1 $HOME/.cache/word-map-ladder-venv/bin/python -I miniature.py
PYTHONDONTWRITEBYTECODE=1 $HOME/.cache/word-map-ladder-venv/bin/python make_miniature_figures.py
```

It worked if the first command raises no assertion error and the second prints `0 crossings` for all six figures.

## Step 6: level 1, the illustrations

**Reads:** the miniature's key. **Leaves:** `wi1` to `wi6` in `1-illustrations/`, with their prompts and status marks in `1-illustrations/diagram-prompts.md`.

```text
/render-diagrams /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/word-map-checker/1-illustrations/diagram-prompts.md
```

## Step 7: the closing page

**Reads:** every rung. **Leaves:** `read-through-a-small-network.md`.

It worked if every link and thumbnail on the page resolves and every decimal on it appears in the key, the numbers file or the paper at its cited page.
