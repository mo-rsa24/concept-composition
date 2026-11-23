# Route: the NoisyCLIP sentence-score zoom ladder

This route rebuilds or extends the [NoisyCLIP ladder](README.md), one rung per step. It lives in the landing folder because `plans/` belongs to another session, and no step has a row on the project's board.

**To reopen it a rung at a time:**

```text
/drip-skill-route --from /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/route-zoom-ladder-noisyclip.md
```

**Environment for every step.** All steps run on the laptop's CPU. No step uses the cluster, a GPU, the NoisyCLIP checkpoint or Claude API tokens. Run them from the landing folder:

```bash
cd /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score
```

## Step 1: the numbers

**Reads:** the three public configs in `numbers/sources/` and the checkpoint comparison. **Leaves:** `numbers/numbers.json`.

```bash
cd numbers && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with torch --with transformers --with numpy python make_numbers.py
```

It worked if the output prints 427,616,513 total parameters and a red clamp share of 26.8%.

## Step 2: level 0.5, the network view

**Reads:** `numbers/numbers.json`. **Leaves:** figures `nc1` to `nc10` in `0.5-network-view/`.

```bash
cd 0.5-network-view && PYTHONDONTWRITEBYTECODE=1 python3 make_network_view.py
```

It worked if the last line is `all written 0`.

## Step 3: level 3, the code walk

**Reads:** the released code at commit `016b913` and `2-classifier-guidance/scripts/method_search/co3_push.py`. **Leaves:** `3-code/walk-noisyclip.md`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 3-code/check_quotes.py
```

It worked if it prints that all 14 quoted blocks match. The released code is read only and never run.

## Step 4: the rebuild companion

**Reads:** the code walk and the learning tree. **Leaves:** `rebuild-from-fundamentals.md`. To refresh it, paste this into a session:

```text
Re-check every link in /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/rebuild-from-fundamentals.md against the disk, and repair any that moved using the learning-tree search. Change only that file, publish nothing and commit nothing.
```

## Step 5: level 2, the miniature

**Reads:** nothing outside its folder. **Leaves:** `2-miniature/miniature_answer_key.json` and figures `mm0` to `mm6`.

```bash
cd 2-miniature && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with torch python miniature.py
PYTHONDONTWRITEBYTECODE=1 python3 make_miniature_figures.py
```

It worked if the first command raises no assertion error and the second ends with `all written 0`.

## Step 6: level 1, the illustrations

**Reads:** the miniature's key and the style reference. **Leaves:** `ni1` to `ni6` in `1-illustrations/`, with their prompts and status marks in `1-illustrations/diagram-prompts.md`.

```text
/render-diagrams /home/molef/PhD/concept-composition/2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/1-illustrations/diagram-prompts.md
```

Check each picture against `scripts/shared/disallowed_subjects.py` and look at it at full resolution before placing it.

## Step 7: the closing page and the local scene

**Reads:** every rung. **Leaves:** `read-through-a-small-network.md` and `noisyclip-ladder.html`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 numbers/check_pages.py
node build_page.cjs
```

It worked if the check prints no unmatched decimal or broken link and the build prints the page's size. The page is local and nothing is published.
