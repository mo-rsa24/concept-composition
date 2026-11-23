# Composing by Regions

Six frames on how region-based methods reconcile overlapping windows, worked on an invented canvas of 8 positions.
Window A denoises positions 0 to 4 and window B denoises 3 to 7, so they disagree at positions 3 and 4. MultiDiffusion's reconcile
step keeps their weighted average at every noise level. Domain Expansion then adds a trained network before that average, and the
last frame compares what each base model saw with Paper 1's setting.

| File | What it is |
|---|---|
| `index.html` | the page source, 6 frames, published as the private artifact below |
| `toy.py` | computes every number on the page in exact fractions; needs only NumPy |

**Which reading it serves**

Reading 7, on Diffusion Domain Expansion and MultiDiffusion. The page is the first picture to open in the
[reading plan's section on composing by regions](../../notes/progress/targets/reading-plan.md#sitting-7-composing-by-regions).

**Published at**

[Composing by Regions](https://claude.ai/artifact/6qY8W3eiRtmoL48yNxUVpx). It is private to its owner until shared from the page's
Share menu. Each frame opens at `#f1` to `#f6`.

**How it was made**

The 8 positions, both windows and every value are invented and illustrative. A small Python script applies an exact Gaussian
denoiser for each window over 5 noise steps. On this toy the gap between the windows at position 3, in brightness units, falls from
2.40 to 0.09 with reconciling on and grows from 2.40 to 3.68 with it off. Codex checked the page's claims and page numbers against
the extracted paper pages, and all 24 numerical cases and both five-step walks matched the script. The script is `toy.py` in this folder. `python3 toy.py toy.json` needs only NumPy and writes the page numbers to `toy.json`.

**What it does not show**

No real model ran, and the toy's falling gap says nothing about how fast a real model's gap falls. Proposition 3.1 does not cover
this toy, because the toy's loss does not reach zero. No coordinator was trained, so its new outputs carry no values. MultiDiffusion's
Equations 4 and 5 can differ with unequal weights, and the page follows Equation 5. The page gives no verdict on the coordinator
against the baselines in Domain Expansion's Table 2, and those pages do not say how many objects each CLEVR image holds.
