# Outside Planners and Critics

Fourteen presses on what RPG's language-model planner and CountLoop's critic add to text-to-image, and what each outside model costs.
RPG's language model rewrites the prompt and splits the canvas into regions. CountLoop's vision models plan a layout, critique the
image and repair the plan in a loop. The last two presses show Revisiting CFG's fixed protocol and a guidance gain that disappears
on another model.

| File | What it is |
|---|---|
| `index.html` | the page source, 14 presses, published as the private artifact below |
| `cells.py` | reads the chart values from Table 1 of Revisiting CFG, PDF page 3 |

**Which reading it serves**

Reading 10, on RPG, CountLoop and Revisiting CFG. The page is the first picture to open in the
[reading plan's section on what an outside planner or critic buys](../../notes/progress/targets/reading-plan.md#sitting-10-what-an-outside-planner-or-critic-buys).

**Published at**

[Outside Planners and Critics](https://claude.ai/artifact/HJYL5qt4kpdn4D4g4dVZME). It is private to its owner until shared from the
page's Share menu. Each press opens at `#p1` to `#p14`.

**How it was made**

The prompts, split ratios, weights and table values are taken from the papers, with page numbers in the page's sources panel. The
scene drawings, planning graph, layout, region pictures and colour tiles are illustrative and are not model outputs. The Revisiting
CFG rows were read from its Table 1, including each cell's printed colour. Codex reviewed the page against the cited source pages.

**What it does not show**

The papers' pages print no calls, time or money per image, so the cost presses name only which outside models each method needs.
CountLoop's counting error (mean absolute error, lower is better) is 29.59 with the vision model alone and 7.59 with the detector
and scorer together, and the paragraph reporting it does not restate the benchmark. The Design and Critic models are counted as two
jobs, because these pages do not say whether they are one model. The page gives no verdict on whether the second CountLoop row of
Table 1 is scored by its own critic.

**Rerunning the chart values**

Run this in an empty folder. It writes `cells.json` there.

```bash
pdftotext -bbox -f 3 -l 3 /home/molef/PhD/concept-composition/artifacts/notes/progress/targets/pdfs/2608.16786.pdf page.html
pdftoppm -f 3 -l 3 -r 200 /home/molef/PhD/concept-composition/artifacts/notes/progress/targets/pdfs/2608.16786.pdf page
python3 /home/molef/PhD/concept-composition/artifacts/scenes/outside-planners-and-critics/cells.py page.html page-03.ppm
```
