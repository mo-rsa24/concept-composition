# Layout sheet: every new figure and table in the review response

One row per figure and table the revision of submission 56173 adds or rebuilds, planned before its result exists. Each slot has a fixed size in the ICLR layout, a placeholder at that size in `paper/iclr-56173/iclr2027_conference.tex`, a row in [the figure register](../../../paper/iclr-56173/figures.md), and one result file. `1-lora-correction/scripts/refresh_paper_slots.py` rebuilds a slot from its result file, so an early result can be placed and replaced without moving a page. Every slot obeys [the comparison protocol](protocol.md).

ICLR text width is 5.5 in; "full" below means 5.5 in. The smallest text in any figure is 7 pt at printed size, and tables are set in `\footnotesize`, which this paper's style file defines as 9 pt on 10 pt leading (`iclr2027_conference.sty` line 175). Result files live under `artifacts/results/<the question the work asked>/`, and every table's CSV carries rule 9's twelve axis columns (`model`, `method`, `adapter`, `sampler`, `steps`, `guidance`, `resolution`, `pair`, `seed`, `group`, `setting`, `nfe`) beside its scores.

## The slots

| Slot | What | Where | Width × height | Rows, columns, panel order | Result file | Board row | Grid it ships with (rule 8) |
|---|---|---|---|---|---|---|---|
| R1 | the benchmark: every SDXL method on the 276 Attend-and-Excite prompts | main, Section 4.1 | natural width, centered, at most full × 1.8 in | rows: joint, PoE, CO3, PoE + Langevin†, ours, then CO3's printed Table 1 as a grey cited row; column groups: animals (66) with Comp., CLIP, BLIP, IR; animals with objects (144) with CLIP, BLIP, IR; objects (66) with CLIP, BLIP, IR; seeds 9 to 12; the showcase prompt "a cat and a dog" marked | `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/main_table.csv`, its rows with `model` SDXL, `group` naming the set | PRZE-1 | R12 |
| R5 | the benchmark on SD 1.5, each method's official code on its own model | appendix | natural width, centered, at most full × 1.8 in | rows: PoE, PoE + measured correction, [Attend-and-Excite](../../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/06-attend-and-excite-bringing-a-missing-subject-back.md), [SynGen](../../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/09-syngen-binding-by-the-prompts-grammar.md), [Divide and Bind](../../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/10-divide-and-bind-attendance-plus-binding.md), [InitNO](../../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/11-initno-choosing-the-starting-noise.md), [CONFORM](../../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/07-conform-pulling-look-alike-subjects-apart.md), ours (SD 1.5 adapter, filled only if route step 12 trains it, otherwise dropped and the limit stated); R1's column groups; the caption says: "SD 1.5 methods run from their official code, which targets SD 1.x; a number compares only within its table." | `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/main_table.csv`, its rows with `model` SD 1.5 | PRZE-1 | R13 |
| R2 | Figure 5 rebuilt: four pairs, every SDXL method, one adapter, one seed | main, the current Figure 5, full width, caption below | full × 4.4 in, drawn at 5.5 in wide by `1-lora-correction/scripts/paper_tile_figure.py` | rows: cat and dog, elephant and penguin, tent and snowy mountain, lighthouse and stormy sea at the shared seed; columns: joint, PoE (Composable Diffusion), CO3, PoE + Langevin†, ours; labels 8 pt | `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-one-adapter-compose-four-pairs/four-pairs-one-adapter.pdf` | PRZE-3a, PRZE-8, PRZE-1 | itself |
| R3 | the ablation: every SDXL method on the 53 held-out census pairs, pooled | appendix | natural width, centered, at most full × 1.3 in | rows: R1's methods; one column over all 53 pairs: BLIP-VQA ± 95% CI (pairs resampled, then seeds); seeds 9 to 16 | `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/held_out_pooled.csv` | PRZE-1, PRZE-3b | R4 |
| R4 | not in the paper: its rows are part of R12 | appendix | full × 6.8 in, labels rotated | rows: one pair per group at seeds 9 and 10, by rule 5; columns: R3's methods | `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-every-sdxl-method-compares-on-held-out-pairs/methods_side_by_side.png` | PRZE-1 | itself |
| R7 | every pair, and what it was used for | appendix | full × 8.0 in (one page) | one row per pair: pair, set (training, tuning, showcase, held out, unseen), whether the joint prompt composed, whether CO3 composed, kept or dropped | `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-pairs/which-pairs-trained-and-which-were-held-out/pair_list.csv` | PRZE-3a | none; a list, no number |
| R9 | failure types as the adapter's window narrows | appendix | full × 2.0 in | three panels in paper/iclr-56173/figures/failure-window/ (failure types from the fixed-prompt judge, images with two animals, ImageReward); bars: adapter off, steps 0 to 10, 0 to 25, all 50 | `1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/does-restricting-the-adapter-to-early-steps-change-the-failures/panel_a_failure_types.pdf` | PRZE-11 | R10 |
| R10 | the early-steps renders | appendix | full width | rows: the four adapter settings; columns: seven held-out pairs at their seed by rule 5 | `1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/does-restricting-the-adapter-to-early-steps-change-the-failures/early_steps_grid_wide.pdf` | PRZE-11 | itself |
| R11 | three concepts | appendix | full × 4.4 in | columns: four kinds of triple, one each (animal-animal-animal: cat, dog, penguin; animal-object-animal: dog, chair, cat; object-object-object: teapot, lamp, book; animal-animal-object: cat, dog, umbrella) at the shared seed; rows: joint prompt, plain product, ours | `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-the-adapter-compose-three-concepts/three_concepts.png` | PRZE-4 | itself |
| R12 | the renders behind R1 and R3, one prompt per set and one pair per group | appendix | full × 6.8 in, labels rotated | rows: two prompts from each of the three sets at seed 9, by rule 5; columns: R1's methods in R1's order | `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/methods_grid.png` | PRZE-1 | itself |
| R14 | the standard benchmarks on their own prompts | appendix | natural width, centered, at most full × 1.6 in | rows: R1's methods; columns: [GenEval](../../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/05-geneval-a-detector-checks-the-objects.md) two objects, GenEval counting, [T2I-CompBench](../../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/04-t2i-compbench-a-fixed-prompt-set-per-category.md) spatial, T2I-CompBench complex, each by its official scorer; seeds and prompt counts as each benchmark prescribes | `1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/standard_benchmarks.csv` | PRZE-1 | R12 shows R1's renders; R14's renders are filed beside its result file |
| R13 | the renders behind R5 | appendix | full × 3.6 in | rows: R12's six prompts at seed 9; columns: R5's methods in R5's order | `artifacts/results/does-the-correction-compose-on-sd15/sd15_grid.png` | PRZE-1 | itself |

† marks a column that differs on more than one axis (rule 7): the Langevin rows run on the labelled subset, and the five SD 1.5 binding methods keep their own noise when their official code cannot take a given latent.

**R1 and the page limit.** The submitted main text ends on page 9, and R1 adds up to 1.8 in to Section 4.1. The adapter-settings ablation on cat and dog sits in appendix C.5, cited from Section 4.1, which makes the room.

## Before a slot is called finished

From [the Codex presentation review](../../artifacts/drips/answer-iclr-reviews/routes/codex-presentation-review.md) of the stand-ins, 2026-11-10. Each is a property the final page must have.

- **No production text.** Every caption is reader-facing (a bold first sentence, then what is needed to read the result), never "Reserved for"; a submission build fails on a missing slot instead of drawing a grey box.
- **Tables as booktabs.** `\toprule`, `\midrule`, `\bottomrule`, no vertical rules, `@{}` outer padding, the caption above the table; a two-row header with `\cmidrule` over each column group (R1, R5, R3, R14); a "Method" heading on the first column; every abbreviation (Comp., CLIP, BLIP, IR, †) defined in the caption with its scale and direction.
- **Numbers aligned and formatted per metric.** `siunitx` columns, a fixed number of decimals per metric, intervals as `0.62 ± 0.08`; the cited CO3 row labelled "reported" under its own rule, not grey text.
- **Nothing shrinks.** A table wider than the text is an error to fix by grouping or wrapping, never scaled; figure text is at least 7 pt at its printed width, in a serif face that sits with the Times body.
- **One lettering for every figure, old and new.** All figure text, in Figures 1 to 8 as well as the new slots, prints at 8 pt for labels and 9 pt for panel titles at its printed width (never under 7 pt), in a serif face that matches the Times body (Liberation Serif or Nimbus Roman when drawn in Python), with the same names for the same methods. A figure is drawn at its final width with `figsize` in inches equal to its printed width, never drawn large and shrunk.
- **Figures readable as printed.** Labels horizontal and short, colour never the only code (a symbol or pattern beside it, a colour-blind-safe palette), seeds labelled per column, method names identical across R1, R2, R4, R10, R11 and R12.
- **R2 rebuilt, not reused.** The submitted Figure 5 image prints its labels at about 3 pt; the rebuild sets them at 7 pt or typesets them in LaTeX over the image grid, marks CO3's column with †, and resets the wrap length so no empty band is left beside it.
- **Pages checked, not assumed.** A filled table does not keep the placeholder's height, so pagination is reviewed after every fill with `--compile --gallery`, and float pages are packed so a lone figure does not sit centred on an empty page.

## The sketches

**R1: the benchmark, SDXL** (main, centered at natural width; seeds 9 to 12)

```
                     Animals (66)                Animals + objects (144)    Objects (66)
                     Comp.  CLIP   BLIP   IR     CLIP   BLIP    IR          CLIP   BLIP   IR
joint                ..     ..     ..     ..     ..     ..      ..          ..     ..     ..
PoE                  ..     ..     ..     ..     ..     ..      ..          ..     ..     ..
CO3                  ..     ..     ..     ..     ..     ..      ..          ..     ..     ..
PoE + Langevin†      ..     ..     ..     ..     ..     ..      ..          ..     ..     ..
ours                 ..     ..     ..     ..     ..     ..      ..          ..     ..     ..
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CO3, as printed in its Table 1 (cited; seeds and settings differ)      grey row
```

Comp. is the share of pictures showing both animals, by the validated instance count, so it appears for the animals set only.

**R5: the benchmark, SD 1.5** (appendix, R1's columns, centered)

```
                       Animals (66)          Animals + objects (144)   Objects (66)
PoE                    ..                    ..                        ..
PoE + measured corr.   ..                    ..                        ..
Attend-and-Excite      ..                    ..                        ..
SynGen                 ..                    ..                        ..
Divide and Bind        ..                    ..                        ..
InitNO                 ..                    ..                        ..
CONFORM                ..                    ..                        ..
ours (SD 1.5 adapter)  ..                    ..                        ..
```

R1 and R5 read one result file and share columns, but a number compares only within its own table, since the two run on different models (rule 7).

**R2: Figure 5 rebuilt** (main, 0.48 text width, square, text wraps on its left)

```
                 joint prompt   plain product   CO3        ours
cat and dog        [tile]         [tile]        [tile]†    [tile]
elephant/penguin   [tile]         [tile]        [tile]†    [tile]
tent/snowy mtn     [tile]         [tile]        [tile]†    [tile]
lighthouse/sea     [tile]         [tile]        [tile]†    [tile]
                 one adapter, one shared seed; † only if CO3 cannot take the seed's latent
```

**R14: the standard benchmarks on their own prompts** (appendix, centered)

```
                     GenEval                   T2I-CompBench
                     two objects   counting    spatial   complex
joint                ..            ..          ..        ..
PoE                  ..            ..          ..        ..
...                  (R1's other methods)
ours                 ..            ..          ..        ..
```

**R3: the ablation on the 53 held-out pairs** (appendix; seeds 9 to 16)

```
                     Hard (3)        Easy but unseen (29)    Not competing (21)
                     BLIP ±CI        BLIP ±CI                BLIP ±CI
joint                ..              ..                      ..
PoE                  ..              ..                      ..
...                  (R1's other methods, same columns)
ours                 ..              ..                      ..
```

**R4, R12, R13: methods side by side** (appendix; the same template, different rows)

```
              method 1   method 2   method 3   ...   ours
prompt/pair   [tile]     [tile]     [tile]           [tile]
  ...  (rows by rule 5; all tiles in a row share the starting noise unless marked †)
```

**R9: failure types as the window narrows** (appendix)

```
count
  |   ▇                       ▇
  |   ▇ ▅          ▇          ▇ ▅
  |   ▇ ▅ ▃ ▂      ▇ ▅ ▃      ▇ ▅ ▃ ▂ ...
  +---------------------------------------------
     intrusion   count error   style loss   joint-prompt   instability
     bars per group: off · 0-10 · 0-25 · all 50 · all-steps-trained
```

**R7, R10, R11** follow the templates above: R7 is a long table, R10 is R4's grid with R9's five settings as columns, and R11 is R2's grid with three columns.
