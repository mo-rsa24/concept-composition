# Layout sheet: every new figure and table in the review response

One row per figure and table the revision of submission 56173 adds or rebuilds, planned before its result exists. Each slot has a fixed size in the ICLR layout, a placeholder at that size in `paper/iclr-56173/iclr2027_conference.tex`, a row in [the figure register](../../paper/iclr-56173/figures.md), and one result file. `scripts/refresh_paper_slots.py` rebuilds a slot from its result file, so an early result can be placed and replaced without moving a page. Every slot obeys [the comparison protocol](protocol.md).

ICLR text width is 5.5 in; "full" below means 5.5 in. The smallest text in any figure is 7 pt at printed size, and tables are set in `\footnotesize`, which this paper's style file defines as 9 pt on 10 pt leading (`iclr2027_conference.sty` line 175). Result files live under `artifacts/results/<the question the work asked>/`, and every table's CSV carries rule 9's twelve axis columns (`model`, `method`, `adapter`, `sampler`, `steps`, `guidance`, `resolution`, `pair`, `seed`, `group`, `setting`, `nfe`) beside its scores.

## The slots

| Slot | What | Where | Width × height | Rows, columns, panel order | Result file | Board row | Grid it ships with (rule 8) |
|---|---|---|---|---|---|---|---|
| R1 | the benchmark: every SDXL method on the 276 Attend-and-Excite prompts | main, Section 4.1 | natural width, centered, at most full × 1.8 in | rows: joint, PoE, CO3, [ToMe](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/08-tome-fusing-an-object-with-its-attributes.md), SuperDiff, PoE + Langevin†, ours, then CO3's printed Table 1 as a grey cited row; column groups: animals (66) with Comp., CLIP, BLIP, IR; animals with objects (144) with CLIP, BLIP, IR; objects (66) with CLIP, BLIP, IR; then NFE; seeds 9 to 12; the showcase prompt "a cat and a dog" marked | `artifacts/results/how-the-adapter-scores-against-other-methods/main_table.csv`, its rows with `model` SDXL, `group` naming the set | PRZE-1 | R12 |
| R5 | the benchmark on SD 1.5, each method's official code on its own model | appendix | natural width, centered, at most full × 1.8 in | rows: PoE, PoE + measured correction, [Attend-and-Excite](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/06-attend-and-excite-bringing-a-missing-subject-back.md), [SynGen](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/09-syngen-binding-by-the-prompts-grammar.md), [Divide and Bind](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/10-divide-and-bind-attendance-plus-binding.md), [InitNO](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/11-initno-choosing-the-starting-noise.md), [CONFORM](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/07-conform-pulling-look-alike-subjects-apart.md), ours (SD 1.5 adapter, filled only if route step 12 trains it, otherwise dropped and the limit stated); R1's column groups; the caption says: "SD 1.5 methods run from their official code, which targets SD 1.x; a number compares only within its table." | `artifacts/results/how-the-adapter-scores-against-other-methods/main_table.csv`, its rows with `model` SD 1.5 | PRZE-1 | R13 |
| R2 | Figure 5 rebuilt: four pairs, every SDXL method, one adapter, one seed | main, the current Figure 5, full width, caption below | full × 3.1 in, drawn at 5.5 in wide by `scripts/paper_tile_figure.py` | rows: cat and dog, elephant and penguin, tent and snowy mountain, lighthouse and stormy sea at the shared seed; columns: joint, PoE (Composable Diffusion), CO3†, ToMe, SuperDiff, PoE + Langevin†, ours; labels 8 pt | `artifacts/results/four-pairs-one-adapter/four-pairs-one-adapter.pdf` | PRZE-3a, PRZE-8, PRZE-1 | itself |
| R3 | the ablation: every SDXL method on the 53 held-out census pairs, by group | appendix | natural width, centered, at most full × 1.3 in | rows: R1's methods; column groups: hard (3), easy but unseen (29), not competing (21), each with Comp. ± 95% CI; seeds 9 to 16 | `artifacts/results/how-the-adapter-scores-against-other-methods/held_out_ablation.csv` | PRZE-1, PRZE-3b | R4 |
| R4 | methods side by side on held-out pairs, the renders behind R3 | appendix | full × 5.0 in | rows: one pair per group at seeds 9 and 10, by rule 5; columns: R3's methods | `artifacts/results/co3-protocol-run-by-us/methods_side_by_side.png` | PRZE-1 | itself |
| R6 | the samplers that correct the product | appendix | full × 3.9 in | rows: the four Figure 5 pairs at the shared seed; columns: joint, joint + Langevin, PoE, PoE + Langevin, PoE + unadjusted HMC, CO3 | `artifacts/results/does-a-langevin-corrector-fix-the-product/six_columns.png` | PRZE-2 | itself |
| R7 | every pair, and what it was used for | appendix | full × 8.0 in (one page) | one row per pair: pair, set (training, tuning, showcase, held out, unseen), whether the joint prompt composed, whether CO3 composed, kept or dropped | `artifacts/results/which-pairs-trained-and-which-were-held-out/pair_list.csv` | PRZE-3a | none; a list, no number |
| R9 | failure types as the adapter's window narrows | appendix | full × 2.0 in | one bar group per failure type (intrusion, count error, style loss, joint-prompt corruption, instability); bars: adapter off, steps 0 to 10, 0 to 25, all 50, the all-steps-trained adapter | `artifacts/results/does-restricting-the-adapter-to-early-steps-change-the-failures/failure_counts.pdf` | PRZE-11 | R10 |
| R10 | the early-steps renders | appendix | full × 4.6 in | rows: pairs and seeds by rule 5; columns: the five settings of R9 | `artifacts/results/does-restricting-the-adapter-to-early-steps-change-the-failures/early_steps_grid.png` | PRZE-11 | itself |
| R11 | three concepts | appendix | full × 3.0 in | rows: 4 concept triples at the shared seed; columns: joint prompt, plain product, ours | `artifacts/results/does-the-adapter-compose-three-concepts/three_concepts.png` | PRZE-4 | itself |
| R12 | the renders behind R1, two prompts per set | appendix | full × 5.2 in | rows: two prompts from each of the three sets at seed 9, by rule 5; columns: R1's methods in R1's order | `artifacts/results/how-the-adapter-scores-against-other-methods/methods_grid.png` | PRZE-1 | itself |
| R14 | the standard benchmarks on their own prompts | appendix | natural width, centered, at most full × 1.6 in | rows: R1's methods; columns: [GenEval](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/05-geneval-a-detector-checks-the-objects.md) two objects, GenEval counting, [T2I-CompBench](../../../../goal-setting/learning/deep-learning/diffusion-models/12-scoring-and-rival-methods/plans/04-t2i-compbench-a-fixed-prompt-set-per-category.md) spatial, T2I-CompBench complex, each by its official scorer; seeds and prompt counts as each benchmark prescribes | `artifacts/results/how-the-adapter-scores-against-other-methods/standard_benchmarks.csv` | PRZE-1 | R12 shows R1's renders; R14's renders are filed beside its result file |
| R13 | the renders behind R5 | appendix | full × 3.6 in | rows: R12's six prompts at seed 9; columns: R5's methods in R5's order | `artifacts/results/does-the-correction-compose-on-sd15/sd15_grid.png` | PRZE-1 | itself |

† marks a column that differs on more than one axis (rule 7): the Langevin rows run on the labelled subset, and the five SD 1.5 binding methods keep their own noise when their official code cannot take a given latent.

**R1 and the page limit.** The submitted main text ends on page 9, and R1 adds up to 1.8 in to Section 4.1. The adapter-settings ablation on cat and dog sits in appendix C.5, cited from Section 4.1, which makes the room.

## Before a slot is called finished

From [the Codex presentation review](../../artifacts/drips/answer-iclr-reviews/routes/codex-presentation-review.md) of the stand-ins, 2026-11-10. Each is a property the final page must have.

- **No production text.** Every caption is reader-facing (a bold first sentence, then what is needed to read the result), never "Reserved for"; a submission build fails on a missing slot instead of drawing a grey box.
- **Tables as booktabs.** `\toprule`, `\midrule`, `\bottomrule`, no vertical rules, `@{}` outer padding, the caption above the table; a two-row header with `\cmidrule` over each column group (R1, R5, R3, R14); a "Method" heading on the first column; every abbreviation (Comp., CLIP, BLIP, IR, NFE, †) defined in the caption with its scale and direction.
- **Numbers aligned and formatted per metric.** `siunitx` columns, a fixed number of decimals per metric, intervals as `0.62 ± 0.08`, NFE as a plain integer; the cited CO3 row labelled "reported" under its own rule, not grey text.
- **Nothing shrinks.** A table wider than the text is an error to fix by grouping or wrapping, never scaled; figure text is at least 7 pt at its printed width, in a serif face that sits with the Times body.
- **One lettering for every figure, old and new.** All figure text, in Figures 1 to 8 as well as the new slots, prints at 8 pt for labels and 9 pt for panel titles at its printed width (never under 7 pt), in a serif face that matches the Times body (Liberation Serif or Nimbus Roman when drawn in Python), with the same names for the same methods. A figure is drawn at its final width with `figsize` in inches equal to its printed width, never drawn large and shrunk.
- **Figures readable as printed.** Labels horizontal and short, colour never the only code (a symbol or pattern beside it, a colour-blind-safe palette), seeds labelled per column, method names identical across R1, R2, R4, R10, R11 and R12.
- **R2 rebuilt, not reused.** The submitted Figure 5 image prints its labels at about 3 pt; the rebuild sets them at 7 pt or typesets them in LaTeX over the image grid, marks CO3's column with †, and resets the wrap length so no empty band is left beside it.
- **Pages checked, not assumed.** A filled table does not keep the placeholder's height, so pagination is reviewed after every fill with `--compile --gallery`, and float pages are packed so a lone figure does not sit centred on an empty page.

## The sketches

**R1: the benchmark, SDXL** (main, centered at natural width; seeds 9 to 12)

```
                     Animals (66)                Animals + objects (144)    Objects (66)          NFE
                     Comp.  CLIP   BLIP   IR     CLIP   BLIP    IR          CLIP   BLIP   IR
joint                ..     ..     ..     ..     ..     ..      ..          ..     ..     ..     ..
PoE                  ..     ..     ..     ..     ..     ..      ..          ..     ..     ..     ..
CO3                  ..     ..     ..     ..     ..     ..      ..          ..     ..     ..     ..
ToMe                 ..     ..     ..     ..     ..     ..      ..          ..     ..     ..     ..
SuperDiff            ..     ..     ..     ..     ..     ..      ..          ..     ..     ..     ..
PoE + Langevin†      ..     ..     ..     ..     ..     ..      ..          ..     ..     ..     ..
ours                 ..     ..     ..     ..     ..     ..      ..          ..     ..     ..     ..
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CO3, as printed in its Table 1 (cited; seeds and settings differ)      grey row
```

Comp. is the share of pictures showing both animals, by the validated instance count, so it appears for the animals set only.

**R5: the benchmark, SD 1.5** (appendix, R1's columns, centered)

```
                       Animals (66)          Animals + objects (144)   Objects (66)       NFE
PoE                    ..                    ..                        ..                 ..
PoE + measured corr.   ..                    ..                        ..                 ..
Attend-and-Excite      ..                    ..                        ..                 ..
SynGen                 ..                    ..                        ..                 ..
Divide and Bind        ..                    ..                        ..                 ..
InitNO                 ..                    ..                        ..                 ..
CONFORM                ..                    ..                        ..                 ..
ours (SD 1.5 adapter)  ..                    ..                        ..                 ..
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
                     Comp. ±CI       Comp. ±CI               Comp. ±CI
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

**R6: the samplers that correct the product** (appendix)

```
             joint   joint+Lang.   PoE   PoE+Lang.   PoE+uHMC   CO3
pair 1       [ ]     [ ]           [ ]   [ ]         [ ]        [ ]
pair 2       [ ]     [ ]           [ ]   [ ]         [ ]        [ ]
pair 3       [ ]     [ ]           [ ]   [ ]         [ ]        [ ]
pair 4       [ ]     [ ]           [ ]   [ ]         [ ]        [ ]
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
