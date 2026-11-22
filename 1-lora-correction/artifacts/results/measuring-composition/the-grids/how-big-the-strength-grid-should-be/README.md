# How big the strength grid should be

The question: Figure 3 of the manuscript looks like it wastes space, and its tiles are hard to
read. What does its generator actually produce, and how much bigger can the tiles be?

## strength-grid-as-it-is-in-the-paper.png

What the manuscript carries today, at `figures/how-much-is-added/strength-grid-0to1.pdf`,
included at `width=0.62\textwidth`. Native 3.44 x 2.49 in, tiles about 0.51 in.

**Where it came from.** It is a crop, not a render. The September 10 output of the generator was
cut down to the columns at λ ≤ 1 during the section 3 proofread walk on 2026-09-22. Cropping
removed columns but not the type, so the labels were oversized for what remained and the figure
had to be set at 0.62 of the text width to keep them in proportion. That is what shrank the
tiles and left 38% of the text line empty beside it.

## strength-grid-fitted-grid-only.png (and .pdf)

The same content re-rendered from the generator at full text width. 5.50 x 4.10 in, tiles
0.88 in, which is 2.9 times the area of the cropped version. Type at its designed 7 to 9 pt.

**Where it came from.** `scripts/compose_rate_vs_correction.py --layout grid5`, run on
2026-09-23 against the dose outputs pulled from
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose` (cat × dog, seed 9), with two
changes to the script that are not yet committed:

- `PANEL` fitted to the canvas. The script sets `GUTTER 0.92 + 5 × PANEL 0.92 + RIGHT_MARGIN
  0.18 = 5.70 in` against a `FIG_W` of 5.5, so the grid overran its own canvas by 0.2 in and
  the note "count = animals detected; filled = composed" was cut off mid-word. `PANEL` becomes
  `(5.5 - 0.92 - 0.18) / 5 = 0.88`.
- The compose-rate curve panel removed, and the λ values drawn as column titles instead. In the
  generator those labels are the curve's x-axis ticks, so removing the curve removes them.

## strength-grid-fitted-with-curve.png (and .pdf)

The same grid with the compose-rate curve underneath, as the generator draws it. 5.50 x 5.79 in,
tiles 0.88 in. The curve is what the crop discarded: the real correction reaches about 94%
compose rate at λ = 1 while all three controls stay near zero across the axis.

**Where it came from.** The same run, with only the `PANEL` fix applied.

## strength-grid-six-columns.png (and .pdf, .svg)

The full-width grid with λ = 0.25 dropped and λ = 0.6 and 0.65 added, so the columns read 0,
0.5, 0.6, 0.65, 0.75, 1. 5.50 in wide, six tiles of 0.91 in, row names in one line under the
grid, no count chips. At 0.6 and 0.65 the real-correction row already shows a cat and a dog
side by side, one step earlier than the five-column grid can show, while the wrong-pair,
wrong-seed and wrong-order rows stay single fused animals.

**Where it came from.** The eight new cells (two strengths × four rows, cat × dog, seed 9) were
rendered on 2026-09-26 by Slurm job 60368 on mscluster106, calling
`scripts/interaction_term_inject.py --pair a_cat__x__a_dog --seed 9 --lambda {0.6,0.65} --row
{oracle,wrong_pair,wrong_seed,wrong_step}` into the dose sweep's own folders
(`outputs/interaction_term/dose/pairs/a_cat__x__a_dog/seed_9/teacher_residual_const_lam06{0,5}*`),
the same sampler and 20 steps as the existing columns. They are not scored: `dose_curves.json`
was not rerun, so no detector count exists for them. The figure is a scratchpad copy of
`scripts/compose_rate_vs_correction.py` with the changes in `the-six-columns-changes.diff`: a
`--lambdas` column list, six panels at 5.5 in, and no count lookup. The SVG is `pdftocairo -svg`
of the PDF, so text and lines are vector and the 24 tiles are embedded rasters.

## What this is for

Deciding what Figure 3 of the ICLR submission should be. The three sit side by side so the
trade is visible: 2.49 in of column for unreadable tiles, 4.10 in for readable ones, or 5.79 in
for readable tiles plus the rate curve. The decision and the walk that produced it are in
[the section 3 proofread map](../../drips/proofread-section-3/PROOF_MAP.md).

<!-- cluster:start -->

## On the cluster

The other 37 files of this result (33.4 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 37 | 33.4 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/<run>/ <run>/
```

<!-- cluster:end -->
