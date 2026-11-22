# CFGRevisited first read notes

PDF: arXiv 2608.16786v1, 13 pages. Page numbers are PDF pages.

## Numbers that could not be read

None.

## Curves that are plot-only

None. Figures 1 to 5 are image grids. The numbers on Figures 3 to 5 are guidance scales of the qualitative sweep, and no result value is printed.

## How the rows are labelled

**Every row has kind `baseline`.** The paper benchmarks no-CFG, CFG and eight guidance methods, and none of them is one of the 19 rivals. The paper has no method of its own.

**The Table 1 margins have their own 10 rows.** The paper defines each margin as the 95th percentile of pooled method-minus-CFG errors, so it applies to a difference from CFG and not to a single score (p2). Those rows name `guided methods minus CFG` in the `paper` column, and the `interval` cell of the score rows stays empty.

**Tables 3 and 4 hold settings only** (p8), so they are marked `no results`.

## Protocol facts a comparison depends on

**Models.** SD3.5 Medium uses 25 steps and FLUX.2 [klein] 4B Base uses 30 steps (p2, p8). Each method's guidance settings differ by model (Tables 3 and 4, p8).

**Benchmarks.** GenEval, DPG-Bench and three OneIG-Bench families (General Object, Text Rendering, Knowledge Reasoning), each on its full prompt set (p2). Style-oriented OneIG families were not generated (p2).

**Held fixed.** Within each model, methods share prompts, seeds, scheduler, resolution and sampling budget (p2).

**Parameter choice.** Settings were chosen by eye on a small fixed prompt set before the benchmarks, and no benchmark score was used for selection (p2).

**Uncertainty.** The margins come from 10,000 paired prompt-bootstrap replicates (p2). Tables 5 to 9 carry no margins, and their colours do not denote statistical significance (p9).

**Scale.** DPG is mean x 100 (p3, p10). GE, Obj, Txt and Rsn are printed on a 0 to 1 scale.

**Not stated.** The resolution value, seed count, images per prompt and prompt counts are not stated.

## Hardware and time

Inference used V100 32GB GPUs (p2). Attention-perturbation methods add one extra forward pass per prediction (p8). No runtime is printed.

## Contradictions inside the paper

None found. GenEval Overall equals the mean of its six categories within rounding on all 20 method rows. The GE and DPG columns of Table 1 equal Tables 5 and 6. The L1 Global column of Table 6 equals the L2 global column of Tables 8 and 9. Every signed difference in Table 2 equals the method's score minus CFG's 0.655.
