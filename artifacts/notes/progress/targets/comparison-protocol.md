# How a comparison against a rival is judged

This page says what has to be true before our result counts as beating a rival, on the numbers and on the pictures. The
targets it applies to are on [the scoreboard](../scoreboard.json), and each one traces to a printed page through
[the numbers table](numbers.csv).

## Words this uses

**Target.** The best rival number on a benchmark, moved one step past it. FID 4.11 becomes 3.11, 91.10% becomes 92.10%, and
a 0 to 1 score moves by 0.01. The drawing script computes it, so nobody types it.

**Benchmark.** One dataset, subset, task, metric and protocol. A score from another protocol is a different benchmark.

**Call.** One judgement on one prompt and seed: `better`, `tied` or `worse`.

## Numbers

**A number counts when the run matches the rival's protocol.** That means the same dataset, split, metric, sample count and
settings the rival's page states. Where the page leaves a setting out, the run records the value it used, and the row says
the protocol is only partly matched.

**The rival is rerun where it can be.** Step D2 reproduces one published number per rival. When a reproduction lands, the
scoreboard uses the harder of the printed and reproduced numbers.

**A win inside the rival's printed uncertainty reads "within noise".** It does not count as beating the rival.

**T2I-CompBench needs a rerun before any win is claimed.** Plain SDXL scores 0.6369 on Color in ToMe's Table 1 and 0.5879 in
CoMat's Table 1, and CoMat does not state its scorer, prompt split or seeds. A gap of one step between our run and a printed
number could come from the setup alone. Plain SDXL, ToMe and CoMat are therefore run on our setup, and our win is read
against those runs.

**Four benchmarks are scored on the prompts the subject guard allows.** [The guard](../../../../scripts/shared/disallowed_subjects.py)
keeps 12 of 1,000 prompts out of T2I-CompBench Complex, 7 of 40 out of RareBench Concat, 12 of 300 out of Color and 7 of 300
out of Shape. On these four, the rival and our method run on the same allowed prompts and seeds, and every number prints its
prompt count, such as "988 of 1,000 prompts". A win needs our score to beat both the rival's rerun on those prompts and the
printed target. A row with no rerun of that rival on the same prompts claims no win. The rerun stays out of the scoreboard's
`reproduced` field, because its sample count differs from the published one.

**GenEval two objects is scored on its released 553 prompts.** FKC released no list of its 1,000 prompts. On the 553, FKC prints
0.78 and plain SDXL prints 0.80 (FKC Table A4, p39), so a win must beat the 0.79 target and also beat plain SDXL's 0.80 on the same
prompts and seeds.

**A win must clear the noise of the rerun.** Our score must beat the rival's rerun by more than 1.96 standard errors on matched
pictures, meaning the same prompt and seed for both methods, with prompts resampled. Rectify Then Diffuse against plain SDXL gives
standard errors of 0.0153 on T2I-CompBench Complex and 0.0983 on RareBench Concat, so the margin is about 0.03 and 0.19 there. Each
row computes its own margin from its own differences. A rival marked `blocked` has no rerun, so its row is read against the printed
target and says the rival was not rerun.

**A disputed number cannot set a target.** `limits.md` marks where a paper contradicts itself about a value. T2I-CompBench
Color is a disagreement between two papers: ToMe quotes CoMat at 0.7774, and CoMat's own page prints 0.7827. The target uses
the number on CoMat's own page.

## Pictures

A side-by-side call is made only when both methods ran on the same prompts and seeds.

1. Render both methods on the same prompt and seed, with the same image size.
2. Save the two pictures under neutral names, so the method cannot be read from the file.
3. Open each pair at full resolution, because downscaled sheets make fused creatures look like two clear ones.
4. Place each picture on the ladder: one thing, a fused hybrid, two bodies or heads, two clear things.
5. Write the call and its reason for the pair: `better` when ours sits higher on the ladder, `worse` when lower, and `tied`
   when both sit on the same rung with no visible difference in faithfulness.
6. Only then reveal which picture came from which method.

Each call is one row in [the side-by-side table](side-by-side.csv), with the columns `paper`, `dataset`, `rival`, `pair`,
`seed`, `call` and `reason`. The scoreboard's `pairs`, `better`, `tied` and `worse` counts must match that table.

## What the targets do not establish

The rivals' figures are chosen by their authors, so [the failure record](failures.md) shows that a failure happens and says
nothing about how often. A rival with no printed failure on a benchmark has an untested failure rate there.

Beating a target on a benchmark says the method wins on that protocol. It says nothing about a dataset we did not run.
