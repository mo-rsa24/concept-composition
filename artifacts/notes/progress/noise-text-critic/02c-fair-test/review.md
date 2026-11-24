# Review: is W2's null fair, and is the setup readable at all? (step W2c)

**State: W2 is inconclusive by the bars fixed beforehand, and no sentence score reads in any fair setting.**
- **The control.** The readout checker on SDXL's features is the positive control. On the clean-pair task it read a mean of 0.637 over steps 0 to 15, just under its floor of 0.65, so the rule calls W2 inconclusive and not null. It does beat frozen CLIP's sentence score by 0.150, with a standard error of 0.035, so the information is there at a level the sentence scores never reach.
- **The sentence scores.** No sentence row reaches 0.75 over steps 0 to 15, in any model, plain or ensembled, on W2's pictures or on the sampler's real states. The best is Latent CLIP's 0.574, on the real states.

The [design](design.md) holds the groups and the bars, and the numbers are in [the result file](result.json), from W&B run `etbecql0`.

## Result

Each cell is mean balanced accuracy on the reading half of the seeds, with the threshold chosen on the other half. 0.5 is chance, and the reading bar is 0.75. "0 to 15" is the push's window. "20 to 29" is NoisyCLIP's training window. "Step 49" is almost the finished picture.

**Group 2, W2's 220 pictures with noise added, four draws per step.**

| Task | Reader | 0 to 15 | 20 to 29 | Step 49 |
|---|---|---|---|---|
| Clean pair (positive against every reject) | readout checker (the control) | **0.637** | 0.801 | 0.74 |
| | NoisyCLIP, plain / ensembled | 0.506 / 0.503 | 0.687 / 0.711 | 0.86 / 0.89 |
| | frozen CLIP, plain / ensembled | 0.488 / 0.486 | 0.513 / 0.523 | 0.92 / 0.91 |
| | Latent CLIP, plain / ensembled | 0.556 / 0.535 | 0.528 / 0.546 | 0.62 / 0.60 |
| Counting (two animals against one or a hybrid) | readout checker | 0.646 | 0.704 | 0.65 |
| | NoisyCLIP, plain / ensembled | 0.514 / 0.521 | 0.540 / 0.516 | 0.71 / 0.65 |
| | frozen CLIP | 0.496 | 0.474 | 0.67 |
| | Latent CLIP, plain / ensembled | 0.561 / 0.522 | 0.499 / 0.514 | 0.54 / 0.53 |
| Prompt match (cat and dog against another pair) | readout checker | 0.600 | 0.792 | 0.73 |
| | NoisyCLIP, plain / ensembled | 0.508 / 0.526 | 0.711 / 0.734 | 0.82 / 0.82 |
| | frozen CLIP | 0.487 | 0.492 | 0.80 |
| | Latent CLIP, plain / ensembled | 0.478 / 0.502 | 0.572 / 0.498 | 0.70 / 0.68 |

The readout checker beats frozen CLIP by more than 1.96 standard errors on every task. Its gaps are 0.150 on clean pair, 0.150 on counting and 0.114 on prompt match. These comparisons are mixed, because the reader and its input both change.

**Group 3, the sampler's real states.** These are 195 held-out cat and dog joint runs, 105 clean, read once per step with no noise added.

| Reader | 0 to 15 | Step 0 | Step 5 | Step 10 | Step 15 | Step 20 |
|---|---|---|---|---|---|---|
| readout checker | 0.696 | 0.50 | 0.63 | 0.73 | 0.87 | 0.87 |
| NoisyCLIP, plain / ensembled | 0.479 / 0.473 | 0.42 | 0.45 | 0.55 | 0.50 | 0.57 |
| frozen CLIP | 0.480 | 0.46 | 0.49 | 0.46 | 0.44 | 0.50 |
| Latent CLIP, plain / ensembled | 0.574 / 0.540 | 0.54 | 0.53 | 0.59 | 0.59 | 0.52 |

On the real states, the readout checker reaches 0.87 by step 15 and beats frozen CLIP by 0.215, with a standard error of 0.042. No sentence row reaches 0.64 at any single step from 0 to 20. The highest is Latent CLIP's 0.63, at step 8.

**What the fair settings changed.**
- **Prompt ensembling** moved no sentence row by more than 0.04 over steps 0 to 15.
- **NoisyCLIP's own window, steps 20 to 29.** It reads prompt match at 0.711 (0.734 ensembled) and the clean pair at 0.687 (0.711), so even where it was trained it stays under 0.75.
- **The real trajectory states** read no better than noise added to finished pictures.
- **The finished-picture check, step 49.** Every sentence model passes the clean pair, and NoisyCLIP and frozen CLIP pass prompt match. Counting fails for all of them (0.71, 0.67 and 0.54), as W2 found. The species-free counting sentences, still to run, test whether the wording causes that.

**Group 4, counting with sentences that name no species.** The right sentence is "a photo of two animals", against "a photo of one animal" and "a photo of one animal with mixed features". It is scored on W2's saved image vectors, so the image side is exactly W2's. The run is job 94210, and its numbers are in `loop-9/w2c/species-free/result.json`.

| Reader | 0 to 15 | Step 49 | W2's sentences at step 49, for comparison |
|---|---|---|---|
| NoisyCLIP, plain / ensembled | 0.553 / 0.522 | 0.517 / 0.729 | 0.71 / 0.65 |
| frozen CLIP, plain / ensembled | 0.477 / 0.498 | 0.704 / **0.750** | 0.67 / not read |
| Latent CLIP, plain / ensembled | 0.503 / 0.499 | 0.538 / 0.538 | 0.54 / 0.53 |

Dropping the species words fixes neither result. No row reads over steps 0 to 15. On the finished picture only frozen CLIP with ensembled wordings reaches 0.75, exactly at the bar, and plain NoisyCLIP falls from 0.71 to 0.517. So the plurality failure on finished pictures belongs to the models' image side as much as to W2's wording. "Two cats" lost to "one animal" half the time in W2, and it still does without species words.

**What this means for W3 (read by the owner, not decided here).** Board question 70 says to stop W3 if W2c leaves the null standing. The rule gives a third outcome, inconclusive, because the control finished 0.013 under its floor. No sentence row's mean over steps 0 to 15 reaches 0.6 in any setting. The highest single step on the reading half is 0.67, Latent CLIP's counting score at step 10 on W2's pictures. The control shows SDXL's own features read these pictures at steps 5 to 15 better than any sentence score does.

## Runs

| Job | Stage | Code | Outcome |
|---|---|---|---|
| 94113 | smoke test: 24 pictures, steps 0 and 15, 8 trajectory runs | `be6a9e0f` | Ran in 81 seconds on mscluster63. Every row scored in both groups and the readout checker loaded. Its numbers come from 24 pictures and carry no verdict |
| 94114 | full run | `be6a9e0f` | First attempt: mscluster63 failed at 07:09 UTC, 65 minutes in, with group A finished and group B at step 3 of 21. The script writes its results only at the end, so nothing was kept. Slurm requeued the job, and it restarted on mscluster49 at about 07:31 UTC. At that hour 31 `bigbatch` nodes were down and 5 drained |
| 94131 | species-free counting rescore (group 4) | `5294ca7c` | Cancelled before it started, to be resubmitted after W2c |
