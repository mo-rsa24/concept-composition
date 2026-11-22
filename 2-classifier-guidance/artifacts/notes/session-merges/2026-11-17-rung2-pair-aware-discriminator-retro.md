# Retrospective: rung 2, a small pair-aware discriminator on SDXL's cat × dog product

The `drip-execute-plan --request` walk over route step 10 ([plan 21](../../../plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md)),
and its follow-up [plan 22](../../../plans/22-rung2-a-discriminator-trained-on-the-benchmark-pairs.md),
on 2026-11-17 from 08:41 to about 18:20 UTC. Built from the walk map, the commits and the run outputs.

## 1.1: the timing

**Request.** Time 4 plain and 4 guided renders on one card; set N for best-of-N.

**Problem.** The three cards the request named were busy with plan 20.

**Done.** Timed on `mscluster107` GPU 0 (Quadro RTX 8000) with plan 18's dog classifier standing in.

**Results.** Plain 27.5 s, guided 42.2 s (medians of 4), so N = 2; about 2.1 GPU-hours for the rung.
`/datasets/.../rung2/compute/timing.json`.

**Links to.** Sets the card and N used by every later piece.

## 1.2 and 2.2: the training set

**Request.** 10 to 15 checked renders of a cat and a dog apart from seeds 1 to 8, lone animals and the
product as negatives.

**Problem.** 9 positives on the seeds' own noise, under the 10 asked for; some lone-animal tiles came
out as drawings or as one cat repeated across the scene.

**Done.** 16 fresh-noise pair draws and 9 replacement negatives, called in a `/loop`; a rule that no
tile repeating one concept enters training.

**Results.** 73 tiles, 17 positives, 24 intact negatives; 23 pair-prompt failures listed for a later
comparison. `kept-positives.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/training-set/strips/kept-positives.png`

**Links to.** The data for 1.3.

## 1.3: the first discriminator

**Request.** Train on noisy latents; check it before it pushes.

**Problem.** 11 million adapter weights against 41 tiles would memorise; the planned check measured fit.

**Done.** Frozen SDXL trunk with plan 18's attention-pool head; the check held out seeds 7 and 8.

**Results.** Passed 13 of 20 sampler steps (0.80 to 1.00 from step 7, 0.59 to 0.73 before).

**Links to.** The push in 1.4.

## 1.4 and 2.4: the result sheet

**Request.** Seeds 1 to 4 and 9 to 12: plain, best-of-N, guided and a same-sized random push at 10, 30, 100.

**Problem.** A session restart killed the queued launch; it was relaunched by hand.

**Done.** 64 tiles on one card; strength 0 identical to plain; random-to-guided norm 1.0000 every step.

**Results.** Draft calls: 0 of 4 composed on seeds 9 to 12 in every column; the cap fired on 100% of
steps at s = 100; draft reading 7. `result-sheet.png` is not shown because it failed the subject check

**Links to.** The diagnostics, which explain it.

## Diagnostics

**Request.** The reader asked whether the discriminator looks for what we want.

**Done.** Filmstrips with attention and push maps, scores through time, every tile kind scored per step.

**Results.** It scored pair-prompt renders positive whatever they showed and looked at one fixed patch.
<a href="../../results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/diagnostics.png"><img src="../../results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/diagnostics.png" width="200" alt="tile kinds through time"></a>

**Links to.** The retrains.

## v2 and plan 22's v3: retrains with hard negatives

**Request.** Pair-prompt failures as negatives; then more data from the benchmark's pairs, with and
without adapters.

**Problem.** `/datasets` went down from about 13:10 UTC; the benchmark's cached noise could not be
regenerated, so its pictures were taken from the laptop's local copies.

**Done.** v2 on 64 tiles; v3 on 400 tiles (81 positives over 38 pairs) called by two independent
passes and settled at full size; held-out check by whole pair.

**Results.** v2, v3 head only and v3 with adapters: 0 of 20 steps each, held-out accuracy 0.53 to 0.69.
<a href="../../results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/discriminator.png"><img src="../../results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/discriminator.png" width="200" alt="held-out accuracy of all four discriminators"></a>

**Links to.** The next rung needs a discriminator that reads the concept.

## The stop point

- **Bar:** one of the seven readings printed with its counts, judged by the reader from full-resolution tiles.
- **Time box:** 3 sittings, about 4.5 hours; used about 9.5 hours in one long sitting.
- **Which came first:** the time box; the reader chose to go on past it (the cap-0.3 rerun, v2, plan 22).
- **Way on taken:** stop here and write up, with a stronger discriminator planned next.

## Decisions

| Decision | Chosen | Options | Who | Why |
|---|---|---|---|---|
| card | `mscluster107` GPU 0 | share 111 with plan 20; wait | reader (defaults) | plan 20 held the named cards |
| discriminator body | frozen trunk, head only | adapters too; network from scratch | reader ("both") | 41 tiles would be memorised |
| check | held out seeds 7 and 8 | fresh noise on training tiles | reader ("both") | the planned check measured fit |
| more positives | fresh noise under seeds 5 to 8 | re-render the same seeds; CO3 | reader ("agree") | a seed's noise repeats its picture |
| after the sheet | cap-0.3 rerun and v2 retrain | either alone | reader ("both") | the diagnostics pointed at the checker |
| during the outage | run v2 from a scratch folder; stop re-rendering the benchmark | wait | reader | re-rendering duplicated judged pictures |
| plan 22's data | local pictures, two-things pairs only | animals only; include places | reader ("go") | more positives; places teach another idea |
| calls | two independent passes, disagreements settled at full size | one pass | reader ("check all") | one pass let hybrids through |

## Dead ends and not done

- The 64 result-tile calls are Claude's drafts; the reader has not confirmed them (A3q's check line open).
- The prescribed cap-0.3 rerun never ran: it needs the first discriminator's weights on `/datasets`,
  which stayed down past 18:00 UTC.
- 16 of 48 benchmark pairs re-rendered into the scratch folder, then stopped as duplication; unused.
- Plan 20's jobs on `mscluster108`, `111` and `112` were gone by 15:30 UTC, most likely killed by the
  outage; that walk belongs to another session.

## The family table

| Id | Kind | Serves | State |
|---|---|---|---|
| this walk | drip-execute-plan session | plans 21 and 22 | closing |
| PIDs 713135, 715633, 727709, 730734 | runs on `mscluster107` | timing, training set, draws | done |
| PIDs 739713, 754977, 768102 | runs on `mscluster107` | discriminator, result sheet, diagnostics | done |
| PID 780883 | v2 chain | v2 discriminator | stopped at its failed check |
| Slurm 89657 to 89659 | bigbatch workers | benchmark re-renders | cancelled |
| eight helper agents | calls on the local pictures | plan 22, 1.2 | done |

---

Validation: every picture linked above was checked on disk before writing; numbers come from
`timing.json`, `calls.csv`, `result.json`, the four `gate.json` files and `calls-final.csv`.
