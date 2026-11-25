# Review: two experts trained on separate colours (step M10)

This file holds what the runs in [the M10 design](design.md) produced. The renders are under way, and no model has trained.

## Runs

Outputs are on the cluster under `/datasets/mmolefe/poe_repair_min/loop-6/m7/`, and logs are in `/datasets/mmolefe/poe_repair_min/loop-6/logs/`.

| Rung | Slurm jobs | Commits | Output | State |
|---|---|---|---|---|
| GPU rendering test, 8 pictures on one RTX 3090 | 94448 | `8932ce01-15d8bdd` | none | stopped: Blender 2.78c asks for NVIDIA's compiler to build its GPU code, and the cluster has none |
| Render smoke, 8 pictures each of r1, r3, b3, c21 and c33 | 94548 to 94552 | `e60212bf-15d8bdd` | `s2-smoke/renders/` | passed, 4.71 to 5.77 seconds per picture on CPU |
| Expert renders, 5,000 per set, red chain r1 to r3 and blue chain b1 to b3 | 94585 to 94590 | `e60212bf-15d8bdd` | `s2/renders/` | done: 5,000 of 5,000 in each of r1, r2, r3, b1, b2 and b3, none missing, about 6.5 to 7 hours per set on one CPU node |
| Labelled datasets for the two experts: each colour's three sets in one zip labelled by count (r1 as 0, r2 as 1, r3 as 2), plus the 1% and overfit subsets | 95420 | `395c2fb4-15d8bdd` | `m10/data/` | done in 12 minutes 35 seconds: `r-full.zip` and `b-full.zip` hold 15,000 pictures each (5,000 per count, labels 0 to 2), `r-one-percent` and `b-one-percent` 150 each, `r-overfit-18` 18; the packer passed a 12-picture test on the cluster first |
| Overfit: the red expert on 18 pictures (6 per count), 256,000 images seen at batch 256, count labels on, 16 samples per count | 95442 | `29db2cb6-15d8bdd` | `m10/overfit/models/r-overfit-18/` | queued on biggpu |
| 1% of the data: each expert on 150 pictures (50 per count), 65,536 images seen at batch 256 | 95443, 95444 | `29db2cb6-15d8bdd` | `m10/one-percent/models/` | cancelled before starting; moved to RunPod |
| Overfit on a RunPod RTX 4090: the red expert on 18 pictures, 256,000 images seen, batch 256 with 128 per pass | pod `47.47.180.34` | `m10_pod/run_ladder.sh` | pod `runs/overfit/r-overfit-18/` | passed in 18 minutes 27 seconds (4.28 seconds per thousand images), final loss −5.29: by eye at double size every one of the 16 samples per count holds the asked count of red objects, against a bar of 15 of 16; the colour counter reads 0.71 of them right, close to its 0.82 on the training pictures. A first try at 256 per pass ran out of the card's 24 GB |
| 1% of the data on RunPod, red and blue | pods `47.47.180.34`, `209.170.80.132` | `m10_pod/run_ladder.sh` | pod `runs/one-percent/` | passed: the loss falls from 0.01 to −2.89 (red) and from 0.21 to −2.96 (blue) over 65,536 images, about 4.7 minutes each |
| Full runs on RunPod, first try | the same pods | `bc2554da` | pod `runs/full/` | failed at start: EDM2 accepts a snapshot interval only in multiples of 1,024 images, and a quarter of 2,000,896 is 500,224 |
| Full runs on RunPod, red and blue, 2,000,896 images, snapshots at 500,224, 1,000,448, 1,500,672 and 2,000,896 | the same pods | `6150cd42` | pod `runs/full-quarters/`, copied back to `m10/full/models/` | training since 13:23 pod time, about 2.4 hours |
| Scorer renders, 250 per set, every combination of 0 to 3 red with 0 to 3 blue, c00 to c33 | 94765 to 94780 | `e60212bf-15d8bdd` | `s2/renders/` | done: 250 of 250 in each of the 16 sets, 18 to 23 minutes per set |

## The render smoke

Every one of the 40 pictures holds exactly the colours and counts asked for, read from its scene file and checked by eye. Objects sit anywhere
on the floor, and in the scenes of 3 red with 3 blue some touch or partly hide one another.

<a href="smoke-renders-by-colour-and-count.png"><img src="smoke-renders-by-colour-and-count.png" width="520" alt="Five rows of 8 CLEVR renders: one red object, three red, three blue, two red with one blue, and three red with three blue, each labelled with its counts"></a>

## The full runs, quarter by quarter

Each quarter is sampled on the pod's CPU from the 0.100 EMA snapshot, 16 seeds per count, and read by the colour counter against its share on
real training pictures of that colour (red 0.82, blue 0.86). The stopping rule in [the design](design.md#watching-the-full-runs-and-when-to-stop-one)
is read at 75%. The records are in each pod's `runs/full-quarters/<colour>-full/quarters/quarters.jsonl`.

| Quarter | Images seen | Red: share read right (1, 2, 3 objects) | Red: mean loss | Blue: share read right (1, 2, 3 objects) | Blue: mean loss |
|---|---:|---|---:|---|---:|
| 1 | 500,224 | 0.81 (1.00, 0.88, 0.56) | −3.37 | 0.83 (0.94, 0.94, 0.63) | −3.36 |
| 2 | 1,000,448 | 0.85 (1.00, 0.88, 0.69) | −3.83 | 0.90 (1.00, 1.00, 0.69) | −3.84 |
| 3 | 1,500,672 | 0.85 (1.00, 0.81, 0.75) | −3.96 | 0.94 (1.00, 1.00, 0.81) | −3.96 |
| 4, the end | 2,000,896 | 0.83 (1.00, 0.88, 0.62) | −4.02 | 0.90 (1.00, 1.00, 0.69) | −4.02 |

At the first quarter both experts draw their colour on the CLEVR floor, and the counter reads them about as often as it reads the real
pictures. Three objects is the weak count, where some samples hold two. At the second quarter every number rose: both shares reach the counter's share on real pictures, three objects rises to 0.69 on both, and the loss falls by about 0.47 on both. **The stopping rule, read at the third quarter, stops neither run.** Red's share holds at 0.85, not above its value at the second quarter, but
far above its stop line of 0.62; blue's rises to 0.94 against a stop line of 0.65; both losses fall again, to −3.96. Both runs go on to the end. At the end both shares sit within the 0.07 that 48 samples move by chance of their third-quarter values, and both losses fell to −4.02.
Both runs' final snapshots (0.050 and 0.100), stats, logs, samples and quarter records are in `/datasets/mmolefe/poe_repair_min/loop-6/m10/full/models/r-full/` and `b-full/`, each snapshot's sha256 matching the pod's copy. The first try at sampling recorded no samples (the sampler is a
generator and was never iterated); that line stays in the file with a share that is not a number.

## The comparison

The settings are in [the design's comparison section](design.md#running-the-comparison). Every GPU row runs on the red expert's RunPod RTX
4090; the scorer trains on the cluster's CPUs.

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Count scorer, first run: overfit on 16 renders, then 40 epochs at a fixed learning rate of 0.001 | 95528 | `bbe809f0-15d8bdd` | `m10/scorer/` | overfit passed, 16 of 16; training missed: 0.518 of the 800 held-out renders counted exactly on the last epoch, against 0.95. The held-out share swung between epochs (0.95 at epoch 29, 0.59 at 19, 0.52 at 39), with red counts of 2 and 3 collapsing on the last; W&B `sxv8ojso` |
| Count scorer, second run: 60 epochs with the rate decaying to zero by a cosine, the last epoch kept, nothing chosen on the held-out renders | 95532 | the commit after `b1ef3752` | `m10/scorer-cosine/` | missed by 3 renders: 0.946 counted exactly against 0.95 (red right 0.973, blue 0.974); the misses sit in the crowded sets, 0.82 at 3 red and 3 blue, 0.84 at 2 and 3, 0.86 at 3 and 2; W&B `acr12hqz` |
| Every model on the pod | | | pod `models/` | blue's and the empty-scene snapshots copied with sha256 matching their sources |
| Sampler check: the comparison's own Heun loop against EDM2's `generate_images` on 4 seeds per count | | `m10_pod/m10_compose.py` | pod `sources/` | the largest difference is 1 grey level of 255 on every seed |
| Plain product, smoke: 8 seeds per cell | | | pod `rows/plain/` | runs; where both experts draw at the same place their colours add, so objects come out dark purple-black, not one red and one blue |
| Plain product, block 0: seeds 0 to 255 in all 9 cells | | | pod `rows/plain/` | 2,304 pictures in 536 seconds, 0.23 seconds a picture |
| Plain product, blocks 1 to 3 | | | pod `rows/plain/` | done, 536 seconds a block; the row holds 9,216 pictures |
| Count checker's training: 3,000 updates of 64 pastes, noise drawn from the push window (σ 3.8 to 10.5) | | `38807290` | pod `checker/` | missed: held-out balanced accuracy 0.645 against 0.75. It calls apart pastes with the right counts right on 0.82, touching pastes on 0.57 and apart pastes asked one count off on 0.37; from update 500 on it sits between 0.636 and 0.646, so it learns one fixed split and no more. It cleared the stop line at half (0.642 above 0.55). The checker row does not run on it |
| Mask paste, blocks 0 to 3 | | `99107ca4` | pod `rows/paste/` | done, 9,216 pictures in 28 seconds on the CPU |
| Diagnosis, after the checker missed: the same training for 1,000 updates at four noise windows, σ 3.8 to 10.5, 1.2 to 3.8, 0.3 to 1.2 and 0.002 to 0.3 | | `99107ca4` | pod `checker-diag/` | done; no window reaches 0.75 (the table below) |
| Corrector row, blocks 0 to 3 | | `99107ca4` | pod `rows/corrector/` | done, 9,216 pictures; 2,067 seconds for block 0 while it shared the card with the diagnosis, then 1,061 to 1,221 seconds a block |
| Every output of the comparison copied back | | | `/datasets/mmolefe/poe_repair_min/loop-6/m10/comparison/m10-comparison.tar` | 209 MB: every row's pictures, the paste sources, the experts-alone samples, the checker and its diagnosis, the quarter reports and the logs; sha256 `0372a324…` matching on the pod and the cluster |

**The checker's noise-window diagnosis.** A diagnosis after the checker missed its bar, not a result: the same training cut to 1,000 updates,
held-out balanced accuracy on the same 1,024 held-out pastes.

| Noise window (σ) | Balanced accuracy, bar 0.75 | Apart, right counts | Touching | Apart, one count off |
|---|---:|---:|---:|---:|
| 3.8 to 10.5, the push window | 0.645 | 0.81 | 0.58 | 0.38 |
| 1.2 to 3.8 | 0.669 | 0.70 | 0.77 | 0.51 |
| 0.3 to 1.2 | 0.685 | 0.65 | 0.82 | 0.62 |
| 0.002 to 0.3, nearly clean | 0.741 | 0.77 | 0.86 | 0.58 |

Less noise lets the checker see touching objects (0.58 to 0.86), and it never reads a count one off: 0.58 even on a nearly clean paste.
So noise is not the main cause. The two maps, each expert's denoised estimate minus the empty scene's, carry little of how many objects a
paste holds, and at σ below 0.3 the layout is already fixed, so a push there cannot move an object.

**The first two quarters, scored provisionally.** Scored with the second-run scorer (0.946 on the held-out renders, under its 0.95 bar), so
these numbers track the run and decide nothing until the owner rules on the scorer. Exact share averaged over the 9 cells:

| Row | Seeds 0 to 255 | Seeds 256 to 511 | 1 red + 1 blue | Every cell of 3 or more objects |
|---|---:|---:|---|---|
| Plain product | 0.026 | 0.033 | 0.20, 0.27 | 0.00 to 0.02 |
| Corrector | 0.028 | 0.033 | 0.24, 0.28 | 0.00 to 0.01 |
| Mask paste | 0.537 | 0.505 | 0.80, 0.75 | 0.27 to 0.66, falling with the count |

The plain product and the corrector both draw dark purple-black objects, so even their 1 red + 1 blue share is suspect: the scorer never
saw an object that is neither red nor blue. The 32-per-cell eye calls will say how often those are truly one red and one blue.

**All four quarters, scored provisionally** (the 0.946 scorer; nothing here is a verdict until the owner rules on the scorer, question 114).

| Row | Seeds 0 to 255 | 256 to 511 | 512 to 767 | 768 to 1,023 | All 9,216 | 1 red + 1 blue | 2 + 2 | 3 + 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain product | 0.026 | 0.033 | 0.026 | 0.030 | 0.029 | 0.23 | 0.00 | 0.00 |
| Corrector | 0.028 | 0.033 | 0.040 | 0.030 | 0.033 | 0.28 | 0.00 | 0.00 |
| Mask paste | 0.537 | 0.505 | 0.523 | 0.496 | 0.515 | 0.77 | 0.40 | 0.33 |

The corrector moves the plain product by 0.004, under the 5 points the bars ask for. The mask paste leads both by about 0.48. Owed before
any verdict: the owner's answers on the scorer (question 114) and the checker row (question 116), the experts-alone check scored, and the
32-per-cell eye calls.

**The eye calls.** Seeds 0 to 31 of every cell in the three rows, 864 pictures, were shuffled under neutral names onto 54 sheets of 16 at
224 pixels and counted by two blind reviewers in opposite orders: red, blue and other-coloured objects, and whether any red object touches
a blue one. A picture is exact when it holds the asked red and blue, nothing else, and no red touching blue. The two agree on that verdict
for 864 of 864 pictures, so no third call was needed, and on all three counts for 796.

| Row | Exact by eye, 288 pictures | 1 red + 1 blue | Pictures holding an object of another colour | Red touching blue |
|---|---:|---:|---:|---:|
| Plain product | 0.000 | 0.00 | 288 | 0 |
| Corrector | 0.000 | 0.00 | 288 | 0 |
| Mask paste | 0.205 | 0.59 | 0 | 206 |

The eye and the scorer part ways on the mask paste: about 0.5 by the scorer, 0.205 by eye. In most pastes the blue object lies over the red
one and leaves a red sliver beside it (seeds 2, 11, 25 and 30 at 1 red + 1 blue), which the reviewers call touching and the scorer reads as
one red and one blue: it trained on real renders, where the generator never lets objects overlap, so it cannot judge "all separate" on a
paste. Every plain and corrector picture holds a dark object that is neither red nor blue, so their scorer share near 0.03 is its error on
objects it never saw. The scorer's call on each of the same 864 pictures (job 95689, the second-run scorer) disagrees with the eye on 127 of 864, 0.147:
111 of 288 mask pastes (0.385; the scorer calls 0.583 of them exact against the eye's 0.205), 10 of 288 plain-product pictures and 6 of
288 corrector pictures. That is more than the 1 in 10 the design allows, so **the stage is inconclusive on the scorer**, whatever a retrained
scorer would reach on real renders: the disagreement comes from overlap that real renders never show.

**The second checker, which reads the noisy picture** (board question 116, written into the design at `cbfdc79a` before it trained). It
trained on a new RTX 4090 pod rebuilt from the verified copies (the same environment, each model's sha256 matching) and **missed its bar**:
held-out balanced accuracy 0.641 against 0.75 after 3,000 updates (216 seconds), between 0.60 and 0.65 from update 250 on. It calls apart
pastes with the right counts right on 0.68, touching on 0.72 and one count off on 0.49. Its row is reported as not run. The result and logs
are in `/datasets/mmolefe/poe_repair_min/loop-6/m10/comparison/m10-part-c.tar` (sha256 `7e778f50…`).

## The verdict

**Inconclusive, by the bars written before the first run.** The count scorer misses its 95% bar (0.946 on real renders), and it disagrees
with the blind eye calls on 127 of 864 pictures, more than 1 in 10. Following the owner's answer to board question 114, the rows are read on
the eye calls.

| Row | Exact share by eye, 288 pictures | Where it breaks: the largest total whose every cell holds at least 50% |
|---|---:|---|
| Plain product | 0.000 | below 2 objects |
| Corrector | 0.000 | below 2 objects |
| Count checker on the experts' maps | not run: missed its bar at every noise window (best 0.741) | |
| Count checker on the noisy picture | not run: missed its bar (0.641) | |
| Mask paste | 0.205 | 2 objects (1 red + 1 blue at 0.59; every 3-object cell at 0.34 or below) |

**What it says.** Two colour experts trained apart, multiplied together, draw no correct scene at any count from 2 to 6: wherever both
draw an object at the same place the colours add into a dark object that is neither, and two Langevin steps a sampler step do not change
that. Pasting one expert's objects over the other's gets one scene in five, nearly all at 2 objects, and fails by overlap as the count
rises. Neither checker could learn to tell a right count from one off by one, on the maps or on the picture, so this test holds no
measurement of Paper 1's method. What a method would have to supply here is a signal that keeps the two experts from drawing at the same
place; the paste's failures say where it must beat pasting: scenes whose objects overlap.

**Each expert alone passes.** Seeds 0 to 31 per count from each expert, 192 pictures shuffled onto 12 blind sheets and counted by two
reviewers in opposite orders, who agree on all 192. Holding exactly k objects of the expert's colour and none of another: red 32, 31 and 29
of 32 at 1, 2 and 3 objects; blue 32, 31 and 31 of 32. Every count clears 0.90, red at 3 objects by one picture (0.906). So the rows'
failures come from composing the experts, not from the experts.
