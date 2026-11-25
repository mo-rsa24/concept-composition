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
