# Review: remove one checker at a time, on digits (step V5)

**Status: inconclusive.** The eleven rows ran on the unseen pairs (W&B run `fsbuivcs`). By the written rules the verdict is inconclusive: the all-three-checkers row is void, and several cells the verdict needs hold fewer than 40 canvases. Board question 73 asks the owner whether to rerun with more seeds.

## The removal table

Job 94172 on `mscluster60` took 7 minutes 18 seconds and wrote `/datasets/mmolefe/poe_repair_min/loop-8/v5/final.json`: 320 canvases per row on the 10 pairs among digits 5 to 9 at seeds 1000 to 1031, each checker's push length 0.229. Each column after the first is scored only on canvases that passed the column before it; the canvas counts are in brackets.

| Row | Count: two digits apart | Right digits, of those | Order, of those | Fully right, of all | Total push, mean share | Pushed steps over 0.6 |
|---|---|---|---|---|---|---|
| Plain product | 0.9% (3) | 0.0% (0) | none | 0.0% | 0 | 0% |
| All three | 41.6% (133) | 40.6% (54) | 88.9% (48) | 15.0% | 0.408 | **5.85%, void** |
| Minus count | **14.1%** (45) | 40.0% (18) | 88.9% (16) | 5.0% | 0.322 | 0% |
| Minus count, random in its place | **10.9%** (35) | 45.7% (16) | 93.8% (15) | 4.7% | 0.399 | 0% |
| Count alone | 33.8% (108) | 31.5% (34) | 52.9% (18) | 5.6% | 0.229 | 0% |
| Minus right digits | 37.5% (120) | **34.2%** (41) | 92.7% (38) | 11.9% | 0.331 | 0% |
| Minus right digits, random in its place | 34.7% (111) | **41.4%** (46) | 95.7% (44) | 13.8% | 0.406 | 0% |
| Right digits alone | 5.9% (19) | 31.6% (6) | 83.3% (5) | 1.6% | 0.229 | 0% |
| Minus order | 44.1% (141) | 44.0% (62) | **45.2%** (28) | 8.8% | 0.319 | 0% |
| Minus order, random in its place | 46.9% (150) | 32.7% (49) | **53.1%** (26) | 8.1% | 0.395 | 0% |
| Order alone | 11.9% (38) | 28.9% (11) | 72.7% (8) | 2.5% | 0.229 | 0% |

**Why the verdict is inconclusive.** The design voids a row whose total push passes 0.6 of the product's norm on more than 5% of pushed steps, and the all-three row does so on 5.85%. The three unit gradients sum to 0.408 of the norm on average, near the 0.4 the calibration aimed at, and occasionally line up enough to pass 0.6. The design also needs at least 40 canvases in every cell its verdict reads; removing count leaves 45 canvases that pass count, 18 with the right digits and 16 in the order column, and removing order leaves 28 in its own column.

**Read without a verdict.** These descriptions carry no claim, because the rules above void them. Removing count lowers the count column from 41.6% to 14.1%, a random push in its place gives 10.9%, and the right-digits and order columns hold at 40.0% and 88.9%. Removing order lowers the order column from 88.9% to 45.2%, a random push gives 53.1%, and count holds at 44.1%. Removing right digits lowers its column only from 40.6% to 34.2%, and the random replacement leaves it at 41.4%, which fits a checker reading near chance (0.556). On this reading, two of the three checkers each control their own property, and the one that cannot read its property does not.

## The three checkers

Job 94153 on `mscluster79` trained the right-digits and order checkers on all 10 pairs among digits 0 to 4 at the 16 timesteps matching SDXL's noise, from the code snapshot of commit `9242d9e9`, and saved them to `/datasets/mmolefe/poe_repair_min/loop-8/v5/`. The count checker is V2's. Scores are balanced accuracy over the 16 steps, where 0.5 is chance.

| Checker | Overfit check, of 16 | Digits 0 to 4 | Digits 5 to 9, never trained on | Source |
|---|---|---|---|---|
| Count | 16 | 0.782 | 0.791 | V2, W&B `tyhrpyov` |
| Right digits | 16 | 0.601 | **0.556** | `v5/train.json` |
| Order | 16 | 0.798 | **0.711** | `v5/train.json` |

The right-digits checker reads near chance, as V2's identity checker did (0.555 on the pairs holding digit 4). The order checker reads the left-right order of two digits on digits it never saw at 0.711, even though telling which digit is which also asks for identity (inferred: on an `apart` canvas, the two maps already sit in opposite halves, so the order only asks which map sits on the left).

## The push length

Job 94154 measured the three checkers' gradients on the 10 seen pairs at seeds 0 to 31 over the pushed steps: 10,560 pairwise cosines with a mean of 0.009. The gradients point in nearly unrelated directions, so the design's rule gives $b = 0.4/\sqrt{3(1 + 2 \times 0.009)} = 0.229$ for each checker. The zero push changed no pixel. The numbers are in `v5/calibrate.json`.

## The ladder

| Rung | Job | Result |
|---|---|---|
| Smoke test, every stage on pair 0 and 1 at 4 seeds with 50-update checkers | 94137 | every row ran, zero push 0.0, no row void |
| The two trainings | 94153 | as the table above |
| Calibration | 94154 | b = 0.229 |
| The eleven rows on the unseen pairs, once | 94172 | inconclusive: the all-three row is void and cells hold under 40 canvases |

GPU-hours spent so far: about 6.9 of 30 (5.76 through step V3, 0.15 on the smoke test, about 0.9 on the trainings and calibration, and 0.12 on the eleven rows).
