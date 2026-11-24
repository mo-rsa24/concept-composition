# Review: remove one checker at a time, on digits (step V5)

**Status: open.** The three checkers are trained and the push length is calibrated. The eleven rows on the unseen pairs have not run, so step V5's proof is not met.

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
| The eleven rows on the unseen pairs, once | | not run |

GPU-hours spent so far: about 6.8 of 30 (5.76 through step V3, 0.15 on the smoke test and about 0.9 on the trainings and calibration).
