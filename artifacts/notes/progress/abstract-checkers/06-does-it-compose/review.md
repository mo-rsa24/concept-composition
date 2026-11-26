# Review: does the count checker's push compose unseen digits (step V3)

**Status: support.** On the 10 pairs among digits 5 to 9, which the count checker never trained on, its push turns the plain product's 0 clean pairs of 320 into 43. The lead is 5.7 times its standard error, and the random push of the same size gives 1. W&B run `qhusod91`. Test 2 does not run.

## The result

| Plain product, digits 5 to 9 | With the count checker's push |
|---|---|
| <a href="unseen-plain.png"><img src="unseen-plain.png" width="420" alt="Ten rows of eight canvases, one row per pair from 5 and 6 to 8 and 9. Every canvas holds one fused glyph; none is framed green. Title: 0 clean pairs of 320"></a> | <a href="unseen-push.png"><img src="unseen-push.png" width="420" alt="The same pairs and seeds with the push. Many canvases now hold two separate digits, one per half, some framed green as clean pairs, others two digits of the wrong identity or one digit. Title: 43 clean pairs of 320"></a> |
| 0 clean pairs of 320; 297 canvases hold one blob | 43 clean pairs of 320; 161 hold two separate digits |

Each sheet shows the first 8 of 32 seeds for each of the 10 pairs, framed green when the judge calls the canvas clean. The same starting seed sits in the same place on both sheets. The numbers are in `/datasets/mmolefe/poe_repair_min/loop-8/v3/test-1/final.json`, from job 94134 on `mscluster79`, which took 2 minutes 55 seconds from the code snapshot of commit `ae80cee6`.

| Row, 320 canvases over 10 unseen pairs | Clean pairs | Two separate digits, any identity | Lead over plain | Standard error | Clears 1.96 standard errors |
|---|---|---|---|---|---|
| Plain product | 0 | 3 | | | |
| Count checker's push, b = 0.4 | **43** | 161 | **43** | 7.58 | **yes** |
| Random push of the same length | 1 | 4 | 1 | 0.96 | no |
| Best of 4 starting seeds, picked by the checker | 0 | 2 | | | |
| Zero push against plain, largest pixel difference | 0.0, bar under 1e-5 | | | | |

The verdict is support by the written rule: the checker's push beats the plain product by more than 1.96 standard errors, the random push does not, and the zero push changes nothing. Picking the best of four plain samples by the checker's own score finds no clean pair, so the gain comes from the gradient during sampling and not from choosing among samples.

## What the push does to a canvas

| Judge's call, 320 canvases | Plain | Push |
|---|---|---|
| Two clear, right digits, one per half | 0 | 43 |
| Two digits, not both clear and right | 3 | 118 |
| Both in the same half | 20 | 3 |
| One blob | 297 | 144 |
| Damaged | 0 | 12 |

The push splits the fused glyph into two digits on half the canvases, and about a quarter of those carry both asked digits clearly. The rest show two digits of which one is the wrong digit, a repeated digit or a stroke too faint for the judge's confidence of 0.9. This matches what the checker reads: where two digits sit, and not which digits they are. The plain product's two labels decide identity, and on these pairs they supply both right digits about one time in four.

Clean pairs by unseen pair: 5 and 6: 9, 5 and 7: 2, 5 and 8: 5, 5 and 9: 2, 6 and 7: 1, 6 and 8: 7, 6 and 9: 6, 7 and 8: 4, 7 and 9: 3, 8 and 9: 4. Every pair gains, and the gain is uneven.

## A rerun on another node

Job 94161 reran the plain and pushed rows on the same pairs and seeds with the same code path, on `mscluster81` in place of `mscluster79`, to record every sampler step for the miniature page (`trace_push.py`, commit `199210af`). The plain row matched exactly: 0 clean pairs on every pair. The pushed row gave 41 clean pairs of 320 in place of 43, and per pair it moved by up to 4 (5 and 6: 5 against 9; 8 and 9: 1 against 4; 5 and 7: 5 against 2). The verdict above stays as recorded from job 94134. The rerun shows that a pushed digit render is not bit-reproducible across nodes, as found earlier for pushed SDXL renders, so per-pair counts carry this much noise beyond the bootstrap error. The rerun's lead of 41 over plain is also more than 1.96 times V3's standard error of 7.58. The record is `/datasets/mmolefe/poe_repair_min/loop-8/v3/trace/trace.json`.

## What the two maps do while the push acts

The same rerun measured, at every sampler step for all 320 canvases, how far apart the two maps' centres of mass sit across the canvas width, and their overlap (the shared share of their mass, 0 to 1).

| Sampler step | 0 | 15 | 20 | 25 | 30 | 40 | 49 |
|---|---|---|---|---|---|---|---|
| Plain: centre separation, pixels | 0.7 | 1.6 | 1.5 | 1.3 | 1.4 | 1.0 | 0.9 |
| Push: centre separation, pixels | 0.7 | 1.6 | **4.1** | **6.6** | 6.4 | 2.6 | 1.1 |
| Plain: overlap | 0.614 | 0.620 | 0.617 | 0.614 | 0.599 | 0.657 | 0.663 |
| Push: overlap | 0.614 | 0.620 | 0.619 | **0.552** | 0.525 | 0.646 | 0.667 |
| Plain: checker logit, mean | −0.21 | −0.23 | −1.26 | −4.14 | −3.66 | −1.33 | −0.15 |
| Push: checker logit, mean | −0.21 | −0.23 | 3.22 | **6.69** | 6.41 | 3.58 | −0.04 |

During the push window (steps 15 to 25), the two maps move apart, from 1.6 to 6.6 pixels on average, while under the plain product they stay within 1.6 pixels of each other. Their overlap falls from 0.62 to 0.55, and the checker's logit rises to 6.69 where the plain product's falls to −4.14. After the window the maps drift back together: by the last step the separation is about 1 pixel in both rows, so the centre of mass of a map at the last steps does not show where the digits ended up (inferred: the late noise predictions differ mostly in stroke detail spread over both digits).

## Choosing the push length

Job 94133 ran the push on the 10 seen pairs among digits 0 to 4 at seeds 0 to 31. Clean pairs of 320 rose with the length: 0 with no push, 8 at b = 0.1, 20 at b = 0.2 and 40 at b = 0.4. The rule froze b = 0.4, the largest of the three choices, so a longer push was not tried.

## The ladder

| Rung | Job | Result |
|---|---|---|
| Smoke test, pair 0 and 1 at 4 seeds | 94115 | failed in the sheet code, which indexed a one-row grid as two-dimensional; fixed in commit `ae80cee6` |
| Smoke test, rerun | 94132 | every row ran; zero push 0.0; the push gave two separate digits on 4 of 4 and no clean pair |
| Choosing b on the seen pairs | 94133 | b = 0.4 |
| The unseen pairs, once | 94134 | support |

<a href="smoke-push.png"><img src="smoke-push.png" width="420" alt="Smoke test sheet with the push: four canvases asked for 0 and 1, each now two separate marks, none framed green"></a>

## What this cannot say

The model here never saw two digits on one canvas, and the push still draws them, so the product's support was enough for a gradient to reach. Step V4 asks the same question on SDXL, whose word maps are read differently and whose test pairs are objects.

The clean count rests on the judge's confidence threshold of 0.9, so a faint but correct digit counts as not clean. The 161 separate-digit count carries no bar and is reported for what it shows about layout.

GPU-hours spent so far: 5.76 of 30 (5.62 through step V2, 0.10 on this step's five jobs and 0.04 on the rerun).

## Cross-references

- The mention of **rerun on another node** in [the context page on the digit-map checkers, one sampler step, section "What is measured where"](../../../../../context/methods/03-the-digit-map-checkers/01-overview.md#what-is-measured-where).
