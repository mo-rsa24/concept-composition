# Why two identity checkers missed on the sampler's own states

Two checkers have missed the 0.75 bar for reading which digits are present on the sampler's own states at steps 27 to 36. V6's checker, trained on 15,000 pasted canvases with noise added, reads them at 0.651. A checker retrained on 2,036 sampler states reads fresh digits 5 to 9 at 0.552. The two misses have different causes. The first checker ranks the states fairly well, but its cutoff sits in the wrong place. The second learned its training pictures. Step V7 is worth less to the papers than its 1.5 GPU-hours, and the lessons carry to V4 as two changes made before V4 runs. Loop 13 asked for this review on 2026-11-24, before the full run that board question 95 approved.

## Contents

- [What each miss is made of](#what-each-miss-is-made-of)
- [What the two share](#what-the-two-share)
- [What the unread papers say](#what-the-unread-papers-say)
- [What other loops found](#what-other-loops-found)
- [Is the gate the right one](#is-the-gate-the-right-one)
- [Is V7 worth its cost](#is-v7-worth-its-cost)
- [What I would change](#what-i-would-change)

## What each miss is made of

All numbers are computed from the saved reads, [V6's sampler read](grid-files/sampler-sampler.json) (172 labelled canvases, 123 clean) and [the retrained read](grid-files/train-read.json) (158 labelled, 108 clean), at the 10 sampler steps 27 to 36 (timesteps 449 to 265).

| | V6's checker | Retrained checker |
|---|---|---|
| Trained on | 15,000 pasted canvases, digits 0 to 3, fresh noise on every draw | 2,036 stored sampler states, digits 0 to 4, no fresh noise |
| Balanced accuracy, cutoff at logit 0 | 0.651 | 0.552 |
| Area under the curve, per step | 0.72 at step 27, rising to 0.83 at step 36 | 0.53 rising to 0.64 |
| Area under the curve, logit averaged over the 10 steps | 0.781 | 0.596 |
| Share of clean canvases called "the asked pair" | 0.68 at step 27, 0.98 to 0.99 from step 32 | 0.78 to 0.85 throughout |
| Share of wrong-digit canvases called "the asked pair" | 0.35 at step 27, 0.73 to 0.78 from step 33 | 0.60 to 0.82 |

**V6's checker ranks, but its cutoff drifts.** Its area under the curve rises through the window, so the later states separate better. Its logits drift upward for both classes, so by step 33 it calls three quarters of wrong-digit canvases "the asked pair". The cutoff at zero was set on pasted canvases, and on sampler states it sits too low.

**The retrained checker learned its pictures.** It reached a training loss of 0.0014. Each of its 550 wrong-digit canvases was drawn about 465 times: 4,000 updates of 64 label-0 examples, over 550 canvases at one of 10 stored steps. No fresh noise was added, and the head has about 483,000 weights. It calls about 80% of every canvas "the asked pair" whatever its label, which is close to a constant answer.

**The base rates differ by pair.** On the fresh read, pair 5 and 6 has 0 wrong-digit canvases against 14 clean, and pair 6 and 7 has 11 against 10. A checker that says "the asked pair" to almost everything scores well on the first and at chance on the second. The pooled balanced accuracy averages over that mix, and each wrong-digit count of about 50 gives a standard error near 0.04.

## What the two share

Both read inputs from a different source than they trained on, or too few inputs to learn from. Pasted canvases with noise added are not what the sampler holds at step 30: the sampler's states come from the product of the two digit labels, and a wrong-digit sampler canvas often holds the same asked digit twice, which pasted wrong-digit canvases never do (see the V6 review, "What this cannot say"). The first checker kept its ranking across that shift and lost its cutoff. The second moved its training onto sampler states but had too few of them, with no fresh noise, so it lost the carry-over to new digits that 15,000 pasted canvases had given.

Whether identity is visible at these steps at all is what question 95's first check measures: the digit judge on the one-step clean estimate of the saved states (job 94717, queued). Its result belongs in the V6 review when it lands.

## What the unread papers say

**[Reduce, Reuse, Recycle](../../score-space/01-reading/reduce-reuse-recycle.md)** (p. 4, Appendix C p. 15). Summing two models' scores gives the score of the product only at $t = 0$; for every $t > 0$ the summed score differs from the score of the noised product. A sampler run on the summed score therefore holds states that are not noised pictures of the product. This is the mechanism behind the first miss: V6's checker learned noised pictures, and the sampler's states are something else at steps 27 to 36.

**[NoisyCLIP](../../noise-text-critic/01-reading/noisyclip.md)** (Fig. 5, p. 10). It trains its critic on the sampler's own latents within a window, and the window matters: trained on latents 20 to 29, it scores 0.709 at latents 21 to 30; trained on 40 to 49, it scores 0.688 there and 0.747 at 41 to 49. Every window beats frozen CLIP at its own steps. Training on the sampler's states works there, with a far larger dataset than 2,036 canvases; the note records no count small enough to compare.

**[The Noise-Aware Latent Reward Model](../../noise-text-critic/01-reading/noise-aware-latent-reward.md)** (p. 4 to 6, 16 to 17). It scores noisy latents to $t = 950$ by passing the true timestep through a U-Net copy, trained with a ranking loss on pairs noised to the same $t$. Two things carry over. Its loss is a ranking loss, so its zero point never matters. And it never takes a gradient of its score: it ranks four candidates and picks one.

## What other loops found

**Loop 6 sees the same pattern on SDXL.** On CO3, its checker's area under the curve sits near 0.70 from step 5 on, while its balanced accuracy at the fixed cutoff is lower (`method-search/01b-balance-the-checker-data/review.md`, lines 33, 52 and 54). Ranking survives the move to the sampler's states, and the cutoff does not.

**Loop 9 found the opposite direction on SDXL.** Its readout checker read SDXL's sampler states better than noised pictures: 0.696 against 0.637 over steps 0 to 15 (`noise-text-critic/05-verdict.md`, line 23). On digits the shift costs accuracy; on SDXL's features it did not. So the digit miss does not predict V4's.

**The digit failures differ from SDXL's.** A digit wrong canvas is a whole other digit or the same digit twice. CO3's misses are borrowed parts between two bodies that are both present (`score-space/12-independent-audit.md`, line 113). V4 is judged on those.

## Is the gate the right one

V7's push is $-\lVert\epsilon\rVert\, b\, g/\lVert g\rVert$, with $g$ the gradient of the checker's logit. Adding a constant to the logit leaves $g$ unchanged, so the push depends on how the checker ranks states and never on where its cutoff sits. The area under the curve measures ranking; balanced accuracy at zero measures ranking and cutoff together. By that reading, V6's checker, at 0.72 to 0.83, is closer to usable for a push than its 0.651 says.

The gate stays as written. Swapping it now, after the result, would be choosing the bar that passes. If the owner wants ranking to gate a push in later steps, it has to be registered before any of those steps' results exist, and V4 is the next such step.

## Is V7 worth its cost

Probably not. V7 tests whether removing one checker removes only its property, on digits. Since board question 86, the SDXL test is one identity checker against CO3, with no removal test, so V7's answer no longer feeds a Paper 2 experiment. The full question-95 run costs about 1.6 GPU-hours and, if it passes, V7 about 1.5 more. The cheap visibility check (job 94717) is worth finishing: if the judge cannot read identity on the clean estimate at these steps, no checker can, and that closes the question.

## What I would change

1. **Finish only the visibility check, then close V7 as not run**, unless the owner wants the digit removal result for its own sake. The record keeps all three readings: 0.861 on pasted digits, 0.651 on sampler states with an area under the curve of 0.78, and 0.552 after retraining.
2. **For V4, register the area under the curve beside the 0.75 bar before step 2 runs**, and say which one gates. This needs the owner's decision, since V4's bar is fixed.
3. **For V4, add fresh noise and many more examples if the checker trains on CO3's states**, and report how the batch mixes finished pictures and states. The retrained digit checker shows what 2,036 stored states without fresh noise do.
4. **For V4, report base rates per pair** beside the pooled score, because CO3's two-clear count varies by pair (8 of 12 pairs make no faithful picture in M1).
