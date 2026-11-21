# Review: bodies and species pushes on every step

The questions and pass marks for [plan 31](../31-bodies-and-species-pushes-on-every-step.md), written on
2026-11-18 before any picture was drawn.

## Words this file uses

**Bodies push.** The readout checker of iteration 6 (`readout-catdog-identity`, step 2,000) at strength 30; it reads
SDXL's own features, and its push moves where the animals are.

**Species push.** Either the CLIP species checker (`finetune/species-clip/best.pt`, two outputs, a clear cat present
and a clear dog present, read on the decoded guess; at step 15 "dog present" 0.847 and "cat present" 0.711), or the
presence-only readout checker (`readout-catdog-presence-only`, two outputs, a cat present and a dog present, read on
SDXL's own features). Its push follows the sum of the two log-probabilities.

**The summed push.** Bodies push plus species push on the same steps, all 50 here, the sum capped at 0.3 of PoE's
prediction. One trajectory per seed from the seed's own starting noise, DDIM eta 0.

**Separate.** Two animals in the frame, not fused into one body. **Clean or near-clean.** A cat and a dog, each
recognisably one species, with at most one small borrowed feature. Calls by eye from full-size tiles on the ladder
one animal < fused < two bodies < near-clean < clean, as in [plan 29's review](29-does-the-early-push-hold-on-new-seeds.md).

## Questions

**Q1. Does the summed push on every step give more clean or near-clean cat-and-dog pictures than the recipe?**

On seeds 3301 to 3332, rendered once for the setting picked on seeds 9 to 16, against the recipe's 28 separate and
14 clean or near-clean (9 clean) on the same seeds:

- Support: clean or near-clean on at least 20 of 32, with at least 26 separate.
- No change: clean or near-clean within 4 of the recipe's 14 (10 to 18).
- Worse: separate below 24, or clean or near-clean below 10.
- Plain PoE and the size-matched random push are drawn on the same sheet; plan 29 gave 1 separate and 0 clean for each.

The comparison with the recipe is mixed: the window (all 50 steps against 0 to 15) and the species push both
differ. The seeds 9 to 16 sheet carries the bodies push alone on all 50 steps, so the species push's own share is
read there, on 8 seeds.

**How the setting is picked.** On seeds 9 to 16 only: the bodies push alone, plus the CLIP species push at strength
10, at strength 30, and plus the presence-only readout checker at strength 30, all on 50 steps. Claude picks the
setting with the most clean or near-clean tiles by eye, writes the pick and its reason below, and stops for the
user's yes. Seeds 3301 to 3332 are not used to choose anything.

## Runs

| Run | What | Status |
| --- | --- | --- |
| identity check | seed 9, species strength 0, steps 0 to 15, against the recipe's tile | done: plain PoE, best-of-2 and strength 0 byte-identical; the pushed tile differs, as the recipe's own pushed tile differs between two runs (mean pixel difference 19 to 22 of 255), so a pushed render is not bit-reproducible and weight 0 matches the recipe as closely as the recipe matches itself; 253 s for two one-seed sheets |
| result-sheet-catdog-every-step-bodies | bodies push alone, steps 0 to 50, seeds 9 to 16 | done: 8 separate, 3 clean or near-clean; 28 s a pushed seed |
| result-sheet-catdog-every-step-species10 | plus CLIP species push at strength 10 | done: fused on most seeds; 80 s a pushed seed |
| result-sheet-catdog-every-step-species30 | plus CLIP species push at strength 30 | done: fused on most seeds |
| result-sheet-catdog-every-step-presence | plus presence-only readout checker at strength 30 | done: 8 separate, 6 clean or near-clean; 20 s a pushed seed |
| result-sheet-catdog-every-step-general | the picked setting on seeds 3301 to 3332 | done: `plan31-general.sh` on mscluster112, 1,585 s for the sheet (18 s a pushed seed, 20 s a best-of-2 pair) |

## The pick on seeds 9 to 16

**Picked: the bodies push plus the presence-only readout checker, all 50 steps.** Calls by eye at 640 px
(`result-sheet-catdog-every-step-presence/calls-seeds-9-to-16.csv`):

| Setting | Separate | Clean or near-clean | Clean |
| --- | --- | --- | --- |
| Bodies push alone | 8 of 8 | 3 of 8 | 0 |
| + CLIP species, strength 10 | 1 of 8 | 0 | 0 |
| + CLIP species, strength 30 | 4 of 8 | 0 | 0 |
| + presence-only readout checker | 8 of 8 | 6 of 8 | 3 |

The CLIP species push pulls the two animals back into one cat-dog head on most seeds (tile-size look; neither
row is a candidate). The presence checker keeps both animals apart on every seed and names them more often: a
clear cat and a clear dog on seeds 11, 14 and 15. Both 50-step rows streak or over-saturate the background on
several seeds. Eight seeds carry run-to-run noise (a pushed seed can render a different picture on a re-run), so
the gap of 3 is a reason to pick, and seeds 3301 to 3332 judge it.

## Answers

**Q1: no change.** The picked setting (bodies push plus the presence-only readout checker, all 50 steps) on seeds
3301 to 3332, called by eye at 600 px (`result-sheet-catdog-every-step-general/calls.csv`):

| | Plain PoE | Random push, same size | The recipe (plan 29) | The picked setting |
| --- | --- | --- | --- | --- |
| Separate | 1 of 32 | 2 of 32 | 28 of 32 | 29 of 32 |
| Clean or near-clean | 0 of 32 | 0 of 32 | 14 of 32 (9 clean) | 15 of 32 (8 clean) |

15 is within 4 of the recipe's 14, so the no-change line is met; the support line (20 or more) is not. Plain PoE and
the random push are scorer counts (two bodies 1 and 2, faithful pair 0 and 0). The scorer, stricter than the eye,
gives the picked setting 30 two bodies and 2 faithful pairs and calls 14 of 32 damaged: pushing on all 50 steps
burns the colour and turns backgrounds and some animals into flat paint (3306, 3307, 3318). The misses are mostly two
dogs (3303, 3314, 3315, 3330) or an animal beside a yellow toy-like object (3327, 3331, 3332). The comparison with
the recipe is mixed (the window and the species push both differ), and the seeds 9 to 16 sheet showed the bodies push
alone on 50 steps at 3 of 8 against 6 of 8 with the presence checker, so the presence push's own share is real on
those 8 seeds and does not survive on 32.
