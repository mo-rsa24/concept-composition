# 📒 Report: can a second push make each body the right animal?

**No experiment here raised the clean cat-and-dog count past the early push alone.** The
[early push](does-an-early-push-separate-the-cat-and-dog.md) makes two bodies on most seeds, and the
usual miss is a borrowed face or two of one kind. Plans 26, 27, 28 and 31 each added something meant
to decide which body becomes the cat and which the dog.

## Table of contents

- [Four attempts and their verdicts](#four-attempts-and-their-verdicts)
- [Pushing for the species on every step](#pushing-for-the-species-on-every-step)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Concepts this uses](#concepts-this-uses)
- [Where this came from](#where-this-came-from)
- [Still open](#still-open)

## Four attempts and their verdicts

Navigation: 📋 [TOC](#table-of-contents) | [Next](#pushing-for-the-species-on-every-step) ➡️

**Each plan wrote its pass mark before drawing, and none reached support on clean pairs.** Seeds are
held-out seeds 9 to 16. Calls are by eye and are drafts unless the row says full size `stated` (each
plan's review file).

| Folder | What changed | Pass mark | Recorded result | Verdict |
|---|---|---|---|---|
| `result-sheet-catdog-species` and plan 26's species checker | a CLIP checker for "clear cat" and "clear dog" | both at least 0.80 at step 15; null below 0.70 | cat 0.711, dog 0.847 | [inconclusive](../../plans/review/26-two-pushes-bodies-early-species-late.md#answers-so-far) |
| `result-sheet-catdog-bodies`, `result-sheet-catdog-both` | readout push on steps 0 to 9, species push on 10 to 49 | at least 2 more "two clear" seeds than the first push alone, with the random push adding none | almost the same pictures; one cat-dog face on most seeds at strengths 30 and 100 | null |
| `not-hybrid`, three strengths | subtract SDXL's direction for "a hybrid animal", weighted by the checker | a separate cat and dog on at least as many seeds as the cap 0.3 push with fewer damaged, and on at least 2 more than plain product | none on any seed; style turns to line drawings and patterns | [null](../../plans/review/27-cat-and-dog-and-not-hybrid.md#answers) |
| `not-hybrid`, fixed-weight control | the same term with the checker's weight fixed at 19 | the checker's weight gives at least 2 more seeds | the review records the same pictures, with the weight at its limit; see [the disagreement below](#where-the-sources-disagree) | null |
| `result-sheet-catdog-boxes` | after step 16, each detected animal pushed toward its own species | clean or near-clean on at least 6 of 8, full size, with the random push adding none | 4 of 8, the same count as the early push alone | [null](../../plans/review/28-each-animal-toward-its-own-species.md#answers) |
| the same run, the detector | how often step 16 holds two boxes | at least 6 of 16 pictures | 16 of 16 | support |
| `result-sheet-catdog-boxes-confident` | only boxes named with confidence of at least 0.8 are pushed | no wrong-species turn, and at least 5 of 8 | 4 of 8, one wrong-species turn remains | null |
| `result-sheet-catdog-presence-added`, seeds 3301 to 3332 | a "cat present, dog present" checker's push added on steps 0 to 15 | at least 7 of 32 faithful pairs with two bodies at 29 or more | scorer: 27 two bodies, 4 faithful pairs, 9 damaged | dropped |

A faithful pair is the automatic scorer's strictest verdict: exactly two bodies, one clear cat, one
clear dog, nothing borrowed and an intact photograph. The scorer is described on
[the 128-seed page](how-close-the-kept-push-comes-to-the-joint-prompt.md#concepts-this-uses).

## Pushing for the species on every step

Navigation: ⬅️ [Previous](#four-attempts-and-their-verdicts) | 📋 [TOC](#table-of-contents) | [Next](#where-the-sources-disagree) ➡️

**Plan 31's verdict is no change: 15 of 32 clean or near-clean, against 14 for the early push.** On
MNIST digits, adding the two pushes on every step lifted two clear, right digits from 215 to 285 of
320 `stated`. Plan 31 tried the same sum on all 50 steps, capped at 0.3. A setting was picked on seeds
9 to 16, then drawn once on seeds 3301 to 3332.

| Setting, seeds 9 to 16, by eye at 640 px | Folder | Separate of 8 | Clean or near-clean | Clean |
|---|---|---|---|---|
| readout push alone | `result-sheet-catdog-every-step-bodies` | 8 | 3 | 0 |
| plus CLIP species push, strength 10 | `result-sheet-catdog-every-step-species10` | 1 | 0 | 0 |
| plus CLIP species push, strength 30 | `result-sheet-catdog-every-step-species30` | 4 | 0 | 0 |
| plus presence-only readout checker (picked) | `result-sheet-catdog-every-step-presence` | 8 | 6 | 3 |

| Seeds 3301 to 3332 | Plain product | Random push | Early push (plan 29) | Picked setting |
|---|---|---|---|---|
| separate, of 32 | 1 | 2 | 28 | 29 |
| clean or near-clean, of 32 | 0 | 0 | 14 (9 clean) | 15 (8 clean) |

The picked setting is in `result-sheet-catdog-every-step-general`. Its pass marks were support at 20 or
more clean or near-clean with at least 26 separate, no change at 10 to 18, and worse at fewer than
24 separate or fewer than 10 clean or near-clean `stated` ([plan 31's review](../../plans/review/31-bodies-and-species-pushes-on-every-step.md#answers)).
The counts of 29 and 15 (8 clean) were recounted from its `calls.csv` `verified`. Plain product and
random push in this table are scorer counts.

## Where the sources disagree

Navigation: ⬅️ [Previous](#pushing-for-the-species-on-every-step) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**The eye and the scorer read the picked setting very differently, and plan 27's two columns match on 5 of 8 seeds.**

| What | One source | The other source | What explains it |
|---|---|---|---|
| clean pairs under the picked setting | eye at 600 px: 15 of 32 clean or near-clean | scorer: 2 faithful pairs, 14 of 32 with the verdict damaged, 28 of 32 with its intact-photograph answer false | a faithful pair needs an intact photograph (`api_label.py` lines 73 and 74). The eye count admits one small borrowed feature |
| the 15 against the early push's 14 | the 15 was called at 600 px | the 14 was called at full size | plan 29's own calls fell from 17 at half size to 14 at full size, so a gap of 1 cannot be read across two viewing sizes `inferred` |
| whether plan 27's weighted and fixed columns draw the same pictures | the review: the weight equalled the fixed 19 and the columns match | at strength 0.03 the tiles are byte-identical on 5 of 8 seeds (10, 11, 14, 15, 16). On seeds 9, 12 and 13 the files differ and the weight fell below 19 on 1, 8 and 3 of steps 0 to 9 | `not-hybrid/record.json` and a file hash. How far the three differing pictures look alike was not measured |

## What this cannot tell you

Navigation: ⬅️ [Previous](#where-the-sources-disagree) | 📋 [TOC](#table-of-contents) | [Next](#concepts-this-uses) ➡️

**The comparison in plan 31 changes two things at once.** The picked setting differs from the early
push in its window (50 steps against 16) and in the added species push, so the difference cannot be
assigned to either. Its 3 of 8 against 6 of 8 on seeds 9 to 16 did not survive on the 32 seeds `stated`.

## Concepts this uses

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

The toy numbers below are invented. The project numbers are from the plans' review files and from
`not-hybrid/record.json`.

**Two pushes under one cap**

One friend pushes a cart 2 units east and another 2 units north. Added together that is about 2.8 units
north-east. With a limit of 2, the sum shrinks to 2 and each friend's share shrinks with it. In plan 31
the two gradients are added first and the sum is capped at 0.3 of the product's prediction, so a strong
species push takes room from the push that keeps the bodies apart.

**The weight on NOT hybrid**

If a checker says a picture is 80% likely to be going wrong, the odds are 0.8 / 0.2 = 4, so the "not
this" term is applied 4 times over. The formula is k = p / (1 - p), where p is the chance of going
wrong. Plan 27 limits p to 0.95, so k cannot pass 0.95 / 0.05 = 19. Across the 24 checker-weighted runs the mean k
over steps 0 to 9 is 15.9 to 19.0, and at strength 0.03 it sits at 19.0 on 5 of the 8 seeds.

**Naming a box without training**

Crop one animal from a photo and ask a text-and-image model which caption fits better, "a cat" or "a
dog". If it answers 0.7 and 0.3, the box is called a cat. "Zero-shot" means no training on this task.
Plan 28 did this on step 16's blurry guess, and the dog box often read as a cat (chance of dog 0.08 to
0.21) `stated`.

## Where this came from

Navigation: ⬅️ [Previous](#concepts-this-uses) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What | Source | Mark |
|---|---|---|
| plan 26 | [the plan](../../plans/26-two-pushes-bodies-early-species-late.md) and [its review](../../plans/review/26-two-pushes-bodies-early-species-late.md) | `stated` |
| plan 27 | [the plan](../../plans/27-cat-and-dog-and-not-hybrid.md) and [its review](../../plans/review/27-cat-and-dog-and-not-hybrid.md) | `stated` |
| plan 28 | [the plan](../../plans/28-each-animal-toward-its-own-species.md) and [its review](../../plans/review/28-each-animal-toward-its-own-species.md) | `stated` |
| plan 31 | [the plan](../../plans/31-bodies-and-species-pushes-on-every-step.md) and [its review](../../plans/review/31-bodies-and-species-pushes-on-every-step.md) | `stated` |
| the added presence push | [the checker loop](../../artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md) and that folder's `scores.jsonl` | `verified` |

## Still open

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents)

- [ ] Confirm or veto the eye calls on plan 31's 32 seeds. Plan 31's task 4.1 is still open.
- [ ] Call plans 26 and 27 from full-size tiles. Their recorded calls were made at tile size.
- [ ] Try the walk's next idea, giving each detected body its own prompt after the layout settles. No plan holds it yet.

## Cross-references

- The mention of **faithful pair** in [the finding "can a small checker tell a clean cat and dog from a fused animal"](can-a-checker-read-a-clean-cat-and-dog.md).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
