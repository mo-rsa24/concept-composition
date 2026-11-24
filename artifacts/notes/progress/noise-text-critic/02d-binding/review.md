# Review: can a sentence score read binding? (step W2d)

**State: no binding score yet.** Part binding has labels that hold, but on CO3's row alone the within-pair test has too few pictures. Colour binding's rebuilt labels missed the 90% eye-check bar, so they were not scored. The [design](design.md) holds the rules and bars.

## Part binding on CO3's pictures

Two labellers called CO3's 96 tuning pictures independently and blind, and agreed on 69 of the 73 pictures both called "own parts" or "borrowed" (94.5%, against an 80% bar). The five disagreements were settled by a third look at full size. The calls are in [the part-binding calls](part-binding-calls.csv): 40 own parts, 33 borrowed and 23 not two separate subjects.

Binding failure is mostly a property of the pair. Cheetah × cougar and duck × goose are borrowed in all 7 of their two-subject pictures, and eagle × fox is clean in 8 of 8. Only bobcat × quail, seal × gull and puma × condor hold both labels: 7 own-parts pictures and 15 borrowed. That is under the design's minimum of 8 per label per half, so the within-pair test fixed before scoring is inconclusive on this row, and no part-binding score was run.

## Colour binding with whole-animal regions

| Job | Stage | Code | Outcome |
|---|---|---|---|
| 94139 | smoke test, 4 pictures | `9315f8e7` | Each picture had exactly two animals, and their masks did not overlap. By eye, whole bodies were coloured, with the dog's front paw patchy in one picture |
| 94209 | regions and copies for the 62 pictures | `9315f8e7` | 57 recoloured; 5 left out because the detector found 1 animal (4 pictures) or 3 (1 picture) |

**The eye check.** One checker read the first 16 pictures in hash order, both copies of each, at full size: 32 copies. The calls are in [the colour eye check](colour-eye-check.csv).

| Result | Copies |
|---|---|
| Pass | 26 of 32 (81%), against a 90% bar |
| Fail | 6, from 3 pictures, each failing in both of its copies |

- **Picture 10.** The dog's head took the cat's colour and its body, lying behind, stayed uncoloured. The cat's mask covered the dog's head, so this copy's label is wrong as well as patchy.
- **Picture 00.** The cat's tail and the dark fur along its back are uncoloured, about an eighth of the cat.
- **Picture 07.** The dog's white front leg and paw are uncoloured, about a sixth of the dog.

The checker passed pictures 01 and 13 as close calls, each with just under a tenth of one animal uncoloured. No copy had a tinted background.

By the design's rule, nothing is scored on these labels.

## Why binding stays untested

The owner chose on 2026-11-24 to wrap up Loop 9 without repairing either label set. The sentence scores missed counting at steps 0 to 15 and on the finished picture ([the W2c review](../02c-fair-test/review.md)), so the image side is unlikely to read which part belongs to which animal (inferred). The text side is not the obstacle: CLIP's text encoder keeps colour binding, with a ratio of 0.72 against 0.13 for random weights ([the word check](../02-can-it-read/review.md#text-only-word-check)). The part-binding labels stay for whoever tests binding next. Their main finding, that CO3's binding failures are fixed by the pair and not by the seed, goes to Loop 6.
