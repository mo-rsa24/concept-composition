# A3q: cat × dog, a small pair-aware discriminator, judged on held-out seeds

Rung 2 of [the scale-up ladder](board.md#the-scale-up-ladder): the fastest real test on SDXL. A
pair-aware discriminator is trained on 10 to 15 renders of a cat and a dog apart, each checked by
eye, from seeds 1 to 8 (the joint prompt plus "a cat on the left and a dog on the right" and "a cat
and a dog sitting apart"). Its negatives are the product's own renders from the same seeds and
renders of a lone cat and a lone dog, all noised the same way. Lone animals are negatives because
plan 18's push turned the fused animal into one cat on 3 of 4 seeds. Its push is judged on held-out seeds 9 to 12 against plain product
and best-of-N at equal compute, with the bar written before the run.

## Outcomes

- [ ] 📝 compute plan written before launch: GPU-hours and wall clock from measured rates, node, card and fallback card, seeds and pairs it buys
- [x] 📝 bar written before the run: the seven readings below, each with its count out of 4 seeds · `verified` · [the rung 2 review file](../../plans/review/a3q-cat-x-dog-quick-discriminator-test.md), 2026-11-17
- [ ] 💻 code ran: the discriminator trained on the checked renders and the product's renders from seeds 1 to 4
- [ ] 🖼️ screenshot: one sheet, seeds 1 to 4 and 9 to 12 side by side, columns plain product, best-of-N at equal compute and guided
- [ ] ✅ check passed: every tile called by eye at full resolution, and the reading below that the calls land on, printed as the rung's pass or miss line

## What it confirms, and what it doesn't

Counts are by eye at full resolution, out of 4 seeds per seed set, at strengths `s` = 10, 30 and 100.
Composed means two separate animals of any species; intact means the photograph does not break at
full size. The readings are checked top to bottom and the first that holds is the verdict; the
definitions are in [the rung 2 review file](../../plans/review/a3q-cat-x-dog-quick-discriminator-test.md).

| If the sheet shows | Count | It means |
|---|---|---|
| two animals, pictures intact | at some `s`, composed intact at least 2 of 4 on seeds 9 to 12, more than best-of-N's, and the random push at that `s` at most 1 of 4 | the push generalises across seeds and its direction does it: worth the big run |
| two animals, but best-of-N does as well | at some `s`, composed intact at least 2 of 4 on seeds 9 to 12, with best-of-N equal or higher at every such `s` | the discriminator works as a picker and the push adds nothing |
| two animals, but a random push of the same size does as well | at some `s`, composed intact at least 2 of 4 on seeds 9 to 12 and more than best-of-N's, but the random push at least 2 of 4 at every such `s` | any disturbance this size separates the animals; the discriminator's direction adds nothing |
| two animals but pictures wrecked | at some `s`, composed at least 2 of 4 on seeds 9 to 12, at most 1 of them intact | inconclusive: rerun at `s` = 3 and the lowest `s` that composed |
| two animals only on seeds 1 to 4 | composed at most 1 of 4 on seeds 9 to 12 at every `s`, at least 2 of 4 on seeds 1 to 4 at some `s` | it memorised the training seeds: the idea needs more data, not more compute |
| no change, and the push acted | at most 1 of 4 on both seed sets at every `s`; at `s` = 100 the picture moved on at least 2 of 8 seeds and the cap fired on fewer than half the steps | a pair-aware push this small cannot fix it: stop before scaling |
| no change, and the push was too weak | the counts above, but either check at `s` = 100 fails | rerun at `s` = 300, or at `s` = 100 with the cap at 0.3, before calling a null |

Only the first reading lets rung 3 start.
