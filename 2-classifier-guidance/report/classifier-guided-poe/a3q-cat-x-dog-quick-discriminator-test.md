# A3q: cat × dog, a small pair-aware discriminator, judged on held-out seeds

Rung 2 of [the scale-up ladder](board.md#the-scale-up-ladder): the fastest real test on SDXL. A
pair-aware discriminator is trained on 10 to 15 renders of a cat and a dog apart, each checked by
eye, from seeds 1 to 8 (the joint prompt plus "a cat on the left and a dog on the right" and "a cat
and a dog sitting apart"). Its negatives are the product's own renders from the same seeds and
renders of a lone cat and a lone dog, all noised the same way. Lone animals are negatives because
[plan 18](../../plans/18-a-dog-classifier-trained-the-papers-way.md)'s push turned the fused animal into one cat on 3 of 4 seeds. Its push is judged on held-out seeds 9 to 12 against plain product
and best-of-N at equal compute, with the bar written before the run.

## Outcomes

- [x] 📝 compute plan written before launch: GPU-hours and wall clock from measured rates, node, card and fallback card, seeds and pairs it buys · `verified` · [plan 21's compute block](../../plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md#considerations): about 2.1 GPU-hours on `mscluster107` GPU 0, fallback `mscluster108` GPU 1, N = 2, cat × dog on seeds 1 to 4 and 9 to 12, 2026-11-17
- [x] 📝 bar written before the run: the seven readings below, each with its count out of 4 seeds · `verified` · [the rung 2 review file](../../plans/review/a3q-cat-x-dog-quick-discriminator-test.md), 2026-11-17
- [x] 💻 code ran: the discriminator trained on the checked renders and the product's renders from seeds 1 to 4 · `verified` · trained on 17 checked positives and 24 negatives from seeds 1 to 8 ([plan 21](../../plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md), task 1.3); held-out check 13 of 20 sampler steps, 2026-11-17
- [x] 🖼️ screenshot: one sheet, seeds 1 to 4 and 9 to 12 side by side, columns plain product, best-of-N at equal compute and guided · `verified` · the result sheet, with the size-matched random push at 10, 30 and 100, 2026-11-17
- [ ] ✅ check passed: every tile called by eye at full resolution, and the reading below that the calls land on, printed as the rung's pass or miss line · `stated` · Claude's draft calls land on reading 7, too weak to read; the reader has not yet confirmed the 64 calls

## The result

**Draft reading 7, too weak to read: no tile on seeds 9 to 12 shows two separate animals, and the
size cap clipped the push on every step at s = 100.** Composed is 0 of 4 on the held-out seeds in
every column, against 1 of 4 on the training seeds (seed 4, a drawing). The guided push changes every
picture while the same-sized random push leaves them as plain, so the discriminator's direction acts,
but toward fused animals and broken pictures.

`result-sheet.png` is not shown because it failed the subject check

**The discriminator did not read the concept.** It scored pair-prompt renders positive whatever
they showed, hybrids and two dogs included, and looked at one fixed patch of every picture. Every
discriminator trained with hybrids and wrong pairs among its negatives failed its held-out check:
17 positives, then 81 positives over 38 benchmark pairs, with and without adapters on SDXL's trunk,
all 0 of 20 sampler steps at the 0.75 bar.

<a href="../../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/diagnostics/kinds-through-time.png"><img src="../../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/diagnostics/kinds-through-time.png" width="200" alt="Discriminator logit against sampler step per kind of training tile: positives, hybrids and wrong pairs from the pair prompts all score above zero; lone animals and plain products below"></a>
<a href="../../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/discriminator/held-out-accuracy-all-checkers.png"><img src="../../artifacts/results/does-a-small-discriminator-steer-cat-x-dog/discriminator/held-out-accuracy-all-checkers.png" width="200" alt="Held-out balanced accuracy against sampler step for four discriminators: v1 rises above 0.75 after step 7 on a check without hard negatives; v2, v3 head and v3 with adapters stay near 0.6 at every step"></a>

The next rung needs a discriminator that reads "two separate things" on pictures it never trained
on, before any push is judged. The record is [the rung 2 review file](../../plans/review/a3q-cat-x-dog-quick-discriminator-test.md)
and [plan 22](../../plans/22-rung2-a-discriminator-trained-on-the-benchmark-pairs.md).

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

## What later experiments add

**The cap 0.3 rerun that reading 7 names has not run, so the draft reading above stands.** The rerun
is [plan 21](../../plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md)'s task 1.4.1, still
unticked `stated`. The later checkers use a different design, and each question has its own page:

| The question | Its page |
|---|---|
| Can a checker tell a clean cat and dog from a fused animal? | [the checker readings](can-a-checker-read-a-clean-cat-and-dog.md) |
| Does a push on the first 16 steps separate the two animals? | [the early push](does-an-early-push-separate-the-cat-and-dog.md) |
| Can a second push make each body the right animal? | [the second push](can-a-second-push-name-each-animal.md) |
| Does the early push work on other pairs? | [the other pairs](does-the-early-push-carry-to-other-pairs.md) |
| How close does the kept push come to the joint prompt and CO3? | [the 128-seed comparison](how-close-the-kept-push-comes-to-the-joint-prompt.md) |
