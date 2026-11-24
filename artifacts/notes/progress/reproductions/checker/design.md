# Choosing the checker architecture (step D3)

## What this asks

Paper 2 adds a push to CO3 inside one SDXL model. CO3 already draws two separate bodies in all 96 tuning pictures, and its failures
are identity: borrowed parts, a wrong species or an extra subject. A trained classifier, called the checker here, supplies the push
through its gradient. This comparison asks which of three ways of reading the half-drawn picture makes the most pictures with two clear
subjects on subject pairs it never trained on.

## Words this uses

**Product of experts (PoE).** The sampler that adds SDXL's noise predictions for the two single-subject prompts.

**Push.** The checker's gradient added to PoE's noise prediction during the early sampling steps.

**Two clear subjects.** A picture on the `two_clear` rung of the [blind count rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md):
one clear subject A and one clear subject B as separate bodies, no borrowed features, drawing style ignored. Damage is called on the same
picture and reported beside the count.

**Neither-subject picture.** A picture of two other subjects, read by the checker under this pair's prompt. A guiding
classifier trained without them called a fridge a cat.

## The run

**Group.** Runs that try ideas (group 2). The result chooses an architecture for Paper 2 and makes no claim about Paper 2's
targets.

**Varied axis.** What the checker reads. Data, labels, loss, training length, push settings, pairs, seeds and scorer are the
same for all three.

**Buys.** Step D3: one named architecture, with the comparison behind it.

**Cost.** Each candidate trains once for 2,000 updates, budgeted at 2.5 GPU-hours, so 7.5 GPU-hours. Seven plain-PoE rows of 96
pictures are rendered (plain PoE, and each candidate's push and its size-matched random push), 672 pictures at a budgeted 40
seconds, so 7.5 GPU-hours. Three CO3 rows add each candidate's push to CO3, 288 pictures at 59 seconds, so 4.7 GPU-hours. CO3 alone
reuses step M1's 96 pictures. The total is about 20 GPU-hours. The smoke rung replaces these budgets with measured times.

**External cost.** Scoring the 672 plain-PoE pictures with the labeller is about 1.46 million input and 67,000 output tokens on the
owner's Anthropic API key, at the 2,170 input and 99 output tokens per picture recorded in the checker loop. The owner approved this
on board question 15. The CO3 rows use no API tokens.

## The three candidates

| Candidate | What it reads | Code | Where the idea comes from |
|---|---|---|---|
| A. SDXL feature reader | SDXL's frozen middle and first two up-block features under the joint prompt | `PairHead` in `2-classifier-guidance/scripts/rung2/train_readout_pair.py` | the existing checker |
| B. Noisy-latent classifier | the noisy latent and timestep directly, conditioned on the joint prompt, with no SDXL features | new, in `2-classifier-guidance/scripts/checker_choice/` | Compositional Sculpting, which trains a classifier on model samples (field table) |
| C. Attention-map reader | SDXL's cross-attention maps for the words of the two subjects | new, in the same folder | Attend-and-Excite, which reads those maps (field table) |

Two other readers are left out. A CLIP checker on the decoded estimate is named in none of the three source files, and in the
checker loop its push changed texture and left the layout alone. Lift Scores rejects finished samples with the model's own
scores, so it has no gradient to push with.

## The data

**Training.** `2-classifier-guidance/scripts/rung2/training_set_multipair.json`, 2,859 tiles over 243 pairs, with the 54 tiles
over 27 pairs that name a subject on the guard's decision list removed, because the pair ledger allows those subjects only for
the review response. That leaves 2,805 tiles over 216 pairs: 1,154 clean and 1,651 rejects.

**Outputs and labels.** Three outputs (clean pair, first subject present, second subject present), with the presence labels of
the existing checker, `presence_catdog.json`.

**Neither-subject negatives.** A quarter of each batch's rejects are tiles from another pair, read under this pair's prompt,
where the two pairs share no word. The training list holds no such tile today.

## Pairs and seeds, fixed before any candidate trains

`2-classifier-guidance/scripts/checker_choice/build_splits.py` writes `splits.json`. An evaluation pair is outside the 275-pair
benchmark pool and the training list, passes the subject guard, and shares no word with any trained pair or with another
evaluation pair. The order comes from the SHA-256 of each pair's name.

| Set | Pairs | Seeds | Use |
|---|---|---|---|
| Tuning | 12: puma × condor, zebra × flamingo, seal × gull, hippopotamus × swallow, giraffe × tortoise, bobcat × quail, duck × goose, crow × raven, eagle × fox, lighthouse × sailboat, boar × pheasant, cheetah × cougar | 6001 to 6008 | the whole comparison |
| Held out | 12: cow × milk churn, bicycle × brick wall, bison × weasel, telescope × beach ball, turkey × peacock, baboon × gazelle, tiger × parrot, squirrel × chipmunk, camel × pelican, hawk × falcon, alpaca × wheelbarrow, reindeer × sled | 7001 to 7008 | nothing in this step; kept for the scaling runs |
| Neither source | the other 33 eligible pairs | their existing joint renders | the neither-subject check |

Seeds 6001 to 6128 already count as tuning seeds, because the kept push was selected on them. Nothing on the void list justifies
any choice here, including that push's 54 of 128.

## The push, held fixed

Strength 30, capped at 0.4 of PoE's prediction, on steps 0 to 15, then plain PoE. These settings were tuned for candidate A, so
they could favour it. Before the comparison, each candidate's strength is scaled until its push reaches the cap on at least 80%
of the pushed steps, measured on one tuning pair and seed. This uses the push's size only and looks at no picture or score.

On the CO3 rows the same strength, steps and cap apply, with the cap measured against CO3's guided prediction. The push enters through
`2-classifier-guidance/scripts/method_search/co3_push.py`, as in step M1.

## The selection rule, fixed before any candidate trains

The constants and the rule are in `2-classifier-guidance/scripts/checker_choice/selection.py`. By board question 57, the CO3 row decides
what goes forward, because CO3 is the rival Paper 2 has to beat.

A candidate is eligible only if all four hold:

1. A zero push reproduces plain PoE, with a largest absolute latent difference under 1e-5 in one loop on one card and dtype.
2. A zero push reproduces CO3's own loop to the same limit.
3. It rejects at least 80% of 200 neither-subject pictures, meaning a clean-pair logit below 0 averaged over steps 0, 5, 10 and
   15.
4. Its size-matched random push on plain PoE makes at most 2 pictures with two clear subjects of 96.

Among eligible candidates, the most pictures with two clear subjects of 96 on the CO3 row wins. A lead under 5 counts as a tie, because
re-rendering the same recipe moved the count by about 2. A tie goes to the fewest parameters. Damage on each row is reported beside the
count and decides nothing. The plain-PoE counts are reported beside it and choose nothing. The owner changed this bar on 2026-11-24, on board question 87, before any result of the search existed. It was faithful pairs, which also required an intact photograph; CO3 alone scores 11 of 96 on that and 32 of 96 on two clear subjects.

The chosen candidate beats CO3 alone when it leads by at least 5 of 96 and by more than 1.96 standard errors, from 10,000 bootstrap
resamples of the 12 pairs. If it does not, D3 still names the best reader and reports that no candidate beat CO3 alone.

**How the CO3 rows are counted.** Step M1's 96 CO3 pictures are reused, and their calls are not, because M1's six reviewers passed
between 0 and 9 of 48 pictures. CO3 alone and the three CO3-plus-checker rows, 384 pictures, are shuffled together under neutral names
and called at full size under the [blind count rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md) by two reviewers,
whose agreement is reported, with a third blind call deciding each picture they disagree on.

The labeller's calls on the plain-PoE rows are checked by eye on 16 random pictures per row at full size, before any count is believed.

## The ladder

Each rung saves its config, metrics, image sheet and commit under `/datasets/mmolefe/poe_repair_min/loop-6/d3/<rung>/`.

1. **Smoke test.** One batch through each candidate, one checkpoint saved and reloaded, one pushed render. Records seconds per
   update and per picture.
2. **Overfit.** Each candidate memorises 16 tiles, reaching at least 15 of 16 right.
3. **1% of the data.** 28 tiles train, and the loss falls.
4. **Controls.** The zero-push checks on plain PoE and on CO3, and the neither-subject check, for each candidate.
5. **Comparison.** The full training list from step M5, then the seven plain-PoE rows and three CO3 rows on the tuning pairs and seeds.

## The unseen-category test

**What it asks.** Does a checker trained only on animals still make pictures with two clear objects? A reader that learned "two separate
things" should carry over, and a reader that learned the species it saw should not.

**Readers.** Candidates A and C, because SDXL's features carry appearance and the word attention maps carry mostly position.

**Training.** Each reader trains once on the animal and animal pictures in the list step M5 passes to D3, with every picture naming an
object removed. On the current list that is 1,749 pictures over 38 pairs and 12 animal words, 660 of them clean and 1,603 of them cat
and dog. Labels, loss and push settings stay as in the main comparison.

**Test pairs.** Six object pairs at seeds 6001 to 6008, so 48 pictures per row: microscope × test tube, chess board × hourglass,
saxophone × music stand, mailbox × pumpkin, picnic basket × watermelon and lighthouse × sailboat. No training list holds them.
Bicycle × birdhouse is left out because bicycle appears in a held-out pair. Lighthouse × sailboat is also one of D3's 12 tuning pairs, so
every count is reported on all six pairs and on the five without it. A reader that passes on six pairs and misses on five is
inconclusive.

**Rows.** Plain PoE, A's push, A's size-matched random push, C's push and C's size-matched random push, 240 pictures in all. They are
counted blind at full size under the [blind count rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md), with no API
tokens.

**The bar, on pictures with two clear subjects of 48, set by the owner's answer to board question 87.** Standard errors come from 10,000 bootstrap resamples of the six pairs. The lead of 3 is the main
comparison's tie rule of 5 of 96, scaled to 48 pictures and rounded up.

| Result, per reader | What it takes |
|---|---|
| Transfers | its push leads plain PoE by at least 3 and by more than 1.96 standard errors |
| Does not transfer | its push has no such lead |
| Inconclusive | its zero-push check fails, or its random push leads plain PoE by at least 3 and by more than 1.96 standard errors |

C carries over better than A when C's push leads A's by at least 3 and by more than 1.96 standard errors. This test stays on plain PoE,
reports beside the selection rule and does not change it. Step M8 reads the same six pairs beside CO3.

**Cost:** about 9 GPU-hours beyond the main comparison: two trainings budgeted at 2.5 GPU-hours each, and 240 pictures at a budgeted 58
seconds. **Buys:** a direct answer to whether either reader works on a category it never trained on.

## What this cannot say

One training per candidate leaves out the variation between trainings. Two CLIP seeds in the checker loop differed by 0.035 in
balanced accuracy. The choice applies to these 12 pairs and this push recipe. A pushed render is not bit-reproducible on the GPU,
so a per-seed comparison between two pushed runs means little, and only the counts are compared.

## Cross-references

- The mention of **checker architecture**, the choice step D3 makes on small data, in [the route to two papers, under The steps](../../../../../plans/route-two-papers.md#the-steps).
