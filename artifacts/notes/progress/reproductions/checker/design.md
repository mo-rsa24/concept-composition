# Choosing the checker architecture (step D3)

## What this asks

Paper 2 adds a push to the product of experts, so that one SDXL model draws two separate subjects where the product fuses
them. A trained classifier, called the checker here, supplies that push through its gradient. This comparison asks which of
three ways of reading the half-drawn picture makes the most faithful pairs on subject pairs it never trained on.

## Words this uses

**Product of experts (PoE).** The sampler that adds SDXL's noise predictions for the two single-subject prompts.

**Push.** The checker's gradient added to PoE's noise prediction during the early sampling steps.

**Faithful pair.** A picture with two separate bodies, a clear subject A and a clear subject B, no borrowed features, and an
intact photograph, as judged by the labeller in `2-classifier-guidance/scripts/rung2/api_label.py`.

**Neither-subject picture.** A picture of two other subjects, read by the checker under this pair's prompt. A guiding
classifier trained without them called a fridge a cat.

## The run

**Group.** Runs that try ideas (group 2). The result chooses an architecture for Paper 2 and makes no claim about Paper 2's
targets.

**Varied axis.** What the checker reads. Data, labels, loss, training length, push settings, pairs, seeds and scorer are the
same for all three.

**Buys.** Step D3: one named architecture, with the comparison behind it.

**Cost.** Each candidate trains once for 2,000 updates, budgeted at 2.5 GPU-hours, so 7.5 GPU-hours. Seven rows of 96 pictures
are rendered (plain PoE, and each candidate's push and its size-matched random push), 672 pictures at a budgeted 40 seconds,
so 7.5 GPU-hours. The total is about 15 GPU-hours. The smoke rung replaces both budgets with measured times.

**External cost.** Scoring 672 pictures with the labeller is about 1.46 million input and 67,000 output tokens on the owner's
Anthropic API key, at the 2,170 input and 99 output tokens per picture recorded in the checker loop. Nothing is scored this
way until the owner says yes on board question 15.

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

## The selection rule, fixed before any candidate trains

The constants and the rule are in `2-classifier-guidance/scripts/checker_choice/selection.py`.

A candidate is eligible only if all three hold:

1. A zero push reproduces plain PoE, with a largest absolute latent difference under 1e-5 in one loop on one card and dtype.
2. It rejects at least 80% of 200 neither-subject pictures, meaning a clean-pair logit below 0 averaged over steps 0, 5, 10 and
   15.
3. Its size-matched random push makes at most 2 faithful pairs of 96.

Among eligible candidates, the most faithful pairs of 96 wins. A lead under 5 counts as a tie, because re-rendering the same
recipe moved the faithful count by about 2. A tie goes to the fewest damaged pictures, then to the fewest parameters.

The labeller's calls are checked by eye on 16 random pictures per row at full size, before any count is believed.

## The ladder

Each rung saves its config, metrics, image sheet and commit under `/datasets/mmolefe/poe_repair_min/loop-5/checker/<rung>/`.

1. **Smoke test.** One batch through each candidate, one checkpoint saved and reloaded, one pushed render. Records seconds per
   update and per picture.
2. **Overfit.** Each candidate memorises 16 tiles, reaching at least 15 of 16 right.
3. **1% of the data.** 28 tiles train, and the loss falls.
4. **Controls.** The zero-push check and the neither-subject check for each candidate.
5. **Comparison.** The full 2,805 tiles, then the seven rows on the tuning pairs and seeds.

## What this cannot say

One training per candidate leaves out the variation between trainings. Two CLIP seeds in the checker loop differed by 0.035 in
balanced accuracy. The choice applies to these 12 pairs and this push recipe. A pushed render is not bit-reproducible on the GPU,
so a per-seed comparison between two pushed runs means little, and only the counts are compared.
