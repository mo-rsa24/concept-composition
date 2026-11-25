# Does correcting for each animal's size let the per-place reader see borrowing? (step M12, second reader)

## What this asks

The first identity reader for Paper 2 asked, cell by cell, which animal's prediction CO3's joint prediction sits closer to, and scored a
picture by the smaller animal's share of cells. On CO3's 96 tuning pictures it was inconclusive, at a best balanced accuracy of 0.688 against
the 0.75 bar ([the first reader's review](../07-a-checker-that-reads-identity/review.md)). Its per-pair table showed two flaws. It reads low
whenever one animal is much larger than the other (hippopotamus and swallow 0.081, with no two-clear pictures to find), and it reads high
when an extra animal inflates the smaller one's share (boar and pheasant 0.403, 0 of 8 two clear, because extra pheasants appear). The owner
answered board question 106 with A and one added rule, on 2026-11-25. Correct each share for the animal's expected size, count any picture
showing more than two animals as a failure, and test on 8 fresh CO3 seeds per pair, never on the 96 tuning trajectories already read.

## Words this uses

**Per-place share.** At one sampling step, SDXL predicts the noise under the joint prompt and under each animal's own prompt. A latent cell
belongs to the animal whose prediction is closer to the joint one, each distance taken over the 4 channels. The first animal's share is the
fraction of the 128 by 128 cells that belong to it, and the second's is the rest.

**Expected share.** How much of the picture each animal fills when SDXL draws it alone, as the detector measures it, relative to the pair
(defined in full below).

**Size-corrected score.** For each animal, its per-place share divided by its expected share; the score is the smaller of the two ratios,
capped at 1. A picture where both animals hold what their sizes predict scores near 1. One where an animal holds much less than its size
predicts scores low, whatever its size.

**Count checker.** The project's validated instance count, `count_instances` in
`poe_repair/experiments/compose_scorer_validation/detection_scorer.py`: GroundingDINO-Tiny asked for "animal", boxes under a quarter of the
image's longer side dropped, overlapping boxes merged. It is validated on animal pairs only.

**Two clear.** A blind call on CO3's finished picture under the [blind count rubric](../01-beat-the-baseline/blind-count-rubric.md): one
clear subject A and one clear subject B as separate bodies, no borrowed parts, drawing style ignored.

## How the expected share is measured, fixed before any run

1. For each of the 24 subject words in the 12 tuning pairs, plain SDXL draws "a <subject>" alone at 1024 by 1024, at seeds 8001 to 8004, in
   30 steps at guidance 5.0, built from the same SDXL weights CO3 loads. These seeds and prompts are used nowhere else.
2. On each picture, the detector is asked for the subject's own word. The highest-confidence box at confidence 0.30 or more gives the area
   fraction (box area over picture area). A seed with no such box is left out; a subject with no box on any seed takes 0.5 and is reported.
3. A subject's area is the mean over its seeds. For a pair, the first animal's expected share is its area over the sum of both areas, and
   the second's is the rest.

A box is a rough measure of a body: it is larger than a thin animal's outline, so a flamingo's expected share comes out high. That bias is
the same for every picture of the pair, so it shifts the pair's scores together and is reported per pair.

## The extra-animal rule

At each read step, the one-step clean estimate (the picture the state would become if the sampler jumped to the end, from CO3's guided
prediction at its released scale of 0.8) is decoded, and the count checker reads it. On the 11 animal pairs, a count above 2 makes that
step's score 0, a failure, whatever the cells say. The rule reads the clean estimate at the same step, so the reader uses only what exists at
that step. Lighthouse and sailboat are objects, which the count checker is not validated on, so the rule does not apply to that pair. The
count on each finished picture is reported beside it.

## The run

**Group.** Runs that try ideas. The result decides whether a reader with no training can carry Paper 2's push, and makes no claim in either
paper.

**Fresh trajectories.** CO3 at its released settings on the 12 tuning pairs at seeds 6009 to 6016, 96 trajectories that no loop has read.
The state at steps 5, 10, 15, 20, 25 and 30 is saved, and the finished picture is kept. The held-out pairs and seeds 7001 to 7008 are not
touched.

**Labels.** The 96 finished pictures are shuffled under neutral names and called blind at full size by two reviewers under the blind count
rubric, with their agreement reported and a third blind call deciding each picture they disagree on. Neither reviewer sees a score.
Giraffe and tortoise is counted and stays off every sheet.

**Code.** New stages in `2-classifier-guidance/scripts/method_search/m5_read_co3.py`, launched by `m5_read.sbatch`: the CO3 read at the
fresh seeds with states saved, the single-subject renders and their areas, and the per-place shares with the count on each clean estimate.
Outputs under `/datasets/mmolefe/poe_repair_min/loop-6/m12/size-corrected/`.

## The bar, written on 2026-11-25 before any run

Balanced accuracy of the call "two clear" when the size-corrected score is at least 0.60, against the blind two-clear calls on the 96 fresh
pictures. The 0.60 is the first reader's call of 0.30 on an even split of 0.50, carried over unchanged.

| Result | What it takes | What follows |
|---|---|---|
| The reader sees the failure | at least 0.75 at any one step from 5 to 30 | a push on the score at steps 15 to 25, judged on Paper 2's bar |
| It does not | under 0.65 at every step, and an area under the curve never above 0.742, the first reader's best | this reader stops |
| Inconclusive | the best step between 0.65 and 0.75 | reported, and the owner decides |
| On ranking only | under 0.75 at every step on the call, but an area under the curve of at least 0.75 | returns to the owner under question 96 |

Reported beside the bar, deciding nothing: the first reader's uncorrected score on the same fresh trajectories, so the size correction and
the extra-animal rule are seen apart from the new seeds; the score with the size correction but without the rule; the area under the curve
at each step; and each pair's expected shares, two-clear count and mean score.

**What would surprise.** The correction lifting hippopotamus and swallow while every other pair stays where it was. That would mean size was
one pair's problem and not the reader's.

## Cost and what it buys

**Cost:** about 1.5 GPU-hours in `bigbatch`, the faulted nodes excluded: 96 CO3 runs at the 41 seconds measured in M5's read (about 1.1),
96 single-subject renders at 30 steps (about 0.2), and the four-prompt passes, decodes and counts on 576 states (about 0.25). This is about
0.3 more than the 1.2 in the owner's answer, because the expected shares need renders of their own. No API tokens: the labels are blind
calls. **Buys:** whether a reader with no training, corrected for the two flaws the first one showed, sees CO3's borrowing early enough to
push on.

## What this cannot say

The per-place share says how much of the picture each animal holds, and not which body belongs to which: on cheetah and cougar both bodies can
read "cheetah". A box overstates a thin animal's size. The count checker is validated on finished pictures; on a blurred clean estimate at
step 5 it may count fewer animals than are there, so the rule is weakest early.
