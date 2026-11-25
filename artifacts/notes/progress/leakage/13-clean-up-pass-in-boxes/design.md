# Design: does Z7's clean-up pass with each animal's own prediction in its box beat the same pass over the whole picture? (step Z13)

## What this asks

Z7 added one extra corrector pass per step at sampler steps 5 to 30 and, inside each animal's box, gave that animal's own prediction alone; it drew 14 of 48 pictures with two clear subjects against 8 for CO3 alone (p = 0.073, null by its bar). Saved pictures show that most of Z7's change came from the extra pass itself: the same pass over the whole picture with no boxes moved pictures by a median of 24.5 grey levels against 21.4 for the boxed pass. Z12's push, which moved pictures by 2.3, could not tell. This step asks the one question neither answered: **with the extra pass held fixed, does putting each animal's own prediction inside its box add two-clear pictures?** The owner said yes to question 115 (2026-11-26, `requests.md`).

## Rows

| Row | What is sampled | Why |
|---|---|---|
| a. Pass with boxes | CO3 plus Z7's extra corrector pass at steps 5 to 30; inside each animal's automatic box, that animal's own clean-picture guess alone; elsewhere CO3's equation-15 weights | the box content |
| b. Pass, whole picture | the same pass with CO3's equation-15 weights everywhere, no boxes | the extra pass alone |
| c. CO3 alone | released settings | the baseline, and the pictures the boxes are read from |

**Varied axis.** Row a against row b changes only what goes inside the boxes. Row b against row c is the extra pass alone, reported and deciding nothing.

**The pass** is Z7's, unchanged: `2-classifier-guidance/scripts/leakage/z3_regions.py`, `Corrector.pass_` with row `own` (row a) and row `whole` (row b), steps 5 to 30, beta 0.9, the largest-entry rescale and the rescale to the state's length.

**Pairs and seeds.** The 12 tuning pairs of `2-classifier-guidance/scripts/checker_choice/splits.json` at seeds **6025 to 6032**, which no Loop 12 run has drawn (Z7 and Z2 used 6001 to 6008, Loop 6's saved states 6001 to 6016, Z12 6017 to 6024): 96 pictures a row. Giraffe × tortoise is drawn and counted and never placed on a sheet. No held-out pair, held-out seed (7001 to 7008) or benchmark prompt.

**Boxes** come from instruments, as in Z12: Grounding DINO places the bodies on row c's finished picture of the same pair and seed, and zero-shot CLIP names them (`z12_box_push.py boxes`). They are a best case for any reader of regions. A picture with no kept box gets the whole-picture weights in row a, which is counted.

**Card.** `bigbatch` RTX 3090 cards, whole nodes, the faulted nodes and the two nodes with a stale `/datasets` (mscluster79 and 87, `environment/hpc/nodes.md`) excluded; every job refuses any other card.

## Controls

| Control | What it does | What comes back |
|---|---|---|
| Empty boxes | row a's code with both boxes empty | row b's picture byte for byte, or the job stops |
| Entering step 5 | the latent entering step 5 in rows a and b against row c | identical |

## How pictures are counted

As Z7 and Z12: two clear subjects with their own parts, drawing style ignored, on [Loop 6's rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md), with `broken`, `faithful`, `damaged`, `a_own` and `b_own`. Two blind reviewers call every picture, one from each end of the sheet, with a third on each rung disagreement. Each batch holds all three rows of the same pairs under neutral ids, so every matched comparison sits inside one batch.

**Pixel change per row** is reported: the mean absolute difference from row c over each picture, median and largest over the 96, in grey levels of 255.

## The bar, written before any picture is drawn

**Support** when all three hold, row a against row b on the 96 matched pictures:

1. Row a has at least 5 more `two_clear` pictures than row b.
2. An exact one-sided McNemar test on the matched seeds gives p < 0.05.
3. Row a's `broken` count is at most row b's plus 3.

Otherwise **null**: putting each animal's own prediction in its box adds nothing to the extra pass, and the box route on CO3 closes with Z7, Z12 and Z13 as its record. Row a and row b each against row c are reported beside the bar and decide nothing.

## A rule that can only stop

**The first half** is the first two pairs of each of the three jobs at all eight seeds: puma × condor, hippopotamus × swallow, zebra × flamingo, giraffe × tortoise, seal × gull and bobcat × quail, 48 pictures a row. Those are called blind first. If row a's two-clear count on them is not above row b's, a lead of 0 or less, the remaining jobs are cancelled and Z13 is reported as stopped for futility. A lead above 0 decides nothing.

## Cost

| Part | RTX 3090 hours |
|---|---:|
| Row c: 96 at about 43 s | 1.15 |
| Row a: 96 at about 56 s (Z7's own row took 55.5 s) | 1.5 |
| Row b: 96 at about 56 s | 1.5 |
| Boxes and controls | about 0.1 |
| **In all** | **about 4.2** |

This is above the approved "about 3": three rows of 96 at Z7's measured times do not fit 3 hours. Rows b and c run first, in parallel; row a waits for row c's boxes.

W&B project `prime_lab/poe-repair-animals-compose`, group `loop-12-z13`. No API tokens.
