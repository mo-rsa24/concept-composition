# Remove one checker at a time, on digits (step V5)

## What this asks

Does removing one checker damage only its own property? If it does, composition is controllable property by property, which is the owner's main experiment. This is the digit rehearsal from [the abstract-checker design](../03-design.md#the-digit-rehearsal), approved by board question 56, run before any SDXL version.

## Words this uses

**The three checkers.** Count (two digits sitting apart?), right digits (are they the two asked digits?) and order (is the asked left digit on the left?). Each is a map reader at the 16 timesteps matching SDXL's noise, the setting V2 settled.

**Fully right.** A canvas the judge calls clean (two confident, right digits, one per half) with the asked digit on the left.

## What each checker is

| Property | Checker | Training label | Score on a finished canvas |
|---|---|---|---|
| Count | V2's count checker, saved as `v2/count-final/count-head-map.pt` | `apart` and `wrong` are 1; `single`, `overlaid`, `same_half` are 0 | share of canvases with two separate digits, one per half |
| Right digits | a map reader trained on `apart` (1) against `wrong` (0) on all 10 pairs among digits 0 to 4 | as V2's identity checker | of the canvases that pass count, the share the judge calls clean |
| Order | a map reader trained on `apart` canvases, each read with its true order (1) and the flipped order (0) | the asked order is given by the order of the two map channels: the first channel is the digit asked to be on the left | of the clean canvases, the share with the asked left digit on the left |

The three scores multiply to the share of canvases that are fully right. Each column is scored only on canvases that passed the column before it.

**Known from V2.** The identity checker read balanced accuracy 0.555 on the pairs holding digit 4 at these noise levels, against a pass mark of 0.75. The right-digits checker is the same recipe on more data, and the order checker also needs to tell digits apart. Both checkers' accuracy on digits 0 to 4 and 5 to 9 is measured and reported before the rows run, and the rows run whatever it is, because a checker with little signal is part of the answer.

## Run settings

**Group.** Ablation (group 7): one checker removed per row, everything else held.

**Varied axis.** Which checker is off.

**Model and sampler.** The `share-0.0` model V3 passed on; DDIM with 50 steps and eta 0, the product of the two asked labels at guidance 1, `cap=1.0`.

**Push window.** Sampler steps 15 to 25 (timesteps 693 down to 489), V3's window.

**Push.** Each active checker adds $-\lVert\epsilon_{\text{product}}\rVert\, b\, g_k/\lVert g_k\rVert$. The length $b$ is the same for every checker and set once by $b = 0.4/\sqrt{3(1 + 2\bar\rho)}$, where $\bar\rho$ is the mean pairwise cosine of the three checkers' gradients over the pushed steps on the 10 seen pairs at seeds 0 to 31, clipped at 0. No shared cap runs. A row is void if its total push passes 0.6 of the product's norm on more than 5% of pushed steps.

**Asked order.** The lower digit is asked on the left for even seeds and on the right for odd seeds.

**Rows.** Eleven rows on the 10 unseen pairs among digits 5 to 9 at seeds 1000 to 1031, 320 canvases per row: the plain product; all three checkers; each removed (3); each removed with a random push of the same length in its place (3); each alone (3).

**Score.** The digit judge `judge-v3/judge.pt` through `two_experts.call_two` and `two_experts.reads`, V3's scoring. Standard errors come from 10,000 bootstrap resamples of the 10 pairs.

| Result | What it takes |
|---|---|
| Support | All three together beat the plain product on fully right canvases by more than 1.96 standard errors. Each removal lowers its own column by at least 6.25 percentage points and more than 1.96 standard errors, its random-push row falls by the same rule, and no other column moves by both more than 4.2 points and more than 1.96 standard errors |
| Partial | At least one property meets that rule and at least one does not |
| Null | No removal lowers its own column by that margin |
| Inconclusive | The zero-push check fails, a row is void by the 0.6 total-length rule, or a cell the verdict needs holds fewer than 40 canvases |

The bars sit in `2-classifier-guidance/scripts/abstract_checkers/v5_remove.py`.

**The ladder.** A smoke test of every stage on one pair at 4 seeds with 50-update checkers; the two trainings with their overfit checks and accuracies; the calibration of $b$ and the zero-push check on the seen pairs; then the eleven rows on the unseen pairs, once.

**Cost:** about 1 GPU-hour of this loop's 30: two trainings of 21 minutes at the measured 316 ms per update, scoring, the calibration, and eleven rows of 320 canvases with up to three gradients per pushed step. **Buys:** step V5, the first measured answer to whether one checker controls one property.

**What would surprise.** Removing the right-digits or order checker lowering its own column even though the checker reads near chance. That would mean a weak checker's gradient still carries the property's direction.

## The rerun (board question 73)

The first run's verdict was inconclusive: the all-three row's total push passed 0.6 of the product's norm on 5.85% of pushed steps, and cells after removing count held as few as 16 canvases. The owner approved one rerun with the bars unchanged.

**Push length.** Chosen on the 10 seen pairs at seeds 0 to 31 before any unseen pair is read: the largest of 0.229, 0.215, 0.2, 0.185, 0.17 and 0.155 whose all-three row passes 0.6 on at most 5% of pushed steps. The same length is used for every checker in every row (`tune-b` stage, `tune-b.json`).

**Seeds.** 1000 to 1095 on the 10 unseen pairs, 960 canvases per row, three times the first run (`final3` stage, `final3.json`). The checkers, their training and every other setting are the first run's.

**Cost:** about 0.5 GPU-hours: up to six all-three rows on the seen pairs, then eleven rows of 960 canvases. **Buys:** a verdict on V5 under the written rules.

The first run's table stays in `review.md` as recorded, and the rerun's verdict is the one V5 reports.

## What this cannot say

Binding, position and size do not exist on a 32 by 64 digit canvas, and order is left or right only. A pass here is about digits; the SDXL version waits for the owner's decision after V4 passes.
