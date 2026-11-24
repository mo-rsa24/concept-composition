# Remove one checker at a time, with a checker that reads identity (step V7)

## What this asks

Does removing one checker damage only its own property, once the right-digits checker actually reads which digits are present? [V5](../08-remove-one-checker/review.md) answered null by its written rule. Its right-digits checker read near chance (0.556), removing it barely moved anything, and removing count or order each moved another column by about 7 points. [V6](../10-reading-identity/review.md) found a checker that reads identity on unseen digits 5 to 9 at 0.861. V7 reruns V5 with that checker in the right-digits place. The owner chose this on 2026-11-24 (option A: the identity grid, then the V5 rerun), and it runs only if V6's two remaining robustness checks pass.

## Words this uses

**The three checkers.** Count (two digits sitting apart?), right digits (are they the two asked digits?) and order (is the asked left digit on the left?), as in [V5's design](../08-remove-one-checker/design.md#what-each-checker-is).

**Fully right.** A canvas the judge calls clean (two confident, right digits, one per half) with the asked digit on the left.

**Count window.** Sampler steps 15 to 25 of 50 (timesteps 693 down to 489), where count and order push, as in V3 and V5.

**Identity window.** Sampler steps 27 to 36 of 50 (timesteps 449 down to 265), the sampler's steps that fall inside V6's later window (timesteps 466 to 263), where the right-digits checker reads at 0.86 to 0.92.

## What changes from V5, and what stays

| | V5 rerun | V7 |
|---|---|---|
| Right-digits checker | two maps, early window, `v5/identity-head-map.pt`, 0.556 on digits 5 to 9 | ten-label signature, later window, `v6/identity-head.pt`, 0.861 on digits 5 to 9 |
| Where it pushes | the count window | the identity window |
| Count checker | `v2/count-final/count-head-map.pt` | same |
| Order checker | `v5/order-head-map.pt`, count window | same |
| Rows, seeds, judge, bars | eleven rows, seeds 1000 to 1095, 960 canvases per row | same |

One thing differs between the two runs: which right-digits checker is used. Its input and its window change together, because V6 chose them together, so V7 against V5 says whether the V6 checker changes the verdict and cannot say which of its two differences did it.

## Run settings

**Group.** Ablation (group 7): one checker removed per row, everything else held.

**Varied axis.** Which checker is off.

**Model and sampler.** The `share-0.0` model; DDIM with 50 steps and eta 0, the product of the two asked labels at guidance 1, `cap=1.0`, as in V5.

**Push.** Each active checker adds $-\lVert\epsilon_{\text{product}}\rVert\, b\, g_k/\lVert g_k\rVert$ at the steps of its own window. Count and order share the count window; the right-digits checker pushes alone in the identity window. The right-digits gradient runs back through the eleven U-Net calls of its signature. A random push that stands in for a removed checker acts in that checker's window.

**Push length.** Chosen as V5's rerun chose it, on the 10 seen pairs at seeds 0 to 31 before any unseen pair is read: the largest of 0.229, 0.215, 0.2, 0.185, 0.17 and 0.155 whose all-three row passes 0.6 of the product's norm on at most 5% of pushed steps. The same length is used for every checker in every row. A row is void if its total push passes 0.6 on more than 5% of pushed steps.

**Asked order.** The lower digit is asked on the left for even seeds and on the right for odd seeds.

**Rows.** Eleven rows on the 10 unseen pairs among digits 5 to 9 at seeds 1000 to 1095, 960 canvases per row: the plain product; all three checkers; each removed (3); each removed with a random push of the same length in its place (3); each alone (3).

**Score.** The digit judge `judge-v3/judge.pt` through `two_experts.call_two` and `two_experts.reads`. Standard errors come from 10,000 bootstrap resamples of the 10 pairs.

**The bars.** V5's, unchanged, in `v5_remove.py` as `OWN_DROP = 6.25`, `OTHER_MOVE = 4.2` and `CELL_MIN = 40`:

| Result | What it takes |
|---|---|
| Support | All three together beat the plain product on fully right canvases by more than 1.96 standard errors. Each removal lowers its own column by at least 6.25 percentage points and more than 1.96 standard errors, its random-push row falls by the same rule, and no other column moves by both more than 4.2 points and more than 1.96 standard errors |
| Partial | At least one property meets that rule and at least one does not |
| Null | No property meets the whole rule above, including a removal that lowers its own column by the margin but moves another column past the limit (board question 75) |
| Inconclusive | The zero-push check fails, a row is void by the 0.6 total-length rule, or a cell the verdict needs holds fewer than 40 canvases |

**What would surprise.** Removing the right-digits checker lowering the count column. The identity window starts after the count window ends, so a drop in count would mean late pushes still move the layout.

**The ladder.** A smoke test of every row on one pair at 4 seeds; the gradient's memory on a 96-seed batch checked before the full run; the zero-push check and the push length on the seen pairs; then the eleven rows on the unseen pairs, once.

**Cost:** about 1.5 GPU-hours of the loop's 30: the push length on the seen pairs, and eleven rows of 960 canvases, six of them with a gradient through eleven U-Net calls on 10 steps. V5's rerun took 20 minutes for its eleven rows. **Buys:** whether one checker controls one property once all three checkers can read their property.

## What this cannot say

Binding, position and size do not exist on a 32 by 64 digit canvas, and order is left or right only. The identity window is later than the count window on purpose, so V7 tests a split where count acts first and identity second; whether the two could share one window is not tested.
