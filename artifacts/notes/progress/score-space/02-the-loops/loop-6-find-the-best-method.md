# Loop 6: race each paper's method against its cheapest baseline, then freeze one recipe

Loop 6 searches tuning data for the best method, checker, labels and settings for both papers, and freezes one recipe per paper before any benchmark run ([the session brief](../../prompts/session-6-find-the-best-method.md)). It serves the separate-experts paper and the CO3 paper. Its owner code is `s6`, and on 2026-11-24 `python3 scripts/progress_board.py show` gives its steps as M1 done, M7 doing, M5 todo (ready), M9 todo (ready), D3 todo, M2 todo, D5 todo, M3 todo, M6 todo, M8 todo and M4 todo.

## The idea

Each paper's method is first raced against the cheapest thing a reviewer would try ([the M1 design](../../method-search/01-beat-the-baseline/design.md)).

**The separate-experts paper.** Two MNIST diffusion models are trained apart: `e04-either` draws digits 0 to 4 and `e59-either` draws digits 5 to 9, each on either side of a 32 by 64 pixel canvas. The cheapest baseline is sample-and-paste: each model draws its digit alone, and the half holding more ink is cut from each and placed on opposite sides of one canvas. The method lets the two models choose their sides over the first 10 of 50 sampling steps, then adds the checker's push, the gradient of a small classifier that reads whether the canvas holds two clear, right digits apart. The real example is the five tuning pairs 1 and 9, 3 and 6, 4 and 7, 2 and 8, 0 and 5, at seeds 0 to 63.

**The CO3 paper.** One SDXL model draws the joint prompt, such as "a puma and a condor". The cheapest baseline is CO3 as released. The method adds the kept push to CO3: strength 30 on sampling steps 0 to 15, capped at 0.4 of CO3's guided prediction. The kept push's own count of 54 of 128 cat and dog pictures is on the [void list](../../void-list.md), so it justifies no choice here. On seed 6001, CO3 alone draws a bird with a puma's face, and CO3 with the push draws a clear puma beside a clear dark bird. The review says this one picture shows the code runs and is no evidence for either side of the bar ([the M1 review](../../method-search/01-beat-the-baseline/review.md#the-smoke-picture)).

**The narrower questions that follow.** Each later step changes one thing and keeps the rest:

| Step | What it asks | Design |
|---|---|---|
| M7 | Does pasting small CLEVR experts, each trained on one object at one position, already reach 7 objects? | [design](../../method-search/01c-clevr-paste-first/design.md) |
| M5 | Does a checker trained on a list balanced across pairs, with cat and dog cut from 60% of clean draws to 0.76%, push better? | [design](../../method-search/01b-balance-the-checker-data/design.md) |
| D3 | Should the checker read SDXL's features, the noisy latent itself, or the word attention maps? | [design](../../reproductions/checker/design.md) |
| M9 | Does letting each subject's prediction act only inside its own region fix which body is which? | [design](../../method-search/04-bind-each-body/design.md) |
| D5 | Does a short fine-tune on inclusive captions make the push stronger on cat and dog? | [design](../../reproductions/inclusive-finetune-and-push/design.md) |
| M2, M3, M6, M8, M4 | Pair-aware or per-subject labels; tuning; a small CLEVR test; Loop 6's checker against Loop 8's count checker; the frozen recipes | no design file yet |

## The principle

**A method earns its place only by beating the cheapest baseline.** The M1 design states it: "Each paper needs a method that does more than the cheapest thing a reviewer would try first." For experts trained apart that is pasting, and for one SDXL model it is CO3, the rival the CO3 paper has to beat, as [the D3 design](../../reproductions/checker/design.md#the-selection-rule-fixed-before-any-candidate-trains) records from the owner's answer to board question 57.

**The push assumes the checker's "yes" is a faithful pair.** The checker is "a small classifier whose gradient pushes the sampler toward two separate, correct objects" ([the M1 design](../../method-search/01-beat-the-baseline/design.md#words-this-uses)). The push can only help where the classifier's accepted region at a noisy step matches what a person calls two clear, separate subjects, and where its gradient at steps 0 to 15 points toward that region. The size-matched random push in every comparison tests the second half: a random direction of the same length should do nothing.

**A checker should learn what a separate pair looks like.** M5 rests on the concern that batches about 60% cat and dog taught the checker "those two animals more than it learned what a separate pair looks like" ([the M5 design](../../method-search/01b-balance-the-checker-data/design.md)). D3's unseen-category test states the same assumption: "A reader that learned 'two separate things' should carry over, and a reader that learned the species it saw should not" ([the D3 design](../../reproductions/checker/design.md#the-unseen-category-test)).

**Regions should fix identity once bodies exist.** CO3 drew two separate bodies in all 96 tuning pictures, and its failures were borrowed parts, wrong species and extra subjects. M9 assumes that if each subject's prediction acts mainly inside its own region, each body takes its own subject ([the M9 design](../../method-search/04-bind-each-body/design.md)).

**Pasting already succeeds when objects do not interact.** M7 assumes objects succeed independently, so a paste rate per object predicts the most objects paste can hold ([the M7 design](../../method-search/01c-clevr-paste-first/design.md#the-bar-fixed-before-the-first-run)).

## The theory, with its equation

**Where the push enters CO3.** The M1 design writes CO3's step as

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat{x}_0(\epsilon_g) + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon_u
$$

and says "The push is subtracted from both, after CO3's corrector has run on that step" ([the M1 design](../../method-search/01-beat-the-baseline/design.md#paper-2-co3-with-the-push-against-co3-alone)). The code writes the guided prediction with the push $p$ as $(u - p) + g\,((c - p) - (u - p)) = u + g\,(c - u) - p$, and builds $p$ as below ([`co3_push.py`](../../../../../2-classifier-guidance/scripts/method_search/co3_push.py), docstring and `Pusher.__call__`; the typeset form is written from those lines):

$$
p_t = \min\!\left(1,\ \frac{0.4\,\lVert \epsilon_u + g\,(\epsilon_c - \epsilon_u)\rVert}{\lVert s\,\sigma_t\,\nabla_{x_t}\log C(\text{both}\mid x_t)\rVert}\right) s\,\sigma_t\,\nabla_{x_t}\log C(\text{both}\mid x_t), \qquad \sigma_t = \sqrt{1-\bar\alpha_t}
$$

for sampling steps 0 to 15, and $p_t = 0$ after.

- $x_{t-1}$: the latent after one step. It is what the push finally moves.
- $\bar\alpha_{t-1}$: how much of the clean picture survives in the latent at the next step. It is near 0 in the early steps the push acts on, so the $\epsilon_u$ term carries most of the step there (inferred).
- $\hat{x}_0(\epsilon_g)$: CO3's guess of the clean picture from its guided prediction. Lowering $\epsilon_g$ by $p$ moves this guess.
- $\epsilon_g$, $\epsilon_u$, $\epsilon_c$: CO3's guided, unconditional and joint-prompt noise predictions. Subtracting $p$ from both $\epsilon_g$ and $\epsilon_u$ makes the step move as the kept recipe's DDIM step does, where one prediction fills both roles.
- $g$: CO3's guidance weight, 0.8 under CFG++. It scales the cap's reference length.
- $s$: push strength, 30. Doubling it doubles the push until the cap fires.
- $C(\text{both}\mid x_t)$: the checker's probability that the noisy latent will become the asked pair. Its gradient is the push's direction; a random push keeps the length and replaces the direction.
- $0.4$: the cap. When the raw push is longer than 0.4 of the guided prediction it is shrunk to that length. In M1's smoke render the cap fired on 2 of the 16 pushed steps ([the M1 review](../../method-search/01-beat-the-baseline/review.md#paper-2-controls)).

**The regional sampler in M9.** The M9 design writes it as

$$
\epsilon_u + g\,[m_A(\epsilon_A-\epsilon_u) + m_B(\epsilon_B-\epsilon_u)]
$$

and for rung 2 replaces CO3's joint-prompt prediction from step 10 by $m_A\epsilon_A + m_B\epsilon_B$ ([the M9 design](../../method-search/04-bind-each-body/design.md#words-this-uses)).

- $m_A$, $m_B$: soft weights over SDXL's 128 by 128 latent grid that sum to 1 at every cell. Where $m_A$ is 1, only subject A's prediction acts. Masks of 0.5 everywhere reduce the sampler to the plain product at half the guidance.
- $\epsilon_A$, $\epsilon_B$: the predictions for each subject's own prompt. Swapping them puts each subject in the other's region.
- $g$: guidance, 7.5 in rung 1.

**The paste prediction in M7.** "Bradley's rule, with objects assumed independent, predicts 7 objects once this rate reaches 0.906, because $0.906^7 \approx 0.50$" ([the M7 design](../../method-search/01c-clevr-paste-first/design.md#the-bar-fixed-before-the-first-run)). The 0.906 is the in-place rate per object, the share of asked objects the scorer finds at their positions on a 0-to-1 scale. Raising it to the 7th power gives the chance all 7 land, and Bradley's rule needs that chance to reach 0.5.

## The objective, with its bar

Two counts carry the comparisons, and higher is better for both.

**Clean digit pairs of 320**, for the separate-experts side: canvases the digit judge `judge-v3` calls "two clear, right, apart": exactly two blobs, one in each half, read as the asked digits with at least 0.9 confidence each. 320 is 5 tuning pairs times 64 seeds.

**Faithful pairs of 96**, for the CO3 side: pictures on the `two_clear` rung of the [blind count rubric](../../method-search/01-beat-the-baseline/blind-count-rubric.md) with `damaged` 0, called by eye at full size without knowing the method. 96 is 12 tuning pairs times seeds 6001 to 6008.

**M1, separate-experts side.** Row 1 is sample and paste, row 2 the method with no push, row 3 the method with the push at strength 10, capped at 0.1 of the prediction, row 4 a size-matched random push. Copied from [the M1 design](../../method-search/01-beat-the-baseline/design.md#paper-1-the-method-against-sample-and-paste):

| Result | What it takes |
|---|---|
| Support | row 3 minus row 1 is more than 1.96 standard errors above zero |
| Null | the difference is within 1.96 standard errors |
| Loses | row 1 minus row 3 is more than 1.96 standard errors above zero |
| Inconclusive | the zero-push check fails, or row 4 beats row 2 by more than 1.96 standard errors |

**M1, CO3 side.** Row 1 is CO3 alone, row 2 CO3 with the push, row 3 CO3 with a random push. Copied from [the M1 design](../../method-search/01-beat-the-baseline/design.md#paper-2-co3-with-the-push-against-co3-alone):

| Result | What it takes |
|---|---|
| Support | row 2 leads row 1 by at least 5 of 96 and by more than 1.96 standard errors |
| Null | neither side has that lead |
| Loses | row 1 leads row 2 by at least 5 of 96 and by more than 1.96 standard errors |
| Inconclusive | the zero-push check fails, or row 3 leads row 1 by at least 5 of 96 and by more than 1.96 standard errors |

**M7, the paste rate on 1,024 pasted CLEVR images with 3 experts.** Copied from [the M7 design](../../method-search/01c-clevr-paste-first/design.md#the-bar-fixed-before-the-first-run):

| Result | What it takes | What happens |
|---|---|---|
| Paste predicted to reach 7 | paste rate at least 0.906 | M7 stops, and the owner gets a replacement test where pasting fails, such as objects close enough to overlap or shade each other |
| Paste falls short | paste rate below 0.906 | step M6 runs on these models |
| Inconclusive | an expert misses 90%, the scorer misses 99%, or the window control fails | the failure is reported, and no verdict is read |

The window control, also fixed before any sample: "On 256 real single-object renders per position, at least 95% of the pixels that differ from the mean empty scene by more than 0.1 on a 0-to-1 scale must fall inside the window."

**M9 rung 1, clear pairs of 8 cat and dog seeds, "written on 2026-11-04".** Copied from [the M9 design](../../method-search/04-bind-each-body/design.md#rung-1-do-the-right-regions-fix-identity-at-all):

| Result | What it takes | What happens |
|---|---|---|
| Support | right boxes reach at least 4 clear pairs of 8, and random boxes reach fewer | rung 2 runs |
| Layout content irrelevant | right boxes reach at least 4, and random boxes are within 1 | rung 2 runs, and the review says the regions' place did not matter |
| Regions without names | right boxes reach at least 4, and swapped or same prompt is within 1 | rung 2 runs, and the review says which prompt a region carries did not matter |
| Inconclusive | right boxes reach 2 or 3 | the idea stops at this rung |
| Null | right boxes reach 0 or 1 | held until the two sampler checks below have run, because a null from this composer alone cannot separate "regions are not enough" from "the composer is wrong" |

**M9 rung 2, faithful pairs of 96.** Before any render, "If more than 25% of reads fail, rung 2 stops at its masks and renders nothing." Then, copied from [the M9 design](../../method-search/04-bind-each-body/design.md#rung-2-can-co3-find-its-own-regions):

| Result | What it takes |
|---|---|
| Support | own masks lead CO3 alone by at least 5 and by more than 1.96 standard errors, and random masks do not |
| Null | own masks have no such lead |
| Regions do nothing | random masks lead CO3 alone by as much as own masks do |
| Inconclusive | the zero-binding check fails |

<details>
<summary>The bars for M5, D3 and D5, which have no result yet</summary>

**M5, faithful pairs of 96, where the CO3 rows decide.** Row 5 is CO3 with the current-list push and row 6 CO3 with the balanced-list push. Copied from [the M5 design](../../method-search/01b-balance-the-checker-data/design.md#the-bar-on-faithful-pairs-of-96):

| Result | What it takes | Which list goes to D3 |
|---|---|---|
| Balanced wins | row 6 leads row 5 by at least 5 and by more than 1.96 standard errors | balanced |
| Current wins | row 5 leads row 6 by at least 5 and by more than 1.96 standard errors | current |
| Tie | neither lead holds | balanced, because it depends least on cat and dog, unless row 6 has at least 5 more damaged pictures than row 5 |
| Inconclusive | a zero-push check fails | none until the check passes |

**D3, the selection rule.** Copied from [the D3 design](../../reproductions/checker/design.md#the-selection-rule-fixed-before-any-candidate-trains): a candidate is eligible only if a zero push reproduces plain PoE and CO3's own loop under 1e-5, "It rejects at least 80% of 200 neither-subject pictures", and "Its size-matched random push on plain PoE makes at most 2 faithful pairs of 96." Then "Among eligible candidates, the most faithful pairs of 96 on the CO3 row wins. A lead under 5 counts as a tie." "The chosen candidate beats CO3 alone when it leads by at least 5 of 96 and by more than 1.96 standard errors, from 10,000 bootstrap resamples of the 12 pairs."

**D3, the unseen-category test, faithful pairs of 48 on six object pairs.** Copied from [the D3 design](../../reproductions/checker/design.md#the-unseen-category-test):

| Result, per reader | What it takes |
|---|---|
| Transfers | its push leads plain PoE by at least 3 and by more than 1.96 standard errors |
| Does not transfer | its push has no such lead |
| Inconclusive | its zero-push check fails, or its random push leads plain PoE by at least 3 and by more than 1.96 standard errors |

**D5, faithful pairs of 32 cat and dog seeds.** Copied from [the D5 design](../../reproductions/inclusive-finetune-and-push/design.md#what-each-result-means-written-before-the-first-render):

| Outcome | What it takes | What the paragraph may say |
|---|---|---|
| Support | B with the push has at least 5 more than A with the push, and more than B without the push | an inclusive fine-tune made the push stronger on this one pair |
| Null | B with the push has no more than A with the push | no gain was measured from inclusive captions |
| Inconclusive | B with the push has 1 to 4 more than A with the push, or the support test's second half fails | the idea remains untested at this scale |

</details>

## Where it stands

| Step | Board state | Verdict file | What it records |
|---|---|---|---|
| M1 | done | [M1 review](../../method-search/01-beat-the-baseline/review.md) | null on both sides |
| M7 | doing | [M7 review](../../method-search/01c-clevr-paste-first/review.md) | "M7 has no verdict yet"; the window control missed |
| M9 | todo, ready | [M9 review](../../method-search/04-bind-each-body/review.md) | rung 1 "regions without names"; rung 2 stopped at its masks |
| M5 | todo, ready | none | design only, no result |
| D3 | todo | none | design only, no result |
| D5 | todo | none | design only, no result |
| M2, M3, M6, M8, M4 | todo | none | no design file and no result |

**M1, separate-experts side: null.** The method with the push draws 295 clean digit pairs of 320 and paste draws 287. "The lead of 8 is under the bar of 13.54, which is 1.96 times its standard error of 6.91." The zero-push check passed (largest absolute difference 0.0 against 1e-5). W&B `vekm3sfh`. The review adds leads read after the run, which carry no bar: the checker push over no push is +36 of 320 (margin 15.61), over a random push +37 (margin 15.47), and no push against paste is -28 (margin 16.57). Its reading is that the checker's direction lifts the method level with paste, and without the push the side-choosing method is 28 pairs below it.

| Row | Clean digit pairs of 320 |
|---|---:|
| Sample alone and paste | 287 |
| Method, no push | 259 |
| Method with the checker push | 295 |
| Method with a random push | 258 |

**M1, CO3 side: null.** CO3 with the kept push makes 7 faithful pairs of 96 and CO3 alone makes 11. "A win needed a lead of at least 5 and more than 10.08, which is 1.96 times the standard error of 5.14. The lead is -4." The random push makes 12. W&B `s7o6404z`. Read after the run: with drawn style ignored, two clear subjects are 32, 30 and 35 of 96 for CO3, push and random, a push deficit of 2 against a margin of 11.25. None of the 288 pictures is one subject or a fused hybrid, so CO3 already separates the bodies on these pairs, and its remaining failures are borrowed parts, wrong species or an extra subject. Eight of the 12 pairs make 0 faithful pairs in all three rows. The kept checker's two presence outputs trained only on cat and dog, so this result covers the kept push as it stands.

**M7: no verdict.** The scorer reached 100% at all three positions on 400 held-out renders, against 99% (W&B `guf6vtho`). The position 1 expert trained on 2,000,896 images, and its first 16 samples each hold one clear object at position 1 (W&B `zv6oef8h`); the 256-sample expert check is still to run. The window control misses its 95% mark at all three positions: 94.4%, 93.9% and 91.7% of changed pixels fall inside the 18 by 18 window. No sample has been scored. Board question 66, open, asks the owner how to fix the window, with the default that M7 gives no paste verdict until answered.

**M9 rung 1 without bootstrapping.** Regions fused from step 0 drew 0 clear pairs of 8 on right, random, swapped and same-prompt boxes, against 7 for the joint prompt. Right boxes drew 6 fused cat-dogs and 2 single animals.

**M9 rung 1 with bootstrapping: "regions without names".** Each region was denoised over a constant colour for the first 15 of 50 steps.

| Row | Clear pairs of 8 | Two separate animals, of 8 | Marked undamaged by both reviewers, of 8 |
|---|---|---|---|
| Joint prompt | 6 | 8 | 5 |
| Right boxes | 8 | 8 | 0 |
| Random boxes | 4 | 4 | 0 |
| Swapped prompts | 7 | 8 | 1 |
| Same prompt twice | 0 | 8 | 1 |

The review says bootstrapping separated the animals and "leaves cut-outs on flat grey", and no right-box picture passed as undamaged.

**M9 rung 2: stopped at its masks.** The zero-binding control passed (0.0 against 1e-5). The step-10 word-attention boxes failed on 11 of 24 reads against a limit of 6, 7 on overlap and 4 on box size, with a median overlap of 0.53, so no rung 2 picture was rendered and the comparison with CO3 alone on faithful pairs of 96 has not been made. Board question 65, open, offers a new region reader, boxes from M1's finished CO3 pictures, or stopping M9.

**The M9 label and its own pictures disagree.** The review records the written verdict "regions without names", which under the bar means "which prompt a region carries did not matter". The same review reports that "The same-prompt row drew two cats and no dog on all 8 seeds, and the swapped row drew a cat and a dog. So the prompt a region carries does decide its species." Both stand as recorded ([the M9 review](../../method-search/04-bind-each-body/review.md#rung-1-count-with-bootstrapping); `artifacts/notes/progress/requests.md`, the s4 line of 2026-11-23 on this reading).

**The board and the M9 review disagree on state.** The board lists M9 as todo, while the review holds rung 1's verdict and rung 2's stop. The board's proof line asks for rung 2 against CO3 alone, which does not exist yet, so the two may both be right; this page does not resolve it.

**Two older sources lag the board.** The board's M7 note still reads "Loop 6 started: reading the CLEVR protocol to write the paste-first design", which predates the M7 review's runs. The [route picture](../../method-search/loop-6-route.png) marks M1 as the current step and has no box for M7, M8 or M9.

## The closest papers from the reading

**Reduce, Reuse, Recycle.** It composes CLEVR models each trained on one cube location and guides samples with a noise-conditioned classifier's gradient, the same two moves as M7's one-object experts and the checker push. A score model with plain reverse diffusion places all 5 cubes in 57.4% of samples, and U-HMC lifts that to 62.3% at equal score evaluations ([its note](../01-reading/reduce-reuse-recycle.md); [Table 2, p. 6](https://arxiv.org/pdf/2302.11552#page=6); classifier guidance at [Eq. 14, p. 7](https://arxiv.org/pdf/2302.11552#page=7)).

**Composable Diffusion.** It adds conditional noise-prediction differences to compose CLEVR positions, reports fused hybrids "when the objects are in the center of the images", and reports "limited success" composing models trained on different datasets, the risk the separate-experts paper carries ([its note](../01-reading/composable-diffusion.md); [p. 13](https://arxiv.org/pdf/2206.01714#page=13)).

**Diffusion Sampling Path Tells More.** It ranks seeds by the summed squared conditional-minus-unconditional length over the early steps, with no training, and keeping 4 of 50 seeds lifts GenEval two-object on SD1.5 from 0.3540 to 0.5177 (fraction of prompts with both objects, 0 to 1). That is a cheap rival of the kind M1 races against, acting on the same early window as the push ([its note](../01-reading/sampling-path-tells-more.md); [Table 2, p. 8](https://arxiv.org/pdf/2505.23343#page=8)).

**SuperDiff.** It composes two models trained on disjoint CIFAR-10 halves ([p. 8](https://arxiv.org/pdf/2412.17762#page=8)), and its AND mode on "a dog" and "a cat" draws one fused animal per tile ([p. 40](https://arxiv.org/pdf/2412.17762#page=40)), the failure the blind count scores as `hybrid` ([its note](../01-reading/superdiff.md)).

## What it suggests for the separate-experts paper

On MNIST the push does real work and still only ties pasting: 295 against 287 clean digit pairs of 320, with the push worth 36 pairs over no push ([the M1 review](../../method-search/01-beat-the-baseline/review.md#questions)). This suggests keeping the push, since the side-choosing method without it is 28 pairs below paste, and resting any claim on a test where pasting fails, which is what M7 and M6 exist to find (inferred). The owner kept this paper in the search through M3 and M6 on board question 52, and M7 cannot read its paste rate until board question 66 settles the window. If paste reaches 0.906 per object on CLEVR, the replacement test with overlapping or shading objects becomes the paper's only measure that separates it from pasting (inferred). A score-path representation could add a checker input that exists for each expert without any joint image: each expert's $d_t$ on the shared canvas over steps 0 to 15, the window the push already acts on (inferred). Seed picking by early score-path length, as Sampling Path Tells More does, is a no-training rival at matched compute that M1 did not include (inferred).

## What it suggests for the CO3 paper

On the 12 tuning pairs CO3 already draws two separate bodies in all 96 pictures, and adding the kept push gives 7 faithful pairs of 96 against CO3's 11 ([the M1 review](../../method-search/01-beat-the-baseline/review.md#questions)). This suggests dropping a push that rewards separation alone and aiming the checker at identity, the borrowed parts, wrong species and extra subjects that remain, which is where M2's choice of labels and the CO3 rows in M5 and D3 now decide (inferred). M9 rung 1 shows that the prompt each region carries decides its species once bodies exist, but only with bootstrapping and with every right-box picture damaged, and rung 2's test on CO3 has no pictures yet ([the M9 review](../../method-search/04-bind-each-body/review.md)). A score-path representation could supply what the step-10 word-attention reader could not: each subject's own $d_t$ at CO3's step 10 is a map of where that subject's prediction acts, and checking whether one subject's map covers both bodies would test the same failure the word-attention boxes showed on 11 of 24 reads (inferred). A checker reading score paths could also be scored on whether it tells a borrowed-part pair from a faithful pair, since that is the distinction left on CO3 (inferred).
