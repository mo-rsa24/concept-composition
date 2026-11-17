# Retrospective: a classifier's push on the exact toy (route step 7)

The walk ported guided-diffusion's push into the exact circle-and-square toy and tested two classifiers on it: one trained on "circle present" and "square present", and one trained to tell exact side-by-side draws from the summed-score sampler's own stacked draws. At pair weight 0.01, seed 7, 300 draws, the second gives 172 crisp pairs at push weight 1 and 300 at weight 2, against 0 unguided. The first gives at most 10. Both halves of the bar pass. The plan was [plan 19](../../../plans/partial/19-classifier-push-on-the-exact-toy.md), and the record is the commits and [A1's page](../../../report/classifier-guided-poe/a1-exact-toy-classifier-push.md).

## Piece 1: the push hook

**The request.** "Port the push from guided-diffusion's condition_score into the toy's sampler, and check that scale 0 reproduces the unguided draws exactly."

**The problem.** The toy's `reverse()` had no per-step hook and drew its noise from a module-wide generator, so two runs could not share starting noise.

**What was done.** `reverse()` gained `push`, `scale`, `rng` and `n`; the push enters through `guided_eps`, typed by Claude at the reader's word as `eps - scale * np.sqrt(1 - a_t) * grad`. One run differed from unguided by 6.2e-8 on 1 draw of 300. Three unguided runs, a zero push and the true correction at scale 0 were bitwise identical in a fresh process, so the gap was numpy's summation order. At the reader's `fix`, the check became bit for bit inside the same loop. Commits `50a1b48b`, `10a88986`, `32c264c5`.

**Where the results live.** `port.json`, `draw-0-unguided-and-pushed.png` in the [card's folder](../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/README.md).

<a href="../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/draw-0-unguided-and-pushed.png"><img src="../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/draw-0-unguided-and-pushed.png" width="400" alt="Draw 0 unguided as one purple blob, the same at scale 0, and a red circle beside a blue square with the true correction at scale 1"></a>

**Before and after.** Bit for bit on 200 of 200 steps at scale 0. The true correction as the push: 0, 112 and 300 crisp of 300 at scales 0, 0.5 and 1; scale 1 matches the true-score sampler to 6.7e-16.

**Links to.** Every later piece guides through this hook.

## Piece 2: the separate-label classifier

**The request.** "Train a separate-label classifier, circle present and square present, on noised toy pictures."

**The problem.** The maths says this push only re-weights the two experts. Whether re-weighting can still make pairs depends on what the classifier saw.

**What was done.** A two-output MLP trained 4,000 steps on draws from the seven pictures at the experts' weights, beside the closed-form best classifier; its push at scales 1, 2, 5 and 10. Commit `32c264c5`, test run `89af6207`.

**Where the results live.** `separate-label-train.json`, `separate-label-guide.json`, `what-the-separate-label-classifier-says.png`.

**Before and after.** Accuracy within 0.012 of the best possible at every checked step. Crisp of 300: trained 0, 0, 2, 10; best possible 0, 1, 13, 34. The side-by-side picture is the one training picture carrying both labels, so sharpening both labels pulls toward it.

**Links to.** Piece 1's hook; the control the pair-aware push is read against.

## Piece 3: the pair-aware discriminator

**The request.** "Train a pair-aware discriminator: exact pair draws against the summed-score sampler's own draws, at every noise level, with the step 3 recipe."

**The problem.** The fake pile has to be the sampler's finished draws noised forward (Kim et al., p15), not its half-finished states.

**What was done.** 3,500 exact draws (seed 101) and 3,500 sampler draws (seeds 201 to 205: 3,470 overlaid, 30 faint), never seed 7. A one-output MLP, binary cross-entropy weighted by β_k / (1 − ᾱ_k). The loss and the log-odds gradient were typed by Claude at the reader's word. The derivation of both lines went into the classifier-guidance journey's chapter 3. Commits `640a9af1`, `76eeea91`.

**Where the results live.** `discriminator-data.npz`, `pair-aware-train.json`, `pair-aware-discriminator.pt`, `the-side-by-side-discriminator.png`.

**Before and after.** Held-out accuracy 1.000 on both piles at steps 10, 50 and 100; 0.956 and 0.980 at step 150; gradient against a finite difference 3.3e-4.

**Links to.** Piece 1's hook carries its push in piece 4.

## Piece 4: guiding with the pair-aware push

**The request.** "Guide the sampler with each push and count crisp pairs and overlays of 300."

**The problem.** None beyond the bar, read at w = 1 as fixed at the open.

**What was done.** `pair-guide` at w = 0.5, 1 and 2, every step, uncapped, the logit clipped at ±11.5 as Kim et al. Commits `cefcf06e`, `06bd9d94`.

**Where the results live.** `pair-aware-guide.json`, `pair-aware-guide-samples.npz`.

**Before and after.** Crisp of 300: 0 unguided, 16, 172, 300. Overlaid at w = 1: 128 against 300. Bar line `pass`.

**Links to.** Pieces 1 and 3.

## Piece 5: the diagnosis

**The request.** "Read each push's cosine per step against the true correction and against the sampler's own ratio, and run the minimal-pair gap beside the sweep."

**The problem.** The first version of the sampler's-ratio score asked for 12 GB on an 8 GB card; it was rewritten with matrix products.

**What was done.** `diagnose`, then two runs beside the bar at the reader's word: `no-clip` and `inclusive`. Commits `3052b828`, `5ccb723f`, `025175fd`.

**Where the results live.** `diagnosis.json`, `push-cosine-per-step.png`, `minimal-pair-gap-per-step.png`, `pair-aware-no-clip.json`, `inclusive-captions-comparison.json`.

<a href="../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/push-cosine-per-step.png"><img src="../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/push-cosine-per-step.png" width="400" alt="Cosine of the push with the sampler's own ratio near 0.85 to 0.96 and with the true correction under 0.21, both falling to 0 by step 80"></a>

**Before and after.** Cosine with the sampler's own ratio 0.85 to 0.96 from step 190 to 90; with the true correction at most 0.21. Minimal-pair gap up to 27.9 log-odds, against within 0.003 of 0 for shuffled labels. The push gives 172 at caps 0.1, 0.3 and 1; the true correction gives 4, 229 and 300. Without the clip: 172 and 300, unchanged. Inclusive captions with no push: 0, 1, 18, 53, 115, 246 crisp at pair weights 0.01 to 0.5.

**Links to.** Pieces 3 and 4.

## The stop point

- Bar: pair-aware at w = 1 at least 150 crisp with overlaid no higher than unguided; separate-label at most 15 at every scale. Met: 172 and 128 against 300; highest 10.
- Time box: 4 sittings, about 3 h 45 of the reader's time. Used: one sitting, about 2 h 20 from the first commit (06:44) to the close (09:04), with the background figure and derivation work running alongside.
- The bar arrived first. Way on: ship.

## Decisions

| Decision | Chosen | Options | Who | Why |
|---|---|---|---|---|
| the cap on the toy | none; 0.1, 0.3, 1 reported beside | 0.1 as step 5 · 0.3 | you, by defaults | under 0.1 the true correction itself composes 4 of 300 |
| where the pair-aware bar is read | w = 1, fixed at the open | best of the sweep | you, by defaults | fixed before the run |
| where the separate-label bar is read | every scale 1 to 10 | scale 1 only | you, by defaults | the strictest reading |
| the overlay half | as written, against this walk's scale-0 run | tightened | you, by defaults | your bar |
| who types the three by-hand lines | Claude | you, as the brief said | you ("you do it", three times) | |
| the scale-0 standard | bit for bit in the same loop | float rounding | you (`fix`) | |
| which classifier the separate-label bar is read on | the trained one, the best possible's 34 beside it | both | you (`a`) | the trained classifier is the one the route uses later |
| the stage figure's tool | hld-diagram's figure mode with real draws | prompt-storyboard single | you (`a`), after the recommendation | prompt-storyboard's own rules send maths to script, and an image model redraws the draws |
| the toy's guidance window | every step | Kim et al.'s stop below a tenth | Claude | matches the SDXL sampler |

## Dead ends and not done

- **Test 2.5 is 🕒.** You closed with a bare "done", so your look at the two diagnosis charts is not recorded; the run's numbers match the test line.
- **The clip as the cause of the 128 failures was wrong.** It was stated as inferred in one round and refuted by the no-clip run; the card and A1's page carry the refuted version nowhere.
- **The board, its history and the route's three new rows and gate tick are written but not committed.** Another session has 566 uncommitted lines in the route and edits in the board; committing would sweep its work in.
- **The derivation in the learning repo is not committed** for the same reason there.
- **Plan 19 task 0.1** (verify-plan and xref-pillar) is running in the background; the plan is filed after it returns.
- **Not run:** both methods together on the toy; the effect of dropping the per-step weight; the pairs' positions against the exact product's proportions.
- **No delivery log row:** this project keeps no `report/delivery-log.md`.

## The family table

| Kind | Id | What | State |
|---|---|---|---|
| this walk | `fe46afa6-5882-43dd-b646-45b184e1f2e9` | route step 7, plan 19 | closed |
| background agent | stage figure | `the-side-by-side-discriminator.png`, seven stages | landed |
| background agent | derivation | chapter 3's section on the two code lines, figure `03-two-penalties.png`, plan 06 task 1.6 | landed, not committed |
| background agent | plan check | verify-plan and xref-pillar on plan 19 | running |

---

Validation: 14 paths checked on disk (plan 19, A1's page, the board, the card, `port.json`, `separate-label-guide.json`, `pair-aware-train.json`, `pair-aware-guide.json`, `diagnosis.json`, `pair-aware-no-clip.json`, `inclusive-captions-comparison.json`, the three linked figures), 0 missing; every count checked against its JSON file.
