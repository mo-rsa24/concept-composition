# 🔬 Report: concept-composition

What this project found, one file per question, every claim paired with a figure and the
statistic that backs it. What the work *means* lives in [context/](../context/00-INDEX.md),
how to redo it in [runbook/](../runbook/00-INDEX.md); this folder is only the findings.

## Read it in this order

The table below is a lookup, sorted by nothing. This section is the thread: eight rungs, each one
the question you can only ask because the rung before it was answered. Read it top to bottom once
and the rest of the folder becomes navigable.

A rung marked ⬜ has its verdict and its figures on disk but no finding file written yet, so it
names the review file and the evidence folder instead. Those are the holes, and they are at the
front of the arc rather than the back.

**1. What product-of-experts is for, and when it works**

Multiplying two experts' predictions is supposed to put both things in one picture, and sometimes
it does. The control pair used throughout the project, a butterfly over a meadow, composes on 8 of
8 held-out seeds, and the joint prompt (one sentence naming both animals, no product at all) also
composes 8 of 8. Those two columns are what every failure below is measured against.

⬜ No finding file. The successful columns live inside
[does a corrector alone produce two animals](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md)
as the control, never on their own. What the method is and what a compose rate counts is in
[what product-of-experts composition is](../context/world/poe-composition.md) and
[the compose rate](../context/world/compose-rate.md).

**2. Where it breaks, and how we know it broke**

Ask for a cat and a dog and the product renders one fused animal: a cat's face on a dog's body.
This is the failure the whole project is about, and it is not a one-off. It recurs across 17 animal
pairs at 8 seeds each.

⬜ No finding file. The gallery of 136 uncorrected renders, one per pair per seed with its sampler
settings beside it, is [the blend gallery](../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/poe-blends-instead-of-composing/README.md).
The instrument that decides a render failed, a GroundingDINO instance count validated against
labelled images, is [the compose-score validation set](../1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/),
with its verdict in [what the current benchmarks score](../1-lora-correction/artifacts/plans/parked/02-can-we-trust-the-compose-rate/review/03-what-the-current-benchmarks-score.md).
Read the instrument before any rate below, because every number in this folder is counted by it.

**3. The correction exists, and it is what causes composition**

Subtract the plain product's prediction from the joint prompt's, step by step, and what is left is
the correction. Add it back and the pair composes. Four checks say the effect is the correction and
not the act of perturbing anything: more of it composes more, matched substitutes of the same size
pointing elsewhere do nothing, it only works in an early window, and the same story holds read from
three sides.

⬜ No finding file, three verdicts. The amount question is in
[more correction, more composition](../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/review/03-more-correction-more-composition.md)
with figures under [how much correction is needed](../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/how-much-correction-is-needed/);
the timing question is in [when in the run it matters](../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md)
with figures under [when the correction must arrive](../1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/when-the-correction-must-arrive/);
the direction and substitute controls are in
[the same story from three sides](../1-lora-correction/plans/03-does-the-correction-cause-composition/review/06-the-same-story-from-three-sides.md)
with figures under [does the interaction term cause composition](../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/does-the-interaction-term-cause-composition/five-checks-from-three-sides/README.md).

**4. What the correction is made of, and where it moves the picture to**

Two findings, and the first one written should be read second. Together they answer whether the
product could have supplied this correction by re-weighting predictions it already has (mostly yes
in principle, not at the weights it uses), and whether the correction moves the render toward the
joint prompt in the scorer's own image space (yes on cat × dog, overlapping bands on held-out
pairs).

- [What is the correction made of, and could PoE have supplied it?](../1-lora-correction/report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md) ❓ inconclusive on the pre-registered bar
- [Where does each condition land in the scorer's image space?](../1-lora-correction/report/when-does-the-outcome-lock-in/where-does-each-condition-land.md) ❌ null on both pre-registered bars, ✅ support on the post-hoc reads

**5. An adapter learns the correction, and which one to show**

The correction is computed from a joint prompt the deployed method never gets to see, so it is not
a fix on its own. A low-rank adapter trained to predict it is. These two say which checkpoint
composes best and what happens if you keep training past it.

- [Which checkpoint composes best, and does more correction help?](../3-understanding-compositionality/report/does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md) ✅ 7 of 8 seeds, ⚪ null on more correction and on the early-only window
- [Does training longer keep improving the held-out fix?](../3-understanding-compositionality/report/does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md) ❌ no: fidelity decays while the weights grow without bound

**6. Whether the fix reaches pairs it never trained on**

The adapter is trained on a pool of pairs and asked about pairs outside it. It carries to some and
not others, and these two ask why: is the gap a fit problem, a drift problem or a property of the
pair, and can the cache predict which pairs will blend before you render anything.

- [Is the held-out gap a fit, a drift, or a pair problem?](../1-lora-correction/report/does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md) ✅ pair problem, ⚪ null on drift
- [Does interaction strength predict which pairs blend?](../1-lora-correction/report/does-the-fix-reach-unseen-pairs/does-interaction-strength-predict-which-pairs-blend.md) ⚪ null

⬜ The transfer rate over fifteen pairs has its verdict in
[does one pooled fix transfer at all](../1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/review/03-does-one-pooled-fix-transfer-at-all.md)
and its figures under [does the fix reach unseen pairs](../1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/),
with no finding file.

**7. Everything tried instead, and what each one ruled out**

Twelve findings, and the reason there are twelve is that a reviewer will ask whether a cheaper fix
works. Most are ways of getting two animals without adding a learned correction; the last two in
the search family keep the correction and spend inference compute on top of it, and the last of all
asks not whether a cheaper fix composes but whether any of them repairs the picture quality the
correction costs. They are ordered by family, and ten of the twelve are nulls or inconclusive. The
two that support both buy their result with 16 or more draws per seed, and neither reaches past what
the correction already delivers at full dose, which is the point: the correction is not obviously
replaceable.

*Correctors, which re-sample the latent without changing the score:*

- [Does a corrector alone produce two animals?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md) ⚪ null
- [Does a model with no CLIP in it still fuse a cat and a dog?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md) 🟢 support: it fuses on 8 of 8
- [Does SDXL still fuse when it is sampled the way PixArt-alpha is?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-sdxl-compose-at-pixarts-settings.md) 🟢 support: it fuses on 8 of 8 at PixArt's settings too
- [Does a Langevin corrector remove part of the correction?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md) ❓ inconclusive at every step size
- [Can a corrector or a clean tail sharpen the adapter's renders?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md) ⚪ null on both bars
- [Can any render-time fix recover the picture quality the adapter costs?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-any-render-time-fix-recover-picture-quality.md) ❓ inconclusive: eight fixes, the edge measure passes two of them, the pictures do not

*A different composition method, SuperDiff, and whether its residual is the same object:*

- [Does SuperDiff compose at its own defaults?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-superdiff-compose-at-its-own-defaults.md) ⚪ null at 200 steps, ✅ the missing piece is the same residual
- [Does the PoE-trained correction carry into SuperDiff?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-the-poe-trained-correction-carry-into-superdiff.md) ⚪ null, it hurts
- [Does an adapter trained on SuperDiff's own residual compose?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-an-adapter-trained-on-superdiffs-own-residual-compose.md) ❓ inconclusive, runs stopped early

*Search and selection, which spend compute at inference instead of training anything:*

- [Does selecting among the product's own proposals compose?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md) ❓ inconclusive
- [Does steering on the scorer find a composing proposal?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-steering-on-the-scorer-find-a-composing-proposal.md) ⚪ null
- [Does steering the corrected sampler produce a clean cat and dog?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-steering-the-corrected-sampler-produce-clean-pairs.md) ✅ support at half dose; ⚪ null on what the resampling itself adds
- [Does searching over the initial noise compose?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-initial-noise-compose.md) ❓ inconclusive, a null in everything but its letter
- [Does searching over the noise sharpen the corrected render?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-noise-sharpen-the-corrected-render.md) ✅ support, 8 of 8 seeds

**8. What is still open**

The one result with a verdict, figures and a W&B run and no home in the arc yet is the energy
penalty: charging the adapter for the size of its correction cut its on-policy energy 27% and moved
the held-out renders back toward plain product-of-experts. Its verdict is in the scope 01 review
file, its figures under
[does charging the adapter for its energy sharpen the fix](../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/making-the-render-crisp/does-charging-the-adapter-for-its-energy-sharpen-the-fix),
and the run is `2cfdtdnl`. The rest of the holes are in [Still open](#still-open) at the foot of
this file.

**The illustrated version** is two pictures waiting to be rendered, in
[the report's illustrated map](../1-lora-correction/report/diagram-prompts.md): the eight rungs above as one line, and what a
single corrector step does inside one noise level.

## What was found

| The question | Verdict | The number | The finding |
|---|---|---|---|
| Can real photographs of both animals train the composition instead of the model's own joint-prompt prediction? | ❌ dead: the corpus is reachable and unusable | of the images an automatic filter accepted, 7 of 16 cat × dog are genuine photographs of both animals and 0 of 11 turtle × tortoise are, the rest watermarked stock and merchandise; fed single-species images the same filter reports both species at 0.323 for turtle × tortoise against a 0.10 bar | [05-training-on-real-photographs.md](../1-lora-correction/report/designing-the-correction-loss/05-training-on-real-photographs.md) |
| Can a third expert built in text space supply what the joint prompt adds to product-of-experts? | ⚪ null: no third expert restores identity | over the 7 judgeable cat × dog seeds, one whole cat beside one whole dog: 0 for every one of eight third experts against a pass mark of 4; two separate animals: 0 for plain PoE, 5 for the phrase "a dog next to a cat", 6 for x_pos. Mean absolute pixel difference from the joint prompt's render, 0 to 255: x_pos used alone 23.43, the three-expert product 39.16, plain PoE 47.05 | [can-a-text-space-third-expert-restore-composition.md](../5-text-embedding-third-expert/report/can-a-text-space-third-expert-restore-composition.md) |
| Where does each condition land in the scorer's image space, and does the correction move PoE toward the joint prompt? | ❌ null on both pre-registered bars; ✅ support on the post-hoc reads | cat × dog both-ness (projection toward the joint-prompt centroid, DINOv2 cosine units, mean of 8 seeds): PoE 0.21, corrected 0.42, joint 0.52; on the 8 held-out pairs the both-ness bands overlap on 5 of 7 counted pairs (null), while the instance count on the same images reads PoE 0.00 to 0.25 and corrected 0.75 to 1.00 on every pair (post-hoc) | [where-does-each-condition-land.md](../1-lora-correction/report/when-does-the-outcome-lock-in/where-does-each-condition-land.md) |
| What is the correction made of, and could PoE have supplied it by re-weighting its own predictions? | ❓ inconclusive on the pre-registered bar | orthogonal share of the correction's squared norm outside the span of the three predictions PoE has, mean over seeds 9 to 16 and steps 0 to 10: 0.374 against bars at 0.5 and 0.25; the reachable part weights each expert at 1 to 3 where PoE weights 7.5; the adapter's orthogonal share 0.19 against the target's 0.37 | [what-is-the-correction-made-of.md](../1-lora-correction/report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md) |
| Is the held-out gap of the pooled rank 8 adapter a fit, a drift, or a pair problem? | ✅ support: pair problem; ⚪ null on drift | fit cosine on cached states: training pairs 0.97, seen words in an unseen pairing 0.84, no seen word 0.80; on the adapter's own corrected trajectory 0.82 against 0.86 cached, cat x dog | [is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md](../1-lora-correction/report/does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md) |
| Does the interaction strength read from the cache (Mitra et al. Theorem 1 turned around) predict which pairs blend? | ⚪ null | lower bound on G·M over steps 10 to 40 spans 100 to 3200 across pairs that all fail 8 of 8; near-synonym pairs sit ten times below distinct-animal pairs | [does-interaction-strength-predict-which-pairs-blend.md](../1-lora-correction/report/does-the-fix-reach-unseen-pairs/does-interaction-strength-predict-which-pairs-blend.md) |
| Which checkpoint of the pooled adapter composes cat x dog best, and does more correction or an early-only window help? | ✅ 7 of 8 at rank 8 @ 30k, rank 16 @ 50k, rank 32 @ 30k to 90k · ⚪ null on λ above 1 and on the early window | seeds composing of 8 at λ 1, all 50 steps: 7 / 7 / 7 against 0 for plain PoE; λ 1.2 to 2.0 never adds a seed on a healthy checkpoint, and on the shipped rank-32 step-30050 checkpoint it changes the scene instead (7 / 6 / 6 / 7 / 6 at λ 1.0 / 1.2 / 1.3 / 1.5 / 2.0, with one seed's animals shrinking out of the detector's reach); the early window (steps 0 to 9) reaches at most 5 of 8 | [which-checkpoint-composes-best-and-does-more-correction-help.md](../3-understanding-compositionality/report/does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md) |
| Does training the pooled adapter longer keep improving the held-out renders? | ❌ no: fidelity decays while the weights grow without bound | 8-seed DINOv2 drift at λ 1 (negative = nearer the joint-prompt image): rank 32 −0.091 at 30k to −0.013 at 90k; rank 8 −0.132 at 30k to +0.041 at 280k; LoRA weight norm rank 8 35.9 at 10k to 93.2 at 450k with weight decay 0 | [does-training-longer-keep-improving-the-held-out-fix.md](../3-understanding-compositionality/report/does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md) |
| Does choosing among the product's own proposals, with no correction added, produce two animals? | ❓ inconclusive | compose fraction over all 12 particles at step 100k: Mono 1.0, PoE control 0.0, twisted SMC 0.0 (10 detector hits of 132 SMC particles at 40k to 80k, each one fused animal by eye); the twist's validation accuracy 0.56, under the 0.60 bar | [does-selecting-among-poe-proposals-compose.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md) |
| Does steering the plain product's particles on the compose scorer (Feynman-Kac steering, Singhal et al.) find a composing proposal? | ⚪ null | cat × dog, seeds 9 to 16: compose rate 0.0 for steering at K 4 and K 16 against 0.0 for the unweighted control and 1.0 for Mono; 0 of 128 unweighted particles composed; the step-10 reward agrees with the final verdict on 0.99 of particles, so the reward was not blind | [does-steering-on-the-scorer-find-a-composing-proposal.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-steering-on-the-scorer-find-a-composing-proposal.md) |
| Does selecting among the corrected sampler's own draws produce a clean cat and dog? | ✅ support on the pre-registered bar; ⚪ null on what the resampling itself adds | cat × dog, seeds 9 to 16, rank-32 correction at λ 0.5 inside the proposal: 8 of 8 seeds read as two animals for steering at K 16 against 2 of 8 for the unweighted first particle and 3 of 8 for the correction alone, a gap of 0.75 against a 0.25 bar; the best of the same 16 unweighted particles is also 8 of 8, so the draws compose and the weights only choose which one survives (mean ImageReward +0.75 steered, +0.45 best of 16, +0.24 the joint prompt, −0.77 the correction alone); at λ 1.2 the control is already 8 of 8 and the gap is 0.00 | [does-steering-the-corrected-sampler-produce-clean-pairs.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-steering-the-corrected-sampler-produce-clean-pairs.md) |
| Does SuperDiff compose at its own defaults, and what is it missing when it does not? | ⚪ null at 200 steps; ✅ the missing piece is the same residual | cat × dog, seed 9: detector count 1 (blend) at 200 steps, 2 (compose) at 50; adding back its own residual `r_t^SD` separates 4 of 4 seeds by λ 0.75 (cat × dog) and by 0.25 (butterfly × meadow) | [does-superdiff-compose-at-its-own-defaults.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-superdiff-compose-at-its-own-defaults.md) |
| Does the PoE-trained correction carry into SuperDiff? | ⚪ null, it hurts | first separating λ "never" on 12 of 12 seeds against 0.5 to 0.75 for SuperDiff alone; median cosine between the adapter's correction and SuperDiff's missing residual 0.10 / 0.34 / 0.18 / 0.06 by step bucket, norms the same order | [does-the-poe-trained-correction-carry-into-superdiff.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-the-poe-trained-correction-carry-into-superdiff.md) |
| Does an adapter trained on SuperDiff's own residual compose inside SuperDiff? | ❓ inconclusive: separates early, smears with training, runs stopped at 61k/62k/36k of 100k | seeds of 4 with two bodies at λ 1, cat × dog: rank 8 peaks at 4 (30k) and holds 3 at 60k; rank 16 4 at 10k, 0 from 50k; rank 32 4 at 10k, 0 from 20k; loss plateaus from 30k to 40k without a matching rise | [does-an-adapter-trained-on-superdiffs-own-residual-compose.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-an-adapter-trained-on-superdiffs-own-residual-compose.md) |
| Does searching over the initial noise, with the compose scorer as verifier, produce two animals? | ❓ inconclusive on the pre-registered bar; a null in everything but its letter | cat × dog seeds 9 to 16: random search over the 8 seeds finds 0 composing; one round of zero-order search (8 candidates per seed at σ 0.1 and 0.3, plain PoE, DDIM eta 0) keeps a counted compose on 1 of 8 seeds at σ 0.1 (0.125 against a 0.10 null margin and 0.25 support margin) and 0 of 8 at σ 0.3; the one counted image is one body with two heads; the scorer read on x0-hat at step 10 keeps 0 of 8 at both σ and loses the butterfly on 3 of 8 control seeds | [does-searching-over-the-initial-noise-compose.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-initial-noise-compose.md) |
| Does searching over the initial noise, with the correction attached, give a crisper two-animal render? | ✅ support on the pre-registered bar; half the seeds by eye | cat × dog seeds 9 to 16, rank-32 step-30050 adapter at λ 1.2: the adapter's own render composes 6 of 8, the best of 8 nearby noises 8 of 8 at σ 0.1 and 0.3 (the count-then-confidence rule reaches the same 8 of 8); median per-seed sharpness ratio kept over adapter 1.47 and 2.88 against a 1.10 bar; by eye a cleaner cat and dog on 4 of 8 seeds, equal on 2, worse on 2 where the edge measure picks drawn texture; the adapter itself loses the butterfly on 4 of 8 control seeds by count | [does-searching-over-the-noise-sharpen-the-corrected-render.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-noise-sharpen-the-corrected-render.md) |
| Does adapting self-attention as well as cross-attention change the result? | ❓ inconclusive: the available comparison is mixed | DINOv2 drift at step 50,000, six-cell mean: cross and self-attention -0.269 against -0.329 for the best cross-attention run; best on cat and dog seed 9 (-0.527 against -0.488), worst on elephant and penguin seed 9 (+0.174 against -0.234); the cross-and-self run also decays its rate and resumed at step 5,000, so two things moved beside the layers | [does-adapting-self-attention-change-the-result.md](../3-understanding-compositionality/report/what-the-adapter-ablations-show/does-adapting-self-attention-change-the-result.md) |
| How much rank does the correction need? | ❓ inconclusive: no trend in rank, every comparison mixed | DINOv2 drift, cat and dog seed 9 at ranks 8, 16, 32: -0.252, -0.516, -0.408; seed 10: +0.222, +0.428, +0.319; the three runs differ in loss and length as well as rank; trainable parameters 0.19%, 0.38% and 0.77% of the model, and 1.38% with self-attention added | [how-much-rank-the-correction-needs.md](../3-understanding-compositionality/report/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.md) |
| Does the joint prompt always draw both concepts? | ❌ no | 11 failing cells of 32 rendered joint prompts on held-out pairs: chess board and hourglass has no hourglass on all 5 seeds, cat and fox has no cat on 4 of 6, dog and dog puts a person in the frame at seed 1, elephant and penguin drops the penguin at seed 9; on the chess cell at seed 10 the plain product holds both objects where the target holds one | [does-the-joint-prompt-always-draw-both-concepts.md](../1-lora-correction/report/is-the-adapter-just-copying-the-joint-prompt/does-the-joint-prompt-always-draw-both-concepts.md) |
| Does the adapter return two objects when both prompts name the same concept? | ✅ support | one concept asked for twice, 5 cells: the plain product draws one animal in 5 of 5, the adapter draws two in 4 of 5 (the exception is dog and dog at seed 1, where the second animal is horse-like), the joint prompt draws two in 3 of 5; adapter v58-03-self at step 50,000, correction on all 50 steps | [does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md](../1-lora-correction/report/is-the-adapter-just-copying-the-joint-prompt/does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md) |
| Does a Langevin corrector remove part of the correction and leave part of it? | ❓ inconclusive at every step size tried | read-zone mean of `‖eps_J − eps_PoE‖ / ‖eps_PoE‖` over the last five steps, cat × dog seed 9 at step-size multiplier 3, by corrector count 0 to 200: 0.153, 0.138, 0.261, 0.203, 0.103, 0.125; the two largest counts differ by 21% against the 5% bar, and the same uncorrected cell reads 0.153, 0.169 and 0.180 on three GPUs | [does-a-langevin-corrector-remove-part-of-the-correction.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md) |
| Does SDXL still fuse when it is sampled the way PixArt-alpha is? | 🟢 support | same SDXL, same cached starting noise, held-out seeds 9 to 16, only the sampling configuration changed: at DPM-Solver++ order 2, 20 steps and guidance 4.5 the product holds two clear animals on 0 of 8 (7 hybrids, 1 tile with two animals of unclear species), the same as the control at DDIM 50 steps and 7.5, which reproduces the published 0 of 8; the joint column holds two separate animals on 8 of 8 at both settings | [does-sdxl-compose-at-pixarts-settings.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-sdxl-compose-at-pixarts-settings.md) |
| Does a model with no CLIP in it still fuse a cat and a dog? | 🟢 support | PixArt-alpha, text encoder Flan-T5-XXL and no CLIP anywhere, at its own defaults of 20 steps and guidance 4.5: the summed-guidance product holds two clear animals on 0 of 8 seeds (6 hybrids, 2 single cats) while the joint prompt "a cat and a dog" holds a clear pair on 7 of 8; every call by eye at 1024² because the instance-count scorer cannot name the animal; the comparison swaps five things at once, so only the fusion direction is read | [does-a-clip-free-model-hybridise.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md) |
| Does a Langevin corrector, adding nothing to the score, produce two animals? | ⚪ null | cat × dog: 0 of 4 seeds composed in every one of the nine ten-step corrector windows and with the corrector on all 50 steps; on the eight held-out seeds joint prompt 8 of 8, plain product-of-experts 0 of 8, corrector 0 of 8; the control pair holds 8, 8, 8 | [does-a-corrector-alone-produce-two-animals.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md) |
| Can a corrector, or handing the tail to the frozen model, sharpen the adapter's renders? | ⚪ null on both bars | corrector on the adapter's own tail: mean Laplacian variance 57.1 to 61.7 over 8 seeds, +8.0% against a 10% bar, composed 7, 8, 7 of 8; clean tail: DINOv2 distance to the joint render 0.472 (adapter alone) to 0.434 at best, a gain of 0.038 against a 0.05 bar, composition held within one seed | [can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md) |
| Can any render-time fix recover the picture quality the adapter costs? | ❓ inconclusive: the edge measure passes two of them, the pictures do not | median Laplacian variance over 8 held-out cells (greyscale 0 to 1, ×1e-4), as a ratio to the adapter's own render: repaint at 0.25 1.14 and at 0.40 1.24, adapter for 20 steps 1.09, guidance 10 1.45 and 12 1.80, experts weighted 3.0 and 3.0 0.78, the joint prompt itself 1.18, against a 1.10 bar borrowed from the noise-search finding after these images existed; by eye the re-weighting smears both subjects on 8 of 8 cells and guidance changes what is in the picture on 3 of 8 | [does-any-render-time-fix-recover-picture-quality.md](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-any-render-time-fix-recover-picture-quality.md) |

## What each project found

**Every other report page, one row each, under the project it belongs to.** The verdict and the number are copied from the page's title line and claim. `no verdict stated` means those two name none, and such a page may still record verdicts on its separate reads.

### The whole repository: how SDXL works

| The question | Verdict | The number | The finding |
|---|---|---|---|
| [How SDXL turns a prompt into an image](../report/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image.md) | no verdict stated | A prompt becomes numbers, those numbers steer a denoising network over 50 steps, and a decoder turns the result into a 1024 by 1024 picture. | [the SDXL pipeline page](../report/how-sdxl-works/how-sdxl-turns-a-prompt-into-an-image.md) |
| [How the SDXL autoencoder is built](../report/how-sdxl-works/how-the-sdxl-vae-is-built.md) | no verdict stated | SDXL's variational autoencoder (VAE) compresses a 1024 by 1024 image into a 4 by 128 by 128 latent, the small code the denoising network works on, and decodes that code back to pixels. | [the SDXL autoencoder page](../report/how-sdxl-works/how-the-sdxl-vae-is-built.md) |
| [Why SDXL multiplies its latent by 0.13025](../report/how-sdxl-works/why-the-latent-is-scaled.md) | no verdict stated | SDXL's autoencoder produces a code with a spread near 7.7, while the noise schedule assumes a spread near 1. One constant, 0.13025, sits between them. | [the latent scaling page](../report/how-sdxl-works/why-the-latent-is-scaled.md) |

### Project 1: the LoRA correction

The ICLR review tickets are on the [paper review board](../1-lora-correction/report/paper-review/board.md).

| The question | Verdict | The number | The finding |
|---|---|---|---|
| [Do the rival methods reproduce the scores their papers print?](../1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md) | ❌ RTD not reproduced on T2I-CompBench Complex · ✅ reproduced on RareBench Concat, a weak test · FKC not tested, by the owner's choice | RTD's ImageReward is 0.2423 on 988 of 1,000 Complex prompts, against 0.4661 printed (pass band 0.1751 to 0.3095). Plain SDXL on FKC's GenEval prompts scores 0.7399 of 396 images correct, against 0.80 printed (pass band 0.6468 to 0.8330, reproduced). | [the rival reproduction finding](../1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md) |
| [How four methods compare on the four pairs of the paper's main grid](../1-lora-correction/report/against-other-methods/how-do-four-methods-compare-on-four-hand-picked-pairs.md) | no verdict stated | On all four rows the adapter ("Ours") draws both named things from the two concept prompts alone. CO3 also draws both on all four, and it reads the sentence naming both while it draws. | [the hand-picked four-pair grid](../1-lora-correction/report/against-other-methods/how-do-four-methods-compare-on-four-hand-picked-pairs.md) |
| [How every SDXL method compares on the 53 held-out pairs](../1-lora-correction/report/against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md) | no verdict stated | Over all 53 pairs the adapter ("Ours") has a mean BLIP-VQA of 0.6306, between PoE at 0.5187 and the joint prompt at 0.7655. | [the held-out method comparison](../1-lora-correction/report/against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md) |
| [How the adapter scores against other methods on the 276 Attend-and-Excite prompts](../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-attend-and-excite-prompts.md) | The one comparison with a rule written before scoring has the recorded verdict SUPPORTS | the adapter scores above plain product of experts on all three prompt sets. | [the Attend-and-Excite scores](../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-attend-and-excite-prompts.md) |
| [How the adapter scores on the GenEval and T2I-CompBench benchmarks](../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-standard-benchmarks.md) | no verdict stated | The adapter ("Ours") is above PoE in all five scored columns and below the joint prompt in all five. CO3 is above the adapter in four columns. On T2I-CompBench shape the adapter reads 0.4631 and CO3 0.4616, a gap of 0.0015 (0.4631 minus 0.4616). | [the GenEval and T2I-CompBench scores](../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-standard-benchmarks.md) |
| [Does one pooled adapter compose on pairs it never trained on?](../1-lora-correction/report/does-the-fix-reach-unseen-pairs/does-one-pooled-adapter-compose-on-pairs-it-never-trained-on.md) | ✅ support · recorded 2026-11-13 | At training step 60000, 58 of 64 pictures (0.906) of eight pairs the adapter never trained on hold two animals, against 80 of 88 (0.909) on its eleven training pairs | [the pooled adapter transfer finding](../1-lora-correction/report/does-the-fix-reach-unseen-pairs/does-one-pooled-adapter-compose-on-pairs-it-never-trained-on.md) |
| [How were the adapter's training pairs chosen?](../1-lora-correction/report/does-the-fix-reach-unseen-pairs/how-were-the-adapters-training-pairs-chosen.md) | no verdict stated | A person kept a pair when the joint prompt drew both named things as separate objects, and the final list holds 48 pairs at 182 pair-and-seed cells over 23 seeds. | [the training pair selection page](../1-lora-correction/report/does-the-fix-reach-unseen-pairs/how-were-the-adapters-training-pairs-chosen.md) |
| [Does handing the last denoising steps back to frozen SDXL bring a render nearer its joint prompt?](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-handing-the-tail-to-frozen-sdxl-bring-renders-nearer-the-joint-prompt.md) | no verdict stated | The script printed `null` for two checkpoints and `composition broke` for the third. | [the frozen tail sweep](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-handing-the-tail-to-frozen-sdxl-bring-renders-nearer-the-joint-prompt.md) |
| [Do the images scored as composed contain both animals?](../1-lora-correction/report/measuring-composition/do-the-scored-successes-contain-both-animals.md) | no verdict stated | At full correction strength it reads 94%, which is 30 of 32 images. A hand sort of those 32 finds 12 with both requested animals, one with two of the same animal, and 17 that the sort could not settle by eye. | [the hand sort of scored successes](../1-lora-correction/report/measuring-composition/do-the-scored-successes-contain-both-animals.md) |
| [How big the paper's strength grid should be](../1-lora-correction/report/measuring-composition/how-big-the-strength-grid-should-be.md) | no verdict stated | A cropped copy set at 0.62 of the text width leaves its tiles small. The section 3 proofreading map records a full-width replacement as accepted, at 5.50 by 4.97 inches with 1.09 inch tiles, and a move of the grid to the appendix | [the strength grid layout page](../1-lora-correction/report/measuring-composition/how-big-the-strength-grid-should-be.md) |
| [What four scorers say about a mug and a wine glass](../1-lora-correction/report/measuring-composition/what-each-scorer-says-about-two-objects.md) | no verdict stated | All four rank the image from the joint prompt above the image from the plain product | [the four-scorer probe](../1-lora-correction/report/measuring-composition/what-each-scorer-says-about-two-objects.md) |
| [What the main study's comparison grids show](../1-lora-correction/report/measuring-composition/what-the-main-study-grids-show.md) | no verdict stated | no number in the claim | [the main study grids](../1-lora-correction/report/measuring-composition/what-the-main-study-grids-show.md) |
| [Does each adapter hold across ten seeds?](../1-lora-correction/report/the-adapter-on-cat-and-dog/does-each-adapter-hold-across-ten-seeds.md) | no verdict stated | Seven grids render seven pairs at ten seeds with fourteen adapters and three reference methods | [the ten-seed grids](../1-lora-correction/report/the-adapter-on-cat-and-dog/does-each-adapter-hold-across-ten-seeds.md) |
| [Does one adapter compose four pairs at one seed?](../1-lora-correction/report/the-adapter-on-cat-and-dog/does-one-adapter-compose-four-pairs.md) | no verdict stated | four pairs the adapter never trained on, five ways of drawing them, one seed. | [the four-pair review sheet](../1-lora-correction/report/the-adapter-on-cat-and-dog/does-one-adapter-compose-four-pairs.md) |
| [Does the two-concept adapter compose three concepts?](../1-lora-correction/report/the-adapter-on-cat-and-dog/does-the-adapter-compose-three-concepts.md) | no verdict stated | It was run on one triple of each of four kinds, at one seed. | [the three-concept renders](../1-lora-correction/report/the-adapter-on-cat-and-dog/does-the-adapter-compose-three-concepts.md) |
| [How the adapter's settings change a cat and a dog](../1-lora-correction/report/the-adapter-on-cat-and-dog/how-the-adapter-settings-change-cat-and-dog.md) | no verdict stated | The first varies the adapter's rank and the steps it was trained on, for a cat and a dog at three held-out seeds. | [the adapter settings tiles](../1-lora-correction/report/the-adapter-on-cat-and-dog/how-the-adapter-settings-change-cat-and-dog.md) |
| [What the adapter ablations show](../1-lora-correction/report/the-adapter-on-cat-and-dog/what-the-adapter-ablations-show.md) | Both ablation questions are recorded as inconclusive | no number in the claim | [the per-seed ablation grids](../1-lora-correction/report/the-adapter-on-cat-and-dog/what-the-adapter-ablations-show.md) |
| [Which adapter composes a cat and a dog?](../1-lora-correction/report/the-adapter-on-cat-and-dog/which-adapter-composes-cat-and-dog.md) | no verdict stated | The project rendered one held-out pair, a cat and a dog, with 23 trained adapters at three seeds no adapter trained on | [the 23-adapter comparison](../1-lora-correction/report/the-adapter-on-cat-and-dog/which-adapter-composes-cat-and-dog.md) |
| [Which adapter composes the two showcase pairs?](../1-lora-correction/report/the-adapter-on-cat-and-dog/which-adapter-composes-the-showcase-pairs.md) | no verdict stated | The showcase rule picked `v58-03-self` at training step 30,000, with seed 10 as the shared seed | [the showcase adapter choice](../1-lora-correction/report/the-adapter-on-cat-and-dog/which-adapter-composes-the-showcase-pairs.md) |
| [Which checkpoint to take each figure cell from](../1-lora-correction/report/the-adapter-on-cat-and-dog/which-checkpoint-to-take-each-figure-cell-from.md) | no verdict stated | On 2026-09-24 a sweep drew saved checkpoints of eight training runs on 13 pairs | [the checkpoint contact sheets](../1-lora-correction/report/the-adapter-on-cat-and-dog/which-checkpoint-to-take-each-figure-cell-from.md) |
| [Which renders show the adapter working and failing](../1-lora-correction/report/the-adapter-on-cat-and-dog/which-renders-show-it-working-and-failing.md) | no verdict stated | no number in the claim | [the working and failing renders](../1-lora-correction/report/the-adapter-on-cat-and-dog/which-renders-show-it-working-and-failing.md) |
| [Does the correction cause composition?](../1-lora-correction/report/what-the-correction-is/does-the-correction-cause-composition.md) | no verdict stated | This page adds no verdict: it gathers the three already recorded and shows the pictures behind them. | [the correction causation page](../1-lora-correction/report/what-the-correction-is/does-the-correction-cause-composition.md) |
| [What does the correction look like?](../1-lora-correction/report/what-the-correction-is/what-does-the-correction-look-like.md) | no verdict stated | The correction has a clear place in the picture and no direction shared across pairs. This page gathers four reads of it in the order they were made. Two carry a recorded verdict, and the other two record none. | [the correction's four reads](../1-lora-correction/report/what-the-correction-is/what-does-the-correction-look-like.md) |
| [When does the correction have to act?](../1-lora-correction/report/what-the-correction-is/when-does-the-correction-have-to-act.md) | no verdict stated | Injected over a sliding 10-step window, the true correction composes best at steps 0 to 10 for all 8 pairs tested. The learned adapter, switched off after step 10, draws two animals in 47% of the animal-pair images, and after step 25 in 92%. The first result has a recorded verdict. The second has a reading written in advance whose outcome nobody has recorded. | [the correction timing page](../1-lora-correction/report/what-the-correction-is/when-does-the-correction-have-to-act.md) |

### Project 2: classifier guidance

| The question | Verdict | The number | The finding |
|---|---|---|---|
| [Does a checker learn plurality on unseen digits?](../2-classifier-guidance/report/abstract-checkers-on-digits/does-a-checker-learn-plurality-on-unseen-digits.md) | ✅ support, by the count checker · the first single checker ⚪ null by its written rule · read 2026-11-24 | The count checker, which reads only where each asked digit acts, scores balanced accuracy 0.791 on digits 5 to 9 it never trained on against 0.782 on digits 0 to 4 (pass mark 0.75, gap 0.009, 0.5 is chance), W&B `tyhrpyov`. The single checker that also had to tell which digits are present scores 0.540, W&B `a562r799`. | [the checker learning finding](../2-classifier-guidance/report/abstract-checkers-on-digits/does-a-checker-learn-plurality-on-unseen-digits.md) |
| [Does removing one checker remove its property, and only its property?](../2-classifier-guidance/report/abstract-checkers-on-digits/does-removing-one-checker-remove-its-property.md) | ⚪ null by the written rule (board question 75) · read 2026-11-24 | Removing count lowers two digits apart from 38.4% to 10.3% and removing order lowers the asked order from 88.2% to 47.8%, of 960 canvases per row, and each moves one other column by 6.8 or 6.9 points, past the 4.2-point limit; removing right digits lowers its own by 2.6 points. W&B `j4anxfgy`. | [the checker removal finding](../2-classifier-guidance/report/abstract-checkers-on-digits/does-removing-one-checker-remove-its-property.md) |
| [Does the count checker's push compose unseen digits?](../2-classifier-guidance/report/abstract-checkers-on-digits/does-the-count-checkers-push-compose-unseen-digits.md) | ✅ support · read 2026-11-24 | On the 10 pairs among digits 5 to 9, the push turns the plain product's 0 clean pairs of 320 into 43 (lead 5.7 standard errors of 7.58); a random push of the same length gives 1 and the best of 4 seeds 0. W&B `qhusod91`. | [the count push finding](../2-classifier-guidance/report/abstract-checkers-on-digits/does-the-count-checkers-push-compose-unseen-digits.md) |
| [Is the small digit environment repeatable?](../2-classifier-guidance/report/abstract-checkers-on-digits/is-the-small-environment-repeatable.md) | ✅ passed · read 2026-11-24 | Two rebuilds give the same data hash on the laptop and on the cluster, and a zero push changes no pixel (largest difference 0.0, bar under 1e-5) on all 20 digit pairs at 8 seeds. W&B `xv6mbklb`. | [the digit environment finding](../2-classifier-guidance/report/abstract-checkers-on-digits/is-the-small-environment-repeatable.md) |
| [A2b: MNIST digits on a canvas, how good the checker must be](../2-classifier-guidance/report/classifier-guided-poe/24-mnist-one-model.md) | no verdict stated | Works on seen pairs only. On 7 & 8, a checker that knows the asked pair turns plain product's fused digit (1 of 64 seeds with a clear 7 beside a clear 8) into two separate digits on 47 of 64 at strength 10; a random push of the same size gives 1, and best-of-2 at the same cost gives 1. On the five pairs never seen together the push still makes two digits, but often the wrong second one: 84 of 320. | [the one-model MNIST page](../2-classifier-guidance/report/classifier-guided-poe/24-mnist-one-model.md) |
| [Experts trained apart: do the step 7b pushes compose them](../2-classifier-guidance/report/classifier-guided-poe/32-mnist-two-experts.md) | no verdict stated | They do not compose. Two digit experts trained apart, a 0–4 expert and a 5–9 expert, give two clear, right digits on 0 of 320 seeds when each draws its digit on either side. | [the two-expert MNIST page](../2-classifier-guidance/report/classifier-guided-poe/32-mnist-two-experts.md) |
| [A1: the exact toy, a pair-aware push against a separate-label push](../2-classifier-guidance/report/classifier-guided-poe/a1-exact-toy-classifier-push.md) | Pass | Pair-aware push at w = 1: 172 of 300 crisp, 128 overlaid, against 0 and 300 unguided; 300 of 300 at w = 2. Separate-label push: at most 10 of 300 over scales 1 to 10. | [the exact toy push finding](../2-classifier-guidance/report/classifier-guided-poe/a1-exact-toy-classifier-push.md) |
| [A3q: cat × dog, a small pair-aware discriminator, judged on held-out seeds](../2-classifier-guidance/report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md) | no verdict stated | Draft reading 7, too weak to read: no tile on seeds 9 to 12 shows two separate animals, and the size cap clipped the push on every step at s = 100. | [the quick discriminator test](../2-classifier-guidance/report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md) |
| [Can a small checker tell a clean cat and dog from a fused animal?](../2-classifier-guidance/report/classifier-guided-poe/can-a-checker-read-a-clean-cat-and-dog.md) | no verdict stated | the best reading of the sampler's own early states is 0.730 against a pass mark of 0.75. | [the checker accuracy page](../2-classifier-guidance/report/classifier-guided-poe/can-a-checker-read-a-clean-cat-and-dog.md) |
| [Can a classifier supply the term the product drops?](../2-classifier-guidance/report/classifier-guided-poe/can-a-classifier-supply-the-missing-term.md) | no verdict stated | No verdict has been recorded, because the eight-seed sweep that would judge this has no result. | [the missing term page](../2-classifier-guidance/report/classifier-guided-poe/can-a-classifier-supply-the-missing-term.md) |
| [Can a second push make each body the right animal?](../2-classifier-guidance/report/classifier-guided-poe/can-a-second-push-name-each-animal.md) | no verdict stated | No experiment here raised the clean cat-and-dog count past the early push alone. | [the second push page](../2-classifier-guidance/report/classifier-guided-poe/can-a-second-push-name-each-animal.md) |
| [Can two separately trained digit models compose while sampling?](../2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md) | ❓ inconclusive (proposed), read 2026-11-22 | Two digit models that never saw each other's digits draw 1,200 clean pairs of 1,280 (94%) on 20 digit pairs kept out of tuning, by the digit-reading judge, when each model works on its own half of the canvas and the side is chosen early and held; On one shared canvas with no halves, the best run reaches 121 of 320 (38%), against the 288 the plan set to support that goal | [the two digit models finding](../2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md) |
| [Does a pair-aware push compose our cat and dog photos?](../2-classifier-guidance/report/classifier-guided-poe/does-a-pair-aware-push-compose-our-photos.md) | no verdict stated | No verdict has been recorded, because the runs that would judge this stopped in a storage outage. | [the pair-aware push page](../2-classifier-guidance/report/classifier-guided-poe/does-a-pair-aware-push-compose-our-photos.md) |
| [Does an early checker push separate the cat and the dog?](../2-classifier-guidance/report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md) | Yes on the 32 seeds kept for this test | two separate animals on 28 of 32, against 1 of 32 for plain product and for a random push of the same size. | [the early push finding](../2-classifier-guidance/report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md) |
| [Does classifier guidance draw a dog on SDXL?](../2-classifier-guidance/report/classifier-guided-poe/does-classifier-guidance-draw-a-dog.md) | The recorded call is bar not met | a trained dog classifier drew a clean dog on 2 of 4 seeds, and the bar was 3 of 4. | [the dog classifier finding](../2-classifier-guidance/report/classifier-guided-poe/does-classifier-guidance-draw-a-dog.md) |
| [Does classifier guidance work on the authors' released weights?](../2-classifier-guidance/report/classifier-guided-poe/does-classifier-guidance-work-on-the-released-weights.md) | The route records all four targets as reproduced, and a strict look at each picture also meets the condition the route wrote for "inconclusive" on three of them | the counts are in the verdict | [the released weights check](../2-classifier-guidance/report/classifier-guided-poe/does-classifier-guidance-work-on-the-released-weights.md) |
| [Does the early push carry from cat and dog to other pairs?](../2-classifier-guidance/report/classifier-guided-poe/does-the-early-push-carry-to-other-pairs.md) | no verdict stated | It carried to one object pair and only partly to a second animal pair. The early push, with a checker trained only on cats and dogs, separated a typewriter and a cactus on 14 of 16 seeds and a lion and a meerkat on 6 of 16. | [the other-pairs page](../2-classifier-guidance/report/classifier-guided-poe/does-the-early-push-carry-to-other-pairs.md) |
| [How close the kept cat and dog push comes to the joint prompt and CO3](../2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md) | no verdict stated | The kept push drew a clean cat and dog on 35 of 128 seeds by the automatic scorer and 54 by eye, below the joint prompt (57 and 69) and CO3 (89 and 101). | [the kept push comparison](../2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md) |
| [Which line of code computes which part of guided sampling](../2-classifier-guidance/report/classifier-guided-poe/which-line-computes-which-part-of-guided-sampling.md) | no verdict stated | Four pieces of code were read line by line, so each step of guided sampling can be traced to the file and line that computes it. | [the guided sampling code walk](../2-classifier-guidance/report/classifier-guided-poe/which-line-computes-which-part-of-guided-sampling.md) |
| [Does the digit model's score path carry which digits are present?](../2-classifier-guidance/report/does-the-score-path-carry-which-digits-are-present.md) | ⚪ null by the written rule · read 2026-11-24 | The digit model's ten label-minus-no-label noise predictions on one noisy canvas read digit presence at best 0.700 balanced accuracy (0.5 is chance) at timestep 480, against a bar of 0.75, and 0.513 at best in the noisier early window; the noisy canvas itself reads 0.593 at timestep 480. W&B `gq51nl74`. | [the score path finding](../2-classifier-guidance/report/does-the-score-path-carry-which-digits-are-present.md) |
| [Does adding the kept push to CO3 beat CO3 alone?](../2-classifier-guidance/report/the-method-search/does-adding-the-kept-push-to-co3-beat-co3-alone.md) | ⚪ null · read 2026-11-24 | On 12 subject pairs at 8 seeds, CO3 with the kept push makes 7 faithful pairs of 96 pictures and CO3 alone makes 11; the lead of -4 is short of the win rule, a lead of at least 5 and more than 10.08 (1.96 standard errors of 5.14). A random push of the same size makes 12, so the inconclusive rule does not fire, and none of the 288 pictures is one subject or a fused hybrid. W&B `s7o6404z`. | [the CO3 with the kept push finding](../2-classifier-guidance/report/the-method-search/does-adding-the-kept-push-to-co3-beat-co3-alone.md) |
| [Does composing two digit models trained apart beat sampling each alone and pasting?](../2-classifier-guidance/report/the-method-search/does-composing-two-digit-models-beat-sampling-and-pasting.md) | ⚪ null · read 2026-11-24 | The method, two digit models each on its own half with a checker push, draws 295 clean digit pairs of 320 and sample-and-paste draws 287; the lead of 8 is under the bar of 13.54 (1.96 standard errors of 6.91). The push lifts the method by 36 over no push, where a random push of the same size changes nothing. W&B `vekm3sfh`. The SDXL half of the step is in [the CO3 with the kept push finding](../2-classifier-guidance/report/the-method-search/does-adding-the-kept-push-to-co3-beat-co3-alone.md). | [the method against paste finding](../2-classifier-guidance/report/the-method-search/does-composing-two-digit-models-beat-sampling-and-pasting.md) |
| [Does giving each subject its own region fix which animal each body becomes?](../2-classifier-guidance/report/the-method-search/does-giving-each-subject-its-own-region-fix-identity.md) | rung 1 with bootstrapping: regions without names · rung 2: stopped at its mask check · read 2026-11-24 | On cat and dog with the right boxes, 0 clear pairs of 8 without bootstrapping and 8 of 8 with it, all 8 damaged (0 undamaged, against 5 of 8 for the joint prompt); swapped prompts give 7 and the same prompt twice 8 pictures with two animals. Word attention at step 10 failed 11 of 24 mask reads against a limit of 6, so no rung 2 picture was rendered. | [the regions finding](../2-classifier-guidance/report/the-method-search/does-giving-each-subject-its-own-region-fix-identity.md) |
| [Can a sentence score read the five properties on a half-drawn picture?](../2-classifier-guidance/report/the-sentence-score/can-a-sentence-score-read-the-five-properties.md) | ⚪ null on counting, prompt match and left-right order · ❓ inconclusive on position and size and on binding · read 2026-11-24 | At steps 0 to 15 of 50, NoisyCLIP, frozen CLIP and Latent CLIP each stay between 0.46 and 0.56 mean balanced accuracy (0.5 is chance) against a 0.75 bar, and none beats frozen CLIP; a later fairness test records the step inconclusive, quoted on the page. W&B `70g3qw2f`. | [the sentence score finding](../2-classifier-guidance/report/the-sentence-score/can-a-sentence-score-read-the-five-properties.md) |
| [Is the sentence-score test fair, and can anything read the half-drawn picture at all?](../2-classifier-guidance/report/the-sentence-score/is-the-sentence-score-test-fair.md) | ❓ inconclusive by the bars fixed beforehand, and no sentence score reads in any fair setting · read 2026-11-24 | A small reader on SDXL's own features, run as a positive control on the reading test's pictures, reads a clean cat and dog at 0.637 mean balanced accuracy over steps 0 to 15 (0.5 is chance), under its floor of 0.65, so the reading test is inconclusive by its rule; it beats frozen CLIP by 0.150 (standard error 0.035). No sentence score reaches the 0.75 bar over steps 0 to 15 with five wordings averaged, in NoisyCLIP's own steps 20 to 29, on the sampler's real states or with species-free counting sentences; the best is Latent CLIP's 0.574 on the real states. The reading test's own review records it null; both verdicts are quoted on the page (board question 79). W&B `etbecql0`. | [the fairness finding](../2-classifier-guidance/report/the-sentence-score/is-the-sentence-score-test-fair.md) |

### Project 3: understanding compositionality

| The question | Verdict | The number | The finding |
|---|---|---|---|
| [Does a camel in the rain come out wet](../3-understanding-compositionality/report/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet.md) | no verdict stated | SDXL draws a wet camel on 4 of 5 seeds when the prompt says "a wet camel", and the reader called every product of experts render of "a camel" and "rain" dry. | [the wet camel page](../3-understanding-compositionality/report/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet.md) |
| [Does a dog in the rain come out wet](../3-understanding-compositionality/report/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet.md) | The test recorded for this sheet did not hold | the joint prompt drew a wet dog on 2 of 5 seeds, and product of experts also drew one on 2 of 5. | [the wet dog page](../3-understanding-compositionality/report/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet.md) |
| [Does an effect need support in the animal's own pictures](../3-understanding-compositionality/report/a-thing-in-the-rain/does-an-effect-need-support.md) | no verdict stated | No verdict has been recorded, because none of the 128 renders in this experiment has been called. | [the effect support page](../3-understanding-compositionality/report/a-thing-in-the-rain/does-an-effect-need-support.md) |
| [Can a classifier's push fix the product on the exact toy](../3-understanding-compositionality/report/the-exact-toy/can-a-classifier-push-fix-the-product.md) | the recorded verdict is pass | At pair weight 0.01 the exact product draws a red circle beside a blue square 300 times of 300, while the sampler that adds the two experts' scores stacks them all 300 times. | [the toy classifier push page](../3-understanding-compositionality/report/the-exact-toy/can-a-classifier-push-fix-the-product.md) |
| [How much co-occurrence the product of two experts needs](../3-understanding-compositionality/report/the-exact-toy/how-much-co-occurrence-the-product-needs.md) | no verdict stated | The exact product prefers the two-shape picture from the smallest overlap tested (a pair weight of 0.001), and its two samplers draw the pair half the time only at a weight between 0.3 and 0.5. | [the co-occurrence page](../3-understanding-compositionality/report/the-exact-toy/how-much-co-occurrence-the-product-needs.md) |
| [How the forward process dissolves an image](../3-understanding-compositionality/report/the-exact-toy/how-the-forward-process-dissolves-an-image.md) | no verdict stated | The forward process mixes a picture with noise so that the picture's share of the variance falls from 1 to almost 0 while the total stays at 1. | [the forward process page](../3-understanding-compositionality/report/the-exact-toy/how-the-forward-process-dissolves-an-image.md) |
| [What product of experts needs from training to draw two things at once](../3-understanding-compositionality/report/the-exact-toy/what-poe-needs-from-training-to-draw-two-things-at-once.md) | no verdict stated | In a small model built like SDXL, product of experts draws two separate shapes only after some both-shapes pictures also carry each single caption. | [the training needs page](../3-understanding-compositionality/report/the-exact-toy/what-poe-needs-from-training-to-draw-two-things-at-once.md) |
| [What the product of two experts draws, and what samplers lose](../3-understanding-compositionality/report/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose.md) | no verdict stated | On 2D points the exact product depends on how the two experts' data overlap, and at a joint weight of 0.05 every sampler draws the shared point far less often than the exact product does (0.23 to 0.31 of draws against 0.91). | [the 2D product page](../3-understanding-compositionality/report/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose.md) |
| [Can an automatic score see which species a render drew](../3-understanding-compositionality/report/what-sdxl-already-knows/can-a-score-see-species.md) | no verdict stated | The best automatic judge agrees with the eye on 13 of 19 confident renders, below the bar of 27 of 30 written for it. | [the species judge page](../3-understanding-compositionality/report/what-sdxl-already-knows/can-a-score-see-species.md) |
| [Does the product of experts match the joint prompt for each kind of composition](../3-understanding-compositionality/report/what-sdxl-already-knows/does-poe-match-the-joint-for-each-kind-of-composition.md) | no verdict stated | The product matches the joint prompt on some kinds of composition and misses on others, and these sheets show examples without measuring a rate. On Stable Diffusion v1-4, the product fuses a cat and a dog on 4 of 4 seeds, paints rain over a dry dog and gives near-identical pictures for a dog and an oil painting. | [the kinds of composition page](../3-understanding-compositionality/report/what-sdxl-already-knows/does-poe-match-the-joint-for-each-kind-of-composition.md) |
| [Does the product of "cat" and "dog" match the joint prompt](../3-understanding-compositionality/report/what-sdxl-already-knows/does-poe-of-cats-and-dogs-match-the-joint.md) | no verdict stated | On seeds 1 to 7, the product of "cat" and "dog" draws one fused animal on six seeds, and one wording that adds "multiple animals" draws separate animals on three. | [the eight wordings sheet](../3-understanding-compositionality/report/what-sdxl-already-knows/does-poe-of-cats-and-dogs-match-the-joint.md) |
| [Does "a photo of a cat" already draw dogs in base SDXL](../3-understanding-compositionality/report/what-sdxl-already-knows/does-the-cat-prompt-already-draw-dogs.md) | no verdict stated | By eye none of the 64 renders holds a dog, the detector reports a dog on all 64, and a classifier on the cropped animal calls all 64 a cat. | [the cat prompt page](../3-understanding-compositionality/report/what-sdxl-already-knows/does-the-cat-prompt-already-draw-dogs.md) |
| [Does a fused cat-dog fit both captions better than a real photograph](../3-understanding-compositionality/report/what-sdxl-already-knows/does-the-hybrid-fit-both-captions.md) | no verdict stated | Fused renders score 0.014 above real photographs on the weaker of "a photo of a cat" and "a photo of a dog", and renders of one sentence naming both animals score 0.017 above them, so no verdict is recorded and the written rule can be read two ways. | [the hybrid caption fit page](../3-understanding-compositionality/report/what-sdxl-already-knows/does-the-hybrid-fit-both-captions.md) |
| [Is the layout the piece the product of experts is missing](../3-understanding-compositionality/report/what-sdxl-already-knows/is-the-layout-the-missing-piece.md) | no verdict stated | This question is unanswered: the boxes and masks exist, one preview was rendered, and the comparison the verdict depends on never ran. | [the layout page](../3-understanding-compositionality/report/what-sdxl-already-knows/is-the-layout-the-missing-piece.md) |
| [What the fused cat-dog is not caused by](../3-understanding-compositionality/report/what-sdxl-already-knows/what-the-fusion-is-not-caused-by.md) | no verdict stated | The fusion survives removing CLIP and survives changing SDXL's sampler, step count and guidance together: the product of "a cat" and "a dog" holds two clear animals on 0 of 8 seeds in all three setups. | [the fusion causes page](../3-understanding-compositionality/report/what-sdxl-already-knows/what-the-fusion-is-not-caused-by.md) |
| [What the training captions say about wet camels and wet dogs](../3-understanding-compositionality/report/what-sdxl-already-knows/what-the-training-captions-show.md) | no verdict stated | Dog captions use a wet word about 6 times as often as camel captions (0.346% against 0.057%), and no picture behind them has been called wet or dry, so no verdict is recorded. | [the training captions page](../3-understanding-compositionality/report/what-sdxl-already-knows/what-the-training-captions-show.md) |
| [Where the product of "a cat" and "a dog" lands](../3-understanding-compositionality/report/what-sdxl-already-knows/where-the-product-lands.md) | no verdict stated | This is an illustration: the surface is drawn by hand and only its four renders are real. | [the product landscape illustration](../3-understanding-compositionality/report/what-sdxl-already-knows/where-the-product-lands.md) |

### Project 4: inclusive captions

This project keeps its own index: every finding, verdict and number is in [the inclusive captions report index](../4-inclusive-captions/report/00-INDEX.md).

### Project 5: a third expert from text embeddings

| The question | Verdict | The number | The finding |
|---|---|---|---|
| [What the "and" vector is made of, and whether it works as a third expert](../5-text-embedding-third-expert/report/what-the-and-vector-is-made-of.md) | no verdict stated | The verdict across the whole series is in the text-space third expert finding. | [the "and" vector page](../5-text-embedding-third-expert/report/what-the-and-vector-is-made-of.md) |
| [What the text already says about one animal or two](../5-text-embedding-third-expert/report/what-the-text-already-says.md) | no verdict stated | no number in the claim | [the text encoder page](../5-text-embedding-third-expert/report/what-the-text-already-says.md) |

## Figures still missing

The evidence pairs waiting on a render, so they are visible rather than quietly absent.

| The finding | What is wanted | Where it saves |
|---|---|---|
| [can a text-space third expert restore composition](../5-text-embedding-third-expert/report/can-a-text-space-third-expert-restore-composition.md) | the rank of the held-out word as a histogram, the subtracted and unsubtracted readouts side by side with the decoy control behind them, from `premise-word-recovery.json` | `5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/premise-word-recovery.png` |

## Reading the pictures

Each finding's figures are read one at a time, in plain words, in a figure explainer beside the
images: [where each condition lands, what these pictures mean](../1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/where-does-each-condition-land/figure-explainer.md)
[is the gap the sampler's or the model's, what these pictures mean](../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/figure-explainer.md)
[what the correction is made of, what these pictures mean](../1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/what-the-correction-is-made-of/figure-explainer.md)
[the corrector figures, what these pictures mean](../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/figure-explainer-the-langevin-corrector.md)
and [which fidelity fix wins, what these pictures mean](../1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/which-fidelity-fix-wins/figure-explainer.md).
The path from two X posts to the twisted-SMC run, for a reader who was not in that conversation, is
[the note beside those pictures](../1-lora-correction/artifacts/notes/from-an-x-post-to-a-twisted-smc-baseline/note.md).
How a finding, its card and its explainer are laid out, with that one as the worked instance, is
in [the finding-and-explainer template](../1-lora-correction/report/finding-and-explainer-template.md).

## The question groups

One folder per high-level research question, mirroring the `artifacts/results/` grouping the
findings embed from; a finding with no sibling yet sits flat.

| Group | The question it answers | Findings |
|---|---|---|
| [does-the-fix-reach-unseen-pairs/](../1-lora-correction/report/does-the-fix-reach-unseen-pairs/) | Why does the pooled adapter's correction carry to some unseen pairs and not others | 2 |
| [does-training-longer-help-the-pooled-lora/](../3-understanding-compositionality/report/does-training-longer-help-the-pooled-lora/) | Which checkpoint of the pooled adapter to show, and what training past it does to the held-out renders | 2 |
| [when-does-the-outcome-lock-in/](../1-lora-correction/report/when-does-the-outcome-lock-in/) | When does a run commit to one animal or two, where does each condition end up, and what is the correction made of | 2 |
| [is-the-gap-the-samplers-or-the-models/](../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/) | Is the gap the samplers or the models | 12 |
| [what-the-adapter-ablations-show/](../3-understanding-compositionality/report/what-the-adapter-ablations-show/) | Which layers the adapter needs and how much rank the correction takes | 2 |
| [is-the-adapter-just-copying-the-joint-prompt/](../1-lora-correction/report/is-the-adapter-just-copying-the-joint-prompt/) | Whether the target itself is right, and what the adapter adds beyond it | 2 |

## The documents that are not findings

Six documents here predate the finding format and answer no question of their own. Two of them are
load-bearing anyway: the bars several findings are judged against were written in them, before the
runs. Where each one should eventually be filed is argued in [Still open](#still-open); this
section is only so a reader can reach them.

**[The stage-by-stage results summary](../1-lora-correction/report/results-summary.md)**

One screen per stage of the work as it stood on 2026-07-22: status, what the stage produced, its
W&B runs, and commands that walk the outputs. Every verdict in it is a candidate finding that has
not been written as one. Its opening warning still bites: artifacts are split across the repo and
`/datasets`, neither root complete, so a bare `artifacts/...` path resolves for some runs and
silently fails for others.

**[The commitment-timing pre-registration](../1-lora-correction/report/experiments-log.md)**

Written before any of those runs. It fixes the axes (pair, seed, step, condition, space, adapter
strength) and the falsification rule for each of the five experiments, and it says the rules are
not revised after seeing results. Read it before quoting a bar from rungs 3 and 4 of the arc.

**[How correction size is measured](../1-lora-correction/report/normalization-preregistration.md)**

Committed 2026-08-05, before any cross-type plot existed. It pins the measure to `‖r_t‖` over
`‖ε̃_PoE‖`, median over steps for a per-run number then median over seeds for a per-pair number,
every tensor upcast to fp32 first. Medians rather than means because the early steps carry heavy
fp16 cancellation noise. Every size number in this folder is measured this way, so this one is
still live rather than historical.

**[What the measuring scripts do on real data](../1-lora-correction/report/instrument-smoke-test.md)**

The recorded output of each measuring script on `a_cat__x__a_dog` seed 9, with the command above
each result, dated 2026-08-05 on node mscluster85. This is instrument provenance: it is what says
the scripts were checked against real cells rather than assumed to work.

**[The decision timeline](../1-lora-correction/report/decision-timeline.md)**

The account, in order, of what each objective asked, what ran, what came back, and the decision it
forced, append-only with superseding banners. The arc above deliberately does not follow this
order, because reading order and the order the work happened are different things.

**[What the paper does with each piece of evidence](../1-lora-correction/report/paper-evidence-index.md)**

One row per evidence folder, saying which figure of the manuscript it feeds and what state it is
in. It carries two live warnings the findings do not: the shared-structure argument behind figure 6
does not stand once the control accounts for the spread in `‖r_t‖`, and several captions have to be
rewritten before the paper may use their result.

## Still open

- [ ] Four links above name results that are not in this checkout: "the blend gallery", "does charging the adapter for its energy sharpen the fix", "where each condition lands, what these pictures mean", "what the correction is made of, what these pictures mean". Each needs its folder restored from the cluster or its link removed.
- [ ] Six older documents in this folder predate the finding format. Against the format's
      boundary with the results folder and this repo's own `CLAUDE.md` (which keeps
      pre-registrations, instrument provenance and results summaries here), they sort as:
      `experiments-log.md` and `normalization_preregistration.md` are pre-registrations and stay;
      `instrument_smoke.md` is instrument provenance and stays; `RESULTS_SUMMARY.md` is a results
      summary and stays, though every verdict it carries is a candidate finding.
      `decision-timeline.md` is a narrative of decisions in order, which the format files under
      the plan tree's history (`CHANGELOG.md` or `artifacts/notes/`), and
      `paper-evidence-index.md` is the manuscript's figure register, which belongs beside
      `writing/iclr/figures.md`. Both carry inbound links from plans, context and the archive, so
      the move is a `tidy-repo` pass with a `RENAMES.md` row each, not a quiet relocation.
- [ ] The dose-response result (root step 4), the transfer result (root step 10) and the
      three-sides result (root step 7) have review verdicts and figures under
      `artifacts/results/` and no finding file.
- [ ] The literature verdict on the three remaining failures (blend, one-animal basin, late haze)
      lives in [the pressure-test route](../1-lora-correction/artifacts/ideas/improving-the-pooled-lora-run/routes/01-pressure-test-poe-failures.md)
      and is a reading, not a measured finding, until a run tests one of its predictions.
- [ ] The energy-penalty null (root step 64, scope 01 plan 22: a Girsanov-weighted running cost on the
      adapter's correction cut its on-policy energy 27% with the fit kept and moved the held-out renders
      toward plain PoE; the adapter already spends half the true correction's energy) has its verdict in
      the review file, its figures under `1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/making-the-render-crisp/does-charging-the-adapter-for-its-energy-sharpen-the-fix/`
      and W&B run `2cfdtdnl`, and no finding file yet.
- [ ] No runbook recipe regenerates the twisted-SMC number.
- [ ] The cross-device fp16 pixel difference (about 2 of 255) quoted in
      [which checkpoint composes best](../3-understanding-compositionality/report/does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md)
      is from the session record, not a file on disk.
- [ ] The decay finding infers its cause (norm growth under weight decay 0); no run has varied
      weight decay or the learning rate, and the trainer has no flag for weight decay yet.
- [ ] The SuperDiff-residual adapters' pre-registered λ sweep on finished checkpoints never ran (trainings stopped at 61k/62k/36k of 100k); the stopped checkpoints and the cache stay under `corrector/superdiff/`.
- [ ] Two numbers in the SuperDiff findings are lifted from review files rather than re-read from scorer output: the eight-cell detector counts and the cache check; each finding's `Still open` names the file to open.
- [ ] The landing figures' three open rungs: RAE axis pictures, per-step tracks with a commit
      step, and the unseen-pair render with trajectories saved (root step 52, tasks 2 to 4).
