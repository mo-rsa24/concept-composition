# What we tried: the five project folders' own reports

This group holds the attempts recorded in the five project folders' own reports, from before and beside the research loops: the ICLR LoRA correction and its transfer to unseen pairs, the pooled adapter's training and ablations, the dose and window experiments, the samplers, searches and steering tried instead of a learned correction, the reward fine-tunes, the energy penalty and the tangent-manifold read, the classifier-guidance pushes, the method search and sentence scores, the exact toy, the inclusive-caption pilots and the text-space third expert. It holds 157 attempts: 64 in project 1, 48 in project 2, 26 in project 3, 10 in project 4 and 9 in project 5. Within each project the entries run in the order the attempts ran, by the date the page records, and attempts whose pages give no date close their project's list. Three results have no report page (the energy penalty, the tangent-manifold read and the reward fine-tunes), so their entries are taken from the review or idea files named in their evidence. Every number is copied from the page it cites, and where two sources disagree the entry records both.

## Project 1: the LoRA correction

### Does the correction point against what the plain product asks for?

- **Asked:** On the cached plain product-of-experts runs, does the correction (the joint prompt's prediction minus the product's prediction, at each denoising step) point against the product's own prediction?
- **Paper:** 2
- **Tried:** The correction was read from 38 cached SDXL runs of the plain product (no new pictures). For each run the cosine between the correction and the sampler's step was measured at every step, along with the cosine between the correction and the product's prediction. Two controls were used: a random direction, and a correction taken from the wrong step.
- **Bar, written first:** none recorded as a number. The review says each cached analysis "carries its own expected shape", and that shape is not quoted on the page.
- **Result:** The median cosine between the correction and the sampler step falls from 0.92 to 0.21 over the 50 steps (`verified`). The per-run average cosine has a median of +0.40, with no run of the 38 below zero (`verified`). The cosine between the correction and the product's prediction is negative in 38 of 38 runs, median -0.14. A random direction reads +0.000 and a wrong-step correction reads +0.11 (`stated`).
- **Verdict:** support (the review's answer is yes).
- **Why it worked:** A sampler step moves along minus the prediction, so a positive cosine with the step means the correction subtracts from what the product asks for, which is what "the product overshoots into a blend" predicts (inferred: argued, not tested separately).
- **Kind of failure:** none, the page records support, but no bar was written first, so test design applies as well. One limit is test design: the page says "The opposition result reads the plain run only", so it measures states the plain product visits, not a corrected run (stated, inferred). Its bar was only an expected shape, with no number recorded.
- **Idea still alive?** alive, on states the plain product visits: the correction opposes the product's prediction in 38 of 38 cached runs. Nothing is shown along a corrected run.
- **Rules out:** Do not describe the correction as adding something the product lacks without saying it subtracts from the product's prediction. Do not claim this read holds on corrected runs.
- **Leaves open:** whether the opposition holds along the corrected run, which the cache does not hold.
- **Evidence:** [the review of the cached runs](../../../../../1-lora-correction/plans/03-does-the-correction-cause-composition/review/04-what-the-cached-runs-already-show.md), [the alignment picture](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/how-much-correction-is-needed/plausibility_climb.png), [the causation finding](../../../../../1-lora-correction/report/what-the-correction-is/does-the-correction-cause-composition.md). No W&B run id given. (hypothesis 04)

### Does the adapter change what a word paints more than where it looks?

- **Asked:** Inside cross-attention, does the learned adapter change a word's content map (what it paints) more than its weight map (where it looks)?
- **Paper:** 2
- **Tried:** Both maps were captured with the adapter off and on at the same noisy picture, on 64 runs (8 held-out pairs at seeds 9 to 16, 384 token-step rows). The comparison removes a single overall rescale before comparing patterns, because the adapter dims the weight map by 25% overall. This was a measurement inside the model and made no new pictures for scoring.
- **Bar, written first:** a median content-to-weight pattern ratio of at least 1.2, and at least 75% of rows above 1, both required. The threshold sits in the source of `reprobe_table.py`.
- **Result:** The median ratio is 1.52 (content pattern change divided by weight pattern change). 373 of 384 rows are above 1. All 8 pairs pass, from 1.16 to 2.15 (`stated`). On the first short run (eagle and hawk, seed 9), plain relative change reads 0.236 for weight against 0.139 for content, so "where it looks moves 1.70 times more". After one rescale it reads 0.034 against 0.054, "what it paints moves 1.57 times more". The elephant and penguin pair, which composes with no adapter, reads 1.45.
- **Verdict:** support ("yes, replicated").
- **Why it worked:** The adapter alters content more than placement once uniform dimming is removed (verified, 64 runs). The first naive measure gave the opposite answer, because the adapter dims the weight map by 25% (stated, so inferred).
- **Kind of failure:** none, it passed its bar. One limit is test design: the effect also shows on elephant and penguin (1.45), a pair that needs no fixing, and "the pool holds no pair the adapter demonstrably fails to help" (verified number, inferred consequence).
- **Idea still alive?** alive, as "the adapter does this to any pair it touches", across 8 held-out pairs and 64 runs. It does not show how the fix works.
- **Rules out:** Do not compare raw relative change of the two maps; remove the overall rescale first. Do not write that this ratio explains how the fix works.
- **Leaves open:** whether the effect is specific to pairs that need fixing, which needs a pair the adapter fails to help.
- **Evidence:** [the review that set the bar](../../../../../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/review/07-what-the-fix-changes-inside-the-model.md), [the note on the measure](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/residual-dynamics/content-change-relative-to-attention-change/measure-fairness.md), [the eagle and hawk maps](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/residual-dynamics/content-change-relative-to-attention-change/smoke_eagle_hawk.png), [the appearance finding](../../../../../1-lora-correction/report/what-the-correction-is/what-does-the-correction-look-like.md). No W&B run id given. (F7)

### Does a subspace fitted on training pairs' corrections carry to unseen pairs?

- **Asked:** Does a low-dimensional subspace fitted to the corrections of 11 training pairs hold the corrections of unseen pairs, as a geometric account of why the adapter transfers?
- **Paper:** 2
- **Tried:** Corrections from the cache of 11 training pairs were stacked, and the share of each unseen pair's correction energy inside the top 64 directions was measured. The current file covers 6 unseen pairs, set beside the adapter's compose rate on those pairs (8 images per pair at training step 60,000).
- **Bar, written first:** none recorded as a number on the page. The review lists it among analyses with an expected shape only.
- **Result:** 6.0% of the unseen pairs' correction energy lies inside the subspace (`verified`), against 63.0% on training pairs (review). Per unseen pair it runs from 3.7% (goose and swan) to 9.0% (leopard and jaguar). The adapter composed 75% to 100% of the 8 images per pair, 91.7% overall in the current file. Corrections are nearly at right angles to one another, cosine about 0.00 even between training pairs (`stated`).
- **Verdict:** null at the level of vectors. The page's recorded answer is "no at the level of vectors, which says nothing about transfer".
- **Why it failed:** No fitted subspace can contain another pair's correction when corrections are mutually near-orthogonal, whether or not the adapter transfers (inferred: stated in the query note).
- **Kind of failure:** not understood. The design assumed corrections shared directions across pairs, but the cosine between pairs is about 0.00 (stated, inferred). Also test design: the test could not separate "transfers" from "does not transfer", since the adapter composed on pairs the subspace missed (verified).
- **Idea still alive?** closed as a test of transfer, because the instrument cannot separate the answers. Transfer itself is measured by compose rate elsewhere.
- **Rules out:** Do not use energy inside a training-pair subspace to predict or explain transfer to unseen pairs. Do not write "shared subspace" wording in the paper.
- **Leaves open:** whether the -0.43 rank correlation between subspace share and compose rate stands. Tied rates allow -0.49 to +0.14.
- **Sources disagree:** The compose rate reads 96.9% on the picture, the query note and the review, against 91.7% in the current file. The old count also scored comparison strips, fixed in commit `0811f029`; frog and toad stays one image off and the code does not explain it (inferred). Energy at 64 directions reads 3.3% on the picture, 13.3% in the review and 6.0% in the current file. The rank correlation reads -0.43 in the query note against -0.19 with tied ranks averaged. All recorded unresolved.
- **Evidence:** [the subspace query note](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/which-way-the-correction-points/does-the-subspace-test-predict-transfer/QUERY.md), [the subspace picture](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/which-way-the-correction-points/does-the-subspace-test-predict-transfer/geometry_vs_transfer.png) (it predates the corrected scoring), [the review of the cached runs](../../../../../1-lora-correction/plans/03-does-the-correction-cause-composition/review/04-what-the-cached-runs-already-show.md). No W&B run id given.

### When in the run does the true correction have to arrive? (sliding 10-step window)

- **Asked:** Does the compose rate peak at some position of a 10-step window in which the true correction is injected, or stay flat across the 50 steps?
- **Paper:** 2
- **Tried:** The true correction was injected over one 10-step window at a time, in nine positions (steps 0 to 10 up to 40 to 50). This used SDXL on 8 pairs at 4 seeds, 288 runs, scored by the compose scorer. Two follow-ups on cat and dog at 4 seeds came afterwards: 16 runs swapping windows and amounts, and 36 runs giving each window the same total correction.
- **Bar, written first:** the compose rate peaks at some window rather than staying flat (a flat curve "would say timing does not matter"). It was also expected to peak near step 16, the step where the plain and corrected paths visibly fork.
- **Result:** Runs composed, of 32 per window: 21 at steps 0 to 10 (rate 0.656), 8 at 5 to 15 (0.250), 3 at 10 to 20 (0.094), 1 at 15 to 25 (0.031), 0 at each of the next three windows, and 1 at each of the last two (0.031) (`verified`). Every pair's best window is steps 0 to 10. The peak did not land near step 16. The correction is about 2.7 times larger late than early, so the window that works is where it is smallest. Given the same total correction, the late window fails on all 4 seeds tested (`stated`). The matched-total curve is 0.75, 0.50, 0.25, then zero, over 4 runs per point.
- **Verdict:** support for "timing decides, at the very start". The second prediction, a peak near step 16, came out no.
- **Why it worked:** The correction acts at the start of the run. Its size there does not explain this, because size rises where the compose rate falls (verified on cat and dog). The fork step is later because it marks a visible difference downstream of the cause (inferred).
- **Kind of failure:** none, it passed its bar. Limits: test design, because the best window is the earliest one measured, so "the true best may be earlier or narrower" (stated); and the amount-matched follow-up is post-hoc, 4 runs per point on one pair, judged by eye. Could not measure it: the detector disagreed with the eye on cat and dog (stated). The second prediction, a peak near step 16, came out no.
- **Idea still alive?** alive, as "the correction must arrive at or before steps 0 to 10, and timing beats amount" over 8 pairs at 4 seeds. Where inside the first ten steps is not settled.
- **Rules out:** Do not argue the correction matters early because it is large there. Do not present the fork step (step 16) as the moment the correction matters, or draw it as agreeing with the window curve. Do not quote the amount-matched runs as population rates.
- **Leaves open:** whether the best window is earlier or narrower than steps 0 to 10, which needs a finer grid inside the first ten steps. Also why the detector disagrees with the eye on cat and dog.
- **Sources disagree:** The timing review says the rate is 0 from steps 20 to 30 onward. The result file has 1 of 32 composing at each of the last two windows (both frog and toad, one seed each). Recorded unresolved.
- **Evidence:** [the review of the timing grid](../../../../../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md), [the timing design](../../../../../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/plans/hypothesis/05-when-in-the-run-it-matters.md), [the window picture](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/when-the-correction-must-arrive/window-vs-commitment/window-vs-commitment.png), [the timing finding](../../../../../1-lora-correction/report/what-the-correction-is/when-does-the-correction-have-to-act.md). The runs went under nohup outside Slurm, so there is no job id and no W&B run id. (F4)

### Does a caption matcher read two animals as the correction is added back?

- **Asked:** As the correction is added at rising strength, does the picture's best-matching caption cross from a blend description to a two-animal description, while control pushes do not?
- **Paper:** 2
- **Tried:** CLIP image-text matching against a hand-written four-way caption bank, on 32 cat and dog runs at each strength (0, 0.25, 0.5, 0.75, 1; strength is how much of the correction is added). The real correction was compared with a same-size random push and another pair's correction. Cached images only.
- **Bar, written first:** at strength 0.75, the two-animal caption's share is at least 50% with a gain of at least 20 points, and each control gains under half the real gain.
- **Result:** The two-animal caption wins 56% of the 32 runs at strength 0.75, up from 3%. The controls gain -3 and +6 points, which is -5.9% and +11.8% of the real gain (`verified`). The blend caption falls from 59% to 25%. At strength 1 the split is 81% against 16%, but that picture is the joint render itself.
- **Verdict:** support (yes).
- **Why it worked:** Anchoring the picture to text separates a blend from two animals, where image-to-image CLIP distance could not (verified by the result). Why that is so is not understood.
- **Kind of failure:** none, it passed its bar. One limit is test design: one pair (cat and dog, 32 runs) and a hand-written caption bank (stated).
- **Idea still alive?** alive, on cat and dog only: at strength 0.75 the two-animal caption wins 56% of runs from 3%, and the controls do not move. Other pairs are untested.
- **Rules out:** Do not quote the strength 1 read as evidence, since at full strength the picture is the joint render by arithmetic. Quote strength 0.75.
- **Leaves open:** why text anchoring separates a blend from two animals when image-to-image distance does not. Also whether the result holds on other pairs.
- **Sources disagree:** The five-checks card says "56% at dose 1.0". The caption file, the picture and the review put 56% at 0.75 and 81% at 1.0. Recorded unresolved.
- **Evidence:** [the review of the five checks](../../../../../1-lora-correction/plans/03-does-the-correction-cause-composition/review/06-the-same-story-from-three-sides.md), [the caption picture](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/does-the-interaction-term-cause-composition/five-checks-from-three-sides/caption_readback.png), [the five-checks card](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/does-the-interaction-term-cause-composition/five-checks-from-three-sides/README.md). No W&B run id given.

### Do pictures slide from the plain picture toward the corrected one as the correction is added? (position in CLIP image space)

- **Asked:** As the correction strength rises, does each picture move along the axis from its plain product picture (0) to its corrected picture (1) in CLIP image space, while a same-size random push does not?
- **Paper:** 2
- **Tried:** 32 cat and dog runs at 5 strengths and 3 rows (the real correction, a random push, and another pair's correction), 480 CLIP embeddings over cached images.
- **Bar, written first:** a mean position of at least 0.30 over strengths 0.25, 0.5 and 0.75, with each control travelling under half the real distance.
- **Result:** The mean position is 0.413. A random push travels 5% of that and another pair's correction 44% (`verified`). At those strengths 66% to 91% of the picture's motion falls off the axis.
- **Verdict:** support (yes).
- **Why it worked:** The real correction moves the picture toward its corrected render and a same-size random push barely does (verified).
- **Kind of failure:** none, it passed its bar. Limits: could not measure it, because "A position on the axis is no compose score" (stated); and test design, because it is one pair and the mis-aimed control at 44% clears the half-travel bar only narrowly (stated).
- **Idea still alive?** alive, as "the real correction moves pictures toward the corrected render and a same-size random push does not" on one pair. It is not a compose claim.
- **Rules out:** Do not read the ends of the strength axis as evidence; at strength 1 the correction rebuilds the joint prediction by arithmetic. Do not treat a position on this axis as a compose score.
- **Leaves open:** whether the 44% for the mis-aimed control is a real margin, which needs more runs of that control.
- **Evidence:** [the review of the five checks](../../../../../1-lora-correction/plans/03-does-the-correction-cause-composition/review/06-the-same-story-from-three-sides.md), [the position picture](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/does-the-interaction-term-cause-composition/five-checks-from-three-sides/manifold_slide_clip.png). No W&B run id given. (F5)

### Is a blended animal wrong content or just a poor-quality picture?

- **Asked:** When the correction turns a plain product picture into a corrected one, do content-blind image quality measures change, or only the content?
- **Paper:** 2
- **Tried:** 749 cached image pairs (plain product against corrected, same seed, prompt and sampler) were scored on four quality measures: blur, noise, contrast and colourfulness. Each effect is a paired change in standard deviations.
- **Bar, written first:** each effect under 0.20 standard deviations, or p above 0.05.
- **Result:** Blur -0.13, noise +0.05, contrast +0.04, colourfulness +0.1994 standard deviations, while the compose rate went from 13.9% to 48.9% of images (323 images gained, 61 lost) (`verified`). The review gives McNemar p = 3.5e-44.
- **Verdict:** support ("yes, with colourfulness on the bar").
- **Why it worked:** The correction changes content and leaves quality nearly unchanged (verified on 749 pairs).
- **Kind of failure:** none, it passed its bar. One limit is test design: colourfulness lands on the 0.20 bar (+0.1994) rather than inside it (verified). The noise sign also differs between sources (-0.05 against +0.0461), though the page says the verdict holds at either sign.
- **Idea still alive?** alive, as "a blend is wrong content, not a poor-quality picture" over 749 image pairs, with colourfulness reported as sitting on the bar.
- **Rules out:** Do not use Laplacian sharpness or CLIP's quality preference as a content-blind quality measure.
- **Leaves open:** the sign of the noise effect, and why the sharpness measure was excluded (see below).
- **Sources disagree:** The card and the review give noise as -0.05 standard deviations; the quality file records +0.0461 and the picture prints +0.05. The card says two animals score higher on sharpness; the file records -0.09 standard deviations, p 0.79. Both recorded unresolved.
- **Evidence:** [the review of the five checks](../../../../../1-lora-correction/plans/03-does-the-correction-cause-composition/review/06-the-same-story-from-three-sides.md), [the quality picture](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/does-the-interaction-term-cause-composition/five-checks-from-three-sides/quality_control_cache.png). No W&B run id given.

### Does the prompt's text alone predict which pairs need a big correction? (additivity gap)

- **Asked:** Does the distance between the joint prompt's text embedding and the sum of the two solo prompts' embeddings (the additivity gap) predict how large a pair's correction is?
- **Paper:** 2
- **Tried:** Text encoders only, over 75 pairs and four views of the prompt. The rank correlation between the gap and the correction size was computed.
- **Bar, written first:** a rank correlation of at least 0.30 in size, with p at most 0.05.
- **Result:** +0.29, +0.00, +0.29 and +0.32 over the four views (`verified`).
- **Verdict:** null (the review answers no). The rule has two readings (below).
- **Why it failed:** A pair's difficulty is not written in its prompt embedding, so the binding lives in how the model processes the two prompts jointly (inferred, from the review). One measurement note: the concatenated view is dominated by CLIP-L, whose median embedding size is 1191 against 105 for bigG (stated).
- **Kind of failure:** the idea (inferred): a broad cloud in every view over 75 pairs, with one view reading +0.00. Also test design: the bar had no rule for combining four views, so the verdict depends on how it is read (verified from the script, line 262).
- **Idea still alive?** unclear. If any one view clearing the bar counts, the answer is yes; if the bar must hold across views, as the review argues, the answer is no. A combining rule fixed before a rerun would tell the two apart.
- **Rules out:** Before testing a text-side predictor over several views, fix how the views combine. Do not treat the concatenated view as an average of its two halves.
- **Leaves open:** whether the concatenated view means anything, given the scale gap between the encoders.
- **Sources disagree:** The review answers no. The text file marks `predicts: true` for the view joining both encoders (+0.3175, p 0.0055). The script applies the bar per view, with no combining rule. The choice is left to the owner.
- **Evidence:** [the review of the five checks](../../../../../1-lora-correction/plans/03-does-the-correction-cause-composition/review/06-the-same-story-from-three-sides.md), [the additivity picture](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/does-the-interaction-term-cause-composition/five-checks-from-three-sides/language_probe_l1_additivity.png). No W&B run id given.

### Do all pairs leave the same leftover direction in the prompt's text? (binding direction)

- **Asked:** After subtracting the two solo prompts from the joint prompt's embedding, do all pairs leave the same leftover direction, a text-side twin of a shared correction?
- **Paper:** 2
- **Tried:** Text encoders only, over 75 pairs and four views of the prompt. The share of leftover energy in the top direction was compared against a control that rebuilds the leftover using a solo prompt from a different pair, which keeps the prompt template and destroys only the binding.
- **Bar, written first:** the top leftover direction holds at least 2.0 times the share of the mismatched-prompt control.
- **Result:** 1.19 to 1.57 times (`verified`). Against an equal-size random baseline it looked like 10 to 12 times, which the review calls the wrong comparison.
- **Verdict:** null (no).
- **Why it failed:** The shared direction is the prompt template ("a X and a Y") and the near-parallel CLIP-L embeddings (mean pairwise cosine +0.999), not binding (stated in the review, inferred).
- **Kind of failure:** the idea (verified against a fair control on 75 pairs). The first baseline was also done wrong: an isotropic random floor that did not control for the shared template (stated).
- **Idea still alive?** closed for a shared binding direction in the prompt's text embedding on this template, since the fair control left 1.19 to 1.57 times against a 2.0 bar.
- **Rules out:** Do not judge shared structure in text embeddings against an isotropic random baseline. Use a control that keeps the template.
- **Leaves open:** nothing further recorded on the page.
- **Evidence:** [the review of the five checks](../../../../../1-lora-correction/plans/03-does-the-correction-cause-composition/review/06-the-same-story-from-three-sides.md), [the binding picture](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/does-the-interaction-term-cause-composition/five-checks-from-three-sides/language_probe_l3_binding.png). No W&B run id given.

### Does the step where a run's picture settles vary by pair?

- **Asked:** Does the settling step vary from pair to pair? The settling step is the first step after which the model's estimate of the finished picture keeps a cosine of at least 0.90 with its final value.
- **Paper:** 2
- **Tried:** Cache only, no GPU. The settling step was measured in latent space on the plain product path over 162 runs and 19 pairs.
- **Bar, written first:** a per-pair median span of at least 5 steps, and between-pair spread at least 1.5 times the across-seed spread.
- **Result:** Per-pair medians span 18 steps, from step 18 (dolphin and porpoise) to step 36 (cat and dog). Between-pair spread is 1.90 times the across-seed spread (`verified`). Every pair settles at step 18 or later, while the correction only works over steps 0 to 10.
- **Verdict:** support (yes, it varies).
- **Why it worked:** The settling step does differ by pair (verified). That it tracks the decision that matters is refuted by the next entry.
- **Kind of failure:** none, it passed its bar. One limit is could not measure it: the settling step is a latent-space stand-in, and the check on decoded frames the registered definition asked for is still owed (stated).
- **Idea still alive?** alive, only as "the latent settling step varies by pair" (an 18-step span against a bar of 5). The window entry shows it does not predict when the correction works.
- **Rules out:** Do not treat the latent settling step as the moment composition is decided. Do not use it without the decoded-frame check.
- **Leaves open:** the settling step on decoded frames.
- **Sources disagree:** The settling note says 19 pairs at 8 seeds each. The file holds 162 runs: 17 pairs at 8 seeds, wolf and husky at 9, cat and dog at 17. Recorded unresolved.
- **Evidence:** [the settling query note](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/when-the-correction-must-arrive/commitment-step/QUERY.md), [what settling looks like](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/when-the-correction-must-arrive/commitment-step/commitment-step-frames.png). No W&B run id given. (EXP-01)

### Does the window where the correction works move with the pair's settling step?

- **Asked:** If a pair's picture settles later, does its best correction window move later too?
- **Paper:** 2
- **Tried:** No new runs. The existing 9-window sweep (8 pairs, 4 seeds per window) was set against each pair's settling step.
- **Bar, written first:** a span of best-window centres of at least 5 steps for "moves", and at most 2 for "does not move".
- **Result:** Every one of the 8 pairs has its best window at steps 0 to 10, a span of 0 steps (`verified`). Those pairs settle between steps 23 and 36. Two secondary summaries came in weak: rank correlation +0.16 for the compose-weighted centre and +0.26 for the latest window that still works.
- **Verdict:** null for "the window moves with the pair" (recorded answer no). The span of 0 meets the pre-registered "does not move" bar.
- **Why it failed:** Whatever locks composition happens earlier than the settling step and does not show in when the estimate stops moving (inferred, from the query note).
- **Kind of failure:** test design (verified): every pair peaks at the earliest window the grid holds, so the result is censored and "the true best window may be earlier still, or narrower". Also 8 pairs at 4 seeds, so a per-pair rate moves in steps of 0.25 (stated). Also could not measure it: the settling step itself is an unvalidated latent stand-in (stated).
- **Idea still alive?** unclear. The grid cannot see whether the best window moves inside steps 0 to 10. A finer grid inside the first ten steps would tell.
- **Rules out:** Do not design per-pair correction schedules around the latent settling step. Do not claim the window is at steps 0 to 10; claim "at or before".
- **Leaves open:** where the window sits inside steps 0 to 10, and whether a wider window could matter for some pairs.
- **Sources disagree:** The window note's table gives centre 15 as cow and buffalo's latest working window; the result file gives 10, with rate 0 at 15. The note's +0.26 matches 10, while recomputing with tied ranks averaged and 15 would give +0.32. Recorded unresolved.
- **Evidence:** [the window query note](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/when-the-correction-must-arrive/window-vs-commitment/QUERY.md), [the window against settling picture](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/when-the-correction-must-arrive/window-vs-commitment/window-vs-commitment.png). No W&B run id given. (EXP-04)

### Do corrections share a few directions across runs? (spectrum against a fair floor)

- **Asked:** Is the stack of cached corrections concentrated in a few directions, which would make it learnable as shared low-dimensional structure?
- **Paper:** 2
- **Tried:** Cache only. Corrections were stacked one row per step per run and the share of energy in the top directions measured. The floor is random directions given the corrections' real row sizes, beside an equal-size random floor.
- **Bar, written first:** none recorded as a number on the page. The review lists the spectrum with an expected shape.
- **Result:** Ratios to the floor at 1 and then 8 directions kept (`verified`):
  - All rows against the equal-size floor: 14.2, then 10.8.
  - All rows against the real-size floor: 1.55, then 1.41.
  - 50 steps of one run, each row scaled to size 1 (median of 6 runs): 11.2, then 4.8.
  - One step from each of 50 runs, scaled the same way: 1.35, then 1.15.
- **Verdict:** null. The page records "no structure against a fair floor".
- **Why it failed:** Rows of unequal size concentrate energy on their own. The structure that survives lives inside a single run, whose correction turns smoothly from step to step, not across runs (verified).
- **Kind of failure:** the idea (verified against a fair, real-size floor). The first floor was done wrong: equal-size random rows gave every row the same size and inflated the ratio to 10.8 at 8 directions (verified).
- **Idea still alive?** closed for shared directions across runs on this cache. The query note says the figure's argument that shared structure makes the correction learnable "does not stand", and that learnability rests on the adapter's measured behaviour.
- **Rules out:** Do not measure low-rank structure against an equal-size random floor. Use real row sizes. Do not argue learnability from a shared spectrum.
- **Leaves open:** what the spectrum adds beyond within-run smoothness and cross-pair orthogonality.
- **Sources disagree:** The query note and the review give 440 rows; the result file has 445. The extra run being wolf and husky's ninth seed is inferred. The file's directions stop at 32, so the note's row at 64 is absent. Recorded unresolved.
- **Evidence:** [the spectrum query note](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/which-way-the-correction-points/what-the-spectrum-measures/QUERY.md), [the spectrum result](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/what-it-looks-like/which-way-the-correction-points/what-the-spectrum-measures/result.json), [the review of the cached runs](../../../../../1-lora-correction/plans/03-does-the-correction-cause-composition/review/04-what-the-cached-runs-already-show.md). There is no picture in its folder and no W&B run id. (F6)

### Does more correction give more composition? (strength sweep with direction and same-size controls)

- **Asked:** As the true correction is added back to the plain product at rising strength, does the compose rate rise, while same-size pushes in the wrong direction stay flat?
- **Paper:** 2
- **Tried:** The correction was injected into SDXL product-of-experts sampling at strength 0, 0.25, 0.5, 0.75 and 1 (strength is how much of the correction is added). Three rows were run: the real correction, a random vector of the same size, and another pair's correction. This covered 8 pairs at seeds 9 to 12, 32 runs per point, 416 images in the experiment. Scoring used the GroundingDINO instance count with a minimum box size of a quarter of the image's longer side.
- **Bar, written first:** the real curve rises monotonically from about 0 at strength 0 to about 80 to 95% at strength 1, with the random vector and the other pair's correction flat and near 0 at every strength.
- **Result:** The compose rate (the share of images with two separate animals) is 3% with none added and 94% at full strength. Both control pushes stay at or below 6%, over 32 runs per point (`stated`). The area under the compose-rate curve on a 0 to 1 scale is 0.387 for the real correction against 0.023 and 0.039 for the controls (review, verified). The real correction rises 91 points and the better control 0. The curve under the page's grid (one pair, cat and dog at seed 9, with three different controls: wrong pair, wrong seed, wrong order) climbs to about 94% with its controls under 10%.
- **Verdict:** support (yes; "these are the paper's numbers").
- **Why it worked:** Direction decides, not size: the controls match the real correction's size at every step (9.53 in all three rows) and differ only in direction (verified).
- **Kind of failure:** none, it passed its bar. Limits: could not measure it, because the detector "cannot tell which animals are present"; a two-dog picture on cat and dog seed 10 at full strength scored as composed, so 94% is an upper bound (verified, one case shown). Test design: there is no working do-no-harm control in the pool (elephant and penguin scored 0 of 4 at strength 0), and the figure strip shows the strongest pair (verified).
- **Idea still alive?** alive, as "adding the correction back causes composition by the detector's count, with direction mattering and size not" over 8 pairs at 4 seeds. 94% must be quoted as an upper bound until a count that can name each animal exists.
- **Rules out:** Do not quote 94% flat; say it is an upper bound from a count that cannot name the animals. Do not use a confidence cutoff to drop spurious boxes: a shadow scored 0.60 against a real penguin's 0.54, so filter by box size. Check that every strength is scored over the same runs. Check that the disk guard reads the filesystem the script writes to.
- **Leaves open:** the per-concept read (query "cat" and "dog" separately) that would measure the two-of-one gap. Also a working do-no-harm control pair.
- **Evidence:** [the review of the strength sweep](../../../../../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/review/03-more-correction-more-composition.md), [the design that set the bar](../../../../../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/plans/hypothesis/03-more-correction-more-composition.md), [the strength grid with its curve](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-grids/how-big-the-strength-grid-should-be/strength-grid-fitted-with-curve.png), [the causation finding](../../../../../1-lora-correction/report/what-the-correction-is/does-the-correction-cause-composition.md). The generation ran outside Slurm, so there is no job id and no W&B run id. (F2)

### SuperDiff at its own defaults, and adding its own missing residual back

- **Asked:** Does SuperDiff, a different rule for combining the two prompts on SDXL, compose cat × dog at its own defaults (200 steps, guidance 7.5)? Does clamping its blending weight κ (kappa) change that? And when it fails, what is it missing?
- **Paper:** 2
- **Tried:** SuperDiff (Skreta et al., arXiv 2412.17762) was reimplemented on SDXL in `poe_repair/composers/superdiff.py`.
  - First, an eight-cell grid: 200 against 50 steps, κ clamped or not, on cat × dog and butterfly × meadow.
  - Then 140 renders added back a fraction λ of SuperDiff's own missing residual, `r_t^SD = ε_J − ε_M`, on every step. λ is the strength it is added at. ε_J is the joint prompt's prediction and ε_M is SuperDiff's blended prediction.
  - The 140 renders ran on seeds 9 to 12 and gave seven sheets across six κ settings for cat × dog.
- **Bar, written first:** two questions were written before the grid ran: does SuperDiff still compose at 50 steps, and does clamping κ change the answer? Cat × dog was read by the validated detector and both pairs by eye. No numeric threshold is recorded. The κ × λ sheets were designed after the grid came back reversed, so the page says "nothing in rung 3 was pre-registered". The review adds that a better result at fewer steps fell outside the pre-registered categories, and recorded it as its own category, "reversed".
- **Result:**
  - **Detector counts, cat × dog seed 9.** 1 animal (a blend) at 200 steps for both clamp settings. 2 animals (composes) at 50 steps clamped. 1 at 50 steps unclamped, where the eye reads compose.
  - **Butterfly × meadow, eye only.** A blur at 200 steps and a distinct butterfly at 50, for both clamp settings.
  - **The clamp.** The bound of [−0.5, 1.5] works on synthetic input (2.3 → 1.5, −1.8 → −0.5) but never fires on a real render. The per-step κ sits between roughly 0.4 and 0.7 after the first steps.
  - **Adding the residual back, cat × dog at κ 0.5, eye read.** Seeds 11 and 12 first separate at λ 0.5 and seeds 9 and 10 at λ 0.75. That is 4 of 4 seeds by 0.75 and 0 of 4 at 0.25.
  - **Butterfly × meadow at κ 0.5.** 4 of 4 seeds by λ 0.25.
  - **At the pipeline's own unclamped κ.** The cat × dog switch is at λ 0.5 on all four seeds.
  - **What κ does.** At λ 0, κ chooses which single animal appears: a plain dog at κ 0, a plain cat at κ 1.
- **Verdict:** two verdicts. Null at 200 steps: SuperDiff does not compose at its own defaults. Support, from the post-hoc sheets, that the missing piece is a residual of the same shape PoE is missing.
- **Why it failed:** why 200 steps collapses where 50 composes has no run behind it.
  - The review names one candidate, the SDE integrator's per-step noise accumulating over 200 steps: inferred.
  - That SuperDiff lacks a residual of the same shape as PoE's was seen in the κ × λ sheets. It is an eye read on four seeds, marked stated on the page.
  - The detector counts are marked inferred: they were lifted from the review and the scorer output was not reopened.
- **Kind of failure:**
  - Test design (verified): the page says "The 50-versus-200 comparison changes the step size and the number of noise injections together, so it is mixed". The κ × λ reads are eye reads on four seeds, and the residual rung had no bar written first.
  - Not understood (inferred): the plan's categories "all assumed cutting steps could only hurt composition, never help it".
  - Could not measure it (verified): butterfly × meadow "is outside the detector's validated contract at every rung".
- **Idea still alive?** Unclear. Two things would settle it:
  - For the 200-step collapse: a run that changes the number of noise injections at a fixed step size.
  - For the residual claim: detector scores on more than four seeds with a bar written first.
- **Rules out:**
  - Do not blame PoE's failure on the combining rule alone. SuperDiff's rule also fails at its defaults and is rescued by adding back its own residual between λ 0.25 and 0.75.
  - Do not read the 50-against-200-step difference as an effect of step count alone.
  - Do not expect clamping κ to change anything: it never fires on real renders.
- **Leaves open:**
  - Why 200 steps collapses.
  - Whether SuperDiff's residual and PoE's point the same way at the same state (the next entry measures this).
  - Detector-scored κ × λ reads.
  - A detector read for the butterfly pair.
- **Evidence:** [the SuperDiff-defaults finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-superdiff-compose-at-its-own-defaults.md), [the review that set the questions](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/05-what-changes-when-superdiff-leaves-its-own-defaults.md), [the cat × dog sheet of λ by seed at κ 0.5](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/superdiff-alone-cat-dog-kappa-050-lambda-sheet.png) and [the butterfly × meadow sheet](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/superdiff-alone-butterfly-meadow-kappa-050-lambda-sheet.png). No W&B run id on the page. Renders are under `corrector/superdiff/parity/grid/` and `corrector/superdiff/lambda_sweep/`. (step 28)

### The PoE-trained correction injected into SuperDiff

- **Asked:** Do the adapters trained against the product-of-experts residual restore the second concept when injected into SuperDiff?
- **Paper:** 2
- **Tried:** the pooled PoE LoRA adapters were injected on every one of SuperDiff's 200 steps across λ, on seeds 9 to 12. The injection is `ε(off) + λ · (ε(on) − ε(off))`.
  - Adapters: rank 8 at checkpoint 410k of 450k, on both pairs; rank 16 at 85k of 100k, on cat × dog.
  - The set stopped at 49 of 96 cells.
  - One diagnostic cell measured direction: cat × dog seed 9, rank 8 at `lora_step_420000.pt`, λ 1, κ 0.5.
- **Bar, written first:** per pair and rank, the first λ at which each seed shows two separate concepts, beside the no-adapter sheet's number. "Transfers if the adapter's λ is lower on more seeds than not, indistinguishable if equal, hurts if higher."
- **Result:**
  - **First separating λ, eye read.** Never, on 12 of 12 seeds rendered. The no-adapter sheet separates at 0.5 to 0.75 for cat × dog and 0.25 for butterfly × meadow.
  - **Cosine to SuperDiff's residual.** Median cosine between the adapter's correction and `r_t^SD`, by step bucket: 0.10 over steps 0 to 19, 0.34 over 20 to 99, 0.18 over 100 to 179, 0.06 over 180 to 199. The per-step minimum is −0.40 and the maximum 0.66.
  - **Sizes.** Median size of the correction against SuperDiff's residual, in L2 units of the noise prediction, by the same buckets: 10.7 vs 8.6, 20.2 vs 15.6, 27.1 vs 17.1, 47.2 vs 15.3.
  - **Adapter attached at λ 0.** A 20-step render of cat × dog seed 9 has the same sha256 as the no-adapter render (`79238174f0b3e244`).
  - **Adapter at λ 1.** The correction is non-zero on all 20 steps, 10.0 to 78.3. Attaching matched 210 modules for each of ranks 8, 16 and 32.
- **Verdict:** null (the page: "null, it hurts", the bar's "hurts if higher" branch).
- **Why it failed:** the correction is the right size and points the wrong way: the PoE residual and the SuperDiff residual are different vectors at the same state. The cosine and norm profile was read from the sidecar, so it is verified, but on one cell, one seed and one rank. The page says it "is not a population statistic".
- **Kind of failure:**
  - The idea (verified, partly): the correction degraded every seed from λ 0.25 and washed out from λ 0.5 on 12 of 12 seeds. Its cosine to the target residual sits near 0.1 for most of the run.
  - Test design (verified): the checkpoints were not finals (410k of 450k, 85k of 100k). There are 4 seeds per cell, 47 cells went unrendered, and the read is by eye with "no detector behind it".
- **Idea still alive?** Unclear, leaning closed. Every seed got worse and the measured direction is wrong. Rendering the 47 missing cells on the final checkpoints, plus the cosine on more than one cell, would tell whether this holds or was a weak test.
- **Rules out:**
  - Do not expect a correction trained on the PoE residual to carry into another combining rule.
  - Before sweeping strengths in a new sampler, measure the cosine between the injected correction and that sampler's own missing residual.
- **Leaves open:**
  - Whether the final checkpoints change the read.
  - The 47 unrendered cells (rank 16 on butterfly × meadow, rank 32 on both pairs).
  - What SuperDiff's residual would need to be learned from.
- **Evidence:** [the PoE-correction-in-SuperDiff finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-the-poe-trained-correction-carry-into-superdiff.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/07-does-the-poe-trained-correction-reach-superdiff.md), [the rank-8 cat × dog sheet](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/poe-adapter-rank8-in-superdiff-cat-dog-lambda-sheet.png) and [the cosine and norm profile](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/poe-adapter-in-superdiff-cosine-profile.png). No W&B run id on the page. Renders are under `corrector/superdiff/transfer/grid/`. (step 49)

### An adapter trained on SuperDiff's own residual

- **Asked:** Does an adapter trained on SuperDiff's own missing residual put two animals into SuperDiff's renders?
- **Paper:** 2
- **Tried:** three LoRA adapters (ranks 8, 16 and 32) were trained on a cache of SuperDiff's residual, using SuperDiff's blend at κ 0.5. The cache is 88 cells, 17,600 step files and 11 GB, with all 200 steps kept.
  - The runs were stopped at 61k, 62k and 36k of 100k optimizer steps.
  - Every 10k checkpoint got a strip of 4 renders per pair (seeds 9 to 12) at λ 1, κ 0.5 and 200 steps.
  - Rank 16 also got a λ sweep at steps 10k and 30k.
- **Bar, written first:** per pair and rank, the first separating λ per seed on the finished adapters, beside the no-adapter and PoE-adapter sheets.
  - Pass if lower than the baseline on more seeds than not, for at least one rank.
  - Fail if equal at every rank.
  - Inconclusive for any rank whose training did not converge.
- **Result:**
  - **Cat × dog, seeds of 4 with two bodies (eye read), per checkpoint from 10k to 60k optimizer steps.** Rank 8: 3, 4, 4, 4, 3, 3. Rank 16: 4, 4, 4, 3, 0, 0. Rank 32: 4, 0, 0 at 10k to 30k.
  - **Butterfly × meadow, seeds with a clear butterfly.** Rank 8: 2, 2, 4, 0, 0, 1. Rank 16: 1, 0, 0, 0, 0, 0. Rank 32: 0, 0, 0.
  - **Training loss at the stop** (mean-squared error against the cached residual, 25-point moving mean). Rank 8: 0.0008 at step 61,000. Rank 16: 0.0025 at 62,000. Rank 32: 0.0017 at 36,000. Every curve falls and flattens, and nothing rises where the renders degrade.
  - **Rank 16 at λ 0.5.** 2 of 4 seeds with two bodies at step 30k, and 3 at 10k. At λ 1 the counts are 4 (silhouettes in a smear) and 3.
  - **The cache is the right target.** The target from one cached cell matches the composer's own norms to a relative error of 3.6 × 10⁻⁵ at steps 0, 50, 100 and 199.
- **Verdict:** inconclusive, by the plan's own criterion: the pre-registered sweep on finished adapters never ran. The page adds "with a clear direction": the adapters separate early, then smear as training goes on.
- **Why it failed:** the runs were stopped on the strip reading before the pre-registered measurement existed (verified: stopped by the user 2026-09-04 01:30).
  - Why the image worsens while the loss falls is open. Two readings fit and nothing separates them: full strength over-applies a correction fit on trajectories the adapter never steered, or the cached residual carries high-frequency texture the adapter learns late. Both inferred.
  - "Part of the smear is dose and part is learned" is an eye read, stated.
- **Kind of failure:**
  - Test design (verified): the pre-registered number "does not exist". Every read is "an eye read on four seeds", and "the detector was part of the un-run final set".
  - Done wrong (inferred): full strength may over-apply the correction. Half strength gave two clean animals on seeds 10 and 12 where λ 1 smeared.
  - Could not measure it (verified): the loss "never sees the smear", so it cannot pick the checkpoint.
- **Idea still alive?** Retry worth it. The fix is named: run the pre-registered λ sweep on the stopped checkpoints (`run_sdlora_sheets.py`), score it with the detector, and compare the per-step size of the correction against `r_t^SD` at rank 8's 30k and 60k checkpoints.
- **Rules out:**
  - Do not choose a checkpoint for this adapter by training loss: it fell and plateaued while the renders smeared.
  - Render a strip from every checkpoint while training runs.
  - Expect a larger rank to peak and degrade sooner.
  - Do not stop a run before the pre-registered measurement exists if a verdict is wanted.
- **Leaves open:**
  - The pre-registered sweep.
  - The dose reading against the learned-texture reading.
  - Whether rank 8 at 30k and λ 0.5 is the adapter to keep.
  - Re-running `check_cache_cell.py`, so the cache check is verified rather than lifted from the review.
- **Evidence:** [the SuperDiff-residual adapter finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-an-adapter-trained-on-superdiffs-own-residual-compose.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/08-adapters-that-learn-superdiffs-own-residual.md), [the separation-by-checkpoint plot](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/superdiff-lora-separation-by-checkpoint.png), [the rank-8 cat × dog timeline](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/superdiff-lora-rank8-cat-dog-checkpoint-timeline.png) and [the loss curves](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/superdiff-lora-loss-curves.png). W&B training runs `t5b00h56`, `5oz1bsa1`, `ue205g6e`; sample runs `sdlora_r8_samples`, `sdlora_r16_samples`, `sdlora_r32_samples`. (step 50)

### Is the pooled adapter's weaker fit on unseen pairs caused by drifting onto states it never trained on?

- **Asked:** The adapter is trained on states from the uncorrected run, but at inference it visits states it has corrected itself. Does that drift explain why it fits unseen pairs worse than training pairs?
- **Paper:** 2
- **Tried:** The rank 8 adapter is a small set of extra weights (a LoRA) trained on 11 animal pairs; this read used its checkpoint at step 30000. On cat and dog, held-out seeds 9 to 16, with the correction at strength 1 on all 50 steps, two measurements were taken. The first is the cosine between the adapter's correction and the true correction at the state the adapter actually reached (on-policy). The second is the same adapter on the cached, uncorrected states. The correction is the joint prompt's prediction minus the plain product's prediction (Slurm job 49421, mscluster72).
- **Bar, written first:** Written in the idea map's "Checks outstanding" before each run, not in a plan file. If the on-policy cosine starts near 0.9 and decays toward 0.4 along the run, the 0.40 is drift. If it is flat from step 0, it measures something else. The run-to-run noise floor was 0.081, which is the across-cell spread from run 1.
- **Result:** On-policy cosine is 0.817 (mean over 8 seeds and 50 steps). Per window it is 0.855 at steps 0 to 4, 0.810 at steps 5 to 24 and 0.815 at steps 25 to 49, and the per-seed means run from 0.732 to 0.884. The cosine on cached states for the same pair and adapter is 0.860. Relative state drift (how far the corrected state has moved from the cached one, as a fraction of the cached state's norm) is 0.04 to 0.08 at step 10, 0.14 to 0.27 at step 25 and 0.69 to 0.86 at step 49. The cosine between the true correction at the reached state and the cached correction at the same step index is 0.015 in the late window. That is why a read comparing the adapter against the cached target along a corrected run reads near zero by construction.
- **Verdict:** null on drift (the page records "⚪ null on drift"). The drift explanation is ruled out.
- **Why it failed:** The adapter fits its own corrected trajectory at 0.82, within 0.04 of its fit on the cache, even though the state drifts by up to 0.86 of its norm `verified`. The plan's original number, a render-time cosine of 0.54 on training cells and 0.40 on held-out cells, measured neither fit nor drift `verified`. In the code it is `direction_metrics`: the cosine between one cell's correction and the pool-mean correction. A state-specific target cannot align with a pool mean, so that number only measures how much of the correction is shared across cells.
- **Kind of failure:** the idea. Drift as the cause was tested against a bar written first and did not hold `verified` (0.817 on-policy against 0.860 cached). The test design limits how far that reaches: one pair (cat and dog), 8 seeds, one rank (8), one checkpoint (step 30000), one strength (1) `verified`. The plan's chasing of 0.40 was a case of not understood, because a pool-mean alignment was read as fit `verified`.
- **Idea still alive?** closed for cat and dog at rank 8, step 30000. unclear elsewhere: the page says whether drift matters more on a pair with a larger correction "is not measured", and the same read on such a pair would tell.
- **Rules out:** Do not read the pool-mean alignment cosine (`direction_metrics`, about 0.40) as the adapter's fit. Do not compare the adapter's output against the cached target along a corrected run, because that reads near zero by construction (0.015).
- **Leaves open:** Drift on a pair with a larger correction. Other ranks and checkpoints. Other strengths and windows. Whether fit predicts composing at all: seed 14 fits at 0.77 and still renders one animal.
- **Evidence:** [the held-out gap finding](../../../../../1-lora-correction/report/does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md) and [the idea map that wrote the bars](../../../../../1-lora-correction/artifacts/ideas/improving-the-pooled-lora-run/IDEA_MAP.md) (runs 1 to 4). Data is on the cluster at `/datasets/mmolefe/poe_repair_min/outputs/showcase/on_policy_fit_r8_030000/on_policy_fit.json`, job `49421`. No W&B id is given.

### Does the pooled adapter learn per animal or per pair?

- **Asked:** If two words the adapter saw in training are put into a pairing it never saw, does it fit them like training pairs (it learned per word) or like unseen pairs (it learned per pair)?
- **Paper:** 2
- **Tried:** The fit cosine of the same rank 8 adapter at step 30000 was measured on cached states, 10 steps per cell, in four groups. The groups were 11 training pairs (880 cells), two pairs whose words were both seen but never together (wolf and horse, lion and horse, 12 seeds each), six pairs with one word seen, and eight pairs with no word seen (Slurm job 49851, mscluster52).
- **Bar, written first:** In the idea map before the run. If seen words in an unseen pairing fit near the training 0.97, the gap is about new words. If they fit near 0.80, it is about new pairs. The noise floor was 0.081.
- **Result:** Mean fit cosine was 0.969 for training pairs. It was 0.839 for both words seen but the pair unseen (wolf and horse 0.815, lion and horse 0.862), 0.769 for one word seen (range 0.705 to 0.827) and 0.800 for no word seen (range 0.677 to 0.877). The both-words-seen group sits 0.13 below training, past the 0.081 floor, and 0.04 above the no-word group, inside the floor. On the showcase renders of cat and dog (rank 8, step 30000, strength 1), the instance-count scorer finds 7 of 8 seeds composed, with seed 14 failing on one instance. At strength 0, all eight count one instance.
- **Verdict:** support (the page records "✅ support: it is a pair problem").
- **Why it worked:** The both-words-seen group sits with the unseen group rather than with training pairs, so the adapter learns one correction per training pair and does not carry it to new pairings `verified`.
- **Kind of failure:** none, it passed its bar. Two kinds limit how far the pass reaches. Test design: the both-words-seen group is two pairs, both sharing the word horse, "so the group mean is two numbers rather than a distribution" `verified`. Could not measure it: composition was checked by an instance counter that is blind to style drift, and seed 16's plain-product render looks like two animals to the eye but counts as one `verified`.
- **Idea still alive?** alive, but only this far: at rank 8 and step 30000, the gap on held-out pairs is a property of the pair and not of fit or drift. The page names the follow-up: train next on new pairings of the 22 seen words, chosen from pairs whose experts disagree strongly (the high band of the interaction-strength page).
- **Rules out:** Do not assume that seeing both words in training lets the adapter handle their new pairing. Do not read fit cosine as compose rate, since seed 14 fits at 0.77 and still renders one animal.
- **Leaves open:** The same tier read at rank 32 and at a late checkpoint. A both-words-seen group with more than two pairs, not all sharing one word.
- **Evidence:** [the held-out gap finding](../../../../../1-lora-correction/report/does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md) and [the idea map that wrote the bars](../../../../../1-lora-correction/artifacts/ideas/improving-the-pooled-lora-run/IDEA_MAP.md). Data is at `/datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_tiers_r8_030000/fit_cosine.json`, job `49851`. No W&B id is given.

### Can a published bound on interaction strength, read from the cache, predict which pairs blend?

- **Asked:** Mitra et al. 2026 bound the error of adding two experts' scores by 2√(GM), where G is how large the interaction is and M is how curved it is. Read per pair from the cached correction, does that bound rise with the plain product's fail rate?
- **Paper:** 2
- **Tried:** The lower bound ‖r_t/σ_t‖²/4 was computed at every cached state with no model call, where r_t is the correction and σ_t is the noise level at step t. It was averaged over steps 10 to 40 per pair and compared with each pair's 8-seed fail rate under plain product-of-experts, scored by the instance counter (CPU only, about four minutes).
- **Bar, written first:** Written in the session that proposed the figure: support if failing pairs sit high and the composing control sits low, null if the bound and the fail rate are unrelated. The page says this was not in a plan file. The two-band reading was made after the run.
- **Result:** The mean bound over steps 10 to 40, in squared score-norm units, was: dolphin and porpoise 102, crow and raven 117, butterfly and flower meadow 118, seal and walrus 189, eagle and hawk 206, wolf and husky 213, rabbit and hare 234, frog and toad 257. Then a gap: cheetah and cougar 1458, cat and dog 1837, lion and tiger 2007, elephant and penguin 2237, tiger and dog 3206. The fail rate is 1.00 (all 8 seeds fail) for every named pair that has one. Post-hoc read: the ratio of band medians is about 10 (low band roughly 100 to 400, high band roughly 1300 to 3200), and the low band is pairs of near-synonyms.
- **Verdict:** null (pre-registered). A post-hoc read says the bound separates near-synonym pairs from distinct-animal pairs, and the page reports this as post-hoc, not as the tested claim.
- **Why it failed:** Every pair failing 8 of 8 fails whether its bound is 100 or 3000 `verified`. The page adds that the outcome barely varies: "Only two pairs have a fail rate below 1, so the x axis of the prediction has almost no spread to correlate against; the null is against a nearly constant outcome" `verified`.
- **Kind of failure:** test design: the outcome is nearly constant, the fail rates come from a different run than the cache, and five pairs (bear and salmon, dog and duck, cat and lion, tiger and dog, lion and dog) have no fail rate `verified`. could not measure it: the fail-rate scorer "cannot distinguish one honest wolf-husky from one cat-dog chimera" `verified` (stated as a property of the instrument). The bound is a lower bound on a product of two unknowns, and neither G nor M is recovered `stated`.
- **Idea still alive?** retry worth it. The fix the page names is fail rates for all pairs, plus a scorer that tells a plausible hybrid from a chimera, run over pairs whose fail rates actually differ.
- **Rules out:** Do not test a predictor of blending against a fail rate that is 1.00 on nearly every pair. Check the outcome has spread before correlating. Do not take the late spike past step 46 as late interaction, because it is the σ_t scaling.
- **Leaves open:** Whether the low-band "blends" are failures at all. Whether the bound predicts anything once the outcome varies.
- **Evidence:** [the interaction-strength finding](../../../../../1-lora-correction/report/does-the-fix-reach-unseen-pairs/does-interaction-strength-predict-which-pairs-blend.md) and [the idea map](../../../../../1-lora-correction/artifacts/ideas/improving-the-pooled-lora-run/IDEA_MAP.md) (run 3). Data is at `/datasets/mmolefe/poe_repair_min/outputs/showcase/interaction_bound/interaction_bound.json`. No job or W&B id is given.

### A Langevin corrector to split the correction into a sampler part and a model part

- **Asked:** If a Langevin corrector settles the SDXL product-of-experts sample before each denoising step, does part of the correction (the joint prompt's prediction minus the plain product's) disappear and part remain, so that the sampler's share can be measured?
- **Paper:** 2
- **Tried:** A Langevin corrector on the product-of-experts score of SDXL. It ran at 0, 1, 5, 20, 100 and 200 corrector steps at every one of the 50 noise levels, at three step-size multipliers (0.3, 3 and 30). Two pairs were used: cat × dog, which blends, and butterfly × meadow, which composes. Each used seed 9 only, so there is one trajectory per corrector count. The step size came from a search over ten multipliers at 20 corrector steps. `residual_curves.json` holds 1800 rows, 600 per step size. The page does not print a picture count, only one saved render per measured chain.
- **Bar, written first:** Three branches, written as constants in `1-lora-correction/scripts/corrector_residual_curve.py` before any corrector existed.
  - Support: over the last five denoising steps, the correction's size relative to the product-of-experts prediction falls by at least `MIN_DROP_FOR_SPLIT = 0.20` of its uncorrected value, with at least `MIN_REMAINDER_FOR_SPLIT = 0.20` still there.
  - Null: the change is under `MAX_DRIFT_FOR_NULL = 0.05` at every step while the chain provably moved.
  - Inconclusive: the 100-step and 200-step curves differ by more than `MAX_K_INSTABILITY = 0.05`, or the chain moved less than `MIN_CHAIN_DISPLACEMENT = 0.05`, or the composing pair behaves like the blending pair.
- **Result:** "Read-zone mean" is the correction's Euclidean norm divided by the product-of-experts prediction's norm, averaged over the last five denoising steps. It is unitless.
  - Cat × dog, seed 9, multiplier 3, at corrector counts 0, 1, 5, 20, 100 and 200: 0.153, 0.138, 0.261, 0.203, 0.103, 0.125. The 100-step and 200-step values differ by 21% of the smaller, against the 5% bar.
  - The uncorrected cell alone, measured on three GPUs: 0.153 (RTX 3090), 0.169 (RTX PRO 6000 Blackwell), 0.180 (Quadro RTX 8000). That is a spread of 15% from fp16 arithmetic with no corrector running. Neighbouring corrector counts differ by 30% to 70%.
  - The step-size search:
    - Multiplier 0.01 stalls: median relative chain displacement 0.049, under the 0.05 floor.
    - Multipliers 0.035 through 30 pass both divergence guards: maximum latent norm 1.001 to 1.113 times the chain's start, against a 1.5 bound.
    - Multipliers 100 and 300 diverge, at 3.09 and 5.24 times.
    - The search rule picked 30, whose renders are texture noise.
  - The composing pair's control, read-zone mean at corrector counts 0 to 200:
    - At multiplier 3: 0.097, 0.095, 0.099, 0.105, 0.090, 0.086, a fall of 10%.
    - At 0.3: 0.092, 0.094, 0.092, 0.093, 0.105, 0.104, a rise of 13%.
    - At 30: 0.090, 0.107, 0.134, 0.169, 0.134, 0.111, a rise of 24%.
    - The drift bar is 5%, so the control fails at two of three step sizes.
  - Results shaped like support but not licensed: at the largest corrector count, cat × dog sits 36%, 18% and 33% under its uncorrected value, with 64%, 82% and 67% still there. They are not licensed because the same pair's 100-step value at the smallest step size is 0.186, above its uncorrected 0.180.
- **Verdict:** inconclusive
- **Why it failed:**
  - The instrument scatters by more than the bars. One seed gives one trajectory per corrector count, and its read-out moves 15% across GPUs and 30% to 70% between neighbouring counts, against 5% bars (verified).
  - The search rule picked a step size whose renders are noise while every numeric guard passed (verified, from the renders). The page says the latent-norm bound cannot trip below about multiplier 170, where the Euler step stops contracting (inferred).
  - By construction, at high noise no number of corrector steps separates the gap left by noising and multiplying not commuting from the model's own gap. So the steps 0 to 10 window is out of reach (inferred).
- **Kind of failure:**
  - **Could not measure it:** an uncorrected read of 0.153, 0.169 and 0.180 across devices against 5% bars (verified).
  - **Test design:** one seed and one pair each, "Every number above is a single trajectory" (verified caveat).
  - **Done wrong:** the pre-registered step-size rule chose multiplier 30, whose renders are texture noise (verified).
- **Idea still alive?** Retry worth it. The page names the fix: seeds 10, 11 and 12 at multiplier 3 and corrector counts 0, 20, 100 and 200, on both pairs and on one device. The same three-way threshold is then applied to the seed-mean curve with the bars unchanged (about two hours on the Blackwell card). The early window stays unreachable by this method.
- **Rules out:**
  - Do not judge a 5% bar on a single-seed trajectory. Measure the uncorrected read's spread across seeds and devices first, and keep every comparison across corrector counts on one device.
  - Do not let a latent-norm guard pick a step size. Save and look at the final image of every chain.
  - Do not use a corrector residual curve to attribute steps 0 to 10.
- **Leaves open:** The sampler and model split itself. A divergence guard that sees what the eye sees. The free bound on the model's share (step 24) has not run, so the remainder has no second route to be checked against.
- **Evidence:** [the Langevin corrector finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-langevin-corrector-remove-part-of-the-correction.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/03-what-is-left-once-the-chain-settles.md), [the step-size review](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/02-the-corrector-and-the-step-size-it-runs-at.md), [the curves at multiplier 3](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/corrector-residual-curves-c3.png), [cat × dog across step sizes](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/corrector-cat-dog-across-step-sizes.png). The page gives no W&B id. Commit `0150704`. (steps 25 and 26)

### A Langevin corrector alone, adding nothing to the score, on cat × dog

- **Asked:** Does a Langevin corrector on SDXL's plain summed score, which adds nothing to it, make the product of "a cat" and "a dog" draw two animals, either in some ten-step window or over the whole run?
- **Paper:** 2
- **Tried:** 20 corrector steps per noise level, which is 60 extra network evaluations inside a window, at one step size ("the multiplier the composing pair's control survives"; the page does not print its value). There were two runs:
  - A window sweep on cat × dog seeds 9 to 12: nine ten-step window positions plus all 50 steps, 40 renders.
  - Eight-seed sheets on held-out seeds 9 to 16, with three columns (joint prompt, plain product-of-experts, corrector on all 50 steps), for cat × dog and for the control pair butterfly × meadow.
- **Bar, written first:**
  - The window question was pre-registered as a comparison recorded either way, with no pass or fail: whether the compose rate peaks in the same window as the injected correction.
  - The sheet carried one bar: the control pair's corrector column may lose at most `FIDELITY_MAX_COMPOSE_LOSS_SEEDS = 1` composed seed against its own plain-product column.
- **Result:**
  - The window sweep composed 0 of 4 seeds in every one of the 10 columns, and every one of the 40 renders counts exactly one animal. The count is the validated instance count: GroundingDINO "animal", confidence at or above 0.30, non-max suppression below 0.5 intersection over union, two or more instances.
  - For comparison, the injected correction on the same pair and seeds composes 0.656 of its runs at steps 0 to 10 and 0.250 at steps 5 to 15.
  - On the eight held-out cat × dog seeds: joint prompt 8 of 8 (1.00), plain product 0 of 8 (0.00), corrector 0 of 8 (0.00).
  - The control, butterfly × meadow, read as both concepts detected above confidence 0.30: 8, 8 and 8 of 8, so the control bar holds.
  - Five of the eight settled renders leave photography for a cartoon or a line drawing.
  - The butterfly's detector confidence, joint prompt against corrector: 0.94 to 0.73 (seed 9), 0.50 to 0.33 (seed 13) and 0.95 to 0.56 (seed 15), with every frame still green.
  - On the detector: the eye read agrees with the count on all 40 window renders and all 16 sheet renders.
- **Verdict:** null
- **Why it failed:**
  - The behaviour is verified: 0 of 8 held-out seeds and 0 of 40 window renders composed.
  - The page's explanation is that a sampler which settles the latent into the distribution the summed score actually describes does not do what adding the missing term does (inferred).
  - The drift toward illustration is verified on the pictures. Reading it as "what the product of the two diffused marginals looks like" is inferred.
- **Kind of failure:**
  - **The idea:** 0 of 8 against the joint prompt's 8 of 8, with eye and detector agreeing on all 56 renders (verified).
  - **Test design:** "One compute budget", "One step size", "One pair for the failing case", and four seeds per window column (verified caveats).
- **Idea still alive?** Unclear. At this budget the result is a clean behavioural null on one pair. The page expects a larger budget to change style, not count, but that is an expectation, not a measurement. Two runs would decide between closed and retry: the same ten columns at 100 corrector steps per level or at the smaller step size (about 40 minutes), and a second failing pair.
- **Rules out:**
  - Do not expect a training-free Langevin corrector on the plain summed score, at 20 steps per level, to produce two animals in any ten-step window or over all 50 steps. It changes drawing style, not count.
  - Do not read this null as attribution: in steps 0 to 10 the sampler's error and the model's cannot be told apart.
- **Leaves open:** Whether a larger budget composes. The style drift has no bar. Only cat × dog was tested.
- **Evidence:** [the corrector-alone finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/04-does-the-corrector-compose-in-the-same-window.md), [the window sweep picture](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/corrector-window-slides-cat-dog.png), [the eight-seed sheet](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/corrector-cat-dog-eight-seed-sheet.png), [the timing verdict it is compared with](../../../../../1-lora-correction/artifacts/plans/completed/03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md). W&B `prime_lab/poe-repair-animals-compose/s61hldbc`, commit `0150704`. (step 27)

### Twisted SMC: reweighting the plain product's own particles

- **Asked:** Does a particle sampler produce two animals if it adds no correction and only reweights and resamples plain PoE? The weights come from a classifier trained to tell joint-prompt latents from PoE latents at the same noise level.
- **Paper:** 2
- **Tried:** twisted SMC with 4 particles per cell on 3 cells: wolf × husky and lion × tiger (training pairs) and cat × dog (held out).
  - Eleven twist-classifier checkpoints up to step 100k, DDIM `eta` 1.0, 20-step runs.
  - The classifier was trained on 120 training latents and validated on 16 held-out cells from 8 pairs.
  - Every particle was scored by the validated detector, 12 per checkpoint and 132 SMC particles in all.
- **Bar, written first:**
  - Pass if the resampled particle composes at least 0.25 more often than the same particles unweighted.
  - Null if within 0.10 while the classifier's validation accuracy is at or above 0.60.
  - Inconclusive if validation accuracy never reaches 0.60.
  - The thresholds are the constants `PASS_MARGIN`, `NULL_MARGIN` and `MIN_VAL_ACC` in `poe_repair/experiments/twisted_smc/train.py`.
- **Result:**
  - **Compose fraction at step 100k** (share of images with two or more animal instances by the detector). Joint prompt alone: 1.0 (12 of 12). PoE control with no weights: 0.0 (0 of 12). Twisted SMC: 0.0 (0 of 12).
  - **SMC at other checkpoints.** 0 of 12 at 0, 10k, 20k, 30k, 50k and 90k; 3 of 12 at 40k and 80k; 2 of 12 at 60k and 70k.
  - **Detector hits against the eye.** The 10 detector hits (9 cat × dog, 1 wolf × husky) are each one animal by eye, with the second box on a collar tag or a patch of coat. So the detector reads 10 of 132 SMC particles and the eye 0 of 132.
  - **The weights were not flat.** Resampling fired 5 to 8 times per 20-step run.
  - **Validation accuracy:** 0.60 at step 1k, a best of 0.66 at 31k, 0.53 at 61k and 0.56 at 100k. Validation loss rose from 2.2 to 5.1. Training accuracy reached 1.0 by step 30k.
  - **Training loss by noise level at step 100k:** 0.00 under timestep 300, 0.00 for 300 to 700, 0.03 for 700 and above, against 0.69 at step 1.
- **Verdict:** inconclusive (the page's verdict). The page says the direction points at null.
- **Why it failed:**
  - The classifier memorised its 120 training images instead of learning composition: verified. At the highest noise the two classes are statistically the same, so a loss of 0.03 against chance 0.69 "can only come from recognising which of the 120 training latents a sample was made from". The measurement is verified and that interpretation is argued on the page.
  - Every selected and every unselected particle was one animal: verified by detector and eye.
- **Kind of failure:**
  - Wrong data (verified): 120 training latents and a head that sees pair identity let the twist memorise.
  - Done wrong (verified): validation ended at 0.56, below the 0.60 bar, with validation loss up from 2.2 to 5.1.
  - Could not measure it (verified): the detector's 10 hits are a collar tag or coat patch by eye.
  - Test design (verified): K was 4, and "the product's proposals at K 16 or 64 were not sampled".
- **Idea still alive?** Retry worth it. The fix is named on the page: "a twist that cannot memorise: more positives and a head that does not see pair identity", plus larger K.
- **Rules out:**
  - Do not train the twist on 120 latents with a head that can see pair identity.
  - Check the high-noise training loss against chance (0.69) as a memorisation alarm before using the weights.
  - Do not let the detector's count stand alone on selected particles: open every hit.
- **Leaves open:**
  - Whether a twist that reads composition changes the result.
  - Larger K.
  - Comparison with figures rendered at `eta` 0: the joint-prompt and control columns here use `eta` 1.0.
- **Evidence:** [the twisted-SMC finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md), [the checkpoint timeline](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/twisted-smc-checkpoint-timeline.png), [the twist training curves](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/twist-training-curves.png) and [every particle's score](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/twisted-smc-all-particles-scores.json). W&B `3cwrxlw0`; run dir `twist_w64_b16_lr1e-04_s100000_20260905-072331`. (plan 09)

### Feynman-Kac steering of plain PoE on the compose scorer

- **Asked:** Does Feynman-Kac steering of plain PoE's particles, with the validated compose scorer as the reward, find a two-animal image?
- **Paper:** 2
- **Tried:** Feynman-Kac steering (Singhal et al., arXiv 2501.06848) over plain PoE.
  - The reward is the detector read on the decoded running estimate.
  - Run at K 4 and K 16 particles (K is the particle count), on the 8 held-out cat × dog seeds 9 to 16, with butterfly × meadow as the composing control.
  - Particles ran at `eta` 1 with systematic resampling.
  - In all: 128 unweighted and 128 steered particles at K 16, and 32 each at K 4.
- **Bar, written first:**
  - Support if steering at K 16 composes at least 0.25 more often than the same particles unweighted.
  - Null if within 0.10 at both K 4 and K 16.
  - Inconclusive if the reward read on the step-10 estimate disagreed with the final verdict on more than half the particles.
  - Pinned as `PASS_MARGIN`, `NULL_MARGIN`, `MAX_REWARD_DISAGREEMENT` and `REWARD_BLIND_STEP` in `steer.py`.
- **Result:**
  - **Compose rate over 8 seeds, cat × dog.** Joint prompt 1.0. Plain PoE 0.0. Unweighted control 0.0. Best of the 16 unweighted 0.0. Steering at K 4 0.0. Steering at K 16 0.0.
  - **Over every particle.** 0 of 128 unweighted and 0 of 128 steered at K 16; 0 of 32 and 0 of 32 at K 4.
  - **Agreement between the reward read and the final verdict, K 16:** 0.38 at step 0, 0.99 at 10, 0.99 at 20, 0.98 at 30, 1.0 at 40.
  - **The reward hardly separated particles:** 0.625 informative resamples of 5, mean over seeds.
  - **Butterfly × meadow.** Joint prompt 0.125, PoE 0.0, control 0.0, best of 16 0.625, K 4 0.125, K 16 0.75. The page says "These are butterfly counts, not composition."
- **Verdict:** null.
- **Why it failed:**
  - The proposals were empty: all 128 unweighted 16-particle draws were single animals, so selection had nothing to find. Verified.
  - The reward was not blind: it agreed with the judge on 0.99 of particles from step 10. Verified.
  - The reward was flat because "nearly every particle scored 1 at every read". Verified.
- **Kind of failure:**
  - The idea (verified): selection can only reach what the proposals span, and 0 of 128 proposals composed. The page shows this was a fair test at K 16 with a non-blind reward.
  - Test design (verified): "16 particles per seed found no composing state; whether 64 or 256 would is not measured". The run covers 8 seeds and one pair.
  - Could not measure it (verified): the instance count cannot read the butterfly × meadow control.
- **Idea still alive?** Closed for selection over plain PoE at K 16 on cat × dog. A much larger K is the only thing left untested. The corrected proposal in the last entry is where selection helps.
- **Rules out:**
  - Do not run selection on plain PoE at K 16 expecting composition.
  - Before a selection method, count composing particles among the unweighted proposals.
  - Do not use the instance count as the compose read on butterfly × meadow.
  - Note that using the scorer as both guide and judge means any detector error would be selected for.
- **Leaves open:**
  - Larger K.
  - Selection on top of the corrected sampler (the last entry).
  - A scorer that reads the butterfly pair.
- **Evidence:** [the steering-on-plain-PoE finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-steering-on-the-scorer-find-a-composing-proposal.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/10-feynman-kac-steering-on-a-detector-reward.md), [the cat × dog sheet](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/fk-steering-cat-dog-sheet.png) and [the run summary](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/fk-steering-summary.json). W&B `czim1n0w`; run dir `fk_K4-16_s50_20260905-165958`. (plan 10)

### Zero-order search over the starting noise on plain PoE

- **Asked:** Is a two-animal image a small nudge away from each held-out seed's failing starting noise, under plain PoE, with the compose scorer choosing?
- **Paper:** 2
- **Tried:** one round of zero-order search (Ma et al., arXiv 2501.09732) with plain PoE at DDIM `eta` 0, 50 steps, guidance 7.5, 1024².
  - Each held-out seed's noise (seeds 9 to 16) was the pivot. Eight candidates were drawn near it at σ 0.1 and σ 0.3, the perturbation size, with cosine 0.995 or 0.958 to the pivot. That is 64 candidates per σ on cat × dog, plus butterfly × meadow.
  - The draw rule `z' = (z + σ·u) / sqrt(1 + σ²)` is named on the page as the project's own choice.
  - Each candidate was picked by the detector on the finished image, or on the step-10 estimate.
  - A random-search read reused the existing 8 seeds.
- **Bar, written first:**
  - Support if the kept image composes at least 0.25 more often than the unperturbed seed at either σ.
  - Null if within 0.10 at both σ.
  - Inconclusive in between.
  - Pinned as `PASS_MARGIN` and `NULL_MARGIN` in `noise_search/search.py`.
  - The hypothesis: two-animal renders on at least two of eight seeds.
- **Result:**
  - **Kept compose rate over 8 seeds, final-image verifier.** 0.125 at σ 0.1 and 0.0 at σ 0.3, against 0.0 for the pivot. That is 1 of 8, then 0 of 8.
  - **Candidates that counted.** 3 of 64 at σ 0.1, all on seed 16, with counts 2, 3 and 2. 0 of 64 at σ 0.3. Opened, the three are one two-headed body and two single animals the detector boxed twice.
  - **Random search over the existing 8 seeds.** Plain PoE composes 0.0 and the λ 1.2 corrected renders 0.875. The best plain seed by both-ness (seed 16, 0.31) still sits under the worst corrected seed (seed 10, 0.34). Both-ness is an embedding axis from the landing finding that reads how much an image looks like both animals.
  - **Step-10 verifier on cat × dog.** It kept compose rate 0.0 at both σ. It agreed with the final verdict on 0.922 of candidates at σ 0.1 and 1.0 at σ 0.3, almost all saying one animal.
  - **Step-10 verifier on the control pair.** At σ 0.3 its pick finished with no detectable animal on seeds 10, 13 and 15.
  - **Both-ness means over 8 seeds.** Joint prompt 0.523, plain PoE pivot 0.200, λ 1.2 renders 0.420. Kept by the final read: 0.235 (σ 0.1) and 0.234 (σ 0.3). Kept by the step-10 read: 0.217 and 0.246.
  - **The two-headed tile** reads 0.442.
- **Verdict:** inconclusive on the pre-registered bar, since 0.125 sits above the 0.10 null margin and under the 0.25 support margin. The page adds "a null in everything but its letter".
- **Why it failed:**
  - No two-animal noise lies in the neighbourhood sampled: 8 draws at cosine 0.995 or 0.958. Verified.
  - The one counted success is a second head, and the other two counted candidates are detector double boxes. Eye reads, marked stated.
  - The step-10 read is blind on cat × dog: verified.
- **Kind of failure:**
  - Could not measure it (verified): "two of the three counted candidates are detector errors and the third is a second head". Both-ness "cannot arbitrate that tile".
  - Test design (verified): "One round, eight candidates, two step sizes" and "Eight seeds, one pair". The paper "iterates and scales to hundreds of candidates".
  - The idea (inferred): the random-search read shows no plain-PoE seed near the corrected band.
- **Idea still alive?** Unclear. The page names what would tell a weak test from a dead idea: a verifier a second head cannot satisfy, used with a larger or iterated search. The iterated climb was left unrun on purpose, as "a climb toward detector errors".
- **Rules out:**
  - Do not climb the noise toward what the instance counter rewards: it rewards a second head and double boxes.
  - Do not verify on the step-10 estimate: it found nothing on cat × dog and dropped the butterfly on 3 of 8 control seeds at σ 0.3.
  - Do not use both-ness to judge a two-headed tile.
- **Leaves open:**
  - A larger or iterated search.
  - A verifier that requires both a cat and a dog.
  - How much of the needed correction lies within what plain PoE can render from some noise (scope 05).
- **Evidence:** [the noise-search finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-initial-noise-compose.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/11-zero-order-search-over-the-initial-noise.md), [the cat × dog sheet](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/noise-search-cat-dog-sheet.png) and [the run summary](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/noise-search-summary.json). W&B `3yxrkwgx`; run dir `zo_N8_s50_20260905-183320`. (plan 11)

### Could re-weighting the two experts have supplied the correction?

- **Asked:** In steps 0 to 10, how much of the correction lies outside anything a per-step re-weighting of the two experts (and the empty prompt) could produce?
- **Paper:** 2
- **Tried:** Four reads of one cached plain product run, cat and dog, held-out seeds 9 to 16, 50 DDIM steps at guidance 7.5. Each quantity was projected onto the span of the empty-prompt prediction and the two experts' departures from it. The rank-32 adapter (step 30050) was evaluated at the same states. Trajectories were embedded with DINOv2. Nothing was rendered.
- **Bar, written first:** the mean over seeds of the orthogonal share of the correction's squared norm, over steps 0 to 10. Above 0.5 is support (re-weighting could not supply it), below 0.25 is null (re-weighting is a candidate fix), and between is inconclusive.
- **Result:** The orthogonal share is 0.374, from 0.16 (seed 9) to 0.65 (seed 16) per seed. Per step it is 0.30 at step 0, 0.51 at step 5, 0.29 at step 10 and 0.09 at step 20. Inside the span, the joint prompt weights each expert at 1 to 3 where the product weights each at 7.5. The two experts' directions have cosine 0.67 at step 0, -0.13 at step 5 and -0.40 at step 10, so they pull against each other. The adapter's own orthogonal share is 0.194 against 0.374; the cosine between orthogonal parts is 0.55 in the early window, and between in-span parts 0.91 to 0.99. From step 10, product runs move about 3.5 times as much in DINOv2 as the joint run (median 2.79 for the plain product against 0.78 for the joint), and 5 of 8 plain runs change animal after step 10. The last three reads carry no bar.
- **Verdict:** inconclusive.
- **Why it failed:** The share sat between the bars, with a wide spread across the 8 seeds (verified). Most of the reachable part is "turn both experts down" (verified from coefficients).
- **Kind of failure:** test design. One pair, eight seeds, one adapter at one checkpoint and one correction strength; per-seed shares of 0.16 to 0.65 straddle both bars (verified). Every read is off-policy, on the plain run's states, so the joint prompt's row is a prediction at a state it did not produce (stated). Also could not measure it: the span is three vectors at one state and does not bound a method that changes the state or the schedule (stated).
- **Idea still alive?** retry worth it. The page names the fix as the experiment the number proposes: a per-step re-weighting that damps both experts to the joint prompt's weights (1 to 3 instead of 7.5). It has no plan yet. More pairs would also narrow the seed spread.
- **Rules out:** Do not claim re-weighting cannot supply the correction, or that it can; the bar was inconclusive. Do not read the span bound as covering correctors or noise search. Check that CUDA is available before a nohup chain, since the first launch stalled.
- **Leaves open:** the split of the orthogonal part into sampler error and model error. Whether the adapter's 0.19 against 0.37 shortfall is a held-out effect or an architecture effect. The joint prompt's estimate along its own run.
- **Evidence:** [the review that set the bar](../../../../../1-lora-correction/plans/05-when-does-the-outcome-lock-in/review/07-what-the-correction-is-made-of.md), [the plan with the bar](../../../../../1-lora-correction/plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md), [the finding](../../../../../1-lora-correction/report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md). W&B run `yb933cr6` in `prime_lab/poe-repair-animals-compose`. GPU reads PIDs 257033 and 257603 on mscluster85. (scope 05, plan 07)

### On cat and dog, does the plain product land apart from the joint prompt and the correction lift it there? (nearest cloud and arrow agreement)

- **Asked:** In DINOv2 image space, is no plain product picture nearest the joint-prompt cloud, and do the correction's arrows at strength 1.0 and 1.2 agree in direction?
- **Paper:** 2
- **Tried:** Five conditions on cat and dog, held-out seeds 9 to 16, the same starting noise: "a cat", "a dog", "a cat and a dog", the plain product, and the product plus the rank-32 correction (step 30050) at strength 1.0 and 1.2. That is 48 final pictures embedded with the compose scorer's DINOv2 encoder, plus decoded running estimates at 13 saved steps.
- **Bar, written first:** support needed both halves. First, no plain product picture's nearest cloud centroid is the joint cloud. Second, the strength 1.0 and 1.2 arrows have cosine above 0.9 on at least 6 of 8 seeds.
- **Result:** 5 of 8 plain product pictures sit nearest the joint centroid. 2 of 8 arrow pairs clear 0.9 (seeds 9 and 13), with the rest from 0.54 to 0.94. Post-hoc reads: all 8 strength 1.2 pictures are nearest the joint cloud. On the both-ness axis (projection toward the joint centroid, in DINOv2 cosine units), the plain product is 0.21, corrected 0.42 and joint 0.52, and the plain and corrected bands do not overlap. Commit step medians: joint 10, corrected 15, plain 22.5. By eye on the contact sheet, the plain column is one animal or one fused animal in every row and the corrected columns are two animals in every row.
- **Verdict:** null on both pre-registered halves. The page also records support on the post-hoc reads, which the review says cannot replace the question.
- **Why it failed:** Nearest centroid is a weak statistic for clouds this wide: cloud widths are 0.28 to 0.36 against centroid gaps of 0.49 to 0.80, and "A global embedding reads a fused face as 'both'" (verified numbers, inferred cause).
- **Kind of failure:** could not measure it (verified): a global embedding cannot tell a fused face from two animals, which is why DINOv2 was rejected as the compose scorer's rule. Also test design: one pair and 8 seeds, and the commit and fork thresholds were "chosen after the curves existed" (stated).
- **Idea still alive?** retry worth it. The bar's instrument could not see the outcome, while the pictures show the correction composing. The fix the page names is to pre-register a cross-pair measure (for example the decoded count axis) before the next render.
- **Rules out:** Do not use nearest-centroid in a global image embedding as a compose test. Render reference images before attaching the LoRA, or in another process: the windowed LoRA sampler leaves the adapter attached, so references rendered after it carried the correction. The contaminated set is kept unused. Check that CUDA is visible before launching on Blackwell cards.
- **Leaves open:** a cross-pair measuring tool for the plane.
- **Evidence:** [the review](../../../../../1-lora-correction/artifacts/plans/completed/05-when-does-the-outcome-lock-in/review/06-where-each-condition-lands.md), [the plan with the bar](../../../../../1-lora-correction/plans/05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md), [the animated scene](../../../../../1-lora-correction/artifacts/scenes/where-each-condition-lands/README.md), [the finding](../../../../../1-lora-correction/report/when-does-the-outcome-lock-in/where-does-each-condition-land.md). Reference renders were PID 397295 on mscluster110. No W&B run id given. (scope 05, plan 06)

### Corrector steps on the adapter's own score in the last fifteen steps, to sharpen its renders

- **Asked:** Do Langevin corrector steps on the adapter's corrected score over the last fifteen denoising steps make the adapter's cat × dog renders sharper while keeping both animals?
- **Paper:** 2
- **Tried:** The rank-32 LoRA adapter (the trained correction attached to SDXL), checkpoint at training step 30050, correction strength 1.2. It ran with 0, 5 and 20 corrector steps per level inside steps 35 to 49, on cat × dog and on the control pair butterfly × meadow, held-out seeds 9 to 16. 48 renders.
- **Bar, written first:**
  - Support: the mean Laplacian variance over the eight seeds at 20 corrector steps is at least `FIDELITY_MIN_SHARPNESS_RISE = 0.10` above the adapter alone, with 5 steps in between and the composed count within `FIDELITY_MAX_COMPOSE_LOSS_SEEDS = 1` seed.
  - Null: inside that band.
  - Composition breaks: two or more composed seeds are lost.
  - Inconclusive: the control pair loses two or more.
- **Result:**
  - Mean Laplacian variance of the greyscale render over the eight cat × dog seeds at 0, 5 and 20 corrector steps: 57.1, 61.1, 61.7. That is a rise of 8.0% against the 10% bar. The page prints no unit.
  - Medians: 37.2, 43.9, 45.4. 20 steps is sharper than none on 4 of the 8 seeds.
  - Composed seeds by the validated instance count: 7, 8 and 7 of 8.
  - The chain moved: median relative displacement 0.36 at 5 steps and 0.67 at 20 on seed 9.
  - Control butterfly × meadow: 8 of 8 in all three conditions. Its Laplacian variance rose from 506 to 726, read as the meadow's texture returning.
- **Verdict:** null
- **Why it failed:**
  - The corrector redraws fine detail seed by seed in both directions (verified on the sheet: 20 steps is sharper on 4 of 8 seeds).
  - The page's reading is that the softness lives in the adapter's low-noise steps, and that settling further into any distribution does not fix it (inferred).
- **Kind of failure:**
  - **The idea:** a rise of 8.0% with the chain shown to move, against a bar written first, on 8 seeds (verified).
  - **Could not measure it:** "Sharpness is a proxy the references break". Laplacian variance on the plain-product renders spans 12 to 499 across eight seeds of one pair (verified caveat).
- **Idea still alive?** Unclear. The bar was written first and the chain moved, but the sharpness measure is one the page itself calls broken by the references. An eye read of the eight seeds, or a fidelity measure that a contrast change cannot pass, would tell a real null from an unreadable one.
- **Rules out:** Do not use corrector steps on the adapter's own score in the tail as a sharpening fix. At 20 steps per level over steps 35 to 49 they moved mean sharpness 8.0% and did not reach the joint prompt's crispness by eye.
- **Leaves open:** Other corrector counts, step sizes and windows. Whether the result holds on a pair the adapter never trained on.
- **Evidence:** [the corrector and clean tail finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/04-does-the-corrector-compose-in-the-same-window.md), [the cat × dog tail sheet](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/corrector-on-adapter-tail-cat-dog-eight-seed-sheet.png). W&B `prime_lab/poe-repair-animals-compose/s61hldbc`, commits `87d6cb2` and `af7eae6`. (step 27, task group 3)

### Handing the tail to frozen SDXL after step 19 or 29, with or without corrector steps

- **Asked:** If the adapter is switched off after step 19 or step 29 and frozen SDXL draws the rest (with 0, 5 or 20 corrector steps on the frozen score), do the renders move nearer the joint prompt's render while keeping both animals?
- **Paper:** 2
- **Tried:** Six conditions (two cutoffs times three corrector counts), with the same rank-32 adapter at step 30050 and strength 1.2. Cat × dog and butterfly × meadow, held-out seeds 9 to 16, 96 renders. The page states that "the adapter stage runs in its own process, per the windowed sampler note". It does not say how much of the adapter's effect remains once it is switched off.
- **Bar, written first:**
  - The primary read is the cosine distance between a render and that seed's joint-prompt render, in the compose scorer's DINOv2 embedding. Lower is nearer.
  - Support: a condition moves at least `CLEAN_MIN_MONO_GAIN = 0.05` nearer than the adapter alone, with composition within one seed.
  - Laplacian variance is a secondary read only. On the plain-product references it is heavy-tailed (mean 152, standard deviation 183 over eight seeds, three of them grainy at 275 to 499). This was measured before the grid.
- **Result:** Mean cosine distance to the seed's joint-prompt render over eight cat × dog seeds, where lower is nearer:

  | Condition | 0 corrector steps | 5 steps | 20 steps |
  |---|---|---|---|
  | Adapter off after step 19 | 0.462 | 0.467 | 0.518 |
  | Adapter off after step 29 | 0.434 | 0.469 | 0.490 |

  - The references are the adapter alone at 0.472 and the plain product at 0.643.
  - The best gain is 0.038, after step 29 with no corrector, under the 0.05 bar.
  - With 20 corrector steps the renders move 0.046 and 0.018 further away, while Laplacian variance rises from 57 to 112 and 94, read as grain.
  - Composed seeds by detector: 5, 6, 7 (cutoff 19) and 6, 7, 7 (cutoff 29) of 8, against the adapter's 7.
  - The detector disagrees with the eye on two tiles. At the cutoff-20 hand-off with no corrector, seeds 10 and 11 count one animal where the eye sees two, so that condition's 5 of 8 is 7 of 8 by eye.
  - Control butterfly × meadow: 8 of 8 in all six conditions. Its distance to the joint render moved from 0.518 to 0.440 at the best condition.
- **Verdict:** null
- **Why it failed:**
  - The best gain, 0.038, is under the 0.05 bar (verified).
  - Corrector steps on the frozen score add grain and move away from the joint render (verified, from the distances and Laplacian variance).
  - The page's reading that "the hand-off is the lever that touches" the softness rests on a mixed comparison (inferred).
- **Kind of failure:**
  - **Test design:** "Mixed on one axis, said plainly". The cutoff and the corrector count change together against a baseline that differs in both. There were only two cutoffs, one adapter and one strength, and only a hard cut was tested (verified caveats).
  - **Could not measure it:** the detector reads 5 of 8 where the eye reads 7 of 8 on one condition (verified).
- **Idea still alive?** Retry worth it. The page names the next lever: correction strength ramped down across the hand-off rather than cut, and a re-noise then re-denoise of the tail. Both would be judged on this page's own bar, a 0.05 gain against the adapter-alone baseline of 0.472.
- **Rules out:**
  - Do not add corrector steps on the frozen score after the hand-off. They moved renders away from the joint render and added grain.
  - Do not read the cutoff-29 gain as a single-axis effect of the hand-off.
  - Check composition by eye beside the detector on this pair, where one body can hide behind another.
- **Leaves open:** A ramped or re-noised hand-off. Other cutoffs, adapters and strengths. Pairs the adapter never trained on.
- **Evidence:** [the corrector and clean tail finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/04-does-the-corrector-compose-in-the-same-window.md), [the plan that owns the next lever](../../../../../1-lora-correction/plans/01-showcase-the-trained-lora/plans/experiments/14-correct-early-then-clean-up.md). The clean-tail sheets are not linked: the page says they failed the subject check. W&B `prime_lab/poe-repair-animals-compose/s61hldbc`, commits `87d6cb2` and `af7eae6`. (step 27, task group 5)

### Noise search with the correction attached, picked for sharpness

- **Asked:** With the rank-32 correction attached at λ 1.2, does the best of eight nearby starting noises (picked by detector count, then sharpness) give a crisper two-animal render while the compose rate holds?
- **Paper:** 2
- **Tried:** the rank-32 adapter at step 30050, at λ 1.2 on every step as `eps_PoE(off) + 1.2 · (eps_PoE(on) − eps_PoE(off))`, with 50 DDIM steps at `eta` 0, guidance 7.5, 1024².
  - Cat × dog seeds 9 to 16, plus butterfly × meadow.
  - 8 candidates per seed at σ 0.1 and 0.3: 64 candidates per σ per pair.
  - Sharpness is the variance of the greyscale render's second differences (the Laplacian variance).
- **Bar, written first:**
  - Support if, at either σ, the kept compose rate is within one seed of the adapter's own and the median over seeds of kept sharpness over the adapter render's is at least 1.10.
  - Null if compose holds and the median ratio is within 0.05 of 1 at both σ.
  - Breaks if compose falls by more than one seed.
  - Pinned as `MAX_COMPOSE_DROP`, `MIN_SHARPNESS_GAIN` and `NULL_SHARPNESS_BAND` in `noise_search/adapter.py`.
- **Result:**
  - **Compose rate over 8 seeds.** Plain PoE 0.0. The adapter's own render 0.75. The kept image 1.0 at both σ. Of the 64 candidates, 0.94 (σ 0.1) and 0.89 (σ 0.3) count two or more.
  - **Sharpness.** Median ratio of kept over adapter render: 1.47 at σ 0.1 and 2.88 at σ 0.3. The kept image reaches plain PoE's sharpness on 5 of 8 seeds at both σ, and the joint prompt's on 5 and 6 of 8. The ratio of plain PoE over adapter is 0.81 and of the joint prompt over adapter 1.16, "so the measure does not read the softness the corrected photographs show".
  - **Eye read against the adapter's own render.** Cleaner on seeds 10, 12, 14 and 16. About equal on 13 and 15. Worse on 9 (drawn cats, no dog) and 11 (a busier sketch with more heads, counts 4 and 6).
  - **Both-ness.** The kept means are 0.39 and 0.38 against the adapter render's 0.41. The four cleaner seeds sit at 0.38 to 0.47. Seed 9 drops to 0.29 and 0.17.
  - **Butterfly × meadow.** The adapter's own render counts 0 on seeds 12, 13, 15 and 16, so it loses the butterfly on 4 of 8 where plain PoE keeps it on 8 of 8. The kept tiles count 2 or 3 on 8 of 8 at σ 0.3 and 7 of 8 at σ 0.1, always as several butterflies. Mean sharpness: joint prompt 65, plain 69, adapter render 503.
  - **The disabled adapter, as the page states it.**
    - In the smoke run, the "disabled adapter reproduces the pre-attach render to 0.0".
    - The cat × dog run on the Quadro RTX 8000 (mscluster107) is exact: mean absolute difference 0.0.
    - On the butterfly × meadow run on the RTX A6000 (mscluster109), the "detach check read 6.0 grey levels".
    - A floor check, plain PoE seed 9 rendered twice in one process with no adapter on the A6000, read 0.0. The page concludes the 6.0 is "the attach changing that card's numerics, within the same scene".
    - The page sums up: "The disabled adapter is inert on one card and within-scene on the other... the references were rendered before the attach, so nothing here depends on it."
- **Verdict:** support on the pre-registered bar. By eye, half the seeds: four gains, two holds, two losses.
- **Why it worked:**
  - Having eight candidates does the composing: random search without the sharpness key reaches the same 8 of 8. Verified.
  - The sharpness key only chooses which composing candidate is shown. Verified.
- **Kind of failure:** none, it passed its bar. Two kinds limit how far the pass reaches.
  - Could not measure it (verified) weakens the support: "The sharpness measure counts edges. It prefers drawn and sketched styles to photographs... the support on the bar is its number. The four-of-eight is the eye's."
  - Test design (verified): "One round, eight candidates, two step sizes, one pair". The adapter render here composes on 6 of 8 where the existing renders do on 7 of 8.
- **Idea still alive?** alive, but for composition only: with the correction attached, eight nearby noises give a composing render on every one of these 8 cat × dog seeds. It does not license a sharpness claim. That needs a verifier that reads clarity rather than edges, which can re-pick from the candidates on disk with no render.
- **Rules out:**
  - This settles that, with the correction attached, the compose gain comes from having candidates, not from the sharpness key.
  - Do not use Laplacian variance as a sharpness verifier for photographs: it rewards line work.
  - Do not report the bar's sharpness ratio without the eye count beside it.
- **Leaves open:**
  - A clarity verifier.
  - Whether the paper's showcase figure uses the kept tiles on seeds 10, 12, 14 and 16.
  - The correction's harm on butterfly × meadow (an illustration style, and the butterfly lost on 4 of 8 by count). The page says this belongs beside the transfer results.
- **Evidence:** [the noise-search-with-the-correction finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-noise-sharpen-the-corrected-render.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/12-noise-search-on-top-of-the-adapter.md), [the cat × dog sheet](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/noise-search-adapter-cat-dog-sheet.png), [the run summary](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/noise-search-adapter-cat-dog-summary.json) and [the disabled-adapter checks](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/noise-search-adapter-detach-checks.json). W&B `kqj26oiz` (cat × dog) and `6oai0d99` (butterfly × meadow). (plan 12)

### Feynman-Kac steering with the rank-32 correction in the proposal

- **Asked:** With the rank-32 correction at half dose inside the proposal, does steering 16 particles on the compose scorer produce two animals more often than the same particles unweighted?
- **Paper:** 2
- **Tried:** Feynman-Kac steering at K 16 on top of the rank-32 adapter at step 30050, at λ 0.5 and λ 1.2.
  - Cat × dog on the 8 held-out seeds 9 to 16, plus butterfly × meadow at λ 1.2.
  - Each tile was also scored by ImageReward against "a cat and a dog". ImageReward is a human-preference score running from about −2 to +2.
- **Bar, written first:**
  - Support if steering at K 16 composes at least 0.25 more often than the unweighted control at λ 0.5.
  - Null if within 0.10 at both λ 0.5 and λ 1.2.
  - Inconclusive if the step-10 reward disagrees with the final verdict on more than half the particles.
  - Pinned as `PASS_MARGIN`, `NULL_MARGIN`, `MAX_REWARD_DISAGREEMENT`, `REWARD_BLIND_STEP`, `ADAPTER_LAMBDAS` and `ADAPTER_JUDGED_LAMBDA` in `steer.py`.
- **Result:**
  - **Seeds of 8 the detector reads as two or more animals, at λ 0.5.** Joint prompt 8. Plain PoE 0. The correction alone 3. First particle with no weights 2. Best of 16 with no weights 8. Steered 8. Steered beats the first-particle control by 0.75, against a bar of 0.25.
  - **Mean ImageReward at λ 0.5.** Steered +0.75. Best of 16 +0.45. Joint prompt +0.24. Correction alone −0.77. Plain PoE −1.51.
  - **Particles and resampling.** The share of all 16 final particles that compose rose from 0.23 unweighted to 0.94 steered. Resampling separated particles on 4.9 of 5 opportunities per run.
  - **At λ 1.2.** The correction alone is 7 of 8, the unweighted control 8 of 8 and steered 8 of 8, so the compose difference is 0.00. Mean ImageReward: +1.16 steered, +0.97 best of 16, +0.05 correction alone, +0.24 joint prompt.
  - **What the detector counts.** The page states the scorer "counts instances of a generic 'animal' query, so it reads two separate bodies and never checks that one is a cat and the other a dog".
  - **Eye read of the 8 steered tiles at λ 0.5.** Two separate animals on 8. One clear cat beside one clear dog on 5 (seeds 9, 10, 12, 15, 16). Seed 14 is two dogs. Seeds 11 and 13 are a small animal beside a cat-dog chimera.
  - **Eye read at λ 1.2.** 7 of 8 are a cat beside a dog. Seed 12 is two cats.
  - **Per-concept detector query.** Asking for "a cat" and "a dog" separately returns both on every tile, including the fused plain-PoE tiles, so it cannot be used.
  - **Butterfly × meadow at λ 1.2.** The joint prompt counts two or more on 1 of 8 and the steered column 8 of 8, by counting several butterflies.
  - **Cost.** About 25 minutes per seed on an RTX 3090, against seconds for one corrected render.
- **Verdict:** two verdicts. Support on the pre-registered bar at λ 0.5. Null on what the resampling itself adds: best of 16 unweighted is also 8 of 8, and at λ 1.2 the compose difference is 0.00, the null branch.
- **Why it worked:**
  - Sixteen draws of the corrected sampler contain composing particles: best of 16 reaches 8 of 8. Verified.
  - The weights choose the picture: steered ImageReward +0.75 against +0.45. Verified.
  - The identity read is by eye and marked stated.
- **Kind of failure:** none, it passed its bar at λ 0.5; the same run is null on what the resampling itself adds. Two kinds limit how far the pass reaches.
  - Could not measure it (verified): the count passes two dogs and a chimera, so only 5 of 8 steered tiles at λ 0.5 are a cat beside a dog by eye. "The guide and the judge are the same scorer."
  - Test design (verified): the bar compares steering with the first particle, not with best of 16, which also reaches 8 of 8. The run covers "One pair, one adapter, one checkpoint" and "Two doses, not a curve".
- **Idea still alive?** alive, but only this far: 16 corrected draws plus selection give two animals by the count on all 8 seeds, and steering picks higher-ImageReward images. It does not license a claim that resampling adds composition, or that the animals are a cat and a dog. A scorer that separates a cat beside a dog from two dogs would tell.
- **Rules out:**
  - Always set a steered column against best-of-N with no weights, not only against the first particle.
  - Do not claim identity from the instance count or from a per-concept detector query, since both fire on a chimera.
  - Report the compute: 16 trajectories against one.
- **Leaves open:**
  - A scorer that separates a cat and a dog from two dogs.
  - Ranks 8 and 16, and pairs the adapter transfers to less well.
  - The compose rate per unit of compute.
  - Doses between 0.5 and 1.2.
- **Evidence:** [the steering-with-the-correction finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-steering-the-corrected-sampler-produce-clean-pairs.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md) and [the run summary](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/fk-steering-on-adapter-summary.json). The sheets failed the subject check and are not linked. W&B `dfrceppl`, Slurm job 50332; run dir `fka_K16_s50_20260906-025826`. (plan 13; the plain-PoE steering page calls this run "plan 11")

### On eight held-out pairs, do the plain and corrected pictures separate on the both-ness axis?

- **Asked:** On the eight held-out pairs, do the plain product's and the corrected pictures' both-ness bands stay apart? Both-ness is the projection toward the pair's joint-prompt cloud in DINOv2 space.
- **Paper:** 2
- **Tried:** 320 SDXL renders: 8 held-out pairs by 8 seeds (9 to 16) by five arms (cat-like solo, dog-like solo, joint, plain product, product plus the rank-32 correction at step 30050). Each pair got its own both-ness axis. Elephant and penguin was named in advance as the exception and counted toward neither side.
- **Bar, written first:** written 2026-09-05 before the render. Support needs bands apart, with the corrected median commit step at or before the plain one, on at least 6 of 8 pairs. Null is overlap on 4 or more.
- **Result:** Of 7 counted pairs, 2 have bands apart (cat and dog; cow and buffalo), 0 pass both halves, and 5 overlap (`verified`). Post-hoc, the share of 8 seeds where GroundingDINO detects two or more animals is 0.00 for the plain product on six of seven counted pairs (0.12 on frog and toad). The corrected arm reads 0.75 to 1.00 on all eight pairs, and the joint prompt 0.62 to 1.00. Elephant and penguin, expected to compose by default, scored 0.25 for the plain product with fusions in 6 of 8 seeds. Its corrected both-ness mean (0.15) sits below its plain mean (0.16), the surprise the bar named in advance.
- **Verdict:** null on the pre-registered bar. Support on the post-hoc instance count, which "becomes a verdict only when a plan names it as the bar".
- **Why it failed:** Both-ness does not travel across pairs. When the two animals look alike, the joint cloud sits close to the solo clouds (solo centroid gap 0.08 on leopard and jaguar, 0.15 on frog and toad), so a picture of two embeds beside a picture of one (verified numbers, inferred cause). The page says the null "says nothing against the correction".
- **Kind of failure:** could not measure it (verified): on goose and swan the plain column is one bird in every row and the corrected column two birds, yet the dots coincide. Also test design: 8 seeds per pair, and the both-ness axis is built from the clouds it judges (stated).
- **Idea still alive?** retry worth it. The fix is to pre-register the instance count, or the decoded count axis defined once on cat and dog, as the bar before the next render.
- **Rules out:** Do not use a per-pair global-embedding axis to compare composition across look-alike pairs. Do not cite the instance-count support here as a pre-registered verdict. The page says nothing about whether the count can name which animal each body is.
- **Leaves open:** a cross-pair coordinate for the plane. Re-checking elephant and penguin's "composes by default" label against a scored run.
- **Evidence:** [the review](../../../../../1-lora-correction/artifacts/plans/completed/05-when-does-the-outcome-lock-in/review/06-where-each-condition-lands.md), [the plan with the bar](../../../../../1-lora-correction/plans/05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md), [the finding](../../../../../1-lora-correction/report/when-does-the-outcome-lock-in/where-does-each-condition-land.md). Renders were PID 2437144 on mscluster106. No W&B run id given. (scope 05, plan 06)

### Keeping the adapter's correction on the manifold (the tangent-manifold read)

- **Asked:** Over steps 0 to 10, does the adapter's correction push further off the base model's manifold than the true correction does, and does dropping that off-manifold part bring the cat and dog render nearer the joint-prompt image?
- **Paper:** 2
- **Tried:** A finite-difference probe of the frozen SDXL unconditional denoiser measured each correction's "normal share" on the cached plain product-of-experts run of cat × dog. The normal share is how much the denoiser's output moves per unit of the direction: 0 means along the manifold and 1 means fully off it. The read covered 400 rows (8 seeds, 9 to 16, by 50 steps). Three conditions were then rendered with the rank-32 adapter at step 30,050: the adapter alone at λ 1.2 (λ is the strength the correction is added at), the normal part dropped on every step, and the normal part dropped from step 10 on. That made 48 renders over cat × dog and a control pair at 8 seeds. The true correction is the guided joint-prompt prediction minus the guided product prediction, and the adapter's correction is the product prediction with the adapter on minus with it off.
- **Bar, written first:** Question 1 used the ratio of the adapter's normal share to the true correction's, averaged over seeds and steps 0 to 10. Support at 1.25 or above, null at 1.05 or below, inconclusive in between or whenever the linearity check fails (`FD_MAX_LINEARITY_GAP = 0.15`). Question 2 used d(joint), the DINOv2 cosine distance to the seed's joint-prompt render (lower is nearer). Support if a projected condition's 8-seed mean d(joint) sits at least 0.05 below the adapter's, with its compose count within one seed and the control pair's compose count within one seed.
- **Result:** Question 1: the linearity gap was 0.182 against the 0.15 bar, so the rule fired. The ratio was 0.985, and 0.981 read on steps 0 to 10 against 0.978 on steps 11 to 49. Over steps 0 to 10 the true correction's normal share read 0.780 and the adapter's 0.766, against 0.994 for a random direction. Over steps 11 to 49 they read 1.665 and 1.628, against 0.902. Question 2: mean d(joint) over 8 seeds was 0.472 for the adapter alone, 0.666 with the part dropped on every step and 0.482 with it dropped from step 10. Those are gains of −0.194 and −0.010 against a +0.05 bar. Seeds where the detector counted two or more animals were 7 of 8, then 8 of 8 and 7 of 8, and the control pair kept 8 of 8 in every condition. On seeds 9, 10, 11 and 15 the every-step condition drew a grey engraving, a sepia painting, or (seed 10) three grey blobs. The projection did not hold steady when applied twice: the adapter's normal share at step 30 went from 2.17 to 2.53 after one subtraction.
- **Verdict:** two verdicts. Question 1 is inconclusive by its own linearity rule and a null on its statistic. Question 2 is null.
- **Why it failed:** `verified`: the adapter's correction is no more off-manifold than the true correction in any window (ratio 0.985). The probe's operator (the noise level times the denoiser's Jacobian) acts as a projector only over about steps 0 to 10. Past that it amplifies (share above 1), so the subtraction overshoots, and the review names that overshoot as the cause of the grey engravings. `verified`: the adapter's error (adapter minus true) reads 0.730 early and 1.090 late, which matches the random direction's profile, so "the part the adapter gets wrong has no geometry of its own". Projecting also moved the adapter slightly away from the true correction (cosine 0.867 unprojected against 0.844 for the tangent parts).
- **Kind of failure:** The idea, for question 1 (`verified`: ratio 0.985 on 400 rows, under the 1.05 null bar, and the review calls the hypothesis "dead"). Not understood (`verified`: the design read the Jacobian probe as the off-manifold projector, and the step-size sweep, jobs 50344 and 50346, showed no linear regime and an amplifying operator past step 10). Done wrong, for question 2's every-step condition (`verified`: it applied an overshoot rather than a projection on about forty of fifty steps). Test design (`verified`: one pair and 8 seeds decide question 2).
- **Idea still alive?** Closed for "the softness comes from the adapter's direction being more off-manifold": the statistic sits under the null bar on both windows. Unclear for a projection as a fix. The review names a least-squares removal built on vector-Jacobian products as the version that would work whatever the eigenvalue, but its own cache read already says the adapter's error does not sit in the normal part. The training-time penalty on the normal part is closed by the review because question 2 is null.
- **Rules out:** Do not read the noise level times the denoiser Jacobian as the off-manifold projector on SDXL past step 10. Check it gives the same answer when applied twice, and run a step-size sweep, before using any finite-difference projector. Do not score sharpness by Laplacian variance on renders whose style changed: the every-step condition read 79.3 against the adapter's 57.1 because engraving lines count as edges. Do not trust the instance count on style-broken renders: it returned 2 on three grey blobs (seed 10) and 3 on a sepia cat and kitten (seed 11).
- **Leaves open:** Why the corrected render is soft, if not from the direction. The same normal-share curve on the corrected trajectory rather than the cached plain-product one. A vector-Jacobian least-squares removal, whose code path (`--conditions projected_ls`) exists but was deliberately not rendered.
- **Evidence:** [the tangent-manifold review](../../../../../1-lora-correction/plans/01-showcase-the-trained-lora/review/15-keep-the-correction-on-the-manifold.md) (a review file, not a report finding); [the design that set the bar](../../../../../1-lora-correction/plans/01-showcase-the-trained-lora/plans/experiments/15-keep-the-correction-on-the-manifold.md); W&B `sgvdb9gl`; Slurm jobs 50342 (main run), 50344 and 50346 (step-size sweep). (scope 01, plan 15)

### Charging the adapter for the energy of its correction

- **Asked:** Does adding a running cost on the size of the adapter's correction during training render the held-out cat and dog nearer the joint-prompt image, with the same seeds still composing?
- **Paper:** 2
- **Tried:** The rank-32 pooled adapter `phase1_r32_100k` was resumed from step 30,050 for 10,000 steps at three prices, with β (the multiplier on the added energy term) at 0, 0.01 and 0.05. The energy term is a Girsanov control energy in nats. All three arms used one checkpoint, one state sequence and one launch shape. At step 40,050 each arm rendered cat × dog at 8 held-out seeds at λ 1 (the strength the correction is added at), and the on-policy energy was read over the same 8 seeds.
- **Bar, written first:** The bar is in source constants of `energy_penalty_readout.py`. Drift is a render's DINOv2 distance to the joint-prompt render minus its distance to the plain-product render, mean of 8 seeds, and negative means nearer the joint prompt. Support if β 0.05 composes at least 6 of 8 and its drift is at least 0.03 below β 0's. Null if it composes under 6, if the drifts differ by less than 0.03, or if β 0.05's drift is 0.03 or more above β 0's. Validity check: the control must compose at least 6 of 8 and sit within 0.03 of the shipped lineage's own drift at 40,050. Secondary bar on contrast (closure of the grey-level standard deviation gap to the joint prompt): support at 0.5, null at 0.1 or less. Energy bar: support if β 0.05's energy is at most 0.7 times the control's and its mismatch at most 1.10 times; null above 0.9 times; inconclusive in between.
- **Result:** β 0.05 composed 6 of 8 seeds with drift +0.019, against the control's 7 of 8 and −0.054. β 0.01 composed 6 of 8 with drift −0.003. Contrast was 35.0 for β 0.05, 36.1 for the control and 47.5 for the joint prompt, a closure of −0.10. On-policy energy was 3,420 nats against the control's 4,671, a ratio of 0.732. Mismatch (the energy of the adapter's correction minus the true one) was 3,034 against 2,987, a ratio of 1.016. The late share of the spend went from 74% to 72%. At every checkpoint the adapter spends 3,400 to 4,700 nats where the true correction at the same states would cost 7,200 to 10,300. The control failed the validity check narrowly: drift −0.054 against the lineage's −0.090, a gap of 0.036 over the 0.03 tolerance, so the arms were read only against each other. By eye on seed 9, β 0.05 fuses the cat and dog into one body with two heads. The sources agree on the headline: the root report index says the price "cut its on-policy energy 27% and moved the held-out renders back toward plain product-of-experts", and the review says the same (27% cut, renders moved away from the joint-prompt image). The index leaves out that the review rates the energy question "inconclusive by the letter" (ratio 0.732 against the 0.7 support bar).
- **Verdict:** Null on the main question ("the price hurt"). Null on contrast. Inconclusive by the letter on the energy question, with the fit kept.
- **Why it failed:** `verified`: the adapter already spends about half the true correction's energy at the states it visits, so there was no excess to remove, and the price took energy the correction needed. `verified`: the penalty shrank the correction at every step alike (late share 74% to 72%), so the prediction that the late share would fall first was wrong. `verified`: the effect grows with the price in the harmful direction: energy 4,671, 3,733 and 3,420 nats, and drift −0.054, −0.003 and +0.019 at β 0, 0.01 and 0.05. `inferred` (asked after, not checked): the running cost may only act as a learned λ below 1.
- **Kind of failure:** Not understood (`verified`: the design assumed the haze was excess energy, and the on-policy read shows the adapter under-spends). The idea (`verified`: a clean comparison where the configs differ in `energy_penalty` alone, and three doses move monotonically the wrong way). Test design, as a weakness (`verified`: one pair, 8 seeds, the control failed the check against the shipped lineage by 0.036 against 0.03, and the arms resume rather than train from step 0).
- **Idea still alive?** Closed for "the haze is excess energy": a one-axis comparison with a monotone harmful dose and a direct energy measurement showing no excess. The review names on-policy training as the follow-on.
- **Rules out:** Do not price the size of the correction to sharpen the render: the adapter spends about half what the true correction costs. Measure the on-policy energy against the true correction's before adding any energy cost. Do not read resumed arms on a different card or cached-state sequence against the shipped lineage without checking the control first (its drift moved from −0.090 to −0.054).
- **Leaves open:** Whether the penalty is just a smaller λ (a cosine check on cached states). Why the adapter composes while spending half the true energy. On-policy training (states from the adapter's own trajectory). A fresh run with the price from step 0.
- **Evidence:** [the energy-penalty review](../../../../../1-lora-correction/plans/01-showcase-the-trained-lora/review/22-charge-the-adapter-for-its-energy.md) (a review file, not a report finding); [the design that set the bar](../../../../../1-lora-correction/plans/01-showcase-the-trained-lora/plans/experiments/22-charge-the-adapter-for-its-energy.md); [the root report index](../../../../../report/00-INDEX.md), section 8; W&B `2cfdtdnl` (readout), `4cv3mtk1` (β 0), `47z6ydvv` (β 0.01) and `whdcr8wy` (β 0.05); Slurm jobs 50353 to 50356. The figures folder the index links is not in this checkout. (root step 64, scope 01 plan 22)

### Training the correction on real photographs of both animals

- **Asked:** Can the adapter train against real photographs holding both animals as separate objects, instead of against the model's own joint-prompt prediction?
- **Paper:** 2
- **Tried:** Candidate photographs for 18 animal pairs were retrieved from 6,259,883 DataComp-1B captions, a 1-in-224 sample. URL liveness was checked, 276 images were downloaded and filtered with a detector, and the accepted images were sorted by eye. No training ran.
- **Bar, written first:** The promotion rule: this idea becomes a plan if the walk returns a reachable corpus. The filter's false-positive bar was 0.10.
- **Result:** Every one of the 18 pairs had candidate captions: cat and dog 3,035, turtle and tortoise 99, cheetah and cougar 2. URLs resolved at 77% to 88%. For cat and dog, the filter accepted 16 images, of which 7 were genuine photographs of both animals. For turtle and tortoise it accepted 11, of which 0 were genuine; the rest were watermarked stock plates and studio product shots. On images whose caption names one species only, the filter reported both species present at 0.070 for cat and dog (under the 0.10 bar) and at 0.323 for turtle and tortoise (over it).
- **Verdict:** closed (the page records "❌ Dead. The corpus is reachable and unusable").
- **Why it failed:** The corpus has few genuine two-animal photographs `verified`. The filter cannot separate look-alike species, and the training pool is look-alike species throughout `verified` (0.323 false positives). The one pair where retrieval worked, cat and dog, is the held-out test pair and appears in no training cell pool, so it could not have been used anyway `stated`. The page also disproves a worry: co-occurring pairs are not already composing. Turtle and tortoise co-occur and fail 8 of 8, elephant and penguin never co-occur and fail 8 of 8, and all 18 animal pairs fail under plain product-of-experts `stated`.
- **Kind of failure:** wrong data (`verified`: 7 genuine of 16 accepted, 0 of 11). could not measure it (`verified`: the filter's 0.323 false positives on look-alikes). Test design weakens the closure: training itself was never run, and the corpus was a 1-in-224 caption sample.
- **Idea still alive?** unclear. The training on real photographs never ran, so the idea itself was not tested. Only the corpus failed, on a sampled caption set and with a filter that passes look-alike species. A larger sample sorted by eye would tell whether a usable corpus exists (inferred). The page routes to a replacement: training on model renders a person selected by eye.
- **Rules out:** Do not filter a look-alike-species corpus with a detector that cannot tell the species apart. Check whether a retrieved pair is a held-out test pair before counting it as training data.
- **Leaves open:** Whether a full-size, hand-sorted corpus exists for the training pairs. Whether training on real photographs helps at all.
- **Evidence:** [the real-photographs finding](../../../../../1-lora-correction/report/designing-the-correction-loss/05-training-on-real-photographs.md), [the idea map](../../../../../1-lora-correction/artifacts/ideas/designing-the-correction-loss/IDEA_MAP.md) (claim 6), and [the replacement plan, training on renders that composed](../../../../../1-lora-correction/artifacts/plans/parked/09-designing-the-correction-loss/plans/hypothesis/05-training-on-the-renders-that-composed.md). The results folder the page links is not on this laptop. No W&B id is given.

### Sweeping the step at which the adapter hands the render back to frozen SDXL

- **Asked:** Does switching the trained adapter off for the last stretch of a 50-step render bring the picture nearer that seed's joint-prompt render ("a cat and a dog") while keeping both animals?
- **Paper:** 2
- **Tried:**
  - `k` is the step at which the adapter is switched off. The adapter runs on steps 0 to k-1 and frozen SDXL's plain product draws the rest. `k` took 25, 29, 33, 36, 40 and 50, where 50 is the control with the adapter on the whole path.
  - Three saved adapters: rank 16 at 30,000 training steps, and the pooled rank 32 at 40,000 and at 60,000 steps.
  - Ten cells each, where a cell is one pair at one seed: cat × dog on seeds 9 to 16, elephant × penguin on seeds 9 and 10.
  - 50 DDIM steps, guidance 7.5, correction strength 1.0, 1024 by 1024. The page prints no total picture count.
- **Bar, written first:**
  - Gain is the control's DINOv2 cosine distance to the joint-prompt render minus this render's distance, so positive means nearer. The required gain is 0.05.
  - The review's support rule: five or more of the eight cat × dog seeds must clear 0.05.
  - The review's composition rule: two or more composed seeds lost "at the `k` that wins on fidelity" breaks composition.
  - The script as run used a different logic: the tied `k` listed first wins, one clearing cell is enough, and a net fall of more than one cell breaks composition.
- **Result:**
  - Only 5 of the 10 cells have a joint-prompt render that shows both animals, so only those 5 count.
  - Results by adapter:

    | Adapter | Readable cells clearing 0.05 | Largest gain on a readable cell | Cells with two or more animals of 10, k = 50 then k = 25 | Script printed |
    |---|---|---|---|---|
    | Rank 16 | 1 of 5 (cat × dog seed 9) | +0.093 (seed 9, k = 25) | 6 then 4 | `composition broke` |
    | Rank 32, 40,000 steps | 0 of 5 | +0.037 (seed 9, k = 29) | 9 then 9 | `null` |
    | Rank 32, 60,000 steps | 0 of 5 | +0.039 (seed 12, k = 25) | 9 then 9 | `null` |

  - On rank 16, k = 25, 29, 33 and 36 tie, each clearing on one cell. Cells with two or more animals at those k: 4, 3, 5 and 6 of 10. Only four of the eight cat × dog seeds (9, 11, 12 and 16) have a usable joint-prompt render, so the review's five-seed rule cannot be met.
  - The page records seven disagreements between sources, left unresolved:
    - Whether `k` is "the step after which the adapter is detached" (the card and the review) or "off from step k" (the script and the sheets).
    - The card says the instance count is "unchanged at 9 of 10 cells at every k" on rank 32. The sidecars say the raw count differs from k = 50 on 5 of the 50 comparisons at each checkpoint.
    - The card says sharpness falls "on almost every cell". The sidecars say 8 of 10 cells at 40,000 steps and 7 of 10 at 60,000.
    - Where the three reference reads are stored.
    - The card says "all four broken" cat × dog references. The script says three broken (seeds 10, 13, 14) and one ambiguous (15).
    - The review's opening says "Nothing has run". Its own run table records the run on 2026-09-19.
    - The `lap` sharpness number is named Laplacian variance, but the function returns 0 on a straight edge, where a 4-neighbour Laplacian variance returns 2097.6 on a 64 by 64 half-black, half-white test image.
- **Verdict:** none recorded: the review's answer boxes are empty. The script printed `null` for both rank-32 checkpoints and `composition broke` for rank 16. Under the review's five-seed rule rank 16 is null. With the tie going to k = 33 or 36, the script's own logic would give support on one cell of five. The page leaves the reading to the owner.
- **Why it failed:**
  - On rank 32 no readable cell reached 0.05 at any k (verified, from the sidecars).
  - The review's explanation: the adapter lowers the contrast of the picture the composition aims at by 56% over steps 0 to 9 and by 1% over steps 40 to 49, so the flattening is finished before step 25. This was measured on one seed (cat × dog seed 9, 40,000-step checkpoint), and the page marks it stated (inferred).
- **Kind of failure:**
  - **Could not measure it:** half the references are broken, with only 5 of 10 cells usable (verified). The `lap` function is not a Laplacian variance (verified on a test image). The review records that distance to the joint prompt's picture is not the standard the project judges by, since a held-out pair has no joint prompt at inference (stated).
  - **Test design:** the rule did not say how to break a tie among four values of k, and its five-seed rule cannot be met with four usable seeds (verified).
  - **Not understood:** if the flattening is finished before step 25, a hand-off at step 25 or later cannot reach it (inferred, one seed).
- **Idea still alive?** Unclear. Rank 32 gives 0 of 5 under every reading, but on five usable cells and against a target the review calls the wrong standard. Three things would tell closed from retry: an eye read of these renders by the project's own standard (both concepts plainly shown and the picture looks good), repaired references, and the contrast measurement repeated on more seeds.
- **Rules out:**
  - Check that every seed's joint-prompt render shows both animals before scoring distance to it.
  - Write how ties between settings are broken before the run.
  - Do not quote these per-seed gains beside the earlier 0.038 mean, which used a different checkpoint, strength 1.2 and a mean over eight seeds.
  - Do not call the `lap` number a Laplacian variance without checking the function.
- **Leaves open:** Which k wins. Whether composition held there. Whether the eye and the distance pick the same k. Whether the training stage was needed.
- **Evidence:** [the hand-back sweep page](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-handing-the-tail-to-frozen-sdxl-bring-renders-nearer-the-joint-prompt.md), [the review that set the bar](../../../../../1-lora-correction/plans/09-designing-the-correction-loss/review/06-fitting-the-whole-path-and-handing-back-the-tail.md), [the design](../../../../../1-lora-correction/plans/09-designing-the-correction-loss/plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md), [the rank-32 40,000-step cat × dog sheet](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/clean-tail-k-sweep-pool43-all50-40k-a-cat-x-a-dog.png). No W&B run, per the review. Run on mscluster109 device 1, 2026-09-19 22:27 to 23:24.

### Every adapter on cat and dog at three held-out seeds

- **Asked:** Which of the trained adapters composes the held-out pair cat and dog, from the same starting noise?
- **Paper:** 2
- **Tried:** 23 trained adapters each rendered cat and dog at seeds 9, 11 and 12, which no adapter trained on. That is 69 adapter tiles plus a no-adapter control per seed, all read by eye. Separately, two adapters at training step 45,000, `v57-09-x0-seeds` (W&B `ek4w6lbu`) and `v57-10-x0-contrast` (W&B `boammtzo`), were set beside the joint-prompt render on four held-out cells.
- **Bar, written first:** none recorded.
- **Result:**
  - By eye (verified): most adapters keep the look their seed has with no adapter, so the seed decides photograph or drawing. The two adapters trained on filtered pictures draw "a photograph in a green field" on all three seeds. Two adapters draw grey mush with no clear animal on seeds 9 and 11: rank 8 at step 450,000, and the 54-cell adapter with the plural-prompt loss term.
  - The joint-prompt render is itself wrong on two of the four cells: two dogs at cat and dog seed 10, and an elephant with no penguin at elephant and penguin seed 9.
  - `v57-09-x0-seeds` draws a tabby and a white dog (cat and dog seed 9), a cat and a dog (seed 10), fused elephant and penguin bodies (elephant and penguin seed 9), and an elephant, a penguin and a small extra elephant (seed 10).
  - `v57-10-x0-contrast` is the same except at cat and dog seed 9, where it draws "a cat and a dog held by two invented people".
- **Verdict:** none recorded: "Three seeds of one pair give no rate, so these sheets cannot rank the adapters."
- **Why it failed:** No adapter was ranked. The page says every read is by eye, and "Columns differ in more than one setting, so a gap between two of them is a mixed comparison" (verified).
- **Kind of failure:** **test design** (verified): three seeds, one pair, by eye, no rate, and mixed columns.
- **Idea still alive?** unclear. These sheets cannot tell a better adapter from a seed effect. A scored rate over more seeds and pairs would.
- **Rules out:** Do not rank adapters from these three-seed sheets. Check the joint-prompt render before using it as the reference: at cat and dog seed 10 it draws two dogs.
- **Leaves open:** The card says like-for-like training of the first-10-steps adapter "made no significant difference", and that comparison is not linked. **Where the sources disagree** (recorded, not resolved): the card says "10 adapter columns plus the no-adapter control" for the paper sheet, while the sheet has 10 columns in all (9 adapters and the control), and the grouping code agrees with the sheet. The card says "Every adapter this project has trained", while the sheets hold 23 and the render script lists 25. The later v58 adapters are in neither.
- **Evidence:** the [every-adapter finding](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/which-adapter-composes-cat-and-dog.md) and its [card](../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-cat-and-dog/README.md). No sheet is shown because none cleared the pair ledger.

### Switching the adapter off early while drawing tiger and parrot

- **Asked:** Does the adapter still separate tiger and parrot when it is switched on for only the first 10 or 20 drawing steps instead of all 50?
- **Paper:** 2
- **Tried:** An adapter labelled "run 9" was switched on for 10, 20 or all 50 drawing steps on tiger and parrot at seeds 9 to 12 (12 adapter pictures), beside the joint prompt, the plain product and columns labelled "original". This was part of a September search for paper figures, read by eye.
- **Bar, written first:** none recorded.
- **Result:** By eye (verified): the joint prompt draws both animals on all four seeds and the plain product fuses all four. With run 9 on for 10 steps, all four seeds stay fused. On for 20 or 50 steps, only seed 9 draws a clean tiger and parrot. Seed 10 gives fused heads. Seed 11 gives a parrot with a hybrid at 20, and a parrot, a tiger and an extra small tiger at 50. Seed 12 gives two parrot-tiger hybrids. The "original" columns draw two bodies at 20 and 50 steps with the two animals' patterns mixed across them.
- **Verdict:** none recorded: the sheets "were made to find good pictures, so they cannot give a success rate."
- **Why it failed:** not recorded. The page gives no reason. "No file defines 'run 9' or 'original'."
- **Kind of failure:** **test design** (verified): one pair, four seeds, by eye, and an unnamed adapter.
- **Idea still alive?** unclear. With four seeds and no named adapter, the record cannot tell a weak short window from a weak adapter. A named adapter on more pairs and seeds with a count would. The 10-step result points the same way as the steps-0-to-9 null above (inferred link across pages).
- **Rules out:** Do not draw with the adapter on for only 10 steps expecting tiger and parrot to separate: all four seeds stayed fused. Name the adapter in a card before running a figure search.
- **Leaves open:** Which adapters "run 9" and "original" are. Whether 20 steps matches 50 on other pairs.
- **Evidence:** the [render-search finding](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/which-renders-show-it-working-and-failing.md) and the [switch-off sheet](../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/adapter-switch-off/a_tiger__x__a_parrot.png).

### Spread match and handing over to a second adapter while drawing

- **Asked:** Do two drawing-time changes turn the fused tiger and parrot seeds clean? The first, spread match, rescales the prediction's spread to the single prompts' average spread and blends it back in. The second hands over to another adapter at step 20.
- **Paper:** 2
- **Tried:** Run 9 with spread match (blend weight not recorded), and run 9 handing over to the "original" adapter at step 20 (`--late-checkpoint`, `--switch-at`). Tiger and parrot at seeds 9 to 12, by eye.
- **Bar, written first:** none recorded.
- **Result:** By eye on the four seeds (verified): "neither change alters what each seed draws: the same seeds stay clean or fused". Spread match and the handover "leave each seed's layout where run 9 put it".
- **Verdict:** none recorded: a figure-search sheet with no rate.
- **Why it failed:** not recorded beyond the observation that each seed's layout stays as run 9 set it (verified by eye).
- **Kind of failure:** **test design** (verified): one pair, four seeds, by eye, unnamed adapters, and the spread-match weight not recorded.
- **Idea still alive?** unclear. One unrecorded setting on four seeds cannot separate a change that does nothing from one set too weakly. A recorded weight swept on more seeds would.
- **Rules out:** Do not expect spread match or a step-20 handover to change which tiger and parrot seeds come out clean at the setting used here. Record the blend weight when rerunning.
- **Leaves open:** Whether any spread-match weight or handover step changes the layout.
- **Evidence:** the [render-search finding](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/which-renders-show-it-working-and-failing.md) and the [spread-and-handover sheet](../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/spread-and-handover/a_tiger__x__a_parrot.png).

### Adding the contrast corrector to the adapter

- **Asked:** Does a contrast corrector, added to the adapter, push the concept that is taking over away and leave two separate animals? The corrector doubles the product's predicted clean picture, subtracts the two concepts with weights summing to 1, then rescales to the product's size.
- **Paper:** 2
- **Tried:** The corrector added to run 9, from commit `4c51b0bd`. The tile name `run9c6` suggests it ran over the first 6 steps (inferred). It was opened on 8 renders: tiger and parrot, and goat and cat, at four seeds each.
- **Bar, written first:** none recorded.
- **Result:** All 8 renders "come out flat or saturated", poster art (verified by eye). Example: goat and cat at seed 11 is a flat black and white poster of a dog-like figure, a horned goat and a small cat-like animal.
- **Verdict:** none recorded: every render was damaged, and no rate was taken.
- **Why it failed:** not recorded. The page gives no reason.
- **Kind of failure:** **test design** (verified): one setting, 8 renders, by eye. **done wrong** is possible but not shown, because the strength setting is not recorded and the window is only inferred from a filename.
- **Idea still alive?** unclear. One setting that flattened every render cannot tell a bad idea from a bad strength. A sweep of the corrector's strength and step range on named seeds would.
- **Rules out:** Do not add the contrast corrector at the `run9c6` setting: it flattened all 8 renders. Record the corrector's strength and steps before rerunning.
- **Leaves open:** Whether a weaker or shorter corrector separates animals without flattening.
- **Evidence:** the [render-search finding](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/which-renders-show-it-working-and-failing.md) and the [goat and cat corrector tile](../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/which-renders-show-it-working-and-failing/contrast-corrector/a_goat__x__a_cat/seed11/ours_run9c6_w50.png).

### Eight render-time fixes for the adapter's picture-quality cost

- **Asked:** Can any fix that needs no retraining recover the picture quality the adapter costs, without changing its layout or subjects?
- **Paper:** 2
- **Tried:**
  - Eight treatments and two references (the joint prompt and the plain product), from the same starting noise and the same checkpoint.
  - The checkpoint is the adapter checkpoint named v58-05-self-contrast at training step 15,000.
  - Eight cells: cat × dog, tiger × dog, elephant × penguin and chess board × hourglass, at held-out seeds 9 and 10.
  - The treatments:
    - the adapter as it ships
    - guidance raised to 10 and to 12
    - the experts weighted explicitly 3.0 and 3.0 (the plain product weights them at 7.5)
    - a repaint pass with the adapter off, at strength 0.25 and at 0.40
    - the adapter for the first 20 steps only
    - the best of 8 nearby starting noises
  - 87 tiles, Slurm jobs 59549 to 59558.
- **Bar, written first:** none recorded. The page applies, after the fact, the noise-search finding's bar: median per-cell Laplacian variance 1.10 times the adapter's own render.
- **Result:** Median Laplacian variance over the eight cells, greyscale on a 0-to-1 scale, in units of 1e-4, with the ratio to the adapter's own render in brackets:
  - guidance 12: 138.96 (1.80)
  - guidance 10: 111.96 (1.45)
  - repaint at 0.40: 95.33 (1.24)
  - the joint prompt: 91.18 (1.18)
  - repaint at 0.25: 87.83 (1.14)
  - adapter for 20 steps: 84.34 (1.09)
  - adapter as it ships: 77.08 (1.00)
  - best of 8, two cells only: 65.11 (0.85)
  - plain product: 60.49 (0.79)
  - experts weighted 3.0 and 3.0: 59.74 (0.78)

  By eye:
  - The two repaints and the 20-step handover redraw the adapter's picture with the same layout and subjects.
  - The explicit re-weighting smears both subjects on all eight cells and invents a wolf statue on chess board × hourglass seed 9.
  - Raised guidance adds or swaps an unrequested object on three of the eight cells: a toy giraffe in cat × dog seed 10, a third animal or a second tiger in tiger × dog seed 10, and a rebuilt hourglass.
  - The joint prompt on cat × dog seed 10 draws two dogs.

  No composition count was scored.
- **Verdict:** inconclusive (the page's verdict): two treatments clear a bar nobody agreed to in advance, on a measure known to disagree with the eye.
- **Why it failed:**
  - The edge measure counts contrast as sharpness. Guidance 12 scores 1.80 while drawing an object nobody asked for, and both repaints score above the joint prompt's own 1.18 while not looking sharper (verified, by eye on the sheets).
  - The hypothesis that the quality cost is the correction's in-span part acting as per-step guidance damping was tested only with a constant 3.0 weighting. The measured damping implies a falling schedule (inferred).
- **Kind of failure:**
  - **Test design:** "Nothing was pinned before this harness ran". Eight cells. The best-of-8 column has 2 of 8 cells (verified).
  - **Could not measure it:** the measure ranks contrast as sharpness, no instance count was run over the 87 tiles, and the cat × dog seed 10 reference has no cat (verified).
  - **Done wrong:** for the re-weighting treatment, the measurement implies about 3 early to about 1 late, and a constant 3.0 was tested (inferred).
- **Idea still alive?**
  - Repaint and the 20-step handover: retry worth it. The fixes are to write, before the run, a fidelity bar that a contrast change cannot pass, and to score the 87 tiles with the validated instance count.
  - Explicit re-weighting: retry worth it with the falling schedule.
  - Raised guidance: unclear. The changed scenes are seen by eye on three of eight cells, but no bar was written first. A pre-written bar plus a count would settle it.
- **Rules out:**
  - Do not judge a fidelity fix by Laplacian variance alone. Write the bar first, in a form a contrast change cannot pass.
  - Do not weight the experts at a constant 3.0 and 3.0. It is worse than the plain product by both the measure and the eye.
  - Count how many held-out cat × dog joint prompts drop the cat before judging against that target again.
- **Leaves open:** Composition counts for all 87 tiles. The falling-schedule re-weighting. The six missing best-of-8 cells. A different checkpoint.
- **Evidence:** [the render-time fixes finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-any-render-time-fix-recover-picture-quality.md), [the figure explainer](../../../../../1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/which-fidelity-fix-wins/figure-explainer.md), [the tiger and dog seed 10 sheet](../../../../../1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/which-fidelity-fix-wins/a_tiger__x__a_dog__seed10.png). No W&B id. Slurm 59549 to 59558.

### Does the joint prompt, the adapter's training target, always draw both concepts?

- **Asked:** Does SDXL's joint prompt (one sentence naming both things) reliably draw both named concepts, given that it is the target the adapter is trained to match?
- **Paper:** 2
- **Tried:** 32 joint-prompt renders already on disk for held-out pairs, over eight pairs and six seeds (50 DDIM steps, guidance 7.5), were read by eye. A cell failed if a named concept was absent or an unrequested object was the subject.
- **Bar, written first:** none recorded ("No bar was pre-registered. The read is descriptive and by eye").
- **Result:** 11 failing cells of 32 rendered joint prompts. Chess board and hourglass failed at seeds 9, 10, 11, 12 and 42 (five of five, no hourglass in any). Cat and fox failed at seeds 2, 9, 11 and 12 (four of six, a fox or two foxes and no cat). Dog and dog drew a person beside one dog at seed 1, and one large dog with a malformed small one at seed 2. Elephant and penguin drew the elephant alone at seed 9. In the chess row at seed 10, the plain product holds an hourglass and chess pieces together.
- **Verdict:** null in the brief's mapping: the page records "❌ no: it drops one on eleven of the thirty-two held-out cells rendered", with no bar written first.
- **Why it failed:** Not established. The failures cluster on objects and on repeated concepts, and "nothing here separates a prompt-parsing failure from a rendering one" `stated`.
- **Kind of failure:** wrong data. The joint prompt as a training target can drop a concept, and "if a training cell's target drops a concept, the adapter is being trained to drop it too" `inferred`, because the training pool's targets were never counted. test design: no bar; 32 renders chosen because they were already on disk, not sampled; one phrasing; and the adapter column is rendered for only 1 of the 6 figure rows `verified`.
- **Idea still alive?** unclear. Whether the adapter beats its own target needs the five unrendered adapter cells, and whether the training pool has wrong targets needs a count over the training cells.
- **Rules out:** Do not assume the joint prompt is a correct target. Check the training cells' targets for dropped concepts before training on them.
- **Leaves open:** The adapter's behaviour on these cells. Whether other phrasings fix the target. The rate of wrong targets in the training pool.
- **Evidence:** [the joint-prompt finding](../../../../../1-lora-correction/report/is-the-adapter-just-copying-the-joint-prompt/does-the-joint-prompt-always-draw-both-concepts.md). The figure is only on the cluster, at `/datasets/mmolefe/poe_repair_min/artifacts-archive/1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/where-the-joint-prompt-itself-fails.png`. No W&B id is given.

### Does the adapter return two objects when both prompts name the same concept?

- **Asked:** When both experts are asked for the same animal, so nothing contends on meaning, does the adapter still return two animals where the plain product returns one?
- **Paper:** 2
- **Tried:** Five cells were read by eye: dog and dog at seeds 1, 2 and 9, cat and cat at seed 9, and tiger and tiger at seed 9. Each was rendered three ways: joint prompt, plain product, and the adapter. The adapter was the checkpoint the project names `v58-03-self` (v58 is the training cell list), at step 50,000, with the correction at strength 1 on all fifty steps.
- **Bar, written first:** none recorded. A cell counts as composed when two whole animals of the named kind are in frame.
- **Result:** The plain product drew one animal in 5 of 5 cells. The adapter drew two animals in 4 of 5. The exception was dog and dog at seed 1, where the second animal is dark and horse-like rather than clearly a dog. The joint prompt drew two animals in 3 of 5, with a person beside one dog at seed 1 and one dog with a malformed second at seed 2.
- **Verdict:** support (the page records "✅ support: two animals on four of five cells, where the plain product returns one on all five"), with no bar written first.
- **Why it worked:** With no competing concepts, the plain product still collapses to one animal and the adapter does not. So what the adapter supplies "is therefore not a way of telling two different concepts apart" `stated` (argued from the by-eye counts).
- **Kind of failure:** none, the page records support, but no bar was written first, so test design applies as well. The test design weakens it: five cells, three pairs at one or three seeds each, one checkpoint, no bar `verified`. could not measure it: the read counts animals rather than species, and at seed 1 the second animal is the wrong kind `verified`. The tiger row's adapter render is softer than its joint prompt, and image quality was not measured.
- **Idea still alive?** alive, but only as a by-eye observation on five cells from one adapter. It does not license a rate. The page names the next step: the same five cells through two more checkpoints, plus a species check.
- **Rules out:** Do not describe the adapter's effect as separating two different concepts, since it also doubles a single concept. Do not count the second animal as correct without a species check.
- **Leaves open:** Whether this holds past three pairs, across checkpoints, and for the species of the second animal.
- **Evidence:** [the same-concept finding](../../../../../1-lora-correction/report/is-the-adapter-just-copying-the-joint-prompt/does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md) and [the figure's tile list](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/the-same-concept-twice.json). The figure itself failed the subject check and is not shown. No W&B id is given.

### Does the adapter carry across kinds of pair, and how does it look beside published models?

- **Asked:** Does the adapter repair object pairs and object-in-a-place pairs as well as animal pairs, and how does SDXL's plain product compare with the joint prompt on published models?
- **Paper:** 2
- **Tried:** Two grids were made. The three-kinds grid has six rows at seed 9 (joint prompt, plain product, adapter). The published-models grid has one row, cat and dog at seed 42, across SD 1.4, SD 2.1, SD 3.5, SDXL, the plain product and CO3, a second composition method that starts from its own noise. Both were read by eye. The adapter was at step 50,000.
- **Bar, written first:** none recorded.
- **Result:** The adapter column is filled on only 2 of 6 rows of the three-kinds grid, both object-in-a-place pairs. For lighthouse and stormy sea, all three columns show both. For tent and snowy mountain, the joint prompt draws an orange tent below a mountain, the plain product a mountain shaped like a tent (fused), and the adapter a dome tent and a white mound below a mountain. The cat and dog, tiger and dog, chess board and hourglass, and picnic basket and watermelon rows have no adapter render. In the published-models grid the adapter tile is empty. By eye, the joint prompt draws a lone cat on SD 1.4, two dogs on SD 2.1, and a cat with a dog on SD 3.5 and SDXL, while SDXL's plain product draws one fluffy animal and CO3 draws two cats.
- **Verdict:** not run for the adapter on 4 of 6 rows and on the published-models row. none recorded for the rest: the page shows single-seed renders with no rate.
- **Why it failed:** The adapter tiles were never rendered `verified` (the tile list has no adapter path). Sources disagree: the grids card lists three pairs for the published-models grid (cat and dog, butterfly and flower meadow, camel and forest), but the picture holds cat and dog only. The grid script lists all three. It was committed at 14:50 on 2026-09-24 and the picture is dated 14:40, so the picture likely predates the three-row version `inferred`.
- **Kind of failure:** test design. The comparison was not completed, and the name "the fix is not about animals" claims more than the filled rows show `verified`.
- **Idea still alive?** unclear. Rendering the adapter tiles for the four empty rows and for cat and dog at seed 42 would tell.
- **Rules out:** Do not cite this grid as showing the adapter carries across kinds of pair. Only the two object-in-a-place rows have adapter renders, and they are one seed each.
- **Leaves open:** The adapter on the animal and object rows. A three-pair published-models grid.
- **Evidence:** [the main-study grids finding](../../../../../1-lora-correction/report/measuring-composition/what-the-main-study-grids-show.md), [the three-kinds grid](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/the-fix-is-not-about-animals.png), [the published-models grid](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/published-models-then-ours.png) and [the grids card](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/README.md). No W&B id is given.

### What does the adapter do to a joint prompt it was not trained to repair?

- **Asked:** If a joint prompt (which already composes) is run through the adapter, does the picture change?
- **Paper:** 2
- **Tried:** Four rows (cat and dog, tiger and dog, chess board and hourglass, picnic basket and watermelon) were each rendered from one starting noise by the base model and through the adapter, then read by eye.
- **Bar, written first:** none recorded.
- **Result:** The picture changes on all four rows. Cat and dog: the same two animals plus extra small animal shapes and an object near their feet. Tiger and dog: a tiger, a tiger-coloured animal with a dog's face, a dog and a small extra dog. Chess board and hourglass: the base shows no hourglass, while the adapter shows chess pieces with an hourglass among them. Picnic basket and watermelon: a different basket, with flowers and more items added.
- **Verdict:** none recorded: the page shows four rows read by eye, with no rate.
- **Why it failed:** Not established. The sources disagree on whether a change is damage. The grids card calls any difference here "damage outside what it was trained to repair". On the chess row, the adapter's picture holds the hourglass the base render lacks, and the page says whether that is damage or a gain is a reading of the picture `inferred`.
- **Kind of failure:** test design: four cells chosen for the figure, no bar, "so it gives no rate of damage" `verified`.
- **Idea still alive?** unclear. A rate over sampled joint-prompt cells, with damage defined before the read, would tell.
- **Rules out:** Do not quote this grid as a rate of damage. Define damage versus gain before reading such a grid.
- **Leaves open:** How often the adapter harms a prompt that already composes.
- **Evidence:** [the main-study grids finding](../../../../../1-lora-correction/report/measuring-composition/what-the-main-study-grids-show.md) and [the grids card](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-grids/what-the-main-study-grids-show/README.md). The figure failed the subject check and is not shown. No W&B id is given.

### Reward fine-tuning the adapter with a KL anchor (refl-kl K0 to K3)

- **Asked:** Does a reward fine-tune anchored to its starting adapter (K0 to K3 are four such runs) improve the adapter's two-animal renders?
- **Paper:** 2
- **Tried:** Four reward fine-tunes, `refl-kl-K0` to `K3` (W&B `t4wiwy5p`, `u73uvqyg`, `yj7l40h5`, `ybjd4trt`), started from v58-05 (a distillation-trained adapter) at step 15,000. They trained on cat and dog and on elephant and penguin at seeds 1 to 8, and each reached 2,000 of 2,000 steps. K2 was launched 2026-09-25 18:08 and K3 2026-09-26 08:26, per the run queue file. A fifth run, K4, is recorded only as "starting".
- **Bar, written first:** none recorded in any repo file checked.
- **Result:** No score, rate or verdict for these runs is in any repo file checked. What is recorded: the protocol lists them as reward fine-tunes that "trained directly on cat and dog and on elephant and penguin, seeds 1 to 8", so they are not candidates. The review route records that Figure 5's elephant and penguin "ours" tile is the plain v58-05 checkpoint, not a fine-tune: mean absolute pixel difference 0.00 on a 0 to 255 scale, against 6.32 to 14.10 for the fine-tunes (`verified`). The ten-seeds page records that its five picked pairs use "the reward fine tuned adapter K0 at 2,000 steps", one hand-picked seed each, rendered 2026-09-26, before the protocol fixed one adapter on 2026-11-10.
- **Verdict:** none recorded: the repo files say only that the runs finished and that they are excluded as candidates because they trained on the showcase pairs. Files checked: the protocol, the review route (`plans/route-answer-the-iclr-reviews.md`, lines 65 and 449), the step-2 walk of the review answers, the ten-seeds page, the idea map, the opening prompt, `scripts/local/run_queue_state.json` and `artifacts/notes/progress/prompts/session-2-artifacts.md`.
- **Why it failed:** not recorded.
- **Kind of failure:** Test design (`verified` in the protocol: trained on the very pairs used to judge the adapter, so no held-out reading is possible; no bar recorded).
- **Idea still alive?** Unclear. A reward run on held-out pairs with a bar written first (run R4 in the idea map) would tell whether the anchored reward helps.
- **Rules out:** Do not cite K0 to K3 renders as held-out evidence: they trained on cat and dog and elephant and penguin. Do not attribute the Figure 5 elephant and penguin tile to a reward fine-tune.
- **Leaves open:** Every question the runs were launched to answer: whether the anchor binds, and whether the reward improves fidelity or composition.
- **Evidence:** [the evaluation protocol](../../../../../1-lora-correction/report/paper-review/protocol.md), [the step-2 walk of the review answers](../../../../../1-lora-correction/artifacts/drips/answering-the-reviewers/answer-iclr-reviews/EXECUTE_WALK-step-2.md), [the ten-seeds page](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/does-each-adapter-hold-across-ten-seeds.md) and [the one-training-recipe idea map](../../../../../3-understanding-compositionality/artifacts/ideas/one-training-recipe/IDEA_MAP.md). No report finding or review file exists for these runs. W&B `t4wiwy5p`, `u73uvqyg`, `yj7l40h5`, `ybjd4trt`.

### The paper's main grid: four methods on four hand-picked pairs

- **Asked:** On the four pairs of the paper's opening figure, which methods draw both named things?
- **Paper:** 2
- **Tried:** One grid of 4 pairs by 4 columns (joint prompt, plain product, CO3, the adapter) at one seed per pair: cat and dog seed 9, elephant and penguin seed 10, tent and snowy mountain seed 9, lighthouse and stormy sea seed 9. The adapter column uses three different adapters, and CO3 starts from its own noise rather than the shared cached noise. Three further CO3 renders (bison and horse, tiger and parrot, wolf and raven) were "chosen by eye".
- **Bar, written first:** none recorded.
- **Result:** Eye reads only. The plain product draws one cat-dog animal and an elephant inside a penguin's outline, and turns the mountain tent-like. CO3 and the adapter draw both things on all four rows, with the adapter adding small stray objects at the dog's feet. All four columns draw a lighthouse in a stormy sea. The sources disagree on the lighthouse row: the card says "the plain product fuses or drops one" across the four cells (`stated`), while the plain-product tile shows a lighthouse and a stormy sea (`verified`), and the Langevin sheet's card reports both things in every column for this pair at seed 10.
- **Verdict:** none recorded: the page says no verdict or rate belongs to the grid because its four cells were picked by hand.
- **Why it failed:** nothing was tested. The page shows hand-picked examples without a rate.
- **Kind of failure:** Test design (`stated`: four hand-picked cells, one seed each, three different adapters in one column, CO3 on different starting noise).
- **Idea still alive?** Unclear. The grid cannot tell. A rate at one checkpoint on fixed pairs and seeds would, and the benchmark and held-out pages are that.
- **Rules out:** Do not quote this grid as evidence of how often any method composes. Do not compare CO3's tiles here pixel for pixel with the other columns.
- **Leaves open:** Which reading of the lighthouse row the card should carry. The settings of the other CO3 renders.
- **Evidence:** [the four-pairs page](../../../../../1-lora-correction/report/against-other-methods/how-do-four-methods-compare-on-four-hand-picked-pairs.md) and [the grid's card](../../../../../1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-four-methods-compare-on-four-pairs/README.md). The grid itself is not shown because of the pair ledger. The CO3 tiles came from Slurm job 59714, rendered 2026-09-25.

### PoE of cat and dog on PixArt-alpha, a model with no CLIP

- **Asked:** Does summed-guidance product-of-experts of "a cat" and "a dog" still fuse into one animal on a model with no CLIP in any role?
- **Paper:** 2
- **Tried:**
  - PixArt-alpha (text encoder Flan-T5-XXL) at its own defaults: 20 steps, guidance 4.5, DPM-Solver++ order 2, 1024², fp16.
  - Seeds 1 to 8, with four columns: "a cat" alone, "a dog" alone, the product and the joint prompt.
  - Before any product render, "a cat" at seed 1 through the released pipeline and through the composition script differed by 0 of 255 grey levels and were byte-identical.
- **Bar, written first:**
  - Support: 2 or fewer of 8 product renders hold two clear animals, while the joint column holds two on 6 or more of 8.
  - Null: 6 or more of 8 product renders compose.
  - Inconclusive: between 3 and 5 of 8, or the joint column below 6 of 8.
- **Result:**
  - Product: 0 of 8 seeds hold two clear animals. Six are one hybrid animal and two (seeds 2 and 3) are one clean cat.
  - Joint prompt: 7 of 8 draw a clear cat beside a clear dog. Seed 2 draws two cats and counts as neither.
  - Every call was made by eye from the full-resolution tile, because the instance-count scorer "counts animals without naming them and reads a hybrid and two cats identically".
- **Verdict:** support
- **Why it worked:**
  - The same summed arithmetic fuses on a model that has never seen a CLIP embedding, so CLIP is not a necessary cause (verified as behaviour, from one reader's eye calls).
  - The joint prompt at guidance 4.5 draws the pair, which rules out a weak-guidance artefact (verified).
- **Kind of failure:** none, it passed its bar. Two things limit how far the pass reaches. Could not measure it: one reader made every call by eye, because the instance-count detector "counts animals without naming them and reads a hybrid and two cats identically" (verified caveat). Test design: the comparison is mixed on five axes (text encoder, denoiser, sampler, step count, guidance) over 8 seeds, so only the fusion direction licenses a claim (verified caveat).
- **Idea still alive?** alive, but the pass reaches only as far as "CLIP is not necessary for the fusion". It does not name the cause, and it does not show that CLIP is not sufficient or plays no part in SDXL.
- **Rules out:** Do not attribute the cat-dog fusion to CLIP's pooled caption representation as a necessary cause.
- **Leaves open:** Which of the five differences carries the failure. A text encoder that is neither CLIP nor T5 (Sana with Gemma-2, not planned). Whether the two single-cat tiles are the product dropping a prompt or out-voting it.
- **Evidence:** [the CLIP-free finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/14-does-a-clip-free-model-hybridise.md), [the PixArt sheet](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/pixart-cat-dog-sheet.png), [the per-tile calls](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-a-clip-free-model-hybridise/calls.json). No W&B id. Run on mscluster84.

### PoE of cat and dog on SDXL sampled at PixArt-alpha's settings

- **Asked:** Does SDXL's product of "a cat" and "a dog" still fuse when the sampler, step count and guidance are switched to PixArt-alpha's defaults?
- **Paper:** 2
- **Tried:** `sdxl_matched_multi_poe.py`, run on held-out seeds 9 to 16 from cached starting latents, with four columns ("a cat", "a dog", the product, the joint prompt), in two runs:
  - A control at DDIM, 50 steps, guidance 7.5.
  - A matched run at DPM-Solver++ order 2, 20 steps, guidance 4.5.
- **Bar, written first:**
  - Support: 2 or fewer of 8 product renders hold two clear animals, while the joint column holds two separate animals on 6 or more of 8.
  - Null: 6 or more of 8 compose.
  - Inconclusive: between 3 and 5 of 8.
  - Before the matched run, the joint column's reading was set in `DRIFT_LOG.md` as two separate animals of any species, because the joint prompt draws two of the same species on 3 of 8 seeds. The product bar was never changed.
- **Result:**
  - Control: product 0 of 8, joint 8 of 8 two separate animals. This matches the corrector finding's counts, which came from a different script.
  - Matched run: product 0 of 8. Seven are one animal with both sets of features. Seed 16 holds two bodies, neither clearly cat nor dog, and is counted in neither category.
  - Joint prompt at the matched settings: 8 of 8 two separate animals. Read for species, 5 of 8 show a clear cat beside a clear dog: seeds 14 and 15 draw two cats and seed 16 draws two dogs. The control's figures are 8 of 8 and 5 of 8.
- **Verdict:** support
- **Why it worked:** Moving the sampler, step count and guidance together left the fusion on every seed (verified, one reader's calls from the sheets).
- **Kind of failure:** none, it passed its bar. Two things limit how far the pass reaches. Test design: "Three settings moved together", and the joint prompt draws two animals of the same species on 3 of 8 seeds, so the reference shows that two subjects can be rendered, not that this model separates these two species (verified caveat). Could not measure it: "The calls are one reader's" (verified caveat).
- **Idea still alive?** alive, but the pass reaches only as far as "the sampler, step count and guidance, as a group, do not carry the fusion". Which of the three does anything is untested, and the denoiser architecture is the difference left unexamined.
- **Rules out:** Do not attribute the fusion to SDXL's sampler, step count or guidance as a group. Do not use these seeds' joint prompt as a cat-beside-dog reference without reading species.
- **Leaves open:** The denoiser architecture (a second CLIP-free model on a different architecture). Which of the three settings does anything. Why the joint prompt draws one species twice on these seeds.
- **Evidence:** [the PixArt-settings finding](../../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-sdxl-compose-at-pixarts-settings.md), [the review that set the bar](../../../../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/review/16-does-sdxl-compose-at-pixarts-settings.md), [the matched sheet](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-sdxl-compose-at-pixarts-settings/sdxl-matched-sheet.png), [the per-tile calls](../../../../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/does-sdxl-compose-at-pixarts-settings/calls.json), [the summary grid of all three setups](../../../../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/what-the-fusion-is-not-caused-by/summary-grid.png). No W&B id. Run on mscluster84.

### Choosing one adapter for the review by a counting rule (protocol rule 1)

- **Asked:** Which single adapter checkpoint draws both showcase pairs clearly on the most seeds, so the review response can use one adapter for all its numbers and images?
- **Paper:** 2
- **Tried:** Nine candidate checkpoints each drew cat and dog, and elephant and penguin, at seeds 9 to 16. Each of the 144 pictures was called clear or not clear by eye on 2026-11-10. The rule: a seed counts only when both of a candidate's pictures at that seed are clear. Ties go to the most clear pictures of 16, then to the most training steps.
- **Bar, written first:** The rule is the protocol's rule 1 (stated). The page does not record whether the rule was written before the calls were made.
- **Result:**
  - `v58-03-self` at step 30,000 is clear on both pairs at 5 of 8 seeds (10, 11, 12, 13, 16), with 12 of 16 pictures clear.
  - Next are `v58-03-self` at 27,500 (4 seeds, 11 of 16) and `v58-02-contrast` at 27,500 (4 seeds, 10 of 16).
  - The other six reach 3 seeds.
  - The shared seed is 10. A read-only rerank gives the same choice (verified).
- **Verdict:** none recorded: the page records a choice made by a rule, not a pass or fail.
- **Why it worked:** The rule was applied as written, and a dry run of the ranking script reproduces it (verified).
- **Kind of failure:**
  - **test design** (verified): "Every call is by eye, 16 per candidate", and the winner leads the next two by one seed.
  - **could not measure it** (inferred, from the page's own doubt): at the stored 256 pixels, the shared seed's cat and dog tile is uncertain. "The smaller animal's face could be read as a cat or a small dog, and a form on the large dog's head cannot be named." The calls file marks it clear.
- **Idea still alive?** unclear. The choice stands as recorded. A one-seed margin from by-eye calls cannot tell a better checkpoint from a luckier one; a full-size re-call of seed 10 and counts on more seeds would. The page says this ranking "gives no success rate", and the protocol keeps these two pairs out of every held-out rate.
- **Rules out:** Do not quote a success rate from this selection or from the two showcase pairs. Repoint the calls file's `picture` column before rerunning the ranking script in full: a full rerun "would therefore change the choice file and then stop with an error" (inferred, not run).
- **Leaves open:** Whether the seed 10 cat and dog call holds at full size. The checkpoint-sweep page (2026-09-24) notes that its sheets hold only four of the nine candidates, so that sweep "cannot be the whole record of how the review's candidates were chosen" (verified).
- **Evidence:** the [showcase finding](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/which-adapter-composes-the-showcase-pairs.md), the [review protocol](../../../../../1-lora-correction/report/paper-review/protocol.md), the [calls file](../../../../../1-lora-correction/report/paper-review/showcase-calls.csv), the [recorded adapter choice](../../../../../artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml), the [chosen adapter's cat and dog at seed 12](../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/which-adapter-composes-the-showcase-pairs/tiles/v58-03-self_s030000__a_cat__x__a_dog__seed12.png).

### Checking four published scorers on one mug and wine glass pair

- **Asked:** Do four published scorers (CLIP score, ImageReward, BLIP-VQA, GenEval two-object) rank the joint-prompt image above the plain-product image for "a mug and a wine glass", before the paper's review tables use them?
- **Paper:** 2
- **Tried:** Each scorer was run on two SDXL images at seed 11, one from the joint prompt and one from the plain product.
- **Bar, written first:** none recorded.
- **Result:** CLIP score (100 times the image-prompt cosine, 0 to 100) was 26.7271 for the joint image and 23.5954 for the product. ImageReward (predicted human preference, higher is preferred) was 0.5668 and -2.2534. BLIP-VQA (product of the yes-probabilities per noun phrase, 0 to 1) was 0.3641 and 0.0044. GenEval two-object (1 if a box of each named class is found) was 1 and 0, the product's 0 reading "expected cup>=1, found 0". The joint image's left object is a tall glass with a mug's handle on a goblet's stem and foot. GenEval puts a `cup` box (confidence 0.945) and a `wine glass` box (0.889) on almost the same pixels of it, and a second `wine glass` box (0.982) on the right-hand glass.
- **Verdict:** none recorded: the page records that all four scorers rank the joint image first, with no error rate.
- **Why it failed:** Not a failure as such. The page warns that GenEval's pass "does not mean a clean mug": it passes a fused mug-goblet `verified`. Sources disagree: the step 9 session note says the joint image "shows both objects", while the picture shows a mug fused onto a goblet.
- **Kind of failure:** could not measure it (`verified`: GenEval passes a fused object because a `cup` box exists). test design: one pair of images, so there is no error rate for any scorer, and the scales differ between scorers `verified`.
- **Idea still alive?** unclear. This probe only shows each scorer runs and orders two images the same way. A labelled set would say which scorer can tell a fused object from two.
- **Rules out:** Do not treat a GenEval two-object pass as two separate objects without looking. Do not compare gaps across scorers, since their scales differ.
- **Leaves open:** Each scorer's error rate. Whether a mug fused to a goblet should count as holding a mug.
- **Evidence:** [the four-scorer finding](../../../../../1-lora-correction/report/measuring-composition/what-each-scorer-says-about-two-objects.md), [the four-scorer figure](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/four-scorers-on-mug-and-wine-glass.png), [the joint image](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-scorers/what-each-scorer-says-about-two-objects/mug-and-wine-glass-seed11-joint.png) and [the step 9 session note](../../../../../artifacts/notes/session-merges/2026-11-11-route-step-9-retro.md) (step 9). No W&B id is given.

### One adapter against four methods on four pairs at one seed

- **Asked:** Does the single review adapter separate four pairs it never trained on, where plain product, CO3 and product plus Langevin may not?
- **Paper:** 2
- **Tried:** `v58-03-self` at step 30,000, on for all 50 drawing steps. It was set beside the joint prompt, plain product, CO3 at its published default, and product plus Langevin (step size c = 3, k = 20 corrector steps at all 50 noise levels). Four pairs (cat and dog, elephant and penguin, tent and snowy mountain, lighthouse and stormy sea) at seed 10: 20 pictures, read by eye at about 296 pixels per tile.
- **Bar, written first:** none recorded.
- **Result:** By eye (verified):
  - The adapter and CO3 show both things on all four rows. The adapter adds a small extra elephant on elephant and penguin, and its cat and dog are touching.
  - The plain product fails on three rows: one fused face, one fused body, a tent-shaped mountain. It draws both the lighthouse and the sea.
  - Product plus Langevin draws one dog, an elephant with penguin markings, a snowy peak above a dark dome, and both on lighthouse.
  - The joint prompt draws two dogs and no cat on the cat and dog row.
- **Verdict:** none recorded: "One seed per pair gives no rate."
- **Why it worked:** not recorded. The sheet is a figure review.
- **Kind of failure:** **test design** (verified): one seed per pair, by eye. The adapter and seed were chosen on the cat and dog and elephant and penguin rows (showcase entry above), so those two rows are not independent of the choice.
- **Idea still alive?** unclear. One chosen seed cannot give a rate; the held-out rate lives in the [comparison over 53 held-out pairs](../../../../../1-lora-correction/report/against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md).
- **Rules out:** Do not write that the adapter composes all four pairs "where the plain product does not": the plain product draws both on the lighthouse row at seed 10. Do not use the joint prompt as the reference on cat and dog at seed 10.
- **Leaves open:** **Where the sources disagree** (unresolved):
  - The paper's Figure 5 prints cat and lion and elephant and turtle (commit `1a891ba7`), while this sheet and the layout sheet name cat and dog and elephant and penguin.
  - The sheet has five method columns, the paper caption six (with ToMe), and the figure register four.
  - The card says Langevin "keeps one of them on the first three" rows, while the tent row shows a dark dome that may be the tent.
- **Evidence:** the [four-pairs finding](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/does-one-adapter-compose-four-pairs.md) and the [seed 10 sheet](../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-one-adapter-compose-four-pairs/figure5-seed10-five-columns.png).

### Does keeping the learned adapter on only for the early steps change the failures?

- **Asked:** If the learned adapter acts only on steps 0 to 10 or 0 to 25 of 50, how do the composition and failure types compare with off and with all 50 steps?
- **Paper:** 2
- **Tried:** The v58-03-self adapter at training step 30,000 (the revision's protocol checkpoint) was run on the 53 held-out pairs at seeds 9 to 16, 424 images per setting, 1,696 images in all. Scores: BLIP-VQA per held-out group, the GroundingDINO share of the 160 animal-pair images with two or more animals, CLIP score and ImageReward. Each image also got one failure label by eye and by a fixed-prompt judge.
- **Bar, written first:** a reading written in the plan before any score existed. If steps 0 to 25 and all 50 overlap on BLIP-VQA in every group, 25 steps give the same result. If steps 0 to 10 sit "below both in a group", ten steps are too short there.
- **Result:** Two or more animals found, of 160 animal-pair images: off 6 (3.75%), steps 0 to 10 75 (46.88%), steps 0 to 25 147 (91.87%), all 50 153 (95.63%) (`verified`). BLIP-VQA (0 to 1) in the easy group: 0.3783, 0.4981, 0.6082, 0.5755. ImageReward over 424 images: -0.1063, 0.4383, 0.5451, 0.4908. Steps 0 to 25 and all 50 have overlapping 95% intervals in all three groups. Hybrids fall from 171 to 104 by eye and 236 to 147 by judge. Intrusions (an unrequested extra) rise from 12 to 157 by eye and 16 to 166 by judge.
- **Verdict:** none recorded. The first branch holds; the second depends on what "below" means. Under a mean reading, ten steps are too short in the hard and easy groups. Under an interval reading, no group is below. The choice is the owner's and the plan's task is unticked.
- **Why it failed:** No verdict, because the pre-written reading did not define "below" (verified from the page).
- **Kind of failure:** test design (verified): an ambiguous pre-written rule, and the hard group has only 3 pairs, so its intervals run from about 0.01 to 0.85. Also could not measure it (verified): the two labellers disagree (off setting, 171 hybrid and 98 one-concept by eye against 236 and 41 by judge), and "Neither set has been checked by a person". The page does not say whether the detector can name the animals; on cow and horse the card reads none of the four as a clear horse (stated).
- **Idea still alive?** unclear. Choosing how "below both" is read and having a person check the 40-image sample would settle it.
- **Rules out:** Do not write a pre-registered reading with an undefined comparison ("below"); say whether means or intervals are compared. Do not report failure-type counts from one labeller without a person checking a sample. Do not read this adapter experiment as contradicting the true correction's early window, since the two ask different things.
- **Leaves open:** which labeller the paper's figure uses. Whether the adapter of record (step 50,000 in the early-training note) behaves the same as step 30,000.
- **Sources disagree:** The early-training note names step 50,000 as the adapter of record, while these renders use 30,000 per the protocol. The grid PNG shows six rows by five columns, while the PDF and its spec show seven by four. The card says two grey cells on the seed 10 sheet; the sheet shows six. All recorded unresolved.
- **Evidence:** [the early-steps card](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/does-restricting-the-adapter-to-early-steps-change-the-failures/README.md), [the renders by setting](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/does-restricting-the-adapter-to-early-steps-change-the-failures/early_steps_grid.png), [the failure counts](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/when-it-acts/does-restricting-the-adapter-to-early-steps-change-the-failures/failure_counts.png), [the plan with the reading](../../../../../1-lora-correction/plans/present-revision-to-supervisor.md). Scoring jobs 85541 (BLIP-VQA), 85539 (CLIP score and ImageReward) and 85540 (two-animal count). No W&B run id given. (Figure 10)

### One pooled adapter on eight pairs it never trained on (the ICLR "58 of 64")

- **Asked:** Does the held-out compose rate of one pooled adapter beat the level plain product-of-experts reaches with no fix?
- **Paper:** 2
- **Tried:** One rank 8 LoRA was trained on all eleven training pairs at once (run `phase1_r8_100k`, 88 pair-and-seed runs, 100000 steps). Each checkpoint was scored from the renders alone by the instance-count scorer: two or more boxes labelled "animal", each spanning at least 25% of the image's long side. At step 60000 that was 64 pictures of 8 unseen pairs (8 seeds each) and 88 pictures of training pairs (job 86591, commit `5cb9329c`).
- **Bar, written first:** A question and a stop rule, with no numeric bar. The run is a no-transfer null if the held-out rate is no better than plain product-of-experts and the direction measure also comes out wrong. The direction measure was wired after the run, so only the first half was read.
- **Result:** At step 60000, 58 of 64 pictures (0.906) of eight unseen pairs held two animals by the detector's count, against 80 of 88 (0.909) on the eleven training pairs. Per unseen pair (8 pictures each): leopard and jaguar 1.0, goose and swan 1.0, cow and buffalo 1.0, elephant and penguin (control) 1.0, eagle and hawk 0.875, seal and walrus 0.875, frog and toad 0.75, cat and dog 0.75. From step 20000 to step 100000 the held-out rate stays between 0.83 and 0.91. At steps 70000 to 100000 it reads 0.88, 0.89, 0.83 and 0.91 (64 pictures each), with training pairs at 0.92 to 0.94 (88 each); one seed per pair is 0.016. Plain product-of-experts is the dotted line at 0.03, from a separate sweep over 32 cells.
- **Verdict:** support (the page records "✅ support", and review 03 records "Answered yes").
- **Why it worked:** The held-out rate sits level with the training-pair rate and far above the plain-product line `verified`. The detector caveat, in the page's words: "The scorer counts animals and cannot name them. Cat and dog scores 0.75, and the review's reading by eye finds a clear cat beside a clear dog on 3 of its 8 seeds" `stated`. Review 03 adds by eye: seed 13 shows two cats, seeds 11 and 15 are layered sketches, and seeds 10 and 14 hold one animal. The control is inside the count: "Elephant and penguin composes with no adapter, and its 8 of 8 pictures are inside the count" `verified`. The note on hard and easy pairs reads the seal and walrus row as two seals, which the count cannot detect `stated`.
- **Kind of failure:** none, the page records support, but no bar was written first, so test design applies as well. could not measure it: the detector counts bodies, not which animal each is (cat and dog 0.75 by count against 3 of 8 clean by eye). test design: the control pair's 8 of 8 is inside the 58. The gap to plain product-of-experts is a mixed comparison, because the 0.03 comes from a different sweep over 32 cells, so the pairs differ as well as the adapter. There is no numeric bar, the direction measure was not wired, and it is one run, not a rate. An earlier scoring fault is fixed: two older rates of 0.961 (123 of 128) and 0.742 (95 of 128) counted the three-panel comparison strips as pictures, fixed in commit `0811f029` (known failure poe-score-003) `verified`.
- **Idea still alive?** alive, but only this far: one pooled rank 8 adapter's held-out pictures hold two bodies by detector count at the training-pair rate. It does not license "both named animals present", and it is not a transfer rate over pairs. Review 03 says the rate "comes from the fifteen runs, and the sentence must not read as though this run supplied it". Review 03 gave the go-ahead for those fifteen runs on 2026-11-13.
- **Rules out:** Do not quote 0.96 or 0.961. Those counted comparison strips, and the true rate is 0.906. Do not count the control pair inside a transfer rate without saying so. Do not report a detector count of two as "both animals" without a by-eye or species check. Do not set the 0.03 plain-product line against this as a clean comparison.
- **Leaves open:** Whether the direction axis agrees, and so whether a pair at the no-fix level got no correction or a wrong-pointing one. Review 03 also leaves three checks unticked: the realised split, the scorer's read scope, and the environment. Sources disagree in four places. The held-out rate is 0.961 in the hard-and-easy note against 0.906 in the review. The review says the draft and figure still carry 0.96, while the figure data and DRAFT_MAP carry 0.906 (the figure was rebuilt by commit `2a3da6f9` at 16:17 on 2026-11-13, after the review's last rewrite at 16:11). The note names frog and toad, eagle and hawk, and seal and walrus as hardest at 0.9375, while the same file has cat and dog lower at 0.875. The note counts the walrus's missing tusks against the adapter, while the joint-prompt picture in that row shows no tusks either.
- **Evidence:** [the pooled transfer finding](../../../../../1-lora-correction/report/does-the-fix-reach-unseen-pairs/does-one-pooled-adapter-compose-on-pairs-it-never-trained-on.md), [the review that records the verdict](../../../../../1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/review/03-does-one-pooled-fix-transfer-at-all.md), [the plan with the stop rule](../../../../../1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/plans/hypothesis/03-does-one-pooled-fix-transfer-at-all.md), [the compose-rate curve](../../../../../writing/iclr/figures/compose-rate-as-the-lora-trains.png), [the scored rates](../../../../../1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/compose_rate.json) and [the note on hard and easy pairs](../../../../../1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/hard-vs-easy-transfer/QUERY.md). Jobs `86591` (current scoring), `86565` and `86587` (strip-inclusive rescore), `86592` (filter-call grids). No W&B id is given.

### Rerunning Rectify Then Diffuse (RTD) at its printed settings

- **Asked:** Does RTD, run from its authors' own code at their printed settings, reproduce the ImageReward its paper prints on T2I-CompBench Complex and RareBench Concat?
- **Paper:** 2
- **Tried:** RTD and plain SDXL rendered 988 of 1,000 Complex prompts and 33 of 40 Concat prompts (the rest are kept out by the subject guard) at 3 seeds each (0, 1 and 2), on RTX 3090s with the authors' pipeline at commit `241eb37`. Each picture was scored by ImageReward, a preference-trained score where higher is better and values can be negative. That is 2,964 RTD scores on Complex and 99 pictures per method on Concat.
- **Bar, written first:** pinned in the RTD design before the first run. Reproduced if |ours − published| ≤ 1.96 × √(SE_ours² + SE_published²), with the published standard error taken as equal to ours, so the pass band is ours ± 1.96 × √2 × SE_ours. Plain SDXL must pass the same test against its printed score.
- **Result:** RTD on Complex scored 0.2423, pass band 0.1751 to 0.3095, against 0.4661 printed: not reproduced. RTD on Concat scored 0.4166, band −0.0268 to 0.8600, against 0.4464 printed: reproduced. Plain SDXL on Complex scored 0.2637, band 0.2019 to 0.3256, against 0.3083 printed: reproduced. Plain SDXL on Concat scored 0.3035, band −0.1269 to 0.7338, against 0.1018 printed: reproduced. Matched picture by picture, RTD minus plain SDXL is −0.0214 on Complex (standard error 0.0153), where the printed scores differ by +0.1578, and +0.1131 on Concat. The sources disagree on the Concat standard error: the page and the review give 0.0983, and a recompute from the score files gives 0.0982. Five checks pass, including 0 of 2,964 RTD scores equal to plain SDXL's and a zero-size update matching plain SDXL with difference 0.0.
- **Verdict:** Null on Complex (not reproduced). Support on Concat (reproduced, a weak test because the band runs from below zero to 0.8600). Plain SDXL support on both.
- **Why it failed:** `inferred`: the review lists three causes still open, and nothing run here separates them: the authors' NVIDIA H200 against our RTX 3090s, an unstated ImageReward package or preprocessing, and the printed number itself. `verified`: our prompts and scorer return plain SDXL's printed scores, so the setup alone does not explain the miss.
- **Kind of failure:** Complex: test design (`verified`: three causes are open, H200 against RTX 3090, an unstated ImageReward package or preprocessing, and the printed number, and nothing run here separates them). Concat: none, it passed its bar, limited by test design (`verified`: 33 prompts and a pass band from −0.0268 to 0.8600, which the page calls a weak test).
- **Idea still alive?** Complex: unclear. Running on the authors' card, or with their exact ImageReward package, would separate the three causes, and the owner keeps the printed 0.466 as Paper 2's bar (board question 26). Concat: alive, but the band is so wide that almost any printed score near RTD's would pass, so the pass does not confirm RTD.
- **Rules out:** Do not treat the RareBench Concat pass as confirmation of RTD. Do not set Paper 2's Complex bar from the 0.2423 rerun: the printed 0.466 stays the bar. Do not claim RTD beats plain SDXL on matched prompts (−0.0214 on Complex).
- **Leaves open:** Which of the three causes explains the Complex miss. RTD on the prompts the guard keeps out.
- **Evidence:** [the rival reproduction finding](../../../../../1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md), [the RTD review](../../../../../artifacts/notes/progress/reproductions/RTD/review.md), [the RTD design that set the bar](../../../../../artifacts/notes/progress/reproductions/RTD/design.md), [the picture of why the Complex score misses](../../../../../artifacts/notes/progress/reproductions/RTD/why-the-complex-score-misses.png) and [plain SDXL beside RTD on eight Complex prompts](../../../../../artifacts/notes/progress/reproductions/RTD/plain-sdxl-against-rtd-eight-random-complex-prompts-seed0.png). W&B `5i9dt7by` (RTD Complex), `tc75fx0i` (RTD Concat), `8802binw` (SDXL Complex), `1gh0vhil` (SDXL Concat); Slurm 93064 and 93068. (D2)

### Feynman-Kac Correctors (FKC) and its plain SDXL baseline on GenEval

- **Asked:** Does FKC reproduce its printed GenEval two-object score, and does plain SDXL reproduce its own printed score on the same prompts?
- **Paper:** 2
- **Tried:** FKC itself was not rerun, by the owner's answer to board question 41. Plain SDXL rendered 99 GenEval two-object prompts at 4 images each (396 images), scored by GenEval's detector (the share of images where both named objects are found). FKC's cost was timed on one prompt.
- **Bar, written first:** the same pass-band rule as the RTD rerun, pinned in the FKC design's run settings.
- **Result:** Plain SDXL scored 0.7399 of 396 images correct against 0.80 printed in FKC Table A4, with a pass band of 0.6468 to 0.8330: reproduced. FKC's printed 0.78 stays as context. One batch of 8 particles takes 205.6 seconds on an RTX 3090 against 12.8 seconds for a plain render, 16.1 times as long, which projects to 22.6 GPU-hours for the 396 images. On the one prompt timed, all 8 particles at the last step descend from one ancestor. The sources disagree in three places. FKC's verdict word is "Not tested, by the owner's choice" in the review but `"not comparable"` in the rows file. The design says FKC "does not run until the owner has seen the measured cost", while the review says it "is not rerun". The guard picture says "all 553 released prompts" are rendered, while the review says 99 two-object prompts.
- **Verdict:** FKC not run. Plain SDXL support (reproduced).
- **Why it failed:** `inferred` (stated by the review): FKC's printed 0.78 sits below plain SDXL's printed 0.80 on the same prompts, so a rerun cannot raise the bar a Paper 2 win must beat.
- **Kind of failure:** plain SDXL: none, it passed its bar (0.7399 of 396 images correct against 0.80 printed, band 0.6468 to 0.8330). FKC: not tested, by the owner's answer to board question 41.
- **Idea still alive?** plain SDXL: alive, but the pass covers only the 99 GenEval two-object prompts at 4 images each, and the sources disagree on whether all 553 GenEval prompts or only those 99 were rendered. It licenses using plain SDXL's printed 0.80 as the GenEval number a Paper 2 win must beat. FKC: unclear, because its printed 0.78 is neither confirmed nor contradicted, and a rerun is projected at 22.6 GPU-hours.
- **Rules out:** Settles that our GenEval prompts and scorer return plain SDXL's printed score. Check the 22.6 GPU-hour cost and the particle collapse to one ancestor before planning an FKC rerun.
- **Leaves open:** Whether FKC reproduces. A plain SDXL GenEval exemplar sheet.
- **Evidence:** [the rival reproduction finding](../../../../../1-lora-correction/report/against-other-methods/do-the-rivals-reproduce-their-printed-scores.md), [the FKC review](../../../../../artifacts/notes/progress/reproductions/FKC/review.md) and [the FKC design](../../../../../artifacts/notes/progress/reproductions/FKC/design.md). W&B `qcdz35cj`; Slurm 93251 (render) and 93311 (scoring); commit pair `ac85826a-47e659d`. (D2)

### Do the corrected and plain runs part sharply or gradually?

- **Asked:** Do the corrected and plain runs separate in meaning sharply, in a window around the fork step (step 16), or gradually? Do pairs that separate more early respond worse to the correction?
- **Paper:** 2
- **Tried:** Both arms of 9 cached 50-step runs (cat and dog, frog and toad, leopard and jaguar, at seeds 9 to 11) were decoded step by step. Each was placed on the axis from its plain picture (0) to its corrected picture (1) in CLIP and DINOv2. The same read was repeated over 19 pairs at 20 steps, 27 curves.
- **Bar, written first:** two checks written as constants in the script. Near-flat separation before step 10, and the steepest rise inside steps 13 to 20. For difficulty, a rank correlation of 0.5 in size to support and 0.3 to kill.
- **Result:** The steepest rise fell inside steps 13 to 20 for 3 of 9 runs in CLIP and 1 of 9 in DINOv2 (`verified`). At step 16 the runs had covered 0.32 to 0.72 of the axis in CLIP (median 0.51) and 0.43 to 0.83 in DINOv2 (median 0.62) (`verified`). The rank correlation between early separation and the area under each pair's compose-rate curve is -0.500 in CLIP, "exactly on the 0.5 bar", and -0.690 in DINOv2, over 8 pairs (`verified`).
- **Verdict:** none recorded on the page: it states the runs part gradually from about step 3 with no sharp fork (`stated`, from the card). Its card calls the sharp-window story falsified, which maps to null for the steepest-rise check, and says the difficulty correlation passes its bar, which maps to support.
- **Why it failed:** The fork step is roughly the midpoint of semantic separation, not its start. The raw-latent fork marks when accumulated drift clears the noise floor (inferred, from the card).
- **Kind of failure:** the idea, for the sharp window (verified: 3 of 9 and 1 of 9). Test design for the difficulty correlation: 8 points, CLIP sitting exactly on the bar, and mixed seed counts (8 pairs at 2 seeds and 11 at 1, stated). The card says "Eight points is support, not proof".
- **Idea still alive?** closed for a sharp semantic fork at steps 13 to 20 on these 9 runs. The difficulty correlation is a retry worth it, with more pairs scored by the strength sweep.
- **Rules out:** Do not describe step 16 as the moment the runs diverge in meaning. Do not read raw pairwise embedding distance between early frames as timing, since the card calls it the instrument warning.
- **Leaves open:** whether early separation predicts difficulty over more than 8 pairs.
- **Sources disagree:** The 20-step picture's title says 19 pairs by 2 seeds. The file holds 27 curves (8 pairs at 2 seeds, 11 at 1). Recorded unresolved.
- **Evidence:** [the separation card](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/how-far-the-corrected-run-separates-from-the-uncorrected-one/README.md), [the nine-run separation picture](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/how-far-the-corrected-run-separates-from-the-uncorrected-one/divergence_eyeball.png), [the 19-pair picture](../../../../../1-lora-correction/artifacts/results/what-the-correction-is/does-it-cause-composition/how-far-the-corrected-run-separates-from-the-uncorrected-one/dose_divergence_eyeball.png). No W&B run id given.

### Hand sort of the 32 images behind the strength sweep's 94%

- **Asked:** Do the images the detector scores as composed at full correction strength actually contain both requested animals?
- **Paper:** 2
- **Tried:** All 32 images at full strength (8 animal pairs at seeds 9 to 12) were sorted by hand into piles and compared cell by cell with the detector's counts in `dose_curves.json`. The detector is the same instance-count rule: "animal" boxes at confidence 0.30 or more, longer side at least a quarter of the image, composed at a count of 2 or more.
- **Bar, written first:** none recorded ("No verdict has been recorded for this question").
- **Result:** The detector called 30 of 32 composed (93.75%, printed as 94%). Hand piles: both requested animals present 12, two of the same animal 1, cannot decide 5, the two animals look alike 12, scored as a failure 2. Recounting whole images: as the detector scored them 30 (93.75%); taking out the two-dog image 29 (90.6%); also taking out the 5 cannot-decide images 24 (75.0%); also taking out the 12 look-alike images 12 (37.5%); counting the seal and walrus image at seed 9 as composed 31 (96.9%). The detector reported a count of 3 on leopard and jaguar at seed 9 for two big cats, the third box being 162 by 473 pixels at confidence 0.309. The 94% does not carry the comparison-strip fault, because the strength-sweep scorer reads only its own run's file `verified`.
- **Verdict:** none recorded: the page shows a hand sort and a range of rates, with the counting choice left to the owner.
- **Why it failed:** The detector's count is not a count of the requested animals. It scores two dogs (cat and dog, seed 10) as composed `verified`, and a third box can clear both floors `stated`. Seventeen images could not be settled by eye `verified`. Sources disagree on the lower end of the true rate: the audit card says about 87% to 94%, while the plan says 75% to 94% and that 87% "counts the 17 uncallable images as successes". The page computes that as 29 of 32, or 90.6%, and finds no arithmetic for 87%. They also disagree on images: on seal and walrus at seed 9, the card calls the failure correct while the picture shows two separate bodies; on elephant and penguin at seed 9, the card says "two elephants and no penguin" while the picture shows one elephant; on elephant and penguin at seed 11, the card says both present while the second creature's identity as a penguin is a reading `inferred`.
- **Kind of failure:** could not measure it (`verified`: one two-dog image scored composed; 17 of 32 not callable by eye). test design: the sort covers one strength, is one person's reading, and three of the 8 pairs were chosen because they blend `stated`.
- **Idea still alive?** unclear. The 94% stands only as a detector count. The card and the strength-sweep review call it an upper bound, which holds only while the seal and walrus image at seed 9 stays a correct failure `inferred`. The three-state labelled set would tell.
- **Rules out:** Do not report the detector's count of two as "both requested animals". Check by eye or with a species-level reader before quoting a compose rate as composition, and say how unsettled images are counted.
- **Leaves open:** How the 17 unsettled images count. Whether seal and walrus at seed 9 is a missed composition. Hand sorts at strengths 0.25, 0.5 and 0.75. The file `scorer_validated.json`, which the strength-sweep scorer refuses to run without, is absent from this laptop.
- **Evidence:** [the hand-sort finding](../../../../../1-lora-correction/report/measuring-composition/do-the-scored-successes-contain-both-animals.md), [the audit card](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md), [the two-dog image scored composed](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/two-of-the-same-animal/a_cat__x__a_dog_seed10_n2.png), [the seal and walrus image scored a failure](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/scored-as-a-failure/a_seal__x__a_walrus_seed9_n1.png), [the plan for trusting the compose rate](../../../../../1-lora-correction/plans/02-can-we-trust-the-compose-rate/MASTER_PLAN.md) and [the three-state labelled set plan](../../../../../1-lora-correction/plans/02-can-we-trust-the-compose-rate/plans/tools/04-the-three-state-labelled-set.md). No W&B id is given.

### Transfer as a rate over fifteen held-out pairs

- **Asked:** Do most held-out pairs compose on an adapter that never saw them, measured as a rate with a spread over fifteen leave-one-pair-out runs?
- **Paper:** 2
- **Tried:** Nothing has run. The design is fifteen adapters, each trained with a different pair held back and tested on exactly that pair, plus a curve of compose rate against how much of the pool was held back.
- **Bar, written first:** The question is written before the runs ("Do most held-out pairs compose on an adapter that never saw them?"). The page says a failure of the threshold closes the plan, but no numeric threshold appears in the review. The design file is said to hold it, and I did not read it.
- **Result:** none. The runs table reads "not started".
- **Verdict:** not run. The two files point at different reviews. The root report index says "The transfer rate over fifteen pairs has its verdict in" review 03. The transfer finding page says the rate over many pairs is the job of review 04. Review 03 itself records only the single pooled run and says "this is one pooled run, not a rate. The rate comes from the fifteen runs". Review 04 says "Nothing has run yet".
- **Why it failed:** Not run. It waits on the one-epoch first run in the three-live-curves review going green `stated`. Review 03's go-ahead (2026-11-13) carries one condition: anchor images exist for only 2 of the 19 sampled pairs, so either generate anchors for the other 17 or read the compose rate from a scoring pass instead of the live curve.
- **Kind of failure:** none: not run.
- **Idea still alive?** unclear until it runs. The question is open, and the go-ahead is given.
- **Rules out:** Do not cite review 03's 0.906 as the fifteen-pair transfer rate. Before launch, print each run's realised training pool and confirm the held-out pair is absent and the hold-out flag selected a non-empty set. Score each run over its own output directory only.
- **Leaves open:** Everything: the rate and its spread, the shape of the degradation curve, and for any pair at the no-fix level, whether the correction never arrived or arrived pointing wrong.
- **Evidence:** [the fifteen-pair review](../../../../../1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/review/04-transfer-as-a-rate-over-fifteen-pairs.md), [the fifteen-pair plan](../../../../../1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/plans/hypothesis/04-transfer-as-a-rate-over-fifteen-pairs.md), [the three-live-curves review it waits on](../../../../../1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/review/02-three-live-curves-while-training.md), [the single-run review the index points at](../../../../../1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/review/03-does-one-pooled-fix-transfer-at-all.md) and [the root report index](../../../../../report/00-INDEX.md) (step 11 in review 03's go-ahead).

### The adapter against other methods on the 276 Attend-and-Excite prompts

- **Asked:** Does the adapter score above the plain product of experts on BLIP-VQA on the Attend-and-Excite prompt sets, paired on prompt, seed and starting noise?
- **Paper:** 2
- **Tried:** One checkpoint was used throughout: `v58-03-self`, a distillation-trained adapter with self-attention layers, at training step 30,000, on all 50 sampling steps. It rendered 276 prompts (66 animals, 144 animals with objects, 66 objects) at four seeds (9 to 12) per prompt. The comparison set was the joint prompt, the plain product, CO3, ToMe and the plain product with Langevin. Scores are BLIP-VQA (0 to 1, the product of yes-probabilities over the prompt's phrases) and a detector count over 264 animal images per method.
- **Bar, written first:** the protocol's rule, fixed before any evaluation image was scored: SUPPORTS if the 95% interval on the adapter minus the plain product sits above 0 in all three sets. The interval comes from resampling prompts and then seeds.
- **Result:** The adapter minus the plain product, in BLIP-VQA: animals 0.278 (interval 0.1997 to 0.3551, 66 prompts), animals with objects 0.1299 (0.0726 to 0.1871, 143 prompts), objects 0.0927 (0.0287 to 0.158, 66 prompts). Mean BLIP-VQA, with the adapter against the plain product, CO3, ToMe and the joint prompt in that order: animals 0.5031 against 0.2251, 0.7437, 0.8059 and 0.7067; animals with objects 0.6753 against 0.5453, 0.8688, 0.8778 and 0.8789; objects 0.5451 against 0.4524, 0.4185, 0.7020 and 0.4806. Share of the 264 animal images where the detector counted two or more animals: adapter 0.9659, plain product 0.0189, joint prompt 0.8902, CO3 0.9394, ToMe 0.9621. On prompts that share no concept with the 48 training pairs the differences are 0.2689 (28 prompts, interval 0.1621 to 0.3723), 0.1722 (88 prompts, 0.0982 to 0.2434) and 0.1023 (55 prompts, 0.0318 to 0.1726), with no verdict recorded for that subset. The sources disagree on the intervals: the review folder's sidecar gives 0.2005 to 0.352, 0.0709 to 0.1869 and 0.0273 to 0.1579, and the page says the shared random stream accounts for the change (`inferred`). They also disagree on CO3's settings: the table labels it DDIM at guidance 7.5, while the per-image score file records CFG++ at 0.8.
- **Verdict:** support (recorded SUPPORTS) on the pre-registered rule. No verdict was recorded for the subset without shared concepts.
- **Why it worked:** The page gives no mechanism. The support is a paired difference on the same prompt, seed and starting noise, with intervals clear of 0 in all three sets (`verified`).
- **Kind of failure:** none, it passed its bar. Two kinds limit how far the pass reaches. Could not measure it, for the count column (stated: the detector is asked for "animal", so "two of the same animal count as a success", and the adapter reads 0.9659 on the count against 0.5031 on BLIP-VQA for the same 264 images). Test design, for the rival columns (stated: CO3, ToMe and the joint prompt read the sentence naming both concepts while drawing and the adapter does not, so only the plain product is a matched comparison).
- **Idea still alive?** alive, but the pass covers only the adapter beating the plain product on BLIP-VQA on the three Attend-and-Excite sets, at one checkpoint (v58-03-self at step 30,000) and four seeds (9 to 12). It says nothing against ToMe, CO3 or the joint prompt, which all score above the adapter on animals, and no verdict was recorded on the subset of prompts sharing no concept with training.
- **Rules out:** Settles that the adapter improves on the plain product on these prompts. Do not quote the count column as a composition rate on its own: it cannot tell two of the same animal from a pair. Do not read the gap to CO3 or ToMe as a method difference, because their inputs differ.
- **Leaves open:** The sampler and guidance labels on the CO3 and ToMe rows. The image list behind ToMe's count. A viewable render grid.
- **Evidence:** [the Attend-and-Excite finding](../../../../../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-attend-and-excite-prompts.md), [the protocol that set the bar](../../../../../1-lora-correction/report/paper-review/protocol.md), [the Table 1 result file](../../../../../1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/main_table.csv) and [the subset file](../../../../../1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/no_overlap_tables.json). No W&B run is named on the page.

### Every SDXL method on the 53 held-out pairs

- **Asked:** How does the adapter compare with the plain product, CO3, the joint prompt and the product with Langevin on 53 pairs kept out of its training?
- **Paper:** 2
- **Tried:** The page does not name the checkpoint. 53 held-out pairs were rendered at eight seeds (9 to 16) and scored by mean BLIP-VQA (0 to 1) with 95% intervals from resampling pairs and then seeds. The pairs were grouped as hard (3 pairs, where the joint prompt shows both things on fewer than 4 of 8 seeds), easy but unseen (29) and not competing (21, object-and-place pairs).
- **Bar, written first:** none recorded: the protocol's pass rule covers only the benchmark prompts.
- **Result:** Over all 53 pairs: adapter 0.6306 ± 0.0699, plain product 0.5187 ± 0.0789, joint prompt 0.7655 ± 0.0586, CO3 0.7757 ± 0.0512. Hard group: adapter 0.4543 ± 0.4163 against plain product 0.3301 ± 0.3854. Easy but unseen: 0.5755 ± 0.0934 against 0.3783 ± 0.0865. Not competing: 0.7319 ± 0.0831 against 0.7395 ± 0.0977, so the adapter is slightly below. On the 23 pairs sharing no concept with training, the adapter's mean is above the plain product's in all three groups (hard, 1 pair: 0.8591 against 0.791; easy, 10: 0.5737 against 0.4693; not competing, 12: 0.7119 against 0.6796), and each pair of intervals overlaps. By eye, the adapter scores 0.95 on seal and walrus seed 11 and 0.01 on cow and horse seed 13, where it draws two cows. On three worst rows it has no bowl, no glacier and no park bench while CO3 shows both things (`verified`). The sources disagree on the adapter's half-widths: this page gives 0.4163, 0.0934 and 0.0831, the review folder's copy 0.4189, 0.0939 and 0.0840. The page attributes the difference to the shared random stream (`inferred`).
- **Verdict:** none recorded: the page reports means with intervals and no pass rule.
- **Why it failed:** Not a failure verdict. The page's eye read is that the adapter "loses or swaps one" thing on its worst cells, where CO3 usually keeps both (`verified` on the sheet; how often is not recorded).
- **Kind of failure:** Test design (`verified`: no bar written first; the hard group holds 3 pairs, with half-widths from 0.2742 to 0.4163 on a 0 to 1 score).
- **Idea still alive?** Unclear. A pass rule written first for held-out pairs, with more hard pairs, would tell whether the adapter's held-out gain over the plain product is real.
- **Rules out:** Do not claim a held-out win on the hard group (3 pairs). Do not read the gap to CO3 or the joint prompt as a method difference: both read the sentence naming both concepts.
- **Leaves open:** The script that wrote the pooled file. The guidance label on CO3's rows (7.5 written against CFG++ 0.8 recorded).
- **Evidence:** [the held-out finding](../../../../../1-lora-correction/report/against-other-methods/how-does-every-sdxl-method-compare-on-the-held-out-pairs.md), [the group scores](../../../../../1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/held_out_ablation.csv) and [the best and worst cells sheet](../../../../../1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-every-sdxl-method-compares-on-held-out-pairs/methods_side_by_side.png). No W&B run is named.

### The adapter on GenEval and T2I-CompBench

- **Asked:** How does the adapter score against the plain product, CO3 and the joint prompt on public benchmarks, each with its own prompts and scorer?
- **Paper:** 2
- **Tried:** The page does not name the checkpoint. Four methods ran on GenEval two objects (396 images per method), colour attribute (400) and position (400) at seeds 9 to 12, and on T2I-CompBench colour (200) and shape (200) at seeds 9 and 10. GenEval scores the fraction of images judged correct. T2I-CompBench scores mean BLIP-VQA.
- **Bar, written first:** none recorded.
- **Result:** Adapter against the plain product, CO3 and the joint prompt, in that order: GenEval two objects 0.4571 against 0.2247, 0.7500 and 0.7626; colour attribute 0.1750 against 0.1025, 0.2125 and 0.2050; position 0.1075 against 0.0225, 0.1425 and 0.1175; T2I-CompBench colour 0.5778 against 0.4511, 0.5990 and 0.6007; shape 0.4631 against 0.3665, 0.4616 and 0.4683. The texture column rests on 0 images and is left out. The sources disagree on which tasks the table holds: the layout sheet lists GenEval two objects and counting and T2I-CompBench spatial and complex. The pair ledger also records counting and spatial runs as complete (99, 80 and 300 prompts), but the result file has no counting or spatial column, and "where the counting and spatial scores went is not recorded".
- **Verdict:** none recorded: the page reports one mean per cell with no intervals.
- **Why it failed:** Not a failure verdict. The adapter is above the plain product in all five columns and below the joint prompt in all five, with CO3 above it in four (`verified` from the result file).
- **Kind of failure:** Test design (`verified`: no intervals and no bar, so the 0.0015 gap on shape cannot be read as a difference).
- **Idea still alive?** Unclear. Intervals on these means and a bar written first would tell.
- **Rules out:** Do not argue a win over CO3 on T2I-CompBench shape from 0.4631 against 0.4616. Do not compare the adapter with CO3 or the joint prompt as matched methods, since both read the sentence naming both concepts.
- **Leaves open:** The counting, spatial and complex tasks the layout sheet lists. How the sheet's rows were chosen. Intervals.
- **Evidence:** [the standard-benchmarks finding](../../../../../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-standard-benchmarks.md), [the benchmark result file](../../../../../1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/standard_benchmarks.csv) and [one benchmark tile where the plain product loses the apple](../../../../../1-lora-correction/artifacts/results/against-other-methods/the-head-to-head-tables/how-the-adapter-scores-against-other-methods/tiles-table2/t2i_color__77__s9__poe.png). No W&B run is named.

### Training the adapter on the first 10, first 25 or all 50 steps

- **Asked:** Does training the adapter only on the first 10 or 25 drawing steps change what it draws for cat and dog, against training on all 50?
- **Paper:** 2
- **Tried:** Three adapters, each switched on for all 50 drawing steps, on cat and dog at three held-out seeds from the same noise, read by eye:
  - first 10 steps: `v57-09-x0-seeds` at 40,000, which also differs in loss and seeds
  - first 25 steps: `pool43-early25` at 40,000
  - all 50 steps: `pool43-all50` at 40,000
- **Bar, written first:** none recorded.
- **Result:** By eye at seed 12 (verified): first 10 gives two animals, first 25 gives two animals with the white one smooth, all 50 gives two animals with the white one smooth. With no adapter the picture is one fused animal.
- **Verdict:** none recorded: the page shows tiles without a rate.
- **Why it failed:** No conclusion was possible. The page says "Both groups of columns are mixed comparisons ... the first-10 column also differs in loss and seeds", and "Three seeds of one pair give no rate" (verified).
- **Kind of failure:** **test design** (verified): mixed comparison, three seeds, by eye.
- **Idea still alive?** unclear. A first-10 adapter trained like the other two would tell the window apart from the loss and seed change. The every-adapter card says such a comparison "made no significant difference", but it is not linked.
- **Rules out:** Do not describe this figure as a drawing window. The render script used a training window: the first-25 tiles match `pool43-early25`'s saved tiles with mean grey-level differences of 0.48, 0.90 and 0.52 of 255 at seeds 9, 11 and 12 (verified), so the window is the training range (inferred). Do not credit the first-10 column's look to the window.
- **Leaves open:** **Where the sources disagree** (unresolved): the paper caption says the correction runs "during the first 10, first 25, or all 50 denoising steps" and "the PoE completes the remaining steps", while the figure heading says "Steps it trains on".
- **Evidence:** the [settings finding](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/how-the-adapter-settings-change-cat-and-dog.md) and the [rank 8 tile at seed 12](../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/tiles/r2c1.png).

### The adapter attached to the joint prompt itself

- **Asked:** What does the adapter do when attached to a prompt that already names both things?
- **Paper:** 2
- **Tried:** An unnamed adapter (the layout file says only "Adapter trained for 1,000 epochs") was attached while drawing the joint prompt for tiger and dog, chess board and hourglass, and cat and dog. Each was compared by eye with the base model from the same noise.
- **Bar, written first:** none recorded.
- **Result:** By eye (stated by the page): the adapter "adds an hourglass the base model left out and adds extra animals to tiger and dog". The same limitations figure names six cat and dog failure outcomes: one fused animal, grey mush with no animal, flat line art of one animal, a third animal in the corner, two animals of the wrong species, and invented people.
- **Verdict:** none recorded: a limitations gallery with no count.
- **Why it failed:** not recorded.
- **Kind of failure:** **test design** (verified): the adapter is unnamed ("Name the adapter and the source of the limitations tiles, which no file records"), with one render per pair and no count.
- **Idea still alive?** unclear. A named adapter on the joint prompt over several seeds, with instances counted, would show whether adding extra objects is systematic.
- **Rules out:** Do not assume attaching the adapter to the joint prompt leaves it unchanged: it added an object and extra animals here. Name the adapter for any limitations figure.
- **Leaves open:** Which adapter drew these tiles. **Where the sources disagree** (unresolved): the layout file titles the first limitations group "product with the adapter", while the caption calls the top-left tile "the uncorrected hybrid". Outside its insets the tile matches the plain product at seed 9 with a mean grey-level difference of 1.19 of 255 (verified), so the caption's reading fits (inferred).
- **Evidence:** the [settings finding](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/how-the-adapter-settings-change-cat-and-dog.md) and the [flat line-art failure tile](../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/where-it-fails/how-the-adapter-settings-change-cat-and-dog/limitations-tiles/fail-r1-c0.png).

### The two-concept adapter inside a three-prompt product (PRZE-4)

- **Asked:** Does the adapter, trained on pairs only, still separate things when the product is given three prompts?
- **Paper:** 2
- **Tried:** `v58-03-self` at step 30,000 inside a three-prompt product (u + w(c1 + c2 + c3 − 3u) at w = 7.5), 50 steps, seed 10. One triple of each of four kinds: cat, dog, penguin; dog, chair, cat; cat, dog, umbrella; teapot, lamp, book. That is 8 pictures, read by eye at 1,024 pixels.
- **Bar, written first:** none recorded. The [review step](../../../../../1-lora-correction/report/paper-review/prze-4-three-concepts.md) leaves its "pass bar written before the run" box unticked.
- **Result:** By eye (verified):

  | Triple | Plain product | Adapter |
  |---|---|---|
  | cat, dog, penguin | one fused body | a dog and a cat, no penguin |
  | dog, chair, cat | one toy-like animal, no chair | two dogs, no cat, no chair |
  | cat, dog, umbrella | one animal and the umbrella | a dog, two cats and the umbrella |

  The teapot, lamp and book render is kept off the page.
- **Verdict:** none recorded: "the renders below are a description."
- **Why it failed:** not recorded. The adapter "was never trained on a triple", so these renders "say nothing about what a three-concept adapter would do" (stated).
- **Kind of failure:** **test design** (verified): one seed and one triple per kind, no bar, and none of the instance counts the review step asks for.
- **Idea still alive?** unclear. Instance counts on the same seeds against a bar written first, as the review step asks, would tell.
- **Rules out:** Do not claim the adapter "draws all three named things when the triple includes objects": the dog, chair and cat render shows two dogs. Write the pass bar before running the counted version.
- **Leaves open:** **Where the sources disagree** (unresolved):
  - The paper's main text says the adapter draws all three things for triples with objects; the caption and the render show two dogs for dog, chair and cat.
  - The main text says that with two or three animals "it stops at two"; the cat, dog and umbrella render holds three animals.
  - The figure register calls the figure judged "against a bar written first"; the review step records no bar.
- **Evidence:** the [three-concepts finding](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/does-the-adapter-compose-three-concepts.md) and the [cat, dog and penguin render](../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-the-adapter-compose-three-concepts/tiles/animal-animal-animal/ours.png).

## Project 2: classifier guidance

### A presence classifier's push on the SDXL cat and dog product (plan 15)

- **Asked:** Does adding a classifier's gradient at each sampling step make the product of a "cat" model and a "dog" model on SDXL draw two separate animals instead of one fused animal?
- **Paper:** 2
- **Tried:** A presence classifier's gradient was added to the SDXL cat and dog product. Renders were made on seed 9 (four pushes labelled very gentle to strong) and on seeds 9 and 10 by a probe of the classifier after 7,500 training steps, at pushes of 0.5 and 2.0 as fractions of the prediction's length. The eight-seed sweep the bar reads has not run.
- **Bar, written first:** two clear animals on at least 4 of 8 held-out seeds (9 to 16) at some classifier weight, called by eye at full resolution, with the damped product (each model's guidance at 3.0 in place of 7.5) no higher. 0 or 1 of 8 at every weight is no support. 2 or 3 of 8 is inconclusive. A count that rises while the pictures are destroyed is its own outcome (stated, the review file).
- **Result:** On seed 9, pushes 1 and 2 show one animal and pushes 3 and 4 are broken patterns (verified). On the probe, push 0.5 leaves one animal on seed 9 and a damaged one on seed 10, and push 2.0 breaks both (verified). No count against the bar exists.
- **Verdict:** not run. The page says "No verdict has been recorded, because the eight-seed sweep that would judge this has no result."
- **Why it failed:** The renders date from 2026-11-04, when the push was rescaled to a fraction of the whole prediction. That made it "roughly 44,000 times what the classifier asked for", so the renders "say nothing yet about the classifier" (inferred: stated from plan 17's quick context). The page also warns that a classifier trained on this sampler's fused animals may learn the sampler's failure without knowing what two animals look like (inferred).
- **Kind of failure:** done wrong: the push was about 44,000 times too large (inferred, stated). test design: two seeds on the probe, and the bar's eight-seed sweep never ran (verified from the page).
- **Idea still alive?** retry worth it. The fix the page names is to run the eight-seed sweep with the unrescaled push.
- **Rules out:** Do not rescale a classifier push to a fraction of the whole prediction. Check the push's size against the prediction before judging a classifier by its renders. Check which classifier made a picture before citing it.
- **Leaves open:** Whether a classifier push composes this product at all. Whether a classifier trained on the sampler's own fused animals generalises. The sources disagree on whether a classifier was trained at all: the card says "No head is trained and no render has been made through a guided sampler", while the folder holds a picture titled as pushed after 7,500 training steps (that it came from the trainer is inferred).
- **Evidence:** [the classifier-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-classifier-supply-the-missing-term.md), [the review that set the bar](../../../../../2-classifier-guidance/plans/review/15-can-a-classifier-supply-the-missing-term.md), [plan 15](../../../../../2-classifier-guidance/plans/15-can-a-classifier-supply-the-missing-term.md). The pushed pictures failed the subject check and are not linked. (plan 15)

### Separate "cat?" and "dog?" tests on fused renders, against one test where the species compete

- **Asked:** Which test can see both animals in a fused cat and dog render, so that a classifier's labels do not teach it that a fused cat-dog is a dog?
- **Paper:** 2
- **Tried:** A floor on CLIP cosine similarity to "cat" and to "dog", each asked on its own, was calibrated on 200 photographs per species. It was then applied to 14 fused SDXL renders from 11 seeds and compared with a two-way test where the species compete in one softmax. Separately, CLIP ranked 8 fused renders against 8 joint-prompt renders by similarity to "a cat and a dog" minus similarity to "one animal that is half cat and half dog".
- **Bar, written first:** none recorded as written before the run. The calibration's own bars were an own-species pass rate of at least 0.90 and a cross-species rate of at most 0.10 (stated, the card).
- **Result:** At the chosen floor of 0.245, own-species pass rates are 0.925 and 0.950 and cross-species rates are 0.070 and 0.035 (stated). Both separate tests fired on 8 of 14 fused renders (verified, counted). The two-way test saw both animals on 1 of 14 (stated). The median cosine on the 14 is 0.2618 to cat and 0.2683 to dog, and a softmax turned that 0.0065 gap into 0.003 against 0.979 (stated). In the ranking, the 8 fused renders score -0.034 to +0.017 and the 8 joint-prompt renders +0.018 to +0.050, so every joint render ranks above every fused one (verified). The call can flip between two renders of one seed: seeds 4, 5 and 42 each appear twice, and in each pair one render clears both floors and the other does not (verified).
- **Verdict:** none recorded. The page shows the separate tests seeing both animals on 8 of 14 against 1 of 14, with calibration rates inside their bars.
- **Why it worked:** A softmax makes two near-equal similarities share one total, so a small gap becomes a confident single-species answer. A floor asks each question on its own (inferred, argued with the page's worked numbers).
- **Kind of failure:** none recorded as a pass against a pre-written bar. Two limits: could not measure it, because the call flips between two renders of one seed (verified on 3 seeds) and the exact sweep rates sit in a `sweep.json` on `/datasets` that was not re-read (stated). test design: 14 renders (verified).
- **Idea still alive?** alive, for labelling fused animals with one floor per species instead of a competing softmax. It does not license steering with CLIP: "Reading a picture and pushing a noisy one are separate abilities."
- **Rules out:** Do not label fused renders with a softmax over competing species names. Ask about each species with its own floor. Do not treat CLIP's ranking of finished pictures as evidence that it can steer a noisy one.
- **Leaves open:** Whether CLIP can steer. The floor's unit is in dispute: the first picture's x-axis says "probability" and the second's titles say p(cat), while the card says raw cosine similarity. The code reads cosine with no softmax, and the axis label is fixed text in the script.
- **Evidence:** [the classifier-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-classifier-supply-the-missing-term.md), [the fused-render sheet with both similarity scores](../../../../../2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/hybrid-check-cosine-sheet.png), [the CLIP ranking of fused against joint renders](../../../../../2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/clip-pair-vs-smoosh-ranked.png). (plan 15)

### A dog classifier on frozen SDXL features (plan 17)

- **Asked:** Can a classifier trained on frozen SDXL features tell whether a half-drawn picture holds a dog, well enough to steer toward "dog"?
- **Paper:** 2
- **Tried:** A dog classifier trained on frozen SDXL features was read at sampler step 20 of 50, on four "a dog" renders and on the four most-pushed renders.
- **Bar, written first:** none recorded on this page.
- **Result:** At step 20 it gave the four "a dog" renders a dog probability of 0.00, 0.85, 0.00 and 0.85. It gave the four most-pushed renders, none holding a dog, 0.90, 0.92, 0.77 and 0.93 (verified, read off its belief sheet).
- **Verdict:** none recorded: the page reads these probabilities as the classifier misreading the pictures. The sources disagree on whether its push drew a dog: plan 18's opening says plan 17's classifier "drew no dog", while plan 17's instruction 6, where that sheet is called, has no call written.
- **Why it failed:** It scored pushed pictures with no dog higher than real dog pictures (verified). The page gives no reason for this.
- **Kind of failure:** could not measure it: the checker could not see a dog at step 20 (verified, the probabilities above). test design: 8 pictures and no bar (verified).
- **Idea still alive?** unclear. Eight pictures and no recorded call cannot tell a bad classifier from a bad idea. A held-out balanced accuracy per sampler step, the check plan 18 later used, would.
- **Rules out:** Check a steering classifier on held-out noisy pictures before using its gradient. A high probability on pushed pictures with no dog means the push is fooling the checker.
- **Leaves open:** What plan 17's dog sheet shows, since no call is written.
- **Evidence:** [the dog-guidance finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-classifier-guidance-draw-a-dog.md), [plan 17](../../../../../2-classifier-guidance/plans/17-classifier-guidance-from-the-paper.md). The belief sheet stays off the page. (plan 17)

### A dog classifier trained the paper's way, steering SDXL from an empty prompt (plan 18)

- **Asked:** Does Dhariwal and Nichol style classifier guidance draw a dog on SDXL when sampling starts from an empty prompt?
- **Paper:** 2
- **Tried:** A dog classifier was trained the way Dhariwal and Nichol train theirs. Its gradient pushed every sampling step of SDXL from an empty prompt, at strengths 10, 30, 100 and 300, on 4 seeds (9 to 12). The same push was also started from the prompts "an animal" and "a cat". Run `dog-01`, W&B `z2k1bdmi`.
- **Bar, written first:** First, a check that the classifier sees a dog in noisy pictures: balanced accuracy of at least 0.75 on at least 10 of the first 20 sampler steps. Then a clean dog on at least 3 of 4 seeds at some strength (stated, the plan's outcomes).
- **Result:** The check failed: 7 of 20 steps. Balanced accuracy rises from 0.62 at the noisiest step to 0.82 at step 19 and passes at steps 13 to 19 (stated). The dog sheet was drawn past the failed check at the user's choice and gave a clean dog on 2 of 4 seeds (seeds 9 and 10). Seeds 11 and 12 show a dog forced in with the picture damaged (stated, Claude's calls open to veto). At strength 0 the tiles match the empty-prompt tiles pixel for pixel on 4 of 4 seeds (verified). Starting from "an animal" gave clean dogs on seeds 10 and 9 at strength 10 and a wolf-like canine on seed 11 at 30, and on seed 12 a dog fused with a second white body. Starting from "a cat" gave a cat on every seed at every strength up to 100 and a broken cat at 300 (stated).
- **Verdict:** null (bar not met: 2 of 4 against 3 of 4). The classifier's check also failed.
- **Why it failed:** The classifier's reading of noisy pictures stays under 0.75 until step 13 (verified: a measured accuracy recorded in plan 18). That this weak early reading is why the push fails is inferred.
- **Kind of failure:** done wrong: the sheet was drawn past a failed classifier check (stated). test design: 4 seeds at four strengths, where the page says this gives "a picture of the behaviour and no rate" (verified). could not measure it: calls are by eye and open to veto (stated).
- **Idea still alive?** unclear. Four seeds cannot separate 2 of 4 from 3 of 4. A classifier that passes the noisy-step check, judged on more seeds, would tell them apart.
- **Rules out:** Do not draw past a failed noisy-step check and read the sheet as a test of the idea. Do not try to estimate a success rate from 4 seeds.
- **Leaves open:** How often a push draws a clean dog. Task 4.1 of plan 18 (the same classifier for a cat, then both) has not run.
- **Evidence:** [the dog-guidance finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-classifier-guidance-draw-a-dog.md), [plan 18](../../../../../2-classifier-guidance/plans/18-a-dog-classifier-trained-the-papers-way.md), W&B `z2k1bdmi`. The dog sheets are kept off the page by the pair ledger. (plan 18)

### The same dog push on the SDXL cat and dog product (plan 18)

- **Asked:** Does a push asking only "is there a dog" pull the product's fused animal apart into a cat and a dog?
- **Paper:** 2
- **Tried:** Plan 18's dog classifier pushed the cat and dog product on seeds 9 to 12 at strengths 10, 30, 100 and 300, shown beside the joint prompt and the plain product.
- **Bar, written first:** none recorded for the product sheet (the 3 of 4 bar was for the dog sheet).
- **Result:** Two separate animals on 0 of 16 pushed tiles (stated). The push moved the fused animal toward one cat, although it asks for a dog (stated). The push measured 0.5%, 1.2%, 1.6% and 1.9% of the prediction's length at the four strengths (verified). Seed 9 stays fused until the picture breaks at 300. Seeds 10, 11 and 12 are recorded as one cat at 30 to 300 (stated). At strength 0 the tiles match the plain product pixel for pixel on 4 of 4 seeds (verified).
- **Verdict:** none recorded: the page records 0 of 16 tiles with two separate animals.
- **Why it failed:** "This classifier asks only 'is there a dog'. The product result says that one question does not separate the animals" (inferred).
- **Kind of failure:** the idea: a single-species presence question gives the push no reason to split a fused animal (inferred). test design: 4 seeds and no bar (verified). done wrong: the classifier had failed its noisy-step check, 7 of 20 steps (stated).
- **Idea still alive?** unclear, leaning against. 0 of 16 tiles is consistent with the argument, but there was no bar and only 4 seeds. A bar written first on 8 or more seeds would settle it.
- **Rules out:** Do not expect a single-species presence push to split a fused animal. Use a pair-aware question, which the page points to on the A3q page.
- **Leaves open:** Whether seeds 10, 11 and 12 at strength 30 are one cat or one fused animal. The sources disagree: plan 18, calling from the sheet, says one cat, while the three full-size tiles opened on 2026-11-22 each show one animal with features of both (inferred, one reading). The owner decides.
- **Evidence:** [the dog-guidance finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-classifier-guidance-draw-a-dog.md), [the product pushed toward dog](../../../../../2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/product-pushed-to-dog/sheet.png), [plan 18](../../../../../2-classifier-guidance/plans/18-a-dog-classifier-trained-the-papers-way.md), W&B `z2k1bdmi`. (plan 18)

### Classifier guidance on the authors' released 256 px ImageNet weights

- **Asked:** Does Dhariwal and Nichol's classifier guidance, run through this project's wrapper on their own released 256 px ImageNet model and classifier with nothing retrained, change nothing at scale 0 and draw the asked-for class at scale 10?
- **Paper:** 2
- **Tried:** The released model and classifier were sampled at scales 0, 1 and 10, with ancestral sampling, 16 seeds per class, for Pembroke Welsh corgi (263), tabby cat (281) and golden retriever (207). Seed i starts from the same noise at both scales. Upstream commit `22e0df8`. The board lists Slurm jobs 89232 and 89235.
- **Bar, written first:** Reproduced if the classifier says yes on at least 14 of 16 at scale 10 and on fewer at scale 1. Inconclusive if the classifier's count minus the eye count at scale 10 is 4 or more, with the eye count named as instruction 13's tile-by-tile call (stated, route step 4). Scale 0 should be identical on 4 of 4 seeds.
- **Result:** At scale 0, renders match plain sampling byte for byte on 4 of 4 seeds, and plain sampling run twice also matches itself (verified, sha256). The classifier said yes to the asked-for class on 46 of 48 renders at scale 10 (corgi 16, tabby 15, golden retriever 15) against 9 of 48 at scale 1 (2, 2, 5). Mean probability of the class at scale 10 was 0.854, 0.737 and 0.907 (verified). By eye from the sheet read, scale 10 gave 13, 12 and 14 (stated). By eye tile by tile, scale 10 gave 9, 11 and 10, and scale 1 gave 0, 0 and 3 (inferred: one reading by Claude, accepted by the owner). All 22 disagreements over the 96 renders are the classifier saying yes and the eye saying no (verified).
- **Verdict:** two verdicts. Support on the reproduced bar: the route ticks targets T2 to T4 as reproduced, and the owner's decision of 2026-11-17 keeps this step (G0) on the classifier's count. Inconclusive on all three classes under the tile-by-tile eye count, where the classifier minus eye gap is 7, 4 and 5. With the sheet-read eye count the gap is 3, 3 and 1, so the inconclusive condition is not triggered.
- **Why it worked:** The wrapper runs the authors' guidance correctly. Scale 0 is byte-identical (verified), and the push moves the classifier's count from 9 to 46 of 48 (verified).
- **Kind of failure:** none, it passed its bar. Limits on how far the pass reaches: could not measure it, because "The same classifier steers each render and then grades it, so its count is partly self-fulfilling", and all 22 disagreements run one way (verified). test design: 16 samples per cell with no confidence interval, and the paper's FID was not attempted (stated).
- **Idea still alive?** alive, for using this wrapper's classifier guidance as a working reproduction on the authors' ancestral sampler. It says nothing about the DDIM form the product sampler uses ("Only ancestral sampling ran") or about composition ("Every render asks for one class").
- **Rules out:** Do not grade guided renders with the classifier that steered them. Use a judge the push was not trained against.
- **Leaves open:** Whether T2 to T4 stay reproduced on the classifier's bar or become inconclusive on the eye count (the owner decides). The sources disagree on the eye count at scale 10: route targets T2 to T4 say 13, 12 and 14 of 16, while instruction 13 and `your_call` say 9, 11 and 10. The card says the sheet read counted corgi-coloured and Labrador-like dogs.
- **Evidence:** [the released-weights finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-classifier-guidance-work-on-the-released-weights.md), [the per-render calls](../../../../../2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/classifier_calls.csv), [the scale 0 checksums](../../../../../2-classifier-guidance/artifacts/results/does-classifier-guidance-work-on-the-released-weights/identity/identity_checksums.csv), [the route that set the bar](../../../../../2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md). No sample sheet is shown, because of the pair ledger. (G0, route step 4)

### A pair-aware classifier push against a separate-label push on the exact circle-and-square toy (A1)

- **Asked:** On a toy where the exact product of two experts draws side-by-side pairs but the sampler that adds the experts' scores stacks them, does a small classifier's gradient turn stacked draws into side-by-side ones with the experts frozen, and does a push built from separate labels fail to?
- **Paper:** both
- **Tried:** On the exact circle-and-square toy at pair weight 0.01, a small classifier's gradient was added at every step to the summed-score sampler, with the experts frozen. One classifier was pair-aware (a discriminator trained on exact product draws against the sampler's own draws, 3,500 pictures). The other was a separate-label classifier. Each count is 300 draws on seed 7, from `toy_classifier_push.py` on the laptop's RTX 4070, commits `50a1b48b` to `5ccb723f`.
- **Bar, written first:** The pair-aware push at w = 1 gives at least 150 of 300 crisp pairs, with overlaid draws no higher than unguided. The separate-label push gives at most 15 of 300 at every scale in 1, 2, 5 and 10 (verified, the plan's Goal, before any classifier was trained).
- **Result:** The pair-aware push at w = 1 gave 172 of 300 crisp and 128 overlaid, against 0 and 300 unguided, and 300 of 300 at w = 2. The separate-label push gave at most 10 of 300 over scales 1 to 10. The push points along the sampler's own ratio (cosine 0.85 to 0.96 from step 190 to step 90). It prefers the side-by-side picture to the stacked one by up to 28 log-odds, against 0 for shuffled labels, and it never reaches a tenth of the summed score's size. The closed-form best separate-label classifier reaches 34 of 300 at scale 10. With no push, raising the pair weight gives 0, 1, 18, 53, 115 and 246 crisp at 0.01, 0.03, 0.1, 0.2, 0.3 and 0.5, so the push at 0.01 beats inclusive captions at 0.3.
- **Verdict:** support (both halves of the bar pass).
- **Why it worked:** A push trained on the sampler's own mistake points along the sampler's own ratio (verified, cosine 0.85 to 0.96).
- **Kind of failure:** none, it passed its bar. Limits: test design, with one seed of 300 draws per count and labels that "do not check that the pairs land in the exact product's positions in the right proportions" (stated). The separate-label half passes on the trained classifier only, since the closed-form best one reaches 34 of 300 (verified).
- **Idea still alive?** alive, as a repair of a sampling failure on a toy whose exact product already composes. It shows no fix where the product's own target is a hybrid: the toy's competing-captions condition, whose exact product draws 4 or 19 crisp of 300, is untested. "On SDXL the gap holds sampler error and model error together, and nothing yet separates them."
- **Rules out:** Do not cite this result as evidence that a push fixes the product's target. It repairs the sampler. Both the push and inclusive captions need pictures of the pair.
- **Leaves open:** Whether a push helps where the exact product itself is a hybrid. How the SDXL gap splits between sampler error and model error.
- **Evidence:** [the exact-toy push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/a1-exact-toy-classifier-push.md), [plan 19](../../../../../2-classifier-guidance/plans/partial/19-classifier-push-on-the-exact-toy.md), [the results card](../../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/README.md), [the discriminator as seven stages](../../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator.png). (A1, plan 19)

### A small pair-aware discriminator's push on SDXL cat and dog, judged on held-out seeds (A3q)

- **Asked:** Does a pair-aware discriminator trained on a few renders checked by eye push the SDXL cat and dog product into two separate animals on held-out seeds, beyond plain product, best-of-N at equal compute and a random push of the same size?
- **Paper:** 2
- **Tried:** A discriminator was trained on 17 checked positives (a cat and a dog apart, from seeds 1 to 8) and 24 negatives (the product's own renders and lone cats and dogs, all noised the same way). Its push was applied at strengths 10, 30 and 100 and judged on held-out seeds 9 to 12 against plain product, best-of-N with N = 2, and a size-matched random push. Compute: about 2.1 GPU-hours on `mscluster107` GPU 0.
- **Bar, written first:** seven readings with counts out of 4 seeds, checked top to bottom (verified, the rung 2 review file, 2026-11-17). Only the first lets the next rung start: at some strength, composed and intact on at least 2 of 4 seeds 9 to 12, more than best-of-N, with the random push at most 1 of 4. Reading 7: at most 1 of 4 on both seed sets, and either push check at strength 100 fails. Its prescribed rerun is strength 300, or strength 100 with the cap at 0.3, before calling a null.
- **Result:** Composed was 0 of 4 on held-out seeds in every column, against 1 of 4 on training seeds (seed 4, a drawing). The size cap clipped the push on every step at strength 100. The guided push changed every picture while the random push left them plain, so the direction acts, but toward fused animals and broken pictures. The discriminator passed 13 of 20 sampler steps on its held-out check (stated).
- **Verdict:** inconclusive: draft reading 7, "too weak to read". These are Claude's draft calls, and the reader has not confirmed the 64 calls. The cap 0.3 rerun (plan 21 task 1.4.1) has not run.
- **Why it failed:** The cap clipped the push on every step at strength 100 (verified, recorded in the result). "The discriminator did not read the concept": it scored pair-prompt renders as positive whatever they showed, hybrids and two dogs included, and looked at one fixed patch of every picture (verified, the diagnostics chart).
- **Kind of failure:** done wrong: the cap cut the push on every step (verified). wrong data: 17 positives taught it the prompt, not two separate animals (verified, diagnostics). test design: 4 held-out seeds, and calls not confirmed (verified).
- **Idea still alive?** retry worth it. The fixes the page names are the cap 0.3 or strength 300 rerun, and a discriminator that reads "two separate things" on pictures it never trained on, before any push is judged. The later early-push finding in this folder, with a different checker and cap 0.3, reached 28 of 32.
- **Rules out:** Check that a discriminator separates clean pairs from hybrids and wrong pairs on held-out pictures before judging its push. Check how often the cap fires before calling a null.
- **Leaves open:** Whether this discriminator's push, uncapped, would compose.
- **Evidence:** [the A3q finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/a3q-cat-x-dog-quick-discriminator-test.md), [the rung 2 review that set the bar](../../../../../2-classifier-guidance/plans/review/a3q-cat-x-dog-quick-discriminator-test.md), [plan 21](../../../../../2-classifier-guidance/plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md), [the discriminator's logits by kind of tile](../../../../../2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/diagnostics.png). The result sheet failed the subject check. (A3q, plan 21)

### Discriminators trained on called cat and dog pairs, and frozen probes (plans 21 and 22)

- **Asked:** Can a small discriminator tell a half-drawn clean cat and dog from a fused animal on pictures it never trained on?
- **Paper:** 2
- **Tried:** Four heads on SDXL: (1) 17 clean pairs against lone animals and product renders; (2) the same plus pair-prompt failures as negatives, 64 tiles; (3) 81 clean pictures over 38 pairs, head only; (4) the same data with adapters on SDXL's trunk. Also a linear layer on frozen CLIP and on DINOv2 (two ways), read on 43 held-out pictures.
- **Bar, written first:** balanced accuracy (the average of the hit rate on clean pairs and on everything else, 0.5 a coin flip) of at least 0.75 on at least 10 of the first 20 sampler steps. The probes were read against the same 0.75.
- **Result:** Head 1: 13 of 20 steps on held-out seeds 7 and 8, on a check with no hybrids (stated). Head 2: 0 of 20, 0.53 to 0.63 (stated). Head 3: 0 of 20, 0.53 to 0.63 (stated, commit `db448f7e`). Head 4: 0 of 20, 0.55 to 0.69 (stated). Probes: 0.710, 0.613 and 0.638 (verified, `probe.json`). The final head fit its own pictures on 20 of 20 steps (stated).
- **Verdict:** null for every discriminator with hybrids among its negatives, and for the probes. Head 1 met the bar only on a check that held no hybrids.
- **Why it failed:** The A3q diagnostics show the discriminator scoring every pair-prompt render as positive, hybrids and two dogs included (verified). Too few clean positives to learn the concept is inferred.
- **Kind of failure:** wrong data: 17 or 81 positives (verified counts), with the prompt learnt instead of the concept (verified on the A3q chart). test design: head 1's held-out check had no hybrids, so its pass could not see the failure (verified).
- **Idea still alive?** retry worth it, with more clean pictures. On the same page, the readout checker rose from 0.639 to 0.737 after adding 350 joint drawings.
- **Rules out:** Do not pass a discriminator on a held-out check that lacks hybrids and wrong pairs among its negatives.
- **Leaves open:** The sources disagree on whether plan 22 ran: its task boxes are all unticked, while commit `db448f7e` and the rung 2 review record both failed checks.
- **Evidence:** [the checker finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-checker-read-a-clean-cat-and-dog.md), [held-out accuracy per step for four discriminators](../../../../../2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/01-the-first-quick-test/discriminator.png), [plan 21](../../../../../2-classifier-guidance/plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md), [plan 22](../../../../../2-classifier-guidance/plans/22-rung2-a-discriminator-trained-on-the-benchmark-pairs.md), [the rung 2 review](../../../../../2-classifier-guidance/plans/review/a3q-cat-x-dog-quick-discriminator-test.md). (plans 21 and 22)

### A pair-aware push on models of our own cat and dog photos (plan 20)

- **Asked:** Does a classifier trained on our own cat and dog images push the product of a cat model and a dog model into two separate animals on more seeds than a push built from separate "cat present" and "dog present" answers?
- **Paper:** 2
- **Tried:** An image set of 550 (200 cat, 200 dog, 150 both; 312 SD3.5 renders and 238 COCO photos) (verified). Two experiments: the authors' frozen 256 px unconditional model with the released classifier's cat and dog classes; and the authors' 64 px model trained on our images, once without the both-animal images and once on all 550. The pair-aware classifier's positives are the 150 both-animal images; its negatives are the product's own draws and the single-animal images (stated).
- **Bar, written first:** At least 4 more seeds of 16 (seeds 1 to 16) with two separate animals than the separate-label push, in at least one experiment, with pictures no more damaged than the plain product's. The pair-aware push is read at weights 1 and 2 (stated, the plan's Goal). Two rules were added after launch: on the 64 px sheets the bar is read against the stronger separate-label scale; and under a ruling of 2026-11-17, made before any tile was called, a tile called "two bodies" counts, with the strict "two clear animals" count reported beside it.
- **Result:** The 64 px model without both-animal images failed at step 32,600 of 50,000 (W&B `wubhxhse`). The one on all images failed at step 27,600 (W&B `pnwlwh0f`). The 256 px classifier on our images had held-out accuracy at step 0 of 0.93 for cat and 0.95 for dog, but its push did not steer the released model at scales 10, 30 or 100 (stated). The pair-aware sheet most likely never ran (stated). The released model pushed by the released classifier's cat and dog classes drew an animal on 4 of 4 seeds, with none showing two separate animals (verified). The walk called draws 2 and 4 hybrids and draws 1 and 3 one dog-like animal (stated).
- **Verdict:** not run. The page says "No verdict has been recorded, because the runs that would judge this stopped in a storage outage."
- **Why it failed:** The runs stopped when `/datasets` went down (stated). Why the own-photo classifier's push did not steer the released model is not recorded.
- **Kind of failure:** none of the six for the main question: the runs never finished. done wrong, for the own-photo classifier: its push did not steer (stated), cause unknown. test design: two bar rules were added after launch (stated).
- **Idea still alive?** retry worth it. The fix the page names: check which checkpoints and draws survived on `/datasets`, resume both 64 px trainings from steps 30,000 and 25,000, and rerun the 256 px pair-aware classifier and its sheet on the 512 draws.
- **Rules out:** Do not add reading rules after launch without recording both readings.
- **Leaves open:** Whether the pair-aware push works on photos. Whether four both-animal images (`both_015`, `both_027` dog only by eye, the collages `both_000` and `both_017`) stay in the set; the owner decides.
- **Evidence:** [the our-photos finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-a-pair-aware-push-compose-our-photos.md), [plan 20](../../../../../2-classifier-guidance/plans/20-our-photos-both-ways.md), [its review file](../../../../../2-classifier-guidance/plans/review/20-our-photos-both-ways.md), [the walk on our photos](../../../../../2-classifier-guidance/artifacts/drips/our-photos-both-ways/EXECUTE_WALK.md), [the released model's four product draws](../../../../../2-classifier-guidance/artifacts/results/does-a-pair-aware-push-compose-our-photos/imagenet-product-check-256.png), W&B `wubhxhse` and `pnwlwh0f`. (plan 20; the board calls it A2r and A2c)

### Readout checkers on SDXL's internal features for cat and dog (the checker loop)

- **Asked:** Can a readout checker, a small head reading SDXL's internal features, reach 0.75 balanced accuracy on held-out cat and dog pictures, and does its push give more two-body pictures?
- **Paper:** 2
- **Tried:** Seven training sets or settings, each read on 220 held-out cat and dog pictures (109 clean, about ±0.05). Each checker's push was then counted on seeds 9 to 16 for two or more bodies, from draft calls.
- **Bar, written first:** the loop's own mark: at least 0.75, and more two-body tiles than plain product and a random push of the same size (stated).
- **Result:** First version, 753 tiles: 0.639, 3 of 8 seeds with two bodies. COCO photos added: 0.600, dropped, 3. 350 new joint drawings without COCO: 0.737, kept, 5. 500 more joint drawings: 0.741, kept, 4. 2,000 training steps in place of 750: 0.755, kept, about 7. Half of each batch's rejects as wrong pairs: 0.756, kept as the loop's result, 3. The 2,000-step checker plus "cat present" and "dog present" outputs: 0.721, kept as a push, 5 (folder matched by name, inferred). All stated.
- **Verdict:** support on the loop's mark: "It closed on the 0.756 row" (stated).
- **Why it worked:** Adding 350 SDXL joint-prompt drawings moved the reading most, from 0.639 to 0.737 (stated).
- **Kind of failure:** none, it passed its bar. Limits: test design, since the readings carry about ±0.05 on 220 pictures and seed counts are draft calls on 8 seeds, "most at tile size" (stated). could not measure it: "A higher reading does not promise a better push", since the 0.755 and 0.756 checkers gave about 7 and 3 of 8 (stated).
- **Idea still alive?** alive, for a readout checker near 0.75 on held-out finished pictures. It does not license choosing a push by the checker's accuracy.
- **Rules out:** Do not choose a checker by balanced accuracy alone; count its push's two-body pictures. Do not add COCO photographs to this checker's training (0.600, dropped).
- **Leaves open:** Which checker gives the best push, since accuracy and push quality do not track.
- **Evidence:** [the checker finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-checker-read-a-clean-cat-and-dog.md), [the checker loop](../../../../../2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md). The render sheets are held off the page by the pair ledger.

### CLIP checkers on finished pictures against half-drawn guesses

- **Asked:** Does a checker that reads finished pictures well also read the sampler's early guesses?
- **Paper:** 2
- **Tried:** DINOv2 ViT-B/14, CLIP ViT-L/14 and CLIP ViT-bigG/14 were fine-tuned on 1,603 finished pictures and read on the 220 held-out ones. CLIP was then trained on 4,966 half-drawn guesses. A readout checker on noised finished pictures was read for comparison.
- **Bar, written first:** none recorded.
- **Result:** Finished pictures: DINOv2 0.846 and 0.847, CLIP ViT-L/14 0.865 to 0.900, CLIP ViT-bigG/14 0.873 and 0.909. Two seeds of CLIP ViT-L/14 differ by 0.035, "as large as any change tried" (stated). By sampler step 0, 5, 10, 15, 20 and 30, then finished: CLIP trained on finished pictures read 0.50, 0.50, 0.51, 0.58, 0.62, 0.81, 0.90. After training on guesses it read 0.51, 0.54, 0.72, 0.82, 0.84, 0.87, 0.90. The readout checker on noised finished pictures read 0.63, 0.74, 0.78 and 0.79 at steps 0 to 15.
- **Verdict:** none recorded: the page gives the readings per step.
- **Why it failed:** A checker trained on finished pictures reads the first 10 steps' guesses at about chance (the readings are stated; the cause is inferred).
- **Kind of failure:** wrong data: trained on finished pictures, read on early states (stated readings). test design: two seeds differ by 0.035, so small differences are noise (stated).
- **Idea still alive?** retry worth it. The fix is to train on the sampler's guesses, which on this page lifted step 10 from 0.51 to 0.72.
- **Rules out:** Do not use a checker trained only on finished pictures to steer early steps. Do not read gaps under 0.035 between CLIP fine-tunes as real.
- **Leaves open:** Whether the guess-trained CLIP steers. Its pushes are in the decoded-picture entry below.
- **Evidence:** [the checker finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-checker-read-a-clean-cat-and-dog.md), [the checker loop](../../../../../2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md).

### A readout checker trained on the sampler's own states (plan 25)

- **Asked:** Does training the readout checker on the product sampler's own early states teach it the early steps, and does its push then make more clean pairs?
- **Paper:** 2
- **Tried:** The checker was trained on 1,972 labelled runs' own states, with labels from the 0.900 CLIP checker. It was scored on 195 held-out joint-prompt runs (105 clean, 90 rejects). A version with 1,000 more joint runs was also trained.
- **Bar, written first:** Q1: mean over steps 0 to 10 of at least 0.75, and at least 0.05 above the earlier checker. Q2: at least 2 more "two clear" seeds of 9 to 16 than plain product and the random push. Q3: at least +0.02 from 1,000 more joint runs (stated, plan 25's review).
- **Result:** Q1: mean 0.730 against 0.677 before (+0.053). Per step 0, 2, 4, 6, 8 and 10, it read 0.591, 0.647, 0.732, 0.774, 0.803 and 0.835, against 0.510, 0.548, 0.644, 0.747, 0.794 and 0.817 before. Q2: no seeds 9 to 16 sheet for this checker. Q3: 0.724 against 0.730. On the 128-seed page, the push from a checker trained on the sampler's own states (`result-sheet-catdog-big-traj`) gave 14 faithful pairs of 128 against 25 for the starting checker, and was dropped (verified).
- **Verdict:** Q1 in between (inconclusive). Q2 none recorded. Q3 null. Its push was dropped on 128 seeds.
- **Why it failed:** not recorded. The page does not say why its push did worse.
- **Kind of failure:** could not measure it: the labels come from a CLIP checker reading 0.900, not ground truth (stated). test design: Q2 was never drawn (verified).
- **Idea still alive?** unclear. The reading improved by the required 0.05 but missed 0.75, and the one push test is a single run where an unchanged rerun kept 25 faithful pairs with only 15 seeds in common. A seeds 9 to 16 sheet against the bar, or a fresh-seed 128 sheet, would tell.
- **Rules out:** Do not compare a checker's reading across the two tests without saying which one is used. Noising finished latents and reading saved sampler states give different numbers.
- **Leaves open:** The sources disagree on the 2,000-step checker's reading at step 0: 0.63 in the checker loop against 0.510 in plan 25's review. The code shows two different tests (noised finished latents against saved sampler states); that this causes the gap is inferred.
- **Evidence:** [the checker finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-checker-read-a-clean-cat-and-dog.md), [plan 25's review](../../../../../2-classifier-guidance/plans/review/25-the-readout-checker-on-poes-own-states.md), [the 128-seed comparison](../../../../../2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md). (plan 25)

### A push that pulls the cat and dog attention maps apart

- **Asked:** Since the product's "cat" and "dog" attention maps overlap more than the joint prompt's, does pushing them apart separate the animals?
- **Paper:** 2
- **Tried:** A push pulled apart the top 10% of the "cat" and "dog" attention maps on steps 0 to 19, on seeds 9 to 16 (`result-sheet-layout`).
- **Bar, written first:** none recorded.
- **Result:** The motivating measure was the overlap of the top 10% of attention (intersection over union, 1 means the same places). The median over 8 seeds at each of steps 2 to 40 runs 0.506 to 0.814 for the product and 0.225 to 0.394 for the joint prompt (verified, `attention/overlap.json`). The push gave 0 of 8 with two animals, and strong pushes zoom into one face (draft calls).
- **Verdict:** none recorded.
- **Why it failed:** not recorded.
- **Kind of failure:** test design: 8 seeds, draft calls and no bar (verified). The page gives no push size, so a weak or misdirected push cannot be told from a wrong idea.
- **Idea still alive?** unclear. A bar written first, plus the push's size as a share of the prediction, would tell a failed build from a failed idea.
- **Rules out:** Record a push's size and a bar before reading an attention push.
- **Leaves open:** The sources disagree on the joint prompt's median overlap: the walk says 0.22 to 0.44 over steps 2 to 40, while `overlap.json` gives 0.225 to 0.394. The walk's 0.44 is unexplained.
- **Evidence:** [the early-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md), [the attention overlap per step](../../../../../2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/03-does-an-early-push-separate-them/attention.png).

### CLIP checker pushes through the decoded picture

- **Asked:** Does a fine-tuned CLIP checker, pushing through the decoded guess, separate the cat and dog?
- **Paper:** 2
- **Tried:** Four sheets. Fine-tuned CLIP on all steps (`clip-push-sheet-from0`). The same on steps 10 to 49 (`from10`). CLIP trained on half-drawn guesses, steps 10 to 49 (`halfdrawn`). CLIP with three backward steps on steps 0 to 15, seeds 3301 to 3308 (`result-sheet-catdog-backward`). The first three used seeds 9 to 16 with draft calls.
- **Bar, written first:** none recorded for the first three. For the backward-step version, a mark of 7 clean pairs of 32.
- **Result:** Two bodies on 1 of 8, 0 of 8 and 0 of 8. With backward steps the layout stays plain product on 8 of 8, while the early readout push separates the same 8.
- **Verdict:** none recorded for the first three. The backward-step version was dropped, with its mark of 7 clean pairs of 32 "out of reach".
- **Why it failed:** "A push through SDXL's own features moved the layout, and pushes through the decoded picture changed faces and fur" (inferred, stated by the page).
- **Kind of failure:** the idea: a push through the decoded picture acts on faces and fur, not layout (inferred). test design: 8 seeds, draft calls and no bar on three of four (verified).
- **Idea still alive?** unclear, leaning against. Four variants all failed to move layout, but the tests used 8 seeds with no bar.
- **Rules out:** Do not push through the decoded picture to change layout. Push through SDXL's own features.
- **Leaves open:** Whether a decoded-picture push helps once the layout is set.
- **Evidence:** [the early-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md), [the checker loop](../../../../../2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md).

### Push, then pick: four pushed tries per seed with CLIP choosing one

- **Asked:** Does making four pushed tries per seed and letting CLIP pick one give clean cat and dog pictures?
- **Paper:** 2
- **Tried:** Four pushed tries per seed on seeds 9 to 16, with CLIP picking one (`push-then-pick`), compared with the best of four plain tries.
- **Bar, written first:** none recorded.
- **Result:** CLIP's picks were called clean on 3 of 8 seeds. By eye those 3 were 2 near-clean and 1 two dogs. The best of four plain tries gave 0.
- **Verdict:** none recorded.
- **Why it failed:** not recorded.
- **Kind of failure:** could not measure it: CLIP passed a two-dog picture as clean, 1 of its 3 picks (stated). test design: 8 seeds, no bar (verified).
- **Idea still alive?** unclear. A picker checked against full-size eye calls on more seeds would tell.
- **Rules out:** Do not trust a CLIP pick as a clean pair without an eye check.
- **Leaves open:** Whether selection adds to the early push.
- **Evidence:** [the early-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md).

### How hard and for how long the readout push acts (cap and window settings)

- **Asked:** Which cap on the push's size and which window of sampler steps keep the separation and the photograph's natural colour?
- **Paper:** 2
- **Tried:** The readout checker push at strengths 30 and 100 on seeds 9 to 16. Variants: cap 0.3 or 0.5 on all 50 steps; cap 0.3 on steps 0 to 15 only; cap 0.1 to step 9 then 0.3 to step 49; cap 0.1 with each of steps 0 to 15 taken 3 times; the early push with 3 re-noised repeats per step. Then cap 0.2 against cap 0.3 on seeds 3301 to 3332. The cap rule is: new push = push times min(1, cap times size of prediction over size of push).
- **Bar, written first:** "No written pass mark exists for these rows except where shown." The cap 0.2 row was compared against the cap 0.3 counts.
- **Result:** Draft calls at tile size (stated): cap 0.3 all steps, a separate cat and dog on most seeds with colour damaged; cap 0.5, like 0.3 at strength 30 with more damage and broken at 100; cap 0.3 on steps 0 to 15, the same seeds separate with colour closer to a photograph and near-clean on 4 of 8 (kept); cap 0.1 then 0.3 late, near-clean on 3 of 8 on different seeds; steps taken 3 times, two animals on about 4 to 5 of 8; re-noised repeats, clean or near-clean on about 3 of 8 with more damage. Cap 0.2 on 32 seeds by eye at 640 px: 22 of 32 separate and 10 clean or near-clean (5 clean), against 28 and 14 at cap 0.3.
- **Verdict:** none recorded: settings were chosen from draft calls, and cap 0.3 on steps 0 to 15 was kept.
- **Why it worked:** Pushing only on steps 0 to 15 spends the push on layout, and "the colour damage from pushing on all 50 steps mostly went away" (inferred, stated).
- **Kind of failure:** test design: tuning on 8 seeds from draft calls at tile size, and "the 4 of 8 for the early push appears to pool strengths 30 and 100" while plan 29 used strength 30 alone (inferred).
- **Idea still alive?** alive, as the chosen setting (cap 0.3, steps 0 to 15) that plan 29 then tested. The rows themselves are tuning, not tests.
- **Rules out:** Do not push on all 50 steps at cap 0.3 or more; colour breaks. Do not read a tuning row on 8 seeds as a result.
- **Leaves open:** The draft calls on the seeds 9 to 16 sheets are unconfirmed.
- **Evidence:** [the early-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md), [the checker loop's cap section](../../../../../2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md).

### The early readout push on 32 new seeds (plan 29)

- **Asked:** Does the early push (a readout checker's gradient on sampler steps 0 to 15 of 50, strength 30, capped at 0.3 of the product's prediction, plain product on steps 16 to 49) separate the cat and the dog on seeds never used before?
- **Paper:** 2
- **Tried:** The SDXL cat and dog product on seeds 3301 to 3332, never used for training, evaluation or tuning when this ran. Plain product, a random push of the same size and the early push were compared, called by eye from full-size tiles (`result-sheet-catdog-general`).
- **Bar, written first:** Two separate animals on at least 19 of 32, and clean or near-clean cat and dog on at least 10 of 32, with each control at most 2 (stated, plan 29's review, written 2026-11-18).
- **Result:** Two separate animals: plain product 1, random push 1, early push 28 of 32. Clean or near-clean: 0, 0, and 14 (9 clean). Of the 18 seeds short of clean or near-clean, 11 show two bodies with a borrowed face or hidden head, 3 two of one kind and 4 not separate. The early push's counts were recounted from `scripts/rung2/calls_plan29.json` (verified). In plan 21 the random push's size over the real push's size was 1.0000 at every step of 24 renders (stated).
- **Verdict:** support (both pass marks and both control limits held).
- **Why it worked:** The checker's direction does it, since a random push of the same size separates 1 of 32 (verified). Pushing only early sets the layout (inferred).
- **Kind of failure:** none, it passed its bar. Limits: could not measure it, because viewing size moves the calls (17 clean or near-clean at half size, 14 at full size, stated) and pushed renders do not repeat pixel for pixel (the same seed and settings gave three different pictures in three runs, stated). test design: one pair.
- **Idea still alive?** alive, for separating a cat and a dog on new seeds. It does not settle species: 14 of 32 clean or near-clean, and the usual miss is a borrowed face or two of one kind. Other pairs are a separate entry.
- **Rules out:** This settles that the checker's direction, not the push's size, separates the bodies. Compare pushed runs by counts, never seed by seed.
- **Leaves open:** Which body becomes which animal. Whether it holds on other pairs.
- **Evidence:** [the early-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-an-early-push-separate-the-cat-and-dog.md), [plan 29](../../../../../2-classifier-guidance/plans/29-does-the-early-push-hold-on-new-seeds.md), [plan 29's review that set the bar](../../../../../2-classifier-guidance/plans/review/29-does-the-early-push-hold-on-new-seeds.md). The pushed views were not cleared against the pair ledger. (plan 29)

### Two pushes: bodies early, species late (plan 26)

- **Asked:** After the readout push makes two bodies, does a CLIP push for "clear cat" and "clear dog" make each body the right animal?
- **Paper:** 2
- **Tried:** A CLIP species checker for "clear cat" and "clear dog". Then the readout push on steps 0 to 9 and the species push on steps 10 to 49 (`result-sheet-catdog-bodies`, `result-sheet-catdog-both`), on seeds 9 to 16 with draft calls at tile size.
- **Bar, written first:** The species checker at least 0.80 for both at step 15, null below 0.70. The two-push sheet at least 2 more "two clear" seeds than the first push alone, with the random push adding none.
- **Result:** The species checker read cat 0.711 and dog 0.847. The two pushes gave "almost the same pictures; one cat-dog face on most seeds at strengths 30 and 100".
- **Verdict:** inconclusive for the species checker, null for the two-push sheet.
- **Why it failed:** not recorded beyond the cat reading of 0.711, under the 0.80 bar.
- **Kind of failure:** could not measure it: the cat checker read 0.711, under the 0.80 the bar asked (stated). test design: 8 seeds and tile-size calls; the page lists calling plan 26 from full-size tiles as still open (stated).
- **Idea still alive?** retry worth it. The fixes are a species checker that reaches 0.80 on cat, and full-size calls.
- **Rules out:** Check that a species checker reaches its bar before reading its push.
- **Leaves open:** Whether a late species push works with a better checker.
- **Evidence:** [the second-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-second-push-name-each-animal.md), [plan 26's review](../../../../../2-classifier-guidance/plans/review/26-two-pushes-bodies-early-species-late.md), [plan 26](../../../../../2-classifier-guidance/plans/26-two-pushes-bodies-early-species-late.md). (plan 26)

### Subtracting SDXL's "a hybrid animal" direction, weighted by the checker (plan 27)

- **Asked:** Does subtracting SDXL's direction for "a hybrid animal", weighted by how likely the checker says the picture is going wrong, give a separate cat and dog?
- **Paper:** 2
- **Tried:** A "not hybrid" term at three strengths, weighted by k = p / (1 - p), where p is the checker's chance of going wrong, limited to 0.95 so k cannot pass 19. A control fixed the weight at 19. Seeds 9 to 16, calls at tile size (`not-hybrid`).
- **Bar, written first:** A separate cat and dog on at least as many seeds as the cap 0.3 push, with fewer damaged, and on at least 2 more than plain product. The checker's weight must give at least 2 more seeds than the fixed weight.
- **Result:** A separate cat and dog on no seed; the style turns to line drawings and patterns. The fixed-weight control drew the same pictures, with the weight at its limit. Across the 24 checker-weighted runs, the mean k over steps 0 to 9 is 15.9 to 19.0, and at strength 0.03 it sits at 19.0 on 5 of the 8 seeds.
- **Verdict:** null (both).
- **Why it failed:** The checker's weight sat at its ceiling of 19 on most steps, so the weighted and fixed runs barely differ (verified, `record.json`). That the term changes style instead of layout is shown in the pictures (stated).
- **Kind of failure:** done wrong: the weight was pinned at 19, so a weight that varies was never tested (verified). the idea: the unweighted term gave no separate pair at three strengths (stated). test design: 8 seeds, tile-size calls (stated).
- **Idea still alive?** unclear. The weighting half was never tested, since k sat at 19. The term itself gave nothing on 8 seeds. A run where k actually varies, called at full size, would tell.
- **Rules out:** Check that an adaptive weight actually moves before comparing it with a fixed one.
- **Leaves open:** The sources disagree on whether the weighted and fixed columns are the same pictures. The review says they match. The files are byte-identical on 5 of 8 seeds (10, 11, 14, 15, 16) and differ on seeds 9, 12 and 13, where the weight fell below 19 on 1, 8 and 3 of steps 0 to 9. How alike those three look was not measured.
- **Evidence:** [the second-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-second-push-name-each-animal.md), [plan 27's review](../../../../../2-classifier-guidance/plans/review/27-cat-and-dog-and-not-hybrid.md), [plan 27](../../../../../2-classifier-guidance/plans/27-cat-and-dog-and-not-hybrid.md). (plan 27)

### Each detected animal pushed toward its own species (plan 28)

- **Asked:** After step 16, does pushing each detected animal's box toward its own species make the right cat and the right dog?
- **Paper:** 2
- **Tried:** After the early push, a detector found boxes at step 16, and each box was named zero-shot (a text-and-image model asked "a cat" or "a dog" on the crop) and pushed toward its species (`result-sheet-catdog-boxes`). A second version pushed only boxes named with confidence of at least 0.8 (`-confident`). Seeds 9 to 16.
- **Bar, written first:** Clean or near-clean on at least 6 of 8, full size, with the random push adding none. The detector finds two boxes at step 16 on at least 6 of 16 pictures. For the confident version, no wrong-species turn and at least 5 of 8.
- **Result:** 4 of 8, the same count as the early push alone. The detector found two boxes on 16 of 16. The confident version gave 4 of 8, with one wrong-species turn remaining. On step 16's blurry guess the dog box often read as a cat (chance of dog 0.08 to 0.21) (stated).
- **Verdict:** null for the push (both versions). Support for the detector check.
- **Why it failed:** Naming a box on step 16's blurry guess often called the dog a cat (inferred, stated).
- **Kind of failure:** could not measure it: the zero-shot namer read the dog box as a cat (chance of dog 0.08 to 0.21, stated). test design: 8 seeds (verified). For the detector check: none, it passed its bar.
- **Idea still alive?** retry worth it. The walk's next idea, giving each detected body its own prompt after the layout settles, has no plan yet. The detector check is alive: two boxes exist by step 16 on 16 of 16, on one pair.
- **Rules out:** Do not name boxes from step 16's blurry guess with a zero-shot namer.
- **Leaves open:** Whether a body-level push works with reliable naming.
- **Evidence:** [the second-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-second-push-name-each-animal.md), [plan 28's review](../../../../../2-classifier-guidance/plans/review/28-each-animal-toward-its-own-species.md), [plan 28](../../../../../2-classifier-guidance/plans/28-each-animal-toward-its-own-species.md). (plan 28)

### Adding "cat present" and "dog present" pushes or outputs to the readout push

- **Asked:** Does adding a presence checker's push, or presence outputs on the readout checker, raise faithful cat and dog pairs?
- **Paper:** 2
- **Tried:** Three tests. (1) A "cat present, dog present" checker's push added on steps 0 to 15, on seeds 3301 to 3332 (`result-sheet-catdog-presence-added`). (2) Presence outputs across every pair at cap 0.4, on 128 seeds (`result-sheet-catdog-big-multipair-presence`). (3) "First animal present" and "second animal present" outputs, cap 0.4, on lion and meerkat (`result-sheet-lion-meerkat-multipair-presence`). All were scored by the automatic scorer. Its "faithful pair" is its strictest verdict: exactly two bodies, one clear cat and one clear dog, nothing borrowed, an intact photograph.
- **Bar, written first:** (1) at least 7 of 32 faithful pairs, with two bodies at 29 or more. (2) The loop's keep rule: faithful pairs up by at least 4 of 128 and two bodies not falling (stated). (3) The loop's stop condition of 6 or more faithful pairs of 16.
- **Result:** (1) 27 two bodies, 4 faithful pairs and 9 damaged of 32, with 21 not intact. (2) 119 two bodies, 24 faithful, 20 damaged and 94 not intact of 128, against 35 faithful for the kept push. (3) 11 two bodies and 0 faithful of 16; "pictures break into posters and collages".
- **Verdict:** null: all three are recorded as dropped or below the stop condition.
- **Why it failed:** not recorded.
- **Kind of failure:** test design: on 32 seeds the unchanged early push moved from 5 to 7 faithful pairs between two draws (verified), so 4 is near that noise. could not measure it: the scorer varies on byte-identical tiles (verified on the other-pairs page).
- **Idea still alive?** unclear, leaning against. Three tests on two pairs were all dropped, but the 32-seed test sits inside the scorer's draw-to-draw noise. A fresh-seed 128 sheet would tell.
- **Rules out:** Do not add presence outputs or a presence push to the early readout push expecting more faithful pairs; it was dropped three times.
- **Leaves open:** Whether presence helps at a different cap or window. Plan 31's version (presence-only readout added on every step) is a separate entry.
- **Evidence:** [the second-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-second-push-name-each-animal.md), [the 128-seed comparison](../../../../../2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md), [the other-pairs finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-the-early-push-carry-to-other-pairs.md), [the checker loop](../../../../../2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md).

### Bodies and species pushes on every step (plan 31)

- **Asked:** Does adding a second push on all 50 steps, as on MNIST where it lifted two clear, right digits from 215 to 285 of 320, raise clean cat and dog pairs past the early push?
- **Paper:** 2
- **Tried:** The readout push plus a second push on all 50 steps, summed and capped at 0.3. A setting was picked on seeds 9 to 16, by eye at 640 px: readout alone gave 8 separate, 3 clean or near-clean and 0 clean of 8; plus CLIP species push at strength 10, 1, 0, 0; at strength 30, 4, 0, 0; plus a presence-only readout checker (picked), 8, 6, 3. The picked setting was then drawn once on seeds 3301 to 3332 (`result-sheet-catdog-every-step-general`).
- **Bar, written first:** Support at 20 or more clean or near-clean with at least 26 separate. No change at 10 to 18. Worse at fewer than 24 separate or fewer than 10 clean or near-clean (stated, plan 31's review).
- **Result:** On 32 seeds: separate 29 (plain product 1, random push 2, early push 28). Clean or near-clean 15 (8 clean), against 14 (9 clean) for the early push (verified, recounted from `calls.csv`). The scorer gave 2 faithful pairs, 14 of 32 with the verdict damaged, and 28 of 32 with its intact-photograph answer false.
- **Verdict:** null ("no change").
- **Why it failed:** "Its 3 of 8 against 6 of 8 on seeds 9 to 16 did not survive on the 32 seeds" (stated). Under one cap, a strong second push takes room from the push that keeps the bodies apart (inferred).
- **Kind of failure:** test design: "The comparison in plan 31 changes two things at once" (window of 50 steps against 16, and the added push), and the 15 was called at 600 px while the 14 was called at full size (stated). could not measure it: eye 15 clean or near-clean against scorer 2 faithful, because a faithful pair needs an intact photograph while the eye admits one small borrowed feature (stated).
- **Idea still alive?** unclear. The fix that would tell is to hold the window at steps 0 to 15, add only the second push, and call both arms at full size.
- **Rules out:** Do not change the window and the added push in one comparison. Do not compare eye counts made at different viewing sizes.
- **Leaves open:** Plan 31's task 4.1 (confirm or veto the eye calls) is open.
- **Evidence:** [the second-push finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-second-push-name-each-animal.md), [plan 31's review](../../../../../2-classifier-guidance/plans/review/31-bodies-and-species-pushes-on-every-step.md), [plan 31](../../../../../2-classifier-guidance/plans/31-bodies-and-species-pushes-on-every-step.md). (plan 31)

### The cat and dog checker's early push on two new pairs (plan 30, first question)

- **Asked:** Does the early push, with a checker trained only on cats and dogs, separate other pairs?
- **Paper:** 2
- **Tried:** The early push on lion and meerkat and on typewriter and cactus, seeds 3301 to 3316, against plain product and a random push of the same size. Called by eye from full-size tiles. Both pairs are held out in `training_set_multipair.json` (verified).
- **Bar, written first:** Both named things separate on at least 10 of 16 seeds, with plain product and the random push each at most 2 (stated, plan 30's review, 2026-11-18).
- **Result:** Lion and meerkat: plain 1, random 1, early push 6 of 16, clean 3 of 16. The push drew two animals on 10 of 16, and 4 of those were two meerkats. Typewriter and cactus: plain 1, random 1, early push 14 of 16, clean 6 of 16.
- **Verdict:** support for typewriter and cactus. Inconclusive ("in between") for lion and meerkat.
- **Why it worked:** The push makes two bodies (verified, counts). Why it partly failed: "the push makes two bodies and leaves which animal each becomes to the product, where the meerkat expert wins on most seeds" (inferred, stated by the loop).
- **Kind of failure:** none, it passed its bar for typewriter and cactus. For lion and meerkat, the idea: two of one kind on 4 of 10 two-animal seeds (stated). Limits on both: test design, with 16 seeds per pair and two pairs (verified). could not measure it: on plain typewriter and cactus the scorer counted 6 of 16 with two bodies against 1 of 16 by eye, because it "counts loose cacti as extra bodies" (stated).
- **Idea still alive?** alive, for a cat-and-dog-trained checker separating one object pair. "Two pairs cannot show how far the push travels." For animal pairs, which animal each body becomes is unsolved.
- **Rules out:** Do not count two bodies as the pair; count the right two things. Check the scorer's body count against the eye on object pairs.
- **Leaves open:** How far the push transfers beyond two pairs.
- **Evidence:** [the other-pairs finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-the-early-push-carry-to-other-pairs.md), [plan 30's review](../../../../../2-classifier-guidance/plans/review/30-does-the-recipe-work-on-other-pairs.md), [plan 30](../../../../../2-classifier-guidance/plans/30-does-the-recipe-work-on-other-pairs.md). The views sheets were not cleared against the pair ledger. (plan 30)

### Improvement rounds on the cat and dog push over 128 seeds

- **Asked:** Which change to the early push raises faithful cat and dog pairs on 128 seeds?
- **Paper:** 2
- **Tried:** Rounds on seeds 6001 to 6128 (some on 32 seeds), each scored by the automatic scorer, a Claude API labeller in `score_sheet.py`. The starting point was the cat and dog checker at cap 0.3 (`result-sheet-catdog-big`).
- **Bar, written first:** A change was kept when faithful pairs rose by at least 4 of 128 and two bodies did not fall. Rounds on 32 seeds carried their own marks (stated, the loop's close).
- **Result:** All recounted from `scores.jsonl` (verified). Each row gives two bodies, faithful, damaged verdict and not intact. Starting point: 114, 25, 13, 80. Unchanged rerun: 116, 25, 11, 83. Checker trained on the sampler's own states: 101, 14, 7, 70, dropped. DDIM eta 1.0 (32 seeds): 28, 2, 7, 27, dropped. Presence outputs removed (32 seeds): 26, 3, 3, 24, outputs kept. Early push rerun unchanged (32 seeds): 30, 7, 3, 19. Presence checker push added (32 seeds): 27, 4, 9, 21, dropped. Window steps 0 to 24: 122, 24, 12, 89, dropped. Checker on 243 pairs: 116, 29, 10, 60, kept. 78 more pairs added: 115, 23, 19, 81, dropped. 4,000 training steps: 117, 13, 23, 89, dropped. Cap 0.4: 121, 35, 11, 65, kept. Presence across every pair at cap 0.4: 119, 24, 20, 94, dropped. A rank-16 adapter inside the checker stopped at training step 500 at chance (0.522), with no sheet (stated). The random push gave 0 or 1 two-body tiles and 0 faithful on every 128-seed sheet that scored it (verified). Drawing the same 128 seeds twice kept faithful at 25 with only 15 seeds faithful both times, and the same verdict on 81 of 128 seeds (verified).
- **Verdict:** kept: the 243-pair checker and cap 0.4. Dropped: every other change listed. The cap 0.4 round has two readings. On the verdict damaged (10 to 11, a rise of 1 against a limit of 4) the limit is met. On `photo_intact` false (60 to 65, a rise of 5 against 4) it is missed. The round stays recorded as kept, and the owner chooses the reading.
- **Why it worked:** a wider training set and a larger cap (stated).
- **Kind of failure:** test design: rounds were kept or dropped on the same 128 seeds, so "Choosing among rounds on them is a form of tuning" (inferred). could not measure it: the scorer gives the verdict damaged only when hybrid, single and wrong pair do not apply first, so a broken hybrid counts as a hybrid; the kept push has damaged 11 but `photo_intact` false on 65 of 128 (verified).
- **Idea still alive?** alive, for the kept setting (243-pair checker, cap 0.4) as the best of these rounds. No fresh seeds have tested it, so the 35 "may flatter the kept push" (inferred).
- **Rules out:** Do not keep or drop rounds on seeds you later report as held out. Do not compare single seeds between pushed runs; only totals hold. Do not read the scorer's damaged verdict as all damage.
- **Leaves open:** The sources disagree on how often the cap cut the push. The loop says "on most steps"; `capped_per_step` gives a mean of 35.5% of steps 0 to 15 over 128 seeds for the multi-pair checker at cap 0.3 (54.0% for the starting checker, 18.9% at cap 0.4). They also disagree on the pair count: "over 200 pairs" against "243 pairs". `training_set_multipair.json` holds 242 pool pairs plus cat and dog, with 200 of 242 having a clean picture. On damaged pictures under the early push on seeds 3301 to 3332, the loop says 3, `plan29.jsonl` says 4 and the re-render says 3.
- **Evidence:** [the 128-seed comparison](../../../../../2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md), [the checker loop's close](../../../../../2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md).

### One MNIST canvas model with a pair-aware checker push: how good the checker must be (A2b, first run)

- **Asked:** In one digit model that saw few two-digit canvases, does a pair-aware checker's push turn plain product's fused digit into two clear, right digits, on a seen pair and on held-out pairs, and how good must the checker be?
- **Paper:** 2
- **Tried:** A class-conditional pixel U-Net per two-digit share (0, 0.01, 0.1) on 32 by 64 canvases of real MNIST digits. Five pairs were held out (0 and 3, 1 and 9, 3 and 4, 3 and 6, 4 and 7). The checker reads the frozen U-Net's decoder features with the asked pair as an input, and checkers were weakened on purpose. Controls were a random push, a separate-label push and best-of-N. 64 seeds on 7 and 8, and 320 on the held-out pairs. Training took 486 ms per step on the laptop and 157 on a `bigbatch` RTX 3090.
- **Bar, written first:** On 7 and 8, at the share where plain product fuses: plain product at most 8 of 64, the pair-aware push at least 32 of 64 at some strength, the random and separate-label pushes within 6 of plain product, and best-of-N at matched compute at least 8 below the pair-aware push. On the five held-out pairs, at least 160 of 320 (verified, review file 2026-11-18).
- **Result:** On 7 and 8: plain product 1 of 64, the pair-aware push 47 of 64 at strength 10, the random push 1, best-of-2 1, the separate-label push 15 of 64. On the five held-out pairs: 84 of 320, eye counts at strength 3 of 3, 14, 25, 11 and 31 (verified). The judge agreed with the eye on 58 of 64 tiles against a bar of 58 (this 58 of 64 is a digit judge, not the ICLR detector). It then misread the model's thick digits, so counts are by eye. Six checkers sat between 0.51 and 0.59 accuracy, all below the 0.655 and 0.771 marks. Two checkers at the same accuracy composed 47 and 27. Raising the cap from 0.1 to 1.0 moved 47 to 49. The 0.1 cap clips most steps at strength 30.
- **Verdict:** "Works on seen pairs only": support on 7 and 8, null on the held-out pairs (84 of 320 against 160). The page names two departures from the bar: the separate-label push composes some (15 of 64), and the 0.1 cap clips most steps at strength 30.
- **Why it failed:** On held-out pairs "the push still makes two digits, but often the wrong second one" (verified by eye counts). The checker-accuracy question went unanswered because "No digit checker passed 0.59 held-out accuracy" (verified).
- **Kind of failure:** For 7 and 8: none, it passed its bar. Limit: the separate-label control came in above its limit, 15 against within 6 of plain product's 1 (verified). For held-out pairs: wrong data, since the pairs were never seen together and the second digit came out wrong (verified). could not measure it: the judge misread the thick digits (stated), and no checker reached the accuracy marks the question needed (verified).
- **Idea still alive?** alive, for a pair-aware push on a pair the model has seen. For held-out pairs, retry worth it: the follow-on (next entry) reached 217 of 320.
- **Rules out:** Do not pick a push by checker accuracy alone; two checkers at the same accuracy gave 47 and 27. Check the judge against the eye on the model's own digits before counting by judge.
- **Leaves open:** Whether rung 2's cat and dog checker (0.655 held out, CLIP's ceiling 0.771) could steer, since no digit checker reached those marks.
- **Evidence:** [the one-model MNIST finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/24-mnist-one-model.md), [the review that set the bar](../../../../../2-classifier-guidance/plans/review/a2b-mnist-digits-on-a-canvas.md), [compose count against checker accuracy](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/compose-against-checker-accuracy.png), [the results card](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/README.md). (A2b, route step 7b)

### A follow-on MNIST checker that sees the pair at every layer, plus the separate-label push (A2b follow-on)

- **Asked:** Can a stronger checker reach the five held-out digit pairs that the first run missed?
- **Paper:** 2
- **Tried:** A checker that sees the asked pair at every layer, retrained twice on its own push's pictures, with a partly random sampler, on the five held-out pairs (320 seeds). Then the separate-label push was added and the result drawn on fresh seeds.
- **Bar, written first:** none restated for the follow-on. The original held-out bar was 160 of 320.
- **Result:** 217 of 320 with two clear, right digits (3 and 4 and 3 and 6 at 62 of 64; 0 and 3 at 25), against 0 for the same sampler without the push. Adding the separate-label push gave 285 of 320 (89%) on fresh seeds, where the pair-aware push alone gave 215. Blind eye counts per pair were 44, 53, 64, 64 and 60 of 64 (verified).
- **Verdict:** none recorded for the follow-on: the page says "A follow-on run reached the new pairs" and gives the counts.
- **Why it worked:** not stated beyond the changes listed.
- **Kind of failure:** none recorded against a bar. Limit: test design, since the checker was "retrained twice on its own push's pictures", and the page does not say on which pairs or seeds (stated).
- **Idea still alive?** alive, for pair-aware plus separate-label pushes composing unseen digit pairs in one model. Plan 31 tried the same sum on SDXL cat and dog with no change, which is a separate entry.
- **Rules out:** Record which pairs and seeds a self-retrained checker saw before counting its held-out score.
- **Leaves open:** Whether the gain holds without retraining on its own push's pictures.
- **Evidence:** [the one-model MNIST finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/24-mnist-one-model.md), [the fresh-seed sheet with both pushes](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/01-one-model-that-saw-a-few-pairs/loop2/final-seeds64.png). (A2b, route step 7b)

### Checkers trained across many pairs, read on lion and meerkat

- **Asked:** Does a readout checker trained across many pairs make the early push produce a faithful lion and meerkat?
- **Paper:** 2
- **Tried:** Five sheets on lion and meerkat (held out of training), 16 seeds each, scored by the automatic scorer. Settings: cat and dog plus the labelled benchmark pictures at cap 0.3; the same at cap 0.4; 78 more pairs' joint-prompt pictures added at cap 0.3 (folder matched by the head in its `result.json`, inferred); trained twice as long at cap 0.3; presence outputs at cap 0.4.
- **Bar, written first:** the loop's stop condition of 6 or more faithful pairs of 16 (stated).
- **Result:** Two bodies 12, 15, 11, 11 and 11 of 16. Faithful pairs 0 of 16 on every sheet (verified, recounted from `scores.jsonl`). By eye (stated): a lion beside a meerkat on about 4 seeds; about 4 seeds with the rest mostly two meerkats; 1 seed; none recorded; and pictures breaking into posters and collages.
- **Verdict:** null (stop condition not met on any sheet).
- **Why it failed:** The push makes two bodies and leaves which animal each becomes to the product, where the meerkat expert wins (inferred, stated by the loop).
- **Kind of failure:** the idea: a bodies push does not decide species (inferred). could not measure it: the eye saw a lion beside a meerkat on about 4 seeds where the scorer gave 0, since the scorer's faithful pair is stricter than the eye's near-clean (stated). On byte-identical plain tiles the scorer counted 3, 2 and 3 across three runs (verified).
- **Idea still alive?** unclear. The record cannot tell a species failure from a strict, noisy scorer. Full-size eye calls against a bar written first would. The walk's next idea (a prompt per detected body) has no plan.
- **Rules out:** Do not expect a checker trained on more pairs to fix which animal each body becomes. Do not read a one-seed scorer difference as real.
- **Leaves open:** Plan 30's second question (next entry).
- **Evidence:** [the other-pairs finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-the-early-push-carry-to-other-pairs.md), [the checker loop's close](../../../../../2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md).

### The kept push against plain product, the joint prompt and CO3 on 128 seeds

- **Asked:** How close does the kept cat and dog push come to the joint prompt "a cat and a dog" and to CO3?
- **Paper:** 2
- **Tried:** Four methods on seeds 6001 to 6128, one picture per seed. Plain product; the kept push (the early push with a checker trained on cat and dog plus labelled pictures of other pairs, strength 30, steps 0 to 15, cap 0.4); the joint prompt; and CO3, another composition method that this page does not define. Each was scored by the automatic scorer and by eye at 330 to 420 px.
- **Bar, written first:** none recorded for this comparison.
- **Result:** Each method gives faithful pairs, not-intact pictures, clean or near-clean by eye, and clean by eye, all of 128. Plain product: 0, 72, 0, 0 (eye counts stated). Kept push: 35, 65, 54, 32. Joint prompt: 57, 42, 69, 64. CO3: 89, 9, 101, 97 (verified, recounted). The joint prompt also drew 14 cat and dog pictures as drawings, which the eye count leaves out. CO3's pictures are mostly a cat and a dog side by side on a plain studio backdrop, so they vary less (stated). On plan 29's 32 pushed tiles the eye and the scorer agreed on 23 of 32, and the scorer never passed a tile the eye rejected (stated).
- **Verdict:** none recorded: "No overall verdict was recorded when the improvement rounds closed on 2026-11-19."
- **Why it failed:** not recorded. The page reads the kept push as sitting between plain product and the joint prompt.
- **Kind of failure:** test design: the 35 is the best of several rounds on these same 128 seeds (inferred), and the eye calls were made at 330 to 420 px, where a smaller view has overcounted before (stated). Every count is one pair.
- **Idea still alive?** unclear. Drawing the kept push, the joint prompt and CO3 once on unused seeds and scoring them the same way would tell how close the push really comes.
- **Rules out:** Do not report the kept push's 35 as a held-out number.
- **Leaves open:** Whether the gap to the joint prompt (35 against 57 faithful pairs) is worth more rounds. Full-size eye calls for all three methods.
- **Evidence:** [the 128-seed comparison](../../../../../2-classifier-guidance/report/classifier-guided-poe/how-close-the-kept-push-comes-to-the-joint-prompt.md), [the checker loop](../../../../../2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md).

### Step 7b's pushes on two digit experts trained apart (A2d, plan 32)

- **Asked:** Do the step 7b pushes compose a 0 to 4 expert and a 5 to 9 expert trained apart into two clear, right digits?
- **Paper:** 1
- **Tried:** Two digit experts trained apart, on 32 by 64 canvases, 320 seeds over five pairs. Tested with each expert drawing its digit on either side, and with each expert always on its own fixed side. Pushes tried: both pushes, a random push, a one-half ink push, and four settings of strength, cap and randomness on two tuning pairs.
- **Bar, written first:** Each expert alone at least 58 of 64 per digit. The fixed-side control at least 256 of 320. Both pushes on the either-side experts at least 224 of 320 and not within 32 of plain product. No harm on the fixed-side experts (verified, review file 2026-11-19).
- **Result:** Experts alone: 9 of 10 digits at or above 58 of 64; the 5 to 9 expert's 8 (55) and, fixed-side, its 7 (57) miss. Either side: plain 0 of 320, both pushes 4, random 2, one-half ink 0, four settings 0. Fixed side: 139 of 320 with no push and 168 with both pushes. Sheets: asked 4 and 7, either side, 0 of 64 with 30 called one blob, and 0 of 64 with 33 one blob with both pushes. Asked 3 and 6, fixed sides, 30 of 64 with no push and 46 of 64 with both. All counts from the counts file (verified).
- **Verdict:** null ("They do not compose").
- **Why it failed:** "The canvases are single saturated blobs: the product is destroyed before a push acts, and a push can only reweight what the product still has." Blob counts are verified on the sheets; the mechanism is inferred.
- **Kind of failure:** the idea: pushing the plain product of experts trained apart cannot recover what the product destroyed (fair test: 320 seeds, bar written first, several settings; verified). Also a failed control: the fixed-side control, where "the argument says the experts compose exactly", reached 139 against 256, and one expert missed its own-digit check (55 and 57 against 58) (verified).
- **Idea still alive?** closed for pushes on the plain whole-canvas product of these experts. retry worth it for "a base the two experts share", which the page says was not tried. Plan 33's own-half sampling later reached 1,200 of 1,280.
- **Rules out:** Do not push the plain product of experts trained apart on one whole canvas and expect two digits. Check that the product still holds two-digit canvases before adding a push.
- **Leaves open:** Why the fixed-side control fell short of the argument. The sources disagree on what the plain product draws (recorded on plan 33's page): plan 32's review says "most hold one fat saturated white blob in one half", while the judge on the same run gives 99 of 320 one shape and 194 two shapes not both clear. Both give 0 clean pairs.
- **Evidence:** [the two-experts finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/32-mnist-two-experts.md), [plan 32's review that set the bars](../../../../../2-classifier-guidance/plans/review/32-two-digit-experts-that-never-met.md), [the results card](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/02-two-models-trained-apart/README.md), [the counts file](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/02-two-models-trained-apart/counts.json), [fixed sides asked for 3 and 6](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/02-two-models-trained-apart/sheets/fixed-none-floor0.5-3-6.png). (A2d, plan 32)

### Repairs to the plain product of two digit models trained apart (plan 33, step 1)

- **Asked:** Can the plain product of a 0 to 4 model and a 5 to 9 model draw clean pairs on one whole canvas with a repair?
- **Paper:** 1
- **Tried:** On the five tuning pairs, 320 canvases each: plain product; an average "no digit" background; a third model as background; both pushes; both pushes with extra steps; and each model kept on one half while seeing the whole canvas.
- **Bar, written first:** On the tuning pairs, 288 of 320 supports a composition, 160 is promising, under 32 is no effect (stated, plan 33's review).
- **Result:** Clean pairs of 320 (a clean pair is two clear, correct digits, one per half, each read at least 0.9 sure by the judge): 0, 0, 0, 1, 0 and 4 (verified). With both pushes on 4 and 7, 0 of 64 are clean and most canvases hold a 4 and a 7 fused into one shape (verified).
- **Verdict:** null (all under the no-effect line of 32 of 320).
- **Why it failed:** A canvas holding one digit from each model has almost no chance under either (inferred: "No run measured these chances, so the zero-chance reading is the plan's explanation of that count").
- **Kind of failure:** the idea: whole-canvas product repairs on the original models (fair: 320 canvases, bar written first; verified).
- **Idea still alive?** closed, for these repairs on the original models. The side road shows that retraining the models with clutter changes it (separate entry).
- **Rules out:** Do not add a background model, an average background, pushes or extra steps to the whole-canvas product of digit models trained apart and expect clean pairs.
- **Leaves open:** Whether a shared base helps, which plan 32 named as untried.
- **Evidence:** [the digit-composition finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md), [the step 1 page](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/01-the-plain-product-draws-nothing/README.md), [the plain product with both pushes, 4 and 7](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/01-the-plain-product-draws-nothing/sheets/plain-product-with-both-pushes/digits-4-and-7.png), [plan 33's review](../../../../../2-classifier-guidance/plans/review/33-experts-that-compose-at-inference.md). (plan 33)

### Additions to the own-half digit sampler: checker pushes, settings and extra denoising steps (plan 33, steps 3 and 4)

- **Asked:** With each model on its own half, do two checker pushes, other settings or extra denoising steps each add the required clean pairs?
- **Paper:** 1
- **Tried:** On the five tuning pairs (320 canvases): two small checkers pushing every sampling step. Then strength, size limit and sampler-noise changes. Then 5 or 20 Langevin corrector steps (a small move along the noise prediction plus a small random kick, at the same noise level) before each sampling step.
- **Bar, written first:** An addition must add at least 64 of 320. Setting changes must add 10. Twenty extra steps must add 10 over five (stated, plan 33's review).
- **Result:** Pushes: the ink checker alone 213, both pushes from the first checkers 241, retrained checkers 257, against 124 with no push, so the retrained pushes add 133. A push of exactly zero gives the same 124 (verified). Settings changed from 257: strength 3, 251; strength 30, 247; size limit 0.3, 211; sampler noise 0.5, 197 (verified). Extra steps: 5 give 298 (+41, 295 noise predictions per canvas); 20 give 306 (8 more, 1,030 noise predictions) (verified).
- **Verdict:** support for the pushes (+133 against 64). Null for every setting change. Null for extra steps (+41 against 64; 8 against 10). The review file still names the five-step setting for step 5, "with no written decision setting the 64 aside" (stated).
- **Why it worked / failed:** Pushes: not explained beyond the step 3 page. Extra steps: gains were smaller than asked (verified).
- **Kind of failure:** For the pushes: none, it passed its bar. Limit: test design, since "The move from 241 to 257 changes two things at once" (the retraining removed the final test's pairs, and it trained the pair checker on pushed canvases of the five tuning pairs), "So 257 is a tuning number" (stated). For extra steps: test design, because the setting was adopted despite missing its bar and the sources disagree on whether it was adopted (stated).
- **Idea still alive?** alive, for checker pushes on own-half sampling of digit models trained apart; 257 is a tuning number. For extra steps: unclear, since the gain is real but under the bar, and it was adopted anyway.
- **Rules out:** Do not retrain a checker on pushed canvases of the tuning pairs and report the count as a test. Do not adopt an addition that missed its bar without writing the decision.
- **Leaves open:** The sources disagree. Review line 145 says the extra steps were "not adopted" while line 146 says the chosen setting includes them. Review line 145 says zero extra steps "leaves the sampler unchanged", while step 4's page shows 58 of 64 saved canvases differ from the default by at least one pixel (verified), with the judge's calls matching on all 64.
- **Evidence:** [the digit-composition finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md), [the step 3 page](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/03-adding-the-checker-pushes/README.md), [the step 4 page](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/04-adding-extra-denoising-steps/README.md), [five extra steps on 4 and 7](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/04-adding-extra-denoising-steps/sheets/five-extra-steps/digits-4-and-7.png), [plan 33's review](../../../../../2-classifier-guidance/plans/review/33-experts-that-compose-at-inference.md). (plan 33)

### Letting the two digit models choose their side at every step (plan 33, step 6)

- **Asked:** Without a side given to them, can the two models choose which half each draws in and still give clean pairs?
- **Paper:** 1
- **Tried:** On the five tuning pairs (320 canvases): soft masks; hard masks; a side chosen at every step with the sides forced apart; and the same with the other half hidden. Each was run with no pushes and with both pushes.
- **Bar, written first:** the plan's lines for a composition on tuning pairs: 288 supports, 160 promising, under 32 no effect (stated).
- **Result:** Clean pairs of 320, without pushes and with both: soft masks 0 and 2; hard masks 0 and 0; side chosen every step 0 and 0; with the other half hidden 34 and 56 (verified). A side chosen afresh at every step swaps 14.1 times per canvas on average, from 10 to 19, on 8 canvases of 4 and 7 (verified).
- **Verdict:** none recorded on the page. At most 56 of 320 is below the promising line of 160.
- **Why it failed:** The side keeps swapping when it is re-chosen at every step (verified, 14.1 swaps per canvas).
- **Kind of failure:** done wrong: re-choosing the side at every step lets it swap (verified). Step 7's fix of holding the side raised clean pairs to 310.
- **Idea still alive?** closed, for choosing the side afresh at every step. The idea of the models choosing their side lives on in step 7.
- **Rules out:** Do not re-choose the side at every step. Hold it once chosen.
- **Leaves open:** none beyond step 7.
- **Evidence:** [the digit-composition finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md), [the step 6 page](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/06-letting-the-models-choose-their-side/README.md), [other half hidden with both pushes on 4 and 7](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/06-letting-the-models-choose-their-side/sheets/side-chosen-every-step-other-half-hidden-with-both-pushes/digits-4-and-7.png). (plan 33)

### Two digit models trained apart, each on its own half, with the side chosen early and held (plan 33 headline, steps 2, 5, 7 and 8)

- **Asked:** Can two digit models that never saw each other's digits compose clean pairs while sampling, on 20 digit pairs kept out of tuning?
- **Paper:** 1
- **Tried:** Step 2: each model sees only its own half, with the side set by a coin flip per seed. Step 5: step 4's setting (both pushes and five extra steps) on the 20 unseen pairs, 1,280 canvases. Step 7: the side chosen by a vote summed over the first 10 of 50 sampling steps, then held. Step 8: step 7's setting on the 20 unseen pairs at new seeds, 1,280 canvases.
- **Bar, written first:** Headline on the 20 unseen pairs: at least 1,152 of 1,280 (90%) and at least 45 of 64 on every pair, counted by eye. On the tuning pairs, 288 of 320 supports a composition and 160 is promising (stated, plan 33's review, 2026-11-19).
- **Result:** Step 2: 124 of 320 (per pair 37, 24, 28, 15, 20), between the no-effect and promising lines. Step 5: 1,088 of 1,280 (85.0%) against 454 with no pushes and no extra steps; adding the 48 clearly right tiles gives 1,136 (88.8%); also adding 38 plausible tiles and an estimate gives about 1,190 (93%). The weakest pair, 4 and 9, reads 42 by the judge and 43 to 46 by eye against 45. Step 7: 310 of 320 (59, 64, 60, 64, 63), against 259 with no pushes and 56 with the side chosen every step. Held, the side swaps 0 times after step 10. Step 8: 1,200 of 1,280 (94%), weakest pair 56 (3 and 8), against 1,043 (81%) and weakest 44 (0 and 8) without pushes (verified). By eye, Claude's calls on 15 rejected tiles (1 clear, 7 plausible, 7 wrong) give 1,201 counting clear tiles and 1,208 counting plausible ones too (verified). The other 65 rejected tiles were not called, and the 1,200 accepted were not re-checked by eye (stated).
- **Verdict:** inconclusive (proposed), for three reasons the page gives: the by-eye count waits for the owner's call on 15 tiles; no run compares the models' vote with a coin flip; and the plan's goal is composition with no cropping, where the best run is 121 of 320. By the judge, step 8 with pushes clears both headline bars and the run without pushes misses both. Step 5 is read as neither met nor missed.
- **Why it worked:** Each model sees only its own half, and holding the side stops the swapping, 14.1 swaps per canvas down to 0 after step 10 (verified).
- **Kind of failure:** none, it passed its bar by the judge. Limits: test design, with no coin-flip control for the vote (on a small exact toy a held coin flip gave 3,859 of 4,000 and the vote 3,840, stated); step 5 had already read these 20 pairs before step 7 was designed, so step 8 is new seeds on the same pairs; step 5's comparison changes the pushes and the extra steps together; and each setting ran once. could not measure it: the review puts the judge about 5% below a careful eye (stated).
- **Idea still alive?** alive, for two digit models trained apart composing on two known fixed halves with the side held after 10 steps. "A cat and a dog in one photo do not come with that split, so these results do not show two models placing objects freely or overlapping them."
- **Rules out:** Do not credit the vote without a held coin-flip control. Do not read step 5 against step 8 as a clean comparison, since method and seeds both differ.
- **Leaves open:** The owner's call on the 15 tiles. Whether the vote helps. A shared-canvas composition with no fixed halves. The sources disagree on the final test's seeds: plan 33, task 1.6 and the launcher's header say 64 to 127, while the launcher's command (`--eval-seed0 128`) and the review's final reading say 128 to 191. The headline count is the judge's 1,200 on the results card but by eye in the review's bars. The review's line 3 says "Nothing has run" while its lines 130 to 168 hold dated readings. Review line 130 calls 124 "the first non-zero composition of two experts that never met", while step 2's page notes 4 of 320 came earlier and plan 32's fixed-side models reached 139.
- **Evidence:** [the digit-composition finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md), [plan 33's review that set the bars](../../../../../2-classifier-guidance/plans/review/33-experts-that-compose-at-inference.md), [plan 33](../../../../../2-classifier-guidance/plans/33-experts-that-compose-at-inference.md), [five ways of composing on the same eight seeds](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/ways-of-composing-side-by-side.png), [the step 7 page](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/07-choosing-the-side-early-and-holding-it/README.md), [the step 8 page](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/08-final-test-on-20-unseen-pairs/README.md), [both pushes on 3 and 8, the weakest pair](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/08-final-test-on-20-unseen-pairs/sheets/with-both-pushes/digits-3-and-8.png). (plan 33)

### Retraining the two digit models with stroke clutter, for one shared canvas (plan 33 side road)

- **Asked:** Does training each model with broken stroke fragments of its own digits beside them let the two compose on one shared canvas with no halves?
- **Paper:** 1
- **Tried:** "Fragment models" were trained with stroke fragments, and "fine-tuned models" were fine-tuned on clutter. Both were combined on the whole canvas with plain product, both pushes, and both pushes plus extra steps, and on their own halves, on the five tuning pairs (320 canvases).
- **Bar, written first:** 288 of 320 supports, 160 is promising, under 32 is no effect (stated).
- **Result:** Clean pairs of 320 for the original, fragment and fine-tuned models. Plain product, whole canvas: 0, 9, 11. Whole canvas with both pushes: 1, 66, 75. With both pushes and extra steps: 0, 94, 121. Own halves, no pushes: 124, 146, 157. Own halves, both pushes: 257, 260, no run (verified). With extra marks allowed the best run reads 123 of 320 (verified). On 3 and 6 the fine-tuned models give 24 of 64 against 0 for the original models with the same sampler (verified).
- **Verdict:** null against the support line of 288. 121 sits between the no-effect line of 32 and the promising line of 160.
- **Why it failed:** not recorded. Both clutter model pairs miss the plan's own-digit check of 58 of 64 on some digits, "so these counts carry that shortfall" (stated).
- **Kind of failure:** wrong data, partly fixed: clutter training lifted the shared canvas from 0 to 121 (verified). done wrong: the clutter models miss their own-digit check (stated).
- **Idea still alive?** unclear. Training data moved the count, but the record cannot tell whether more clutter training would reach 288. Models that pass the own-digit check, run on the same sampler, would tell.
- **Rules out:** Check that retrained models still pass their own-digit check before reading a composition count.
- **Leaves open:** A shared-canvas composition with no fixed halves reaching 288 of 320.
- **Evidence:** [the digit-composition finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-two-separately-trained-digit-models-compose-while-sampling.md), [the side road page](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/side-road-training-with-clutter/README.md), [fine-tuned models on 3 and 6](../../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/side-road-training-with-clutter/sheets/fine-tuned-models-whole-canvas-with-both-pushes-and-extra-steps/digits-3-and-6.png). (plan 33)

### Composing two digit models trained apart against sampling each alone and pasting

- **Asked:** Does Paper 1's method, two digit models trained apart and composed while sampling with a checker push, draw more clean two-digit canvases than sampling each model alone and pasting the halves side by side?
- **Paper:** 1
- **Tried:** Two small MNIST diffusion models trained apart (`e04-either` draws digits 0 to 4, `e59-either` draws digits 5 to 9) worked on a canvas 32 pixels tall and 64 wide, run as four rows on the same seeds. The rows were sample alone and paste; the method with no push (the two experts choose a side for each digit over the first 10 of 50 steps, then each predicts noise on its own half); the method with the checker push (the gradient of two small classifiers trained only on expert samples, strength 10, capped at 0.1 of the prediction's size); and the method with a random push scaled to the same length. Each row covered 5 tuning digit pairs at seeds 0 to 63, so 320 canvases per row, scored by the digit judge `judge-v3`, which counts a canvas clean when it holds exactly two blobs, one per half, each read as the asked digit with at least 0.9 confidence.
- **Bar, written first:** In the step's design before any run, and in code in `beat_baseline_bars.py`. Support needed the pushed method minus paste to be more than 1.96 standard errors above zero, with the standard error taken from 10,000 bootstrap resamples of matched canvas differences within each digit pair. Null was a difference within 1.96 standard errors. Loses was paste ahead by more than 1.96. Inconclusive was a failed zero-push check, or the random push beating no push by more than 1.96.
- **Result:** Clean digit pairs, of 320 canvases per row: paste 287, method with no push 259, method with the checker push 295, method with a random push 258. The pushed method leads paste by 8 clean pairs, under the bar of 13.54 (1.96 times a standard error of 6.91). The following leads were read after the run and carry no bar of their own, each lead in clean pairs of 320 with its 1.96 standard error margin: checker push over no push +36 (margin 15.61); checker push over random push +37 (margin 15.47); random push over no push -1 (margin 8.75); no push against paste -28 (margin 16.57). On 64 canvases called by eye before the key was opened, the eye and the judge agreed on 58. Checks passed before the run: both checkers memorised 16 of 16 canvases (bar 15), and a zero push reproduced the unpushed method with a largest absolute difference of 0.0 (bar 1e-5).
- **Verdict:** null
- **Why it failed:** Pasting is already strong on digits. Each expert draws its own digit right about 94.7% of the time, and the design predicted from the experts' solo counts that paste would reach about 285 of 320; it reached 287 (inferred: a prediction from the solo rate, assuming each pasted object succeeds independently). With two fixed halves on a 32 by 64 canvas, pasting cannot overlap or shade anything (inferred). Why the method without its push draws 28 fewer clean pairs than paste is recorded and no run separates its causes.
- **Kind of failure:** test design (inferred). The design itself predicted paste at about 285 of 320, and paste reached 287. That leaves at most 33 canvases of headroom on fixed-half canvases where "pasting cannot overlap or shade anything", so the setting could barely separate the two methods. It also ran on 5 tuning digit pairs only (verified). The bar was written first, there were 320 canvases per row, and the controls passed, so the null is sound for this setting and says little about the idea.
- **Idea still alive?** retry worth it. The fix is a setting where pasting fails. The 20 held-out digit pairs are named on the page, and after the CLEVR paste test the two-colour design (experts told what to draw and not where) is named as Paper 1's next test.
- **Rules out:** Do not use fixed-half two-digit canvases to separate Paper 1's method from pasting: paste reaches 287 of 320 there. Do not read paste against the pushed method as a one-axis comparison: it changes how the digits reach the canvas and the push together, so it is mixed. Keep a random push of equal length beside every push: here it showed that the checker's direction, and not any disturbance of that size, gave the 36-pair lift.
- **Leaves open:** The same comparison on the 20 digit pairs kept out of tuning (not touched), and on CLEVR with more objects, where pasting may fail. Why the unpushed method trails paste by 28. No paper claim, because tuning pairs and seeds support none.
- **Evidence:** [the digit-half finding](../../../../../2-classifier-guidance/report/the-method-search/does-composing-two-digit-models-beat-sampling-and-pasting.md), [the step's review](../../../../../artifacts/notes/progress/method-search/01-beat-the-baseline/review.md), [the design that set the bar](../../../../../artifacts/notes/progress/method-search/01-beat-the-baseline/design.md), [the four rows at seeds 0 and 1](../../../../../artifacts/notes/progress/method-search/01-beat-the-baseline/paper-1-four-rows-first-two-seeds.png); W&B `vekm3sfh`; Slurm jobs 93461, 93583, 93598; commit pair `7df3e0b1-f2bdb6d` (M1)

### Adding the kept checker push to CO3 on SDXL

- **Asked:** Does adding the kept checker push to CO3 make more faithful two-subject SDXL pictures than CO3 alone?
- **Paper:** 2
- **Tried:** CO3 is a published sampler correction that combines SDXL's per-subject predictions as a product and corrects each step with the joint prompt's and the single subjects' predictions. It ran at its released settings (CFG++ at guidance 0.8, 50 steps, 6 corrected steps) in three rows: CO3 alone; CO3 with the kept push (the gradient of the kept multi-pair readout checker `head_step_002000.pt`, a small network reading SDXL's own features, at strength 30 on steps 0 to 15, capped at 0.4 of CO3's guided prediction); and CO3 with a random push of the same length. Each row covered 12 tuning subject pairs at seeds 6001 to 6008, so 96 pictures per row and 288 in all, called one at a time in a blind count by eye.
- **Bar, written first:** In the step's design before any run, and in code as `paper2_verdict` in `beat_baseline_bars.py`. Support needed CO3 with the push to lead CO3 alone by at least 5 of 96 faithful pairs and by more than 1.96 standard errors (bootstrap over the 12 pairs). Null was neither side holding that lead. Loses was CO3 alone ahead by the same margin. Inconclusive was a failed zero-push check, or the random push leading CO3 alone by that margin. The floor of 5 exists because a pushed render is not bit-reproducible and moved by about 2 of 96 when repeated.
- **Result:** Faithful pairs (exactly one clear first subject and one clear second subject, separate, nothing borrowed, an intact photograph), of 96 pictures per row: CO3 alone 11, CO3 with the push 7, CO3 with a random push 12. The lead is -4, against a winning bar of at least 5 and more than 10.08 (1.96 times a standard error of 5.14). Two clear subjects with style ignored, of 96: 32, 30 and 35; the push trails by 2 against a margin of 11.25. None of the 288 pictures was one subject or a fused hybrid. Not an intact photograph, of 96: 73, 80 and 72. Eight of the 12 pairs made 0 faithful pictures in all three rows. Checks passed first: a zero push reproduced CO3's own loop with a largest final-latent difference of 0.0 (bar 1e-5); the real push changed the final latent by up to 3.79 and the cap fired on 2 of its 16 steps.
- **Verdict:** null
- **Why it failed:** CO3 already separates the two bodies on all 288 pictures. What fails is borrowed parts, wrong species, extra subjects and drawn style, and the reviewers' notes name a drawn style on 53, 41 and 53 of the 96 pictures in the three rows (inferred: readings after the run, copied from the review; the style counts were not recomputed). The push's two presence outputs trained only on 1,603 cat and dog pictures and never on the 12 subjects asked for, and none of the 12 tuning pairs is in the checkpoint's training list (inferred, from the design).
- **Kind of failure:** wrong data (verified from the design). The push's two presence outputs trained only on 1,603 cat and dog pictures, and none of the 12 tuning pairs is in the checkpoint's training list. Could not measure it (verified). Each picture got one call. The six reviewers passed 7, 0, 6, 4, 4 and 9 of their 48. Drawn style counts as damage, with style named on 53, 41 and 53 of 96. Test design (inferred). With CO3 alone at 11 of 96, a win needed at least 5 and more than 10.08, a large share of a small count.
- **Idea still alive?** closed for the kept checkpoint as it stands on CO3: the page says the result "covers the kept push as it stands", and the lead is -4 with a clean three-row comparison. A checker push on CO3 in general is retry worth it, with a checker trained on the asked subjects (step D3, retrained on the cleaned list), a second caller on a sample, and style separated from damage.
- **Rules out:** Do not expect the kept cat and dog push to lift CO3: on 12 other pairs it trails CO3 alone by 4 of 96. Do not drive a push from presence outputs that never trained on the subjects asked for. Do not score faithfulness on prompts that do not ask for a photograph without separating drawn style from damage, because style drives most of the damage count.
- **Leaves open:** Whether a checker built for this paper and retrained on a cleaned list helps CO3 (step D3 on the board). Agreement between callers: each picture had one call, and the six reviewers passed 7, 0, 6, 4, 4 and 9 of their 48 pictures. The 12 held-out pairs and seeds 7001 to 7008 were not touched. The sources also disagree twice. First, the smoke sheet labels the pushed puma and condor at seed 6001 "clear puma, clear bird", while the blind count calls the full run's pushed picture at the same pair and seed `two_bodies` (pushed renders are not bit-reproducible, so they may not be the same picture). Second, the design gives a median of 38.4 seconds per picture against the review's smoke timing of 43.2 seconds.
- **Evidence:** [the CO3 push finding](../../../../../2-classifier-guidance/report/the-method-search/does-adding-the-kept-push-to-co3-beat-co3-alone.md), [the step's review](../../../../../artifacts/notes/progress/method-search/01-beat-the-baseline/review.md), [the design that set the bar](../../../../../artifacts/notes/progress/method-search/01-beat-the-baseline/design.md), [the scored blind count](../../../../../artifacts/notes/progress/method-search/01-beat-the-baseline/paper-2-blind-result.json), [the three rows at seed 6002](../../../../../artifacts/notes/progress/method-search/01-beat-the-baseline/paper-2-three-rows-seed-6002.png); W&B `s7o6404z`; Slurm jobs 93468, 93481, 93523, 93524; commit pair `99ac88a8-f2bdb6d` (M1)

### Giving the cat and the dog their own regions on SDXL, without bootstrapping

- **Asked:** When SDXL's cat prediction acts only inside the cat's true box and the dog prediction only inside the dog's, does each body come out as the right animal?
- **Paper:** 2
- **Tried:** A regional sampler on SDXL mixed the unconditioned prediction with each subject's guided difference weighted by a soft mask per latent cell, the two masks summing to 1. The masks came from the right boxes (both animals' boxes detected on the joint prompt's own render of the same seed), random boxes, swapped prompts, and the same prompt twice, all beside the joint prompt and the plain product. It ran for cat and dog only, on 8 seeds, giving 48 pictures, each called blind at full size by two reviewers with a third call for the one split.
- **Bar, written first:** Yes, on 2026-11-04 in the oracle-layout plan, copied into the step's design before any run. Rung 1 passes at 4 or more clear pairs of 8 on the right boxes. 2 or 3 stops the idea. 0 or 1 is a null held until bootstrapping has run, because a null from this composer alone cannot separate "regions are not enough" from "the composer is wrong".
- **Result:** Clear pairs, of 8 seeds: joint prompt 7, plain product 0, right boxes 0, random boxes 0, swapped prompts 0, same prompt twice 0. Pictures with two separate animals, of 8: joint prompt 8, every other row 0. Right boxes drew 6 fused cat-dogs and 2 single animals. Reviewers agreed on 47 of 48. Controls on seed 9 passed: the joint render matched its box source with a mean absolute difference of 0.0 of 255, and masks of 0.5 at guidance 7.5 differed from the plain product at guidance 3.75 by 1.53 of 255 (limit 3).
- **Verdict:** null by the bar (0 of 8 on right boxes), held under the design's rule until bootstrapping ran
- **Why it failed:** Fusing the two predictions by region from step 0 does not separate the animals even with the layout taken from a render that succeeded (verified: 0 of 8 across every region row). The bootstrapped run that followed separated them, so the missing piece was MultiDiffusion's bootstrapping stage (verified by that run, which changes only bootstrapping).
- **Kind of failure:** done wrong (verified). The plain fusion left out MultiDiffusion's bootstrapping stage, and adding only that stage moved right boxes from 0 to 8 of 8. Test design (verified): one pair, 8 seeds.
- **Idea still alive?** closed for plain region fusion from step 0. Right boxes taken from successful renders gave 0 of 8 against 7 of 8 for the joint prompt, which is a wide gap even at 8 seeds. The region idea itself lives on in the bootstrapped run.
- **Rules out:** Do not expect a region mask alone, applied from step 0 on SDXL, to separate two subjects: right boxes taken from a successful render still gave 0 of 8.
- **Leaves open:** Whether any region method without bootstrapping can work. The sources disagree on the eighth seed: the oracle-layout plan lists seed 20, while the step's design, both sheets and both result files use seed 30. They also disagree on what a count under 4 does: the binding design says it "stops the idea", while the step's design holds 0 or 1 until bootstrapping, and the step followed the second.
- **Evidence:** [the regions finding](../../../../../2-classifier-guidance/report/the-method-search/does-giving-each-subject-its-own-region-fix-identity.md), [the step's review](../../../../../artifacts/notes/progress/method-search/04-bind-each-body/review.md), [the step's design](../../../../../artifacts/notes/progress/method-search/04-bind-each-body/design.md), [six conditions by eight seeds](../../../../../artifacts/notes/progress/method-search/04-bind-each-body/rung-1-six-rows-eight-seeds.png); Slurm job 93794 (M9)

### Cat and dog regions on SDXL with bootstrapping

- **Asked:** With MultiDiffusion's bootstrapping added, do right regions give a clear cat beside a clear dog, and does it matter where the boxes sit and which prompt each carries?
- **Paper:** 2
- **Tried:** The same regional sampler with bootstrapping, the one part of MultiDiffusion (arXiv 2302.08113) the plain fusion leaves out: for the first 15 of 50 steps each region is denoised over a constant colour and the latents are fused (unrelated to the statistical bootstrap). Five rows (joint prompt, right boxes, random boxes, swapped prompts, same prompt twice) ran on the same 8 cat and dog seeds, giving 40 pictures called blind by two reviewers, with a third call on 3 splits.
- **Bar, written first:** The same rung 1 table, written 2026-11-04. At least 4 of 8 on right boxes with random boxes fewer is support. Right boxes at 4 or more with random boxes within 1 is "layout content irrelevant". Right boxes at 4 or more with swapped or same prompt within 1 is "regions without names". 2 or 3 is inconclusive, and 0 or 1 is null.
- **Result:** Clear pairs, of 8: joint 6, right boxes 8, random boxes 4, swapped prompts 7, same prompt twice 0 (this row asks for two cats, so the bar counts pictures with two separate animals: 8 of 8). Two separate animals, of 8: 8, 8, 4, 8, 8. Marked undamaged by both reviewers, of 8: joint 5, right boxes 0, random 0, swapped 1, same prompt 1. Every right-box picture is a cut-out on flat grey. Reviewers agreed on `two_clear` for 38 of 40.
- **Verdict:** support on the 4-of-8 bar, recorded as "regions without names". The page notes that on these counts the support row's two conditions also hold, and the bar table does not say which row applies when two do (inferred).
- **Why it worked:** Bootstrapping is what separated the animals: the same fusion without it reached 0 of 8 (verified, a one-axis comparison). The prompt a region carries decides its species, since the same-prompt row drew two cats on all 8 seeds and the swapped row drew a cat and a dog (verified on the sheet). The review says any two bootstrapped regions make two objects wherever the boxes sit (inferred), but random boxes gave two animals on only 4 of 8 against 8 of 8 for right boxes, and the page records that the two disagree.
- **Kind of failure:** none, it passed its bar. Two kinds limit how far the pass reaches. Test design (verified): it ran on one pair (cat and dog) and 8 seeds, with boxes taken from joint renders that already drew both animals, so it is an upper bound. The calls also move by 1 between counts (the joint prompt scored 7 clear pairs in the first count and 6 in this one), and the bar table lets both the "support" row and the "regions without names" row apply to the same counts (inferred). Could not measure it (verified): the bar counts clear pairs, while 0 of the 8 right-box pictures were marked undamaged by both reviewers, against 5 of 8 for the joint prompt.
- **Idea still alive?** alive, only as an upper bound. With right boxes and bootstrapping, SDXL separates a cat and a dog on 8 of 8 seeds, and the prompt each region carries decides its species. It is not a usable method (every right-box picture is a cut-out on flat grey), it covers one pair, and it supports no paper claim.
- **Rules out:** Do not count bootstrapped regions as a usable method: 0 of 8 right-box pictures were marked undamaged, against 5 of 8 for the joint prompt. Do not read a difference of 1 between rows as real: the joint renders scored 7 clear pairs in the first count and 6 in this one.
- **Leaves open:** Whether bootstrapping can separate the animals without leaving cut-outs on flat grey; no run separates separation from damage. Rung 1 covers only cat and dog with boxes taken from successful joint renders, so it is an upper bound and supports no claim. MultiDiffusion's own code on these layouts was not run. The sources also disagree on the joint prompt's seed 25, a clear pair in the first count and absent from the 6 here.
- **Evidence:** [the regions finding](../../../../../2-classifier-guidance/report/the-method-search/does-giving-each-subject-its-own-region-fix-identity.md), [the step's review](../../../../../artifacts/notes/progress/method-search/04-bind-each-body/review.md), [five conditions by eight seeds with bootstrapping](../../../../../artifacts/notes/progress/method-search/04-bind-each-body/rung-1-bootstrapped-five-rows-eight-seeds.png); Slurm job 93906; commit pair `100772b7-15d8bdd` (M9)

### CO3 with regions read from word attention

- **Asked:** Can CO3 find its own subject regions from SDXL's word attention at step 10, and do those regions beat CO3 alone on faithful pairs?
- **Paper:** 2
- **Tried:** Each subject word's cross-attention map at step 10 (a 32 by 32 grid) was smoothed, cut with Otsu's threshold and reduced to the box around its largest connected part. This was done on the 12 tuning pairs at seeds 6001 and 6002, 24 reads, as a mask check before any render. No rung 2 picture was rendered.
- **Bar, written first:** Yes, in the step's design and in `m9_rung2.py`. A read fails if either box covers at least 80% of the grid, or if the two boxes overlap with an intersection over union (shared area divided by total area, 0 to 1) of at least 0.8. If more than 25% of reads fail (more than 6 of 24), rung 2 stops and renders nothing. The render comparison would have needed CO3 with regions to beat CO3 alone by 5 of 96.
- **Result:** 11 of 24 reads failed, against a limit of 6: 7 on overlap and 4 on box size, with a median overlap of 0.527 (0.53 on the page's headline). A zero-binding control passed: with the regions computed and discarded, the final latent matched plain CO3 with a largest absolute difference of 0.0 (limit 1e-5). GPU time for the whole step was 3.53 of the 6.3 GPU-hours added for it.
- **Verdict:** not run (stopped at its mask check by the written rule)
- **Why it failed:** The mask failures are verified (counted in `rung-2-mask-reads.json`). The reasons are inferred, read off the attention maps by the review. In reads such as bobcat and quail at seed 6001 the second word lights up both bodies and its largest part is the first subject's body. In reads such as puma and condor the second word's map keeps a background floor, so its box spreads across the frame. The same run showed the first box reader returning the whole grid for both subjects.
- **Kind of failure:** done wrong. The box reader failed 11 of 24 reads against a limit of 6 (verified). Its two reasons are inferred from the maps: the second word lights both bodies, or its map keeps a background floor so the box spreads over the frame. The region idea on CO3 was never tested.
- **Idea still alive?** retry worth it. The fix is a box reader that passes the mask check (6 or fewer failures of 24), then about 4.8 GPU-hours of renders by the review's estimate.
- **Rules out:** Do not take subject regions from a single word-attention read at step 10 with Otsu boxes: 11 of 24 reads give overlapping or frame-filling boxes.
- **Leaves open:** Whether regions fix identity on CO3 at all. That needs a reader that passes the mask check, then about 4.8 GPU-hours of renders by the review's estimate (the design and the ladder picture say about 5.3; the page records the disagreement). The progress board marks step M9 `todo` while the review records rungs 1 and 2 as run.
- **Evidence:** [the regions finding](../../../../../2-classifier-guidance/report/the-method-search/does-giving-each-subject-its-own-region-fix-identity.md), [the step's review](../../../../../artifacts/notes/progress/method-search/04-bind-each-body/review.md), [word attention at step 10 with its boxes](../../../../../artifacts/notes/progress/method-search/04-bind-each-body/rung-2-step-10-word-attention-boxes.png); Slurm job 93936; commit pair `d5aeb4b6-15d8bdd` (M9)

### Reading five properties of a half-drawn cat and dog with a sentence score

- **Asked:** At SDXL steps 0 to 15 of 50, can an image-and-text model's sentence score tell a right sentence from a wrong one for counting, prompt match, left-right order, position and size, and attribute binding?
- **Paper:** 2
- **Tried:** Three models were tested. NoisyCLIP is CLIP ViT-L/14 with its image side retrained on noisy latents. Frozen CLIP is the same model with its original weights, used as the baseline. Latent CLIP is a smaller CLIP trained on clean SDXL latents. Each scored the cosine of a picture against the right sentence minus the best wrong sentence. The pictures were the 220 labelled cat and dog pictures in `eval_set_catdog_v2.json`, noised back to each step from the finished latent (185, 155 and 128 pictures depending on the property). The cut-off was chosen on one half of the seeds and balanced accuracy read on the other half. Nothing was trained.
- **Bar, written first:** Yes, in the reading design, with constants in `w2_read.py`. A model reads a property when its mean balanced accuracy over steps 0 to 15 on the reading half is at least 0.75 (balanced accuracy: the mean of the share of yes pictures called yes and the share of no pictures called no; 0.5 is chance), and it beats frozen CLIP by more than 1.96 bootstrap standard errors. Null is a passed label check with the 0.75 missed. Inconclusive is a failed label check, fewer than 8 pictures of either label in either half, or a failed pre-scoring check.
- **Result:** Mean balanced accuracy over steps 0 to 15, as NoisyCLIP, frozen CLIP, Latent CLIP. Counting: 0.515, 0.496, 0.560. Prompt match: 0.503, 0.478, 0.479. Left-right order: 0.479, 0.506, 0.507. Position and size: 0.538, 0.482, 0.510, on 1 right picture per half. Attribute binding: 0.492, 0.467, 0.481. Later steps: NoisyCLIP reads prompt match at 0.67, 0.78, 0.86 and 0.82 at steps 20, 30, 40 and 49. At step 49 NoisyCLIP's counting call is right on 107 of 109 clean pairs, 23 of 46 two-of-a-kind pictures (two cats or two dogs), 18 of 29 hybrids and the 1 single animal. Label eye check (bar 90%): left-right 62 of 64 passes, size 37 of 57 fails, recolours 18 of 32 fail.
- **Verdict:** Two verdicts exist. The reading review records null on counting, prompt match and left-right order, and inconclusive on position and size and on attribute binding, because their labels failed the eye check. The later fairness review records the whole step as inconclusive by the bars fixed beforehand, because a positive control read 0.637 against its floor of 0.65. The two disagree.
- **Why it failed:** Reading starts later than the push window. NoisyCLIP's noise training moves reading earlier, to about step 20 of 50, and not into steps 0 to 15 (verified from the per-step file; the reading is the review's). Counting fails even on finished pictures because the wrong sentence "part cat and part dog" shares both species words with the right one and scores within 0.03 of it on clean pairs (inferred from the cosines). The size and colour labels fail because the word "cat" lands mostly on the cat's head, so the cat region is often the head alone (verified by the eye check counts; the head-only reason is the review's reading). A text-only check shows CLIP's text side carries colour binding at 0.72 of a red-blue swap surviving against 0.13 with random weights, and left-right order weakly at 0.28 against 0.08 (verified).
- **Kind of failure:** test design (verified by the later fairness test). No positive control ran, and when one was added it read 0.637 against its 0.65 floor, so the protocol was not shown to be readable at these steps. Could not measure it (verified). The position-and-size and binding labels failed the eye check (37 of 57 and 18 of 32 against 90%), and position and size had 1 right picture per half. Done wrong (inferred): the counting sentences share both species words with the right one. Wrong data (stated): finished pictures noised back are not the sampler's own states.
- **Idea still alive?** unclear on this test alone, because the missing control leaves a blind reader and an unreadable protocol indistinguishable. The fairness test that followed is what settles counting, prompt match and the clean pair. Position and size and binding are retry worth it, with labels that pass the eye check and at least 8 right pictures per half.
- **Rules out:** Do not steer at SDXL steps 0 to 15 with a NoisyCLIP, frozen CLIP or Latent CLIP sentence score for these five properties. Do not write counting sentences whose wrong option shares both species words with the right one. Do not build size or colour labels from the region a species word's attention lands on without checking that it covers the whole body. Run a reader already known to work beside a sentence score before calling its miss a null.
- **Leaves open:** Whether the sampler's real states read the same as finished pictures noised back (taken up by the fairness test). Whether a score that reads would also steer. Position and size needs passing labels and at least 8 right pictures per half. Binding has no score. Latent CLIP's comparisons are mixed. The sources disagree in four more places. Latent CLIP's counting mean is 0.560 in the review and 0.561 in the loop verdict. NoisyCLIP's step 30 lead is 0.16 in the review and 0.167 in the file. Latent CLIP's 7 steps that passed on the threshold half average 0.56 in the review and 0.538 in the file. The loop summary says position and size and binding were not scored, while the review gives scores.
- **Evidence:** [the reading finding](../../../../../2-classifier-guidance/report/the-sentence-score/can-a-sentence-score-read-the-five-properties.md), [the reading review](../../../../../artifacts/notes/progress/noise-text-critic/02-can-it-read/review.md), [the reading design that set the bar](../../../../../artifacts/notes/progress/noise-text-critic/02-can-it-read/design.md), [the reading result file](../../../../../artifacts/notes/progress/noise-text-critic/02-can-it-read/result.json), [balanced accuracy by step](../../../../../artifacts/notes/progress/noise-text-critic/02-can-it-read/balanced-accuracy-by-step.png); W&B `70g3qw2f`; Slurm jobs 93753 to 94105, full scoring job 94105 at commit `385dae85` (W2)

### Testing whether the sentence-score null is fair, with a positive control and fairer settings

- **Asked:** Can a reader known to work read the half-drawn cat and dog pictures under the reading test's own protocol, and does any sentence score read at steps 0 to 15 once given its best fair chance?
- **Paper:** 2
- **Tried:** The reading test's pictures, noise draws, halves and scoring code were kept. Four groups were added. Group 1 is a positive control: the kept readout checker `readout-catdog-more3-long/head_step_002000.pt`, a small network reading SDXL's inner features. Group 2 is the reading test's noised pictures with four noise draws per step. Group 3 is 195 held-out cat and dog joint runs read on the sampler's real states, 105 of them clean. Group 4 rescores counting with sentences naming no species. The fairer settings were five wordings averaged per sentence, NoisyCLIP's own training steps 20 to 29, the real states, and a step 49 check on the finished picture.
- **Bar, written first:** Fixed in the fairness design after the reading test's scores were seen and before any score of this test existed. The control holds if the readout checker reads the clean pair at a mean of at least 0.65 over steps 0 to 15 on group 2, and beats frozen CLIP by more than 1.96 standard errors. A sentence score reads if any of the 24 sentence rows reaches 0.75 over steps 0 to 15 on any task in group 2 or 3. The reading test stands if the control holds and no sentence row reads; it is inconclusive if the control does not hold.
- **Result:** On group 2 the control read the clean pair at 0.637 over steps 0 to 15, 0.013 under its 0.65 floor, and beat frozen CLIP by 0.150 (standard error 0.035). On group 3 the control read 0.696 over steps 0 to 15, climbing from 0.50 at step 0 to 0.87 at step 15. The sentence rows there were NoisyCLIP 0.479 plain and 0.473 averaged, frozen CLIP 0.480, and Latent CLIP 0.574 plain and 0.540 averaged; no sentence row reached 0.64 at any step from 0 to 20. No sentence row's mean over steps 0 to 15 reaches 0.6 in any setting, and the highest single step is 0.67 (Latent CLIP counting at step 10). Averaging five wordings moved no row by more than 0.04. In its own training window, steps 20 to 29, NoisyCLIP reads prompt match at 0.711 (0.734 averaged) and the clean pair at 0.687 (0.711). With species-free counting sentences over steps 0 to 15: NoisyCLIP 0.553 and 0.522, frozen CLIP 0.477 and 0.498, Latent CLIP 0.503 and 0.499. At step 49 only frozen CLIP with averaged wordings reaches the bar, at 0.750, and plain NoisyCLIP falls from 0.71 to 0.517. The one clean comparison, NoisyCLIP minus frozen CLIP on the clean pair, is 0.019, under 1.96 standard errors of 0.025.
- **Verdict:** Two verdicts exist. This test's own verdict is inconclusive by the bars fixed beforehand, because the control missed its floor, and no sentence score reads in any fair setting. The reading test's own review records null on counting, prompt match and left-right order, and inconclusive on two properties. The two disagree, and the owner chose (board question 79) that each page keeps its own test's verdict.
- **Why it failed:** The control missed by 0.013 on pictures that had been used to choose which readout checker to keep, so its row is slightly optimistic, and a fresh control could land on either side (inferred, from the design). The sentence scores fail while SDXL's own features read 0.87 by step 15 on the real states. The review concludes that the early information sits mostly in the diffusion model's features, but that comparison is mixed because the reader and its input both change (inferred). Removing the species words fixed neither the early steps nor the finished picture, so the counting failure sits in the models' image side as much as in the wording (inferred, from the review).
- **Kind of failure:** the idea (verified). Across 24 sentence rows, including averaged wordings, NoisyCLIP's own steps 20 to 29, 195 real sampler runs and species-free sentences, no mean over steps 0 to 15 reaches 0.6 against a bar of 0.75. SDXL's own features reach 0.87 by step 15 on the same real states. Test design (verified from the design's caveat): the control was chosen on these pictures and missed its floor by 0.013, and the bars were fixed after the reading test's scores were seen. Wrong data (stated): group 3's labels come from a model's call, not an eye call.
- **Idea still alive?** closed for NoisyCLIP, frozen CLIP and Latent CLIP sentence scores at SDXL steps 0 to 15, since every fair setting stayed under 0.6 with a 0.15 gap to the bar. Whether the original reading test counts formally as a null is unclear, and a control not chosen on these pictures would settle that.
- **Rules out:** Do not steer SDXL at steps 0 to 15 with any of these three sentence scores, plain or averaged, on noised pictures or real states. Do not choose a positive control on the same pictures it then judges. Do not expect averaged wordings or species-free sentences to rescue an early-step sentence score.
- **Leaves open:** Whether a control not chosen on these pictures passes the floor. Whether the readout checker generalises beyond cat and dog. Whether a score that reads would steer, since the push test was "not run, because no step was readable". The sources disagree on how many pictures this test scored: 220 in the review and design, while `group_a.pictures` in the result file is 185. The review says "every sentence model passes the clean pair" at step 49, while Latent CLIP reads 0.62 and 0.60 there. The runs table lists the species-free rescore as job 94131, cancelled, while its numbers come from job 94210. Latent CLIP's counting mean is 0.560 in the reading review and 0.561 here.
- **Evidence:** [the fairness finding](../../../../../2-classifier-guidance/report/the-sentence-score/is-the-sentence-score-test-fair.md), [the fairness review](../../../../../artifacts/notes/progress/noise-text-critic/02c-fair-test/review.md), [the fairness design that set the bars](../../../../../artifacts/notes/progress/noise-text-critic/02c-fair-test/design.md), [the fairness result file](../../../../../artifacts/notes/progress/noise-text-critic/02c-fair-test/result.json), [who reads early on the real states](../../../../../artifacts/notes/progress/noise-text-critic/who-reads-early.png), [the push test's review](../../../../../artifacts/notes/progress/noise-text-critic/03-does-it-steer/review.md); W&B `etbecql0`; Slurm jobs 94113, 94114, 94210; commit `be6a9e0f` (W2c)

### Sampling each CLEVR generator alone and pasting

- **Asked:** Before any tuning of Paper 1's method on CLEVR, is cutting out and pasting objects from single-object generators already good enough to make counting objects a useless test?
- **Paper:** 1
- **Tried:** Four small EDM2 `edm2-img64-xs` models were trained for 2,000,896 images seen: three experts, each on 5,000 CLEVR pictures of one object at one fixed position, and a background model on 5,000 empty scenes. All were rendered at 320 by 240 and squashed to 64 by 64. Each expert's sample was cut in a window around its position (18 by 18, with position 3 at 18 wide and 24 tall), its outer 2 pixels faded, and laid on a background sample. 1,024 pasted images were made with two experts and 1,024 with three, scored by a small convolutional classifier that says whether an object sits at each of the three positions.
- **Bar, written first:** Yes, in the step's design. Paste is predicted to reach 7 objects if the paste rate (the mean of the three positions' in-place rates on the three-expert paste, 0 to 1) is at least 0.906, which is where Bradley's counting rule with independent objects gives 0.906^7 of about 0.50; the step then stops. Below 0.906, the small CLEVR comparison of every method runs. The result is inconclusive if an expert misses 90%, the scorer misses 99%, or the window check fails. The window check changed after its written form failed: the owner chose a new check and a taller position 3 window on board question 66, before any paste existed.
- **Result:** Both objects landed in place in 1,024 of 1,024 images with two experts, and all three in 1,024 of 1,024 with three, with 0 objects where none was asked. The paste rate was 1.0 against a bar of 0.906. Each expert alone drew exactly its object at its position in 256 of 256 samples (bar 90%), and the background model drew no object in 256 of 256. The scorer memorised 16 of 16 training pictures at each position (bar 15) and was right at all three positions on 100% of 400 held-out renders (bar 99%). On the first window check, the share of changed pixels inside the 18 by 18 window was 0.9439, 0.9394 and 0.9165 at positions 1 to 3, against a mark of 95% (failed). The check in use (each object's own footprint) gave 1.0, 1.0 and 0.998. The eye agreed with the scorer on 64 of 64 enlarged images.
- **Verdict:** support for "paste predicted to reach 7 objects", so the step stops and the small CLEVR comparison of every method (step M6) does not run on these models
- **Why it worked:** Each generator alone already draws its object right every time, and at a per-object success of 1.0 every count up to the test's cap of 12 is predicted to succeed (verified for the counts; the reach to 7 is a prediction assuming objects succeed independently, since three experts cannot draw 7). Objects at three fixed, separated places never overlap or shade each other, so pasting has nothing to get wrong (inferred, from the design).
- **Kind of failure:** none, it passed its bar. Two kinds limit how far the pass reaches. Test design (verified): the two-expert and three-expert pastes share the same top two objects tile for tile, so they are not independent draws. Three experts cannot draw 7 objects, so reaching 7 is a prediction that assumes objects succeed independently. Only tuning positions were used, none of Bradley's 12. The window check was rewritten after its written form failed, by the owner before any paste existed. Could not measure it (stated): the in-place rate does not score the lighter seam around pasted objects.
- **Idea still alive?** alive, for this setting only. Pasting places every object at three fixed, separated positions (1,024 of 1,024 images), so counting objects there cannot separate Paper 1's method from pasting. The pass says nothing about crowded or overlapping scenes, nothing about whether 7 objects are reached in practice, and nothing against Paper 1 itself. Paper 1's next test moves to experts told what to draw and not where.
- **Rules out:** Do not use counting objects at fixed, separated places to tell Paper 1's method apart from pasting: pasting places every object in 1,024 of 1,024. Paper 1's next test uses experts told what to draw and not where, which leaves no fixed place to paste into.
- **Leaves open:** Whether pasting reaches 7 objects in practice. Crowded or overlapping scenes. Image quality: the lighter seam where an expert's floor meets the background's is not measured by the in-place rate. The two-expert and three-expert pastes share the same top two objects tile for tile, so they are not independent draws. No paper claim: tuning positions only, none of Bradley's 12. The sources disagree in four places. The review's line 3 says "M7 has no verdict yet" while the same review and the board record the verdict. The review says the expert check is "still to run" and also records it passed. The board marks M6 `todo` while the review says it does not run. The paste-reach chart's footer says "No CLEVR paste run exists yet".
- **Evidence:** [the CLEVR paste finding](../../../../../2-classifier-guidance/report/the-method-search/does-sample-and-paste-already-solve-small-clevr.md), [the step's review](../../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/review.md), [the design that set the bar](../../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/design.md), [three experts pasted, first 32 images](../../../../../artifacts/notes/progress/method-search/01c-clevr-paste-first/paste-three-experts-first-32.png), [the replacement two-colour design](../../../../../artifacts/notes/progress/method-search/05-two-experts-by-colour/design.md); W&B `guf6vtho` (scorer) and `zv6oef8h` (position 1 expert); Slurm jobs 94081 to 94084 for training and 94086, 94087, 94169 and 94181 for sampling; about 8.5 GPU-hours (M7)

### Reading which digits are present from the digit model's own noise predictions

- **Asked:** Do the digit diffusion model's noise predictions for all ten labels on one noisy canvas carry which digits are on that canvas, at noise levels where a push would act?
- **Paper:** both (one class-conditioned digit model that never saw two digits together, which is Paper 1's setting, read at the noise levels of SDXL's first 16 steps for Paper 2)
- **Tried:** A frozen digit model was used, a pixel U-Net with eleven labels that drew one digit per canvas in training and never two together. On canvases of digits 0 to 4 (32 by 64 pixels), it was called 11 times per noisy canvas to build the signature: ten maps, each the prediction for one digit label minus the prediction with no label, signed and unnormalised. One small convolutional reader per timestep was fitted on 4,000 canvases and read on 2,000 others with different noise. The signature was compared with the noisy canvas, the raw noise, an error readout (each label's prediction against the true noise), the path length (the squared size of each map), and shuffled labels. Two windows of 16 timesteps were read separately: an early one, t 999 to 693 (steps 0 to 15 of 50), and one matched to SDXL's steps 0 to 15 by remaining signal, t 712 to 480.
- **Bar, written first:** Pinned in the score path design before the full read, and in `x5_signal.py` as `MIN_PRESENCE = 0.75` and `MIN_OVER_NOISE = 0.10`. The signature must read presence (balanced accuracy of "this digit is on the canvas", averaged over digits 0 to 4, 0.5 is chance) at 0.75 or more at some step of a window, and at least 0.1 above the same reader on raw noise. Support if this holds in the early window; "support at SDXL's noise only" if it holds only in the matched window; null if the positive control and overfit test pass and it misses both; inconclusive if the canvas row misses the positive control or the overfit test fails. Before the full read, on the owner's word (board questions 72 and 74), the reader was changed from linear to convolutional and the control rerun at full size; no bar was changed.
- **Result:** The signature's best is 0.513 at t 754 in the early window (raw noise 0.493 there) and 0.700 at t 480 in the matched window (raw noise 0.499 there, so 0.20 above noise but under 0.75). It is still rising when the window ends (0.680 at t 495). At t 480, as presence accuracy: noisy canvas 0.593, path length 0.621, error readout 0.742 (its peak is 0.746 at t 495), shuffled labels 0.49 to 0.51 throughout. On two separate digits, asked pair against one wrong digit, the signature reaches 0.655 at t 480. Stacking all 16 steps reads 0.607 in the early window and 0.674 in the matched window. Positive control at t 100: noisy canvas 0.950 against a floor of 0.9 (passed). Overfit: 16 of 16 (bar 15). Cost about 0.74 GPU-hours against a budget of about 0.5.
- **Verdict:** null
- **Why it failed:** At the matched window's noise the signature carries some presence and is still rising at the window's end, but does not reach 0.75 (verified). In the early window the canvas holds at most 8.7% of its signal and every input except the error readout sits at chance (verified). The error readout's early reading partly reflects access to the true noise, which cannot be had during sampling (inferred, by the review). The stacked early read's 0.607 may come from the 16 steps sharing one noise draw that a combination across steps can partly cancel (inferred).
- **Kind of failure:** the idea, for the early window (verified). The signature's best is 0.513, with the positive control at 0.950 and the overfit test at 16 of 16. Test design for the matched window (verified): the signature is still rising at the window's end (0.680 at t 495, then 0.700 at t 480), there is one run on 2,000 canvases with no confidence interval, and the stacked read shares one noise draw across its 16 steps.
- **Idea still alive?** closed for the early window, t 999 to 693. The matched window is unclear. What would tell them apart: a read along the sampler's own path with unshared noise, a reader across a later window, and a confidence interval on the 0.700 against 0.75.
- **Rules out:** Do not build the text-aligned encoder on the score path for steps matching SDXL's 0 to 15: the path carries presence at 0.700 at best. Do not count the error readout as a usable signal, because it needs the true noise. Do not read a stacked multi-step result as evidence that a sampled path carries the digits when the steps share one noise draw.
- **Leaves open:** Whether a reader trained across both windows, or a later window, reaches 0.75. Whether the signature along the sampler's own path, where noise is not shared across steps, reads the same. No confidence interval: one run on 2,000 canvases. A miss here says nothing about SDXL. The sources disagree on the reader and input sizes: the probe-input diagram says "linear probe per digit", with the signature "pooled 4x4, 1,280 numbers" and the canvas "128 pooled", while the design and result file say a convolutional probe on maps pooled 2 by 2. The text-aligned encoder and the two arithmetic tests were not run (board question 76).
- **Evidence:** [the score path finding](../../../../../2-classifier-guidance/report/does-the-score-path-carry-which-digits-are-present.md), [the score path review](../../../../../artifacts/notes/progress/score-space/05-signal/review.md), [the score path design that set the bar](../../../../../artifacts/notes/progress/score-space/04-design.md), [the score path result file](../../../../../artifacts/notes/progress/score-space/05-signal/full.json), [presence accuracy by timestep](../../../../../artifacts/notes/progress/score-space/05-signal/presence-by-timestep.png); W&B `gq51nl74`; Slurm jobs 94160, 94180, 94184 (X5)

### Training our own noise-aware image encoder against CLIP's frozen text side, on digits

- **Asked:** Does a sentence score learned on some digit concepts read digits it never trained on, as a cheap test before any SDXL version?
- **Paper:** 2
- **Tried:** A new small image encoder (four convolution blocks of widths 64, 128, 256 and 256, told the step through a learned embedding) was trained on noisy 32 by 64 digit canvases against CLIP ViT-L/14's frozen text encoder. The loss picked the true caption out of a bank naming only digits 0 to 4 (temperature 0.07), with canvases noised to the digit sampler's steps 0 to 25. It ran 8,000 updates at batch 256 in 158 seconds. It was read on 1,000 seen canvases (digits 0 to 4) and 2,000 unseen canvases (digits 5 to 9), judged over steps 14 to 25, the window that keeps as much of the canvas as SDXL's steps 0 to 15.
- **Bar, written first:** Yes, in the design before the run, with constants in `w4_train.py`. Support needed three things: unseen balanced accuracy over steps 14 to 25 of at least 0.75; within 0.05 of the seen score; and at least 0.75 on two digits apart against canvases with a wrong digit, where layout alone scores 0.5. Null is a passed overfit test with any of the three missed. Inconclusive is a failed overfit test, or a seen score below 0.55 at every step.
- **Result:** Balanced accuracy over steps 14 to 25: seen 0.758, unseen 0.590, a gap of 0.168 against 0.05 allowed. Digits apart against wrong digits: seen 0.637, unseen 0.502 (chance). Steps 0 to 15, read second: seen 0.536, unseen 0.504. By step, seen climbs from 0.64 at step 14 to 0.89 at step 25, while unseen goes 0.52 to a peak of 0.67 at step 22 and ends at 0.63. All three checks are recorded false. The overfit test placed 16 of 16 (15 needed), and a 1% run read seen canvases at 0.660.
- **Verdict:** null
- **Why it failed:** The image encoder never saw digits 5 to 9, and no caption naming them was in the bank, so nothing in training tied their shapes to CLIP's words for them (inferred by the review from the two splits). The loop verdict reads this as "CLIP's frozen words do not carry an image encoder to concepts it never trained on".
- **Kind of failure:** the idea (verified). Seen digits passed at 0.758 and the overfit test placed 16 of 16. Unseen digits apart against wrong digits read 0.502, which is chance, on 2,000 unseen canvases. Test design (stated): one model trained once with fixed settings.
- **Idea still alive?** closed for this recipe on digits, because CLIP's frozen words did not carry the encoder to digits outside its caption bank. Unclear for SDXL, since the page says a miss on 32 by 64 canvases with ten concepts "does not show that an SDXL version would fail". Only an SDXL run would tell, estimated at about 70 GPU-hours (inferred), and the review does not recommend it.
- **Rules out:** Do not expect a noise-aware encoder trained against CLIP's frozen text side to read concepts outside its caption bank: unseen digits apart against wrong digits read 0.502. Do not spend about 70 GPU-hours (the review's inferred estimate, against the loop's 8.7) on an SDXL version of this recipe on the strength of this run; the review does not recommend it.
- **Leaves open:** Whether an SDXL version would fail the same way, since a 32 by 64 canvas with ten concepts says little about SDXL. Whether a reader would steer, since no sampler was in this step. How much is the recipe and how much is one run: one model was trained once. The SDXL cost rests on inferred figures, and the measured 0.128 seconds per update was not used. The sources disagree in four places. The review says both splits stay between 0.49 and 0.60 at steps 0 to 15, while the file holds seen 0.6350 at step 14. The design gives timesteps 712 to 480 for the judged window, while the file holds 714 and 489. The design trains on steps 0 to 25, while the two drawings and the scene card say steps 0 to 15. The design builds the SDXL cost from measured time, while the review says the figures come from no run.
- **Evidence:** [the own-encoder finding](../../../../../2-classifier-guidance/report/the-sentence-score/does-our-own-noise-aware-model-read-unseen-digit-pairs.md), [the own model review](../../../../../artifacts/notes/progress/noise-text-critic/04-train-our-own/review.md), [the own model result file](../../../../../artifacts/notes/progress/noise-text-critic/04-train-our-own/final.json); W&B `en52x031`; Slurm jobs 94208 and 94211 at commit `8cbdaf49` (W4)

### One trained layer on frozen image vectors, read on unseen pairs

- **Asked:** At SDXL steps 0 to 15, can a single trained linear layer on a frozen image encoder's vector read counting and prompt match on subject pairs it never trained on?
- **Paper:** 2
- **Tried:** One logistic regression was fitted per model, property and step on the frozen image vectors of NoisyCLIP, frozen CLIP and Latent CLIP (768, 768 and 640 numbers). It used 4,039 pictures noised to steps 0 to 15 with one shared noise draw: 3,268 to fit, from 119 pairs after the subject guard removed 28 pictures from 4 pairs, 349 in the seen test, and 422 from 7 unseen pairs (bison and horse, donkey and duck, guitar and traffic cone, horse and pig, llama and alpaca, rhino and zebra, windmill and tractor). Labels came from an earlier labeller's `kind` field. A control probe was fitted on 768 fixed random numbers per picture.
- **Bar, written first:** Yes, in the design before the run, with constants in `w2b_probe.py`. Support needed unseen balanced accuracy over steps 0 to 15 of at least 0.75, within 0.05 of the seen score. Null is missing either. Inconclusive is fewer than 30 pictures in either unseen class, or a seen score below 0.55 at every step. The random-number control must stay at most 0.6.
- **Result:** Mean balanced accuracy over steps 0 to 15 (0.5 is chance), as counting seen, counting unseen, prompt match seen, prompt match unseen. NoisyCLIP: 0.637, 0.560, 0.644, 0.586. Frozen CLIP: 0.579, 0.526, 0.614, 0.552. Latent CLIP: 0.575, 0.527, 0.579, 0.527. Random numbers: 0.502, 0.506, 0.505, 0.497. Every encoder falls by 0.048 to 0.077 from seen to unseen. The unseen classes held 195 and 227 pictures for counting and 109 and 86 for prompt match, all above 30. NoisyCLIP leads frozen CLIP on every cell, but no standard error is recorded, so the lead is untested.
- **Verdict:** null for all three encoders
- **Why it failed:** The frozen vectors carry real but weak information: each encoder sits above the random control on seen pairs, and loses part of that on new pairs (verified from the result file). The review compares this with the readout checker on SDXL's features, which reaches 0.87 by step 15 on real states, and concludes that the early information sits mostly in the diffusion model's features. It marks that inferred and mixed, because the two readers differ in input and pictures.
- **Kind of failure:** the idea (verified). The best unseen score is 0.586 against 0.75. The random-number control sat at 0.506 and 0.497. Every unseen class held 86 to 227 pictures, above the minimum of 30. Could not measure it (stated): the `kind` labels' error rate is not measured. Wrong data (stated): finished pictures noised back, not the sampler's states.
- **Idea still alive?** closed for a linear probe on NoisyCLIP, frozen CLIP or Latent CLIP vectors at steps 0 to 15, since the probe found real but weak information and the gap to the bar is at least 0.164. Whether NoisyCLIP's lead over frozen CLIP is real stays unclear, because no standard error was recorded.
- **Rules out:** Do not use a linear probe on NoisyCLIP, frozen CLIP or Latent CLIP vectors as an early-step reader for unseen pairs: the best unseen score is 0.586 against a bar of 0.75.
- **Leaves open:** Whether a probe that read would steer. Whether the sampler's real states read the same. How accurate the `kind` labels are: not measured. Whether NoisyCLIP's lead over frozen CLIP is real. No W&B run id is recorded. The sources disagree in three places. The review's line 3 gives an unseen low of 0.527, while frozen CLIP's unseen counting is 0.526 (0.5263 in the file). The review says the random control passes, while the file's `random_numbers.counting.verdict` reads "inconclusive" and `random_control_passes` is true. The smoke test is 32 pictures in the design and 34 in the review.
- **Evidence:** [the linear probe finding](../../../../../2-classifier-guidance/report/the-sentence-score/does-a-linear-probe-on-frozen-image-vectors-read-unseen-pairs.md), [the linear probe review](../../../../../artifacts/notes/progress/noise-text-critic/02b-linear-probe/review.md), [the linear probe result file](../../../../../artifacts/notes/progress/noise-text-critic/02b-linear-probe/result.json); no W&B run id recorded; Slurm jobs 94110 (failed at the fit stage), 94219 (smoke) and 94220 (full run) at commit `8cbdaf49` (W2b)

### A readout checker built on 1,920 labelled drawings (plan 23)

- **Asked:** Does a readout checker trained on plan 23's 1,920 drawings, labelled by an API, read a clean cat and dog on held-out pictures?
- **Paper:** 2
- **Tried:** Plan 23's 1,920 drawings were labelled and a readout checker was built on them (folders `api-labels/drawings.jsonl`, `dataset-v4`, `result-sheet-v4c`, `result-sheet-v4c-early`).
- **Bar, written first:** not recorded on this page.
- **Result:** "no held-out reading is written in any plan, walk or card".
- **Verdict:** none recorded: no held-out reading exists.
- **Why it failed:** not recorded.
- **Kind of failure:** test design: no reading recorded (verified).
- **Idea still alive?** unclear. Recording the held-out reading would tell; the check files are on the cluster's scratch folder.
- **Rules out:** Record a checker's held-out reading in its plan before building pushes on it.
- **Leaves open:** Whether plan 23's checker reads the concept. The sources disagree on what it trains on: the plan says no labels, joint drawings against product drawings, while the script counts as real only joint drawings that the API called a faithful pair (`train_readout_pair.py` lines 16 and 17).
- **Evidence:** [the checker finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-checker-read-a-clean-cat-and-dog.md), [plan 23](../../../../../2-classifier-guidance/plans/23-rung2-a-discriminator-trained-the-kim-et-al-way-at-scale.md). (plan 23)

### A checker trained on 78 pairs plus cat and dog, tested on both new pairs (plan 30, second question)

- **Asked:** Does a checker trained on 78 pairs plus cat and dog carry the early push to lion and meerkat and to typewriter and cactus?
- **Paper:** 2
- **Tried:** Not run as designed. The nearest run, the `multipair2` checker, also holds the benchmark pictures and has no typewriter and cactus sheet.
- **Bar, written first:** in plan 30's review (not restated on this page).
- **Result:** no recorded verdict.
- **Verdict:** not run.
- **Why it failed:** not run. Tasks 1.1 to 1.3 are unticked, although the 78-pair runs were drawn and labelled.
- **Kind of failure:** test design: the nearest run changes the training set as well (verified).
- **Idea still alive?** unclear. Running it as designed, or retiring it, would settle it.
- **Rules out:** Do not cite the `multipair2` run as plan 30's second question.
- **Leaves open:** The question itself.
- **Evidence:** [the other-pairs finding](../../../../../2-classifier-guidance/report/classifier-guided-poe/does-the-early-push-carry-to-other-pairs.md), [plan 30](../../../../../2-classifier-guidance/plans/30-does-the-recipe-work-on-other-pairs.md). (plan 30)

## Project 3: understanding compositionality

### The pooled adapter at full strength on held-out cat and dog

- **Asked:** Does the pooled adapter compose the held-out pair cat and dog where plain PoE does not? The pooled adapter is one LoRA trained on many animal pairs to supply the correction (the joint prompt's prediction minus the plain product's).
- **Paper:** 2
- **Tried:** SDXL with 50-step DDIM, plain PoE plus the adapter's correction at λ 1 (λ is the strength the correction is added at) on every step. Held-out cat and dog at seeds 9 to 16 (8 pictures per checkpoint). Checkpoints: rank 8 at 30,000 steps, rank 16 at 50,018 and rank 32 at 30,050, plus rank 32 at every checkpoint probed from 30k to 90k. Each picture was scored by the instance-count scorer.
- **Bar, written first:** none recorded. The page says "No criterion was pre-registered". The 7 of 8 figure is judged against the instance-count scorer validated in the compose-score validation.
- **Result:** 7 of 8 seeds held two animals by the instance-count scorer at each of rank 8 at 30,000, rank 16 at 50,018 and rank 32 at 30,050. The strict read (a cat box and a dog box overlapping under 0.5) gives the same 7 in each case. Plain PoE (λ 0) composed 0 of 8 at every checkpoint. Rank 32 holds a flat 7 of 8 from 30k to 90k. Seed 14 never composes, and its joint-prompt target is itself two cats. Rendering seed 9 twice on one device gave bit-identical pictures (verified). Across GPU generations the same render differs by about 2 of 255 in mean pixel value and the compose calls agree (stated, from the session record).
- **Verdict:** support (the page marks "✅ 7 of 8").
- **Why it worked:** The page gives no measured mechanism. Its only "why" is about λ saturating (see the next entry), and it labels that an inference.
- **Kind of failure:** none, the page records support, but no bar was written first, so test design applies as well. Two kinds limit how far the pass reaches. **Test design** (verified, the page's own caveat): "One pair, eight seeds. Every count here is cat x dog." Rank and step also move together across the three best checkpoints, "so nothing here ranks the ranks". **Could not measure it** (verified as a caveat): "The scorer counts instances, not correctness of species. The strict cat-and-dog read is a box-level check with the same detector and is not separately validated."
- **Idea still alive?** alive. The pass reaches one pair only: the pooled adapter at λ 1 (the strength the correction is added at) drew two animals on 7 of 8 held-out cat and dog seeds, where plain PoE drew 0 of 8. It gives no rate over other held-out pairs and no ordering of ranks.
- **Rules out:** Do not rank adapter ranks from rank 8 at 30k, rank 16 at 50k and rank 32 at 30k, because rank and step differ together. Check a seed's joint-prompt target before calling a missed seed an adapter failure: seed 14's target is two cats. Do not average over repeated renders on one device, because the sampler is deterministic (η 0, pinned starting latents).
- **Leaves open:** Whether the other seven held-out pairs compose at this rate. Whether a re-cached seed 14 target composes. Whether the strict cat-and-dog read is valid, since it has no validation set. The cross-device pixel figure is not backed by a file on disk.
- **Evidence:** the [checkpoint and λ finding](../../../../../3-understanding-compositionality/report/does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md), the [compose-score validation](../../../../../1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score), the [recipe that renders the held-out grid](../../../../../runbook/running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md). W&B `8vl2uzak`, `n1w4fw5b`, `gcej8ib9`, `6xc2l8ix` in `prime_lab/poe-repair-animals-compose`. Results are in `figure_r8_030000/`, `figure_r16_050018/` and `figure_r32_030050/` under `/datasets/mmolefe/poe_repair_min/outputs/showcase/`.

### Raising the correction strength above 1

- **Asked:** Does adding the correction at λ above 1 (λ is the strength the correction is added at) compose more held-out cat and dog seeds?
- **Paper:** 2
- **Tried:** Same 8 held-out seeds (9 to 16), correction on all 50 steps. λ was swept over 1.0, 1.2, 1.3, 1.4, 1.5 and 2.0 on rank 16 at 50k, rank 32 at 30,050, 60k and 70k, and rank 8 at 240k, 280k and 300k. Pictures were scored by the instance-count scorer and the strict cat-and-dog read.
- **Bar, written first:** none recorded. The read is descriptive: "a seed count that rises with λ ... would have supported the hypothesis, a flat or falling count is what a saturated correction looks like."
- **Result:**
  - Seeds of 8 holding two animals, at λ 1.0 / 1.2 / 1.5 / 2.0: rank 16 at 50k 7 / 6 / 7 / 7; rank 32 at 60k 7 / 7 / 7 / 7; rank 32 at 70k 7 / 7 / 7 / 8; rank 8 at 240k 5 / 7 / 4 / 3, where the strict cat-and-dog count at λ 2.0 is 0 of 8; rank 8 at 280k 4 / not recorded at 1.2 / 4 / 1.
  - Rank 8 at 300k, λ 1.0 / 1.2 / 1.3 / 1.4 / 1.5 / 2.0: two animals 5 / 5 / 4 / 4 / 5 / 5 of 8, strict cat and dog 5 / 5 / 4 / 3 / 4 / 2. Its DINOv2 drift at λ 1 is +0.014. DINOv2 drift is the corrected render's distance to the joint-prompt render minus its distance to the plain PoE render, and negative means nearer the joint prompt. So +0.014 means no nearer than plain PoE.
  - Rank 32 at 30,050, λ 0 / 1.0 / 1.2 / 1.3 / 1.5 / 2.0: two animals 0 / 7 / 6 / 6 / 7 / 6 of 8, strict count identical, DINOv2 drift +0.531 / -0.122 / -0.162 / -0.134 / -0.135 / -0.124.
  - The largest per-step correction norm the sampler applied, averaged over seeds, falls as λ rises: 60.4 at λ 1.0, 51.4 at 1.2, 47.8 at 1.3, 44.8 at 1.5, 31.5 at 2.0.
  - By eye: from λ 1.3, seed 10's two close-up dogs become a distant shack with two thumbnail-sized animals. At λ 2.0, seeds 9, 11 and 15 change from photograph or pencil sketch to coloured illustration.
- **Verdict:** null (the page marks "⚪ null on λ above 1").
- **Why it failed:** The falling correction norm is verified. "The adapter is reading a trajectory that λ has already moved, so a larger multiplier does not produce a proportionally larger correction" is the page's reading of that number, so it is inferred. The deeper cause is inferred too: the page says "the correction's magnitude is already matched to the target (‖Δ̂‖/‖Δ‖ is 0.87 on held-out cells in the fit read), so scaling it up overshoots ... That is an inference, not a result."
- **Kind of failure:**
  - **could not measure it** (verified by eye on the grid): "The count is blind to the scene change." Seed 10's drop from 7 to 6 at λ 1.3 and above "reads as a composition failure when the grid shows a framing change."
  - **test design** (verified): no bar written first, one pair, 8 seeds. On healthy checkpoints λ 1 already reached 7 of 8, so at most one seed could be added (inferred from the counts).
  - **the idea** (inferred): on rank 16 and rank 32 the count sits flat from λ 1.0 to 2.0, which fits saturation.
- **Idea still alive?** unclear. The instance count had one seed of headroom and cannot see framing or medium changes. What would tell the two apart: a pair where λ 1 composes well below ceiling, scored with a measure that sees framing and species (inferred).
- **Rules out:** Do not raise λ above 1 expecting more seeds on a healthy checkpoint for cat and dog. Do not judge a λ setting from the instance count alone. Do not raise λ on a degraded checkpoint: at rank 8 240k, λ 2.0 took the strict cat-and-dog count to 0 of 8.
- **Leaves open:** Why λ saturates, which is not measured. How far the framing and medium move per unit of λ, which is read by eye only and has no number. Whether λ helps on pairs other than cat and dog.
- **Evidence:** the [checkpoint and λ finding](../../../../../3-understanding-compositionality/report/does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md). Results are in `lambda_boundary_probe_r16_050018/`, `lambda_boundary_probe_r32_030050/`, `_r32_060050/`, `_r32_070050/`, `lambda_boundary_probe_240k/`, `lambda_boundary_probe/` (280k) and `lambda_boundary_probe_300k/` under `/datasets/mmolefe/poe_repair_min/outputs/showcase/`.

### Applying the correction on steps 0 to 9 only

- **Asked:** Does applying the adapter's correction only on the first ten drawing steps (steps 0 to 9), with plain guided PoE after, compose as many held-out cat and dog seeds as applying it on all 50?
- **Paper:** 2
- **Tried:** Same 8 held-out seeds and the same rank 8, 16 and 32 checkpoints. Correction on steps 0 to 9 only, swept over λ (the strength the correction is added at). Scored by the instance-count scorer.
- **Bar, written first:** none recorded. The page says no criterion was pre-registered for the window sweep.
- **Result:** Best λ per checkpoint with the early window: rank 8 at most 4 of 8 seeds holding two animals, rank 16 at most 4, rank 32 at most 5. "No bottom-row line reaches its top-row partner": the full window gives 7 of 8 at λ 1.
- **Verdict:** null (the page marks "⚪ null ... on the early window").
- **Why it failed:** not recorded. The page gives no measured or argued reason.
- **Kind of failure:** **test design** (verified): no bar written first, one pair, 8 seeds, and the same instance count, which "counts instances, not correctness of species". The comparison itself changes one axis, the window, on fixed checkpoints and seeds. The gap appears on all three ranks.
- **Idea still alive?** unclear. One pair at 8 seeds cannot tell a weaker early-only correction from a pair-specific effect. The same comparison on more held-out pairs would.
- **Rules out:** Do not expect the correction on steps 0 to 9 alone to reach the full window's count for cat and dog. On these checkpoints it peaked at 4 or 5 of 8 against 7 of 8.
- **Leaves open:** Why the later steps matter. Whether the gap holds on other pairs.
- **Evidence:** the [checkpoint and λ finding](../../../../../3-understanding-compositionality/report/does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md). The sampler is `run_lora_residual_inject_windowed_poe` in `1-lora-correction/scripts/showcase/lambda_window_grid.py`.

### Training the pooled adapter longer

- **Asked:** Does training the pooled adapter past its best checkpoint keep improving the held-out cat and dog renders?
- **Paper:** 2
- **Tried:** The rank 8 adapter ran 100k to 200k steps (`8vl2uzak`) and 200k to 450k (`n1w4fw5b`). Rank 16 ran to 100k (`gcej8ib9`) and rank 32 to 100k (`6xc2l8ix`). All trained on 88 cells with AdamW, weight decay 0.0 and a constant learning rate of 1e-4. Two instruments read them: the 8-seed held-out cat and dog grid at λ 1 at several checkpoints, and a 2-cell W&B tracking set (cat and dog seeds 9 and 10) logged every 10k steps. The weight norm was read from every checkpoint.
- **Bar, written first:** The [train longer plan](../../../../../1-lora-correction/plans/01-showcase-the-trained-lora/plans/experiments/08-experiment-a-resume-to-200k.md) planned to read whether the held-out tracking curves keep improving. "No numeric bar was written for 'worse'". The decay criterion (drift returning to 0) "is post-hoc and is said so."
- **Result:**
  - DINOv2 drift, mean of 8 seeds at λ 1. Drift is the corrected render's distance to the joint-prompt render minus its distance to the plain PoE render; negative means nearer the joint prompt.
    - Rank 32: -0.091 at 30,050, -0.060 at 60,050, -0.003 at 70,050, -0.013 at 90,050.
    - Rank 8: -0.132 at 30,000, -0.004 at 240,000, +0.041 at 280,000, +0.014 at 300,000.
    - Rank 16: -0.102 at 50,018, +0.042 at 60,018.
  - Rank 8's absolute distance to the joint-prompt image grows from 0.551 at 30k to 0.850 at 280k.
  - Tracking-set drift for rank 32 (mean of 2 held-out cells): -0.264 at 30,050, -0.234 at 60,050, -0.021 at 70,050, +0.010 at 80,050, +0.018 at 90,050, -0.045 at 100,050. The 2 training cells stay between -0.05 and -0.15 at every save on every run. Rank 8's held-out drift is at or above 0 from 140k on.
  - Weight norm (Frobenius norm of the LoRA weights): rank 8 is 35.9 at 10k, 60.1 at 100k, 73.2 at 200k and 93.2 at 450k. Rank 16 is 45.1 at 10k and 72.5 at 100k. Rank 32 is 58.9 at 10k, 68.8 at 30k, 79.0 at 60k, 86.9 at 90k and 89.3 at 100k.
  - Seeds of 8 holding two animals: rank 32 stays at 7 at 30k, 60k, 70k and 90k. Rank 8 goes 7 (30k), 5 (240k), 4 (280k), 5 (300k).
  - By eye: by 240k every rank 8 tile has lost saturation, and at 90k rank 32's seed 9 dog has a malformed head.
- **Verdict:** null. The page's answer is "❌ no", and its decay criterion is post hoc.
- **Why it failed:** Three things are verified. Held-out fidelity decays from about 70k (rank 32) and 160k (rank 8). The weights grow without bound. The training cells do not decay. The cause is inferred: "Norm growth, the loss having flattened, and off-trajectory evaluation ... all rise together; no run varied weight decay or the learning rate, so which of them drives the decay is untested." Off-trajectory evaluation means the adapter trains on plain-PoE states and is used on corrected ones.
- **Kind of failure:**
  - **done wrong** (inferred): weight decay hard-wired to 0.0, a constant learning rate and no early stop, under which the norm keeps rising (verified numbers above).
  - **could not measure it** (verified): the compose count "lags the fidelity loss by tens of thousands of steps and is not the instrument to stop training on". Rank 32 holds 7 of 8 while drift goes from -0.091 to -0.013.
  - **test design** (verified): the decay criterion is post hoc, "Held-out means cat x dog", the 2-cell tracking set "swing[s] by 0.3 between saves", and the rank 8 lineage "crossed three runs and two GPU generations".
- **Idea still alive?** retry worth it. The fix the page names: "A run with weight decay on ... and an early stop on held-out drift, to test the inferred cause." The trainer has no flag for weight decay yet.
- **Rules out:** Do not select or stop checkpoints on the compose count; read held-out DINOv2 drift on the 8-seed grid. Do not trust the 2-cell tracking set alone. Do not train past about 70k (rank 32) or 160k (rank 8) with weight decay 0 and a constant learning rate expecting held-out gains.
- **Leaves open:** Which of norm growth, flat loss or off-trajectory evaluation drives the decay. Whether the other seven held-out pairs decay on the same schedule.
- **Evidence:** the [training-length finding](../../../../../3-understanding-compositionality/report/does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md) and the [train longer plan](../../../../../1-lora-correction/plans/01-showcase-the-trained-lora/plans/experiments/08-experiment-a-resume-to-200k.md). W&B `6xc2l8ix`, `8vl2uzak`, `n1w4fw5b`, `gcej8ib9`, `jii2mtb0` (rank 16, aborted at 37,418) and `le2tp2ti` (rank 32, aborted at 6,500).

### Adapting self-attention as well as cross-attention

- **Asked:** Does covering the self-attention projections as well as the cross-attention ones make the correction better or worse?
- **Paper:** 2
- **Tried:** The page compares DINOv2 drift at step 50,000 on six tracking cells (two trained on, four held out). DINOv2 drift is the corrected render's distance to the joint-prompt render minus its distance to the plain product render, running from -2 to +2, lower is better. The runs, all rank 32 on the same pool:
  - cross and self-attention: `v57-11b-x0-self-decay` (W&B `tiiz0c0c`)
  - cross-attention only: `v57-09-x0-seeds` (`ek4w6lbu`), `v57-08-x0-w25` (`a5rv38qy`) and `v57-10-x0-contrast` (`boammtzo`)
- **Bar, written first:** none recorded. "No bar was pre-registered for this question, so the verdict is a descriptive read." The compose rate was dropped because "every configuration here reads 1.0 on every cell where it is scored".
- **Result:**
  - Cross and self against the best cross-attention run: cat and dog seed 9, -0.527 against -0.488. Elephant and penguin seed 9, +0.174 against -0.234. Six-cell mean, -0.269 against -0.329.
  - The other two cross-attention runs have six-cell means of -0.288 and -0.271.
  - Trainable parameters:
    - cross-attention at rank 32: 19,824,640 of 2,587,288,324 (0.77%), over 210 projections
    - adding self-attention: 35,799,040 of 2,603,262,724 (1.38%), over 420 projections
    - self-attention at rank 64: 51,773,440 of 2,619,237,124 (1.98%)
- **Verdict:** inconclusive.
- **Why it failed:** Verified from the run settings: the comparison is mixed. `v57-11b-x0-self-decay` "also decays its learning rate to zero and was resumed from a step-5,000 checkpoint, while the three cross-attention runs hold the rate constant and train from scratch."
- **Kind of failure:**
  - **test design** (verified): a mixed comparison with two axes beside the layers, and no bar.
  - **could not measure it** (verified): the compose rate reads 1.0 everywhere, so it has no headroom. The 1-lora ablations page adds that this rate came from the trainer scoring the render alone. It "is 1 when the render's DINOv2 embedding is nearer the joint-prompt anchor than either single-prompt anchor ... It does not count animals" (verified from the code at commit `e289646`; that every run used this code is inferred).
- **Idea still alive?** retry worth it. The fix named on the page is the clean pair `v58-01-plain` (W&B `qx5ffxzp`) against `v58-03-self` (W&B `qtk5gi37`): same pool and schedule, differing only in the layer set. The page records them as "training" on 2026-09-23 and gives no read.
- **Rules out:** Do not credit the gap between `v57-11b-x0-self-decay` and the cross-attention runs to the layers. Do not use the trainer's compose rate to separate adapters.
- **Leaves open:** The clean comparison's verdict. A self-attention-only adapter has never been run, and the current `--lora-rank-self` flag cannot express rank 0 on cross-attention. **Where the sources disagree** (from the 1-lora page, unresolved): this finding says the compose rate was scored on "the two held-out cat and dog cells", while the rank finding says "four cells per run". The logs are not on the laptop, and both may hold because the findings read different runs (inferred).
- **Evidence:** the [self-attention finding](../../../../../3-understanding-compositionality/report/what-the-adapter-ablations-show/does-adapting-self-attention-change-the-result.md), the [layers chart](../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/which-layers-carry-the-adapter.png), the [1-lora ablations page](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/what-the-adapter-ablations-show.md).

### How much rank the correction needs

- **Asked:** Does a higher-rank adapter (rank 8, 16 or 32, meaning how many patterns the adapter's weights can hold) give a better correction?
- **Paper:** 2
- **Tried:** Three cross-attention runs with the epsilon-space loss: `phase1_r8_200k` (200,000 steps), `phase1_r16_100k` (100,018) and `phase1_r32_100k` (100,050). Each was read by DINOv2 drift at the end of training on the only two cells both tracking sets share, cat and dog seeds 9 and 10. Separately, the paper's settings figure shows rank 8 (at 450,000), rank 16 (at 30,000) and rank 32 (at 30,050) on cat and dog at three held-out seeds, read by eye.
- **Bar, written first:** none recorded. "No bar was pre-registered, so this is a descriptive read."
- **Result:**
  - DINOv2 drift (negative means nearer the joint-prompt render than plain product), rank 8 / 16 / 32: cat and dog seed 9, -0.252 / -0.516 / -0.408. Seed 10, +0.222 / +0.428 / +0.319. At seed 10, no rank moved the cell nearer its joint-prompt render than plain product.
  - Parameters: rank 8 is 4,956,160 of 2,572,419,844 (0.19%), rank 16 is 9,912,320 of 2,577,376,004 (0.38%), rank 32 is 19,824,640 of 2,587,288,324 (0.77%).
  - By eye at seed 12 (settings page, verified): every adapter separates the two animals more than plain product, rank 8 is the haziest, and rank 16 "looks almost the same as rank 32".
- **Verdict:** inconclusive.
- **Why it failed:** Verified: the comparison is mixed. The runs "differ in training length (200k, 100k, 100k steps)" and are read on tracking sets whose other cells differ, so it rests on two cells. On the settings figure, rank 8 is at 450,000 against about 30,000 for the others. Inferred, linking to the training-length entry above: end-of-training drift compares checkpoints past the point where held-out fidelity decays (about 160k for rank 8, 70k for rank 32).
- **Kind of failure:**
  - **test design** (verified): mixed comparison, two cells, no bar.
  - **could not measure it** (verified): the compose rate "reads 1.0 for every configuration".
- **Idea still alive?** retry worth it. The fix named on the page is "Three runs at the same loss, the same pool and the same length, differing only in rank", about 14 hours each on one 49 GB card. Rank 64 (`v57-12-x0-self64`) was stopped at step 42,500 before any evaluation.
- **Rules out:** Do not order ranks from `phase1_r8_200k`, `phase1_r16_100k` and `phase1_r32_100k`, or from the settings figure's rank columns. Do not print a run's end-of-training drift under a picture from an earlier checkpoint.
- **Leaves open:** Whether rank matters at all at equal length and loss. **Where the sources disagree** (from the 1-lora ablations page, unresolved):
  - The rank grids print -0.252, -0.516 and -0.408 under pictures rendered from checkpoints at steps 105k, 50k and 30k. Those numbers are the end-of-training values (200,000, 100,018, 100,050), so "the picture and the number come from different checkpoints of one run" (verified).
  - The card and the rank sidecar's `why_not_clean` say the three runs "differ in loss space". The sidecar's run entries and this finding give all three the epsilon-space loss. The run configs are on the cluster, so this stays unsettled.
  - The compose rate was scored on four cells per run here, against two cells on the self-attention page.
- **Evidence:** the [rank finding](../../../../../3-understanding-compositionality/report/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.md), the [rank chart](../../../../../1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/which-adapter-wins/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.png), the [1-lora ablations page](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/what-the-adapter-ablations-show.md), the [settings page](../../../../../1-lora-correction/report/the-adapter-on-cat-and-dog/how-the-adapter-settings-change-cat-and-dog.md).

### An automatic judge that names the species of each body

- **Asked:** Can an automatic judge, which counts the bodies in a render and then names each one with CLIP, agree with the eye on whether a render shows a cat beside a dog, two of one species, or a hybrid?
- **Paper:** 2
- **Tried:** Two passes over the same 29 renders labelled by eye. The renders come from one trained adapter checkpoint, `v58-05-self-contrast` (a LoRA correction checkpoint on SDXL), on cat and dog, tiger and dog, and cat and fox, with two seeds per pair. The first pass used CLIP ViT-B/32 with two names per body. The second used CLIP ViT-L/14, added "half of each" as a third answer, and redid the eye labels at full size.
- **Bar, written first:** agreement on 27 or more of 30 labelled cells (the walk's bar, `stated`). The labels file holds 29.
- **Result:** First pass agreed on 10 of 23 renders the eye called confidently. Second pass agreed on 13 of 19 (13 / 19 = 0.68, against the bar's 0.90). Per pair on the second pass: cat and dog 9 of 12, tiger and dog 4 of 7, cat and fox no confident render.
- **Verdict:** null (below the bar). The walk records the judge as usable for ranking checkpoints and short of stopping a run on its own.
- **Why it failed:** On the second pass the eye called 2 renders "two of one" that the judge called composed. A striped body with a dog's face (tiger and dog, seed 42) was called "two of one" by the judge and hybrid by the eye. Two white animals (cat and dog, seed 42) were called a pair by the judge and two dogs by the eye (`verified`, from the verdict files). The reference also moved: 16 of 29 rows changed label or confidence when relabelled at full size (`verified`).
- **Kind of failure:** could not measure it: the judge misnames small white animals and striped hybrids, 13 of 19 against a bar of 27 of 30 (`verified`). test design: 29 renders, two seeds per pair, one reader, and a bar written for 30 labels while 29 exist (`verified`).
- **Idea still alive?** retry worth it. The fix the page names: label a larger set at full size (including cat and fox), and settle whether the bar stays 27 of 30.
- **Rules out:** Do not use this judge alone to stop a run or to give a verdict. Do not label reference renders at thumbnail size: 16 of 29 labels changed at full size.
- **Leaves open:** whether the judge works on cat and fox; whether the bar holds now that 29 renders are labelled and 19 are confident.
- **Evidence:** [the species judge finding](../../../../../3-understanding-compositionality/report/what-sdxl-already-knows/can-a-score-see-species.md), [the recipe walk with the bar](../../../../../3-understanding-compositionality/artifacts/ideas/one-training-recipe/IDEA_MAP.md), [an annotated render the judge overcalls](../../../../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/run-02/annotated/a_cat__x__a_dog_s42_v58-05-self-contrast_20000.png) (runs 2 and 4 of the walk)

### Whether each prompt word attends to its own animal

- **Asked:** Does each animal word's cross-attention fall more on its own animal than an even spread would put there, so that a loss on attention has something to push on?
- **Paper:** 2
- **Tried:** Four captures: cat and dog at seed 9 and tiger and dog at seed 42, each with the adapter and with the plain product. For each capture, the share of a word's 32 by 32 attention map that falls inside each GroundingDINO box, averaged over 50 steps.
- **Bar, written first:** none recorded.
- **Result:** Cat and dog, seed 9, with the adapter: the cat box covers 0.270 of the frame and holds 0.289 of the "cat" word's attention (1.07 times an even spread). The dog box covers 0.439 and holds 0.451. In the other three captures the two boxes overlap almost completely (intersection over union of 0.992 to 0.996, on a 0 to 1 scale), so "inside its own animal" and "inside the other animal" are one number.
- **Verdict:** none recorded: the walk reads the shares as showing no preference for the word's own animal (`stated`).
- **Why it failed:** Each word puts about as much attention inside a box as the box's share of the frame (`stated`, the walk's reading). The box detector put both concept boxes on one body in 3 of 4 captures (`verified`, from the overlap values).
- **Kind of failure:** could not measure it: per-concept boxes landed on one body in 3 of 4 captures, overlap 0.992 to 0.996 (`verified`). test design: 4 captures, only one of which can compare a word's own animal with the other, and no bar (`verified`).
- **Idea still alive?** retry worth it. The page's fix is to repeat the read with boxes on separate bodies (instance boxes plus the judge's per-body label).
- **Rules out:** Do not use per-concept GroundingDINO boxes ("a cat", "a dog") as the regions for attention shares. Check the box overlap before reading any share.
- **Leaves open:** whether word attention prefers its own animal in renders where the bodies are separate.
- **Evidence:** [the species judge finding](../../../../../3-understanding-compositionality/report/what-sdxl-already-knows/can-a-score-see-species.md#whether-each-word-looks-at-its-own-animal), [the attention maps for cat and dog at seed 9](../../../../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/can-a-score-see-species/attention-probe/catdog_s9_lora/read/maps_seed_9.png) (run 5 of the walk)

### Does a dog in the rain come out wet under the product

- **Asked:** Does SDXL's joint prompt "a dog and rain" draw a wet dog, while the product of "a dog" and "rain" mostly does not?
- **Paper:** 2
- **Tried:** One SDXL sheet over seeds 1, 2, 4, 9 and 10 with five columns: the dog alone, rain alone, the joint prompt, the product, and SuperDiff (a sampler that reweights the two prompts, run for 200 steps from its own noise). The first four columns used 50 DDIM steps at guidance 7.5. Push recordings were also made on seed 10.
- **Bar, written first:** the joint prompt wets the dog on at least 3 of 5 seeds and the product on at most 1 of 5.
- **Result:** "a wet dog": wet on 5 of 5 seeds (`verified`). "a dog in the rain": wet 4, half 1 (`verified`). Joint prompt: wet 2, half 1 (`stated`). Product: wet 2 (`stated`), and on seed 10 a person with an umbrella replaces the dog. The guess fails whether a half call counts as wet or dry. On seed 10, between 32% and 70% of the joint prompt's push lies outside the flat plane of the dog and rain pushes at all six steps drawn (share of the joint push's length outside the plane). The product cannot produce that part (`stated`).
- **Verdict:** null (the test recorded for this sheet did not hold).
- **Why it failed:** The guess was off in both directions: the joint prompt was less wet and the product more wet than predicted (`verified`, from the counts). The page says the sheet "leaves open why". Seed 10 sits on a switch between a dog and a person that batch shape and graphics card can each flip (`stated`, the walk).
- **Kind of failure:** test design: 5 seeds, no rule written for how a half call counts, and the joint and product counts are taken from the walk with no per-seed calls (`verified`, the walk did not record which seeds). the idea: the guess that the joint prompt reliably wets the dog came out 2 of 5 (`stated`).
- **Idea still alive?** unclear. Five seeds cannot tell 2 of 5 from 3 of 5. Seed-by-seed calls on more seeds (step 7 of the causal route) would tell.
- **Rules out:** Do not settle a 3-of-5 bar from a 5-seed sheet. Write how a half call counts before calling. Do not assume a push recording reproduces a sheet tile: batch shape changes the fp16 output (one recording kept the dog, another matched the person with a pixel gap of 6.5 of 255).
- **Leaves open:** why the joint prompt is dry on most seeds; the walk's pieces 3 to 5 (text embeddings, "raining", "wet" and "water" sheets, seed 10 across four models).
- **Evidence:** [the dog in the rain finding](../../../../../3-understanding-compositionality/report/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet.md), [the dog and rain walk with the test](../../../../../3-understanding-compositionality/artifacts/drips/probing-sdxl/a-dog-and-rain/EXECUTE_WALK.md), [the dog and rain sheet](../../../../../3-understanding-compositionality/artifacts/results/a-thing-in-the-rain/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain.png). Jobs `65969`, `65970` (sheet), `65971`, `65972` (push recordings).

### The product against the joint prompt across kinds of composition

- **Asked:** Does the product match the joint prompt for each kind of composition: structural, relational, causal, shared and temporal?
- **Paper:** 2
- **Tried:** 11 pairs over 4 seeds each, 10 on Stable Diffusion v1-4 and 1 on Liu et al.'s CLEVR relation model. The same pairs on SD3.5 with two joint wordings. Seven SD3.5 grids of 6 pairs by 4 seeds, with the pairs chosen over several selection rounds.
- **Bar, written first:** none recorded.
- **Result:** Counts are out of 4 seeds (`stated`, the five-kinds page). Cat and dog: one fused cat-dog in 4. Dog and oil painting: a near-identical painted dog in 4. A cat behind a tree: no occlusion in any. Wet dog and rain: rain streaks painted over the image. Camel and sunset shadow: a long shadow in none (joint prompt in 2). Butterfly and meadow: among plants in 4 for both. Burning log and smoke: smoke in 3 (joint prompt in 1).
- **Verdict:** none recorded: the sheets show examples without measuring a rate.
- **Why it failed:** No rate was measured. The SD3.5 grids kept rows where the joint prompt shows the interaction and the product fails (`stated`, the selection prompt), so they cannot say how often the product fails.
- **Kind of failure:** test design: no bar, 4 seeds per pair, SD3.5 rows selected for failing, and a mixed comparison across model families. SD v1-4 runs a total guidance of 15 against 4.5 for SD3.5, and the two architectures differ (`stated`).
- **Idea still alive?** unclear. Measuring a failure rate on pairs chosen without looking at the product's output would tell.
- **Rules out:** Do not cite the SD3.5 grids as a failure rate. Do not read SD v1-4 rows and SD3.5 rows as one axis.
- **Leaves open:** how often each kind fails; where the generation scripts are (the card names `scripts/joint_vs_poe_grid/`, which is in neither the repo nor its history).
- **Evidence:** [the per-kind comparison finding](../../../../../3-understanding-compositionality/report/what-sdxl-already-knows/does-poe-match-the-joint-for-each-kind-of-composition.md), [the eleven pairs by seed](../../../../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pairs-by-seed.png)

### Does a camel in the rain come out wet under the product

- **Asked:** Can the product only draw an overlap of two concepts that SDXL saw in training, so that a camel (rarely photographed wet) stays dry under the product of "a camel" and "rain"?
- **Paper:** 2
- **Tried:** SDXL sheets over seeds 1, 2, 4, 9 and 10. One asked four prompts directly (a wet camel, a camel in the rain, a wet dog, a dog in the rain). The other had five camel-with-rain columns: camel, rain, joint prompt, product, and SuperDiff at 200 steps.
- **Bar, written first:** the walk's written test, with a half call counted as dry. Part 1: the product's wet count is no more than the wet count of "a ... in the rain". Part 2: the animal with fewer wet training pictures (by eye) comes out wet on fewer product seeds. The test says "inconclusive" when "a caption count is not reachable".
- **Result:** Camel, over 5 seeds: "a wet camel" wet 4, half 1. "a camel in the rain" dry 4, half 1. Joint prompt dry 5. Product dry 5. SuperDiff broken 3, half 1, cannot tell 1 (`verified`). Part 1 holds for both animals: dog 2 against 4, camel 0 against 0. Part 2 has no input, because no by-eye count of wet training pictures exists.
- **Verdict:** none recorded: the second half of the test has no input. Whether the missing count is "not reachable" (which gives inconclusive) or still to be made (which leaves the test open) is the owner's choice.
- **Why it failed:** The by-eye count of wet camels in training pictures was never made (`verified`). The sources also disagree on seed 9: the reader called the product dry, but the full-size render shows white drops hanging from the camel's chin and neck.
- **Kind of failure:** test design: 5 seeds and a missing input (`verified`). could not measure it: the attempt to count wet animals with CLIP failed its own check (next entries).
- **Idea still alive?** unclear. The by-eye call of the training pictures (step 4 of the causal route) and rendering the product of "a camel" with "wet" would tell.
- **Rules out:** Do not read a dry product camel as caused by missing training pictures, because the product on SDXL also carries sampler error (`stated`). Do not compare the SuperDiff column with its row or with the dog sheet: it differs in starting noise, sampler, step count and graphics card.
- **Leaves open:** whether the seed 9 call stays dry; the 90 uncalled camel renders on the cluster.
- **Evidence:** [the camel in the rain finding](../../../../../3-understanding-compositionality/report/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet.md), [the camel with rain sheet](../../../../../3-understanding-compositionality/artifacts/results/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet/a-camel-and-rain.png), [the walk holding the written test](../../../../../3-understanding-compositionality/artifacts/drips/probing-sdxl/a-dog-and-rain/EXECUTE_WALK.md). Jobs `72389`, `72470`.

### Counting wet animals in caption pictures with CLIP

- **Asked:** Can CLIP pick out the wet animals among public training pictures whose captions name a camel or a dog?
- **Paper:** 2
- **Tried:** 100 camel and 100 dog captions sampled from Re-LAION-2B-en. 63 and 59 pictures still loaded. CLIP said the animal was there in 43 and 47 of those, and called 4 and 3 of them wet. Claude then checked the 7 wet calls by eye.
- **Bar, written first:** precision of at least 80% on the wet calls (the pilot script, line 53).
- **Result:** 0 of the 7 pictures CLIP called wet showed a wet animal by eye. Precision is 0 / 7 = 0%, against the 80% pass mark. 78 of the 200 links no longer load (39%).
- **Verdict:** null (the method failed its own check).
- **Why it failed:** All 7 wet calls were wrong by eye (`stated`, Claude's eye check). Why CLIP was wrong is not recorded. The sources disagree on the denominator: the write-up says 7 of 122 fetched pictures, while the script and the figure count the 90 pictures CLIP accepted as showing the animal.
- **Kind of failure:** could not measure it: 0 of 7 against a pass mark of 80% (`stated`). test design: only 7 wet calls in total (`verified`).
- **Idea still alive?** closed for CLIP as the wetness counter (`inferred`: even on 7 calls, 0% is far from 80%). The question it served stays open through by-eye calls.
- **Rules out:** Do not use CLIP similarity to count wet animals in web photos without a by-eye check.
- **Leaves open:** what CLIP's 7 calls are out of (the saved scores on the cluster would settle it).
- **Evidence:** [the camel in the rain finding](../../../../../3-understanding-compositionality/report/a-thing-in-the-rain/does-a-camel-in-the-rain-come-out-wet.md#what-the-public-captions-hold), [the write-up, section 7.2](../../../../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md#72-the-census-what-the-training-captions-hold), [the pilot script](../../../../../3-understanding-compositionality/scripts/caption_wetness.py)

### How often camel and dog captions mention wetness

- **Asked:** How often do camel and dog captions in SDXL's public stand-in training set use a wet word, and do the pictures behind them show wet animals?
- **Paper:** 2
- **Tried:** Whole-word counts over 2 of the 128 Re-LAION-2B-en files (32,776,409 captions). Animal-name counts over all 128 files. Pictures fetched for 8 caption groups.
- **Bar, written first:** none for the counts. The walk's test needs by-eye calls that were not made.
- **Result:** Captions with a wet word: camel 7 of 12,291 (0.057% of camel captions), dog 781 of 225,544 (0.346%), so dog captions use one 6.08 times as often. 95% Wilson ranges: 0.028% to 0.118% for camel and 0.323% to 0.371% for dog. Weather words: 0.179% against 0.289%. Water words: 1.733% against 1.532%. The 2 files hold within 0.5% of an even share of all 128 on the animal-name counts. Between 42.9% and 68.8% of links per group still gave a picture. None has been called wet or dry.
- **Verdict:** none recorded: no picture has been called wet or dry.
- **Why it failed:** The by-eye tally was never made: `calls.json`, `tally.md` and `tally.png` are not on disk (`verified`). The sources disagree on the search: the write-up gives 2,574 wet-word hits in 4 files, and the saved counts give 788 in 2 files (twice the files holds 3.27 times the hits).
- **Kind of failure:** could not measure it: a caption hit does not show a wet animal, and 1,078 of 2,574 wet-word captions mention food (`stated`). wrong data: the stand-in is not SDXL's training set (`stated`).
- **Idea still alive?** unclear. By-eye calls on the loaded pictures would tell.
- **Rules out:** Do not read a wet-word caption share as a wet-animal share. Do not assume the pictures that still load are a random part of each group.
- **Leaves open:** the word list behind the 2,574; the wet, weather and water counts over all 128 files.
- **Evidence:** [the training captions finding](../../../../../3-understanding-compositionality/report/what-sdxl-already-knows/what-the-training-captions-show.md), [the saved caption counts](../../../../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-captions-say/what-the-training-captions-show/counts.json), [the walk on camels and dogs in the training data](../../../../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-camel-and-dog-in-the-training-data.md)

### Adding "wet" as a third prompt for a camel and a dog

- **Asked:** If the product can only show an effect that the animal's own training pictures hold, does adding "wet" as a third prompt bring wetness back for a dog (often photographed wet) and leave a camel dry?
- **Paper:** 2
- **Tried:** 128 laptop renders on an RTX 4070 Laptop: seeds 1 to 8 by eight prompt columns, for camel and for dog. Column 7 is the product of the animal, "rain" and "wet". Column 8 is the animal, "wet" and "wet", which holds the total guidance weight the same with no rain.
- **Bar, written first:** Counted seeds are those where column 2 or 4 shows the effect and column 5 does not. Supported: the dog is recovered by column 7 on at least half its counted seeds and the camel on at most 1. Falsified: the camel is recovered on 3 or more. Inconclusive: the camel on exactly 2, or either animal has fewer than 2 counted seeds. Column 8 has its own pass marks.
- **Result:** None of the 128 renders has been called. The one recorded remark is the reader's note on camel seeds 1 to 6: "tile 7 wet where 5 and 6 are dry" (`stated`, no per-seed calls).
- **Verdict:** none recorded: 128 renders exist, and none has been called.
- **Why it failed:** It was not read, so nothing has failed yet. The settings also differ from the cluster sheets: the empty prompt is zeros, there is 1 prediction per UNet call, and the card is different (`verified` for the code, `inferred` for the zeroed empty prompt).
- **Kind of failure:** test design: 8 seeds per animal. Column 7 against column 5 is mixed, because it adds the word "wet" and also a third unit of guidance weight (column 8 exists to separate them). Two result combinations fit more than one pass mark. Column 8's marks were written after the reader had seen camel seeds 1 to 6 (`stated`).
- **Idea still alive?** unclear. Calling the 128 tiles against the pass marks would tell.
- **Rules out:** Do not match a laptop tile to a cluster tile: the starting noise, empty prompt, batch and card all differ. Do not compare three-prompt and two-prompt products without the weight-matched column.
- **Leaves open:** whether the cluster rerun should use the laptop's empty prompt or the earlier sheets'.
- **Evidence:** [the effect-needs-support finding](../../../../../3-understanding-compositionality/report/a-thing-in-the-rain/does-an-effect-need-support.md), [the third-concept plan with its pass marks](../../../../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-the-third-concept.md), [the camel sheet](../../../../../3-understanding-compositionality/artifacts/results/a-thing-in-the-rain/does-an-effect-need-support/laptop-third-concept/camel-sheet.jpg)

### What samplers lose from an exact product on 2D points

- **Asked:** On 2D points, what does the exact product of two experts draw for each way their data can overlap, and how much of it do samplers lose?
- **Paper:** both
- **Tried:** Five overlaps plus a ring-times-bar row. Ideal experts were written as formulas and also trained as ResNets. The exact product was computed on a 241 by 241 grid and compared with four samplers (reverse diffusion, ULA, MALA, UHA), 1,500 draws per panel, from one data draw and one seed.
- **Bar, written first:** none recorded.
- **Result:** Joint share means the fraction of draws within 0.15 of the point both experts hold. At joint weight 0.05: perfect draws 0.91, reverse diffusion 0.26, ULA 0.24, MALA 0.23, UHA 0.31. Total variation from the exact answer (0 identical, 1 no overlap): 0.10 for perfect draws against 0.61 to 0.68 for the samplers. At joint weight 0.2: 0.97 against 0.70 to 0.88. At joint weight 0.01, the product's share on the joint point falls from about 1 to near 0 as the mode spread goes from 0.05 to 0.1 (read off the panel).
- **Verdict:** none recorded: pictures with no pass rule.
- **Why it failed:** It did not fail. Every sampler draws the shared point less often than the exact product does (`verified`, sampler numbers). Why is not measured here.
- **Kind of failure:** none recorded as a failure. test design weakens it: one data draw and one seed per cell, so small differences between samplers are not established (`stated`).
- **Idea still alive?** The reading that samplers lose the shared point stands on one seed. More seeds, and total variation for the main table, would size the gaps.
- **Rules out:** Do not rank the samplers from this table. Do not read the ideal ring row as matching its data: the ideal expert is wider than the ring's own spread (0.05 smoothing against 0.03).
- **Leaves open:** total variation for the main table; why the disjoint-support row is empty in one table and a thin ridge in the others.
- **Evidence:** [the two-experts finding](../../../../../3-understanding-compositionality/report/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose.md), [the sampler run picture](../../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/sampler-run.png)

### How much co-occurrence the exact product and its samplers need

- **Asked:** How much weight on the two-shape picture inside each expert does the exact product need to prefer that picture, and how much do its samplers need?
- **Paper:** both
- **Tried:** A circle expert and a square expert, each an exact mixture of Gaussians over 16 by 32 colour pictures. Pair weight from 0 to 0.99. The exact product, reverse diffusion and annealed MALA, 300 draws each. Reverse diffusion was also run with the true score of the noised product in place of the summed noised scores.
- **Bar, written first:** a table in the plan, written before any run. This page does not reproduce it, and the owner's step of reading the figure against it is unticked.
- **Result:** Crisp pairs out of 300. Exact product: 300 at every weight from 0.001. Reverse diffusion: 0, 10, 122, 238, 300 at weights 0.001, 0.1, 0.3, 0.5, 0.9. MALA: 0, 30, 140, 249, 300. With summed noised scores: 0, 0, 15, 109 at weights 0.001, 0.01, 0.1, 0.3. With the true noised-product score: 300 at each of those weights.
- **Verdict:** none recorded: the plan's reading step is unticked.
- **Why it worked:** The pair is lost where the experts' noised scores are added. The same sampler given the true noised score draws 300 of 300 (`verified`).
- **Kind of failure:** none: diagnosis. Weakening: one spread (σ = 0.1) fixes every number, and every crisp pair is the one picture both experts were given (`stated`).
- **Idea still alive?** On the toy this licenses "the sampler, not the product, loses the pair". It does not say where a real model sits.
- **Rules out:** Do not blame the exact product for missing pairs here, since it composes at weight 0.001. Check summed scores against the true noised score first. Do not quote "three orders of magnitude": the grid gives log10 gaps of 2.48 to 2.70 from its floor, or 4.48 to 4.70 from the closed form's lowest weight.
- **Leaves open:** Reverse diffusion's count at weight 0.1 differs between runs (10, 14, 15, 18).
- **Evidence:** [the co-occurrence finding](../../../../../3-understanding-compositionality/report/the-exact-toy/how-much-co-occurrence-the-product-needs.md), [the co-occurrence plan](../../../../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-how-much-co-occurrence-the-product-needs.md), [the weight sweep](../../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/weight-sweep.png)

### Experts that like each other's shapes, and captions that compete

- **Asked:** How much may each expert put on the other's lone shapes before the exact product stops drawing the pair, and what happens when three competing captions set the overlap?
- **Paper:** both
- **Tried:** A leak sweep at pair weight 0.1, where the leak is each expert's weight on each of the other's three shapes. A competing-caption product, compared with a hand-weighted product at the same implied weight. 300 draws each.
- **Bar, written first:** the plan's table. No reading of the competing-caption rule is written down.
- **Result:** A leak of 0.00567 undoes a pair weight of 0.1. Exact crisp pairs out of 300: 300, 201, 113, 20 at leak 0, 0.003, 0.01, 0.1. Reverse diffusion: 10, 9, 9, 8. Competing captions imply pair weights of 0.0461 and 0.0458. At those weights the hand-set product draws 300 of 300, and the competing-caption product draws 4 and 19 crisp, with 296 and 281 single shapes.
- **Verdict:** none recorded.
- **Why it failed:** The card's account is that each expert gives the other's lone shapes more weight than the pair (`stated`, argued, not measured).
- **Kind of failure:** the idea: here the exact product itself, not a sampler, stops drawing the pair (4 and 19 of 300, `verified`). Weakening: one spread fixes the numbers, and the caption model's support is a modelling choice (`stated`).
- **Idea still alive?** unclear. Measuring whether a real "a cat" expert's weight on dog pictures sits above or below the toy's 0.00567 leak would tell.
- **Rules out:** Do not expect a sampler fix to rescue composition when each expert puts weight on the other's lone shapes: the exact product draws single shapes there.
- **Leaves open:** the unexplained `divide by zero in log` warning in the competing-caption run.
- **Evidence:** [the co-occurrence finding](../../../../../3-understanding-compositionality/report/the-exact-toy/how-much-co-occurrence-the-product-needs.md#how-much-the-experts-may-disagree), [the leak dial](../../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/the-leak-dial.png), [the competing-captions draws](../../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/competing-captions.png)

### Dividing the product by a background expert

- **Asked:** Can circle times square, divided by a background expert, draw layouts that no training picture holds?
- **Paper:** both
- **Tried:** A hand-written toy on grey and on white backgrounds. Exact draws, the true noised score and summed scores, compared with a joint caption. 300 draws each.
- **Bar, written first:** none on this page.
- **Result:** On grey, draws landing on the four unseen layouts: 202 of 300 exact, 197 with the true noised score, 210 with summed scores, against 0 for a joint caption. On white: 0, 0 and 9. The plain product puts 145 and 155 on the two trained layouts and 0 on the two unseen ones.
- **Verdict:** support (the recorded verdict for that test).
- **Why it worked:** The page records the counts and does not explain the mechanism. The model is written by hand (`stated`).
- **Kind of failure:** none, the page records support, but no bar was written first, so test design applies as well. Limits: it fails on white (0, 0 and 9 of 300, against 202, 197 and 210 on grey), and the model is written by hand, so it says nothing about how a captioned model learns (`stated`).
- **Idea still alive?** alive, only for a hand-written toy on a grey background.
- **Rules out:** Settles that the plain product returns only layouts its experts hold (0 on the unseen layouts).
- **Leaves open:** why white fails; any learned version.
- **Evidence:** [the co-occurrence finding](../../../../../3-understanding-compositionality/report/the-exact-toy/how-much-co-occurrence-the-product-needs.md#whether-it-can-draw-a-layout-nobody-trained-on), [the unseen-layout draws](../../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/can-the-product-draw-a-layout-nobody-trained-on.png)

### Two separately trained shape networks

- **Asked:** Do two small networks, trained separately (one per shape), compose a separate pair as the share of training pictures they hold in common rises?
- **Paper:** 1
- **Tried:** Two trained experts at shared fractions from 0 to 0.5, 6,000 training steps, 300 samples per setting. A separate preflight check trained one expert alone.
- **Bar, written first:** none on this page.
- **Result:** 0 of 300 separate pairs at every shared fraction. The preflight check's draws are noise (final loss 0.58087), yet the nearest-picture labeller still names 169 of them "one circle" at 6,000 steps. The trained experts file reports 300 of 300 own shape at the same 6,000 steps (final loss 0.01287).
- **Verdict:** null.
- **Why it failed:** Unresolved. The null rests on the second run. The check probably ran on an earlier version that predicted noise rather than the picture (`inferred`), and both scripts arrived in one commit (`734a074e`), so the order is unrecorded.
- **Kind of failure:** could not measure it: the nearest-picture labeller names pure noise as shapes, 169 "one circle" (`verified`). done wrong is possible: the two runs disagree about whether one expert even reproduces its own pictures (`verified` that they disagree).
- **Idea still alive?** unclear. Establishing why the check draws noise while the trained experts file reports clean shapes would tell.
- **Rules out:** Do not trust the nearest-picture labeller without looking at the draws. Check that one expert reproduces its own pictures before reading a product null.
- **Leaves open:** whether the null for two trained networks stands.
- **Evidence:** [what product of experts needs from training](../../../../../3-understanding-compositionality/report/the-exact-toy/what-poe-needs-from-training-to-draw-two-things-at-once.md#two-separately-trained-networks), [the one-expert check draws](../../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/can-one-expert-learn-its-own-pictures.png)

### The caption model with a random share of pictures relabelled

- **Asked:** In a small captioned model, does filing a random share of both-shapes pictures under each single caption as well let the product draw a separate pair?
- **Paper:** 2
- **Tried:** The dial (the share of both-shapes pictures also trained under each single caption) at 0, 0.1, 0.3, 0.6 and 1. Three trainings per setting, 300 samples. A second sweep over guidance 1, 2, 3, 5 and 7.
- **Bar, written first:** Support: at least 150 separate pairs while both single captions stay at 240 or more. Null: under 30 wherever both captions pass. Inconclusive: a rise only where single-caption purity has already fallen.
- **Result:** Best setting with both single captions intact: 118 of 300. Separate pairs at dials 0.1, 0.3, 0.6: 118, 114, 55. Three trainings at guidance 3 give 118, 149, 110 at dial 0.1 and 174, 226, 205 at dial 1. Guidance sweep: at dial 0 no cell passes 1 of 300, and other cells range from 1 to 264 with no best guidance across dials.
- **Verdict:** inconclusive, as the script prints it. The result falls outside the three written outcomes, and the owner has not chosen a reading.
- **Why it failed:** 118 sits between null (30) and support (150), and the rise happens where purity holds, which no written outcome covers (`verified`).
- **Kind of failure:** test design: the written outcomes do not cover the result, and there are only three trainings per setting, with the largest-minus-smallest gap reaching 52 at one dial (`verified`).
- **Idea still alive?** unclear. A reading chosen for this case, and more trainings per setting, would tell. The fixed-size relabelling design under project 4 is the follow-on.
- **Rules out:** Write outcomes that cover every result between support and null. Do not read one training per setting. Stronger guidance does not rescue a model with no relabelled pictures (dial 0: no cell above 1 of 300).
- **Leaves open:** The card says two guidance cells meet the rule, and the file shows three.
- **Evidence:** [what product of experts needs from training](../../../../../3-understanding-compositionality/report/the-exact-toy/what-poe-needs-from-training-to-draw-two-things-at-once.md#what-the-caption-model-draws), [what the caption model draws](../../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/what-the-caption-model-draws.png)

### Does a fused cat-dog fit both captions better than a photograph

- **Asked:** Does a fused cat-dog render score higher than real photographs on the weaker of "a photo of a cat" and "a photo of a dog", as the explanation that the product rewards an ambiguous animal predicts?
- **Paper:** 2
- **Tried:** CLIP scored 137 images against three phrases: 91 COCO photographs, 7 fused product renders, 32 renders of the one sentence "a cat and a dog", and 7 renders of the product of "cats" and "dogs". Nothing was trained.
- **Bar, written first:** the claim holds if the product renders sit up and to the right of the photographs, and fails if the photographs sit at least as high on the weaker caption. No minimum gap was set.
- **Result:** Weaker single-animal cosine similarity (scale -1 to 1), mean ± sd: photographs 0.226 ± 0.021, fused 0.240 ± 0.017, one sentence 0.243 ± 0.016, "cats" times "dogs" 0.226 ± 0.020. Fused minus photographs: +0.014 (0.66 sd). One sentence minus photographs: +0.017. On the pair phrase, fused scores 0.241 against 0.310 for one sentence, and the sets do not overlap (fused maximum 0.264, one-sentence minimum 0.270).
- **Verdict:** none recorded. Read as written it gives support. Read against the one-sentence renders it neither supports nor refutes. The choice is the owner's.
- **Why it failed:** Renders with two separate animals also beat the photographs, by more (0.017 against 0.014) (`verified`).
- **Kind of failure:** test design: the comparison is mixed, because generated against photographed changes together with fused against two animals. The fused set is only 7 images, and resolutions differ (1024 px, 320 px crops, about 640 px) (`stated`). could not measure it: a similarity is one number per phrase and cannot say where in the image the match came from (`stated`).
- **Idea still alive?** unclear. Rendering the one-sentence prompt from the cached noise on seeds 1 to 7, then comparing within each seed, would tell.
- **Rules out:** Do not test the fusion explanation against real photographs alone. Include generated two-animal renders from the same starting noise.
- **Leaves open:** whether seed numbers mean the same starting noise across the two render sets (at seed 19 the sheet tile and the repo render differ by 26.47 of 255).
- **Evidence:** [the hybrid caption fit finding](../../../../../3-understanding-compositionality/report/what-sdxl-already-knows/does-the-hybrid-fit-both-captions.md), [the plan with its rule](../../../../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/does-the-hybrid-fit-both-captions.md), [the summary file](../../../../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/summary.json)

### Giving each prompt its own region of the picture

- **Asked:** If each prompt is given its own region, taken from a render where the joint prompt already drew both animals, does that alone give two clear animals?
- **Paper:** 2
- **Tried:** Boxes were taken from joint-prompt renders: 8 of 24 candidate seeds were usable, and 4 of those needed a name corrected by hand. Each box became a soft mask. One preview was rendered on seeds 9 and 19, plus an identity check that flat masks reproduce the plain product at half guidance.
- **Bar, written first:** Support: the layout reaches 4 or more clear pairs of 8 and random boxes reach fewer. Null: 1 or 0, after running MultiDiffusion's own code. Inconclusive: 2 or 3. Further rows cover random boxes and swapped prompts.
- **Result:** The six-column comparison never ran. Preview: the layout alone draws one animal on both seeds. With 15 opening steps against a flat colour, a cat and a dog appear as cut-outs on grey (`inferred`, by eye, 2 seeds). Identity check: mean absolute difference of 1.531 grey levels of 255, against the script's limit of 3.0 (largest single difference 121). Box naming: 12 of 16 CLIP names were final, 2 named the wrong animal, 2 were uncertain.
- **Verdict:** not run.
- **Why it failed:** It was not run.
- **Kind of failure:** none observed yet. Known risks on the page: the boxes come from renders that already succeeded, and the seeds were chosen for clean boxes (8 of 24) (`stated`).
- **Idea still alive?** unclear. Running the six-column sheet on the eight seeds would tell.
- **Rules out:** Settles that the regional sampler reproduces the plain product (1.531 of 255, inside the 3.0 limit). Do not count the two-seed preview as evidence.
- **Leaves open:** which reading of the usable-seed rule applies (4 seeds on CLIP names alone, 8 with hand corrections).
- **Evidence:** [the layout finding](../../../../../3-understanding-compositionality/report/what-sdxl-already-knows/is-the-layout-the-missing-piece.md), [the layout plan with the bar](../../../../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/is-the-layout-the-missing-piece.md), [the preview](../../../../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/is-the-layout-the-missing-piece/overlap-readings.png)

### A classifier push on the exact toy

- **Asked:** Can the gradient of a classifier trained on what the sampler gets wrong turn the sampler's stacked draws into side-by-side pairs, with the experts frozen?
- **Paper:** both
- **Tried:** Pair weight 0.01, seed 7, 300 draws per count. A discriminator was trained on 3,500 exact draws against 3,500 sampler draws (500 of each held out) and compared with a separate-label classifier. The push was added at every step.
- **Bar, written first:** (plan 19) at least 150 crisp pairs at push weight 1 with no more overlaid than unguided, and at most 15 crisp at every scale for the separate-label push.
- **Result:** Crisp pairs of 300. Unguided: 0 crisp, 300 overlaid. True correction at scale 1: 300. Discriminator at push weights 0.5, 1, 2: 16, 172, 300. Trained separate-label classifier at scales 1, 2, 5, 10: 0, 0, 2, 10. Closed-form best separate-label classifier: 0, 1, 13, 34. With no push, raising the pair weight to 0.3 gives 115, against 172 for the push at 0.01.
- **Verdict:** support (pass, read on the trained classifier, the owner's choice). Read on the closed-form best classifier, the separate-label half misses (34 against a limit of 15).
- **Why it worked:** The push follows the sampler's own error: its median cosine with the sampler's own ratio is 0.96 at step 190 and 0.85 at step 100, and with the true correction at most 0.21 (`verified`). The discriminator prefers the pair by 27.9 log-odds at step 10, against 3.6 for the separate-label classifier (`stated`).
- **Kind of failure:** none, it passed its bar. test design limits it: every count is one seed (7) of 300 draws, and the labels check the picture and not whether the pairs land in the exact product's positions and proportions (`stated`). The separate-label half passes only when read on the trained classifier (10 crisp against a limit of 15); on the closed-form best classifier it reads 34 (`verified`).
- **Idea still alive?** alive, on the exact toy only, where the exact product already composes. It repairs a sampling failure and says nothing where the product's own target is a hybrid, as under competing captions (`stated`).
- **Rules out:** Settles that the toy's sampler loss can be repaired without retraining the experts. Do not expect this push to help where the product itself draws single shapes.
- **Leaves open:** positions and proportions of the crisp pairs; counts on more than one seed; the owner's own look at the two diagnosis charts.
- **Evidence:** [the classifier push finding](../../../../../3-understanding-compositionality/report/the-exact-toy/can-a-classifier-push-fix-the-product.md), [finding A1](../../../../../2-classifier-guidance/report/classifier-guided-poe/a1-exact-toy-classifier-push.md), [plan 19 with the pass rule](../../../../../2-classifier-guidance/plans/partial/19-classifier-push-on-the-exact-toy.md), [the discriminator in seven stages](../../../../../3-understanding-compositionality/artifacts/results/the-exact-toy/can-a-classifier-push-fix-the-product/the-side-by-side-discriminator.png) (plan 19, A1)

### Reward fine-tuning the adapter, first runs (refl-03 and refl-04-fidelity)

- **Asked:** Does fine-tuning the correction adapter on a plurality reward (identity, distinctness and picture quality) give a clean two-animal render?
- **Paper:** 2
- **Tried:** `refl-03-self-contrast-15k` (W&B `95o900mc`) fine-tuned on the reward, starting from the raw weights of v58-05 at step 15,000. v58-05 is a distillation-trained adapter with self- and cross-attention layers and a contrast term. The run trained on exactly the two pairs being judged, at a constant learning rate of 1e-5 with no EMA. `refl-04-fidelity` added the ImageReward picture-quality term. The number of pictures is not recorded.
- **Bar, written first:** none recorded.
- **Result:** refl-03's reward rose through the distinctness term alone (0.40 to 0.54), while identity stayed at about 0.27. The ImageReward term had weight 0 because it did not exist yet. The gradient sat above the clip on nearly every step, and the run crashed at 1,331 steps. refl-04-fidelity crashed at step 49. The opening prompt's table credits refl-03 with "the first clean elephant and penguin", checkpoint unknown (250 to 1,250 exist). A separate progress prompt records "the fidelity term that was silently off in reward fine-tuning before commit `a72f748`".
- **Verdict:** none recorded: the opening prompt says "The reward idea has not had a fair test."
- **Why it failed:** `inferred` (the source states it, nothing in a run measured it): the reward rose through distinctness alone, the picture-quality term was missing, and both runs crashed before a planned length.
- **Kind of failure:** Done wrong (stated: ImageReward weight 0 on refl-03, the fidelity term silently off before `a72f748`, crashes at 1,331 and 49 steps). Test design (stated: trained on exactly the pairs being judged, no held-out pairs, no bar).
- **Idea still alive?** Retry worth it. The idea map names the fix: identity scored as the worse of the two concepts, ImageReward on, an L2 or KL anchor to the distillation adapter it starts from, EMA, and pairs held out of the reward (run R4, which "waits for two additions to the reward trainer").
- **Rules out:** Do not read refl-03's rising reward as progress: it rose through distinctness while identity stayed flat. Check the picture-quality term is actually non-zero before launching a reward run. Hold the judged pairs out of the reward.
- **Leaves open:** Whether a reward run with the four fixes buys fidelity without costing composition (claim 5 of the idea map, "needs a check").
- **Evidence:** [the recipe walk's opening prompt](../../../../../3-understanding-compositionality/artifacts/ideas/one-training-recipe/OPENING_PROMPT.md) and [the one-training-recipe idea map](../../../../../3-understanding-compositionality/artifacts/ideas/one-training-recipe/IDEA_MAP.md). Neither is a report finding or a review file, and no review file exists for these runs. W&B `95o900mc`.

### Eight ways of asking SDXL for cats and dogs

- **Asked:** Does the product of "cat" and "dog" match the joint prompt, and does any wording draw separate animals?
- **Paper:** 2
- **Tried:** One SDXL sheet over seeds 1 to 7 with eight columns: three single prompts and five products. Each row starts from the same noise, with 50 steps and guidance 7.5 per prompt.
- **Bar, written first:** none recorded.
- **Result:** One by-eye reading at sheet scale (`inferred`). Product of "cat" and "dog": one fused animal on 6 of 7 seeds. Joint prompt "cats and dogs": illustrated crowds on 7 of 7. Product of "dog and cat" and "multiple animals": separate animals in a photo on 3 (seeds 2, 6, 7). Product of "dog", "cat" and "single": one fused animal on 7.
- **Verdict:** none recorded: one by-eye reading, with no card, no pass rule and no per-tile calls.
- **Why it failed:** No measurement was made.
- **Kind of failure:** test design: 7 seeds and no bar. Column 8 is mixed, because its first prompt already names both animals. Columns differ in total guidance (7.5, 15, 22.5) as well as wording (`stated`). could not measure it: sheet-scale reads overcall composition (`stated`).
- **Idea still alive?** unclear. Full-size calls per tile and a rerun at matched total guidance would tell.
- **Rules out:** Do not credit column 8's separate animals to the product. Compare columns only at matched total guidance.
- **Leaves open:** whether any wording composes at a rate.
- **Evidence:** [the cats and dogs finding](../../../../../3-understanding-compositionality/report/what-sdxl-already-knows/does-poe-of-cats-and-dogs-match-the-joint.md), [the cats and dogs sheet](../../../../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/cats-and-dogs-sheet.jpg)

### Does base SDXL's cat prompt already draw dogs

- **Asked:** Does base SDXL's "a photo of a cat" already draw dogs, which would set a floor under the inclusive-captions effect?
- **Paper:** 2
- **Tried:** 64 renders (seeds 0 to 63) from base SDXL at 1024 px, 50 DDIM steps and guidance 7.5. Each was read three ways: by eye, by the detector asked for "dog", and by a zero-shot CLIP classifier on the cropped animal and on the whole image.
- **Bar, written first:** The census script sets 0 or 1 dog hits as a clean floor, 2 to 4 as a weak floor, and 5 or more as the base model already leaking dogs. The classifier has its own three bars: at least 62 of 64 crops called cat, a median margin of at least 0.50, and anything under a 0.20 margin reported uncertain.
- **Result:** By eye, 0 of 64 hold a dog. The detector reports one dog on 64 of 64, with median confidence 0.889 for "dog" against 0.943 for "cat". Its cat-minus-dog margin has median 0.054, 54 renders are under 0.10, and "dog" scores higher on 3. The classifier calls 64 cat, 0 dog, 0 uncertain, with median margin 0.990. The whole-image read gives the same.
- **Verdict:** Two readings. With the detector, 64 hits means the base model leaks dogs. With the classifier or the eye, 0 hits means a clean floor. Which one settles the floor is the owner's choice. The classifier cleared its own three bars (support: "the instrument works").
- **Why it failed:** The detector cannot tell these cats from dogs: it puts a "dog" box on every single-cat render (`verified`). This is the detector that counts bodies but cannot name the animal.
- **Kind of failure:** none, it passed its bar (the crop classifier's three bars: 64 of 64 called cat, median margin 0.990). could not measure it applies to the other instrument: the detector reports a dog on 64 of 64 single-cat renders (`verified`).
- **Idea still alive?** alive, for naming the animal in single-animal renders. The classifier is not validated on two-animal frames: in the layout experiment, 12 of 16 boxes matched the final name, 2 were wrong and 2 were uncertain.
- **Rules out:** Do not count "dog hits" with the per-class detector.
- **Leaves open:** the crop classifier's accuracy on labelled two-animal renders.
- **Evidence:** [the cat prompt finding](../../../../../3-understanding-compositionality/report/what-sdxl-already-knows/does-the-cat-prompt-already-draw-dogs.md), [the crop classifier sheet](../../../../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-prompts-draw/does-the-cat-prompt-already-draw-dogs/crop_classifier_sheet.png), [the census script with the floor rule](../../../../../3-understanding-compositionality/scripts/census/cat_condition_dog_rate.py)

## Project 4: inclusive captions

### How many cat-and-dog photos the pilot can train on

- **Asked:** How many COCO photos holding both a cat and a dog can the inclusive-captions pilot train on?
- **Paper:** 2
- **Tried:** Counted COCO train2017, applied a size filter (each animal's box at least 0.25 of the image's longer side), and had the assistant audit the 131 survivors by eye at 448 px. A second pool used a vision-language judge plus generated photos.
- **Bar, written first:** The target was 300 cat-only, 300 dog-only and 200 both-animal photos. Plan 01's audit step says more than about 20 rejected both-animal photos means the single-name caption shares must rise before training.
- **Result:** 208 both-animal photos in COCO. 131 after the size filter (69 short of the target). The eye audit kept 91, rejected 33 and left 7 uncertain. Second pool: 38 of 64 COCO candidates kept by the judge and 112 generated photos used (243 of 306 passed the judge). Trained on: 150 both-animal, 200 cat and 200 dog, 550 in all.
- **Verdict:** none recorded as support or null. The counts file records "131 usable both-animal photos", and the audit then rejected 33, above the plan's limit of about 20.
- **Why it failed:** COCO holds too few clean both-animal photos. Rejected tiles include drawings, collages and photos with one animal (`verified`, audit verdicts).
- **Kind of failure:** wrong data: 208 labelled both-animal photos became 91 clear by eye (`verified`).
- **Idea still alive?** The count is settled. The pilot went ahead with 150, of which 112 were generated, so the idea itself is tested under the pilot entry.
- **Rules out:** Do not take COCO's both-label count as usable photos. Do not reuse a criterion written for the 491-photo pool without checking it against the 550 trained.
- **Leaves open:** the 7 uncertain tiles; an audit of the single-animal sets.
- **Evidence:** [the photo pool finding](../../../../../4-inclusive-captions/report/what-the-training-data-holds/how-many-photos-the-inclusive-captions-pilot-can-train-on.md), [the counts file](../../../../../4-inclusive-captions/artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/counts.json), [plan 01](../../../../../4-inclusive-captions/plans/11-do-inclusive-captions-help-poe-compose/plans/01-the-inclusive-captions-pilot.md) (plan 01)

### Do public captions say a cat and a dog, and do the pictures show both

- **Asked:** How many Re-LAION-2B-en captions name both a cat word and a dog word, and do their pictures show both animals?
- **Paper:** 2
- **Tried:** A word search over 2,097,693,557 captions, then 60 captions drawn at random from each of four caption forms (seed 20261103) and their pictures fetched for marking.
- **Bar, written first:** none. The check page asks for B, O or N per picture and sets no share of B that would count.
- **Result:** 1,102,469 captions name both (0.053% of all captions). Forms: bare 4,971 (0.45%), short 423,432 (38.41%), mid 588,332 (53.37%), long 85,734 (7.78%). Of 240 pictures drawn, 155 loaded, and the page lists 117. 0 of 117 rows are marked.
- **Verdict:** none recorded.
- **Why it failed:** No row was marked (`verified`).
- **Kind of failure:** test design: no share written first (`verified`). could not measure it: a caption naming both animals says nothing about the picture, and 85 of 240 links are dead (`stated`).
- **Idea still alive?** unclear. Writing the share of B that counts and then marking the 117 rows would tell.
- **Rules out:** Do not read a caption naming both animals as a picture holding both.
- **Leaves open:** whether the 155 web pictures, with unchecked licences, may stay.
- **Evidence:** [the caption search finding](../../../../../4-inclusive-captions/report/what-the-training-data-holds/does-any-caption-say-a-cat-and-a-dog.md), [the check page](../../../../../4-inclusive-captions/artifacts/results/does-any-caption-say-a-cat-and-a-dog/caption-check.md)

### SDXL fine-tunes that caption both-animal photos under each name

- **Asked:** Does training SDXL with both-animal photos also captioned "cat" and "dog" (run B) make the product of "a cat" and "a dog" compose more often than captioning them only "a cat and a dog" (run A)?
- **Paper:** 2
- **Tried:** Two LoRAs on the same 550 photos (200 cat, 200 dog, 150 both). Run A captions a both-animal photo "a cat and a dog" 90% of the time and blank 10%. Run B uses "cat" 40%, "dog" 40%, joint 10% and blank 10%. A sheet of 6 seeds (0, 1, 3, 5, 6, 7) and 8 watching seeds read through step 2,000.
- **Bar, written first:** (review file) On the same 128 seeds: support if run B's compose rate exceeds run A's by at least the band (two standard errors of their difference); no support if B is at or below A; inconclusive if B is above A by less than the band. A fourth outcome: all three rates below 0.10 and run B's cat column worse by eye on 5 or more of 8 sheet seeds.
- **Result:** 0 of 6 seeds show two animals in any row (plain SDXL, run A, run B). On the 8 watching seeds the product shows two animals on 0 of 8 at every checkpoint. Run A at step 2,000 draws three fused heads, and run B mostly clean single animals. No compose rate exists. Run A's 128 scoring renders were discarded at 52 seeds after a contrast bug, and run B's were never started (`stated`).
- **Verdict:** not run (the planned 128-seed test). The card says "a sheet and not a verdict".
- **Why it failed:** The scoring pass was halted by a contrast bug in both renderers (`stated`).
- **Kind of failure:** test design: 6 seeds by eye cannot give a rate (`stated`). done wrong: the renderer contrast bug forced the discard (`stated`). could not measure it: a cleaner animal may be a less degraded one and no better composed (`stated`). The photo pool also changed from 491 to 550 after the criterion was written.
- **Idea still alive?** unclear. The 128-seed scoring pass would tell, once the owner decides the criterion applies to 550 photos. The crop classifier still needs testing on two-animal frames.
- **Rules out:** Do not read run B's cleaner single animals as movement toward composition. Check the renderer's contrast before a scoring pass.
- **Leaves open:** the compose rates. How often run B's "cat" draws hold a dog is recorded as both 16.9% and 25.1%.
- **Evidence:** [the SDXL pilot finding](../../../../../4-inclusive-captions/report/do-inclusive-captions-help-the-product-compose/does-captioning-both-animal-photos-under-each-name-help-poe-compose.md), [the review file with the criterion](../../../../../4-inclusive-captions/plans/11-do-inclusive-captions-help-poe-compose/review/01-does-captioning-widen-the-cat-condition.md), [the six-seed sheet](../../../../../4-inclusive-captions/artifacts/results/does-captioning-both-animal-photos-under-each-name-help-poe-compose/poe-cat-x-dog-plain-run-a-run-b-six-seeds.png) (plan 01)

### Smallest number of relabelled pictures, adding to the set

- **Asked:** In the toy caption model, what is the smallest number of both-shapes pictures, added under each single caption, at which the product draws the pair in all three trainings?
- **Paper:** 2
- **Tried:** 0, 1, 5, 20, 50, 200, 600, 1,200 and 2,000 relabelled pictures added to 2,000 per single caption. Three trainings each, 300 product samples.
- **Bar, written first:** A training draws the pair at 30 or more of 300 separate pairs while each single caption keeps 240 of 300 in its own colour. The threshold is the smallest count where all three trainings do.
- **Result:** `threshold: null`. From 0 to 50 relabelled, at most 3 of 300 across 15 models. At 200: 1, 153, 0. At 600: 157, 17, 134. At 1,200: 183, 29, 57. At 2,000: 252, 74, 238, where the third training's single caption falls to 223. The joint caption draws the pair in at least 267 of 300 in all 27 models.
- **Verdict:** null (no threshold found; "one picture is enough" contradicted).
- **Why it failed:** Trainings disagree widely at the same setting: 252 against 74 at 2,000, a gap of 178 (`verified`).
- **Kind of failure:** test design: three trainings per number. The comparison is mixed, because adding pictures also grows the set from 6,000 to 10,000 at a fixed 8,000 training steps (`stated`).
- **Idea still alive?** retry worth it. The fix is to hold the set size fixed, which the replacing design (next entry) does.
- **Rules out:** Do not add relabelled pictures without holding the set size. Do not order neighbouring counts from three trainings.
- **Leaves open:** The card's figure entry orders the counts at 200 as 153, 1, 0, while the file has 1, 153, 0.
- **Evidence:** [the toy relabelling finding](../../../../../4-inclusive-captions/report/do-inclusive-captions-help-the-product-compose/does-relabelling-both-shapes-pictures-help-the-toy-product-compose.md), [what the product draws at each relabelled count](../../../../../4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/what-poe-draws-at-each-relabelled-count.png)

### A Langevin corrector on the toy caption model, adding design

- **Asked:** Does a Langevin corrector raise how often the product draws the pair in models trained with relabelled pictures, and does it work in models without them?
- **Paper:** 2
- **Tried:** Four models with relabelled pictures and four without. 5 and 20 corrector steps at every noise level, each step 0.05 of that level's noise variance.
- **Bar, written first:** support if the median of four trainings rises by 30 or more with 5 corrector steps, with each single caption at 240 or more.
- **Result:** Separate pairs of 300: median 260.5 plain (plain counts 67, 252, 269, 290) and 300 with 5 steps. One 2,000-picture model reaches only 162 at 5 steps and 300 at 20. Without relabelled pictures (0 and 50), 0 of 300 plain and corrected. Dial corrector, one model per row: plain 0, 110, 103, 51, 205, and with 20 steps 0, 300, 300, 300, 300.
- **Verdict:** support (with relabelled pictures). null (without them).
- **Why it worked:** The single captions let the product find the pair, and the corrector raises how often it does (`verified`, counts).
- **Kind of failure:** none, it passed its bar (median of four trainings 260.5 plain, 300 with 5 corrector steps). Limits: the same corrector gives 0 of 300 in models without relabelled pictures, and the dial-corrector rows are one model each (`verified`).
- **Idea still alive?** alive, as an amplifier of a pair that training already made reachable. It cannot create the pair from zero, and with the set size fixed it misses its rule at four of the seven shares tested (see the fixed-size corrector entry).
- **Rules out:** Do not expect a corrector to compose where single captions hold no both-shapes pictures.
- **Leaves open:** whether this holds with the set size fixed (next entries).
- **Evidence:** [the toy relabelling finding](../../../../../4-inclusive-captions/report/do-inclusive-captions-help-the-product-compose/does-relabelling-both-shapes-pictures-help-the-toy-product-compose.md#2-what-the-corrector-adds), [the corrector with and without relabelling](../../../../../4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/the-corrector-with-and-without-relabelling.png), [what product of experts needs from training](../../../../../3-understanding-compositionality/report/the-exact-toy/what-poe-needs-from-training-to-draw-two-things-at-once.md#what-a-corrector-adds)

### Share of both-shapes pictures with the set size fixed

- **Asked:** With the training set held at 6,000 pictures, what share of both-shapes pictures under each single caption makes the toy product draw the pair?
- **Paper:** 2
- **Tried:** 1 to 1,000 of each single caption's 2,000 pictures were swapped for both-shapes pictures. Three trainings each, 300 samples. A size check ran at a 50% share.
- **Bar, written first:** "share drives it" if all three 50% models reach 30 of 300.
- **Result:** Separate pairs of 300, one per training. Up to a 1% share: 0, 0, 0. 2.45%: 0, 0, 2. 9.1%: 13, 5, 72. 23.1%: 66, 64, 146. 37.5%: 239, 129, 75. 50%: 209, 150, 84. Size check: 119, 281, 182.
- **Verdict:** support on the size check ("share drives it"). The card's overall "Yes, once about 9%" sits beside a threshold rule that returns null, and at 9.1% one training of three passes 30. Whether "Yes" stands is the owner's call.
- **Why it worked:** From 23.1% all three trainings pass 30 of 300 (`verified`).
- **Kind of failure:** none, it passed its bar (the size check: 119, 281 and 182 of 300 at a 50% share, all above 30). test design limits it: three trainings per share, with a spread of 164 at 37.5%. Two runs with 200 swapped pictures and the same training seed numbers give 0, 0, 3 and 175, 0, 37, and no file records why (`verified`).
- **Idea still alive?** alive, in the toy, from a 23.1% share, where all three trainings pass 30 of 300. The card's "about 9%" is not supported, because only 1 training of 3 passes there. The toy gives no share for real photos (`stated`).
- **Rules out:** Settles that set size alone does not explain the effect. Do not quote "about 9%" as a threshold (1 training of 3).
- **Leaves open:** why the two 200-swapped runs differ; any share for real photos.
- **Evidence:** [the toy relabelling finding](../../../../../4-inclusive-captions/report/do-inclusive-captions-help-the-product-compose/does-relabelling-both-shapes-pictures-help-the-toy-product-compose.md#1-how-many-pictures-must-hold-both-shapes)

### Inclusive labels on MNIST, by the dial

- **Asked:** On one MNIST diffusion model with labels, does also training two-digit canvases under each single label let the product of "7" and "8" draw two separate right digits?
- **Paper:** 2
- **Tried:** A small diffusion model on 32 by 64 canvases. The dial (how often a two-digit canvas is also trained under each single label) at 0, 0.1, 0.3, 0.6 and 1. Three trainings each at 6,000 steps. The corrected 7 times 8 product was read by the judge on 256 seeds and by eye on 64. A held-out 4 times 7 pair was also drawn.
- **Bar, written first:** (bar file, before any model drew) A dial counts if "7" alone and "8" alone each draw their lone digit at or above the dial's floor in all three trainings. Support: at a counting dial, at least 128 of 256 in all three trainings while dial 0 stays at 12 or fewer. Null: no counting dial's three-training mean reaches 39 of 256 (9.75 of 64 by eye). Otherwise inconclusive.
- **Result:** Counting dials 0, 0.1, 0.3: judge means 0, 0.33 and 15.33 of 256. Eye mean at dial 0.3: 8.0 of 64. Dial 0.6 has judge mean 52.33 and eye mean 22.0, but is excluded because one training drew a lone 7 on 205 of 256 against a floor of 207. Dial 1 (55.67 and 24.0) misses its floors too. 4 times 7: 0 of 256 at every dial.
- **Verdict:** null (plan 02). Admitting dial 0.6 would give inconclusive.
- **Why it failed:** The product rises only at dials where the single labels lose purity (`inferred` from the table). The choice of reader changed after the data was seen, from judge to blind eye calls (`stated`).
- **Kind of failure:** test design: mixed, because raising the dial raises the share and also lowers how often the joint label "7 and 8" is trained (`stated`). could not measure it: the judge agrees with the eye on 42 of 64 on the corrected row and 50 of 64 at dial 0.6, against the 58 of 64 plan 02 requires (`stated`).
- **Idea still alive?** unclear. The share run (next entry) holds the label mix fixed, and its own null waits on a judge check.
- **Rules out:** Do not move two things with one dial. Check the judge against the eye before reading a verdict from it.
- **Leaves open:** plan 02 tasks 5.2, 5.3 and 6.1.
- **Evidence:** [the MNIST labels finding](../../../../../4-inclusive-captions/report/do-inclusive-captions-help-the-product-compose/do-inclusive-labels-help-the-product-compose-on-mnist.md), [the bar file](../../../../../poe_repair/experiments/inclusive_captions/mnist/bar.py), [plan 02](../../../../../4-inclusive-captions/plans/11-do-inclusive-captions-help-poe-compose/plans/02-the-mnist-stage.md), [the dial read](../../../../../4-inclusive-captions/artifacts/results/do-inclusive-labels-help-the-product-compose-on-mnist/the-leak-read.png) (plan 02)

### A Langevin corrector with the set size fixed

- **Asked:** With the set held at 6,000 pictures, do 20 corrector steps bring every model with 100 or more swapped pictures to the pair?
- **Paper:** 2
- **Tried:** Swapped 0, 100, 200, 300, 500, 700 and 1,000 of 2,000 pictures. Three trainings each, plain sampling and 20 corrector steps.
- **Bar, written first:** with 100 or more replaced, 20 steps give at least 290 of 300 in all three trainings, and with none at most 3 (the replacing corrector script).
- **Result:** With 20 steps: 0%: 0, 0, 0. 5%: 31, 0, 185. 10%: 300, 253, 195. 15%: 300, 300, 300. 25%: 300, 36, 198. 35%: 300, 299, 300. 50%: 296, 262, 132. Of 15 models at 10% or more, 9 reach 290 or more and the lowest reaches 36.
- **Verdict:** none recorded: the files hold no verdict field. Against the written rule it holds at 0%, 15% and 35% and misses at 5%, 10%, 25% and 50% (the page's reading).
- **Why it failed:** Trainings at the same share diverge (300, 36, 198 at 25%) (`verified`). Why is not recorded.
- **Kind of failure:** test design: three trainings per share with large spread (`verified`). the idea, in part: the corrector does not reliably finish the job at a fixed share (`verified`, 9 of 15).
- **Idea still alive?** unclear. More trainings per share would tell apart a corrector that is unreliable and training noise.
- **Rules out:** Do not cite the card's sentence that the corrector "takes every model tested at 9% or more to 300 of 300".
- **Leaves open:** The seven replacing-corrector files have no card entry.
- **Evidence:** [the toy relabelling finding](../../../../../4-inclusive-captions/report/do-inclusive-captions-help-the-product-compose/does-relabelling-both-shapes-pictures-help-the-toy-product-compose.md#2-what-the-corrector-adds), [the replacing corrector script](../../../../../4-inclusive-captions/scripts/toy_replace_corrector.py)

### Share of two-digit pictures on MNIST, at guidance 3

- **Asked:** If a single-digit label's pictures show a second digit with probability s, does the product of "7" and "8" compose?
- **Paper:** 2
- **Tried:** Shares of 0, 10, 20, 40 and 80%, with 30, 50, 60 and 70% also scored. Three trainings per share. The corrected product was read by the pinned judge on 256 seeds at guidance 3.
- **Bar, written first:** (share bar file, before any share model trained) A share counts if "7" alone and "8" alone each draw their lone digit on at least round(0.95 × 256 × (1 − s)) seeds in all three trainings. Support: at least 128 in all three trainings at a counting share while 0% stays at 12 or fewer. Null: no counting share's mean reaches 39. The judge is trusted only if it agrees with blind eye calls on at least 58 of 64.
- **Result:** Only the 10% share counts. There the corrected product draws 17, 28 and 24, mean 23 of 256 against the null line of 39. At 20%, "7" alone reads 166, 207 and 197 against a floor of 195, so that share does not count. None of the extra shares counts.
- **Verdict:** null (card and counts file). Plan 03 has 0 of 9 tasks ticked, and the judge check is pending.
- **Why it failed:** Above 10%, the single labels lose purity, so only one share is readable (`verified`).
- **Kind of failure:** could not measure it: the judge check is pending, and the same judge undercounted on the dial run (`stated`). test design: the purity floor excludes every share above 10%, so the test reads one share (`inferred`).
- **Idea still alive?** unclear. The judge-against-eye check (plan 03 task 4.2) would tell whether the null stands.
- **Rules out:** Do not record a judge-read verdict before its eye check. "With a second digit" does not mean "with an 8".
- **Leaves open:** The card says 15 models, and the counts file holds 27.
- **Evidence:** [the MNIST share finding](../../../../../4-inclusive-captions/report/do-inclusive-captions-help-the-product-compose/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist.md), [the share bar file](../../../../../poe_repair/experiments/inclusive_captions/mnist/share_bar.py), [plan 03](../../../../../4-inclusive-captions/plans/11-do-inclusive-captions-help-poe-compose/plans/03-the-mnist-share-run.md), [the count against share figure](../../../../../4-inclusive-captions/artifacts/scenes/how-the-inclusive-labels-mnist-stage-works/figures/s2-the-product-against-what-each-label-draws-alone.png) (plan 03)

### The same MNIST share models redrawn at guidance 1

- **Asked:** At guidance 1, does the product compose beyond what its single labels already draw?
- **Paper:** 2
- **Tried:** The share models redrawn at guidance 1 and read by the judge.
- **Bar, written first:** two readings, written after the guidance-3 null and before any guidance-1 picture. "Composes beyond its labels" needs the plain product to reach 39 and to reach twice the larger single label's with-a-second count, in all three trainings.
- **Result:** Plain product against the needed count. 10%: 4, 8, 9 against 30, 30, 34. 20%: 36, 30, 43 against 72, 60, 74. 40%: 115, 111, 126 against 182, 154, 180. 80%: 189, 180, 189 against 366, 382, 390. At 80%, training 2, the corrected product reads 81 at guidance 3 and 206 at guidance 1.
- **Verdict:** null (the product composes beyond its labels at no judged share).
- **Why it failed:** The file records that guidance made the extra pairs (`stated`). Guidance 3 may be why "7" alone draws pairs far above its trained share (`inferred`, the page says "may be").
- **Kind of failure:** the idea: the product never exceeds twice what the single labels draw (`verified`, counts). could not measure it: same unchecked judge (`stated`).
- **Idea still alive?** unclear until the judge check.
- **Rules out:** Compare the product's pair count against what each single label already draws before crediting the product.
- **Leaves open:** The guidance-1 task is cited as "plan 03, task 5", which plan 03 does not contain.
- **Evidence:** [the MNIST share finding](../../../../../4-inclusive-captions/report/do-inclusive-captions-help-the-product-compose/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist.md#2-the-same-models-redrawn-at-guidance-1), [the guidance 1 picture](../../../../../4-inclusive-captions/artifacts/results/does-the-share-of-two-digit-pictures-help-the-product-compose-on-mnist/guidance-1-first-six-models.png) (plan 03)

## Project 5: a third expert from text embeddings

### Whether pairs with less similar words compose more easily

- **Asked:** Do animal pairs whose two words are less alike in the text encoder compose more easily under the correction?
- **Paper:** 2
- **Tried:** Eight pairs. For each, the area under its curve of compose rate against correction strength (4 seeds at each of 5 strengths, on a 0 to 1 scale) was plotted against a word cosine, and a rank correlation was computed.
- **Bar, written first:** none (the picture calls itself "a first look, not a verdict").
- **Result:** The y-axis values match `dose_curves.json` (`verified`). The x-axis values run from 0.9992 to 0.9997 and cannot be traced to one calculation. The printed rank correlation is +0.17. Breaking ties in other orders gives +0.17, +0.24 or +0.31.
- **Verdict:** none recorded: "a first look, not a verdict".
- **Why it failed:** The x-axis matches the range of a first run that the script's notes say was replaced. The saved numbers are absent (`stated`).
- **Kind of failure:** test design: 8 pairs and no bar (`verified`). could not measure it: the x-axis is untraceable (`verified`).
- **Idea still alive?** unclear. Recovering `text_orthogonality_probe.json` and recomputing on a known cosine would tell.
- **Rules out:** Do not cite +0.17. Save the numbers beside any picture that carries a statistic.
- **Leaves open:** everything about this link between word similarity and composition.
- **Evidence:** [what the text already says](../../../../../5-text-embedding-third-expert/report/what-the-text-already-says.md#whether-similar-words-compose-more-easily), [the word similarity script](../../../../../scripts/text_orthogonality_probe.py)

### The "and" vector built by subtraction as a third expert

- **Asked:** Does the "and" vector (the joint prompt's text embedding minus its two halves), fed in as a third expert beside "dog" and "cat", bring back one whole cat beside one whole dog?
- **Paper:** 2
- **Tried:** Three constructions on SDXL, rendered on seeds 1 to 8 from the same cached noise at guidance 7.5. x_raw is the plain difference over all 77 token slots. x_fix is x_raw plus the empty prompt's embedding, which cancels the start token. x_aff is x_raw plus twice the empty prompt. x_fix was also swept at weights 0.1 to 1.
- **Bar, written first:** one whole cat beside one whole dog on at least 4 of the 7 judgeable seeds. Seed 8 is not judgeable because the joint prompt draws two dogs there. The mark was not changed after the counts.
- **Result:** Clear cat and dog: x_raw 0 (a flat grey tile on all 8 seeds). x_fix 1 or 0 (the sources disagree on seed 3). x_aff 0. x_raw's start-token slot has length 1190, where a word sits at about 15. x_fix's repair was meant to bring its padding under 0.1 of the joint prompt's, and the ratio is 0.766 and 0.487 for "a cat and a dog" and 0.572 and 0.555 for "dog and cat" (smaller and larger encoder). Weight sweep: the product's hybrid at low weights and wallpaper at high ones.
- **Verdict:** null.
- **Why it failed:** x_raw's start-token error of 1190 is multiplied by the guidance weight of 7.5, which flattens the picture. x_fix leaves a sequence unlike a real prompt (`verified`, norms).
- **Kind of failure:** done wrong: the vectors as built are not prompt-like embeddings, and the repair missed its own 0.1 mark (`verified`). test design: one pair, 7 seeds, by-eye calls (`stated`).
- **Idea still alive?** closed for subtraction-built vectors on this pair (`inferred`). The construction built faithfully (x_and, below) is also null, and the page says the zeros are firm because they are far from the mark.
- **Rules out:** Do not subtract prompt embeddings slot by slot without checking per-slot norms. The start token dominates and is multiplied by the guidance weight.
- **Leaves open:** the pooled channel alone, never rendered as a condition.
- **Evidence:** [what the "and" vector is made of](../../../../../5-text-embedding-third-expert/report/what-the-and-vector-is-made-of.md), [the per-position norms](../../../../../5-text-embedding-third-expert/artifacts/results/the-and-vector/what-the-and-vector-is-made-of/a-cat-and-a-dog-norms.png), [the "and" vector idea map](../../../../../5-text-embedding-third-expert/artifacts/ideas/the-and-vector-as-a-third-expert/IDEA_MAP.md) (runs 2 to 7 of the idea map)

### Written phrases as a third expert

- **Asked:** Does a plain written phrase as a third expert beside "dog" and "cat" bring back one whole cat beside one whole dog?
- **Paper:** 2
- **Tried:** Seven phrases ("a dog next to a cat", "side by side", "together", "two different animals", "a pair of animals", "two animals", and the joint prompt itself) plus "single animal", on seeds 1 to 8. As a reference, the joint prompt alone draws one whole dog and one whole cat on 28 of 32 seeds (Claude's calls, `verified` count).
- **Bar, written first:** one whole cat beside one whole dog on at least 4 of 7 judgeable seeds.
- **Result:** Clear cat and dog: 0 for every phrase ("two animals" is 0 or 1, since the sources disagree on seed 4). Two separate animals, of 7: "a dog next to a cat" 5, the joint prompt 4, "two animals" 2. The idea map lists 6 tiles out of 8 for "a dog next to a cat".
- **Verdict:** null.
- **Why it failed:** Two animals come back, but each is a hybrid: the owner called the two strongest phrases "two hybrid animals" (`stated`). Why identity does not return is not measured.
- **Kind of failure:** the idea: co-occurrence moves from 0 to 5 while identity stays at 0 (`stated`). test design: one pair, 7 seeds, one weight (the run's 7.5), by-eye calls, and Claude's first calls overcalled from a downscaled sheet (`stated`).
- **Idea still alive?** unclear. One pair at one weight. A weight sweep on the best phrase and a second pair would tell.
- **Rules out:** Do not credit text arithmetic for co-occurrence that a plain sentence already gives (5 of 7). Score identity separately from co-occurrence.
- **Leaves open:** the remaining phrases' calls, which nobody has checked.
- **Evidence:** [the text-space third expert finding](../../../../../5-text-embedding-third-expert/report/can-a-text-space-third-expert-restore-composition.md#2-co-occurrence-comes-back-identity-does-not), [the phrases sheet](../../../../../5-text-embedding-third-expert/artifacts/results/the-and-vector/what-the-and-vector-is-made-of/dog-and-cat-phrases-sheet.png)

### Steering the starting noise apart by attention

- **Asked:** Does a gradient step on the image latent at steps 0 to 9, reducing the overlap between where the dog and cat experts attend, give one clear dog beside one clear cat?
- **Paper:** 2
- **Tried:** Three step sizes, 2.5e4, 1.2e5 and 5e5 (1%, 5% and 20% of the latent's length), on 8 seeds.
- **Bar, written first:** clear dog and cat on 4 of 7.
- **Result:** 0 of 7 at every step size, plus one borderline tile on seed 6 at the large size. Small steps give one animal, wallpaper or a hybrid. Medium and large give flat graphics or saturated blocks. The attention maps' largest value is 0.0012, against 0.00098 for a flat map. About 98% of each update comes from the terms that sharpen each map.
- **Verdict:** null (not supported).
- **Why it failed:** The maps are almost flat at these steps, so the step barely separates them (`stated`).
- **Kind of failure:** not understood: the design assumed attention at steps 0 to 9 carries a spatial overlap to push on, and the peak is 0.0012 against a flat 0.00098 (`stated`). done wrong: 98% of the update sharpens maps instead of separating them (`stated`). test design: one loss and one pair.
- **Idea still alive?** unclear. A loss on the overlap term alone was not run, and whether the flat maps leave anything to push would tell.
- **Rules out:** Do not steer on cross-attention overlap at steps 0 to 9 without first checking that the maps carry spatial structure.
- **Leaves open:** The card says small steps keep the photograph style, while the sheet shows drawings on seeds 5 to 8.
- **Evidence:** [what the "and" vector is made of](../../../../../5-text-embedding-third-expert/report/what-the-and-vector-is-made-of.md#steering-the-starting-noise), [the steering sheet](../../../../../5-text-embedding-third-expert/artifacts/results/the-and-vector/what-the-and-vector-is-made-of/separation-steering-sheet.png), [the steering walk](../../../../../5-text-embedding-third-expert/artifacts/drips/separation-steering/EXECUTE_WALK.md)

### A learned "and" token in place of the vector

- **Asked:** Can one fitted token, `<and>`, used as the third expert, close the gap between the plain product and the joint prompt?
- **Paper:** 2
- **Tried:** One trainable row in each text encoder, fitted for 8,000 updates on 182 cached examples over 48 animal and object pairs, with cat and dog and seeds 9 to 16 held out.
- **Bar, written first:** the share of the gap removed must be above 0.10 at some step to continue (a stop rule). A picture test (step 6 of the plan) would give the verdict.
- **Result:** Share of the gap removed (1 closes the whole gap, 0 none, negative is further away), mean of seeds 9 to 16. Before training: −0.142, −0.064, −0.086, −0.052 at steps 0, 10, 25, 40. After 8,000 updates: −0.070, +0.013, −0.005, −0.008. The sheets show one animal mixing dog and cat features on every held-out seed (`stated`). The deck records every other token variant stopping on the same rule.
- **Verdict:** null on the stop rule (it fired). The picture test was not run.
- **Why it failed:** One fitted token removed under 10% of the gap at every checked step (`stated`, from the plan's record of the W&B run).
- **Kind of failure:** the idea, as far as the stop rule reads (`stated`). could not measure it: the share removed reads noise predictions at four steps and does not show whether a picture composes (`stated`).
- **Idea still alive?** unclear. The plan's picture test (step 6) would tell, and so would the per-step search for any embedding that hits the target (running as of 2026-11-04).
- **Rules out:** Do not expect one fitted token row to close the gap to the joint prompt.
- **Leaves open:** what separates the run6a, run6b and run6c2 sheets.
- **Evidence:** [what the "and" vector is made of](../../../../../5-text-embedding-third-expert/report/what-the-and-vector-is-made-of.md#a-learned-token-in-place-of-the-vector), [the learned-token plan](../../../../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-a-learned-and-token.md). W&B run `c6f1pmmq`.

### Does the joint prompt's embedding minus one half carry the other word

- **Asked:** Does E("A and B") − E("A") carry B, where E is a prompt's text embedding?
- **Paper:** 2
- **Tried:** 200 random pairs scored against a 276-noun dictionary (chance median rank 138), text encoders only, with a decoy word as control.
- **Bar, written first:** none recorded on the page.
- **Result:** The held-out word comes back at median rank 1. Top-1 is 75.0% on the pooled summary vector and 98.5% on the content mean. The decoy word is at chance in both.
- **Verdict:** none recorded: the page reads the premise as holding ("It does").
- **Why it worked:** The ranks show it (`verified`, the premise file).
- **Kind of failure:** none: support. The content-mean row is partly bound by arithmetic to land near B (`stated`), so the pooled row is the one that counts.
- **Idea still alive?** It licenses this: the third-expert failures are not caused by text arithmetic being meaningless.
- **Rules out:** Settles that the difference carries the other word (pooled top-1 75.0%). Do not cite the 98.5% content row as independent.
- **Leaves open:** the pooled channel alone as a rendered condition.
- **Evidence:** [the text-space third expert finding](../../../../../5-text-embedding-third-expert/report/can-a-text-space-third-expert-restore-composition.md#1-the-arithmetic-recovers-the-word), [the positional card](../../../../../5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/README.md)

### The positional "and" vector as a third expert

- **Asked:** Does a third expert that subtracts each animal word only at the slot holding it bring back one whole cat beside one whole dog?
- **Paper:** 2
- **Tried:** x_pos is the joint prompt's own sequence with each animal's word subtracted at its own slot, so it edits 2 of 77 slots. Rendered on mscluster108 outside Slurm on seeds 1 to 8, alone and inside the product of dog, cat and x_pos.
- **Bar, written first:** one whole cat beside one whole dog on at least 4 of 7 judgeable seeds.
- **Result:** Two separate animals on 6 of 7. Clear cat and dog on 0 of 7. Mean absolute pixel distance from the joint prompt's image (0 to 255, averaged over 8 seeds): x_pos alone 23.43, the product with x_pos 39.16, plain product 47.05. x_pos alone gives a clear pair on 2 of 7, which drops to 0 when the dog and cat experts are added. For comparison, the exact target in the third slot renders the joint image to within 2.23 (the deck).
- **Verdict:** null.
- **Why it failed:** Adding the two single-animal experts moves the picture away from the target (23.43 to 39.16) (`verified`).
- **Kind of failure:** the idea: the single experts pull identity apart even when the third expert resembles the joint prompt (`verified`). test design: one pair, 7 seeds (`stated`).
- **Idea still alive?** closed for a text-space third expert beside the two single experts on this pair (`inferred`: zero clear pairs, far from the mark of 4). What stays open is a correction inside the denoiser.
- **Rules out:** Do not keep the two single-animal experts at full weight beside a third expert that already resembles the joint prompt.
- **Leaves open:** whether any embedding can hit the target at every step (the per-step optimiser, running as of 2026-11-04).
- **Evidence:** [the text-space third expert finding](../../../../../5-text-embedding-third-expert/report/can-a-text-space-third-expert-restore-composition.md#3-the-product-is-worse-than-the-third-expert-alone), [the target, plain product and the product with x_pos on all seeds](../../../../../5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/all-seeds-target-poe-xpos.png), [the tile calls](../../../../../5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/calls.json)

### Only what the joint adds, on the empty prompt's scaffolding

- **Asked:** Does a third expert carrying only what the joint prompt adds, written onto the empty prompt's start, end and padding slots, restore the pair?
- **Paper:** 2
- **Tried:** x_and in the product with dog and cat, on seeds 1 to 8.
- **Bar, written first:** one whole cat beside one whole dog on at least 4 of 7 judgeable seeds.
- **Result:** 0 of 8 tiles show two separate animals, and 0 of 7 judgeable seeds show a clear pair. Distance from the joint prompt's image: 48.49, against the plain product's 47.05 (mean absolute pixel difference, 0 to 255). One fused animal in every tile, with seeds 4 and 5 as flat graphics.
- **Verdict:** null.
- **Why it failed:** A condition enters the guided sum as w·(ε_x − ε_∅), the weight times its prediction minus the empty prompt's. So a third expert built faithfully as a difference sits near the empty prompt and contributes nothing (`inferred`: argued on the page, and consistent with the measured 48.49 against 47.05).
- **Kind of failure:** not understood: the design expected a faithful difference to carry the joint's addition, but the guided sum sees only its gap from the empty prompt (`inferred`).
- **Idea still alive?** closed (`inferred`).
- **Rules out:** Do not build a third expert as a difference written onto the empty prompt's scaffolding.
- **Leaves open:** nothing further for this construction.
- **Evidence:** [the text-space third expert finding](../../../../../5-text-embedding-third-expert/report/can-a-text-space-third-expert-restore-composition.md#still-open), [the positional card](../../../../../5-text-embedding-third-expert/artifacts/results/the-and-vector/can-a-positional-and-vector-restore-composition/README.md)

### Do the text encoders already say one animal or two

- **Asked:** Before any image exists, do SDXL's text encoders already place "a cat" toward one thing and "a cat and a dog" toward two?
- **Paper:** 2
- **Tried:** A one-versus-many direction: the mean of E("two X") − E("a X") over 40 nouns, checked on 41 held-out nouns against 200 random directions. Prompts for 49 pairs were then placed on it, using the larger encoder's pooled summary.
- **Bar, written first:** The direction must beat chance at least 3 times. The joint prompt must sit "at least one such margin toward plural".
- **Result:** The direction beats chance 43.24 times. Positions run from -0.5 (one thing) to +0.5 (two things). Single-animal prompts: median −0.53 and −0.54 (cat −0.40, dog −0.38). Joint prompt: median +0.11 (cat and dog +0.05), a move of +0.643 from the single-animal median. Other views of the prompt give joint positions of −0.045, −0.327 and −0.219, with moves of +0.576, +0.302 and +0.422.
- **Verdict:** direction check: support (43.24 against 3). Joint prompt: none recorded, with two readings. Read as reaching the two-thing position (needs +0.50, measures 0.111), it gives null. Read as moving that far from the singles (0.643), it gives support. The owner chooses.
- **Why it failed:** The rule's wording allows both readings (`verified`).
- **Kind of failure:** none, it passed its bar (the one-versus-many direction check). test design limits the second claim: the rule for the joint prompt reads two ways (+0.111 against +0.50 needed, or a move of +0.643 against 0.50), and the position changes with the view of the prompt (`verified`).
- **Idea still alive?** alive, for the direction itself. Whether the joint prompt already reads as two things waits on the owner's choice of reading, and nothing here reaches the image.
- **Rules out:** Write whether a margin is measured from the singles or as an absolute position before running. Report every view of the prompt, not one.
- **Leaves open:** the same axis on a text encoder with no CLIP in it; how much of the joint prompt lies outside its parts.
- **Evidence:** [what the text already says](../../../../../5-text-embedding-third-expert/report/what-the-text-already-says.md), [where the prompts land](../../../../../5-text-embedding-third-expert/artifacts/results/what-the-text-already-says/does-one-prompt-already-say-one-animal/where-the-prompts-land.png), [the plan with the rule](../../../../../3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-does-one-prompt-already-say-one-animal.md)


- 3-understanding-compositionality/report/the-exact-toy/how-the-forward-process-dissolves-an-image.md: an explainer of the forward noising process; the page says no pass rule or verdict applies.
- 3-understanding-compositionality/report/what-sdxl-already-knows/where-the-product-lands.md: an illustration with a hand-drawn surface; the page says nothing is measured.
- 4-inclusive-captions/report/00-INDEX.md: the project index. Every row is covered by the entries above.
- 4-inclusive-captions/report/walks-scenes-notes-and-decks.md: a listing of 406 working files, no attempt (read only its opening, as it was outside the assigned folders).
- what-the-fusion-is-not-caused-by.md: left to another reader, as instructed.


1. Most nulls here were not fair tests. They rest on 5 to 8 seeds or one pair (dog in the rain, camel, third concept, cats-and-dogs sheet, every text-space third expert), three trainings with spreads of 150 or more of 300 (toy relabelling and corrector), or one seed (the exact-toy pushes). The thin seed counts are `verified` from the pages. That this is why the results are inconclusive is `inferred`.
2. Instruments that cannot see the outcome come up again and again. The detector reported a dog on 64 of 64 single-cat renders, CLIP precision on wetness was 0 of 7, the nearest-picture labeller named noise as shapes, the MNIST judge agreed with the eye on 42 of 64, and the species judge agreed on 13 of 19. All `verified` on the pages.
3. Mixed comparisons are built into several designs. Examples: dial raises share and lowers joint-label training; adding pictures grows the set; column 7 adds a word and a guidance unit; fused renders against real photographs; SD v1-4 against SD3.5. Each is named on its page (`stated`).
4. Many attempts stop before the verdict, with renders made but no calls or tally. Examples: 128 third-concept tiles, the camel training-picture count, 117 caption rows, the 128-seed SDXL pilot, the layout's six columns. Several more have rules that can be read two ways and leave the choice to the owner.
5. The fair, far-from-the-mark results are the text-space third experts and the exact-toy diagnosis. The third experts show co-occurrence returning (up to 6 of 7) and identity staying at 0 against a mark of 4, and x_and shows the guided sum ignoring a faithful difference (argued, `inferred`). The exact toy shows the sampler, not the product, losing the pair (`verified`: 300 of 300 with the true noised score).


- RAN dates not printed on the page itself: species judge and attention probe (2026-09-25, from the recipe walk); the per-kind sheets (2026-10-14, from their card); the CLIP pilot (2026-10-21, the script's first commit); the caption census (2026-10-21, from its walk); the exact-toy sweeps, layout, hybrid fit, one-vs-many axis and x_pos (2026-11-04, from plans or cards); threshold and corrector (2026-11-05, first commit of their JSON); the toy replacing sweep (2026-11-18, from its card); the photo pool, caption search and SDXL pilot (2026-11-03, from first commits and the pilot walk); x_and (2026-11-04, from the idea map's run row); arithmetic vectors, phrases, steering and learned token (2026-10-27 and 2026-10-28, from the idea map, card and plan). Undated: the cats-and-dogs sheet, the cat-prompt base rate, the MNIST share runs and the replacing corrector.
- The toy caption-model experiments (threshold and corrector, with and without relabelled pictures) appear on both the project 3 training page and the project 4 relabelling page. I filed them once, under project 4, and linked the project 3 page from the corrector entry.
- MNIST dial page: the opening says the counting dials draw "a mean of 0, 0.33 and 8.0 separate right pairs of 64 by eye". In the table, 0 and 0.33 are judge means out of 256, and dials 0 and 0.1 have no eye calls. The opening mixes two readers and two denominators.
- The seven replacing-corrector files and the other toy-caption JSONs are not on the laptop. `on-the-cluster.tsv` in the toy relabelling results folder lists them under `/datasets/mmolefe/poe_repair_min/artifacts-archive/...`, so I linked the page and the script, not the files.
- Counts the sources disagree on, recorded rather than resolved: x_fix clear pairs 1 against 0 (seed 3); "two animals" 0 against 1 (seed 4); the best phrase's two-animal tiles 5 against 6; the number of third experts tried (eight, nine, or x_pos as the seventh); plain product distance from the joint image, 47.05 against 34.69 in the deck.
- The ICLR "58 of 64" count does not appear on any of my pages. The detector that cannot name the animal appears twice. On the cat-prompt page it reported a dog on 64 of 64 single-cat renders, with median "dog" confidence 0.889. The text-space page says the project's instance-count scorer "cannot tell a cat from a dog", which is why the third-expert calls were made by eye.
- None of my pages says anything about a disabled or detached adapter.
