# Paper 2's method, chosen from where CO3 leaks (step Z4)

**Paper 2 does not yet have a method that removes borrowed parts, and the candidate to test next is a fix in the text encoder.** Z2 found that no reading at the text encoder, the attention layers or the sum of predictions predicts which pairs borrow ([Z2's review](02-what-predicts-borrowing/review.md)). Z3 found that correcting each animal's region later in sampling, even with the regions drawn by hand, leaves the count of clean pictures unchanged, and on cheetah × cougar it moves the borrowed cheetah features from the cougar's body to its face ([Z3's review](03-one-fix/review.md)). Board question 80 declined the checker choosing CO3's settings, because the checker does not see CO3's failure. The remaining training-free candidate is the one that acts on the "cougar" word itself.

## The owner's constraints

**Prefer methods that leave the diffusion model's weights untouched.** The owner said on 2026-11-24 that classifier guidance is the current best fit because no diffusion model is trained, and that it is not a commitment; a candidate that trains the diffusion model, even as a LoRA, needs a reason the others cannot meet. **Every candidate is judged on one starting noise per seed following its own trajectory**: restarting seeds or keeping the best of several tries is selection, ruled out as the method on 2026-11-18, and appears only as a baseline.

## The bar and the rival

[Loop 4's leakage rivals page](../targets/leakage-rivals.md) sets them. Paper 2's results are counted as pictures with two clear subjects that keep their own parts, drawing style ignored, with broken anatomy as a guard and two blind reviewers (board questions 87 and 92). The bar is CO3 alone on the same pictures: on 96 pictures by M1's rule (a lead of at least 5 and more than 1.96 standard errors), and on the 48 pictures of the six borrowing pairs by Z3's rule (a lead of at least 4 with an exact one-sided McNemar p below 0.05). **Bounded Attention is the rival** Paper 2's method must beat: it is the one leakage method with released SDXL code (`run_xl.py`), and its row must say where its boxes came from, since boxes from finished pictures make it an upper bound.

## The candidates

| Candidate | What it does | Trains the diffusion model? | Evidence from this loop and its neighbours | Cost to test | Call |
|---|---|---|---|---|---|
| A text-side fix | replaces the second subject's word vectors inside the joint prompt with its context-free encoding, which CO3 already computes ([Follow the Flow's note](01-reading/follow-the-flow.md)), or edits the joint prompt's embedding once ([A Cat Is A Cat's note](01-reading/a-cat-is-a-cat.md)) | no | Z3: the cougar keeps a cheetah's face wherever the correction acts, which fits the "cougar" prediction itself carrying cheetah features (inferred); A Cat Is A Cat: CLIP's causal mask copies the first-named subject into the second subject's token (pp. 4 and 5); Z2: the joint-prompt cosine of the two nouns does not predict which pairs borrow ($\rho$ 0.137), so the fix is untested here and unsupported | about 0.7 GPU-hours for 48 pictures, plus 0.7 for the word-order check | **test next** |
| Attention bounds (Bounded Attention) | keeps each subject's attention inside its own box ([its note](01-reading/bounded-attention.md)) | no | Z2: attention overlap does not predict borrowing in any window (best $\rho$ 0.489, bar 0.706); Z3: a correction by region with exact boxes changed nothing; it acts on attention, a different place from Z3's correction, and is untested on CO3 | about 2 GPU-hours for 48 pictures, estimate; boxes already drawn | run as **the rival**, with the boxes Z3 drew |
| CO3 with a weight per cell | CO3's own corrector with per-cell weights ([the CO3 note](ideas/what-co3-does-that-our-loops-did-not.md), difference 2) | no | the [free check](03-one-fix/free-check/README.md): on the recorded seed CO3's per-cell signal reads "cheetah" on both bodies at step 5, so per-cell weights would push the cheetah off its own body; Z3 is its upper bound with drawn regions, and it is null | about 0.8 GPU-hours, not to be spent | **closed** |
| Regions from each subject's own push | each subject's prediction acts only in a region read from its $d_t$ map (Loop 10's audit) | no | Z3 gives the same correction with exact regions, and it is null; regions read from $d_t$ can do no better | under 0.1 GPU-hours (Z5's rerun 4), not to be spent | **closed** |
| The checker chooses CO3's settings | picks CO3's $\beta$ at each corrected step ([the choosing test](../score-space/11-the-checker-chooses-on-co3.md)) | no | board question 80 declined it: the checker's yes-or-no call on CO3's failure is right at best 60% of the time against a 65% floor (Loop 6's full read) | about 3.1 GPU-hours for its first read (its design), declined | **closed** |
| A checker on the failure signal | a small checker reads only the early reading that predicts borrowing ([the idea note](ideas/a-checker-on-the-failure-signal.md)) | no | Z2 found no such reading for CO3's borrowing; the one lead, end-of-text token overlap ($\rho$ 0.74, the largest of 20 secondary readings and so possibly chance), is untested | not measured: 96 or more pictures labelled by eye, a training run, and a push | **closed for CO3**, unless a pre-registered test of the end-of-text overlap on new pairs finds a signal |
| An identity reward LoRA | trains a LoRA on a reward read from the finished picture | yes, a LoRA | the only line that has fixed identity in this project is the correction, which a LoRA learns; the owner's constraint asks for a reason the training-free candidates cannot meet | not measured: a LoRA training run on SDXL, of the order of the project's earlier LoRA runs (31.8 to 34.9 hours on an RTX 8000, environment/hpc/throughput.md) | **held**: it becomes the candidate only if the text-side fix and the rival also fail |
| A combination | any two of the above | depends | none | the sum of its parts | after the text-side test |

## What the next test is

**A cheap check first: the word-order swap** Z2 left as an optional column. CO3 on "a cougar and a cheetah" at the same seeds, labelled blind. If the borrowed features follow the order of the words (the second-named animal borrows from the first, whichever it is), the text encoder is where borrowing enters, and the token swap is aimed by evidence. 48 renders, about 0.7 GPU-hours.

**Then the text-side fix on the same 48 pictures, beside Bounded Attention, against CO3 alone**, one render per picture per row, counted blind on the same measure, with the bar above.

| Row | What it is | GPU-hours |
|---|---|---|
| CO3 alone | Z2's one-card re-renders, re-called blind in the same set | 0 |
| CO3 with the token swap | the second subject's word vectors in the joint prompt replaced by its encoding alone, in every U-Net call that reads the joint prompt | about 0.7 |
| Bounded Attention on SDXL | the released code with Z3's drawn boxes, the rival's upper bound | about 2, estimate |

These runs are outside Loop 12's brief and need the owner's yes; [the verdict](06-verdict.md) puts them as one question.
