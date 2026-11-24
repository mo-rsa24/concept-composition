# Method review: why giving each subject its own region missed three times (step M9)

Loop 13 asked for this review after three misses in a row. No new M9 run starts before the owner answers board question 83, which this
review is written to inform. The runs and numbers are in [the M9 review](review.md).

## The three misses

| Miss | What was tried | Result |
|---|---|---|
| Rung 1, regions from step 0 | each subject's prediction weighted by a mask around its true box, on the plain product | 0 clear pairs of 8 |
| Rung 1, with bootstrapping | each region denoised alone over a constant colour for the first 15 of 50 steps | 8 of 8 clear pairs, every one damaged |
| Rung 2, two region readers | boxes read from word attention at step 10, then from where one word's map beats the other | 11 and 10 of 24 reads failed, against a limit of 6 |

## What they have in common

**Each one separates the subjects with a signal that has not yet separated them.** At step 0 each subject's prediction describes a whole
picture of that subject, so weighting two such predictions by masks still blends them, and the plain fusion fused 8 of 8. At step 10 one
word's attention covers both bodies on 4 of the 12 tuning pairs, so no reading of the two maps gives each subject its own cells. Bootstrapping
is the one move that separated the subjects, and it did so by cutting each region off from the other, which is also why every result looks
pasted. The method has two gaps: no signal it can read early separates the subjects, and once regions are imposed nothing blends them back.

## Papers that bear on it and that this loop has not read

None of these was read in full. Each claim below is what a search found, and it is marked as such.

- **Bounded Attention** ([Dahary et al., arXiv 2403.16990](https://arxiv.org/abs/2403.16990), ECCV 2024). It reports that subjects leak through both
  cross-attention and self-attention, and that bounding both is needed. M9 masked only the noise predictions, so self-attention could still
  mix the bodies. *Found by search.*
- **StreamMultiDiffusion** ([arXiv 2403.09055](https://arxiv.org/abs/2403.09055)). It reports that MultiDiffusion's random-colour bootstrapping
  leaves abrupt boundaries when too few steps remain to blend the regions, and replaces it with a white background mixed with the other
  regions' content. This matches the pasted look of rung 1. *Found by search.*
- **Composite Diffusion** ([arXiv 2307.13720](https://arxiv.org/abs/2307.13720)). It builds regions under a scaffold, then drops the scaffold
  for a harmonising stage in which each region develops beside the others. *Found by search.*
- **Training-free regional prompting for diffusion transformers** ([arXiv 2411.02395](https://arxiv.org/abs/2411.02395)). It applies regional
  control only in early steps, on the grounds that they set the layout. *Found by search.*
- **Attention overlap and missing entities** ([arXiv 2410.20972](https://arxiv.org/abs/2410.20972)). Its title says attention overlap causes
  missing entities, the effect DAAM measures and the failure the step-10 readers met. *Found by search.*
- **AttenCraft** ([arXiv 2405.17965](https://arxiv.org/abs/2405.17965)). It reports that cross-attention maps are too blurry for masks at very
  noisy timesteps, and starts its masks from less noisy ones. *Found by search.*

## Whether other loops already explain it

**They explain the readers' failures.** Loop 8's identity read on digits rises from 0.563 in the early window to 0.702 in SDXL's steps 16 to
31 (`abstract-checkers/10-reading-identity/review.md`), and Loop 9's readout on SDXL's own features reaches 0.87 by step 15
(`noise-text-critic/05-verdict.md`). Both say who-is-where becomes readable later than step 10. Loop 12's reading of DAAM (p. 7) finds that
pairs whose words' attention overlaps little draw both objects in 71.7% of images, against 9.75% when it overlaps a lot
(`leakage/01-reading/mechanism-table.md`), which is the overlap seen on 4 of 12 pairs.

**They suggest a cause for the damage.** Loop 12 notes that CO3 re-noises with the empty prompt's prediction and rescales the latent to its
length after every edit (`leakage/ideas/what-co3-does-that-our-loops-did-not.md`). M9's composer did neither. That is an untested inference.

## Whether M9 is still worth its cost for Paper 2

**Not in its present form.** Rung 1 used boxes taken from pictures that had already succeeded, which is the best case any reader could
supply, and its only success was damaged on 8 of 8. So a reader that passed would hand rung 2 a composer that is already known to damage
pictures. Rendering rung 2 costs about 4.8 GPU-hours, and on present evidence it would measure the damage again. Board question 83's option A,
closing M9, is the review's recommendation.

## What would change in a redesign

These would make a new method, not a further rung, and any of them is the owner's call.

1. **Read regions where they are readable.** Read after step 16, or from CO3's own finished picture, so the regions describe bodies that
   already exist.
2. **Bound attention as well as predictions.** Restrict self-attention between the regions, as Bounded Attention does, so the bodies stop
   mixing through the U-Net itself.
3. **Blend after separating.** Replace the constant-colour bootstrap with a background mixed from the other region, and leave late steps
   free to harmonise, or follow CO3's renoise and rescale after each edit.
4. **Make damage a bar, not a note.** Every rung counts undamaged pictures beside clear pairs, and a clear pair that is damaged does not pass.
5. **One reader per fresh seed set.** Each new reader is judged once, on seeds no earlier reader saw, so readers are not tuned until one passes.
