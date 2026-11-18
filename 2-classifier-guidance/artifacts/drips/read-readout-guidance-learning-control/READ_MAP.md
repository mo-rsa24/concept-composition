# Read map: Readout Guidance: Learning Control from Diffusion Features (lean)

Luo, Darrell, Wang, Goldman and Holynski, CVPR 2024, arXiv 2312.02150. Code:
github.com/google-research/readout_guidance (Apache-2.0), read at commit `a8a699c` (2025-06-26).
Lean walk opened 2026-11-17. Step 10f of [the classifier-guided PoE route](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md).
Page numbers are the arXiv PDF's (27 pages; the supplement starts at p12 and numbers its sections
8 to 10). Plain readings in the rounds are the walk's own reading of the quoted text.

## Setup

| Field | Value |
|---|---|
| Why it is read | rung 2 needs a discriminator that reads "two separate things" on pictures it never trained on; every one tried failed its held-out check (0.53 to 0.71 balanced accuracy against a 0.75 bar). A count head in this paper's style is built in [the count head script](../../../scripts/rung2/train_count_head.py), and this read decides what it must copy |
| Grain | paragraph |
| Mode | lean; numbers only through `test`, the full walk only through `deep` |
| Rules | read only: no code, no plan edits; this map is the one file the walk writes |
| Bar | the five asks below answered with a page each, and every caveat carries a line on whether the count head script meets it |
| Time box | one sitting for sections 1 to 4, one for 5 and the supplement's sections 8 and 10 |
| Paper text | the arXiv source and the PDF, fetched 2026-11-17 into the session scratchpad with the released code; not kept, refetch from arXiv 2312.02150 |
| Journey | 13, `13-classifier-guidance-by-hand`: chapter 6 and plan 11 hold the Readout Guidance ladder, with the network view under `artifacts/figures/readout-guidance/` and two code walks ([the first](https://claude.ai/artifact/SHnMMo46yDvhj2GaXuEs7H), [the second](https://claude.ai/artifact/2VgqMaBHqrXNoHdTWjxro1)) |

## Spine

The paper in one paragraph: a frozen text-to-image model already knows a lot about the picture
it is drawing, even at noisy steps. Train a small head on its decoder features to read off one
property (a pose, a depth map, edges, or a feature that matches across two pictures), using about
100 to 8,500 pictures labelled by off-the-shelf models. At sampling time, compare the head's
reading with a target and nudge the noisy latent down the gradient of that difference, over the
first half of the steps. One recipe covers dragging, identity and pose control, with far less
data than ControlNet.

| Part | Pages | What it does in the argument | You |
|---|---|---|---|
| Abstract | 1 | the claim: small heads on frozen features, used for guidance | ← start |
| 1 Introduction | 1 to 2 | the problem: control without training on big datasets; as few as 100 examples | ask 2 |
| 2 Related work | 2 to 3 | background: guidance, ControlNet-style adapters, readouts of diffusion features | |
| 3.1 Background | 3 | background: classifier guidance on DDIM | |
| 3.2 A revised guidance recipe | 3 to 4 | method: a regressor and a distance replace the classifier's log-probability | ask 3 |
| 4.1 Common architecture | 4 | method: which decoder layers, timestep conditioning, the empty prompt | ask 1 |
| 4.2 Spatially aligned heads | 4 to 5 | method: pose, depth, edge heads; pseudo-labels; the sparse-loss reweighting | ask 2, the count head's nearest relative |
| 4.3 Relative heads | 5 | method: correspondence and appearance heads trained with contrastive and triplet losses | skippable for the count head |
| 5 Results | 5 to 8 | results: drag, appearance, identity, pose control; 100 against 1k against 8.5k pictures | ask 2 |
| 5.1 Limitations, 6 Conclusion | 8 | where it fails: memory and time, improbable pictures | ask 4 |
| 8.1 to 8.6 | 12 to 14 | the recipe's numbers: window, weight, eta, the normalised update rule, compute, improbable images | asks 3 and 4 |
| 9 Relative control | 14 | identity guidance only inside a face mask | skippable |
| 10.1 Readout evolution | 15 | the composition shows in the readout within the first step | ask 5 |
| 10.2, 10.3 | 15 to 16 | the pose numbers; why SDXL's mid block is left out; heads that hallucinate | asks 1 and 4 |

## Chunks

| # | Section | Paragraph | State | Note |
|---|---|---|---|---|
| 1 | Abstract | whole | walked lean ← current | |

## What the read-ahead found for the five asks

**1. Which layers and timesteps the heads read, and why.**
- Every resnet of the U-Net's decoder, the up blocks: 9 on SDXL, 12 on SD 1.5 (section 4.1 p4;
  10.2 p16; code `dhf/stable_diffusion/resnet.py` `collect_layers`, which walks `up_blocks[*].resnets`
  only). SDXL's mid block is left out because "we found in early experiments [it] did not improve the
  quality of our spatially-aligned readouts" (p16).
- Each layer is resized to the latent grid (128 on SDXL; code `rg_helpers.collect_and_resize_feats`),
  brought to 384 channels by a bottleneck, and mixed by learnt weights (p14, Figure 12; config
  `projection_dim: 384`).
- The features come from the empty prompt "" for every picture (p4).
- Why per timestep: Diffusion Hyperfeatures averaged over timesteps; a guidance head must answer
  at each sampling step, so it aggregates over layers only and gets the timestep as an input, the
  U-Net's own timestep embedding projected and added inside each bottleneck (p4, p14).
- Training noise: one timestep per batch, uniform over 1 to 999, the clean picture noised forward
  to it (code `dhf/diffusion_extractor.py:143`).

**2. How many examples a head needs, and how its labels are made.**
- As few as 100 for pose (p2, Figure 9 p7). On SDXL, PCK at α = 0.05 (the share of keypoints
  within 5% of the pose's box size, in percent): no control 1.19, 100 pictures 15.24, 1k 19.65,
  8.5k 24.29, T2I-Adapter on 3M 24.06 (Table 2, p15).
- Labels: pseudo-labels from off-the-shelf models, OpenPose, MiDaS and HED, on PascalVOC (p4);
  CoTracker tracks on DAVIS videos for correspondence; SDEdit-made hard negatives for appearance
  and identity (p5).
- Loss: per-pixel MSE; for pose and edges the empty and non-empty regions are reweighted to count
  equally, "as both controls are inherently sparse" (p5).
- Training: batch 8, at most 5k steps, 2 h 25 min for the SDXL pose head on one A100 (p13).

**3. The guidance rule.**
- Written: ε̂ ← ε_θ(x_t) + w ∇ d(r, f_ψ(x_t)), a distance between the target r and the head's
  reading, in place of classifier guidance's log-probability (section 3.2, p3).
- Relative heads compare the reading on x_t with the reading on a second, reference diffusion
  process z_t (p4).
- Strength: every result uses the normalised update, step size bounded by α = 0.02 whatever the
  gradient's size (8.2, p12 to 13). The timestep rescaling of classifier guidance "has a
  negligible effect" (p13).
- Steps: 100 sampling steps; guidance over the first half for spatial control (`rg_ratio [0.0, 0.5]`),
  starting later (0.05) for appearance and identity so the reference's layout is not copied; never
  to the end, "to avoid the guidance signal competing with photorealism" (Table 1, p12).
- η = 1 (stochastic) for spatial control; η = 0 gives the improbable pictures of Figure 13 (p12, p14).
- In the released code the push edits the latent, not ε: before each ordinary step a fresh AdamW
  (lr = `rg_weight` = 0.02, weight decay 0) takes one step on x_t, with the loss summed over the
  unconditional and conditional halves of the CFG batch (`rg_pipeline.py:154-188`).

**4. Where it fails, in the paper's own words.**
- "our method generally requires more memory and runtime during sampling" (5.1, p8): on SDXL pose
  control 44.0 s against 14.2 s, 27.7 GB against 15.6 GB on an A100 (8.3, p13).
- "our method can sometimes produce improbable images, in the form of cartoonish or unrealistic
  imagery satisfying the readout constraints" (5.1, p8); "they start to diverge from the natural
  image manifold, tending towards cartoonish colors or surreal imagery with relatively texture-less
  scenes", fixed by a lower weight, stopping earlier, or another seed (8.6, p14).
- A head with no timestep input gives "improbable outputs that respect the input control but
  quickly fall off the manifold of natural imagery" (p13).
- "our pose readout can sometimes hallucinate extra detail, for example detecting fingers ... or
  detecting the dog" (10.3, p16); the depth readout "can struggle with producing a smooth and uniform
  depth across the entire object" (p16).

**5. Early steps.** "image composition is largely determined early in the diffusion process, where
our readouts show a central figure in the foreground within the first step of sampling and a coarse
silhouette within the first 10%" (10.1, p15, Figure 15).

## Caveats for PoE, carried forward

Each is inferred from the passages named unless a source says otherwise; none is tested. Named R1
to R10 so they do not collide with the Kim et al. read's C1 to C10. The last column judges
[the count head script](../../../scripts/rung2/train_count_head.py) as of commit `c5e61cec`.

| # | Caveat | Mitigation to try | Knob that measures it | Count head script |
|---|---|---|---|---|
| R1 | The paper's labels come from models that are good at their task. Ours come from GroundingDINO, which calls 80 of 105 hand-called hybrids two bodies (balanced accuracy 0.632; 71 of 105 with "animal"; [the rung 2 walk](../rung2-pair-aware-discriminator/EXECUTE_WALK.md)). A head that copies it learns "hybrid = plural", the exact split the main check scores | relabel with the 443 hand calls where they exist (the paper shows 100 suffice); drop or down-weight drawings whose boxes overlap; check the labels against hand calls before training | label source (detector with nouns, detector with "animal", hand calls); share of hybrid drawings labelled plural | does not meet: every target comes from `detector/labels.json`; the check uses hand calls, so it will show the problem but nothing in training corrects it |
| R2 | The paper calls timestep conditioning "very important"; without it guidance falls off the manifold (p13) | add the U-Net's timestep embedding, projected, to each projection (the paper's Figure 12) | the head with and without a time input, judged by the push's pictures | does not meet: `CountHead` takes features only; time reaches it only through the features |
| R3 | The paper reads all 9 SDXL up-block resnets at 128 px with learnt mixing weights and found the mid block did not help (p14, p16) | read the paper's 9 layers; or keep the coarse ones and test against them | layer set; per-layer mixing weights after training | partly: reads the mid block (which the paper dropped), up block 0 and up block 1, skips up block 2 (the finest), pools all to 32 px and sums them with no mixing weights. Coarse layers may be where bodies show, so the choice is defensible but untested |
| R4 | The paper trains on the clean picture noised forward. Our push acts on the product's own trajectory, where the fused body forms | train on trajectory states (as the script does) and check on both kinds | balanced accuracy on trajectory states against forward-noised states, step by step | partly: trains on plan 23's trajectory states (steps 0, 5, ..., 45 and the final latent) labelled from each finished drawing, which suits the push; but the check noises finished tiles forward, so training and check see different states |
| R5 | The composition shows within the first step and the silhouette within the first 10% (p15), and our cat and dog attention is already apart by step 2. A step-0 state is pure noise, so a label taken from the finished drawing asks the head to guess the outcome from noise | weight the loss toward steps where the outcome is readable; report accuracy per step; start the push at step 0 and end it by the middle, as Table 1 does | per-step balanced accuracy over steps 0 to 19; guidance window start and end | meets as a check: the bar is 10 of the first 20 of 50 steps, printed per step. Training includes step-0 states with finished-drawing labels, which may teach the prior and not the picture |
| R6 | Pose and edge targets are sparse, so the paper reweights empty and non-empty pixels (p5). A blob map is mostly zeros | reweight the blob pixels against the background; or train the count sum alone first | predicted count on blank and single-animal pictures | partly: no reweighting; the 0.1 × count-sum term pushes the total, not the blob placement |
| R7 | Guidance can satisfy the readout without the content: cartoonish, textureless pictures (p8, p14), and a pose head that "detects the dog" (p16). A push can draw two blobs without two animals, and the detector that made the labels is not an independent judge of the push | judge the push by hand calls from full-resolution tiles and a second measure; η = 1, weight at most 1, stop by mid-run (Table 1) | count by the head against hand calls after the push; the gap between them | not in scope: the script trains and checks, it does not push. Its check uses hand calls, which are independent of the detector |
| R8 | The released step moves most latent elements by about 0.02 per guided step whatever the gradient's size (elements with a gradient tiny beside Adam's epsilon move less; see the slip row on the push's optimizer) (one fresh AdamW step with lr 0.02; the first Adam step is close to lr times the gradient's sign). Over 50 guided steps that is up to 1.0 per element, with no link to the size of the product's missing correction; the paper's `rg_weight` 0.5 maps to a `loss_rescale` the Adam step ignores (inferred from `rg_pipeline.py:154-188` and `rg_operators.py:27-31`) | copy the released step first (10h reproduces it), then sweep lr; compare the step with the true correction's size on the toy | lr per element; ratio of push size to the correction's size by step | not in scope: no push in the script |
| R9 | A count reads bodies, not which animals: two dogs is plural, and the detector cannot tell a cat from a dog (64 of 64 false dog calls, project memory) | pair the count push with a separate identity read (CLIP crop names) or accept plurality as the rung's target | balanced accuracy against every negative (two dogs, wrong pair) beside the main one | meets as reported: the script prints the all-negatives balanced accuracy beside the main one and says wrong pairs are expected to fail |
| R10 | Held out must mean unseen. The 7 held-out pairs are kept out of plan 23 by a guard (`render_kim.py:47`), but cat × dog is one of plan 23's 80 training pairs at seeds 7 and 8, drawn from the seed's own starting noise. The check's `seed07__joint`, `seed07__product`, `seed08__joint` and `seed08__product` start from that same noise, so up to 4 of the 20 cat × dog check tiles may be pictures the head trained on (inferred: sampler settings and prompt wording not compared) | drop those 4 tiles from the check, or report the 7 whole pairs on their own | balanced accuracy on the 7 whole pairs alone against the full check | partly: the 7 pairs are clean; the cat × dog part of the check may leak |

Two costs sit beside the caveats rather than in them. A push needs a backward pass through the
U-Net at each guided step: 27.7 GB and 44 s on an A100 for SDXL pose control (p13), so a 24 GB 3090
needs gradient checkpointing, which the released code already uses. And every head result in the
paper is a regression onto a picture-shaped target; nothing in it reads a count of separate
objects, so the count head is an extension, not a reproduction.

## Slips in the printed paper

| Where | What | Evidence |
|---|---|---|
| section 3.2 (p3) against the code | the rule is printed as an edit to ε with weight w; the released code takes an AdamW step on the latent itself before the ordinary step | `rg_pipeline.py:154-188`; equivalent in direction for DDIM, different in what the weight scales |
| 8.2 (p13) against the code | the normalised rule divides by "the L2-norm"; a fresh AdamW normalises each element on its own, closer to the gradient's sign | `rg_pipeline.py:159-160` builds the optimizer inside each step, so no state carries over |
| training against guidance, in the code | the head's time input may differ between the two: training feeds it e(j), a step index, while guidance feeds it e(t), the timestep; at one noise level that is e(8) against e(991), whose time vectors have cosine −0.462. If it holds, a time input added for R2 must use one convention in both places | stated by the reproduce session's code walk of google-research/readout_guidance at `a8a699c`; read from code, not tested on a render |
| the push's optimizer, in the code | on SDXL the latents are float16, so the push runs 8-bit AdamW (`rg_pipeline.py:159`, `low_memory` set at line 264); most elements move about 0.02 per step, and an element whose gradient is tiny beside Adam's epsilon moves less: at gradient scale 1e-6 the median move is 0.0197 and the smallest 0.000022 (at 1e-3 the smallest is 0.0105; at 1 every element moves 0.02) | stated by the same session, measured in a Codex review of its code walk (journey 13's `rg_numbers.json`, `push.first_adam_step`, and the code-walk card's `rig.json`); not tested here |

## Disagreements

| Chunk | What | Both sides |
|---|---|---|
| R1 | what the head is taught | the paper: labels from a reliable off-the-shelf model; the count head: labels from a detector that calls most hybrids plural |

## Given back

| Round | To | What | Answer |
|---|---|---|---|

## Friction

| Chunk | What happened | Rule it points at |
|---|---|---|

## Timings

| Round | Start | End | Reader typed |
|---|---|---|---|
| setup, read-ahead and caveats | 2026-11-17 21:48 | 2026-11-17 21:56 | the invocation |
| 1 abstract | 2026-11-17 21:56 | | next |
