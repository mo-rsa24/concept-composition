# Read map: Refining Generative Process with Discriminator Guidance (lean)

Kim, Kim, Kwon, Kang and Moon, ICML 2023, arXiv 2211.17091. Code: github.com/alsdudrla10/DG.
Lean walk opened 2026-11-12. Step 1 of [the classifier-guided PoE route](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md).
Page numbers are the arXiv PDF's (48 pages). Plain readings in the rounds are the walk's own reading
of the quoted text.

## Setup

| Field | Value |
|---|---|
| Why it is read | rung A1 of the route trains a discriminator, joint pictures against the product's own draws, and adds its log-odds gradient to the product sampler; this paper is that method on unconditional and class-conditional image models |
| Grain | paragraph |
| Mode | lean; numbers only through `test`, the full walk only through `deep` |
| Rules | read only: no code, no plan edits; this map is the one file the walk writes |
| Bar | the closing section names the discriminator's inputs, loss, noise levels and the sampler formula, each with its page, and the route gets one row per settled fact |
| Time box | one sitting for sections 1 to 3, one for 4 to 7 and the appendix rows A1 needs |
| Paper text | the arXiv source `camera-ready.tex` and the PDF, fetched 2026-11-16 into the session scratchpad; not kept, refetch from arXiv 2211.17091 |
| Journey | none: classifier guidance and density ratio have no journey (gap); density ratio is on the shelf in the little book of generative AI foundations, p162 |

## Spine

The paper in one paragraph: a trained diffusion model's score is slightly wrong, so its samples
drift from the data. Train a second, small network to tell noised real pictures from noised
model pictures at every noise level. The gradient of its log-odds is exactly the missing piece of
the score when the network is perfect, so adding it at each sampling step pulls the samples back
toward the data. It needs no retraining of the diffusion model.

| Part | Pages | What it does in the argument | Route |
|---|---|---|---|
| Abstract | 1 | the claim: a discriminator trained after the score fixes the score at sampling time | ← start |
| 1 Introduction | 1 to 2 | the problem: reuse a frozen model and still improve its samples | |
| 2 Preliminary | 2 to 3 | background: the diffusion SDE, its reverse, the score loss; Algorithm 1 trains the discriminator | Algorithm 1 is the training recipe |
| 3.1 Correction of the score | 3 | mathematics: Theorem 1, the score plus the gradient of log (data density over model density) reverses the data exactly | the correction A1 measures |
| 3.2 Discriminator guidance | 4 | method: the loss (Eq. 4), the best discriminator's log-odds is that ratio, the guided SDE (Eq. 5) | the derivation round |
| 3.3 Theoretical analysis | 4 to 5 | mathematics: Theorem 2 and Table 1, the KL gap shrinks by how well the correction is learnt | a check the toy can run |
| 3.4 Optimality analysis | 5 | mathematics: the weight w tilts between data and model, (p_r)^w (p_θ)^(1−w) | the strength λ |
| 3.5 Connection with classifier guidance | 5 to 6 | method: Algorithm 2, the sampler with both guidances and their weights | the sampler change |
| 4 Related work | 6 | background: GAN hybrids, rejection and MCMC reweighting | |
| 5.1 Toy 2D case | 6 | how they checked it: a wrong two-mode score corrected by an MLP discriminator | the nearest thing to the exact toy |
| 5.2 Image generation | 6 to 8 | results: the discriminator network, FID tables, diagnostics by noise level | features the toy lacks |
| 5.3 Image-to-image | 8 | results: SDEdit plus guidance | skippable for A1 |
| 6 Discussion, 7 Conclusion | 8 to 9 | other losses for the same ratio, joint training as future work | |
| Appendix A.4, D.1 | 15, 21 to 24 | the forward-marginal choice, every training and sampling setting, the weight schedule | the recipe's numbers |

## Chunks

| # | Section | Paragraph | State | Note |
|---|---|---|---|---|
| 1 | Abstract | whole | tested | sentence 6: holds with a condition. 1D, one noise level, data N(1,1), model draws N(0,1.5²); mean abs score error on x in [-3,3]: model alone 1.134, plus optimal correction 0.0000, plus a trained quadratic logistic discriminator (2,000 real, 2,000 fake) 0.080. When the score used is not the score of the model's own draws (score of N(0.3,1), draws N(0,1.5²)): 0.700 alone, 0.862 corrected, so the correction made it worse. Script in the session scratchpad, test6/test6.py, not kept |
| 2 | 1 | the field | walked lean | Figure 2 (p2, the ImageNet samples; the round misnamed it Figure 1) shown on the reader's ask; reader asked the PoE caveats of test 6 and how the idea meets SDXL's text pathway, answered in the round |
| 3 | 1 | reuse of pretrained models | walked lean | reader asked what the fixed sampler is called, whether it needs MCMC or joint prompts, and asked to carry the caveats forward with mitigations and knobs (section below) |
| 4 | 1 | the approach, and contributions | tested | Figure 1 (p1) shown. Sentence 5 (term matches the data score at the optimal discriminator): same claim as abstract sentence 6, same verdict; Theorem 1's printed condition is that log p_θ equals its ELBO and s_θ(x, T) = ∇ log π (p3). Sentence 7 (stable, fast, cheap): holds with conditions. Figure 3 (p4) plots FID, not loss: 2.03 unguided, 2.05 at epoch 0, about 1.85 after 1 epoch, best 1.77 at epoch 10, 1.77 to 1.82 to epoch 60. Table 6 (p7), CIFAR-10: 200M score evaluations (480 hr) to train, 1M discriminator evaluations (10 min); the budget leaves out the pretrained ADM classifier the discriminator is built on. 1D two-mode toy (real sd 0.3, fake sd 0.8, 4,096 each, MLP 2-64-64-1): BCE within 0.0008 nats of the best by epoch 400, but correction error over true correction size 0.30, 0.21, 0.40 at t = 0.1, 0.3, 0.5 and 2.96, 18.77 at t = 0.7, 0.9. Script test7/test7.py in the session scratchpad, not kept |
| 5 | 2 | KL bound on latent variable models | tested | sentence 1 (end KL at most chain KL): holds. 1D, two noising steps (x0 ~ N(1, 0.5²), b = 0.5), KL in nats, end then chain: exact reversal 0.0000, 0.0000; exact reverse steps from N(0,1) 0.0117, 0.1351; a wrong model 0.5438, 0.7023; right finished pictures with a wrong chain 0.0000, 0.8050, which is the loose-bound case Theorem 1's condition rules out. Script test-s2p1/chain_kl.py in the session scratchpad, not kept |
| 6 | 2 | DDPM's forward and generative chains | walked lean ← current | |
| 7 | 2 | the forward SDE, Eq. 1 | not reached | |
| 8 | 2 | the reverse SDE, Eq. 2, and the generative SDE | not reached | |
| 9 | 2 | the score loss | not reached | |
| 10 | 2 | other ways to sharpen the score | not reached | |
| 11 | 2 | Algorithm 1 | not reached | |
| 12 | 3.1 | the generative process, Eq. 3, and Theorem 1 | not reached | |
| 13 | 3.2 | the loss, Eq. 4 | not reached | |
| 14 | 3.2 | the correction from the best discriminator, and Eq. 5 | not reached | the derivation round: prediction asked first; `prove` and `/picture-speak --derive` offered |
| 15 | 3.3 | Theorem 2 | not reached | |
| 16 | 3.3 | the gain and Table 1 | not reached | |
| 17 | 3.3 | the extra degree of freedom | not reached | |
| 18 | 3.4 | the weighted score | not reached | |
| 19 | 3.5 | classifier guidance as a joint score | not reached | |
| 20 | 3.5 | combining the two, Eq. 6, and Table 2 | not reached | |
| 21 | 3.5 | Algorithm 2 | not reached | |
| 22 | 4 | GAN hybrids | not reached | |
| 23 | 4 | rejection and MCMC reweighting | not reached | |
| 24 | 5.1 | the 2D toy and Figure 6 | not reached | |
| 25 | 5.2 | datasets | not reached | |
| 26 | 5.2 | the discriminator network | not reached | the training round: `example` offered |
| 27 | 5.2 | quantitative results | not reached | |
| 28 | 5.2 | ImageNet results | not reached | |
| 29 | 5.2 | the compute budget | not reached | |
| 30 | 5.2 | the regeneration check | not reached | |
| 31 | 5.2 | density ratio along trajectories | not reached | |
| 32 | 5.2 | fewer steps | not reached | |
| 33 | 5.2 | precision and recall | not reached | |
| 34 | 5.2 | loss by noise level | not reached | |
| 35 | 5.3 | image-to-image | not reached | |
| 36 | 6 | Bregman losses | not reached | |
| 37 | 6 | joint training and the f-divergence | not reached | |
| 38 | 7 | conclusion | not reached | |

## What the read-ahead found that the route has to decide

**The model density the discriminator learns is the final samples noised forward, never the
sampler's own intermediate states.** Appendix A.4 (p15) says p_θ^t is the forward-time marginal:
only the finished samples are kept as the fake set, and each is noised to a random t during
training. The route's plan describes q_sampler as what the summed-score sampler draws at each step.
Those are two different densities on the toy, because the toy card shows the summed noised scores
are not the score of any forward-noised density ("Where the samplers lose the pair"). So rung A1 has
three targets to compare against, not two: the true correction (the true noised product's score
minus the summed scores), the paper's learnable target (the joint's noised score minus the noised
score of the product's finished draws), and the trajectory version the route described.

**Theorem 1's equality needs an assumption PoE breaks.** Theorem 1 (p3) holds when the model's
log-likelihood equals its ELBO, which in practice means the model's score is the score of its own
forward-noised samples. Summed expert scores are not. The guided sampler then lands on the data
only as far as the score plus the learnt correction matches the true noised joint score, and on the
toy that can be computed exactly.

**A fourth fake set makes the correction exact for PoE.** The summed noised scores are the score
of the renormalised product of the noised experts, p_cat^t · p_dog^t / p_null^t, at each noise level. A
fake set drawn from that product at each t (not the sampler's finished pictures) makes the best
discriminator's log-odds gradient equal ∇ log p_joint^t minus the summed score, which is the
interaction term with no leftover. It also never goes stale, since it does not depend on the guided
sampler. On the toy that product is a Gaussian mixture with a closed form, so it can be drawn exactly;
on SDXL it needs MCMC at each noise level. Inferred in the walk, 2026-11-16, from test 6 on the
abstract; untested.

## Caveats for PoE, carried forward

Raised in the walk on 2026-11-16 (abstract test 6, and the reader's questions at introduction
paragraphs 1 and 2). Each is inferred unless a source is named; none is tested. Steps 7 and 9 of the
route read this section before the discriminator is designed.

| # | Caveat | Mitigation to try | Knob that measures it |
|---|---|---|---|
| C1 | The summed score plus the learnt correction reaches the joint only as far as the summed score is the score of the fake set; the paper's fake set (finished draws noised forward, p15) leaves a leftover | draw the fake set at each noise level from the product of the noised experts, which makes the correction exactly the interaction term; exact on the toy, MCMC per noise level on SDXL | fake-set kind (finished-noised, trajectory, per-noise-level product); on the toy, cosine of learnt against true correction per step |
| C2 | Building the per-noise-level product set on SDXL needs MCMC at every noise level, the Langevin cost the project already met | build it only over the early steps where the pair is decided; reuse the Langevin settings already learnt (c = 3, k = 20); or train on finished-noised fakes and measure the leftover on the toy first | number of noise levels given an MCMC fake set; MCMC steps per level |
| C3 | The discriminator learns every difference between real and fake, so a photo-vs-render or joint-vs-product style gap is learnt and pushed along with the pairing (a reviewer's comment asks the same of the LoRA) | real set from the same SDXL at the same settings; identical random style augmentation (colour, blur, compression) on both sets so style stops predicting the label; fused-but-realistic hard negatives in the fake set; many pair kinds (animal-animal, animal-object, object-object) so only "two separate things" separates the sets across all pairs | style-only control (two sets that differ only in style; the push must not change the count); count against style change under the push; held-out pairs |
| C4 | A real set of joint-prompt renders inherits the joint prompt's failures (wrong on 4 of 8 held-out seeds) | real set curated by eye, as the v58 LoRA targets were, or real photographs of pairs (150 in the inclusive-captions pool) with C3's augmentation | real-set kind (raw joint renders, curated joint renders, photographs) |
| C5 | Real and fake look most alike at high noise, where PoE loses the pair, so the classifier has the least signal where it is needed most | more training latents at high noise (importance sampling over t, as the paper does); a time input the classifier can use | training latents per noise level; on the toy, correction error by step against the exact best discriminator (Table 1, p4) |
| C6 | The gradient may add a pattern only the classifier responds to instead of changing content (recalled, a known failure of plain classifier gradients) | noise-aware training; push only over high-noise steps; average the gradient over small perturbations | judge the push with an independent measure (detector count with CLIP crop names, eye calls) beside the classifier's own score; the gap between them |
| C7 | Section 3.4's reading of w (w = 1 lands on the data) assumes no leftover | sweep w | w, with t_mid |
| C9 | A converged loss does not certify the correction: on a 1D two-mode toy the loss came within 0.0008 nats of the best while the gradient at high noise stayed 2 to 19 times the true correction's size (chunk 4's test) | judge the classifier by its correction error per step on the toy before trusting it on SDXL; regularise the input gradient, or fit the log-ratio's gradient directly | correction error by step against loss by step, on the toy |
| C10 | The steps where the correction matters vary by pair and by what is decided. The true correction on steps 0 to 9 then plain PoE gave single 2, hybrid 14, two bodies 16, two clear 0 of 32 tiles (4 pairs, seeds 9 to 16); the adapter cut at step 10 gave 1, 13, 18, 0. On lion × meerkat the adapter cut at step 10 loses the lion, at 25 keeps a meerkat on most seeds, at 50 turns the meerkat into a lion cub on 7 of 8 (impression, labels pending). Source: 1-lora-correction/artifacts/drips/experiments/second-animal-species/EXECUTE_WALK.md | treat the guidance window end as a per-pair knob; find each pair's window with the true correction first; consider two classifiers, one for two bodies early and one for the right species later | guidance window end, swept per pair; the four-level ladder per tile |
| C8 | Binding: an attention-based binding loss (recalled: Attend-and-Excite and its kin act on cross-attention maps) does not fit a classifier that sees only latents | an auxiliary head that predicts the count or each object's label, so its features must encode separation; or a separate binding term in the sampler, tested beside the push | auxiliary head on or off |

## Slips in the printed paper

| Where | What | Evidence |
|---|---|---|
| Table 8 (p22), DiT column | the maximum discriminator clip is printed 10⁻⁵ in both the training and sampling rows; every other column prints 1 − 10⁻⁵ | a maximum of 10⁻⁵ would pin the discriminator near zero |
| Appendix D.1 (p23), the Heun step | the second stage's weight is printed t_{t,2nd} in the text and w_{t,1nd} in the last line | the line before defines it as w_{t,2nd} |

## What A1 takes from this

Drafted from the read-ahead; each line is confirmed when the walk reaches its chunk.

**The training recipe.** Algorithm 1 (p3), Eq. 4 (p4), Table 8 (p22).
- Inputs: a noised picture x_t and its time t (the network is "noise-embedded"), plus the class
  for a class-conditional model. Real set: the data. Fake set: the frozen model's finished samples,
  generated once (25,000 to 1.28 million of them).
- Noise levels: every t from a minimum of 10⁻⁵ (0.01 for the latent model) to T, drawn by
  importance sampling, each picture noised fresh each time it is used.
- Loss: binary cross-entropy, half of each batch real and half fake, each term weighted by
  λ(t) = g²(t)/σ²(t). Batch 128 (512 on ImageNet), 7 to 280 epochs, one GPU, about 10 minutes on
  CIFAR-10 (Table 6, p7).
- Features: a frozen, noise-aware ImageNet classifier's encoder up to its last pooling layer
  (8 × 8 × 512), then a shallow U-Net encoder that is trained (section 5.2, p7). The latent
  model (LSGM) trains the U-Net from scratch with no pretrained part; the latent DiT model first
  trains a classifier of the same shape on latents. The 2D toy uses a 4-layer MLP of 256 units
  trained to convergence (p6).
- What would replace the features on the toy: an MLP from scratch on the 16 × 32 × 3 picture with
  a time embedding, which is the paper's own LSGM and 2D-toy route. On SDXL latents, the DiT route:
  a classifier trained on noisy latents first, then the shallow head.

**The sampler change.** Eq. 5 (p4), Algorithm 2 (p5), Appendix D.1 (p21 to 24).
- The score is replaced by s_θ + w · ∇ₓ log(d / (1 − d)), and the correction is multiplied by the
  same −(1 + η²)/2 · g² factor as the score. Nothing is rescaled to the score's norm.
- Weight w: 2 on CIFAR-10 (and only at Heun's first stage; 1.67 at the first and 0 at the second
  was best), 1 on DiT ImageNet, otherwise "adaptive": 2 for samples whose log density ratio is
  below 0 on odd steps, 1 otherwise.
- Schedule: guidance is on from T down to t_mid and off below it; t_mid is 0.01 for EDM and 0.1 for
  LSGM, ADM and DiT.
- The discriminator's output is clipped to [10⁻⁵, 1 − 10⁻⁵] (10⁻³ for LSGM) at sampling.
- Section 3.4 (p5): with the best discriminator, s + w·c is the score of
  (p_r)^w (p_θ)^(1−w), so w = 1 targets the data and w between 0 and 1 interpolates.

**The checks the toy can run.**
- Figures 6, 15 and 16 (p6, p17, p18): a wrong score made right by the learnt correction.
  The toy can do this with the true answer known.
- Table 1 (p4): the error E_{θ,φ} = ½∫ g² E‖c_true − c_φ‖² is E_θ for a blind discriminator and 0
  for the best one. The toy computes both exactly, per step.
- Figure 11 (p8): the discriminator's loss sits mostly at large diffusion time. On the toy the
  summed score is wrong mostly at high noise, so this should repeat.
- Figure 8 (p8), the log-ratio along trajectories: guided draws should end with the real data's log-ratio.
- Section 3.4 and Figure 5 (p5): a sweet spot in w; on the toy, the crisp-pair count of 300
  against w.

## Rows for the route's What we know so far table

Each is added to the route only on the reader's word.

| Fact, answer or limit | Source |
|---|---|
| Discriminator guidance adds w · ∇ log(d/(1−d)) to the frozen score at each step, with the score's own g² factor and no rescaling; w is 1 or 2 | Kim et al. 2023, Eq. 5 p4, Algorithm 2 p5, Table 8 p22 |
| The discriminator is trained once, BCE on noised real against noised finished model samples at every t, weighted g²/σ², half and half per batch | Algorithm 1 p3, Eq. 4 p4, Table 8 p22 |
| The fake set is the finished samples noised forward, never the sampler's trajectory; the learnt target is the joint's noised score minus the noised score of the product's draws | Appendix A.4 p15 |
| Theorem 1's exact correction assumes the model's score is the score of its own noised samples, which summed expert scores are not | Theorem 1 p3 |
| Guidance runs from pure noise to t_mid only (0.01 or 0.1) and the output is clipped to 10⁻⁵ from 0 and 1 | Appendix D.1 p21, Table 8 p22 |
| With no pretrained classifier the paper trains the head from scratch (LSGM, the 2D toy's 4 × 256 MLP) | section 5.2 p7, section 5.1 p6, Appendix D.1 p21 |

## Disagreements

| Chunk | What | Both sides |
|---|---|---|
| route | what q_sampler is | the route: the sampler's draws at each step; the paper: the finished draws noised forward (p15) |

## Given back

| Round | To | What | Answer |
|---|---|---|---|

## Friction

| Chunk | What happened | Rule it points at |
|---|---|---|
| 3 | the reader had to ask for the caveats raised in a round to be carried forward | caveats and limits raised in a round's answer are written to the map in the same round, with a mitigation and a measuring knob each |
| 2 | the round named the ImageNet samples Figure 1 from the source's figure order; the PDF numbers it Figure 2 | number figures from the compiled PDF, as SKILL.md already says for lemmas |

## Timings

| Round | Start | End | Reader typed |
|---|---|---|---|
| setup and spine | 2026-11-12 | | the invocation |
| 1 abstract | 2026-11-16 20:17 | 2026-11-16 20:30 | the resume invocation |
| 1 test 6, example; 2 intro paragraph 1 | 2026-11-16 20:30 | 2026-11-16 20:46 | test 6. example. next |
| 2 Figure 2, caveats, SDXL why; 3 intro paragraph 2 | 2026-11-16 20:46 | 2026-11-16 21:05 | four questions and next |
| 3 naming, MCMC, caveats, knobs, why, simplify; 4 intro paragraph 3 | 2026-11-16 21:05 | 2026-11-16 21:30 | six asks |
| 4 test 7, example | 2026-11-16 21:30 | 2026-11-16 21:50 | test 7, example |
| 4 test 5, why; LoRA and oracle comparison; 5 section 2 paragraph 1 | 2026-11-16 21:50 | 2026-11-16 22:10 | test 5, why, next, a question |
| 5 example, why, test 1; 6 section 2 paragraph 2 | 2026-11-16 22:10 | | example, why, test 1, next |
