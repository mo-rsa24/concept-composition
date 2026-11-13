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
| 1 | Abstract | whole | not reached | |
| 2 | 1 | the field | not reached | |
| 3 | 1 | reuse of pretrained models | not reached | |
| 4 | 1 | the approach, and contributions | not reached | |
| 5 | 2 | KL bound on latent variable models | not reached | |
| 6 | 2 | DDPM's forward and generative chains | not reached | |
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

## Timings

| Round | Start | End | Reader typed |
|---|---|---|---|
| setup and spine | 2026-11-12 | | the invocation |
