# 🌍 Context: Latent CLIP, how its push differs from NoisyCLIP's

⬅️ Previous: [reading a latent with no VAE](02-reading-a-latent-with-no-vae.md) | 📋 [The pages of this method](01-overview.md#the-pages-of-this-method) | Next: [the linear probe on frozen vectors](04-the-linear-probe-on-frozen-vectors.md) ➡️

The push formula is the same for both readers: the gradient of the right-minus-wrong score, set to a fixed length and subtracted from CO3's predictions. What differs is the road the gradient takes back to the latent. NoisyCLIP's road passes a clamp and a rounding step that block it in places; Latent CLIP's passes only a division and a pooling, so it reaches every latent value, though in identical 2 × 2 blocks. This page follows that road backward, then sets Loop 9's proposed use beside the one Latent CLIP's authors published. The push equation itself is on [the overview](01-overview.md#the-push-by-what-moves), and NoisyCLIP's own backward road is drawn in [the sentence-score ladder](../../../2-classifier-guidance/artifacts/scenes/noisyclip-sentence-score/README.md#the-deep-dives-in-the-overviews-order).

<a href="../../../artifacts/notes/progress/noise-text-critic/way-2-latent-clip-push.png"><img src="../../../artifacts/notes/progress/noise-text-critic/way-2-latent-clip-push.png" width="200" alt="Flowchart top to bottom: a half-drawn SDXL latent of 4 by 128 by 128 at steps 0 to 15, divided by 0.13025 and average-pooled by 2 to 4 by 64 by 64 with no RGB step and no clamp, Latent CLIP's image encoder ViT-B/4-plus trained on clean latents only with no step input, the right sentence a photo of a puma and a condor and wrong ones through Latent CLIP's own text encoder, the checker D as S right minus highest S wrong, then in orange the gradient of D with respect to x_t straight through the pooling and a push of length 0.4 times CO3's prediction only on steps that read at 0.6 or better, ending at CO3's next step using its prediction minus the push"></a>

*Loop 9's design for this push, top to bottom. Blue boxes are inputs: the latent ("4 x 128 x 128, steps 0 to 15 of 50") and the sentences ("right: a photo of a puma and a condor; wrong: a photo of two pumas, ..."). Grey: "divide by 0.13025, average-pool by 2; 4 x 64 x 64, no RGB step, no clamp". Green: the image encoder, joined to the checker by a dotted arrow, "never tested on a noisy latent". Lilac: "Latent CLIP's own text encoder". The diamond is the checker D and the question W2 had to answer first. The two orange boxes are this page: "gradient of D with respect to x_t, straight through the pooling" and "push length 0.4 x size of CO3's prediction, only on steps that read at 0.6 or better".* ✅ *Read on the picture at full size and in `way-2-latent-clip-push.mmd`; a design drawing.*

## Table of contents

- [Words this page uses](#words-this-page-uses)
- [The gradient, back through the pooling](#the-gradient-back-through-the-pooling)
- [The names, taken apart](#the-names-taken-apart)
- [Beside NoisyCLIP's road](#beside-noisyclips-road)
- [Beside how its authors used the gradient](#beside-how-its-authors-used-the-gradient)
- [How it fails and how you would know it worked](#how-it-fails-and-how-you-would-know-it-worked)
- [Where this came from](#where-this-came-from)
- [Read more](#read-more)

## Words this page uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-gradient-back-through-the-pooling) ➡️

**Gradient ∇_{x_t} D**: for each of the latent's 65,536 values, how much D would rise per unit that value rises. It has the latent's shape, 4 × 128 × 128.

**Pooled cell**: one of the 4 × 64 × 64 values Latent CLIP reads, each the mean of a 2 × 2 block of the divided latent (see [the previous page](02-reading-a-latent-with-no-vae.md#the-input-by-what-moves)).

**Clamp**: cutting a value off at a limit. NoisyCLIP's colour map clamps each colour to 0 to 255.

**ReNO**: reward-based noise optimisation, the method Latent CLIP's authors plugged it into. It changes the starting noise of a one-step generator so that the finished picture scores higher.

**W2**: Loop 9's reading test, which scored right-against-wrong sentences with NoisyCLIP, frozen CLIP and Latent CLIP on cat and dog latents at steps 0 to 15, with no training, to see whether the score reads each of the five properties before anything is pushed.

**W3**: Loop 9's step planned to push CO3's renders with the sentence score, allowed only on steps where its reading test, W2, found the score readable.

## The gradient, back through the pooling

Navigation: ⬅️ [Words this page uses](#words-this-page-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-names-taken-apart) ➡️

**Each of the four latent values under a pooled cell receives the same share of that cell's gradient, 1 / (4 × 0.13025) = 1.919 times it, so the push is constant across every 2 × 2 block.** ✅ The pooled cell is u = (x₁ + x₂ + x₃ + x₄) / (4 × 0.13025), so ∂u/∂x_i = 1 / 0.521 = 1.919 for each of the four; nothing on the road clips or rounds. Read in `latent_to_latent_clip`, `common.py` line 125.

$$
\textcolor{#c86a0a}{\underbrace{\frac{\partial D}{\partial x_i}}_{\text{the gradient at one latent value}}}
= \textcolor{#1d6b3a}{\underbrace{\frac{\partial D}{\partial u}}_{\text{at its pooled cell}}}
\times
\textcolor{#6b7a88}{\underbrace{\frac{1}{4}}_{\text{the pool's share}}}
\times
\textcolor{#6b7a88}{\underbrace{\frac{1}{0.13025}}_{\text{the division}}}
$$

🧸 **Toy:** a 4 × 4 corner of one latent channel, which pools to 2 × 2 cells. Suppose Latent CLIP's score rises with the top-left and bottom-right cells and falls with the other two, with slopes (0.8, −0.3 | −0.3, 0.8) at the pooled cells (illustrative). Expected before computing: each 2 × 2 block of the latent gets one value, 1.919 times its cell's slope.

| Latent rows | Columns 1 and 2 | Columns 3 and 4 |
|---|---|---|
| rows 1 and 2 | 1.536, 1.536 / 1.536, 1.536 | −0.576, −0.576 / −0.576, −0.576 |
| rows 3 and 4 | −0.576, −0.576 / −0.576, −0.576 | 1.536, 1.536 / 1.536, 1.536 |

✅ Computed: 0.8 × 1.919 = 1.536 and −0.3 × 1.919 = −0.576. The slopes are illustrative; the factor 1.919 is the code's.

| Move | What changes in the gradient |
|---|---|
| a slope doubled at one pooled cell | its 4 latent values double, the other 12 stay |
| the pool removed (if the model accepted 128 × 128) | each latent value gets its own slope, 1 / 0.13025 = 7.678 times it, and the blocks disappear |
| the division removed | each latent value gets a quarter of its cell's slope in place of 1.919 times it; the push's direction does not change, and its length is fixed anyway |
| the push set to its fixed length | the blocks survive: dividing by the gradient's length rescales all 16 values by one number |

✅ Computed from the factors above. Whether a blocky push leaves a visible 2 × 2 pattern on the decoded picture was never tested; the reading note names it as a risk. 🔍 [The reading note](../../../artifacts/notes/progress/noise-text-critic/01-reading/latent-clip.md#7-the-three-ways-to-use-it-here), line 228.

## The names, taken apart

Navigation: ⬅️ [The gradient, back through the pooling](#the-gradient-back-through-the-pooling) | 📋 [TOC](#table-of-contents) | [Next](#beside-noisyclips-road) ➡️

**"Straight through the pooling" on the picture means the gradient meets nothing that cuts it.**

🔤 **The name:** straight through the pooling
- **straight through**: passing an obstacle without being stopped or bent. On the picture, the first orange box.
- **pooling**: the 2 × 2 mean from [the previous page](02-reading-a-latent-with-no-vae.md#the-names-taken-apart).
- **Formula, in those words:** a mean is a sum times a constant, so its slope with respect to each input is that constant, 1/4, everywhere and for every value. No input is ever cut off.
- **Consequence:** every one of the 65,536 latent values receives some push. On NoisyCLIP's road a colour clamped at 0 or 255 passes a slope of 0, and for a latent of unit spread the clamp holds about 26.8% of red values, 11.4% of green and 7.0% of blue. ✍️ [The NoisyCLIP reading note](../../../artifacts/notes/progress/noise-text-critic/01-reading/noisyclip.md#what-the-released-code-does), line 91, which marks those shares inferred from the weights.

🔤 **The name:** fixed push length
- **fixed**: set in advance and not adjusted afterwards.
- **push length**: the size of the push vector, here 0.4 times the size of CO3's guided prediction at that step. On the picture, the second orange box.
- **Formula, in those words:** p_t = 0.4 × ‖ε_g‖ × (∇D / ‖∇D‖).
- **Consequence:** the factor 1.919, or a gradient ten times longer because Latent CLIP is surer, changes where the push points and never how hard it pushes. 🔍 Inferred from the formula. The length 0.4 is the one the abstract checkers' design fixes for one checker. ✍️ [The W3 design](../../../artifacts/notes/progress/noise-text-critic/03-does-it-steer/design.md#how-the-push-enters-co3), line 79.

## Beside NoisyCLIP's road

Navigation: ⬅️ [The names, taken apart](#the-names-taken-apart) | 📋 [TOC](#table-of-contents) | [Next](#beside-how-its-authors-used-the-gradient) ➡️

**The two roads differ at every stage before the encoder, and the same at every stage after it.** ✍️ The NoisyCLIP column is the reading notes' account of its released code; the Latent CLIP column is ✅ read in `common.py` and `w3_steer.py`.

| Stage, read backward | Latent CLIP | NoisyCLIP |
|---|---|---|
| the push into CO3 | 0.4 × ‖ε_g‖ along ∇D, subtracted from both predictions | the same |
| the score D | right minus highest wrong | the same |
| the encoder | 12-layer ViT-B/4-plus, never trained on noise | ViT-L/14, its picture side retrained on noisy latents at steps 20 to 29 |
| the step back to the input | 4 × 64 × 64 pooled cells | a resize from 128 to 224 and a normalisation |
| the colour map | none | fixed 3 × 4 weights from latent channels to red, green and blue |
| the clamp | none | 0 to 255; clamped values pass no gradient, so W3 kept a soft clamp ready |
| rounding | none | `.byte()` in the released code, which has no gradient and must be dropped |
| what reaches x_t | every value, in identical 2 × 2 blocks | every value whose colours were not clamped |

> [!NOTE]
> W3's code chooses the clamp only for NoisyCLIP and records "Latent CLIP reads the latent directly" when Latent CLIP is chosen. ✅ `stage_clamp` in `w3_steer.py`, lines 189 to 190. The soft clamp is `soft_clamp`, lines 100 to 102, at β = 0.05.

🔤 **The name:** soft clamp
- **clamp**: cut a value off at a limit, here a colour at 0 and at 255.
- **soft**: with the corners rounded, so a value near or past a limit is bent toward it instead of cut flat, and some slope always gets through.
- **Formula, in those words:** start from the value, take off a smooth "how far past 255" and add back a smooth "how far below 0". The smooth version of max(z, 0) is softplus, log(1 + e^z). ✅ `soft_clamp`, `w3_steer.py` lines 100 to 102.

$$
\textcolor{#1d6b3a}{\underbrace{\text{soft}(v)}_{\text{the clamped colour}}}
= \textcolor{#1f5fd0}{\underbrace{v}_{\text{the colour}}}
- \textcolor{#6b7a88}{\underbrace{\tfrac{1}{\beta}\,\mathrm{softplus}\big(\beta\,(v-255)\big)}_{\text{how far past 255, smoothed}}}
+ \textcolor{#6b7a88}{\underbrace{\tfrac{1}{\beta}\,\mathrm{softplus}(-\beta\,v)}_{\text{how far below 0, smoothed}}}
$$

| The colour v | Hard clamp | Soft clamp at β = 0.05 | Slope passed back, hard | Slope passed back, soft |
|---|---|---|---|---|
| −50 | 0 | 1.58 | 0 | 0.076 |
| 128 | 128 | 128.00 | 1 | 0.997 |
| 255 | 255 | 241.14 | none, a corner | 0.5 |
| 300 | 255 | 253.00 | 0 | 0.095 |

✅ Computed this session from the formula; the colour values are the toy's.

- **Consequence:** at a colour of 300 the hard clamp passes a slope of 0, so no push reaches the latent values behind it, and the soft clamp passes 0.095. The price is that colours near a limit move: 255 comes out as 241.14. Raise β from 0.05 to 0.5 and the soft clamp moves closer to the hard one: 255 comes out as 253.61, and the slope at 300 falls to about 1.7 × 10⁻¹⁰. ✅ Computed.

## Beside how its authors used the gradient

Navigation: ⬅️ [Beside NoisyCLIP's road](#beside-noisyclips-road) | 📋 [TOC](#table-of-contents) | [Next](#how-it-fails-and-how-you-would-know-it-worked) ➡️

**Latent CLIP's authors used its gradient on a finished latent to change a one-step model's starting noise; Loop 9 proposed it on a half-drawn latent at each of up to 16 steps of a 50-step sampler, which nobody has tested.** ✍️ [The reading note](../../../artifacts/notes/progress/noise-text-critic/01-reading/latent-clip.md#4-how-its-gradient-enters), lines 33 to 39 and 166 to 184.

| | The authors' use, ReNO | Loop 9's proposal, W3 |
|---|---|---|
| model | SDXL-Turbo, one step, 512 pixels | SDXL with CO3, 50 steps, 1,024 pixels |
| what Latent CLIP reads | the finished latent | the half-drawn latent x_t at steps 0 to 15 |
| what the gradient changes | the starting noise | the noise prediction at that step |
| how many times | 50 gradient steps of SGD at learning rate 5.0, momentum 0.9, gradient length clipped to 0.1 | once per pushed step, at a fixed length |
| what it steers toward | the prompt (a CLIPScore reward), human preference, or away from a negative sentence | the right sentence over the highest wrong one |
| result | [GenEval two-object](../../../1-lora-correction/report/measuring-composition/what-each-scorer-says-about-two-objects.md#geneval-two-object) 0.66 to 0.82 and counting 0.45 to 0.60 for B/4-plus; spatial relation 0.24 to 0.24 | gated on the reading test, whose result is in [the reading finding](../../../2-classifier-guidance/report/the-sentence-score/can-a-sentence-score-read-the-five-properties.md) |

The ReNO column is ✍️ the reading note's account of the paper (p. 3 to 7) and of its released code; the W3 column is ✅ read in `w3_steer.py` and the W3 design.

**The ReNO "how many times" cell stacks four settings; here they are one at a time.**

- **Gradient step**: moves the starting noise a little along the gradient of the reward.
- **SGD**: stochastic gradient descent, the rule that takes those steps; it descends the negative reward, which raises the reward, and [the gradient descent card](../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Gradient%20Descent.md#where-to-learn-this) works it.
- **Clipped to 0.1**: a gradient longer than 0.1 is shrunk to length 0.1 before it is used.
- **Learning rate 5.0**: multiplies it into the step, so a clipped step is 5.0 × 0.1 = 0.5 long; [the learning rate card](../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Learning%20Rate.md#where-to-learn-this) shows what a larger or smaller one does.
- **Momentum 0.9**: adds 0.9 of the previous step to each new one, so steps in a steady direction grow.

🧸 **Toy:** a gradient of length 1 pointing the same way at every step (illustrative). Expected before computing: clipping shrinks it to 0.1, and the steps then grow under momentum.

| Step | Without momentum | With momentum 0.9 | Distance moved so far, with momentum |
|---|---|---|---|
| 1 | 0.5 | 0.5 | 0.5 |
| 2 | 0.5 | 0.95 | 1.45 |
| 3 | 0.5 | 1.355 | 2.805 |
| 4 | 0.5 | 1.720 | 4.524 |
| 5 | 0.5 | 2.048 | 6.572 |

✅ Computed this session: momentum stored as the running sum v ← 0.9 v + g, the step 5.0 × v. After 5 steps the noise has moved 6.572 with momentum against 2.5 without, and a steady step tends to 0.5 / (1 − 0.9) = 5.0. Remove the clip and the first step is 5.0 × 1 = 5.0, ten times longer. The toy's gradient is invented; 5.0, 0.9 and 0.1 are ReNO's settings as the reading note gives them.

🔤 **The name:** ReNO (reward-based noise optimisation)
- **reward**: a score the method tries to raise, here Latent CLIP's similarity to the prompt.
- **noise**: the generator's starting noise, the only thing changed.
- **optimisation**: repeated small steps that each raise the reward, 50 of them.
- **Consequence:** the gradient never touches a noisy latent, because a one-step generator turns the starting noise straight into a finished latent, and Latent CLIP reads that. So the authors' results say the gradient can raise a benchmark when it reads clean latents; they say nothing about what it does at step 5 of 50. 🔍 The reading note, line 85, marks this reading inferred.

## How it fails and how you would know it worked

Navigation: ⬅️ [Beside how its authors used the gradient](#beside-how-its-authors-used-the-gradient) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**W3's design writes three checks to run before any picture is judged.** ✍️ [The W3 design](../../../artifacts/notes/progress/noise-text-critic/03-does-it-steer/design.md), lines 101 to 103. Whether the push acts at all rests on the reading test, in [the reading finding](../../../2-classifier-guidance/report/the-sentence-score/can-a-sentence-score-read-the-five-properties.md); [the W3 review](../../../artifacts/notes/progress/noise-text-critic/03-does-it-steer/review.md), line 7, records the step.

| Check | What shows it worked |
|---|---|
| zero push | the push path multiplied by 0 reproduces CO3's own run, with a largest latent difference under 1e-5 |
| the push acts | the real push changes the final latent, and its length equals 0.4 × ‖ε_g‖ on every pushed step |
| a random push of the same length | CO3 with the real push draws more faithful pairs than CO3 with the random one |

💥 **Broken run (without):** a gradient of length 0, as from a score that does not depend on x_t.
- **Predict:** what does the push add?
- **Wrong:** nothing; the direction ∇D / ‖∇D‖ is undefined.
- **Right**, as the code handles it: the step is counted as dead and the push is set to zero, so the run still completes and reports how many steps went dead. ✅ `SentencePusher`, `w3_steer.py` lines 162 to 165.

> [!WARNING]
> A push that runs, changes the picture and has the right length can still steer nothing useful. The project's earlier CLIP checker raised its own score by 1 to 13.5 logits over a random push without moving the animals. ✍️ [The NoisyCLIP reading note](../../../artifacts/notes/progress/noise-text-critic/01-reading/noisyclip.md#what-to-copy-what-to-avoid), line 182. The random push of the same length is the control that would catch this.

## Where this came from

Navigation: ⬅️ [How it fails and how you would know it worked](#how-it-fails-and-how-you-would-know-it-worked) | 📋 [TOC](#table-of-contents) | [Next](#read-more) ➡️

| What | How it was established | When |
|---|---|---|
| The input path and its slope 1.919 | Read in `common.py` line 125; the slope computed | 2026-11-24 |
| The push's length, the gate, the dead-step rule, the clamp choice, the soft clamp | Read in `w3_steer.py`, lines 34 to 42, 100 to 102, 136 to 169 and 189 to 190 | 2026-11-24 |
| The toy's gradient values and the moves | Computed this session; the slopes are illustrative | 2026-11-24 |
| NoisyCLIP's road: colour map, clamp shares, `.byte()`, resize | Stated by the NoisyCLIP reading note, lines 78 to 91 | 2026-11-23 |
| ReNO's settings and results | Stated by the Latent CLIP reading note, lines 33 to 39 and 164 to 186 | 2026-11-23 |
| The three checks | Stated by the W3 design and review | 2026-11-24 |
| The opening picture's labels | Read on the picture at full size and in its `.mmd` source; checked with `is_showable` | 2026-11-24 |

## Read more

Navigation: ⬅️ [Where this came from](#where-this-came-from) | 📋 [TOC](#table-of-contents)

| What | Where |
|---|---|
| A push as a step on the input with the weights held, worked on one small network | [the readout guidance chapter, the push](../../../../../goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/textbook/06-readout-guidance-on-one-small-network.md#push-a-step-on-the-input-not-the-weights) |
| Where a fixed-length push enters CO3 in this project's code | [the readout checker's push page](../02-the-readout-checker/03-the-push-and-its-cap.md#where-the-push-enters-the-sampler) |
| W3's full design, every bar written before any reading | [the W3 design](../../../artifacts/notes/progress/noise-text-critic/03-does-it-steer/design.md) |

⬅️ Previous: [reading a latent with no VAE](02-reading-a-latent-with-no-vae.md) | 📋 [The pages of this method](01-overview.md#the-pages-of-this-method) | Next: [the linear probe on frozen vectors](04-the-linear-probe-on-frozen-vectors.md) ➡️

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
