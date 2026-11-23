# Controlling Latent Diffusion Using Latent CLIP

arXiv 2503.08455v1, read from the full PDF (21 pages, appendix included). The PDF prints no venue. The released code in three public repositories was read and not run, and no weights were downloaded.

## What it does

Latent-CLIP is a CLIP model whose image tower reads a 64 by 64 by 4 SDXL-family VAE latent directly, so no VAE decode is needed before scoring (p. 1, p. 3). The authors train it from scratch on 2.7 billion VAE-encoded images with captions (p. 1, p. 3). They then use its similarity as a reward in ReNO, which optimises the starting noise of one-step SDXL-Turbo over 50 gradient steps, and report benchmark scores close to pixel CLIP at about 21% less run time (p. 4 to 7).

## What the model or score reads

**Image tower.** It takes a latent of shape 64 by 64 by 4, cuts it into patches, and runs a vision transformer (p. 3). Two sizes are released: Latent-ViT-B/8 with 64 tokens, and Latent-ViT-B/4-plus with 256 tokens (p. 3, Table 6 on p. 12).

**Text tower.** Its own CLIP text transformer, trained with the image tower. The width and embedding size are 512 for B/8 and 640 for B/4-plus (Table 6, p. 12).

**Score.** Cosine similarity between the image and text embeddings (p. 2). The model receives no timestep or noise level (library `src/latent_clip/transformer.py:869-889`).

**Latent scale.** The input is the unscaled VAE latent, meaning SDXL's latent divided by 0.13025 (demo notebook cell 11; ReNO fork `models/RewardStableDiffusionXL.py:326`).

## What it trained on

**Data.** LAION-2B-en and COYO images, 2.7 billion in total, resized through augmentations to 512 by 512 and VAE-encoded to 64 by 64 by 4. Each image has four augmented latents (p. 3).

**Pairs and loss.** Each latent is paired with its caption under the OpenCLIP contrastive training code (p. 2, p. 3).

**Training length.** Effective global batch size 81,920, until 34 billion latents had been sampled with repetition. AdamW, weight decay 0.2, cosine schedule with peak learning rate 0.001 (p. 3). B/8 took 128 A100 GPUs for 4 days and 9.5 hours, and B/4-plus took 256 A100 GPUs for 4 days and 1.9 hours (p. 3).

**Noise levels.** Clean latents only. The training loader reads `latent.pt` and adds no noise or timestep (library `src/training/data.py:394-401`). The authors note that VAE-encoded latents may differ slightly from generated ones (p. 3).

**PickScore variant.** One reward is Latent-CLIP fine-tuned on Pick-a-Pic, which holds over 500,000 human preference judgements (p. 5).

## How its score or gradient enters

A gradient enters through ReNO, which changes the starting noise $\epsilon$ and leaves the model alone (p. 3):

$$
\epsilon^* = \arg\max_{\epsilon}\Big(\sum_i \lambda_i R_i(G(\epsilon, c), c) - L_{\text{reg}}(\epsilon)\Big)
$$

$R_i$ is the $i$-th reward, $\lambda_i$ its weight, $G(\epsilon,c)$ the latent generated from noise $\epsilon$ and prompt $c$, and $L_{\text{reg}}$ a regulariser on the noise (p. 3). The model is SDXL-Turbo with one denoising step at 512 pixels, optimised for 50 gradient ascent steps (p. 4, p. 5). The gradient runs from the Latent-CLIP score, through the finished latent and the one-step U-Net, back to $\epsilon$ (`training/trainer.py:101-120, 164`).

The paper prints no reward weights, learning rate or gradient cap. The released code's values are in [how its gradient enters](#4-how-its-gradient-enters) below.

## When adding scores is valid

This paper does not address it. The only sum it uses is the weighted reward sum in the ReNO objective (p. 3), and it says nothing about when that sum is valid.

## Which of the five properties it bears on

All numbers below are from generated images scored by each benchmark's own evaluator after ReNO. They use the CLIPScore reward on SDXL-Turbo, on a 0 to 1 scale where higher is better. None measures Latent-CLIP's own ability to tell a right sentence from a wrong one.

**Counting.** GenEval counting is 0.45 with no reward, 0.53 for B/8, 0.60 for B/4-plus, 0.60 for pixel CLIP ViT-g/14 and 0.62 for the full ReNO ensemble (Table 3, p. 7).

**Attribute binding.** T2I-CompBench colour is 0.62 with no reward, 0.68 for B/8, 0.69 for B/4-plus, 0.69 for ViT-g/14 and 0.78 for ReNO (Table 2, p. 6). GenEval colour attribution is 0.20 with no reward, 0.32 for B/8, 0.30 for B/4-plus, 0.27 for ViT-g/14 and 0.36 for ReNO (Table 3, p. 7). On "a photo of a white bottle and a blue sheep", both Latent-CLIP rewards show a blue bottle and a white sheep, and the ReNO ensemble shows the right colours (Fig. 6, p. 17, read from the figure). This is one example chosen by the authors.

**Spatial relation.** T2I-CompBench spatial is 0.24 with no reward, 0.25 for B/8, 0.24 for B/4-plus, 0.23 for ViT-g/14 and 0.25 for ReNO (Table 2, p. 6). The authors say spatial tasks "remain challenging" (p. 5).

**Position and size.** GenEval position is 0.09 with no reward, 0.12 for B/8, 0.14 for B/4-plus, 0.11 for ViT-g/14 and 0.12 for ReNO (Table 3, p. 7). Size is not tested.

**Prompt match.** GenEval mean is 0.54 with no reward, 0.60 for B/8, 0.62 for B/4-plus, 0.62 for ViT-g/14 and 0.64 for ReNO. GenEval two-object is 0.66 with no reward, 0.78 for B/8, 0.82 for B/4-plus, 0.82 for ViT-g/14 and 0.86 for ReNO (Table 3, p. 7).

## Unseen concepts

This paper does not test held-out concepts. Its ImageNet test is zero-shot classification from class names (p. 4), and the paper does not check whether those classes appear in the training captions.

## Numbers

| Metric (scale, direction) | Latent-CLIP | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Top-1 zero-shot accuracy, %, higher is better | B/8 68.8; B/4-plus 73.5 | same-size pixel CLIP: B/32 D1B 72.8; B/16 D1B 73.5 | ImageNet validation, VAE-encoded at 512 | SDXL VAE | p. 5, Table 1 |
| Top-1 accuracy on generated latents read directly, %, higher is better | B/8 81.7; B/4-plus 84.6 | decoded pixel CLIP: B/32 D1B 84.5; B/16 D1B 85.1 | generated ImageNet | SDXL-Turbo | p. 5, Table 1 |
| Colour binding, 0 to 1, higher is better | B/4-plus 0.69 | ViT-g/14 0.69; no reward 0.62; ReNO 0.78 | T2I-CompBench | SDXL-Turbo | p. 6, Table 2 |
| Spatial relationship, 0 to 1, higher is better | B/4-plus 0.24 | no reward 0.24; ReNO 0.25 | T2I-CompBench | SDXL-Turbo | p. 6, Table 2 |
| Two-object score, 0 to 1, higher is better | B/4-plus 0.82 | ViT-g/14 0.82; no reward 0.66; ReNO 0.86 | GenEval | SDXL-Turbo | p. 7, Table 3 |
| Counting, 0 to 1, higher is better | B/4-plus 0.60 | ViT-g/14 0.60; no reward 0.45; ReNO 0.62 | GenEval | SDXL-Turbo | p. 7, Table 3 |
| Position, 0 to 1, higher is better | B/4-plus 0.14 | ViT-g/14 0.11; no reward 0.09; ReNO 0.12 | GenEval | SDXL-Turbo | p. 7, Table 3 |
| Total ReNO time per image, seconds, lower is better | B/4-plus 9.01 | same-size pixel CLIP B/16 D1B 11.70 | 50 gradient steps on an A100 | SDXL-Turbo | p. 6, Table 2 |
| Probability that an output is inappropriate, 0 to 1, lower is better | B/4-plus 0.16 | pixel ViT-B/16 0.19; no reward 0.32; SLD 0.13 | I2P, all categories | SDXL-Turbo | p. 7, Table 4 |

**Where the text and tables differ.** The text says counting "rises from 53% to 60%" (p. 5). In Table 3, 0.53 is B/8 and 0.60 is B/4-plus, and the no-reward baseline is 0.45 (p. 7). The figures labelled "Inference Step" and "t = 1" to "t = 50" (p. 18 to 20) plot the 50 ReNO gradient iterations on a one-step model, so they are not diffusion timesteps (p. 6, p. 7) (inferred).

## What to copy, what to avoid

- Copy the direct latent input. It has no RGB clamp or 8-bit step, so the gradient path into $x_t$ is cleaner than NoisyCLIP's (inferred).
- Copy the scale rule: divide SDXL's latent by 0.13025 before the model. Feeding the scaled latent would put the input about 7.7 times below the range it was trained on (inferred).
- Avoid reading the ReNO results as evidence for a push on $x_t$. Every published score reads a finished latent, and the gradient updates the starting noise of a one-step model (p. 3 to 5) (inferred).
- Avoid expecting a position or relation gain from this score. Latent-CLIP's spatial score stayed at 0.24 to 0.25 against 0.24 with no reward, which matches this project's result that a CLIP checker raised its own score without moving the animals (Table 2, p. 6) (inferred).
- Avoid assuming it binds attributes. The colour-attribution gain is real on average (0.20 to 0.30 or 0.32), but the printed bottle and sheep example moves the colour onto the wrong object (p. 7, p. 17) (inferred).
- Report it with the same right-against-wrong pairwise accuracy as NoisyCLIP, with pure noise as a baseline, since the paper has no such measurement (inferred).

## Can this project use it

**Can it be used directly with PoE?** Not as released. The push formula works with any differentiable score, so PoE itself is no obstacle, but the model rejects SDXL's 128 by 128 latent and has never been shown a noisy one (inferred).

**How does it compare with NoisyCLIP?** Latent-CLIP has the cleaner input path and a gradient that has been used in generation. NoisyCLIP is the one trained on noisy SDXL latents at SDXL's own resolution (inferred).

**Can its gradients steer sampling?** The gradient exists and is computable with respect to $x_t$. The published evidence covers a different use: updating the starting noise of a one-step model from a finished latent. Nothing shows what a push on a noisy $x_t$ does (inferred).

### 1. Which latent space

**Trained for.** The SDXL VAE family at 512 pixels, which gives 64 by 64 by 4 latents, the default for SDXL-Turbo (p. 2, p. 3). The authors state it was built within SDXL's latent space and that FLUX's 16-channel space is future work (p. 8). Both released configs name `madebyollin/sdxl-vae-fp16-fix` as the encoder ([B/4-plus config](https://huggingface.co/wendlerc/latent-clip-b-4-512-plus-34b-80k/blob/main/Latent-ViT-B-4-512-plus.json)). Its scaling factor is 0.13025, the same as the [stock SDXL VAE](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/vae/config.json) this project loads (`poe_repair/_sdxl/runtime.py:114`).

**SDXL's 128 by 128 latents are not accepted as they stand.**

| Requirement | SDXL latent in this project | Latent-CLIP expects | Match |
|---|---|---|---|
| Channels | 4 | 4 | yes |
| Size | 128 by 128 | 64 by 64 | **no** |
| Scale | scaled by 0.13025 | unscaled, so divide by 0.13025 | needs one division |
| Noise | noisy $x_t$ at steps 0 to 15 | clean latent | **no** |

A 128 by 128 input yields 257 tokens for B/8 and 1,025 for B/4-plus, against position tables of 65 and 257, so the addition at `transformer.py:492` fails (inferred from the code, not run). The checkpoint loader does not resize the table either, because `resize_pos_embed` looks up `visual.positional_embedding` (`model.py:488`) and this checkpoint stores it under `visual.latent_vit.` (`factory.py:115`) (inferred). The authors name this limit for SDXL at its other native resolution (p. 9).

**What would have to change.** The smallest change is `F.avg_pool2d(x_t / 0.13025, 2)`, which gives a 64 by 64 unscaled latent. The authors tried this once in their [library notebook](https://github.com/wendlerc/latent_clip/blob/renamed/notebooks/Minimum%20Usage.ipynb) (cells 34 to 38): 10 clean 1024-pixel latents from a pipeline named miniSDXL, where 9 of 10 images ranked their own caption first among the 10 (read from the saved heat map). That test is outside the paper, uses clean latents and covers 10 images. The other route is to interpolate the position table to the larger grid, which nothing in the paper or code tests (inferred).

### 2. Is it released

Yes. Weights and code are public.

| Part | Location | Size or commit | Licence |
|---|---|---|---|
| B/8 weights | [`wendlerc/latent-clip-b-8-512-34b-80k`](https://huggingface.co/wendlerc/latent-clip-b-8-512-34b-80k), `checkpoints/epoch_34.pt` | 1,926,593,435 bytes (1.93 GB) | none stated |
| B/4-plus weights | [`wendlerc/latent-clip-b-4-512-plus-34b-80k`](https://huggingface.co/wendlerc/latent-clip-b-4-512-plus-34b-80k), `checkpoints/epoch_34.pt` | 2,260,598,171 bytes (2.26 GB) | none stated |
| Same-size copies | [`anonymous-author-42/latent-clip-b-8`](https://huggingface.co/anonymous-author-42/latent-clip-b-8) and [`latent-clip-b-4-plus`](https://huggingface.co/anonymous-author-42/latent-clip-b-4-plus) | identical byte counts, hashes not compared | none stated |
| Model library | [`wendlerc/latent_clip`, branch `renamed`](https://github.com/wendlerc/latent_clip/tree/renamed) | `716e61a` | MIT-style, inherited from OpenCLIP |
| ReNO with latent rewards | [`jsonBackup/Latent-ReNO`](https://github.com/jsonBackup/Latent-ReNO) | `4fff53d` | MIT |
| Demo | [`jsonBackup/Latent-CLIP-Demo`](https://github.com/jsonBackup/Latent-CLIP-Demo) | `7b9a4eb` | no licence file |

The repositories `wendlerc/latent-clip-b-4-512-13b-80k` and `wendlerc/latent-clip-b-4-512-plus-13b-80k` hold a README and no weights. The demo README lists a `Latent_ReNO/` folder that the repository does not contain, and the notebook clones it from the separate ReNO repository (demo notebook cell 0).

**How a model is loaded** (demo notebook cells 0 to 3):

```python
# pip install git+https://github.com/wendlerc/latent_clip.git@renamed
model, _, preprocess = latent_clip.create_model_and_transforms(
    "Latent-ViT-B-4-512-plus", pretrained="<path>/epoch_34.pt", device=device)
tokenizer = latent_clip.get_tokenizer("Latent-ViT-B-4-512-plus")
```

**Three things to handle before loading.** The checkpoint is a Python pickle and the library calls `torch.load` with no `weights_only` argument (`factory.py:88`), so a file from the Hub can run code when loaded on a version of torch that defaults to full unpickling (inferred). The constructor also fetches `madebyollin/sdxl-vae-fp16-fix` (`transformer.py:784`), so that VAE must be cached on the cluster. The demo environment pins PyTorch 2.3.0, diffusers 0.28.0 and transformers 4.38.2 (`environment.yml:182, 253, 293`), and the `co3` environment was not compared with these.

### 3. Does it read noise

No published result shows that it does.

**Training.** Clean VAE-encoded latents only (p. 3; `src/training/data.py:394-401`). The model has no timestep input (`transformer.py:869-889`).

**Testing.** Every test scores a clean or finished latent. The generated ImageNet set is made by noising real latents to a 66% noise level and then denoising them with SDXL-Turbo, and Latent-CLIP reads the denoised result (p. 4).

**Guidance.** ReNO reads the finished output of the one-step generator, $G(\epsilon,c)$, and never an intermediate $x_t$ (p. 3 to 5; `trainer.py:101-120`). On a 50-step SDXL run, the nearest equivalent would be a clean estimate $\hat x_0$, which is poor at steps 0 to 15 (inferred).

**Measurement at high noise.** None exists in the paper or the released notebooks.

**How far outside its training range this project's steps are.** Under SDXL's schedule with 50 DDIM steps, the weight on the clean picture in $x_t$ is small (computed from the [SDXL scheduler config](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/scheduler/scheduler_config.json), inferred):

| Step of 50 | Timestep | Weight on clean latent, $\sqrt{\bar\alpha_t}$ | Weight on noise, $\sqrt{1-\bar\alpha_t}$ |
|---|---|---|---|
| 0 | 981 | 0.076 | 0.997 |
| 5 | 881 | 0.131 | 0.991 |
| 10 | 781 | 0.209 | 0.978 |
| 15 | 681 | 0.308 | 0.952 |

After dividing by 0.13025 and average pooling by 2, the noise at step 0 has a standard deviation of about 3.8 per latent pixel. Clean encoder latents have a per-channel standard deviation of 5.7 to 10.5 (library notebook cell 23). So the input magnitude is plausible while its content is mostly noise the model never saw (inferred).

### 4. How its gradient enters

**Printed update.** The ReNO objective above (p. 3). The released code minimises this loss on the starting noise $\epsilon$:

$$
L(\epsilon) = w\Big(100 - e^{\text{logit scale}}\cos\big(\hat V(G(\epsilon,c)/0.13025),\,T(c)\big)\Big) + 0.01\Big(\tfrac12\|\epsilon\|^2 - (d-1)\log\|\epsilon\|\Big)
$$

$\hat V$ is the Latent-CLIP image tower, $T$ its text tower, $w$ the reward weight and $d$ the number of noise dimensions (`rewards/latent_clip.py:66-73`, `training/trainer.py:135-139`).

| Item | Value | Source |
|---|---|---|
| Differentiates through | Latent-CLIP, the finished latent, one SDXL-Turbo U-Net step | `trainer.py:101-120, 164` |
| Updated variable | the starting noise $\epsilon$ | `helper.py:148` |
| Steps | 50 gradient iterations on a 1-step generator | p. 4; `helper.py:33-34` |
| Weight $w$ | 0.01 default; the notebook uses 0.05 or 0.1; the paper's benchmark weights are not printed | `helper.py:51`; demo notebook cells 17, 19 |
| Normalisation | embeddings normalised to unit length | `rewards/latent_clip.py:83-84` |
| Cap | gradient norm clipped to 0.1 | `helper.py:37`; `trainer.py:165` |
| Optimiser | SGD, learning rate 5.0, Nesterov momentum 0.9 | `helper.py:32, 35-36`; `training/optim.py` |
| Steers toward | the prompt (CLIPScore), human preference (PickScore), or away from a negative sentence with $R = 1 - \text{CLIPScore}(z, N)$ | p. 5, p. 6 |
| Model | SDXL-Turbo, one step, 512 pixels | p. 4, p. 5 |

**Layout against texture, style or a metric.** The paper does not measure layout separately from texture or style. Its reward raises the benchmarks for object presence and attributes: two-object from 0.66 to 0.82 and colour attribution from 0.20 to 0.30 for B/4-plus (Table 3, p. 7). The benchmarks closest to layout barely move: spatial relation is 0.24 before and 0.24 after for B/4-plus (Table 2, p. 6), and position goes from 0.09 to 0.14 (Table 3, p. 7). Whole images change under the reward in Figs. 4 and 6 (p. 15, p. 17), which is expected when the starting noise changes. This result cannot separate a change of layout from a new sample that happens to score better (inferred).

### 5. The five properties

The strict reading: the paper measures generated images after optimisation with benchmark evaluators. It never tests whether the Latent-CLIP score itself separates a right sentence from a wrong one for any property, and no test uses a noisy latent.

| Property | Tested? | What the paper shows (B/4-plus, CLIPScore reward) |
|---|---|---|
| Counting | on generated images | 0.45 with no reward to 0.60 (p. 7) |
| Attribute binding | on generated images | colour attribution 0.20 to 0.30 (p. 7); one printed example swaps the colours (p. 17) |
| Spatial relation | on generated images | 0.24 to 0.24, no gain (p. 6) |
| Position and size | position only | 0.09 to 0.14 (p. 7); size not tested |
| Prompt match | on generated images | GenEval mean 0.54 to 0.62 (p. 7) |

### 6. Side by side with NoisyCLIP

NoisyCLIP facts come from [the NoisyCLIP note](noisyclip.md).

| | Latent-CLIP | NoisyCLIP |
|---|---|---|
| **Input** | 64 by 64 by 4 unscaled latent, read directly (p. 3) | noisy SDXL latent through a fixed linear map to RGB, clamped, then CLIP's image input |
| **Latent space** | SDXL VAE family at 512 pixels (p. 2, p. 8) | SDXL at its default 128 by 128 latent |
| **Trained noise levels** | none, clean latents only (p. 3) | latents 20 to 29 of 50 |
| **What was tuned or frozen** | both towers trained from scratch, VAE frozen (p. 8; `transformer.py:791-795`) | image tower fine-tuned; text tower and both projection heads equal base CLIP |
| **Text encoder** | its own CLIP text tower, 512 or 640 wide (p. 12) | frozen OpenAI CLIP ViT-L/14 text encoder |
| **Released weights** | yes, 1.93 GB and 2.26 GB pickles, no licence stated | yes, `asiimo/noisyclip` |
| **Gradient tested** | yes, on the starting noise of one-step SDXL-Turbo through a finished latent (p. 3 to 5). Never on $x_t$ | no gradient enters a sampler |
| **Early-step evidence** | none; no noisy latent is ever scored | right caption ranks first below chance before latent 10 of 50 (Fig. 4 of that paper) |
| **Word-swap evidence** | none; one printed colour swap under its reward (p. 17) | none; a named failure swaps attributes between objects |

### 7. The three ways to use it here

**(a) Replace NoisyCLIP in the right-minus-wrong push on PoE or CO3: the evidence does not support this yet.**

The push would be, with $P$ the 2 by 2 average pool:

$$
\hat\epsilon_t = \epsilon_{\text{base}} - \lambda\sigma_t\nabla_{x_t}\big[S(P(x_t/0.13025),\text{right}) - S(P(x_t/0.13025),\text{wrong})\big]
$$

*Needs.* The `latent_clip` package, one checkpoint and the pooled input path. The gradient flows through this path with no clamp or `.byte()` step (`transformer.py:883-888`) (inferred).

*What could block it.* Four incompatibilities. It was never trained or tested on noise, and steps 0 to 15 carry 0.076 to 0.308 of the clean signal. It needs a resize that the authors flag as difficult (p. 9). The pooled gradient is constant across each 2 by 2 block of $x_t$, so it may add a blocky pattern. Its published gradient left the spatial benchmark flat (p. 6) (inferred).

*Rough cost.* About a day to wire and test, plus one forward and backward pass per sentence per pushed step on a ViT-B. It should wait for the result of (b) (inferred).

**(b) A read-only comparison beside NoisyCLIP in step W2: the evidence supports this as the next test.**

*Needs.* A third model in the read, using the input `F.avg_pool2d(z_t / 0.13025, 2)` on the same pictures, noise draws, sentences and thresholds as [the W2 design](../02-can-it-read/design.md). No gradient is required.

*What could block it.* The comparison with NoisyCLIP is mixed. Weights, architecture, resolution, input path, text tower and training data all differ, so a gap cannot be credited to noise training. W2's varied axis is the image encoder weights with an identical input path, so Latent-CLIP should be reported as a separate read outside W2's fixed verdict table. It also needs its own input check, because the authors have no pooled path to compare against beyond the 10-image notebook test (inferred).

*Rough cost.* Well under 1 extra GPU-hour for about 40,000 image encoder passes, since its image tower has 85.8 million parameters against 304 million for ViT-L/14 (Table 6, p. 12), plus a 2.26 GB download to `/datasets` (inferred).

**(c) Train or fine-tune a noise-aware Latent-CLIP for SDXL: a full retrain is out of reach, and a fine-tune is possible but premature.**

*Needs.* Noised latents across the steps this project pushes on, including 0 to 15, with captions. A fine-tune could start from the released B/4-plus, keep the 64 by 64 pooled input and freeze the text tower, as NoisyCLIP did. A timestep input would be an architecture change (inferred).

*What could block it.* The original training took 13,504 A100 GPU-hours for B/8 and 25,062 for B/4-plus (128 GPUs for 105.5 hours and 256 for 97.9 hours, computed from p. 3), which this project's limit of one `biggpu` and six `bigbatch` jobs cannot reach. The original latents sit in an S3 bucket named in the job scripts, and no public download is given (`scripts/latentclip_preprocessed_vitb8.sbatch`). The authors report that fine-tuning pixel CLIP onto the VAE latent space did worse than training from scratch (p. 8). That result concerns a different starting point, so it is a warning and does not settle a clean-to-noisy fine-tune (inferred).

*Rough cost, assumed and unmeasured.* At NoisyCLIP's scale of 50,000 latents and 10 epochs, training is about 1 to 3 GPU-hours. Building the data with `add_noise` on encoded images is about 1 to 3 GPU-hours. Sampling the real 50-step path instead, at an assumed 5 seconds per image, is about 69 GPU-hours. Add 2 to 4 days of engineering. Decide this only after (b) shows whether the stock model reads anything at steps 0 to 15 (inferred).

## Closer work it cites

- **ReNO: Enhancing one-step text-to-image models through reward-based noise optimization**, arXiv 2406.04312 (p. 9, reference 14). It optimises starting noise with text-image reward gradients, and it is the method this paper plugs Latent-CLIP into (p. 2, p. 3).
- **CLIP guided diffusion**, Crowson, GitHub, no arXiv id printed (p. 9, reference 8). The paper cites it as the early community method that optimised CLIP similarity to guide diffusion models (p. 8).
- **GLIDE**, no arXiv id printed (p. 10, reference 26). This paper cites it only for image quality (p. 8). GLIDE also trains a CLIP on noised images for guidance, a fact from outside this PDF, which makes it the direct precedent for option (c) (inferred).
