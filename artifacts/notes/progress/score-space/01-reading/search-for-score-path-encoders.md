# The search for encoders trained on noise predictions or sampling paths

Step X1 asks whether any paper trains an encoder on a diffusion model's noise predictions, score fields or sampling trajectories, contrastively or otherwise. This page records the search, run on 2026-11-24, and what it found.

## What was searched

**arXiv API, abstract field, 12 queries, up to 25 results each, sorted by relevance.** The script and every hit with its abstract are kept in [the saved search](tools/search-2026-11-24/), as `q.py` and `arxiv_search.json`.

| Query (arXiv abstract field) | Matching papers |
|---|---|
| "noise prediction" AND contrastive AND encoder AND diffusion | 4 |
| "denoising trajectory" AND contrastive | 13 |
| "diffusion trajectory" AND embedding AND encoder | 5 |
| "sampling trajectory" AND representation AND diffusion AND encoder | 2 |
| "score function" AND "representation learning" AND encoder AND diffusion | 0 |
| "noise predictions" AND features AND classifier AND diffusion | 2 |
| "epsilon prediction" AND representation AND diffusion | 0 |
| "score field" AND encoder AND contrastive | 0 |
| "denoising trajectories" AND features | 49 |
| trajectory AND "text-to-image" AND contrastive AND noise AND embedding | 0 |
| "classifier-free guidance" AND difference AND representation AND encoder | 5 |
| "diffusion classifier" AND noise AND learned AND embedding | 1 |

The 81 hits are 55 distinct papers, and none of them trains an encoder whose input is a noise prediction or a sequence of them. They are segmentation, editing, caching, recommendation, policy and video papers that use the words in other senses. The nearest, ConDiff (arXiv 2405.00858), classifies wound photographs by distance after guided synthesis, and DeFlowCritic (arXiv 2611.28265) scores latents. Both were judged from the abstracts saved in `arxiv_search.json`.

**Web search, five queries, the same day:**

1. contrastive encoder trained on diffusion model noise predictions across timesteps as representation
2. image-text matching using diffusion noise prediction error fine-tuned hard negatives
3. classifier trained on diffusion denoising trajectory features or reconstruction error across noise levels
4. detect generated images classifier trained on diffusion reconstruction error, denoising trajectory
5. noise prediction or score sequence along sampling path, encoded embedding, text alignment, 2025 to 2026

## What it found

Five papers came closest, and all five were read in full for this step.

| Paper | arXiv | How close it comes |
|---|---|---|
| Are Diffusion Models Vision-And-Language Reasoners? (DiffusionITM) | 2305.16397 | scores a caption by how much it lowers the noise-prediction error, and fine-tunes the diffusion model with hard-negative captions |
| Guiding Diffusion-based Reconstruction with Contrastive Signals for Balanced Visual Representation | 2603.04803 | a contrastive loss on a frozen diffusion model's noise predictions, used to train a CLIP image encoder |
| NoiseCLR | 2312.05390 | a contrastive loss on noise predictions across steps, used to find editing directions |
| Diffusion Sampling Path Tells More | 2505.23343 | sums the conditional-minus-unconditional score along the sampling path and uses it as a quality signal |
| DIRE for Diffusion-Generated Image Detection | 2303.09295 | trains a classifier on the error left after a diffusion model reconstructs an image |

Other related lines were seen and not read, because their input is internal features or clean images rather than noise predictions: diffusion features aggregated across timesteps, diffusion autoencoders whose encoder reads clean images, and noise-level classifiers (arXiv 2407.08946, 2402.10095).

## The finding

The search found no paper that trains an encoder whose input is a stack of noise predictions $d_t$ along a sampling path, and none that places such a path in a shared space with text or with noisy images. The five closest papers each use one part of that: a score difference read as a text match (DiffusionITM), a contrastive loss on noise predictions (2603.04803, NoiseCLR), the conditional-minus-unconditional score accumulated over the path (2505.23343), and a classifier trained on a diffusion-derived residual (DIRE). Their notes are in the [reading table](reading-table.md).

The absence holds only for the 12 arXiv queries and 5 web queries above. A paper using other words for the same thing could have been missed.
