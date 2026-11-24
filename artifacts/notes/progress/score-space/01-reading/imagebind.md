# ImageBind: One Embedding Space To Bind Them All

arXiv 2305.05665 (v2), read from the full PDF (15 pages, appendix included). No venue is printed on the PDF.

## What it does

ImageBind trains encoders for audio, depth, thermal images and IMU (motion sensor) readings so that each one matches a frozen OpenCLIP image encoder, using only data that comes naturally paired with images or video (p. 2, p. 4). Since OpenCLIP had already aligned text to images, the new modalities end up aligned to text without ever seeing a (modality, text) pair, which the paper calls emergent alignment (p. 3). The paper then uses the shared space for zero-shot classification, retrieval, embedding arithmetic, and for swapping audio embeddings into a text-prompted detector and a DALLE-2 diffusion decoder (p. 6).

## What it reads

There is no diffusion model inside ImageBind, so nothing reads a noisy latent or a noise prediction. Each encoder reads a clean input of its own modality and outputs a normalized $d$-dimensional embedding through a linear projection head (p. 3, p. 4):

- Images at 224 pixels through a frozen OpenCLIP ViT-H (630M parameters); video as 2-frame clips through the same encoder with an inflated patch layer (p. 3, p. 4, p. 13).
- Text through the frozen OpenCLIP text encoder (302M parameters) (p. 4).
- Audio as 2-second clips at 16 kHz turned into 128-bin log-mel spectrograms, encoded by a ViT with patch 16 and stride 10 (p. 3, p. 13).
- Depth converted to disparity and thermal as one-channel images, each through a ViT (p. 4).
- IMU as a 6×2000 tensor (5 seconds of accelerometer and gyroscope), a 1D convolution with kernel 8, then a transformer (p. 4, p. 13).

For generation, the temperature-scaled audio embedding replaces the CLIP text embedding that a reimplemented DALLE-2 decoder expects (p. 13).

## What it trains

The audio, depth, thermal and IMU encoders are trained; the image and text encoders stay frozen (p. 4). Positive pairs come from co-occurrence: a video clip and its own audio, an image and its own registered depth or thermal map, a video and its own IMU signal (p. 4). Every other example in the minibatch is a negative (p. 3). The loss as printed, used symmetrically as $L_{I,M} + L_{M,I}$ (p. 3):

$$
L_{I,M} = -\log \frac{\exp(q_i^\top k_i/\tau)}{\exp(q_i^\top k_i/\tau) + \sum_{j \neq i}\exp(q_i^\top k_j/\tau)}
$$

- $q_i = f(I_i)$ is the image embedding, $k_i = g(M_i)$ the other modality's embedding, both normalized, and $\tau$ a fixed temperature (p. 3).
- Data: about 2M unlabeled AudioSet videos for (video, audio); about 5K SUN RGB-D (image, depth) pairs replicated 50 times; 12,025 LLVIP (image, thermal) pairs replicated 25 times by Table 9 (50 by the main text); 510,142 Ego4D (video, IMU) clips (p. 4, p. 12, p. 13, p. 14).
- Settings: batch size 2048 for (video, audio) and 512 for the others; 64 epochs (8 for IMU); temperature 0.05 for audio, 0.2 for depth, 0.1 for thermal, 0.2 for IMU; AdamW (p. 14, Table 9).
- Ablations use ViT-B encoders trained for 16 epochs and report ESC fold-1 and SUN-D accuracy (p. 6, p. 14).

## How noise predictions are combined or compared

ImageBind has no noise predictions. Embeddings are compared by cosine distance after temperature scaling (p. 13). Embedding arithmetic, as described in the appendix (p. 13), normalizes each embedding, scales each by 0.5 and sums:

$$
e_{\text{sum}} = 0.5\,\frac{q_{\text{image}}}{\lVert q_{\text{image}} \rVert} + 0.5\,\frac{k_{\text{audio}}}{\lVert k_{\text{audio}} \rVert}
$$

The sum is then used for nearest-neighbour image retrieval by cosine distance (p. 13). For text-to-video retrieval with audio added, the two embeddings of one clip are combined with weight 0.95 on video and 0.05 on audio, the weights that "was found to perform the best" (p. 14). No timesteps, noise samples or shared noise are involved.

## When adding scores or embeddings is valid

The paper claims that adding embeddings from different modalities "naturally composes their semantics" (p. 1) and shows it with retrieved images for image plus audio sums, such as fruit on a table plus chirping birds retrieving birds among fruit (p. 6, Figure 4). That evidence is qualitative: no success rate, no count of tried pairs and no failure case is reported. The one quantitative sum is the weighted video plus audio embedding, which lifts MSR-VTT text-to-video Recall@1 from 36.1% with video alone to 36.8% (p. 5, Tables 2 and 4), with the 0.95 and 0.05 weights tuned on that benchmark (p. 14). The paper gives no condition under which adding is valid and proves nothing about it. Its ethics section warns that a joint embedding "may also create unintentional associations" (p. 15).

## What it says about the noise and the seed

Not addressed in the sense of diffusion noise or seeds. No error bars or repeated training seeds are reported. ESC-50 numbers are a 5-fold average (p. 12), and few-shot runs fix the same k training samples for ImageBind and every baseline (p. 13). The closest finding is about dataset size: for the small (image, depth) set, a smaller batch (512 gives 47.3% on NYU-D against 39.9% at 4k) and a smaller depth encoder work better (p. 8, Tables 6 and 7).

## Which timesteps carry the signal

Not measured. The paper has no diffusion timesteps. The nearest result is that temporally aligned audio and video clips beat unaligned ones, 56.7% against 55.7% ESC fold-1 top-1 accuracy (p. 8, Table 5g).

## Unseen concepts and composition

Partly. What is held out is the pairing: no (audio, text), (depth, text), (thermal, text) or (IMU, text) pair is ever trained on, and the evaluation class labels are never used in training (p. 3, p. 4, p. 12). Concepts are not held out from the text side, because the frozen OpenCLIP encoders were trained on billions of (image, text) pairs (p. 4). Results on these emergent tasks are compared against models trained with paired text: for example ESC-50 top-1 66.9% for ImageBind against 68.6% for AudioCLIP, which trained on AudioSet class names (p. 5, Table 2). Composition is tested only by the embedding sums above. There is no test of two objects, relations, counting or attribute binding.

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Emergent zero-shot top-1 accuracy (%), higher is better | 66.9 | 68.6, AudioCLIP trained with AudioSet class names | ESC-50, 5-fold mean | ImageBind ViT-H | p. 5 Tables 2, 3 |
| Text-to-audio Recall@1 (%), higher is better | 6.0 | 3.0, AVFIC | Clotho | ImageBind ViT-H | p. 5 Table 3 |
| Text-to-audio Recall@10 (%), higher is better | 28.4 | 17.5, AVFIC | Clotho | ImageBind ViT-H | p. 5 Table 3 |
| Text-to-audio Recall@1 (%), higher is better | 9.3 | 8.7, AVFIC | AudioCaps | ImageBind ViT-H | p. 5 Table 3 |
| Emergent zero-shot top-1 accuracy (%), higher is better | 35.1 | 25.4, OpenCLIP ViT-H on depth rendered as grayscale | SUN-D | ImageBind ViT-H | p. 5 Table 2 |
| Emergent zero-shot top-1 accuracy (%), higher is better | 54.0 | 41.9, OpenCLIP ViT-H on grayscale depth | NYU-D | ImageBind ViT-H | p. 5 Table 2 |
| Text-to-video Recall@1 (%), video plus audio sum | 36.8 | 36.1, ImageBind video alone | MSR-VTT 1k-A | ImageBind ViT-H | p. 5 Tables 2, 4 |
| ESC fold-1 top-1 (%) by temperature: learnable, 0.05, 0.07, 0.2, 1.0 | 54.8, 56.7, 52.4, 45.4, 24.3 | | ESC-50 | ViT-B ablation | p. 8 Table 5a |
| Top-1 (%), linear against MLP projection head | SUN-D 26.7 vs 26.5, ESC 56.7 vs 51.0 | | SUN-D, ESC | ViT-B ablation | p. 8 Table 5b |
| SUN-D top-1 (%), spatially aligned against unaligned crops | 26.7 vs 16.0 | | SUN-D | ViT-B ablation | p. 8 Table 5e |

## What it suggests for this loop

- For a score-path encoder (inferred): ImageBind shows that one encoder trained against a frozen hub can inherit the hub's alignment to a third space. The loop could freeze an existing text-and-latent pair (CLIP text, or the W4 text-with-latent encoder once it works) and train only the score-path encoder against the noisy latent of the same canvas, seed and step. Alignment of score paths to captions might then appear without any score-path and caption pairs, which is the cheapest version of the third representation.
- For a score-path encoder (inferred): the small-data settings that worked for about 5K (image, depth) pairs are the ones to copy for a few thousand digit canvases: fixed temperature, a linear projection head, a small encoder, a batch of about 512 and data replication.
- For a score-path encoder (inferred): spatially misaligned crops cost 10.7 points on SUN-D. Any crop or augmentation of a canvas must be applied identically to its score path and its latent.
- For the seed problem (inferred): emergent alignment improved with a stronger hub (7 and 4 points for depth and audio going from ViT-B to ViT-H). A strong hub cannot remove seed content from the score path, so the seed problem has to be solved in the score-path encoder's positives.
- For "sum lands on joint" (inferred): ImageBind's own sum test is 0.5 plus 0.5 of normalized embeddings judged by nearest-neighbour retrieval and shown only as pictures. The loop's version should report a retrieval rate over all digit pairs with a random-pair control, since ImageBind gives no rate to compare against. The 0.95 and 0.05 video and audio weights show that equal weights can be wrong when one input is much stronger.
- For "gap predicts the term" (inferred): ImageBind never subtracts embeddings, so it gives no evidence on embedding differences.
- For the separate-experts paper (inferred): the DALLE-2 swap shows a diffusion decoder conditioned on CLIP text embeddings will accept another encoder's embedding if it was trained into the same space. A score-path embedding aligned to CLIP text could be fed back as a condition in the same way.
- For the CO3 paper (inferred): no direct bearing.

## Closer work it cites

None. ImageBind cites no work that trains on noise predictions, score fields, reconstruction errors or sampling trajectories, or that aligns a diffusion model's internals to a text or image space. The diffusion model it uses, DALLE-2 (arXiv 2204.06125, reference on p. 11), conditions on CLIP embeddings without aligning its internals.

## Where the paper disagrees with itself

- The AudioSet audio-only metric is mAP in Table 1 (p. 4) and the Table 2 caption (p. 5), while Appendix A says "The metric used is top-1 accuracy" for the same evaluation (p. 12).
- The Table 5 caption says a fixed temperature beats a learnable one "for all modalities" (p. 8), while on ESC the fixed values 0.2 and 1.0 score 45.4% and 24.3% against 54.8% for learnable; only 0.05 beats it.
- The main text says SUN RGB-D and LLVIP are both replicated 50 times for training (p. 4), while Table 9 lists a sample replication of 50 for SUN and 25 for LLVIP (p. 14).
- Appendix C says Table 9 lists hyperparameters for "the models reported in Table 4" (p. 14), which is the MSR-VTT retrieval table; the main results are in Table 2.

No text in the paper was addressed to the reader of this note.
