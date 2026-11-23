# The six illustrations

These pictures tell the idea before the sizes and the maths. Every tile in them is an image model's teaching drawing, so none is an SDXL sample and none shows a result. Their prompts and faithfulness notes are in [the illustrated map](diagram-prompts.md).

**1. A cheap colour view of the latent**

<a href="ni1-a-cheap-colour-view-of-the-latent.png"><img src="ni1-a-cheap-colour-view-of-the-latent.png" width="640" alt="Four grey latent channels pass a fixed linear map into a rough colour view and then the image encoder, with the VAE decoder faded and skipped."></a>

A half-drawn latent is not a picture, so a fixed sum of its four channels makes a rough colour view with no decoder.

**2. One picture against two sentences**

<a href="ni2-one-picture-against-two-sentences.png"><img src="ni2-one-picture-against-two-sentences.png" width="640" alt="One noisy picture meets a right and a wrong sentence at two product nodes, giving two score chips and their difference, on toy numbers."></a>

The checker asks the same noisy picture about two sentences with the same words in a different order, and keeps the difference. The numbers are the miniature's.

**3. Training on half-drawn pictures**

<a href="ni3-training-on-half-drawn-pictures.png"><img src="ni3-training-on-half-drawn-pictures.png" width="640" alt="Noisy pictures and captions enter a tuned image encoder and a frozen text encoder, then a similarity matrix with a pale blue diagonal and an attract and repel panel."></a>

Only the image side learns. Each half-drawn picture is pulled toward its own caption and away from the others.

**4. The push bends the noise prediction**

<a href="ni4-the-push-bends-the-noise-prediction.png"><img src="ni4-the-push-bends-the-noise-prediction.png" width="640" alt="A vector picture: the base noise prediction, an orange push from its tip, and their sum, with the gradient taken at the latent tile."></a>

The gradient is taken at the latent, and what changes is the noise prediction the sampler is about to use.

**5. Two places the signal is cut**

<a href="ni5-two-places-the-signal-is-cut.png"><img src="ni5-two-places-the-signal-is-cut.png" width="640" alt="A pipeline from the latent to the image encoder, with the orange return lane cut at the byte conversion and ending at a clamped colour."></a>

The released map blocks the gradient at the conversion to bytes and at any clamped colour.

**6. Trained later than this loop would push**

<a href="ni6-trained-later-than-we-push.png"><img src="ni6-trained-later-than-we-push.png" width="640" alt="A row of an armchair emerging from noise, with this loop's steps 0 to 15 bracketed in orange and NoisyCLIP's training steps 20 to 29 bracketed in green."></a>

NoisyCLIP practised on the middle of the sampling run, and this loop would use it earlier, on noisier latents.
