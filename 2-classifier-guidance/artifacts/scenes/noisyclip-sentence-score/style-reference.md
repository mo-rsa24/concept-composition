# Style reference: CLIP contrastive

Every figure on this ladder is an original drawing in the layout below, and every Codex brief attaches this file. The three pictures in `reference/` are third-party figures kept for reference only. No picture, wording or logo is copied from them, none is placed on a page, and that folder is not committed.

## Colour key

| Meaning | Colour |
|---|---|
| text | `#b9a7e8` |
| image | `#a8dcb5` |
| matching pairs | `#bfe3f5` |
| the push (nothing else is orange) | `#c86a0a` |
| frozen or fixed | `#6b7a88` |

## The shared layout

**Page.** White background, thin dark arrows, sans-serif labels, generous space and no drop shadows.

**Two lanes.** Text runs left to right on top and image below. Inputs sit at the far left as a small stack of caption cards and a small stack of image thumbnails.

**Encoders.** Solid trapezoids that narrow toward the output, or rounded blocks, with the name inside: "Text Encoder" in lilac and "Image Encoder" in green.

**Embeddings.** Thin strips of small cells that shade from light to dark. Text embeddings run across the top of the matrix as T1, T2, T3, ..., TN, and image embeddings run down its left side as I1, I2, I3, ..., IN. Thin elbow arrows drop from the text encoder to each T column.

**Similarity matrix.** Each cell prints its product, such as I1·T2, on white or very pale grey. The diagonal is pale blue. A small boxed label beside it reads "Contrastive loss: maximise the diagonal, minimise the off-diagonal".

**Variants.** A "joint embedding space" box with paired dots suits the figure that explains the shared space. Horizontal lanes separated by dotted lines, each named at its right end, suit the figure showing training, reading a noisy latent and pushing the sampler. One encoder opened into a small tower of blocks at the far right (embedding, multi-head attention, MLP, each with add and norm) suits the figure that opens the ViT.

## Devices from the three reference pictures

**From `multi-view-contrastive-loss.png`, the main layout of the read and push figure.** Rounded pastel encoder blocks, shaded embedding strips, small ⊗ nodes where two embeddings meet, a labelled chip above each ⊗ holding that pair's score, and one wide grey bar across the top that every score feeds. Braces under the inputs name each group. The image embedding meets the right sentence at one ⊗ and the wrong sentence at another, giving an `S right` chip and an `S wrong` chip. The top bar reads "S right minus S wrong", and in the training figure it reads "contrastive loss".

**From `clip-frame-beside-own-model.png`.** A bold frame with a lighter inner frame around the part that is stock CLIP, pill-shaped modules, feature stacks written f1, f2, ..., fn, a template box for the sentence pairs, and NoisyCLIP's own parts outside the frame.

**From `dual-contrastive-attract-repel.png`.** Inputs in dashed columns, right-hand panels in rounded grey frames with titles, and one attract-and-repel panel for the training figure. A "weights sharing" arrow is drawn only if the released code confirms that the same encoder reads every noise level.

These devices do not change the colour key. The image encoder stays green and only the push is orange.

## NoisyCLIP's own additions

**Frozen parts** keep their shape and colour family, fade, and carry a small "frozen" tag. The tuned image encoder stays fully saturated.

**The image projection head needs a check before it is drawn as frozen.** The style request lists both projection heads as frozen. The released `train_model_classic.py` (lines 47 to 51) freezes only parameter names containing "text" under `--only_vision`, which leaves `visual_projection` trainable. Which flags produced the released checkpoint is not stated in the code.

**The latent-to-RGB map** is a small grey box before the image encoder, labelled "no VAE decode".

**The right and wrong sentences** are two T columns side by side, with the score difference drawn across them.

**The push** is orange, in its own lane below the forward path, and runs back to `x_t`.
