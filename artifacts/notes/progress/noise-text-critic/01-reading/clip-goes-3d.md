# CLIP goes 3D: Leveraging Prompt Tuning for Language Grounded 3D Recognition

arXiv 2303.11313, read from the full PDF (14 pages, appendix included).

## What it does

The paper adds a third encoder for 3D point clouds beside CLIP's image and text encoders (p. 1, 3). It trains the new encoder with contrastive losses so that its features match CLIP's features for the same object category (p. 4). A few learned prompt tokens adapt the frozen image encoder to rendered images (p. 4 to 5).

**Plain answer on the two questions for this loop.** The model never reads a noisy input, because its inputs are clean point clouds, clean rendered images and captions (p. 3 to 4, 6). No gradient of its score is used to change anything. The paper contains no diffusion model or sampler, so it has no evidence on layout, texture or any generated picture.

## What the model or score reads

- **3D encoder:** a clean point cloud, encoded by PointTransformer or PointMLP, then a projection layer to the shared feature size (p. 3, 7).
- **Image encoder:** a rendered view of the same object through CLIP's visual encoder, using SLIP pre-trained weights with a ViT-Base backbone (p. 3, 7). Five learned prompt tokens are added at the input of every transformer layer (p. 5, 7).
- **Text encoder:** CLIP's text encoder, reading a caption built from a template such as "A photo of a {OBJECT}" (p. 4, 6).
- **Score:** the inner product of normalised features (p. 4). At test time the class is the prompt with the highest similarity to the 3D feature (Eq. 10, p. 5).

## What it trained on

- **Data:** ShapeNet, 52,460 textured CAD models in 55 categories, each giving a sampled point cloud, a Blender-rendered view and a template caption (p. 6).
- **Positives:** in the 3D-to-image loss, features of the **same class** are positive pairs and the rest are negatives (p. 4).
- **Loss on the 3D encoder:** InfoNCE, $L_{3D} = L_{(3D,2D)} + L_{(3D,text)}$ (Eq. 4 to 7, p. 4).
- **Loss on the prompt tokens:** a contrastive loss between image and text features (Eq. 9, p. 5).
- **What moves:** only the 3D encoder and the visual prompt tokens train, and everything else is frozen (Fig. 2 caption, p. 4). The paper reports that unfreezing CLIP's visual encoder caused a substantial performance drop that it attributes to catastrophic forgetting, without giving a number (p. 4).
- **Length:** 100 epochs at batch size 32 (p. 7).
- **Noise levels:** none.

## How its score or gradient enters

No gradient enters a sampler, because the paper has no sampler. The score is only used to classify or retrieve (p. 5 to 6):

$$
y_{pred} = \max(\text{softmax}(\langle f^{3D}, F_{text} \rangle))
$$

Here $f^{3D}$ is the point-cloud feature and $F_{text}$ is the set of text features for the candidate class prompts (Eq. 10, p. 5).

## When adding scores is valid

This paper does not address it. The only addition is of two training losses (Eq. 7, p. 4).

## Which of the five properties it bears on

**Counting.** Not tested.

**Attribute binding.** Not tested. Captions name a single object category (p. 6).

**Spatial relation.** Not tested.

**Position and size.** Not tested as a property of an image or sentence. Scene querying picks which k-means cluster matches a word, and this is shown only in example figures (p. 6, 12 to 13).

**Prompt match.** Not tested on sentences. It measures single-category classification and retrieval (p. 6 to 7).

## Unseen concepts

The paper partly tests this. It trains on ShapeNet and reports zero-shot accuracy on ModelNet10, ModelNet40 and ScanObjectNN (p. 6 to 7), and it calls the ModelNet10 classes "unseen categories" (p. 8). It describes no step that removes categories shared with ShapeNet. These datasets share everyday classes such as chair, table, sofa and bed, so the test is a new-dataset test more than a clean held-out-concept test (inferred). Accuracy falls from 67.3% on ModelNet10 to 25.6% on real scans in ScanObjectNN for the PointTransformer encoder (Table 1, p. 7).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Zero-shot accuracy (%, higher better) | 67.3 | 30.2 PointCLIP | ModelNet10 | PointTransformer + CG3D | p. 7, Table 1 |
| Zero-shot accuracy (%, higher better) | 50.6 | 20.2 PointCLIP | ModelNet40 | PointTransformer + CG3D | p. 7, Table 1 |
| Zero-shot accuracy (%, higher better) | 25.6 | 15.4 PointCLIP | ScanObjectNN | PointTransformer + CG3D | p. 7, Table 1 |
| Fine-tuned accuracy (%, higher better) | 85.78 ± 0.75 | 85.18 Point-MAE | ScanObjectNN PB-T50-RS | PointMLP + CG3D | p. 7, Table 2 |
| Fine-tuned accuracy (%, higher better) | 93.35 ± 0.18 | 93.8 Point-MAE | ModelNet40 | PointMLP + CG3D | p. 7, Table 2 |
| Zero-shot accuracy, 3D-to-image loss only (%, higher better) | 19.1 | n/a | ScanObjectNN | PointTransformer | p. 8, Table 3 |
| Zero-shot accuracy, 3D-to-text loss only (%, higher better) | 19.7 | n/a | ScanObjectNN | PointTransformer | p. 8, Table 3 |
| Zero-shot accuracy, both losses (%, higher better) | 23.9 | n/a | ScanObjectNN | PointTransformer | p. 8, Table 3 |
| Zero-shot accuracy, both losses plus visual prompts (%, higher better) | 25.6 | n/a | ScanObjectNN | PointTransformer | p. 8, Table 3 |

PointCLIP's ModelNet40 zero-shot accuracy is 20.2% in Table 1 (p. 7) and 10.1% in Table 4 (p. 14). The text around Table 4 calls that comparison single-view (p. 14).

## What to copy, what to avoid

- Copy: the training shape for a new encoder beside a frozen CLIP. A noise-aware image encoder could be pulled toward both CLIP's clean-image feature and its text feature, with CLIP left frozen (inferred).
- Copy: use both loss terms. Together they beat either one alone on the paper's ablation, 23.9% against 19.1% and 19.7% (Table 3, p. 8), though on one dataset and with no error bars (inferred).
- Avoid: class-level positives and single-object template captions for this loop's sentence pairs. An encoder trained this way learns which category is present and has no training signal for "red cat and blue dog" against its swap (inferred).
- Avoid: taking a high zero-shot number as evidence that an aligned encoder can steer. The score here is only ever read, and our CLIP gradient already raised its score without moving the animals (inferred).

## Closer work it cites

None.
