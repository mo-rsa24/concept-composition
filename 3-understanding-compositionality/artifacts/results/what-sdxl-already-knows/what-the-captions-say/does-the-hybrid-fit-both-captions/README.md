# Does the hybrid fit both captions?

This folder tests one prediction of the repo's explanation for why the product of a cat expert and a dog expert draws one fused animal. The explanation says multiplying "is this a cat?" by "is this a dog?" scores highest on one animal that is ambiguously both. If that is right, a fused render should match "a photo of a cat" and "a photo of a dog" better than a real photograph of a cat beside a dog does. CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) scored every image against three phrases. Nothing was trained and the compose scorer labelled nothing.

## The answer

The fused renders do beat the photographs on the weaker of their two single-animal similarities, by 0.014. The renders of one sentence naming both animals, which hold two separate animals, beat the photographs by 0.017. The gap separates generated images from photographs, so this test does not support the explanation and does not refute it.

The phrase "a photo of a cat and a dog" does separate the fused renders from both other sets, by 0.043 below the photographs and 0.069 below the two-animal renders. Scoring each animal on its own cannot tell a fused animal from two animals; scoring the pair phrase can.

## How to read the numbers

Cosine similarity between a CLIP image embedding and a CLIP text embedding runs from -1 to 1. On these 137 images it only spans 0.165 to 0.331, so a difference of 0.01 is about a twentieth of the whole observed range and about half the photographs' standard deviation (0.021 on the weaker single-animal score). A difference of 0.014 is two thirds of a photograph standard deviation, and the fused and photograph dots overlap almost completely. A difference of 0.043 on the pair phrase is 1.8 photograph standard deviations, and the fused and two-animal dots barely overlap.

| Set | Images | Weaker single-animal similarity, mean ± sd | "a photo of a cat and a dog", mean ± sd |
|---|---|---|---|
| Real photographs (COCO, passed the by-eye audit) | 91 | 0.226 ± 0.021 | 0.285 ± 0.024 |
| Plain product, "a cat" × "a dog" | 7 | 0.240 ± 0.017 | 0.241 ± 0.018 |
| One sentence, "a cat and a dog" | 32 | 0.243 ± 0.016 | 0.310 ± 0.015 |
| Plain product, "cats" × "dogs" | 7 | 0.226 ± 0.020 | 0.233 ± 0.015 |

On the 7 seeds the fused and one-sentence sets share, the fused minus one-sentence difference on the weaker single-animal score is -0.006 ± 0.021, with 3 of 7 seeds positive.

## Items

**`three-sets-side-by-side.png`**

<a href="three-sets-side-by-side.png"><img src="three-sets-side-by-side.png" width="200" alt="three rows of four images: fused renders, one-sentence renders on the same seeds, real photographs, each with its pair-phrase score"></a>

Three rows of four images: fused renders on seeds 2, 3, 6 and 7, one-sentence renders on the same seeds, and four photographs spread from the lowest to the highest pair-phrase score. Each tile carries its similarity to "a photo of a cat and a dog". Seed 4 is left out because it renders a wallpaper of flat cat and dog silhouettes and holds no animal.

**`cat-dog-caption-fit.png`**

<a href="cat-dog-caption-fit.png"><img src="cat-dog-caption-fit.png" width="200" alt="two strip panels sharing a cosine-similarity y axis, one dot per image, black bars at set means"></a>

Two panels on one shared y axis (cosine similarity), one dot per image in four columns, one per set, with a black bar at each set's mean. The left panel is the weaker single-animal similarity, the right panel the pair phrase. Four example images sit underneath, each near its set's mean, carrying both of its numbers.

**`cat-dog-similarity-plane.png`**

<a href="cat-dog-similarity-plane.png"><img src="cat-dog-similarity-plane.png" width="200" alt="scatter of similarity to a cat against similarity to a dog, three colours, six thumbnails"></a>

The figure the request specified: x is similarity to "a photo of a cat", y is similarity to "a photo of a dog", one point per image, three sets in three colours, six thumbnails pinned beside their points.

**`similarities.json`**

One record per image with `set`, `key`, `path`, `seed`, `sim_cat`, `sim_dog`, `sim_both` and `sim_weaker` (the smaller of `sim_cat` and `sim_dog`), plus the model name and the three phrases.

**`summary.json`**

Per-set count, mean, standard deviation, minimum and maximum of each similarity, the paired fused-minus-one-sentence difference per seed, and the two gaps against the photographs.

**`sets.json`, `joint-tiles/`, `photo-thumbs/`**

The list of every image and its source path; the 32 one-sentence renders cut from the contact sheet `supervisor-meeting/1-inference-only/0-reference-where-the-joint-prompt-succeeds.png`; and local copies of the eight photographs the figures show.

**How it was made**

The photographs are the `both` list in `/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions_pool/manifest.json` on the cluster, which already excludes the 40 rejected by eye. The fused renders are `artifacts/results/does-poe-of-cats-and-dogs-match-the-joint/tiles/cat__x__dog/seed*/poe.png`, and the plural ones sit beside them under `cats__x__dogs/`. The scripts are in `scripts/hybrid_caption_fit/`: `build_sets.py`, `crop_joint_sheet.py`, `push_renders.py`, `embed_clip.py` (run on the cluster GPU, about 90 seconds), `pull_results.py`, `summarise.py`, `plot_plane.py` and `plot_columns.py`. The side-by-side grid was drawn with PIL from the same `similarities.json`.

<details>
<summary>Limits</summary>

- Render against photograph changes at the same time as fused against two animals, which is why the single-animal gap cannot be read as a fusing effect. The one-sentence renders are the control that exposes it.
- 7 fused images. One (seed 4) holds no animal, and one (seed 1) has a fused animal in front of a second striped body, judged from full-resolution files.
- The sets differ in resolution before CLIP resizes them to 224 px: fused renders are 1024 px, one-sentence renders are 320 px crops of a contact sheet, photographs are about 640 px.
- A CLIP similarity is one number per phrase. It cannot say where in the image a match came from, so a high score does not show two animals are present, and it is not a species call.
</details>

<!-- cluster:start -->

## On the cluster

The other 8 files of this result (1.1 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `photo-thumbs` | 8 | 1.1 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/photo-thumbs/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions/<run>/ <run>/
```

<!-- cluster:end -->
