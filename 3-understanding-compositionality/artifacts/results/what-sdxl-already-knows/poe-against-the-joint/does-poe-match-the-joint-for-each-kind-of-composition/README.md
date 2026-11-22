# Joint against PoE, by kind of composition

Pictures for each kind of composition named in [the composition terms](../../ideas/poe-overlap-dial/composition-terms.md). The [build sheet](MANIFEST.md) says, row by row, which paper, model and checkpoint made each cell.

## joint-against-poe-by-kind.png

<a href="joint-against-poe-by-kind.png"><img src="joint-against-poe-by-kind.png" width="200" alt="Eleven rows, one per kind of composition, joint sample on the left and product of experts on the right"></a>

**What it shows.** Eleven rows, one per kind. The left cell is the joint sample, or the nearest thing the source has, and the right cell is the product of experts. A grey box marks a side that no released model or paper provides, with the reason under it. Each row uses its source paper's own model: Liu et al. 2022's CLEVR models, Stable Diffusion v1-4 and v2, PixArt-Σ, GLIDE, and Bradley et al.'s EDM2 models.

**What to notice.** Spatial co-occurrence composes: two objects land at the two positions in 4 of 4 seeds. Relation does not: the PoE of "large red cube" and "large blue sphere" gives one object, and in 3 of 4 seeds its attributes are crossed (a blue cube once, a red sphere twice). Style and content give almost the same image on both sides. Physical interaction does not: the PoE paints rain streaks over a dry dog. Negation stacks several red cubes rather than giving one clean one.

**How it was made.** Six rows are cropped at 300 dpi from the papers' own PDFs on the paper shelf by `scripts/joint_vs_poe_grid/crop_figures.py`, each cited under its cell. Five rows were generated on mscluster109 (RTX A6000) on 2026-10-14. The CLEVR rows use Liu et al.'s released `clevr_pos` and `clevr_rel` checkpoints (899,228,792 and 900,135,692 bytes, from the Dropbox links in their `download.py`) through `scripts/joint_vs_poe_grid/gen_clevr.py`. That script copies the model construction, `model_fn` and 100-step ancestral sampler from Liu's `image_sample_compose_clevr_{pos,rel}.py`, and drops their final save line, which uses an undefined `guidance_scale`. The empty second object is coded `[3, 2, 8, 2, 2]` with relation 6, read from the 9,928 single-object rows of Liu's relational training labels. The Stable Diffusion rows use Liu's `ComposableStableDiffusionPipeline` unchanged except one import path (`diffusers.pipeline_utils` became `diffusers.pipelines.pipeline_utils` for diffusers 0.29.2), with DDIM, 50 steps and weight 7.5 per prompt, through `scripts/joint_vs_poe_grid/gen_sd.py`. Every generated cell is seed 0, and both sides of a pair share the seed. `scripts/joint_vs_poe_grid/build_grid.py` assembles the figure.

## real-radiographs-by-kind.png

<a href="real-radiographs-by-kind.png"><img src="real-radiographs-by-kind.png" width="140" alt="Five real chest radiographs: comorbidity, compound disease, physical interaction, consequence, diagnosis from findings"></a>

**What it shows.** One real radiograph each for comorbidity, compound disease (silicotuberculosis), physical interaction (a pneumothorax in a COPD lung), consequence (a chest drain) and a diagnosis built from findings (heart failure). These have no product-of-experts side.

**How it was made.** Each image comes from an open-access source with its licence quoted and its citation under the cell. Sources, licences, captions and rejected candidates are in [the medical examples card](medical-examples/README.md). The heart-failure film rests on NIH's text-mined labels, and calling it heart failure is our reading.

## pairs-by-seed.png

<a href="pairs-by-seed.png"><img src="pairs-by-seed.png" width="200" alt="11 rows of generated pairs by 4 seed columns, each cell joint beside PoE"></a>

**What it shows.** The 11 generated pairs as rows, grouped by interaction (structural, relational, causal, shared, temporal), and seeds 0 to 3 as columns. Each cell is the joint-prompt sample on the left and the product of experts on the right, at the same seed. The row title names the interaction, the joint prompt, and the PoE prompts.

**How it was made.** `scripts/joint_vs_poe_grid/build_seed_grid.py` from the images in `generated/`. Cropped and radiograph examples have no seeds, so they are not in this grid; they are in `five-interactions.md`.

## five-interactions.md

**What it shows.** The five kinds of interaction (structural, relational, causal, shared, temporal) with what each is, its key question, a success and a failure, then a table of 24 example pairs. Columns are the kinds and rows are the pairs, and each cell is a side-by-side figure of the joint sample and the product of experts, or a cited radiograph for the medical pairs.

**How it was made.** 11 generated pairs (Stable Diffusion v1-4 through Liu's pipeline, and Liu's CLEVR relation model), seed 0 per cell, with `scripts/joint_vs_poe_grid/build_pairs.py` making the cell figures in `pairs/` and `build_tiles.py` making the four-seed tiles in `tiles/` that each caption's counts come from. Five cropped cells came from the user's screenshots of Bradley et al. 2025a and T2I-CompBench++, copied into `crops/`.

## interaction-grids/

Seven grids, one per kind of interaction: 1a structural objects, 1b structural binding, 2a relational spatial, 2b relational action, 3 causal, 4 associative (out of context, file `associative-out-of-context.png`), 5 temporal. Each has six pairs as rows and seeds 0 to 3 as columns. Each cell is the joint prompt "A and B" (guidance 4.5) beside the PoE A + B (2.25 each), on SD3.5 medium, same seed, steps and sampler. Joint and PoE always see the same words.

**How the pairs were chosen.** Three Codex rounds, all in `pair-selection/`. In round 1, Codex proposed 8 candidates per grid from `brief.md`, which holds the user's definitions and the fairness rules. In round 2, it reviewed them adversarially and rewrote causal pairs to name the cause and the object, never the effect. In round 3, it looked at the rendered grids from round 2, kept rows where the joint shows the interaction and the PoE fails, and replaced the rest. `round3.json` is the selection drawn. Generated by `scripts/joint_vs_poe_grid/gen_sd35_sets.py` and assembled by `build_interaction_grids.py`.

## generated/

The generated images, four seeds per cell, plus one four-seed contact sheet each for the first CLEVR and Stable Diffusion batches. Two cells are extra: `negation-a-only` and `spatial-a-only` show concept A alone, for comparison.

## crops/

The 12 cropped figure panels, named by kind, paper and figure number.

## scouting/

The two reports that chose each row's model and code: [the CLEVR rows](scouting/clevr-rows.md) and [the text-to-image rows](scouting/text-to-image-rows.md).

<!-- cluster:start -->

## On the cluster

The other 4 files of this result (0.2 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `pair-selection` | 4 | 0.2 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/pair-selection/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/<run>/ <run>/
```

<!-- cluster:end -->
