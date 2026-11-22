# Does relabelling both-shapes pictures help the toy product compose?

Yes, once about 9% of each single caption's pictures hold both shapes. Below 2.5%, the product of
"a red circle" × "a blue square" draws a purple hybrid in every training. At 9.1% one training of
three draws a separate red circle and blue square on 30 or more of 300, and from 23% every training
does. This holds with the training set fixed at 6,000 (replacing, the main design) and with it growing
(adding, the first design, kept as the check). A Langevin corrector on the sampler takes every model
tested at 9% or more to 300 of 300; at 0% and at 2.4% it stays at 0.

## toy-caption-replace-sweep.json

The main design's full sweep: each single caption keeps 2,000 pictures, and k of its lone shapes are
swapped for both-shapes pictures, at k = 1, 5, 20, 49, 182, 462, 750 and 1,000 (shares 0.05% to 50%,
matched to the adding design), three seeds each, 24 models. Separate pairs out of 300: 0 at every
training up to 1%; 0, 0, 2 at 2.45%; 13, 5, 72 at 9.1%; 66, 64, 146 at 23%; 239, 129, 75 at 37.5%;
209, 150, 84 at 50%. The single captions drew only their own shape on at least 247 of 300 at every
share, and the joint caption drew the pair on at least 298. Trained on an RTX 3090 (cluster job 90137,
mscluster88, 21 minutes) by `4-inclusive-captions/scripts/toy_caption_replace.sbatch`; the log is
`4-inclusive-captions/data/toy-relabelling-ladder/toy-replace-90137.out` and the samples
`toy-caption-replace-sweep-samples.npz`. Drawn as
[the share-by-share grid](../../scenes/how-the-toy-relabelling-experiment-works/network-views/14-what-the-product-draws-at-each-share-replacing.png)
and beside the adding design in
[the replacing-against-adding chart](../../scenes/how-the-toy-relabelling-experiment-works/network-views/13-replacing-against-adding.png).
`verified` from the file.

This is the toy version of inclusive captioning: a photo of a cat beside a dog also captioned "a
cat" and "a dog". The toy is a caption-conditional latent diffusion model built the way SDXL is
(a VAE, one denoiser, a caption lookup table), trained on 16 × 32 pictures of a red circle, a blue
square, or both.

## the-training-set-with-and-without-relabelling.png

<a href="the-training-set-with-and-without-relabelling.png"><img src="the-training-set-with-and-without-relabelling.png" width="600" alt="three captions by three relabelling levels, eight real training pictures each; both-shapes pictures under the single captions go from 0% to 9% to 50%"></a>

What relabelling changes. Rows are the caption a picture is trained under, columns are 0, 200 and
2,000 of the 2,000 both-shapes pictures also filed under each single caption. The share of each
single caption's pool holding both shapes goes 0%, 9%, 50%; the joint caption's pool stays the same
2,000 pictures. `verified` by looking at the figure.

## what-poe-draws-at-each-relabelled-count.png

<a href="what-poe-draws-at-each-relabelled-count.png"><img src="what-poe-draws-at-each-relabelled-count.png" width="600" alt="nine rows of relabelled counts by three training runs: purple blobs up to 50, separate red circle and blue square from 200"></a>

The main result. Rows are 0, 1, 5, 20, 50, 200, 600, 1,200 and 2,000 relabelled pictures; columns
are three trainings with different seeds; each cell shows four product samples and how many of 300
are a separate red circle and blue square. Up to 50 relabelled, at most 3 of 300 in any of the 15
models. From 200, some trainings reach 30 or more (outlined): 153, 1 and 0 at 200; 252, 74 and 238
at 2,000. In all 27 models the joint caption alone drew the pair on at least 267 of 300. Counts in
`toy-caption-threshold.json` and `toy-caption-threshold-600-1200-2000.json`. `verified` against
both files and by eye.

## the-corrector-with-and-without-relabelling.png

<a href="the-corrector-with-and-without-relabelling.png"><img src="the-corrector-with-and-without-relabelling.png" width="600" alt="rows of PoE samples: 0 and 50 picture models stay purple under the corrector, 200 and 2,000 picture models reach 300 of 300 separate"></a>

Rows are eight retrained models (three at 0 relabelled, one at 50, one at 200, three at 2,000);
columns are plain sampling, 5 and 20 Langevin corrector steps per noise level, from the same
starting noise. With 20 steps the models at 200 and 2,000 relabelled draw the pair on 300 of 300; the 0 and 50
models stay at 0 and are tidied into purple hybrids. The rule (the middle count of the four
relabelled models rises by 30 or more while each single caption keeps 240 of its own colour) was
written before the run and is met: 260.5 plain, 300 with 5 steps. The 0 and 50 rows were added
after seeing that result. Counts in `toy-caption-corrector.json` and
`toy-caption-corrector-no-single-captions.json`. `verified`.

## every-caption-with-and-without-the-corrector.png

<a href="every-caption-with-and-without-the-corrector.png"><img src="every-caption-with-and-without-the-corrector.png" width="600" alt="five rows for five shares, columns for the circle caption, the square caption, PoE, PoE plus corrector and the joint caption: 0 of 300 at 0 relabelled, 300 of 300 in every relabelled row with the corrector"></a>

One model per row at 0%, 10%, 30%, 60% and 100% relabelled (about 0, 200, 600, 1,200, 2,000
pictures), asked five ways. Plain product 0, 110, 103, 51, 205 of 300; with the corrector 0, 300,
300, 300, 300. The single captions stay pure until 2,000, where "a red circle" draws only red on 244
and "a blue square" only blue on 239. Counts in `toy-caption-dial-corrector.json`. `verified`.

## the-caption-model-across-guidance.png

<a href="the-caption-model-across-guidance.png"><img src="the-caption-model-across-guidance.png" width="600" alt="dial 0 to 1 down, guidance 1 to 7 across: dial 0 gives at most 1 of 300 apart at every strength"></a>

The product at guidance strengths 1, 2, 3, 5 and 7 for the same five settings. Stronger guidance
does not rescue the unrelabelled model: at most 1 of 300 apart at any strength. Counts in
`toy-caption-dial-ldm-guidance.json`; the guidance-1 counts and VAE checks in
`toy-caption-dial-ldm.json`. `verified`.

## toy-caption-replace.json

The size check. Adding grows the training set from 6,000 to 10,000 pictures, so this run replaces
instead: k of each single caption's 2,000 lone shapes become both-shapes pictures, every caption keeps
2,000 and the set stays at 6,000. At share 10% (200 replaced) the separate-pair counts are 0, 0 and 3
of 300; at share 50% (1,000 replaced) they are 119, 281 and 182, against 252, 74 and 238 for adding at
50%. The rule written before the run (in `4-inclusive-captions/scripts/toy_caption_replace.py`): if all
three 50% models reach 30 of 300, the share and not the size drives the result. It did. Trained on CPU
with the experiment's own code, seeds 20261022 to 20261024; the log is
`4-inclusive-captions/data/toy-relabelling-ladder/toy-caption-replace.log` and the samples are
`toy-caption-replace-samples.npz`. Drawn beside the adding design in
[the size-check figure](../../scenes/how-the-toy-relabelling-experiment-works/network-views/13b-the-size-check-at-two-shares.png).
`verified` from the file.

## toy-caption-threshold-no-noise.json

The same exact-count training with the pixel noise on every training copy set to zero, three
trainings each at 0, 200 and 2,000 relabelled. The separate-pair counts out of 300 are 0, 0, 0;
56, 40, 0; and 162, 114, 266, against 0, 0, 0; 1, 153, 0; and 252, 74, 238 with the noise. The
rule written before the run (in `toy_caption_no_noise.py`) held: the noise does not change the
result. At 0 relabelled, the no-noise models draw a single shape where the noisy ones draw a purple
blob, and neither draws the pair. `stated` from the walkthrough; the figure that checks it against
this file is network view 12 in
[the ladder](../../scenes/how-the-toy-relabelling-experiment-works/network-views/README.md).

**How it was made.** These are copies, taken on 2026-11-18 at commit `db1325d7`, of files whose
originals live in
`3-understanding-compositionality/artifacts/results/the-exact-toy/how-much-co-occurrence-the-product-needs/`,
where the toy's full argument is told (the four figures above with a number prefix under
`plain-figures/`). The scripts sit in
`3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/` and write to
the originals: `toy_caption_threshold.py`, `toy_caption_dial_ldm.py`, `toy_caption_corrector.py`
and `toy_caption_dial_corrector.py` train and count; `plain_figures.py` draws the first four
figures and `toy_caption_guidance_sheet.py` the fifth. A rerun updates the originals, not these
copies. The sample arrays (`*-samples.npz`, about 6 MB) stay with the originals.

**What it cannot say.** Three trainings per count is few: trainings at one count differ by up to
150 of 300, so the order between 600 and 1,200 is not reliable; the jump from at most 50 to 200 and
above is. Relabelling by adding also grows the training set (6,000 to 10,000 pictures at a fixed
8,000 steps); the replacing run above holds the size fixed and still composes at 50%, but it covers
two shares only. The scorer misses a
reddish circle beside a bluish square. A toy caption's pool is three pictures plus pixel noise, so
this says the share matters, not what share a real "a cat" pool needs.

<!-- cluster:start -->

## On the cluster

The other 25 files of this result (6.0 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 25 | 6.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/4-inclusive-captions/artifacts/results/does-relabelling-both-shapes-pictures-help-the-toy-product-compose/<run>/ <run>/
```

<!-- cluster:end -->
