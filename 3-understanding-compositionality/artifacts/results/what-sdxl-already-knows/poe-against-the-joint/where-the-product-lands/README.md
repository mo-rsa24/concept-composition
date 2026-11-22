# Where the product lands

Illustrations of where plain PoE composition lands relative to the prompts it combines. The
surface is drawn by hand; the four renders pinned to it are real.

## where-the-product-lands-on-the-manifold-seed12-surface (.pdf, .svg, .png)

<a href="where-the-product-lands-on-the-manifold-seed12-surface.png"><img src="where-the-product-lands-on-the-manifold-seed12-surface.png" width="200" alt="Two red peaks labelled a cat and a dog with their renders above them, the PoE render floating over the blue saddle between them with a callout marking where PoE lands, and a front peak carrying the a cat and a dog render"></a>

**What it shows.** Two peaks for "a cat" and "a dog", a third peak in front for "a cat and a
dog", and an X on the low saddle between the first two, where the product of the two
single-animal densities peaks. The product's render (a cat with a dog's coat and ears fused
in) floats above that saddle; the joint prompt's render (a cat and a dog side by side) sits
on the front peak.

**What is measured and what is not.** Nothing about the surface is measured: the plane is not
an embedding, the peaks are hand-placed Gaussians at equal height, and the only geometric fact
honoured is that the product of two Gaussians peaks at their midpoint. The four renders come
from cell `a_cat__x__a_dog`, seed 12, one starting latent, 50 steps, guidance 7.5, no adapter,
read from `/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/pairs/a_cat__x__a_dog/<condition>/seed_12/step_050.png`.

**How it was made.** A vector rebuild of the image-model illustration
`paper/overleaf-iclr/figures/where-the-product-lands-on-the-manifold-seed12-surface.png`,
which has no source. The four frames were copied off mscluster84 and the figure drawn with

    python3 scripts/where_poe_lands_on_the_manifold.py --layout surface --seeds 12 \
        --cell-root <local copy of the cell folder> --out-dir artifacts/results/where-the-product-lands

In the PDF and SVG the surface, words and marker are vector; the four renders are embedded
bitmaps at about 490 px each.

<!-- cluster:start -->

## On the cluster

The other 1 files of this result (1.2 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/where-the-product-lands/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 1 | 1.2 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/where-the-product-lands/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/where-the-product-lands/<run>/ <run>/
```

<!-- cluster:end -->
