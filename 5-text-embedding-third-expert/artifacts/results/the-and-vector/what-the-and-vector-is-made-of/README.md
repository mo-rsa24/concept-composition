# What the "and" vector is made of

Can the text-space difference between a joint prompt and its two halves stand in as a third
expert in SDXL's product of experts? The walk that asks it is
[the "and" vector idea map](../../ideas/the-and-vector-as-a-third-expert/IDEA_MAP.md).

## a-cat-and-a-dog-norms.png

The norm of each of the 77 token positions for x_raw = E(m) − E(a) − E(b), x_fix = x_raw + E(""),
and the joint prompt E(m), CLIP ViT-L slice on top and bigG below, for "a cat and a dog". Numbers in
`a-cat-and-a-dog-norms.json` beside it.

**How it was made.** `scripts/and_vector_norms.py --joint "a cat and a dog" --a "a cat" --b "a dog"`
on mscluster84, fp32, text encoders only, 2026-10-28. The same check on "dog and cat", "dog", "cat"
was written on the cluster at
`/datasets/mmolefe/poe_repair_min/outputs/and_vector_norms/dog-and-cat-norms.json`.

## dog-and-cat-sheet.png

Twelve columns from the same starting noise per seed, seeds 1 to 8 down the page: dog alone, cat
alone, the joint prompt "dog and cat", PoE of dog and cat, then x_raw alone, x_fix alone, PoE with
x_raw, PoE with x_fix, x_aff alone (x_raw + 2E("")), PoE with x_aff, PoE with "single animal", PoE
with "two animals". Calls and the reading against the pass marks written before rendering are in
the idea map, claim 2.

**How it was made.** `scripts/and_vector_sheet.py --render` through `scripts/and_vector_sheet.sh`
on mscluster106 device 1, outside Slurm, in two passes ending 2026-10-28 00:17 and 00:34 cluster
time; tiles at `/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/and_vector/dog__x__cat/`.
Sheet built on the laptop with `scripts/and_vector_sheet.py --sheet`.

## separation-steering-sheet.png

Does nudging the starting latent so the dog expert and the cat expert stop attending to the same
pixels make plain PoE draw one dog and one cat? Columns, left to right: plain PoE of "dog" and
"cat", plain PoE with the latent steered at three step sizes (small, medium, large), then the joint
prompt "dog and cat" as the reference. Rows are seeds 1 to 8, each seed with the same starting
noise in every column.

**Pass marks, written before any steered tile was rendered.** Counted on seeds 1 to 7, because
seed 8 is outside the set where the joint prompt draws one of each. A tile counts when it shows
one dog and one cat, each whole. A tile that shows half-and-half animals (a mosaic) or one animal
is a failure. The question is supported if some step size gives that on at least 4 of the 7 seeds.
The plain PoE column is re-rendered by the same script, device and pass as the steered columns,
so it is the no-steering count read by the same rule. The tiles are read by eye, and the reading is open to a veto.

**What steering does.** At steps 0 to 9 only, before each PoE update, the latent takes one
gradient step. The loss is the overlap of the two experts' attention maps (the dog expert's map
for its "dog" word times the cat expert's map for its "cat" word, summed over the 32 x 32 grid)
plus, for each map, one minus its largest value. Maps are averaged over heads and over every
32 x 32 cross-attention layer, then divided to sum to 1. The UNet is frozen and nothing is trained.

**Step sizes, fixed from one calibration run on seed 1 and not changed after.** In that run, with
no steering applied, the latent has norm about 256 and the loss gradient has norm 1e-4 on average
(steps 0 to 9, range 6.6e-5 to 1.8e-4). The overlap term alone has gradient norm 1e-6 to 8e-6, so
the two "largest value" terms supply about 98% of every update. The maps are close to flat at
these steps: the largest map value is 0.0012 against 0.00098 for a perfectly flat map. Step size
eta is set so one step moves the latent by a fixed fraction of its norm at the average gradient:
small 2.5e4 (1%), medium 1.2e5 (5%), large 5e5 (20%).

**Cost.** One steering step is a forward and backward pass over the two experts' prompts, and the
PoE update is the usual forward over three prompts. Seconds per tile and pass counts are in each
tile's JSON sidecar and in `separation-steering-sheet.json`.

**How it was made.** `scripts/separation_steering.py`, launched through
`scripts/separation_steering.sh` on mscluster109, outside Slurm. Calibration output:
`/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/separation_steering/calibration_seed1.json`.

**Result.** Seeds 1 to 7, read by eye: no steering 0 of 7, small 0 of 7, medium 0 of 7, large 0 of 7
clean, plus one borderline tile (large, seed 6: two small animals, a white one and a cat, on a
background that has turned into saturated blocks). The mark was 4 of 7, so the question is not
supported. Small steps keep the photograph style and give one animal, a wallpaper of many animals
(seeds 1 and 4) or a dog-cat hybrid. Medium and large steps turn the picture into flat graphics or
saturated blocks on every seed, which counts as destroyed, not as separated.

**Cost.** Plain 14.8 s per tile (8 tiles, range 14.2 to 16.3), steered 18.7, 18.6 and 18.6 s (8
tiles each, range 17.8 to 18.8), on an RTX A6000, 1024 x 1024, 8.4 GB. Per steered tile: 10
forward-and-backward passes over two prompts at steps 0 to 9, and 50 forward passes over three
prompts. Plain PoE is 50 forward passes over three prompts.

**Caveat.** The no-steering tile from this script differs from the earlier `and_vector` plain PoE
tile on seed 1 by a mean of 12.5 grey levels with the same composition; the original script
rendered again on this device differs from its own earlier tile by 13.9. Plain PoE in fp16 does not
reproduce pixel for pixel across runs here.

<!-- cluster:start -->

## On the cluster

The other 3 files of this result (0.0 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/5-text-embedding-third-expert/artifacts/results/the-and-vector/what-the-and-vector-is-made-of/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 3 | 0.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/5-text-embedding-third-expert/artifacts/results/the-and-vector/what-the-and-vector-is-made-of/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/5-text-embedding-third-expert/artifacts/results/the-and-vector/what-the-and-vector-is-made-of/<run>/ <run>/
```

<!-- cluster:end -->
