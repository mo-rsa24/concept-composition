# SDXL from the prompt down to one attention head: the zoom ladder

SDXL read at four levels, to show where CO3's fix and Loop 12's step Z2 readings sit inside the network: one real
picture from the prompt to the pixels, three figures inside the network, one attention head worked by hand, and the
code. Start at [the reading page](index.md). The ladder's record is `ladder.json`, its symbols are in
[the notation file](NOTATION.md), and the rungs still owed are in [the route file](route-zoom-ladder-sdxl.md).

| File | What it is |
|---|---|
| `index.md` | the reading page: the overview first, each deeper figure linked from the text about its part, and the existing figures linked in their places |
| `l0-sdxl-from-prompt-to-picture.png`, `.svg` | level 0: "an eagle and a fox", seed 6001, from the prompt through both text encoders, the U-Net 50 times and the VAE decoder, with CO3's box, Z2's three readings and code boxes |
| `l05a-inside-the-unet.png`, `.svg` | level 0.5: the U-Net's seven blocks with grids, channels, skip widths, ResBlock and transformer-block counts, and where the time and word vectors enter |
| `l05b-one-transformer-block.png`, `.svg` | level 0.5: one transformer block at 32 × 32, with cross-attention opened and Z2's capture marked |
| `l05c-noise-guess-and-clean-guess.png`, `.svg` | level 0.5: the share of picture at steps 0, 5, 20 and 45, the clean guess formula, and the slot waiting for job 94450 |
| `l2-one-attention-head-by-hand.png`, `.svg` | level 2: one head worked by hand beside the real eagle and fox columns at steps 3, 20 and 40 |
| `code-walk.md` | level 3: the lines that compute each level, byte for byte with line numbers |
| `sdxl_numbers.py`, `sdxl_numbers.json` | every size and value any figure or page prints |
| `sdxl_figkit.py` | the shared colours and the check that refuses a figure where a line crosses a node, box or label |
| `draw_*.py`, `gen_codewalk.py` | one script per figure, and the code page's generator |
| `ladder.json`, `NOTATION.md`, `route-zoom-ladder-sdxl.md` | the ladder's record, its symbols, and the route with the rungs still owed |
| `source/sdxl-config/` | SDXL base 1.0's config files, copied read-only from the cluster |
| `source/diffusers-0.29.2/` | the seven diffusers files the code page cites, copied read-only from the cluster |

## How each item was made

**The numbers.** `python3 sdxl_numbers.py` reads SDXL base 1.0's config files (`unet`, both text encoders, scheduler,
VAE, `model_index.json`), copied on 2026-11-24 from mscluster's Hugging Face cache, snapshot
`462165984030d82259a11f4367a4eed129e94a7b` (the same snapshot journey 13's sizes came from). It derives the U-Net's
blocks, skips and counts by following the constructor lines of diffusers 0.29.2, asserting each line it leans on, and
stops if a sum fails: 768 + 1,280 = 2,048 = `cross_attention_dim`, and 1,280 + 6 × 256 = 2,816 =
`projection_class_embeddings_input_dim`. It finds 17 ResBlocks and 70 transformer blocks (10 at 64 × 64, 60 at 32 × 32),
and checks the 70 against the layer count Z2's smoke run recorded. It rebuilds the DDIM schedule in numpy from the
scheduler config (scaled-linear betas from 0.00085 to 0.012 over 1,000 steps, leading spacing, offset 1), checks every
step's $\bar\alpha$ at $t-1$ against the CO3 ladder's `co3_numbers.json`, and checks that the share of picture rounds to
0.6%, 1.7%, 18% and 92% at steps 0, 5, 20 and 45 under both readings. It reads Z2's read steps from `z2_bars.py`, the
smoke run's token columns, text cosines, layer count and maps, and computes the hand-worked head (4 patches, 3 words,
2 numbers per vector, chosen so the eagle column spills onto the fox's patch).

**The figures.** Each `draw_*.py` reads only `sdxl_numbers.json`, the copied source files, the smoke run's maps and the
picture, and draws with matplotlib through `sdxl_figkit.py`, a copy of the CO3 ladder's kit with one added colour (gold,
for Z2's readings). The kit registers every box, node, label and line, and refuses to write a figure where a line crosses
a node, box or label, or two labels overlap; all five figures pass. Each figure was looked at after drawing, and fixed
where the check could not see a fault: a code-box title running off the canvas, and the level 2 maps drawn upside down
against their outlines. Blue marks inputs, green SDXL's frozen networks, red outputs, orange CO3, grey what is fixed,
purple the eagle and teal the fox, as in the CO3 ladder.

**The picture and the maps.** The picture is the Z2 smoke run's render of "an eagle and a fox", seed 6001 (W&B
`lmm75rgv`), at `/home/molef/.claude/jobs/0071c318/tmp/z2-smoke2/ours.png`. The maps are
`/home/molef/.claude/jobs/0071c318/tmp/z2-smoke/an_eagle__x__a_fox/seed-6001-maps.npz`: for each of 14 read steps,
the head-mean cross-attention of the eagle, fox, end-of-text and mean padding columns, each layer upscaled to
128 × 128 and summed over the 70 layers. Level 2 rescales each map to run from 0 to 1 and prints its raw minimum and
maximum under it. The animals' outlines are traced from the picture's colours (saturation above 0.28 or brightness
below 0.35, leaving out the top 50 of 128 rows and the first 8 columns, box-blurred and contoured), and miss the eagle's white head and the fox's
white chest. The smoke run's stored overlap numbers read 1.0 at every step, from before the rescaling fix the design
note records, so no overlap number is printed here.

**The noised pictures in 0.5c.** The finished picture is mixed with seeded Gaussian noise (numpy seed 0) at SDXL's
$\bar\alpha$ for steps 0, 5, 20 and 45, in pixels. It shows the share of picture; the sampler noises the
4 × 128 × 128 latent, and the figure says so.

**The code page.** `python3 gen_codewalk.py` copies each block from the diffusers 0.29.2 files under
`source/diffusers-0.29.2/` (read with `sed` from
`/home-mscluster/mmolefe/miniforge3/envs/co3/lib/python3.10/site-packages/diffusers/` on mscluster107, 2026-11-24),
CO3's checkout at commit `35ea972313276014716dca3d821023728e8bc917`, and `z2_predictors.py`, asserting each block's
first and last lines.

**The slot for job 94450.** The bottom row of 0.5c is empty on purpose. On 2026-11-24 job 94450 (`l12-z2`) was pending
in the bigbatch queue and `/datasets/mmolefe/poe_repair_min/loop-12/co3-inside/` did not exist. When it lands, the
decoded clean guess at each of steps 0 to 5, before and after CO3's passes, goes in that row.

**Rebuild everything.** From this folder: `python3 sdxl_numbers.py`, then each `draw_*.py`, then `gen_codewalk.py`.
No GPU, no model weights and no network access are needed once `source/` is filled.
