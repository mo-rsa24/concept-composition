# What each line of Readout Guidance computes

Two code walks for Readout Guidance (Luo et al., CVPR 2024, [arXiv 2312.02150](https://arxiv.org/pdf/2312.02150#page=4)), level 3 of its zoom ladder, the way classifier guidance has [the authors' walk](../what-each-line-of-classifier-guidance-computes/README.md) and [ours](../what-each-line-of-our-guided-sampler-computes/README.md):

- the [Readout Guidance loop walk](https://claude.ai/artifact/SHnMMo46yDvhj2GaXuEs7H) (private): the released code, google-research/readout_guidance at `a8a699c`, cloned read-only at `/home/molef/PhD/replications/readout_guidance`;
- the [count head walk](https://claude.ai/artifact/2VgqMaBHqrXNoHdTWjxro1) (private): this project's `scripts/rung2/train_count_head.py` at `c5e61cec`, with every place it departs from the released code named by [the read map's](../../drips/read-readout-guidance-learning-control/READ_MAP.md) caveat R1 to R10.

The [Readout Guidance rebuild](/home/molef/goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-readout-guidance-from-fundamentals.md) beside the shelved PDF takes both walks' equations down to 24 rungs, each linked to where the learning tree teaches it, with five gaps.

codex stills: not answered

## readout-guidance-loop-codewalk.html

<a href="readout-guidance-loop-high-level-design-explainer-light.png"><img src="readout-guidance-loop-high-level-design-explainer-light.png" width="400" alt="Twelve boxes in two rows: the head's training on top, one guided sampling step below, grey frozen model, green head, red readout, orange distance and push"></a>

**What it shows**

The top row trains the SDXL pose head: a picture noised forward to one random step, one frozen U-Net step on the empty prompt, nine decoder resnets caught and resized to 128 px, nine bottlenecks with the time vector added, mixed by learnt shares, a 3-channel readout, and the sparse MSE against an OpenPose label. The bottom row is one guided step: the U-Net on the CFG batch of four, the same taps and head, the distance to the pose on both halves, one fresh AdamW step on the latent, then the ordinary DDIM step at η = 1. A `Code` pill opens ten files of the clone byte for byte in 33 sections; each title starts with its role.

**What the rig ran, on CPU**

<a href="pictures/eq_readout.png"><img src="pictures/eq_readout.png" width="400" alt="The pose, the head's dark reading at t = 991, the squared error, and where the head's error gradient lands on the features"></a>

`rg_rig.py` imports the authors' functions unchanged and runs SDXL's U-Net with float16 weights and float32 arithmetic, the released `readout_sdxl_spatial_pose.pt` (Hugging Face g-luo/readout-guidance, re-saved to CPU), at the released pose config's first guided step: seed 100, batch 2, t = 991, the shipped prompt and pose for 000000013291. One forward over the CFG batch of four took 165 s. Numbers in `pictures/rig.json`, log in `logs/rig.log`.

| Line | Shape on the rig |
|---|---|
| latents, the batch of two (same seed, so identical) | (2, 4, 128, 128) |
| the U-Net's input and context, CFG batch | (4, 4, 128, 128) and (4, 77, 2048) |
| the 9 hidden states, up blocks 0, 1, 2 | (4, 1280, 32, 32) ×3, (4, 640, 64, 64) ×3, (4, 320, 128, 128) ×3 |
| `collect_and_resize_feats` | (4, 128, 128, 6720), 1680 MiB (about 1.76 GB) in float32 |
| the time vector | (2, 1280) |
| one bottleneck's output | (1, 384, 128, 128) |
| the readout | (1, 3, 128, 128), −0.498 to −0.188 |
| the head's error gradient on the features | nonzero on rows 1 and 3 only (the guided image), length 4.99e-4 and 6.59e-4 |
| one fresh AdamW step on a stand-in gradient of the latent's shape | each element of the guided image moved 0.019995 to 0.020000 at gradient scale 1, 0.020000 at 1000, and 0.000080 to 0.019956 at 1e-6; 0 on the reference image; 8-bit AdamW's largest move 0.020000 |

The released head has 5,907,436 parameters (the paper's Table 2 says 5.9M), was saved at step 4000, and its softmax mixing shares run from 0.154 (layer 1) to 0.092. The guidance loss is 0.3551 on the unconditional half and 0.3830 on the conditional half. The backward pass through the U-Net does not fit in this laptop's memory, so the push's real gradient size is not reported; on the stand-in gradients the step rule moves each element by about 0.02 unless the gradient is small enough for Adam's ε to matter.

**Matched against the network view's numbers.** All 21 overlapping values in journey 13's `artifacts/figures/readout-guidance/rg_numbers.json` match the rig (`logs/match-rg-numbers.log`): the nine tap names, widths and sides, the count head's three taps, the latent, the 1280 time vector, 384 and 96, 5,907,436 and 706,049 parameters, the output head widths, 50 guided steps, lr 0.02, loss_rescale 0.5, the 32 × 32 output, the 1.0 bound and the first Adam step. One entry there disagrees with the code: it names `torch.optim.AdamW` as the optimizer, while `rg_pipeline.py` line 159 picks bitsandbytes' 8-bit AdamW whenever the latents are float16 (line 264), which they are on SDXL.

**What the code shows that the paper does not**

- The time input at training and at guidance are two different numbers for the same noise level. Training gives the head e(j), j the generation step index with 1 near pure noise (`diffusion_extractor.py` line 148); guidance gives e(t), t the scheduler timestep (`rg_pipeline.py` line 165). At the first guided step that is e(8) against e(991), cosine −0.462 on the rig. Read from the code and computed, not tested on a render; the released weights may come from another commit.
- The push is one fresh AdamW step on the latent itself (lines 154 to 179), not the printed edit to ε; the paper's rg weight 0.5 is `loss_rescale`, which the step ignores (read map slip and R8).
- The stacked features have 6720 channels (3 × 1280 + 3 × 640 + 3 × 320).
- The batch must be at least two: image 0 is the unguided reference, the loss compares image 1 with the pose, and both images share every noise draw, so `orig` and `rg` in `script_spatial.py` differ only by the push.
- `load_aggregation_network` calls `torch.load` with no `map_location`, so the released weights do not load on a CPU-only machine as shipped.
- The checkpoint's run id says lr 1e-4 and batch 8; its stored config says lr 1e-3 and batch 4.

## our-count-head-codewalk.html

<a href="our-count-head-high-level-design-explainer-light.png"><img src="our-count-head-high-level-design-explainer-light.png" width="400" alt="Eleven boxes: saved states into the frozen U-Net, three hooks, pooling and the density map on top; the detector boxes, blob target, loss and AdamW below; the held-out check and its bar"></a>

**What it shows**

The script's 185 lines in 11 sections: the three hooks, pooling to 32 px, the density head, the blob target, the training batch from plan 23's states, the loss, and the held-out check with its bar. There is no push in the script. The rig compiles lines 1 to 91 of the committed file under its own path and runs its `Readout` and `CountHead` on SDXL's real features: taps (1, 1280, 32, 32), (1, 1280, 64, 64), (1, 640, 128, 128), each pooled to 32 × 32 and projected to 128 channels; 706,049 parameters; the untrained map sums to 679, a shape check only. `target_map` on two boxes typed for the check gives σ = 2.625 and 3.0 cells and a map summing to 2.000.

**Where it departs from the released code, by caveat**

| Caveat | Where in the script | Meets? |
|---|---|---|
| R1, detector labels | lines 103 and 106: every target from GroundingDINO's boxes | no |
| R2, time input | `CountHead`, lines 69 to 79: features only | no |
| R3, layers and mixing | line 58 reads the mid block and up blocks 0 and 1, skips up block 2; line 78 pools to 32 px and sums with no mixing weights | partly |
| R4, trajectory against forward-noised states | lines 167 to 170 train on plan 23's states; line 135 noises check tiles forward | partly |
| R5, early steps | lines 126 to 166 score steps 0 to 19 against the bar; lines 168 to 170 include step-0 states with finished-drawing labels | meets as a check |
| R6, sparse target | line 175: no reweighting of blob against background (70 percent of the typed target's cells are below 0.001); the count term moves the total | partly |
| R7 and R8, the push | no push in the script | not in scope |
| R9, which animals | line 146 reports the all-negatives balanced accuracy | meets as reported |
| R10, held out | line 121's check tiles may include cat × dog seeds 7 and 8 | partly |
| none covers it | line 58 reads whole block outputs, after attention and the upsampler (up block 0 at 64 px), where the released taps keep each resnet's hidden state before the skip add | |
| none covers it | line 79: one softplus channel summing to a count, where the released head gives three tanh channels | |
| none covers it | line 117: AdamW at lr 3e-4, weight decay 0.01, 2000 steps, where the released head trains with 8-bit AdamW at lr 1e-3, weight decay 0, up to 5000 steps | |

**How we got it**

`rg_rig.py` writes `pictures/rig.json` and `pictures/rig_arrays.npz`; `equation_pictures.py` draws the 12 term figures from them with `~/.claude/figkit/termfig.py` (`logs/equation-pictures.log`); `gen_codewalk.py` reads the ten clone files and the committed count head byte for byte, asserts 180 cited lines against their text, takes every caption number from the two JSON files, and writes `readout-guidance-loop-code.json`, `our-count-head-code.json`, `codewalk-released.json`, `codewalk-count-head.json` and the two role-coloured SVGs (drawn by the hld-diagram skill's `draw_hld.py`, overlaps checked by eye in the light PNGs). `build_codewalk.py ... --strict` exits 0 for both: 12 of 12 boxes and 33 sections over 2447 lines in 10 files, and 11 of 11 boxes and 11 sections (`logs/build-codewalk-*.log`). `probe_codewalk.py --dark` prints `ok` for both (`logs/probe-*.log`). `urls.json` holds the two links so each page links the other.

Colours follow journey 13's key: blue `#1f5fd0` inputs, green `#1d7a3e` the head, red `#c0262d` its readout, orange `#c86a0a` the error gradient and the push, grey `#6b7a88` the frozen model and the ordinary sampler. Symbols follow journey 13's [notation](/home/molef/goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/NOTATION.md) (R_ψ(x_t), R\*, ℓ, ω_l, g_b, σ_b); the walks' new symbols were appended there.

**What it lets you say**

Which line of the released code computes which part of the paper, at what shape on SDXL; that a guided step moves every latent element by about 0.02 whatever the gradient's size; and, line by line, where our count head copies the released head and where it does not, with the caveat that covers each difference.

**What it cannot say**

The size of the real push gradient through SDXL (the backward pass does not fit on the laptop), whether the time-input mismatch hurts a render, and anything about the count head's trained behaviour: its labels, states and check tiles are on the cluster mount, and no training run or `check.json` existed when the page was built. The training code is read, not run.

**Where it came from**

The clone at `a8a699c`, the released SDXL pose head, SDXL base 1.0's local snapshot, `train_count_head.py` at `c5e61cec`, the read map, journey 13's notation and `rg_numbers.json`, and the shelved PDF.

**Where it is registered.**

Both walks are in the scene library, in journey 13's `artifacts/scenes/scenes.json`, and in the HLD register as "Readout Guidance Loop" and "Our Count Head", with their links. The Readout Guidance ladder record in journey 13 (`artifacts/ladder/readout-guidance/ladder.json`) names both as its level 3, and the teaching files the companion links carry back-references to it. The HLD gallery is not rebuilt from the register yet, for every walk alike.

**Reviewed by Codex: the Readout Guidance loop walk**

A read-only review (job `20261118-061115-905446`) checked the 33 sections against the cited lines, `pictures/rig.json` and `rg_numbers.json`. It found all 33 ranges holding their code, and the CFG rows, the detached range, the two time arguments, the DDIM algebra and the optimizer construction consistent. Applied, in `gen_codewalk.py` and the diagram spec:

- The push: the measured moves are given as ranges (0.019995 to 0.020000 at scale 1; 0.000080 to 0.019956 at 1e-6), where the page said every element moved 0.020000; the 1.0 bound covers the pushes alone, with the DDIM steps between them outside it; gradient checkpointing reruns forward pieces in the backward.
- 1680 is MiB (about 1.76 GB); the stacked features keep their batch axis of 4; the nearest-neighbour resize keeps sampled colours exact but can lose thin lines.
- The maths: the resnet branch adds W SiLU(e) and the block returns S(x) + F with S the shortcut; output_head already ends in tanh, so the readout is no longer wrapped in a second one; the bottleneck's time vector passes a ReLU (line 127); ν's range comes from the detached latents (line 171); renormalize is a linear map with no clipping; bilinear sampling clamps at the edges.
- The sparse loss takes the plain mean when nothing is null (line 56), and its picture caption now says the drawn elements carry most of the plain mean; the training loop can take 5001 updates (lines 118 to 119); all generators share one seed (line 271).
- The four CFG rows are named; the loss section runs to its `return` at line 212; the time edge on the diagram reads "training e(j), guidance e(t)"; the time caption says equal arguments, not equal noise levels, would give cosine 1.
- Cards no longer print the repository's absolute path; the not-run note drops a reason that did not hold for the DDIM helper; a bare caveat label (R8) became the exact-arithmetic condition.

Not applied:

- Equations laid out one move per line, card paragraphs under bold headings, a larger desktop diagram, hover against selected highlighting and the always-visible key: `build_codewalk.py` under `~/.claude`, which this review did not edit.
- A target-pose node and a second U-Net box on the diagram: the page is at its twelve-box limit, and the p_step section names the second U-Net call.

**Reviewed by Codex: the count-head walk**

A read-only review (job `20261118-061115-301bcc`) checked the 11 sections against the 185-line script, `READ_MAP.md` and `pictures/rig.json`. It found the tap shapes, density shape, parameter counts, blob widths and target sum consistent. Applied:

- The loss is the batch mean over 8 entries, in the equation and on the diagram (map MSE and count MSE, ∇ψ L); the 437 against 45815 comparison is one untrained example; the count term's contrast frame is gone.
- The 70% near-zero share belongs to the two typed boxes; p is a cell centre (0.5 to 31.5); the batch is drawn with replacement; evaluation also runs before the first update; the docstring fixes the rule before training, rather than making it impossible to move.
- The head has no separate time input while the U-Net still reads t; the pooling caption separates shapes before and after pooling; the held-out caveat says the cat × dog check tiles may overlap with training drawings; the schedule figure names √ᾱ as the clean latent's weight and states the real bar.
- The diagram's verdict reads "balanced accuracy ≥ 0.75" on 10 of 20 steps from a mean of 4 counts, and the head's update arrow and readout eyebrow say the readout is trained too.

Not applied:

- The renderer changes (bold-headed paragraphs, diagram width, hover) for the reason above; trimming the rig notes, since they say which lines a breakpoint stops on.

### Versions

- `readout-guidance-loop-high-level-design-explainer__2026-11-17.svg`: version 1, twelve boxes and thirteen arrows, published as version 3 of its page (cross-links and notation).
- `our-count-head-high-level-design-explainer__2026-11-17.svg`: version 1, eleven boxes and eleven arrows, published as version 3 of its page.
- `readout-guidance-loop-high-level-design-explainer__2026-11-18.svg`: the time edge labelled with both arguments, published as version 4 with the Codex review applied.
- `our-count-head-high-level-design-explainer__2026-11-18.svg`: the loss, update and verdict relabelled, published as version 4 with the Codex review applied.

## Cross-references

- The mention of **code walk's folder** in [the Readout Guidance rebuild, words this uses](../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-readout-guidance-from-fundamentals.md#words-this-uses).

<!-- cluster:start -->

## On the cluster

The other 42 files of this result (2.4 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-readout-guidance-computes/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 23 | 1.8 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-readout-guidance-computes/` |
| `logs` | 8 | 0.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-readout-guidance-computes/logs/` |
| `pictures` | 11 | 0.6 MB | [eq_readout.png](pictures/eq_readout.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-readout-guidance-computes/pictures/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-readout-guidance-computes/<run>/ <run>/
```

<!-- cluster:end -->
