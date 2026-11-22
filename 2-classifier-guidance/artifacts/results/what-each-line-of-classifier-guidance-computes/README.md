# What each line of classifier guidance computes

Dhariwal and Nichol's classifier guidance ([arXiv 2105.05233](https://arxiv.org/pdf/2105.05233#page=6), Section 4, Algorithms 1 and 2 on page 7), walked line by line through the authors' code beside the VS Code debugger. The page is the [classifier guidance loop code walk](https://claude.ai/artifact/SjQVHwtg1EUgBxrBgJsB5h) (private).

The [classifier guidance rebuild](/home/molef/goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/rebuild-classifier-guidance-from-fundamentals.md) follows this page's 17 sections in order and takes each section's equation down to the fundamentals it rests on, 32 rungs in all, each linked to where the learning tree teaches it.

[Classifier guidance, read through a small network](/home/molef/goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/guided-sampling/classifier-guidance-read-through-a-small-network.md) is the third way in: the same method on a two-pixel image and two networks small enough to compute by hand, each idea worked on them before the paper's equation.

codex stills: not answered

## classifier-guidance-loop-codewalk.html

<a href="classifier-guidance-loop-high-level-design-explainer-light.png"><img src="classifier-guidance-loop-high-level-design-explainer-light.png" width="400" alt="Eight boxes in three rows: the ordinary sampler on top in grey, the steering in the middle in orange, the scale s alone at the bottom in magenta"></a>

**What it shows**

Which lines of the guided loop are the paper's own. The top row, grey, is the ordinary sampler with no classifier: the loop, the network's mean and variance, the noise draw, the DDIM update. The middle row, orange, is the steering: `cond_fn` asks a noisy classifier for the gradient of log p(y | x_t), and `condition_mean` (Algorithm 1, Eq. 10) or `condition_score` (Algorithm 2, Eq. 14) moves the step with it. The paper reviews that steering from Sohl-Dickstein 2015 and Song 2020 in Sections 4.1 and 4.2, and the code's docstrings say the same. The bottom box, magenta, is the gradient scale Section 4.3 studies: line 61 of `classifier_sample.py` multiplies the gradient by the scale s, which defaults to 1.0 and is 10.0 in the README's command for the unconditional 256 px model.

A `Code` pill opens both upstream files byte for byte, numbered as the debugger numbers them, in 17 sections. Each section title starts with its role. Each gives the breakpoint line and what to read in the Variables pane, the equation its lines compute (paper's number and page, reduced to the code's form), every symbol with a plain name, a figure taking the equation apart on the line toy, and what the lines did to the seeded corgi run.

**What the method buys, on the exact rig**

`three_ways_mean.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/pictures/three_ways_mean.png`

The line toy (cat's clean values from N(+1, 1), dog's from N(−1, 1)) run through the authors' own sampler, with the UNet and the classifier replaced by their exact answers. 4096 particles, 250 steps, seed 0, the same starting points and the same noise in every run. Numbers from `pictures/rig.json`.

| Run, at the end | mean | sd | share on the cat side (x > 0) |
|---|---|---|---|
| unguided (the guidance lines deleted) | −0.012 | 1.385 | 0.494 |
| scale 0 | identical to unguided, largest difference 0.0 | | |
| scale 1 (the paper's scale deleted) | 0.964 | 0.988 | 0.837 |
| exact cat class, N(1, 1) | 1.000 | 1.000 | 0.841 |
| scale 10 (the full loop) | 2.448 | 0.683 | 1.000 |
| p(x)·p(cat \| x)^10 / Z, Section 4.3's reading | 1.843 | 0.681 | 1.000 |
| DDIM, scale 1 / scale 10 | 0.984 / 3.574 | 0.986 / 0.472 | 0.840 / 1.000 |

Scale 1 samples the cat class within a few hundredths. Scale 10 follows the mean of p_t(x)·p_t(cat | x)^10 until about step 150, then separates from it and ends 0.6 further out with almost the same spread (0.683 against 0.681). So at s = 10 the loop does not end on the p(x)·p(y | x)^s that Section 4.3 describes, and this run does not show why. It is one seed on a one-dimensional toy with exact scores.

**What the code shows that the paper's equations do not**

The derivation (Eq. 5 and 6) takes the gradient at the mean μ of the next state and the next state's noise level; Algorithm 1 and line 365 take it at the current x_t with the classifier at step t. On the toy, for one particle at x_t = 0 at the step nearest √ᾱ = 0.8 with s = 10, one step's shift is 0.128 in the code against 0.117 for the exact product N(μ, Σ)·p(cat | x)^10 (`pictures/eq_pictures.json`). At x_t = 0 the mean μ is also 0, so that gap comes from the Taylor step and the noise level, and the example does not isolate where g is taken. Algorithm 2 as printed has no s; its caption names s and line 61 applies it. Line 383 re-derives the noise guess from x̂₀, clipped when `clip_denoised` is on, so under `condition_score` the guess can differ from the UNet's raw output. With exact scores and s = 1, Eq. 14 gives the cat-only noise prediction at every x_t (largest difference 7e-16; −0.480 at x_t = 0, as the journey's answer key says).

**What those lines did to the seeded image run**

<a href="pictures/run_grad.png"><img src="pictures/run_grad.png" width="400" alt="Two log-scale line plots against step: the classifier's raw gradient norm and the push as a percentage of the mean's norm, three classes at scales 1 and 10"></a>

Read from the per-step logs of [the seeded run of the released weights](../does-classifier-guidance-work-on-the-released-weights/README.md), seed 0. No GPU run was made for this page. At s = 10 the corgi's raw gradient falls from 0.52 at step 199 to 0.11 at step 49, and the final image is called a corgi; at s = 1 it reaches 2.64 at step 149 and is still 1.85 at step 49, and the final image is not. The norms measure how sensitive the classifier is, not when the picture becomes a corgi. The push that line 367 adds is at most 0.13 percent of the mean's norm at s = 10. Numbers in `pictures/run_numbers.json`.

**How we got it**

`rig.py` imports the upstream `gaussian_diffusion.py` and `respace.py` from the guided-diffusion clone at `22e0df8` unchanged (with bytecode writing off, so the clone stays clean: 0 changed files afterwards), builds the same 250-step `SpacedDiffusion` the seeded run used, and replaces the UNet and the classifier with exact functions; `cond_fn` and `model_fn` are copied from `classifier_sample.py` lines 54 to 65. Two settings differ from the image run: FIXED_SMALL variance (the released model learns its own, which the `eq_loop` figure shows sits on the same lower bound except for one-step jumps at steps 208 and 125) and no clipping, because the toy's clean values are not confined to [−1, 1]. Run with `PYTHONDONTWRITEBYTECODE=1 /home/molef/PhD/replications/fkc-diffusion/.venv/bin/python -B rig.py` (25 s on CPU, torch 2.15.0), log in `logs/rig.log`. `equation_pictures.py` draws the eight term figures with `~/.claude/figkit/termfig.py` (log in `logs/equation-pictures.log`), and `seeded_run_pictures.py` draws the seeded-run plots from the existing logs.

`gen_codewalk.py` reads both upstream files byte for byte, asserts 46 cited lines against their text, takes every caption number from the three JSON files, and writes `classifier-guidance-loop-code.json`, `codewalk.json` and the role-coloured SVG (from `classifier-guidance-loop-high-level-design-explainer.json`, drawn by the hld-diagram skill's `draw_hld.py`, overlap check clean). The page is built with `python3 -I ~/.claude/skills/hld-diagram/scripts/build_codewalk.py codewalk.json classifier-guidance-loop-codewalk.html --strict`, which exits 0 with 8 of 8 boxes explained, 17 sections over 1039 lines in 2 files, 26 pictures. `python3 -I ~/.claude/skills/hld-diagram/scripts/probe_codewalk.py classifier-guidance-loop-codewalk.html --dark` prints `ok`: each box opens its section with its lines in view, no maths left as raw TeX, no symbol printed twice, every lead figure under its maths (`logs/probe.log`).

To step it on a laptop, run `rig.py` under the VS Code debugger with the fkc-diffusion venv's Python and set breakpoints in the clone's `gaussian_diffusion.py`; the first run is unguided, so the condition `int(t[0]) == 51` lands on the toy's step in a guided run.

**Reviewed by Codex**

A read-only Codex review (job `20261117-060929-caf1d0`) checked the line numbers, the maths against pages 6 to 8, the roles, the symbols, the numbers and the prose; it found the cited lines and the numbers consistent with their files. Applied:

- The scale-10 result no longer claims the loop samples something sharper; the spreads match (0.683 against 0.681) and only the location differs.
- `model_fn`'s maths splits the six-channel output into the noise guess and the variance channels.
- `p_mean_variance`'s maths shows the clip, and its terms say the code does not hold f inside [0, 1].
- `condition_score` names the raw noise guess and the recovered `eps` apart, and says only the rig's DDIM runs reach it.
- Algorithm 1's derivation uses u for the next state and names the derivation's gradient g_μ apart from line 365's g_t; its figure caption says the x_t = 0 example cannot separate the two.
- The DDIM section shows the reduction at η = 0 to Algorithm 2's update.
- The role wording says "the sampling here" and "unscaled guidance" where it said "every diffusion sampler" and "the earlier works' guidance".
- The scale-1 reference mean is integrated on a wider grid and reads 0.800 (it read 0.798); the scale-10 reference mean moved from 1.844 to 1.851.
- The seeded-run captions no longer infer when the picture becomes a corgi from gradient norms.
- The debugger setup is written once in the page's notes, with the interpreter path and the order of the rig's runs.
- The diagram's branches are labelled "default: ancestral" and "--use_ddim: DDIM", and its reading notes separate call order from what each box reads.
- Missing symbols were added to the terms of the shorter sections.

Not applied:

- A larger overview, or an "Expand diagram" control: the diagram's size inside the page is set by `build_codewalk.py` under `~/.claude`, which this work did not edit.
- Splitting the Taylor block into three labelled blocks: it is one aligned block with a sentence after it saying why completing the square leaves a constant.

**What it lets you say**

Which line of the authors' code computes which equation of Section 4, with each reduction written out; which lines are the ordinary sampler, which steer, and which one line is the paper's own; and, on an exact toy, that scale 1 samples the class and scale 10 ends further out than p(x)·p(y | x)^10 with about the same spread.

**What it cannot say**

The image run is seen through per-step norms and final tiles, since no snapshots of the image at six steps exist and no GPU run was made. DDIM (Algorithm 2) never ran on the images; its pictures are from the toy. The scale-10 overshoot is one seed on a one-dimensional toy with exact scores and says nothing yet about the 256 px model. The page loads KaTeX's script from cdnjs and draws the maths as MathML.

**Where it came from**

The paper PDF in this session's scratchpad (pages 6 to 8), the guided-diffusion clone at `22e0df8`, the seeded run's card beside this one, and the journey's [notation](/home/molef/goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/NOTATION.md) and [gate](/home/molef/goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/GATE.md) for the line toy.

**Held since 2026-11-17.**

The scene-library section in `Scenes.md` (with `scene-cards.py --apply` and `--refresh-marks`) is owed for this page. The HLD Gallery republish is held: the rebuilt gallery lists 23 cards but drops "Product validation evolution", which the live page carries, so it is rebuilt from the full registry before it replaces the live page. Plan 17's Figure Catalog carries this page.

### Versions

- `classifier-guidance-loop-high-level-design-explainer__2026-11-17.svg`: version 1, eight boxes and eight arrows, the ordinary sampler, the steering and the paper's scale in three labelled rows, published as Version 1.

## guided-step-stepper.html

**What it shows**

One step of Algorithm 1 on the line toy, one press per term: the current point on the cat and dog halves of the noised density, the unconditional score, the classifier's gradient g = 2√ᾱ·p(dog | x_t), the push s·Σ·g, the guided mean, the noise √Σ·z, and x_{t−1} with the dashed total move. A strength control (s = 0, 1, 3, 10) and a draggable x_t redraw it, every symbol opens its meaning and value, and two questions ask for a prediction before the push and before the update. The [guided step stepper](https://claude.ai/artifact/1g1Y617YgRmzxDxuPgRnQy) (private) is linked from the code walk's See also and from its Algorithm 1 section.

**What one step looks like, on the exact rig**

At x_t = 0.5 the push is 0.008 at s = 1 and 0.080 at s = 10, against a noise draw of 0.195, because the step's variance Σ is 0.016. The model's own move is 0.0009: the score pulls by −0.0032 and dividing by √α_t stretches x_t out by +0.0042. At s = 10 the push exceeds that noise draw for every x_t at or below −0.75. At x_t = 0 and s = 1 the guided noise prediction (Algorithm 2's form) equals the cat-only one, both −0.4792, difference 9e-9; the answer key's −0.480 uses √ᾱ = 0.8 and σ = 0.6 exactly, where this step has 0.8018 and 0.5976.

**How we got it**

`export_stepper_data.py` copies `rig.py`'s construction (it cannot import it, since `rig.py` samples and draws on import), checks the copy against the rig's saved step-51 arrays (`pictures/rig_toy_step_s10.npz`, 4096 particles, largest difference 0.0), and calls the upstream `p_mean_variance`, `condition_mean`, `condition_score` and `p_sample` at step i = 51 of 250 for 101 values of x_t from −2.5 to 2.5 and the four strengths, with `torch.manual_seed(0)` before each `p_sample` so every point shares one noise draw z = 1.541. It writes `guided-step-stepper-data.json`, which the page fetches; no number is typed into the page. Run with `PYTHONDONTWRITEBYTECODE=1 /home/molef/PhD/replications/fkc-diffusion/.venv/bin/python -B export_stepper_data.py`, log in `logs/export-stepper-data.log`; the clone has 0 changed files afterwards.

Codex wrote the page (job `20261117-063908-082926`, two rounds). Each round was run in Chrome at 1440 px light and 390 px dark with the full data file: every number shown matched the file, no horizontal scroll, no script errors, MathML rendered. Round 2 moved the labels onto their arrows and removed colliding tick labels; the numbers did not change between rounds.

**What it cannot say**

It shows one step at one noise draw. It says nothing about the 249 other steps, or about the image model, whose variance is learned.

<!-- cluster:start -->

## On the cluster

The other 37 files of this result (5.9 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 14 | 3.9 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/` |
| `logs` | 6 | 0.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/logs/` |
| `pictures` | 17 | 2.0 MB | [run_grad.png](pictures/run_grad.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/pictures/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-classifier-guidance-computes/<run>/ <run>/
```

<!-- cluster:end -->
