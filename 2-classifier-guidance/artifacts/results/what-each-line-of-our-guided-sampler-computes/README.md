# What each line of our guided sampler computes

This project's classifier-guided SDXL sampler, `poe_repair/methods/_classifier_guided.py` at the current commit (last changed in `445cef9f`), walked line by line beside [the authors' code walk](../what-each-line-of-classifier-guidance-computes/README.md). The page is [our guided sampler code walk](https://claude.ai/artifact/WwzF3fXBFFb6arZySDA8qv) (private). It is level 3 of the zoom ladder for classifier guidance, beside the authors' walk; the [system picture](https://claude.ai/artifact/FBbwapTjp1dbEmzpLspRBT) is level 0 and the journey's miniature is level 2.

codex stills: not answered

## our-guided-sampler-codewalk.html

<a href="our-guided-sampler-high-level-design-explainer-light.png"><img src="our-guided-sampler-high-level-design-explainer-light.png" width="400" alt="Eleven boxes in three rows: the SDXL step on top in grey with the teal product combine, our two checks in teal in the middle beside the orange classifier and shift, the magenta strength, orange noise level and teal cap at the bottom"></a>

**What it shows**

Which lines of our sampler are the ordinary SDXL step, which are Dhariwal and Nichol's push, and which are ours alone. The top row is one SDXL step: the loop over 50 DDIM timesteps, one UNet call over "a cat", "a dog" and the empty prompt with ordinary guidance at 7.5 on each, and the DDIM update, in grey. The teal box between them is ours: `poe_eps` adds the two prompts' predictions and takes the empty prompt's away once, ε_PoE = ε_cat + ε_dog − ε_uncond. With no head, at strength 0, or on a step outside the window, ε_PoE goes straight to the DDIM step. Otherwise it passes our two checks (strength not 0, step inside the window, teal), the classifier's gradient (orange), the strength s (magenta), σ_t (orange), our cap at 0.1 of |ε_PoE| (teal), and the subtraction of Algorithm 2's Eq. 14 (orange) before the DDIM step.

A `Code` pill opens both files byte for byte, `_classifier_guided.py` (262 lines) and the four runtime helpers it calls in `poe_repair/_sdxl/metrics.py` (46 lines), in 30 sections. Each title starts with its role. Each section gives the breakpoint line and what to read in the Variables pane, the equation in the journey's [notation](/home/molef/goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/NOTATION.md), the matching line of the authors' `condition_score` or `cond_fn` where one exists, every symbol named, and the miniature's numbers run through this file's own lines. The second entry point, `run_cfg_classifier_guided`, is the paper's single-prompt form (no product, no window, no cap); its sections name the same boxes.

**The miniature, through this file's own lines**

`eq_push.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-our-guided-sampler-computes/pictures/eq_push.png`

`miniature_check.py` imports `capped_push`, `guided_eps`, `poe_eps`, `tweedie_mean` and `ddim_prev_from_x0_eps` unchanged, and runs lines 136 and 142 to 146 of the sampler (read from the file, asserted, and compiled under the file's own path and line numbers) on the journey's [miniature answer key](/home/molef/goal-setting/learning/deep-learning/diffusion-models/13-classifier-guidance-by-hand/artifacts/substrate/miniature_answer_key.json). Numbers from `pictures/miniature.json`.

| Run of lines 136 and 142 to 146 | ε after line 146 | what it checks |
|---|---|---|
| `poe_eps` on the key's three predictions | (2.1, −1.8) | the key's ε_PoE |
| w = 0 | (2.1, −1.8), head never called | the canary: line 136 is False |
| w = 1, cap off (`push_cap=None`) | (−8.7, −7.8) | the key's ε_true, equal to float precision |
| w = 1, default cap 0.1 | (1.858, −1.934) | the cap fires: the push of length 12.35 is 4.47 times \|ε_PoE\| = 2.766, and is shortened to 0.277 |

On the miniature the default cap keeps the push from reaching ε_true. The miniature's interaction term is 4.5 times the product's prediction by construction, so this says what the cap does to a large push, and nothing about how large SDXL's pushes are.

**What those lines did to seeded SDXL runs**

<a href="pictures/run_push_vs_cap.png"><img src="pictures/run_push_vs_cap.png" width="400" alt="Two dot plots of the largest and median push over the prediction's length per render at strengths 10 to 300, with the 0.1 cap as a dashed line"></a>

Read from logs already on this machine; no GPU job was run. The product run of 2026-11-05 (`../can-a-classifier-supply-the-missing-term/product-pushed-to-dog/push.json`) ran this function at commit `7bae82b0`, the same push with no cap: plan 18's dog-01 classifier at step 3,000, seeds 9 to 12, strengths 10, 30, 100 and 300. Strength 0 matched the plain product on all four seeds (the tiles' md5s compared here, 4 of 4). A cap of 0.1 would have fired on some step in 9 of its 16 renders (1 of 4 at strength 30, all at 100 and 300); the median step's push stayed between 0.04 and 2.7 percent of |ε_PoE|. Plan 18 called the renders: two separate animals in 0 of 16 pushed tiles, and the push toward dog turned seeds 10 to 12 into one cat. On plan 18's paper-form dog sheet the cap would have fired on every render at 100 and 300 and none at 10 or 30, as the comment on lines 43 to 45 says. Numbers in `pictures/run_numbers.json`.

That product log kept the gradient's length per step and the median and largest push over |ε_PoE| per render, so the page rebuilds the push per step as strength × σ_t × |∇|, with σ_t from SDXL's scheduler config. The same rebuild reproduces the paper-form log's own push lengths within 0.05 percent at every step once σ_t is taken in float16, as line 236 takes it.

**What the code shows that the paper's equations do not**

The product combine (line 130), the cap (lines 46 to 61), the window (lines 133 and 134) and the canary (line 136) have no line in the authors' code. The push is subtracted from ε_PoE directly (line 146); the authors' `condition_score` first re-derives ε from a clean guess that may be clipped (`gaussian_diffusion.py` line 383). The strength enters at line 145; the authors multiply inside `cond_fn` (`classifier_sample.py` line 61). The paper-form entry point reads ᾱ_t in float16 (line 236), which makes the last step's σ_t 0.0383 against 0.0413 in float32, 7.3 percent smaller for the same gradient, while the other 49 steps differ by under 0.3 percent; the product sampler takes it in float32 (line 142).

**How we got it**

`miniature_check.py` runs on CPU with `/home/molef/PhD/replications/fkc-diffusion/.venv/bin/python` (torch 2.15), log in `logs/miniature-check.log`; it also traces which lines of the sampler file ran (136, 142 to 146, and 55 to 60 in `capped_push`), so a breakpoint on those lines stops on the miniature in VS Code. `seeded_run_pictures.py` draws the three run pictures from the two logs (`logs/seeded-run-pictures.log`), and `equation_pictures.py` draws the five term figures on the miniature with `~/.claude/figkit/termfig.py` (`logs/equation-pictures.log`).

`gen_codewalk.py` reads both files byte for byte, asserts 75 cited lines against their text (including the authors' lines in the guided-diffusion clone at `22e0df8`), takes every caption number from the three JSON files, and writes `our-guided-sampler-code.json`, `codewalk.json` and the role-coloured SVG (from `our-guided-sampler-high-level-design-explainer.json`, drawn by the hld-diagram skill's `draw_hld.py`, overlap check clean in light and dark). The page is built with `python3 -I ~/.claude/skills/hld-diagram/scripts/build_codewalk.py codewalk.json our-guided-sampler-codewalk.html --strict`, which exits 0 with 11 of 11 boxes explained and 30 sections over 308 lines in 2 files (`logs/build-codewalk.log`). `python3 -I ~/.claude/skills/hld-diagram/scripts/probe_codewalk.py our-guided-sampler-codewalk.html --dark` prints `ok`: each box opens its own section with its lines in view, no maths left as raw TeX, no symbol printed twice, every lead figure under its maths (`logs/probe.log`).

Colours follow the ladder's concordance: grey `#7A8699` the ordinary SDXL sampler, orange `#c86a0a` the push, magenta `#C2185B` the strength, and teal `#14a38b` for what is ours alone, a fourth colour the authors' walk does not need.

**Reviewed by Codex**

A read-only Codex review (job `20261117-123810-8f2f15`) checked the line numbers, the maths, the roles, the numbers against the three JSON files and the prose; it found the miniature results and the cap counts consistent with their files. Each finding was checked against the files before it was applied. Applied:

- The product combine is described as the score of p(x | cat)·p(x | dog) / p(x) at γ = 1, with the guided factors at γ = 7.5.
- The cap is written as two cases, so a zero push is returned unchanged as the code does, with its limit L named.
- The canary section separates strength 0 (the whole render is the plain product) from a step outside the window (that step goes unpushed while the latent keeps what earlier steps did); the diagram's bypass arrow reads "push off".
- The paper form's base is ordinary guidance on its prompt slot, the unconditional prediction only with the empty prompt in both slots.
- The first-step gradients agree to three decimals across strengths, and 6 of 16 product renders peak after the first step; the largest rebuilt push, 223, is at the first step of seed 10 at strength 300.
- The float16 σ_t at the last step is 7.3 percent below float32, and under 0.3 percent apart at the other 49 steps; σ_t shrinking does not by itself shrink the push.
- The logs section says line 131 records |ε_PoE| on every step and the other four lists get zeros when the push does not run.
- `metrics.py` ends at line 46; lines 174 and 175 restore the head's training mode; the paper form's setup and closing sections name their steering lines and our logs apart from the ordinary sampler.
- The push panels are orange and the capped push teal, keeping magenta for the strength alone; each vector figure's caption says where its arrows start, what its axes are, and where a panel is zoomed.
- The debugger note names the workspace, the interpreter and which lines the miniature reaches.

Not applied:

- A larger overview, an "Expand diagram" control, and a closed box for provenance: the page layout is set by `build_codewalk.py` under `~/.claude`, which this work did not edit.
- Independent checks of the cluster-only claims (route step 5's canary, plan 21's 2 to 6 percent cap firing): their files are on `/datasets`, so the page quotes them with their source rather than drawing them.

**What it lets you say**

Which line of our sampler computes which part of Algorithm 2 on the product base, which lines are ours alone, and, on the miniature, that the file's own push lines land on ε_true at w = 1 with the cap off while the default cap of 0.1 holds the push to a tenth of |ε_PoE|.

**What it cannot say**

How often the cap fires on real renders of the capped sampler: those logs (route step 5's canary, plan 21's timing run, where the cap fired on 2 to 6 percent of steps at strength 30) are on `/datasets`, which this machine cannot read, so the page quotes them from plan 21 and the report's board and draws only the uncapped run of 2026-11-05. The window was never set in a logged run. W&B run `z2k1bdmi` (dog-01) holds the classifier's training curves, not the sampler's steps. The SDXL run cannot be stepped on the laptop; only the miniature can.

**Where it came from**

The sampler and `metrics.py` at `5d753591`, the two callers in `2-classifier-guidance/scripts/`, the 2026-11-05 logs beside this folder, plans 15, 18 and 21, the [report's board](../../../report/classifier-guided-poe/board.md), the guided-diffusion clone at `22e0df8`, and the journey's notation and miniature answer key.

**Held since 2026-11-17.**

The scene-library section in `Scenes.md` (with `scene-cards.py --apply` and `--refresh-marks`) is owed for this page, as it is for the authors' walk. The page is in the HLD register (`register_hld.py --add`, published version 1); the HLD Gallery republish is held for the same reason as the authors' walk: the rebuilt gallery lists 24 cards but drops "Product validation evolution", which the live page carries.

### Versions

- `our-guided-sampler-high-level-design-explainer__2026-11-17.svg`: version 1, eleven boxes and twelve arrows, the SDXL step, our checks and the push in three rows, published as Version 1.

<!-- cluster:start -->

## On the cluster

The other 24 files of this result (9.5 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-our-guided-sampler-computes/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 12 | 8.9 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-our-guided-sampler-computes/` |
| `logs` | 5 | 0.0 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-our-guided-sampler-computes/logs/` |
| `pictures` | 7 | 0.7 MB | [run_push_vs_cap.png](pictures/run_push_vs_cap.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-our-guided-sampler-computes/pictures/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/what-each-line-of-our-guided-sampler-computes/<run>/ <run>/
```

<!-- cluster:end -->
