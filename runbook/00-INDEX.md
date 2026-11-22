# 🧭 Runbook: concept-composition

How to do the recurring things here by hand: checking the plan tree's state, launching or
harvesting a run on the cluster, reading what a run produced, making the paper's figures, and
moving manuscript edits between this repo and Overleaf.

## Table of contents

- [Lookup](#lookup)
- [Themes](#themes)
- [Still open](#still-open)

## Lookup

Navigation: 📋 [TOC](#table-of-contents) | [Next](#themes) ➡️

| I want to... | Go to |
|---|---|
| See whether the plan tree has drifted (stale run claims, unharvested output, orphaned files, unjudged reviews) | [checking-the-plan-tree.md §1](checking-the-plan-tree.md#1-run-the-state-check) |
| Decide where a run goes: `sbatch`, or an SSH launch the owner has approved | [launching-and-harvesting-a-run.md §1](running-things-on-the-cluster/launching-and-harvesting-a-run.md#1-decide-where-a-run-goes) |
| Launch something on a shared half-used `biggpu` node, once the owner has said yes | [launching-and-harvesting-a-run.md §2](running-things-on-the-cluster/launching-and-harvesting-a-run.md#2-launch-on-a-shared-device) |
| Find every run in flight, including the ones Slurm can't see | [launching-and-harvesting-a-run.md §3](running-things-on-the-cluster/launching-and-harvesting-a-run.md#3-harvest-what-is-running) |
| Serve a static scene from `artifacts/scenes/` in a tmux session | [showing-a-scene-in-the-local-browser.md §1](looking-at-what-a-run-produced/showing-a-scene-in-the-local-browser.md#1-serve-the-scenes-folder-from-a-tmux-session) |
| See a scene served on the node in my local browser (the SSH forward) | [showing-a-scene-in-the-local-browser.md §2](looking-at-what-a-run-produced/showing-a-scene-in-the-local-browser.md#2-forward-the-port-and-open-it-on-the-laptop) |
| Know what a training run records and how to read each curve and image | [reading-a-training-run.md §1](looking-at-what-a-run-produced/reading-a-training-run.md#1-the-per-step-curves-and-what-each-one-means) |
| See what gets saved at every checkpoint (the tracking set) | [reading-a-training-run.md §2](looking-at-what-a-run-produced/reading-a-training-run.md#2-the-tracking-set-saved-at-every-10k-step-checkpoint) |
| Build the SuperDiff trajectory cache the adapters train on | [training-and-watching-a-superdiff-adapter.md §1](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#1-build-the-superdiff-trajectory-cache---ran-2026-09-03) |
| Train an adapter on SuperDiff's own residual (or resume one) | [training-and-watching-a-superdiff-adapter.md §2](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#2-launch-one-training---ran-2026-09-03) |
| See what a training adapter does inside SuperDiff, every 10k steps, without stopping it | [training-and-watching-a-superdiff-adapter.md §3](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#3-render-a-strip-from-every-checkpoint-while-it-trains---ran-2026-09-03) |
| Put every checkpoint strip on one sheet per rank and pair | [training-and-watching-a-superdiff-adapter.md §4](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#4-assemble-the-timelines---ran-2026-09-05) |
| Sweep λ on one checkpoint of a SuperDiff adapter | [training-and-watching-a-superdiff-adapter.md §5](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#5-sweep-λ-on-one-checkpoint---ran-2026-09-03) |
| Render SuperDiff alone across κ and λ (the seeds-by-λ sheets) | [training-and-watching-a-superdiff-adapter.md §6](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#6-render-superdiffs-own-kappa-by-lambda-sheets---ran-2026-09-03) |
| Inject a PoE-trained adapter into SuperDiff and measure which way its correction points | [training-and-watching-a-superdiff-adapter.md §7](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#7-inject-a-poe-trained-adapter-into-superdiff-and-measure-its-direction---ran-2026-09-03) |
| Stop a `nohup` run on a node without killing my own shell | [training-and-watching-a-superdiff-adapter.md §8](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#8-stop-a-run-cleanly---ran-2026-09-04) |
| Get the authors' Reduce, Reuse, Recycle code running from an empty folder (clone, 2D environment, notebook fixes) | [setting-up-the-reduce-reuse-recycle-code.md §1](setting-up-the-reduce-reuse-recycle-code.md#1-clone-at-the-commit-the-walk-uses---ran-2026-10-13) |
| Download the Reduce, Reuse, Recycle CLEVR checkpoint past Google Drive's virus-scan page | [setting-up-the-reduce-reuse-recycle-code.md §4](setting-up-the-reduce-reuse-recycle-code.md#4-get-the-clevr-checkpoint---ran-2026-10-13) |
| Build the Reduce, Reuse, Recycle 2D notebook's CUDA jax environment on a cluster GPU node | [setting-up-the-reduce-reuse-recycle-code.md §7](setting-up-the-reduce-reuse-recycle-code.md#7-build-the-2d-notebooks-environment-on-the-cluster---ran-2026-10-13) |
| Render the 8 held-out cat x dog seeds from one pooled-LoRA checkpoint, with the scorer's verdict on every tile | [probing-a-pooled-lora-checkpoint.md §1](running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#1-render-the-8-seed-held-out-grid-for-one-checkpoint---ran-2026-09-03) |
| Measure how well a checkpoint fits the cached correction, per pair and step bucket, with no sampling | [probing-a-pooled-lora-checkpoint.md §2](running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#2-measure-how-well-a-checkpoint-fits-the-cached-correction---ran-2026-09-03) |
| Resume a pooled (PoE) training from a checkpoint on a free device, once the owner has approved an SSH launch | [probing-a-pooled-lora-checkpoint.md §3](running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#3-resume-a-pooled-training-from-a-checkpoint-on-a-free-device---ran-2026-09-01) |
| Train two ranks at once on an idle node with one Slurm job | [probing-a-pooled-lora-checkpoint.md §4](running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#4-train-two-ranks-on-one-idle-node-with-one-slurm-job---ran-2026-09-01) |
| Estimate how long a training run will take before launching it | [how long a training run takes](../environment/hpc/throughput.md#projected-durations-for-experiments-a-and-b) |
| Prove the Langevin corrector is off when it is switched off | [running-the-langevin-corrector.md §1](running-things-on-the-cluster/running-the-langevin-corrector.md#1-prove-the-corrector-is-off-when-it-is-switched-off) |
| Fix the corrector's step size before reading any curve | [running-the-langevin-corrector.md §2](running-things-on-the-cluster/running-the-langevin-corrector.md#2-fix-the-step-size-before-reading-any-curve) |
| Run the corrector-count grid and print its three-way verdict | [running-the-langevin-corrector.md §3](running-things-on-the-cluster/running-the-langevin-corrector.md#3-run-the-corrector-count-grid-and-print-its-verdict) |
| Slide a corrector window across the run, or render the eight-seed sheets and the adapter tails | [running-the-langevin-corrector.md §4](running-things-on-the-cluster/running-the-langevin-corrector.md#4-slide-a-corrector-window-across-the-run), then [§5](running-things-on-the-cluster/running-the-langevin-corrector.md#5-render-the-eight-seed-sheets-the-adapter-tail-and-the-clean-tail) |
| Run a Langevin corrector on the product-of-experts score, fix its step size, and read what survives it | [running-the-langevin-corrector.md §1](running-things-on-the-cluster/running-the-langevin-corrector.md#1-prove-the-corrector-is-off-when-it-is-switched-off) to [§3](running-things-on-the-cluster/running-the-langevin-corrector.md#3-run-the-corrector-count-grid-and-print-its-verdict) |
| Slide a corrector window across the run, or render the eight-seed corrector sheets and the adapter tails | [running-the-langevin-corrector.md §4](running-things-on-the-cluster/running-the-langevin-corrector.md#4-slide-a-corrector-window-across-the-run) to [§6](running-things-on-the-cluster/running-the-langevin-corrector.md#6-draw-the-figures-and-log-the-set-to-wb) |
| Re-render the joint prompt, each concept alone, PoE and the corrected run on the held-out seeds, with per-step frames | [reproducing-where-each-condition-lands.md §1](running-things-on-the-cluster/reproducing-where-each-condition-lands.md#1-render-the-references-and-the-per-step-frames-on-a-shared-device) |
| Redraw the endpoint figures and the sidecar behind the landing finding | [reproducing-where-each-condition-lands.md §2](running-things-on-the-cluster/reproducing-where-each-condition-lands.md#2-make-the-endpoint-figures-and-their-sidecar) |
| Rebuild the animated "where each condition lands" page (Diffusion Explorer's UI over our frames) and open it | [reproducing-where-each-condition-lands.md §3](running-things-on-the-cluster/reproducing-where-each-condition-lands.md#3-embed-the-frames-and-export-the-pages-data), then [§4](running-things-on-the-cluster/reproducing-where-each-condition-lands.md#4-build-the-page-from-diffusion-explorers-ui-and-open-it) |
| Show that page on a laptop with no cluster tunnel | [reproducing-where-each-condition-lands.md §5](running-things-on-the-cluster/reproducing-where-each-condition-lands.md#5-open-the-built-page-on-a-laptop-with-no-cluster-tunnel) |
| Build the two-panel correction-window figure the paper uses | [making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md (opening)](making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md#read-this-before-you-start) |
| Draw a grid panel at the exact width it will occupy on the page | [making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md §1](making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md#1-make-the-panels) |
| Put two panels side by side under one caption in Inkscape | [making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md §3](making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md#3-set-the-page-size-first-before-anything-else) to [§7](making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md#7-tidy-up-before-exporting) |
| Export an Inkscape figure as a PDF that renders correctly in the ICLR template | [making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md §8](making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md#8-export-the-pdf) |
| Place a finished figure in the paper without ruining its label sizes | [making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md §9](making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md#9-put-it-in-the-paper) |
| Re-letter a figure whose labels are baked into the image, at its printed size | [making-the-papers-figures/relettering-an-image-with-baked-in-text.md](making-the-papers-figures/relettering-an-image-with-baked-in-text.md#1-set-the-page-to-the-printed-width) |
| Compose two panels with LaTeX and skip Inkscape entirely | [making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md §10](making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md#10-the-latex-only-route-if-you-skip-inkscape) |
| Get what I typed in the Overleaf editor into the repo | [syncing-the-paper-with-overleaf.md §1](syncing-the-paper-with-overleaf.md#1-pull-your-overleaf-edits-into-the-repo---ran-2026-09-21) |
| See what an `Update on Overleaf.` commit actually changed | [syncing-the-paper-with-overleaf.md §2](syncing-the-paper-with-overleaf.md#2-read-what-came-down-before-you-commit-it---ran-2026-09-21) |
| Push a repo-side edit up to Overleaf without deleting files from the project | [syncing-the-paper-with-overleaf.md §3](syncing-the-paper-with-overleaf.md#3-send-a-repo-side-edit-back-up-to-overleaf) |
| Rebuild the zip that starts a fresh Overleaf project | [syncing-the-paper-with-overleaf.md §4](syncing-the-paper-with-overleaf.md#4-rebuild-the-upload-zip-instead---unverified) |
| Remember which of the three copies of the paper is which | [syncing-the-paper-with-overleaf.md (where the paper lives)](syncing-the-paper-with-overleaf.md#where-the-paper-lives) |

## Themes

Navigation: ⬅️ [Previous](#lookup) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Grouped by the kind of work. The group is a listing convenience; every row above still points
straight at the recipe that answers it.

**Running things on the cluster**

| File | One place, one purpose | Recipes |
|---|---|---|
| [launching-and-harvesting-a-run.md](running-things-on-the-cluster/launching-and-harvesting-a-run.md) | Submit through `sbatch`, launch over SSH only with the owner's yes, and find what is running | 3 |
| [reproducing-where-each-condition-lands.md](running-things-on-the-cluster/reproducing-where-each-condition-lands.md) | The landing figures and their animated page, from renders to `dist/` | 5 (4 `ran`, 1 `unverified`) |
| [running-the-langevin-corrector.md](running-things-on-the-cluster/running-the-langevin-corrector.md) | The Langevin corrector of scope 06, on a `biggpu` device or the session node: the two leak checks, the step-size search, the corrector-count grid and its verdict, the sliding window, the eight-seed sheets and the two adapter tails, the figures and the W&B log | 6 (all `ran`) |
| [probing-a-pooled-lora-checkpoint.md](running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md) | A `biggpu` device or one Slurm job: render a pooled checkpoint's held-out grid, read its fit on the cache, resume or launch a pooled training | 4 (all `ran`) |
| [training-and-watching-a-superdiff-adapter.md](running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md) | SuperDiff on the cluster: its cache, an adapter trained on its residual, strips from every checkpoint, λ sweeps, its own κ × λ sheets, a PoE adapter injected into it | 8 (all `ran`) |

**Looking at what a run produced**

| File | One place, one purpose | Recipes |
|---|---|---|
| [reading-a-training-run.md](looking-at-what-a-run-produced/reading-a-training-run.md) | W&B: the curves and the tracking set a training run writes | 3 reference sections |
| [showing-a-scene-in-the-local-browser.md](looking-at-what-a-run-produced/showing-a-scene-in-the-local-browser.md) | A static server on the node and the tunnel from the laptop | 2 |

**Making the paper's figures**

| File | One place, one purpose | Recipes |
|---|---|---|
| [composing-a-two-panel-figure-in-inkscape.md](making-the-papers-figures/composing-a-two-panel-figure-in-inkscape.md) | A laptop with Inkscape: draw panels at final size, compose them under one caption, export a PDF the ICLR template places at 1:1 | 10 (all `unverified`) |
| [relettering-an-image-with-baked-in-text.md](making-the-papers-figures/relettering-an-image-with-baked-in-text.md) | A laptop with Inkscape: replace labels baked into an image with real type at its printed size | 6 (all `unverified`) |

**Flat, no group yet**

| File | One place, one purpose | Recipes |
|---|---|---|
| [checking-the-plan-tree.md](checking-the-plan-tree.md) | The repo on the session node: the plan-tree state check | 1 |
| [syncing-the-paper-with-overleaf.md](syncing-the-paper-with-overleaf.md) | The repo and the Overleaf clone beside it: moving manuscript edits between the two, and the zip route | 4 (2 `ran`, 2 `unverified`) |
| [setting-up-the-reduce-reuse-recycle-code.md](setting-up-the-reduce-reuse-recycle-code.md) | The laptop and a cluster GPU node: clone the authors' code, build its environments, fetch the CLEVR checkpoint | 7 (6 `ran`, 1 `unverified`) |

Two declared exceptions apply across this folder. Recorded commands keep placeholders such as
`<node>` because a recorded command does not change. On `launching-and-harvesting-a-run.md`,
`checking-the-plan-tree.md`, `showing-a-scene-in-the-local-browser.md`,
`running-the-langevin-corrector.md`, `reproducing-where-each-condition-lands.md`,
`composing-a-two-panel-figure-in-inkscape.md` and `relettering-an-image-with-baked-in-text.md`, the
freshness mark sits on the line under the heading, which keeps every section link stable.

## Still open

Navigation: ⬅️ [Previous](#themes) | 📋 [TOC](#table-of-contents)

- **No diagram prompts exist for this folder.**
- **The screenshot places in `reading-a-training-run.md` are empty.** They want captured W&B
  panels, or the local wandb-API figures in that file's section 3.
- **The three recipes in `launching-and-harvesting-a-run.md` are `unverified`.** Section 1 was
  written on 2026-11-22 and sections 2 and 3 transcribed on 2026-08-24, all from
  `environment/hpc/execution-protocol.md`, and the next real launch or harvest
  should upgrade them via `--capture`.
- **Which Inkscape recipe has been run is not recorded.** An earlier count named 1 of the 10 as
  run without saying which, so all 10 carry `unverified` until that is known.
- **12 figure recipes state no success check.** They are all 10 Inkscape recipes and re-lettering
  recipes 3 and 5, so nothing on the page says what a correct result looks like. Each gets a
  success line the first time someone walks it, and until then stays `unverified`.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
