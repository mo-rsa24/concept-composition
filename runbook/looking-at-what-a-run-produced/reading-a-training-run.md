# 🧭 Runbook: reading an adapter training run

What gets recorded while a pooled adapter trains, where each number lands, and how to read
it. The recorded things come in two kinds: **per-step curves** (logged to W&B on every
optimizer step) and the tracking set, the fixed list of images and curves saved at every
10k-step checkpoint. The tracking set is frozen before a run launches so that every
checkpoint of every run is comparable; its contents are decided in
[the showcase scope's decision ledger](../../1-lora-correction/plans/01-showcase-the-trained-lora/decisions-taken-here.md).

W&B project: [prime_lab/poe-repair-animals-compose](https://wandb.ai/prime_lab/poe-repair-animals-compose).
Run folders: `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/pooled_lora/<run_id>/`
(config, checkpoints, samples, history).

The numbered sections below are reference tables, so they carry a status line and no recipe mark.

## Table of contents

- [1. The per-step curves, and what each one means](#1-the-per-step-curves-and-what-each-one-means)
- [2. The tracking set, saved at every 10k-step checkpoint](#2-the-tracking-set-saved-at-every-10k-step-checkpoint)
- [3. Filling in the screenshots](#3-filling-in-the-screenshots)
- [Cross-references](#cross-references)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)

## 1. The per-step curves, and what each one means

Navigation: 📋 [TOC](#table-of-contents) | [Next](#2-the-tracking-set-saved-at-every-10k-step-checkpoint) ➡️

Status: unverified (transcribed from `phase1_r8_100k`'s logged history; upgrade via
`--capture` on the next live run).

| Curve (W&B key) | What it is | How to read it |
|---|---|---|
| `train/loss` | How far the adapter's correction is from the cached true correction on the pairs and seeds it trains on | Falling means still learning. Flat and noisy (as `phase1_r8_100k` was, around 1e-4) means more steps buy nothing at this capacity. |
| `train/loss_bucket/early`, `/commit`, `/late` | The same loss split by where in the 50 sampling steps the training example came from: before step 5, steps 5 to 25 (where the image picks its layout), after step 25 | If `commit` stays high while the others fall, the adapter is failing exactly where composition is decided. |
| `train/delta_target_norm` vs `train/delta_hat_norm` | The size of the true correction beside the size of the learned one | Learned much smaller than true means the adapter under-corrects. The learned size is known to stop rising once it covers about 40% of the true one. |
| `train/grad_norm`, `train/lr` | Training health | Spikes in `grad_norm` with no loss change usually mean one bad training example rather than a broken run. |

## 2. The tracking set, saved at every 10k-step checkpoint

Navigation: ⬅️ [Previous](#1-the-per-step-curves-and-what-each-one-means) | 📋 [TOC](#table-of-contents) | [Next](#3-filling-in-the-screenshots) ➡️

Status: unverified until the extra recording is wired in; items marked *new* are adopted but
not yet wired.

| Item | Where it lands | How to read it |
|---|---|---|
| Comparison strips: joint-prompt render, plain PoE, adapter, one row per pair and seed | `samples/per_epoch/epoch_*/`, and W&B panels | The adapter column should drift from PoE-like toward the joint-prompt column over checkpoints. Left two columns never change; only the third evolves. |
| The repair example (cat and dog, seed 1) | same folders | The one pair and seed where the joint prompt itself fails. Watch for the step where the dog first appears. |
| Compose rate over the held-out pool | `compose_rate.json`, W&B | Fraction of held-out renders where the scorer counts two animals. The headline transfer number. |
| `direction_cosine` | W&B | Does the learned correction point the same way as the pool-average true correction: 1 same direction, 0 unrelated. |
| `frac_distance_reached` | W&B | How much of the true correction's length the learned one covers; it is known to stop rising at about 0.4. |
| *new* Learned-vs-true [cosine](../../../../goal-setting/learning/deep-learning/diffusion-models/08-spectral-structure-of-the-correction/plans/01-dot-product-norm-cosine.md) at sampling steps 7, 15, 22 | to be wired | Same idea as `direction_cosine` but against this example's own true correction at three fixed sampling steps. Shows which part of the trajectory is learned first. |
| *new* Embedding drift in the two scoring spaces (DINOv2, CLIP) | to be wired | Distance of the render to the joint-prompt image minus distance to the PoE image. Below zero means the render now sits closer to the target than to the failure. |
| *new* Spectral share of learned corrections | to be wired | Whether the learned corrections across training examples concentrate in a few shared directions. Diagnostic only: the paper's low-rank claim is made on true corrections, never on these. |
| *new* Correction size across the 50 sampling steps | to be wired | Where in the sampling run the adapter acts. It should act before the layout is decided (steps 18 to 36 is where samples commit; the best injection window is steps 0 to 10). |

> Held-out means the pairs the adapter never saw during training, kept aside so the compose rate
> measures transfer to new pairs rather than memorisation of the training ones.

> Low-rank means the corrections, seen as a big pile of vectors, mostly lie in a handful of
> shared directions rather than pointing every which way.

## 3. Filling in the screenshots

Navigation: ⬅️ [Previous](#2-the-tracking-set-saved-at-every-10k-step-checkpoint) | 📋 [TOC](#table-of-contents) | [Next](#cross-references) ➡️

No screenshots yet: each table row above wants one captured W&B panel with a caption
saying what the healthy shape looks like.

They get captured while the showcase experiments run: the session capturing a run also
captures its W&B panels and files each one here, beside its table row, with a caption saying
what the healthy shape looks like. That needs the two connectors below, which only a human
can authorize (a non-interactive session cannot run the OAuth or install step).

**Both connectors are registered at user scope on this machine and connect** (check any
time with `claude mcp list`; they load at session start, so a session started before the
registration will not see them). The exact registered commands, should either ever need
re-creating:

```bash
# W&B MCP: serves runs, histories and panels as tools. Auth comes from ~/.netrc,
# no key in the config. The git+pin form is required: the package is not on PyPI,
# and its HEAD needs mcp pinned below 2.
claude mcp add --scope user wandb -- /home-mscluster/mmolefe/.local/bin/uvx \
    --with 'mcp<2' --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server

# Playwright MCP: drives a browser and takes screenshots. npx lives in the
# miniforge base, not on PATH; chromium 151 is installed under
# ~/.cache/ms-playwright and launches on this node.
claude mcp add --scope user playwright -- /home-mscluster/mmolefe/miniforge3/bin/npx \
    -y @playwright/mcp@latest
```

One caveat before relying on Playwright for W&B pages: wandb.ai panels need a logged-in
browser, and the Playwright MCP uses its own profile, so either log in once in headed mode or
let the W&B MCP pull the panel data and render the figure locally.

- **Generate the same figures locally today**, no browser and no new connectors, from the
  run history:

  ```bash
  PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
  $PY - << 'PYEOF'
  import wandb, matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  run = wandb.Api().run("prime_lab/poe-repair-animals-compose/<run_id>")
  df = run.history(keys=["train/loss", "train/delta_target_norm",
                         "train/delta_hat_norm"], samples=2000)
  ax = df.plot(x="_step", logy=True)
  ax.set_xlabel("optimizer step")
  plt.savefig("run_curves.png", dpi=150)
  PYEOF
  ```

  Save the output beside the run's other evidence and caption it from the tables above.

## Cross-references

Navigation: ⬅️ [Previous](#3-filling-in-the-screenshots) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

- The mention of **fraction-of-distance-reached**, stopping at 20% where earlier runs reached 40%, in [the failure catalog entry poe-lora-001](../../environment/known-failures.md#entry-id-poe-lora-001).
- The mention of **comparison strips**, the three-panel files the trainer saves beside each render that the compose-rate count scored as renders, in [the failure catalog entry poe-score-003](../../environment/known-failures.md#entry-id-poe-score-003).

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#cross-references) | 📋 [TOC](#table-of-contents)

The tables above name each logged curve. This block shows what two real loss-curve figures look like, the lines of code that write the curves, and where the run's saved files sit.

### The code behind it

The trainer logs the curves named in section 1 on every optimizer step. Excerpt from `poe_repair/experiments/cross_pair_lora_pooling/multi_pair_trainer.py` (lines 148 to 157):

```python
                "train/loss_bucket/early": state.bucket_loss_running["early"],
                "train/loss_bucket/commit": state.bucket_loss_running["commit"],
                "train/loss_bucket/late": state.bucket_loss_running["late"],
                "train/grad_norm": grad_norm_pre,
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/batch_size": int(info["batch_size"]),
                "train/epoch": state.epoch,
                "train/optimizer_step": state.optimizer_step,
                "train/delta_target_norm": info["delta_target_norm"],
                "train/delta_hat_norm": info["delta_hat_norm"],
```

- The three `loss_bucket` keys are the early, commit and late split of the section 1 table. Each step's whole-batch loss goes to the one bucket most of the batch falls in, as a running average (lines 122 to 133).
- `delta_target_norm` and `delta_hat_norm` are the pair the table reads as "true size beside learned size".

The logged buckets split by sampling step through `cfg.probe.commit_window`, which is `[5, 25]` in the `config.json` of `phase1_r8_100k`. `poe_repair/experiments/cross_pair_lora_pooling/_inline_sampling.py` keeps the same split as its standalone defaults (lines 171 to 172):

```python
TRACKING_BUCKETS_DEFAULT = {"early": (0, 5), "commit": (5, 25), "late": (25, 49)}
TRACKING_REP_STEPS_DEFAULT = (7, 15, 22)
```

- `(5, 25)` is the commit window the page's table describes, and `(7, 15, 22)` are the three sampling steps the *new* learned-versus-true cosine row names.

### What comes out

The run folder named at the top holds `config.json`, `history.json`, `compose_rate.json` and `verdict.json` for `phase1_r8_100k` `verified` (listed in the folder). Its `verdict.json` reads `"verdict": "ok"` at `optimizer_step` 100000 and epoch 2000 `verified` (opened). No W&B panel screenshot of this run is recorded, which is what section 3 says.

### Further illustrations

Two real loss figures from other trainings, one of three adapters and one of a twist head, to show what the per-bucket curves look like. Neither is the pooled run above.

**Evidence**

<a href="../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/superdiff-lora-loss-curves.png"><img src="../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/superdiff-lora-loss-curves.png" width="200" alt="Four panels of training loss against optimizer step for three adapter ranks"></a>

Training loss (mean squared error, log axis) against optimizer step for the three SuperDiff-residual adapters of rank 8, 16 and 32, in four panels: total, and three buckets of SuperDiff step (0 to 19, 20 to 99, 100 to 199) `verified` (read from the panel titles). Rank 8 sits lowest late in the total and commit panels, and the three ranks lie close together in the early panel `verified` (by eye). Rank 32's curve stops near step 36,000 `verified`. The card adds that all three plateau from 30k to 40k, from W&B runs `t5b00h56`, `5oz1bsa1`, `ue205g6e` `stated`. The card also says rank 8 is lowest everywhere, while in the early panel rank 8 is at times above the other two, so the two differ. The buckets here are 200-step SuperDiff ranges and differ from the 50-step early, commit and late split above.

<a href="../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/twist-training-curves.png"><img src="../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/twist-training-curves.png" width="200" alt="Twist head accuracy and loss by timestep bucket against optimizer step"></a>

A different trainer, the twisted-SMC twist head. Left: training accuracy sits near 1.0 while validation accuracy on 16 held-out cells ends at 0.56, under the 0.60 bar `verified` (printed on the panel). Right: loss per timestep bucket on a log axis with chance at 0.69 `verified`. The card says the loss reaches zero in every bucket, and the right panel's title says it does so even at high noise `stated`. In the curves the high-noise bucket still swings from the floor up to the chance line and past it through the run `verified`, and the figure's sidecar file records 0.034 for that bucket at step 100,000 `stated`. These sources differ, and neither is changed here. The buckets are noise levels (timestep), unlike the sampling-step buckets of this page.

### Go deeper

| What | Link |
|---|---|
| The results card holding both figures | [the sampler-or-model results card](../../1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/README.md) |
| The decision ledger that froze the tracking set | [the showcase scope's decision ledger](../../1-lora-correction/plans/01-showcase-the-trained-lora/decisions-taken-here.md) |
| Failure catalog entries the cross-references name | [fraction of distance reached](../../environment/known-failures.md#entry-id-poe-lora-001) |

### Still open

No W&B panel of the pooled run is captured yet (section 3). No loss-curve figure of `phase1_r8_100k` itself is filed under the artifacts.
<!-- related:end -->

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
