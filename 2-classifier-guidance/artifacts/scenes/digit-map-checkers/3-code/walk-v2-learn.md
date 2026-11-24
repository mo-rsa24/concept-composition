# The code behind the two checkers, block by block

This walk reads the code that builds, trains and scores the count checker and the identity checker, in the order one training update runs, then the code that saves the count checker, pushes the sampler with its gradient, and records what both do. Each block quotes the lines, says what they compute in one equation, and links the figure that draws it. Every path is from the repository root, and the line numbers hold at commit `cbff8c0e`. Blocks 1 to 9 quote lines that are the same in `6c3f8a16`, the code snapshot the split run (job 94085) ran from.

## Table of contents

- [1. The noise schedule and the sixteen timesteps](#1-the-noise-schedule-and-the-sixteen-timesteps)
- [2. Noising a canvas](#2-noising-a-canvas)
- [3. The two maps](#3-the-two-maps)
- [4. The time features](#4-the-time-features)
- [5. The network](#5-the-network)
- [6. Swapping the first layer to two channels](#6-swapping-the-first-layer-to-two-channels)
- [7. One training update](#7-one-training-update)
- [8. The two label sets](#8-the-two-label-sets)
- [9. The split run](#9-the-split-run)
- [10. Saving the count checker and its real maps](#10-saving-the-count-checker-and-its-real-maps)
- [11. The push](#11-the-push)
- [12. Recording every layer and every sampler step](#12-recording-every-layer-and-every-sampler-step)

## 1. The noise schedule and the sixteen timesteps

`2-classifier-guidance/scripts/mnist/canvas_unet.py`, lines 35 to 38:

```python
NULL = 10
T = 1000
BETAS = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
ABAR = torch.cumprod(1 - BETAS, 0).float()
```

$\bar\alpha_t = \prod_{s \le t} (1 - \beta_s)$, the share of the clean canvas's variance left at timestep $t$. Label 10 is the "no digit" class the U-Net was trained with.

`2-classifier-guidance/scripts/abstract_checkers/v2_learn.py`, line 46:

```python
SDXL_MATCHED = torch.tensor([712, 696, 680, 664, 648, 632, 616, 601, 585, 570, 555, 539, 524, 510, 495, 480])
```

These 16 timesteps give the toy the same $\bar\alpha$ as SDXL's first 16 sampler steps, 0.0058 to 0.0949. The split run uses them through `--window sdxl` (`v2.sbatch` line 56). Figure: [the canvas and its noise](../0.5-network-view/dc1-the-canvas-and-its-noise.png).

## 2. Noising a canvas

`v2_learn.py`, lines 101 to 107:

```python
def inputs(cfg, model, x0, t, pair, noise):
    a = ABAR.to(x0.device)[t][:, None, None, None]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
    if cfg["reader"] == "raw":
        return xt
    with torch.no_grad():
        return maps(model, xt, t, pair, cfg["input"])
```

$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$. The raw-canvas reader gets $x_t$ itself; the map reader gets the two maps, computed under `torch.no_grad`, so training never sends a gradient into the U-Net. `x0` comes from `to_x` (`scripts/mnist/pixel_checker.py` line 152), which maps pixel 0 to −1 and 255 to +1.

## 3. The two maps

`v2_learn.py`, lines 85 to 98:

```python
def maps(model, xt, t, pair, variant):
    """The two maps of where each asked label changes the noise prediction, each adding to 1, then scaled by the pixel count."""
    n = xt.shape[0]
    y = torch.cat([torch.full((n,), NULL, device=xt.device), pair[:, 0], pair[:, 1]])
    e0, ea, eb = model(xt.repeat(3, 1, 1, 1), t.repeat(3), y).chunk(3)
    channels, totals = [], []
    for d in (ea - e0, eb - e0):
        total = d.abs().flatten(1).sum(1).clamp_min(1e-12)[:, None, None, None]
        channels.append((d if variant == "signed" else d.abs()) / total * PIXELS)
        totals.append(total.log())
    out = torch.cat(channels, 1)
    if variant == "totals":
        out = torch.cat([out, torch.cat(totals, 1).expand(-1, -1, H, W) / 5], 1)
    return out
```

One batched call runs the U-Net three times: with no digit, with digit $a$, with digit $b$. Then, for the base variant the split run uses,

$$m_a = 2048 \cdot \frac{|\varepsilon_a - \varepsilon_\varnothing|}{\sum_{\text{pixels}} |\varepsilon_a - \varepsilon_\varnothing|}, \qquad m_b \text{ the same with } b.$$

The digit labels enter only through which calls are made, so the checker that reads `out` never sees them. Figure: [where the two maps come from](../0.5-network-view/dc2-where-the-two-maps-come-from.png).

## 4. The time features

`2-classifier-guidance/scripts/mnist/checker.py`, lines 69 to 73:

```python
def time_features(t):
    a = ABAR.to(t.device)[t]
    lsnr = torch.log(a / (1 - a))[:, None]
    f = torch.exp(torch.linspace(0, 3, 8, device=t.device))[None] * lsnr / 10
    return torch.cat([f.sin(), f.cos()], 1)
```

$f_k = e^{3k/7}\,\log\frac{\bar\alpha_t}{1-\bar\alpha_t}\,/\,10$ for $k = 0$ to $7$, and the features are $\sin f_k$ and $\cos f_k$: 16 numbers per timestep. Figure: [the timestep as sixteen numbers](../0.5-network-view/dc4-the-timestep-as-sixteen-numbers.png), whose values `numbers/make_numbers.py` computes with the same lines in numpy.

## 5. The network

`2-classifier-guidance/scripts/mnist/pixel_checker.py`, lines 33 to 56:

```python
class PixelHead(nn.Module):
    """Noisy canvas (or half) and step in; out logits. The step and the pair scale and shift the middle maps."""

    def __init__(self, out, pair_aware=False):
        super().__init__()
        self.pair_aware = pair_aware
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU())
        self.c1, self.c2 = nn.Conv2d(128, 128, 3, padding=1), nn.Conv2d(128, 128, 3, padding=1)
        self.pair = nn.Embedding(10, 64) if pair_aware else None
        self.film = nn.Linear(16 + (64 if pair_aware else 0), 256)
        self.mlp = nn.Sequential(nn.Linear(256 + 16 + (64 if pair_aware else 0), 256), nn.SiLU(), nn.Linear(256, out))

    def forward(self, x, t, pair=None):
        cond = [time_features(t)]
        if self.pair_aware:
            cond.append(self.pair(pair[:, 0]) + self.pair(pair[:, 1]))
        cond = torch.cat(cond, 1)
        g, b = self.film(cond).chunk(2, 1)
        h = F.silu(self.c1(self.stem(x)))
        h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(self.c2(h))
        return self.mlp(torch.cat([h.mean((2, 3)), h.amax((2, 3)), cond], 1))
```

The map reader builds it with `out = 1` and `pair_aware=False`, so `cond` is only the 16 time features. The shapes, counted in `numbers/numbers.json`:

| Line | Output shape | Weights |
|---|---|---|
| stem conv 1, 3×3, then SiLU | 64 × 32 × 64 | 1,216 (after the swap in block 6) |
| stem conv 2, 3×3 stride 2, then SiLU | 64 × 16 × 32 | 36,928 |
| stem conv 3, 3×3 stride 2, then SiLU | 128 × 8 × 16 | 73,856 |
| `c1`, then SiLU | 128 × 8 × 16 | 147,584 |
| `film`: 16 → 256, split into a scale $\gamma$ and a shift $\beta$ of 128 each | 256 | 4,352 |
| $h \leftarrow h\,(1+\gamma) + \beta$ | 128 × 8 × 16 | 0 |
| `c2`, then SiLU | 128 × 8 × 16 | 147,584 |
| mean and max over the 128 positions, plus the 16 time features | 272 | 0 |
| `mlp`: 272 → 256, SiLU, 256 → 1 | 1 | 70,145 |
| total | | 481,665 |

A stride-2 convolution with kernel 3 and padding 1 turns $n$ cells into $\lfloor (n + 2 - 3)/2 \rfloor + 1$: 32 → 16 → 8 down the height, 64 → 32 → 16 across. Figure: [the checker network](../0.5-network-view/dc3-the-checker-network.png).

## 6. Swapping the first layer to two channels

`v2_learn.py`, lines 77 to 82:

```python
def make_head(cfg, dev):
    if cfg["reader"] == "raw":
        return PixelHead(1, pair_aware=True).to(dev)
    head = PixelHead(1, pair_aware=False)
    head.stem[0] = nn.Conv2d(4 if cfg["input"] == "totals" else 2, 64, 3, padding=1)
    return head.to(dev)
```

The map reader's first convolution takes 2 channels, $m_a$ and $m_b$. The raw-canvas reader keeps 1 input channel and is told the asked pair through a learned embedding, which the map reader never has.

## 7. One training update

`v2_learn.py`, lines 110 to 133 (the loop body is lines 119 to 129):

```python
    for step in range(1, updates + 1):
        i = np.concatenate([rng.choice(pos, batch // 2), rng.choice(neg, batch - batch // 2)])
        x0 = to_x(split["x"][i], dev)
        pair = torch.from_numpy(split["asked"][i].astype(np.int64)).to(dev)
        y = torch.from_numpy(split["target"][i].astype(np.float32)).to(dev)
        t = STEPS16[torch.randint(0, len(STEPS16), (len(i),), generator=gen)].to(dev)
        noise = torch.randn(x0.shape, generator=gen).to(dev)
        loss = F.binary_cross_entropy_with_logits(head(inputs(cfg, model, x0, t, pair, noise), t, pair)[:, 0], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
```

Each update draws 128 label-1 canvases and 128 label-0 canvases with replacement, one of the 16 timesteps and a fresh noise draw per canvas, and takes one Adam step (learning rate 0.001, line 116) on

$$L = -\big[\,y \log\sigma(\ell) + (1-y)\log(1-\sigma(\ell))\,\big], \qquad \frac{\partial L}{\partial \ell} = \sigma(\ell) - y.$$

`loss.backward()` reaches every weight of the head and stops there, because the maps came out of `torch.no_grad`. There are 4,000 updates. Figure: [training backprop into the weights](../0.5-network-view/dc6-training-backprop-into-the-weights.png).

## 8. The two label sets

`v2_learn.py`, lines 296 to 304:

```python
def relabel(split, task):
    """count: two digits apart, so apart and wrong are label 1. identity: apart (1) against wrong (0) only."""
    if task == "count":
        out = dict(split)
        out["target"] = split["apart"].copy()
        return out
    if task == "identity":
        return take(split, np.isin(split["kind"], [KINDS.index("apart"), KINDS.index("wrong")]))
    raise ValueError(task)
```

Count keeps every canvas and labels it by layout: two digits one per half is label 1, whatever they are. Identity keeps only `apart` and `wrong` canvases and their stored label, 1 for `apart`. On the training pairs among digits 0 to 3 that is 6,000 `apart` against 1,004 `wrong`, so the balanced batches of block 7 redraw each `wrong` canvas about 510 times over 4,000 updates (counted in `numbers/numbers.json`). Figure: [two checkers, two label sets](../0.5-network-view/dc5-two-checkers-two-label-sets.png).

## 9. The split run

`v2_learn.py`, lines 307 to 334, runs for each task an overfit check on 16 fixed canvases, the 4,000-update training, and a scoring pass over the 16 timesteps with 4 fixed noise draws per canvas (`evaluate`, lines 142 to 174). The bars sit in the code:

```python
            checks = {"holding_4_ba_at_least_0.75": u["balanced_accuracy"] >= MIN_BA}
            if task == "count":
                checks["gap_at_most_0.05"] = gap <= MAX_GAP
```

`stage_split` keeps no trained head. The count checker that the push uses is trained again and saved by the `final-count` stage in block 10.

## 10. Saving the count checker and its real maps

`v2_learn.py`, lines 365 to 376 (the `final-count` stage):

```python
def stage_final_count(args, model, data, dev, data_hash):
    """Board question 69: retrain the count checker on digits 0 to 4 at the given window, save it, and read digits 5 to 9 once."""
    out = args.out / "final-count.json"
    if out.exists():
        raise SystemExit(f"{out} exists: digits 5 to 9 are read once")
    split = {s: relabel(data[s], "count") for s in ("train", "seen_test", "unseen_test")}
    result = {"task": "count", "window": args.window, "timesteps": STEPS16.tolist(), "data_hash": data_hash, "model": str(args.model)}
    for name, cfg in READERS.items():
        fit = overfit_one(cfg, model, split["train"], dev)
        head, info = train(cfg, model, split["train"], dev)
        torch.save({"head": head.state_dict(), "cfg": cfg, "task": "count", "window": args.window,
                    "timesteps": STEPS16.tolist(), "data_hash": data_hash}, args.out / f"count-head-{name}.pt")
```

The count checker trains on all 10 pairs among digits 0 to 4 with the training update of block 7 and the labels of block 8, and is saved as `count-head-map.pt` before digits 5 to 9 are scored once. Job 94108 ran it through `v2.sbatch` line 61 with `--window sdxl`. It scored 0.791 on digits 5 to 9 and 0.782 on digits 0 to 4 (`numbers/real/count-final.json`; [the V2 review](../../../../../artifacts/notes/progress/abstract-checkers/05-does-it-learn/review.md)).

Lines 384 and 385 then call `explain` on the saved map reader:

```python
        if name == "map" and args.examples:
            np.savez_compressed(args.out / "count-explained.npz", **explain(head, model, args.examples, dev))
```

`explain` (lines 341 to 362) builds the ladder's four example canvases with the ladder's own noise draw and, with gradients on, records the maps, the logit and $g = \nabla_{x_t}\ell$ (lines 353 to 357):

```python
            with torch.enable_grad():
                xt = (ab.sqrt() * x0 + (1 - ab).sqrt() * noise).requires_grad_(True)
                m = maps(model, xt, t, pair, "base")
                logit = head(m, t, pair)[:, 0]
                g, = torch.autograd.grad(logit.sum(), xt)
```

This is block 3's `maps` called outside `torch.no_grad`, so the gradient runs back through the head, the absolute value and the scaling, and all three U-Net calls. The file is `numbers/real/count-explained.npz`. Figures: the real maps in [where the two maps come from](../0.5-network-view/dc2-where-the-two-maps-come-from.png) and the real g in [the push gradient into the canvas](../0.5-network-view/dc7-the-push-gradient-into-the-canvas.png).

## 11. The push

`2-classifier-guidance/scripts/abstract_checkers/v3_compose.py`, lines 32 and 33:

```python
B_CHOICES = (0.1, 0.2, 0.4)
WINDOW = {int(t) for t in ddim_timesteps(50) if 480 <= int(t) <= 712}
```

The window is the 50-step sampler's timesteps between 712 and 480, which are sampler steps 15 to 25 (timesteps 693 to 489). The push length $b$ was chosen from the three choices on the 10 seen pairs among digits 0 to 4: 8, 20 and 40 clean pairs of 320, so $b = 0.4$.

Lines 50 to 66:

```python
def make_push(kind, b, head, model, a, bb, seed):
    gen = torch.Generator().manual_seed(seed)

    def push(x, t, eps):
        if int(t[0]) not in WINDOW:
            return torch.zeros_like(eps)
        if kind == "random":
            d = torch.randn(x.shape, generator=gen).to(x.device)
        else:
            with torch.enable_grad():
                xx = x.detach().requires_grad_(True)
                pair = torch.tensor([[a, bb]], device=x.device).expand(x.shape[0], 2)
                logit = head(maps(model, xx, t, pair, "base"), t, pair)[:, 0]
                d, = torch.autograd.grad(logit.sum(), xx)
        size = eps.flatten(1).norm(dim=1)[:, None, None, None]
        return -size * b * d / d.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None, None]
    return push
```

The sampler adds what `push` returns to the product's noise prediction (`scripts/mnist/canvas_unet.py` lines 217 to 219), so

$$\hat\varepsilon = \varepsilon_{\text{product}} - \lVert\varepsilon_{\text{product}}\rVert\; b\; \frac{g}{\lVert g\rVert}, \qquad b = 0.4 .$$

The `random` row swaps $g$ for a random direction of the same length. The sampler's cap of 1.0 times the product's length never binds at $b = 0.4$. On the 10 pairs among digits 5 to 9 at seeds 1000 to 1031 the push gave 43 clean pairs of 320 against 0 for the plain product, 1 for the random push and 0 for the best of 4 seeds (job 94134; [the V3 review](../../../../../artifacts/notes/progress/abstract-checkers/06-does-it-compose/review.md)). Figure: [the push gradient into the canvas](../0.5-network-view/dc7-the-push-gradient-into-the-canvas.png).

## 12. Recording every layer and every sampler step

`2-classifier-guidance/scripts/abstract_checkers/explain_stages.py`, lines 27 to 46, runs the saved count checker's forward pass one layer at a time and keeps each output; lines 86 and 87 stop the run if that pass differs from the head's own output by more than 1e-4:

```python
    if worst > 1e-4:
        raise SystemExit(f"the layer-by-layer forward pass differs from the head's own by {worst}")
```

Job 94116 ran it on the same 16 cells as block 10, and its file is `numbers/real/count-stages.npz`.

`2-classifier-guidance/scripts/abstract_checkers/trace_push.py`, lines 39 to 59, hooks a recorder into the sampler in place of the push. At every one of the 50 steps it computes $g$ for every canvas, applies the push only on the pushed row and only inside the window (line 49), and records where the two maps sit:

```python
        push = -size * B * unit if (row == "push" and step in WINDOW) else torch.zeros_like(eps)
        md = m.detach()
        col = md.sum(2)                                    # (n, 2, W): each map's mass per column
        com = (col * XS.to(x.device)).sum(2) / col.sum(2).clamp_min(1e-12)
```

`com` is each map's centre of mass across the canvas width, and the overlap on line 55 is the shared share of the two maps' mass, 0 to 1. Job 94161 ran it and wrote `numbers/real/trace/`: the pushed row gave 41 clean pairs of 320 on this rerun, and inside the window the two maps' centres move from 1.6 to 6.6 pixels apart.
