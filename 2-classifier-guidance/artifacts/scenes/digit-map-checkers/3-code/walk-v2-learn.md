# The code behind the two checkers, block by block

This walk reads the code that builds, trains and scores the count checker and the identity checker, in the order one training update runs. Each block quotes the lines, says what they compute in one equation, and links the figure that draws it. Every path is from the repository root, and the line numbers hold at commit `a737766d`. The four files quoted are unchanged between that commit and `6c3f8a16`, the code snapshot the split run (job 94085) ran from.

The push has no code yet. It is step V3's design, and the last section quotes the design instead.

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
- [10. The push, as designed](#10-the-push-as-designed)

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

It never saves either trained head; only `stage_final` (line 347) calls `torch.save`. That is why the rung with real logits and real gradients on example canvases is owed: it needs a run that saves the two heads.

## 10. The push, as designed

`artifacts/notes/progress/abstract-checkers/03-design.md`, section "How the push is computed":

$$\hat\varepsilon_t = \varepsilon_{\text{product}} - \lVert\varepsilon_{\text{product}}\rVert \sum_k b\,\frac{\nabla_{x_t}\ell_k}{\lVert\nabla_{x_t}\ell_k\rVert}, \qquad b = 0.4 \text{ for one checker.}$$

To compute $\nabla_{x_t}\ell$, the maps must be built with gradients on, which `inputs` does not do: the push code needs its own call of `maps` outside `torch.no_grad`, with `xt.requires_grad_(True)`, so the gradient runs back through the head, the absolute value and the scaling, and all three U-Net calls. Figure: [the push gradient into the canvas](../0.5-network-view/dc7-the-push-gradient-into-the-canvas.png).
