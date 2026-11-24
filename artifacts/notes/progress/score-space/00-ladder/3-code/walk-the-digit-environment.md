# The code a score path would be built from

This walk reads the digit environment's real code, block by block, and puts each block beside the equation it computes. Nothing in the repository builds a score path yet, so the walk stops at the three lines that would: the three U-Net calls, the two subtractions, and the sixteen timesteps they would run at.

Every quote is taken from commit `2dcb2bebf268aa7f97783615b23cb17559182976`, and `python3 check_quotes.py` in this folder compares each one with the file at that commit. The code was read and not run, except that [the numbers script](../numbers/make_numbers.py) builds `CanvasUNet` on the CPU with untrained weights to count them.

## Table of contents

- [1. The noise schedule and the no-digit label](#1-the-noise-schedule-and-the-no-digit-label)
- [2. How the step and the label enter the network](#2-how-the-step-and-the-label-enter-the-network)
- [3. Training: one label per call, a pair as two examples](#3-training-one-label-per-call-a-pair-as-two-examples)
- [4. The product: three calls and two subtractions](#4-the-product-three-calls-and-two-subtractions)
- [5. The sixteen steps](#5-the-sixteen-steps)
- [6. The DDIM step that moves the canvas](#6-the-ddim-step-that-moves-the-canvas)
- [7. Where d_t already exists: V2's maps](#7-where-d_t-already-exists-v2s-maps)
- [8. The canvas and the captions](#8-the-canvas-and-the-captions)
- [What the code settles and what it leaves open](#what-the-code-settles-and-what-it-leaves-open)

## 1. The noise schedule and the no-digit label

<!-- quote 2-classifier-guidance/scripts/mnist/canvas_unet.py 35-38 -->
```python
NULL = 10
T = 1000
BETAS = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
ABAR = torch.cumprod(1 - BETAS, 0).float()
```

$$\bar\alpha_t = \prod_{s \le t} (1 - \beta_s), \qquad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$$

Label 10 means no digit, so $\epsilon_\theta(x_t, t, \varnothing)$ is the call with $y = 10$. $\bar\alpha_t$ is the share of the clean canvas's power left in $x_t$; [the sixteen steps figure](../0.5-network-view/sp2-sixteen-steps-make-a-score-path.png) prints its square root at each step of the window.

## 2. How the step and the label enter the network

<!-- quote 2-classifier-guidance/scripts/mnist/canvas_unet.py 44-48 -->
```python
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], 1)
```

<!-- quote 2-classifier-guidance/scripts/mnist/canvas_unet.py 110-110 -->
```python
        emb = self.temb(timestep_embedding(t, self.ch)) + self.cemb(y)
```

<!-- quote 2-classifier-guidance/scripts/mnist/canvas_unet.py 62-62 -->
```python
        h = h + self.e(F.silu(emb))[:, :, None, None]
```

$$\mathrm{emb} = \mathrm{MLP}\big(\mathrm{sincos}_{64}(t)\big) + \mathrm{table}[c], \qquad \mathrm{table} \in \mathbb{R}^{11 \times 256}$$

The step becomes 64 sines and cosines, a two-layer MLP lifts them to 256 numbers, and the label adds one row of an 11-row table. That sum is added to every ResBlock's features after its first convolution (line 62). The condition $c$ is one label, so $\epsilon_\theta(x_t, t, c)$ takes a digit or no digit and never a pair. [Inside the digit U-Net](../0.5-network-view/sp1-inside-the-digit-unet.png) draws where this enters, with the real sizes.

## 3. Training: one label per call, a pair as two examples

<!-- quote 2-classifier-guidance/scripts/mnist/canvas_unet.py 140-144 -->
```python
        if rng.random() < share:
            a, b = SEEN[rng.integers(len(SEEN))]
            c = cv.two(a, b)
            xs += [c, c]
            ys += [a, b]
```

<!-- quote 2-classifier-guidance/scripts/mnist/canvas_unet.py 161-161 -->
```python
        y[torch.from_numpy(rng.random(n) < 0.1)] = NULL
```

<!-- quote 2-classifier-guidance/scripts/mnist/canvas_unet.py 293-294 -->
```python
            pred = model(a.sqrt() * x + (1 - a).sqrt() * noise, t, y)
        loss = F.mse_loss(pred.float(), noise)
```

$$\mathcal{L} = \big\lVert \epsilon_\theta(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t,\ c) - \epsilon \big\rVert^2$$

With share 0.01, one canvas in a hundred holds two digits from the 40 pairs not held out, and it enters training twice, once labelled $a$ and once $b$. Ten percent of labels are replaced by 10. So `share-0.01` has seen a 3 beside a 7 on some canvases, but it has no label meaning "3 and 7": a joint prediction for a pair does not exist in either model. The held-out pairs are (0, 3), (1, 9), (3, 4), (3, 6) and (4, 7), from `canvas_sheet.py` line 25, so every pair among digits 5 to 9 is among those `share-0.01` saw.

## 4. The product: three calls and two subtractions

<!-- quote 2-classifier-guidance/scripts/mnist/canvas_unet.py 172-177 -->
```python
def product_eps(model, x, t, labels, g=1.0):
    """eps_null + g * sum over labels of (eps_label - eps_null), one forward over the stacked batch."""
    b = x.shape[0]
    ys = [torch.full((b,), NULL, device=x.device)] + [torch.full((b,), l, device=x.device) for l in labels]
    out = model(x.repeat(len(ys), 1, 1, 1), t.repeat(len(ys)), torch.cat(ys)).chunk(len(ys))
    return out[0] + g * sum(e - out[0] for e in out[1:])
```

$$\epsilon_{\mathrm{PoE}} = \epsilon_\varnothing + g\,\big(d_t(a) + d_t(b)\big), \qquad d_t(c) = \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)$$

`out[0]` is $\epsilon_\varnothing$ and each `e - out[0]` is one $d_t(c)$. At $g = 1$ this is the session note's $\epsilon_{\mathrm{PoE}}$. The function is decorated `@torch.no_grad()` (line 171), and it returns only the sum, so a score path cannot be read off it: the two $d_t$ terms are formed and added inside the return line and never kept. The miniature's [first figure](../2-miniature/mm1-one-step-three-calls.png) lays these three calls and two subtractions out in four pixels.

## 5. The sixteen steps

<!-- quote 2-classifier-guidance/scripts/mnist/canvas_unet.py 167-168 -->
```python
def ddim_timesteps(n=50):
    return torch.linspace(T - 1, 0, n).round().long()
```

<!-- quote 2-classifier-guidance/scripts/noise_text_critic/w4_train.py 47-48 -->
```python
STEPS16 = ddim_timesteps(50)[:16]
TAU = 0.07
```

<!-- quote 2-classifier-guidance/scripts/abstract_checkers/v2_learn.py 46-46 -->
```python
SDXL_MATCHED = torch.tensor([712, 696, 680, 664, 648, 632, 616, 601, 585, 570, 555, 539, 524, 510, 495, 480])
```

Steps 0 to 15 of a 50-step DDIM run are $t = 999, 979, \dots, 714, 693$. W4 trains on that window. V2 also defines a second window whose noise level matches SDXL's first 16 steps, $t = 712$ down to $480$. At $t = 693$ the clean canvas is scaled by 0.087 inside $x_t$, and at $t = 480$ by more, so the two windows are different questions. The session note says "steps 0 to 15" without naming the window, and step X4 has to name it.

## 6. The DDIM step that moves the canvas

<!-- quote 2-classifier-guidance/scripts/mnist/canvas_unet.py 231-233 -->
```python
        a = abar[t]
        a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
```

<!-- quote 2-classifier-guidance/scripts/mnist/canvas_unet.py 239-239 -->
```python
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
```

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}, \qquad x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t'}}\,\epsilon$$

With eta 0 the run is fixed by its seed. A score path read along this run evaluates each $d_t$ at the product's own $x_t$, which is what the miniature does with its toy step $x - \epsilon_{\mathrm{PoE}}$ ([the two steps figure](../2-miniature/mm2-two-steps-make-a-score-path.png)). A score path read on a test canvas noised once per step evaluates them at $\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ instead. Both are possible from this code.

## 7. Where d_t already exists: V2's maps

<!-- quote 2-classifier-guidance/scripts/abstract_checkers/v2_learn.py 85-93 -->
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
```

$$m_c = 2048 \cdot \frac{\lvert d_t(c) \rvert}{\sum_{\mathrm{pixels}} \lvert d_t(c) \rvert}$$

This is the only place the repository keeps $d_t(a)$ and $d_t(b)$ apart. It reads one step at a time and, in its base form, divides out each map's size. Loop 8's count checker reads these maps on digits 5 to 9 and passes its bar, while its identity checker misses ([the V2 review](../../../abstract-checkers/05-does-it-learn/review.md)). A score path is these 16 maps kept together, raw or scaled; which of the two is for step X4. The digit-map ladder's [where the two maps come from](../../../../../../2-classifier-guidance/artifacts/scenes/digit-map-checkers/0.5-network-view/dc2-where-the-two-maps-come-from.png) draws this block.

## 8. The canvas and the captions

<!-- quote 2-classifier-guidance/scripts/abstract_checkers/build_env.py 65-69 -->
```python
def make_canvas(kind, a, b, digits, source, rng):
    """One canvas of a kind for the asked pair (a, b). For apart and wrong, holds[0] is the left digit."""
    if kind == "apart":
        shown = (a, b) if rng.random() < 0.5 else (b, a)
        corners = [half_corner(rng, 0), half_corner(rng, 1)]
```

<!-- quote 2-classifier-guidance/scripts/noise_text_critic/w4_train.py 54-54 -->
```python
TEXT_MODEL = "openai/clip-vit-large-patch14"
```

The ladder's input canvas is `make_canvas("apart", 3, 7, ...)` on MNIST's test images ([its script](../overview/make_canvas_3_7.py)), which put the 7 on the left. W4's captions always name the lower digit first, so "the digit three and the digit seven, apart" says nothing about left and right. Its bank has 35 captions naming digits 0 to 4: 5 single-digit captions and 3 for each of the 10 pairs.

## What the code settles and what it leaves open

**Settled by the code.** The condition is one of 11 labels. $d_t(c)$ is one call minus the no-digit call at the same $x_t$ and $t$. The product adds the two $d_t$ at each step. The window W4 uses is $t = 999$ to $693$. The U-Net has 6,141,185 weights.

**Left open, for step X4.** Which $x_t$ the path is read at (the product's own run, or a test canvas noised per step). Which window. Raw or scaled $d_t$. What a "joint" score path is, since no pair label exists: one reading is the product's path on `share-0.01` for a pair it saw, which is still a sum of two single-label calls.
