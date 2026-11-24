# 🌍 Context: the readout checker, the code that runs it

⬅️ Previous: [the push and its cap](03-the-push-and-its-cap.md) | 📋 [The pages of this method](01-overview.md#the-pages-of-this-method) | Next: [the methods index](../00-INDEX.md) ➡️

This page reads the checker's own code block by block, where it lives, and ties each block to a box of [the overview's picture](01-overview.md) or [the deep dive](02-inside-the-kept-checker.md). The code does five things: read three of the frozen U-Net's maps, shrink them to three logits, train the head, turn the summed logits into a capped push, and slip that push into CO3 without changing a file of CO3's own code. Line numbers are as the files stand at commit `e7787922` on 2026-11-24, every file named here unmodified.

<a href="../../../2-classifier-guidance/artifacts/results/what-each-line-of-readout-guidance-computes/our-count-head-high-level-design-explainer-light.png"><img src="../../../2-classifier-guidance/artifacts/results/what-each-line-of-readout-guidance-computes/our-count-head-high-level-design-explainer-light.png" width="720" alt="Eleven boxes for the count head script: saved states into the frozen U-Net, three hooks, pooling to 32 and projecting, a density map whose sum is the count; below, detector boxes, one blob per box, the map loss and AdamW on the head; a held-out check of balanced accuracy at least 0.75 on 10 of 20 steps"></a>

*The count head, the kept checker's older sibling in the same folder, drawn from `train_count_head.py` at commit `c5e61cec`. Top row, left to right: the saved half-drawn latents x_t of an earlier set of joint-prompt and product drawings (the diagram's "plan 23"), with their timesteps t; the frozen SDXL U-Net run on the empty prompt; the hooks on the middle block and up blocks 0 and 1 (lines 58 to 59); pool to 32, project to 128 channels and sum (lines 72 to 73 and 78), a 128 × 32² stack; a trained readout ρ, a 32 × 32 map whose sum is the count of bodies; and the check, balanced accuracy at least 0.75 on 10 of the first 20 steps, a tile counting as plural when the mean of 4 readings is at least 1.5. Bottom row: GroundingDINO's boxes, one blob per box as the 32 × 32 target, a loss of 1,000 × the map's squared error plus 0.1 × the count's (line 175), and AdamW updating the head. Below the U-Net: held-out tiles noised to steps 0 to 19.* ✅ read on the picture at full size; the hook lines still read 58 to 59 in the current file.

The kept checker reuses the same `Readout` and the same three 1 × 1 projections, and replaces the density map with two stride-2 convolutions, an attention pool and three logits; it also reads SDXL under the pair's joint prompt in place of the empty one. ✅ `train_readout_pair.py`, line 54 imports `Readout` from the count head's file.

## Table of contents

- [Words this page uses](#words-this-page-uses)
- [The blocks and where they live](#the-blocks-and-where-they-live)
- [Reading the three taps](#reading-the-three-taps)
- [The head](#the-head)
- [From logits to one score](#from-logits-to-one-score)
- [Training the head](#training-the-head)
- [The push](#the-push)
- [Slipping the push into CO3](#slipping-the-push-into-co3)
- [How it fails and how you know it worked](#how-it-fails-and-how-you-know-it-worked)
- [Where this came from](#where-this-came-from)

## Words this page uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-blocks-and-where-they-live) ➡️

**Forward hook**: a function PyTorch calls with a module's output each time the module runs.

**Autograd**: PyTorch's record of every operation, which `torch.autograd.grad` replays backwards to give a gradient.

**Checkpoint**: a saved file of trained weights plus a few settings, here `head_step_002000.pt`.

**Wrapper**: an object that stands in for another, passes every call through, and changes one thing on the way.

**The M1 review**: the file recording what the runs of the progress board's step M1 gave; step M1 tests CO3 with this checker's push against CO3 alone.

## The blocks and where they live

Navigation: ⬅️ [Words this page uses](#words-this-page-uses) | 📋 [TOC](#table-of-contents) | [Next](#reading-the-three-taps) ➡️

**Every block is this project's own code, read where it lives.** ✅

| Block | Where it lives | What it computes | Box it draws |
|---|---|---|---|
| reading the taps | `Readout` in [the count head script](../../../2-classifier-guidance/scripts/rung2/train_count_head.py), lines 50 to 69 | one U-Net pass; the middle, up 0 and up 1 outputs | the three green dots |
| the head | `PairHead` in [the readout checker's training script](../../../2-classifier-guidance/scripts/rung2/train_readout_pair.py), lines 61 to 79 | pool, project, add, two convolutions, attention pool, three logits | the trained head band |
| one score | `ReadoutPairClassifier`, the same script, lines 82 to 94 | L = the sum of the three logits | "summed into L" |
| training | `manifest_sampler` and the loop in `main`, lines 112 to 154 and 412 to 469 | 4 clean and 4 reject pictures per pass, noised; cross-entropy on all three outputs | not drawn |
| the push | `Pusher` in [the CO3 push module](../../../2-classifier-guidance/scripts/method_search/co3_push.py), lines 39 to 76 | 30 σ_t ∇L, capped at 0.4 of CO3's guided prediction, on steps 0 to 15 | the orange formula |
| into CO3 | `PushingUNet` and `attach`, the same module, lines 16 to 36 and 79 to 92 | subtracts the push from CO3's main two-entry U-Net call | the orange arrow back to x_t |
| the kept settings | [the CO3 rows script](../../../2-classifier-guidance/scripts/method_search/paper2_rows.py), lines 30 to 32 and 43 to 64 | strength 30, cap 0.4, window (0, 16); loads the checkpoint; points it at each pair's prompt | "steps 0 to 15 of 50" |
| the same push on the plain product | `run_poe_classifier_guided` in [the guided sampler](../../../poe_repair/methods/_classifier_guided.py), lines 49 to 61 and 201 to 232 | ε_PoE − push, capped at the given fraction of ε_PoE | where the recipe was tuned |

## Reading the three taps

Navigation: ⬅️ [The blocks and where they live](#the-blocks-and-where-they-live) | 📋 [TOC](#table-of-contents) | [Next](#the-head) ➡️

**Three hooks copy the three block outputs into a dictionary while the U-Net runs once; the U-Net's own output is thrown away.** ✅

```python
class Readout(nn.Module):
    """SDXL's mid block and up blocks 0 and 1, read through forward hooks on a frozen UNet."""

    def __init__(self, unet, seq_e, pool_e):
        super().__init__()
        self.unet, self.feats = unet, {}
        self.register_buffer("seq_e", seq_e, persistent=False)
        self.register_buffer("pool_e", pool_e, persistent=False)
        for name, mod in (("mid", unet.mid_block), ("up0", unet.up_blocks[0]), ("up1", unet.up_blocks[1])):
            mod.register_forward_hook(lambda m, i, o, n=name: self.feats.__setitem__(n, o if torch.is_tensor(o) else o[0]))

    def forward(self, x_t, t, seq=None, pool=None):
        """seq, pool: per-picture prompt embeddings (B, 77, 2048) and (B, 1280); the stored prompt when None."""
        b = x_t.shape[0]
        seq = self.seq_e.expand(b, -1, -1) if seq is None else seq
        pool = self.pool_e.expand(b, -1) if pool is None else pool
        cond = {"text_embeds": pool,
                "time_ids": add_time_ids(height=1024, width=1024, batch_size=b, device=x_t.device, dtype=x_t.dtype)}
        self.unet(x_t, t, encoder_hidden_states=seq, added_cond_kwargs=cond)
        return [self.feats[k] for k in ("mid", "up0", "up1")]
```

`seq_e` and `pool_e` are the stored prompt's two text encodings. The CO3 run overwrites them with each pair's joint prompt before rendering that pair (`point_checker`, `paper2_rows.py` lines 63 to 64), and it hands `Readout` the U-Net it saved before wrapping, so the checker's own pass never triggers the push. ✅ lines 87 to 90 there.

## The head

Navigation: ⬅️ [Reading the three taps](#reading-the-three-taps) | 📋 [TOC](#table-of-contents) | [Next](#from-logits-to-one-score) ➡️

**Eight lines of forward pass carry the whole middle band of the deep-dive picture.** ✅

```python
class PairHead(nn.Module):
    def __init__(self, channels=(1280, 1280, 640), width=128, n_out=1):
        super().__init__()
        self.n_out = n_out
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, width, 1), nn.GroupNorm(16, width), nn.SiLU())
                                   for c in channels])
        self.mix = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU())   # 32 -> 8
        self.pos = nn.Parameter(torch.randn(1, 65, width) / width ** 0.5)
        self.attn = nn.MultiheadAttention(width, 8, batch_first=True)
        self.out = nn.Linear(width, n_out)

    def forward(self, feats):
        h = sum(p(F.adaptive_avg_pool2d(f.float(), GRID)) for p, f in zip(self.proj, feats))
        tok = self.mix(h).flatten(2).transpose(1, 2)
        x = torch.cat([tok.mean(1, keepdim=True), tok], 1) + self.pos
        pooled, _ = self.attn(x[:, :1], x, x)
        o = self.out(pooled[:, 0])
        return o[:, 0] if self.n_out == 1 else o          # (B,) or (B, 3): clean pair, a cat present, a dog present
```

| Line of `forward` | Shape after it, one picture | Box on the deep dive |
|---|---|---|
| `h = sum(...)`, with `GRID` 32 | 128 × 32 × 32 | pool, 1 × 1 convolution, the ⊕ |
| `tok = self.mix(h)...` | 64 tokens of 128 | the two stride-2 stacks |
| `x = torch.cat([...]) + self.pos` | 65 tokens of 128 | the pale blue mean token and the green tokens |
| `pooled, _ = self.attn(x[:, :1], x, x)` | 1 vector of 128 | "attention pool: the mean token asks, 8 heads" |
| `o = self.out(...)` | 3 numbers | "linear 128 to 3" and the three red circles |

The kept checkpoint is built with `n_out` 3, read from its saved `heads` field (`paper2_rows.py`, line 47). Counting this class's weights gives 780,675, worked layer by layer on [the deep dive](02-inside-the-kept-checker.md#the-weights-counted). ✅

## From logits to one score

Navigation: ⬅️ [The head](#the-head) | 📋 [TOC](#table-of-contents) | [Next](#training-the-head) ➡️

**The wrapper the push calls adds the three logits for each picture, then adds over the batch.** ✅

```python
    def forward(self, x_t, timestep):
        o = self.head(self.reader(x_t.to(self.reader.seq_e.dtype), timestep))
        return o if o.ndim == 1 else o.sum(-1)              # identity push: clean pair + cat present + dog present

    def log_prob_both(self, x_t, timestep):
        return self.forward(x_t, timestep).sum()
```

The push renders one picture at a time, so the batch sum is over one picture and `log_prob_both` returns that picture's L. ✅ `Pusher` passes `sample[:1]` (`co3_push.py`, line 35). The quantity is a sum of raw logits; [the push page](03-the-push-and-its-cap.md#what-the-push-follows) shows what that means for the push.

## Training the head

Navigation: ⬅️ [From logits to one score](#from-logits-to-one-score) | 📋 [TOC](#table-of-contents) | [Next](#the-push) ➡️

**Each pass reads 8 noised pictures through the frozen U-Net without a gradient, then scores the head's three outputs with cross-entropy, the presence outputs only where a label exists.** ✅ lines 448 to 466.

```python
        y = torch.tensor([1.0] * 4 + [0.0] * 4, device=ctx.device)
        for _ in range(args.accum):
            x, t, pairs, ids = batch(ctx.scheduler, ctx.device, ctx.dtype)
            with (torch.enable_grad() if lora_params else torch.no_grad()):
                feats = read(x, t, pairs)
            o = head(feats)
            if o.ndim == 1:
                loss = F.binary_cross_entropy_with_logits(o, y)
            else:
                pres = torch.tensor([presence.get(i, [-1, -1]) for i in ids], dtype=torch.float32, device=ctx.device)
                mask = (pres >= 0).float()
                ...
                loss = ... F.binary_cross_entropy_with_logits(o[:, 0], y) + \
                    (F.binary_cross_entropy_with_logits(o[:, 1:], pres.clamp(min=0), reduction="none") * mask).sum() \
                    / mask.sum().clamp(min=1) * 2
            (loss / args.accum).backward()
        opt.step()
```

The dots mark the presence-only checker's loss (lines 460 to 461, and the choice between the two at the start of line 462), which the kept checker does not use. A picture with no presence label gets `[-1, -1]`, so its mask is 0 and it adds nothing to the presence term. The optimiser is AdamW at learning rate 3e-4 with weight decay 0.01 (line 310). The `batch` function noises each picture's finished latent with the scheduler's own `add_noise` at a sampler timestep (lines 147 to 151). ✅

**The training sentence above, one term at a time.**
- **Optimiser**: the rule that turns each update's gradient into a change of the head's weights.
- **AdamW**: Adam, short for adaptive moment estimation, scales each weight's step by running averages of that weight's own past gradients, so every weight moves at a similar pace whatever the size of its gradient. The W says the weight decay below is applied on its own, outside that scaling.
- **Learning rate 3e-4**: the size of that step, so a weight moves by roughly 0.0003 per update. 🔍 "roughly" because Adam's scaling makes the step only close to the rate. The [learning rate card](../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Learning%20Rate.md#where-to-learn-this) teaches it.
- **Weight decay 0.01**: each update also shrinks every weight by the learning rate times the decay, 3 × 10⁻⁶ of itself, so a weight no gradient pushes keeps 0.994 of its size after the kept checker's 2,000 updates. ✅ computed this session.
- **add_noise**: the scheduler's noising, x_t = √ᾱ_t x_0 + √(1 − ᾱ_t) ε, worked on one number at steps 0, 15 and 49 on [the deep dive](02-inside-the-kept-checker.md#what-the-head-learned-from).

## The push

Navigation: ⬅️ [Training the head](#training-the-head) | 📋 [TOC](#table-of-contents) | [Next](#slipping-the-push-into-co3) ➡️

**Outside the window the push is zero; inside it, the gradient of L is scaled by 30 σ_t and cut to 0.4 of CO3's guided prediction.** ✅ `Pusher.__call__`, `co3_push.py` lines 48 to 76; the values come from `paper2_rows.py`, lines 30 to 32.

```python
        if not self.window[0] <= self.step < self.window[1]:
            ...
            return torch.zeros_like(x)
        ...
        with torch.enable_grad():
            z = x.detach().float().requires_grad_(True)
            log_p = self.checker.log_prob_both(z, timestep.reshape(1).to(x.device))
            grad = torch.autograd.grad(log_p, z)[0]
        sigma = (1.0 - self.alpha_bar) ** 0.5
        strength = 0.0 if self.mode == "zero" else self.strength
        raw = strength * sigma * grad
        uncond, cond = eps[:1].float(), eps[1:2].float()
        limit = self.cap * (uncond + self.guidance * (cond - uncond)).norm()
        raw_norm = raw.norm()
        fired = bool(raw_norm > limit)
        push = raw * (limit / raw_norm) if fired else raw
```

The first dots are two lines that log a zero; the second are the random mode (lines 56 to 61), which draws a random direction and scales it to the length the real push had at the same step. `self.guidance` is CO3's guidance scale, 0.8. The plain-product sampler does the same with `capped_push`, where the limit is the cap times the length of ε_PoE (`_classifier_guided.py`, lines 49 to 61). ✅

| Line | Symbol on [the push page](03-the-push-and-its-cap.md#the-push-by-what-moves) |
|---|---|
| `grad = torch.autograd.grad(log_p, z)[0]` | ∇_{x_t} L |
| `sigma = (1.0 - self.alpha_bar) ** 0.5` | σ_t |
| `raw = strength * sigma * grad` | the raw push, 30 σ_t ∇L |
| `limit = self.cap * (...).norm()` | 0.4 × the length of CO3's guided prediction |
| `push = raw * (limit / raw_norm) if fired else raw` | the cap |

## Slipping the push into CO3

Navigation: ⬅️ [The push](#the-push) | 📋 [TOC](#table-of-contents) | [Next](#how-it-fails-and-how-you-know-it-worked) ➡️

**A wrapper stands in for CO3's U-Net and changes only the call with a batch of two, which is CO3's main prediction; every other call passes through untouched.** ✅

```python
    def forward(self, sample, timestep, *args, **kwargs):
        out = self.inner(sample, timestep, *args, **kwargs)
        if self.pusher is None or sample.shape[0] != 2:
            return out
        eps = out["sample"]
        push = self.pusher(sample[:1], timestep, eps).to(device=eps.device, dtype=eps.dtype)
        return UNet2DConditionOutput(sample=eps - push)
```

`eps - push` subtracts the same push from both entries, the unconditional and the joint prompt's, which lowers CO3's guided prediction by exactly the push; [the push page](03-the-push-and-its-cap.md#where-the-push-enters-the-sampler) works the identity on numbers. `attach` (lines 79 to 92) wraps CO3's denoising step so the pusher always knows the step index and ᾱ_t. CO3's corrector calls carry at least three entries, so they pass through unpushed. ✅ the module's docstring, lines 3 to 5.

## How it fails and how you know it worked

Navigation: ⬅️ [Slipping the push into CO3](#slipping-the-push-into-co3) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**The code stops itself on four mistakes, and two checks on the cluster showed the push does nothing at strength 0 and something at strength 30.** ✅ for the guards, read in the code; ✍️ for the two checks, [the M1 review](../../../artifacts/notes/progress/method-search/01-beat-the-baseline/review.md), lines 39 to 43.

| What goes wrong | What stops it | What you see |
|---|---|---|
| a checkpoint trained on the empty prompt, or with adapters inside SDXL | `build_checker`, `paper2_rows.py` lines 45 to 46 | `this script expects the kept checker: joint prompt, no trunk adapters` |
| CO3 makes a second two-entry call in one step, so the push would land twice | `Pusher.__call__`, `co3_push.py` lines 49 to 51 | `RuntimeError: CO3 made more than one two-entry UNet call on step ...` |
| a held-out pair, or cat and dog seeds 9 to 16, in the training list | `manifest_sampler`, `train_readout_pair.py` lines 121 to 124 | `guard: a held-out pair is in the training tiles` |
| a prompt the subject guard refuses | `check`, `paper2_rows.py` lines 83 to 85 | `the subject guard refuses: [...]` |

| Check | Pass mark | Recorded |
|---|---|---|
| zero push: the push path run at strength 0 against CO3's own loop, one process, one card | largest absolute difference in the final latent under 1e-5 | 0.0 on a Quadro RTX 8000 at float16 |
| the real push changes the render | the final latent moves | largest absolute difference 3.79; the push acted on all 16 steps from 0 to 15 and the cap fired on 2 |

🔨 **Build:** count the head's weights on a machine with PyTorch, from `2-classifier-guidance/scripts/rung2/`: `python -c "from train_readout_pair import PairHead; print(sum(p.numel() for p in PairHead(n_out=3).parameters()))"`. It must print `780675`. Any other number means the head's layers changed and the deep dive's sizes no longer hold.

> [!WARNING]
> That import loads the project package and writes a `__pycache__` folder beside the script. It was not run for these pages; the count was made by hand from the layer shapes and matches `trained_weights_kept_checker` in `make_deep_dives.py`.

## Where this came from

Navigation: ⬅️ [How it fails and how you know it worked](#how-it-fails-and-how-you-know-it-worked) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| Every excerpt and line number | Read in `train_count_head.py`, `train_readout_pair.py`, `co3_push.py`, `paper2_rows.py` and `poe_repair/methods/_classifier_guided.py` at commit `e7787922`, all five clean | 2026-11-24 |
| The checkpoint path the CO3 run loads | Read in `m1_paper2.sbatch`, line 14: `readout_pair/readout-multipair/head_step_002000.pt` | 2026-11-24 |
| The opening picture's boxes and line numbers | Read at full size; path and labels passed `is_showable` and are absent from the held-pictures list | 2026-11-24 |
| The zero-push and real-push checks | Stated by the M1 review, lines 39 to 43 | 2026-11-24 |
| That `import train_readout_pair` writes a cache folder | Inferred from Python's default behaviour; not run | 2026-11-24 |

⬅️ Previous: [the push and its cap](03-the-push-and-its-cap.md) | 📋 [The pages of this method](01-overview.md#the-pages-of-this-method) | Next: [the methods index](../00-INDEX.md) ➡️

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
