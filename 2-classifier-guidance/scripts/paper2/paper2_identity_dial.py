"""The identity dial: a rank-4 adapter on SDXL's attention layers with a strength knob, its trainer and its drawer.

Paper 2's first result (plans/paper-2-dials/plans/03-identity-dial-first-result.md, steps 3 to 6) and
the full ladder's step 3.3 train and draw with this file. How a dial is trained is the ladder's
"How a dial is trained" (plans/paper-2-dials/plans/04-full-ladder.md); the five guards below are the
owner's (artifacts/notes/guarding-the-first-results/paper-2.md, rows 1, 2, 3 and 9).

The adapter. Every attention projection of the U-Net (`to_q`, `to_k`, `to_v`, `to_out.0` inside each
`attn` block) gets a rank-4 pair of matrices, down (in -> 4) and up (4 -> out), and its output becomes

    y = W x + strength * up(down(x))

with up started at zero. At strength 0 the second term is skipped, not multiplied by zero, so the
output is the frozen layer's own, bit for bit.

The random dial. Same shapes, weights drawn at random, then scaled so the size of what it adds equals
an asked norm: sqrt(sum over layers of ||up @ down||^2), the Frobenius norm of every layer's weight
change together. `random` writes one from a trained checkpoint's shapes and norm, with no model loaded.

The rollout is CO3's own sampler. CO3 is loaded the way its row in
2-classifier-guidance/scripts/method_search/paper2_rows.py loads it (load, set_prompt, start_noise),
and every step of every rollout is CO3's `denoise_step`, its joint prompt and its corrector at steps 0
to 5 included. The step that keeps the graph is CO3's guided joint-prompt prediction at that step, so
the window must start after CO3's corrected steps.

The dial acts from `--dial-from-step` onward (default: the window's first step, never earlier), in
training and when `draw` renders the dial and random-dial rows, so both rows act on the same steps.
"Off", the CO3 row, is an untouched U-Net in its own process, never this adapter at strength 0 or
disabled (04-full-ladder.md, rule 14); `load_dial` refuses strength 0.

The trainer is the one-step trainer of poe_repair/experiments/reward_finetune/train.py on CO3's
rollout. The reward is an argument, so both routes in the first result's step 2 use one loop:

    gradient   loss = -reward(decoded estimate at step k), k drawn from the window, the reward being
               the gradient judge's two-clear score through the crops, each body's box held fixed
    kept       CO3's own draw (no dial) is finished and judged without gradient; a draw judged two
               clear is noised back to step k and trains the dial with the denoising loss on the
               joint prompt's own prediction (no guidance); any other draw is skipped

The closeness term is read outside the dial's purpose (setup plan task 3, item 2). `closeness-cache` runs first,
in its own process, with no dial attached to the U-Net at all: for every prompt in a file of single-animal and
unlike-pair prompts (paper2_closeness_prompts.txt) and every starting noise, CO3's own rollout under that prompt
to a step drawn once from the window, where it saves the state x_k and the frozen guided prediction. Each training
step draws a small batch of those states (--closeness-batch, 4) and computes only the dial's prediction on the
stored x_k, so both sides see the same input:
drift = mean((eps_dial - eps_frozen)^2) / mean(eps_frozen^2), averaged over the batch, and the loss adds
closeness_weight * drift. Its weight is an argument, chosen from the drift logged at the smoke's step 50
(closeness.json) so the term is at least a tenth of the loss.

Training pairs come from a file with a weight per pair (--pairs-file, "slug weight" per line), so look-alike pairs
can be drawn at least half the time; each pair's reward is printed and logged to W&B on its own key,
reward/pair/<slug>.

Guards against a term that is silently off (setup plan task 3, item 4; the earlier reward fine-tune ran with its
fidelity term off until commit a72f748): a closeness weight of 0 or less stops the run before the first step. The
reward term and the adapter's gradient norm must not be exactly 0 at step 1, or, when step 1 gives the reward
nothing to differentiate (fewer than two bodies, or a kept-route draw that was skipped), at the first step that
does, within 10 steps; a 0 stops the run with the term's name. The closeness term is checked at the first step
after the dial first moved, not at step 1, because, in item 2's own words, "a new adapter's drift is zero at step
1": up starts at zero, so a check there would stop every run.

    python3 paper2_identity_dial.py train --dry-run --out <dir>
    python paper2_identity_dial.py closeness-cache --closeness-prompts paper2_closeness_prompts.txt \\
        --seeds 21001:21005 --window 15 25 --out <paper2-first-closeness-cache>/closeness-cache.pt
    python paper2_identity_dial.py train --pairs-file pairs.txt --seeds 11001:11065 --route gradient \\
        --reward clip --judge-settings <paper2-judge-clip>/settings.json --window 15 25 \\
        --closeness-cache <closeness-cache.pt> --closeness-weight <from the smoke> \\
        --steps 1000 --ckpt-steps 500 1000 \\
        --out /datasets/mmolefe/poe_repair_min/paper2/paper2-first-identity-train
    python paper2_identity_dial.py random --like <ckpt-1000/dial.pt> --out <random/dial.pt> --seed 0
    python paper2_identity_dial.py draw --ckpt <dial.pt> --strength 1 --dial-from-step 15 \\
        --pairs-file test-pairs.txt --seeds 13001:13004 --out <folder>
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_common import (CROP_MARGIN, PAPER2_DATASETS, REPO, decide_by_shares,  # noqa: E402
                           neither_names, species_from_slug, with_article)
from paper2_kinds import TWO_CLEAR, kind_of  # noqa: E402

RANK = 4
ATTN_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
FIRST_RESULT_CKPT_STEPS = (500, 1000)
CLOSENESS_REPORT_STEP = 50      # the smoke's step at which the closeness weight is chosen
REWARD_GUARD_WITHIN = 10        # the reward term must show a gradient within this many steps
WANDB_PROJECT = "prime_lab/poe-repair-animals-compose"


# ---------------------------------------------------------------- the adapter

class DialLinear(nn.Module):
    """A frozen linear layer plus a rank-r change, scaled by `strength`."""

    def __init__(self, base: nn.Linear, rank: int = RANK):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.lora_down = nn.Linear(base.in_features, rank, bias=False).to(dev, torch.float32)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False).to(dev, torch.float32)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.strength = 0.0

    def forward(self, x):
        y = self.base(x)
        if self.strength == 0.0:
            return y
        d = self.lora_up(self.lora_down(x.to(self.lora_down.weight.dtype)))
        return y + (self.strength * d).to(y.dtype)

    def delta(self) -> torch.Tensor:
        return self.lora_up.weight @ self.lora_down.weight


class Dial:
    """Every DialLinear attached to one model, handled together."""

    def __init__(self, layers: dict[str, DialLinear], rank: int):
        self.layers = layers
        self.rank = rank
        self.strength = 0.0

    def set_strength(self, s: float) -> None:
        self.strength = float(s)
        for m in self.layers.values():
            m.strength = float(s)

    def parameters(self) -> list[nn.Parameter]:
        return [p for m in self.layers.values() for p in (m.lora_down.weight, m.lora_up.weight)]

    def state(self) -> dict[str, torch.Tensor]:
        out = {}
        for n, m in self.layers.items():
            out[f"{n}.lora_down"] = m.lora_down.weight.detach().cpu().clone()
            out[f"{n}.lora_up"] = m.lora_up.weight.detach().cpu().clone()
        return out

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        missing = [n for n in self.layers if f"{n}.lora_down" not in state]
        if missing:
            raise SystemExit(f"checkpoint lacks {len(missing)} layers, e.g. {missing[0]}")
        with torch.no_grad():
            for n, m in self.layers.items():
                m.lora_down.weight.copy_(state[f"{n}.lora_down"])
                m.lora_up.weight.copy_(state[f"{n}.lora_up"])

    def norm(self) -> float:
        return state_norm(self.state())


def attach_dial(model: nn.Module, rank: int = RANK, targets=ATTN_TARGETS) -> Dial:
    """Wrap every attention projection of `model` in a DialLinear, at strength 0."""
    found = [name for name, m in model.named_modules()
             if isinstance(m, nn.Linear) and "attn" in name and any(name.endswith(t) for t in targets)]
    if not found:
        raise SystemExit(f"no attention projections ending in {targets} found")
    layers = {}
    for name in found:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = DialLinear(getattr(parent, child), rank)
        setattr(parent, child, wrapped)
        layers[name] = wrapped
    return Dial(layers, rank)


def state_norm(state: dict[str, torch.Tensor]) -> float:
    """sqrt(sum over layers of ||up @ down||_F^2), computed in float64."""
    total = 0.0
    for k, down in state.items():
        if k.endswith(".lora_down"):
            up = state[k[: -len(".lora_down")] + ".lora_up"]
            total += float((up.double() @ down.double()).pow(2).sum())
    return math.sqrt(total)


def random_state_like(state: dict[str, torch.Tensor], norm: float, seed: int) -> dict[str, torch.Tensor]:
    """Same shapes as `state`, weights drawn from a normal by `seed`, scaled to `norm`."""
    g = torch.Generator().manual_seed(seed)
    out = {k: torch.randn(v.shape, generator=g, dtype=torch.float64) for k, v in sorted(state.items())}
    now = state_norm(out)
    if now == 0:
        raise SystemExit("random draw has zero norm")
    for k in out:
        if k.endswith(".lora_up"):
            out[k] = out[k] * (norm / now)
    return {k: v.to(torch.float32) for k, v in out.items()}


def save_dial(dial_state: dict, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dial_state": dial_state, "rank": RANK, "targets": list(ATTN_TARGETS),
                "norm": state_norm(dial_state), **meta}, path)


def load_dial(model: nn.Module, ckpt: Path, strength: float) -> Dial:
    """Attach a saved dial for a row that draws with it, at `strength` once switched on.

    Strength 0 is refused: the CO3 row is an untouched U-Net in its own process, because a
    disabled adapter has been seen to keep acting (04-full-ladder.md, rule 14).
    """
    if strength == 0:
        raise SystemExit("'off' is an untouched U-Net in its own process, not a dial at strength 0")
    saved = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    dial = attach_dial(model, saved.get("rank", RANK), tuple(saved.get("targets", ATTN_TARGETS)))
    dial.load_state(saved["dial_state"])
    return dial


# ---------------------------------------------------------------- the differentiable crop judge

def torch_crops(image: torch.Tensor, boxes: list[list[int]], margin: float = CROP_MARGIN) -> list[torch.Tensor]:
    """paper2_common.crop_body on a (3, H, W) tensor in [0, 1], with the graph kept through the crop.

    The boxes and the mean colour are constants: the gradient flows through the pixels, not the box.
    """
    _, H, W = image.shape
    mean = image.detach().mean(dim=(1, 2), keepdim=True)
    crops = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        cx0, cy0 = max(0, int(round(x0 - mx))), max(0, int(round(y0 - my)))
        cx1, cy1 = min(W, int(round(x1 + mx))), min(H, int(round(y1 + my)))
        crop = image[:, cy0:cy1, cx0:cx1]
        keep = torch.ones_like(crop[:1])
        for j, (ox0, oy0, ox1, oy1) in enumerate(boxes):
            if j == i or (ox0 <= x0 and oy0 <= y0 and ox1 >= x1 and oy1 >= y1):
                continue
            ix0, iy0, ix1, iy1 = max(cx0, ox0), max(cy0, oy0), min(cx1, ox1), min(cy1, oy1)
            if ix1 > ix0 and iy1 > iy0:
                keep[:, iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0] = 0
        crops.append(crop * keep + mean * (1 - keep))
    return crops


def resize_center(crop: torch.Tensor, size: int, resize_to: int | None = None) -> torch.Tensor:
    """Shortest side to `resize_to` (default `size`), bicubic, then the centre `size` square."""
    _, h, w = crop.shape
    r = (resize_to or size) / min(h, w)
    nh, nw = max(size, round(h * r)), max(size, round(w * r))
    x = F.interpolate(crop[None], size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)[0]
    top, left = (nh - size) // 2, (nw - size) // 2
    return x[:, top:top + size, left:left + size]


def two_clear_from_shares(s_a: torch.Tensor, flags: list[bool]) -> torch.Tensor:
    """paper2_common.share_two_clear in torch: 0 below two bodies, else the plan's formula."""
    if s_a.numel() < 2:
        return s_a.sum() * 0
    best = torch.minimum(s_a.max(), (1 - s_a).max())
    flagged = [torch.minimum(s, 1 - s) for s, f in zip(s_a, flags) if f]
    return best * (1 - torch.stack(flagged).max()) if flagged else best


def to_pil(image: torch.Tensor):
    from PIL import Image
    arr = (image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


class BodyBoxes:
    """The body counter on the detached decoded picture: the boxes the crops are held to."""

    def __init__(self, device):
        sys.path.insert(0, str(REPO))
        from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances
        self.count = count_instances
        self.device = device

    def __call__(self, image: torch.Tensor) -> list[list[int]]:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "estimate.png"
            to_pil(image).save(p)
            _, boxes = self.count(p, device=self.device)
        return [[int(round(v)) for v in b["box"]] for b in boxes]


class CropShareReward:
    """The gradient judge's two-clear score, differentiable through the crops (CLIP or ImageNet).

    Each body's call and flag are decided on detached scores with the judge's frozen settings
    (m and p from its settings.json), as the judge decides them; the score keeps the graph.
    """

    def __init__(self, kind: str, device, settings: dict, class_table: Path | None = None):
        self.kind, self.device = kind, device
        self.margin, self.flag_share = settings["margin"], settings["flag_share"]
        self.boxes = BodyBoxes(device)
        if kind == "clip":
            from paper2_judge_clip import ClipNamer
            self.namer = ClipNamer(device)
            for p in self.namer.model.parameters():
                p.requires_grad_(False)
            ip = self.namer.proc.image_processor
            self.mean = torch.tensor(ip.image_mean, device=device).view(3, 1, 1)
            self.std = torch.tensor(ip.image_std, device=device).view(3, 1, 1)
        elif kind == "imagenet":
            import timm
            from timm.data import resolve_data_config
            from paper2_judge_imagenet import TIMM_NAME, load_class_table
            self.table = load_class_table(class_table)
            self.model = timm.create_model(TIMM_NAME, pretrained=True).to(device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            cfg = resolve_data_config({}, model=self.model)
            self.size = cfg["input_size"][-1]
            self.resize_to = int(math.floor(self.size / cfg.get("crop_pct", 1.0)))
            self.mean = torch.tensor(cfg["mean"], device=device).view(3, 1, 1)
            self.std = torch.tensor(cfg["std"], device=device).view(3, 1, 1)
        else:
            raise SystemExit(f"the gradient route needs a judge with a gradient (clip or imagenet), not {kind!r}")

    def probs(self, crop: torch.Tensor, a: str, b: str):
        """(p_a, p_b) with the graph, and the best "neither" probability detached."""
        if self.kind == "clip":
            x = (resize_center(crop, 224) - self.mean) / self.std
            model = self.namer.model
            img = model.get_image_features(pixel_values=x[None].to(model.dtype))[0].float()
            img = img / img.norm()
            names = [a, b] + neither_names(a, b)
            txt = torch.stack([self.namer.text_embedding(n) for n in names]).float()
            p = (model.logit_scale.exp().float() * txt @ img).softmax(-1)
            return p[0], p[1], float(p[2:].max().detach()) if len(names) > 2 else 0.0
        x = (resize_center(crop, self.size, self.resize_to) - self.mean) / self.std
        p = self.model(x[None])[0].softmax(-1)
        neither = [n for n in neither_names(a, b) if n in self.table]
        pn = max((float(p[self.table[n]].sum().detach()) for n in neither), default=0.0)
        return p[self.table[a]].sum(), p[self.table[b]].sum(), pn

    def __call__(self, image: torch.Tensor, a: str, b: str):
        boxes = self.boxes(image)
        s_list, calls = [], []
        for crop in torch_crops(image, boxes):
            pa, pb, pn = self.probs(crop, a, b)
            calls.append(decide_by_shares({"p_a": float(pa), "p_b": float(pb), "p_neither_best": pn},
                                          self.margin, self.flag_share))
            s_list.append(pa / (pa + pb).clamp_min(1e-12))
        s_a = torch.stack(s_list) if s_list else image.sum()[None] * 0
        kind, _ = kind_of(calls)
        return two_clear_from_shares(s_a, [f for _, f in calls]), {"kind": kind, "n_bodies": len(boxes)}


class JudgeKeep:
    """The kept-draws route's reward: 1 when a crop judge calls the finished draw two clear, else 0."""

    def __init__(self, judge: str, device, settings_file: Path):
        from types import SimpleNamespace
        from paper2_common import crop_body
        mod = __import__(f"paper2_judge_{judge}")
        cls = next(v for k, v in vars(mod).items() if k.endswith("Judge") and k != "Judge")
        saved = json.loads(Path(settings_file).read_text())["settings"]
        cli = {"margin": saved.get("margin"), "flag_share": saved.get("flag_share"),
               "borrow_gap": saved.get("borrow_gap"), "n_refs": saved.get("n_refs"),
               "neither": saved.get("neither", False), "yes_threshold": saved.get("yes_threshold"),
               "references": saved.get("references"), "class_table": None}
        self.judge = cls(SimpleNamespace(dry_run=False, device=device, **cli))
        self.settings = self.judge.settings()
        self.crop_body = crop_body
        self.boxes = BodyBoxes(device)

    def __call__(self, image: torch.Tensor, a: str, b: str):
        boxes = self.boxes(image)
        im = to_pil(image)
        calls = []
        for i in range(len(boxes)):
            crop, _ = self.crop_body(im, boxes, i)
            calls.append(self.judge.decide(self.judge.raw(crop, a, b), self.settings))
        kind, _ = kind_of(calls)
        return torch.tensor(1.0 if kind == TWO_CLEAR else 0.0), {"kind": kind, "n_bodies": len(boxes)}


class StandInReward:
    """Dry run only: a smooth, differentiable score of the decoded picture (its mean red minus blue)."""

    def __call__(self, image, a, b):
        return image[0].mean() - image[2].mean(), {"kind": "stand-in", "n_bodies": 2}


class StandInKeep:
    """Dry run only, kept route: keeps every draw."""

    def __call__(self, image, a, b):
        return torch.tensor(1.0), {"kind": TWO_CLEAR, "n_bodies": 2}


# ---------------------------------------------------------------- the sampler: CO3, or a stand-in with CO3's shape

class Sampler:
    """CO3's rollout around a CO3-shaped model: `denoise_step(x, t, i)`, `scheduler.timesteps`,
    `alpha(t)`, `unet`. Every rollout step is the model's own `denoise_step`; the dial is switched
    on from `dial_from` onward and off before it."""

    def __init__(self, model, decode, guided_eps, cond_eps, prompt_eps, set_pair, set_text, start,
                 corrected_steps: int, kind: str):
        self.model = model
        self.unet = model.unet
        self.decode = decode
        self.guided_eps = guided_eps            # (x, k) -> CO3's guided joint-prompt prediction, graph kept
        self.cond_eps = cond_eps                # (x, k) -> the joint prompt's own prediction, no guidance
        self.prompt_eps = prompt_eps            # (x, k, prompt) -> guided prediction under another prompt
        self.set_pair = set_pair                # (a, b) -> CO3 pointed at "a A and a B"
        self.set_text = set_text                # (prompt) -> CO3 pointed at any prompt (the closeness cache)
        self.start = start
        self.kind = kind                        # "co3" or "stand-in": a cache is read only by its own kind
        self.attention = "stand-in"             # CO3's attention kernel: "xformers" or "pytorch-sdpa"
        self.corrected_steps = corrected_steps  # CO3's own corrector runs at steps below this
        self.n_steps = len(model.scheduler.timesteps)

    def rollout(self, x, start: int, end: int, dial: Dial | None, dial_from: int, strength: float):
        """CO3's steps start..end-1 without gradient, the dial at `strength` from `dial_from` on."""
        for i in range(start, end):
            t = self.model.scheduler.timesteps[i]
            if dial is not None:
                dial.set_strength(strength if i >= dial_from else 0.0)
            with torch.no_grad():
                x = self.model.denoise_step(x, t, i)
        return x

    def alpha(self, k):
        return self.model.alpha(self.model.scheduler.timesteps[k]).float()


class StandInModel:
    """A CO3-shaped stand-in on CPU: one attention block as the U-Net, ten steps, prompts as offsets.

    It records every `denoise_step` call and the dial strength in force, so a test can check the
    rollout runs through CO3's step and switches the dial on where it should.
    """

    class _Sched:
        def __init__(self, n):
            self.timesteps = torch.arange(n - 1, -1, -1)

    def __init__(self, seed: int = 0, n_steps: int = 10):
        torch.manual_seed(seed)
        self.unet = StandInUNet()
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self.decoder = nn.Linear(16 * 8, 3 * 8 * 8)
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.scheduler = self._Sched(n_steps)
        self.alphas = torch.linspace(0.05, 0.95, n_steps)
        self.prompt = "a cat and a dog"
        self.calls: list[tuple[int, float]] = []

    def alpha(self, t):
        return self.alphas[len(self.alphas) - 1 - int(t)]

    def eps(self, x, prompt):
        g = torch.Generator().manual_seed(sum(map(ord, prompt)))
        return self.unet(x + 0.1 * torch.randn(1, 1, 8, generator=g))

    def denoise_step(self, x, t, i):
        to_q = self.unet.attn1.to_q
        self.calls.append((i, float(getattr(to_q, "strength", 0.0))))
        at = self.alpha(t)
        e = self.eps(x, self.prompt)
        x0 = (x - (1 - at).sqrt() * e) / at.sqrt()
        an = self.alphas[min(i + 1, len(self.alphas) - 1)]
        return an.sqrt() * x0 + (1 - an).sqrt() * e


class StandInUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = StandInAttention(8)

    def forward(self, x):
        return self.attn1(x)


class StandInAttention(nn.Module):
    """The shape of a diffusers attention block: to_q, to_k, to_v and to_out.0 as linear layers."""

    def __init__(self, d):
        super().__init__()
        self.to_q, self.to_k, self.to_v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.to_out = nn.ModuleList([nn.Linear(d, d)])

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        att = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)
        return self.to_out[0](att @ v)


def stand_in_sampler(seed: int = 0) -> Sampler:
    m = StandInModel(seed)

    def set_pair(a, b):
        m.prompt = f"{with_article(a)} and {with_article(b)}"

    def set_text(prompt):
        m.prompt = prompt

    return Sampler(m, decode=lambda x0: torch.sigmoid(m.decoder(x0.reshape(1, -1))).reshape(3, 8, 8),
                   guided_eps=lambda x, k: m.eps(x, m.prompt), cond_eps=lambda x, k: m.eps(x, m.prompt),
                   prompt_eps=lambda x, k, prompt: m.eps(x, prompt), set_pair=set_pair, set_text=set_text,
                   start=lambda seed: torch.randn(1, 16, 8, generator=torch.Generator().manual_seed(seed)),
                   corrected_steps=1, kind="stand-in")


def co3_sampler(device, out: Path) -> Sampler:
    """CO3 on SDXL, loaded through its row in paper2_rows.py so the trainer and the row share it."""
    for p in (HERE.parent / "method_search", HERE.parent / "method_search" / "rung2",
              HERE.parent / "rung2", REPO):
        sys.path.insert(0, str(p))
    from paper2_co3_patch import use_pytorch_attention_if_no_xformers
    attention = use_pytorch_attention_if_no_xformers()   # before CO3 is built: co3_bw has no xformers
    import paper2_rows as rows
    from poe_repair._sdxl.sdipc_utils import decode_latents_with_grad
    parser = rows.load_parser()
    model, config = rows.load("a cat and a dog", parser, out)
    for p in model.unet.parameters():
        p.requires_grad_(False)
    vae = model.vae.to(torch.float32)   # the fp16 SDXL VAE overflows; decode in fp32
    enc_cache: dict[str, tuple] = {}

    def guided(x, t, seq_u, pool_u, seq_c, pool_c):
        cond = {"time_ids": model.add_time_ids.repeat(2, 1), "text_embeds": torch.cat([pool_u, pool_c])}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out_ = model.unet(torch.cat([x, x]), t, encoder_hidden_states=torch.cat([seq_u, seq_c]),
                              added_cond_kwargs=cond)["sample"].float()
        return out_[:1] + config.guidance_scale * (out_[1:2] - out_[:1])

    def cond_eps(x, k):
        seq, pool = model.text_embeds
        cond = {"time_ids": model.add_time_ids, "text_embeds": pool[1:2]}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model.unet(x, model.scheduler.timesteps[k], encoder_hidden_states=seq[1:2],
                              added_cond_kwargs=cond)["sample"].float()

    def guided_eps(x, k):
        # CO3's own prediction after its corrected steps: the empty prompt and the joint prompt,
        # text_embeds rows 0 and 1, as Co3.denoise_step builds them.
        seq, pool = model.text_embeds
        return guided(x, model.scheduler.timesteps[k], seq[0:1], pool[0:1], seq[1:2], pool[1:2])

    def prompt_eps(x, k, prompt):
        if prompt not in enc_cache:
            enc_cache[prompt] = rows.encode(model, prompt)
        seq_c, pool_c = enc_cache[prompt]
        seq, pool = model.text_embeds
        return guided(x, model.scheduler.timesteps[k], seq[0:1], pool[0:1], seq_c, pool_c)

    def set_pair(a, b):
        rows.set_prompt(model, config, f"{with_article(a)} and {with_article(b)}", parser)

    def set_text(prompt):
        rows.set_prompt(model, config, prompt, parser)

    def rollout_step(x, t, i, _step=model.denoise_step):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return _step(x, t, i)
    model.denoise_step = rollout_step
    sampler = Sampler(model, decode=lambda x0: decode_latents_with_grad(vae, x0.float())[0],
                   guided_eps=guided_eps, cond_eps=cond_eps, prompt_eps=prompt_eps, set_pair=set_pair,
                   set_text=set_text, start=lambda seed: rows.start_noise(model, seed),
                   corrected_steps=int(config.num_ts_to_correct), kind="co3")
    sampler.attention = attention
    return sampler


# ---------------------------------------------------------------- inputs

def read_pairs(path: Path) -> list[tuple[str, str, float]]:
    """Training pairs with a weight each: one "slug weight" per line (the weight defaults to 1), or a JSON list of
    {"pair": slug, "weight": w}. # starts a comment. Draws follow the weights."""
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        items = [(d["pair"], float(d.get("weight", 1.0))) if isinstance(d, dict) else (d, 1.0)
                 for d in json.loads(text)]
    else:
        items = []
        for ln in text.splitlines():
            parts = ln.split("#")[0].split()
            if parts:
                items.append((parts[0], float(parts[1]) if len(parts) > 1 else 1.0))
    out = []
    for slug, w in items:
        if w <= 0:
            raise SystemExit(f"{path}: pair {slug} has weight {w}; weights must be above 0")
        a, b = species_from_slug(slug)
        out.append((a, b, w))
    if not out:
        raise SystemExit(f"{path}: no pairs")
    return out


def read_prompts(path: Path) -> list[str]:
    return [ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines() if ln.split("#")[0].strip()]


def parse_seeds(items: list[str]) -> list[int]:
    """'11001:11065' is the range 11001 to 11064; plain numbers are themselves."""
    out = []
    for it in items:
        if ":" in it:
            lo, hi = it.split(":")
            out.extend(range(int(lo), int(hi)))
        else:
            out.append(int(it))
    return out


def slug_of(a: str, b: str) -> str:
    def part(x):
        return with_article(x).replace(" ", "_")
    return f"{part(a)}__x__{part(b)}"


# ---------------------------------------------------------------- the closeness cache

STAND_IN_CLOSENESS_PROMPTS = ["a cat", "a horse and a parrot"]


def build_closeness_cache(sampler: Sampler, prompts: list[str], seeds: list[int], window: tuple[int, int],
                          seed: int, out: Path) -> dict:
    """The frozen side of the closeness term, computed once by a sampler with no dial attached.

    For every prompt and starting seed, a step k drawn by `seed` from the window, CO3's own rollout under that
    prompt to step k, and there the state x_k and the frozen prediction under the prompt (the same function the
    trainer later calls with the dial on). Saved to one file the trainer reads.
    """
    if any(isinstance(m, DialLinear) for m in sampler.unet.modules()):
        raise SystemExit("the closeness cache must come from a U-Net with no dial attached")
    rng = random.Random(seed)
    entries = []
    for prompt in prompts:
        sampler.set_text(prompt)
        for s in seeds:
            k = rng.randrange(window[0], window[1])
            x = sampler.rollout(sampler.start(s), 0, k, None, sampler.n_steps, 0.0)
            with torch.no_grad():
                eps = sampler.prompt_eps(x, k, prompt).float()
            entries.append({"prompt": prompt, "seed": s, "k": k, "x": x.detach().cpu(), "eps": eps.cpu()})
            print(f"closeness cache: {prompt!r} seed {s} step {k}", flush=True)
    cache = {"kind": sampler.kind, "attention": sampler.attention, "dial_attached": False, "prompts": prompts, "seeds": seeds,
             "window": list(window), "seed": seed, "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "entries": entries}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    print(f"wrote {out}: {len(entries)} states and frozen predictions, no dial attached", flush=True)
    return cache


def closeness_cache(a: argparse.Namespace) -> None:
    if a.dry_run:
        sampler = stand_in_sampler(0)
        prompts, seeds = STAND_IN_CLOSENESS_PROMPTS, [21001, 21002]
        window = tuple(a.window) if a.window else (2, 5)
    else:
        for need in ("closeness_prompts", "seeds", "window"):
            if not getattr(a, need):
                raise SystemExit(f"--{need.replace('_', '-')} is required unless --dry-run")
        sampler = co3_sampler(a.device, Path(a.out).parent)
        prompts, seeds, window = read_prompts(a.closeness_prompts), parse_seeds(a.seeds), tuple(a.window)
    build_closeness_cache(sampler, prompts, seeds, window, a.seed, Path(a.out))


def load_closeness_cache(path: Path, kind: str, n_steps: int) -> list[dict]:
    cache = torch.load(str(path), map_location="cpu", weights_only=False)
    if cache.get("dial_attached", True):
        raise SystemExit(f"{path} was not made by a U-Net with no dial attached")
    if cache.get("kind") != kind:
        raise SystemExit(f"{path} was made by the {cache.get('kind')} sampler, not {kind}")
    if not cache["entries"] or any(not 0 <= e["k"] < n_steps for e in cache["entries"]):
        raise SystemExit(f"{path} holds no entries, or a step outside 0..{n_steps - 1}")
    return cache["entries"]


# ---------------------------------------------------------------- the loop

def closeness(eps: torch.Tensor, eps_frozen: torch.Tensor) -> torch.Tensor:
    return ((eps.float() - eps_frozen.float()) ** 2).mean() / (eps_frozen.float() ** 2).mean().clamp_min(1e-12)


def grad_snapshot(params) -> list[torch.Tensor]:
    return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]


def grad_norm_since(params, before) -> float:
    total = 0.0
    for p, b in zip(params, before):
        g = torch.zeros_like(p) if p.grad is None else p.grad.detach()
        total += float((g - b).double().pow(2).sum())
    return math.sqrt(total)


class Tracker:
    """W&B when it is on (each pair's reward on its own key), and nothing when it is off."""

    def __init__(self, mode: str, project: str, name: str, config: dict):
        self.run = None
        if mode != "disabled":
            import wandb
            entity, _, proj = project.rpartition("/")
            self.run = wandb.init(project=proj, entity=entity or None, name=name, mode=mode, config=config,
                                  tags=["paper2", "identity-dial"])

    def log(self, row: dict, step: int) -> None:
        if self.run is not None:
            self.run.log(row, step=step)

    def finish(self, summary: dict) -> None:
        if self.run is not None:
            self.run.summary.update(summary)
            self.run.finish()


def train(a: argparse.Namespace) -> dict:
    if a.closeness_weight is None or a.closeness_weight <= 0:
        raise SystemExit("closeness weight must be above 0: every dial run carries the closeness term "
                         "(the earlier reward fine-tune ran with it silently off until a72f748)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    if a.dry_run:
        window = tuple(a.window) if a.window else (2, 5)
        cache_path = a.closeness_cache
        if cache_path is None:
            # The frozen side comes from another process, as on the cluster.
            cache_path = out / "closeness-cache.pt"
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "closeness-cache", "--dry-run",
                                "--window", str(window[0]), str(window[1]), "--out", str(cache_path)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"the stand-in closeness cache failed: {r.stdout}{r.stderr}")
        sampler = stand_in_sampler(a.seed)
        pairs = [("cheetah", "cougar", 3.0), ("duck", "goose", 1.0)]
        seeds = [11001, 11002]
        reward = StandInReward() if a.route == "gradient" else StandInKeep()
    else:
        for need in ("pairs_file", "seeds", "window", "closeness_cache", "judge_settings"):
            if not getattr(a, need):
                raise SystemExit(f"--{need.replace('_', '-')} is required unless --dry-run")
        cache_path = a.closeness_cache
        sampler = co3_sampler(a.device, out)
        pairs, seeds = read_pairs(a.pairs_file), parse_seeds(a.seeds)
        window = tuple(a.window)
        if a.route == "gradient":
            settings = json.loads(Path(a.judge_settings).read_text())["settings"]
            reward = CropShareReward(a.reward, a.device, settings, a.class_table)
        else:
            reward = JudgeKeep(a.reward, a.device, a.judge_settings)
    cache = load_closeness_cache(cache_path, sampler.kind, sampler.n_steps)
    if not sampler.corrected_steps <= window[0] < window[1] <= sampler.n_steps:
        raise SystemExit(f"window {window} must sit in {sampler.corrected_steps}..{sampler.n_steps}: the step "
                         "with the graph is CO3's plain guided prediction, after its corrected steps")
    dial_from = window[0] if a.dial_from_step is None else a.dial_from_step
    if dial_from < window[0]:
        raise SystemExit(f"--dial-from-step {dial_from} is before the window's first step {window[0]}: "
                         "the dial would act at steps it never trained on")

    dial = attach_dial(sampler.unet, RANK)
    params = dial.parameters()
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr)
    ckpt_steps = sorted(set(a.ckpt_steps) | {a.steps})
    report_step = min(a.closeness_report_step, a.steps)
    weights = [w for _, _, w in pairs]
    meta = {"route": a.route, "reward": "stand-in" if a.dry_run else a.reward, "window": list(window),
            "dial_from_step": dial_from, "closeness_weight": a.closeness_weight, "lr": a.lr,
            "steps": a.steps, "seed": a.seed, "dry_run": a.dry_run, "n_layers": len(dial.layers),
            "pairs": {f"{x}|{y}": w for x, y, w in pairs}, "n_seeds": len(seeds),
            "attention": sampler.attention,
            "closeness_cache": str(cache_path), "closeness_batch": a.closeness_batch,
            "closeness_prompts": sorted({e["prompt"] for e in cache})}
    (out / "run.json").write_text(json.dumps({**meta, "ckpt_steps": ckpt_steps,
                                              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    tracker = Tracker("disabled" if a.dry_run and a.wandb_mode == "online" else a.wandb_mode,
                      a.wandb_project, out.name, meta)
    print(f"dial: {len(dial.layers)} attention projections at rank {RANK}, "
          f"{sum(p.numel() for p in params):,} trainable weights; route {a.route}; window {window}; "
          f"dial on from step {dial_from}; pairs and weights {meta['pairs']}; {len(seeds)} seeds; "
          f"closeness on {a.closeness_batch} of {len(cache)} cached states a step, weight {a.closeness_weight}",
          flush=True)

    log = open(out / "train-log.jsonl", "a")
    rewards_by_pair: dict[str, list[float]] = {}
    reward_checked = False
    closeness_checked = False
    moved_at = None  # the first step whose optimiser step had a task gradient above 0
    skipped = 0
    t0 = time.time()
    for step in range(1, a.steps + 1):
        pa, pb, _ = rng.choices(pairs, weights=weights)[0]
        seed = rng.choice(seeds)
        k = rng.randrange(window[0], window[1])
        batch = rng.sample(cache, min(a.closeness_batch, len(cache)))
        sampler.set_pair(pa, pb)
        x = sampler.start(seed)
        at = sampler.alpha(k)
        opt.zero_grad(set_to_none=True)

        # The task term: the reward's gradient (gradient route) or the denoising loss on a kept draw.
        if a.route == "gradient":
            x = sampler.rollout(x, 0, k, dial, dial_from, 1.0)
            dial.set_strength(1.0)
            eps = sampler.guided_eps(x, k)
            x0 = (x.float() - (1 - at).sqrt() * eps) / at.sqrt()
            score, info = reward(sampler.decode(x0), pa, pb)
            task_loss = -score
        else:
            finished = sampler.rollout(x, 0, sampler.n_steps, dial, sampler.n_steps, 0.0)
            with torch.no_grad():
                score, info = reward(sampler.decode(finished.float()), pa, pb)
            noise = torch.randn(finished.shape, generator=torch.Generator().manual_seed(seed * 1000 + k))
            noise = noise.to(finished.device, torch.float32)
            state = (at.sqrt() * finished.float() + (1 - at).sqrt() * noise).to(finished.dtype)
            dial.set_strength(1.0)
            eps = sampler.cond_eps(state, k)
            task_loss = float(score) * ((eps.float() - noise) ** 2).mean()
            if float(score) == 0:
                skipped += 1
        task_value = float(task_loss.detach()) if torch.is_tensor(task_loss) else float(task_loss)
        task_has_graph = torch.is_tensor(task_loss) and task_loss.requires_grad
        before = grad_snapshot(params)
        if task_has_graph:
            task_loss.backward()
        task_grad = grad_norm_since(params, before)

        # The closeness term: the dial's prediction on cached states of prompts outside its purpose, against the
        # frozen prediction an untouched U-Net made on the same states in another process.
        dial.set_strength(1.0)
        before = grad_snapshot(params)
        drifts = []
        for e in batch:
            xe = e["x"].to(next(sampler.unet.parameters()).device)
            d = closeness(sampler.prompt_eps(xe, e["k"], e["prompt"]),
                          e["eps"].to(xe.device))
            if step == 1 and not d.requires_grad:
                raise SystemExit("the closeness term carries no gradient: the dial is not in its prediction")
            (a.closeness_weight * d / len(batch)).backward()
            drifts.append(float(d.detach()))
        drift = sum(drifts) / len(drifts)
        close_grad = grad_norm_since(params, before)
        term = a.closeness_weight * drift
        dial.set_strength(0.0)

        gnorm = float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip))
        opt.step()

        # The guards (item 4): no term silently off. The reward term and the adapter's gradient norm are checked at
        # the first step the reward can act (two or more bodies, and on the kept route a draw that was kept).
        reward_live = task_has_graph and info.get("n_bodies", 2) >= 2 and not (a.route == "kept" and float(score) == 0)
        if not reward_checked:
            if reward_live:
                if task_value == 0.0 or task_grad == 0.0:
                    raise SystemExit(f"reward term is exactly 0 at step {step} (value {task_value}, gradient "
                                     f"norm {task_grad}): it is not reaching the dial")
                if gnorm == 0.0:
                    raise SystemExit(f"the adapter's gradient norm is exactly 0 at step {step}")
                reward_checked = True
            elif step >= REWARD_GUARD_WITHIN:
                raise SystemExit(f"no step in the first {REWARD_GUARD_WITHIN} gave the reward a gradient "
                                 "(fewer than two bodies, or every kept draw skipped)")
        # The closeness term is checked at the first step after the dial first moved: "a new adapter's drift is
        # zero at step 1" (setup plan task 3, item 2), so a check at step 1 would stop every run.
        if not closeness_checked and moved_at is not None and step > moved_at:
            if term == 0.0 or close_grad == 0.0:
                raise SystemExit(f"closeness term is exactly 0 at step {step}, after the dial moved at step "
                                 f"{moved_at} (term {term}, gradient norm {close_grad}): it is not reaching the dial")
            closeness_checked = True
        if moved_at is None and task_grad > 0:
            moved_at = step
        if not closeness_checked and moved_at is None and step >= REWARD_GUARD_WITHIN:
            raise SystemExit(f"the dial has not moved in the first {REWARD_GUARD_WITHIN} steps")

        share = term / (abs(task_value) + term) if abs(task_value) + term > 0 else 0.0
        key = f"{pa}|{pb}"
        value = float(score.detach()) if torch.is_tensor(score) else float(score)
        rewards_by_pair.setdefault(key, []).append(value)
        row = {"step": step, "pair": key, "seed": seed, "k": k,
               "closeness_states": [f"{e['prompt']}|{e['seed']}|{e['k']}" for e in batch],
               "reward": value, "kind": info.get("kind"), "n_bodies": info.get("n_bodies"),
               "task_loss": task_value, "task_grad_norm": task_grad, "drift": drift,
               "closeness_term": term, "closeness_grad_norm": close_grad, "closeness_share": share,
               "grad_norm": gnorm, "seconds": round(time.time() - t0, 2)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        tracker.log({f"reward/pair/{slug_of(pa, pb)}": value, "reward/all": value, "train/task_loss": task_value,
                     "train/drift": drift, "train/closeness_share": share, "train/grad_norm": gnorm,
                     "train/k": k}, step)
        if step % a.log_every == 0 or step <= 2:
            print(f"step {step}/{a.steps} {pa} and {pb} seed {seed} k {k}: reward {value:.4f} "
                  f"(grad {task_grad:.3g}); closeness on {len(batch)} cached states: drift {drift:.3g}, "
                  f"share {share:.3f} (grad {close_grad:.3g})", flush=True)
        if step == report_step:
            (out / "closeness.json").write_text(json.dumps(
                {"step": step, "closeness_weight": a.closeness_weight, "drift": drift,
                 "closeness_term": term, "task_loss": task_value, "closeness_share": share,
                 "at_least_a_tenth": share >= 0.1,
                 "weight_for_a_tenth": (abs(task_value) / 9 / drift) if drift > 0 else None}, indent=1))
        if step in ckpt_steps:
            path = out / f"ckpt-{step}" / "dial.pt"
            save_dial(dial.state(), path, {**meta, "step": step})
            print(f"checkpoint {path} (norm {state_norm(dial.state()):.6g})", flush=True)
    log.close()

    per_pair = {k: {"n": len(v), "mean": statistics.fmean(v),
                    "spread": statistics.pstdev(v) if len(v) > 1 else 0.0} for k, v in sorted(rewards_by_pair.items())}
    every = [r for v in rewards_by_pair.values() for r in v]
    print("reward per pair (n, mean, spread):")
    for k, v in per_pair.items():
        print(f"  {k:28s} {v['n']:4d}  {v['mean']:+.4f}  {v['spread']:.4f}")
    print(f"  {'all':28s} {len(every):4d}  {statistics.fmean(every):+.4f}  "
          f"{statistics.pstdev(every) if len(every) > 1 else 0.0:.4f}")
    summary = {**meta, "skipped_draws": skipped, "reward_per_pair": per_pair,
               "reward_mean": statistics.fmean(every),
               "reward_spread": statistics.pstdev(every) if len(every) > 1 else 0.0,
               "seconds": round(time.time() - t0, 2), "seconds_per_step": round((time.time() - t0) / a.steps, 3)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    tracker.finish({f"reward/pair/{k}/mean": v["mean"] for k, v in per_pair.items()})
    return summary


def random_dial(a: argparse.Namespace) -> None:
    like = torch.load(str(a.like), map_location="cpu", weights_only=False)
    norm = a.norm if a.norm is not None else like["norm"]
    state = random_state_like(like["dial_state"], norm, a.seed)
    save_dial(state, Path(a.out), {"random": True, "seed": a.seed, "like": str(a.like), "asked_norm": norm,
                                   "window": like.get("window"), "dial_from_step": like.get("dial_from_step")})
    print(f"random dial {a.out}: {len(state) // 2} layers, norm {state_norm(state):.6f} (asked {norm:.6f})")


def draw(a: argparse.Namespace) -> None:
    """Draw one row: CO3 with a saved dial (trained or random) at `--strength` from `--dial-from-step`."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        sampler = stand_in_sampler(0)
        pairs, seeds = [("cheetah", "cougar", 1.0)], [13001]
    else:
        sampler = co3_sampler(a.device, out)
        pairs, seeds = read_pairs(a.pairs_file), parse_seeds(a.seeds)
    dial = load_dial(sampler.unet, a.ckpt, a.strength)
    saved = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    dial_from = a.dial_from_step if a.dial_from_step is not None else saved.get("dial_from_step")
    if dial_from is None:
        raise SystemExit("--dial-from-step is required: the checkpoint does not record it")
    drawn = []
    for pa, pb, _ in pairs:
        sampler.set_pair(pa, pb)
        for seed in seeds:
            x = sampler.rollout(sampler.start(seed), 0, sampler.n_steps, dial, dial_from, a.strength)
            with torch.no_grad():
                image = sampler.decode(x.float())
            path = out / slug_of(pa, pb) / f"seed-{seed}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            to_pil(image).save(path)
            drawn.append({"pair": f"{pa}|{pb}", "seed": seed, "picture": str(path)})
            print(f"{pa} and {pb} seed {seed} -> {path}", flush=True)
    (out / "draw.json").write_text(json.dumps({"ckpt": str(a.ckpt), "strength": a.strength, "attention": sampler.attention,
                                               "dial_from_step": dial_from, "pictures": drawn}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train the identity dial")
    t.add_argument("--out", type=Path, required=True,
                   help=f"run folder; cluster runs use {PAPER2_DATASETS}/<run-name>")
    t.add_argument("--dry-run", action="store_true", help="stand-in CO3, reward and pairs, CPU, 2 steps")
    t.add_argument("--route", choices=("gradient", "kept"), default="gradient")
    t.add_argument("--reward", choices=("clip", "imagenet", "dac", "vlm"), default="clip",
                   help="the judge: the gradient route needs clip or imagenet; the kept route takes any")
    t.add_argument("--judge-settings", type=Path, default=None,
                   help="the judge's frozen settings.json from species judge steps 0.18 to 0.20")
    t.add_argument("--class-table", type=Path, default=None, help="ImageNet judge only")
    t.add_argument("--pairs-file", type=Path, default=None,
                   help="training pairs with a weight each: 'slug weight' per line, or JSON [{pair, weight}]")
    t.add_argument("--seeds", nargs="+", default=None, help="seeds; 11001:11065 is a range")
    t.add_argument("--window", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                   help="the step with the graph is drawn from [LO, HI) of CO3's steps")
    t.add_argument("--dial-from-step", type=int, default=None,
                   help="the dial acts from this step on; default and minimum: the window's first step")
    t.add_argument("--closeness-cache", type=Path, default=None,
                   help="the frozen states and predictions from `closeness-cache` (another process, no dial)")
    t.add_argument("--closeness-batch", type=int, default=4, help="cached states scored for closeness each step")
    t.add_argument("--closeness-weight", type=float, default=None,
                   help="must be above 0; chosen from the drift at the smoke's step 50 (closeness.json)")
    t.add_argument("--closeness-report-step", type=int, default=CLOSENESS_REPORT_STEP)
    t.add_argument("--steps", type=int, default=None, help="training steps (first result: 1000)")
    t.add_argument("--ckpt-steps", type=int, nargs="*", default=None,
                   help=f"steps that save a checkpoint (first result: {' '.join(map(str, FIRST_RESULT_CKPT_STEPS))})")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online",
                   help="each pair's reward is logged on its own key; the dry run is always disabled")
    t.add_argument("--wandb-project", default=WANDB_PROJECT)
    c = sub.add_parser("closeness-cache", help="the frozen side of the closeness term, from a U-Net with no dial")
    c.add_argument("--out", type=Path, required=True, help="the cache file (.pt)")
    c.add_argument("--closeness-prompts", type=Path, default=None,
                   help="single-animal and unlike-pair prompts, one per line (paper2_closeness_prompts.txt)")
    c.add_argument("--seeds", nargs="+", default=None, help="the starting noises; 21001:21005 is a range")
    c.add_argument("--window", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                   help="each prompt and seed gets one step drawn from [LO, HI)")
    c.add_argument("--seed", type=int, default=0, help="draws the steps")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--device", default="cuda")
    r = sub.add_parser("random", help="a random dial of matched norm, from a checkpoint's shapes")
    r.add_argument("--like", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--norm", type=float, default=None, help="default: the --like checkpoint's own norm")
    r.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("draw", help="draw one row with a saved dial")
    d.add_argument("--ckpt", type=Path, required=True)
    d.add_argument("--strength", type=float, default=1.0)
    d.add_argument("--dial-from-step", type=int, default=None, help="default: the checkpoint's own")
    d.add_argument("--pairs-file", type=Path, default=None)
    d.add_argument("--seeds", nargs="+", default=None)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.cmd == "random":
        return random_dial(a)
    if a.cmd == "draw":
        return draw(a)
    if a.cmd == "closeness-cache":
        return closeness_cache(a)
    if a.dry_run:
        a.device = "cpu"
        a.steps = a.steps or 2
        a.ckpt_steps = a.ckpt_steps if a.ckpt_steps is not None else [a.steps]
        a.closeness_weight = 1.0 if a.closeness_weight is None else a.closeness_weight
        a.log_every = min(a.log_every, a.steps)
    elif a.steps is None or a.ckpt_steps is None or a.closeness_weight is None:
        raise SystemExit("--steps, --ckpt-steps and --closeness-weight are required")
    print(json.dumps(train(a)))


if __name__ == "__main__":
    main()
