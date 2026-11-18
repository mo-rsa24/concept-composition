"""A fine-tuned CLIP checker as a push head: it scores the sampler's decoded estimate of the finished picture.

The readout checkers read SDXL's own features at the noisy latent. This one reads pictures, so at each step it
forms the sampler's estimate of the finished latent, x0 = (x_t - sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_bar_t),
decodes it with the VAE, resizes it to the checker's input, and returns log p(clean cat and dog). The push
function passes the product's noise prediction eps (``reads_estimate``); eps is held fixed, so the gradient
with respect to x_t is the gradient with respect to x0 divided by sqrt(alpha_bar_t) and no UNet backward pass
is needed. The decode runs in a float32 copy of the VAE, because gradients through the half-precision decoder
can overflow. Called without eps (best-of-2's final pick) the latent is read as already finished.

Checkpoints are finetune_checker.py's or finetune_estimates.py's best.pt.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from finetune_checker import SPECS, Checker  # noqa: E402
from poe_repair._sdxl.sdipc_utils import decode_latents_with_grad  # noqa: E402


class ClipEstimateChecker(nn.Module):
    reads_estimate = True

    def __init__(self, ckpt: Path, vae, device):
        super().__init__()
        state = torch.load(ckpt, map_location="cpu")
        self.net = Checker(state["model"], n_out=state.get("n_out", 1))
        self.net.load_state_dict(state["state"])
        self.net = self.net.float().eval().to(device).requires_grad_(False)
        spec = SPECS[state["model"]]
        self.res = spec["res"]
        self.register_buffer("mean", torch.tensor(spec["mean"], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(spec["std"], device=device).view(1, 3, 1, 1))
        object.__setattr__(self, "vae32", copy.deepcopy(vae).float().eval().requires_grad_(False))
        self.epoch = state.get("epoch")

    def forward(self, x_t, timestep, eps=None, alpha_bar=None):
        x_t = x_t.float()
        x0 = x_t if eps is None else (x_t - (1 - alpha_bar).sqrt() * eps.float()) / alpha_bar.sqrt()
        img = decode_latents_with_grad(self.vae32, x0)
        img = F.interpolate(img, size=(self.res, self.res), mode="bicubic", antialias=True, align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = self.net((img - self.mean) / self.std)
        return logit.float()

    def log_prob_both(self, x_t, timestep, eps=None, alpha_bar=None):
        # one output: log p(clean pair); two (the species checker): log p(a clear cat) + log p(a clear dog)
        return F.logsigmoid(self.forward(x_t, timestep, eps, alpha_bar)).sum()


class TwoStageHead(nn.Module):
    """Plan 26's combined push: one checker on the early sampler steps, another on the rest.

    ``early`` (the readout checker, which reads SDXL's own features and moves the layout) answers on sampler
    steps before ``split``; ``late`` (the species checker, a ClipEstimateChecker with a clear-cat and a
    clear-dog output) answers from ``split`` on. The step is found from the timestep's place in the sampler's
    schedule. Called without eps (best-of-2's final pick), the late checker scores the finished latent.
    """
    reads_estimate = True

    def __init__(self, early, late, sampler_timesteps, split: int = 10):
        super().__init__()
        self.early, self.late, self.split = early, late, split
        self.order = [int(t) for t in sampler_timesteps]

    def step_of(self, timestep) -> int:
        return self.order.index(int(timestep.reshape(-1)[0]))

    def forward(self, x_t, timestep, eps=None, alpha_bar=None):
        return self.late(x_t, timestep, eps, alpha_bar)

    def log_prob_both(self, x_t, timestep, eps=None, alpha_bar=None):
        if self.step_of(timestep) == self.split and hasattr(self.late, "reset"):
            self.late.reset()   # a new picture reaches the late steps: find its own animals
        if self.step_of(timestep) < self.split:
            return self.early.log_prob_both(x_t, timestep)
        return self.late.log_prob_both(x_t, timestep, eps=eps, alpha_bar=alpha_bar)


class BoxSpeciesHead(nn.Module):
    """Plan 28's late push: each animal pushed toward its own species, through the decoded guess.

    At the first late step (``reset`` is called there by TwoStageHead) the sampler's guess at the finished picture is
    decoded and saved, the project's animal detector (GroundingDINO, count_instances) boxes the animals, the two
    largest boxes are kept, and zero-shot CLIP ViT-L/14 names them: the box with the higher p("a photo of a cat") is
    the cat, the other the dog. From then on log_prob_both returns log p(cat | cat box) + log p(dog | dog box), each
    from zero-shot CLIP's two-way softmax over "a photo of a cat" and "a photo of a dog" on a square crop of the
    decoded guess around the box. The noise prediction is held fixed, as in ClipEstimateChecker. With fewer than two
    boxes the push is zero for that picture, and ``fallbacks`` counts it.
    """
    reads_estimate = True

    def __init__(self, vae, device, scratch: Path):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).float().eval().requires_grad_(False)
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        with torch.no_grad():
            t = self.clip.get_text_features(**proc(text=["a photo of a cat", "a photo of a dog"], return_tensors="pt",
                                                     padding=True).to(device))
        self.register_buffer("text", F.normalize(t.float(), dim=-1))
        spec = SPECS["clip"]
        self.register_buffer("mean", torch.tensor(spec["mean"], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(spec["std"], device=device).view(1, 3, 1, 1))
        object.__setattr__(self, "vae32", copy.deepcopy(vae).float().eval().requires_grad_(False))
        self.device, self.scratch = device, Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.boxes, self.fallbacks, self.detections = None, 0, []

    def reset(self):
        self.boxes = None

    def _x0(self, x_t, eps, alpha_bar):
        x_t = x_t.float()
        return x_t if eps is None else (x_t - (1 - alpha_bar).sqrt() * eps.float()) / alpha_bar.sqrt()

    def _probs(self, crop):  # crop: (1, 3, h, w) in [0, 1]
        x = F.interpolate(crop, size=(224, 224), mode="bicubic", antialias=True, align_corners=False)
        f = F.normalize(self.clip.get_image_features(pixel_values=(x - self.mean) / self.std).float(), dim=-1)
        return (self.clip.logit_scale.exp() * f @ self.text.T).log_softmax(-1)[0]  # log p(cat), log p(dog)

    @staticmethod
    def _square(box, size):
        x0, y0, x1, y1 = box
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2
        a, b = int(max(0, cx - half)), int(max(0, cy - half))
        return a, b, int(min(size, cx + half)), int(min(size, cy + half))

    def _detect(self, img):
        from PIL import Image
        from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances
        path = self.scratch / "late-start-guess.png"
        Image.fromarray((img[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()).save(path)
        n, found = count_instances(path, device=self.device)
        found = sorted(found, key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))[:2]
        self.detections.append(n)
        print(f"  late start: the detector found {n} animal(s){'' if len(found) >= 2 else '; no late push for this picture'}",
              flush=True)
        if len(found) < 2:
            self.fallbacks += 1
            return []
        size = img.shape[-1]
        boxes = [self._square([float(v) for v in d["box"]], size) for d in found]
        with torch.no_grad():
            p_cat = [float(self._probs(img[:, :, b:d, a:c])[0]) for a, b, c, d in boxes]
        return [boxes[0], boxes[1]] if p_cat[0] >= p_cat[1] else [boxes[1], boxes[0]]   # [cat box, dog box]

    def forward(self, x_t, timestep, eps=None, alpha_bar=None):
        img = decode_latents_with_grad(self.vae32, self._x0(x_t, eps, alpha_bar))
        if self.boxes is None:
            with torch.no_grad():
                self.boxes = self._detect(img.detach())
        if not self.boxes:
            return (img * 0).sum().reshape(1)
        (a0, b0, c0, d0), (a1, b1, c1, d1) = self.boxes
        return (self._probs(img[:, :, b0:d0, a0:c0])[0] + self._probs(img[:, :, b1:d1, a1:c1])[1]).reshape(1)

    def log_prob_both(self, x_t, timestep, eps=None, alpha_bar=None):
        return self.forward(x_t, timestep, eps, alpha_bar).sum()
