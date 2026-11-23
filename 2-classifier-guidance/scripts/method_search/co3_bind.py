"""Give each subject its own region inside CO3 from step 10, without changing a file of the authors' code (step M9, rung 2).

CO3's main prediction is the only UNet call in a step with a batch of two (unconditional, joint prompt). From step 10,
BindingUNet replaces that call's joint-prompt entry with

    c = m_A * eps_A + m_B * eps_B

where eps_A and eps_B are the predictions for CO3's two concept prompts on the same latent, and the masks sum to 1 at every
cell. CO3's guided prediction u + g * (c - u) is then the regional formula in the M9 design.

The masks are read once, at step 10, from the joint prompt's word attention on that step's latent. Each word's 32 by 32 map is
smoothed, thresholded at half its peak, and reduced to the box around its largest connected part. The two boxes become soft
masks through poe_repair/composers/_regional_masks.py. The attention read is a separate UNet call whose output is discarded,
so the sampler's own predictions are computed by the original attention processors.
"""
import numpy as np
import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from scipy.ndimage import gaussian_filter, label

from attention_push import _KeepCrossAttention
from poe_repair.composers._regional_masks import masks_from_boxes
from poe_repair.composers.poe_regional import random_boxes_like

START_STEP = 10
THRESHOLD = 0.5          # of the smoothed map's peak
SMOOTH_SIGMA = 0.5       # in cells of the 32 by 32 grid, the value Attend-and-Excite uses
GRID = 32
LATENT_CELLS_PER_GRID_CELL = 4
MODES = ("own", "swapped", "random", "zero")


def word_tokens(tokenizer, prompt, word):
    """The positions of a word's tokens in the prompt, and how many word tokens the prompt has."""
    ids = tokenizer(prompt).input_ids
    target = tokenizer(word, add_special_tokens=False).input_ids
    for start in range(len(ids) - len(target) + 1):
        if ids[start:start + len(target)] == target:
            return list(range(start, start + len(target))), len(ids) - 2
    raise ValueError(f"{word!r} is not in {prompt!r}")


def box_from_map(attention):
    """(x0, y0, x1, y1) in latent cells, around the largest connected part above half the smoothed peak."""
    smooth = gaussian_filter(attention, SMOOTH_SIGMA)
    peak = smooth.max()
    if not peak > 0:
        raise ValueError("an attention map has no positive value")
    parts, _ = label(smooth >= THRESHOLD * peak)
    sizes = np.bincount(parts.ravel())
    sizes[0] = 0
    ys, xs = np.where(parts == sizes.argmax())
    scale = LATENT_CELLS_PER_GRID_CELL
    return int(xs.min()) * scale, int(ys.min()) * scale, (int(xs.max()) + 1) * scale, (int(ys.max()) + 1) * scale


def box_iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    shared = width * height
    total = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    return shared / total if total else 0.0


class BindingUNet(torch.nn.Module):
    """The UNet, with an optional binder applied to CO3's main two-entry prediction."""

    def __init__(self, unet):
        super().__init__()
        self.inner = unet
        self.binder = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(self, sample, timestep, *args, **kwargs):
        out = self.inner(sample, timestep, *args, **kwargs)
        if self.binder is None or sample.shape[0] != 2:
            return out
        eps = out["sample"]
        cond = self.binder(sample[:1], timestep, eps, self.inner)
        return UNet2DConditionOutput(sample=torch.cat([eps[:1], cond.to(dtype=eps.dtype)]))


class Binder:
    """One render's regions. mode: own, swapped (each subject in the other's region), random (same sizes, random places), or zero."""

    def __init__(self, model, prompt, word_a, word_b, mode, box_seed=None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        seq, pool = model.text_embeds
        if seq.shape[0] != 5:
            raise ValueError(f"expected the null, joint, two concepts and joint again, found {seq.shape[0]} embeddings")
        if word_a not in model.prompts_single[0] or word_b not in model.prompts_single[1]:
            raise ValueError(f"concept order differs from the pair: {model.prompts_single[:2]} for {word_a}, {word_b}")
        self.mode, self.box_seed, self.time_ids = mode, box_seed, model.add_time_ids
        self.seq_joint, self.pool_joint = seq[1:2], pool[1:2]
        self.seq_subjects, self.pool_subjects = seq[2:4], pool[2:4]
        self.tokens_a, self.word_count = word_tokens(model.tokenizer, prompt, word_a)
        self.tokens_b, _ = word_tokens(model.tokenizer, prompt, word_b)
        self.step, self.m_a, self.m_b, self.record = None, None, None, {}
        self.main_calls = {}

    def read_masks(self, x, timestep, unet):
        kept, originals = [], dict(unet.attn_processors)
        unet.set_attn_processor({n: (_KeepCrossAttention(kept) if ".attn2." in n else p) for n, p in originals.items()})
        try:
            cond = {"text_embeds": self.pool_joint, "time_ids": self.time_ids.repeat(1, 1)}
            unet(x, timestep, encoder_hidden_states=self.seq_joint, added_cond_kwargs=cond)
        finally:
            unet.set_attn_processor(originals)
        words = torch.stack(kept).mean(0)[0, :, 1:1 + self.word_count]
        words = words / words.sum(-1, keepdim=True)
        map_a = words[:, [i - 1 for i in self.tokens_a]].sum(-1).reshape(GRID, GRID).cpu().numpy()
        map_b = words[:, [i - 1 for i in self.tokens_b]].sum(-1).reshape(GRID, GRID).cpu().numpy()
        read_a, read_b = box_from_map(map_a), box_from_map(map_b)
        box_a, box_b = read_a, read_b
        if self.mode == "random":
            box_a, box_b = random_boxes_like(read_a, read_b, np.random.default_rng(self.box_seed))
        masks = masks_from_boxes(box_a, box_b, reading="even")
        first, second = (masks.b, masks.a) if self.mode == "swapped" else (masks.a, masks.b)
        self.m_a = torch.from_numpy(first)[None, None].to(device=x.device, dtype=torch.float32)
        self.m_b = torch.from_numpy(second)[None, None].to(device=x.device, dtype=torch.float32)
        self.record = {"box_a_read": list(read_a), "box_b_read": list(read_b), "box_a_used": list(box_a), "box_b_used": list(box_b),
                       "read_boxes_iou": round(box_iou(read_a, read_b), 4), "mask_a_share": round(float(self.m_a.mean()), 4),
                       "max_sum_deviation": float(masks.max_sum_deviation)}

    def __call__(self, x, timestep, eps, unet):
        self.main_calls[self.step] = self.main_calls.get(self.step, 0) + 1
        if self.main_calls[self.step] > 1:
            raise RuntimeError(f"CO3 made more than one two-entry UNet call on step {self.step}")
        if self.step < START_STEP:
            return eps[1:2]
        if self.m_a is None:
            self.read_masks(x, timestep, unet)
        cond = {"text_embeds": self.pool_subjects, "time_ids": self.time_ids.repeat(2, 1)}
        subjects = unet(x.repeat(2, 1, 1, 1), timestep, encoder_hidden_states=self.seq_subjects, added_cond_kwargs=cond)["sample"]
        if self.mode == "zero":
            return eps[1:2]
        return self.m_a * subjects[:1].float() + self.m_b * subjects[1:2].float()


def attach(model):
    """Wrap the UNet once and make each step tell the current binder where it is. Returns the wrapper."""
    wrapper = BindingUNet(model.unet)
    model.unet = wrapper
    original = model.denoise_step

    def denoise_step(x, t, step):
        if wrapper.binder is not None:
            wrapper.binder.step = int(step)
        return original(x, t, step)

    model.denoise_step = denoise_step
    return wrapper
