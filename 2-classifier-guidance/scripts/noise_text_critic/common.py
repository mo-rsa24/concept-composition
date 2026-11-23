"""What steps W2 and W3 share: the sentences, the tile lookup, and the path from an SDXL latent to CLIP.

The latent path copies NoisyCLIP's released map (src/latent_to_image.py) without its 8-bit step, because
converting to bytes has no gradient. The resize and normalisation are CLIP ViT-L/14's, since the NoisyCLIP
checkpoint ships no processor file.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/shared"))
from disallowed_subjects import check  # noqa: E402

LOOP = Path("/datasets/mmolefe/poe_repair_min/loop-9")
ROOTS = (Path("/datasets/mmolefe/poe_repair_min"), Path("/home-mscluster/mmolefe/poe_repair_min_scratch"))
EVAL_SET = REPO / "2-classifier-guidance/scripts/rung2/eval_set_catdog_v2.json"
NOISYCLIP_DIR = LOOP / "models/noisyclip"
FROZEN = "openai/clip-vit-large-patch14"

WEIGHTS = ((60.0, -60.0, 25.0, -70.0), (60.0, -5.0, 15.0, -50.0), (60.0, 10.0, -5.0, -35.0))
BIASES = (150.0, 140.0, 130.0)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

SENTENCES = {
    "counting": ("a photo of a cat and a dog, two separate animals",
                 ("a photo of one animal", "one animal that is part cat and part dog")),
    "prompt_match": ("a photo of a cat and a dog",
                     ("a photo of two cats", "a photo of two dogs", "a photo of a horse and a sheep")),
    "spatial": ("a cat to the left of a dog", ("a dog to the left of a cat",)),
    "position_size": ("a large cat on the left and a small dog on the right",
                      ("a small cat on the left and a large dog on the right",
                       "a small dog on the left and a large cat on the right",
                       "a large dog on the left and a small cat on the right")),
    "binding": ("a red cat and a blue dog", ("a blue cat and a red dog",)),
}
AND_SENTENCES = ("a cat and a dog", "a cat", "a dog", "")


def all_sentences() -> list[str]:
    out = []
    for right, wrong in SENTENCES.values():
        out += [right, *wrong]
    out += list(AND_SENTENCES)
    out = list(dict.fromkeys(out))
    for s in out:
        check(s)
    return out


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def eval_tiles() -> list[dict]:
    return json.loads(EVAL_SET.read_text())["tiles"]


def find(rel: str) -> Path:
    """A tile's file: 200 of the 220 live on the scratch disk and 20 on /datasets."""
    for root in ROOTS:
        if (root / rel).exists():
            return root / rel
    raise FileNotFoundError(rel)


def load_tile_latent(tile: dict) -> torch.Tensor:
    return torch.load(find(tile["latent"]), map_location="cpu", weights_only=True).float()


def latent_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """(B, 4, H, W) latent to (B, 3, H, W) on the 0 to 255 scale, before any clamp."""
    w = torch.tensor(WEIGHTS, device=x.device, dtype=x.dtype)
    b = torch.tensor(BIASES, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return torch.einsum("blhw,rl->brhw", x, w) + b


def rgb_to_clip(rgb: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) on 0 to 255 to CLIP's normalised 224 by 224 input."""
    y = F.interpolate(rgb / 255.0, size=(224, 224), mode="bicubic", align_corners=False, antialias=False)
    mean = torch.tensor(CLIP_MEAN, device=y.device, dtype=y.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=y.device, dtype=y.dtype).view(1, 3, 1, 1)
    return (y - mean) / std


def latent_to_clip(x: torch.Tensor) -> torch.Tensor:
    """The hard-clamped path W2 reads through."""
    return rgb_to_clip(latent_to_rgb(x).clamp(0, 255))


def load_clip(which: str, device, dtype=torch.float32):
    from transformers import CLIPModel, CLIPTokenizer

    src = str(NOISYCLIP_DIR) if which == "noisyclip" else FROZEN
    model = CLIPModel.from_pretrained(src, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    model.requires_grad_(False)
    return model, CLIPTokenizer.from_pretrained(FROZEN)


@torch.no_grad()
def text_features(model, tokenizer, sentences, device) -> torch.Tensor:
    tok = tokenizer(list(sentences), padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    return F.normalize(model.get_text_features(**tok), dim=-1)


def image_features(model, pixels: torch.Tensor) -> torch.Tensor:
    return F.normalize(model.get_image_features(pixel_values=pixels), dim=-1)
