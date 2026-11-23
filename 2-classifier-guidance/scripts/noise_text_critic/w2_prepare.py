#!/usr/bin/env python
"""Step W2: build the labelled pictures for spatial relation, position and size, and attribute binding.

Takes the first 64 `positive` tiles by the SHA-256 of their id, reads where "cat" and "dog" land on each
finished picture at t = 200, and writes each picture's left-right order and boxes. Then it makes four copies
through SDXL's VAE: unflipped, mirrored, red cat with blue dog, and blue cat with red dog. Sheets for the eye
check are written beside them. No text and image model is loaded, so no score exists when the labels are made.

Writes under loop-9/w2/prep[-smoke]/: labels.json, derived.pt, sheets/labels-NN.png, sheets/colours-NN.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LOOP, REPO, check, eval_tiles, load_tile_latent, sha  # noqa: E402

sys.path.insert(0, str(REPO))
from poe_repair._sdxl.runtime import decode_latents_to_tensor  # noqa: E402
from poe_repair.methods._sampling import add_time_ids  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402

N_PICTURES = 64
MAP_T = 200
MAP_DRAWS = 3
TOP_FRACTION = 0.10
BOX_LEVEL = 0.5
LARGE_RATIO = 1.25
RED_HUE = 0.0
BLUE_HUE = 220.0 / 360.0
MIN_SATURATION = 0.75
MIN_BRIGHTNESS = 0.40
EDGE_PIXELS = 16
PROMPT = "a cat and a dog"


class Store:
    def __init__(self):
        self.maps = []


class StoreCrossAttention:
    """The plain attention computation, keeping the probabilities of the 32 by 32 layers."""

    def __init__(self, store: Store):
        self.store = store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        batch, seq, _ = hidden_states.shape
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
        probs = attn.get_attention_scores(query, key, attention_mask)
        if seq == 32 * 32:
            self.store.maps.append(probs.view(batch, attn.heads, seq, -1).mean(1).float().cpu())
        out = attn.batch_to_head_dim(torch.bmm(probs, value))
        return attn.to_out[1](attn.to_out[0](out))


def token_positions(tokenizer) -> tuple[int, int, int]:
    ids = tokenizer(PROMPT).input_ids
    cat = tokenizer("cat", add_special_tokens=False).input_ids
    dog = tokenizer("dog", add_special_tokens=False).input_ids
    assert len(cat) == 1 and len(dog) == 1
    return ids.index(cat[0]), ids.index(dog[0]), len(ids) - 2


@torch.no_grad()
def word_maps(ctx, z0, seq, pool, cat_i, dog_i, n_words, seed):
    """Average 32 by 32 maps of "cat" and "dog" over layers, heads and noise draws."""
    unet, device, dtype = ctx.models["unet"], ctx.device, ctx.dtype
    gen = torch.Generator().manual_seed(seed)
    z = z0.repeat(MAP_DRAWS, 1, 1, 1).to(device)
    noise = torch.randn(z.shape, generator=gen).to(device)
    t = torch.full((MAP_DRAWS,), MAP_T, device=device, dtype=torch.long)
    x = ctx.scheduler.add_noise(z, noise, t).to(dtype)
    store = Store()
    originals = dict(unet.attn_processors)
    unet.set_attn_processor({n: (StoreCrossAttention(store) if ".attn2." in n else p) for n, p in originals.items()})
    try:
        cond = {"text_embeds": pool.expand(MAP_DRAWS, -1),
                "time_ids": add_time_ids(height=1024, width=1024, batch_size=MAP_DRAWS, device=device, dtype=dtype)}
        unet(x, t, encoder_hidden_states=seq.expand(MAP_DRAWS, -1, -1), added_cond_kwargs=cond)
    finally:
        unet.set_attn_processor(originals)
    m = torch.stack(store.maps).mean(0)
    words = m[:, :, 1:1 + n_words]
    words = words / words.sum(-1, keepdim=True)
    cat = words[:, :, cat_i - 1].mean(0).view(32, 32).numpy()
    dog = words[:, :, dog_i - 1].mean(0).view(32, 32).numpy()
    return cat, dog


def top_centroid_x(m: np.ndarray) -> float:
    k = max(1, int(TOP_FRACTION * m.size))
    idx = np.argsort(m.ravel())[-k:]
    return float(np.unravel_index(idx, m.shape)[1].mean())


def box(m: np.ndarray) -> list[int]:
    ys, xs = np.where(m >= BOX_LEVEL * m.max())
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def area(b: list[int]) -> int:
    return (b[2] - b[0]) * (b[3] - b[1])


def soft_masks(cat: np.ndarray, dog: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each word's region at 1,024 pixels; where both qualify, the word with the larger share keeps it."""
    c, d = cat / cat.sum(), dog / dog.sum()
    mc = (cat >= BOX_LEVEL * cat.max()) & (c >= d)
    md = (dog >= BOX_LEVEL * dog.max()) & (d > c)
    up = lambda a: gaussian_filter(np.kron(a.astype(np.float32), np.ones((32, 32), np.float32)), EDGE_PIXELS)
    mc, md = up(mc), up(md)
    total = mc + md
    over = total > 1
    mc[over] /= total[over]
    md[over] /= total[over]
    return mc, md


def recolour(img: np.ndarray, mc: np.ndarray, md: np.ndarray, cat_hue: float, dog_hue: float) -> np.ndarray:
    hsv = rgb_to_hsv(img)
    def tinted(h):
        o = hsv.copy()
        o[..., 0] = h
        o[..., 1] = np.maximum(o[..., 1], MIN_SATURATION)
        o[..., 2] = np.maximum(o[..., 2], MIN_BRIGHTNESS)
        return hsv_to_rgb(o)
    a, b = mc[..., None], md[..., None]
    return np.clip(img * (1 - a - b) + tinted(cat_hue) * a + tinted(dog_hue) * b, 0, 1)


@torch.no_grad()
def encode(vae, img: np.ndarray, device) -> torch.Tensor:
    px = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None].to(device=device, dtype=vae.dtype) * 2 - 1
    return (vae.encode(px).latent_dist.mean * vae.config.scaling_factor).to(torch.float16).cpu()


def to_pil(a: np.ndarray, size: int = 768) -> Image.Image:
    return Image.fromarray((a * 255).round().astype(np.uint8)).resize((size, size), Image.LANCZOS)


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def label_tile(img: np.ndarray, r: dict) -> Image.Image:
    p = to_pil(img)
    d = ImageDraw.Draw(p)
    scale = 768 / 32
    for key, colour in (("cat_box", (0, 220, 255)), ("dog_box", (255, 150, 0))):
        d.rectangle([v * scale for v in r[key]], outline=colour, width=4)
    size = {"cat": "cat larger", "dog": "dog larger", None: "no clear size"}[r["larger"]]
    text = f"{r['index']:02d}  {'cat LEFT of dog' if r['cat_left'] else 'dog LEFT of cat'}  {size} ({r['area_ratio']:.2f})"
    d.rectangle([0, 0, 768, 34], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255), font=font(21))
    d.text((6, 742), "cyan box: cat   orange box: dog", fill=(255, 255, 255), font=font(18), stroke_width=2, stroke_fill=(0, 0, 0))
    return p


def colour_tile(img: np.ndarray, index: int, cat_colour: str, dog_colour: str) -> Image.Image:
    p = to_pil(img)
    d = ImageDraw.Draw(p)
    d.rectangle([0, 0, 768, 34], fill=(0, 0, 0))
    d.text((6, 5), f"{index:02d}  meant: {cat_colour} cat, {dog_colour} dog", fill=(255, 255, 255), font=font(21))
    return p


def write_sheets(tiles: list[Image.Image], folder: Path, stem: str) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(0, len(tiles), 4):
        sheet = Image.new("RGB", (768 * 2, 768 * 2), "white")
        for j, t in enumerate(tiles[i:i + 4]):
            sheet.paste(t, ((j % 2) * 768, (j // 2) * 768))
        sheet.save(folder / f"{stem}-{n:02d}.png")
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="4 pictures only, written to prep-smoke")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    check(PROMPT)
    out = LOOP / ("w2/prep-smoke" if args.smoke else "w2/prep")
    out.mkdir(parents=True, exist_ok=True)
    n = 4 if args.smoke else N_PICTURES
    tiles = sorted((t for t in eval_tiles() if t["kind"] == "positive"), key=lambda t: sha(t["id"]))[:n]
    assert len(tiles) == n and all(t["pair"] == "a_cat__x__a_dog" for t in tiles)

    ctx = make_ctx(output_root=out / "raw")
    for k in ("vae", "unet", "text_encoder", "text_encoder_2"):
        ctx.models[k].requires_grad_(False).eval()
    seq, pool = encode_prompt_sdxl(PROMPT, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    cat_i, dog_i, n_words = token_positions(ctx.models["tokenizer"])
    vae = ctx.models["vae"]

    records, ids, latents, label_sheet, colour_sheet = [], [], [], [], []
    t0 = time.time()
    for index, tile in enumerate(tiles):
        z0 = load_tile_latent(tile)
        cat, dog = word_maps(ctx, z0, seq, pool, cat_i, dog_i, n_words, int(sha(tile["id"])[:8], 16))
        cat_box, dog_box = box(cat), box(dog)
        ca, da = area(cat_box), area(dog_box)
        ratio = max(ca, da) / max(min(ca, da), 1)
        larger = None if ratio < LARGE_RATIO else ("cat" if ca > da else "dog")
        r = {"index": index, "id": tile["id"], "seed": tile["seed"], "cat_left": top_centroid_x(cat) < top_centroid_x(dog),
             "cat_box": cat_box, "dog_box": dog_box, "larger": larger, "area_ratio": ratio}
        records.append(r)
        with torch.no_grad():
            img = decode_latents_to_tensor(vae, z0.to(ctx.device))[0].permute(1, 2, 0).float().cpu().numpy()
        mc, md = soft_masks(cat, dog)
        red_blue = recolour(img, mc, md, RED_HUE, BLUE_HUE)
        blue_red = recolour(img, mc, md, BLUE_HUE, RED_HUE)
        for name, arr in (("unflipped", img), ("mirror", img[:, ::-1].copy()), ("red_cat_blue_dog", red_blue),
                          ("blue_cat_red_dog", blue_red)):
            ids.append(f"{tile['id']}::{name}")
            latents.append(encode(vae, arr, ctx.device))
        label_sheet.append(label_tile(img, r))
        if index < 16:
            colour_sheet += [colour_tile(red_blue, index, "red", "blue"), colour_tile(blue_red, index, "blue", "red")]
        print(f"[{time.time() - t0:6.0f}s] {index + 1}/{n} {tile['id']} cat_left={r['cat_left']} larger={larger} ratio={ratio:.2f}", flush=True)

    torch.save({"ids": ids, "latents": torch.cat(latents)}, out / "derived.pt")
    (out / "labels.json").write_text(json.dumps({"map_t": MAP_T, "prompt": PROMPT, "records": records}, indent=1))
    a = write_sheets(label_sheet, out / "sheets", "labels")
    b = write_sheets(colour_sheet, out / "sheets", "colours")
    print(f"cat left of dog: {sum(r['cat_left'] for r in records)} of {n}; clear size difference: "
          f"{sum(r['larger'] is not None for r in records)} of {n}; sheets: {a} label, {b} colour", flush=True)
    print("PREPARE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
