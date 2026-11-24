#!/usr/bin/env python
"""Text-only check: can CLIP's sentence vectors tell word order and colour binding apart at all?

If a sentence vector were a sum of one vector per word slot, swapping two nouns would change it by an amount that
does not depend on which nouns they are, and the part that averages both word orders would be exactly zero. This
script measures that part, w, for left-right order and for colour binding, on CLIP ViT-L/14's text encoder (the
encoder NoisyCLIP keeps frozen) and on the same architecture with random weights as a control. No image is used.

    w_order = [u(X left of Y) + u(Y right of X)] - [u(Y left of X) + u(X right of Y)]
    w_bind  = [u(red X and blue Y) + u(blue Y and red X)] - [u(blue X and red Y) + u(red Y and blue X)]

u is the unit-length sentence vector. The ratio ||w|| / (2 ||swap difference||) is 0 for a sum of word slots and 1
when the whole swap difference survives averaging both orders.

Usage: python word_check.py --out <folder>
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL = "openai/clip-vit-large-patch14"
PAIRS = [("cat", "dog"), ("puma", "condor"), ("zebra", "flamingo"), ("horse", "sheep"), ("duck", "goose"),
         ("lighthouse", "sailboat")]
COLOURS = ("red", "blue")


def order_sentences(x, y):
    return {"xLy": f"a {x} to the left of a {y}", "yLx": f"a {y} to the left of a {x}",
            "xRy": f"a {x} to the right of a {y}", "yRx": f"a {y} to the right of a {x}"}


def bind_sentences(x, y, c1, c2):
    return {"c1x_c2y": f"a {c1} {x} and a {c2} {y}", "c2y_c1x": f"a {c2} {y} and a {c1} {x}",
            "c2x_c1y": f"a {c2} {x} and a {c1} {y}", "c1y_c2x": f"a {c1} {y} and a {c2} {x}"}


@torch.no_grad()
def embed(model, tok, sentences):
    t = tok(list(sentences), padding="max_length", truncation=True, max_length=77, return_tensors="pt")
    return F.normalize(model(**t).text_embeds, dim=-1).numpy()


def measure(model, tok) -> dict:
    out = {"order": [], "bind": []}
    swaps_o, swaps_b = {}, {}
    for x, y in PAIRS:
        s = order_sentences(x, y)
        u = dict(zip(s, embed(model, tok, s.values())))
        swap = u["xLy"] - u["yLx"]
        w = (u["xLy"] + u["yRx"]) - (u["yLx"] + u["xRy"])
        noun = embed(model, tok, [f"a {x} to the left of a {y}", f"a {x} to the left of a tree"])
        swaps_o[(x, y)] = swap
        out["order"].append({"pair": f"{x}, {y}", "cosine_of_swapped_sentences": float(u["xLy"] @ u["yLx"]),
                             "swap_difference": float(np.linalg.norm(swap)), "w": float(np.linalg.norm(w)),
                             "ratio": float(np.linalg.norm(w) / (2 * np.linalg.norm(swap))),
                             "noun_change_difference": float(np.linalg.norm(noun[0] - noun[1]))})
        b = bind_sentences(x, y, *COLOURS)
        v = dict(zip(b, embed(model, tok, b.values())))
        swap_b = v["c1x_c2y"] - v["c2x_c1y"]
        wb = (v["c1x_c2y"] + v["c2y_c1x"]) - (v["c2x_c1y"] + v["c1y_c2x"])
        swaps_b[(x, y)] = swap_b
        out["bind"].append({"pair": f"{x}, {y}", "cosine_of_swapped_sentences": float(v["c1x_c2y"] @ v["c2x_c1y"]),
                            "swap_difference": float(np.linalg.norm(swap_b)), "w": float(np.linalg.norm(wb)),
                            "ratio": float(np.linalg.norm(wb) / (2 * np.linalg.norm(swap_b)))})
    for name, swaps in (("order", swaps_o), ("bind", swaps_b)):
        cos = [float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in itertools.combinations(swaps.values(), 2)]
        out[f"{name}_swap_direction_shared_across_pairs"] = {"mean_cosine": float(np.mean(cos)), "min": float(np.min(cos)),
                                                             "max": float(np.max(cos))}
    for name in ("order", "bind"):
        out[f"{name}_mean_ratio"] = float(np.mean([r["ratio"] for r in out[name]]))
        out[f"{name}_mean_cosine_of_swapped_sentences"] = float(np.mean([r["cosine_of_swapped_sentences"] for r in out[name]]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(MODEL)
    trained = CLIPTextModelWithProjection.from_pretrained(MODEL, attn_implementation="eager").eval()
    torch.manual_seed(0)
    random = CLIPTextModelWithProjection(CLIPTextConfig.from_pretrained(MODEL)).eval()
    result = {"model": MODEL, "pairs": PAIRS, "colours": COLOURS,
              "trained": measure(trained, tok), "random_weights": measure(random, tok)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "word-check.json").write_text(json.dumps(result, indent=1))
    for k in ("trained", "random_weights"):
        r = result[k]
        print(f"{k}: order ratio {r['order_mean_ratio']:.3f}, cosine of swapped {r['order_mean_cosine_of_swapped_sentences']:.4f}, "
              f"shared swap direction {r['order_swap_direction_shared_across_pairs']['mean_cosine']:.3f}; "
              f"bind ratio {r['bind_mean_ratio']:.3f}, cosine of swapped {r['bind_mean_cosine_of_swapped_sentences']:.4f}, "
              f"shared swap direction {r['bind_swap_direction_shared_across_pairs']['mean_cosine']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
