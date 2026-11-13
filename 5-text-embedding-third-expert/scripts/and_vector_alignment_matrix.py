"""Token by prompt matrix for the "and" vector: which token each prompt puts at each position, and the norm of
x_raw, x_fix and the joint prompt's own hidden state there (CLIP ViT-L slice). Reads the norms file written by
5-text-embedding-third-expert/scripts/and_vector_norms.py, draws nothing it does not contain.

    python3 5-text-embedding-third-expert/scripts/and_vector_alignment_matrix.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("5-text-embedding-third-expert/artifacts/results/the-and-vector/what-the-and-vector-is-made-of/a-cat-and-a-dog-norms.json")
OUT = Path("5-text-embedding-third-expert/artifacts/results/the-and-vector/what-the-and-vector-is-made-of/token-alignment-matrix.png")
PROMPTS = ["a cat and a dog", "a cat", "a dog", ""]
HEADS = ['joint\n"a cat and a dog"', 'a\n"a cat"', 'b\n"a dog"', 'empty\n""']
GREEN, RED, GREY, BLUE = "#cfe8cf", "#f4c7c3", "#ececec", "#d6e4f5"


def short(tok: str) -> str:
    return {"<|startoftext|>": "<start>", "<|endoftext|>": "<end>"}.get(tok, tok.replace("</w>", ""))


def main() -> None:
    d = json.loads(SRC.read_text())
    toks = d["tokens_clip_l"]
    norms = d["norms"]
    pad = d["summary"]["clip_l"]["pad_median"]
    rows = list(range(8))
    n_tok, n_val = len(PROMPTS), 3
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, n_tok + n_val + 0.7)
    ax.set_ylim(len(rows) + 1.6, -1.2)
    ax.axis("off")
    cols = HEADS + ["x_raw\nnorm", "x_fix\nnorm", "joint\nnorm"]
    for c, h in enumerate(cols):
        ax.text(c + 1.0 + (0.12 if c == 0 else 0), -0.55, h, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0.5, -0.55, "pos.", ha="center", va="center", fontsize=10, fontweight="bold")
    vmax = 40.0
    for r, pos in enumerate(rows + ["pads"]):
        label = str(pos) if pos != "pads" else "pads,\nmedian"
        ax.text(0.5, r + 0.5, label, ha="center", va="center", fontsize=10)
        for c, p in enumerate(PROMPTS):
            if pos == "pads":
                ax.add_patch(plt.Rectangle((c + 0.6, r + 0.05), 0.8, 0.9, fc="white", ec="white"))
                continue
            tok = toks[p][pos]
            if c == 0:
                fc = BLUE
            elif tok == toks[PROMPTS[0]][pos]:
                fc = GREEN
            else:
                fc = RED
            if tok == "<|endoftext|>" and c > 0 and fc == GREEN:
                fc = GREY
            ax.add_patch(plt.Rectangle((c + 0.6, r + 0.05), 0.8, 0.9, fc=fc, ec="white"))
            ax.text(c + 1.0, r + 0.5, short(tok), ha="center", va="center", fontsize=10)
        for k, name in enumerate(["x_raw", "x_fix", "joint"]):
            v = pad[name] if pos == "pads" else norms[name]["clip_l"][pos]
            shade = plt.cm.Oranges(min(v / vmax, 1.0) * 0.85 + 0.05) if v > 0 else (1, 1, 1, 1)
            c = n_tok + k
            ax.add_patch(plt.Rectangle((c + 0.6, r + 0.05), 0.8, 0.9, fc=shade, ec="white"))
            ax.text(c + 1.0, r + 0.5, f"{v:,.0f}" if v >= 100 else f"{v:.1f}", ha="center", va="center", fontsize=10)
    ax.text(0.5, len(rows) + 1.35,
            "Token cells: green = same token as the joint prompt at that position, red = a different token, grey = padding\n"
            "that matches. Norm cells darken up to 40; position 0 is 1,190 for x_raw and for the joint prompt, off this scale.\n"
            "Every prompt starts with <start>, so E(\"\") cancels position 0 exactly (x_fix = 0). Positions 1 to 5 stay misaligned:\n"
            "\"dog\" sits at position 5 of the joint prompt but at position 2 of \"a dog\", so the subtraction there mixes words.",
            ha="left", va="top", fontsize=9)
    ax.set_title("Where the null token helps and where it does not: tokens by position, CLIP ViT-L slice", fontsize=11)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
