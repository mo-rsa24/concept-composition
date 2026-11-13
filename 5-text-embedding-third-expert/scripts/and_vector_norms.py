#!/usr/bin/env python
"""How big the text-space "and" vector is at each of the 77 token positions, before and after + E("").

    x_raw = E(joint) - E(a) - E(b)            the arithmetic on SDXL's two text encoders, as written
    x_fix = E(joint) - E(a) - E(b) + E("")    cancels the start token exactly, the pads as far as they are shared
    x_pos                                     E(joint), with each single prompt's word subtracted only at the
                                              joint position holding that same word; the start token, the end
                                              token and every pad stay exactly as the joint prompt has them

E is poe_repair.runtime.encode_prompt_sdxl: the second-to-last hidden layer of CLIP ViT-L (columns
0-767) and OpenCLIP bigG (768-2047) side by side, plus bigG's 1280-wide pooled vector. For each
position this prints and saves the norm of x_raw, x_fix and E(joint) on each encoder's slice, and the
three pooled norms. Text encoders only, no UNet, fp32 on whatever device is free; about a minute on
a CPU.

    python scripts/and_vector_norms.py --joint "a cat and a dog" --a "a cat" --b "a dog"

Writes <out>/<joint slug>-norms.json and <out>/<joint slug>-norms.png.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

CLIP_L_WIDTH = 768
# The pass reading, set before the first run: the repair works when x_fix's median norm over the
# pad positions is under this fraction of the joint prompt's own pad norm.
PAD_RATIO_WORKS = 0.1
# x_pos copies every position that carries no word straight from the joint prompt, so the difference
# there is exactly zero or the alignment is wrong. The bar sits here, in the source, not in prose.
SCAFFOLD_MAX_DIFF = 0.0


def load_text_encoders(model_id: str):
    import torch
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    def load(cls, sub):
        try:
            m = cls.from_pretrained(model_id, subfolder=sub)
        except (OSError, ValueError):   # a cache holding only the fp16 files
            m = cls.from_pretrained(model_id, subfolder=sub, variant="fp16")
        return m.float().eval()

    with torch.no_grad():
        return {
            "tokenizer": CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer"),
            "tokenizer_2": CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2"),
            "text_encoder": load(CLIPTextModel, "text_encoder"),
            "text_encoder_2": load(CLIPTextModelWithProjection, "text_encoder_2"),
        }


def tokens(tokenizer, prompt: str) -> list[str]:
    ids = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True).input_ids[0]
    return tokenizer.convert_ids_to_tokens(ids)


def word_pairs(tokenizer, joint: str, singles: list[tuple[str, str]]) -> list[tuple[str, int, int]]:
    """Where each single prompt's words sit in the joint prompt: (key, joint position, single position).

    Only the positions between <|startoftext|> and <|endoftext|> are matched, on the token itself, left
    to right, each joint position claimed once. So "a dog" taken after "a cat" pairs its "a" with the
    joint prompt's second "a", and the subtraction is word by the same word rather than by position.
    A word of a single prompt that the joint prompt does not hold is an error, not a silent skip.
    """
    jt = tokens(tokenizer, joint)
    j_end = jt.index("<|endoftext|>")
    claimed: set[int] = set()
    pairs = []
    for key, prompt in singles:
        st = tokens(tokenizer, prompt)
        for i in range(1, st.index("<|endoftext|>")):
            j = next((k for k in range(1, j_end) if k not in claimed and jt[k] == st[i]), None)
            if j is None:
                raise SystemExit(f"{prompt!r} has {st[i]!r} at position {i}, and {joint!r} has no free "
                                 f"position holding that token: the two prompts cannot be aligned")
            claimed.add(j)
            pairs.append((key, j, i))
    return pairs


def positional_x(seq_joint, seqs_single: dict, tokenizers: dict, joint: str,
                 singles: list[tuple[str, str]], clip_l_width: int = CLIP_L_WIDTH):
    """x_pos, built per encoder slice because the two tokenizers may segment a prompt differently.

    Each slice is the joint prompt's own hidden states with that slice's own word alignment subtracted
    at the matched positions; every unmatched position, the start token and the pads included, is left
    at the joint prompt's value.
    """
    x = seq_joint.clone()
    slices = ((0, clip_l_width, tokenizers["tokenizer"]),
              (clip_l_width, seq_joint.shape[-1], tokenizers["tokenizer_2"]))
    for lo, hi, tk in slices:
        for key, j, i in word_pairs(tk, joint, singles):
            x[:, j, lo:hi] -= seqs_single[key][:, i, lo:hi]
    return x


def and_only_x(seq_joint, seqs_single: dict, seq_empty, tokenizers: dict, joint: str,
               singles: list[tuple[str, str]], clip_l_width: int = CLIP_L_WIDTH):
    """x_and: only what the joint prompt adds, written onto a real prompt's scaffolding.

    The empty prompt supplies the start token, the end token and every pad, so the sequence is one
    SDXL can read; that is the objection that killed x_fix. Onto it goes, at each word slot, the
    joint's state minus the single prompt's state for that same word, and at a word slot no single
    prompt supplies (here "and") the joint's state as it stands. A slot where the parts already
    explain the joint therefore contributes nothing, which is the whole idea.
    """
    x = seq_empty.clone()
    for lo, hi, tk in ((0, clip_l_width, tokenizers["tokenizer"]),
                       (clip_l_width, seq_joint.shape[-1], tokenizers["tokenizer_2"])):
        jt = tokens(tk, joint)
        content = range(1, jt.index("<|endoftext|>"))
        matched = {j: (key, i) for key, j, i in word_pairs(tk, joint, singles)}
        for j in content:
            if j in matched:
                key, i = matched[j]
                x[:, j, lo:hi] += seq_joint[:, j, lo:hi] - seqs_single[key][:, i, lo:hi]
            else:
                x[:, j, lo:hi] += seq_joint[:, j, lo:hi]
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/results/what-the-and-vector-is-made-of")
    args = ap.parse_args()

    import torch

    from poe_repair.config import DEFAULT_MODEL_ID
    from poe_repair.runtime import encode_prompt_sdxl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_text_encoders(DEFAULT_MODEL_ID)
    for k in ("text_encoder", "text_encoder_2"):
        models[k].to(device)
    enc = lambda p: encode_prompt_sdxl(p, models=models, device=device, dtype=torch.float32)

    (s_m, p_m), (s_a, p_a), (s_b, p_b), (s_0, p_0) = (enc(p) for p in (args.joint, args.a, args.b, ""))
    s_pos = positional_x(s_m, {"a": s_a, "b": s_b}, models, args.joint,
                         [("a", args.a), ("b", args.b)])
    seqs = {"x_raw": s_m - s_a - s_b, "x_fix": s_m - s_a - s_b + s_0, "x_pos": s_pos, "joint": s_m}
    # x_pos has no arithmetic to do on the pooled vector, which carries no token positions, so it keeps
    # the joint prompt's own.
    pooled = {"x_raw": p_m - p_a - p_b, "x_fix": p_m - p_a - p_b + p_0, "x_pos": p_m, "joint": p_m}

    norms = {}
    for name, s in seqs.items():
        s = s[0].cpu()
        norms[name] = {"clip_l": s[:, :CLIP_L_WIDTH].norm(dim=-1).tolist(),
                       "bigg": s[:, CLIP_L_WIDTH:].norm(dim=-1).tolist()}

    tok = {p: tokens(models["tokenizer"], p) for p in (args.joint, args.a, args.b, "")}
    tok_2 = {p: tokens(models["tokenizer_2"], p) for p in (args.joint, args.a, args.b, "")}
    # The joint prompt is the longest, so its pads start last: from there on every prompt is padding.
    pad_start = tok[args.joint].index("<|endoftext|>") + 1

    summary = {}
    for half in ("clip_l", "bigg"):
        pads = lambda name: torch.tensor(norms[name][half][pad_start:])
        ratio = float((pads("x_fix") / pads("joint")).median())
        summary[half] = {
            "position_0": {k: norms[k][half][0] for k in norms},
            "content_positions": {k: norms[k][half][1:pad_start] for k in norms},
            "pad_median": {k: float(pads(k).median()) for k in norms},
            "x_fix_over_joint_pad_median": ratio,
            "repair_works": ratio < PAD_RATIO_WORKS,
        }
    summary["pooled"] = {k: float(v.norm()) for k, v in pooled.items()}

    # The build check for x_pos: every position that carries no word must still be the joint prompt's,
    # bit for bit. Those are position 0 (the start token), pad_start - 1 (the end token) and the pads.
    end_pos = pad_start - 1
    scaffold = [0, end_pos] + list(range(pad_start, s_m.shape[1]))
    diff = (s_pos - s_m)[0].abs()
    worst = float(diff[scaffold].max())
    summary["x_pos_scaffolding"] = {
        "positions_checked": {"start_token": 0, "end_token": end_pos,
                              "pads": [pad_start, int(s_m.shape[1]) - 1]},
        "max_abs_difference_from_joint": worst,
        "allowed": SCAFFOLD_MAX_DIFF,
        "build_is_right": worst <= SCAFFOLD_MAX_DIFF,
        "positions_that_differ_at_all": [i for i in range(s_m.shape[1]) if float(diff[i].max()) > 0],
    }

    slug = re.sub(r"[^a-z0-9]+", "-", args.joint.lower()).strip("-")
    args.out.mkdir(parents=True, exist_ok=True)
    record = {"prompts": {"joint": args.joint, "a": args.a, "b": args.b, "empty": ""},
              "model_id": DEFAULT_MODEL_ID, "layer": "hidden_states[-2]", "dtype": "float32",
              "pad_start": pad_start, "pad_ratio_works": PAD_RATIO_WORKS,
              "tokens_clip_l": {p: t[:pad_start + 1] for p, t in tok.items()},
              "tokens_bigg": {p: t[:pad_start + 1] for p, t in tok_2.items()},
              "summary": summary, "norms": norms}
    (args.out / f"{slug}-norms.json").write_text(json.dumps(record, indent=1))

    for half in ("clip_l", "bigg"):
        s = summary[half]
        print(f"\n{half}: position 0  " + "  ".join(f"{k} {v:.3f}" for k, v in s["position_0"].items()))
        print("  position  " + "  ".join(f"{i:>7d}" for i in range(1, pad_start)))
        print("  token     " + "  ".join(f"{t.replace('</w>', ''):>7.7}"
                                         for t in tok[args.joint][1:pad_start]))
        for k in norms:
            print(f"  {k:<8}  " + "  ".join(f"{v:7.2f}" for v in s["content_positions"][k]))
        print(f"  pads {pad_start}-76 median: " + "  ".join(f"{k} {v:.3f}" for k, v in s["pad_median"].items())
              + f"   x_fix/joint {s['x_fix_over_joint_pad_median']:.3f} -> repair works: {s['repair_works']}")
    print("\npooled norms: " + "  ".join(f"{k} {v:.3f}" for k, v in summary["pooled"].items()))

    sc = summary["x_pos_scaffolding"]
    print(f"\nx_pos build check: largest absolute difference from the joint prompt at the start token, "
          f"the end token (position {sc['positions_checked']['end_token']}) and the pads: "
          f"{sc['max_abs_difference_from_joint']:.6g}, allowed {sc['allowed']:.6g} "
          f"-> build is right: {sc['build_is_right']}")
    print(f"  positions where x_pos differs from the joint prompt at all: "
          f"{sc['positions_that_differ_at_all']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pos = np.arange(77)
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    colors = {"x_raw": "#c0392b", "x_fix": "#2471a3", "x_pos": "#1e8449", "joint": "#7f8c8d"}
    bar_w = 0.8 / len(norms)
    for ax, half, title in zip(axes, ("clip_l", "bigg"), ("CLIP ViT-L slice (768 wide)", "OpenCLIP bigG slice (1280 wide)")):
        rest_max = max(max(norms[k][half][1:]) for k in norms)
        cap = 1.15 * rest_max
        for j, k in enumerate(norms):
            vals = np.array(norms[k][half])
            off = (j - (len(norms) - 1) / 2) * bar_w
            ax.bar(pos + off, np.minimum(vals, cap), width=bar_w, color=colors[k], label=k)
            if vals[0] > cap:
                ax.annotate(f"{vals[0]:.0f}, off scale", (0 + off, cap), fontsize=7,
                            rotation=90, va="top", ha="center", color="white")
        ax.axvline(pad_start - 0.5, color="k", ls=":", lw=1)
        ax.text(pad_start, cap * 0.95, "padding from here", fontsize=8, va="top")
        ax.set_ylim(0, cap)
        ax.set_ylabel("norm at this position")
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8)
    short = {"<|startoftext|>": "<start>", "<|endoftext|>": "<end>"}
    labels = [short.get(t, t.replace("</w>", "")) for t in tok[args.joint][:pad_start]] \
        + [""] * (77 - pad_start)
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels([f"{i}\n{l}" if i < pad_start else (str(i) if i % 10 == 0 else "")
                             for i, l in enumerate(labels)], fontsize=7, rotation=45, ha="right")
    axes[1].set_xlabel(f'token position in "{args.joint}" (0 to 76)')
    fig.suptitle(f'Where the "and" vector\'s size sits, one bar per token position: '
                 f'x_raw = E(joint) - E(a) - E(b), x_fix = x_raw + E(""), '
                 f'x_pos = E(joint) with each word subtracted only where that word sits, '
                 f'beside E(joint) itself', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / f"{slug}-norms.png", dpi=150)
    print(f"\nwrote {args.out / f'{slug}-norms.json'} and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
