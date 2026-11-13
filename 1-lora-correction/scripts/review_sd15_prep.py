"""Compute job 10's inputs on SD 1.5: one saved starting latent per seed, and the token indices two of the methods need.

Plan: _archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/plan-sd15-baselines-and-syngen-sdxl.md, "The starting latent" and
"The token indices".

Latents. Protocol rule 2 gives every method on one model the same starting noise per seed. The four SD 1.5 methods run
under three torch versions (1.12.1, 2.0.1, 2.5.1), so the noise is drawn once here and saved, and every method loads the
file rather than redrawing it: torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(seed),
dtype=torch.float32), written to <OUT>/latents_sd15/seedNN.pt with its md5 printed. A file that exists is kept (its md5
is printed again), so a rerun cannot change the noise under tiles already rendered.

Token indices. Attend-and-Excite and Divide-and-Bind take the position of each subject noun in the CLIP token sequence
of the joint prompt (position 0 is the start token). The noun is the last word of each concept ("a horse with a glasses"
gives [2, 5]); Divide-and-Bind also takes the position of the colour word in front of each noun, or 0 where a concept has
none ("a pink crown and a purple bow" gives nouns [3, 7], colours [2, 6]). Every word must be a single word piece of SD
1.5's tokenizer, which is asserted. Written to data/prompt-sets/attend-and-excite/token_indices_sd15.json, keyed by the
pair slug. SynGen parses its own prompts and needs none of this.

    python 1-lora-correction/scripts/review_sd15_prep.py                      # both
    python 1-lora-correction/scripts/review_sd15_prep.py --indices-out <path> # write the index file elsewhere
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO / "scripts"))
from review_baseline_cell import slug  # noqa: E402

OUT = Path(os.environ.get("REVIEW_SD15_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review"))
LATENTS = OUT / "latents_sd15"
SNAPSHOT = ("/home-mscluster/mmolefe/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/"
            "451f4fe16113bff5a5d2269ed5ad43b0592e9a14")
SEEDS = range(9, 13)
PAIRS = REPO / "scripts" / "review_benchmark_pairs.txt"
INDICES = REPO / "data" / "prompt-sets" / "attend-and-excite" / "token_indices_sd15.json"


def md5(x: torch.Tensor) -> str:
    return hashlib.md5(x.contiguous().cpu().numpy().tobytes()).hexdigest()


def latents() -> None:
    LATENTS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        path = LATENTS / f"seed{seed:02d}.pt"
        draw = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)
        if path.exists():
            x = torch.load(path)
            state = "kept" if torch.equal(x, draw) else "kept, DIFFERS FROM A FRESH DRAW"
        else:
            x = draw
            torch.save(x, path)
            state = "written"
        print(f"seed {seed}: {state} {tuple(x.shape)} {x.dtype} std {x.std():.4f} md5 {md5(x)} -> {path}", flush=True)


def benchmark() -> list[tuple[str, str, str, str]]:
    rows = []
    for ln in PAIRS.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            a, b, joiner, group = ln.strip().split("|")
            rows.append((a, b, joiner, group))
    return rows


def indices(out: Path) -> None:
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(SNAPSHOT, subfolder="tokenizer")
    table = {}
    for a, b, joiner, group in benchmark():
        joint = f"{a} {joiner} {b}"
        words = joint.split()
        for w in words:
            pieces = tok.tokenize(w)
            assert len(pieces) == 1, f"{w!r} in {joint!r} is {pieces}, not one word piece"
        ids = tok(joint)["input_ids"]
        assert len(ids) == len(words) + 2, (joint, ids)
        # Word k of the prompt sits at token position k + 1, after the start token.
        na = len(a.split())
        noun_a, noun_b = na, na + 1 + len(b.split())
        colour = lambda concept, pos: pos - 1 if len(concept.split()) == 3 else 0
        nouns = [noun_a, noun_b]
        colours = [colour(a, noun_a), colour(b, noun_b)]
        for pos, word in zip(nouns, (a.split()[-1], b.split()[-1])):
            assert tok.decode([ids[pos]]).strip() == word, (joint, pos, word)
        for pos, concept in zip(colours, (a, b)):
            assert pos == 0 or tok.decode([ids[pos]]).strip() == concept.split()[1], (joint, pos)
        table[slug(a, b)] = {"prompt_a": a, "prompt_b": b, "joint_prompt": joint, "set": group,
                             "token_indices": nouns, "color_indices": colours}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1) + "\n")
    n_col = sum(1 for v in table.values() if any(v["color_indices"]))
    print(f"{len(table)} prompts, {n_col} with a colour word -> {out}", flush=True)
    for k in ("a_horse__x__a_glasses", "a_pink_crown__x__a_purple_bow", "a_cat__x__a_dog"):
        if k in table:
            print(f"  {table[k]['joint_prompt']!r}: nouns {table[k]['token_indices']} colours {table[k]['color_indices']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--indices-out", type=Path, default=INDICES)
    p.add_argument("--skip-latents", action="store_true")
    args = p.parse_args()
    if not args.skip_latents:
        latents()
    indices(args.indices_out)


if __name__ == "__main__":
    main()
