#!/usr/bin/env python3
"""Loop 12 step Z7, stage 1: how far each noun's vector inside the joint prompt moves from its encoding alone, per tuning pair.

Design: artifacts/notes/progress/leakage/07-own-prediction/design.md, "Stage 1". For each pair, CO3's own prompt embeddings
(the 2,048 numbers per token the U-Net receives: CLIP-L then OpenCLIP bigG penultimate layers) for "a {A} and a {B}" and
for each single-subject prompt; the shift of a noun is 1 minus the cosine between its last token in the two encodings.
The test: Spearman rho between the second noun's shift and W2d's borrowed count, one-sided permutation p < 0.05.

  python z7_text_shift.py --splits <json> --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rung2"))
from co3_common import load, load_parser, set_prompt  # noqa: E402
from train_readout_pair import joint_prompt  # noqa: E402
import z2_bars  # noqa: E402
from z2_predictors import noun_index, nouns  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pairs = json.loads(args.splits.read_text())["tuning_pairs"]
    parser_nlp = load_parser()
    model, config = load(joint_prompt(pairs[0]), parser_nlp, args.out / "_config")
    rows = {}
    for pair in pairs:
        prompt = joint_prompt(pair)
        assert set_prompt(model, config, prompt, parser_nlp) == 2
        a, b = nouns(pair)
        emb = model.text_embeds[0].float()  # [empty, joint, A alone, B alone, joint]
        tok = model.tokenizer
        ja, jb = noun_index(tok, prompt, a), noun_index(tok, prompt, b)
        sa, sb = noun_index(tok, model.prompts_single[0], a), noun_index(tok, model.prompts_single[1], b)
        cos = lambda u, v: float(F.cosine_similarity(u.double(), v.double(), dim=0))
        rows[pair] = {"shift_first": 1 - cos(emb[1, ja], emb[2, sa]), "shift_second": 1 - cos(emb[1, jb], emb[3, sb]),
                      "single_prompts": list(model.prompts_single[:2]), "tokens": {"joint": [ja, jb], "alone": [sa, sb]}}
        print(pair, json.dumps({k: round(v, 4) for k, v in rows[pair].items() if k.startswith("shift")}), flush=True)
    z2_bars.ALPHA_ONE_SIDED = 0.05  # one pre-registered test (design, stage 1)
    second = z2_bars.verdict({p: r["shift_second"] for p, r in rows.items()})
    first = z2_bars.verdict({p: r["shift_first"] for p, r in rows.items()})
    result = {"per_pair": rows, "test_second_noun": second, "reported_first_noun": first,
              "text_fix_worth_testing": second["call"] == "support"}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "text-shift.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"second": second, "first": first}), flush=True)


if __name__ == "__main__":
    main()
