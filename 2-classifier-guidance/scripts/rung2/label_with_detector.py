#!/usr/bin/env python
"""Plurality labels from the project's validated instance counter, and how far they agree with the hand calls.

``count_instances`` (``compose_scorer_validation/detection_scorer.py``) asks GroundingDINO for boxes
and counts distinct ones spanning at least a quarter of the picture's longer side; two or more is
plural, one is a single thing or a hybrid. The query here is the pair's two nouns ("bear . boar"),
since some pairs are objects.

--validate: count every hand-called picture of ``training_set_v3.json`` and report agreement with the
calls (positive against everything else, and by kind). Run first: the labels are only used if they
agree with the hand calls.
--drawings: count every finished drawing of plan 23 (decoded from its final latent) and write one
label per drawing.

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/detector/:
    validate.json, or labels.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poe_repair import paths  # noqa: E402
from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances  # noqa: E402

RUNG2 = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2"
OUT = RUNG2 / "detector"
HERE = Path(__file__).resolve().parent


QUERY_MODE = "nouns"


def nouns(pair: str) -> str:
    if QUERY_MODE == "animal":
        return "animal"
    strip = lambda s: s.split("_", 1)[1].replace("_", " ")
    a, b = pair.split("__x__")
    return f"{strip(a)} . {strip(b)}"


def validate(device) -> int:
    tiles = json.loads((HERE / "training_set_v3.json").read_text())["tiles"]
    rows, agree = [], Counter()
    by_kind = defaultdict(Counter)
    for t in tiles:
        pic = (paths.MOUNT_ROOT / t["picture"]) if t["source"] == "local" else \
            RUNG2 / "training_set/tiles" / f"{t['id']}.png"
        n, _ = count_instances(pic, query=nouns(t["pair"]), device=device)
        plural = n >= 2
        ok = plural == bool(t["label"])
        agree[ok] += 1
        by_kind[t["kind"]][f"counted {'2+' if plural else '0-1'}"] += 1
        rows.append({"id": t["id"], "pair": t["pair"], "kind": t["kind"], "label": t["label"], "count": n})
    tp = sum(1 for r in rows if r["label"] == 1 and r["count"] >= 2)
    tn = sum(1 for r in rows if r["label"] == 0 and r["count"] < 2)
    pos, neg = sum(r["label"] == 1 for r in rows), sum(r["label"] == 0 for r in rows)
    bal = (tp / pos + tn / neg) / 2
    report = {"n": len(rows), "agreement": agree[True] / len(rows), "balanced_agreement": bal,
              "positives_counted_plural": f"{tp} of {pos}", "negatives_counted_single": f"{tn} of {neg}",
              "by_kind": {k: dict(v) for k, v in by_kind.items()}, "rows": rows}
    (OUT / "validate.json").write_text(json.dumps(report, indent=1))
    print(f"agreement with the hand calls: {agree[True]} of {len(rows)}; balanced {bal:.3f}; "
          f"positives counted 2+: {tp}/{pos}; negatives counted 0-1: {tn}/{neg}", flush=True)
    for k, v in by_kind.items():
        print(f"  {k:22s} {dict(v)}", flush=True)
    print("VALIDATE DONE", flush=True)
    return 0


def drawings(device) -> int:
    from poe_repair.methods._sampling import write_decoded_image
    from poe_repair.run import make_ctx
    from poe_repair.runtime import decode_latents
    ctx = make_ctx(output_root=OUT / "raw")
    labels = {}
    files = sorted((RUNG2 / "kim/states").glob("*.pt"))
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(files):
            stem = f.stem
            pair = stem.split("__seed")[0]
            final = torch.load(f)[-1:].to(ctx.device, ctx.dtype)
            with torch.no_grad():
                img = decode_latents(ctx.models, final)
            path = Path(td) / "pic.png"
            write_decoded_image(img.cpu(), path)
            n, keep = count_instances(path, query=nouns(pair), device=device)
            labels[stem] = {"count": n, "boxes": [[float(v) for v in d["box"]] for d in keep], "size": 1024}
            if i % 100 == 0:
                print(f"{i} of {len(files)} counted", flush=True)
    (OUT / "labels.json").write_text(json.dumps(labels, indent=0))
    c = Counter(("joint" if k.endswith("joint") else "product", v["count"] >= 2) for k, v in labels.items())
    print({f"{m}, {'plural' if p else 'single or hybrid'}": n for (m, p), n in sorted(c.items())}, flush=True)
    print("LABELS DONE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--drawings", action="store_true")
    ap.add_argument("--query", choices=("nouns", "animal"), default="nouns",
                    help="nouns: the pair's two nouns; animal: the validated scorer's generic query")
    args = ap.parse_args()
    global QUERY_MODE, OUT
    QUERY_MODE = args.query
    if args.query != "nouns":
        OUT = OUT / f"query-{args.query}"
    if not torch.cuda.is_available():
        raise SystemExit("guard: torch sees no CUDA card")
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    return validate(device) if args.validate else drawings(device)


if __name__ == "__main__":
    raise SystemExit(main())
