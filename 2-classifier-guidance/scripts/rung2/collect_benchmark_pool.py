#!/usr/bin/env python
"""Collect the 276-prompt Attend-and-Excite benchmark renders as a labelling pool for the checker.

The paper review rendered every benchmark prompt (animals 66, animals with objects 144, objects 66) at seeds 9 to 12
with several methods, under /datasets/mmolefe/poe_repair_min/outputs/paper_review/. The methods that can draw a clean
pair are collected here: the joint prompt (r1_joint, mono.png), the adapter (r1_ours, ours_*.png), CO3
(co3_default/r1, co3.png) and ToMe (tome_default/r1, tome.png). Plain PoE is left out: it is almost all hybrids, and
negatives are plentiful. Left out by rule: cat x dog at seeds 9 to 16 (the held-out sheet) and the test pairs
lion x meerkat and typewriter x cactus.

Each picture is written as a 768 x 768 JPEG (quality 92) to cut labelling tokens, with an index row naming the method,
pair, seed and source path. Runs on the cluster login node (CPU only).

Writes, under MOUNT_ROOT/outputs/classifier_guidance_ladder/rung2/benchmark-pool/: pictures/<method>__<pair>__seedNN.jpg,
index.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from poe_repair import paths  # noqa: E402

SRC = Path("/datasets/mmolefe/poe_repair_min/outputs/paper_review")
OUT = paths.MOUNT_ROOT / "outputs/classifier_guidance_ladder/rung2/benchmark-pool"
METHODS = {"joint": ("r1_joint", "mono.png"), "ours": ("r1_ours", "ours_*.png"),
           "co3": ("co3_default/r1", "co3.png"), "tome": ("tome_default/r1", "tome.png")}
LEAVE_OUT = {"a_lion__x__a_meerkat", "a_typewriter__x__a_cactus"}


def main() -> int:
    (OUT / "pictures").mkdir(parents=True, exist_ok=True)
    index = []
    for method, (folder, pattern) in METHODS.items():
        n = 0
        for pair_dir in sorted((SRC / folder).iterdir()):
            pair = pair_dir.name
            if not pair_dir.is_dir() or "__x__" not in pair or pair in LEAVE_OUT:
                continue
            for seed_dir in sorted(pair_dir.glob("seed*")):
                seed = int(seed_dir.name[4:])
                if {"cat", "dog"} <= set(pair.replace("__x__", "_").split("_")) and 9 <= seed <= 16:
                    continue
                for src in sorted(seed_dir.glob(pattern))[:1]:
                    stem = f"{method}__{pair}__seed{seed:02d}"
                    dst = OUT / "pictures" / f"{stem}.jpg"
                    if not dst.exists():
                        Image.open(src).convert("RGB").resize((768, 768), Image.BICUBIC).save(dst, quality=92)
                    index.append({"id": stem, "method": method, "pair": pair, "seed": seed, "source": str(src)})
                    n += 1
        print(f"{method}: {n} pictures", flush=True)
    (OUT / "index.json").write_text(json.dumps(index))
    print(f"POOL DONE {len(index)} pictures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
