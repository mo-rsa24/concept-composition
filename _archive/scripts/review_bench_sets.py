"""The four prompt sets of slot R14 (compute job 13), read from the benchmarks' own files.

    geneval_two    GenEval two_object, 99 prompts, 4 images each       data/prompt-sets/geneval/
    geneval_count  GenEval counting, 80 prompts, 4 images each
    t2i_spatial    T2I-CompBench spatial_val, 300 prompts, 10 images   data/prompt-sets/t2i-compbench/
    t2i_complex    T2I-CompBench complex_val, 300 prompts, 10 images

Images per prompt are each benchmark's own count. The seeds are not: GenEval starts its generator at
seed 42 and T2I-CompBench at 42, 43, ..., and under protocol rule 2 every method here starts from
the shared latent of an integer seed instead. Seeds 9 to 12 (GenEval) and 9 to 16 then 59 to 60
(T2I-CompBench) are used, because no run trained on them, while the starting noise of seeds 17 to 58
is in the adapter's training cache.

A prompt naming a subject of scripts/showcase/disallowed_subjects.py is dropped and counted, never
rendered: BLOCKED always, ASK_FIRST unless the pair ledger allows that word for the set: frog and mouse for R1 and R5, frog for
the two T2I-CompBench sets (a set's "allow" below). Each prompt keeps its index in the benchmark file, so a dropped prompt leaves a gap.

PoE, ours and CO3 run on geneval_two only (the user's decision of 2026-11-11, job 13's row of
report/paper-review/prze-10-compute.md). Its prompts split at " and " into the two concepts, the
rule R1 uses ("a photo of a bench and a sports ball" gives "a photo of a bench" and "a sports ball").

    python3 scripts/review_bench_sets.py --counts
    python3 scripts/review_bench_sets.py --pairs geneval_two          "a|b" per line, for the renderers
    python3 scripts/review_bench_sets.py --co3-cells geneval_two      review_baseline_cell.py --cells lines
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "showcase"))
from disallowed_subjects import ASK_FIRST, is_allowed, words  # noqa: E402

GENEVAL = REPO / "data/prompt-sets/geneval/evaluation_metadata.jsonl"
T2I = REPO / "data/prompt-sets/t2i-compbench"
SETS = {
    "geneval_two": {"bench": "geneval", "tag": "two_object", "seeds": [9, 10, 11, 12], "objects": True},
    "geneval_count": {"bench": "geneval", "tag": "counting", "seeds": [9, 10, 11, 12], "objects": True},
    "t2i_spatial": {"bench": "t2icompbench", "file": "spatial_val.txt", "seeds": list(range(9, 17)) + [59, 60],
                    "allow": {"frog", "frogs", "mouse"}, "objects": True},
    "t2i_complex": {"bench": "t2icompbench", "file": "complex_val.txt", "seeds": list(range(9, 17)) + [59, 60],
                    "allow": {"frog", "frogs"}},
    # Table 2 as decided on 2026-11-12: every set names two things, so all four methods run on it.
    "geneval_colorattr": {"bench": "geneval", "tag": "color_attr", "seeds": [9, 10, 11, 12], "objects": True},
    "geneval_position": {"bench": "geneval", "tag": "position", "seeds": [9, 10, 11, 12], "objects": True},
    "t2i_color": {"bench": "t2icompbench", "file": "color_val.txt", "seeds": [9, 10], "objects": True, "sample": 100},
    "t2i_shape": {"bench": "t2icompbench", "file": "shape_val.txt", "seeds": [9, 10], "objects": True, "sample": 100},
    "t2i_texture": {"bench": "t2icompbench", "file": "texture_val.txt", "seeds": [9, 10], "objects": True, "sample": 100, "off": True},
}
PAIRED = {"geneval_two", "geneval_colorattr", "geneval_position", "t2i_color", "t2i_shape", "t2i_texture"}
SAMPLE_SEED = 20261112  # the fixed draw of 100 two-concept prompts per T2I-CompBench subset
# Objects that share a word with a listed creature. In a set marked "objects" the phrase is set aside
# before the subject check (pair ledger, 2026-11-12), so the creature word stays blocked on its own;
# spider and web are never set aside.
OBJECT_PHRASES = re.compile(r"\b(computer mouse|baseball bats?)\b", re.I)  # the only set PoE, ours and CO3 render


def load(name: str) -> tuple[list[dict], list[dict]]:
    """(kept, dropped): each {'idx', 'prompt', 'meta'} with idx the line number in the benchmark file."""
    s = SETS[name]
    if s["bench"] == "geneval":
        lines = [json.loads(ln) for ln in GENEVAL.read_text().splitlines() if ln.strip()]
        items = [{"idx": i, "prompt": m["prompt"], "meta": m} for i, m in enumerate(lines) if m["tag"] == s["tag"]]
    else:
        lines = (T2I / s["file"]).read_text().splitlines()
        items = [{"idx": i, "prompt": p.strip(), "meta": None} for i, p in enumerate(lines) if p.strip()]
    allow = s.get("allow", set())  # ask-first words the pair ledger allows for this set

    def ok(prompt: str) -> bool:
        if s.get("objects"):
            prompt = OBJECT_PHRASES.sub(" object ", prompt)
        if is_allowed(prompt):
            return True
        return is_allowed(prompt, allow_ask_first=True) and (words(prompt) & ASK_FIRST) <= allow

    kept = [x for x in items if ok(x["prompt"])]
    dropped = [x for x in items if not ok(x["prompt"])]
    if s.get("off"):  # dropped from Table 2 on 2026-11-12, by the user
        return [], items
    if s.get("sample"):  # only prompts naming exactly two things, then a fixed random draw kept in file order
        two = [x for x in kept if x["prompt"].count(" and ") == 1]
        pick = set(x["idx"] for x in random.Random(SAMPLE_SEED).sample(two, s["sample"]))
        kept = [x for x in two if x["idx"] in pick]
    return kept, dropped


def split(item: dict) -> tuple[str, str, str]:
    """(concept a, concept b, joiner) with f"{a} {joiner} {b}" equal to the benchmark prompt."""
    m = item["meta"]
    if m and m.get("tag") == "position":  # "a photo of a dog right of a teddy bear"
        rel = m["include"][1]["position"][0]
        a, b = item["prompt"].split(f" {rel} ", 1)
        assert f"{a} {rel} {b}" == item["prompt"], item["prompt"]
        return a, b, rel
    a, b = item["prompt"].split(" and ")
    return a.strip(), b.strip(), "and"


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--counts", action="store_true")
    g.add_argument("--pairs", choices=sorted(PAIRED))
    g.add_argument("--co3-cells", choices=sorted(PAIRED))
    a = ap.parse_args()
    if a.counts:
        for name, s in SETS.items():
            kept, dropped = load(name)
            n = len(s["seeds"])
            print(f"{name:14s} prompts {len(kept) + len(dropped):3d}  dropped {len(dropped):2d}  kept {len(kept):3d}  "
                  f"x {n} images = {len(kept) * n:4d}  methods {'joint, poe, ours, co3' if name in PAIRED else 'joint'}")
        return
    name = a.pairs or a.co3_cells
    kept, _ = load(name)
    for x in kept:
        pa, pb, joiner = split(x)
        if a.pairs:
            print(f"{pa}|{pb}")
        else:
            for seed in SETS[name]["seeds"]:
                print(f"{name}|{pa}|{pb}|{joiner}|{seed}")


if __name__ == "__main__":
    main()
