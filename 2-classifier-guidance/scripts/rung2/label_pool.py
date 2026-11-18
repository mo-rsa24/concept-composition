#!/usr/bin/env python
"""Label a picture pool for the checker with the strict Claude API labeller, and apply the training rule.

Runs on the laptop (api_label.py's client; the key is never printed). Each picture of <pool>/index.json is labelled with
api_label's prompt in strict mode (a borrowed body shape, build or size also counts as a hybrid). The training rule:
    clean (label 1)      verdict "faithful pair": exactly two bodies, a clear A and a clear B, no mixing, photo intact
    reject (label 0)     verdict "hybrid", "single" or "wrong pair"
    left out             verdict "damaged" (drawings, collages, broken pictures: style must not carry the label), errors
Writes <pool>/labels.jsonl (one line per picture; a stopped run resumes) and <pool>/labelled.json, the rule applied,
and prints the counts per method and the tokens per picture.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import api_label


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    api_label.STRICT = True
    index = json.loads((args.pool / "index.json").read_text())
    items = [{"id": r["id"], "pair": r["pair"], "method": r["method"], "seed": r["seed"],
              "path": args.pool / "pictures" / f"{r['id']}.jpg"} for r in index]
    if args.limit:
        items = items[: args.limit]
    out = args.pool / "labels.jsonl"
    asyncio.run(api_label.run(items, out, args.concurrency))
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    rule = {"faithful pair": 1, "hybrid": 0, "single": 0, "wrong pair": 0}
    labelled = {r["id"]: {"pair": r["pair"], "method": r["method"], "seed": r["seed"], "verdict": r["verdict"],
                          "label": rule[r["verdict"]]} for r in rows if r.get("verdict") in rule}
    (args.pool / "labelled.json").write_text(json.dumps(labelled))
    ok = [r for r in rows if "verdict" in r]
    tin = [r["input_tokens"] for r in ok if "input_tokens" in r]
    tout = [r["output_tokens"] for r in ok if "output_tokens" in r]
    print(f"labelled {len(ok)} of {len(rows)} (errors {dict(Counter(r.get('error') for r in rows if 'verdict' not in r))})")
    if tin:
        print(f"tokens: input {sum(tin)} total, {sum(tin) / len(tin):.0f} per picture; output {sum(tout)} total")
    for m in sorted({r["method"] for r in ok}):
        c = Counter(r["verdict"] for r in ok if r["method"] == m)
        print(f"  {m:6s}: {dict(c)}")
    c = Counter(v["label"] for v in labelled.values())
    print(f"training rule: clean {c[1]}, reject {c[0]}, left out {len(ok) - len(labelled)}")
    print("LABEL POOL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
