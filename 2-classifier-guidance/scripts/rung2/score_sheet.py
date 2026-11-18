#!/usr/bin/env python
"""Score a push sheet with the Claude API labeller, so a verdict can rest on 100+ seeds instead of eye calls on 8 to 32.

Runs on the laptop (api_label.py's client and key; the key is never printed). Labels every tile of the chosen columns
(seedNNNN__<column>.png, as render_results.py names them) with api_label's prompt for the pair, and reports per column:
    two bodies       the labeller counts 2 or more separate main bodies (eye call: "separate")
    faithful pair    exactly two bodies, a clear A and a clear B, no mixing, photo intact (eye call: "clean")
With --calls (a JSON of eye calls, such as calls_plan29.json), it also reports agreement per seed with the eye calls,
which is the check that must pass before this scorer stands in for eye calls. Token use per picture is reported, so the
cost of a larger sheet can be stated before it runs.

Writes the JSONL given by --out (one line per tile; a stopped run resumes).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

import api_label


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", type=Path, required=True, help="a folder of seedNNNN__<column>.png tiles")
    ap.add_argument("--pair", required=True, help="e.g. a_cat__x__a_dog")
    ap.add_argument("--columns", nargs="+", default=["plain", "s30", "r30"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calls", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="label only the first N tiles (a cost probe)")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    api_label.STRICT = args.strict
    items = []
    for p in sorted(args.sheet.glob("seed*__*.png")):
        m = re.match(r"seed(\d+)__(.+)$", p.stem)
        if m and m.group(2) in args.columns:
            items.append({"id": p.stem, "pair": args.pair, "seed": int(m.group(1)), "column": m.group(2), "path": p})
    if args.limit:
        items = items[: args.limit]
    asyncio.run(api_label.run(items, args.out, args.concurrency))

    rows = [json.loads(l) for l in args.out.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "verdict" in r]
    errs = Counter(r.get("error") for r in rows if "verdict" not in r)
    tok_in = [r["input_tokens"] for r in ok if "input_tokens" in r]
    tok_out = [r["output_tokens"] for r in ok if "output_tokens" in r]
    print(f"labelled {len(ok)} tiles, errors {dict(errs)}")
    if tok_in:
        print(f"tokens per tile: input {sum(tok_in) / len(tok_in):.0f}, output {sum(tok_out) / len(tok_out):.0f} "
              f"(over {len(tok_in)} tiles)")
    for col in args.columns:
        rc = [r for r in ok if r["column"] == col]
        if not rc:
            continue
        two = sum(r["bodies"] >= 2 for r in rc)
        fp = sum(r["verdict"] == "faithful pair" for r in rc)
        print(f"{col:8s}: {len(rc)} tiles; two bodies {two}; faithful pair {fp}; verdicts {dict(Counter(r['verdict'] for r in rc))}")

    if args.calls:
        calls = json.loads(args.calls.read_text())
        eye = calls.get("s30", {})
        api = {str(r["seed"]): r for r in ok if r["column"] == "s30"}
        both = [s for s in eye if s in api]
        sep_eye = {s for s in both if eye[s] != "not_separate"}
        sep_api = {s for s in both if api[s]["bodies"] >= 2}
        clean_eye = {s for s in both if eye[s] in ("clean", "near")}
        clean_api = {s for s in both if api[s]["verdict"] == "faithful pair"}
        agree = lambda a, b: sum((s in a) == (s in b) for s in both)
        print(f"pushed column, {len(both)} seeds against eye calls:")
        print(f"  separate: eye {len(sep_eye)}, API {len(sep_api)}, agree on {agree(sep_eye, sep_api)} of {len(both)}")
        print(f"  clean or near-clean (eye) against faithful pair (API): eye {len(clean_eye)}, API {len(clean_api)}, "
              f"agree on {agree(clean_eye, clean_api)} of {len(both)}")
        for s in sorted(both):
            if (s in clean_eye) != (s in clean_api) or (s in sep_eye) != (s in sep_api):
                print(f"    {s}: eye {eye[s]}; API {api[s]['verdict']}, bodies {api[s]['bodies']}: {api[s]['note']}")
        for col, key in (("plain", "plain_separate"), ("r30", "r30_separate")):
            rc = {str(r["seed"]): r for r in ok if r["column"] == col}
            if rc:
                e = set(calls.get(key, []))
                a = {s for s, r in rc.items() if r["bodies"] >= 2}
                print(f"  {col}: two bodies eye {len(e)}, API {len(a)}, agree on {sum((s in e) == (s in a) for s in rc)} of {len(rc)}")
    print("SCORE SHEET DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
