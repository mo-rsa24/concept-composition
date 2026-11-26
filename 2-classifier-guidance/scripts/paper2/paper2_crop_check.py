"""Read the species judge's crop check (step 0.15): does each judge name the 64 single cats as cats?

Every judge runs on the 64 single-cat renders (paper2_cats_64.csv), asked "a cat" (A) against "a dog" (B), at the
middle of its grid. This reads each judge's calls.json and writes cats.json (`paper2-judge-check`).

The bar, CROP_CHECK_MIN (62 of 64, paper2_bars.py), is judged per picture: a picture counts as named cat when the
counter found at least one body and every body found is called A with no flag. Beside it, as Loop 18 asked on
2026-11-26: the share of all crops called A, and the bodies-per-picture histogram, so a cat split into two boxes shows
as two boxes both called cat rather than as a miss. A judge under the bar is wired wrong and is not read further until
fixed (species judge plan, "What each result means").

    python3 paper2_crop_check.py --calls clip=<dir>/calls.json imagenet=<dir>/calls.json ... --out <dir>/cats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_bars import CROP_CHECK_MIN  # noqa: E402

EXPECTED_PICTURES = 64


def read_judge(calls: dict) -> dict:
    pictures = calls["pictures"]
    named, misses, crops_a, crops = 0, [], 0, 0
    hist = Counter()
    for p in pictures:
        bodies = p["bodies"]
        hist[len(bodies)] += 1
        crops += len(bodies)
        crops_a += sum(b["call"] == "A" for b in bodies)
        if bodies and all(b["call"] == "A" and not b["flag"] for b in bodies):
            named += 1
        else:
            misses.append({"picture": p["picture"], "calls": [b["call"] + ("*" if b["flag"] else "") for b in bodies],
                           "kind": p.get("kind")})
    return {"pictures": len(pictures), "named_cat": named, "passes": named >= CROP_CHECK_MIN,
            "crops": crops, "crops_called_cat": crops_a, "crop_share_cat": round(crops_a / crops, 4) if crops else None,
            "bodies_per_picture": {str(k): hist[k] for k in sorted(hist)}, "settings": calls.get("settings"),
            "dry_run": calls.get("dry_run", False), "misses": misses}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calls", nargs="+", required=True, metavar="JUDGE=CALLS.JSON")
    ap.add_argument("--out", type=Path, required=True, help="cats.json")
    a = ap.parse_args(argv)
    result = {"bar": f"at least {CROP_CHECK_MIN} of {EXPECTED_PICTURES} pictures with every body called cat, no flag",
              "judges": {}}
    for item in a.calls:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--calls takes judge=path, not {item!r}")
        r = read_judge(json.loads(Path(path).read_text()))
        if r["pictures"] != EXPECTED_PICTURES:
            print(f"note: {name} read {r['pictures']} pictures, not {EXPECTED_PICTURES}")
        result["judges"][name] = {"calls": path, **r}
    print(f"{'judge':10s} {'named cat':>10s} {'bar':>5s}  {'crops called cat':>18s}  bodies per picture")
    for name, r in result["judges"].items():
        share = f"{r['crops_called_cat']}/{r['crops']} ({r['crop_share_cat']:.3f})" if r["crops"] else "no crops"
        print(f"{name:10s} {r['named_cat']:>4d} of {r['pictures']:<3d} {'pass' if r['passes'] else 'MISS':>5s}  "
              f"{share:>18s}  {r['bodies_per_picture']}{'  (dry run)' if r['dry_run'] else ''}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {a.out}")
    return 0 if all(r["passes"] for r in result["judges"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
