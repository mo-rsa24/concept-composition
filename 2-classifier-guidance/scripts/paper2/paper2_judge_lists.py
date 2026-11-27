"""The judges' picture lists on the split labelled set, their GPU cost, and each judge's setting chosen on choosing pairs.

Species judge steps 0.17 to 0.20 (`paper2-judge-old`, `-clip`, `-imagenet`, `-vlm`; Detect-and-Compare is out of these
steps since step 0.15, see the hand-off), and the follow-on judges `paper2-judge-dreamsim` and `paper2-judge-bioclip`. The plan ("The judges"): every free setting is chosen on choosing pairs, from
the judge's grid, by the highest six-kind agreement, and frozen before any test pair is called; the test pairs are
called without the confirmation slice. Three subcommands, run in this order:

  lists   reads paper2_label_set.py's labelled-set.csv and split.json and writes, under --out:
            choosing.csv  every settled picture on a choosing pair (part "choosing")
            test.csv      every picture on a test pair that is read or spare (the unselected sample is among them),
                          plus crow and raven (part "apart"), reported apart; never a confirmation picture
            pictures.csv  the two together, one row per picture file, for paper2_bodies.py (renamed pictures share
                          their original's file and so its boxes)
          Each list carries picture, a, b (the judges' columns) and the labelled set's item, pair, pair_type, part,
          unselected_sample, renamed and renaming. confirm.json is never opened, and a confirmation row in either list
          stops the script.
  cost    the GPU time each judge would take on the two lists: the seconds a body each judge measured in step 0.15
          (its calls.json on the 64 single cats, `seconds` over the bodies called) times the bodies bodies.json gives
          the listed pictures. Prints the projection per judge and in all, against GPU_HOURS_STOP.
  choose  the setting a judge keeps: its calls on choosing.csv (made at the grid's middle, which stores every body's raw
          scores) re-decided at every grid point with paper2_common.redecide, no model; the point whose six-kind
          agreement with the settled kind is highest is kept, ties going to the earlier point in grid order. Writes
          settings-chosen.json (the setting, every grid point's agreement, the time) and prints the flags for the test
          call. paper2_score.py checks the test call's settings.json matches it and was written after it.

    "$PY" paper2_judge_lists.py lists --labelled-set <paper2-label-set>/labelled-set.csv \\
        --split <paper2-label-set>/split.json --out /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-lists
    "$PY" paper2_judge_lists.py cost --lists <paper2-judge-lists> --bodies <paper2-find-bodies-split>/bodies.json \\
        --check-dir /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-check
    "$PY" paper2_judge_lists.py choose --judge clip --calls <paper2-judge-clip>/choosing/calls.json \\
        --lists <paper2-judge-lists> --out <paper2-judge-clip>/settings-chosen.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_bars import GPU_HOURS_STOP  # noqa: E402

LIST_COLUMNS = ["picture", "a", "b", "item", "pair", "pair_type", "kind", "part", "unselected_sample", "renamed",
                "renaming", "call_1", "call_2"]
TEST_PARTS = ("read", "spare", "apart")
JUDGES = ("old", "clip", "imagenet", "vlm", "dreamsim", "bioclip")
OUT_OF_STEPS = {"dac": "Detect-and-Compare failed the crop check at every grid value in step 0.15 and is out of steps "
                       "0.17 to 0.21 (paper-2-handoff.md, 'Step 0.15 on mscluster112')"}


def refuse_confirm(*paths) -> None:
    for p in paths:
        if p is not None and Path(p).name == "confirm.json":
            raise SystemExit("confirm.json is sealed until owner-choice.json exists; this script never opens it")


def item_key(picture: str, renaming: str) -> str:
    return f"{picture}|renamed {renaming}" if renaming else picture


def read_labelled(path: Path) -> list[dict]:
    refuse_confirm(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["item"] = item_key(r["picture"], r.get("renaming", "") if r.get("renamed") == "True" else "")
    return rows


def make_lists(rows: list[dict], split: dict) -> tuple[list[dict], list[dict]]:
    choosing_pairs, test_pairs = set(split["choosing_pairs"]), set(split["test_pairs"])
    choosing = [r for r in rows if r["part"] == "choosing"]
    test = [r for r in rows if r["part"] in TEST_PARTS and (r["part"] == "apart" or r["pair"] in test_pairs)]
    bad = [r["item"] for r in choosing + test if r["part"] == "confirmation"]
    if bad:
        raise SystemExit(f"{len(bad)} confirmation pictures reached a list; nothing written")
    stray = [r["item"] for r in choosing if r["pair"] not in choosing_pairs]
    if stray:
        raise SystemExit(f"{len(stray)} choosing rows sit on pairs split.json does not name as choosing: {stray[:3]}")
    return choosing, test


def write_csv(path: Path, rows: list[dict], cols=LIST_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cmd_lists(a) -> int:
    rows = read_labelled(a.labelled_set)
    refuse_confirm(a.split)
    split = json.loads(Path(a.split).read_text())
    choosing, test = make_lists(rows, split)
    write_csv(a.out / "choosing.csv", choosing)
    write_csv(a.out / "test.csv", test)
    files, seen = [], set()
    for r in choosing + test:
        if r["picture"] not in seen:
            seen.add(r["picture"])
            files.append(r)
    write_csv(a.out / "pictures.csv", files, ["picture", "a", "b"])
    parts = Counter(r["part"] for r in test)
    print(f"choosing.csv: {len(choosing)} pictures on {len({r['pair'] for r in choosing})} choosing pairs, by kind "
          f"{dict(sorted(Counter(r['kind'] for r in choosing).items()))}")
    print(f"test.csv: {len(test)} pictures ({dict(sorted(parts.items()))}), "
          f"{sum(r['unselected_sample'] == 'True' for r in test)} in the unselected sample; no confirmation picture")
    print(f"pictures.csv: {len(files)} picture files for paper2_bodies.py")
    return 0


def seconds_per_body(calls_path: Path) -> float:
    c = json.loads(Path(calls_path).read_text())
    bodies = sum(max(p.get("n_bodies", len(p.get("bodies", []))), 1) for p in c["pictures"])
    return c["seconds"] / bodies


def cmd_cost(a) -> int:
    from paper2_common import load_bodies
    bodies = load_bodies(a.bodies)
    lists = {name: read_csv(a.lists / f"{name}.csv") for name in ("choosing", "test")}
    missing = sorted({r["picture"] for rows in lists.values() for r in rows if r["picture"] not in bodies})
    if missing:
        raise SystemExit(f"{len(missing)} listed pictures have no boxes in {a.bodies}, for example {missing[:2]}")
    n_bodies = {name: sum(max(len(bodies[r["picture"]]), 1) for r in rows) for name, rows in lists.items()}
    total = 0.0
    print(f"bodies to call: choosing {n_bodies['choosing']}, test {n_bodies['test']}")
    for judge in JUDGES:
        calls = a.check_dir / judge / "calls.json"
        if not calls.exists():
            print(f"  {judge:9s} no step 0.15 calls at {calls}: no measured rate")
            continue
        spb = seconds_per_body(calls)
        hours = spb * (n_bodies["choosing"] + n_bodies["test"]) / 3600
        total += hours
        print(f"  {judge:9s} {spb:.2f} seconds a body (step 0.15) -> choosing {spb * n_bodies['choosing'] / 3600:.2f} h, "
              f"test {spb * n_bodies['test'] / 3600:.2f} h, in all {hours:.2f} GPU-hours")
    print(f"all judges: {total:.2f} GPU-hours against GPU_HOURS_STOP {GPU_HOURS_STOP} "
          f"({'under' if total <= GPU_HOURS_STOP else 'OVER: stop for the owner'})")
    return 0


def grid_of(judge: str) -> tuple[list[str], list[tuple]]:
    if judge == "clip":
        import paper2_judge_clip as m
        return ["margin", "flag_share"], list(itertools.product(m.MARGIN_GRID, m.FLAG_SHARE_GRID))
    if judge == "imagenet":
        import paper2_judge_imagenet as m
        return ["margin", "flag_share"], list(itertools.product(m.MARGIN_GRID, m.FLAG_SHARE_GRID))
    if judge == "vlm":
        import paper2_judge_vlm as m
        return ["yes_threshold"], [(t,) for t in m.YES_THRESHOLD_GRID]
    if judge == "bioclip":
        import paper2_judge_clip as m                 # BioCLIP runs on the CLIP judge's grids
        return ["margin", "flag_share"], list(itertools.product(m.MARGIN_GRID, m.FLAG_SHARE_GRID))
    if judge == "dreamsim":
        import paper2_judge_dreamsim as m
        return ["margin", "flag_gap", "floor"], list(itertools.product(m.MARGIN_GRID, m.FLAG_GAP_GRID, m.FLOOR_GRID))
    raise SystemExit(OUT_OF_STEPS.get(judge, f"{judge} has no free setting to choose (the old judge runs as committed)"))


def deciding_judge(judge: str):
    """A judge object whose decide and two_clear_score work with no model (its stand-in is never called)."""
    ns = argparse.Namespace(dry_run=True, device="cpu", margin=0.15, flag_share=0.25, yes_threshold=0.5,
                            class_table=None, flag_gap=0.10, floor=0.4, references=None, n_refs=8)
    if judge == "clip":
        from paper2_judge_clip import ClipJudge
        return ClipJudge(ns)
    if judge == "imagenet":
        from paper2_judge_imagenet import ImageNetJudge
        return ImageNetJudge(ns)
    if judge == "bioclip":
        from paper2_judge_bioclip import BioClipJudge
        return BioClipJudge(ns)
    if judge == "dreamsim":
        from paper2_judge_dreamsim import DreamSimJudge
        return DreamSimJudge(ns)
    from paper2_judge_vlm import VlmJudge
    return VlmJudge(ns)


FLAGS = {"margin": "--margin", "flag_share": "--flag-share", "yes_threshold": "--yes-threshold",
         "flag_gap": "--flag-gap", "floor": "--floor"}


def choose(judge: str, calls: dict, choosing: list[dict]) -> dict:
    from paper2_common import redecide
    names, points = grid_of(judge)
    settled = {(r["picture"], r["a"], r["b"]): r["kind"] for r in choosing}
    pics = [p for p in calls["pictures"] if (p["picture"], p["a"], p["b"]) in settled]
    if len(pics) != len(settled):
        raise SystemExit(f"the choosing calls hold {len(pics)} of the {len(settled)} choosing pictures")
    stray = [p["picture"] for p in calls["pictures"] if (p["picture"], p["a"], p["b"]) not in settled]
    if stray:
        raise SystemExit(f"the choosing calls hold {len(stray)} pictures that are not on choosing pairs: {stray[:2]}")
    jd = deciding_judge(judge)
    base = dict(calls["settings"])
    scores = []
    for point in points:
        s = {**base, **dict(zip(names, point))}
        again = redecide(jd, pics, s)
        right = sum(p["kind"] == settled[(p["picture"], p["a"], p["b"])] for p in again)
        scores.append({"setting": dict(zip(names, point)), "right": right, "of": len(again),
                       "agreement": round(right / len(again), 4)})
    best = max(range(len(scores)), key=lambda i: (scores[i]["right"], -i))
    return {"judge": judge, "chosen": scores[best]["setting"], "agreement": scores[best]["agreement"],
            "grid": scores, "rule": "highest six-kind agreement on choosing pairs; ties to the earlier grid point",
            "choosing_pictures": len(pics), "written": time.strftime("%Y-%m-%dT%H:%M:%S")}


def cmd_choose(a) -> int:
    refuse_confirm(a.calls, a.out)
    calls = json.loads(Path(a.calls).read_text())
    result = choose(a.judge, calls, read_csv(a.lists / "choosing.csv"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    for g in result["grid"]:
        print(f"  {g['setting']}: {g['right']} of {g['of']} ({g['agreement']:.3f})")
    flags = " ".join(f"{FLAGS[k]} {v}" for k, v in result["chosen"].items())
    print(f"{a.judge}: chose {result['chosen']} at {result['agreement']:.3f} on {result['choosing_pictures']} choosing "
          f"pictures; wrote {a.out}\ntest call flags: {flags}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("lists")
    s.add_argument("--labelled-set", type=Path, required=True)
    s.add_argument("--split", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("cost")
    s.add_argument("--lists", type=Path, required=True)
    s.add_argument("--bodies", type=Path, required=True)
    s.add_argument("--check-dir", type=Path, required=True, help="step 0.15's folder: <judge>/calls.json")
    s = sub.add_parser("choose")
    s.add_argument("--judge", required=True, choices=list(JUDGES) + list(OUT_OF_STEPS))
    s.add_argument("--calls", type=Path, required=True, help="the judge's calls.json on choosing.csv")
    s.add_argument("--lists", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True, help="settings-chosen.json")
    a = ap.parse_args(argv)
    return {"lists": cmd_lists, "cost": cmd_cost, "choose": cmd_choose}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
