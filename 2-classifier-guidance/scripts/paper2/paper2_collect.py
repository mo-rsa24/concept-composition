"""Collect the candidate pictures for the species judge's labelled set (step 0.4, `paper2-collect`).

The sources are the six of plans/paper-2-dials/plans/02-species-judge.md, "The labelled check set", each read from its
own committed record (the uncommitted ones are copied into records/ beside this file; see records/README.md). Every
picture is tied to its pair and seed by a record, never by how it looks or where its folder sits:

  m1        CO3 alone on its twelve borrowing pairs, seeds 6001 to 6008 (96; plan: 96, 88 without the
            lighthouse); artifacts/notes/progress/noise-text-critic/02d-binding/part-binding-calls.csv
  joint     SDXL on the joint prompt, the same pairs, seeds 6001 to 6016 (192); records/m14-key.json and the calls
  box       Loop 12's rows on CO3 (867): rounds 03 (/datasets loop-12 z2 for CO3 alone, z3 for its pass rows), 07 (z2,
            z7), 08 (z8 cases), 12 and 13 (half and rest); artifacts/notes/progress/leakage/<round>/blind/
  dial      the pooled LoRA's weight dial (200); artifacts/notes/progress/leakage/10-co-occurrence-dial/blind/
  binding   the LoRA binding check, option A (128); 1-lora-correction/binding/blind/
  old       the old species judge's 29 renders; records/old-judge-labels.json

Paths, rows and methods are from each round's own scripts (named beside each reader), mapped on 2026-11-26. No picture
in these sources was drawn by a push whose checker read CLIP or ImageNet features: the only checker push (round 08's
push-k) reads SDXL's U-Net readout features, and CLIP only named which detected box was which animal in round 12's
`box` row and round 13's `own` row (`clip_named_boxes`). Two called sets the plan's table leaves out are not collected:
M1's push and random rows, and M12's fresh CO3 pictures.

Left out, and counted: the lighthouse and sailboat (not animals; the body counter asks for "animal"), any pair the
subject guard refuses, and the old judge's cat and dog cell at seed 9, step 10000, which its finding records as having
failed the subject check (3-understanding-compositionality/report/what-sdxl-already-knows/can-a-score-see-species.md,
line 22). Pictures called in more than one round (round 03's and 07's CO3 rows are the same 48 files; the binding
check's plain and lora rows are the dial's weight 0 and 1 files) are kept once, on the first record, with every other
call kept in `also_called_in`.

The 100 unselected pictures: drawn by seed 2 (paper2_bars.UNSELECTED_SEED) from every CO3-alone picture (the M1 row,
the box rounds' co3 rows and round 08's co3 without a push) with no regard to earlier calls, and marked `unselected`.

Writes candidates.csv (one row per distinct picture: picture, a, b, then the record's columns, readable by
paper2_sheets.py and every judge) and counts.txt (per source against the plan, per pair, per pair type, per earlier
call). --check, on the node, adds whether each file opens at 1024 by 1024, and the step's pass line: at least 1,000
candidates open, and the earlier calls give at least 80 hybrid and 150 two-clear candidates.

    python3 paper2_collect.py --out outputs/paper2/paper2-collect                    # laptop: records only
    "$PY" paper2_collect.py --check --out /datasets/mmolefe/poe_repair_min/paper2/paper2-collect   # on the node
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RECORDS = HERE / "records"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts/shared"))

from paper2_bars import REPORTED_APART, UNSELECTED_SEED, UNSELECTED_TEST, pair_type  # noqa: E402
from paper2_common import PAPER2_DATASETS, species_from_slug  # noqa: E402

ROOT = Path("/datasets/mmolefe/poe_repair_min")
M1 = ROOT / "loop-6/m1/paper-2/full"                       # leakage/z2.sbatch line 14
JOINT = ROOT / "paper2/paper2-upload-joint"                # environment/storage.md, the upload of 2026-11-26
LOOP12 = ROOT / "loop-12"
DIAL = LOOP12 / "z10/full/renders"                         # z10_dial.py line 166, z10.sbatch line 14
BINDING = ROOT / "lora-binding/option-a"                   # binding/option_a.py line 154
ABLATION = ROOT / "outputs/ablation_grid"                  # 3-understanding-compositionality/scripts/sweep_probe.sbatch line 21
LEAKAGE = REPO / "artifacts/notes/progress/leakage"

PLAN_COUNTS = {"m1": 96, "joint": 192, "box": 867, "dial": 200, "binding": 128, "old": 29}
NOT_ANIMALS = {"a_lighthouse__x__a_sailboat"}
SUBJECT_CHECK_FAILED = {("old", "a_cat__x__a_dog", 9, "v58-05-self-contrast", 10000)}
TWO_CLEAR_LIKE = {"two_clear", "own_parts", "composed"}
COLUMNS = ["picture", "a", "b", "pair", "pair_type", "seed", "source", "round", "row", "made", "earlier_call",
           "earlier_scale", "notes", "push_checker", "clip_named_boxes", "reported_apart", "unselected",
           "also_called_in"]

FOUR_STEP = "four-step (one, hybrid, two_bodies, two_clear)"
CO3_ALONE = "CO3 alone at its released settings (CFG++ 0.8, 50 steps, 6 corrected steps)"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def notes_by_id(*files: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in files:
        if f.exists():
            for r in read_csv(f):
                if r.get("note"):
                    out.setdefault(r["id"], []).append(r["note"].strip())
    return out


def settle(folder: Path) -> dict[str, str]:
    """The common settling rule: reviewer 1's rung where reviewers 1 and 2 agree, else the third call, else split."""
    one = {r["id"]: r["rung"] for r in read_csv(folder / "calls-1.csv")}
    two = {r["id"]: r["rung"] for r in read_csv(folder / "calls-2.csv")}
    three = {r["id"]: r["rung"] for r in read_csv(folder / "calls-3.csv")} if (folder / "calls-3.csv").exists() else {}
    return {i: (one[i] if one[i] == two.get(i) else three.get(i, "split")) for i in one}


def row(picture, pair, seed, source, round_, row_, made, call, scale, notes="", checker="none", clip_boxes=False):
    a, b = species_from_slug(pair)
    return {"picture": str(picture), "a": a, "b": b, "pair": pair, "seed": int(seed), "source": source,
            "round": round_, "row": row_, "made": made, "earlier_call": call, "earlier_scale": scale,
            "notes": " | ".join(notes) if isinstance(notes, list) else notes, "push_checker": checker,
            "clip_named_boxes": clip_boxes}


# ---------------------------------------------------------------- the six readers

def m1() -> list[dict]:
    f = REPO / "artifacts/notes/progress/noise-text-critic/02d-binding/part-binding-calls.csv"
    return [row(M1 / r["picture"], r["pair"], r["seed"], "m1", "m1", "co3", CO3_ALONE, r["final"],
                "binding (own_parts, borrowed, not_two)", [n for n in (r["note_1"], r["note_2"]) if n])
            for r in read_csv(f)]


def joint() -> list[dict]:
    key = json.loads((RECORDS / "m14-key.json").read_text())
    final = {r["id"]: r["rung"] for r in read_csv(RECORDS / "m14-calls-final.csv")}
    notes = notes_by_id(*(RECORDS / f"m14-calls-{w}.csv" for w in ("first", "second", "third")))
    made = "SDXL on the joint prompt alone, CFG 7.5, 50 DDIM steps, from CO3's starting noise for the seed"
    return [row(JOINT / f"{i}.png", k["pair"], k["seed"], "joint", "m14", "joint", made, final[i],
                "two-step (two_bodies, two_clear)", notes.get(i, [])) for i, k in sorted(key.items())]


BOX_MADE = {
    ("03", "co3"): CO3_ALONE,
    ("03", "whole"): "CO3 plus one extra corrector pass per step at steps 15 to 30, CO3's weights everywhere",
    ("03", "boxes"): "the same pass, with twice the joint prompt's guess minus the other animal's inside each eye-drawn box",
    ("07", "co3"): CO3_ALONE,
    ("07", "whole"): "CO3 plus one extra whole-picture corrector pass at steps 5 to 30",
    ("07", "own"): "CO3 with each animal's own prediction inside its eye-drawn box, steps 5 to 30",
    ("12", "co3"): CO3_ALONE,
    ("12", "box"): "CO3 pushed at steps 5 to 29 toward each animal's own prediction inside its detected box, cap 0.4",
    ("13", "co3"): CO3_ALONE,
    ("13", "whole"): "CO3 plus one extra whole-picture corrector pass at steps 5 to 30",
    ("13", "own"): "CO3 with each animal's own prediction inside its detected box, steps 5 to 30",
}
BASE_MADE = {"co3": "CO3", "poe": "the plain product of experts on SDXL", "joint": "SDXL on the joint prompt"}
READOUT = "readout head on SDXL U-Net features (rung2 readout-multipair head_step_002000.pt)"


def box() -> list[dict]:
    out = []
    for round_, folder, z in (("03", "03-one-fix", "z3"), ("07", "07-own-prediction", "z7")):
        d = LEAKAGE / folder / "blind"      # z3_blind.py / z7_blind.py lines 35 and 37
        key = json.loads((d / "key.json").read_text())
        rungs, notes = settle(d), notes_by_id(*(d / f"calls-{n}.csv" for n in (1, 2, 3)))
        for i, k in sorted(key.items()):
            pic = (LOOP12 / "z2/full" / k["pair"] / f"seed-{k['seed']}.png" if k["row"] == "co3"
                   else LOOP12 / z / "full" / k["row"] / k["pair"] / f"seed-{k['seed']}.png")
            out.append(row(pic, k["pair"], k["seed"], "box", round_, k["row"], BOX_MADE[(round_, k["row"])], rungs[i],
                           FOUR_STEP, notes.get(i, [])))
    d = LEAKAGE / "08-first-correction-test/blind"   # z8_blind.py line 37, z8_cases.py lines 52 to 56
    key = json.loads((d / "key.json").read_text())
    rungs, notes = settle(d), notes_by_id(*(d / f"calls-{n}.csv" for n in (1, 2, 3)))
    for i, k in sorted(key.items()):
        arm = k["arm"]
        if arm == "without":
            made, checker = f"{BASE_MADE[k['base']]}, no push", "none"
        elif arm.startswith("push"):
            made, checker = f"{BASE_MADE[k['base']]} with the readout checker's push (strength 30, steps 0 to 15, cap 0.4), draw {arm[-1]}", READOUT
        else:
            made, checker = f"{BASE_MADE[k['base']]} with a random push of the same size, draw {arm[-1]}", "none"
        out.append(row(LOOP12 / "z8/cases" / k["base"] / k["pair"] / f"seed-{k['seed']}" / f"{arm}.png", k["pair"],
                       k["seed"], "box", "08", f"{k['base']}/{arm}", made, rungs[i], FOUR_STEP, notes.get(i, []), checker))
    for round_, folder, z in (("12", "12-aligned-classifier-on-co3", "z12"), ("13", "13-clean-up-pass-in-boxes", "z13")):
        for half in ("half", "rest"):        # z12_box_push.py lines 290 to 300; z13_pass.py lines 77 to 84
            d = LEAKAGE / folder / "blind" / half
            settled = json.loads((d / "settled.json").read_text())["settled"]
            notes = notes_by_id(*(d / f"calls-{n}.csv" for n in (1, 2, 3)))
            for i, s in sorted(settled.items()):
                out.append(row(LOOP12 / z / s["row"] / s["pair"] / f"seed-{s['seed']}.png", s["pair"], s["seed"], "box",
                               f"{round_}-{half}", s["row"], BOX_MADE[(round_, s["row"])], s["rung"], FOUR_STEP,
                               notes.get(i, []), clip_boxes=(round_, s["row"]) in {("12", "box"), ("13", "own")}))
    return out


DIAL_MADE = ("the plain product of experts (CFG 7.5 per expert, 50 DDIM steps) with the pooled LoRA v58-03-self "
             "(step 30000) at weight {w:.2f}; weight 0 is the plain product")


def dial() -> list[dict]:
    out = []
    # Only folders: before Python 3.11, glob("*/") also returns files (the blind folder holds bar.json).
    for d in sorted(p for p in (LEAKAGE / "10-co-occurrence-dial/blind").iterdir() if p.is_dir()):
        settled = json.loads((d / "settled.json").read_text())["settled"]
        notes = notes_by_id(*(d / f"calls-{n}.csv" for n in (1, 2, 3)))
        for i, s in sorted(settled.items()):
            w = float(s["weight"])
            out.append(row(DIAL / s["pair"] / f"seed-{s['seed']}-w{w:.2f}.png", s["pair"], s["seed"], "dial", d.name,
                           f"w{w:.2f}", DIAL_MADE.format(w=w), s["rung"], FOUR_STEP, notes.get(i, [])))
    return out


BINDING_MADE = {"plain": "the plain product of experts, no adapter",
                "lora": "the plain product with the pooled LoRA at weight 1",
                "own": "the plain product with each animal's own prediction inside its eye-drawn box, steps 5 to 30",
                "both": "the pooled LoRA plus each animal's own prediction inside its eye-drawn box"}


def binding() -> list[dict]:
    d = REPO / "1-lora-correction/binding/blind"        # binding/option_a_blind.py lines 30 to 35
    settled = json.loads((d / "settled.json").read_text())["settled"]
    notes = notes_by_id(*(d / f"calls-{n}.csv" for n in (1, 2, 3)))
    out = []
    for i, s in sorted(settled.items()):
        r = s["row"]
        pic = (DIAL / s["pair"] / f"seed-{s['seed']}-w{1.0 if r == 'lora' else 0.0:.2f}.png" if r in ("plain", "lora")
               else BINDING / r / s["pair"] / f"seed-{s['seed']}.png")
        out.append(row(pic, s["pair"], s["seed"], "binding", "option-a", r, BINDING_MADE[r], s["rung"], FOUR_STEP,
                       notes.get(i, [])))
    return out


def old() -> list[dict]:
    cells = json.loads((RECORDS / "old-judge-labels.json").read_text())["cells"]
    out = []
    for c in cells:                                    # species_judge.py line 126
        pic = ABLATION / c["slug"] / f"seed{c['seed']:02d}" / f"ours_{c['run']}_s{c['step']:06d}_w50.png"
        r = row(pic, c["slug"], c["seed"], "old", "can-a-score-see-species", f"{c['run']} step {c['step']}",
                "the pooled-LoRA adapter correction (the ablation grid's ours column)", c["verdict"],
                "by eye at full size (composed, two_of_one, hybrid, one_body)",
                "" if c["confident"] else "the eye was unsure")
        r["_key"] = ("old", c["slug"], c["seed"], c["run"], c["step"])
        out.append(r)
    return out


READERS = {"m1": m1, "joint": joint, "box": box, "dial": dial, "binding": binding, "old": old}


# ---------------------------------------------------------------- the collection

def collect() -> tuple[list[dict], dict]:
    from disallowed_subjects import is_allowed
    report = {"sources": {}, "left_out": Counter(), "merged": 0}
    kept: dict[str, dict] = {}
    for name, reader in READERS.items():
        rows = reader()
        report["sources"][name] = {"resolved": len(rows), "plan": PLAN_COUNTS[name]}
        animal = 0
        for r in rows:
            if r["pair"] in NOT_ANIMALS:
                report["left_out"]["not animals (lighthouse and sailboat)"] += 1
                continue
            animal += 1
            a, b = r["a"], r["b"]
            if not is_allowed(f"a {a} and a {b}"):
                report["left_out"]["refused by the subject guard"] += 1
                continue
            if r.pop("_key", None) in SUBJECT_CHECK_FAILED:
                report["left_out"]["its finding records it failed the subject check"] += 1
                continue
            if r["picture"] in kept:
                first = kept[r["picture"]]
                first.setdefault("_also", []).append(f"{r['source']}/{r['round']}/{r['row']}: {r['earlier_call']}")
                report["merged"] += 1
                continue
            r["pair_type"] = pair_type(a, b)
            r["reported_apart"] = (a, b) in REPORTED_APART
            r["unselected"] = False
            kept[r["picture"]] = r
        report["sources"][name]["animal"] = animal
    rows = list(kept.values())
    for r in rows:
        r["also_called_in"] = " ; ".join(r.pop("_also", []))
    pool = sorted(r["picture"] for r in rows if r["made"] == CO3_ALONE or r["made"] == "CO3, no push")
    for pic in random.Random(UNSELECTED_SEED).sample(pool, UNSELECTED_TEST):
        kept[pic]["unselected"] = True
    report["co3_alone_pool"] = len(pool)
    return rows, report


def check_files(rows: list[dict]) -> Counter:
    from PIL import Image
    seen = Counter()
    for r in rows:
        p = Path(r["picture"])
        if not p.exists():
            r["opens_at_1024"] = "missing"
        else:
            try:
                with Image.open(p) as im:
                    r["opens_at_1024"] = "yes" if im.size == (1024, 1024) else f"size {im.size[0]}x{im.size[1]}"
            except Exception as e:
                r["opens_at_1024"] = f"unreadable ({type(e).__name__})"
        seen[r["opens_at_1024"] if r["opens_at_1024"] in ("yes", "missing") else "other"] += 1
    return seen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=PAPER2_DATASETS / "paper2-collect")
    ap.add_argument("--check", action="store_true", help="on the node: does each file exist and open at 1024 by 1024")
    a = ap.parse_args(argv)
    rows, report = collect()
    lines = ["per source: resolved from its records against the plan's count, then on animal pairs"]
    for name, s in report["sources"].items():
        flag = "matches" if s["resolved"] == s["plan"] else f"SHORT BY {s['plan'] - s['resolved']}"
        lines.append(f"  {name:8s} {s['resolved']:4d} of the plan's {s['plan']:4d} ({flag}); {s['animal']} on animal pairs")
    lines.append(f"left out: {dict(report['left_out'])}")
    lines.append(f"called in more than one round, kept once: {report['merged']}")
    lines.append(f"distinct candidate pictures: {len(rows)}; unselected CO3-alone drawn: "
                 f"{sum(r['unselected'] for r in rows)} of {report['co3_alone_pool']} by seed {UNSELECTED_SEED}")
    calls = Counter(r["earlier_call"] for r in rows)
    hybrid, two_clear = calls["hybrid"], sum(calls[c] for c in TWO_CLEAR_LIKE)
    lines.append(f"earlier calls: {dict(sorted(calls.items()))}")
    lines.append(f"  hybrid candidates {hybrid} (need 80); two-clear candidates {two_clear} (two_clear, own_parts and "
                 f"composed; need 150)")
    lines.append(f"per pair type: {dict(Counter(r['pair_type'] for r in rows))}; reported apart (crow and raven): "
                 f"{sum(r['reported_apart'] for r in rows)}")
    lines.append(f"push checkers: {dict(Counter(r['push_checker'] for r in rows))}; boxes named by CLIP: "
                 f"{sum(r['clip_named_boxes'] for r in rows)}; read CLIP or ImageNet features: 0")
    ok = hybrid >= 80 and two_clear >= 150
    columns = list(COLUMNS)
    if a.check:
        seen = check_files(rows)
        columns.append("opens_at_1024")
        lines.append(f"on disk: {dict(seen)}")
        ok = ok and seen["yes"] >= 1000
        lines.append(f"step 0.4 {'passes' if ok else 'does NOT pass'}: {seen['yes']} open at 1024 (need 1000), "
                     f"{hybrid} hybrid (need 80), {two_clear} two-clear (need 150)")
    else:
        lines.append("files not checked: run with --check on the node")
    lines.append("per pair: " + ", ".join(f"{p} {n}" for p, n in sorted(Counter(r['pair'] for r in rows).items())))
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["source"], r["pair"], r["seed"], r["round"], r["row"])))
    (a.out / "counts.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {a.out / 'candidates.csv'} ({len(rows)} rows) and {a.out / 'counts.txt'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
