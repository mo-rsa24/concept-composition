#!/usr/bin/env python3
"""Paper 1's first result: its bars as named constants, and the reader that turns the settled blind calls into the answer.

The constants are plans/paper-1-clevr-touching/plans/02-first-result.md, "The bars", line for line; the rule is the verdict file's,
plans/paper-1-clevr-touching/review/02-first-result.md, "The question written before the run". A value exactly at a bar meets it.
Committed before the first result's first job, so no bar can move after an answer is seen without showing in a diff. Needs no torch.

The reader (task 16 of the first result):
  1. settles the calls with paper1_sheets.settle: field by field, the two reviewers where they agree, the third where they do not;
  2. joins each settled picture to its row, seed, count and side through the sheet's key;
  3. calls a picture right: on a touching seed, red count and blue count both the asked count, no blended band, occlusion correct;
     on an apart seed, both counts right and no blended band; a real scene is read as a touching picture;
  4. counts right per row per count and pooled on the touching seeds; ours and mask paste on the apart seeds; the real scenes;
  5. pairs ours with the region sampler seed by seed: the 2 by 2 table, the gain (ours right and region wrong, minus the reverse),
     and the one-sided exact binomial p on the disagreeing seeds (the paired McNemar test); the random push minus the region sampler;
  6. answers yes, no or inconclusive by the rule, and writes answer.json, answer.txt and the 8-seed picture strip.

The weight and window (task 13): choose-w reads the owner's calls on the tuning sheet and contact.json, and writes
chosen-setting.json; it exits 3 when no setting keeps contact within FIRST_CONTACT_DROP_MAX of the region sampler.

Usage (laptop, after the calls are back):
  python paper1_first_bars.py read --run paper1-first-read --sheet paper1-first-eye-sheet --calls calls-a.csv calls-b.csv --third calls-c.csv
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------------------------------------------------------------------------------
# The bars (02-first-result.md, "The bars")

FIRST_TOUCH_SEEDS = 96              # touching evaluation seeds called per row, 32 per object count
FIRST_APART_SEEDS = 96              # apart evaluation seeds called for ours and mask paste
FIRST_GAIN_MIN = 15                 # seeds: ours right minus the region sampler right, on the 96 touching seeds, for a yes
FIRST_P_MAX = 0.05                  # the one-sided paired McNemar p must be below this for a yes
FIRST_RANDOM_BAND = 5               # seeds: the random push right minus the region sampler right within -5 to +5 for a yes
FIRST_NULL_GAIN_MAX = 5             # seeds: a gain at or below this is a no
FIRST_REAL_RIGHT_MIN = 82           # unused, kept for the record: the bar as first written, 82 of 96 real touching scenes (0.85 of 96).
                                    # FIRST_REAL_RIGHT_SHARE_MIN replaced it on 2026-11-27 by the owner's decision, because the blind
                                    # sheet held 76 real touching scenes, not 96, so 82 could never be met
FIRST_REAL_RIGHT_SHARE_MIN = 0.85   # of the real touching scenes on the sheet, at least this share right (ceil(0.85 x n): 82 of 96,
                                    # 65 of 76), or the rubric cannot be read; the full plan's OCCLUSION_REAL_MIN
FIRST_PUSH_MIN_GREY = 1             # step 10: the push's median change at w = 1, grey levels on 0 to 255; below it the plan stops
FIRST_DIRECTION_COS_MIN = 0.10      # step 10: the push's median cosine with the denoising target on the 64 training scenes; under it in
                                    # both of the two lowest sigma bands, the head trains again with the alignment loss on
FIRST_W_ANCHOR_GREY = 5             # step 10: the in-band change the smallest tuning weight w1 is set to give; a setting, not a pass mark
FIRST_CONTACT_DROP_MAX = 0.10       # step 11: a weight is allowed only if its contact kept is within 0.10 of the region sampler's
FIRST_GPU_HOURS_STOP = 4            # measured rates projecting this plan past it stop it for the owner
FIRST_GPU_HOURS_STOP_ALIGNED = 7    # the same stop once the aligned head runs

ROWS = {"ours": "ours", "region": "region", "random": "randompush", "paste": "paste", "plainsum": "plainsum",
        "paste_lower": "paste-lower-on-top", "real": "real"}   # the row names paper1_sheets.py select is given


# ---------------------------------------------------------------------------------------------------------------------------------------
# The rule


def one_sided_binomial_p(wins, losses):
    """P(X >= wins) for X ~ Binomial(wins + losses, 1/2): the one-sided exact McNemar p on the disagreeing seeds."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(wins, n + 1)) / 2 ** n


RUBRIC_REASON = "could not measure it: the rubric at 64 pixels"


def real_right_needed(real_called):
    """How many of real_called real touching scenes must be called right: ceil(FIRST_REAL_RIGHT_SHARE_MIN x real_called)."""
    return math.ceil(round(FIRST_REAL_RIGHT_SHARE_MIN * real_called, 9))


def answer_and_reason(gain, p, random_minus_region, real_right=None, real_called=FIRST_TOUCH_SEEDS):
    """(answer, reason). The instrument first, as the full plan reads its instrument rows before any result row: fewer than
    real_right_needed(real_called) of the real_called real touching scenes called right makes the answer inconclusive whatever the
    gain, p or random push (decided on board question 123; read as a share by the owner on 2026-11-27). Then the verdict file's rule,
    in this order: no, yes, inconclusive."""
    if real_right is not None and real_right < real_right_needed(real_called):
        return "inconclusive", RUBRIC_REASON
    if gain <= FIRST_NULL_GAIN_MAX:
        return "no", f"gain {gain} at or below {FIRST_NULL_GAIN_MAX}"
    if gain >= FIRST_GAIN_MIN and p < FIRST_P_MAX and abs(random_minus_region) <= FIRST_RANDOM_BAND:
        return "yes", f"gain {gain} at least {FIRST_GAIN_MIN}, p {p:.4f} below {FIRST_P_MAX}, random push within {FIRST_RANDOM_BAND}"
    return "inconclusive", "between the bars: a gain of 6 to 14, or p at 0.05 or above, or the random push more than 5 seeds off"


def answer(gain, p, random_minus_region, real_right=None, real_called=FIRST_TOUCH_SEEDS):
    """yes, no or inconclusive (see answer_and_reason)."""
    return answer_and_reason(gain, p, random_minus_region, real_right, real_called)[0]


def is_right(call, count, side):
    """A settled call is right: both counts the asked count and no blended band; on a touching picture also occlusion correct."""
    try:
        counts_right = int(call["red_count"]) == count and int(call["blue_count"]) == count
    except ValueError:
        return False
    clean = call["blended_band"] == "no"
    if side == "apart":
        return counts_right and clean
    return counts_right and clean and call["occlusion_correct"] == "yes"


def failing_fields(call, count, side):
    out = []
    for field, ok in (("red_count", call["red_count"] == str(count)), ("blue_count", call["blue_count"] == str(count)),
                      ("blended_band", call["blended_band"] == "no"),
                      ("occlusion_correct", side == "apart" or call["occlusion_correct"] == "yes")):
        if not ok:
            out.append(field)
    return out


def read(settled, key, rows=ROWS):
    """The whole answer from settled calls ({picture: fields}) and the sheet's key (list of {picture, row, seed, count, side})."""
    right, fails = {}, {}
    for item in key:
        call = settled.get(item["picture"])
        if call is None:
            raise SystemExit(f"no settled call for {item['picture']}")
        k = (item["row"], item["side"], item["count"], item["seed"])
        right[k] = is_right(call, item["count"], item["side"])
        for f in failing_fields(call, item["count"], item["side"]):
            fails.setdefault((item["row"], item["side"]), {}).setdefault(f, 0)
            fails[(item["row"], item["side"])][f] += 1

    def per_row(row, side):
        per = {}
        for (r, s, c, _), ok in right.items():
            if r == row and s == side:
                per.setdefault(c, [0, 0])
                per[c][0] += int(ok)
                per[c][1] += 1
        return {f"count-{c}": {"right": v[0], "called": v[1]} for c, v in sorted(per.items())}, sum(v[0] for v in per.values()), \
            sum(v[1] for v in per.values())

    table = {}
    for name, row in rows.items():
        for side in ("touching", "apart"):
            per, n_right, n = per_row(row, side)
            if n:
                table.setdefault(name, {})[side] = {"right": n_right, "called": n, "per_count": per}

    def seeds(row):
        return {(c, s): ok for (r, side, c, s), ok in right.items() if r == row and side == "touching"}
    ours, region, rand = seeds(rows["ours"]), seeds(rows["region"]), seeds(rows["random"])
    shared = sorted(set(ours) & set(region))
    if len(shared) != len(ours) or len(shared) != len(region):
        raise SystemExit("ours and the region sampler are not called on the same touching seeds")
    both = sum(1 for s in shared if ours[s] and region[s])
    ours_only = sum(1 for s in shared if ours[s] and not region[s])
    region_only = sum(1 for s in shared if region[s] and not ours[s])
    neither = len(shared) - both - ours_only - region_only
    gain = ours_only - region_only
    p = one_sided_binomial_p(ours_only, region_only)
    random_minus_region = sum(rand.values()) - sum(region.values())
    real = table.get("real", {}).get("touching", {"right": 0, "called": 0})
    verdict, reason = answer_and_reason(gain, p, random_minus_region, real["right"] if real["called"] else None, real["called"])
    out = {
        "answer": verdict, "reason": reason,
        "gain": gain, "p_one_sided": round(p, 6), "disagreeing_seeds": ours_only + region_only,
        "two_by_two": {"ours right, region right": both, "ours right, region wrong": ours_only,
                       "ours wrong, region right": region_only, "ours wrong, region wrong": neither},
        "random_push_minus_region": random_minus_region,
        "touching_seeds": len(shared), "expected_touching_seeds": FIRST_TOUCH_SEEDS,
        "real_scenes_right": real["right"], "real_scenes_called": real["called"],
        "real_scenes_needed": real_right_needed(real["called"]),
        "rubric_readable": real["right"] >= real_right_needed(real["called"]),
        "rows": table,
        "failing_fields": {f"{r} {s}": v for (r, s), v in sorted(fails.items())},
        "bars": {"FIRST_GAIN_MIN": FIRST_GAIN_MIN, "FIRST_P_MAX": FIRST_P_MAX, "FIRST_RANDOM_BAND": FIRST_RANDOM_BAND,
                 "FIRST_NULL_GAIN_MAX": FIRST_NULL_GAIN_MAX,
                 "FIRST_REAL_RIGHT_SHARE_MIN": FIRST_REAL_RIGHT_SHARE_MIN},
    }
    if "ours" in table and "paste" in table and "apart" in table["ours"] and "apart" in table["paste"]:
        out["apart_ours_minus_paste"] = table["ours"]["apart"]["right"] - table["paste"]["apart"]["right"]
    if "paste_lower" in table and "ours" in table:
        out["touching_paste_lower_minus_ours"] = table["paste_lower"]["touching"]["right"] - table["ours"]["touching"]["right"]
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------


def strip(key, rows, path, sheet_dir, seeds_per=8):
    """The picture strip: the first 8 touching seeds the region sampler and ours share (columns) across the rows (rows), read from the
    blind sheet's own 4-times pictures, so it builds wherever the sheet was pulled to."""
    from PIL import Image, ImageDraw
    by = {(it["row"], it["count"], it["seed"]): str(Path(sheet_dir) / it["picture"]) for it in key if it["side"] == "touching"}
    cols = sorted({(c, s) for (r, c, s) in by if r == rows["ours"]} & {(c, s) for (r, c, s) in by if r == rows["region"]})[:seeds_per]
    order = [r for r in ("plainsum", "paste", "region", "ours", "random", "paste_lower") if any(rr == rows[r] for rr, _, _ in by)]
    out = Image.new("RGB", (120 + len(cols) * 258, len(order) * 258), "white")
    draw = ImageDraw.Draw(out)
    for i, name in enumerate(order):
        draw.text((4, i * 258 + 120), rows[name], fill="black")
        for j, (c, s) in enumerate(cols):
            src = by.get((rows[name], c, s))
            if src and Path(src).is_file():
                out.paste(Image.open(src).convert("RGB").resize((256, 256), Image.NEAREST), (120 + j * 258, i * 258))
    out.save(path)


def cmd_read(args):
    import paper1_sheets as sh
    folder = Path(args.root) / args.run
    folder.mkdir(parents=True, exist_ok=True)
    sheet = Path(args.root) / args.sheet
    first, second = (sh.read_calls(sheet / SHEET_SUB / c if not Path(c).is_absolute() else Path(c)) for c in args.calls)
    third = sh.read_calls(sheet / SHEET_SUB / args.third if not Path(args.third).is_absolute() else Path(args.third)) if args.third else {}
    settled, decided_by, unresolved = sh.settle(first, second, third)
    if unresolved:
        (folder / "unresolved.json").write_text(json.dumps(unresolved, indent=1) + "\n")
        raise SystemExit(f"{len(unresolved)} fields have neither agreement nor a third call; see unresolved.json")
    key = json.loads((sheet / "key.json").read_text())["key"]
    out = read(settled, key)
    out["settled_by"] = decided_by
    (folder / "answer.json").write_text(json.dumps(out, indent=1) + "\n")
    t = out["two_by_two"]
    lines = [f"answer: {out['answer']} ({out['reason']})",
             f"gain {out['gain']} (bar {FIRST_GAIN_MIN} for yes, {FIRST_NULL_GAIN_MAX} or below is no); one-sided p {out['p_one_sided']} "
             f"on {out['disagreeing_seeds']} disagreeing seeds (bar below {FIRST_P_MAX})",
             f"random push minus region sampler: {out['random_push_minus_region']} (bar within +-{FIRST_RANDOM_BAND})",
             "", "                     region right   region wrong",
             f"ours right           {t['ours right, region right']:>12}   {t['ours right, region wrong']:>12}",
             f"ours wrong           {t['ours wrong, region right']:>12}   {t['ours wrong, region wrong']:>12}", "",
             f"real touching scenes right: {out['real_scenes_right']} of {out['real_scenes_called']} (needed {out['real_scenes_needed']}, {FIRST_REAL_RIGHT_SHARE_MIN} of those called; "
             f"rubric readable: {out['rubric_readable']})"]
    for name, sides in out["rows"].items():
        for side, v in sides.items():
            lines.append(f"{name:12s} {side:9s} right {v['right']:3d} of {v['called']:3d}   " +
                         "  ".join(f"{c}: {x['right']}/{x['called']}" for c, x in v["per_count"].items()))
    (folder / "answer.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    strip(key, ROWS, folder / "strip.png", sheet / SHEET_SUB)


SHEET_SUB = "sheet"


# ---------------------------------------------------------------------------------------------------------------------------------------
# The weight and window, from the owner's tuning calls (the first result's task 13)


def setting_of(row):
    """'win15-22-w0.5' to (0.5, '15-22'): the tuning sheet's pushed rows are named by window and weight."""
    window, _, w = row[len("win"):].rpartition("-w")
    return float(w), window


def choose_setting(calls, key, contact, region_row="region", default_window="15-22"):
    """Among the pushed settings whose contact kept is no more than FIRST_CONTACT_DROP_MAX below the region sampler's, the one with
    the most pictures right on the tuning sheet; on a tie the smaller w, then the default window. Returns (chosen or None, table).
    calls: {picture: fields} from the owner; contact: {row: contact-kept share on the touching tuning seeds}."""
    right = {}
    for item in key:
        call = calls.get(item["picture"])
        if call is not None:
            right.setdefault(item["row"], 0)
            right[item["row"]] += int(is_right(call, item["count"], "touching"))
    region_contact = contact[region_row]
    table = []
    for row in sorted(r for r in right if r.startswith("win")):
        w, window = setting_of(row)
        drop = region_contact - contact[row]
        table.append({"row": row, "w": w, "window": window, "right": right[row], "contact_kept": contact[row],
                      "drop_from_region": round(drop, 4), "allowed": drop <= FIRST_CONTACT_DROP_MAX + 1e-12})
    allowed = [t for t in table if t["allowed"]]
    if not allowed:
        return None, {"region_right": right.get(region_row), "region_contact": region_contact, "settings": table}
    best = min(allowed, key=lambda t: (-t["right"], t["w"], t["window"] != default_window))
    return best, {"region_right": right.get(region_row), "region_contact": region_contact, "settings": table}


def cmd_choose_w(args):
    import paper1_sheets as sh
    sheet = Path(args.root) / args.sheet
    folder = Path(args.root) / args.run
    folder.mkdir(parents=True, exist_ok=True)
    calls = sh.read_calls(sheet / SHEET_SUB / args.calls if not Path(args.calls).is_absolute() else Path(args.calls))
    key = json.loads((sheet / "key.json").read_text())["key"]
    by_folder = json.loads(Path(args.contact).read_text())
    row_folder = {it["row"]: str(Path(it["path"]).parents[1]) for it in key}
    contact = {}
    for row, path in row_folder.items():
        hit = [v for k, v in by_folder.items() if k == path or Path(k).parts[-2:] == Path(path).parts[-2:]]
        if not hit:
            raise SystemExit(f"no contact kept for row {row} ({path}) in {args.contact}")
        contact[row] = hit[0]["pooled_share"]
    chosen, table = choose_setting(calls, key, contact)
    out = {"rule": f"among pushed settings whose contact kept is at most {FIRST_CONTACT_DROP_MAX} below the region sampler's, the most "
                   "right on the tuning sheet; ties to the smaller w, then the window 15-22", "chosen": chosen, **table}
    (folder / "chosen-setting.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"chosen": chosen}), flush=True)
    if chosen is None:
        print("STOP: no setting keeps contact within the bar; the push separates the objects at every setting tried", flush=True)
        raise SystemExit(3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("read")
    p.add_argument("--run", required=True)
    p.add_argument("--root", default="/datasets/mmolefe/poe_repair_min/paper1")
    p.add_argument("--sheet", required=True, help="the eye-sheet run folder, holding key.json and sheet/")
    p.add_argument("--calls", nargs=2, required=True, help="the two reviewers' filled call files (in sheet/ unless absolute)")
    p.add_argument("--third", help="the third reviewer's calls")
    p.set_defaults(func=cmd_read)
    p = sub.add_parser("choose-w", help="the weight and window from the owner's tuning calls")
    p.add_argument("--run", required=True)
    p.add_argument("--root", default="/datasets/mmolefe/poe_repair_min/paper1")
    p.add_argument("--sheet", required=True, help="the tuning sheet's run folder, holding key.json and sheet/")
    p.add_argument("--calls", default="calls-owner.csv", help="the owner's filled calls (in sheet/ unless absolute)")
    p.add_argument("--contact", required=True, help="contact.json from paper1_sample.py contact over every tuning row and the region sampler")
    p.set_defaults(func=cmd_choose_w)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
