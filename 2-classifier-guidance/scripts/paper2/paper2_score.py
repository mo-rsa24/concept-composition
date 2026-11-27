"""Score every judge against the bars on the test pairs, the confirmation slice unread (species judge step 0.21).

Run name `paper2-judge-score`. Reads paper2_label_set.py's labelled-set.csv and split.json (through
paper2_judge_lists.make_lists, so the test pictures are the very ones the judges were given), each judge's calls on
test.csv (`--judge clip=<paper2-judge-clip>`, reading <dir>/test/calls.json and, where the judge has a free setting,
<dir>/settings-chosen.json), the owner's 60 when called (`--owner-key`, `--owner-calls`), and the collector's
candidates (`--candidates`, for which checker drew a picture). Every bar is read from paper2_bars.py. It refuses to
open confirm.json, and refuses a judge whose calls include a confirmation picture.

The read set is the test pairs' read pictures (40 a kind). Per judge, written to bars.json and printed, each number with
the owner's label check beside it:

  agreement         in all and per kind, each split by pair type, against AGREE_ALL_MIN and AGREE_KIND_MIN scaled to
                    the pictures read (rounded up); an uncovered picture (the ImageNet judge's) counts as a miss, and
                    the share covered and the agreement on covered pictures are printed beside it
  look-alike        agreement on look-alike pairs against unlike pairs, at the bars' rates (216 of 240, 33 of 40); a
                    judge that passes on unlike pairs and misses on look-alike pairs misses (the plan's results table).
                    A kind whose look-alike minimum is under 20 (LOOKALIKE_MIN_BY_KIND: extra subject 8) is marked
                    underpowered on look-alike pairs
  reviewer gap      the judge's agreement with each reviewer against the reviewers' agreement with each other, minus
                    REVIEWER_GAP_MAX in all and REVIEWER_GAP_KIND_MAX within each settled kind
  two-clear         precision of the judge's two-clear calls on the read set (TWO_CLEAR_PRECISION_MIN), printed also on
  precision         the unselected sample at its natural share and on pictures whose boxes overlap apart from the rest
  within-pair AUC   two clear against the other five kinds by the judge's two-clear score, compared only within a pair,
                    over the pairs holding at least AUC_PAIR_MIN_EACH of each (WITHIN_PAIR_AUC_MIN, AUC_PAIRS_MIN); the
                    pooled AUC is printed beside the pooled AUC of a lookup that knows only the pair (each pair's share
                    of two clear in the read set)
  drop one pair     every read bar again with each test pair dropped, counts scaled to what is left; a bar that
                    flips names the pair (DROP_PAIR_HOLDS)
  apart             natural and renamed wrong species, and per renaming entry; pictures drawn by a push whose checker
                    read CLIP or ImageNet features; crow and raven. None of these enters a bar
  settings          the test call's settings equal settings-chosen.json, and settings-chosen.json was written first

Which bars are read follows the owner's check (OWNER_KIND_MIN, OWNER_ALL_MIN, OWNER_ABSOLUTE_MIN): at 57 of 60 or more
every bar; at 54 to 56 the absolute agreement bars (in all, per kind, look-alike) are printed and not read and the
verdict is "limited by labels"; under 54, or under 8 in any kind, the labels cannot be trusted and nothing is read;
while the owner's calls are missing the absolute bars are printed as pending. The old judge is the floor row, read on
the kinds it can name (two clear, two of one kind, hybrid).

Writes under --out: bars.json, confusion-<judge>.csv (eye kind against judge kind), disagreements.csv (every read
picture where a judge and the eye differ), and disagreements/<judge>/ for every judge that meets every read bar: each
such picture at full size with its boxes, each body's call, and both kinds printed.

    "$PY" paper2_score.py --labelled-set <paper2-label-set>/labelled-set.csv --split <paper2-label-set>/split.json \\
        --judge old=<paper2-judge-old> --judge clip=<paper2-judge-clip> --judge imagenet=<paper2-judge-imagenet> \\
        --judge vlm=<paper2-judge-vlm> --owner-key <paper2-label-set>/owner-sheet/key/key.json \\
        --owner-calls <paper2-label-set>/owner-calls.csv --candidates <paper2-collect>/candidates.csv \\
        --out /datasets/mmolefe/poe_repair_min/paper2/paper2-judge-score
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paper2_bars as bars  # noqa: E402
from paper2_judge_lists import make_lists, read_labelled, refuse_confirm  # noqa: E402
from paper2_kinds import KINDS, TWO_CLEAR, WRONG_SPECIES  # noqa: E402

OLD_JUDGE_READ_KINDS = sorted({k for k in bars.OLD_JUDGE_KINDS.values() if k in KINDS})
PUSH_FEATURES = ("clip", "imagenet")
KIND_RATE = bars.AGREE_KIND_MIN / bars.PER_KIND_TEST_READ                         # 33 of 40
ALL_RATE = bars.AGREE_ALL_MIN / (bars.PER_KIND_TEST_READ * len(KINDS))            # 216 of 240


def norm(k) -> str | None:
    if k in (None, ""):
        return None
    return "_".join(str(k).strip().lower().replace("-", " ").replace("_", " ").split())


def key_of(p: dict) -> tuple:
    return (p["picture"], p["a"], p["b"])


def need(rate_num: int, rate_den: int, n: int) -> int:
    """A count bar scaled to n pictures, rounded up."""
    return math.ceil(rate_num * n / rate_den)


def agree(xs, ys) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    return round(sum(x == y for x, y in pairs) / len(pairs), 4) if pairs else None


def boxes_overlap(boxes: list) -> bool:
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                return True
    return False


def auc(pos: list[float], neg: list[float]) -> tuple[float, int]:
    """Share of (positive, negative) comparisons ranked right, ties counting half; and the number of comparisons."""
    n = len(pos) * len(neg)
    if not n:
        return float("nan"), 0
    right = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return right / n, n


# ---------------------------------------------------------------- the bars on one set of read pictures

def bar_set(rows: list[dict], judge: str) -> dict:
    """Every bar on these read rows. Each row carries kind (settled), call_1, call_2, pair, pair_type, judge (its kind)
    and score (its two-clear score)."""
    n = len(rows)
    kinds = OLD_JUDGE_READ_KINDS if judge == "old" else list(KINDS)
    out = {}
    right = sum(r["judge"] == r["kind"] for r in rows)
    total = bars.PER_KIND_TEST_READ * len(KINDS)
    out["agree_all"] = {"right": right, "of": n, "need": need(bars.AGREE_ALL_MIN, total, n),
                        "pass": right >= need(bars.AGREE_ALL_MIN, total, n)}
    per = {}
    for k in kinds:
        mine = [r for r in rows if r["kind"] == k]
        rk = sum(r["judge"] == k for r in mine)
        nk = need(bars.AGREE_KIND_MIN, bars.PER_KIND_TEST_READ, len(mine))
        per[k] = {"right": rk, "of": len(mine), "need": nk, "pass": bool(mine) and rk >= nk}
        for pt in ("look-alike", "unlike"):
            sub = [r for r in mine if r["pair_type"] == pt]
            per[k][pt] = {"right": sum(r["judge"] == k for r in sub), "of": len(sub)}
    out["agree_kind"] = {"kinds": per, "pass": all(v["pass"] for v in per.values())}
    by_type = {}
    for pt in ("look-alike", "unlike"):
        sub = [r for r in rows if r["pair_type"] == pt]
        by_type[pt] = {"right": sum(r["judge"] == r["kind"] for r in sub), "of": len(sub)}
        by_type[pt]["rate"] = round(by_type[pt]["right"] / len(sub), 4) if sub else None
    la, un = by_type["look-alike"]["rate"], by_type["unlike"]["rate"]
    out["lookalike"] = {**by_type, "bar_rate": round(ALL_RATE, 4),
                        "pass": not (un is not None and un >= ALL_RATE and (la is None or la < ALL_RATE))}

    both = [r for r in rows if r["call_1"] and r["call_2"]]
    r12 = agree([r["call_1"] for r in both], [r["call_2"] for r in both])
    j1 = agree([r["judge"] for r in both], [r["call_1"] for r in both])
    j2 = agree([r["judge"] for r in both], [r["call_2"] for r in both])
    gap_ok = None not in (r12, j1, j2) and j1 >= r12 - bars.REVIEWER_GAP_MAX and j2 >= r12 - bars.REVIEWER_GAP_MAX
    out["reviewer_gap"] = {"reviewers": r12, "judge_vs_1": j1, "judge_vs_2": j2, "max": bars.REVIEWER_GAP_MAX,
                           "pass": bool(gap_ok)}
    gk = {}
    for k in kinds:
        mine = [r for r in both if r["kind"] == k]
        a12 = agree([r["call_1"] for r in mine], [r["call_2"] for r in mine])
        a1 = agree([r["judge"] for r in mine], [r["call_1"] for r in mine])
        a2 = agree([r["judge"] for r in mine], [r["call_2"] for r in mine])
        ok = None not in (a12, a1, a2) and a1 >= a12 - bars.REVIEWER_GAP_KIND_MAX and a2 >= a12 - bars.REVIEWER_GAP_KIND_MAX
        gk[k] = {"reviewers": a12, "judge_vs_1": a1, "judge_vs_2": a2, "pass": bool(ok)}
    out["reviewer_gap_kind"] = {"kinds": gk, "max": bars.REVIEWER_GAP_KIND_MAX, "pass": all(v["pass"] for v in gk.values())}

    called = [r for r in rows if r["judge"] == TWO_CLEAR]
    prec = round(sum(r["kind"] == TWO_CLEAR for r in called) / len(called), 4) if called else None
    out["two_clear_precision"] = {"precision": prec, "called_two_clear": len(called), "min": bars.TWO_CLEAR_PRECISION_MIN,
                                  "pass": prec is not None and prec >= bars.TWO_CLEAR_PRECISION_MIN}

    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["pair"]].append(r)
    per_pair, right_sum, comps = {}, 0.0, 0
    for p, rs in sorted(by_pair.items()):
        pos = [r["score"] for r in rs if r["kind"] == TWO_CLEAR and r["score"] is not None]
        neg = [r["score"] for r in rs if r["kind"] != TWO_CLEAR and r["score"] is not None]
        if len(pos) >= bars.AUC_PAIR_MIN_EACH and len(neg) >= bars.AUC_PAIR_MIN_EACH:
            a, nc = auc(pos, neg)
            per_pair[p] = round(a, 4)
            right_sum += a * nc
            comps += nc
    within = round(right_sum / comps, 4) if comps else None
    scored = [r for r in rows if r["score"] is not None]
    pooled, _ = auc([r["score"] for r in scored if r["kind"] == TWO_CLEAR],
                    [r["score"] for r in scored if r["kind"] != TWO_CLEAR])
    share = {p: sum(r["kind"] == TWO_CLEAR for r in rs) / len(rs) for p, rs in by_pair.items()}
    lookup, _ = auc([share[r["pair"]] for r in rows if r["kind"] == TWO_CLEAR],
                    [share[r["pair"]] for r in rows if r["kind"] != TWO_CLEAR])
    out["within_pair_auc"] = {"auc": within, "pairs": per_pair, "pairs_needed": bars.AUC_PAIRS_MIN,
                              "min": bars.WITHIN_PAIR_AUC_MIN, "pooled_auc": None if math.isnan(pooled) else round(pooled, 4),
                              "pooled_pair_lookup_auc": None if math.isnan(lookup) else round(lookup, 4),
                              "pass": len(per_pair) >= bars.AUC_PAIRS_MIN and within is not None
                              and within >= bars.WITHIN_PAIR_AUC_MIN}
    return out


ABSOLUTE = ("agree_all", "agree_kind", "lookalike")
RELATIVE = ("reviewer_gap", "reviewer_gap_kind", "two_clear_precision", "within_pair_auc")


# ---------------------------------------------------------------- the owner's check

def owner_check(key_path: Path | None, calls_path: Path | None, settled: dict) -> dict:
    if not key_path or not calls_path or not Path(calls_path).exists():
        return {"state": "pending", "text": "owner's check pending (owner-calls.csv not yet written)"}
    refuse_confirm(key_path, calls_path)
    key = json.loads(Path(key_path).read_text())["pictures"]
    with open(calls_path, newline="") as f:
        calls = {r["name"]: norm(r["kind"]) for r in csv.DictReader(f) if (r.get("kind") or "").strip()}
    per, right, n = defaultdict(lambda: [0, 0]), 0, 0
    for name, e in key.items():
        k = settled.get((e.get("picture_path") or e.get("picture"), e["a"], e["b"]))
        if k is None:
            raise SystemExit(f"owner sheet entry {name} is not a settled read picture of the labelled set")
        if name not in calls:
            continue
        n += 1
        per[k][1] += 1
        if calls[name] == k:
            right += 1
            per[k][0] += 1
    per = {k: {"right": v[0], "of": v[1]} for k, v in per.items()}
    if n < len(key):
        return {"state": "pending", "right": right, "of": n, "per_kind": per,
                "text": f"owner's check pending ({n} of {len(key)} called)"}
    kind_ok = all(v["right"] >= bars.OWNER_KIND_MIN for v in per.values())
    if right < bars.OWNER_ALL_MIN or not kind_ok:
        state = "untrusted"
    elif right >= bars.OWNER_ABSOLUTE_MIN:
        state = "absolute"
    else:
        state = "limited"
    return {"state": state, "right": right, "of": n, "per_kind": per,
            "text": f"labels: owner {right} of {n}" + {"absolute": "", "limited": ", limited by labels",
                                                       "untrusted": ", labels cannot be trusted"}[state]}


# ---------------------------------------------------------------- reading the judges

def read_judge(name: str, folder: Path, test: list[dict], confirm_keys: set) -> dict:
    calls_path = folder / "test" / "calls.json"
    refuse_confirm(calls_path)
    calls = json.loads(calls_path.read_text())
    by_key = {key_of(p): p for p in calls["pictures"]}
    touched = [k for k in by_key if k in confirm_keys]
    if touched:
        raise SystemExit(f"{name} called {len(touched)} confirmation pictures; its calls are not scored")
    missing = [key_of(r) for r in test if key_of(r) not in by_key]
    if missing:
        raise SystemExit(f"{name} has no call for {len(missing)} test pictures, for example {missing[0]}")
    settings_check = {"needed": name != "old"}
    chosen_path = folder / "settings-chosen.json"
    test_settings = folder / "test" / "settings.json"
    if name != "old":
        if not chosen_path.exists():
            settings_check.update(ok=False, why=f"no {chosen_path.name}")
        else:
            chosen = json.loads(chosen_path.read_text())
            used = json.loads(test_settings.read_text()) if test_settings.exists() else {"settings": calls["settings"]}
            same = all(used["settings"].get(k) == v for k, v in chosen["chosen"].items())
            before = ("written" in used and
                      datetime.fromisoformat(chosen["written"]) <= datetime.fromisoformat(used["written"]))
            settings_check.update(ok=same and before, chosen=chosen["chosen"], same=same, chosen_first=before)
    return {"calls": by_key, "settings": calls["settings"], "settings_check": settings_check}


def push_checkers(paths: list[Path]) -> dict | None:
    if not paths:
        return None
    out = {}
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                out[r["picture"]] = r.get("push_checker", "none")
    return out


# ---------------------------------------------------------------- one judge

def score_judge(name: str, j: dict, read: list[dict], test: list[dict], owner: dict, checkers: dict | None) -> dict:
    def judged(rows):
        out = []
        for r in rows:
            p = j["calls"][key_of(r)]
            out.append({**r, "judge": p.get("kind") if p.get("covered", True) else None,
                        "score": p.get("two_clear_score"), "covered": p.get("covered", True),
                        "boxes": [b["box"] for b in p.get("bodies", [])], "bodies": p.get("bodies", [])})
        return out

    rd, ts = judged(read), judged(test)
    res = {"judge": name, "labels": owner["text"], "settings": j["settings"], "settings_check": j["settings_check"]}
    res["bars"] = bar_set(rd, name)
    covered = [r for r in rd if r["covered"]]
    res["coverage"] = {"covered": len(covered), "of": len(rd),
                       "agreement_on_covered": agree([r["judge"] for r in covered], [r["kind"] for r in covered])}

    # two-clear precision at the natural share, and by whether the boxes overlap
    def precision(rows):
        c = [r for r in rows if r["judge"] == TWO_CLEAR]
        return {"precision": round(sum(r["kind"] == TWO_CLEAR for r in c) / len(c), 4) if c else None,
                "called_two_clear": len(c), "pictures": len(rows)}
    uns = [r for r in ts if r.get("unselected_sample") == "True" and r["part"] in ("read", "spare")]
    res["two_clear_precision_unselected"] = precision(uns)
    res["two_clear_precision_overlap"] = {"boxes overlap": precision([r for r in rd if boxes_overlap(r["boxes"])]),
                                          "boxes apart": precision([r for r in rd if not boxes_overlap(r["boxes"])])}

    # drop one pair
    full = res["bars"]
    flips = defaultdict(list)
    for p in sorted({r["pair"] for r in rd}):
        again = bar_set([r for r in rd if r["pair"] != p], name)
        for b in ABSOLUTE + RELATIVE:
            if full[b]["pass"] and not again[b]["pass"]:
                flips[b].append(p)
    res["drop_one_pair"] = {"flips": dict(flips), "pass": not flips}

    # apart
    ws = [r for r in rd if r["kind"] == WRONG_SPECIES]
    nat = [r for r in ws if r.get("renamed") != "True"]
    ren = [r for r in ws if r.get("renamed") == "True"]
    by_entry = defaultdict(list)
    for r in ren:
        by_entry[r["renaming"]].append(r)
    res["wrong_species_apart"] = {
        "natural": {"right": sum(r["judge"] == WRONG_SPECIES for r in nat), "of": len(nat)},
        "renamed": {"right": sum(r["judge"] == WRONG_SPECIES for r in ren), "of": len(ren)},
        "by_renaming": {e: {"right": sum(r["judge"] == WRONG_SPECIES for r in rs), "of": len(rs)}
                        for e, rs in sorted(by_entry.items())}}
    if checkers is None:
        res["clip_or_imagenet_push"] = "not checked (no --candidates)"
    else:
        pushed = [r for r in ts if any(f in str(checkers.get(r["picture"], "none")).lower() for f in PUSH_FEATURES)]
        res["clip_or_imagenet_push"] = {"pictures": len(pushed),
                                        "agreement": agree([r["judge"] for r in pushed], [r["kind"] for r in pushed])}
    apart = [r for r in ts if r["part"] == "apart"]
    res["crow_and_raven"] = {"pictures": len(apart), "right": sum(r["judge"] == norm(r["kind"]) for r in apart),
                             "by_kind": dict(Counter(r["kind"] for r in apart))}
    res["underpowered_lookalike"] = {k: f"look-alike minimum {m}, underpowered on look-alike pairs"
                                     for k, m in bars.LOOKALIKE_MIN_BY_KIND.items() if m < bars.LOOKALIKE_PER_KIND_MIN}

    # which bars are read, and the verdict
    if name == "old":
        res["verdict"] = "the floor row: read on " + ", ".join(OLD_JUDGE_READ_KINDS) + "; not a candidate"
        res["read_bars"] = []
    else:
        state = owner["state"]
        read_bars = [] if state == "untrusted" else list(RELATIVE) + ["drop_one_pair"] + \
            (list(ABSOLUTE) if state == "absolute" else [])
        res["read_bars"] = read_bars
        missed = [b for b in read_bars if not (res["drop_one_pair"] if b == "drop_one_pair" else full[b])["pass"]]
        if not j["settings_check"].get("ok", False):
            missed.append("settings chosen before the test call")
        one_kind = [b for b in ("agree_kind", "reviewer_gap_kind") if b in missed and
                    sum(not v["pass"] for v in full[b]["kinds"].values()) == 1]
        if state == "untrusted":
            res["verdict"] = "nothing read: the labels cannot be trusted"
        elif not missed:
            res["verdict"] = "meets every read bar" + {"absolute": "", "limited": " (limited by labels)",
                                                       "pending": " read so far (the absolute bars wait for the owner's check)"}[state]
        elif set(missed) == set(one_kind) and one_kind:
            res["verdict"] = "meets every read bar but one kind: " + "; ".join(
                f"{b} on {next(k for k, v in full[b]['kinds'].items() if not v['pass'])}" for b in one_kind)
        else:
            res["verdict"] = "misses: " + ", ".join(missed)
        res["missed"] = missed
    res["_rows"] = rd
    return res


# ---------------------------------------------------------------- outputs

def lines_for(res: dict) -> list[str]:
    b, L = res["bars"], res["labels"]
    out = [f"{res['judge']}: {res['verdict']}   [{L}]"]
    a = b["agree_all"]
    out.append(f"  agreement {a['right']} of {a['of']} (need {a['need']}){'' if a['pass'] else ' MISS'}; covered "
               f"{res['coverage']['covered']} of {res['coverage']['of']}, {res['coverage']['agreement_on_covered']} on covered   [{L}]")
    for k, v in b["agree_kind"]["kinds"].items():
        note = res["underpowered_lookalike"].get(k, "")
        out.append(f"    {k:16s} {v['right']} of {v['of']} (need {v['need']}){'' if v['pass'] else ' MISS'}; look-alike "
                   f"{v['look-alike']['right']} of {v['look-alike']['of']}, unlike {v['unlike']['right']} of "
                   f"{v['unlike']['of']}{'; ' + note if note else ''}   [{L}]")
    la = b["lookalike"]
    out.append(f"  look-alike {la['look-alike']['rate']} against unlike {la['unlike']['rate']} at the bar's rate "
               f"{la['bar_rate']}{'' if la['pass'] else ': passes on unlike, misses on look-alike'}   [{L}]")
    g = b["reviewer_gap"]
    out.append(f"  reviewer gap: reviewers {g['reviewers']}, judge with reviewer 1 {g['judge_vs_1']}, with reviewer 2 "
               f"{g['judge_vs_2']}{'' if g['pass'] else ' MISS'}; per kind misses: "
               f"{[k for k, v in b['reviewer_gap_kind']['kinds'].items() if not v['pass']] or 'none'}   [{L}]")
    p = b["two_clear_precision"]
    out.append(f"  two-clear precision {p['precision']} of {p['called_two_clear']} called{'' if p['pass'] else ' MISS'}; "
               f"unselected {res['two_clear_precision_unselected']['precision']} "
               f"({res['two_clear_precision_unselected']['pictures']} pictures); boxes overlap "
               f"{res['two_clear_precision_overlap']['boxes overlap']['precision']}, apart "
               f"{res['two_clear_precision_overlap']['boxes apart']['precision']}   [{L}]")
    w = b["within_pair_auc"]
    out.append(f"  within-pair AUC {w['auc']} over {len(w['pairs'])} pairs{'' if w['pass'] else ' MISS'}; pooled "
               f"{w['pooled_auc']} beside a pair lookup's {w['pooled_pair_lookup_auc']}   [{L}]")
    out.append(f"  drop one pair: {'holds' if res['drop_one_pair']['pass'] else res['drop_one_pair']['flips']}   [{L}]")
    ws = res["wrong_species_apart"]
    out.append(f"  wrong species: natural {ws['natural']['right']} of {ws['natural']['of']}, renamed "
               f"{ws['renamed']['right']} of {ws['renamed']['of']} {ws['by_renaming'] or ''}; crow and raven "
               f"{res['crow_and_raven']['right']} of {res['crow_and_raven']['pictures']}; CLIP or ImageNet pushes "
               f"{res['clip_or_imagenet_push']}")
    sc = res["settings_check"]
    if sc["needed"]:
        out.append(f"  settings: {'chosen on choosing pairs and frozen before the test call' if sc.get('ok') else 'CHECK FAILED ' + json.dumps(sc)}")
    return out


def draw_disagreements(res: dict, out: Path) -> int:
    from PIL import Image, ImageDraw
    folder = out / "disagreements" / res["judge"]
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in res["_rows"]:
        if r["judge"] == r["kind"]:
            continue
        im = Image.open(r["picture"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for i, body in enumerate(r["bodies"]):
            x0, y0, x1, y1 = body["box"]
            d.rectangle((x0, y0, x1, y1), outline=(230, 30, 30), width=4)
            d.text((x0 + 6, y0 + 6), f"{i}: {body['call']}{'*' if body.get('flag') else ''}", fill=(230, 30, 30))
        d.rectangle((0, 0, im.width, 26), fill=(255, 255, 255))
        d.text((6, 6), f"asked {r['a']} and {r['b']}; eye {r['kind']}; {res['judge']} {r['judge']}", fill=(0, 0, 0))
        n += 1
        im.save(folder / f"{n:03d}.png")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labelled-set", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--judge", action="append", required=True, help="name=folder, the folder holding test/calls.json")
    ap.add_argument("--owner-key", type=Path, default=None)
    ap.add_argument("--owner-calls", type=Path, default=None)
    ap.add_argument("--candidates", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-sheet", action="store_true", help="skip drawing the disagreement pictures")
    a = ap.parse_args(argv)
    refuse_confirm(a.labelled_set, a.split, a.owner_key, a.owner_calls, *a.candidates)
    rows = read_labelled(a.labelled_set)
    split = json.loads(a.split.read_text())
    _, test = make_lists(rows, split)
    for r in test:
        r["kind"] = norm(r["kind"])
        r["call_1"], r["call_2"] = norm(r.get("call_1")), norm(r.get("call_2"))
    read = [r for r in test if r["part"] == "read"]
    confirm_keys = {key_of(r) for r in rows if r["part"] == "confirmation"}
    settled = {key_of(r): r["kind"] for r in read}
    owner = owner_check(a.owner_key, a.owner_calls, settled)
    checkers = push_checkers(a.candidates)

    a.out.mkdir(parents=True, exist_ok=True)
    results, lines = {}, [f"read set: {len(read)} pictures on {len({r['pair'] for r in read})} test pairs; "
                          f"confirmation slice unread; {owner['text']}"]
    dis_rows = []
    for spec in a.judge:
        name, _, folder = spec.partition("=")
        res = score_judge(name, read_judge(name, Path(folder), test, confirm_keys), read, test, owner, checkers)
        lines += lines_for(res)
        conf = Counter((r["kind"], r["judge"] or "uncovered") for r in res["_rows"])
        cols = sorted({c for _, c in conf})
        with open(a.out / f"confusion-{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["eye \\ judge"] + cols)
            for k in KINDS:
                w.writerow([k] + [conf[(k, c)] for c in cols])
        dis_rows += [{"judge": name, "picture": r["picture"], "a": r["a"], "b": r["b"], "pair": r["pair"],
                      "eye": r["kind"], "judge_kind": r["judge"], "renamed": r.get("renamed", "")}
                     for r in res["_rows"] if r["judge"] != r["kind"]]
        if not a.no_sheet and name != "old" and res["verdict"].startswith("meets every read bar"):
            res["disagreement_pictures"] = draw_disagreements(res, a.out)
        res.pop("_rows")
        results[name] = res
    with open(a.out / "disagreements.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["judge", "picture", "a", "b", "pair", "eye", "judge_kind", "renamed"])
        w.writeheader()
        w.writerows(dis_rows)
    (a.out / "bars.json").write_text(json.dumps({"owner": owner, "read_pictures": len(read),
                                                 "test_pairs": split["test_pairs"], "judges": results}, indent=1))
    print("\n".join(lines))
    print(f"wrote {a.out / 'bars.json'}, the confusion tables and disagreements.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
