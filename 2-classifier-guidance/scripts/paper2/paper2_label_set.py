"""Fill the labelled set and split the pairs (species judge step 0.8, `paper2-label-set`).

Reads the blind batches named in a manifest (a JSON list, one object a batch):

    {"name": "batch-01", "key": ".../batch-01/key/key-relinked.json",
     "calls": [".../calls-1.csv", ".../calls-2.csv", ".../calls-3.csv"]}

Each calls file has the sheet's columns name, kind, note (calls-template.csv filled in); the third holds only the
pictures reviewers 1 and 2 split on. A kind is written as the sheet lists it ("two clear") or as in code ("two_clear").
The rubric version of a batch is its key's "rules_version" (a key with none was built with version 1); a manifest
"rules_version", if given, must agree.

What it does, in order:

  1. settles each batch: reviewer 1's kind where reviewers 1 and 2 agree, else the third reviewer's, else the picture is
     unsettled and left out (counted);
  2. merges the batches by picture path, a later batch in the manifest replacing an earlier one's label (a recall);
  3. refuses to mix rubric versions where they differ (VERSION_KINDS): once any batch of a version is in the set, a
     picture whose final kind that version governs must take its label from a batch of that version or later. Version 2
     governs two clear and borrowed part (it changed what counts as a borrowed part); version 3 governs hybrid (a coat
     never makes a body mixed, which can move a hybrid to borrowed part); version 4 governs wrong species and extra
     subject (what counts as a body, and a close relative is OTHER). Every other label stands from any version;
  4. reports reviewer agreement per settled kind (REVIEWER_AGREE_KIND_MIN) and on two clear against borrowed part
     (CLEAR_VS_BORROWED_AGREE_MIN), each read on the batch that gave the picture its final label;
  5. splits the pairs into choosing and test pairs by the first draw from SPLIT_SEED (0) that meets the size table:
     on choosing pairs at least PER_KIND_CHOOSING_MIN (15) of each kind; on test pairs, for each kind, a confirmation
     slice of PER_KIND_CONFIRM (10) drawn by CONFIRM_SEED (1), and from the rest PER_KIND_TEST_READ (40) read pictures,
     at least LOOKALIKE_PER_KIND_MIN (20) on look-alike pairs, from at least PER_KIND_TEST_PAIRS_MIN (4) pairs, at most
     PER_KIND_ONE_PAIR_MAX (12) from one pair, at most RENAMED_PER_KIND_MAX (20) renamed. Crow and raven are in neither
     (REPORTED_APART). Each draw puts every pair on the choosing side with probability 0.3; over every draw tried,
     counts.txt and split-diagnostic.json report how often each kind missed each rule, so what blocks a split shows;
  6. draws the owner's 60, 10 a kind from the read test pictures, by OWNER_SEED (4), and marks the unselected CO3
     pictures that landed on test pairs (outside the confirmation slice) as the natural-mix sample; if fewer than
     UNSELECTED_TEST (100) did, writes unselected-next.csv, the shortfall's worth of uncalled CO3-alone candidates on
     test pairs drawn by UNSELECTED_SEED (2), for the next batch.

Writes, under --out: labelled-set.csv (every settled picture: its kind, its rubric version, its part, choosing, read,
spare, confirmation or apart; a confirmation picture's kind is left blank there), confirm.json (the confirmation
slice's kinds, sealed: paper2_score.py refuses to open it until owner-choice.json exists), split.json, counts.txt
(each kind per pair type and per pair, the agreement, and every shortfall), and owner-list.csv (the owner's 60 as a
picture list for paper2_sheets.py, with no kind). If no draw meets the table within SPLIT_ATTEMPTS_MAX, nothing is
split: counts.txt names each short kind, by how many, overall and on look-alike pairs, and next-candidates.csv lists the
uncalled pictures of --candidates whose earlier call points at a short kind.

Renamed pictures (paper2_renamed.py: a settled two-clear picture shown with one asked name replaced by its stand-in)
come in as a batch like any other, their key entries marked renamed with their renaming entry. A settled picture's
identity is its path, and for a renamed one its renaming too, so a renamed copy never replaces the natural label of the
same file. They keep the label the reviewers settle, are held to at most 20 of a kind's 40 read, and are reported apart
from the natural pictures, per kind and per renaming entry; a kind whose read holds fewer than 20 natural pictures is
marked MOSTLY RENAMED.

    python3 paper2_label_set.py --batches batches.json --candidates <paper2-collect>/candidates.csv --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_bars import (CLEAR_VS_BORROWED_AGREE_MIN, CONFIRM_SEED, LOOKALIKE_PER_KIND_MIN, OWNER_SEED,  # noqa: E402
                         PER_KIND_CHOOSING_MIN, PER_KIND_CONFIRM, PER_KIND_ONE_PAIR_MAX, PER_KIND_TEST_PAIRS_MIN,
                         PER_KIND_TEST_READ, REPORTED_APART, REVIEWER_AGREE_KIND_MIN, SPLIT_ATTEMPTS_MAX, SPLIT_SEED,
                         RENAMED_PER_KIND_MAX, UNSELECTED_SEED, UNSELECTED_TEST, pair_type)
from paper2_collect import CO3_ALONE  # noqa: E402
from paper2_common import species_from_slug  # noqa: E402
from paper2_kinds import (ALL_KINDS, BORROWED_PART, EXTRA_SUBJECT, HYBRID, KINDS, OTHER_KIND, TWO_CLEAR,  # noqa: E402
                          WRONG_SPECIES)

FLAG_KINDS = {TWO_CLEAR, BORROWED_PART}     # the kinds the rubric's flag decides between
# Each rubric version names the kinds its change can move a picture into or out of. Once any batch of that version is
# in the set, a picture labelled with one of those kinds must take its label from a batch of that version or later.
# v2 changed what counts as a borrowed-part flag (two clear against borrowed part); v3 says a coat never makes a body
# mixed, which can move a hybrid to borrowed part but moves no picture out of two clear or borrowed part.
# v4 says what counts as a body and that a close relative is OTHER: that decides wrong species, and the body count it
# changes can move a picture out of extra subject (a blurred third bird no longer counts).
VERSION_KINDS = {2: FLAG_KINDS, 3: {HYBRID}, 4: {WRONG_SPECIES, EXTRA_SUBJECT}}
CHOOSING_SHARE = 0.3
OWNER_PER_KIND = 10
EARLIER_CALL_POINTS_AT = {             # which earlier calls suggest a candidate may be a short kind
    TWO_CLEAR: {"two_clear", "own_parts", "composed"},
    BORROWED_PART: {"borrowed", "two_bodies"},
    "hybrid": {"hybrid"},
    "two_of_one_kind": {"two_of_one", "two_bodies", "one"},
    "wrong_species": {"not_two", "two_bodies"},
    "extra_subject": {"not_two", "two_bodies"},
}


# ---------------------------------------------------------------- reading and settling

def norm_kind(k: str) -> str:
    k = k.strip().lower().replace("-", " ").replace("_", " ")
    k = "_".join(k.split())
    if k not in ALL_KINDS:
        raise SystemExit(f"unknown kind {k!r}; the kinds are {', '.join(ALL_KINDS)}")
    return k


def read_calls(path: Path) -> dict[str, dict]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("kind") or "").strip()]
    out = {}
    for r in rows:
        if r["name"] in out:
            raise SystemExit(f"{path}: {r['name']} is called twice")
        out[r["name"]] = {"kind": norm_kind(r["kind"]), "note": (r.get("note") or "").strip()}
    return out


def picture_of(entry: dict) -> str:
    p = entry.get("picture_path") or entry.get("picture")
    if not p:
        raise SystemExit("a key entry has no picture path; use the batch's relinked key")
    return p


def item_of(picture: str, renaming: str = "") -> str:
    """A settled picture's identity: its path, and for a renamed picture its renaming entry too, so a renamed picture
    never replaces the natural label of the same file."""
    return f"{picture}|renamed {renaming}" if renaming else picture


def settle_batch(b: dict) -> dict:
    key = json.loads(Path(b["key"]).read_text())
    version = int(key.get("rules_version", 1))
    if "rules_version" in b and int(b["rules_version"]) != version:
        raise SystemExit(f"{b['name']}: the manifest says rubric version {b['rules_version']}, its key says {version}")
    calls = [read_calls(Path(c)) for c in b["calls"]]
    one, two = calls[0], calls[1]
    three = calls[2] if len(calls) > 2 else {}
    settled, unsettled = {}, []
    for name, entry in key["pictures"].items():
        if name not in one or name not in two:
            unsettled.append(name)
            continue
        agree = one[name]["kind"] == two[name]["kind"]
        kind = one[name]["kind"] if agree else three.get(name, {}).get("kind")
        if kind is None:
            unsettled.append(name)
            continue
        a, b_ = (entry.get("a"), entry.get("b")) if entry.get("a") else species_from_slug(entry["pair"])
        renamed = str(entry.get("renamed", "")) == "True"
        renaming = entry.get("renaming", "") if renamed else ""
        settled[item_of(picture_of(entry), renaming)] = {
            "item": item_of(picture_of(entry), renaming), "renamed": renamed, "renaming": renaming,
            "picture": picture_of(entry), "a": a, "b": b_, "pair": entry.get("pair"), "seed": entry.get("seed"),
            "source": entry.get("source", ""), "unselected": str(entry.get("unselected", "")) == "True",
            "kind": kind, "rules_version": version, "batch": b["name"], "name": name,
            "call_1": one[name]["kind"], "call_2": two[name]["kind"], "agreed": agree,
            "notes": " | ".join(n for n in (one[name]["note"], two[name]["note"], three.get(name, {}).get("note", "")) if n)}
    return {"name": b["name"], "version": version, "settled": settled, "unsettled": unsettled}


def merge(batches: list[dict]) -> tuple[dict, list[str]]:
    labels: dict[str, dict] = {}
    for b in batches:
        for pic, row in b["settled"].items():
            if pic in labels:
                row = {**row, "replaces": f"{labels[pic]['batch']} ({labels[pic]['kind']}, v{labels[pic]['rules_version']})"}
            labels[pic] = row
    newest = max((b["version"] for b in batches), default=1)
    refused = []
    for r in labels.values():
        for version, kinds in sorted(VERSION_KINDS.items()):
            if version <= newest and r["kind"] in kinds and r["rules_version"] < version:
                refused.append(f"{r['batch']}/{r['name']}: {r['kind']} under rubric v{r['rules_version']}, "
                               f"where rubric v{version} is in the set and governs {r['kind']}")
                break
    return labels, sorted(refused)


def agreement(labels: dict) -> dict:
    by_kind = defaultdict(list)
    for r in labels.values():
        by_kind[r["kind"]].append(r["agreed"])
    per_kind = {k: {"n": len(v), "agreed": sum(v), "share": round(sum(v) / len(v), 4)} for k, v in sorted(by_kind.items())}
    both = [r for r in labels.values() if r["call_1"] in FLAG_KINDS and r["call_2"] in FLAG_KINDS]
    cvb = {"n": len(both), "agreed": sum(r["call_1"] == r["call_2"] for r in both)}
    cvb["share"] = round(cvb["agreed"] / cvb["n"], 4) if cvb["n"] else None
    return {"per_kind": per_kind, "clear_vs_borrowed": cvb,
            "kinds_under_bar": [k for k, v in per_kind.items() if k in KINDS and v["share"] < REVIEWER_AGREE_KIND_MIN],
            "clear_vs_borrowed_under_bar": cvb["share"] is not None and cvb["share"] < CLEAR_VS_BORROWED_AGREE_MIN}


# ---------------------------------------------------------------- the split

REASONS = ("choosing under 15", "confirmation under 10", "test under 50 (confirmation 10 and read 40)",
           "read under 40: the 12-a-pair cap", "read under 40: the 20-renamed cap", "look-alike under 20",
           "pairs under 4")


def read_selection(pictures: list[dict], rng: random.Random) -> tuple[list[dict], list[str]]:
    """40 read pictures of one kind: look-alike first up to 20, then any; at most 12 a pair and at most
    RENAMED_PER_KIND_MAX renamed. Returns the pictures and every rule of the size table they miss (none if met)."""
    order = pictures[:]
    rng.shuffle(order)
    chosen, taken, per_pair = [], set(), Counter()
    look = renamed = 0
    skipped_renamed = False
    for phase in ("look-alike", "any"):
        for r in order:
            if len(chosen) == PER_KIND_TEST_READ:
                break
            if r["item"] in taken or per_pair[r["pair"]] >= PER_KIND_ONE_PAIR_MAX:
                continue
            if phase == "look-alike" and (r["pair_type"] != "look-alike" or look >= LOOKALIKE_PER_KIND_MIN):
                continue
            if r["renamed"] and renamed >= RENAMED_PER_KIND_MAX:
                skipped_renamed = True
                continue
            chosen.append(r)
            taken.add(r["item"])
            per_pair[r["pair"]] += 1
            look += r["pair_type"] == "look-alike"
            renamed += r["renamed"]
    fails = []
    if len(chosen) < PER_KIND_TEST_READ:
        fails.append("read under 40: the 20-renamed cap" if skipped_renamed else "read under 40: the 12-a-pair cap")
    if look < LOOKALIKE_PER_KIND_MIN:
        fails.append("look-alike under 20")
    if len(per_pair) < PER_KIND_TEST_PAIRS_MIN:
        fails.append("pairs under 4")
    return chosen, fails


def judge_draw(by_kind: dict[str, list[dict]], choosing: set[str]) -> tuple[dict, dict[str, list[str]]]:
    """Every kind under one draw of the pair split: its parts, and every rule of the size table it misses."""
    parts, fails = {}, {}
    for kind in KINDS:
        mine = by_kind.get(kind, [])
        why = []
        if sum(r["pair"] in choosing for r in mine) < PER_KIND_CHOOSING_MIN:
            why.append("choosing under 15")
        test = [r for r in mine if r["pair"] not in choosing]
        if len(test) < PER_KIND_CONFIRM:
            why.append("confirmation under 10")
        if len(test) < PER_KIND_CONFIRM + PER_KIND_TEST_READ:
            why.append("test under 50 (confirmation 10 and read 40)")
        else:
            confirm = random.Random(f"{CONFIRM_SEED}-{kind}").sample(test, PER_KIND_CONFIRM)
            held = {r["item"] for r in confirm}
            read, read_fails = read_selection([r for r in test if r["item"] not in held],
                                              random.Random(f"{SPLIT_SEED}-read-{kind}"))
            why += read_fails
            parts[kind] = {"confirm": confirm, "read": read}
        fails[kind] = why
    return parts, fails


def split(rows: list[dict]) -> tuple[set[str] | None, dict | None, int, dict]:
    """The first draw from SPLIT_SEED that meets the size table, and over every draw tried, how often each rule failed."""
    pairs = sorted({r["pair"] for r in rows})
    by_kind = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["item"]):
        by_kind[r["kind"]].append(r)
    rng = random.Random(SPLIT_SEED)
    diag = {"draws": 0, "fails": {k: Counter() for k in KINDS}, "only_kind_failing": Counter()}
    for attempt in range(1, SPLIT_ATTEMPTS_MAX + 1):
        choosing = {p for p in pairs if rng.random() < CHOOSING_SHARE}
        parts, fails = judge_draw(by_kind, choosing)
        diag["draws"] = attempt
        failing = [k for k, why in fails.items() if why]
        for k, why in fails.items():
            diag["fails"][k].update(why)
        if len(failing) == 1:
            diag["only_kind_failing"][failing[0]] += 1
        if not failing:
            return choosing, parts, attempt, diag
    return None, None, SPLIT_ATTEMPTS_MAX, diag


def diagnostic_lines(diag: dict) -> list[str]:
    n = diag["draws"]
    out = [f"split diagnostic over {n} draws (a draw fails when any kind misses any rule; shares are of all draws):"]
    for kind in KINDS:
        f = diag["fails"][kind]
        parts = ", ".join(f"{why} {f[why] / n:.0%}" for why in REASONS if f[why])
        out.append(f"  {kind:16s} {parts or 'never fails'}; the only kind failing in {diag['only_kind_failing'][kind]} draws")
    return out


def shortfalls(rows: list[dict]) -> dict:
    """What the size table needs of each kind across the whole set, whatever the split: a necessary condition."""
    need_all = PER_KIND_CHOOSING_MIN + PER_KIND_CONFIRM + PER_KIND_TEST_READ
    out = {}
    for kind in KINDS:
        mine = [r for r in rows if r["kind"] == kind]
        look = sum(r["pair_type"] == "look-alike" for r in mine)
        pairs = len({r["pair"] for r in mine})
        short = {}
        if len(mine) < need_all:
            short["pictures"] = need_all - len(mine)
        if look < LOOKALIKE_PER_KIND_MIN:
            short["look-alike pictures"] = LOOKALIKE_PER_KIND_MIN - look
        if pairs < PER_KIND_TEST_PAIRS_MIN + 1:
            short["pairs"] = PER_KIND_TEST_PAIRS_MIN + 1 - pairs
        out[kind] = {"have": len(mine), "look_alike": look, "pairs": pairs, "need": need_all, "short": short}
    return out


CO3_ALONE_MADE = {CO3_ALONE, "CO3, no push"}     # the collector's CO3-alone rows (M1, the box rounds' co3, round 08's)


def unselected_next(candidates: Path, called: set[str], test_pairs: set[str], short: int) -> list[dict]:
    """The top-up for a short unselected sample: uncalled CO3-alone candidates on test pairs, drawn by UNSELECTED_SEED,
    with no regard to earlier calls, in the collector's columns."""
    with open(candidates, newline="") as f:
        rows = list(csv.DictReader(f))
    pool = sorted((r for r in rows if r["made"] in CO3_ALONE_MADE and r["pair"] in test_pairs and r["picture"] not in called),
                  key=lambda r: r["picture"])
    return random.Random(UNSELECTED_SEED).sample(pool, min(short, len(pool)))


def next_candidates(candidates: Path, called: set[str], short_kinds: list[str]) -> list[dict]:
    with open(candidates, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r["picture"] in called:
            continue
        aims = [k for k in short_kinds if r.get("earlier_call") in EARLIER_CALL_POINTS_AT.get(k, set())]
        if aims:
            out.append({**r, "may_be": " ".join(aims)})
    out.sort(key=lambda r: (r["pair_type"] != "look-alike", r["may_be"], r["picture"]))
    return out


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=Path, required=True, help="the manifest: a JSON list of batches")
    ap.add_argument("--candidates", type=Path, default=None, help="paper2_collect.py's candidates.csv, for next candidates")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    manifest = json.loads(a.batches.read_text())
    batches = [settle_batch(b) for b in manifest]
    labels, refused = merge(batches)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"batches: " + "; ".join(f"{b['name']} rubric v{b['version']}: {len(b['settled'])} settled, "
                                      f"{len(b['unsettled'])} unsettled" for b in batches)]
    if refused:
        msg = (f"refused: {len(refused)} pictures hold a label from an older rubric than the newest one that governs "
               "that kind; call them again under that version:\n  " + "\n  ".join(refused))
        (out / "counts.txt").write_text("\n".join(lines) + "\n" + msg + "\n")
        print("\n".join(lines))
        raise SystemExit(msg)

    rows = []
    for r in labels.values():
        a_, b_ = r["a"], r["b"]
        r["pair_type"] = pair_type(a_, b_)
        r["apart"] = (a_, b_) in REPORTED_APART
        rows.append(r)
    agree = agreement(labels)
    lines.append(f"settled pictures: {len(rows)}; kinds {dict(sorted(Counter(r['kind'] for r in rows).items()))}")
    lines.append("reviewer agreement per settled kind (bar " + f"{REVIEWER_AGREE_KIND_MIN}): " + ", ".join(
        f"{k} {v['agreed']}/{v['n']} ({v['share']:.2f})" for k, v in agree["per_kind"].items()))
    cvb = agree["clear_vs_borrowed"]
    lines.append(f"two clear against borrowed part (bar {CLEAR_VS_BORROWED_AGREE_MIN}): {cvb['agreed']}/{cvb['n']}"
                 + (f" ({cvb['share']:.3f})" if cvb["share"] is not None else ""))
    if agree["kinds_under_bar"] or agree["clear_vs_borrowed_under_bar"]:
        lines.append(f"UNDER THE AGREEMENT BAR: {agree['kinds_under_bar']}"
                     + (" and two clear against borrowed part" if agree["clear_vs_borrowed_under_bar"] else ""))

    in_split = [r for r in rows if not r["apart"] and r["kind"] != OTHER_KIND]
    choosing, parts, attempts, diag = split(in_split)
    (out / "split-diagnostic.json").write_text(json.dumps(
        {"draws": diag["draws"], "fails": {k: dict(v) for k, v in diag["fails"].items()},
         "only_kind_failing": dict(diag["only_kind_failing"])}, indent=1))
    for r in rows:
        r["part"] = "apart" if r["apart"] else ("other" if r["kind"] == OTHER_KIND else "not split")
    result = 0
    if parts is None:
        sf = shortfalls(in_split)
        lines.append(f"NOT SPLIT: no draw of the {len({r['pair'] for r in in_split})} pairs in {attempts} met the size "
                     f"table. Each kind needs at least {PER_KIND_CHOOSING_MIN + PER_KIND_CONFIRM + PER_KIND_TEST_READ} "
                     f"pictures, {LOOKALIKE_PER_KIND_MIN} on look-alike pairs, over at least {PER_KIND_TEST_PAIRS_MIN + 1} pairs:")
        for kind, v in sf.items():
            lines.append(f"  {kind:16s} {v['have']:4d} pictures, {v['look_alike']:3d} look-alike, {v['pairs']:2d} pairs"
                         f"   short: {v['short'] or 'none overall (the split itself fails)'}")
        short_kinds = [k for k, v in sf.items() if v["short"]]
        lines += diagnostic_lines(diag)
        if a.candidates and short_kinds:
            nxt = next_candidates(a.candidates, {r["picture"] for r in rows}, short_kinds)
            with open(out / "next-candidates.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(nxt[0]) if nxt else ["picture"])
                w.writeheader()
                w.writerows(nxt)
            lines.append(f"next candidates for the short kinds: {len(nxt)} uncalled pictures in next-candidates.csv "
                         f"({sum(r['pair_type'] == 'look-alike' for r in nxt)} on look-alike pairs)")
        if "wrong_species" in short_kinds:
            lines.append("wrong species is short: once the whole pool is called, the renamed pictures come from "
                         "paper2_renamed.py (paper2_bars.RENAMING), called blind as a batch")
        lines.append(f"unselected sample: the top-up to {UNSELECTED_TEST} waits for the split, since it is drawn from test pairs")
        result = 1
    else:
        confirm_items = {r["item"] for p in parts.values() for r in p["confirm"]}
        read_items = {r["item"] for p in parts.values() for r in p["read"]}
        for r in in_split:
            r["part"] = ("choosing" if r["pair"] in choosing else "confirmation" if r["item"] in confirm_items
                         else "read" if r["item"] in read_items else "spare")
        owner = []
        for kind in KINDS:
            owner += random.Random(f"{OWNER_SEED}-{kind}").sample(parts[kind]["read"], OWNER_PER_KIND)
        test_pairs = sorted({r["pair"] for r in in_split} - choosing)
        unselected = [r for r in rows if r["unselected"] and not r["renamed"] and r["pair"] in test_pairs
                      and r["part"] != "confirmation"]
        for r in rows:
            r["unselected_sample"] = r in unselected
        (out / "split.json").write_text(json.dumps({"seed": SPLIT_SEED, "attempt": attempts, "choosing_pairs": sorted(choosing),
                                                    "test_pairs": test_pairs}, indent=1))
        (out / "confirm.json").write_text(json.dumps(
            {"sealed": "paper2_score.py does not open this until owner-choice.json exists", "seed": CONFIRM_SEED,
             "pictures": {r["item"]: r["kind"] for p in parts.values() for r in p["confirm"]}}, indent=1))
        with open(out / "owner-list.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["picture", "a", "b"])
            w.writerows((r["picture"], r["a"], r["b"]) for r in owner)
        lines.append(f"split at draw {attempts} of seed {SPLIT_SEED}: {len(choosing)} choosing pairs, {len(test_pairs)} test pairs")
        for kind in KINDS:
            read = parts[kind]["read"]
            lines.append(f"  {kind:16s} choosing {sum(r['kind'] == kind and r['part'] == 'choosing' for r in rows):3d}; "
                         f"read {len(read)} ({sum(r['pair_type'] == 'look-alike' for r in read)} look-alike, "
                         f"{len({r['pair'] for r in read})} pairs, most from one pair "
                         f"{max(Counter(r['pair'] for r in read).values())}; natural {sum(not r['renamed'] for r in read)}, "
                         f"renamed {sum(r['renamed'] for r in read)}"
                         + ("; MOSTLY RENAMED" if sum(not r["renamed"] for r in read) < PER_KIND_TEST_READ // 2 else "")
                         + f"); confirmation {PER_KIND_CONFIRM} (sealed)")
        renamed_read = [r for p in parts.values() for r in p["read"] if r["renamed"]]
        if renamed_read:
            lines.append("  renamed pictures read, by renaming entry: "
                         + ", ".join(f"{k} {n}" for k, n in sorted(Counter(r["renaming"] for r in renamed_read).items())))
        lines.append(f"unselected sample on test pairs: {len(unselected)} (the plan asks {UNSELECTED_TEST})")
        short = UNSELECTED_TEST - len(unselected)
        if short > 0:
            if a.candidates:
                top = unselected_next(a.candidates, {r["picture"] for r in rows}, set(test_pairs), short)
                with open(out / "unselected-next.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(top[0]) if top else ["picture", "a", "b"])
                    w.writeheader()
                    w.writerows(top)
                lines.append(f"  short by {short}: unselected-next.csv holds {len(top)} uncalled CO3-alone pictures on test "
                             f"pairs drawn by seed {UNSELECTED_SEED}, for the next batch"
                             + (f"; the pool has only {len(top)}" if len(top) < short else ""))
            else:
                lines.append(f"  short by {short}: give --candidates to draw the top-up for the next batch")
        lines.append(f"owner's 60: owner-list.csv, {OWNER_PER_KIND} a kind from the read pictures by seed {OWNER_SEED}")
    nat = Counter(r["kind"] for r in rows if not r["renamed"])
    ren = Counter(r["kind"] for r in rows if r["renamed"])
    if ren:
        lines.append("settled natural and renamed apart: " + ", ".join(
            f"{k} {nat[k]} natural, {ren[k]} renamed" for k in KINDS if ren[k]))
    lines.append("per pair: " + ", ".join(f"{p} {dict(Counter(r['kind'] for r in rows if r['pair'] == p))}"
                                          for p in sorted({r['pair'] for r in rows})))
    cols = ["picture", "renamed", "renaming", "a", "b", "pair", "pair_type", "seed", "source", "kind", "rules_version", "part",
            "unselected_sample", "batch", "name", "call_1", "call_2", "agreed", "replaces", "notes"]
    with open(out / "labelled-set.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["pair"], str(r["seed"]), r["item"])):
            sealed = r["part"] == "confirmation"
            w.writerow({**r, "kind": "" if sealed else r["kind"],
                        "call_1": "" if sealed else r["call_1"], "call_2": "" if sealed else r["call_2"],
                        "notes": "" if sealed else r["notes"], "unselected_sample": r.get("unselected_sample", False)})
    (out / "agreement.json").write_text(json.dumps(agree, indent=1))
    (out / "counts.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
