#!/usr/bin/env python3
"""Paper 2's first result: the readers. The checkpoint pick (step 5), the runner-up judge's shares while the dial trained
(step 4), and the verdict (step 8).

plans/paper-2-dials/plans/03-identity-dial-first-result.md and its review file. The verdict comes from the settled blind
calls only, by paper2_first_result_bars.py's rule; the judges are printed beside it, never instead of it. The kept judge
is the trained species judge (its calls from paper2_first_checks.py judge); the runner-up judge is the vision-language
judge (Qwen3-VL-8B, paper2_judge_vlm.py at its chosen threshold), a different model family, so a dial that pleases the
trained head without improving the picture opens a gap between the two.

Settling. Two reviewers call every page of a sheet into calls-template.csv's layout (name, kind, note); a page takes their
kind where they agree and the third reviewer's where they split. A page with no settled kind stops the reader, which
writes unresolved.json.

  pick        step 5: the 96 tuning pictures (CO3's first draw, the 500-step and the 1,000-step checkpoints). Each row's
              six-kind shares by blind calls, the two judges' two-clear shares, the gaming gap (the trained head's
              two-clear share minus the runner-up's) and its rise from CO3, the not-two count (hybrid, extra subject and
              other: pictures that do not hold two animal bodies) against CO3's. Picks the checkpoint with the higher
              blind two-clear share (a tie to 500). Writes pick.json; exits 3 when the gaming alarm fires.
  runner-up   step 4: the runner-up's calls on the logged tuning pictures, joined to the trained head's shares at each
              logged step. Writes judges-during-training.json and adds the runner-up's share and the gap to the training
              run's W&B record on the judge/train_step axis.
  read        step 8: the 240 test pictures. Writes verdict.json, verdict.txt, matched-sheet.png (the 16 test pairs at
              the first seed, CO3, the dial and the random dial side by side with each settled kind) and ceiling-sheet.png.

    python3 paper2_first_read.py pick --key <pick>/blind/key/key.json --calls c1.csv c2.csv --third c3.csv \\
        --trained <pick>/trained/trained-calls.json --runner-up <pick>/runner-up/calls.json --out <pick>/pick.json
    python3 paper2_first_read.py read --key <test>/key.json --calls c1.csv c2.csv --third c3.csv \\
        --trained <test>/trained/trained-calls.json --sets <checks>/sets.json --out <paper2-first-identity-read>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paper2_first_result_bars as fb  # noqa: E402
from paper2_kinds import ALL_KINDS, BORROWED_PART, EXTRA_SUBJECT, HYBRID, OTHER_KIND, TWO_CLEAR, TWO_OF_ONE_KIND, \
    WRONG_SPECIES  # noqa: E402

NOT_TWO = (HYBRID, EXTRA_SUBJECT, OTHER_KIND)     # pictures that do not hold two animal bodies
NOT_TWO_RISE_SHOW = 3                             # pictures above CO3: shown to the owner before step 6
TRADE_RISE_REPORT = 3                             # pictures above CO3 in one failure kind: reported with the pictures
JUDGE_AGREE_MIN = 43                              # of 48: step 0's 0.90 bar rate, the review file's question


def norm_kind(k: str) -> str:
    k = "_".join(str(k or "").strip().lower().replace("-", " ").replace("_", " ").split())
    if k not in ALL_KINDS:
        raise SystemExit(f"unknown kind {k!r}; the kinds are {', '.join(ALL_KINDS)}")
    return k


def read_calls(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("kind") or "").strip()]
    out = {}
    for r in rows:
        if r["name"] in out:
            raise SystemExit(f"{path}: {r['name']} is called twice")
        out[r["name"]] = norm_kind(r["kind"])
    return out


def settle(names, first: dict, second: dict, third: dict) -> tuple[dict, dict, list]:
    """{name: kind}, {name: who decided}, and the names with no settled kind."""
    settled, by, unresolved = {}, {}, []
    for n in names:
        if n in first and n in second and first[n] == second[n]:
            settled[n], by[n] = first[n], "both reviewers"
        elif n in third and n in first and n in second:
            settled[n], by[n] = third[n], "the third reviewer"
        else:
            unresolved.append(n)
    return settled, by, unresolved


def load_settled(key_path: Path, calls: list[Path], third: Path | None, out_dir: Path) -> tuple[dict, dict]:
    """The key's pictures {name: entry}, each with its settled kind, or a stop naming the unresolved pages."""
    key = json.loads(Path(key_path).read_text())["pictures"]
    first, second = read_calls(calls[0]), read_calls(calls[1])
    settled, by, unresolved = settle(key, first, second, read_calls(third) if third else {})
    if unresolved:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "unresolved.json").write_text(json.dumps(unresolved, indent=1))
        print(f"STOP: {len(unresolved)} pages have no settled kind (the two reviewers split with no third call, or a "
              f"call is missing); {out_dir / 'unresolved.json'}")
        raise SystemExit(4)
    agreed = sum(1 for n in key if first.get(n) == second.get(n))
    for n, e in key.items():
        e["kind"], e["settled_by"], e["seed"] = settled[n], by[n], int(e["seed"])
    return key, {"pages": len(key), "reviewers_agree": agreed, "reviewer_agreement": round(agreed / len(key), 4)}


def judge_kinds(path: Path) -> dict[str, str | None]:
    """{picture: kind} from a trained-calls.json or a crop judge's calls.json."""
    return {p["picture"]: p.get("kind") for p in json.loads(Path(path).read_text())["pictures"]}


def shares(kinds: list[str]) -> dict:
    n = len(kinds)
    c = Counter(kinds)
    return {k: {"count": c.get(k, 0), "share": round(c.get(k, 0) / n, 4) if n else None} for k in ALL_KINDS}


# ---------------------------------------------------------------------------------------------------------------- the pick

def pick(key: dict, trained: dict, runner: dict) -> dict:
    rows = defaultdict(list)
    for e in key.values():
        rows[e["row"]].append(e)
    if "co3" not in rows or not {"ckpt-500", "ckpt-1000"} <= set(rows):
        raise SystemExit(f"the pick sheet's rows are {sorted(rows)}; it needs co3, ckpt-500 and ckpt-1000")
    table = {}
    for name, es in rows.items():
        kinds = [e["kind"] for e in es]
        missing = [e["picture_path"] for e in es if e["picture_path"] not in trained or e["picture_path"] not in runner]
        if missing:
            raise SystemExit(f"{len(missing)} {name} pictures lack a judge's call, for example {missing[0]}")
        t_share = sum(trained[e["picture_path"]] == TWO_CLEAR for e in es) / len(es)
        r_share = sum(runner[e["picture_path"]] == TWO_CLEAR for e in es) / len(es)
        table[name] = {"n": len(es), "blind": shares(kinds), "blind_two_clear_share": round(kinds.count(TWO_CLEAR) / len(es), 4),
                       "not_two": sum(k in NOT_TWO for k in kinds), "trained_two_clear_share": round(t_share, 4),
                       "runner_up_two_clear_share": round(r_share, 4), "gaming_gap": round(t_share - r_share, 4)}
    co3 = table["co3"]
    for c in ("ckpt-500", "ckpt-1000"):
        t = table[c]
        t["gap_rise_from_co3"] = round(t["gaming_gap"] - co3["gaming_gap"], 4)
        t["gaming_alarm"] = fb.gaming_alarm(co3["gaming_gap"], t["gaming_gap"])
        t["not_two_above_co3"] = t["not_two"] - co3["not_two"]
        t["show_owner_not_two"] = t["not_two_above_co3"] >= NOT_TWO_RISE_SHOW
    picked = 500 if table["ckpt-500"]["blind_two_clear_share"] >= table["ckpt-1000"]["blind_two_clear_share"] else 1000
    alarm = table["ckpt-500"]["gaming_alarm"] or table["ckpt-1000"]["gaming_alarm"]
    return {"picked": picked, "checkpoint": f"ckpt-{picked}",
            "rule": "the higher two-clear share by blind calls on the 32 tuning pictures; a tie to the 500-step checkpoint",
            "gaming_alarm": alarm, "gaming_bar": fb.FIRST_GAMING_GAP_MAX_RISE,
            "show_owner_not_two": [c for c in ("ckpt-500", "ckpt-1000") if table[c]["show_owner_not_two"]],
            "rows": table}


def cmd_pick(a) -> int:
    out = Path(a.out)
    key, agreement = load_settled(a.key, a.calls, a.third, out.parent)
    result = {**pick(key, judge_kinds(a.trained), judge_kinds(a.runner_up)), "reviewers": agreement,
              "key": str(a.key), "calls": [str(c) for c in a.calls] + ([str(a.third)] if a.third else [])}
    result["settled"] = {n: {k: e[k] for k in ("picture_path", "row", "pair", "seed", "kind", "settled_by")}
                         for n, e in key.items()}
    out.write_text(json.dumps(result, indent=1))
    for name, t in result["rows"].items():
        print(f"{name:10s} blind two clear {t['blind_two_clear_share']:.3f}; trained head {t['trained_two_clear_share']:.3f}, "
              f"runner-up {t['runner_up_two_clear_share']:.3f}, gap {t['gaming_gap']:+.3f}"
              + (f" (rise {t['gap_rise_from_co3']:+.3f})" if "gap_rise_from_co3" in t else "")
              + f"; not two {t['not_two']} of {t['n']}")
    print(f"picked {result['checkpoint']}; gaming alarm {'FIRES' if result['gaming_alarm'] else 'quiet'} "
          f"(bar: a rise of more than {fb.FIRST_GAMING_GAP_MAX_RISE}); wrote {out}")
    if result["show_owner_not_two"]:
        print(f"SHOW THE OWNER: {', '.join(result['show_owner_not_two'])} hold {NOT_TWO_RISE_SHOW} or more not-two "
              "pictures above CO3; the owner sees them before step 6")
    if result["gaming_alarm"]:
        print("STOP: the gaming alarm fired; the owner chooses between kept draws and a smaller step size")
        return 3
    return 0


# ---------------------------------------------------------------------------------------------------------------- the runner-up during training

def cmd_runner_up(a) -> int:
    train = Path(a.train)
    runner = judge_kinds(a.calls)
    steps = []
    for line in (train / "logged.jsonl").read_text().splitlines():
        r = json.loads(line)
        pics = [p["picture"] for p in r["pictures"]]
        missing = [p for p in pics if p not in runner]
        if missing:
            raise SystemExit(f"step {r['step']}: {len(missing)} logged pictures have no runner-up call, e.g. {missing[0]}")
        share = sum(runner[p] == TWO_CLEAR for p in pics) / len(pics)
        steps.append({"step": r["step"], "n": len(pics), "trained_two_clear_share": r["trained_two_clear_share"],
                      "runner_up_two_clear_share": round(share, 4),
                      "gap": round(r["trained_two_clear_share"] - share, 4)})
    first = steps[0]["gap"]
    for s in steps:
        s["gap_rise_from_step_0"] = round(s["gap"] - first, 4)
    out = {"judges": {"trained": "the trained species judge (the dial's reward)",
                      "runner_up": "the vision-language judge, Qwen3-VL-8B"},
           "axis": "y: two-clear share of the logged tuning pictures; x: training step", "steps": steps}
    (train / "judges-during-training.json").write_text(json.dumps(out, indent=1))
    for s in steps:
        print(f"step {s['step']:5d}: trained head {s['trained_two_clear_share']:.3f}, runner-up "
              f"{s['runner_up_two_clear_share']:.3f}, gap {s['gap']:+.3f} (rise {s['gap_rise_from_step_0']:+.3f})")
    run_file = train / "wandb-run.json"
    if a.wandb_mode != "disabled" and run_file.exists():
        import wandb
        info = json.loads(run_file.read_text())
        run = wandb.init(id=info["id"], project=info["project"], entity=info["entity"], resume="must", mode=a.wandb_mode)
        run.define_metric("judge/train_step")
        run.define_metric("judge/*", step_metric="judge/train_step")
        for s in steps:
            run.log({"judge/train_step": s["step"], "judge/runner_up_two_clear_share": s["runner_up_two_clear_share"],
                     "judge/gaming_gap": s["gap"]})
        run.finish()
        print(f"added the runner-up's shares to W&B run {info['id']}")
    return 0


# ---------------------------------------------------------------------------------------------------------------- the verdict

def matched(rows_by: dict, row_a: str, row_b: str, set_: str, pairs=None) -> dict:
    """McNemar's counts for row_a against row_b on matched pictures (same pair, same seed)."""
    a_ = {(e["pair"], e["seed"]): e["kind"] for e in rows_by[(row_a, set_)] if pairs is None or e["pair"] in pairs}
    b_ = {(e["pair"], e["seed"]): e["kind"] for e in rows_by[(row_b, set_)] if pairs is None or e["pair"] in pairs}
    if set(a_) != set(b_):
        raise SystemExit(f"{row_a} and {row_b} on {set_} are not drawn on the same pairs and seeds")
    wins = [k for k in a_ if a_[k] == TWO_CLEAR and b_[k] != TWO_CLEAR]
    losses = [k for k in a_ if b_[k] == TWO_CLEAR and a_[k] != TWO_CLEAR]
    n = len(a_)
    lead = (sum(v == TWO_CLEAR for v in a_.values()) - sum(v == TWO_CLEAR for v in b_.values())) / n
    return {"n": n, "wins": len(wins), "losses": len(losses), "lead": round(lead, 4),
            "p_one_sided": round(fb.one_sided_mcnemar_p(len(wins), len(losses)), 6),
            "win_keys": wins, "a": a_, "b": b_}


def answer_on(rows_by, pairs=None) -> tuple[str, str, dict]:
    d = matched(rows_by, "dial", "co3", "test", pairs)
    r = matched(rows_by, "random", "co3", "test", pairs)
    n = d["n"]
    dial_b = sum(v == BORROWED_PART for v in d["a"].values()) / n
    co3_b = sum(v == BORROWED_PART for v in d["b"].values()) / n
    verdict, reason = fb.answer_and_reason(d["lead"], d["p_one_sided"], dial_b, co3_b, r["lead"])
    return verdict, reason, {"dial": d, "random": r, "dial_borrowed": round(dial_b, 4), "co3_borrowed": round(co3_b, 4)}


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("a", "b", "win_keys")}


def verdict(key: dict, trained: dict | None, sets: dict | None) -> dict:
    rows_by = defaultdict(list)
    for e in key.values():
        rows_by[(e["row"], e["set"])].append(e)
    counts = {f"{r} on {s}": len(es) for (r, s), es in sorted(rows_by.items())}
    fair = all(len(rows_by[(r, "test")]) == fb.MATCHED_PICTURES for r in ("co3", "dial", "random")) and all(
        len(rows_by[(r, "ceiling")]) == fb.MATCHED_PICTURES for r in ("co3", "dial"))
    ans, reason, parts = answer_on(rows_by)
    d, r = parts["dial"], parts["random"]
    per_row = {f"{rw} on {st}": shares([e["kind"] for e in es]) for (rw, st), es in sorted(rows_by.items())}

    # the sign test over pairs, and the lead per pair
    pairs = sorted({e["pair"] for e in rows_by[("dial", "test")]})
    per_pair, up, down = {}, 0, 0
    for p in pairs:
        dc = sum(v == TWO_CLEAR for (q, _), v in d["a"].items() if q == p)
        cc = sum(v == TWO_CLEAR for (q, _), v in d["b"].items() if q == p)
        rc = sum(v == TWO_CLEAR for (q, _), v in r["a"].items() if q == p)
        n_p = sum(1 for (q, _) in d["a"] if q == p)
        per_pair[p] = {"dial_two_clear": dc, "co3_two_clear": cc, "random_two_clear": rc, "of": n_p,
                       "lead": round((dc - cc) / n_p, 4)}
        up, down = up + (dc > cc), down + (dc < cc)
    sign = {"pairs_up": up, "pairs_down": down, "ties": len(pairs) - up - down,
            "p_one_sided": round(fb.one_sided_mcnemar_p(up, down), 6)}

    # the yes rule with each pair dropped
    drop = {}
    for p in pairs:
        v, why, _ = answer_on(rows_by, set(pairs) - {p})
        drop[p] = {"answer": v, "reason": why}
    carried_by = [p for p, v in drop.items() if ans == "yes" and v["answer"] != "yes"]

    # where the new two-clear pictures came from, and failures traded for others
    came_from = dict(Counter(d["b"][k] for k in d["win_keys"]))
    trade = {}
    for k in (EXTRA_SUBJECT, TWO_OF_ONE_KIND, WRONG_SPECIES, HYBRID):
        dk, ck = sum(v == k for v in d["a"].values()), sum(v == k for v in d["b"].values())
        trade[k] = {"dial": dk, "co3": ck, "rise": dk - ck, "report": dk - ck >= TRADE_RISE_REPORT}

    # species-novel test pairs
    novel = None
    if sets:
        names = {f"{x['a']}|{x['b']}" for x in sets["species_overlap"] if x["species_novel"]}
        sub = [p for p in pairs if p in names]
        if sub:
            v, why, sp = answer_on(rows_by, set(sub))
            novel = {"pairs": sub, "n_pairs": len(sub), "dial_vs_co3": strip(sp["dial"]),
                     "random_vs_co3": strip(sp["random"]), "rule_read_on_them": v}
        else:
            novel = {"pairs": [], "n_pairs": 0}

    ceiling = strip(matched(rows_by, "dial", "co3", "ceiling")) if rows_by.get(("dial", "ceiling")) else None

    # the kept judge against the blind calls
    judge = None
    if trained:
        judge = {}
        total, agree = 0, 0
        for (rw, st), es in sorted(rows_by.items()):
            ok = sum(trained.get(e["picture_path"]) == e["kind"] for e in es)
            judge[f"{rw} on {st}"] = {"agree": ok, "of": len(es), "below_43_of_48": st == "test" and ok < JUDGE_AGREE_MIN}
            if st == "test":
                total, agree = total + len(es), agree + ok
        judge["test_pictures_disagreeing_share"] = round(1 - agree / total, 4) if total else None
        judge["more_than_1_in_10_disagree"] = bool(total) and (total - agree) / total > 0.10

    return {"answer": ans, "reason": reason, "lead": d["lead"], "p_one_sided": d["p_one_sided"],
            "mcnemar": {"dial_two_clear_co3_not": d["wins"], "co3_two_clear_dial_not": d["losses"]},
            "random_minus_co3": r["lead"], "random_mcnemar": strip(r),
            "borrowed_share": {"dial": parts["dial_borrowed"], "co3": parts["co3_borrowed"]},
            "bars": {k: getattr(fb, k) for k in ("FIRST_TWO_CLEAR_LEAD_MIN", "FIRST_MCNEMAR_P_MAX",
                                                  "FIRST_BORROWED_MUST_FALL", "FIRST_RANDOM_DIAL_CLOSE",
                                                  "FIRST_NO_LEAD_MAX")},
            "rows": per_row, "pictures_per_row": counts, "comparison_fair": fair,
            "sign_test_over_pairs": sign, "per_pair": per_pair, "drop_one_pair": drop, "carried_by_pairs": carried_by,
            "new_two_clear_came_from_co3_kind": came_from, "traded_failures": trade, "species_novel": novel,
            "ceiling_dial_vs_co3": ceiling, "kept_judge_against_blind_calls": judge}


def sheet(key: dict, set_: str, rows: list[str], path: Path, tile: int = 256) -> None:
    from PIL import Image, ImageDraw
    by = {(e["row"], e["pair"], e["seed"]): e for e in key.values() if e["set"] == set_}
    seed = min(e["seed"] for e in key.values() if e["set"] == set_)
    pairs = sorted({e["pair"] for e in key.values() if e["set"] == set_})
    page = Image.new("RGB", (160 + len(rows) * tile, 20 + len(pairs) * (tile + 16)), "white")
    d = ImageDraw.Draw(page)
    for j, rw in enumerate(rows):
        d.text((160 + j * tile + 4, 4), rw, fill="black")
    for i, p in enumerate(pairs):
        y = 20 + i * (tile + 16)
        d.text((4, y + tile // 2), p.replace("|", " and "), fill="black")
        for j, rw in enumerate(rows):
            e = by.get((rw, p, seed))
            if e and Path(e["picture_path"]).is_file():
                with Image.open(e["picture_path"]) as im:
                    page.paste(im.convert("RGB").resize((tile, tile)), (160 + j * tile, y))
                d.text((160 + j * tile + 4, y + tile), e["kind"], fill="black")
    page.save(path)


def cmd_read(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key, agreement = load_settled(a.key, a.calls, a.third, out)
    v = verdict(key, judge_kinds(a.trained) if a.trained else None,
                json.loads(Path(a.sets).read_text()) if a.sets else None)
    v["reviewers"] = agreement
    v["key"], v["calls"] = str(a.key), [str(c) for c in a.calls] + ([str(a.third)] if a.third else [])
    v["settled"] = {n: {k: e[k] for k in ("picture_path", "row", "set", "pair", "seed", "kind", "settled_by")}
                    for n, e in key.items()}
    (out / "verdict.json").write_text(json.dumps(v, indent=1))
    lines = [f"answer: {v['answer']} ({v['reason']})",
             f"two-clear lead of the dial over CO3: {v['lead']:+.4f} (yes at {fb.FIRST_TWO_CLEAR_LEAD_MIN} or more, no at "
             f"{fb.FIRST_NO_LEAD_MAX} or less); McNemar {v['mcnemar']['dial_two_clear_co3_not']} up, "
             f"{v['mcnemar']['co3_two_clear_dial_not']} down, one-sided p {v['p_one_sided']}",
             f"borrowed part: dial {v['borrowed_share']['dial']}, CO3 {v['borrowed_share']['co3']}; random dial minus "
             f"CO3 {v['random_minus_co3']:+.4f} (within {fb.FIRST_RANDOM_DIAL_CLOSE} for a yes)",
             f"sign test over pairs: {v['sign_test_over_pairs']}", f"fair comparison (48 a row): {v['comparison_fair']}",
             f"new two-clear pictures came from CO3's: {v['new_two_clear_came_from_co3_kind']}",
             f"carried by one pair: {v['carried_by_pairs'] or 'no'}",
             f"species-novel pairs: {(v['species_novel'] or {}).get('n_pairs')}",
             f"ceiling: {v['ceiling_dial_vs_co3']}"]
    if v["kept_judge_against_blind_calls"]:
        lines.append(f"kept judge against blind calls: {v['kept_judge_against_blind_calls']}")
    (out / "verdict.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sheet(key, "test", ["co3", "dial", "random"], out / "matched-sheet.png")
    if any(e["set"] == "ceiling" for e in key.values()):
        sheet(key, "ceiling", ["co3", "dial"], out / "ceiling-sheet.png")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    r = sub.add_parser("read")
    for s in (p, r):
        s.add_argument("--key", type=Path, required=True, help="the sheet's key.json")
        s.add_argument("--calls", type=Path, nargs=2, required=True, help="the two reviewers' filled calls")
        s.add_argument("--third", type=Path, default=None, help="the third reviewer's calls on the split pages")
    p.add_argument("--trained", type=Path, required=True, help="paper2_first_checks.py judge's trained-calls.json")
    p.add_argument("--runner-up", type=Path, required=True, help="paper2_judge_vlm.py's calls.json on the same pictures")
    p.add_argument("--out", type=Path, required=True, help="pick.json")
    r.add_argument("--trained", type=Path, default=None, help="the trained head's calls on the 240 pictures")
    r.add_argument("--sets", type=Path, default=None, help="step 2's sets.json, for the species-novel pairs")
    r.add_argument("--out", type=Path, required=True, help="the paper2-first-identity-read folder")
    u = sub.add_parser("runner-up")
    u.add_argument("--train", type=Path, required=True, help="the training run folder (logged.jsonl)")
    u.add_argument("--calls", type=Path, required=True, help="the runner-up's calls.json on every logged picture")
    u.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    a = ap.parse_args(argv)
    return {"pick": cmd_pick, "read": cmd_read, "runner-up": cmd_runner_up}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
