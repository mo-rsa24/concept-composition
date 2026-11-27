"""Paper 2's first result: step 2's checks (paper2_first_checks.py) and the readers (paper2_first_read.py).

The sets, the redraw count, checks.txt and the readers need no torch and run on the laptop's python; the window read,
the trained judge's calls and the rollout check run on CPU stand-ins under torch (skipped without it):

    python3 -m pytest tests/test_paper2_first_checks.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER2 = REPO / "2-classifier-guidance/scripts/paper2"
sys.path.insert(0, str(PAPER2))
sys.path.insert(0, str(REPO / "scripts/shared"))

import paper2_first_checks as fc  # noqa: E402
import paper2_first_read as fr  # noqa: E402
import paper2_first_result_bars as fb  # noqa: E402
from disallowed_subjects import is_allowed, is_showable  # noqa: E402

CHECKS = PAPER2 / "paper2_first_checks.py"
READ = PAPER2 / "paper2_first_read.py"


def split(look=(("cheetah", "cougar"), ("cat", "dog"), ("tiger", "dog"), ("lion", "tiger"), ("rabbit", "hare"),
                ("goat", "sheep"))):
    """A stand-in for step 0's split.json, built from step 0's own pair tables."""
    test = list(look) + [("boar", "pheasant"), ("eagle", "fox"), ("lobster", "whale")]
    choosing = [("duck", "goose"), ("wolf", "fox"), ("horse", "donkey"), ("bobcat", "quail"), ("zebra", "flamingo")]
    return {"test_pairs": [fc.slug(*p) for p in test], "choosing_pairs": [fc.slug(*p) for p in choosing]}


def pool():
    return sorted(json.loads(fc.POOL_FILE.read_text()))


def k(r):
    return frozenset((r["a"], r["b"]))


# ---------------------------------------------------------------------------------------------------------------- the sets

@pytest.mark.parametrize("n_look", [0, 3, 6])
def test_sets_meet_every_pass_line_of_the_plan(n_look):
    look = [("cheetah", "cougar"), ("cat", "dog"), ("tiger", "dog"), ("lion", "tiger"), ("rabbit", "hare"),
            ("goat", "sheep")][:n_look]
    s = fc.fix_sets(split(look), pool())
    assert s["passed"], s["checks"]
    assert len(s["tuning"]) == 4 and len(s["test"]) == 16 and len(s["ceiling"]) == 16
    assert all(r["type"] == "look-alike" for r in s["tuning"] + s["test"] + s["ceiling"])
    tun, tes, tra = ({k(r) for r in s[x]} for x in ("tuning", "test", "training"))
    assert not (tun & tes) and not (tun & tra) and not (tes & tra)
    assert {k(r) for r in s["ceiling"]} <= tra
    test_species = {x for r in s["test"] for x in (r["a"], r["b"])}
    assert not test_species & {x for r in s["new_training"] for x in (r["a"], r["b"])}
    assert len(s["new_training"]) >= 9 and sum(r["type"] == "look-alike" for r in s["training"]) >= 16
    assert s["lookalike_draw_share"] >= 0.5
    # step 0's look-alike test pairs are all test pairs, and the rest are new pairs from the candidates
    assert {fc.key(*p) for p in look} <= tes
    assert sum(r["source"].startswith("new") for r in s["test"]) == 16 - n_look


def test_tuning_pairs_come_from_the_eleven_and_never_from_step_0s_test_side():
    s = fc.fix_sets(split(), pool())
    eleven = {fc.key(*p) for p in fc.ELEVEN}
    assert {k(r) for r in s["tuning"]} <= eleven
    assert not {k(r) for r in s["tuning"]} & {fc.key(*fc.species_from_slug(x)) for x in split()["test_pairs"]}
    # drawn by seed 0: the same draw every time
    assert [r["slug"] for r in fc.fix_sets(split(), pool())["tuning"]] == [r["slug"] for r in s["tuning"]]


def test_every_pair_passes_the_subject_guard_and_every_shown_pair_its_display_hold():
    s = fc.fix_sets(split(), pool())
    for r in s["training"] + s["smoke"]:
        assert is_allowed(f"a {r['a']} and a {r['b']}"), r
    for r in s["tuning"] + s["test"] + s["ceiling"]:
        assert is_showable(f"a {r['a']} and a {r['b']}"), r
    left_out = {e["pair"]: e["why"] for e in s["excluded"]}
    assert left_out["a_crocodile__x__an_alligator"] == "fails the subject guard"
    assert left_out["a_turtle__x__a_tortoise"] == "held from display"


def test_training_is_the_pool_and_the_eleven_minus_tuning_and_test_plus_new_pairs():
    s = fc.fix_sets(split(), pool())
    tra = {k(r) for r in s["training"]}
    excluded = {fc.key(*fc.species_from_slug(e["pair"])) for e in s["excluded"] if e["why"] == "fails the subject guard"}
    expect = ({fc.key(*fc.species_from_slug(x)) for x in pool()} | {fc.key(*p) for p in fc.ELEVEN}) - excluded
    expect -= {k(r) for r in s["tuning"] + s["test"]}
    assert tra - {k(r) for r in s["new_training"]} == expect


def test_smoke_pairs_are_two_look_alike_and_two_unlike_training_pairs():
    s = fc.fix_sets(split(), pool())
    assert sorted(r["type"] for r in s["smoke"]) == ["look-alike", "look-alike", "unlike", "unlike"]
    assert {k(r) for r in s["smoke"]} <= {k(r) for r in s["training"]}


def test_every_species_has_an_order_and_every_candidate_is_look_alike_and_new():
    for x in pool():
        fc.pair_type(*fc.species_from_slug(x))
    seen = set()
    for a, b in fc.NEW_CANDIDATES:
        assert fc.pair_type(a, b) == "look-alike", (a, b)
        assert not {a, b} & seen, (a, b)
        seen |= {a, b}
        assert is_showable(f"a {a} and a {b}")
    with pytest.raises(SystemExit, match="no taxonomic order"):
        fc.pair_type("zorilla", "cat")


def test_sets_command_writes_every_file_step_2_names(tmp_path):
    (tmp_path / "split.json").write_text(json.dumps(split()))
    r = subprocess.run([sys.executable, str(CHECKS), "sets", "--split", str(tmp_path / "split.json"), "--out",
                        str(tmp_path / "out")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for f in ("sets.json", "overlaps.txt", "species-overlap.csv", "train-pairs.txt", "smoke-pairs.txt",
              "tuning-pairs.txt", "test-pairs.txt", "ceiling-pairs.txt", "choosing-pairs.txt"):
        assert (tmp_path / "out" / f).stat().st_size > 0, f
    lines = [ln.split("#")[0].split() for ln in (tmp_path / "out/train-pairs.txt").read_text().splitlines()]
    assert all(len(x) == 2 and float(x[1]) >= 1 for x in lines)
    assert len((tmp_path / "out/test-pairs.txt").read_text().splitlines()) == 16


def test_choose_block_takes_the_highest_mean_and_ties_go_late():
    agreement = [0.1] * 6 + [0.5] * 10 + [0.9] * 10 + [0.5] * 24
    assert fc.choose_block(agreement, 6) == (16, 26, pytest.approx(0.9))
    flat = [0.7] * 50
    assert fc.choose_block(flat, 6)[:2] == (40, 50)
    with pytest.raises(SystemExit):
        fc.choose_block([0.5] * 12, 6)


# ---------------------------------------------------------------------------------------------------------------- redraw and checks.txt

def calls_file(path, kinds):
    path.write_text(json.dumps({"pictures": [{"pair": f"p{i % 4}|q", "seed": 12001 + i // 4, "kind": kd}
                                             for i, kd in enumerate(kinds)]}))
    return path


@pytest.mark.parametrize("changed,within", [(0, True), (2, True), (3, False)])
def test_redraw_counts_kind_changes_against_the_bar(tmp_path, changed, within):
    first = ["two_clear"] * 32
    second = ["borrowed_part"] * changed + ["two_clear"] * (32 - changed)
    a, b = calls_file(tmp_path / "1.json", first), calls_file(tmp_path / "2.json", second)
    assert fc.main(["redraw", "--first", str(a), "--second", str(b), "--out", str(tmp_path / "redraw.json")]) == 0
    r = json.loads((tmp_path / "redraw.json").read_text())
    assert r["changed"] == changed and r["within_bar"] is within
    assert r["random_dial_band_needed"] == (None if within else pytest.approx(changed / 32))


def test_checks_stop_on_missing_parts_and_pass_when_all_are_there(tmp_path):
    assert fc.main(["checks", "--dir", str(tmp_path)]) == 1
    assert "STOP" in (tmp_path / "checks.txt").read_text()
    (tmp_path / "split.json").write_text(json.dumps(split()))
    fc.main(["sets", "--split", str(tmp_path / "split.json"), "--out", str(tmp_path)])
    for i in (1, 2):
        (tmp_path / f"co3-draw-{i}").mkdir()
        (tmp_path / f"co3-draw-{i}/draw.json").write_text(json.dumps({"pictures": [{}] * 32}))
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory/summary.json").write_text(json.dumps(
        {"seconds_per_step": 30.0, "memory": {"peak_mib": 47000, "card_mib": 97000, "judge_mib": 2000}}))
    (tmp_path / "memory/train-log.jsonl").write_text(json.dumps(
        {"reward": 0.2, "task_grad_norm": 0.1, "grad_norm": 0.1}) + "\n")
    (tmp_path / "rollout").mkdir()
    (tmp_path / "rollout/rollout.json").write_text(json.dumps(
        {"passed": True, "trainer_minus_co3_first_grey": 0.4, "co3_second_minus_first_grey": 0.6}))
    (tmp_path / "window").mkdir()
    (tmp_path / "window/window-read.json").write_text(json.dumps(
        {"window": [30, 40], "window_mean_agreement": 0.6, "plan_steps": {"15": 0.4}, "plan_bar": 0.75,
         "any_step_reaches_bar": False, "what_the_plan_rule_would_say": "kept draws"}))
    (tmp_path / "redraw.json").write_text(json.dumps({"changed": 1, "of": 32, "bar": 2, "within_bar": True}))
    assert fc.main(["checks", "--dir", str(tmp_path)]) == 0
    text = (tmp_path / "checks.txt").read_text()
    assert "step 2 passes" in text and "peak 47000" in text and "steps 30 to 39" in text


# ---------------------------------------------------------------------------------------------------------------- the readers

def key_and_calls(tmp_path, rows, split_names=()):
    """A sheet key from rows of (row, set, pair, seed, kind), and two reviewers' calls agreeing except on split_names,
    where the third settles to the true kind."""
    key, c1, c2, c3 = {}, [], [], []
    for i, (rw, st, pair, seed, kind) in enumerate(rows):
        name = f"n{i:04d}"
        key[name] = {"a": pair.split("|")[0], "b": pair.split("|")[1], "row": rw, "set": st, "pair": pair,
                     "seed": str(seed), "picture_path": f"/x/{rw}/{st}/{pair}/{seed}.png"}
        c1.append((name, kind))
        c2.append((name, "hybrid" if name in split_names else kind))
        if name in split_names:
            c3.append((name, kind))
    (tmp_path / "key.json").write_text(json.dumps({"pictures": key}))
    for f, rows_ in (("c1.csv", c1), ("c2.csv", c2), ("c3.csv", c3)):
        with open(tmp_path / f, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "kind", "note"])
            w.writerows([(n, kd, "") for n, kd in rows_])
    return key


def judge_file(path, key, kind_of):
    path.write_text(json.dumps({"pictures": [{"picture": e["picture_path"], "kind": kind_of(e)} for e in key.values()]}))
    return path


def pick_rows(two_clear_500=10, two_clear_1000=12, not_two_1000=0):
    rows = []
    for rw, tc, nt in (("co3", 8, 0), ("ckpt-500", two_clear_500, 0), ("ckpt-1000", two_clear_1000, not_two_1000)):
        for i in range(32):
            kind = "two_clear" if i < tc else ("hybrid" if i < tc + nt else "borrowed_part")
            rows.append((rw, "tuning", f"p{i % 4}|q", 12001 + i // 4, kind))
    return rows


def run_pick(tmp_path, rows, runner_two_clear=None):
    key = key_and_calls(tmp_path, rows)
    trained = judge_file(tmp_path / "trained.json", key, lambda e: "two_clear" if e["row"] != "co3" else "hybrid")
    runner = judge_file(tmp_path / "runner.json", key, runner_two_clear or (lambda e: "two_clear" if e["row"] != "co3" else "hybrid"))
    return fr.main(["pick", "--key", str(tmp_path / "key.json"), "--calls", str(tmp_path / "c1.csv"),
                    str(tmp_path / "c2.csv"), "--third", str(tmp_path / "c3.csv"), "--trained", str(trained),
                    "--runner-up", str(runner), "--out", str(tmp_path / "pick.json")])


def test_pick_takes_the_higher_blind_share_and_a_tie_goes_to_500(tmp_path):
    assert run_pick(tmp_path, pick_rows(10, 12)) == 0
    assert json.loads((tmp_path / "pick.json").read_text())["picked"] == 1000
    assert run_pick(tmp_path, pick_rows(12, 12)) == 0
    p = json.loads((tmp_path / "pick.json").read_text())
    assert p["picked"] == 500 and p["gaming_alarm"] is False
    assert p["rows"]["ckpt-500"]["gap_rise_from_co3"] == 0


def test_pick_stops_when_the_trained_head_rises_alone(tmp_path):
    # the trained head calls every checkpoint picture two clear; the runner-up calls none: the gap rises by 1.0
    assert run_pick(tmp_path, pick_rows(), runner_two_clear=lambda e: "hybrid") == 3
    assert json.loads((tmp_path / "pick.json").read_text())["gaming_alarm"] is True


def test_pick_flags_three_more_not_two_pictures_for_the_owner(tmp_path):
    assert run_pick(tmp_path, pick_rows(10, 12, not_two_1000=3)) == 0
    assert json.loads((tmp_path / "pick.json").read_text())["show_owner_not_two"] == ["ckpt-1000"]


def test_unsettled_pages_stop_the_reader(tmp_path):
    key = key_and_calls(tmp_path, pick_rows(), split_names=("n0001",))
    (tmp_path / "c3.csv").write_text("name,kind,note\n")
    judge_file(tmp_path / "t.json", key, lambda e: "two_clear")
    with pytest.raises(SystemExit) as e:
        fr.main(["pick", "--key", str(tmp_path / "key.json"), "--calls", str(tmp_path / "c1.csv"),
                 str(tmp_path / "c2.csv"), "--third", str(tmp_path / "c3.csv"), "--trained", str(tmp_path / "t.json"),
                 "--runner-up", str(tmp_path / "t.json"), "--out", str(tmp_path / "pick.json")])
    assert e.value.code == 4 and json.loads((tmp_path / "unresolved.json").read_text()) == ["n0001"]


def test_rows(dial_wins=15, dial_losses=5, both=8, random_two_clear=None, borrowed_dial=5, borrowed_co3=12):
    """48 matched pictures on 16 pairs at 3 seeds: CO3 two clear on both + losses, the dial on both + wins."""
    rows = []
    keys = [(f"a{p}|b{p}", 13001 + s) for p in range(16) for s in range(3)]
    for i, (pair, seed) in enumerate(keys):
        dial_tc = i < both + dial_wins
        co3_tc = i < both or both + dial_wins <= i < both + dial_wins + dial_losses
        co3_kind = "two_clear" if co3_tc else ("borrowed_part" if i >= 48 - borrowed_co3 else "hybrid")
        dial_kind = "two_clear" if dial_tc else ("borrowed_part" if i >= 48 - borrowed_dial else "hybrid")
        rnd = random_two_clear if random_two_clear is not None else both + dial_losses
        rows += [("co3", "test", pair, seed, co3_kind), ("dial", "test", pair, seed, dial_kind),
                 ("random", "test", pair, seed, "two_clear" if i < rnd else "hybrid"),
                 ("co3", "ceiling", f"c{pair}", seed, co3_kind), ("dial", "ceiling", f"c{pair}", seed, dial_kind)]
    return rows


test_rows.__test__ = False


def run_read(tmp_path, rows, sets=None):
    key = key_and_calls(tmp_path, rows, split_names=("n0000", "n0007"))
    trained = judge_file(tmp_path / "trained.json", key, lambda e: "two_clear")
    args = ["read", "--key", str(tmp_path / "key.json"), "--calls", str(tmp_path / "c1.csv"), str(tmp_path / "c2.csv"),
            "--third", str(tmp_path / "c3.csv"), "--trained", str(trained), "--out", str(tmp_path / "read")]
    if sets:
        (tmp_path / "sets.json").write_text(json.dumps(sets))
        args += ["--sets", str(tmp_path / "sets.json")]
    assert fr.main(args) == 0
    return json.loads((tmp_path / "read/verdict.json").read_text())


def test_read_gives_the_review_files_worked_yes(tmp_path):
    # 15 towards the dial and 5 away: the lead is 10 of 48 (0.208) and the one-sided p 0.021
    v = run_read(tmp_path, test_rows())
    assert v["answer"] == "yes", v["reason"]
    assert v["mcnemar"] == {"dial_two_clear_co3_not": 15, "co3_two_clear_dial_not": 5}
    assert v["lead"] == pytest.approx(10 / 48, abs=1e-4) and v["p_one_sided"] == pytest.approx(0.0207, abs=1e-4)
    assert v["random_minus_co3"] == 0 and v["comparison_fair"] is True
    assert v["ceiling_dial_vs_co3"]["lead"] == pytest.approx(10 / 48, abs=1e-4)
    assert v["new_two_clear_came_from_co3_kind"] == {"hybrid": 15}
    assert v["kept_judge_against_blind_calls"]["more_than_1_in_10_disagree"] is True
    assert (tmp_path / "read/matched-sheet.png").exists() and (tmp_path / "read/ceiling-sheet.png").exists()


@pytest.mark.parametrize("change,answer", [
    (dict(dial_wins=2, dial_losses=0), "no"),                      # lead 2 of 48, at most 0.05
    (dict(dial_wins=7, dial_losses=0), "inconclusive"),            # lead 7 of 48, between the bars
    (dict(random_two_clear=16), "inconclusive"),                   # the random dial moved 3 of 48
    (dict(borrowed_dial=12, borrowed_co3=5), "inconclusive"),      # borrowed parts rose
])
def test_read_follows_the_rule(tmp_path, change, answer):
    assert run_read(tmp_path, test_rows(**change))["answer"] == answer


def test_read_reports_the_species_novel_pairs_and_the_sign_test(tmp_path):
    sets = {"species_overlap": [{"a": f"a{p}", "b": f"b{p}", "species_novel": p < 4} for p in range(16)]}
    v = run_read(tmp_path, test_rows(), sets)
    assert v["species_novel"]["n_pairs"] == 4
    s = v["sign_test_over_pairs"]
    assert s["pairs_up"] + s["pairs_down"] + s["ties"] == 16
    assert set(v["drop_one_pair"]) == {f"a{p}|b{p}" for p in range(16)}


def test_runner_up_joins_its_calls_to_the_logged_steps(tmp_path):
    lines = []
    for step, share in ((0, 0.25), (250, 0.5)):
        pics = [{"picture": f"/x/{step}/{i}.png"} for i in range(8)]
        lines.append(json.dumps({"step": step, "trained_two_clear_share": share, "pictures": pics}))
    (tmp_path / "logged.jsonl").write_text("\n".join(lines) + "\n")
    calls = {"pictures": [{"picture": f"/x/{s}/{i}.png", "kind": "two_clear" if i < 2 else "hybrid"}
                          for s in (0, 250) for i in range(8)]}
    (tmp_path / "calls.json").write_text(json.dumps(calls))
    assert fr.main(["runner-up", "--train", str(tmp_path), "--calls", str(tmp_path / "calls.json"),
                    "--wandb-mode", "disabled"]) == 0
    steps = json.loads((tmp_path / "judges-during-training.json").read_text())["steps"]
    assert [s["runner_up_two_clear_share"] for s in steps] == [0.25, 0.25]
    assert [s["gap_rise_from_step_0"] for s in steps] == [0, 0.25]


# ---------------------------------------------------------------------------------------------------------------- the card's subcommands on CPU stand-ins

def _stub_head(path):
    torch = pytest.importorskip("torch")
    import paper2_trained_judge as tj
    torch.manual_seed(0)
    head = tj.JudgeHead(tj.StubBackbone.dim)
    torch.save({"state": head.state_dict(), "config": head.config, "kinds": list(tj.KINDS), "bank": ["owl"],
                "backbone": "stub", "logit_scale": tj.StubBackbone.logit_scale, "epoch": 1}, path)
    return path


def test_window_read_writes_agreement_at_every_step_and_a_block_after_the_corrected_steps(tmp_path):
    head = _stub_head(tmp_path / "head.pt")
    (tmp_path / "choosing.txt").write_text("a_duck__x__a_goose\na_wolf__x__a_fox\n")
    assert fc.main(["window", "--choosing", str(tmp_path / "choosing.txt"), "--seeds", "14001", "14002", "--block", "4",
                    "--judge-head", str(head), "--out", str(tmp_path / "w"), "--dry-run"]) == 0
    w = json.loads((tmp_path / "w/window-read.json").read_text())
    assert w["n_pictures"] == 4 and len(w["agreement_by_step"]) == 10
    lo, hi = map(int, (tmp_path / "w/window.txt").read_text().split())
    assert [lo, hi] == w["window"] and hi - lo == 4 and lo >= w["corrected_steps"]
    assert all(0 <= v <= 1 for v in w["agreement_by_step"].values())
    assert w["window_mean_agreement"] == pytest.approx(sum(w["agreement_by_step"][str(i)] for i in range(lo, hi)) / 4,
                                                       abs=1e-3)
    assert isinstance(w["any_step_reaches_bar"], bool) and w["plan_bar"] == 0.75
    assert len(w["head_sha256"]) == 64


def test_judge_writes_calls_and_bodies_the_crop_judges_read(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image
    from paper2_common import load_bodies
    head = _stub_head(tmp_path / "head.pt")
    rows = []
    for i in range(3):
        p = tmp_path / f"p{i}.png"
        Image.new("RGB", (64, 64), (40 * i, 100, 200)).save(p)
        rows.append({"picture": str(p), "a": "cat", "b": "dog", "pair": "cat|dog", "seed": 12001 + i})
    with open(tmp_path / "list.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert fc.main(["judge", "--pictures", str(tmp_path / "list.csv"), "--judge-head", str(head), "--out",
                    str(tmp_path / "j"), "--dry-run"]) == 0
    calls = json.loads((tmp_path / "j/trained-calls.json").read_text())["pictures"]
    assert [c["seed"] for c in calls] == ["12001", "12002", "12003"]
    assert all(c["kind"] in fr.ALL_KINDS and 0 < c["two_clear_score"] < 1 for c in calls)
    assert load_bodies(tmp_path / "j/bodies.json")[str(tmp_path / "p0.png")] == [[0, 0, 32, 64], [32, 0, 64, 64]]


def test_rollout_check_passes_on_the_stand_in(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "tuning.txt").write_text("a_wolf__x__a_husky\n")
    assert fc.main(["rollout", "--pairs-file", str(tmp_path / "tuning.txt"), "--out", str(tmp_path / "r"),
                    "--dry-run"]) == 0
    r = json.loads((tmp_path / "r/rollout.json").read_text())
    assert r["passed"] and r["trainer_minus_co3_first_grey"] == 0
    assert r["latents_equal"] and r["latent_largest_difference"] == 0
    assert r["co3_steps_then_finish_minus_co3_first_grey"] == 0
    assert len(list((tmp_path / "r").glob("*.png"))) == 4
