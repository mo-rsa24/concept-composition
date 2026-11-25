#!/usr/bin/env python3
"""The shared progress board for the two-paper route.

Three looping Claude sessions and the calendar agent read and write one file,
artifacts/notes/progress/board.json. Every write holds a file lock, so sessions cannot
overwrite each other. The progress map (map.mmd, map.png) is drawn from that file.

    python3 scripts/progress_board.py show               every step and its state
    python3 scripts/progress_board.py ready --owner s2   steps this session may start now
    python3 scripts/progress_board.py set B3 doing       change a step (todo, doing, done)
    python3 scripts/progress_board.py set B3 done --note "new 0, changed 0"
    python3 scripts/progress_board.py add R1 "R1 Label" --owner s1 --proof "..." --needs B2 --lane L1 --holds B3
                                                         add a step to a lane; --holds makes another step wait on it
    python3 scripts/progress_board.py line               the one-line status
    python3 scripts/progress_board.py render             redraw the map
    python3 scripts/progress_board.py blocks             calendar blocks and what each needs
    python3 scripts/progress_board.py block-event lane1 --event ID --start ISO --end ISO
    python3 scripts/progress_board.py owner s1 --session 3d5aedc9 --name "Loop 1"
                                                         which live session works an owner's steps
    python3 scripts/progress_board.py add C5 "C5 New step" --owner s3 --needs C4 --proof "..." --serves "Paper 1" --objective "Know the field"
    python3 scripts/progress_board.py serve C5 --serves "Paper 1" --objective "Know the field"
    python3 scripts/progress_board.py edit M4 --needs M3 M8 M10 --proof "..."
    python3 scripts/progress_board.py edit M6 --close "closed, not run: the owner's answer to question 84"
    python3 scripts/progress_board.py brief may_decide "reversible choices with a recommended default"
    python3 scripts/progress_board.py ask "Which split?" --holds "E1" --default "the rival's split" --by s3
    python3 scripts/progress_board.py answer 1 "the rival's split"
    python3 ~/.claude/skills/progress-map/scripts/progress_board.py init --repo . --outcome "what done looks like"
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "artifacts" / "notes" / "progress"
BOARD = DIR / "board.json"
LOCK = DIR / ".board.lock"
MERMAID = Path.home() / ".claude" / "skills" / "drip" / "scripts" / "mermaid_png.py"
STATES = ("todo", "doing", "done")


@contextlib.contextmanager
def locked():
    """Hold the board's lock, then write the board back atomically if it changed."""
    DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        board = json.loads(BOARD.read_text())
        before = json.dumps(board, sort_keys=True)
        yield board
        if json.dumps(board, sort_keys=True) != before:
            tmp = BOARD.with_suffix(".tmp")
            tmp.write_text(json.dumps(board, indent=1, ensure_ascii=False) + "\n")
            os.replace(tmp, BOARD)


def read():
    return json.loads(BOARD.read_text())


def now():
    return dt.datetime.now().astimezone().replace(microsecond=0)


def by_id(board):
    return {s["id"]: s for s in board["steps"]}


def needs_met(step, steps):
    return all(steps[n]["state"] == "done" for n in step.get("needs", []))


def status_line(board):
    steps = board["steps"]
    done = sum(s["state"] == "done" for s in steps)
    doing = [s["id"] for s in steps if s["state"] == "doing"]
    where = ", ".join(doing) if doing else "none"
    return (f"Map: {done} of {len(steps)} steps done, {len(doing)} in progress ({where}), "
            f"{len(board['detours'])} detours, finish line: {board['outcome']}")


def cmd_show(args):
    board = read()
    steps = by_id(board)
    for s in board["steps"]:
        if args.owner and s["owner"] != args.owner:
            continue
        gate = "ready" if s["state"] == "todo" and needs_met(s, steps) else ""
        print(f"{s['id']:3} {s['state']:5} {s['owner']:5} {gate:5} {s['label']}")
        if s["state"] == "doing":
            print(f"    done when: {s['proof']}")
    print(status_line(board))


def cmd_ready(args):
    board = read()
    steps = by_id(board)
    for s in board["steps"]:
        if s["state"] == "todo" and needs_met(s, steps) and (not args.owner or s["owner"] == args.owner):
            print(f"{s['id']}\t{s['label']}\tdone when: {s['proof']}")


def cmd_set(args):
    if args.state not in STATES:
        sys.exit(f"state must be one of {STATES}")
    with locked() as board:
        steps = by_id(board)
        if args.id not in steps:
            sys.exit(f"no step {args.id}")
        step = steps[args.id]
        if args.state in ("doing", "done") and not needs_met(step, steps):
            waiting = [n for n in step["needs"] if steps[n]["state"] != "done"]
            sys.exit(f"{args.id} is waiting on {', '.join(waiting)}")
        if args.state == "doing":
            other = [s["id"] for s in board["steps"]
                     if s["state"] == "doing" and s["owner"] == step["owner"] and s["id"] != args.id]
            if other:
                sys.exit(f"{step['owner']} already has {other[0]} in progress; finish it first")
        moment = now().isoformat()
        if args.state == "todo":
            step.pop("started", None)
            step.pop("finished", None)
        else:
            step.setdefault("started", step["updated"] if step["state"] == "doing" and step.get("updated") else moment)
        if args.state == "done":
            step["finished"] = moment
        step["state"] = args.state
        step["updated"] = moment
        if args.note:
            step["note"] = args.note
        line = status_line(board)
    if not args.no_render:
        render()
    print(line)


def cmd_add(args):
    with locked() as board:
        steps = by_id(board)
        if args.id in steps:
            sys.exit(f"step {args.id} already exists")
        missing = [n for n in args.needs + args.holds if n not in steps]
        if missing:
            sys.exit(f"no step {', '.join(missing)}")
        if args.lane:
            lane = next((l for g in board["groups"] for l in g.get("lanes", []) if l["id"] == args.lane), None)
            if lane is None:
                sys.exit(f"no lane {args.lane}")
        else:
            lane = next((g for g in board["groups"] if g["id"] == args.group and "steps" in g), None) or \
                next(g for g in reversed(board["groups"]) if "steps" in g)
        at = max([board["steps"].index(steps[i]) for i in lane["steps"]], default=len(board["steps"]) - 1) + 1
        step = {"id": args.id, "label": args.label, "owner": args.owner, "proof": args.proof, "needs": args.needs, "state": "todo"}
        if args.serves:
            step["serves"] = args.serves
        if args.objective:
            step["objective"] = args.objective
            if args.objective not in board.setdefault("objectives", []):
                board["objectives"].append(args.objective)
        board["steps"].insert(at, step)
        lane["steps"].append(args.id)
        for held in args.holds:
            steps[held].setdefault("needs", []).append(args.id)
        line = status_line(board)
    render()
    print(line)


def label(step):
    text = step["label"].replace('"', "'")
    return f'{text}<br/>YOU ARE HERE' if step["state"] == "doing" else text


def record_history(board):
    """Keep one line per change in steps done or steps planned, which the burn-up chart draws."""
    path = DIR / "history.jsonl"
    point = {"done": sum(s["state"] == "done" for s in board["steps"]), "total": len(board["steps"])}
    try:
        last = json.loads(path.read_text().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        last = {}
    if (last.get("done"), last.get("total")) != (point["done"], point["total"]):
        with open(path, "a") as handle:
            handle.write(json.dumps({"t": now().timestamp(), **point}) + "\n")


def render():
    board = read()
    try:
        record_history(board)
    except OSError as error:
        print(f"history not recorded: {error}", file=sys.stderr)
    steps = by_id(board)
    out = ["%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'monospace', 'primaryColor': '#ffffff', "
           "'primaryBorderColor': '#222222', 'primaryTextColor': '#111111', 'lineColor': '#222222'}, "
           "'flowchart': {'nodeSpacing': 18, 'rankSpacing': 28}}}%%",
           "flowchart TB",
           "  classDef done fill:#b2f2bb,stroke:#2b8a3e,color:#111111",
           "  classDef doing fill:#ffd8a8,stroke:#e8590c,stroke-width:3px,color:#111111",
           "  classDef todo fill:#e9ecef,stroke:#868e96,color:#111111",
           "  classDef detour fill:#ffffff,stroke:#868e96,stroke-dasharray:5 5,color:#111111",
           "  classDef goal fill:#ffffff,stroke:#222222,stroke-width:3px,color:#111111"]

    def nodes(ids, joiner):
        chain = f" {joiner} ".join(f'{i}["{label(steps[i])}"]:::{steps[i]["state"]}' for i in ids)
        return f"    {chain}"

    names = []
    for group in board["groups"]:
        names.append(group["id"])
        out.append(f'  subgraph {group["id"]}["{group["title"]}"]')
        out.append("    direction LR")
        if group.get("lanes"):
            for lane in group["lanes"]:
                out.append(f'    subgraph {lane["id"]}["{lane["title"]}"]')
                out.append("      direction TB")
                out.append("  " + nodes(lane["steps"], "-->"))
                out.append("    end")
        else:
            out.append(nodes(group["steps"], group.get("joiner", "-->")))
        out.append("  end")
    out.append(f'  GOAL["FINISH: {board["outcome"]}"]:::goal')
    first = board["detours_join"]
    for n, text in enumerate(board["detours"]):
        out.append(f'  X{n}["{text}"]:::detour -.-> {first}')
    for group in board["groups"]:
        if group.get("lanes"):
            out.append("  " + " ~~~ ".join(lane["id"] for lane in group["lanes"]))
    out.append("  " + " --> ".join(names + ["GOAL"]))
    for group in board["groups"]:
        out.append(f'  style {group["id"]} {group.get("style", "fill:#fafafa,stroke:#adb5bd")}')
        for lane in group.get("lanes", []):
            out.append(f'  style {lane["id"]} fill:#ffffff,stroke:#a8c7fa')
    mmd = DIR / "map.mmd"
    mmd.write_text("\n".join(out) + "\n")
    subprocess.run([sys.executable, "-I", str(MERMAID), str(mmd), str(DIR / "map.png")],
                   check=True, stdout=subprocess.DEVNULL)
    redraw_priority(board)
    redraw_sessions()


def redraw_sessions():
    """Redraw the picture of who is on what. A failure here never stops a change to the board."""
    script = Path.home() / ".claude" / "skills" / "session-status" / "scripts" / "session_status.py"
    try:
        subprocess.run([sys.executable, "-I", str(script), "--repo", str(ROOT), "--draw"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except Exception as error:
        print(f"sessions picture not redrawn: {error}", file=sys.stderr)


def redraw_priority(board):
    """Take finished steps off the priority chart. A failure here never stops a change to the board."""
    chart = DIR / "priority.mmd"
    if not chart.exists():
        return
    finished = {s["id"] for s in board["steps"] if s["state"] == "done"}
    try:
        lines = chart.read_text().splitlines()
        kept = [l for l in lines if not ((m := re.match(r"\s*([A-Z]\d+)\b[^:]*:\s*\[", l)) and m[1] in finished)]
        if kept != lines or not (DIR / "priority.png").exists():
            tmp = chart.with_suffix(".tmp")
            tmp.write_text("\n".join(kept) + "\n")
            os.replace(tmp, chart)
            subprocess.run([sys.executable, "-I", str(MERMAID), str(chart), str(DIR / "priority.png")],
                           check=True, stdout=subprocess.DEVNULL)
    except Exception as error:
        print(f"priority chart not redrawn: {error}", file=sys.stderr)


def sync_key(block):
    """Changes whenever the block's colour or description changes, so the calendar agent knows to refresh it."""
    text = f"{block.get('color', '')}|{block.get('description', '')}"
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def block_action(block, steps, moment):
    state, action = timing_action(block, steps, moment)
    if action == "none" and block.get("event_id") and block.get("synced") != sync_key(block):
        action = "refresh"
    return state, action


def timing_action(block, steps, moment):
    covered = [steps[i] for i in block["steps"]]
    event = block.get("event_id")
    if all(s["state"] == "done" for s in covered):
        return "done", ("finish" if event and not block.get("finished") else "none")
    # A block is ready only when its own prerequisites are done and none of its unfinished steps still
    # waits on another step: a paused step (one whose needs include unfinished work) is not running time.
    ready = all(steps[n]["state"] == "done" for n in block.get("needs", [])) and all(
        steps[n]["state"] == "done" for s in covered if s["state"] != "done" for n in s.get("needs", []))
    running = any(s["state"] == "doing" for s in covered)
    state = "running" if running else "ready" if ready else "waiting"
    if not event:
        return state, ("create" if ready or block.get("tentative") else "none")
    start = dt.datetime.fromisoformat(block["start"])
    end = dt.datetime.fromisoformat(block["end"])
    if block["kind"] == "loop" and ready and start > moment + dt.timedelta(minutes=10):
        return state, "pull-forward"
    if state in ("running", "ready") and end < moment:
        return state, "extend" if block["kind"] == "loop" else "reschedule"
    return state, "none"


def cmd_blocks(args):
    board = read()
    steps = by_id(board)
    moment = now()
    rows = []
    for block in board["blocks"]:
        state, action = block_action(block, steps, moment)
        rows.append({**block, "state": state, "action": action, "sync_key": sync_key(block),
                     "outcome": [f"{i}: {steps[i]['proof']}" for i in block["steps"]]})
    if args.todo:
        rows = [r for r in rows if r["action"] != "none"]
    print(json.dumps({"now": moment.isoformat(), "blocks": rows}, indent=1, ensure_ascii=False))


def cmd_block_event(args):
    with locked() as board:
        block = next((b for b in board["blocks"] if b["id"] == args.id), None)
        if block is None:
            sys.exit(f"no block {args.id}")
        if args.event:
            block["event_id"] = args.event
        if args.start:
            block["start"] = args.start
        if args.end:
            block["end"] = args.end
        if args.finished:
            block["finished"] = True
        if args.synced:
            block["synced"] = args.synced
    print(f"{args.id}: recorded")


def cmd_owner(args):
    with locked() as board:
        board.setdefault("sessions", {})[args.owner] = {"id": args.session, "name": args.name}
    print(f"{args.owner}: {args.name} ({args.session})")


def cmd_init(args):
    """Give a new project a board. Run from the template in ~/.claude; it copies itself into the project."""
    repo = Path(args.repo).resolve()
    folder = repo / "artifacts" / "notes" / "progress"
    if (folder / "board.json").exists():
        sys.exit(f"{folder / 'board.json'} already exists")
    folder.mkdir(parents=True, exist_ok=True)
    (repo / "scripts").mkdir(exist_ok=True)
    if not (repo / "scripts" / "progress_board.py").exists():
        shutil.copy2(__file__, repo / "scripts" / "progress_board.py")
    fresh = {"outcome": args.outcome, "detours": [], "detours_join": "", "steps": [], "blocks": [], "sessions": {},
             "groups": [{"id": "STEPS", "title": "STEPS", "steps": []}], "objectives": [], "brief": {}, "questions": []}
    (folder / "board.json").write_text(json.dumps(fresh, indent=1, ensure_ascii=False) + "\n")
    print(f"board created at {folder / 'board.json'}")


def cmd_serve(args):
    with locked() as board:
        step = by_id(board).get(args.id)
        if step is None:
            sys.exit(f"no step {args.id}")
        if args.serves:
            step["serves"] = args.serves
        if args.objective:
            step["objective"] = args.objective
            if args.objective not in board.setdefault("objectives", []):
                board["objectives"].append(args.objective)
    print(f"{args.id}: serves {args.serves or step.get('serves')}, objective {args.objective or step.get('objective')}")


def cmd_edit(args):
    if args.needs is None and args.proof is None and args.close is None:
        sys.exit("give --needs, --proof or --close")
    with locked() as board:
        steps = by_id(board)
        step = steps.get(args.id)
        if step is None:
            sys.exit(f"no step {args.id}")
        if args.needs is not None:
            unknown = [n for n in args.needs if n not in steps]
            if unknown:
                sys.exit(f"no step {', '.join(unknown)}")
            step["needs"] = list(args.needs)
        if args.proof is not None:
            step["proof"] = args.proof
        moment = now().isoformat()
        if args.close is not None:
            step["state"] = "done"
            step["closed"] = True
            step.setdefault("started", moment)
            step["finished"] = moment
            step["note"] = args.close
        step["updated"] = moment
        line = status_line(board)
    render()
    print(f"{args.id}: needs {', '.join(step['needs']) or 'nothing'}; {step['state']}{' (closed, not run)' if step.get('closed') else ''}")
    print(line)


def cmd_brief(args):
    with locked() as board:
        board.setdefault("brief", {})[args.key] = args.value
    print(f"brief {args.key}: {args.value}")


def cmd_ask(args):
    with locked() as board:
        questions = board.setdefault("questions", [])
        number = max([q["id"] for q in questions] + [0]) + 1
        questions.append({"id": number, "question": args.question, "holds": args.holds, "default": args.default,
                          "by": args.by, "state": "open", "asked": now().isoformat()})
    print(f"question {number} filed")


def cmd_answer(args):
    with locked() as board:
        question = next((q for q in board.get("questions", []) if q["id"] == args.id), None)
        if question is None:
            sys.exit(f"no question {args.id}")
        question.update(state="answered", answer=args.answer, answered=now().isoformat())
    print(f"question {args.id} answered")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show")
    show.add_argument("--owner")
    show.set_defaults(run=cmd_show)
    ready = sub.add_parser("ready")
    ready.add_argument("--owner")
    ready.set_defaults(run=cmd_ready)
    setter = sub.add_parser("set")
    setter.add_argument("id")
    setter.add_argument("state")
    setter.add_argument("--note")
    setter.add_argument("--no-render", action="store_true")
    setter.set_defaults(run=cmd_set)
    adder = sub.add_parser("add")
    adder.add_argument("id")
    adder.add_argument("label")
    adder.add_argument("--owner", required=True)
    adder.add_argument("--proof", required=True)
    adder.add_argument("--lane", help="a lane inside a group; leave out on a board with plain groups")
    adder.add_argument("--group")
    adder.add_argument("--serves", help="the outcome this step serves")
    adder.add_argument("--objective")
    adder.add_argument("--needs", nargs="*", default=[])
    adder.add_argument("--holds", nargs="*", default=[], help="steps that must now wait for this one")
    adder.set_defaults(run=cmd_add)
    sub.add_parser("line").set_defaults(run=lambda a: print(status_line(read())))
    sub.add_parser("render").set_defaults(run=lambda a: render())
    blocks = sub.add_parser("blocks")
    blocks.add_argument("--todo", action="store_true", help="only blocks the calendar agent must act on")
    blocks.set_defaults(run=cmd_blocks)
    event = sub.add_parser("block-event")
    event.add_argument("id")
    event.add_argument("--event")
    event.add_argument("--start")
    event.add_argument("--end")
    event.add_argument("--finished", action="store_true")
    event.add_argument("--synced", help="the sync_key of the colour and description just written")
    event.set_defaults(run=cmd_block_event)
    owner = sub.add_parser("owner")
    owner.add_argument("owner")
    owner.add_argument("--session", required=True, help="the start of the Claude session id")
    owner.add_argument("--name", required=True, help="the short name shown in the live view")
    owner.set_defaults(run=cmd_owner)
    init = sub.add_parser("init")
    init.add_argument("--repo", default=".")
    init.add_argument("--outcome", required=True)
    init.set_defaults(run=cmd_init)
    serve = sub.add_parser("serve")
    serve.add_argument("id")
    serve.add_argument("--serves")
    serve.add_argument("--objective")
    serve.set_defaults(run=cmd_serve)
    edit = sub.add_parser("edit", help="change an existing step's needs or proof, or close it without running it")
    edit.add_argument("id")
    edit.add_argument("--needs", nargs="*", help="replaces the step's needs")
    edit.add_argument("--proof")
    edit.add_argument("--close", metavar="NOTE", help="mark the step done as closed without running, with the reason")
    edit.set_defaults(run=cmd_edit)
    brief = sub.add_parser("brief")
    brief.add_argument("key")
    brief.add_argument("value")
    brief.set_defaults(run=cmd_brief)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--holds")
    ask.add_argument("--default")
    ask.add_argument("--by")
    ask.set_defaults(run=cmd_ask)
    answer = sub.add_parser("answer")
    answer.add_argument("id", type=int)
    answer.add_argument("answer")
    answer.set_defaults(run=cmd_answer)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
