#!/usr/bin/env python3
"""Step 3 of plans/regroup-into-project-folders.md: move every inventory item to its project.

Reads scripts/regroup/inventory.json, built from plans/regroup-into-project-folders-inventory.md
(one row per item: kind, name, project letter, why it was placed there, the checkouts it was found
in, and its new name or ""). Tracked paths move with `git mv`, so history follows them; ignored
paths (most of artifacts/results/) move with a plain rename. Prints a dry run unless --apply is
given, and on --apply writes scripts/regroup/moves.tsv (old path, new path), which step 4 reads to
repair links and imports.

    python3 scripts/regroup/move.py            # dry run
    python3 scripts/regroup/move.py --apply
"""
import argparse
import collections
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT = {"L": "1-lora-correction", "C": "2-classifier-guidance", "K": "3-understanding-compositionality",
           "I": "4-inclusive-captions", "T": "5-text-embedding-third-expert", "A": "_archive"}
SOURCE = {"scopes": "plans", "report": "report", "drips": "artifacts/drips", "ideas": "artifacts/ideas",
          "results": "artifacts/results", "notes": "artifacts/notes", "scenes": "artifacts/scenes",
          "scripts": "scripts", "experiments": "poe_repair/experiments"}
DEST = {"scopes": "plans", "report": "report", "drips": "artifacts/drips", "ideas": "artifacts/ideas",
        "results": "artifacts/results", "notes": "artifacts/notes", "scenes": "artifacts/scenes",
        "scripts": "scripts"}
STAYS = {"R", "M", "?"}       # shared root items stay; medical items leave in step 5
STAYS_KIND = {"experiments"}  # imported by name from the package; moved in a later step


def tracked(path):
    out = subprocess.run(["git", "-C", ROOT, "ls-files", "--", path], capture_output=True, text=True).stdout
    return bool(out.strip())


def destination(kind, name, proj, new):
    if new.startswith("moves to "):
        return new.split("moves to ", 1)[1].split(" ", 1)[0]
    if proj == "A":
        return os.path.join("_archive", SOURCE[kind], name)
    target = new if new and not new.startswith("?") else name
    return os.path.join(PROJECT[proj], DEST[kind], target)


def move_one(src, dst):
    os.makedirs(os.path.dirname(os.path.join(ROOT, dst)), exist_ok=True)
    if not tracked(src):
        os.rename(os.path.join(ROOT, src), os.path.join(ROOT, dst))
        return
    subprocess.run(["git", "-C", ROOT, "mv", src, dst], check=True)
    rest = os.path.join(ROOT, src)  # ignored files left behind inside a tracked folder
    if os.path.isdir(rest):
        for dirpath, _, files in os.walk(rest):
            for f in files:
                s = os.path.join(dirpath, f)
                d = os.path.join(ROOT, dst, os.path.relpath(s, rest))
                os.makedirs(os.path.dirname(d), exist_ok=True)
                os.rename(s, d)
        subprocess.run(["find", rest, "-depth", "-type", "d", "-empty", "-delete"], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    rows = json.load(open(os.path.join(ROOT, "scripts/regroup/inventory.json")))
    moves, missing, stays = [], [], collections.Counter()
    for kind, name, proj, _why, _where, new in rows:
        if proj in STAYS or kind in STAYS_KIND or (kind == "scripts" and name == "regroup"):
            stays["experiments stay in poe_repair" if kind in STAYS_KIND else proj] += 1
            continue
        src = os.path.join(SOURCE[kind], name)
        if not os.path.lexists(os.path.join(ROOT, src)):
            missing.append((proj, src))
            continue
        moves.append((proj, src, destination(kind, name, proj, new)))
    clashes = [d for _, _, d in moves if os.path.lexists(os.path.join(ROOT, d))]
    dups = [d for d, c in collections.Counter(d for _, _, d in moves).items() if c > 1]
    print("moves per project:", {PROJECT[p]: n for p, n in sorted(collections.Counter(p for p, _, _ in moves).items())})
    print("left in place:", dict(stays))
    print(f"not found ({len(missing)}):", *[f"\n   {PROJECT.get(p, p)}  {s}" for p, s in missing])
    print("destination already exists:", len(clashes), clashes[:5])
    print("two items to one destination:", dups[:5])
    if not args.apply:
        print("dry run only; rerun with --apply")
        return
    if clashes or dups:
        sys.exit("refusing to move: fix the clashes first")
    with open(os.path.join(ROOT, "scripts/regroup/moves.tsv"), "w") as log:
        for _proj, src, dst in moves:
            move_one(src, dst)
            log.write(f"{src}\t{dst}\n")
    print("moved", len(moves), "items; log in scripts/regroup/moves.tsv")


if __name__ == "__main__":
    main()
