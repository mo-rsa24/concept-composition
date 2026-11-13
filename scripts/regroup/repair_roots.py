#!/usr/bin/env python3
"""Step 4.1 of plans/regroup-into-project-folders.md: fix how moved scripts find the repo root.

Every file moved in step 3 now sits exactly one folder deeper than before. A lookup that climbed
from the file to the repo root (or above it) must climb one level more; a lookup that stopped
inside the old folder keeps its depth. For each moved .py, .sh and .sbatch file, the old path comes
from scripts/regroup/moves.tsv, and these lookups are rewritten when they reached the root:
`Path(__file__)...parents[k]`, `Path(__file__).parent.parent...`, nested `os.path.dirname(...
__file__)`, and bash `$(dirname "${BASH_SOURCE[0]}")/..`. Dry run unless --apply.
"""
import argparse, os, re, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_moves():
    moves = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "scripts/regroup/moves.tsv"))]
    return {new: old for old, new in moves}


def old_path(new, back):
    for n, o in back.items():
        if new == n or new.startswith(n + "/"):
            return o + new[len(n):]
    return None


def fix(text, depth):
    """depth: number of folders between the repo root and the file at its old place (scripts/x.py -> 1)."""
    changes = []

    def bump(levels_up):  # levels_up: how many directories above the file's own folder the lookup lands
        return levels_up >= depth

    def parents(m):
        k = int(m.group(1))
        if bump(k):
            changes.append(m.group(0)); return f"parents[{k + 1}]"
        return m.group(0)

    def chain(m):
        n = m.group(0).count(".parent")
        if bump(n - 1):
            changes.append(m.group(0)); return m.group(0) + ".parent"
        return m.group(0)

    def dirn(m):
        n = m.group(0).count("dirname(")
        if bump(n - 1):
            changes.append(m.group(0)); return "os.path.dirname(" + m.group(0) + ")"
        return m.group(0)

    def bash(m):
        n = m.group(2).count("/..")
        if bump(n):
            changes.append(m.group(0)); return m.group(1) + m.group(2) + "/.."
        return m.group(0)

    out = []
    for line in text.splitlines(keepends=True):
        if "__file__" in line:
            line = re.sub(r"parents\[(\d)\]", parents, line)
            line = re.sub(r"(?<![\w.])(?:\.parent)+(?=[^\w.]|$)", chain, line) if "Path(" in line else line
            line = re.sub(r"(?:os\.path\.dirname\()+(?:os\.path\.(?:abspath|realpath)\()?__file__\)+", dirn, line)
        if "BASH_SOURCE" in line or 'dirname "$0"' in line:
            line = re.sub(r'(\$\(dirname "(?:\$\{BASH_SOURCE\[0\]\}|\$0)"\))((?:/\.\.)*)', bash, line)
        out.append(line)
    return "".join(out), changes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); args = ap.parse_args()
    back = load_moves(); total = 0; files = 0
    for new in back:
        base = os.path.join(ROOT, new)
        paths = [base] if os.path.isfile(base) else [os.path.join(d, f) for d, _, fs in os.walk(base) for f in fs]
        for p in paths:
            if not p.endswith((".py", ".sh", ".sbatch")):
                continue
            rel = os.path.relpath(p, ROOT); old = old_path(rel, back)
            depth = old.count("/")
            if rel.count("/") == depth:  # back at its original depth: restore the original, no bump
                fixed = subprocess.run(["git", "-C", ROOT, "show", f"pre-regroup-move:{old}"], capture_output=True, text=True).stdout
                ch = ["restored"] if fixed and fixed != open(p, errors="ignore").read() else []
            else:
                text = subprocess.run(["git", "-C", ROOT, "show", f"pre-regroup-move:{old}"], capture_output=True, text=True).stdout or open(p, errors="ignore").read()
                fixed, ch = fix(text, depth)
                if not ch and text != open(p, errors="ignore").read():
                    ch = ["restored"]
            if ch:
                files += 1; total += len(ch)
                if args.apply:
                    open(p, "w").write(fixed)
                elif files <= 6:
                    print(f"{rel} (was {old}): {ch[:3]}")
    print(f"{'rewrote' if args.apply else 'would rewrite'} {total} lookups in {files} files")


if __name__ == "__main__":
    main()
