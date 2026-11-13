#!/usr/bin/env python3
"""Move files or folders and repair every link and path mention to them, in one pass.

Takes a tab-separated list of moves (old path, new path, from the repo root). For each move it
uses `git mv` for tracked paths and a rename for ignored ones. Then, in every tracked text file,
it re-resolves each relative markdown link from where the file sat before this pass, maps the
target through this pass's moves, and writes the link from where the file sits now; paths written
from the repo root are replaced the same way. Appends the moves to scripts/regroup/moves.tsv's
history file scripts/regroup/later-moves.tsv. Dry run unless --apply; --check reports links this
pass broke.

    python3 scripts/regroup/move_and_relink.py <moves.tsv> [--apply]
"""
import argparse, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXT = (".md", ".py", ".sh", ".sbatch", ".yaml", ".yml", ".tex", ".txt")
SKIP = {"RENAMES.md", "CHANGELOG.md", "DRIFT_LOG.md", "scripts/regroup/moves.tsv", "scripts/regroup/later-moves.tsv",
        "scripts/regroup/inventory.json", "plans/regroup-into-project-folders-inventory.md"}
LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)


def remap(p, pairs):
    for o, n in pairs:
        if p == o or p.startswith(o + "/"):
            return n + p[len(o):]
    return p


def links_ok(files):
    bad = []
    for f in files:
        if not f.endswith(".md") or not os.path.isfile(os.path.join(ROOT, f)):
            continue
        for m in LINK.finditer(open(os.path.join(ROOT, f), errors="ignore").read()):
            t = m.group(2).split("#")[0]
            if not t or t.startswith(("http:", "https:", "mailto:", "/")):
                continue
            if not os.path.lexists(os.path.join(ROOT, os.path.dirname(f), t)):
                bad.append((f, m.group(2)))
    return bad


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("moves"); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    pairs = [tuple(l.rstrip("\n").split("\t")) for l in open(a.moves) if l.strip()]
    pairs = sorted(pairs, key=lambda x: -len(x[0]))
    missing = [o for o, _ in pairs if not os.path.lexists(os.path.join(ROOT, o))]
    clash = [n for _, n in pairs if os.path.lexists(os.path.join(ROOT, n))]
    print(f"{len(pairs)} moves; {len(missing)} sources missing {missing[:3]}; {len(clash)} destinations taken {clash[:3]}")
    if not a.apply:
        return
    if missing or clash:
        sys.exit("refusing: fix the list first")
    files_before = git("ls-files").stdout.split()
    broken_before = {(remap(f, pairs), t) for f, t in links_ok(files_before)}
    for o, n in pairs:
        os.makedirs(os.path.dirname(os.path.join(ROOT, n)) or ROOT, exist_ok=True)
        if git("ls-files", "--", o).stdout.strip():
            git("mv", o, n)
            rest = os.path.join(ROOT, o)
            if os.path.isdir(rest):  # ignored files left behind inside a tracked folder
                for d, _, fs in os.walk(rest):
                    for x in fs:
                        s = os.path.join(d, x); t = os.path.join(ROOT, n, os.path.relpath(s, rest))
                        os.makedirs(os.path.dirname(t), exist_ok=True); os.rename(s, t)
                subprocess.run(["find", rest, "-depth", "-type", "d", "-empty", "-delete"])
        else:
            os.rename(os.path.join(ROOT, o), os.path.join(ROOT, n))
    back = [(n, o) for o, n in pairs]
    alt = "|".join(re.escape(o) for o, _ in pairs)
    plain = re.compile(r"(?:(?<=^)|(?<=[\s\"'`(=:,\[])|(?<=poe_repair_min/))(" + alt + r")(?=[/\s\"'`),:#.\]]|$)", re.M)
    table = dict(pairs); edits = 0
    for f in git("ls-files").stdout.split():
        if f in SKIP or not f.endswith(TEXT) or not os.path.isfile(os.path.join(ROOT, f)):
            continue
        s = open(os.path.join(ROOT, f), errors="ignore").read(); t = s
        if f.endswith(".md"):
            old_dir, new_dir = os.path.dirname(remap(f, back)), os.path.dirname(f)
            def fix(m):
                tgt = m.group(2)
                if tgt.startswith(("http:", "https:", "mailto:", "#", "/")):
                    return m.group(0)
                p, _, anc = tgt.partition("#")
                before = os.path.normpath(os.path.join(old_dir, p))
                if not os.path.lexists(os.path.join(ROOT, remap(before, pairs))):
                    return m.group(0)
                r = os.path.relpath(remap(before, pairs), new_dir or ".") + ("/" if p.endswith("/") else "")
                return m.group(1) + r + ("#" + anc if anc else "") + m.group(3)
            t = LINK.sub(fix, t)
        t = plain.sub(lambda m: table[m.group(1)], t)
        if t != s:
            open(os.path.join(ROOT, f), "w").write(t); edits += 1
    with open(os.path.join(ROOT, "scripts/regroup/later-moves.tsv"), "a") as h:
        for o, n in pairs:
            h.write(f"{o}\t{n}\n")
    after = links_ok(git("ls-files").stdout.split())
    new_broken = [x for x in after if x not in broken_before]
    print(f"moved {len(pairs)}; {edits} files repointed; links broken by this pass: {len(new_broken)}")
    for x in new_broken[:10]:
        print("  ", x)


if __name__ == "__main__":
    main()
