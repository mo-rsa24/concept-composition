#!/usr/bin/env python3
"""Step 4.2 of plans/regroup-into-project-folders.md: repoint links and path mentions after the move.

Reads scripts/regroup/moves.tsv. In every tracked text file:
- a relative markdown link is re-resolved: target from the file's old place, the target's new
  place, then a new relative path from the file's new place (anchors kept);
- a path written from the repo root (`artifacts/results/x`, `scripts/x.py`, also after
  `concept-composition/` in an absolute path) is replaced by its new path.
Files that record old paths on purpose are skipped. Dry run unless --apply; --check counts broken
relative markdown links now and at the tag pre-regroup-move, and lists the ones the move broke.
"""
from urllib.parse import unquote
import argparse, os, re, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP = {"RENAMES.md", "CHANGELOG.md", "DRIFT_LOG.md", "scripts/regroup/moves.tsv", "scripts/regroup/inventory.json",
        "plans/regroup-into-project-folders-inventory.md"}
TEXT = (".md", ".py", ".sh", ".sbatch", ".yaml", ".yml", ".tex", ".txt", ".cfg", ".toml")
LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def moves():
    m = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "scripts/regroup/moves.tsv"))]
    return sorted(((o, n) for o, n in m if o != n), key=lambda x: -len(x[0]))


def remap(path, pairs):
    for o, n in pairs:
        if path == o or path.startswith(o + "/"):
            return n + path[len(o):]
    return path


def files_at(ref):
    out = subprocess.run(["git", "-C", ROOT, "ls-tree", "-r", "--name-only", ref] if ref else
                         ["git", "-C", ROOT, "ls-files"], capture_output=True, text=True).stdout.splitlines()
    return out


def fix_text(rel, text, fwd, back_old):
    old_dir = os.path.dirname(back_old)
    new_dir = os.path.dirname(rel)
    n = 0
    if rel.endswith(".md"):
        def link(m):
            nonlocal n
            tgt = m.group(2)
            if tgt.startswith(("http:", "https:", "mailto:", "#", "/")):
                return m.group(0)
            path, _, anchor = tgt.partition("#")
            old_target = os.path.normpath(os.path.join(old_dir, path))
            new_target = remap(old_target, fwd)
            new_rel = os.path.relpath(new_target, new_dir or ".")
            if path.endswith("/") and not new_rel.endswith("/"):
                new_rel += "/"
            new = new_rel + ("#" + anchor if anchor else "")
            if new != tgt and os.path.lexists(os.path.join(ROOT, new_target)):
                n += 1
                return m.group(1) + new + m.group(3)
            return m.group(0)
        text = LINK.sub(link, text)
    pat, table = combined(fwd)
    def sub(m):
        nonlocal n
        n += 1
        return table[m.group(1)]
    text = pat.sub(sub, text)
    return text, n


_COMBINED = {}


def combined(fwd):
    key = id(fwd)
    if key not in _COMBINED:
        alt = "|".join(re.escape(o) for o, _ in fwd)  # longest first, so a folder never shadows a file in it
        pat = re.compile(r"(?:(?<=^)|(?<=[\s\"'`(=:,\[])|(?<=concept-composition/))(" + alt + r")(?=[/\s\"'`),:#.\]]|$)", re.M)
        _COMBINED[key] = (pat, dict(fwd))
    return _COMBINED[key]


def broken(ref):
    names = set(files_at(ref)); count = []
    for f in names:
        if not f.endswith(".md"):
            continue
        try:
            t = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:{f}"], capture_output=True, text=True).stdout if ref else open(os.path.join(ROOT, f), errors="ignore").read()
        except Exception:
            continue
        for m in LINK.finditer(t):
            tgt = m.group(2).split("#")[0]
            if not tgt or tgt.startswith(("http:", "https:", "mailto:", "/")):
                continue
            p = os.path.normpath(os.path.join(os.path.dirname(f), unquote(tgt)))
            if ref:
                ok = p in names or any(x.startswith(p + "/") for x in names) or os.path.lexists(os.path.join(ROOT, p))
            else:
                ok = os.path.lexists(os.path.join(ROOT, p))
            if not ok:
                count.append((f, tgt))
    return count


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        before = broken("pre-regroup-move"); after = broken(None)
        fwd = moves()
        before_keys = {(remap(f, fwd), os.path.normpath(os.path.join(os.path.dirname(remap(f, fwd)), t))) for f, t in before}
        new_broken = [(f, t) for f, t in after if (f, os.path.normpath(os.path.join(os.path.dirname(f), t))) not in before_keys]
        print(f"broken relative links: {len(before)} before the move, {len(after)} now, {len(new_broken)} broken by the move")
        for f, t in new_broken[:15]:
            print("  ", f, "->", t)
        return
    fwd = moves(); back = [(n, o) for o, n in fwd]
    total = files = 0
    for rel in files_at(None):
        if rel in SKIP or not rel.endswith(TEXT) or rel.startswith("_archive/"):
            continue
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        text = open(p, errors="ignore").read()
        fixed, n = fix_text(rel, text, fwd, remap(rel, back))
        if n:
            total += n; files += 1
            if a.apply:
                open(p, "w").write(fixed)
    print(f"{'rewrote' if a.apply else 'would rewrite'} {total} links and paths in {files} files")


if __name__ == "__main__":
    main()
