#!/usr/bin/env python3
"""Rewrite the old repo name poe_repair_min to concept-composition across every tree that names it.

What changes and what stays is the table in the docstring's rules below. The data folder on
/datasets keeps its old name, so every /datasets path is left alone, along with the names of files
inside it and the retired worktree folders.

    python3 scripts/regroup/rename_repo.py              # dry run: counts per root, sample lines
    python3 scripts/regroup/rename_repo.py --apply      # write the laptop paths and prose names
    python3 scripts/regroup/rename_repo.py --apply --cluster   # also the cluster checkout path,
                                                                # only after it is renamed there
"""
import argparse
import re
import subprocess
from pathlib import Path

OLD, NEW = "poe_repair_min", "concept-composition"
HOME = Path("/home/molef")
ROOTS = {
    "repo": HOME / "PhD/concept-composition",
    "medical": HOME / "PhD/medical-imaging/compositional_diffusion",
    "learning": HOME / "goal-setting/learning",
    "obsidian": HOME / "Obsidian",
    "claude": HOME / ".claude",
}
# Files in ~/.claude that record history rather than point at a place.
SKIP_FILES = {"rename_repo.py", "pairs.json", "RENAMES.md", "settings.json", "settings.local.json"}
SKIP_DIRS = {".git", "projects", "node_modules", "cache", "backups", "skills-archive", "cadence-queue"}

# Occurrences that keep the old name: the data folder and what sits in it, and retired folders.
KEEP = re.compile(
    r"datasets/mmolefe/poe_repair_min"
    r"|data/poe_repair_min"
    r"|poe_repair_min/outputs"
    r"|poe_repair_min/tracking_set"
    r"|poe_repair_min-(?:dog-and-rain|reviews|classifier-guided|playwright|large|dog)"
)
LAPTOP = re.compile(r"(?:/home/molef/PhD|~/PhD|\\PhD)[/\\]poe_repair_min")
CLUSTER = re.compile(r"Playground/PhD/poe_repair_min")
GITHUB = re.compile(r"mo-rsa24/poe_repair_min")
BARE = re.compile(r"(?<![/A-Za-z0-9_.~\\-])poe_repair_min(?![-A-Za-z0-9_])")


def rewrite(text, cluster):
    # Shield the kept forms, rewrite the rest, then restore the shields.
    kept = []

    def shield(m):
        kept.append(m.group(0))
        return f"\0{len(kept) - 1}\0"

    text = KEEP.sub(shield, text)
    text = LAPTOP.sub(lambda m: m.group(0).replace(OLD, NEW), text)
    text = GITHUB.sub(f"mo-rsa24/{NEW}", text)
    if cluster:
        text = CLUSTER.sub(f"Playground/PhD/{NEW}", text)
    else:
        text = CLUSTER.sub(shield, text)
    text = BARE.sub(NEW, text)
    return re.sub(r"\0(\d+)\0", lambda m: kept[int(m.group(1))], text)


def files_naming(root):
    # Tracked files only: logs, staging folders and run output are gitignored and record history.
    out = subprocess.run(["git", "-C", str(root), "grep", "-lI", OLD],
                         capture_output=True, text=True).stdout.split()
    return [root / p for p in out
            if Path(p).name not in SKIP_FILES and not set(Path(p).parts) & SKIP_DIRS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--cluster", action="store_true")
    ap.add_argument("--only", help="comma-separated root names")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(ROOTS)
    for name in names:
        root = ROOTS[name]
        changed, lines = 0, 0
        samples = []
        for f in files_naming(root):
            try:
                old = f.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            new = rewrite(old, a.cluster)
            if new == old:
                continue
            changed += 1
            diff = [(o, n) for o, n in zip(old.splitlines(), new.splitlines()) if o != n]
            lines += len(diff)
            if len(samples) < 4 and diff:
                samples.append((f.relative_to(root), diff[0][1].strip()[:140]))
            if a.apply:
                f.write_text(new)
        print(f"== {name}: {changed} files, {lines} lines {'written' if a.apply else 'would change'}")
        for p, s in samples:
            print(f"   {p}: {s}")


if __name__ == "__main__":
    main()
