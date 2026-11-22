#!/usr/bin/env python3
"""List artifact files that nothing in the repository names, with size, reason and proposed move.

Moves nothing. A file is listed only when its name appears in no page, card, plan, paper, script or
data file. The search cannot see a filename a script or scene builds at run time, so files a built
scene, deck or walk may load carry a `check first` risk. Paths naming a guarded subject are counted
here and written only to the archive's own list.

    python3 scripts/regroup/prune_list.py
"""
import collections
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "shared"))
from disallowed_subjects import is_allowed  # noqa: E402

PROGRESS = "artifacts/notes/progress"
ARCHIVE = "/home/molef/PhD/_archive/concept-composition-offload-2026-11-22"
PROJECTS = ("1-lora-correction", "2-classifier-guidance", "3-understanding-compositionality",
            "4-inclusive-captions", "5-text-embedding-third-expert")
FIXED = ("artifacts/_shared/", "artifacts/_quarantine/", "artifacts/caches/", "artifacts/rung2-survive-noise/",
         "artifacts/rung3-group-wise/", "artifacts/rung4-scale/", f"{PROGRESS}/")
TEXT = (".md", ".html", ".tex", ".py", ".sh", ".sbatch", ".yaml", ".yml", ".json", ".jsonl", ".csv", ".tsv",
        ".txt", ".mmd", ".ipynb", ".js", ".css")
SKIP = {".git", ".claude", "__pycache__", "node_modules", ".venv", "venv", "outputs", "temp", "_archive"}
TOKEN = re.compile(r"[\w.\-]+\.[A-Za-z0-9]{1,6}")
KEEP_NAMES = {"README.md", "on-the-cluster.tsv", "file-origins.tsv", ".gitkeep", ".gitignore"}


def named_files():
    names = set()
    for folder, subdirs, files in os.walk("."):
        subdirs[:] = [d for d in subdirs if d not in SKIP]
        if folder.startswith(f"./{PROGRESS}"):
            subdirs[:] = []
            continue
        for name in files:
            if not name.endswith(TEXT) or name == "on-the-cluster.tsv":
                continue
            path = os.path.join(folder, name)
            try:
                if os.path.getsize(path) > 20_000_000:
                    continue
                names.update(t.lower() for t in TOKEN.findall(open(path, errors="ignore").read()))
            except OSError:
                pass
    return names


def risk(path):
    return "check first: a built page may load it" if re.search(r"/(scenes|decks|drips)/", "/" + path) else "low"


def size(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def main():
    os.chdir(ROOT)
    names = named_files()
    rows, hidden, totals, listed, bytes_listed = [], [], collections.Counter(), collections.Counter(), collections.Counter()
    groups, risks = collections.Counter(), collections.Counter()
    for label, base in [(p, f"{p}/artifacts") for p in PROJECTS] + [("repository root", "artifacts")]:
        for folder, subdirs, files in os.walk(base):
            if (folder + "/").startswith(FIXED):
                subdirs[:] = []
                continue
            subdirs[:] = [d for d in subdirs if d not in SKIP and not f"{folder}/{d}/".startswith(FIXED)]
            for name in files:
                path = os.path.join(folder, name)
                if os.path.islink(path):
                    continue
                totals[label] += 1
                if name in KEEP_NAMES or name.lower() in names:
                    continue
                n = os.path.getsize(path)
                row = (path, str(n), "no page, card, plan, paper, script or data file names it", risk(path), f"{ARCHIVE}/{path}")
                (rows if is_allowed(path) else hidden).append(row)
                listed[label] += 1
                bytes_listed[label] += n
                parts = path.split("/")
                groups[(label, "/".join(parts[parts.index("artifacts") + 1:parts.index("artifacts") + 3]))] += n
                risks[row[3]] += n
    with open(f"{PROGRESS}/prune-list.tsv", "w") as out:
        out.write("path\tbytes\treason\trisk\tproposed_move\n")
        for row in sorted(rows):
            out.write("\t".join(row) + "\n")
    with open(f"{ARCHIVE}/prune-guarded.tsv", "w") as out:
        out.write("path\tbytes\treason\trisk\tproposed_move\n")
        for row in sorted(hidden):
            out.write("\t".join(row) + "\n")
    lines = ["# 🧭 Progress: prune list", "",
             "Artifact files that nothing in the repository names, so they can leave the repository once the owner "
             "approves. Every file listed here is still in the repository. The [full list](prune-list.tsv) gives each file's path, size, reason, risk "
             "and proposed move to the laptop archive.", "",
             '<a href="prune-decision.png"><img src="prune-decision.png" width="520" alt="The 6,594 files split into 3,764 low-risk files and 2,830 check-first files, with what each answer moves"></a>', "",
             "The search reads every page, card, plan, paper, script and data file for the file's name. It cannot see a "
             "name a script or scene builds at run time, so files under `scenes/`, `decks/` and `drips/` are marked "
             "`check first`.", "", "## By project", "",
             "| Project | Artifact files | Named by nothing | Size |", "|---|---|---|---|"]
    for label in totals:
        lines.append(f"| {label} | {totals[label]:,} | {listed[label]:,} | {size(bytes_listed[label])} |")
    lines.append(f"| **All** | {sum(totals.values()):,} | {sum(listed.values()):,} | {size(sum(bytes_listed.values()))} |")
    lines += ["", "## Largest folders", "", "| Project | Folder | Size |", "|---|---|---|"]
    for (label, group), n in groups.most_common(12):
        lines.append(f"| {label} | `{group}` | {size(n)} |")
    lines += ["", "## By risk", "", "| Risk | Size |", "|---|---|"]
    for name, n in risks.most_common():
        lines.append(f"| {name} | {size(n)} |")
    if hidden:
        lines += ["", f"{len(hidden):,} of these files name a guarded subject. Their paths are kept in the archive's own list."]
    open(f"{PROGRESS}/prune-list.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[8:]))


if __name__ == "__main__":
    main()
