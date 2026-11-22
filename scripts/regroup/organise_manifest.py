#!/usr/bin/env python3
"""Build the file list for organising result folders: one picture per run folder stays on the laptop.

Moves nothing. For each file under a project's artifacts/results/ it writes one row saying whether the
file stays or moves, where it goes, and the checksum the cluster copy must match. The rule is in
artifacts/notes/progress/organise-plan.md.

    python3 scripts/regroup/organise_manifest.py --picks artifacts/notes/progress/organise-picks.tsv \
        --shipped <ship-manifest.tsv> [<ship-manifest.tsv> ...]

Rows naming a guarded subject are written outside the repository, to the archive's own list.
"""
import argparse
import collections
import json
import os
import re
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "shared"))
from disallowed_subjects import is_allowed  # noqa: E402

PROJECTS = ("2-classifier-guidance", "4-inclusive-captions")
DISCRIMINATOR = "2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog"
WEB_IMAGES = "4-inclusive-captions/artifacts/results/does-any-caption-say-a-cat-and-a-dog/images/"
ARCHIVE = "/home/molef/PhD/_archive/concept-composition-offload-2026-11-22"
CLUSTER = "/datasets/mmolefe/poe_repair_min/artifacts-archive"
OUT = "artifacts/notes/progress/organise-manifest.tsv"
GROUPS = "artifacts/notes/progress/organise-groups.json"
IMG = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")
SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", "outputs", "temp", "_archive"}
TOKEN = re.compile(r"[\w.\-]+/[\w.\-]+\.(?:png|jpe?g|webp|gif|svg|json|jsonl|csv|tsv|npz|log|md)", re.I)
LINK = re.compile(r'(?:\]\(|src="|href=")([^)"#?]+)')


def kind(path):
    """Which sort of document names a file."""
    if path.startswith("writing/"):
        return "paper"
    if re.match(r"(\d-[^/]+/)?(report|context|environment|runbook)/", path):
        return "page"
    if re.match(r"(\d-[^/]+/)?plans/", path):
        return "plan"
    if path.endswith((".py", ".sh", ".sbatch", ".yaml", ".yml")):
        return "code"
    return "card"


def names():
    """Every file path a page, plan, paper, card or script names, with the kinds that name it."""
    exact, tails = collections.defaultdict(set), collections.defaultdict(set)
    for folder, subdirs, files in os.walk("."):
        subdirs[:] = [d for d in subdirs if d not in SKIP]
        for name in files:
            if not name.endswith((".md", ".html", ".tex", ".py", ".sh", ".sbatch", ".yaml", ".yml")):
                continue
            path = os.path.join(folder, name)[2:]
            if path.startswith("artifacts/notes/progress/"):
                continue
            try:
                if os.path.getsize(path) > 4_000_000:
                    continue
                text = open(path, errors="ignore").read()
            except OSError:
                continue
            which = kind(path)
            for token in TOKEN.findall(text):
                tails[token.lower()].add(which)
            if name.endswith((".md", ".html")):
                for link in LINK.findall(text):
                    link = urllib.parse.unquote(link.strip())
                    if not link.startswith(("http", "mailto")):
                        exact[os.path.normpath(os.path.join(os.path.dirname(path), link))].add(which)
    return exact, tails


def regrouped(path, groups):
    """Where a kept discriminator file sits once its run folder is under its report page's group."""
    if not path.startswith(DISCRIMINATOR + "/"):
        return path
    parts = path[len(DISCRIMINATOR) + 1:].split("/")
    if len(parts) > 1 and parts[0] in groups:
        return "/".join([DISCRIMINATOR, groups[parts[0]]] + parts)
    return path


def read_picks(path, groups):
    """result, folder, picked path, why. Returns {picked path: new path on the laptop}."""
    picks = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or parts[0] == "result" or parts[2].strip() in ("", "none"):
            continue
        source = parts[2].strip()
        picks[source] = source
        if source.startswith(DISCRIMINATOR + "/"):
            run = source[len(DISCRIMINATOR) + 1:].split("/")[0]
            if run in groups:
                picks[source] = f"{DISCRIMINATOR}/{groups[run]}/{run}{os.path.splitext(source)[1]}"
    return picks


def read_shipped(paths):
    shipped = {}
    for path in paths:
        for line in open(path):
            old, cluster, size, digest = line.rstrip("\n").split("\t")
            shipped[old] = (cluster, int(size), digest)
    return shipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--picks", required=True)
    parser.add_argument("--shipped", nargs="+", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    groups = json.load(open(GROUPS))
    picks, shipped = read_picks(args.picks, groups), read_shipped(args.shipped)
    exact, tails = names()
    rows, guarded, tally, problems = [], [], collections.Counter(), []
    sizes = collections.Counter()
    for project in PROJECTS:
        for folder, _, files in os.walk(f"{project}/artifacts/results"):
            for name in files:
                path = os.path.join(folder, name)
                size = os.path.getsize(path)
                named = exact.get(path, set()) | tails.get("/".join(path.split("/")[-2:]).lower(), set())
                if not is_allowed(path):
                    action, new = "archive only, guarded subject", f"{ARCHIVE}/{path}"
                elif path.startswith(WEB_IMAGES):
                    action, new = "stays until the owner decides on web images", path
                elif path in picks:
                    action, new = "keep, picked picture", picks[path]
                elif name in ("README.md", "on-the-cluster.tsv", "file-origins.tsv"):
                    action, new = "keep, card", regrouped(path, groups)
                elif "paper" in named:
                    action, new = "keep, the paper names it", regrouped(path, groups)
                elif not name.lower().endswith(IMG) and named & {"page", "plan", "code"}:
                    action, new = "keep, data a page, plan or script names", regrouped(path, groups)
                else:
                    action, new = "move to the cluster and laptop archive", f"{ARCHIVE}/{path}"
                cluster, digest = "", ""
                if action.startswith("move"):
                    if path not in shipped:
                        problems.append(f"no cluster copy: {path}")
                    else:
                        cluster, shipped_size, digest = shipped[path]
                        if shipped_size != size:
                            problems.append(f"size changed since the copy: {path}")
                row = (path, action, new, str(size), cluster, digest)
                (guarded if action.startswith("archive only") else rows).append(row)
                tally[(project, action)] += 1
                sizes[(project, action)] += size
    for source in picks:
        if not os.path.exists(source):
            problems.append(f"picked picture is missing: {source}")
    with open(OUT, "w") as out:
        out.write("old_path\taction\tnew_path\tbytes\tcluster_path\tsha256\n")
        for row in sorted(rows):
            out.write("\t".join(row) + "\n")
    os.makedirs(ARCHIVE, exist_ok=True)
    with open(f"{ARCHIVE}/guarded-files.tsv", "w") as out:
        out.write("old_path\taction\tnew_path\tbytes\tcluster_path\tsha256\n")
        for row in sorted(guarded):
            out.write("\t".join(row) + "\n")
    for (project, action), count in sorted(tally.items()):
        print(f"{project[:12]:13}{count:7d} {sizes[(project, action)] / 1e6:9.1f} MB  {action}")
    print(f"rows {len(rows)} in the repo list, {len(guarded)} in the archive list, problems {len(problems)}")
    for problem in problems[:12]:
        print(" ", problem)


if __name__ == "__main__":
    main()
