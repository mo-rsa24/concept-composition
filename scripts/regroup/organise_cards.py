#!/usr/bin/env python3
"""Give each result a list of what sits on the cluster and a card section that cites it.

Reads artifacts/notes/progress/restore-manifest.tsv. For each result folder it writes
on-the-cluster.tsv (every moved file with its size, checksum and cluster path) and a marked
"On the cluster" section in the folder's README.md, one row per run. Text outside the markers is
never touched, and running it twice changes nothing.

    python3 scripts/regroup/organise_cards.py
"""
import collections
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROGRESS = "artifacts/notes/progress"
CLUSTER = "/datasets/mmolefe/poe_repair_min/artifacts-archive"
DISCRIMINATOR = "2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog"
MNIST = "2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist"
ROOTS = ("2-classifier-guidance/artifacts/results", "4-inclusive-captions/artifacts/results")
START, END = "<!-- cluster:start -->", "<!-- cluster:end -->"
TITLES = {
    "01-the-first-quick-test": "The first quick test of a small discriminator on cat and dog",
    "02-can-a-checker-read-a-clean-pair": "Can a checker read a clean cat and dog pair",
    "03-does-an-early-push-separate-them": "Does an early push separate the cat and the dog",
    "04-can-a-second-push-name-each-animal": "Can a second push name each animal",
    "05-does-it-carry-to-other-pairs": "Does the early push carry to other pairs",
    "06-how-close-it-comes-to-the-joint-prompt": "How close the kept push comes to the joint prompt",
    "pools-of-training-pictures": "The pools of training pictures",
}


def units():
    found = [DISCRIMINATOR]
    for root in ROOTS:
        for name in sorted(os.listdir(root)):
            path = f"{root}/{name}"
            if not os.path.isdir(path) or path == DISCRIMINATOR:
                continue
            if path == MNIST:
                found += [f"{path}/{s}" for s in sorted(os.listdir(path)) if os.path.isdir(f"{path}/{s}")]
            else:
                found.append(path)
    found += [f"{DISCRIMINATOR}/{g}" for g in TITLES if os.path.isdir(f"{DISCRIMINATOR}/{g}")]
    return sorted(found, key=len, reverse=True)


def question_folder(path, groups):
    """The first folder under results/ that is a result and not a group of results."""
    parts = path.split("/")
    i = parts.index("results") + 1
    folder = "/".join(parts[:i + 1])
    while folder in groups and i + 2 < len(parts):
        i += 1
        folder = "/".join(parts[:i + 1])
    return folder


def size(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def block(unit, rows):
    runs = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        relative = r["cluster_path"][len(f"{CLUSTER}/{unit}/"):]
        run = relative.split("/")[0] if "/" in relative else "(files at this level)"
        runs[run][0] += 1
        runs[run][1] += int(r["bytes"])
    lines = [START, "", "## On the cluster", ""]
    lines.append(f"The other {len(rows):,} files of this result ({size(sum(int(r['bytes']) for r in rows))}) sit at "
                 f"`{CLUSTER}/{unit}/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.")
    lines += ["", "| Run | Files | Size | Kept on the laptop | Cluster folder |", "|---|---|---|---|---|"]
    for run, (count, total) in sorted(runs.items()):
        kept = [e for e in sorted(os.listdir(unit)) if os.path.splitext(e)[0] == run and os.path.isfile(f"{unit}/{e}")]
        inside = f"{unit}/{run}"
        if os.path.isdir(inside):
            for folder, _, names in sorted(os.walk(inside)):
                kept += [os.path.relpath(os.path.join(folder, f), unit) for f in sorted(names) if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))]
        shown = ", ".join(f"[{os.path.basename(k)}]({k})" for k in kept) or "no picture"
        where = f"`{CLUSTER}/{unit}/{run}/`" if run != "(files at this level)" else f"`{CLUSTER}/{unit}/`"
        lines.append(f"| `{run}` | {count:,} | {size(total)} | {shown} | {where} |")
    lines += ["", "Bring one run back to the laptop from this folder:", "", "```bash",
              f"rsync -a mscluster84:{CLUSTER}/{unit}/<run>/ <run>/", "```", "", END]
    return "\n".join(lines)


def main():
    os.chdir(ROOT)
    rows = [r for r in csv.DictReader(open(f"{PROGRESS}/restore-manifest.tsv"), delimiter="\t") if r["cluster_path"]]
    ordered = units()
    groups = {l.strip() for l in open("scripts/regroup/result-groups.txt") if l.strip()}
    assigned = collections.defaultdict(list)
    unplaced = 0
    for r in rows:
        path = r["cluster_path"][len(CLUSTER) + 1:]
        unit = next((u for u in ordered if path.startswith(u + "/")), None)
        if unit is None and "/results/" in "/" + path:
            unit = question_folder(path, groups)
        if unit is None:
            unplaced += 1
        else:
            assigned[unit].append(r)
    written = []
    for unit, members in sorted(assigned.items()):
        os.makedirs(unit, exist_ok=True)
        with open(f"{unit}/on-the-cluster.tsv", "w") as out:
            out.write("file\tbytes\tsha256\tcluster_path\n")
            for r in sorted(members, key=lambda r: r["cluster_path"]):
                relative = r["cluster_path"][len(f"{CLUSTER}/{unit}/"):]
                out.write("\t".join([relative, r["bytes"], r["sha256"], r["cluster_path"]]) + "\n")
        card = f"{unit}/README.md"
        text = open(card).read() if os.path.exists(card) else f"# {TITLES.get(os.path.basename(unit), os.path.basename(unit).replace('-', ' ').capitalize())}\n"
        section = block(unit, members)
        if START in text:
            text = text[:text.index(START)] + section + text[text.index(END) + len(END):]
        else:
            text = text.rstrip("\n") + "\n\n" + section + "\n"
        open(card, "w").write(text)
        written.append((unit, len(members)))
    for unit, count in written:
        print(f"{count:6d}  {unit}")
    print(f"{len(written)} cards, {sum(c for _, c in written)} files cited, {unplaced} not placed")
    json.dump([u for u, _ in written], open(f"{PROGRESS}/organise-cards.json", "w"), indent=0)


if __name__ == "__main__":
    main()
