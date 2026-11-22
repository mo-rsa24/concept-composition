#!/usr/bin/env python3
"""Carry out organise-manifest.tsv: keep the picked files, move the rest to the laptop archive, fix links.

Nothing is deleted. Every moved file is checked against the checksum of its cluster copy first, and
every move is written to the restore manifest as it happens. A page's link to a moved file becomes a
plain citation of its cluster path. Links in plans go to a list for Session 3, which owns plans/.

    python3 scripts/regroup/organise_apply.py --dry-run
    python3 scripts/regroup/organise_apply.py
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROGRESS = "artifacts/notes/progress"
MANIFEST = f"{PROGRESS}/organise-manifest.tsv"
RESTORE = f"{PROGRESS}/restore-manifest.tsv"
PLAN_LINKS = f"{PROGRESS}/organise-plan-links.tsv"
GROUPS = f"{PROGRESS}/organise-groups.json"
ARCHIVE = "/home/molef/PhD/_archive/concept-composition-offload-2026-11-22"
CLUSTER = "/datasets/mmolefe/poe_repair_min/artifacts-archive"
DISCRIMINATOR = "2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog"
RESULT_ROOTS = ("2-classifier-guidance/artifacts/results", "4-inclusive-captions/artifacts/results")
SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", "outputs", "temp", "_archive"}
WRAPPED = re.compile(r'<a\s+href="([^"]+)"[^>]*>\s*<img\s[^>]*>\s*</a>')
IMAGE = re.compile(r'<img\s+[^>]*src="([^"]+)"[^>]*>')
MD_IMAGE = re.compile(r'!\[[^\]]*\]\(([^)\s]+)[^)]*\)')
MD_LINK = re.compile(r'(?<!!)\[([^\]]*)\]\(([^)\s]+)[^)]*\)')
ANCHOR = re.compile(r'<a\s+href="([^"]+)"[^>]*>([^<]*)</a>')


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load():
    rows = list(csv.DictReader(open(MANIFEST), delimiter="\t"))
    guarded = list(csv.DictReader(open(f"{ARCHIVE}/guarded-files.tsv"), delimiter="\t"))
    return rows, guarded


def preflight(rows, guarded):
    problems = []
    moving = [r for r in rows if r["action"].startswith("move")]
    for r in rows + guarded:
        if not os.path.exists(r["old_path"]):
            problems.append(f"missing: {r['old_path']}")
        elif r["new_path"] != r["old_path"] and os.path.exists(r["new_path"]):
            problems.append(f"target exists: {r['new_path']}")
    check = [r for r in moving if os.path.exists(r["old_path"])]
    with ThreadPoolExecutor(8) as pool:
        for r, value in zip(check, pool.map(lambda r: digest(r["old_path"]), check)):
            if value != r["sha256"]:
                problems.append(f"differs from its cluster copy: {r['old_path']}")
    return problems


def move(rows, guarded):
    new_file = not os.path.exists(RESTORE)
    with open(RESTORE, "a") as out, open(f"{ARCHIVE}/guarded-restore.tsv", "a") as hidden:
        if new_file:
            out.write("old_path\tnew_path\tbytes\tsha256\tcluster_path\n")
        for r in rows + guarded:
            if r["new_path"] == r["old_path"]:
                continue
            value = r["sha256"] or digest(r["old_path"])
            os.makedirs(os.path.dirname(r["new_path"]), exist_ok=True)
            os.rename(r["old_path"], r["new_path"])
            target = hidden if r["action"].startswith("archive only") else out
            target.write("\t".join([r["old_path"], r["new_path"], r["bytes"], value, r["cluster_path"]]) + "\n")
            target.flush()
    removed = 0
    for root in RESULT_ROOTS:
        for folder, _, _ in os.walk(root, topdown=False):
            if folder != root and not os.listdir(folder):
                os.rmdir(folder)
                removed += 1
    return removed


def documents():
    for folder, subdirs, files in os.walk("."):
        subdirs[:] = [d for d in subdirs if d not in SKIP]
        for name in files:
            if name.endswith((".md", ".html")):
                path = os.path.join(folder, name)[2:]
                if not path.startswith((f"{PROGRESS}/", "writing/overleaf-iclr/")):
                    yield path


class Links:
    """Decides what each link in a document becomes after the move."""

    def __init__(self, rows, guarded, groups):
        self.groups = groups
        self.renamed = {r["old_path"]: r["new_path"] for r in rows if r["action"].startswith("keep") and r["new_path"] != r["old_path"]}
        self.moved = {r["old_path"]: r["cluster_path"] for r in rows if r["action"].startswith("move")}
        self.hidden = {r["old_path"] for r in guarded}
        self.back = {new: old for old, new in self.renamed.items()}
        self.gone_folders = set()
        for path in list(self.moved) + list(self.renamed) + list(self.hidden):
            folder = os.path.dirname(path)
            while folder and folder not in self.gone_folders:
                self.gone_folders.add(folder)
                folder = os.path.dirname(folder)

    def cluster_folder(self, path):
        if path.startswith(DISCRIMINATOR + "/"):
            parts = path[len(DISCRIMINATOR) + 1:].split("/")
            if parts[0] in self.groups:
                path = "/".join([DISCRIMINATOR, self.groups[parts[0]]] + parts)
        return f"{CLUSTER}/{path}"

    def decide(self, doc, link):
        """None to leave it, ("link", new) to repoint it, ("cluster", path) to cite it."""
        if link.startswith(("http", "mailto", "#", "/")):
            return None
        bare, _, fragment = link.partition("#")
        suffix = f"#{fragment}" if fragment else ""
        old_doc = self.back.get(doc, doc)
        target = os.path.normpath(os.path.join(os.path.dirname(old_doc), urllib.parse.unquote(bare)))
        here = os.path.dirname(doc)
        if target in self.renamed:
            return "link", os.path.relpath(self.renamed[target], here) + suffix
        if target in self.moved:
            return "cluster", self.moved[target]
        if target in self.hidden:
            return "cluster", "held in the laptop archive"
        if target in self.gone_folders and not os.path.exists(target):
            return "cluster", self.cluster_folder(target)
        if old_doc != doc and os.path.exists(target):
            ending = "/" if bare.endswith("/") else ""
            return "link", os.path.relpath(target, here) + ending + suffix
        return None

    def rewrite(self, doc, text):
        changes = []

        def cite(link, where):
            name = os.path.basename(urllib.parse.unquote(link.partition("#")[0]).rstrip("/"))
            changes.append((link, where))
            return f"`{name}` is on the cluster at `{where}`" if where.startswith("/") else f"`{name}` is {where}"

        def whole(m):
            d = self.decide(doc, m.group(1))
            if d is None:
                return m.group(0)
            if d[0] == "cluster":
                return cite(m.group(1), d[1])
            changes.append((m.group(1), d[1]))
            return m.group(0).replace(m.group(1), d[1])

        def labelled(m, label, link):
            d = self.decide(doc, link)
            if d is None:
                return m.group(0)
            if d[0] == "cluster":
                changes.append((link, d[1]))
                return f"{label} (on the cluster at `{d[1]}`)" if label else cite(link, d[1])
            changes.append((link, d[1]))
            return m.group(0).replace(link, d[1])

        text = WRAPPED.sub(whole, text)
        text = IMAGE.sub(whole, text)
        text = MD_IMAGE.sub(whole, text)
        text = MD_LINK.sub(lambda m: labelled(m, m.group(1), m.group(2)), text)
        text = ANCHOR.sub(lambda m: labelled(m, m.group(2), m.group(1)), text)
        return text, changes


def fix_links(rows, guarded, write):
    links = Links(rows, guarded, json.load(open(GROUPS)))
    edited, plan_rows, tally = [], [], collections.Counter()
    for doc in documents():
        try:
            if os.path.getsize(doc) > 4_000_000:
                continue
            text = open(doc, errors="surrogateescape").read()
        except OSError:
            continue
        new, changes = links.rewrite(doc, text)
        if not changes:
            continue
        if re.match(r"(\d-[^/]+/)?plans/", doc):
            plan_rows += [(doc, old, value) for old, value in changes]
            tally["plan"] += len(changes)
            continue
        tally["page or card"] += len(changes)
        edited.append(doc)
        if write:
            backup = f"{ARCHIVE}/pages-before-links/{doc}"
            os.makedirs(os.path.dirname(backup), exist_ok=True)
            if not os.path.exists(backup):
                shutil.copy2(doc, backup)
            open(doc, "w", errors="surrogateescape").write(new)
    if write:
        with open(PLAN_LINKS, "w") as out:
            out.write("plan\told_link\tbecomes\n")
            for row in plan_rows:
                out.write("\t".join(row) + "\n")
    return edited, tally, sorted({r[0] for r in plan_rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    rows, guarded = load()
    problems = preflight(rows, guarded)
    moving = sum(r["action"].startswith("move") for r in rows)
    renamed = sum(r["action"].startswith("keep") and r["new_path"] != r["old_path"] for r in rows)
    print(f"checked {moving} files against their cluster checksums; {renamed} kept files change path; {len(guarded)} guarded")
    if problems:
        print(f"STOP: {len(problems)} problems")
        for problem in problems[:12]:
            print(" ", problem)
        sys.exit(1)
    if args.dry_run:
        edited, tally, plans = fix_links(rows, guarded, write=False)
        print("dry run: links", dict(tally), "documents", len(edited), "plans", plans)
        return
    # Links are decided while every old path can still be told apart, then written after the move.
    removed = move(rows, guarded)
    print(f"moved; {removed} empty folders removed")
    edited, tally, plans = fix_links(rows, guarded, write=True)
    print("links", dict(tally), "documents", len(edited), "plans", plans)
    with open(f"{ARCHIVE}/edited-documents.txt", "w") as out:
        out.write("\n".join(sorted(edited)) + "\n")


if __name__ == "__main__":
    main()
