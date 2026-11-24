"""Census of loop figures, scenes and reviews that no pillar page cites.

Run from the repository root:  python3 artifacts/notes/progress/drain/census.py [--json OUT]

A pillar page is any markdown file under report/, context/, runbook/ or environment/, at the root or
inside one of the five project folders. An item counts as placed when a pillar page links or embeds
its exact path. A page that links only the item's folder counts as "folder only".
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
PILLARS = ["report", "context", "runbook", "environment"]
PROJECTS = [".", "1-lora-correction", "2-classifier-guidance", "3-understanding-compositionality",
            "4-inclusive-captions", "5-text-embedding-third-expert"]

LOOP_ROOTS = {
    "artifacts/notes/progress/targets": "Loop 4",
    "artifacts/notes/progress/reproductions": "Loop 5",
    "artifacts/notes/progress/method-search": "Loop 6",
    "artifacts/notes/progress/abstract-checkers/01-reading": "Loop 7",
    "artifacts/notes/progress/abstract-checkers": "Loop 8",
    "artifacts/notes/progress/noise-text-critic": "Loop 9",
    "artifacts/notes/progress/score-space": "Loop 10",
    "artifacts/notes/progress/leakage": "Loop 12",
    "2-classifier-guidance/artifacts/scenes": "scenes",
    "2-classifier-guidance/artifacts/results": "results",
    "1-lora-correction/artifacts/results": "results",
}
PIC = {".png", ".jpg", ".jpeg", ".svg", ".gif"}
SKIP_PARTS = {"__pycache__", "pdfs", "tools", "rows"}
LINK = re.compile(r"""\]\(([^)\s#]+)|(?:src|href)=["']([^"'#]+)|`([^`\s]+\.(?:png|jpe?g|svg|html|md|json|gif))`""")


def pillar_pages():
    for proj in PROJECTS:
        for p in PILLARS:
            base = ROOT / proj / p
            if base.is_dir():
                yield from base.rglob("*.md")


def references():
    refs: dict[str, set[str]] = {}
    for page in pillar_pages():
        try:
            text = page.read_text(errors="ignore")
        except OSError:
            continue
        rel_page = str(page.relative_to(ROOT))
        for m in LINK.finditer(text):
            target = next(g for g in m.groups() if g)
            if target.startswith(("http", "mailto")):
                continue
            for cand in (page.parent / target, ROOT / target):
                try:
                    r = os.path.normpath(os.path.relpath(cand.resolve(), ROOT))
                except (OSError, ValueError):
                    continue
                refs.setdefault(r, set()).add(rel_page)
    return refs


def loop_of(path: str) -> str:
    best = ""
    for root, name in LOOP_ROOTS.items():
        if path.startswith(root + "/") and len(root) > len(best):
            best = root
    return LOOP_ROOTS.get(best, "?")


def items():
    seen = set()
    for root in LOOP_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file() or SKIP_PARTS & set(f.parts):
                continue
            rel = str(f.relative_to(ROOT))
            if rel in seen:
                continue
            ext = f.suffix.lower()
            kind = None
            if ext in PIC:
                if ext == ".svg" and f.with_suffix(".png").exists():
                    continue  # the png twin stands for it
                kind = "picture"
            elif ext == ".html":
                kind = "scene"
            elif f.name == "review.md" or (f.parent.name == "review" and ext == ".md"):
                kind = "review"
            if kind:
                seen.add(rel)
                yield rel, kind


LOOPS_STARTED = "2026-11-21"
DESC = re.compile(r'<img[^>]*src="([^"]+)"[^>]*alt="([^"]*)"|<img[^>]*alt="([^"]*)"[^>]*src="([^"]+)"'
                  r'|!\[([^\]]+)\]\(([^)\s]+)\)|\[([^\]]{6,})\]\(([^)\s#]+\.(?:png|svg|jpg|html))\)')


def first_added():
    import subprocess
    roots = list(LOOP_ROOTS)
    log = subprocess.run(["git", "log", "--diff-filter=A", "--name-only", "--format=@@%ad", "--date=short", "--"]
                         + roots, capture_output=True, text=True).stdout
    first, date = {}, None
    for line in log.splitlines():
        if line.startswith("@@"):
            date = line[2:]
        elif line.strip():
            first[line.strip()] = date
    tracked = set(subprocess.run(["git", "ls-files", "--"] + roots, capture_output=True, text=True).stdout.split())
    return first, tracked


def descriptions():
    out = {}
    for r in ["artifacts/notes/progress", "2-classifier-guidance/artifacts", "plans"]:
        for f in (ROOT / r).rglob("*.md"):
            try:
                t = f.read_text(errors="ignore")
            except OSError:
                continue
            for m in DESC.finditer(t):
                g = m.groups()
                src, text, rank = ((g[0], g[1], 2) if g[0] else (g[3], g[2], 2) if g[3]
                                   else (g[5], g[4], 2) if g[5] else (g[7], g[6], 1))
                try:
                    rel = os.path.normpath(os.path.relpath((f.parent / src).resolve(), ROOT.resolve()))
                except (OSError, ValueError):
                    continue
                text = text.replace("`", "")
                if rel not in out or (rank, len(text)) > out[rel][:2]:
                    out[rel] = (rank, len(text), text)
    return {k: v[2] for k, v in out.items()}


def review_status(path):
    paras = re.split(r"\n\s*\n", (ROOT / path).read_text(errors="ignore"))
    for para in paras:
        s = " ".join(para.split())
        if s and not s.startswith(("#", "<", "|", "!")):
            s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s).replace("**", "")
            return ". ".join(s.split(". ")[:2])[:330]
    return ""


def held_set():
    f = ROOT / "artifacts/notes/progress/held-pictures.md"
    return set(re.findall(r"`([^`]+\.(?:png|jpe?g|svg))`", f.read_text())) if f.exists() else set()


def main():
    refs = references()
    out = []
    for rel, kind in items():
        pages = sorted(refs.get(rel, ()))
        folder_pages = set()
        parent = os.path.dirname(rel)
        while parent and parent not in LOOP_ROOTS:
            folder_pages |= refs.get(parent, set())
            parent = os.path.dirname(parent)
        state = "placed" if pages else ("folder only" if folder_pages else "unplaced")
        out.append({"path": rel, "loop": loop_of(rel), "kind": kind, "state": state,
                    "pages": pages, "folder_pages": sorted(folder_pages)})
    first, tracked = first_added()
    desc = descriptions()
    held = held_set()
    sys.path.insert(0, str(ROOT / "scripts/shared"))
    import disallowed_subjects as guard
    for o in out:
        p = o["path"]
        o["added"] = first.get(p) or ("untracked" if p not in tracked else "?")
        o["loop_made"] = o["added"] == "untracked" or (o["added"] != "?" and o["added"] >= LOOPS_STARTED)
        o["held"] = p in held or not guard.is_showable(p)
        if o["kind"] == "review":
            o["shows"], o["shows_src"] = review_status(p), "review"
        elif desc.get(p) or (p.endswith(".png") and desc.get(p[:-4] + ".svg")):
            o["shows"], o["shows_src"] = desc.get(p) or desc[p[:-4] + ".svg"], "caption"
        else:
            o["shows"], o["shows_src"] = Path(p).stem.replace("-", " ").replace("_", " "), "file name"
    if "--markdown" in sys.argv:
        Path(sys.argv[sys.argv.index("--markdown") + 1]).write_text(render(out))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1))
    from collections import Counter
    c = Counter((o["loop"], o["kind"], o["state"]) for o in out)
    for k in sorted(c):
        print(f"{c[k]:5d}  {k[0]:8s} {k[1]:8s} {k[2]}")
    print("total", len(out), "unplaced", sum(o["state"] == "unplaced" for o in out))



UP = "../../../../"


def _cell(s, n=240):
    s = " ".join(str(s).split()).replace("|", "/")
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _link(path):
    return f"[{Path(path).name}]({UP}{path})"


def _page(page):
    if not page:
        return ""
    exists = (ROOT / page).exists()
    return f"`{page}`" + ("" if exists else " (to write)")


def render(out):
    import datetime
    sys.path.insert(0, str(Path(__file__).parent))
    from landing import landing
    today = datetime.date.today().isoformat()
    loopish = [o for o in out if o["state"] != "placed"
               and (o["loop"].startswith("Loop") or (o["loop"] == "scenes" and o["loop_made"]))]
    older = [o for o in out if o["state"] != "placed" and o not in loopish]
    for o in out:
        o["land"], o["note"] = landing(o["path"])
    L = []
    w = L.append
    w("# 🗂️ Drain census: the loops' figures and results that no pillar page shows")
    w("")
    w(f"This page lists every picture, scene page and review file the looping sessions made that no page under "
      f"`report/`, `context/`, `runbook/` or `environment/` links or embeds, with the page each should land on. "
      f"It was written on {today} by `python3 artifacts/notes/progress/drain/census.py --markdown "
      f"artifacts/notes/progress/drain/01-census.md`, run from the repository root, and the landing pages come from "
      f"[the landing rules](landing.py). Rerun it to refresh the page.")
    w("")
    w("## Words this page uses")
    w("")
    w("**Placed** means a pillar page links or embeds the file's exact path. **Folder only** means a pillar page links "
      "a folder above the file but not the file. **Unplaced** means neither. **Made by a loop** means the file first "
      f"entered git on or after {LOOPS_STARTED}, when the looping sessions started, or is not in git yet. "
      "Results folders count as made before the loops even where a folder move on 2026-11-22 re-added them to git. "
      "**What it shows** is the caption the loop wrote where it embeds or links the picture, the review's own "
      "opening sentences for a review, and otherwise the file name in plain words, marked *(file name)*.")
    w("")
    w("## Why the drain never found these")
    w("")
    w("The drain's sweep (`/artifacts-drain . --dry-run`) found 8 new and 4 changed files on this run, all of them "
      "scene cards under the root `artifacts/scenes/`. None of the loops' work was among them, for two reasons:")
    w("")
    w("1. The drain ledger, `report/.artifacts-drain-ledger.json`, holds `artifacts/notes/progress` out of every "
      "drain, and every loop folder (`targets/`, `reproductions/`, `method-search/`, `abstract-checkers/`, "
      "`noise-text-critic/`, `score-space/`) sits inside it.")
    w("2. The sweep reads only the root `artifacts/`. It never looks in `2-classifier-guidance/artifacts/` or "
      "`1-lora-correction/artifacts/`, where the loops' scenes and the older results live.")
    w("")
    w("## The counts")
    w("")
    from collections import Counter, defaultdict
    w("| Where it came from | Pictures and scene pages, unplaced | Reviews holding a verdict | Reviews with no verdict yet |")
    w("|---|---|---|---|")
    groups = defaultdict(list)
    for o in loopish:
        groups[o["loop"]].append(o)
    order = ["Loop 4", "Loop 5", "Loop 6", "Loop 7", "Loop 8", "Loop 9", "Loop 10", "Loop 12", "scenes", "results"]
    names = {"scenes": "scenes in `2-classifier-guidance/artifacts/scenes/`",
             "results": "results folders, made by a loop"}

    def has_verdict(o):
        return o["kind"] == "review" and not (o["note"].startswith("waits") or o["note"].startswith("no finding"))
    for g in order:
        items = groups.get(g, [])
        if not items:
            continue
        pics = sum(o["kind"] != "review" for o in items)
        v = sum(has_verdict(o) for o in items)
        nv = sum(o["kind"] == "review" and not has_verdict(o) for o in items)
        w(f"| {names.get(g, g)} | {pics} | {v} | {nv} |")
    w(f"| older results and scenes, made before the loops | {len(older)} | not counted | not counted |")
    w("")
    w("## Results with a verdict, for step Y2")
    w("")
    w("Each row is a review file whose verdict is recorded. The verdict column quotes the review's opening sentences; "
      "Y2 copies the full verdict, numbers, bar and run id from the review itself.")
    w("")
    w("| File | Loop | What the review records | Lands on |")
    w("|---|---|---|---|")
    for o in loopish:
        if has_verdict(o):
            w(f"| {_link(o['path'])} `{Path(o['path']).parent.name}` | {o['loop']} | {_cell(o['shows'], 330)} | {_page(o['land'])} |")
    w("")
    w("## Reviews with no verdict yet")
    w("")
    w("These get no finding until their verdict lands. Y5 checks them on every tick.")
    w("")
    w("| File | Loop | What the review records | Why it waits |")
    w("|---|---|---|---|")
    for o in loopish:
        if o["kind"] == "review" and not has_verdict(o):
            w(f"| {_link(o['path'])} `{Path(o['path']).parent.name}` | {o['loop']} | {_cell(o['shows'], 260)} | {o['note']} |")
    w("")
    w("## Pictures and scene pages, by loop")
    w("")
    w("Folders with more than ten files are folded; open one to see every file.")
    w("")
    for g in order:
        items = [o for o in groups.get(g, []) if o["kind"] != "review"]
        if not items:
            continue
        w(f"### {names.get(g, g).capitalize() if g in names else g}")
        w("")
        byfolder = defaultdict(list)
        for o in items:
            byfolder[str(Path(o["path"]).parent)].append(o)
        for folder in sorted(byfolder):
            rows = byfolder[folder]
            head = f"`{folder}/`: {len(rows)} file{'s' if len(rows) != 1 else ''}"
            table = ["| File | What it shows | Lands on |", "|---|---|---|"]
            for o in rows:
                shows = _cell(o["shows"]) + (" *(file name)*" if o["shows_src"] == "file name" else "")
                if o["held"]:
                    shows += " **Held: fails the subject guard or is on the held-pictures list, so it stays off every page.**"
                land = _page(o["land"]) or o["note"]
                if o["land"] and o["note"]:
                    land += f", {o['note']}"
                if o["state"] == "folder only":
                    land += f". Its folder is already linked from {', '.join('`'+p+'`' for p in o['folder_pages'][:2])}"
                table.append(f"| {_link(o['path'])} | {shows} | {land} |")
            if len(rows) > 10:
                w(f"<details><summary>{head}</summary>")
                w("")
                L.extend(table)
                w("")
                w("</details>")
            else:
                w(f"**{head}**")
                w("")
                L.extend(table)
            w("")
    w("## Older results and scenes the census also found")
    w("")
    w("These sit in the results and scenes folders but were made before the loops started, so this drain does not "
      "place them. They are listed so the count is complete; whether each has a finding was not checked here.")
    w("")
    nheld = sum(o["held"] for o in older)
    w(f"{nheld} of them fail the subject guard (`scripts/shared/disallowed_subjects.py`, read by file name) or sit on "
      f"the held-pictures list, and stay off every page. No picture a loop made fails either check.")
    w("")
    w("| Folder | Unplaced files | Note |")
    w("|---|---|---|")
    c = Counter("/".join(str(Path(o["path"]).parent).split("/")[:5]) for o in older)
    for k, v in sorted(c.items()):
        w(f"| `{k}/` | {v} | {landing(k + '/x')[1]} |")
    w("")
    w("## Folders the drain does not look in")
    w("")
    w("| Folder | Why the sweep misses it | Who covers it now |")
    w("|---|---|---|")
    w("| `artifacts/notes/progress/` and every loop folder inside it | held in the drain ledger | this census, rerun by step Y5 |")
    w("| `2-classifier-guidance/artifacts/` | the sweep reads the root `artifacts/` only | this census, for `scenes/` and `results/` |")
    w("| `1-lora-correction/artifacts/` | the sweep reads the root `artifacts/` only | this census, for `results/` |")
    w("| `3-understanding-compositionality/artifacts/`, `4-inclusive-captions/artifacts/`, `5-text-embedding-third-expert/artifacts/` | the sweep reads the root `artifacts/` only | nobody: no loop writes there |")
    w("| `/datasets/mmolefe/poe_repair_min/loop-5/` to `loop-10/` | on the cluster, outside the repository | each loop's review cites what it needs from there |")
    w("")
    return "\n".join(L) + "\n"

if __name__ == "__main__":
    main()
