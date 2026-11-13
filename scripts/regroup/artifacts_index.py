#!/usr/bin/env python3
"""Write an index README for each project's walk maps, idea maps, notes and scenes.

The same idea as results_index.py, for the other four kinds of artifact: `artifacts/drips/`,
`artifacts/ideas/`, `artifacts/notes/` and `artifacts/scenes/` in every project and at the root.
Each entry shows the item's own title, its first sentence and when it last changed; a walk also
names the plan or route it walks, and a scene shows a thumbnail. Groups are the folders listed in
scripts/regroup/artifact-groups.txt; each group gets its own index. Re-run when items change.

    python3 scripts/regroup/artifacts_index.py
"""
import glob, os, re, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARK = "<!-- index: written by scripts/regroup/artifacts_index.py; edit the items, not this file -->"
GROUPS = {l.strip() for l in open(os.path.join(ROOT, "scripts/regroup/artifact-groups.txt")) if l.strip()}
KINDS = {"drips": "Walks", "ideas": "Idea maps", "notes": "Notes", "scenes": "Scenes"}
MAIN = ("README.md", "EXECUTE_WALK.md", "REPO_WALK.md", "READ_MAP.md", "IDEA_MAP.md", "ROUTE_WALK.md", "COMBINE_MAP.md")
IMG = (".png", ".jpg", ".jpeg", ".svg")


def main_file(d):
    for m in MAIN:
        if os.path.exists(os.path.join(d, m)):
            return os.path.join(d, m)
    mds = sorted(glob.glob(os.path.join(d, "*.md")))
    return mds[0] if mds else None


def describe(d):
    f = main_file(d)
    title, first, walks = None, "", None
    if f:
        text = open(f, errors="ignore").read()
        for line in text.splitlines():
            s = line.strip()
            if not title and s.startswith("#"):
                title = s.lstrip("# ").strip(); continue
            if title and s and not s.startswith(("#", "|", "<", "!", "```", "-", "*", ">")) and not first:
                first = re.split(r"(?<=[.!?])\s", re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s))[0][:150]
        m = re.search(r"\]\(([^)\s]*plans/[^)\s]*\.md)", text)
        if m:
            walks = m.group(1)
    target = os.path.relpath(f or d, ROOT)
    args = ["git", "-C", ROOT, "log", "-1", "--format=%cd", "--date=short", "--diff-filter=AM"] + (["--follow"] if f else [])
    when = subprocess.run(args + ["--", target], capture_output=True, text=True).stdout.strip() or "not committed"
    return f, title, first, walks, when


def write(d, label, kind):
    kids = sorted(e for e in os.listdir(d) if os.path.isdir(os.path.join(d, e)) and not e.startswith((".", "__")))
    groups = [k for k in kids if os.path.relpath(os.path.join(d, k), ROOT) in GROUPS]
    items = [k for k in kids if k not in groups]
    out = [f"# {label}", "", MARK, ""]
    if groups:
        out += ["| Group | What it holds |", "|---|---|"]
        for g in groups:
            sub = sorted(e for e in os.listdir(os.path.join(d, g)) if os.path.isdir(os.path.join(d, g, e)) and not e.startswith((".", "__")))
            names = [describe(os.path.join(d, g, s))[1] or s for s in sub]
            out.append(f"| [{g.replace('-', ' ').capitalize()}]({g}/README.md) | {len(sub)}: " + " · ".join(names) + " |")
        out.append("")
    if items:
        head = {"drips": "| Walk | What it is | Walks | Last change |", "scenes": "| Scene | What it is | Picture | Last change |"}.get(kind, "| Item | What it is | Last change |")
        out += [head, "|---" * head.count("|")[:-1] if False else "|" + "---|" * (head.count("|") - 1)]
        for k in items:
            p = os.path.join(d, k); f, title, first, walks, when = describe(p)
            link = os.path.relpath(f, d) if f else f"{k}/"
            cell = f"[{title or k}]({link})"
            if kind == "drips":
                w = f"[the plan]({os.path.relpath(os.path.normpath(os.path.join(os.path.dirname(f), walks)), d)})" if walks and f and os.path.exists(os.path.normpath(os.path.join(os.path.dirname(f), walks))) else "—"
                out.append(f"| {cell} | {first or '—'} | {w} | {when} |")
            elif kind == "scenes":
                imgs = sorted(i for i in glob.glob(os.path.join(p, "**", "*"), recursive=True) if i.lower().endswith(IMG))
                pic = f'<a href="{os.path.relpath(imgs[0], d)}"><img src="{os.path.relpath(imgs[0], d)}" width="160" alt="{k}"></a>' if imgs else "—"
                out.append(f"| {cell} | {first or '—'} | {pic} | {when} |")
            else:
                out.append(f"| {cell} | {first or '—'} | {when} |")
        out.append("")
    readme = os.path.join(d, "README.md")
    if os.path.exists(readme) and MARK not in open(readme, errors="ignore").read():
        readme = os.path.join(d, "00-INDEX.md")  # never overwrite a hand-written README
    open(readme, "w").write("\n".join(out))
    for g in groups:
        write(os.path.join(d, g), g.replace("-", " ").capitalize(), kind)


def main():
    bases = sorted(glob.glob(os.path.join(ROOT, "[0-9]-*"))) + [ROOT]
    for b in bases:
        proj = os.path.basename(b) if b != ROOT else ""
        for kind, label in KINDS.items():
            d = os.path.join(b, "artifacts", kind)
            if os.path.isdir(d) and any(os.path.isdir(os.path.join(d, e)) for e in os.listdir(d)):
                name = re.sub(r"^[0-9]-", "", proj).replace("-", " ").capitalize().replace("Lora", "LoRA") if proj else "Shared"
                write(d, f"{label}: {name}", kind)
                print("indexed", os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
