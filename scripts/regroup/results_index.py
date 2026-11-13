#!/usr/bin/env python3
"""Write an index README at every level of every artifacts/results/ tree, so a result is found by
reading down from the project, never by opening folders blind.

A folder is a *question folder* when it holds a result (its own card, figures or data); it is a
*group* when it only holds other folders. Each group and each results root gets a README.md
marked as generated, listing its children: a group by the questions it holds, a question folder
by its card's title, its first sentence and a thumbnail of its first figure. Re-run it whenever
results change; it never touches a question folder's own card.

    python3 scripts/regroup/results_index.py
"""
import glob, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARK = "<!-- index: written by scripts/regroup/results_index.py; edit the cards, not this file -->"
IMG = (".png", ".jpg", ".jpeg", ".svg", ".gif")


GROUPS = {l.strip() for l in open(os.path.join(ROOT, "scripts/regroup/result-groups.txt")) if l.strip()}


def is_group(d):
    """Groups are listed in scripts/regroup/result-groups.txt; every other folder is a result."""
    return os.path.relpath(d, ROOT) in GROUPS


def card(d):
    r = os.path.join(d, "README.md")
    if not os.path.exists(r):
        return None, ""
    title, first = None, ""
    for line in open(r, errors="ignore"):
        s = line.strip()
        if not title and s.startswith("#"):
            title = s.lstrip("# ").strip(); continue
        if title and s and not s.startswith(("#", "|", "<", "!", "```", "-", "*", ">")):
            first = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
            first = re.split(r"(?<=[.!?])\s", first)[0][:160]
            break
    return title, first


def thumb(d):
    for pat in ("*", "*/*", "*/*/*"):
        imgs = sorted(f for f in glob.glob(os.path.join(d, pat)) if f.lower().endswith(IMG))
        if imgs:
            return imgs[0]
    return None


def nice(name):
    return name.replace("-", " ").capitalize()


def questions_under(d):
    if not is_group(d):
        return [d]
    out = []
    for e in sorted(os.listdir(d)):
        p = os.path.join(d, e)
        if os.path.isdir(p) and not e.startswith((".", "__")):
            out += questions_under(p)
    return out


def write_index(d, top_label):
    kids = sorted(e for e in os.listdir(d) if os.path.isdir(os.path.join(d, e)) and not e.startswith((".", "__")))
    lines = [f"# {top_label}", "", MARK, ""]
    groups = [k for k in kids if is_group(os.path.join(d, k))]
    qs = [k for k in kids if k not in groups]
    if groups:
        lines += ["| Group | What it holds |", "|---|---|"]
        for g in groups:
            q = questions_under(os.path.join(d, g))
            names = [card(x)[0] or nice(os.path.basename(x)) for x in q]
            lines.append(f"| [{nice(g)}]({g}/README.md) | {len(q)} results: " + " · ".join(names) + " |")
        lines.append("")
    if qs:
        lines += ["| Result | What it shows | Picture |", "|---|---|---|"]
        for k in qs:
            p = os.path.join(d, k); title, first = card(p); t = thumb(p)
            link = f"{k}/README.md" if os.path.exists(os.path.join(p, "README.md")) else f"{k}/"
            pic = f'<a href="{os.path.relpath(t, d)}"><img src="{os.path.relpath(t, d)}" width="160" alt="{k}"></a>' if t else "no figure"
            lines.append(f"| [{title or nice(k)}]({link}) | {first or 'no card yet'} | {pic} |")
        lines.append("")
    open(os.path.join(d, "README.md"), "w").write("\n".join(lines))
    for g in groups:
        write_index(os.path.join(d, g), nice(g))


def main():
    roots = sorted(glob.glob(os.path.join(ROOT, "*/artifacts/results"))) + [os.path.join(ROOT, "artifacts/results")]
    for r in roots:
        if not os.path.isdir(r):
            continue
        proj = os.path.relpath(r, ROOT).split("/")[0]
        label = "Shared results" if proj == "artifacts" else "Results: " + nice(re.sub(r"^[0-9]-", "", proj)).replace("Lora", "LoRA")
        write_index(r, label)
        print("indexed", os.path.relpath(r, ROOT))


if __name__ == "__main__":
    main()
