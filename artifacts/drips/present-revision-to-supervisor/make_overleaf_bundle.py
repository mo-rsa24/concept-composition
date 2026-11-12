"""Build the Overleaf upload bundle for the ICLR 56173 revision.

Overleaf compiles from the project root, so the bundle is laid out as the root of the
supervisor's project will be after the upload:

    overleaf-bundle/
        iclr2027_conference_revision.tex   the revised tex, every path rewritten to revision/
        revision/                          every file it reads, flat

Only paths change in the tex. Run from the repo root:

    python3 artifacts/drips/present-revision-to-supervisor/make_overleaf_bundle.py
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "paper/iclr-56173"
OUT = Path(__file__).resolve().parent / "overleaf-bundle"
MAIN = "iclr2027_conference_revision.tex"
TEMPLATE = ["iclr2027_conference.sty", "iclr2027_conference.bst", "iclr2027_conference.bib",
            "natbib.sty", "fancyhdr.sty", "math_commands.tex"]


def live_tex(text: str) -> str:
    return "".join(re.sub(r"(?<!\\)%.*", "", l) for l in text.splitlines(True)
                   if not l.lstrip().startswith("%"))


def resolve(path: str, exts=("", ".pdf", ".png", ".jpg")) -> Path | None:
    for base in ("", "figures/"):
        for e in exts:
            p = PAPER / f"{base}{path}{e}"
            if p.is_file():
                return p
    return None


def included_files(tex: str) -> tuple[dict[str, Path], list[str]]:
    """Map each include argument in the live tex to the file it resolves to."""
    live = live_tex(tex)
    found: dict[str, Path] = {}
    pending: list[str] = []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}#]+)\}", live):
        found[m.group(1)] = resolve(m.group(1))
    for m in re.finditer(r"\\reviewslot(?:\[([^\]]*)\])?\{([^}]+)\}", live):
        sid = m.group(2)
        hit = next((PAPER / f"slots/{sid}{e}" for e in (".pdf", ".png", ".tex")
                    if (PAPER / f"slots/{sid}{e}").is_file()), None)
        if hit:
            found[f"slot:{sid}"] = hit
        else:
            pending.append(sid)
    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise SystemExit(f"unresolved includes: {missing}")
    return found, pending


def rewrite(tex: str, found: dict[str, Path]) -> str:
    """Prefix every path the tex reads with revision/, flattening directories."""
    for arg, path in found.items():
        if arg.startswith("slot:"):
            continue
        tex = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{)" + re.escape(arg) + r"\}",
                     lambda m: m.group(1) + "revision/" + path.name + "}", tex)
    tex = tex.replace("{slots/#2.pdf}", "{revision/#2.pdf}").replace("{slots/#2.png}", "{revision/#2.png}")
    tex = tex.replace("{slots/#2.tex}", "{revision/#2.tex}")
    tex = tex.replace(r"\graphicspath{{figures/}}", r"\graphicspath{{revision/}}")
    # The template RequirePackages fancyhdr and natbib by name, which outside the paper folder finds
    # the system copies, and the newer fancyhdr drops the running header. Load the paper's own copies
    # from revision/ and mark them loaded, so those RequirePackage lines are no-ops.
    tex = tex.replace(r"\usepackage{iclr2027_conference,times}",
                      r"\usepackage{revision/fancyhdr}\usepackage{revision/natbib}" + "\n"
                      + r"\makeatletter\@namedef{ver@fancyhdr.sty}{}\@namedef{ver@natbib.sty}{}\makeatother" + "\n"
                      + r"\usepackage{revision/iclr2027_conference}" + "\n" + r"\usepackage{times}")
    tex = tex.replace(r"\input{math_commands.tex}", r"\input{revision/math_commands.tex}")
    tex = tex.replace(r"\bibliography{iclr2027_conference}", r"\bibliography{revision/iclr2027_conference}")
    tex = tex.replace(r"\bibliographystyle{iclr2027_conference}", r"\bibliographystyle{revision/iclr2027_conference}")
    return tex


def main() -> None:
    tex = (PAPER / "iclr2027_conference.tex").read_text()
    found, pending = included_files(tex)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "revision").mkdir(parents=True)
    files = sorted(set(found.values()) | {PAPER / t for t in TEMPLATE})
    names = [p.name for p in files]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"two files would share a name in revision/: {dup}")
    for p in files:
        shutil.copy2(p, OUT / "revision" / p.name)
    new = rewrite(tex, found)
    (OUT / MAIN).write_text(new)
    left = [l for l in live_tex(new).splitlines() if re.search(r"\{(slots|ablation|figures|final_figures)/", l)]
    if left:
        raise SystemExit("paths not rewritten:\n" + "\n".join(left))
    # The editable reviews-and-replies document: needs no file from revision/, sits at the root.
    shutil.copy2(REPO / "artifacts/scenes/iclr-review-response/review.tex", OUT / "review.tex")
    zpath = OUT.parent / "overleaf-bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT))
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"{MAIN} + revision/ ({len(files)} files), {total / 1e6:.1f} MB; zip {zpath.stat().st_size / 1e6:.1f} MB")
    print("pending slots (print their placeholder box):", pending)
    for f in sorted((OUT / "revision").iterdir()):
        print(f"  revision/{f.name}  {f.stat().st_size / 1e6:.2f} MB  <- {os.path.relpath(next(p for p in files if p.name == f.name), PAPER)}")


if __name__ == "__main__":
    main()
