#!/usr/bin/env python3
"""Rebuild the review-response slots of the paper from their result files.

Every slot (R1, R2, ...) is a row of report/paper-review/layout-sheet.md naming one result file.
A PNG or PDF result is copied to <paper>/slots/<slot>.<ext>; a CSV result is rendered as a LaTeX
tabular into <paper>/slots/<slot>.tex. The tex's \\reviewslot macro picks the file up, and shows a
placeholder box at the slot's final size while none exists. Result files are only ever read.

    python3 scripts/refresh_paper_slots.py --check                 every slot wired, PASS or MISS
    python3 scripts/refresh_paper_slots.py                         refresh every slot whose result exists
    python3 scripts/refresh_paper_slots.py --slot R1 --compile     refresh one slot, then build on mscluster
    python3 scripts/refresh_paper_slots.py --compile --gallery     build, then one sheet of every slot's page
    python3 scripts/refresh_paper_slots.py --slot R1 --from f.csv --paper <copy> --remote <dir> --compile
                                                                   test a slot on a copy of the paper

A CSV must carry the twelve axis columns of protocol rule 9 (report/paper-review/protocol.md);
--check and the refresh both refuse one that does not.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHEET = REPO / "report/paper-review/layout-sheet.md"
PAPER = REPO / "paper/iclr-56173"
TEX_NAME = "iclr2027_conference.tex"
REGISTER_NAME = "figures.md"
REMOTE = "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/paper/iclr-56173"
TECTONIC = "/home-mscluster/mmolefe/.local/bin/tectonic"
AXES = ["model", "method", "adapter", "sampler", "steps", "guidance", "resolution",
        "pair", "seed", "group", "setting", "nfe"]
KINDS = {".png", ".pdf", ".csv"}


def slots() -> list[dict]:
    """Slot rows of the layout sheet: id, size cell and result file."""
    out = []
    for line in SHEET.read_text().splitlines():
        m = re.match(r"^\|\s*(R\d+)\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        result = re.search(r"`(artifacts/results/[^`]+)`", line)
        out.append({"id": m.group(1), "size": cells[3], "result": result.group(1) if result else None})
    return out


def tex_escape(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", s).replace("±", r"$\pm$").replace("†", r"$^\dagger$")


def missing_axes(path: Path) -> list[str]:
    with path.open(newline="") as f:
        header = next(csv.reader(f), [])
    return [a for a in AXES if a not in header]


# Printed column labels, short enough that R1's columns fit the text width at 8 pt.
# The caption carries the key; a CSV column not listed here prints under its own name.
LABELS = {"compose rate ±CI": "Comp.", "CLIP": "CLIP", "ImageReward": "IR", "BLIP-VQA": "BLIP",
          "T2I-CompBench++": "CB++", "GenEval": "GE", "GenEval two-obj": "GE-2",
          "GenEval counting": "GE-N", "T2I-CB spatial": "CB-S", "T2I-CB complex": "CB-C", "nfe": "NFE"}
FIRST_BLOCK = "SDXL"
# A slot that shows one model's rows of a shared result file (the layout sheet says which).
MODEL_OF_SLOT = {"R1": "SDXL", "R5": "SD 1.5"}


def csv_to_tabular(path: Path, sid: str) -> str:
    """One row per CSV row, grouped into blocks by the `model` column, SDXL first.

    Every block after the first sits under a rule with a label row naming its model, which is
    protocol rule 7's mixed marking. Columns: the method, every score column, then network
    evaluations. A slot in MODEL_OF_SLOT keeps only that model's rows. The table is centered at its natural width with 6 pt between columns and rows
    1.25 times line height. When it is wider than the text it is shrunk to fit and the log gets a
    'reviewslot <id> shrunk' line, which --compile reports, because a shrunk table has text below
    the layout sheet's 7 pt minimum.
    """
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if sid in MODEL_OF_SLOT:
        rows = [r for r in rows if r["model"] == MODEL_OF_SLOT[sid]]
        if not rows:
            raise SystemExit(f"{sid}: {path} has no rows with model {MODEL_OF_SLOT[sid]}")
    scores = [c for c in rows[0] if c not in AXES]
    cols = ["method"] + scores + ["nfe"]
    head = [""] + [LABELS.get(c, c) for c in scores] + [LABELS["nfe"]]
    models = sorted({r["model"] for r in rows}, key=lambda m: (m != FIRST_BLOCK, m))
    lines = [r"\setbox0=\hbox{\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.25}%",
             r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\hline",
             " & ".join(tex_escape(h) for h in head) + r" \\", r"\hline"]
    for k, model in enumerate(models):
        if k:
            lines += [r"\hline", r"\multicolumn{" + str(len(cols)) + r"}{l}{\textit{" + tex_escape(model)
                      + r" only, each method's official code on its own model}} \\"]
        lines += [" & ".join(tex_escape(r[c]) for c in cols) + r" \\" for r in rows if r["model"] == model]
    lines += [r"\hline", r"\end{tabular}}%",
              r"\ifdim\wd0>\linewidth\typeout{reviewslot " + sid + r" shrunk: \the\wd0}%",
              r"\centerline{\resizebox{\linewidth}{!}{\box0}}\else\centerline{\box0}\fi", ""]
    return "\n".join(lines)


def refresh(slot: dict, paper: Path, source: Path | None) -> str:
    src = source or (REPO / slot["result"] if slot["result"] else None)
    if src is None or not src.exists():
        return f"{slot['id']}: placeholder (no result at {slot['result']})"
    ext = src.suffix.lower()
    if ext not in KINDS:
        raise SystemExit(f"{slot['id']}: {src} is {ext}; a slot takes .png, .pdf or .csv")
    out_dir = paper / "slots"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob(f"{slot['id']}.*"):
        old.unlink()
    if ext == ".csv":
        gaps = missing_axes(src)
        if gaps:
            raise SystemExit(f"{slot['id']}: {src} lacks axis columns {gaps} (protocol rule 9)")
        (out_dir / f"{slot['id']}.tex").write_text(csv_to_tabular(src, slot["id"]))
        return f"{slot['id']}: table from {src} -> {out_dir / (slot['id'] + '.tex')}"
    shutil.copy2(src, out_dir / f"{slot['id']}{ext}")
    return f"{slot['id']}: image from {src} -> {out_dir / (slot['id'] + ext)}"


def check(paper: Path) -> bool:
    tex = (paper / TEX_NAME).read_text()
    register = (paper / REGISTER_NAME).read_text()
    found, problems, filled = slots(), [], 0
    for s in found:
        sid = s["id"]
        if not re.search(r"\\reviewslot(\[[^\]]*\])?\{" + sid + r"\}", tex):
            problems.append(f"{sid}: no \\reviewslot{{{sid}}} in {TEX_NAME}")
        if not re.search(r"^\|\s*" + sid + r"\s*\|", register, re.M):
            problems.append(f"{sid}: no row in {REGISTER_NAME}")
        if not s["result"]:
            problems.append(f"{sid}: no result file named on the layout sheet")
            continue
        res = REPO / s["result"]
        if Path(s["result"]).suffix.lower() not in KINDS:
            problems.append(f"{sid}: result {s['result']} is not .png, .pdf or .csv")
        if res.exists():
            filled += 1
            if res.suffix.lower() == ".csv" and missing_axes(res):
                problems.append(f"{sid}: {s['result']} lacks axis columns {missing_axes(res)}")
    extra = set(re.findall(r"\\reviewslot(?:\[[^\]]*\])?\{(R\d+)\}", tex)) - {s["id"] for s in found}
    problems += [f"{sid}: in the tex but not on the layout sheet" for sid in sorted(extra)]
    for p in problems:
        print("  " + p)
    verdict = "PASS" if found and not problems else "MISS"
    print(f"{verdict}: {len(found)} slots, {filled} with a result file, {len(found) - filled} placeholders")
    return verdict == "PASS"


def compile_remote(paper: Path, remote: str) -> int:
    """rsync the paper folder to mscluster, build with Tectonic there, pull the PDF and log back."""
    run = lambda cmd: subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    r = run(["ssh", "mscluster84", f"mkdir -p {remote}"])
    r = r if r.returncode else run(["rsync", "-az", "--delete", "--exclude", "build/", f"{paper}/", f"mscluster84:{remote}/"])
    if r.returncode:
        print(r.stderr); return r.returncode
    r = run(["scripts/cluster.sh", "sh", f"cd {remote} && {TECTONIC} --keep-logs --keep-intermediates "
             f"{TEX_NAME} > tectonic.out 2>&1; echo exit=$?; grep -h 'reviewslot .* shrunk' *.log || echo 'no slot shrunk'"])
    print(r.stdout.strip())
    build = paper / "build"
    build.mkdir(exist_ok=True)
    for name in ["iclr2027_conference.pdf", "iclr2027_conference.log", "iclr2027_conference.aux", "tectonic.out"]:
        run(["rsync", "-az", f"mscluster84:{remote}/{name}", str(build / name)])
    pdf = build / "iclr2027_conference.pdf"
    pages = run(["pdfinfo", str(pdf)]).stdout
    print(f"built {pdf}: " + (re.search(r"Pages:\s+(\d+)", pages).group(1) + " pages" if "Pages" in pages else "no PDF"))
    return 0 if "exit=0" in r.stdout else 1


def gallery(paper: Path) -> int:
    """One sheet of every slot's page from the last --compile build, labelled by slot and state."""
    build = paper / "build"
    aux, pdf = build / "iclr2027_conference.aux", build / "iclr2027_conference.pdf"
    if not aux.exists() or not pdf.exists():
        raise SystemExit(f"no build in {build}; run with --compile first")
    pages = dict(re.findall(r"\\newlabel\{reviewslot:(R\d+)\}\{\{[^}]*\}\{(\d+)\}", aux.read_text()))
    from PIL import Image, ImageDraw, ImageFont
    tex = (paper / TEX_NAME).read_text()
    tiles = []
    for s in slots():
        sid, page = s["id"], pages.get(s["id"])
        if not page:
            print(f"{sid}: no page in {aux.name}"); continue
        stem = build / f"gallery-{sid}"
        subprocess.run(["pdftoppm", "-f", page, "-l", page, "-r", "40", "-png", "-singlefile",
                        str(pdf), str(stem)], check=True)
        filled = any((paper / "slots").glob(f"{sid}.*"))
        fallback = re.search(r"\\reviewslot\[[^\]]+\]\{" + sid + r"\}", tex)
        state = "filled" if filled else "submitted image" if fallback else "placeholder"
        tiles.append((sid, page, state, Image.open(f"{stem}.png").convert("RGB")))
    if not tiles:
        return 1
    w, h = tiles[0][3].size
    per_row, top = 5, 26
    sheet = Image.new("RGB", (per_row * (w + 10) + 10, ((len(tiles) + per_row - 1) // per_row) * (h + top + 10) + 10), "white")
    d = ImageDraw.Draw(sheet)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
        f = ImageFont.load_default()
    for i, (sid, page, state, im) in enumerate(tiles):
        x, y = 10 + (i % per_row) * (w + 10), 10 + (i // per_row) * (h + top + 10)
        d.text((x, y), f"{sid} · page {page} · {state}", fill="black" if state == "filled" else "#666666", font=f)
        sheet.paste(im, (x, y + top))
    out = build / "slot-gallery.png"
    sheet.save(out)
    for t in build.glob("gallery-R*.png"):
        t.unlink()
    print(f"gallery: {len(tiles)} slots -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report whether every slot is wired; change nothing")
    ap.add_argument("--slot", help="refresh one slot, e.g. R1")
    ap.add_argument("--from", dest="source", type=Path, help="read this file instead of the slot's result file")
    ap.add_argument("--paper", type=Path, default=PAPER, help="paper folder to write slots into")
    ap.add_argument("--remote", default=REMOTE, help="folder on mscluster to build in")
    ap.add_argument("--compile", action="store_true", help="build on mscluster after refreshing")
    ap.add_argument("--gallery", action="store_true", help="after any compile, sheet every slot's page into build/slot-gallery.png")
    a = ap.parse_args()
    if a.check:
        return 0 if check(a.paper) else 1
    chosen = [s for s in slots() if not a.slot or s["id"] == a.slot]
    if not chosen:
        raise SystemExit(f"no slot {a.slot} on {SHEET}")
    if a.source and len(chosen) != 1:
        raise SystemExit("--from needs --slot")
    for s in chosen:
        print(refresh(s, a.paper, a.source))
    code = compile_remote(a.paper, a.remote) if a.compile else 0
    return gallery(a.paper) if a.gallery and not code else code


if __name__ == "__main__":
    sys.exit(main())
