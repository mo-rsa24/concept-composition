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

REPO = Path(__file__).resolve().parents[2]
SHEET = REPO / "report/paper-review/layout-sheet.md"
PAPER = REPO / "paper/iclr-56173"
TEX_NAME = "iclr2027_conference.tex"
REGISTER_NAME = "figures.md"
# Built outside the cluster checkout: copying the paper into the checkout leaves untracked copies
# of tracked files there, and the next scripts/cluster.sh sync cannot fast-forward past them.
REMOTE = "/datasets/mmolefe/poe_repair_min/outputs/paper-build/iclr-56173"
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


# A slot that shows one model's rows of a shared result file (the layout sheet says which).
MODEL_OF_SLOT = {"R1": "SDXL", "R5": "SD 1.5"}
# Slots that are lists, not scores: printed column for column, no axis columns required.
LIST_SLOTS = {"R7"}


GROUP_ORDER = [("hard", "Hard"), ("easy", "Easy but unseen"), ("not competing", "Not competing"), ("showcase", "Showcase")]


def csv_to_grouped_list(path: Path) -> str:
    """R7: one run-in paragraph per group, each pair with its joint-prompt count, * for held out."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for key, title in GROUP_ORDER:
        sel = [r for r in rows if (r["Group"] or r["Set"]) == key]
        if not sel:
            continue
        items = []
        for r in sel:
            name = tex_escape(r["Pair"]).replace("×", r"$\times$") + ("$^*$" if r["Set"] == "held out" else "")
            if r["Joint"] not in ("", "–"):
                name += f" ({r['Joint']})"
            items.append(name)
        out.append(r"\noindent\textbf{" + f"{title} ({len(sel)})." + "} " + ", ".join(items) + ".\\par\\smallskip")
    return "\n".join(out) + "\n"


def csv_to_list(path: Path) -> str:
    """Every column of the CSV as a plain booktabs table at natural width, rows as written."""
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    def numeric(i: int) -> bool:
        vals = [r[i] for r in body if r[i] not in ("", "–", "-")]
        return bool(vals) and all(v.replace(".", "", 1).isdigit() for v in vals)
    align = "".join("r" if numeric(i) else "l" for i in range(len(head)))
    lines = [r"\setbox0=\hbox{\setlength{\tabcolsep}{5pt}%",
             r"\begin{tabular}{@{}" + align + "@{}}", r"\toprule",
             " & ".join(tex_escape(h) for h in head) + r" \\", r"\midrule"]
    lines += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}}%",
              r"\ifdim\wd0>\linewidth\reviewoverflow{list}{\the\wd0}\fi",
              r"\centerline{\box0}", ""]
    return "\n".join(lines)


# One layout per table slot: column groups (the CSV's `group` value, in order, with its printed
# title) and the metrics under each, then columns that close the row. A metric is a CSV column; a
# metric with a `<name>_ci` column beside it prints as "value ± interval". Each table is set at
# natural width with booktabs rules and siunitx columns, and never scaled: a table wider than the
# text is reported and, in a final build, stops the build (\ifreviewfinal in the tex).
METRIC = {  # CSV column: (printed label, decimals)
    "comp": ("Comp.", 2), "clip": ("CLIP", 1), "blip": ("BLIP", 2), "ir": ("IR", 2),
    "geneval_two": ("Two obj.", 2), "geneval_count": ("Counting", 2),
    "geneval_colorattr": ("Attr. bind.", 2), "geneval_position": ("Position", 2),
    "t2i_color": ("Colour", 2), "t2i_shape": ("Shape", 2), "t2i_texture": ("Texture", 2),
    "t2i_spatial": ("Spatial", 2), "t2i_complex": ("Complex", 2), "nfe": ("NFE", 0),
}
BENCH_GROUPS = [("animals", "Animals", ["comp", "clip", "blip", "ir"]),
                ("animals_objects", "Animals + objects", ["clip", "blip", "ir"]),
                ("objects", "Objects", ["clip", "blip", "ir"])]
SCHEMA = {
    "R1": {"groups": BENCH_GROUPS, "tail": []},
    "R5": {"groups": BENCH_GROUPS, "tail": []},
    "R3": {"groups": [("all", "53 held-out pairs", ["blip"])], "tail": []},
    "R14": {"groups": [("geneval", "GenEval", ["geneval_two", "geneval_colorattr", "geneval_position"]),
                       ("t2icompbench", "T2I-CompBench", ["t2i_color", "t2i_shape"])], "tail": []},
}
# A method whose name ends in this is a published number we cite, never a run of ours; it sits
# under its own rule at the foot of the table.
REPORTED = "(reported)"
# Methods that evaluate the joint prompt at every step. They sit above a rule as references; bold
# marks the best value in each column over every row.
REFERENCE = ("Joint prompt", "CO3", "ToMe")
# Methods kept out of the tables: the Langevin corrector appears in Figure 5 only.
EXCLUDED = ("PoE + Langevin",)


def fmt(value: str, decimals: int) -> str:
    if value in ("", None):
        return "{--}"
    v = float(value)
    return f"{v:.{decimals}f}" if decimals else f"{round(v):d}"


def csv_to_tabular(path: Path, sid: str) -> str:
    """Pivot a long CSV (one row per method and group) into the slot's grouped booktabs table."""
    if sid not in SCHEMA:
        raise SystemExit(f"{sid}: no table layout in SCHEMA; add one before filling this slot")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if sid in MODEL_OF_SLOT:
        rows = [r for r in rows if r["model"] == MODEL_OF_SLOT[sid]]
        if not rows:
            raise SystemExit(f"{sid}: {path} has no rows with model {MODEL_OF_SLOT[sid]}")
    sch = SCHEMA[sid]
    rows = [r for r in rows if r["method"] not in EXCLUDED]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    run = [m for m in methods if not m.endswith(REPORTED)]
    refs = [m for m in REFERENCE if m in run]
    rest = [m for m in run if m not in refs]
    ours = refs + rest + [m for m in methods if m.endswith(REPORTED)]
    cell = {(r["method"], r["group"]): r for r in rows}
    best = {}  # (group, metric) -> highest value over every row; all metrics here are higher-is-better
    if rows:
        for key, _, metrics in sch["groups"]:
            for m in metrics:
                vals = [float(cell[(x, key)][m]) for x in refs + rest if cell.get((x, key), {}).get(m) not in ("", None)]
                if vals:
                    best[(key, m)] = max(vals)
    spec, head1, head2, rules, col = ["l"], ["Method"], [""], [], 2
    for key, title, metrics in sch["groups"]:
        for m in metrics:
            has_ci = any(r.get(f"{m}_ci") for r in rows if r["group"] == key)
            d = METRIC[m][1]
            whole = 2 if m == "clip" else 1
            sign = "-" if m == "ir" else ""
            spec.append(f"S[table-format={sign}{whole}.{d}" + (f"({d})" if has_ci else "") + "]")
            head2.append("{" + METRIC[m][0] + r" $\uparrow$}")
        head1.append(r"\multicolumn{" + str(len(metrics)) + "}{c}{" + tex_escape(title) + (r" $\uparrow$" if len(metrics) == 1 else "") + "}")
        rules.append(r"\cmidrule(lr){" + f"{col}-{col + len(metrics) - 1}" + "}")
        col += len(metrics)
    for m in sch["tail"]:
        spec.append("S[table-format=3.0]")
        head1.append("")
        head2.append("{" + METRIC[m][0] + "}")
    one_metric = all(len(m) == 1 for _, _, m in sch["groups"]) and not sch["tail"]
    header = ([" & ".join(head1) + r" \\", r"\midrule"] if one_metric else
              [" & ".join(head1) + r" \\", " ".join(rules), " & ".join(head2) + r" \\", r"\midrule"])
    lines = [r"\setbox0=\hbox{\setlength{\tabcolsep}{2.1pt}\sisetup{separate-uncertainty=true,detect-weight=true,mode=match}%",
             r"\begin{tabular}{@{}" + "".join(spec) + "@{}}", r"\toprule"] + header
    for i, meth in enumerate(ours):
        if meth.endswith(REPORTED) and (i == 0 or not ours[i - 1].endswith(REPORTED)):
            lines.append(r"\midrule")
        if refs and i == len(refs):
            lines.append(r"\midrule")
        vals = [tex_escape(meth)]
        for key, _, metrics in sch["groups"]:
            r = cell.get((meth, key), {})
            for m in metrics:
                v = fmt(r.get(m, ""), METRIC[m][1])
                ci = r.get(f"{m}_ci", "")
                txt = v if not ci or v == "{--}" else f"{v}({round(float(ci) * 10 ** METRIC[m][1]):d})"
                if v != "{--}" and (key, m) in best and abs(float(r[m]) - best[(key, m)]) < 1e-12:
                    d = METRIC[m][1]
                    txt = ("{\\boldmath$" + v + (r"\pm" + f"{float(ci):.{d}f}" if ci else "") + "$}")
                vals.append(txt)
        for m in sch["tail"]:
            any_row = next((r for (mm, _), r in cell.items() if mm == meth and r.get(m)), {})
            vals.append(fmt(any_row.get(m, ""), METRIC[m][1]))
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}%",
              r"\ifdim\wd0>\linewidth\reviewoverflow{" + sid + r"}{\the\wd0}\fi",
              r"\centerline{\box0}", ""]
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
    if ext == ".csv" and slot["id"] in LIST_SLOTS:
        (out_dir / f"{slot['id']}.tex").write_text(csv_to_grouped_list(src) if slot["id"] == "R7" else csv_to_list(src))
        return f"{slot['id']}: list from {src} -> {out_dir / (slot['id'] + '.tex')}"
    if ext == ".csv":
        gaps = missing_axes(src) if src.name != "placeholder" else []
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
            if res.suffix.lower() == ".csv" and sid not in LIST_SLOTS and missing_axes(res):
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
             f"{TEX_NAME} > tectonic.out 2>&1; echo exit=$?; grep -h 'reviewslot .* too wide' *.log || echo 'no slot too wide'"])
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
