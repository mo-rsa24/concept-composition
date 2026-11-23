"""Write page-marked text for each paper, in reading order and in layout order."""
import subprocess, sys, pathlib
root = pathlib.Path(sys.argv[1])
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    for mode, name, args in (("read", "paged.txt", []), ("layout", "paged-layout.txt", ["-layout"])):
        raw = subprocess.run(["pdftotext", *args, str(d / "paper.pdf"), "-"], capture_output=True).stdout.decode("utf8", "replace")
        pages = raw.split("\f")
        if pages and not pages[-1].strip():
            pages = pages[:-1]
        (d / name).write_text("".join(f"\n=== PAGE {i} ===\n{t}" for i, t in enumerate(pages, 1)))
    print(d.name, len(pages))
