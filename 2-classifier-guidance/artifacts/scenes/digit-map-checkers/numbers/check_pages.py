"""Check that every relative link and thumbnail on the ladder's markdown pages resolves on disk.

Run: ~/.claude/venvs/figures/bin/python -I check_pages.py   (prints "0 broken" when all resolve)
"""
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r'\]\(([^)\s]+)\)|(?:href|src)="([^"]+)"')
broken, seen = [], 0
for page in sorted(ROOT.rglob("*.md")):
    if "env" in page.parts:
        continue
    text = re.sub(r"```.*?```", "", page.read_text(), flags=re.S)
    for m in LINK.finditer(text):
        target = m.group(1) or m.group(2)
        if target.startswith(("http", "#", "mailto:")):
            continue
        path = unquote(target.split("#")[0])
        if not path:
            continue
        seen += 1
        if not (page.parent / path).exists():
            broken.append(f"{page.relative_to(ROOT)}: {target}")
print(f"{seen} links checked, {len(broken)} broken")
for b in broken:
    print("  " + b)
print("0 broken" if not broken else "")
