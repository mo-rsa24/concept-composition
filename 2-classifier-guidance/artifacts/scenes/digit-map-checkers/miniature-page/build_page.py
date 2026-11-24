"""Write the miniature page: page_template.html with page_key.json inlined, so the page opens as one file.

Run after make_page_key.py:  PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I build_page.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
key = (HERE / "page_key.json").read_text()
assert "</script" not in key
page = (HERE / "page_template.html").read_text().replace("/*PAGE_KEY*/", key, 1)
assert "/*PAGE_KEY*/" not in page
(HERE / "digit-map-checkers-miniature.html").write_text(page)
print(f"wrote digit-map-checkers-miniature.html, {len(page) / 1024:.0f} KB")
