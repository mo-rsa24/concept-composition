"""Write the miniature page as one file: page_template.html with the key, the typeset equations and KaTeX's CSS inlined.

Every equation is typeset at build time by KaTeX 0.16.11 (render_math.mjs, from ./node_modules, git-ignored; install it with
`npm install --no-save --no-package-lock katex@0.16.11` in this folder). KaTeX's CSS goes in with its woff2 fonts as data
URIs, so the page fetches nothing to show its mathematics.

Run after make_page_key.py:  PYTHONDONTWRITEBYTECODE=1 ~/.claude/venvs/figures/bin/python -I build_page.py
"""
import base64
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
KATEX = HERE / "node_modules" / "katex" / "dist"


def katex_css():
    css = (KATEX / "katex.min.css").read_text()

    def woff2_only(m):
        name = m.group(1)
        data = base64.b64encode((KATEX / "fonts" / f"{name}.woff2").read_bytes()).decode()
        return f'src:url(data:font/woff2;base64,{data}) format("woff2")'
    css, n = re.subn(r'src:url\(fonts/([A-Za-z0-9_-]+)\.woff2\) format\("woff2"\),url\(fonts/[^)]+\.woff\) format\("woff"\),'
                     r'url\(fonts/[^)]+\.ttf\) format\("truetype"\)', woff2_only, css)
    assert n == 20 and "url(fonts/" not in css, n
    return css


def main():
    key = (HERE / "page_key.json").read_text()
    eq = subprocess.run(["node", str(HERE / "render_math.mjs"), str(HERE / "equations.json")],
                        check=True, capture_output=True, text=True).stdout
    n_eq = len(json.loads(eq))
    css = katex_css()
    for blob in (key, eq):
        assert "</script" not in blob
    assert "</style" not in css
    page = (HERE / "page_template.html").read_text()
    for marker, value in (("/*PAGE_KEY*/", key), ("/*EQUATIONS*/", eq), ("/*KATEX_CSS*/", css)):
        assert page.count(marker) == 1, marker
        page = page.replace(marker, value, 1)
    (HERE / "digit-map-checkers-miniature.html").write_text(page)
    print(f"wrote digit-map-checkers-miniature.html, {len(page.encode()) / 1024:.0f} KB; {n_eq} equations typeset by KaTeX; "
          f"KaTeX CSS with 20 fonts inlined, {len(css) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
