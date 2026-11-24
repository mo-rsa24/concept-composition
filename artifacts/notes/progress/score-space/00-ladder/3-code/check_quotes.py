"""Check every quoted block in walk-the-digit-environment.md against its file at the pinned commit.

A block is a `<!-- quote <path> <first>-<last> -->` line followed by a fenced code block. Run from this folder:
    python3 check_quotes.py
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMMIT = '2dcb2bebf268aa7f97783615b23cb17559182976'
text = (HERE / 'walk-the-digit-environment.md').read_text()
blocks = re.findall(r'<!-- quote (\S+) (\d+)-(\d+) -->\n```python\n(.*?)```', text, re.S)
bad = 0
for path, a, b, body in blocks:
    src = subprocess.run(['git', '-C', str(REPO), 'show', f'{COMMIT}:{path}'], capture_output=True, text=True, check=True).stdout
    want = '\n'.join(src.splitlines()[int(a) - 1:int(b)]) + '\n'
    if want != body:
        bad += 1
        print(f'MISMATCH {path} {a}-{b}')
print(f'{len(blocks)} quotes checked against {COMMIT[:8]}, {bad} mismatches')
sys.exit(1 if bad or not blocks else 0)
