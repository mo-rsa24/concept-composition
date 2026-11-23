"""Checks every quoted code block in walk-noisyclip.md against the file and lines it cites, and the released checkout's commit."""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = Path('/home/molef/.claude/jobs/bed4762f/tmp/code/noisyclip')
ROOTS = {'src/': CODE, 'co3_push.py': HERE.parents[3] / 'scripts/method_search'}
COMMIT = '016b91356a43e455542fd018bc16ea3b419779d2'

head = subprocess.run(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
assert head == COMMIT, f'checkout is at {head}'
text = (HERE / 'walk-noisyclip.md').read_text()
blocks = re.findall(r'<!-- cite: (\S+):(\d+)-(\d+) -->\n.*?\n```python\n(.*?)\n```', text, re.S)
for path, a, b, body in blocks:
    root = next(v for k, v in ROOTS.items() if path.startswith(k))
    lines = (root / path).read_text().split('\n')[int(a) - 1:int(b)]
    assert body.split('\n') == lines, f'{path}:{a}-{b} does not match'
print(f'{len(blocks)} quoted blocks match their files at commit {COMMIT[:7]}')
