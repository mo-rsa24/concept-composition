"""The "and" vector sheet: the references, then the two ways of building the vector as a third expert.

x is the global build (the repo's code calls it x_fix): E(joint) - E(a) - E(b) + E(""), which cancels the
start token everywhere and leaves the pads and the word positions unlike any real prompt's. x_pos is the
positional build: E(joint) with each single prompt's word subtracted only at the position holding that
same word, so the start token, the end token and the pads stay the joint prompt's.

Reuses the tiles scripts/and_vector_sheet.py rendered; only the columns and their labels change.

    python3 scripts/and_vector_x_sheet.py <tiles_dir> <out.png>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from and_vector_sheet import sheet  # noqa: E402

COLUMNS = [
    ("a", ["{a}", "alone"], []),
    ("b", ["{b}", "alone"], []),
    ("joint", ["Joint prompt", "{joint}"], []),
    ("poe", ["PoE", "{a}, {b}"], []),
    ("x_fix", ["x alone", "subtracted everywhere"], []),
    ("poe_x_fix", ["PoE", "{a}, {b}, x"], []),
    ("x_pos", ["x_pos alone", "subtracted in place"], []),
    ("poe_x_pos", ["PoE", "{a}, {b}, x_pos"], []),
]

if __name__ == "__main__":
    tiles, out = Path(sys.argv[1]), Path(sys.argv[2])
    sheet(COLUMNS, 'A third expert beside "dog" and "cat": subtracted everywhere, then subtracted in place',
          tiles, out, "dog", "cat", "dog and cat", list(range(1, 9)))
