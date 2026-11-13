"""The exact-count sweep of toy_caption_threshold.py with the training pictures' pixel noise set to
zero: every copy of a picture is then the identical clean picture. Counts 0, 200 and 2,000
relabelled, three training runs each, everything else unchanged.

Written before the run: the noise does not matter for this result if each no-noise count falls
within the spread of the noisy runs at the same number (0 to 3 of 300 at 0; 0 to 153 at 200; 74 to
252 at 2,000)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_caption_dial_ldm as m
import toy_caption_threshold as th

m.t.SIGMA0 = 0.0                       # corpus_k reads this at call time
th.COUNTS = [0, 200, 2000]
th.TAG = "-no-noise"
th.main()
