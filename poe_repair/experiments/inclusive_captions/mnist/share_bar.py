"""The MNIST share run's bar, written before any share model trains (plan 03 of scope 11).

The share s is the probability that a training draw labelled with a single digit d shows d beside a
second digit (8 for "7", 7 for "8"). Counts are out of SEEDS seeds, read by the pinned judge.
Change nothing here after a result is in without saying so in the diff.
"""

SEEDS = 256
SHARES = (0.0, 0.1, 0.2, 0.4, 0.8)
JUDGED = (0.1, 0.2, 0.4, 0.8)          # shares the verdict may be read at; 0.0 is the baseline


def purity_floor(s):
    """A share counts only if "7" alone and "8" alone each draw their lone digit at least this often,
    in all three trainings: 95% of what training implies, round(0.95 * SEEDS * (1 - s))."""
    return round(0.95 * SEEDS * (1 - s))


PURITY_FLOOR = {s: purity_floor(s) for s in SHARES}
assert PURITY_FLOOR == {0.0: 243, 0.1: 219, 0.2: 195, 0.4: 146, 0.8: 49}, PURITY_FLOOR

# The verdict, read off the corrected 7 x 8 product's "two separate right digits".
SUPPORT_MIN = 128          # at a counting share in JUDGED, in all three trainings
SHARE0_MAX = 12            # 0% must stay at or under this, in all three trainings
NULL_MAX_MEAN = 39         # null when no counting share's three-training mean reaches this

# The judge's reading holds at a share only if, on one 64-tile sheet of training 1's corrected 7 x 8
# product, it agrees with blind eye calls on at least JUDGE_EYE_MIN of EYE_SEEDS. Otherwise that share is
# read by eye on the first EYE_SEEDS seeds of each training, with the thresholds scaled to 64.
EYE_SEEDS = 64
JUDGE_EYE_MIN = 58
EYE_SUPPORT_MIN = 32
EYE_SHARE0_MAX = 3
EYE_NULL_MAX_MEAN = 9.75
