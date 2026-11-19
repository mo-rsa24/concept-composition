"""The MNIST stage's bar, written before any model draws (route step 4, round 8 of the route walk).

Counts are out of SEEDS seeds. Change nothing here after a result is in without saying so in the diff.
"""

SEEDS = 256
DIALS = (0.0, 0.1, 0.3, 0.6, 1.0)

# A dial setting counts only if both single labels draw their own digit alone at least this often,
# in all three trainings: each label's training line minus 5 points, rounded up to a whole seed.
PURITY_FLOOR = {
    "7": {0.0: 244, 0.1: 231, 0.3: 217, 0.6: 207, 1.0: 199},
    "8": {0.0: 244, 0.1: 230, 0.3: 214, 0.6: 203, 1.0: 196},
}

# The verdict, read off the corrected 7 x 8 product's "two separate right digits".
SUPPORT_MIN = 128          # at a counting setting, in all three trainings
DIAL0_MAX = 12             # dial 0 must stay at or under this, in all three trainings
NULL_MAX_MEAN = 39         # null when no counting setting's three-training average reaches this

# The verdict reads the corrected 7 x 8 product by eye: blind full-size calls on the first EYE_SEEDS
# seeds of each training, settled by the user, with the judge's count on all SEEDS reported beside it.
# The shares are the ones above: 128 of 256 is 32 of 64, 39 of 256 is 9.75 of 64, 12 of 256 is 3 of 64.
# The judge could not read this row (dial 0.6, training 1: 42 of 64 agreements current, 54 ten-class).
EYE_SEEDS = 64
EYE_SUPPORT_MIN = 32
EYE_DIAL0_MAX = 3
EYE_NULL_MAX_MEAN = 9.75

# The stop: the plain 7 x 8 product at dial 0 above this means dial 0 already composes.
STOP_PLAIN_DIAL0 = 12

# The two side readings.
CORRECTOR_NOT_NEEDED_MIN = 128   # plain product at the supported setting, all three trainings
UNSEEN_MIN = 39                  # corrected 4 x 7 at the supported setting, all three trainings
UNSEEN_FACTOR = 3                # and at least this many times its own dial-0 count

# The judge's two checks, before any count is used.
JUDGE_PASTED_MIN = 0.95          # share right on 500 pasted canvases per call
JUDGE_EYE_MIN = 58               # agreements with eye calls, of 64, per dial sheet
