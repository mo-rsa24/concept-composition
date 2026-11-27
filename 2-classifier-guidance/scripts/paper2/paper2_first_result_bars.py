#!/usr/bin/env python3
"""Paper 2's first result (the identity dial alone on CO3): its bars as named constants, and the rule that reads them.

The constants are plans/paper-2-dials/plans/03-identity-dial-first-result.md, "The bars", line for line. The rule is the
review file's, plans/paper-2-dials/review/03-identity-dial-first-result.md, "The question written before the run". A value
exactly at a bar meets it. Committed in the plan's instruction 1.1, after step 2's CO3 redraw and before any dial trains,
so no bar can move after an answer is seen without showing in a diff. Every bar is read on the 48 matched test pictures
per row, called blind, except the redraw bar and the gaming alarm, which are read on the 32 tuning pictures.

The judge the dial trains through (the trained species judge, by the owner's choice of 2026-11-27) is a steering signal
only. Nothing in this file reads a judge's call on a test picture: the verdict comes from the settled blind calls.

Needs no torch, so it runs on the laptop.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------------------------------------------------
# The bars (03-identity-dial-first-result.md, "The bars")

FIRST_TWO_CLEAR_LEAD_MIN = 0.20         # the dial's two-clear share minus CO3's, on the 48 matched pictures, for a yes
FIRST_MCNEMAR_P_MAX = 0.05              # the one-sided McNemar p of that lead must be below this for a yes
FIRST_BORROWED_MUST_FALL = True         # the dial's borrowed-part share must be below CO3's for a yes
FIRST_RANDOM_DIAL_CLOSE = 0.05          # the random dial's two-clear share within this of CO3's for a yes
FIRST_NO_LEAD_MAX = 0.05                # a two-clear lead at or below this is a no
FIRST_GAMING_GAP_MAX_RISE = 0.10        # the gradient judge's two-clear share minus the runner-up judge's, on the 32
                                        # tuning pictures, may rise by no more than this from CO3's first draw to either
                                        # checkpoint
FIRST_REDRAW_KIND_CHANGES_MAX = 2       # of the 32 tuning pictures CO3 draws twice in step 2, at most this many change
                                        # kind by the kept judge
FIRST_GPU_HOURS_STOP = 24               # measured rates projecting past this many GPU-hours stop the plan for the owner

MATCHED_PICTURES = 48                   # 16 look-alike test pairs at seeds 13,001 to 13,003, per row
TUNING_PICTURES = 32                    # 4 tuning pairs at seeds 12,001 to 12,008
EPS = 1e-12                             # shares are fractions of 48; a value exactly at a bar meets it


# ---------------------------------------------------------------------------------------------------------------------
# The rule


def one_sided_mcnemar_p(wins: int, losses: int) -> float:
    """P(X >= wins) for X ~ Binomial(wins + losses, 1/2): the one-sided exact McNemar p, counting only the matched
    pictures where the two rows disagree (wins: the dial two clear and CO3 not; losses: the reverse)."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(wins, n + 1)) / 2 ** n


def answer_and_reason(lead: float, p: float, dial_borrowed: float, co3_borrowed: float,
                      random_minus_co3: float) -> tuple[str, str]:
    """(answer, reason) by the review file's rule.

    lead               the dial's two-clear share minus CO3's, on the 48 matched test pictures
    p                  the one-sided McNemar p of that lead
    dial_borrowed      the dial's borrowed-part share; co3_borrowed CO3's
    random_minus_co3   the random dial's two-clear share minus CO3's
    """
    borrowed_fell = dial_borrowed < co3_borrowed if FIRST_BORROWED_MUST_FALL else True
    random_close = abs(random_minus_co3) <= FIRST_RANDOM_DIAL_CLOSE + EPS
    if lead >= FIRST_TWO_CLEAR_LEAD_MIN - EPS and p < FIRST_MCNEMAR_P_MAX and borrowed_fell and random_close:
        return "yes", (f"lead {lead:.4f} at least {FIRST_TWO_CLEAR_LEAD_MIN}, p {p:.4f} below {FIRST_MCNEMAR_P_MAX}, "
                       f"borrowed parts fell, random dial within {FIRST_RANDOM_DIAL_CLOSE}")
    if lead <= FIRST_NO_LEAD_MAX + EPS:
        return "no", f"lead {lead:.4f} at or below {FIRST_NO_LEAD_MAX}"
    missed = []
    if lead < FIRST_TWO_CLEAR_LEAD_MIN - EPS:
        missed.append(f"lead {lead:.4f} below {FIRST_TWO_CLEAR_LEAD_MIN}")
    if p >= FIRST_MCNEMAR_P_MAX:
        missed.append(f"p {p:.4f} at or above {FIRST_MCNEMAR_P_MAX}")
    if not borrowed_fell:
        missed.append(f"borrowed parts did not fall ({dial_borrowed:.4f} against CO3's {co3_borrowed:.4f})")
    if not random_close:
        missed.append(f"random dial moved {random_minus_co3:+.4f}, outside {FIRST_RANDOM_DIAL_CLOSE}")
    return "inconclusive", "; ".join(missed)


def answer(lead, p, dial_borrowed, co3_borrowed, random_minus_co3) -> str:
    """yes, no or inconclusive (see answer_and_reason)."""
    return answer_and_reason(lead, p, dial_borrowed, co3_borrowed, random_minus_co3)[0]


def gaming_alarm(gap_co3: float, gap_checkpoint: float) -> bool:
    """True when the gaming gap (the gradient judge's two-clear share minus the runner-up judge's, on the 32 tuning
    pictures) rose by more than FIRST_GAMING_GAP_MAX_RISE from CO3's first draw to this checkpoint: step 5 stops."""
    return gap_checkpoint - gap_co3 > FIRST_GAMING_GAP_MAX_RISE + EPS


def redraw_band_needed(kind_changes: int, n: int = TUNING_PICTURES) -> float | None:
    """Instruction 1.1.0: None when CO3's redraw changed kind on at most FIRST_REDRAW_KIND_CHANGES_MAX of the n tuning
    pictures; otherwise the smallest FIRST_RANDOM_DIAL_CLOSE the owner may set before committing, the measured share."""
    if kind_changes <= FIRST_REDRAW_KIND_CHANGES_MAX:
        return None
    return kind_changes / n


def projects_past_stop(gpu_hours: float) -> bool:
    """True when measured rates project this plan past FIRST_GPU_HOURS_STOP GPU-hours."""
    return gpu_hours > FIRST_GPU_HOURS_STOP


if __name__ == "__main__":
    for name in ("FIRST_TWO_CLEAR_LEAD_MIN", "FIRST_MCNEMAR_P_MAX", "FIRST_BORROWED_MUST_FALL", "FIRST_RANDOM_DIAL_CLOSE",
                 "FIRST_NO_LEAD_MAX", "FIRST_GAMING_GAP_MAX_RISE", "FIRST_REDRAW_KIND_CHANGES_MAX",
                 "FIRST_GPU_HOURS_STOP"):
        print(f"{name} = {globals()[name]}")
