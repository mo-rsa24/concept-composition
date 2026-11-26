"""The seven rules: each body's call and flag in, one kind for the picture out.

Every judge and every reviewer sheet uses this one function, so a picture's kind means the same
thing wherever it is counted. The rules and their order are copied from the species judge plan
(plans/paper-2-dials/plans/02-species-judge.md, "What happens"); the first rule that fits wins.

Each body is called A (the first asked species), B (the second), MIXED (both at once) or OTHER
(neither), and separately flagged when it carries a visible part of the other asked species.

    1. 3 or more bodies, with at least one plain A and one plain B    -> extra subject
    2. any body called other                                          -> wrong species
    3. exactly 2 bodies, one A and one B, either flagged              -> borrowed part
    4. exactly 2 bodies, one A and one B, neither flagged             -> two clear
    5. some body mixed, and no clear body of one asked species        -> hybrid
    6. 2 or more bodies, all A, or all B                              -> two of one kind
    7. anything else (one plain body, no body)                        -> other

Two words the plan leaves open are read this way, the same in rules 1 and 5: a "plain" or
"clear" body of a species is a body called that species (A or B), whatever its flag. A flag
marks a borrowed part on a body that is still that species; only MIXED takes a body out.

paper2_bars.py (species judge step 0.2) imports KINDS and kind_of from here.
"""
from __future__ import annotations

from typing import Iterable, Sequence

A, B, MIXED, OTHER = "A", "B", "mixed", "other"
CALLS = (A, B, MIXED, OTHER)

TWO_CLEAR = "two_clear"
BORROWED_PART = "borrowed_part"
HYBRID = "hybrid"
TWO_OF_ONE_KIND = "two_of_one_kind"
WRONG_SPECIES = "wrong_species"
EXTRA_SUBJECT = "extra_subject"
OTHER_KIND = "other"

# The six kinds a picture is counted in, then "other", which is called and kept but in no count.
KINDS = (TWO_CLEAR, BORROWED_PART, HYBRID, TWO_OF_ONE_KIND, WRONG_SPECIES, EXTRA_SUBJECT)
ALL_KINDS = KINDS + (OTHER_KIND,)


def kind_of(bodies: Iterable[Sequence]) -> tuple[str, int]:
    """Return (kind, rule number) for a picture from its bodies' (call, flag) pairs.

    `bodies` holds one (call, flag) per body found, call in CALLS and flag a bool. The rule
    number is returned beside the kind so a sheet can show which rule decided.
    """
    bodies = [(c, bool(f)) for c, f in bodies]
    for c, _ in bodies:
        if c not in CALLS:
            raise ValueError(f"unknown call {c!r}; expected one of {CALLS}")
    n = len(bodies)
    calls = [c for c, _ in bodies]
    n_a, n_b = calls.count(A), calls.count(B)

    if n >= 3 and n_a >= 1 and n_b >= 1:
        return EXTRA_SUBJECT, 1
    if OTHER in calls:
        return WRONG_SPECIES, 2
    if n == 2 and n_a == 1 and n_b == 1:
        if any(f for _, f in bodies):
            return BORROWED_PART, 3
        return TWO_CLEAR, 4
    if MIXED in calls and (n_a == 0 or n_b == 0):
        return HYBRID, 5
    if n >= 2 and (n_a == n or n_b == n):
        return TWO_OF_ONE_KIND, 6
    return OTHER_KIND, 7
