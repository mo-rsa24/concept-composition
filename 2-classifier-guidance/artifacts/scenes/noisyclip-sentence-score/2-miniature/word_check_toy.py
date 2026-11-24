"""The text-only word check on a toy: a sentence vector built as a sum of one vector per word slot.

With a pure sum of word slots, averaging both word orders cancels the swap exactly, so w is zero. One interaction term,
which depends on which noun sits on which side of the relation, survives the averaging. Every number on the toy's
figures comes from the key this writes. The vectors are hand-set, have 3 numbers each and are not made unit length,
so the toy explains the cancellation and predicts nothing about CLIP. Run: python3 word_check_toy.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIRST = {"cat": np.array([2., 1, 0]), "dog": np.array([0., 1, 2])}       # the noun in the first slot
SECOND = {"cat": np.array([1., 0, 1]), "dog": np.array([0., 2, 1])}      # the noun in the second slot
RELATION = {"left": np.array([0., 0, 1]), "right": np.array([1., 0, 0])}
SIDE = {"left": 1.0, "right": -1.0}                                       # +1 when the first noun is on the left
NOUN = {"cat": np.array([1., 0, 0]), "dog": np.array([0., 1, 0])}        # what the interaction term reads of each noun
C = 0.5
SENTENCES = {"xLy": ("cat", "left", "dog"), "yRx": ("dog", "right", "cat"), "yLx": ("dog", "left", "cat"), "xRy": ("cat", "right", "dog")}
TEXT = {"xLy": "a cat to the left of a dog", "yRx": "a dog to the right of a cat", "yLx": "a dog to the left of a cat", "xRy": "a cat to the right of a dog"}


def sentence(first, rel, second, c):
    slots = FIRST[first] + RELATION[rel] + SECOND[second]
    interaction = c * SIDE[rel] * (NOUN[first] - NOUN[second])
    return slots, interaction, slots + interaction


def run(c):
    rows = {k: sentence(*v, c) for k, v in SENTENCES.items()}
    u = {k: r[2] for k, r in rows.items()}
    plus = u["xLy"] + u["yRx"]
    minus = u["yLx"] + u["xRy"]
    w = plus - minus
    swap = u["xLy"] - u["yLx"]
    return {"sentences": {k: {"text": TEXT[k], "slots": rows[k][0].tolist(), "interaction": rows[k][1].tolist(), "u": rows[k][2].tolist()} for k in rows},
            "same_meaning_sum": plus.tolist(), "swapped_sum": minus.tolist(), "w": w.tolist(), "w_length": round(float(np.linalg.norm(w)), 4),
            "swap": swap.tolist(), "swap_length": round(float(np.linalg.norm(swap)), 4),
            "ratio": round(float(np.linalg.norm(w) / (2 * np.linalg.norm(swap))), 4)}


plain, mixed = run(0.0), run(C)
assert plain["w_length"] == 0.0, "a pure sum of word slots must cancel"
expected = 4 * C * (NOUN["cat"] - NOUN["dog"])
assert np.allclose(mixed["w"], expected), "w must equal 4c times the noun difference"
real = json.loads(Path('/home/molef/PhD/concept-composition/artifacts/notes/progress/noise-text-critic/02-can-it-read/word-check/word-check.json').read_text())
cd = {m: {p: next(r for r in real[m][p] if r["pair"] == "cat, dog") for p in ("order", "bind")} for m in ("trained", "random_weights")}
key = {
    "note": "Toy with hand-set 3-number vectors, not made unit length. It explains why a sum of word slots cancels and predicts nothing about CLIP.",
    "slot_vectors": {"first": {k: v.tolist() for k, v in FIRST.items()}, "second": {k: v.tolist() for k, v in SECOND.items()},
                     "relation": {k: v.tolist() for k, v in RELATION.items()}, "noun_read": {k: v.tolist() for k, v in NOUN.items()}},
    "interaction_weight": C,
    "sum_of_slots": plain,
    "with_interaction": mixed,
    "w_formula_check": "w = 4 c (n_cat - n_dog) = " + str(expected.tolist()),
    "real_cat_dog": {m: {p: round(cd[m][p]["ratio"], 2) for p in cd[m]} for m in cd},
    "real_mean": {m: {"order": round(real[m]["order_mean_ratio"], 2), "bind": round(real[m]["bind_mean_ratio"], 2)} for m in ("trained", "random_weights")},
}
(HERE / "word_check_toy_key.json").write_text(json.dumps(key, indent=1))
print(json.dumps({k: key[k] for k in ("real_cat_dog", "real_mean")}))
print("plain w", plain["w"], "swap", plain["swap"], "ratio", plain["ratio"])
print("mixed w", mixed["w"], "swap", mixed["swap"], "ratio", mixed["ratio"])
for k, s in mixed["sentences"].items():
    print(k, s["slots"], s["interaction"], s["u"])
