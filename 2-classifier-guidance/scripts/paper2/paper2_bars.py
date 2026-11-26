"""The species judge's bars, written before any picture is called (species judge step 0.2).

Every constant in the bars table of plans/paper-2-dials/plans/02-species-judge.md ("The bars"), at its value, so no
bar can move after a number arrives without showing in a diff; the seven rules (from paper2_kinds.py, so the judges and
the scoring read one copy); the judges' grids (from the judge scripts); the taxonomic order of every species and the
look-alike rule; the plan's pairs with their types; eight new look-alike pairs for the conditional render; the
renaming table; the old judge's mapping to the six kinds; the "neither" names; and the ImageNet class table.

The ImageNet class table (IMAGENET_CLASSES) maps a species to the ImageNet-1k classes that are that animal or a kind
of it (a breed, a sex, a species of it), never to a broader class or a look-alike: "lynx" is not a bobcat, "vulture"
is not a condor, "box turtle" is not a tortoise, "tiger cat" is a house cat, "Madagascar cat" is a lemur. Each index's
name is kept beside it in IMAGENET_CLASS_NAMES, checked against torchvision's list of the 1,000 classes by
tests/test_paper2_bars.py. A species with no class leaves its pair out of the ImageNet judge's count;
`python3 paper2_bars.py` prints the coverage.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper2_kinds import ALL_KINDS, KINDS, kind_of  # noqa: E402,F401  (the seven rules, one copy)
from paper2_common import neither_names  # noqa: E402,F401  (the 43 names of the pooled dial's pair file)
from paper2_judge_clip import FLAG_SHARE_GRID, MARGIN_GRID  # noqa: E402,F401
from paper2_judge_dac import BORROW_GAP_GRID, N_REFS_GRID  # noqa: E402,F401
from paper2_judge_vlm import YES_THRESHOLD_GRID  # noqa: E402,F401

# ---------------------------------------------------------------- the bars (the plan's table, in its order)

REVIEWER_AGREE_KIND_MIN = 0.80
CLEAR_VS_BORROWED_AGREE_MIN = 0.85
OWNER_KIND_MIN = 8            # of 10
OWNER_ALL_MIN = 54            # of 60
OWNER_ABSOLUTE_MIN = 57       # of 60
REVIEWER_GAP_MAX = 0.05
REVIEWER_GAP_KIND_MAX = 0.10
AGREE_ALL_MIN = 216           # of 240
AGREE_KIND_MIN = 33           # of 40
LOOKALIKE_PER_KIND_MIN = 20   # of 40
TWO_CLEAR_PRECISION_MIN = 0.90
WITHIN_PAIR_AUC_MIN = 0.85
AUC_PAIRS_MIN = 4
DROP_PAIR_HOLDS = True
CONFIRM_MIN = 51              # of 60
CROP_CHECK_MIN = 62           # of 64
GPU_HOURS_STOP = 4.0

# The counts the bars are read against (the plan's "The size" table).
PER_KIND_CHOOSING_MIN = 15
PER_KIND_TEST_READ = 40
PER_KIND_TEST_PAIRS_MIN = 4
PER_KIND_ONE_PAIR_MAX = 12
PER_KIND_CONFIRM = 10
UNSELECTED_TEST = 100
OWNER_SAMPLE = 60
AUC_PAIR_MIN_EACH = 3          # two-clear and other pictures a pair needs to enter the within-pair AUC
SINGLE_SPECIES_TELL_MIN = 0.90  # below this on one species' own pictures: "cannot tell at all", no bar

# Seeds fixed before any judge runs.
SPLIT_SEED = 0
CONFIRM_SEED = 1
UNSELECTED_SEED = 2

# The middle of each judge's grid, where step 0.15 reads the 64 cats.
CROP_CHECK_SETTINGS = {"margin": 0.15, "flag_share": 0.25, "yes_threshold": 0.5, "borrow_gap": 0.2, "n_refs": 8}

# ---------------------------------------------------------------- species, orders and pairs

# The taxonomic order of every species the plan names (its pairs, the renaming table, the new look-alike pairs, the
# "neither" names and the closeness prompts). Whales are written as Cetacea, the order most references still give.
ORDER = {
    "cheetah": "Carnivora", "cougar": "Carnivora", "puma": "Carnivora", "bobcat": "Carnivora", "cat": "Carnivora",
    "dog": "Carnivora", "tiger": "Carnivora", "lion": "Carnivora", "leopard": "Carnivora", "jaguar": "Carnivora",
    "fox": "Carnivora", "coyote": "Carnivora", "wolf": "Carnivora", "seal": "Carnivora", "badger": "Carnivora",
    "otter": "Carnivora", "weasel": "Carnivora", "panda": "Carnivora", "raccoon": "Carnivora", "skunk": "Carnivora", "walrus": "Carnivora",
    "duck": "Anseriformes", "goose": "Anseriformes", "swan": "Anseriformes",
    "crow": "Passeriformes", "raven": "Passeriformes", "magpie": "Passeriformes", "swallow": "Passeriformes",
    "pheasant": "Galliformes", "quail": "Galliformes", "peacock": "Galliformes",
    "boar": "Artiodactyla", "giraffe": "Artiodactyla", "hippopotamus": "Artiodactyla", "goat": "Artiodactyla",
    "sheep": "Artiodactyla", "llama": "Artiodactyla", "camel": "Artiodactyla", "moose": "Artiodactyla",
    "pig": "Artiodactyla", "yak": "Artiodactyla",
    "zebra": "Perissodactyla", "rhinoceros": "Perissodactyla", "horse": "Perissodactyla", "donkey": "Perissodactyla",
    "rabbit": "Lagomorpha", "hare": "Lagomorpha",
    "hamster": "Rodentia",
    "tortoise": "Testudines", "condor": "Cathartiformes", "gull": "Charadriiformes", "eagle": "Accipitriformes",
    "flamingo": "Phoenicopteriformes", "spoonbill": "Pelecaniformes", "pelican": "Pelecaniformes",
    "owl": "Strigiformes", "parrot": "Psittaciformes", "toucan": "Piciformes", "ostrich": "Struthioniformes",
    "penguin": "Sphenisciformes", "whale": "Cetacea", "kangaroo": "Diprotodontia", "koala": "Diprotodontia",
    "sloth": "Pilosa", "hedgehog": "Eulipotyphla", "lobster": "Decapoda", "crab": "Decapoda", "bee": "Hymenoptera",
    "ant": "Hymenoptera", "butterfly": "Lepidoptera", "ladybug": "Coleoptera", "squid": "Teuthida",
    "octopus": "Octopoda", "frog": "Anura", "snail": "Stylommatophora", "snake": "Squamata", "spider": "Araneae",
    "shark": "Lamniformes", "goldfish": "Cypriniformes",
}


def pair_type(a: str, b: str) -> str:
    """The look-alike rule: two animals of one taxonomic order are a look-alike pair, else an unlike pair."""
    return "look-alike" if ORDER[a] == ORDER[b] else "unlike"


# The plan's pairs ("The pairs"), with the type its table gives and where their pictures come from.
PAIRS = [
    ("cheetah", "cougar", "CO3's twelve borrowing pairs"), ("duck", "goose", "CO3's twelve borrowing pairs"),
    ("crow", "raven", "CO3's twelve borrowing pairs"),
    ("boar", "pheasant", "CO3's twelve borrowing pairs"), ("bobcat", "quail", "CO3's twelve borrowing pairs"),
    ("giraffe", "tortoise", "CO3's twelve borrowing pairs"), ("hippopotamus", "swallow", "CO3's twelve borrowing pairs"),
    ("puma", "condor", "CO3's twelve borrowing pairs"), ("seal", "gull", "CO3's twelve borrowing pairs"),
    ("zebra", "flamingo", "CO3's twelve borrowing pairs"), ("eagle", "fox", "CO3's twelve borrowing pairs"),
    ("cat", "dog", "the old species judge's 29 renders"), ("tiger", "dog", "the old species judge's 29 renders"),
    ("cat", "fox", "the old species judge's 29 renders"),
    ("lobster", "whale", "the pooled LoRA's 8 unseen pairs"), ("bee", "owl", "the pooled LoRA's 8 unseen pairs"),
    ("butterfly", "crab", "the pooled LoRA's 8 unseen pairs"), ("kangaroo", "squid", "the pooled LoRA's 8 unseen pairs"),
    ("goat", "rhinoceros", "the pooled LoRA's 8 unseen pairs"), ("ladybug", "octopus", "the pooled LoRA's 8 unseen pairs"),
    ("flamingo", "sloth", "the pooled LoRA's 8 unseen pairs"), ("fox", "llama", "the pooled LoRA's 8 unseen pairs"),
    ("badger", "magpie", "the pooled LoRA dial's control 3"),
]
# Crow and raven are called and reported apart, in neither split.
REPORTED_APART = {("crow", "raven")}

# Eight new look-alike pairs for steps 0.9 and 0.10, rendered only if a kind is short and the owner says yes. The first
# four are the plan's own examples; the last four are proposed here, each two clearly different species of one order
# (two forms of one species, such as a farmed pig and a wild boar, are not used: a part borrowed between them cannot be
# called). Every one passes scripts/shared/disallowed_subjects.py and is in no other pair table
# (tests/test_paper2_bars.py checks the order, the guard and the tables).
NEW_LOOKALIKE_PAIRS = [
    ("lion", "tiger"), ("wolf", "fox"), ("horse", "donkey"), ("goat", "sheep"),
    ("leopard", "jaguar"), ("rabbit", "hare"), ("camel", "llama"), ("otter", "weasel"),
]
NEW_LOOKALIKE_SEEDS = range(8001, 8009)

# Wrong-species pictures made by renaming, if natural ones are short: one subject shown under a look-alike name.
RENAMING = {"cougar": "leopard", "goose": "swan", "fox": "coyote", "flamingo": "spoonbill"}
RENAMED_PER_KIND_MAX = 20     # of the 40 read wrong-species pictures
NATURAL_WRONG_SPECIES_NEEDED = 65   # 40 read, 10 confirmation, 15 choosing

# The old judge's verdicts (3-understanding-compositionality/scripts/species_judge.py) as the six kinds.
OLD_JUDGE_KINDS = {"composed": "two_clear", "two_of_one": "two_of_one_kind", "hybrid": "hybrid", "one_body": "other"}

# ---------------------------------------------------------------- the ImageNet class table

IMAGENET_CLASS_NAMES = {
    1: "goldfish", 2: "great white shark", 3: "tiger shark", 4: "hammerhead", 9: "ostrich", 18: "magpie",
    22: "bald eagle", 24: "great grey owl", 30: "bullfrog", 31: "tree frog", 32: "tailed frog",
    52: "thunder snake", 53: "ringneck snake", 54: "hognose snake", 55: "green snake", 56: "king snake",
    57: "garter snake", 58: "water snake", 59: "vine snake", 60: "night snake", 61: "boa constrictor",
    62: "rock python", 63: "Indian cobra", 64: "green mamba", 65: "sea snake", 66: "horned viper", 67: "diamondback",
    68: "sidewinder", 72: "black and gold garden spider", 73: "barn spider", 74: "garden spider", 75: "black widow",
    76: "tarantula", 77: "wolf spider", 84: "peacock", 85: "quail", 87: "African grey", 88: "macaw",
    89: "sulphur-crested cockatoo", 90: "lorikeet", 96: "toucan", 97: "drake", 98: "red-breasted merganser",
    99: "goose", 100: "black swan", 104: "wallaby", 105: "koala", 113: "snail", 118: "Dungeness crab",
    119: "rock crab", 120: "fiddler crab", 121: "king crab", 122: "American lobster", 123: "spiny lobster",
    129: "spoonbill", 130: "flamingo", 144: "pelican", 145: "king penguin", 147: "grey whale", 148: "killer whale",
    150: "sea lion", 269: "timber wolf", 270: "white wolf", 271: "red wolf", 272: "coyote", 277: "red fox",
    278: "kit fox", 279: "Arctic fox", 280: "grey fox", 281: "tabby", 282: "tiger cat", 283: "Persian cat",
    284: "Siamese cat", 285: "Egyptian cat", 286: "cougar", 288: "leopard", 290: "jaguar", 291: "lion", 292: "tiger",
    293: "cheetah", 301: "ladybug", 309: "bee", 310: "ant", 321: "admiral", 322: "ringlet", 323: "monarch",
    324: "cabbage butterfly", 325: "sulphur butterfly", 326: "lycaenid", 330: "wood rabbit", 331: "hare",
    332: "Angora", 333: "hamster", 339: "sorrel", 340: "zebra", 341: "hog", 342: "wild boar", 344: "hippopotamus",
    348: "ram", 349: "bighorn", 350: "ibex", 354: "Arabian camel", 355: "llama", 356: "weasel", 360: "otter", 361: "skunk",
    362: "badger", 364: "three-toed sloth", 388: "giant panda",
}
DOG_BREEDS = range(151, 269)   # ImageNet's 118 dog breeds, Chihuahua (151) to Mexican hairless (268); 269 is a wolf

IMAGENET_CLASSES = {
    "cheetah": [293], "cougar": [286], "puma": [286], "leopard": [288], "jaguar": [290], "lion": [291], "tiger": [292],
    "cat": [281, 282, 283, 284, 285], "dog": list(DOG_BREEDS), "wolf": [269, 270, 271], "coyote": [272],
    "fox": [277, 278, 279, 280], "badger": [362], "otter": [360], "weasel": [356], "skunk": [361], "panda": [388],
    "seal": [150],           # a sea lion is an eared seal
    "duck": [97, 98], "goose": [99], "swan": [100], "magpie": [18], "quail": [85], "peacock": [84],
    "eagle": [22], "owl": [24], "flamingo": [130], "spoonbill": [129], "pelican": [144], "parrot": [87, 88, 89, 90],
    "toucan": [96], "ostrich": [9], "penguin": [145],
    "boar": [342], "pig": [341], "hippopotamus": [344], "goat": [350],   # an ibex is a wild goat
    "sheep": [348, 349], "llama": [355], "camel": [354], "zebra": [340], "horse": [339],
    "rabbit": [330, 332], "hare": [331], "hamster": [333],
    "kangaroo": [104],       # a wallaby is a small kangaroo
    "koala": [105], "sloth": [364], "whale": [147, 148],
    "lobster": [122, 123], "crab": [118, 119, 120, 121], "bee": [309], "ant": [310],
    "butterfly": [321, 322, 323, 324, 325, 326], "ladybug": [301], "frog": [30, 31, 32], "snail": [113],
    "snake": list(range(52, 69)), "spider": [72, 73, 74, 75, 76, 77], "shark": [2, 3, 4], "goldfish": [1],
}
# No ImageNet-1k class is these animals or a kind of them, so their pairs are out of the ImageNet judge's count.
IMAGENET_UNCOVERED = ["bobcat", "crow", "raven", "swallow", "pheasant", "giraffe", "tortoise", "condor", "gull",
                      "rhinoceros", "squid", "octopus", "donkey", "moose", "yak", "raccoon", "walrus",
                      "hedgehog"]


def imagenet_covers(a: str, b: str) -> bool:
    return bool(IMAGENET_CLASSES.get(a)) and bool(IMAGENET_CLASSES.get(b))


def coverage() -> dict:
    """Which of the plan's pairs and new look-alike pairs the ImageNet judge can count."""
    rows = [(a, b, "plan") for a, b, _ in PAIRS] + [(a, b, "new look-alike") for a, b in NEW_LOOKALIKE_PAIRS]
    out = {"pairs": [], "covered": 0, "total": len(rows)}
    for a, b, where in rows:
        ok = imagenet_covers(a, b)
        out["covered"] += ok
        missing = [s for s in (a, b) if not IMAGENET_CLASSES.get(s)]
        out["pairs"].append({"pair": f"{a} and {b}", "where": where, "type": pair_type(a, b), "covered": ok,
                             "missing": missing})
    return out


if __name__ == "__main__":
    c = coverage()
    for row in c["pairs"]:
        mark = "covered" if row["covered"] else f"not covered (no class for {', '.join(row['missing'])})"
        print(f"{row['pair']:28s} {row['type']:10s} {row['where']:15s} {mark}")
    plan = [r for r in c["pairs"] if r["where"] == "plan"]
    look = [r for r in plan if r["type"] == "look-alike"]
    print(f"\nthe ImageNet judge covers {sum(r['covered'] for r in plan)} of {len(plan)} plan pairs "
          f"({sum(r['covered'] for r in look)} of {len(look)} look-alike) and "
          f"{sum(r['covered'] for r in c['pairs'] if r['where'] != 'plan')} of {len(NEW_LOOKALIKE_PAIRS)} new look-alike pairs")
