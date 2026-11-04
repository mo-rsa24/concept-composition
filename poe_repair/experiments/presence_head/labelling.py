"""The labelling rule: what counts as a cat present, a dog present, and neither.

This file is the lever the whole experiment turns on. Nothing in a caption says that a cat and a
dog have to be two separate things, which is why the product of "a photo of a cat" and "a photo
of a dog" is maximised by one animal that reads as both. The rule below is where that requirement
first enters a loss: a fused animal is labelled as neither animal present, never as both.

The rule is constants, not prose, so changing it after seeing the renders shows up in a diff.

Two entry points, because the two label sources are not the same kind of evidence:

``label_from_coco`` reads COCO's own instance annotations. The species is given, so the only
judgment is the box-size floor.

``label_from_detections`` reads what the instruments say about a render: GroundingDINO's generic
"animal" boxes, each one named by zero-shot CLIP on its padded crop. Measured on 64 single-cat
renders, that crop classifier called cat 64 of 64 at a median margin of 0.990, where the
detector's own confidences separated the classes by 0.054. It has never been tested on a frame
holding two animals whose boxes overlap, so a box it cannot call is reported rather than forced.
"""
from __future__ import annotations

from dataclasses import dataclass

LABEL_NAMES = ("cat_present", "dog_present")

# ---------------------------------------------------------------------------
# The rule.
# ---------------------------------------------------------------------------
# Both labels 1 requires two separate animal boxes, one named cat and one named dog. One box can
# never carry both labels, however much of both animals it contains. This is the whole lever.
MIN_INSTANCES_FOR_BOTH = 2

# A box whose |p_cat - p_dog| falls below this is not assigned a species. The crop classifier's
# own uncertain floor is 0.20 and its measured median margin on known-single-species renders is
# 0.990, so 0.50 sits far above the noise and far below the signal.
MIN_SPECIES_MARGIN = 0.50

# A box smaller than this fraction of the image's longer side is not an animal being present, the
# same floor the compose scorer and the photo pool already use, so a training photo and a scored
# render agree about what counts.
MIN_BOX_FRACTION = 0.25

# A single animal carrying features of both is labelled (0, 0). The alternative, (1, 1), is what
# the product already maximises, so training on it would teach the head to want the failure.
HYBRID_IS_NEITHER = True

RULE_CONSTANTS = {
    "MIN_INSTANCES_FOR_BOTH": MIN_INSTANCES_FOR_BOTH,
    "MIN_SPECIES_MARGIN": MIN_SPECIES_MARGIN,
    "MIN_BOX_FRACTION": MIN_BOX_FRACTION,
    "HYBRID_IS_NEITHER": HYBRID_IS_NEITHER,
}

RULE_IN_WORDS = (
    "both labels 1 needs two separate animal boxes, one named cat and one named dog, each box at "
    f"least {MIN_BOX_FRACTION} of the image's longer side and named with a species margin of at "
    f"least {MIN_SPECIES_MARGIN}; one box is never both animals, so a fused animal is labelled "
    "neither animal present"
)


@dataclass(frozen=True)
class Label:
    cat_present: int
    dog_present: int
    reason: str
    n_boxes_kept: int = 0
    n_boxes_unnamed: int = 0

    def as_tuple(self) -> tuple[int, int]:
        return (self.cat_present, self.dog_present)

    def as_row(self) -> dict:
        return {"cat_present": self.cat_present, "dog_present": self.dog_present,
                "reason": self.reason, "n_boxes_kept": self.n_boxes_kept,
                "n_boxes_unnamed": self.n_boxes_unnamed}


def box_fraction(box, image_size) -> float:
    """A box's longer side as a fraction of the image's longer side."""
    x0, y0, x1, y1 = box
    return max(abs(x1 - x0), abs(y1 - y0)) / max(image_size)


def label_from_coco(categories, boxes, image_size) -> Label:
    """Label a COCO photograph from its own annotations.

    ``categories`` and ``boxes`` are parallel; each category is "cat", "dog" or anything else.
    """
    kept = [c for c, b in zip(categories, boxes)
            if c in ("cat", "dog") and box_fraction(b, image_size) >= MIN_BOX_FRACTION]
    cat = int("cat" in kept)
    dog = int("dog" in kept)
    if cat and dog:
        reason = "a cat and a dog, both above the size floor"
    elif cat or dog:
        reason = f"only {'cat' if cat else 'dog'} above the size floor"
    else:
        reason = "no cat or dog above the size floor"
    return Label(cat, dog, reason, n_boxes_kept=len(kept))


def label_from_detections(detections, image_size) -> Label:
    """Label a render from its animal boxes and each box's species call.

    Each detection is a mapping with ``box`` and ``margin``, where margin is p_cat - p_dog from
    the crop classifier. A box the classifier cannot call is counted, never guessed at.
    """
    kept = [d for d in detections if box_fraction(d["box"], image_size) >= MIN_BOX_FRACTION]
    named_cat = [d for d in kept if d["margin"] >= MIN_SPECIES_MARGIN]
    named_dog = [d for d in kept if d["margin"] <= -MIN_SPECIES_MARGIN]
    unnamed = len(kept) - len(named_cat) - len(named_dog)

    if not kept:
        return Label(0, 0, "no animal above the size floor", 0, 0)

    both = (len(kept) >= MIN_INSTANCES_FOR_BOTH and named_cat and named_dog)
    if both:
        return Label(1, 1, "two boxes, one named cat and one named dog",
                     len(kept), unnamed)

    if unnamed and not named_cat and not named_dog:
        # Every box present and none of them nameable: a fused animal, which the rule says is
        # neither animal present rather than both.
        return Label(0, 0, "one or more boxes, no species nameable: a hybrid", len(kept), unnamed)

    if named_cat and not named_dog:
        return Label(1, 0, "every named box is a cat", len(kept), unnamed)
    if named_dog and not named_cat:
        return Label(0, 1, "every named box is a dog", len(kept), unnamed)

    # Both species named with fewer than two boxes kept. The margin is one signed number per box,
    # so a single box cannot be named both, which leaves this unreachable; it is a guard rather
    # than a case, and it refuses to guess a label instead of inventing one.
    raise AssertionError(
        f"both species named across {len(kept)} kept box(es), which the margin rule cannot "
        "produce; the detection rows are malformed"
    )
