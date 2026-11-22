# Can a classifier supply the term the product drops

Evidence for [the plan](../../../plans/15-can-a-classifier-supply-the-missing-term.md),
piece 1: building the labelled set the two-output presence head trains on. The head and the
composer are not built yet, so nothing here says whether classifier guidance composes. What is
settled is which instrument can see a fused animal, which the labelling rule needs before any
label is written.

## The labelling rule

Both labels 1 needs two separate animal boxes, one named cat and one named dog, each box at least
0.25 of the image's longer side. One box is never both animals, so a fused animal is labelled
neither animal present. The constants are in
[the labelling module](../../../../poe_repair/experiments/presence_head/labelling.py), not in prose,
so changing them after seeing a render shows up in a diff.

## evidence_a_cat__x__a_dog.png

`evidence_a_cat__x__a_dog.png` is on the cluster at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/evidence_a_cat__x__a_dog.png`

**What it shows.** Every plain product-of-experts render of `a_cat__x__a_dog` held outside the
eight sweep seeds, 33 rows, each labelled with the species the first rule named and the margin it
named it at. Red titles are renders where the rule called a species on a chimera.

**What it says.** 32 of 33 were named a confident single species, at margins out to 0.993, on
pictures that are fused animals by eye. The rule's hybrid class was therefore empty, and training
on these labels would have taught the head that a fused cat-dog is a dog.

**What the picture showed that the counts did not.** The 33 rows are 14 distinct images. Seed 4
appears ten times and seed 42 thirteen, because the same render is reachable under several output
families and the first pass de-duplicated by path rather than by content. Every count since is
over distinct images.

## species-test-calibration-sweep.png

<a href="species-test-calibration-sweep.png"><img src="species-test-calibration-sweep.png" width="260" alt="Four pass-rate curves against the cosine floor, with the chosen floor and the two bars drawn on"></a>

**What is on each axis.** The fraction of photographs passing a presence test on y, a 0-to-1
scale; the floor that each species' raw cosine similarity must clear on x, from 0.150 to 0.300.
Four curves: a cat-only photo passing the cat test and a dog-only photo passing the dog test, both
wanted high, and each photo set passing the other species' test, both wanted low. 200 photographs
per species, species annotated by COCO, read through the same detector crop the renders are read
through. Numbers from `sweep.json` beside it.

**The bars, written before the run.** Every true pass rate at least 0.90 and every false pass rate
at most 0.10, at the floor that maximises their separation. In
[the calibration script](../../../scripts/presence_head_calibrate_species.py) as `MIN_TRUE_PASS`
and `MAX_FALSE_PASS`.

**What it says.** Usable. At a floor of 0.245, a cat-only photo passes the cat test 0.925 of the
time and a dog-only photo passes the dog test 0.950, while the cross rates are 0.070 and 0.035.

## hybrid-check-cosine-sheet.png

<a href="hybrid-check-cosine-sheet.png"><img src="hybrid-check-cosine-sheet.png" width="260" alt="14 distinct fused renders, each with its two independent cosine similarities and whether both cleared the floor"></a>

**What it shows.** The 14 distinct fused renders, each with its raw cosine similarity to "a photo
of a cat" and to "a photo of a dog" read on its own, and whether both cleared 0.245. Green means
both fired, so the rule can call it a hybrid. The margin the two-way softmax gave the same render
is printed underneath.

**What it says.** 8 of 14 are now seen as holding both species, against 1 of 14 under the rule
that makes the species compete. The six misses are the renders where one species visibly
dominates, such as the dark dog face in the lavender field at `p(cat) 0.25, p(dog) 0.28`.

**The measurement underneath it.** In raw cosine, the fused renders sit at a median 0.2618 to cat
and 0.2683 to dog, both at the level a clean photograph reaches for its own species, while a clean
dog photograph sits at 0.2130 to cat. The information that both animals are present survives in
the similarities and is destroyed by any normalisation that makes the two species compete: a
softmax turns that 0.0065 gap into 0.003 against 0.979.

## What this cannot tell you

**Nothing here is about composition.** No head is trained and no render has been made through a
guided sampler. These are instrument results only.

**The six missed renders are not a calibration problem.** They are renders whose fusion is
one-sided, and a lower floor that caught them would also start calling clean photographs hybrids,
which is what the sweep curve shows.

**Every hybrid in this set comes from the model being corrected.** A head trained on them learns
to dislike this sampler's particular failure, which is not the same as knowing what two animals
look like.

## Cross-references

- The mention of **the labelling rule** in [plan 15, can a classifier supply the missing term](../../../plans/15-can-a-classifier-supply-the-missing-term.md#why-this-plan-exists).

<!-- cluster:start -->

## On the cluster

The other 46 files of this result (63.5 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 18 | 25.3 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/` |
| `product-pushed-to-dog` | 28 | 38.3 MB | [sheet.png](product-pushed-to-dog/sheet.png) | `/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/product-pushed-to-dog/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/2-classifier-guidance/artifacts/results/can-a-classifier-supply-the-missing-term/<run>/ <run>/
```

<!-- cluster:end -->
