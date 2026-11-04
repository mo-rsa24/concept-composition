# Route 03: four journeys for the fourteen words this walk had no teacher for

Four prompts. Paste each into its own fresh session. They do not depend on each other and can run
in any order, or at the same time.

## Why there are four and not fourteen

Every reply in this walk ended on a `gap:` line naming field words that no note, plan or textbook
chapter in the learning tree teaches. Fourteen words accumulated:

masking, mimicry, occlusion, profusion, the ILO classification, pleural thickening, consolidation,
infiltration, intersection over union, DICOM, windowing, MONOCHROME1, interobserver agreement,
Fleiss' kappa, AUROC, joint conditioning, prevalence recalibration.

Seventeen words.

They are not fourteen subjects. They are four, and the grouping is what one sitting can learn at
once:

| Journey | The words it covers | Why these belong together |
|---|---|---|
| A: what two diseases do to each other on one film | masking, mimicry, occlusion, profusion, the ILO classification, pleural thickening, consolidation, infiltration | all eight are things a reader says about a radiograph, and the first four are the overlap regimes this walk's claim 1.10 rests on |
| B: where a finding is, and how much two findings share | intersection over union, DICOM, windowing, MONOCHROME1, bounding box, box fraction, boolean mask | the box and the pixels it sits on. Run 5 of this walk turned on an IoU floor and the answer changed, so this is the measure the regimes are read with |
| C: scoring a classifier when the test set is biased | AUROC, joint conditioning, prevalence recalibration, AUROC by subgroup, hidden stratification, shortcut learning, confidence interval, binomial bound, Clopper-Pearson, McNemar's test | the whole chain in [the composition route](../../../../plans/route-compose-x-rays-and-score-them.md) ends in an AUROC table with an interval per cell, and none of the measures in it is taught anywhere yet |
| D: agreement between three readers | interobserver agreement, Fleiss' kappa | `ai-engineering/chance-corrected-agreement` already teaches percent agreement through Cohen's κ for two raters. VinDr has three, which that journey does not reach, so this is `--over` and not a new journey |

The register of every word a walk has ever left uncovered is
`~/goal-setting/learning/artifacts/notes/words-to-learn/WORDS.md`, most-seen first. Journey C picks
up the four highest-count statistics entries in it (confidence interval 15, McNemar's test 15,
binomial bound 14, delta method 13), so it closes more than this walk's share.

## Shared facts every one of the four needs

- The data is VinDr-CXR at `/datasets/mmolefe/vinbigdata/`, on the cluster only. `train.csv` holds
  67,914 boxes over 4,394 annotated films, and every annotated film was read by exactly three
  radiologists.
- Every count or run goes through
  `scripts/cluster.sh sh "<command>"` from `/home/molef/PhD/poe_repair_min-dog-and-rain`, with
  `CLUSTER_DIR=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`. Figures come
  back only with `scripts/cluster.sh pull`. Nothing is copied off the cluster by hand.
- The walk these words came from is
  `artifacts/ideas/silicotb-conversation/IDEA_MAP.md` in that repo. Its runs 3, 4 and 5 are the
  measurements each journey below is anchored to.
- No journey in the tree teaches anything medical. A search for an existing home will come back
  empty for journeys A and B; that is expected, not a reason to stop.

---

## A: what two diseases do to each other on one film

```text
/submerge I cannot read a chest radiograph, and the argument I am making in my PhD depends on
reading one. I need to know what a radiologist means by masking, mimicry, occlusion and profusion,
what the ILO classification actually classifies and on what scale, and what pleural thickening,
consolidation and infiltration look like on a film well enough to tell them apart when both are
present. The thing I am trying to settle is whether two diseases on one film sit in separate places,
share a region while staying recognisable, merge into a lesion neither builds alone, or one hides
the
other outright.

Starts from: artifacts/ideas/silicotb-conversation/IDEA_MAP.md in
/home/molef/PhD/poe_repair_min-dog-and-rain, claim 1.10 and the runs table. Ten NIOH teaching
radiographs of silicosis and tuberculosis in gold miners, with an expert reader's verdict in each
filename, four of them "doubtful silicosis". The source is
https://www.nioh.ac.za/teaching-training/training-and-learning-repository-of-materials-on-the-radiology-of-silicotuberculosis/chest-radiographs-showing-features-of-silicosis-and-tuberculosis-and-tuberculosis-in-silica-exposed-individuals-2/
and the accompanying paper is Maboso et al. 2022, "Difficulties in distinguishing silicosis and
pulmonary tuberculosis in silica exposed gold miners: a report of four cases", which covers
radiographs 3, 5, 9 and 10. The films may be looked at and learned from; their licence reads
"Copyright 2023 NIOH. All Rights Reserved", so none of them goes into a committed figure or a
training set.

Settled facts: Ehrlich et al. 2024 (https://doi.org/10.1183/16000617.0168-2024) name three things
two diseases can do to each other, each disease hiding the other, each distorting the other's
pattern, and the two together building a lesion neither builds alone. Masking is the one with no
picture anywhere in my repository. NIOH radiographs 3 and 4 are the same disease pair with the same
two labels and opposite reader verdicts, and the visible difference is how much lung the first
disease left to read in, which says masking is a dose effect rather than a category.

Outputs: the journey under /home/molef/goal-setting/learning/, its row in JOURNEYS.md, and a figure
set in its artifacts/figures/ where every regime has a film with the regions outlined and labelled
on the image itself, never in a legend underneath.

Target: I can take a film I have not seen, say which of the four regimes each pair of findings on it
is in, and say what in the picture made me call it that. On the ten NIOH films my calls agree with
the reader's verdict in the filename on at least eight.

Progress: one regime per sitting, each ending on a film I called correctly before being told.

Proof: my calls on the ten NIOH films against the verdicts in their filenames, written down before
I look at the filename.

Limits: patient data stays on the cluster. Nothing whose licence forbids it is committed anywhere.
Do not download model weights onto a cluster node without asking.

Hand-off: the regime definitions go back into claim 1.10 of the walk, and the figure set is what
route 01 of that walk was asking for.
```

---

## B: where a finding is, and how much two findings share

```text
/submerge I want to be able to say, from annotations, whether two findings on an image are in the
same place, and I want to know what the measure I use is actually doing. Intersection over union,
what the number means at 0.1, 0.25 and 0.5, when it lies, what containment measures instead, and
what a bounding box is as opposed to a mask. Underneath that, what a DICOM file is: how the pixels
are stored, what MONOCHROME1 means and why the image comes out inverted, and what windowing does.

Starts from: /datasets/mmolefe/vinbigdata/train.csv on the cluster, 67,914 boxes over 4,394 films,
read through scripts/cluster.sh sh "<command>" from /home/molef/PhD/poe_repair_min-dog-and-rain with
CLUSTER_DIR=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain. The walk that raised
this is artifacts/ideas/silicotb-conversation/IDEA_MAP.md, run 5.

Settled facts: I counted two findings as sharing a region whenever any pair of their boxes
overlapped at all, and the answer changed when a floor was added. Lung opacity with pulmonary
fibrosis reads 76% at any overlap and 46% once the boxes must share half their area, over 666 films,
median intersection over union 0.405. Pleural thickening with pulmonary fibrosis falls from 46% to
7%, so what looked like a pair spanning two regimes was boxes clipping at a corner. Pleural effusion
with pleural thickening reads 94% with a median intersection over union of exactly 1.00, which is
one
rectangle twice: on 737 of its 839 films one radiologist drew one box and attached both labels to
it.
A measure that cannot tell that apart from a real overlap is the thing this journey is about.

Outputs: the journey under /home/molef/goal-setting/learning/, its row in JOURNEYS.md, and in its
artifacts/figures/ one picture per threshold showing two boxes at that intersection over union on a
real film, so I can see what 0.25 looks like rather than compute it.

Target: I can derive intersection over union from areas by hand on two rectangles whose corners I
chose, predict roughly what it will be before computing it, say which of two findings' boxes is
contained in the other, and open a DICOM in Python and display it the right way up without
copying code.

Progress: one measure per sitting, each ending on a number I worked by hand that matches the one
the code prints.

Proof: my hand-computed intersection over union on three box pairs against what a four-line script
prints for the same coordinates.

Limits: the films stay on the cluster. Figures come back only with scripts/cluster.sh pull.

Hand-off: the surviving half of run 5's check in the walk, which is to drop every identical-box film
and re-read the per-pair split on what remains.
```

---

## C: scoring a classifier when the test set is biased

```text
/submerge I am about to report an AUROC table and I cannot defend any part of it. I need AUROC
itself, what it is the area under and what it does not tell me, how to put a confidence interval on
one and how wide that interval is at the counts I will actually have, AUROC read per subgroup, what
hidden stratification and shortcut learning are, what prevalence recalibration does to a classifier
trained on a biased set, and what joint conditioning is as the baseline a generative method has to
beat. McNemar's test for comparing two classifiers on the same films, and the binomial and
Clopper-Pearson intervals for a rate.

Starts from:
artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md
in /home/molef/PhD/poe_repair_min-dog-and-rain, read whole, and the chain in
plans/route-compose-x-rays-and-score-them.md in the same repo, whose last step is exactly this
table.

Settled facts: prevalence recalibration alone takes AUROC from 0.226 to 0.726 on films where the
shortcut points the wrong way, which is most of what a generative method would have to add on top
(Kina and Petersen 2026, arXiv 2609.07922). At 25 positives against 25 negatives and AUROC 0.75 the
Hanley-McNeil 95% interval is about ±0.14, and at 100 against 100 about ±0.07, so a table at the
smaller counts cannot separate 0.70 from 0.80. The register of words I have met and not learned is
~/goal-setting/learning/artifacts/notes/words-to-learn/WORDS.md, where confidence interval has been
seen 15 times, McNemar's test 15, binomial bound 14 and delta method 13, all of them this journey's.

Outputs: the journey under /home/molef/goal-setting/learning/, its row in JOURNEYS.md, and in its
artifacts/figures/ one ROC curve I drew from a table of scores by hand, with the area shaded and the
interval drawn on it.

Target: I can compute AUROC by hand from a list of ten scored examples, say what it would be for a
classifier that reads the shortcut and nothing else, compute the interval for my own counts, and say
in one sentence each what prevalence recalibration and joint conditioning do and when each is the
right baseline.

Progress: one measure per sitting, each ending on a number worked by hand that matches the code.

Proof: my hand-computed AUROC and interval on a ten-row table against scikit-learn's on the same ten
rows.

Limits: no cluster run is needed for this one; it is worked on small tables I write myself.

Hand-off: the bar in the composition route's step 3, which is where these numbers get used, and
Introduction to probability for data science p. 605 is the shelf page the register already found for
AUROC.
```

---

## D: agreement between three readers

```text
/submerge --over ai-engineering/chance-corrected-agreement I need the same journey's argument with
three raters instead of two. Fleiss' kappa, what it reduces to when there are two raters, and how to
read a per-item vote count of 1, 2 or 3 out of 3 as a measure of how hard an item was to see.

Starts from: the existing journey at
/home/molef/goal-setting/learning/ai-engineering/chance-corrected-agreement/, which teaches percent
agreement, Scott's π, Cohen's κ, φ and the kappa paradox on a judge's yes-or-no verdicts. Extend it;
do not start a second journey.

Settled facts: VinDr-CXR's 4,394 annotated films were each read by exactly three radiologists, and
the share of findings only one of the three marked runs from 68% for "other lesion" and 66% for
consolidation down to 21% for cardiomegaly and 24% for aortic enlargement. I am using that
single-reader share as a proxy for a finding being present and hard to see, and nothing has checked
whether that is a legitimate reading or whether it is measuring reader disagreement about what a
label means. The data and the counts are in run 4 of
artifacts/ideas/silicotb-conversation/IDEA_MAP.md in /home/molef/PhD/poe_repair_min-dog-and-rain.

Outputs: new plan files inside the existing journey, its ledger row in JOURNEYS.md updated, and one
figure in its artifacts/figures/ showing the same twelve findings' agreement as both a kappa and a
vote-count share, so I can see where the two disagree.

Target: I can compute Fleiss' kappa by hand on a small table of three readers over six items, say
what the chance-agreement term is doing there that percent agreement misses, and say whether a
single-reader share of 66% on consolidation is evidence that consolidation is hard to see or
evidence
that the three readers mean different things by it.

Progress: one sitting, ending on the hand computation matching the code.

Proof: my Fleiss' kappa by hand on six items against what a short script prints, and the twelve-row
VinDr table computed both ways.

Limits: the CSV stays on the cluster; it is read through scripts/cluster.sh sh "<command>".

Hand-off: the answer goes into claim 1.10 of the walk, which currently treats the single-reader
share
as the hidden-disease signal without having checked it.
```

## What is not routed here

Three words in this walk's gap lines are covered by something that already exists and need no
journey: AUROC has a shelf page, and masking and occlusion have the Ehrlich review. They stay in
journeys A and C anyway, because a page read once is not the same as being able to use the word.

The remaining register entries are another pass. About a third of them are not field words at all
(`watching sheet`, `route file`, `none new`, `bundled`), and the register is worth pruning before
anything else is minted from it.
