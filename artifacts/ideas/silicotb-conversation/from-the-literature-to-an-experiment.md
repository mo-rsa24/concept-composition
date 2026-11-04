# From the literature to an experiment: the five decisions the report has to settle

Read by: [the claim ledger](IDEA_MAP.md), route 05.

[Route 05](routes/05-deep-research-the-disease-pairs.prompt.md) is a literature search, and a
literature search is only worth running if something is waiting on its answer. Five things are. This
file names each one, says which of the six returned files answers it, writes the options out now so
the walk picks between written options rather than inventing one under time pressure, and names what
decides between them.

Walk it one decision per round, in this order, once the six files land in [the return
folder](routes/05-return/). Each decision is settled in writing before the next opens, because every
later one is shaped by the earlier ones: the pairs we can choose depend on the dataset, and the
dataset depends on what we are claiming to do.

## Words this uses

**Regime.** One of the four ways two diseases sit relative to each other in one region of lung:
separate places, same region with both readable, merged beyond decomposition, one present and
unreadable. Set out in full in the route file.

**The dial.** The share of real both-disease films the generator is allowed to see during training:
100%, 25%, 5%, 1%, 0%. Composition can only beat joint conditioning where the share is too small for
a joint model to fit, so the dial is the axis the comparison lives on.

**Proxy pair.** A pair of findings that is not silicosis with tuberculosis but sits in the same
regime, chosen because a dataset holds it. Everything we measure on a proxy pair transfers to
silicotuberculosis only by argument, never by measurement, and that argument has to be made out loud
wherever a proxy pair appears.

## What is already settled, so no round re-opens it

**The four regimes are real for one pair and measurable for two of the four.** Aortic enlargement
with cardiomegaly sits in separate places on 1,951 of the 1,972 VinDr-CXR films carrying both, 98%,
and the share does not move at any intersection-over-union floor. Lung opacity with pulmonary
fibrosis shares a region on 201 of 489 films, 41%, at a floor of 0.25, median intersection over
union 0.405. The merged regime and the hidden regime cannot be read off annotation boxes at all,
because both are cases where the boxes say less than the film does.

**Our own dataset cannot do silicotuberculosis.** VinDr-CXR has fourteen finding classes and neither
tuberculosis nor silicosis is one of them: aortic enlargement, atelectasis, calcification,
cardiomegaly, consolidation, interstitial lung disease, infiltration, lung opacity, nodule or mass,
other lesion, pleural effusion, pleural thickening, pneumothorax, pulmonary fibrosis. So every
measurement we have made so far is on proxy pairs, and the only silicotuberculosis films we hold are
ten look-only teaching radiographs from NIOH whose licence forbids reuse.

**Recalibrating prevalence alone takes AUROC from 0.226 to 0.726** on films whose disease mix does
not match the training set (Kina and Petersen 2026, https://arxiv.org/abs/2609.07922). Any
generative method has to beat that, and it is cheap, so it is the baseline rather than a comparison.

**Twenty-five films per class against twenty-five gives an AUROC confidence interval of about
±0.14** at a true AUROC of 0.75, by Hanley and McNeil. A hundred against a hundred gives ±0.07. So a
scorecard built on fewer than about a hundred films per class cannot separate two methods that
differ by less than 0.1, and the cell counts decide what the experiment can conclude before anything
is trained.

## What route 05's report settled

The six files are in [the return folder](routes/05-return/). Sixteen pairs, sixty-four regime
verdicts, twenty-five candidate figures, twenty of them with a verified reuse licence.

**The merged regime has no verified example on any modality.** Every one of the seven candidates is
the report's own inference from a caption that describes an altered appearance without establishing
that neither disease can be named. No chest radiograph caption establishes it at all. So the regime
our work treats as the interesting case may not be a documented phenomenon.

**The separate-places regime is almost never established either**, because our own definition
requires each disease to look exactly as it would alone, and no paper tests that counterfactual.
Two of sixteen pairs reach "documented", both on CT.

**There is a fifth state, and our four cannot hold it.** Both diseases confirmed, neither visible on
the film. Yoo et al. 2011 report four of fifty-four HIV-positive patients with normal radiographs
carrying two simultaneous respiratory infections, tuberculosis with cryptococcosis in two and
tuberculosis with aspergillosis in two (https://pmc.ncbi.nlm.nih.gov/articles/PMC3126910/). All four
of our regimes require at least one disease to be visible.

**There is also a sixth failure, of a different kind: one disease changes where the other can
appear.** Jackson et al. 1986 report lung segments occluded by chronic emboli being spared from
pulmonary oedema (https://pubmed.ncbi.nlm.nih.gov/3948563/). The two end up in separate places
because they interacted, which is exactly what the separate-places regime says does not happen.

**HIV with tuberculosis is not a disease pair in our sense.** HIV has no lesion on a chest
radiograph, so the object is one disease plus a systemic modifier. The modifier is strong: normal
radiographs in culture-confirmed tuberculosis occurred in 52 of 323 patients with CD4 at or below
150 cells per microlitre, 19 of 525 above 150, and 26 of 1,085 HIV-negative patients (Chamie et al.
2010, https://pmc.ncbi.nlm.nih.gov/articles/PMC3033769/). A second study did not reproduce the
normal-film contrast at its own CD4 split, p = 0.44.

**Nobody has published the four regimes, or three of them.** The closest is the official statement on
combined pulmonary fibrosis and emphysema, which distinguishes separated from intermingled
distributions, matching our first two without their readability requirement (Cottin et al. 2022,
Table 10, https://pmc.ncbi.nlm.nih.gov/articles/PMC7615200/). No textbook chapter on how to read a
film carrying two diseases exists; radiology teaching is organised one disease at a time.

**The best-evidenced pair is not silicosis with tuberculosis.** Combined pulmonary fibrosis and
emphysema has a named syndrome with an official statement, serial CT showing isolated emphysema
becoming admixed with fibrosis over two years, both arrangements present in one examination, and a
combination-specific lesion with a real comparison behind it: thick-walled cystic lesions in 16 of
22 patients with the combination, 0 of 8 with fibrosis alone, 0 of 17 with emphysema alone (Inomata
et al. 2014, https://link.springer.com/article/10.1186/1471-2466-14-104). That 16 against 0 and 0 is
the only published number in the whole report that measures a pair doing something neither disease
does alone.

## Decision 1: what the paper claims

**The question.** Is the four-regime account a general property of chest disease pairs, a property
of silicotuberculosis alone, or the wrong axis entirely?

**Answered by** `pairs-and-their-regimes.md` and `where-the-four-regimes-fail.md`.

| If the report shows | Then the claim is | And the contribution is |
|---|---|---|
| three or more pairs each documenting three or more regimes | the regimes are general | a target set: composition has four things to get right, not one |
| only silicosis with tuberculosis documenting more than two | the regimes are specific to one compound disease | a case study, and the paper's scope narrows to it |
| a fifth regime, documented | the account is incomplete | the taxonomy itself, with the fifth named and pictured |
| every published scheme cutting the combined case a different way | the regimes are our axis and nobody else's | an argument we have to win before any experiment counts |

**What decides it.** Count the filled cells in the pairs table, separating the ones supported by a
figure caption from the ones inferred from prose. A regime supported only by our own reading of an
image does not count toward generality.

**What this produces.** One paragraph stating the claim, which every later round is checked against.

## Decision 2: what we are actually doing

**The question.** Are we synthesising films for a disease pair nobody has films of, or measuring
what a product of two experts drops on pairs we do have?

**Answered by** `candidate-figures.md`, for whether a real test set exists at all, and
`pairs-and-their-regimes.md` for whether a published reader error gives us something to predict.

**Option A: synthesise for an unseen pair, score a classifier.** Train the composition on pairs we
hold, compose a pair withheld entirely, train a classifier on the composed films, and score it on
real films of the withheld pair. This is the strongest claim and it needs a real held-out test set
of that pair, roughly a hundred films per class by the interval above.

**Option B: synthesise along the dial.** Same machinery, but the withheld pair is seen at 100%, 25%,
5%, 1% and 0%, and the question is where composition overtakes joint conditioning. Weaker claim,
measurable with the films we have, and it is the only version where composition's advantage is the
thing being measured rather than assumed.

**Option C: measure the drop, synthesise nothing.** Show on real pairs that the product of two
experts loses exactly what the regimes name, and that the correction term recovers it. No classifier
scorecard, no synthesis claim, and it is the version this repository's existing results already
nearly support.

**What decides it.** Whether any pair has about a hundred real both-disease films that could be held
out without destroying the training set. If none does, Option A is unmeasurable and saying so is the
honest scoping decision rather than a retreat.

**What this produces.** The chosen option in one sentence, with the count that made it possible or
impossible.

## Decision 3: the dataset

**The question.** Which films do we train and test on, and where does the regime label come from?

**Answered by** `candidate-figures.md` and `the-textbook-shelf.md`, plus our own disk.

**What we hold.** VinDr-CXR, 15,002 chest radiographs at `/datasets/mmolefe/vinbigdata/`, 161 GB,
every one of the 15,000 training films read by exactly three radiologists with boxes carrying the
reader's identity. Fourteen classes, no tuberculosis, no silicosis.

**What we would have to get, and what each one costs.** A tuberculosis set (TBX11K, Shenzhen,
Montgomery) brings the disease but not the pairing, because none of them labels a second disease.
MIMIC-CXR brings free-text reports, which is the only label source that can carry the merged and
hidden regimes, and it needs credentialing. CT brings the two regimes a radiograph cannot show, and
changes the modality the whole paper is about. The published figures bring reader-confirmed regimes
and are too few to train on at any licence.

**The label problem, stated plainly.** Two of the four regimes cannot be labelled from boxes. So
either the paper covers the two that can, or the label comes from reports, or a radiologist reads
films for us. Decide which, and say what the paper cannot claim as a result.

**What this produces.** The dataset, the label source per regime, and the sentence saying which
regimes we cannot label and therefore cannot measure.

## Decision 4: which pairs

**The question.** Which disease pairs do the experiments run on?

**Answered by** `pairs-and-their-regimes.md` and `one-pair-many-regimes.md`.

**Four things a chosen set has to satisfy.** At least two regimes documented in the literature for
that pair. Enough films in the chosen dataset to carry the interval above. One pair with a mechanism
and one without, so the comparison has a control rather than only cases. And at least one pair where
the second disease is confirmed by something other than the image, since that is the only way the
hidden regime can be scored at all.

**The set we can build today, with what each one is for.**

| Pair | Films we hold | What it is for |
|---|---|---|
| aortic enlargement + cardiomegaly | 1,972 in VinDr, 98% separate places | the control: two findings that do not interact, so a product over a partition should be enough |
| lung opacity + pulmonary fibrosis | 489 in VinDr after the double-label drop, 201 same region | the shared-region case, and the only honest one our own measurement found |
| pulmonary fibrosis + emphysema | none in VinDr; both are labels in NIH ChestX-ray14, count not yet taken | the pair with the published evidence, including the one measured combination-specific lesion |
| pleural effusion + atelectasis | both are labels in VinDr and in ChestX-ray14 | the pair whose shared region is documented and whose hidden case is prose-supported |
| silicosis + tuberculosis | none in any dataset we hold | the pair the work names, reachable only by argument from a proxy |

**The first count to take before this decision closes.** How many films in NIH ChestX-ray14 carry
both Emphysema and Fibrosis, and how many carry both Effusion and Atelectasis. Both label pairs
exist in that dataset's fourteen classes, and the counts decide whether the best-evidenced pair in
the literature is also a pair we can train on. Neither count has been taken.

**The decision this forces.** The pair the paper names has no films anywhere we hold, and the pairs
with films are not it. Either the paper is explicitly about proxy pairs and says so in its title, or
a tuberculosis dataset joins the work and the pairing is constructed rather than found. The report
adds a third way out: make combined pulmonary fibrosis and emphysema the pair, since it is the one
with both published evidence and a plausible dataset, and carry silicotuberculosis as the motivating
case rather than the measured one.

**What this produces.** The chosen pairs with their film counts, and for each one the sentence
carrying it to silicotuberculosis.

## Decision 5: the experiments

**The question.** What gets run, in what order, and what result would change the paper's mind?

**Answered by** every file, and by
[the composition chain already written](../../../plans/route-compose-x-rays-and-score-them.md), whose
first step reads one idea across six papers and picks the operator.

**The candidates, with the one axis each varies.**

**Joint conditioning on a single prompt.** The ceiling at 100% on the dial and the thing composition
has to beat below it. Not a contribution, a reference line.

**The product of two experts, uncorrected.** What PoE literally computes. The experiment is which
regime it fails in, measured per regime, which is the first thing the regimes are actually for.

**The product with the learned correction.** This repository's existing adapter, applied to disease
pairs instead of animal pairs. One axis differs from the line above, so the comparison is clean.

**Composition inside one conditional model**, in the FCFG shape, with the noise prediction built as
the unconditional one plus a weighted sum of each condition's difference from it
(https://arxiv.org/abs/2506.14399). It is the one published method that both composes a chest film
and scores a classifier on the result, and it is the comparison the paper cannot omit.

**Editing an existing film**, in the RadEdit shape: take a film with one disease and add the second.
It has no dial, because it never needs both-disease films, so it wins trivially at 0% and cannot be
compared along the axis the rest of the work lives on. Decide whether it is in the paper as a
different kind of method or out of it as a different question.

**What decides the order.** The cheapest comparison that could kill the premise goes first. If the
uncorrected product already composes two findings correctly in the separate-places regime, the
correction has nothing to do there and the paper narrows to the other three regimes before any
adapter is trained.

**What this produces.** The experiment list as a table of one row per run, each with the axis it
varies, the comparison it belongs to, the number that would support the claim, the number that would
be a null, and the band in between where the answer is inconclusive. That table is the thing
`/drip-idea --hypothesis` turns into a falsification criterion, one part per round, before anything
runs.

## What this walk hands over

A scoped problem with its claim written, a dataset with its label source per regime, a chosen set of
pairs with film counts, and an experiment table with a falsification criterion owed on every row.
That is enough to open a plan scope. Nothing in this file is a plan yet, and no task should be
written from it until the five decisions carry answers.
