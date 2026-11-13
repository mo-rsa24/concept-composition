# Route 01: an annotated radiograph for every overlap regime

Paste this whole file into a fresh session.

---

## The task

Build a reference set of real chest radiographs, each one annotated on the image, in which two
findings occur together, so that every overlap regime named below has at least three pictures
showing it and every picture says in its own caption which regime it is and where on the film to
look.

The set exists so that a compositional diffusion model of chest radiographs can be judged against
real combined films rather than against a description of them. A regime with no picture is a regime
we cannot check a sample against.

## What it starts from

**The films are already on our own disk.** VinDr-CXR sits at `/datasets/mmolefe/vinbigdata/` on
mscluster: 15,002 DICOM files under `train/`, 161 GB, named `<image_id>.dicom`, with every box in
`train.csv`. Every one of the 15,000 training films was read by exactly three radiologists, and each
box carries the `rad_id` of the reader who drew it. So every regime below is selected by a query
rather than searched for in the literature.

Read, in this order:

1. `artifacts/ideas/silicotb-conversation/IDEA_MAP.md`, claim 1.10 and the whole Runs table. Runs 3,
   5 and 6 are the measurements this route's regime assignments come from.
2. `3-understanding-compositionality/artifacts/ideas/poe-overlap-dial/composition-terms.md`, the composition vocabulary this project
   already uses. Use its words in every caption. Do not invent a parallel vocabulary.
3. `3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/radiograph-highlights/` and the
   `README.md` beside it: nine radiographs this repository already holds. They are a starting set,
   not the target set.

## The settled facts, with their sources

1. **Four regimes, not three.** The conversation's own figure
   (`artifacts/drips/silicotb-conversation/source/screenshots/three-overlap-regimes.png`) draws
   three. Ehrlich et al. 2024 (https://doi.org/10.1183/16000617.0168-2024) name three *effects*,
   which are a different cut, and one of them, each disease hiding the other, has no panel. The four
   regimes are kinds of region, not kinds of patient, so every caption names a region and never a
   whole film.

2. **Each regime now has a query that selects its films.** From `train.csv`, after dropping every
   film where one radiologist drew one box and attached both labels to it (which is a labelling
   habit and not an overlap):

   | Regime | How to select it | What that returns |
   |---|---|---|
   | separate lesions | aortic enlargement with cardiomegaly, boxes not overlapping | 1,951 films, 98% of that pair |
   | same region | lung opacity with pulmonary fibrosis, boxes at intersection over union ≥ 0.25 | 201 films, 41% of what is left of that pair |
   | merged presentation | the same pair at intersection over union ≥ 0.5, then judged by eye for whether the two findings are still separately nameable | 666 films before the drop; the eye is what splits same-region from merged, and no query can do it |
   | one finding hidden | a film where exactly one of the three readers marked a finding whose Fleiss' kappa is below 0.40, with another finding marked by all three | low-kappa findings are consolidation 0.355, other lesion 0.356, atelectasis 0.365, lung opacity 0.406 |

3. **The hidden-finding regime has a measured proxy, and it is kappa rather than the single-reader
   share.** Of the films where consolidation was marked at all, 65% were marked by one reader of
   three, and its Fleiss' kappa over all 15,000 films is 0.355. Pneumothorax also has a high
   single-reader share, 39%, but its kappa is 0.743, so for that finding the single-reader calls are
   rarity and not difficulty. Select on kappa. A film selected this way carries a radiologist's own
   disagreement about whether the finding is there, which is what the hidden regime is.

4. **Licence, which decides what may ship.** VinDr-CXR's own terms have not been read yet. Read them
   before any film goes into a committed figure or a paper, and record the exact licence string in
   `sources.json`. Until that is done, treat every VinDr film as look-only, same as NIOH's.

5. **The NIOH teaching set is the comparison, not the source.** Ten chest radiographs of silicosis
   and tuberculosis in silica-exposed gold miners, eight of them full-resolution DICOM, each with an
   expert reader's verdict in its filename and four of them reading "doubtful silicosis":
   https://www.nioh.ac.za/teaching-training/training-and-learning-repository-of-materials-on-the-radiology-of-silicotuberculosis/chest-radiographs-showing-features-of-silicosis-and-tuberculosis-and-tuberculosis-in-silica-exposed-individuals-2/
   The accompanying paper is Maboso et al. 2022, "Difficulties in distinguishing silicosis and
   pulmonary tuberculosis in silica exposed gold miners: a report of four cases", covering
   radiographs 3, 5, 9 and 10. Its footer reads "Copyright 2023 NIOH. All Rights Reserved" and
   states no reuse permission, so these films may be looked at and learned from and may not be
   committed, published or trained on. NIOH invites contact at info@nioh.ac.za; asking is task 6
   below. Use them to check that a VinDr film called "hidden" looks like a film a reader actually
   called doubtful.

6. **Radiopaedia is not usable for anything that ships.** Its licence is CC-NC-BY-SA 3.0 and
   non-commercial machine-learning use needs an application and a 250 USD fee. Use it to find which
   appearance to look for, never for a picture.

## What to produce

| Path | What right looks like |
|---|---|
| `artifacts/results/overlap-regimes-on-real-films/README.md` | one entry per picture, under a bold heading, saying which regime, which region of which film, which readers marked what, the query that selected it, and what its licence permits |
| `artifacts/results/overlap-regimes-on-real-films/<regime>-<pair>-<nn>.png` | the annotated film: the original, with a coloured outline on each region and the label written on the outline, never in a legend below |
| `artifacts/results/overlap-regimes-on-real-films/sources.json` | one record per picture: `image_id`, the pair, the regime, the query that selected it, each box with its `rad_id`, how many of the three readers marked each finding, the licence string, and whether reuse is granted |
| `artifacts/ideas/silicotb-conversation/routes/01-annotated-radiographs-for-each-regime.return.md` | what each regime now has, and which licences are unresolved |

## The target, as a number

**Twelve annotated pictures: three per regime, across at least three different finding pairs.** One
pair is not enough, because the whole point is that the regimes are a property of a region and not
of a disease.

**The hidden-finding regime is the one that decides whether this route succeeded.** It has zero
pictures today. Three is the bar. A film only counts for it when the thing that is hard to see can
be pointed at on the image: a box one reader drew that the other two did not, outlined, with a
sentence saying what in the surrounding lung is covering it.

## How to read a DICOM here

The VinDr files are `MONOCHROME1`, so the raw array displays inverted and has to be flipped before
anything is drawn on it. Open one with `pydicom`, flip it, window it, then draw. Install into a
throwaway virtual environment rather than the system Python, which refuses installs. The box
coordinates in `train.csv` are in the original pixel space, so any resize has to be applied to the
boxes too or the outlines land in the wrong place.

## How to annotate

Annotate the picture, do not describe it. One coloured outline per region, following the finding,
with the label written on or beside the outline in the image itself. One colour per regime, used the
same way across all twelve. No legend box, no caption-only annotation, no arrow pointing at an
unlabelled thing.

Where a film carries boxes from more than one reader, draw each reader's box in its own line style
and say in the card which reader drew which, because for the hidden regime the disagreement is the
content.

## How progress shows

Run this after each picture lands:

```
python3 - <<'PY'
import json, collections
recs = json.load(open("artifacts/results/overlap-regimes-on-real-films/sources.json"))
by = collections.Counter(r["regime"] for r in recs)
for k in ["separate", "same-region", "merged", "hidden"]:
    print(f"{k:14s} {by.get(k,0)}/3   pairs: {sorted({r['pair'] for r in recs if r['regime']==k})}")
print(f"pairs overall: {len({r['pair'] for r in recs})}/3")
print(f"unresolved licences: {[r['image_id'] for r in recs if not r['reuse_granted']]}")
PY
```

Done is all four rows at 3/3, pairs overall at 3 or more, and the unresolved-licence list either
empty or explicitly accepted as look-only.

## Limits

- Patient data lives only on the cluster under `/datasets/mmolefe/`. Every count and every render
  runs through `scripts/cluster.sh sh "<command>"` from
  `/home/molef/PhD/poe_repair_min-dog-and-rain` with
  `CLUSTER_DIR=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`, and the finished
  pictures come back only with `scripts/cluster.sh pull`.
- `artifacts/results/` is gitignored in this repository, so nothing here is committed. The card and
  `sources.json` are what make a picture re-findable.
- Do not download model weights onto a node without asking.
- Do not assign a regime by eye where the boxes contradict it. Where the query says same-region and
  the picture reads as two separate things, that is a finding about the measure: write it in the
  return file rather than relabelling the film.

## The hand-off

The return file feeds claim 1.10 of the silicotuberculosis walk and its outstanding checks. A regime
that finishes with three pictures across three pairs is ready for `/drip-idea --hypothesis`; a regime
that cannot be pictured at all is a finding about the regime, not a failure of the route, and the
return file says which it is.
