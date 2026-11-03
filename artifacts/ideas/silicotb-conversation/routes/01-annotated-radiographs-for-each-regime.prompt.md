# Route 01: an annotated radiograph for every overlap regime

Paste this whole file into a fresh session.

---

## The task

Build a reference set of real chest radiographs, each one annotated, in which two lung diseases
occur together, so that every overlap regime named below has at least three pictures showing it
and every picture says in its own caption which regime it is and where on the film to look.

The set exists so that a compositional diffusion model of chest radiographs can be judged against
real combined films rather than against a description of them. A regime with no picture is a
regime we cannot check a sample against.

## What it starts from

The silicotuberculosis walk at `artifacts/ideas/silicotb-conversation/IDEA_MAP.md`. Claim 1.5 is
the claim this route serves. Read the claim, the "Checks outstanding" row named
"1.5, the regimes against the review's effects", and the "What the words are" table.

The composition vocabulary this project already uses is in
`artifacts/ideas/poe-overlap-dial/composition-terms.md`. Read it before writing any caption, and
use its words. Do not invent a parallel vocabulary.

Nine radiographs that already exist in this repo are in
`artifacts/results/joint-against-poe-by-kind-of-composition/radiograph-highlights/`, with their
provenance in the `README.md` beside them. They are the starting set, not the target set.

## The settled facts, with their sources

1. **The combined condition is a disjunction.** Silicotuberculosis is "the combination of
   silicosis and active TB and/or post-tuberculous lung disease" (Ehrlich et al. 2024,
   https://doi.org/10.1183/16000617.0168-2024, Definitions). The review is quoted at section
   level in `artifacts/ideas/poe-overlap-dial/routes/02-findings-and-diagnoses-literature.return.md`.
   Read that return file before fetching the review, which refuses automated fetches.

2. **Four regimes, not three.** The conversation's own figure
   (`artifacts/drips/silicotb-conversation/source/screenshots/three-overlap-regimes.png`) draws
   three. Ehrlich et al. name three *effects* which are a different cut, and one of them, each
   disease hiding the other, has no panel. The four this route must cover are:

   | Regime | What the film shows | Where it is already pictured |
   |---|---|---|
   | separate lesions | both diseases present, in different parts of the lung, each looking as it looks alone | `tuberculosis-silicosis-silicotuberculosis.png`, right panel, away from the cavity |
   | same region | both diseases in the same region, each still recognisable | the nodules ringing the cavity in the same film |
   | merged presentation | a lesion neither disease builds alone | `disease-hidden-by-a-changed-lung.png`, left panel: pneumonia in emphysema |
   | one disease hidden | the overlap shows **fewer** findings than either disease alone; the second disease is present and unreadable | **nothing in this repo pictures this yet** |

3. **All four can appear in one patient.** Martins et al. 2010
   (https://pmc.ncbi.nlm.nih.gov/articles/PMC2929613/) report one silicotuberculosis HRCT
   carrying nodules away from the masses, nodules fused into masses, and a cavity inside a mass.
   In their cited series of 44 silicosis patients with conglomerate masses, 8 masses cavitated
   and 6 of those 8 had tuberculosis. So the regimes are kinds of region, not kinds of patient,
   and a caption must name a region, never a whole film.

4. **The NIOH teaching repository is the best source found so far, and it is already downloaded.**
   The National Institute for Occupational Health (South Africa) publishes ten chest radiographs
   of silicosis, tuberculosis and combined disease in silica-exposed gold miners, eight of them
   as full-resolution DICOM, each carrying an expert reader's verdict in its filename:
   https://www.nioh.ac.za/teaching-training/training-and-learning-repository-of-materials-on-the-radiology-of-silicotuberculosis/chest-radiographs-showing-features-of-silicosis-and-tuberculosis-and-tuberculosis-in-silica-exposed-individuals-2/

   Four of the ten are labelled "doubtful silicosis" or "silicosis in previous tuberculosis",
   which is the hidden-disease regime recorded by a reader. The accompanying paper is Maboso et
   al. 2022, "Difficulties in distinguishing silicosis and pulmonary tuberculosis in silica
   exposed gold miners: a report of four cases", which covers radiographs 3, 5, 9 and 10. Read
   it for the per-case reasoning behind each verdict.

   **The licence is the blocker.** The repository footer reads "Copyright 2023 NIOH. All Rights
   Reserved" and states no reuse permission. The films may be looked at and learned from. They
   may not be committed to this repository, published in a paper figure, or put in a training
   set, until written permission exists. NIOH invites contact at info@nioh.ac.za. Asking is a
   task in this route, not an afterthought.

5. **Radiopaedia is not usable for anything that ships.** Its licence is CC-NC-BY-SA 3.0,
   non-commercial only, and non-commercial machine-learning use needs an application and a
   250 USD fee. It is already recorded as a dead end in the walk's map for exactly this reason.
   Use it to find *which* appearance to look for, then source the picture elsewhere.

## What to produce

| Path | What right looks like |
|---|---|
| `artifacts/results/overlap-regimes-on-real-films/README.md` | one entry per picture, under a bold heading, saying which regime, which region of which film, who reported it, where it came from, and what its licence permits |
| `artifacts/results/overlap-regimes-on-real-films/<regime>-<pair>-<nn>.png` | the annotated film: the original, with a coloured outline on each region and a label on the outline, never in a legend below |
| `artifacts/results/overlap-regimes-on-real-films/sources.json` | one record per picture: source URL, licence string, whether reuse was granted, the reader's verdict verbatim, the pair of diseases, the regime, and the pixel box of each annotated region |
| `artifacts/ideas/silicotb-conversation/routes/01-annotated-radiographs-for-each-regime.return.md` | what was found, what each regime now has, and which licences are still unresolved |

## The target, as a number

**Twelve annotated pictures: three per regime, across at least three different disease pairs.**
Silicotuberculosis alone is not enough, because the whole point of a proxy pair is that the
regimes are not specific to one disease. Combined pulmonary fibrosis and emphysema is the
strongest second pair (see below). A third pair is the route's own choice.

**The hidden-disease regime is the one that decides whether this route succeeded.** It has zero
pictures today. Three is the bar. If after a full search it still has fewer than three, say so
plainly and say what was searched, rather than promoting a weak example to fill the slot.

## Where to look, in order

1. **The NIOH repository**, already downloaded, read first. The four doubtful-silicosis films
   are the hidden-disease candidates.
2. **PubMed Central open-access case reports.** Filter for CC-BY. Searches that have already
   worked: "cavitated conglomerate mass silicosis tuberculosis", "combined pulmonary fibrosis
   and emphysema CT upper lobe emphysema lower lobe fibrosis". The CPFE autopsy case at
   https://pmc.ncbi.nlm.nih.gov/articles/PMC4585953 is the strongest published hidden-disease
   case found so far: upper-zone lung read as plain emphysema on CT held dense usual-interstitial-
   pneumonia fibrosis at post mortem.
3. **The Radiology Assistant** (https://radiologyassistant.nl/chest/chest-x-ray/lung-disease) for
   the pattern vocabulary and for which appearances to search for. Check its terms before reusing
   any image.
4. **Radiopaedia**, for finding the appearance only, never for a picture that ships.
5. **NIH ChestX-ray14**, which this project already holds, for pairs with enough films to measure
   on. The co-occurrence counts are in
   `artifacts/results/which-findings-arrive-after-which/cooccurrence.json`: effusion with
   atelectasis 3,275 films, consolidation with effusion 1,287, cardiomegaly with effusion 1,063,
   pneumonia with emphysema 23. Its labels are report-derived and carry no regions, so it can
   supply films but never an annotation.

## How to annotate

Annotate the picture, do not describe it. For each region: a coloured outline following the
lesion, and the label written on or beside the outline in the image itself. One colour per
regime, used consistently across all twelve. No legend box, no caption-only annotation, no
arrows pointing at unlabelled things.

Where a published figure is already annotated by its authors, keep their annotation and add only
the regime label, and say in the card that the original marks are theirs.

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
print(f"unresolved licences: {[r['source_url'] for r in recs if not r['reuse_granted']]}")
```

Done is all four rows at 3/3, pairs overall at 3 or more, and the unresolved-licence list either
empty or explicitly accepted as look-only.

## Limits

- Patient data lives only on the cluster under `/datasets/mmolefe/`. Any count or run against
  NIH films goes through
  `scripts/cluster.sh sh "<command>"` with
  `CLUSTER_DIR=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`, and figures
  come back only with `scripts/cluster.sh pull`.
- Nothing whose licence forbids it gets committed to this repository. When in doubt, put the file
  in the session scratchpad, record its URL in `sources.json`, and say in the card that the
  picture must be re-fetched rather than read from disk.
- Do not download model weights on a node without asking first.
- Do not invent a regime assignment. If a published figure's own caption does not support the
  regime, either find the sentence in the paper's body that does, or leave the picture out.

## The hand-off

The return file feeds claim 1.5 of the silicotuberculosis walk, the "1.5, the regimes against the
review's effects" check, and the proxy-pair check. A regime that finishes with three pictures
across three pairs is ready for `/drip-idea --hypothesis`; a regime that cannot be pictured at all
is a finding about the regime, not a failure of the route, and the return file says which it is.
