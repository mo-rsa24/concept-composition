# Return: annotated radiographs for each overlap regime

Feeds claim 1.5 of [the silicotuberculosis walk](../IDEA_MAP.md), the check named "1.5, the regimes
against the review's effects", and the proxy-pair check.

## What this run did

It assigned the four overlap regimes to films the repository already holds, and stopped before
sourcing new ones. Nothing was newly fetched for the set, and no picture has been annotated yet.
The assignment is in [the set's card](../../../results/overlap-regimes-on-real-films/README.md) and
in `sources.json` beside it, which the route's progress check reads.

## Where each regime stands

| Regime | Pictures | Bar | Pairs | Short by |
|---|---|---|---|---|
| separate lesions | 4 | 3 | 4 | nothing |
| same region | 2 | 3 | 1 | one picture of a second pair |
| merged presentation | 3 | 3 | 3 | nothing |
| one disease hidden | 1 | 3 | 1 | two pictures |

Nine disease pairs appear across the set against a bar of three, so the spread requirement is met
overall and fails only inside the same-region row.

## Two facts in the route that did not hold

**The NIOH films were not downloaded.** The route states they are already on disk. No DICOM and no
NIOH file existed anywhere in the repository. They have now been fetched to the session scratchpad
for looking only, because their licence forbids committing them.

**The starting set is larger and better documented than stated.** The route names nine films in
`radiograph-highlights/` with provenance in a README beside them. That folder holds seven composite
figures and no README. The provenance lives one folder across, in
[the medical examples card](../../../../3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-match-the-joint-for-each-kind-of-composition/medical-examples/README.md),
682 lines covering 29 films with each one's caption, licence and citation quoted. That card is why
this run needed no web search: the sourcing was already done, and only the regime assignment was
missing.

## What the regimes turned out to need

**Separate lesions and merged presentation are covered.** Both reached three pictures across three
or more pairs from films already held.

**Same region fails on spread, not on count.** Its two pictures are both a fungal ball inside an old
tuberculous cavity. A cavity inside a silicotic conglomerate mass is the natural second pair, and
the Martins 2010 film shows exactly that, but its cavitation is marked on the companion CT and not
on the radiograph, so assigning it would break the route's own rule against assigning a regime the
figure's caption does not support. Finding a radiograph whose caption names both the mass and the
cavity on the film itself is the one piece of searching this row needs.

**The hidden regime is where the route is genuinely blocked, and the reason is licensing.** The
best recorded instances of a reader seeing fewer diseases than were present are the four NIOH films
whose filenames read "doubtful silicosis" or "silicosis in previous tuberculosis". Those verdicts
are exactly the regime, written by an expert reader, and they cannot be used until NIOH grants
written permission. The one picture the set does have, a lung cancer read as pneumonia against
honeycombing until biopsy settled it, qualifies because two radiologists disagreed in print.

## A finding about the regimes, not about the search

The merged and hidden rows are mostly CT. Of the four merged and hidden pictures, three are axial
CT and one is a radiograph, and the radiograph is the case where the merging is geometric, a
pneumothorax taking its shape from an emphysematous lung. A lesion that neither disease builds alone
is largely a matter of texture, and plain film resolves texture poorly.

This bears on claim 1.5 directly. If two of the four regimes are not reliably visible on a plain
radiograph, then a compositional model trained on radiographs cannot be asked to produce them, and
the regimes that can be judged against real radiographs are separate lesions and same region. That
is a narrower claim than the walk's figure draws, and it is checkable: it predicts that any
radiograph-only search for merged or hidden examples keeps returning CT.

## What is still unresolved

- **Three annotated pictures do not exist for any row.** Every entry names its source film, and the
  coloured outline with its on-image label is still to be drawn. `annotated_file` is null in every
  record until then.
- **The NIOH permission request has not been written or sent.**
- **Same region needs one radiograph of a second pair** whose caption names both diseases on the
  film.
- **Hidden needs two more pictures.** The CPFE autopsy case at
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4585953, where upper-zone lung read as plain emphysema on
  CT held dense usual-interstitial-pneumonia fibrosis at post mortem, is the strongest unexamined
  candidate and was not fetched on this run.

## Hand-off

Separate lesions and merged presentation are ready for `/drip-idea --hypothesis` once their pictures
are annotated. Same region and hidden are not ready. Whether the hidden regime can be pictured on a
plain radiograph at all is a finding about the regime rather than a failure of the route, and the
medium split above is the evidence for it.
