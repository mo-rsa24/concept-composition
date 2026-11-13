# Route 05: what the literature knows about two chest diseases in one chest

Paste this whole file into ChatGPT with deep research turned on. It is written for a researcher who
has none of our files.

---

## The task

Find out what the medical and radiology literature already knows about two chest diseases occurring
together in one patient, and in particular whether the four ways they can sit relative to each other
(set out below) are documented for disease pairs other than silicosis with tuberculosis.

Return six markdown files. Their contents will be read one claim at a time and argued over, so every
claim is one sentence carrying one source, and a claim nobody published is marked as ours rather
than written as if it were established.

## Why this exists

We are building generative models that synthesise a chest image showing two diseases at once, for
disease pairs too rare for anyone to have collected films of. To judge a synthesised image we need
to know what a real image of that pair looks like, and we have found that "two diseases in one
chest" is not one thing. Silicosis with tuberculosis gave us four ways it can go. Whether those four
are a property of that pair or of disease pairs in general decides whether our model has one target
or many, and nothing we have read answers it.

## The four ways two diseases can sit in one chest

These are ours, built from one review paper and nine published figures. They describe a region of
lung, not a whole patient: one chest can hold more than one of them at once.

1. **Separate places.** Each disease sits in its own part of the chest and looks exactly as it would
   alone. A reader names both, and neither changes the other's appearance. Example: a tuberculous
   cavity in the right upper zone and a large silicotic opacity in the left.

2. **Same region, both still readable.** Both diseases affect the same tissue and a reader can still
   name both. Example: a fungus ball growing inside a cavity that healed tuberculosis left behind.
   The cavity and the ball occupy one place and both are separately identifiable.

3. **Merged, neither nameable alone.** The two together produce an appearance neither produces
   alone, and a reader cannot decompose it. Example: tuberculosis making discrete silicotic nodules
   conglomerate into a mass, where the mass belongs to both diseases and to neither.

4. **One disease present and unreadable.** Both diseases are there, confirmed by something other
   than the image (culture, biopsy, a later film), and the image shows only one of them, because the
   other has destroyed or buried the lung that would have shown it. Example: a cancer growing inside
   honeycombed fibrosis, read as pneumonia until biopsy.

## What we already hold, so you need not re-find it

Each of these is already read and quoted in our own files. Finding them again is not useful; finding
what they do not cover is.

| What it gives us | Source |
|---|---|
| three effects of tuberculosis on a silicotic chest in prose (each hides the other, each distorts the other's pattern, together they build a lesion neither builds alone), with no figure for the hiding effect | Ehrlich et al. 2024, *European Respiratory Review*, https://doi.org/10.1183/16000617.0168-2024 |
| a mineworker's radiograph with a silicotic mass and a tuberculous cavity circled by the authors, GeneXpert positive | Pyana Kitenge et al. 2025, *IJTLD Open*, https://doi.org/10.5588/ijtldopen.24.0564, Fig. 1 Case 2 |
| chronic silicosis with conglomerate masses on radiograph, cavitation visible only on CT, culture positive, and the warning that a cavity in a conglomerate mass can also be ischaemic necrosis without infection | Martins et al. 2010, https://pmc.ncbi.nlm.nih.gov/articles/PMC2929613/ |
| tuberculosis risk in silicosis 2.8 to 39 times higher than in controls | Martins et al. 2010, Discussion |
| a fungus ball appearing in an old tuberculous cavity five years later, radiograph and CT | Pattamapaspong et al. 2024, *BJR Open*, https://doi.org/10.1093/bjro/tzae031, Fig. 2 |
| pneumonia in an emphysematous lobe, consolidation as a net of strands around black holes rather than a solid patch | Ragnoli et al. 2026, *Diagnostics*, https://doi.org/10.3390/diagnostics16091329, Fig. 2A |
| squamous cell carcinoma grown within honeycombed fibrosis, read as pneumonia by the first reader | Kim et al. 2026, *Journal of Clinical Medicine*, https://doi.org/10.3390/jcm15176863, Fig. 2a |
| an acute exacerbation laid over pulmonary fibrosis, before-and-after CT, infection clinically excluded | Hobbs et al. 2021, *Radiology: Cardiothoracic Imaging*, https://doi.org/10.1148/ryct.2021200279, Figs. 17a and 17b |
| COPD with an old calcified granuloma, two diagnoses with no shared cause | Ibraheem et al. 2024, *Cureus* 16(2):e54343, https://doi.org/10.7759/cureus.54343, Fig. 1 |
| a four-panel CT figure naming simple silicosis, complicated silicosis, active tuberculosis and post-tuberculous lung disease as four separate appearances | unattributed figure from a saved conversation; finding its real source is task 7 below |

We also hold annotation data, not literature: VinDr-CXR, 15,000 chest radiographs each read by three
radiologists with boxes. From it we measured that aortic enlargement with cardiomegaly is in
separate places on 98% of the 1,972 films carrying both, and that lung opacity with pulmonary
fibrosis shares a region on 41% of 489 films at an intersection-over-union of 0.25 or more. So the
first two ways above are measurable on annotation boxes and the last two are not.

## The six files to return

### 1. `pairs-and-their-regimes.md`

One section per disease pair. For each pair, in this order:

- **What each disease looks like on a chest radiograph alone**, in plain words a non-radiologist can
  follow, with the source that says so.
- **What changes when both are present**, again in plain words, with the source.
- **Which of the four ways above is documented for this pair**, each one either supported by a named
  figure and its own caption, or marked "not documented". A regime you infer from prose rather than
  from a figure is marked as inferred, with the sentence you inferred it from quoted.
- **Whether a reader error is reported**, and if a number is published (missed-diagnosis rate, delay
  in months, reader agreement), the number with its denominator.

Cover at least these pairs, and add any others the literature treats as a named combination:

silicosis + tuberculosis · HIV + tuberculosis · pneumonia + emphysema · pleural effusion +
atelectasis · pulmonary fibrosis + emphysema · pulmonary fibrosis + lung cancer · old tuberculosis +
aspergilloma · heart failure + pneumonia · COPD + pneumonia · diabetes + tuberculosis · silicosis +
lung cancer · asbestosis + tuberculosis · coal workers' pneumoconiosis + tuberculosis · COVID-19 +
tuberculosis · sarcoidosis + tuberculosis

HIV with tuberculosis matters most of the four named in our own question, because the published
claim is that advanced HIV changes tuberculosis from an upper-zone cavitating disease into something
that looks different or shows nothing at all. If that is right it is the fourth way above, happening
through immune state rather than through destroyed lung, and it is a second mechanism for the same
regime. Say whether it is right, with the study and its numbers.

### 2. `the-words-the-field-uses.md`

Our four ways, and our own vocabulary below, mapped onto terms that already exist.

For each of the four, give every term the literature uses for it, with the specialty that uses it
(chest radiology, occupational medicine, infectious disease, machine learning on chest X-rays) and
one source per term. Then say whether any single source names all four, or three, or two, and which.
If nobody has named the fourth, say so plainly; that is a useful answer.

Our vocabulary, for the same treatment:

- **Comorbidity**, for us: two diagnoses on one film with no shared cause and no effect on each
  other's appearance.
- **Superimposed disease**, for us: two diseases acting on the same tissue so each changes how the
  other looks.
- **Co-occurrence**: both diseases present, each in its own region.
- **Intersection**: both diseases claiming the same pixels.
- **Interaction**: the pair producing something neither produces alone.

For each, say whether the term is standard, whether it means something else in standard use, and
what the field's own word is. We already know "comorbidity" is Feinstein's 1970 term and is broader
than our use of it, and that "compound disease" is not a radiology term.

### 3. `the-textbook-shelf.md`

Radiology textbooks, teaching websites and review articles that teach two diseases on one film as a
topic in its own right, rather than one disease at a time.

One row per source: title, edition, year, chapter title, page numbers, and one sentence saying what
it teaches about the combined case. Include teaching collections with annotated films (Radiology
Masterclass, Radiopaedia, STATdx, Learning Radiology, the Royal College collections) and say for
each whether its images can be reused and under what terms.

At least six books with chapter and page numbers. A book you cannot confirm a page number for is
listed with "page not confirmed" rather than a guess.

Also answer directly: is there a textbook chapter anywhere on how to read a film when two diseases
are present? If the answer is that radiology teaching is organised one disease at a time, say that,
with what you searched to establish it.

### 4. `one-pair-many-regimes.md`

The question that prompted this: can one disease pair show different ways at different times, or in
different parts of one chest, in the same patient?

Silicosis with tuberculosis is the test case, because tuberculosis has two states. Active
tuberculosis builds a cavity, which sits in silicotic lung and is a shared region. Healed
tuberculosis leaves scarring and distortion, which may be in a different place and may be
indistinguishable from the silicosis around it. So the same named disease pair may be in the second
way while the tuberculosis is active and in the fourth once it has healed.

Find out whether anyone has documented that, and whether the same happens for other pairs. What we
want back:

- Published cases where one patient's film shows two or more of the four ways at once, in different
  parts of the chest, with the figure and the authors' description of each part.
- Published series of the same patient over time where the way the two diseases sit changes between
  films, with the interval and what changed.
- Whether disease stage is what does it (active against healed tuberculosis, acute against chronic
  anything), or whether it is where in the lung the two happen to land.
- Any pair where all four ways are documented, which would be the single best example to build our
  work around.

### 5. `candidate-figures.md`

A table of published figures we could annotate and reuse. One row each:

| column | what goes in it |
|---|---|
| figure | the paper, the figure number, the panel |
| link | a direct URL to the article or the figure |
| licence | the exact licence string (CC BY 4.0, CC BY-NC-SA 3.0, "all rights reserved", or "could not determine") |
| pair | the two diseases |
| way | which of the four, and whether the authors' own caption supports it |
| author marks | whether the authors drew arrows, circles or letters on the image, and what they point at |
| proof | what establishes the second disease: culture, biopsy, a later film, or the image alone |
| modality | chest radiograph or CT |

At least 25 rows, weighted toward the third and fourth ways, which is where we have almost nothing.
Open-licensed figures only for anything we would reuse; a paywalled figure may appear in the table
marked as look-only.

One thing to watch for, because it changes what we can claim: our own set found that the first two
ways appear on plain radiographs and the third and fourth appear only on CT. If that holds in your
search, say so; if you find a radiograph whose own caption describes a merged appearance, that
single figure is the most valuable thing this whole report can return.

### 6. `where-the-four-regimes-fail.md`

Attack the four ways. What we want is the case they cannot describe.

- A documented pair of chest diseases that fits none of the four.
- A case where the four are the wrong question, because the two diseases interact through something
  other than the image (treatment, timing, the patient's immune state).
- Any published scheme that divides the combined case differently, with its divisions and its
  source, whether from radiology, pathology, or machine learning on chest X-rays.
- Whether the machine-learning literature on multi-label chest X-ray classification has its own
  account of why co-occurring findings are hard, with the papers and any measured drop in accuracy
  when two labels appear together rather than alone.

A fifth way, if you find one, is the single most useful finding in this file. Say what it is, what
documents it, and why the four do not cover it.

## What right looks like, and what wrong looks like

**A claim with no source is wrong.** Every sentence that asserts something about a disease, a film
or a reader carries a citation with enough detail to find the exact passage: author, year, journal
or book, and the figure number, page number or section heading.

**A regime assignment is supported by what the authors wrote, not by what the image looks like to
you.** Where you assign a regime from prose, the sentence is quoted. Where you assign it from a
caption, the caption is quoted. Where you assign it by eye, say so and mark it as unsupported.

**"Not documented" is a finding and gets written.** An empty cell in the pairs table is more useful
than a plausible guess, because we will go and look for it.

**A number carries its denominator.** Not "readers often miss it" but "the first reader called it
pneumonia in 1 of 1 cases" or "missed in 34 of 211 films".

**Two diseases present is not the same as two diseases visible.** Keep them apart everywhere. The
fourth way exists only because they come apart.

## How much is enough

| File | Enough |
|---|---|
| `pairs-and-their-regimes.md` | 15 pairs, each with a verdict in all four cells |
| `the-words-the-field-uses.md` | every one of the four and the five vocabulary entries mapped or declared unmapped |
| `the-textbook-shelf.md` | 6 books with chapter and page, plus a direct answer on whether the chapter exists |
| `one-pair-many-regimes.md` | 3 documented cases of one pair showing two or more ways |
| `candidate-figures.md` | 25 rows, at least 6 of them in the third or fourth way |
| `where-the-four-regimes-fail.md` | a direct yes or no on a fifth way, with what you searched |

Close the report with a self-audit table: one row per file, the number above, the number you
reached, and one line on what you could not find and what you searched for it.

## Limits

- No patient-identifying material of any kind, and nothing from a source that looks like it was
  published without consent.
- Do not reproduce a paywalled figure. Link it and describe it.
- Never invent a DOI, a page number or a figure number. "Could not confirm" everywhere you cannot.
- Do not translate a non-English source's clinical claim without saying it is translated.
- Where two sources disagree, write both and say they disagree. Do not pick one.

## One more thing to settle

**Task 7: find the real source of the four-panel CT figure.** We hold a four-panel CT figure whose
panels are labelled simple silicosis, complicated silicosis with progressive massive fibrosis,
active pulmonary tuberculosis, and post-tuberculous lung disease. It came from a conversation with
no citation. Find where it is from, and say whether its four panels are as we read them: two of
silicosis without tuberculosis and two of tuberculosis without silicosis, so that none of the four
panels shows the combined disease. If the original figure is making a different point, say what it
is.
