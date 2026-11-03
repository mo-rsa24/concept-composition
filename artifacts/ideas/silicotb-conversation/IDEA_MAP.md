# 💡 Every claim in the silicotuberculosis conversation, marked

## Position in the idea

| Turn | Claims | Marked | Mark spread |
|---|---|---|---|
| **1 (current)**: TB then silicosis, how each changes the other | 7 | 2 | 1 wrong as stated, 1 needs a check |
| 2: where AI fails, and what the label should be | 18 | 0 | all open |
| 3: public-data pairs with the same three regimes | 15 | 0 | all open |
| 4: pairs that genuinely interact | 11 | 0 | all open |
| 5: product of experts, the interaction term, inclusive labels | 21 | 0 | all open |
| 6: multi-label LoRA on cats and dogs | 9 | 9 | routed, not walked here |
| 7: is there a mechanism or a context for cats and dogs | 12 | 0 | all open |
| 8: does inclusive training generalise across seeds and pairs | 12 | 0 | all open |
| **total** | **105** | **11** | |

Load-bearing: claim 5.7, that a composition which divides out the base image distribution equals
the true joint whenever the two labels are conditionally independent given the whole image. Every
turn-5, turn-7 and turn-8 claim about whether a learned correction is needed at all hangs off it,
and this repo's own findings can speak to it.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [The claims](#the-claims)
- [What the words are](#what-the-words-are)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Routes](#routes)
- [Sources](#sources)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

A saved ChatGPT conversation argues that two chest diseases occurring together break classifiers
that each work alone, that public datasets carry pairs with the same structure, and that a learned
correction to a product of two generative experts could synthesise a disease pair nobody has data
for. This walk marks every claim in it as holding, needing a named check, wrong as stated, or dead.

**Where the walk is**

Turn 1, claim 1.2, round 1. Claim 1.1 is marked wrong as stated and repaired into 1.7; turn 6 is
routed elsewhere.

**What compile would produce today**

Nothing worth producing: 10 of 105 claims carry a mark, and nine of them are the routed turn.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

The conversation runs from medicine to generative modelling in eight turns. Turns 1 to 4 claim that
silicosis and tuberculosis together produce three imaging regimes (lesions in separate places, in
the same place, and merged beyond decomposition), that classifiers fail on the combined case through
mimicry, masking or interaction, and that public chest X-ray sets hold pairs with the same shape:
cardiomegaly with effusion, effusion with atelectasis, pneumonia with fibrosis, fibrosis with
emphysema. Turns 5 to 8 recast the same problem as composition in a diffusion model: whether the
product of a tuberculosis expert and a silicosis expert is silicotuberculosis, what the product
drops, and whether a low-rank adapter can learn the dropped piece well enough to transfer to a pair
withheld from training.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

A claim is ticked when it carries a mark that is not `open`. Numbering is turn-first and never
renumbered; a claim found mid-walk is appended within its turn and said out loud.

### 1. 🔍 TB then silicosis, how each changes the other  ← current

The saved answer text for this turn is empty: only the question and three screenshots survive. Its
claims are therefore the question's own premises plus what the three images assert.

- [x] 1.1 contracting silicosis after TB changes how the silicosis presents on a radiograph — **wrong as stated**
- [ ] 1.2 the combined condition is properly called silicotuberculosis
- [ ] 1.3 the posteroanterior film shows scarring on one side and progressive massive fibrosis on the other
- [ ] 1.4 CT separates the combined condition into four distinguishable patterns (`source/screenshots/four-ct-patterns.png`)
- [ ] 1.5 the three overlap regimes are as drawn: separate lesions, same-region lesions, merged presentation (`source/screenshots/three-overlap-regimes.png`)
- [ ] 1.6 combined fibrosis and emphysema has the CT appearance shown (`source/screenshots/cpfe-ct-example.png`), which belongs to turn 4's content, not turn 1's
- [x] 1.7 what the order of onset was supposed to carry is carried instead by two films of the same patient at different times: a rapid rise in how many opacities there are, in their size, or a new cavity, is what favours tuberculosis — **needs a check** *(appended round 2, the repair of 1.1)*
- [ ] 1.2 the combined condition is properly called silicotuberculosis  ← current

**1.7, settled.** Mark: needs a check, on the asymmetry ratio in Checks outstanding. The axis is
real (9,167 of 13,302 repeat-film patients change label) and the conversation's own pair is not
(12 patients one way, 8 the other). Nothing downstream depends on it: every design in turns 2 to 8
treats the pair as symmetric, so 1.7 is an axis the conversation could gain, never one it needs.

**1.1, settled.** Mark: wrong as stated. Every source found defines the combined condition by
coexistence in any sequence, and includes old tuberculosis damage, so the ordering the claim turns
on is not part of the definition. No source says the appearance depends on which disease arrived
first. The repair is 1.7: the discriminating signal the conversation reaches for is time, not
order. Cost of the repair: none downstream. The symmetric treatment in turns 2 to 8 (four
evaluation groups, the interaction term, the adapter) loses nothing, and what it gains is a
second time point per patient as an axis nobody had named.

### 2. ⬜ Where AI fails, and what the label should be

- [ ] 2.1 a 2020 study of 330 chest X-rays and four AI systems found that adding silicotuberculosis films reduced each system's ability to tell the single diseases apart
- [ ] 2.2 a 2022 study of 501 films and two commercial systems reached AUC 0.84 to 0.89 against expert reads of radiographically defined silicotuberculosis
- [ ] 2.3 that study's combined score was the mean of the TB and silicosis scores, which is not an estimate of both diseases being present
- [ ] 2.4 its inter-reader agreement for the combined condition was Cohen's kappa 0.320 and 0.384
- [ ] 2.5 a 2026 study of about 2,000 Southern African miners' films found local and international sets differ, and halving resolution cost silicosis detection up to five percentage points while barely touching TB
- [ ] 2.6 the premise that South Africa has no benchmark and no usable classifier for the combined condition is too strong to defend
- [ ] 2.7 classifier failure on combined disease does not by itself establish biological interaction as the cause
- [ ] 2.8 mimicry: silicotic nodules and progressive massive fibrosis can read as TB, and TB scarring can read as silicosis, without the second disease being present
- [ ] 2.9 masking: fibrosis developing over silicotic nodules makes them unrecoverable on a two-dimensional film, so no classifier can find them
- [ ] 2.10 interaction: TB is associated with progressive massive fibrosis developing, and TB inflammation in silica-exposed lungs produces a stronger fibrotic response
- [ ] 2.11 the combined condition is defined clinically by coexistence, including post-TB lung damage, and requires no new radiological feature
- [ ] 2.12 the label should be three bits (silicosis, active TB, post-tuberculous lung disease) with the combined condition as silicosis and either TB state
- [ ] 2.13 a radiologist reading a film cannot establish microbiologically active TB
- [ ] 2.14 a CNN false positive is a classification error, and calling it a hallucination misuses a term that belongs to generative models
- [ ] 2.15 only a four-group evaluation (neither, TB alone, silicosis alone, both) with a sensitivity difference per classifier separates whether failure happens from why
- [ ] 2.16 multiplying two independent classifiers' scores assumes conditional independence that should not be assumed here
- [ ] 2.17 joint multi-task training can transfer negatively, so it is not guaranteed to beat two separate specialists
- [ ] 2.18 the first two hypotheses (the gap exists, the gap tracks visual mechanisms) must be answered before a new architecture is justified

### 3. ⬜ Public-data pairs with the same three regimes

- [ ] 3.1 the same research questions can be asked on NIH ChestX-ray14, PadChest and VinBigData with no silicotuberculosis set at all
- [ ] 3.2 co-occurrence, visual interference and pathological interaction are three different research problems, not one
- [ ] 3.3 cardiomegaly with effusion serves the first two problems and is weak for the third
- [ ] 3.4 effusion with atelectasis is the stronger analogue for the third, because a large effusion physically compresses lung and causes the atelectasis
- [ ] 3.5 the dataset table is right: NIH 112,120 films and no access friction, VinBigData 18,000 with boxes behind Kaggle terms, PadChest over 160,000 by request
- [ ] 3.6 the PhysioNet VinDr-CXR release and the Kaggle competition version are separate datasets reached by separate routes
- [ ] 3.7 NIH's labels were extracted automatically from reports and are noisy
- [ ] 3.8 NIH single-label counts are 13,317 effusion, 11,559 atelectasis, 2,776 cardiomegaly
- [ ] 3.9 a 2026 paper studies how adding one finding to a film changes predictions of other findings, and found cardiomegaly overpredicted when oedema was present
- [ ] 3.10 because that paper exists, showing that one finding shifts another's prediction is not a novel contribution in 2026
- [ ] 3.11 a model can score effusion high from an enlarged heart alone, which is a shortcut that overcalls effusion in cardiomegaly and undercalls it without
- [ ] 3.12 four further pairs are worth investigating: oedema with emphysema, pneumothorax with effusion, consolidation with atelectasis, and the air-fluid level of combined air and fluid is a genuinely distinct combined appearance
- [ ] 3.13 a co-labelled failure on NIH cannot support a claim of biological interaction, and two overlapping boxes in VinBigData show projected overlap only
- [ ] 3.14 the pasted pandas snippet gives the four-group distribution from the metadata file alone, with no images downloaded
- [ ] 3.15 splitting by patient and keeping both isolated and combined findings in some baseline's training data is what stops the evaluation becoming an artificial zero-shot task

### 4. ⬜ Pairs that genuinely interact

- [ ] 4.1 pneumonia with fibrosis, pneumonia with emphysema, fibrosis with emphysema and HIV with TB are all closer analogues than cardiomegaly with effusion
- [ ] 4.2 advanced HIV immunosuppression changes TB's radiographic appearance: less cavitation, more lymphadenopathy, more miliary pattern, more lower-zone involvement, more normal-looking films despite active disease
- [ ] 4.3 the machine-learning statement of that is that the image distribution given TB differs by HIV status, and HIV itself cannot be read off a chest film
- [ ] 4.4 NIH labels neither TB nor HIV and the VinBigData competition labels neither, so the HIV-and-TB experiment cannot be built from their released labels
- [ ] 4.5 pneumonia on a fibrotic lung is the closest practical analogue, and infection there is hard to separate from an acute exacerbation of the fibrosis
- [ ] 4.6 NIH carries both a pneumonia and a fibrosis label, but the fibrosis label is nonspecific, the pneumonia label is report-derived, and the joint count needs checking before committing
- [ ] 4.7 combined pulmonary fibrosis and emphysema is an established syndrome whose published imaging patterns match the three regimes, so the regime framework does not have to be invented
- [ ] 4.8 that pair shares causes, smoking among them, so it is not two independently caused diseases the way TB and silicosis are
- [ ] 4.9 the full VinDr-CXR release carries 28 labels including tuberculosis, pneumonia, emphysema and fibrosis, and not all of them appear in the competition version
- [ ] 4.10 a public set of more than 25,000 chest CT studies with reports and multi-abnormality annotations exists behind a Hugging Face account and terms
- [ ] 4.11 the number of both-positive patients has to be inspected before either track is chosen, because 112,000 films can still hold too few of one combination

### 5. ⬜ Product of experts, the interaction term, inclusive labels

- [ ] 5.1 the normalised product of two conditional distributions means "images both experts consider probable", which is not the clinical population of patients with both diseases
- [ ] 5.2 in the four-image cat-and-dog table the exact product puts all its mass on the one image with positive probability under both experts
- [ ] 5.3 if the two experts' supports are disjoint the normalising constant is zero and no sampler can repair the missing overlap
- [ ] 5.4 Du et al. (arXiv 2302.11552) show ordinary reverse diffusion does not sample the specified product, and their MCMC work addresses the sampling error rather than the choice of target
- [ ] 5.5 sampling error and distributional error are separate obstacles, and an exact product sampler fixes only the first
- [ ] 5.6 the raw product carries the base image distribution twice, so dividing by it once is the better-founded composition
- [ ] 5.7 when the two labels are conditionally independent given the whole image, the base-divided composition equals the true joint conditional exactly, and no interaction term is needed  **(load-bearing)**
- [ ] 5.8 the interaction factor is the ratio of the joint label probability given the image to the product of the two single label probabilities, and its gradient is the score correction that turns the composed score into the joint score
- [ ] 5.9 that factor measures conditional statistical dependence between the labels, not biological interaction: it can be 1 while the diseases genuinely interact, because the interaction is already inside the image distribution
- [ ] 5.10 noising a product is not the product of the noised pieces, and the interaction factor can be nontrivial at a noisy step even when it is exactly 1 on clean images
- [ ] 5.11 the correction to the user's Stable Diffusion account: the autoencoder makes the latent, the denoiser cross-attends to text, and the objective is noise prediction rather than per-word matching
- [ ] 5.12 "a photo of silicoTB" and "a photo of silicosis and TB" are not guaranteed to be the same conditional, and a general-domain text encoder may represent the compound term poorly
- [ ] 5.13 captions that are mutually exclusive teach the model the distribution of TB-only films, not the distribution of films from patients who have TB
- [ ] 5.14 including combined films under each single condition and dropping condition components at random fixes that, and it needs combined examples to exist
- [ ] 5.15 two single-label probabilities of 0.5 each leave the joint anywhere from 0 to 0.5, so the relationship between the conditions is unidentified from the experts alone: an information limit, not a capacity limit
- [ ] 5.16 an adapter trained on known combinations to predict the gap between the true joint score and the composed score could be applied to a pair never jointly observed
- [ ] 5.17 one undifferentiated interaction operator will not carry across pairs whose mechanisms differ, and conditioning the correction on a named mechanism is the stronger design
- [ ] 5.18 four holdout kinds are distinct and get harder in order: the phrase, the paired label, every combined image, the mechanism itself
- [ ] 5.19 every generated film can look medically plausible while the model produces cavitation twice as often as it should, so radiologist review of single images does not establish a correct distribution
- [ ] 5.20 the four-experiment progression is the right order: a synthetic set where the joint is identifiable, the three regimes imposed deliberately, a transferable residual with one combination withheld, then real films
- [ ] 5.21 a Soweto study found TB patients with HIV had fewer cavitary presentations than those without

### 6. ✅ Multi-label LoRA on cats and dogs: routed, not walked here

Mark: routed. All nine sections carry the same mark and are not opened in this walk. The handoff
names `plans/route-inclusive-captions-pilot.md` as their home, and that file exists in neither
checkout, so the mark is provisional on the route file being found or rewritten.

- [x] 6.1 the experiment as proposed
- [x] 6.2 the three LoRA training strategies
- [x] 6.3 the captions carry the important distinction
- [x] 6.4 COCO rather than a cats-versus-dogs set
- [x] 6.5 how to implement label dropout
- [x] 6.6 what to evaluate
- [x] 6.7 the most interesting single experiment
- [x] 6.8 what would count as a meaningful result
- [x] 6.9 the proposed initial configuration

### 7. ⬜ Is there a mechanism or a context for cats and dogs

- [ ] 7.1 for cats and dogs, co-occurrence is a property of the training set and its labels rather than any mechanism
- [ ] 7.2 a mechanism and a context can still be defined there: spatial arrangement, occlusion or coupled poses, against scene context, positions, scale and view
- [ ] 7.3 two things being in one image is the joint event, not an interaction mechanism
- [ ] 7.4 captions that name only what is there and nothing else teach the model that naming a cat implies no dog
- [ ] 7.5 sampling a subset of the true labels teaches the single conditional to include combined images, while the combined caption separately teaches the joint
- [ ] 7.6 that subset sampling does not make the single-concept conditional equal to the joint conditional, and should not
- [ ] 7.7 the dropout probabilities have to match the intended dataset distribution or the learned conditionals come out distorted
- [ ] 7.8 if both conditionals have learned that the concepts coexist and both are identifiable from the image, the base-divided composition recovers the joint, so subset-sampled conditioning may remove the need for a learned correction entirely
- [ ] 7.9 with no combined images in the fine-tuning set, both conditionals put zero probability on combined images and the composition cannot produce one; a pretrained SDXL doing it anyway is pretraining transfer, not invented co-occurrence
- [ ] 7.10 a semantically interacting scene can need no score correction, and a non-interacting one can need a large correction at noisy steps
- [ ] 7.11 the mechanism and context should be written out explicitly (the three named configurations) rather than left as learned embeddings
- [ ] 7.12 cat and dog labels are readable off the image while disease labels may not be, so a cats-and-dogs result cannot establish that the method recovers biological interaction

### 8. ⬜ Does inclusive training generalise across seeds and pairs

- [ ] 8.1 subset-sampled multi-label training generalises across both random seeds and object pairs and scales to many concepts
- [ ] 8.2 one general-purpose compositional adapter is the better experiment than one adapter per pair
- [ ] 8.3 a multi-label vector with a sampled conditioning subset is the scalable mechanism, and its cost grows with labels processed rather than with all label subsets
- [ ] 8.4 the data needed to cover diverse combinations still grows fast even so
- [ ] 8.5 such an adapter may have learned better single-concept conditionals rather than any universal co-occurrence operator
- [ ] 8.6 four obstacles to scaling: unseen combinations, both experts wanting the centre of the frame, captions that name one of two visible objects, and damage to the base model's broader knowledge
- [ ] 8.7 a held-out-combination paper found multi-object generation less reliable as more combinations are withheld, spatial relations hardest, and class imbalance insufficient as an explanation
- [ ] 8.8 another result finds the diversity of combinations matters more than more examples of the same ones
- [ ] 8.9 the pair-holdout design (five concepts, ten pairs, train eight, withhold two) only works if images holding a withheld pair plus a third object are removed too
- [ ] 8.10 50 to 100 inference seeds and at least three training seeds are needed, and the two seed kinds are different sources of variation
- [ ] 8.11 holding image count fixed while varying how many pairs it spreads over is what isolates combinatorial diversity from data volume
- [ ] 8.12 subset-sampled conditioning alone would not solve the silicotuberculosis problem, because the morphological effects of coexistence still have to be learned

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| the three regimes | spatially distinct, overlapping and confluent presentation | the conversation's own cut; the third regime matches the published combined fibrosis-and-emphysema patterns | confident on the conversation's usage, claim 4.7 checks the published match |
| two diseases at once breaking a classifier | compositional generalisation failure | the model saw each concept alone and fails on their conjunction | confident |
| what the product drops | the interaction term | the ratio correcting a product of independently learned conditionals into the true joint | confident, this repo's own term |
| captions that admit other things are present | inclusive conditioning with condition dropout | a combined image is shown under each single label as well as the joint one | confident on the mechanism |
| one disease hiding another | masking | the evidence a classifier used is no longer observable in the image | confident |
| one disease reading as another | mimicry, a specificity failure | the imaging pattern does not uniquely identify its cause | confident |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|
| 6.1 to 6.9 | marked routed, but `plans/route-inclusive-captions-pilot.md` is in neither checkout | finding that route file under another name, or rewriting it |

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|
| 1.1, 1.3 | take the reference radiographs from Radiopaedia, which has the best-labelled silicosis and progressive massive fibrosis cases | its licence is CC-NC-BY-SA 3.0, non-commercial only, and non-commercial machine-learning use needs an application and a 250 USD fee, so nothing from there can enter a paper figure or a training set here |
| 1.7 | use NIH's gained labels as the stand-in for the radiology criterion the NIOH guide names, a rapid rise in profusion | profusion is an ILO score and no public set on disk carries it; NIH's labels are binary presence, so the measurable thing is not the thing the criterion names |
| 1.1 | read the ERS critical narrative review directly for its definition | both the publisher and PubMed Central refuse an automated fetch (403, and a browser check), so the definition is taken from the Cureus case report and the search summary instead, and the review stays unread |

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 1.7 | the asymmetry ratio, cell(A then B) over cell(B then A), with a binomial confidence interval, on the pairs whose counts can carry one, starting with effusion and atelectasis at 247 against 124 | an interval excluding 1 on a pair with a documented physical mechanism says the labels do track a sequence; an interval covering 1 says the matrix's asymmetry is prevalence and the ordering axis is not readable from report-derived labels |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|
| 2 | 3.8, 4.6 | `count_cooccurrence.py` over the same CSV, every pair of findings counted over films | returned 2026-11-03 | all four named pairs reproduce the route's facts table exactly: 11, 36, 1,063 and 3,275 films. [The matrix and its card](../../results/which-findings-arrive-after-which/README.md) |
| 1 | 1.7 | `count_arrival_order.py` over NIH's metadata on mscluster, every ordered pair of findings counted over patients with two or more films | returned 2026-11-03 | 9,167 of 13,302 repeat-film patients changed label, so the axis exists; fibrosis then pneumonia is 12 patients and pneumonia then fibrosis is 8, so the conversation's own pair cannot be scored. [The matrix and its card](../../results/which-findings-arrive-after-which/README.md) |

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | The ask | Return line | State |
|---|---|---|---|---|
| 1 | 1.7 | `/drip-idea --hypothesis` on the asymmetry ratio, to write the support, null and inconclusive bands before the ratio is computed | bring back `HYPOTHESIS_arrival_order.md` and the band that the measured ratio falls in | emitted, not yet run |

## Sources

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Everything the conversation cites is unverified until a round checks it. The four links it carries
are listed so a later round does not re-fetch them blind.

| Source | What it gives the idea | Confidence |
|---|---|---|
| pubmed.ncbi.nlm.nih.gov/32317070/ | the 2020 four-system study behind claim 2.1 | link saved in the transcript, content unverified |
| pmc.ncbi.nlm.nih.gov/articles/PMC9565035/ | the 2022 commercial-system study behind claims 2.2 to 2.4 | link saved, content unverified |
| pmc.ncbi.nlm.nih.gov/articles/PMC12904126/ | the 2026 Southern African miners study behind claim 2.5 | link saved, content unverified |
| arxiv.org/abs/2302.11552 | Du et al., the sampling result behind claims 5.4 and 5.10 | confident, already cited in this repo |
| this repo's [report index](../../../report/00-INDEX.md) | the measured findings that mark turns 5, 7 and 8 | confident, this project's own verdicts |
| the [disjoint-support idea map](../disjoint-marginal-support-and-poe-chimeras/IDEA_MAP.md) | claim 5.3 is already an idea here, with a two-stage design | confident |
| Elango et al., *Cureus* 16(7):e65411 (2024), DOI 10.7759/cureus.65411 | the definition as plain coexistence, and the posteroanterior reference radiograph saved beside the walk under CC BY 4.0 | confident, article read in full |
| the [NIOH short guide](https://www.nioh.ac.za/teaching-training/training-and-learning-repository-of-materials-on-the-radiology-of-silicotuberculosis/short-guide-on-the-radiologic-features-of-silicosis-and-silicotuberculosis-combined-disease/) to the radiology of the combined disease | the list of features that favour tuberculosis over silicosis, and the statement that order of onset is not discussed while change over time is | likely but verify: the fetched page carried unrelated advertising and a 404 fragment, so the host may be serving injected content |
| Ehrlich et al., *Silicotuberculosis: a critical narrative review*, Eur Respir Rev 33(174):240168 | the definition covering all temporal sequences and including post-tuberculous lung disease | can name the source only: both the publisher and PubMed Central refused the fetch, so this is the search engine's summary of it, unread |
| the ILO 2022 classification standard radiographs, `ilo.org/radiographs` | free standard films for profusion scoring, named by the NIOH guide as the comparison set | likely but verify, not yet fetched |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Claim 1.2, round 1: whether the combined condition is properly called silicotuberculosis. Three
fetched sources already use the term, so the round is about what the term commits a label to
rather than about whether it exists.
