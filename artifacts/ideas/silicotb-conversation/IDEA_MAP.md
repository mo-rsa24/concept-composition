# 💡 Every claim in the silicotuberculosis conversation, marked

## Position in the idea

| Turn | Claims | Marked | Mark spread |
|---|---|---|---|
| **1 (current)**: TB then silicosis, how each changes the other | 10 | 8 | 3 holds, 4 wrong as stated, 1 needs a check |
| 2: where AI fails, and what the label should be | 18 | 0 | all open |
| 3: public-data pairs with the same three regimes | 15 | 0 | all open |
| 4: pairs that genuinely interact | 11 | 0 | all open |
| 5: product of experts, the interaction term, inclusive labels | 21 | 0 | all open |
| 6: multi-label LoRA on cats and dogs | 9 | 9 | routed, not walked here |
| 7: is there a mechanism or a context for cats and dogs | 12 | 0 | all open |
| 8: does inclusive training generalise across seeds and pairs | 12 | 0 | all open |
| **total** | **108** | **17** | |

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

Turn 1, claim 1.9. Claims 1.1, 1.3, 1.4 and 1.5 are wrong as stated and repaired into
1.7, 1.8, 1.9 and 1.10; 1.6 holds and shows only the first regime; 1.10 holds with a VinDr query for three of its four regimes; 1.2 holds with a disjunction consequence carried to turns 5 and 7; turn 6
is routed elsewhere.

**What compile would produce today**

Nothing worth producing: 17 of 108 claims carry a mark, and nine of them are the routed turn.

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
- [x] 1.2 the combined condition is properly called silicotuberculosis — **holds**, and the label behind the name is a disjunction
- [x] 1.3 the posteroanterior film shows scarring on one side and progressive massive fibrosis on the other — **wrong as stated**, it is a question and never an assertion; repaired into 1.8
- [x] 1.4 CT separates the combined condition into four distinguishable patterns (`source/screenshots/four-ct-patterns.png`) — **wrong as stated**, the four are the atoms the condition is built from, not four looks of the condition; repaired into 1.9
- [x] 1.5 the three overlap regimes are as drawn: separate lesions, same-region lesions, merged presentation (`source/screenshots/three-overlap-regimes.png`) — **wrong as stated**, the three are real but not all of them and they are kinds of region rather than kinds of patient; repaired into 1.10
- [x] 1.6 combined fibrosis and emphysema has the CT appearance shown (`source/screenshots/cpfe-ct-example.png`), which belongs to turn 4's content, not turn 1's — **holds**, and the figure shows only the separate-places regime; its own paper names the hidden-disease regime in prose and pictures none of it
- [ ] 1.9 the four CT patterns are the four atomic predicates of the condition's Boolean formula, (simple silicosis OR complicated silicosis) AND (active tuberculosis OR post-tuberculous lung disease), so the formula has four atoms and two disjunction nodes rather than three atoms and one  ← current *(appended round 4, the repair of 1.4)*
- [ ] 1.8 the three regimes in `source/screenshots/three-overlap-regimes.png` are the project's co-occurrence, intersection and interaction, in that order *(appended round 3, the repair of 1.3)*
- [x] 1.7 what the order of onset was supposed to carry is carried instead by two films of the same patient at different times: a rapid rise in how many opacities there are, in their size, or a new cavity, is what favours tuberculosis — **needs a check** *(appended round 2, the repair of 1.1)*
- [x] 1.10 the overlap has four regimes, not three, and they apply to a region of a lung rather than to a whole film: both diseases in separate places, both in the same place and both readable, a lesion neither builds alone, and one disease present and unreadable because the other has destroyed or buried the lung it would show in — **holds**, measured per pair on VinDr-CXR, with one named pair for each of the first two regimes and a kappa-based selection for the fourth *(appended round 5, the repair of 1.5)*

**1.9, what attacking it found.** No mark yet. Four things break, and the first two are visible in
the figure the claim is built from.

**The silicosis pair is not two atoms.** Panel B draws conglomerate masses with small nodules still
scattered around them, which is what complicated silicosis is: the large-opacity grade of the ILO
classification is defined as sitting on a background of small opacities. So complicated silicosis
contains simple silicosis, `B` implies `A`, and `A OR B` reduces to `A`. One graded predicate, how
much silicosis, not two Boolean ones.

**The tuberculosis pair is a clock, not a disjunction.** Active tuberculosis and post-tuberculous
lung disease are the same disease at two times, and a patient can carry both at once (healed
scarring in one lobe from an earlier episode, active disease in another). An ordering over time
cannot be a node in a formula whose atoms are simultaneous.

**The formula is over a patient and 1.10 settled that the unit is a region.** A patient-level
conjunction says both diseases are present and says nothing about where, so it cannot carry the four
regimes. Read per region it fails differently: the hidden regime and the merged regime are both
cases where the conjunction is true and the film shows one finding or neither. Presence and
readability are different predicates, and the formula only has the first.

**The atoms are three-reader votes, not facts.** Run 6's Fleiss' kappa is 0.355 for consolidation
and 0.365 for atelectasis. For the findings that matter to the hidden regime, the atom's own truth
value is a chance-corrected agreement of about a third, so a Boolean formula over them is a formula
over calls that two of three readers do not make.

**The direction a repair would take.** Per region, one graded silicosis extent and one
tuberculosis state with a time on it, plus a separate readability predicate that the four regimes of
1.10 are the values of. That is two graded predicates and a visibility term, not four Booleans, and
it is what `fix` would write as 1.11.

**1.10, settled.** Mark: holds. The four regimes are a per-pair property that can be read off
annotation boxes, and three of the four now have a query over VinDr-CXR that selects their films.
Separate places is aortic enlargement with cardiomegaly: 1,951 of 1,972 films, 98%, and the share
does not move at any intersection-over-union floor. Same region is lung opacity with pulmonary
fibrosis: 201 films at a floor of 0.25 and 41% of what survives the double-label drop, median
intersection over union 0.405. One finding hidden is selected on Fleiss' kappa rather than on the
single-reader share, because the two disagree: pneumothorax has 39% single-reader calls at kappa
0.743, which is rarity, while consolidation has 65% at kappa 0.355, which is difficulty. The merged
regime has no query and cannot have one, because whether two findings are still separately nameable
is a judgement made by eye on the film.

What the claim does not carry: the regimes are regions, so a film is never one regime, and a caption
that names a whole film has already lost the claim. Two of the words in the original figure's own
caption, mimicry and masking, are effects rather than places and belong to Ehrlich et al.'s cut,
which has no fourth panel. The outstanding work is the twelve annotated pictures in
[route 01](routes/01-annotated-radiographs-for-each-regime.prompt.md), where the hidden regime still
has none.

**1.4, settled.** Mark: wrong as stated. The four panels are A simple silicosis, B complicated
silicosis (progressive massive fibrosis), C active pulmonary tuberculosis and D post-tuberculous
lung disease. Two are silicosis with no tuberculosis and two are tuberculosis with no silicosis, so
none of the four is the combined condition and CT is not shown separating it into anything. What the
figure is instead is the atom list for the Boolean formula, drawn: it names the four predicates the
condition is a formula over. The repair is 1.9. Panel B's own picture settles one of the two
disjunction nodes against exclusivity as well, because it draws conglomerate masses with small
nodules still around them, so complicated silicosis contains simple silicosis rather than excluding
it.

**1.3, settled.** Mark: wrong as stated, because it is not a claim. Turn 1's text is the user's own
question, "Is it scarring on the left, progressive massive fibrosis on the right for a PA x-ray?",
and the answer that would have made it an assertion is the empty one. No screenshot shows such a
film either: the three saved images are the four CT patterns, the three overlap regimes and the
fibrosis-with-emphysema CT, and none is a posteroanterior radiograph. The repair is 1.8: the
separated arrangement the question guesses at is regime 1 of the conversation's own three-regime
figure, and that figure maps onto three operators this repo has already defined in
[the composition terms](../../../3-understanding-compositionality/artifacts/ideas/poe-overlap-dial/composition-terms.md).

**1.2, settled.** Mark: holds, as the name of the condition. Three fetched sources use the term
and the Cureus title is *Silico-Tuberculosis Mimicking Malignancy*. The consequence is the round's
real content and it is carried to claims 5.12 and 7.5, not marked here: claim 2.12 spells the label
out as silicosis AND (active TB OR old TB damage), so one word stands for two different-looking
chests. A product of two experts is the operator for "both", and the operator for "either" is a
mixture, which nobody in the conversation reaches for. The three operators are already in print as
a calculus (Du, Li and Mordatch, NeurIPS 2020, `arXiv:2004.06030`): conjunction is the sum of the
energies (their equations 4 and 5), disjunction is the log of the sum of their exponentials (7 and
8), negation is one energy minus a scaled other (9). Disjunction is the expensive one, because
equation 6 needs each concept's partition function and equation 7 only works by assuming they are
all equal, which the authors state is "not guaranteed to hold in general".

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
- [ ] 5.12 *(carries 1.2's disjunction: the label is silicosis AND (active TB OR old TB damage), and the "either" operator is a mixture, not a product)* "a photo of silicoTB" and "a photo of silicosis and TB" are not guaranteed to be the same conditional, and a general-domain text encoder may represent the compound term poorly
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
names `4-inclusive-captions/plans/route-inclusive-captions-pilot.md` as their home, and that file exists in neither
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
- [ ] 7.5 *(carries 1.2's disjunction, same as 5.12)* sampling a subset of the true labels teaches the single conditional to include combined images, while the combined caption separately teaches the joint
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
| 1.2's consequence | the conversation only ever composes by multiplying, and the condition's label contains an "either". Whether the mixture operator can be sampled here is undecided, because its own authors assume every concept's partition function is equal and say that assumption is "not guaranteed to hold in general" | answered in print: the exact calculus (`arXiv:2602.05549`, Proposition 3.1) needs the two branches of the "either" to be mutually exclusive or conditionally independent given the noisy image, and both of its disjunction rules weight each branch's score by that branch's posterior probability, so the price is a classifier over noisy images. Answered against the hope: Ehrlich et al. define the condition as "the combination of silicosis and active TB and/or post-tuberculous lung disease" (Definitions, quoted in [route 2 of the overlap-dial walk](../../../3-understanding-compositionality/artifacts/ideas/poe-overlap-dial/routes/02-findings-and-diagnoses-literature.return.md)). "And/or" means a patient can carry both branches, so they are not mutually exclusive and the OR node must fall back on conditional independence, the same condition the AND node needs. The whole formula therefore reduces to claim 5.7. Lands in claims 5.12 and 7.5 |

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|
| 1.1, 1.3 | take the reference radiographs from Radiopaedia, which has the best-labelled silicosis and progressive massive fibrosis cases | its licence is CC-NC-BY-SA 3.0, non-commercial only, and non-commercial machine-learning use needs an application and a 250 USD fee, so nothing from there can enter a paper figure or a training set here |
| 1.7 | use NIH's gained labels as the stand-in for the radiology criterion the NIOH guide names, a rapid rise in profusion | profusion is an ILO score and no public set on disk carries it; NIH's labels are binary presence, so the measurable thing is not the thing the criterion names |
| 1.5 | express the hidden-disease regime with the negation operator, score(x,A) minus a weight times score(x,B), which is the one operator in [Product, Mixture And Negation](/home/molef/Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/Composition/Product,%20Mixture%20And%20Negation.md) that removes evidence | it gives the right picture under the wrong label. Negation samples from "A and not B", whose images are the ones where B is genuinely absent. A masked film is one where B is present and unreadable. The two sets of images overlap almost exactly, which is the whole clinical problem, but the conditioning differs, so a model trained this way can never be asked for "both present, one hidden" and cannot be scored for whether it got the hidden disease right. The negation weight is also a free scalar chosen by hand with no quantity in the data to set it against |
| 1.1 | read the ERS critical narrative review directly for its definition | both the publisher and PubMed Central refuse an automated fetch (403, and a browser check), so the definition is taken from the Cureus case report and the search summary instead, and the review stays unread |

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 1.7 | the asymmetry ratio, cell(A then B) over cell(B then A), with a binomial confidence interval, on the pairs whose counts can carry one, starting with effusion and atelectasis at 247 against 124 | an interval excluding 1 on a pair with a documented physical mechanism says the labels do track a sequence; an interval covering 1 says the matrix's asymmetry is prevalence and the ordering axis is not readable from report-derived labels |
| the three regimes against the three operators | the conversation's regime figure has three panels and [the composition terms](../../../3-understanding-compositionality/artifacts/ideas/poe-overlap-dial/composition-terms.md) has three matching operators, but nothing has checked the mapping panel by panel, and regime 2 has two candidate operators rather than one | reading each regime's caption against the operator's "what it is" and "failure" lines, and saying for regime 2 whether both diseases stay recognisable (intersection with contention as the failure) or merge (the equal-density blend, where fusion is the target). Feeds claim 1.8 |
| 1.5, the regimes against the review's effects | the conversation cuts the overlap three ways by **where** the lesions sit (separate, same region, merged). [Ehrlich et al. 2024](https://doi.org/10.1183/16000617.0168-2024) cuts it three ways by **what the overlap does** (each disease can hide the other, each can distort the other's pattern, together they build a lesion neither builds alone), quoted in [the composition terms](../../../3-understanding-compositionality/artifacts/ideas/poe-overlap-dial/composition-terms.md). The two threes are different cuts and nothing has reconciled them: masking has no regime, and separate-lesion co-occurrence is not one of the effects | laying the two lists side by side and saying, for each of the review's three effects, which regime it falls in. An effect with no regime means the three regimes are not exhaustive and the figure needs a fourth panel. If masking lands in regime 1, the regimes stop being a difficulty ordering, because a hidden disease is harder than two separate lesions, not easier |
| 1.10 | whether the hidden-disease regime is a failure of the composition operator or a failure of each expert's own conditional. The Bayes form's interaction factor is p(A,B\|x) / (p(A\|x) p(B\|x)). On a masked film, A is read with near-certainty and the posterior on B is some small e, so p(A,B\|x) is about e and the denominator is about e, and the factor is near 1. By that arithmetic masking is not something the interaction term has to supply. What is wrong instead is p(x\|B): a B expert trained only on films where B is visible puts almost no density on a film where B is present and unreadable, because no such film ever appeared under the B label | computing the factor numerically on a trained pair rather than by hand, and separately measuring the B expert's likelihood on masked films against unmasked ones. If the factor is near 1 and the likelihood collapses, the fix is inclusive captioning, not a new operator, and the whole hidden-disease regime lands in the pilot already routed at [the inclusive-captions plan](../../../4-inclusive-captions/plans/route-inclusive-captions-pilot.md) rather than in the interaction term. **Applied there on 2026-11-03** (`0af529a6`): that plan's per-class scorer now has three outcomes rather than two, because in 53% of its 169 both-animal photos the smaller animal is under 5% of the frame, so a both-visible scorer would read the inclusive arm as worse exactly when it works |
| 1.10 | whether readability of the hidden disease falls with the extent of the hiding disease. NIOH radiographs 3 and 4 are the same pair and the same two labels, and the reader called silicosis on 3 and "doubtful" on 4; the visible difference is how much lung the previous tuberculosis left intact | scoring the ten NIOH films for the fraction of lung destroyed and plotting the reader's confidence in the second disease against it. A falling curve makes masking a dose effect of the first disease with a measurable axis, which is what a regime needs before it can be imposed deliberately in a synthetic set |
| 1.10 | whether the 201 lung opacity with pulmonary fibrosis films that survive the double-label drop at an intersection-over-union floor of 0.25 read as one region to a person. Run 6 settles the counting; nobody has looked at the films | opening twenty of the 201 and twenty of the 288 that the same query calls separate, and saying for each whether the two findings are in one region. A query whose calls a person reverses more than twice in forty is measuring boxes rather than regions, and the regime needs a different selector |
| the proxy pair | no pair has been chosen. Effusion with atelectasis is named a candidate in [the route](../../../plans/route-every-silicotb-chat-claim.md) and in claim 3.4, and nothing marks it | counting how many of the three regimes the candidate shows. On the NIH counts, effusion with atelectasis has 3,275 films both and a documented mechanism, pneumonia with emphysema has a documented appearance change but only 23 films, and cardiomegaly with effusion has 1,063 films and no third pattern. Feeds claims 3.4 and 3.12 |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|
| 7 | 1.10, the dataset decision | the co-occurrence count for every label pair in NIH ChestX-ray14 and its box coverage, and the overlap split for five more VinDr pairs | returned 2026-11-04 | the pair with the published evidence cannot be measured and the pair that can be measured is not two diseases. Combined pulmonary fibrosis and emphysema: 36 of ChestX-ray14's 112,120 films, against 2,516 emphysema alone and 1,686 fibrosis alone, and neither class appears in that dataset's 984 boxes. Pleural effusion with atelectasis: 3,275 films in ChestX-ray14, the second-largest pair after effusion with infiltration at 4,000, and 20 of the 880 box-carrying films have boxes for both. In VinDr the same pair is 90 films, 12 of them one box labelled twice, leaving 78 of which 68 are separate at any floor with a median best overlap of 0.000, while the literature says the fluid causes the collapse. Also in VinDr: atelectasis with consolidation 57 films, 27 double-labelled, 30 left, 19 shared-region, median overlap 0.465, the highest measured; pleural effusion with consolidation 190 left, 50 shared-region; pleural effusion with infiltration 222 left, 24 shared-region |
| 6 | 1.10 | the identical-box films dropped from every pair and the split re-read at an intersection-over-union floor of 0.25, and separately Fleiss' kappa per finding over the three readers | returned 2026-11-04 | regime 2 survives the drop on one pair and the hidden-disease proxy changes. Lung opacity with pulmonary fibrosis: 177 of 666 films dropped, 201 of the remaining 489 same-region, 41%. Pleural effusion with pleural thickening: 737 of 839 dropped, leaving 102 films of which 24 are same-region, so that pair carries no regime. Aortic enlargement with cardiomegaly is unchanged at 98% separate over 1,972 films. Fleiss' kappa over all 15,000 three-reader films: consolidation 0.355, other lesion 0.356, atelectasis 0.365, lung opacity 0.406, calcification 0.415, infiltration 0.438, pleural thickening 0.461, ILD 0.476, nodule/mass 0.528, pulmonary fibrosis 0.675, pleural effusion 0.704, pneumothorax 0.743, aortic enlargement 0.780, cardiomegaly 0.789. Kappa and the single-reader share rank differently: pneumothorax has 39% single-reader calls at kappa 0.743, which is rarity, and consolidation 65% at kappa 0.355, which is difficulty. Select the hidden regime on kappa |
| 5 | 1.10 | the same box overlap re-run at intersection-over-union floors of 0.1, 0.25 and 0.5 instead of any-overlap, plus the containment fraction and the median IoU per pair, and separately how many films carry one box bearing both labels | returned 2026-11-04 | the separate-places end survives every floor and the same-region end splits in two. Aortic enlargement with cardiomegaly stays 1% overlapping at any floor over 1,973 films, and with pleural thickening 0% over 1,339, so regime 1 is real. Pleural effusion with pleural thickening reads 94% any-overlap, 89% at IoU 0.5 and median IoU 1.00, which is one box labelled twice by one radiologist on 737 of its 839 films: not a measurement. Lung opacity with pulmonary fibrosis is 76% any-overlap, 46% at IoU 0.5, median IoU 0.405, with the double-label artefact on only 26% of 666 films, so that is the pair carrying regime 2. Pleural thickening with pulmonary fibrosis falls from 46% to 7% at IoU 0.5 over 971 films, so its "spans both regimes" reading was corner-clipping |
| 4 | 1.10 | reader agreement on VinDr-CXR: for each finding, over the films where that finding was marked at all, how many of the three radiologists who read the film marked it | returned 2026-11-03 | the single-reader rate is the hidden-disease signal and it is large. Consolidation 66% called by 1 of 3, Other lesion 68%, Calcification 61%, ILD 61%, Infiltration 60%, Lung Opacity 59%, Pleural thickening 55%, Nodule/Mass 51%, Pleural effusion 39%, Pulmonary fibrosis 37%, Cardiomegaly 21%, Aortic enlargement 24%. All 15,000 training films have exactly three readers, so this is a per-film three-way vote rather than a pooled rate. 4,394 of the 15,000 carry at least one box |
| 3 | 1.10 | box overlap on VinDr-CXR: for each co-occurring finding pair, how many of the films where both appear have at least one overlapping box pair | returned 2026-11-03 | the regimes separate cleanly by pair, from `train.csv`, 67,914 boxes over 4,394 films. Aortic enlargement with cardiomegaly 97% separate over 1,973 films; pleural effusion with pleural thickening 94% same region over 839; lung opacity with pulmonary fibrosis 76% same region over 666; pleural thickening with pulmonary fibrosis 46% same region over 971, the only pair that spans both |
| 2 | 3.8, 4.6 | `count_cooccurrence.py` over the same CSV, every pair of findings counted over films | returned 2026-11-03 | all four named pairs reproduce the route's facts table exactly: 11, 36, 1,063 and 3,275 films. [The matrix and its card](../../results/which-findings-arrive-after-which/README.md) |
| 1 | 1.7 | `count_arrival_order.py` over NIH's metadata on mscluster, every ordered pair of findings counted over patients with two or more films | returned 2026-11-03 | 9,167 of 13,302 repeat-film patients changed label, so the axis exists; fibrosis then pneumonia is 12 patients and pneumonia then fibrosis is 8, so the conversation's own pair cannot be scored. [The matrix and its card](../../results/which-findings-arrive-after-which/README.md) |

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | The ask | Return line | State |
|---|---|---|---|---|
| 1 | 1.7 | `/drip-idea --hypothesis` on the asymmetry ratio, to write the support, null and inconclusive bands before the ratio is computed | bring back `HYPOTHESIS_arrival_order.md` and the band that the measured ratio falls in | emitted, not yet run |
| 2 | 1.5 | [an annotated radiograph for every overlap regime](routes/01-annotated-radiographs-for-each-regime.prompt.md): build twelve annotated real films, three per regime across three disease pairs, with the hidden-disease regime as the one that decides success | bring back `01-annotated-radiographs-for-each-regime.return.md`, the picture count per regime, and which licences are still unresolved | written, not yet run |
| 3 | the gap lines | [four journeys for the words this walk had no teacher for](routes/03-submerge-the-gap-words.prompt.md): `/submerge` on what two diseases do to each other on a film, on where a finding is and how much two share, and on scoring a classifier when the test set is biased, plus `/submerge --over ai-engineering/chance-corrected-agreement` for three readers instead of two | bring back each journey's row in `JOURNEYS.md` and its hand-computed check against the code | written, not yet run |
| 4 | 1.10 | [the chain from six composition papers to a classifier scorecard](../../../plans/route-compose-x-rays-and-score-them.md): read one idea across six papers, pick the construction, reproduce one figure, then compose films for the picked VinDr pair and score a classifier across the rare-cell dial | bring back the operator table with the pick, then the AUROC-against-dial scorecard with counts per cell | written, step 0 not yet run |
| 5 | 1.10, 1.6 | [what the literature knows about two chest diseases in one chest](routes/05-deep-research-the-disease-pairs.prompt.md): a ChatGPT deep-research run over fifteen disease pairs, asking which of the four regimes each one documents, whether anyone names the fourth at all, whether one pair shows different regimes at different disease stages, and whether a fifth regime exists | bring back six markdown files: the pairs table with a verdict in every cell, the field's own words for our four, six textbooks with chapter and page, three cases of one pair showing two regimes, twenty-five licensed candidate figures, and a yes or no on a fifth regime | written, not yet run |

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
| [Martins et al. 2010, the cavitated conglomerate mass](https://pmc.ncbi.nlm.nih.gov/articles/PMC2929613/) | one silicotuberculosis patient carrying a cavity inside a conglomerate mass with small nodules elsewhere on the same CT, plus a 44-patient series where 8 masses cavitated and 6 of those 8 had tuberculosis | confident, already cited in [the composition terms](../../../3-understanding-compositionality/artifacts/ideas/poe-overlap-dial/composition-terms.md) |
| [Zeng, Liang and Ou 2025, the CPFE narrative review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12711510/) | the source of the conversation's own CPFE screenshot, uncropped and CC BY, filed at [the regime set](../../results/overlap-regimes-on-real-films/README.md). Its conclusion names "mutual masking of the imaging characteristics of pulmonary emphysema and early pulmonary fibrosis" while its figure shows only the separate-places regime | confident, quoted |
| [Fatal CPFE with fibrosis unseen on CT](https://pmc.ncbi.nlm.nih.gov/articles/PMC4585953) | an autopsy case where upper-zone lung read as plain emphysema on CT held dense usual-interstitial-pneumonia fibrosis, so the overlap showed fewer findings than either disease alone | confident on the finding, one case |
| this repo's [report index](../../../report/00-INDEX.md) | the measured findings that mark turns 5, 7 and 8 | confident, this project's own verdicts |
| the [disjoint-support idea map](../../../3-understanding-compositionality/artifacts/ideas/disjoint-marginal-support-and-poe-chimeras/IDEA_MAP.md) | claim 5.3 is already an idea here, with a two-stage design | confident |
| Elango et al., *Cureus* 16(7):e65411 (2024), DOI 10.7759/cureus.65411 | the definition as plain coexistence, and the posteroanterior reference radiograph saved beside the walk under CC BY 4.0 | confident, article read in full |
| the [NIOH short guide](https://www.nioh.ac.za/teaching-training/training-and-learning-repository-of-materials-on-the-radiology-of-silicotuberculosis/short-guide-on-the-radiologic-features-of-silicosis-and-silicotuberculosis-combined-disease/) to the radiology of the combined disease | the list of features that favour tuberculosis over silicosis, and the statement that order of onset is not discussed while change over time is | likely but verify: the fetched page carried unrelated advertising and a 404 fragment, so the host may be serving injected content |
| Ehrlich et al., *Silicotuberculosis: a critical narrative review*, Eur Respir Rev 33(174):240168 | the definition covering all temporal sequences and including post-tuberculous lung disease | can name the source only: both the publisher and PubMed Central refused the fetch, so this is the search engine's summary of it, unread |
| Du, Li and Mordatch, *Compositional Visual Generation with Energy Based Models*, NeurIPS 2020, `arXiv:2004.06030` | the three logical operators as a calculus over energies: conjunction at equations 4 and 5, disjunction at 6 to 8, negation at 9, with the equal-partition-function assumption the disjunction needs | confident, section 3 read as extracted text |
| Alesiani, Warrell, Bien, Christiansen, Ferraz and Niepert, *Logical Guidance for the Exact Composition of Diffusion Models*, `arXiv:2602.05549` (v2, 23 March 2026) | the exact Boolean calculus for diffusion guidance, and its sufficient condition: conjunctions over conditionally independent subformulas, disjunctions over subformulas that are conditionally independent or mutually exclusive | confident, abstract and sections 2 to 3 read as extracted text |
| Gaudi, Sreekumar and Boddeti, *CoInD: Enabling Logical Compositions in Diffusion Models*, `arXiv:2503.01145` | trains the conditional-independence violation down rather than assuming it, which is the precondition the exact calculus needs | confident, already in this repo's literature notes |
| the ILO 2022 classification standard radiographs, `ilo.org/radiographs` | free standard films for profusion scoring, named by the NIOH guide as the comparison set | likely but verify, not yet fetched |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Claim 1.3, round 1: whether the posteroanterior film shows scarring on one side and progressive
massive fibrosis on the other. The saved screenshot is gone, so the round is about whether the
claim is checkable at all against the Cureus radiograph beside the walk, which shows both patterns
in the same chest rather than one per side.
