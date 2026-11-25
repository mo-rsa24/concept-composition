# Dials for abstract concepts: co-occurrence strength, count and binding strength

Board step H3, Loop 15 (`s15`). For each of the owner's three dials: what exists, whether the outcome
rises steadily as the strength rises (a strength-against-outcome curve with its source), and where it
breaks. Fourteen papers were read for this page from their PDFs, on top of
[the H1 page](01-directions-that-generalise.md) and [the H2 page](02-shared-representations.md), and
three curves come from this project's own runs. Every page number is a PDF page.

**The answer.** Only one of the three dials has a curve that rises steadily with its strength, and it is
this project's own: adding the exact interaction term back at weight λ takes runs with two animals
counted from 1 to 30 of 32 as λ goes from 0 to 1 (8 animal pairs, 4 seeds each), while wrong corrections of the
same size stay at 3 of 32 or fewer. No published paper read here sweeps a strength against "two subjects
together" on a text-to-image model and finds a steady rise: blending two words' embeddings peaks at 62%
both-present at the halfway mix and falls, and MagicMix jumps from one object through a hybrid to two.
The count dial is the requested number, and accuracy falls at every step up even before steering
(SDXL below half from 3 objects, SD3 from 4); the best attention-map methods push that point out to 5 to
7, and two object types in one prompt collapse it (SDXL 14% against 60% with one noun). No method sets a
count by repeating a token or scaling its attention, and writing the count as "a times b" halves
accuracy. Binding strength has no steady dial anywhere: every binding strength read here either barely
moves the score, peaks and then floods the attribute across the picture, or acts as a switch.

<a href="03-strength-against-outcome.png"><img src="03-strength-against-outcome.png" width="720" alt="Three panels. Left, co-occurrence: the project's exact correction rises from 3 to 94 percent as weight goes 0 to 1 while wrong corrections stay under 10; an embedding blend peaks at 62 percent at 0.5 then falls; InteractDiffusion rises and levels at about 29 mAP. Middle, count: exact accuracy falls with requested count for SDXL and SD3, and CountCluster and ISAC lift it but it still falls. Right, binding: guidance scale leaves colour attribution flat between 24 and 31 percent, the project's species push drops from 8 of 8 to 1 then 4, and Relation Rectification rises from 47 to 73 percent"></a>

*Strength against outcome for the three dials. Left: x is each method's own 0-to-1 strength (λ, r or ω),
y is the share of pictures with both subjects (percent) or InteractDiffusion's interaction-detection
score (mAP points). Middle: x is the requested count, y is exact-count accuracy in percent; the grey line
is 50%. Right: x is the strength as a share of the largest value each method tried, y in percent. One
point is one setting; the number of pictures behind each point is in the tables below. Drawn by
[the figure script](03-strength-against-outcome.py) from the numbers in its legend, each named with its
table or figure and page.*

## Table of contents

- [Words this uses](#words-this-uses)
- [The main table: one row per dial](#the-main-table-one-row-per-dial)
- [1. Co-occurrence strength](#1-co-occurrence-strength)
- [2. Count](#2-count)
- [3. Binding strength](#3-binding-strength)
- [What a working dial would need](#what-a-working-dial-would-need)
- [Where the sources disagree](#where-the-sources-disagree)
- [Still open](#still-open)
- [Where this came from](#where-this-came-from)

## Words this uses

**Co-occurrence strength.** How firmly two named subjects appear together: from one fused hybrid,
through two touching or interacting, to two clearly separate subjects.

**Count.** The number of instances of one subject, set from the prompt ("dog" times 2 gives two dogs).

**Binding strength.** How firmly an attribute (a colour, a texture, a part) stays on its own subject
rather than leaking onto the other.

**Steady, peaks, saturates, switch.** Steady: the outcome moves the same way at every step of the sweep.
Peaks: it rises, then falls past some strength. Saturates: it rises, then stays flat. Switch: it moves
from one state to another between two settings, with nothing measured in between.

**Compose rate.** The share of runs with two separate animals, as the project's instance-count scorer
reads them ([its definition](../../../../context/world/compose-rate.md)).

**Read marks.** *read from the paper*: the PDF was opened at that page for this page; *by eye*: read from
a plot rendered from the PDF; *read from the project's findings*: copied from a report page or a results
file named in the row; *inferred*: this page's reasoning.

## The main table: one row per dial

| Dial | What exists | Does the outcome rise steadily with strength? The curve and its source | Where it breaks | Mark |
|---|---|---|---|---|
| **Co-occurrence strength** | the project's exact correction added back at weight λ (score space); SuperDiff's own missing term added back; blending two words' embeddings (Matsuhira et al.); MagicMix's layout-content mix; InteractDiffusion's relation control; Separate-and-Enhance's training strength | **Yes, in this project only.** Two animals counted by the instance scorer (which cannot tell a cat and a dog from two dogs) in 1, 3, 8, 23, 30 of 32 runs at λ 0, 0.25, 0.5, 0.75, 1; wrong pair, wrong seed and wrong order corrections 0 to 3 of 32 (`outputs/interaction_term/dose/dose_curves.json`, scorer `instance_count`, pairs and seeds read from the file). Published: both-present 14.3% rising to a peak of 62.1% at blend 0.5, then 25.7% at 0.9 (Matsuhira et al., Table 1, p. 4); interaction detection rises from about 0.5 to 29 mAP and is flat from ω 0.6 (InteractDiffusion, Fig. 10, p. 11, by eye) | the exact correction needs the joint prompt's prediction, which an unseen pair lacks; the trained adapter that stands in for it washes out from λ 0.5 when moved into SuperDiff ([H2, section 5](02-shared-representations.md#5-the-sampler-mathematics-of-composition)); the best published mix setting differs by pair, 0.1 for corgi and husky against 0.9 for corgi and coffee machine (MagicMix, p. 8); no published score tells a hybrid from two subjects (Matsuhira, p. 4) | read from the project's findings and from the paper |
| **Count** | requested N with steering from the noun's attention maps (CountCluster, ISAC, Make It Count, NUMINA), a learned count token (YOLO-Count, Detection-Driven), a detector-guided starting noise (D2D, Numerosity), a prompt contrast rebuilt every step (ATHENA); this project's count checker push on digits | **No: accuracy falls with every step up in N**, steered or not. SDXL exact accuracy 62, 55, 40, 20, 10, 11, 4, 5, 3, 2% for N = 1 to 10; SD3 74, 72, 50, 46, 24, 12, 10, 10, 8, 5% (GeckoNum, human counts, Fig. A2, p. 19). With CountCluster on SDXL, 93, 74, 93, 63, 63, 37, 37, 26, 15% for N = 2 to 10 (Fig. 4, p. 7, by eye); with ISAC 96, 89, 71, 47% for N = 2 to 5 (Table 3, p. 12). The push itself does rise steadily: this project's count checker push on digits gave 0, 8, 20, 40 clean pairs of 320 at push length 0, 0.1, 0.2, 0.4 | below half from N = 3 on plain SDXL, N = 7 with CountCluster; SDXL overshoots, 80% of its N = 10 pictures hold more than 10 objects (GeckoNum, p. 19); two object types in one prompt: SDXL 14% against 60% for one noun (GeckoNum, Fig. 3, p. 7), and 2.44% when each count runs to 10 (D2D, Table 1, p. 7); "a times b" phrasing cuts FLUX 1.1 from 0.38 to 0.16 (T2ICountBench, Table 3, p. 11) | read from the paper (and by eye where marked); the digit push read from the project's findings |
| **Binding strength** | SynGen's and EBAMA's attention losses; Attention Regulation's penalty; FRAP's per-token weights; Baumann's per-subject token direction; guidance scale; Relation Rectification (a relation, not binding); this project's CLIP species push | **No.** SDXL colour attribution across guidance scale 5.5 to 16: 26.88, 28.00, 26.00, 26.00, 25.00, 27.00, 24.00, 31.00, 25.00% (Guidance Matters, Table 2, p. 9), no trend. EBAMA's regulariser weight 0 to 1 moves text-caption similarity on two objects by at most 0.012 (Fig. 10, p. 22, by eye). The project's species push at strength 0, 10, 30 gave 8, 1, 4 separate animals of 8. The only steady relation curve: Relation Rectification position accuracy 46.7, 56.4, 64.6, 69.7, 72.9% at λ 0, 0.2, 0.4, 0.6, 1, with FID 83.73 to 110.01 (Table 1, p. 7) | EBAMA's step size floods the attribute across the scene at 40 (p. 23); SynGen blurs or flips attributes above about 20 (p. 15); Baumann's colour leaks when pushed far (p. 12); FRAP's fixed weight 1.4 costs realism, 0.746 to 0.694 (Table 9, p. 25) | read from the paper (and by eye where marked); the species push read from the project's findings |

## 1. Co-occurrence strength

| Source | Space and strength | Strength against outcome | Carries to new subjects | Where it breaks | Mark |
|---|---|---|---|---|---|
| This project, the strength sweep ([the correction finding](../../../../1-lora-correction/report/what-the-correction-is/does-the-correction-cause-composition.md), [the strength grid page](../../../../1-lora-correction/report/measuring-composition/how-big-the-strength-grid-should-be.md)) | score: the exact correction (joint prompt's prediction minus the product's) added at weight λ every step | two separate animals 1, 3, 8, 23, 30 of 32 at λ 0, 0.25, 0.5, 0.75, 1; wrong pair 1, 1, 1, 2, 1; wrong seed 1, 1, 1, 2, 3; wrong order 1, 1, 1, 1, 0 (strength grid page, lines 43 to 48); area under the curve 0.387 against 0.023 to 0.047 for the controls (the results file, key `auc`) | 8 pairs at seeds 9 to 12: cat and dog, cow and buffalo, frog and toad, goose and swan, leopard and jaguar, seal and walrus, eagle and hawk, elephant and penguin (the results file, key `scores`) | the exact term needs the joint prompt's prediction; the steepest step is 0.5 to 0.75; the scorer counts animals and cannot tell a cat and a dog from two dogs (cat and dog, seed 10, counted 2, shows two dogs: [the compose rate page](../../../../context/world/compose-rate.md), "What comes out") | read from the project's findings |
| This project, SuperDiff's own term ([the SuperDiff finding](../../../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-superdiff-compose-at-its-own-defaults.md)) | score: SuperDiff's missing term at weight λ | cat and dog: one animal on all 4 seeds at 0 and 0.25, two by 0.75 on all 4; butterfly and meadow two by 0.25 (lines 52 to 63) | 2 pairs | where the switch sits inside 0.25 to 0.75 is not measured | read from the project's findings |
| Matsuhira et al., "Investigating Conceptual Blending of a Diffusion Model for Improving Nonword-to-Image Generation", arXiv 2411.03595, ACM MM 2024 | text embedding: $r e_A + (1-r) e_B$, SD 1.4 | both words present (CLIP score over 0.15) 0.143, 0.301, 0.348, 0.521, 0.621, 0.593, 0.416, 0.257, 0.257 at r 0.1 to 0.9; word A alone rises 0.411 to 0.948 between r 0.3 and 0.5 (Table 1, p. 4); about 1,000 random noun pairs, 101 to 125 per ratio, 10 pictures each | random pairs, so new subjects by design | **peaks** at 0.5; at the peak 38% of pairs still show only one; "both present" counts hybrids and side-by-side subjects alike (p. 4) | read from the paper |
| Liew et al., "MagicMix: Semantic Mixing with Diffusion Models", arXiv 2210.16056 | latent: $\nu \cdot$ generated $+ (1-\nu) \cdot$ noised layout, between 0.6T and 0.3T (p. 8) | corgi layout, "coffee machine" content: a coffee machine at ν 1, leaning corgi-shaped at 0.8 and 0.6, a corgi-like coffee machine at 0.5 (as the paper reports it), a corgi beside a separate cup at 0.4 and below (Fig. 7, p. 8, by eye); no metric | shown on several pairs only | a **switch** between one object, a hybrid and two; best ν 0.1 for corgi and husky, 0.9 for corgi and coffee machine (p. 8); unlike shapes are pasted side by side (p. 10) | read from the paper |
| Hoe et al., "InteractDiffusion: Interaction Control in Text-to-Image Diffusion Models", arXiv 2312.05849, CVPR 2024 | self-attention: a trained gated block over subject, action and object boxes, on for a share ω of the steps | interaction detection on the full set about 0.5, 6, 18, 24, 27, 28.4, 29, then 29.1 to 29.4 at ω 0 to 1 by 0.1; FID 35.8 falling to 18.8 (Figs. 10 and 11, p. 11, by eye) | **measured**: held-out interaction classes 20.75 against 23.85 seen (Table 4, p. 14) | **saturates** from 0.6; needs boxes and a trained module | read from the paper |
| Bao et al., "Separate-and-Enhance: Compositional Finetuning for Text2Image Diffusion Models", arXiv 2312.06712, CVPR 2024 (venue inferred) | cross-attention key projection fine-tuned with losses pushing object masks apart and raising each object's peak | no loss-weight sweep (p. 7); training strength: success 0.372, 0.410, 0.388, 0.393 at four learning-rate and step settings, FID 36.41 to 54.80 (Table C, p. 12) | **measured**: success 0.299 seen pairs, 0.305 seen with unseen, 0.294 unseen pairs, against SD 0.212, 0.227, 0.203 (Table 2, p. 7) | absolute success stays near 0.3 (p. 7) | read from the paper |
| Zhang et al., "Enhancing Semantic Fidelity in Text-to-Image Synthesis: Attention Regulation in Diffusion Models", arXiv 2403.06381 | cross-attention edited toward target tokens with a proximity penalty β | CLIP similarity 0.3457, 0.3467, 0.3485, 0.3267, 0.3266 at β 0.001, 0.01, 0.1, 1, 10 (Fig. 5, p. 13, by eye): on below 0.1, off from 1 | both objects detected 72.5% against 55.9% on SD 2 (Table 1, p. 11) | a **switch**; SD 2 still fuses spider and glasses (Fig. 8, p. 14) | read from the paper |

## 2. Count

| Source | Space | Requested N against outcome | Two object types in one prompt | Where it breaks | Mark |
|---|---|---|---|---|---|
| Kajić et al., "Evaluating Numerical Reasoning in Text-to-Image Models" (GeckoNum), arXiv 2406.14774, NeurIPS 2024 Datasets and Benchmarks | benchmark, nothing steered; human counts | SDXL 62, 55, 40, 20, 10, 11, 4, 5, 3, 2% and SD3 74, 72, 50, 46, 24, 12, 10, 10, 8, 5% for N = 1 to 10 (Fig. A2, p. 19) | **tested**: SDXL 60% one noun, 14% two, 18% three; SD3 82%, 42%, 41% (Fig. 3, p. 7) | SDXL overshoots: more than 10 objects in 13% of N = 1, 25% of N = 5, 80% of N = 10 pictures (p. 19); digits score worse than number words for 10 of 12 models (p. 8) | read from the paper |
| Lee et al., "CountCluster", arXiv 2508.10710 | cross-attention of the object token at the first step, split into N clusters, latent pushed toward them | SDXL 93, 74, 93, 63, 63, 37, 37, 26, 15% against plain 62, 47, 45, 30, 23, 24, 15, 2, 4% for N = 2 to 10 (Fig. 4, p. 7, by eye); pooled 27.88% to 55.75% (Table 1, p. 6) | not tested | bumps at N = 4 and 9, which the authors tie to grid layouts (p. 7); the push size is fixed at 10,000 and never swept (p. 13) | read from the paper |
| Jo et al., "ISAC", arXiv 2505.20935 | self-attention clustered into N instance masks, cross-attention binds the class to them | SDXL 90, 71, 49, 32% to 96, 89, 71, 47% for N = 2 to 5 (Table 3, p. 12); FLUX.1-dev plain 97, 89, 82, 66% (Table 5, p. 14) | partly: one each of k classes, SD3.5-M 62, 23, 12, 3% to 98, 51, 40, 20% for k = 2 to 5 (Table 2, p. 11) | instance term alone: count right 65%, class right 10% (Table 7, p. 15) | read from the paper |
| Yoo et al., "D2D: Detector-to-Differentiable Critic for Improved Numeracy", arXiv 2510.19278, NeurIPS 2026 | starting noise modified under a detector's differentiable count | CoCoCount: SDXL-Turbo 27.38% to 55.62% (Table 21, p. 25) | **tested**: SDXL 2.44%, SDXL-Turbo 2.12% to 9.81% with each count up to 10 (Table 1, p. 7) | 78% of first draws at N = 11 to 20 overshoot (p. 24) | read from the paper |
| Zeng et al., "YOLO-Count", arXiv 2508.00728, ICCV 2025 | a learned count token against a differentiable counter | mean count error at N = 25, 50, 75, 100: plain 20, 29.5, 36, 45.5; guided 13, 16, 17.5, 17; unseen categories 15.5, 17, 19.5, 21 (Fig. 5, p. 7, by eye) | not tested | cannot reduce an overcount (p. 13) | read from the paper |
| Guo et al., "Text-to-Image Diffusion Models Cannot Count, and Prompt Refinement Cannot Help", arXiv 2503.06884 | benchmark and prompt rewrites | FLUX 1.1 0.75, 0.22, 0.06 and SD3.5 Large Turbo 0.59, 0.13, 0.02 for N 1 to 5, 6 to 10, 11 to 15 (Fig. 1, p. 7; any of 4 draws right) | not tested | "a times b" phrasing: FLUX 1.1 0.38 to 0.16, average over 15 models 0.42 to 0.26 (Table 3, p. 11) | read from the paper |
| Sun et al., "When Numbers Speak" (NUMINA), arXiv 2604.08546 | self- and cross-attention heads of a video model | Wan2.1 plain 68.7, 44.5, 38.6, 31.7, 18.6, 15.2, 11.3% to 71.0, 60.7, 41.6, 40.7, 30.3, 26.3, 20.7% for N = 2 to 8 (Fig. 6, p. 7) | pictures only | the number word's own cross-attention is diffuse while the noun's is local (p. 2) | read from the paper |
| Xie et al., "Guidance Matters", arXiv 2602.22570, ICLR 2026 | guidance scale | SDXL GenEval counting 37.66, 42.50, 41.25, 43.75, 48.75, 47.50, 48.75, 46.25, 36.25% at scale 5.5 to 16.05 (Table 2, p. 9): **peaks** near 7.6 | not separately | about 100 prompts per category, so one prompt is one point (inferred) | read from the paper |
| This project, the count checker's push on digits ([the count push finding](../../../../2-classifier-guidance/report/abstract-checkers-on-digits/does-the-count-checkers-push-compose-unseen-digits.md)) | score: a checker reading where each asked digit acts | push length b 0, 0.1, 0.2, 0.4 gave 0, 8, 20, 40 clean pairs of 320 on seen digits (lines 174 to 176); 43 of 320 on unseen digits at 0.4 | two digits, each a different identity | b above 0.4 not tried | read from the project's findings |
| Already in [H1](01-directions-that-generalise.md#4-scene-relations-and-counts-as-probes-or-directions) | ATHENA, Numerosity, Make It Count, CountSteer | accuracy falls with N in ATHENA (Fig. 6, p. 8) | | | cited, not reread |

**On "dog times 2".** No paper read sets a count by repeating a subject's token or scaling its attention
map; the searches found only clustering, layout and optimisation methods. The methods that move the
curve read instances from the noun's attention maps, never from the number word, whose map is diffuse
(NUMINA, p. 2). The nearest prompt test, the count written as "a times b", made accuracy worse.

## 3. Binding strength

| Source | Space and strength | Strength against outcome | Where it breaks | Mark |
|---|---|---|---|---|
| Xie et al., "Guidance Matters", arXiv 2602.22570, ICLR 2026 | guidance scale | SDXL GenEval colour attribution 26.88, 28.00, 26.00, 26.00, 25.00, 27.00, 24.00, 31.00, 25.00% at 5.5 to 16.05; two objects 73.74 to 78.79% with no trend (Table 2, p. 9) | CLIPScore, HPSv2 and ImageReward rise with guidance scale on their own (Fig. 3, p. 5), so a dial scored by them can mislead | read from the paper |
| Zhang et al., "Object-Conditioned Energy-Based Attention Map Alignment" (EBAMA), arXiv 2404.07389, ECCV 2024 | cross-attention binding loss, latent step α, regulariser weight λ | text-caption similarity on two objects 0.828, 0.839, 0.840, 0.840, 0.831 at λ 0 to 1 (Fig. 10, p. 22, by eye); α 1, 20, 40 shown only | **peaks**: at α 40 blue spreads across the picture and black onto the buildings behind (p. 23) | read from the paper |
| SynGen, 2306.08877 | cross-attention, step α | larger α separates more "only to a point", blur and flipped attributes above about 20 (p. 15), pictures only | parser misses leave it as plain SD (p. 10) | read from [the project's note](../abstract-checkers/01-reading/syngen-linguistic-binding.md) and the paper |
| Baumann et al., 2403.17064 | a direction on one subject's token | subject-specificity 3.35 in two-subject prompts (Table 1, p. 6) | colour leaks to the other object when pushed far (p. 12) | read from the paper ([H1, section 2](01-directions-that-generalise.md#2-concept-sliders-and-text-token-directions)) |
| FRAP, 2408.11706 | per-token weight | fixed 1.4: alignment 0.815 to 0.841, realism 0.746 to 0.694 (Table 9, p. 25) | quality falls with a fixed weight | read from the paper ([H2, section 1](02-shared-representations.md#1-the-text-embedding)) |
| This project, the CLIP species push ([the second-push finding](../../../../2-classifier-guidance/report/classifier-guided-poe/can-a-second-push-name-each-animal.md)) | a push toward each animal's own species on top of the separating push | separate animals of 8 at strength 0, 10, 30: 8, 1, 4 (lines 50 to 54) | moves texture, never which animal is which | read from the project's findings |
| Relation Rectification, 2403.20249 (a relation, the nearest steady curve) | offset on the end token | position accuracy 46.7, 56.4, 64.6, 69.7, 72.9% at λ 0, 0.2, 0.4, 0.6, 1; FID 83.73 to 110.01 (Table 1, p. 7) | needs a larger λ, and more quality lost, when the model lacks the concept (p. 13) | read from the paper ([H2, section 1](02-shared-representations.md#1-the-text-embedding)) |

## What a working dial would need

Inferences, resting on the rows above and on H1 and H2.

1. **A target that is a measurement, not a word.** The only steady curves here have a measured or exact
   target: the exact correction, the count checker's push, Relation Rectification's offset trained on
   images. H1's Measured Sliders comparison (9.6% against 99.8% ordered sweeps on matched training,
   p. 9) points the same way.
2. **A scale that spreads the switch.** The co-occurrence curve does most of its moving between λ 0.5
   and 0.75. A 1-to-10 dial over it would sit near the bottom for half its range unless the strength is
   mapped through the measured curve.
3. **A score that tells a hybrid from two subjects.** Every published co-occurrence score read here
   counts a hybrid as success; the project's own scorers can separate them, and the dial is only as good
   as that score.
4. **A count dial that reads instances from the noun's maps and acts on the first steps.** The number
   word carries little, the starting noise carries a preferred count, and two subject types in one
   prompt is where every method fails.
5. **Binding needs a definition of strength before it needs a method.** No paper measures leakage
   against a strength; the nearest scores (CLIP similarity, GenEval colour attribution) barely move.

## Where the sources disagree

Reported, not resolved.

- **MagicMix at ν 0.5.** The paper calls it a corgi-like coffee machine; the reader of the rendered figure
  saw a corgi with a cup beside it (Fig. 7, p. 8).
- **Separate-and-Enhance's smallest learning rate.** The text says 5e-7, Table C prints 7e-7 (p. 12).
- **YOLO-Count's baseline.** Fig. 5 labels it "Vanilla SDXL"; the text says SDXL-Turbo (p. 7).
- **Which controls stay low in the project's strength sweep.** The strength grid page gives three
  controls (wrong pair, wrong seed, wrong order) at up to 3 of 32; the review of the sweep, as that page
  quotes it, gives two controls at or below 6%. The results file holds four controls, with wrong step at
  most 1 of 32 and a random one at most 1 of 32.

## Still open

- **No published strength sweep on "two subjects together" scores hybrid and two separately.** The
  project's own sweep is 8 pairs at 4 seeds; how it behaves past λ 1, and on pairs where the exact term is
  not available, is untested here.
- **No paper tests "dog times 2" as token repetition or attention scaling.** Whether scaling a noun's
  attention map sets a count is unknown, not shown to fail.
- **No paper plots attribute leakage against a strength.** Binding strength has no agreed measure.
- **Not opened:** TokeBi (not found under that name); InteractDiffusion's results beyond ω.

## Where this came from

Two reading passes on 2026-11-25 opened every PDF with `pdftotext` and rendered plotted pages with
`pdftoppm` for the values marked by eye; their notes are kept as
[the count notes](03-reading/count-reader-notes.md) and
[the co-occurrence and binding notes](03-reading/cooccurrence-and-binding-reader-notes.md). Every arXiv
id was checked against its title. The project's curves come from
`outputs/interaction_term/dose/dose_curves.json` (keys `lambdas`, `curves`, `auc`, `scores`, read
2026-11-25) and the report pages linked in each row. The figure is drawn by
[its script](03-strength-against-outcome.py) with
`~/.claude/venvs/figures/bin/python 03-strength-against-outcome.py` from this folder.
