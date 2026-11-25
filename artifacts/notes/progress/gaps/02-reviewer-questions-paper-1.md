# What a reviewer will ask of Paper 1

Paper 1 composes diffusion models trained on separate data, so that no model has seen its concepts
together ([the two problem statements](../problem-statements.md#paper-1-composing-experts-that-never-met)).
This page lists 19 questions a reviewer will ask, the papers that bear on each, and the measure that would
answer it. 15 of the 19 have a measure. 11 of those 15 map to a measure already in
[Loop 4's evaluation plan](../targets/paper-1-evaluation-plan.md) (CLEVR measures 1 to 5, chest X-ray
measures A to C), 1 maps to [the CLEVR scaling plan](../targets/clevr-scaling-plan.md), and 3 need a new
row, marked **new**; question 8 also adds a new sign for its second X-ray pair. The other 4 are positioning
or reporting questions that no measure settles.

## Table of contents

- [Words this uses](#words-this-uses)
- [The questions](#the-questions)
- [Is the owner's link between the two papers new](#is-the-owners-link-between-the-two-papers-new)
- [The owner's chest X-ray hypothesis against the literature](#the-owners-chest-x-ray-hypothesis-against-the-literature)
- [Where the sources disagree](#where-the-sources-disagree)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [How this was read](#how-this-was-read)

## Words this uses

**Expert.** A diffusion model trained on one kind of image only, such as one CLEVR colour or one chest
X-ray finding.

**Paste.** Each expert draws its own image and one image's objects are laid over the other's. It is the
control Paper 1 must beat; on CLEVR at fixed, separated places it puts every object in place in 1,024 of
1,024 images ([the paste finding](../../../../2-classifier-guidance/report/the-method-search/does-sample-and-paste-already-solve-small-clevr.md)).

**Interaction term.** What a product of experts leaves out: effects that need both subjects at once, such
as one object's shadow on another or a nearer object hiding a farther one
([the product of experts miniature](../../../../context/methods/01-the-product-of-experts/03-the-miniature.md);
[the manager's picture](../manager/experts-interact.png)).

**Marks.** **read**: read from the paper at the page given. **summary**: read from a search result, a web
article with no page numbers, or another loop's note. **inferred**: this page's reasoning.

## The questions

Ordered by how likely a reviewer is to raise it first (inferred). "Measure" names the row in Loop 4's plan
that answers it, or says **new**.

| # | The question | Why a reviewer asks it | Papers that bear on it | Measure that answers it |
|---|---|---|---|---|
| 1 | Is the method better than pasting the experts' samples together? | Counting objects cannot tell the two apart: paste places every object in 1,024 of 1,024 CLEVR images (the paste finding, section 1, verified there) | Bradley et al., Projective Composition (arXiv 2502.04549): score addition is exact only when each object's pixels are independent of the others', a condition shadows and occlusion break (p. 7, p. 16, read) | CLEVR measures 1 to 4 together, each scored against paste |
| 2 | Is it better than pasting followed by an off-the-shelf shadow generator? | Adding shadows to pasted objects is a published task with diffusion methods, so "paste has no shadows" is a weak control | MultiShadow (arXiv 2603.02743) generates shadows for several objects pasted into one scene, built on DESOBAv2 (p. 1, read); the DESOBAv2 diffusion shadow generator (arXiv 2403.15234, summary); a composition survey (arXiv 2106.14490) reports most methods give no reasonable shadow (summary) | **new**: a "paste plus shadow generator" row on CLEVR measures 2, 3 and 4. Whether MultiShadow runs on 64-pixel CLEVR is unknown |
| 3 | Do the objects agree on one light? | Two experts each trained on scenes with jittered lights can be drawn lit from different sides | Sarkar et al., Shadows Don't Lie (arXiv 2311.17138): object and shadow relations in generated images differ reliably from real ones (p. 2, read); its object-shadow classifier separates real from generated images at ROC area 0.63 to 0.86 across the generators in its Figure 3 (p. 6, read) | CLEVR measure 2. Sarkar's classifier could serve as a second reader if it reads 64-pixel CLEVR, which is untested (inferred) |
| 4 | Does one object's shadow fall on the other? | It is the clearest effect needing both objects, which neither expert has seen | Bradley et al. name shadows cast on each other as a violation of their exactness condition, and say the effect grows with more objects (p. 7, Figure 3a; p. 16, read). No paper read measures it for composed models | CLEVR measure 3, by two blind reviewers |
| 5 | Does the nearer object hide the farther one? | Occlusion needs both objects; paste gets it right only when the layer order happens to match depth | DepthArb (arXiv 2603.23924): OcclBench, 310 prompts, on SDXL and FLUX, scores occlusion by FOCR, the share of the two detected boxes' overlap covered by the nearer object's box, read by Grounding DINO, plus how recognisable the hidden object stays (p. 5, p. 6, read). Bradley et al. name occlusion as a violation (p. 7, read) | CLEVR measure 1. FOCR is a box-level version a reviewer may ask for beside the pixel-level reader (inferred) |
| 6 | Is the "missing interaction term" framing new? | Product-of-experts composition with a conditional independence assumption is well known | Bradley et al. (p. 7, p. 16, read); CoInD (arXiv 2503.01145) trains with a loss that drives the gap between the joint score and the summed scores to zero ([the project's CoInD note](../../../../3-understanding-compositionality/artifacts/notes/coind-conditional-independence-loss/README.md), summary); DCFM (arXiv 2606.10153) enforces conditional independence across silos (p. 2, read) | None: a positioning question, answered in [the section below](#is-the-owners-link-between-the-two-papers-new) |
| 7 | How does Paper 1 differ from DCFM, which composes models trained on separate silos, including chest X-rays? | DCFM is the closest published setting: separate data sources, novel combinations, NIH ChestX-ray14 | DCFM (arXiv 2606.10153): models are exchanged between peers during training and a global conditional independence penalty is enforced (p. 1, p. 2, read); on ChestX-ray14 with 4 silos it reports FID and "utility", the recall of a classifier trained on its samples for each disease combination (p. 8, read) | **new**: DCFM as a rival row on the chest X-ray measures; its utility score as a fourth X-ray measure beside A to C. DCFM's peers communicate during training while Paper 1's experts never do, which is the difference to state (inferred) |
| 8 | On chest X-rays, does one finding change how the other appears? | It is the X-ray version of question 4, and DCFM's independence penalty removes exactly this (inferred) | Felson's silhouette sign: a lesion touching a border of the heart erases that border (Radiopaedia, summary); silicotuberculosis: cavitation inside silicotic masses usually signals tuberculosis, 6 of 8 cavitated masses in one series (PMC2929613, summary) | X-ray measure C for effusion with cardiomegaly. **new** for tuberculosis with silicosis: cavitation inside a silicotic mass, read by a radiologist, as the cross-finding sign |
| 9 | Are ribs, spine, lungs and heart kept when two X-ray experts are multiplied? | The owner's hypothesis says the product keeps what both experts agree on | XReal (arXiv 2403.09240) controls anatomy and pathology by masks (summary); DCFM shows qualitative samples only (p. 8, Figure 8, read) | X-ray measure A |
| 10 | Are both findings present at all? | A product can drop the finding each expert treats as unlikely | DCFM's utility recall (p. 8, read) | X-ray measure B |
| 11 | Is the gain from a better sampler rather than the checker? | Du et al. show summed scores are sampled wrongly and fix it with MCMC correctors | Du et al., Reduce, Reuse, Recycle (arXiv 2302.11552; [its note](../score-space/01-reading/reduce-reuse-recycle.md), summary) | The corrector row in Loop 4's plan, read beside every measure |
| 12 | Does it hold beyond two or three objects? | Bradley et al. find interactions start and images degrade as more objects are composed (p. 16, read) | Bradley et al., Local Mechanisms (arXiv 2509.16447), Kmax 5 on CLEVR ([the benchmark spec](../benchmark-spec.md), summary) | The CLEVR scaling plan's count of objects, plus measures 1 to 3 at each count |
| 13 | Where do the regions come from, and is this layout-conditioned generation? | The proposed method has each expert denoise its own region (the problem statement); MultiDiffusion and layout methods already fill given regions | MultiDiffusion (reading plan sitting 7); DepthArb and OcclBench supply layouts (p. 5, read) | **new**: a row that gives paste and Bradley's rule the same regions, so the comparison does not reward the layout itself. Loop 6's two-colour test tells experts what to draw and not where, which removes the question there (summary, its design) |
| 14 | Has the checker seen images holding both concepts? | The setting allows a helper trained on the experts' samples; a reviewer will ask whether pasted or composed images leak a joint picture into it | Compositional Sculpting trains a classifier on pretrained models' samples (problem statements, summary); the digit test's checker trained on pasted canvases (two-colour design, summary) | None by a score: an audit listing every image the checker trained on and how it was made |
| 15 | Do the composites look real, with no seams? | Pasted objects carry lighter seams (paste finding, read there) | Kernel Inception distance and a real-or-fake classifier (Loop 4's plan) | CLEVR measure 4 |
| 16 | Does each colour stay with its object? | A method can gain on measures 1 to 4 by blending two objects into one | ToMe and attribute-leakage papers (see [G1's ToMe note](notes/tome.md)) | CLEVR measure 5, as a guard |
| 17 | Are these effects readable at 64 by 64 pixels? | CLEVR objects cover a few dozen pixels and shadows fewer | Bradley et al. ran 128 by 128 ([the benchmark spec](../benchmark-spec.md), summary) | The reader checks in measures 2 and 3, which can declare them unreadable |
| 18 | Are Bradley's printed numbers a fair comparison? | Bradley's models trained about 16 times longer | [The benchmark spec](../benchmark-spec.md), lines 55 to 57 (summary) | None: printed numbers stay context; every row runs on the same experts |
| 19 | Which chest X-ray data holds both findings, and is silicosis available at all? | A measure needs real images with both findings for its readers | DCFM uses NIH ChestX-ray14, 14 labels, with about 99% of images holding 3 or fewer diseases and at most 9 together (p. 8, read). Whether silicosis is among its 14 labels was not checked here | None until a dataset is chosen; Loop 4's plan lists it as open |

## Is the owner's link between the two papers new

**Partly.** The idea that a product of separately trained experts breaks where objects interact is
published; measuring what breaks is not, in the papers read here.

| What the link says | Published already | Where |
|---|---|---|
| Shadows and occlusion break the exactness of composing by adding scores | yes, as a caveat | Bradley et al., p. 7 ("objects can cast shadows or even occlude each other") and p. 16 ("overlap or other interaction (shadows, etc.) between objects can clearly violate all three criteria"), read |
| The gap between the joint score and the summed scores is a term that can be written down | yes, as a training penalty driven to zero | CoInD, [the project's CoInD note](../../../../3-understanding-compositionality/artifacts/notes/coind-conditional-independence-loss/README.md), summary |
| Independence between factors should be enforced across separate data sources | yes, the opposite stance | DCFM, p. 2, read |
| What the product loses, measured on effects that need both subjects (cross shadows, occlusion, one X-ray finding changing another) | not found | searched: arXiv and web for interaction terms in composed diffusion models with occlusion or shadows, chest X-ray composition of co-occurring findings, and occlusion benchmarks; the closest are OcclBench (single model, not composed) and DCFM (independence, not interaction) |

**Recommendation (inferred).** Paper 1 can claim the measurement: the first evaluation of composed
separately trained experts on effects that need both subjects, with paste as the control. It cannot claim
the observation that interaction breaks composition, which Bradley et al. print. Adopting the link in the
problem statements stays the owner's decision. For Paper 2, the link explains plain PoE's failure and not
the joint prompt's ([G1's section](01-is-paper-2-still-open.md#does-the-missing-interaction-term-change-paper-2s-case)).

## The owner's chest X-ray hypothesis against the literature

The hypothesis: both experts agree on anatomy, so a product keeps ribs, spine and lungs, while each expert
treats the other's finding as unlikely.

| Part | What the literature read here says | Mark |
|---|---|---|
| A product keeps shared anatomy | no paper read tests a product of X-ray experts; XReal and DCFM generate anatomy with one model | not tested in the literature found |
| Each expert treats the other's finding as unlikely | consistent with DCFM's finding that experts trained on skewed silos fail to produce combinations none of them saw, which its global penalty addresses (p. 2, p. 4, read) | supported in kind, on DCFM's own data |
| Effusion changes how the heart's edge looks | the silhouette sign says a lesion touching a heart border erases it (Radiopaedia, summary). No source found states it for pleural effusion beside cardiomegaly specifically; one review notes cardiomegaly with a left-dominant effusion can instead signal a pericardial effusion (summary, web) | partly supported; the specific pair is not sourced |
| Tuberculosis with silicosis has a combined appearance | plain films often cannot tell the two apart, and cavitation inside a silicotic mass usually means tuberculosis is present (NIOH silicotuberculosis guide; PMC2929613, summary) | supported: cavitation is the cross-finding sign for measure C's second pair |

## Where the sources disagree

None found between sources read for this page. Loop 4's plan expects paste to get occlusion right "about
half" the time (predicted there, not measured); no source here measures it.

## What this cannot tell you

- Whether MultiShadow or Sarkar's classifier work on 64-pixel CLEVR renders.
- Whether NIH ChestX-ray14 or CheXpert hold enough images with effusion and cardiomegaly together; DCFM's
  counts are for all 14 labels together.
- Which questions reviewers will actually raise; the order is this page's judgement.
- Whether any paper not found by these searches already measures cross-object shadows for composed models.

## How this was read

Read from full PDFs fetched from arXiv for this page: DCFM (2606.10153), Projective Composition
(2502.04549), Shadows Don't Lie (2311.17138), DepthArb (2603.23924) and MultiShadow (2603.02743). Every page
number was found by locating the line in the extracted text and counting page breaks. Read from search
results only: DESOBAv2 (2403.15234), the composition survey (2106.14490), XReal (2403.09240), the
Radiopaedia silhouette sign article, the NIOH silicotuberculosis guide and PMC2929613. Cited from project
notes: CoInD, Reduce Reuse Recycle, Local Mechanisms, MultiDiffusion, Compositional Sculpting. Searches run:
shadow consistency in generated images; shadow generation for composites; interaction terms in composed
diffusion with occlusion or shadows; chest X-ray composition of co-occurring findings; occlusion
benchmarks for text-to-image; the silhouette sign with effusion and cardiomegaly; silicotuberculosis on
radiographs.
