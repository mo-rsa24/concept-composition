# Open problems in composing diffusion models

Eight open problems, ranked by how large the published gap is and how close it sits to this project's two
papers. Each comes with the papers that leave it open, why it is open, a first small experiment and a rough
cost. Two further sections follow: what each SDXL counterpart of Paper 2 still gets wrong, which is the list
Paper 2 has to beat first in the owner's order of claims, and where DCFM's open gaps meet this project's
questions.

## Table of contents

- [Words this uses](#words-this-uses)
- [The ranked problems](#the-ranked-problems)
- [What each SDXL counterpart still gets wrong](#what-each-sdxl-counterpart-still-gets-wrong)
- [DCFM's gaps against this project's questions](#dcfms-gaps-against-this-projects-questions)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [How this was read](#how-this-was-read)

## Words this uses

**Look-alike pair.** Two subjects similar in shape or kind, such as a duck and a goose, or a cat and a dog.

**Borrowed part.** One of two separately drawn subjects carries a part of the other: a face, markings, a
head. **Extra subject.** A third subject drawn when the prompt names two.

**Finished experts.** Models trained once and then composed only at sampling time, with no retraining.

**Cost.** GPU-hours on a cluster RTX 8000 unless named. SDXL costs use the measured 43.2 to 58.3 seconds a
picture at 50 steps ([the reviewer's check](../reviewer/2026-11-24-review.md), the assumptions table). Every
cost below is an estimate (inferred) and runs nothing in this session.

**Marks.** **read**: read from the paper at the page given. **summary**: read from another loop's
page-cited note or a search result. **inferred**: this page's reasoning.

## The ranked problems

| Rank | Open problem | Papers that leave it open | Why it is open | First small experiment | Rough cost | Serves |
|---|---|---|---|---|---|---|
| 1 | Look-alike pairs: borrowed parts and extra subjects | MMDiT similar subjects: plain SD3 draws both look-alike subjects in 53.1% of images, the best frozen fix 68.8% (Table 1, p. 8, summary); DeLeaker: 33% of plain animal-pair images leak on FLUX.1-dev (summary, [the manager's search](../manager/who-counts-borrowed-parts-and-extra-subjects.md)); CO3 on SDXL: 34 clean, 33 borrowed and 21 not two subjects of 88 hand-labelled pictures, 20 of the 21 with an extra animal (the manager's search, from `noise-text-critic/02d-binding/part-binding-calls.csv`) | No paper counts borrowed parts on SDXL or for any SDXL method, and no paper counts extra subjects on a two-subject prompt (the manager's search). The usual scores (BLIP-VQA, GenEval's detector, ImageReward) cannot see either failure (same source) | Count extra animals above two per picture with the project's detector on CO3's 88 existing pictures, then on the joint prompt's and ToMe's pictures of the same 11 pairs and 8 seeds | First part: no GPU, existing pictures. Joint prompt and ToMe renders: 176 pictures, about 2.1 to 2.9 GPU-hours | Paper 2 |
| 2 | Composing finished experts on combinations none of them saw | DCFM: its baselines that route between finished experts fail on novel combinations (p. 4, Figure 3; p. 7, Table 1, read); Catastrophic Compositional Generation: FKC with separately trained models shows a painting in none of 8 rooms, and one out-of-distribution picture is destroyed into noise (Fig. 3, p. 28, as in [the failures catalogue](../targets/failures.md), summary) | Every method that succeeds retrains or distils its experts (DCFM-A, DCFM-B, CoInD); a sampling-time correction on finished experts has no published success on novel combinations (inferred from the above) | Paper 1's checker on DCFM's Colored MNIST split with finished experts, scored by DCFM's own novel-combination FID and recall | unknown: DCFM states cost in passes, not hours (p. 9, read); the project's digit runs are the nearest size | Paper 1 |
| 3 | What a product of separately trained experts loses where objects interact | Bradley et al., Projective Composition: shadows cast on each other and occlusion break exact composition, and the effect grows with more objects (p. 7, p. 16, read); CoInD and DCFM drive the interaction to zero by design (see [G2](02-reviewer-questions-paper-1.md#is-the-owners-link-between-the-two-papers-new)) | Named as a caveat and never measured in any paper found ([G2](02-reviewer-questions-paper-1.md)) | Loop 4's CLEVR measure 3 (one object's shadow on the other) on paste, Bradley's rule and Paper 1's method | about 18 CPU node-hours of reference renders plus under 1 GPU-hour ([Loop 4's plan](../targets/paper-1-evaluation-plan.md), "What it costs") | Paper 1 |
| 4 | Composition when almost no combinations were seen | DCFM requires every value seen somewhere and a minimum share of combinations seen, and runs only at half and three quarters (p. 3, Definition 3.1; p. 6 to p. 8, as in [Loop 4's DCFM note](../targets/dcfm-compositional-flow-matching-from-separate-silos.md#what-dcfm-leaves-open), summary) | Below that share, DCFM's guarantee does not apply and nothing is tested; Paper 1's setting, experts that never met, sits there | DCFM's Colored MNIST at a coverage below its minimum, with DCFM's penalty and with Paper 1's method on the same experts | unknown, as in rank 2 | Paper 1 |
| 5 | Medical images where one finding changes how another looks | DCFM's chest X-ray utility is below 0.25 for every method, with no anatomy check and no per-combination breakdown (p. 8, Figure 9, summary from Loop 4's note); the silhouette sign says a lesion touching the heart erases its border (Radiopaedia, summary, in [G2](02-reviewer-questions-paper-1.md#the-owners-chest-x-ray-hypothesis-against-the-literature)) | No paper composes X-ray findings and checks the cross-finding effect; the data for both findings together is thin (G2, question 19) | X-ray measure A (anatomy kept) on DCFM-style samples and on a two-expert product, on ChestX-ray14 effusion and cardiomegaly | not costed: no X-ray data or expert exists in the project ([Loop 4's plan](../targets/paper-1-evaluation-plan.md#chest-x-ray-measures)) | Paper 1 |
| 6 | Prompts with three or more subjects, or relations | T2I-CompBench++ complex, 3-in-1 metric 0 to 1: SDXL 0.3237, SD3 0.3771, FLUX 0.3703 (Table XIII, p. 12, read); plain SD3 draws three look-alike subjects in 37.9% of images and four in 33.1% (MMDiT paper, Table 1, p. 8, summary); CO3 and ToMe drop objects in complex scenes (the failures catalogue, RTD Figs D.3 and D.4, pp. 17 and 18, summary) | Every model and method read here stays below 0.40 on the complex category ([G1](01-is-paper-2-still-open.md#how-far-published-methods-get-on-each-kind-of-pair)) | Out of Paper 2's scope by the owner's design. A first look: the kept push on three-animal prompts, scored by the detector's count | about 1.2 to 1.6 GPU-hours for 100 SDXL pictures | neither paper now |
| 7 | Binding a colour or attribute to each object | GenEval colour attribution: 0.43 to 0.60 on every released or frozen model, 0.86 only after RL training ([G1](01-is-paper-2-still-open.md#the-numbers-by-model), read) | Open, but crowded: ToMe, CoMat, SynGen and RL fine-tuning all target it ([G1's ToMe note](notes/tome.md)) | None proposed: outside both papers | none | neither |
| 8 | Occlusion order in one model's picture | DepthArb: OcclBench, 310 prompts on SDXL and FLUX, scored by the nearer object's cover of the overlap (p. 5, p. 6, read) | A single model drawing occlusion is a layout problem with its own methods (DepthArb, layout-to-image work); it touches this project only through rank 3 | None for this project; CLEVR measure 1 covers the composed case | none | neither |

**Why this order (inferred).** Rank 1 is the largest unmeasured gap on Paper 2's own pairs, and its first
experiment needs no GPU. Ranks 2 to 5 are where DCFM, the closest published work to Paper 1, stops. Ranks 6
to 8 are open but outside both papers' scope or already crowded.

## What each SDXL counterpart still gets wrong

Ranked by how much each still fails on Paper 2's four kinds of pairs (two animals, an animal and an
object, two objects, two subjects with attributes), from the failures each paper or a rival prints.
Most rows come from [Loop 4's failures catalogue](../targets/failures.md), which read every figure at full
size; authors choose their own figures, so a missing failure says nothing.

| Rank | Counterpart | What it still gets wrong, with the page | Rate where one exists | Inputs it needs |
|---|---|---|---|---|
| 1 | CO3 | Fuses a lion with a horse and a frog with a mouse (Rectify Then Diffuse, Fig. D.1 p. 15, Fig. D.2 p. 16); adds a tiny extra animal beside a horse (CO3 Fig. 7, p. 19); fuses a brown seat into a white bench and moves colours across objects (CO3 Fig. 3, p. 8); drops the second object on several T2I-CompBench prompts (RTD Figs D.3 and D.4, pp. 17 and 18). All summary, from the catalogue | 33 of 88 borrowed and 20 extra animals on SDXL in the project's hand labels (the manager's search); a clean cat and dog on 89 of 128 seeds by the scorer, 101 by eye ([the report index](../../../../report/00-INDEX.md), the kept push comparison) | the prompt and each subject's prompt; no gradients (CO3 p. 7, read) |
| 2 | ToMe | Fuses a lion with a horse and draws two frogs with no mouse (RTD Fig. D.1 p. 15, Fig. D.2 p. 16); adds a second turtle (CO3 Fig. 3, p. 8); loses the second subject in all three spliced-prompt pictures (ToMe Fig. 13, p. 19). Summary, from the catalogue | BLIP-VQA on animal pairs 0.6257, below plain SDXL's 0.6950 (CO3 Table 1, p. 7, read) | a parser for objects and attributes, gradients through cross-attention (ToMe p. 7, read) |
| 3 | Attend-and-Excite | Its own limitations figure draws a dog alone, a dog with a plush toy, and a bear head on a dog body for "a dog and a bear"; a cat and a dog share one jacket (catalogue, Attend-and-Excite rows, summary). Its authors ran SD1.4 only (Loop 7's note, p. 10, summary), so no SDXL number exists from them | Fused creatures in 11.5% of images on SD1.4 (A Cat Is A Cat, Table 4, p. 8, in the manager's search) | each subject's token; gradients through cross-attention |
| 4 | FKC | On SDXL with two coloured objects, a TV gains two extra toothbrushes and a red bag gains green handles (FKC Fig. A5, p. 39); with separately trained models, no out-of-distribution room of eight shows a painting (Catastrophic Fig. 3, p. 28). Summary, from the catalogue | GenEval two objects 0.78 on SDXL, a measure that ignores extra objects and borrowed parts (FKC Table A4, p. 39; GenEval p. 5; summary) | particle weights over several samples per picture |
| 5 | Bounded Attention | Needs a box for every subject from the user (p. 5); its SDXL results on look-alike subjects are pictures only, with no number (p. 8, p. 9); its own authors say its counting metrics "have limitations in capturing semantic leakage" (p. 9). Summary, from [Loop 12's note](../leakage/01-reading/bounded-attention.md) | Counting precision 0.83 and recall 0.88 on SD (Table 1, p. 9, summary); none on SDXL | a box per subject; gradients through attention |

**What Paper 2 must beat first (inferred).** Borrowed parts and extra subjects on look-alike animal pairs
appear in the failures of every counterpart above, and only the project's hand labels count them on SDXL.
The first comparison is CO3, ToMe and Attend-and-Excite on the project's 11 animal pairs, counted for clean,
borrowed and extra, against the same with Paper 2's correction. FKC and Bounded Attention need inputs
(several particles, user boxes) that make a matched comparison harder, so they come after.

## DCFM's gaps against this project's questions

Each gap is one of the eight limits in
[Loop 4's DCFM note](../targets/dcfm-compositional-flow-matching-from-separate-silos.md#what-dcfm-leaves-open),
with its page there. The questions weighed: Paper 1's measures where pasting fails
([Loop 4's evaluation plan](../targets/paper-1-evaluation-plan.md)), the interaction a product drops
([G2](02-reviewer-questions-paper-1.md)), and Paper 2's early correction
([Paper 2's design](../manager/paper2-four-way-design.md)). "Closed" means one of our questions would answer
the gap if run; nothing here has run.

| # | DCFM's gap | Which of our questions meets it | Status | Experiment that would show it |
|---|---|---|---|---|
| 1 | What forcing independence costs is never measured (p. 3, p. 5, p. 6, p. 9) | The interaction a product drops: CLEVR measures 1 to 3 and X-ray measure C; Paper 2's correction exists to supply that term on plain PoE | closed by our question | X-ray measure C (the heart's left border contrast) on DCFM-style samples and on ours, effusion with cardiomegaly |
| 2 | It composes attributes of one thing, on three small datasets, never above 256 pixels (p. 6 to p. 8) | Paper 1 composes separate objects (two-colour CLEVR); Paper 2 runs natural images from text on SDXL at 1,024, though in one model, not silos | partly closed | Paper 1's two-colour CLEVR test, where each expert draws separate objects, reported beside DCFM's single-digit task |
| 3 | The experts never stay finished: DCFM-A retrains them, DCFM-B distils them, and the finished-expert baselines fail on novel combinations (p. 4 to p. 7) | Paper 1's question: finished experts plus a checker at sampling time | closed by our question | rank 2 above: Paper 1's checker on DCFM's Colored MNIST split with finished experts, scored by DCFM's FID and recall |
| 4 | Every value must be seen somewhere and a minimum share of combinations seen; nothing below is tested (p. 3) | Paper 1's setting, experts that never met | closed by our question | rank 4 above: Colored MNIST below DCFM's minimum share, both methods on the same experts |
| 5 | Sample counts, seen against unseen X-ray FID, per-combination results and any anatomy check are missing (p. 7, p. 8) | X-ray measures A and B check anatomy and each finding per image | partly closed: our plan names counts and per-image reads; seen against unseen X-ray FID stays limited by rare images | X-ray measures A and B, reported per combination with counts |
| 6 | DCFM is not best on seen combinations, and FID drops on the non-IID split unexplained (p. 7, p. 8) | none | untouched | none from our questions; a seen-combination row in Paper 1's tables would at least report the same thing |
| 7 | Its own figures show a malformed digit and, by eye, ribs below the diaphragm (p. 4, p. 6, p. 8) | X-ray measure A would catch anatomy errors | partly closed | measure A's anatomy segmenter on DCFM-style X-rays, with failures shown beside ours |
| 8 | Cost grows with the number of attributes, approximated by pairs at about four passes a step (p. 9) | Paper 1's checker adds one push per step whatever the number of experts (inferred from the problem statement's formula) | partly closed, untested | time per picture against the number of experts, two to four, for both methods |

**Do our questions make a difference against DCFM? (inferred)** Yes on three gaps: the cost of forcing
independence (1), finished experts (3) and coverage below DCFM's minimum (4) are each answered by a
question this project already asks. Gap 6 is untouched. G4 turns each row into an experiment with its
picture, its number against DCFM's printed one, and its cost.

## What this cannot tell you

- Whether any counterpart's failure rate on look-alike pairs is above or below CO3's on SDXL: only CO3 has
  been counted.
- The GPU cost of DCFM-sized runs: DCFM prints passes, not hours.
- Whether DCFM's code and splits are public: G4 checks this.
- Problems outside the papers this project and its loops have read; the ranking covers what the reading
  tables, the failures catalogue and these searches reached.

## How this was read

Sources for the ranked problems: [G1](01-is-paper-2-still-open.md), [G2](02-reviewer-questions-paper-1.md),
[the manager's borrowed-parts search](../manager/who-counts-borrowed-parts-and-extra-subjects.md),
[Loop 4's DCFM note](../targets/dcfm-compositional-flow-matching-from-separate-silos.md), [Loop 4's failures
catalogue](../targets/failures.md) (359 figures from 25 papers), Loop 12's
[mechanism table](../leakage/01-reading/mechanism-table.md) and its Bounded Attention note, and the PDFs read
in full for G1 and G2 (T2I-CompBench++, Projective Composition, DCFM, DepthArb). Failure examples marked
summary were read from the catalogue, which gives the figure and page; they were not reopened here.
