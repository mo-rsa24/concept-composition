# Fairness audit: is each benchmark a like-for-like comparison?

This page judges every benchmark the two papers have committed to beating, and the CLEVR plan that Session 5 is about to train.
It reads the repository at commit `2d7c2599`, and every finding rests on a rival's page, released code or a run's output.

## Verdict counts

| What was audited | Rows | FAIR | FIXABLE | NOT COMPARABLE | CANNOT TELL |
|---|---:|---:|---:|---:|---:|
| [Scoreboard targets](#the-10-targets) | 10 | 0 | 7 | 3 | 0 |
| [Side-by-side rows](#the-30-side-by-side-rows) | 30 | 0 | 17 | 13 | 0 |
| [Reproduction rows](#the-28-reproduction-rows) | 28 | 0 | 15 | 13 | 0 |
| [Judging rules](#the-16-judging-rules) | 16 | 6 | 10 | 0 | 0 |
| [Other route commitments](#other-commitments-in-the-route) | 7 | 1 | 4 | 1 | 1 |
| [CLEVR scaling plan](#the-clevr-scaling-plan) | 23 | 4 | 17 | 2 | 0 |
| **All** | **114** | **11** | **70** | **32** | **1** |

No target, side-by-side row or reproduction row is fair as it stands. The scoreboard now holds 30 side-by-side rows, up from 28, because
Paper 2 gained two CLEVR rows. **Session 5 should train nothing for CLEVR until the first five fixes in the [CLEVR section](#what-to-change-before-session-5-trains) are written into the plan.**

## The five most serious problems

**1. None of Paper 1's four targets tests experts trained apart, and three cannot host its method.**

Shapes3D, Colored MNIST and CelebA ask for several attributes on one object or one face, from one network. Paper 1 gives each model
its own region of the picture, and a smiling man cannot be split into a "smiling" region and a "male" region. [CoInD's task](#the-10-targets)
is an AND over all six Shapes3D attributes ([p21][coind21]) with one U-Net ([p22][coind22]). Logical Guidance trains one model on `cmnist_uniform`
([its config][logdiff-cfg]), so all 100 digit and colour combinations are in its training data. A win on these rows would say nothing about
"experts that never met".

**2. "6 objects at 128 pixels" would not beat the printed CLEVR numbers.**

Bradley trains each model on 134,217,728 images ([Projective Composition p15][proj15]). The plan's full length is 8 million, one sixteenth of that
and half of Bradley's earliest checkpoint of 16.8 million ([Local Mechanisms p26][loc26]), and Bradley shows the object count growing with training
([p3][loc3]). The bar is also too low on the plan's own second metric: the rival rule already places all 6 objects in 97 of 100 hand-counted images
([p17][proj17]). Its first clear failure is at 7 ([Figure 3, p3][proj3]).

**3. The plan has no paste control, and fixed-position experts are close to a paste by construction.**

Bradley says that with an empty background the sampler "acts as if each model individually pastes an object" ([p2][proj2], also [p6][proj6]). With
one expert per fixed position, sampling each expert alone and pasting the objects onto a background may reach a high count with no composition at all.
[The target-change page](proposed-target-changes.md) says the plan holds this control (lines 54 and 55), and [the plan's table of methods](clevr-scaling-plan.md) has no such row.

**4. Paper 1's stated setting is contradicted by how its checker was trained.**

The [problem statement](../problem-statements.md) (lines 23 and 24) allows a helper trained on samples from the two models and excludes any model trained on the union
of the two datasets. The digit checkers trained on [MNIST training digits of every class, including pasted two-digit canvases][digit-review] (lines 97 to 103 and 227 to 231).
The pair checker also trained on [its own pushed canvases of the five cross-range tuning pairs][digit-checkers]. The 1,200 of 1,280 result depends on these checkers.

**5. Paper 2's checker trained on the Attend-and-Excite benchmark that the route still lists as an evaluation.**

After the subject guard, 216 of the 231 allowed Attend-and-Excite prompts have training tiles in [the checker's list][training-list], and 1,202 of its 2,805 tiles are renders
of that benchmark ([the collection script][pool-script]). Those tiles include 234 ToMe and 149 CO3 pictures labelled as clean, so the checker learned from rival outputs on the same prompts.
The [route](../../../../plans/route-two-papers.md) (line 81) still promises "a held-out set of Attend-and-Excite prompts", and only 15 allowed prompts remain unseen.

**The next problems, in order**

6. GenEval's two-object score can pass one fused object. [Its evaluator][geneval-code] checks each class separately (lines 176 to 179, 231 to 245), and [our own probe][scorer-probe] found `cup` and `wine glass` boxes on almost the same pixels of one hybrid.
7. Paper 2's checker has two subject outputs, while 383 to 513 of the 988 allowed Complex prompts name three or more objects, depending on the parser. On Texture, 60 of 300 do.
8. The route's Tune rung has no held-out prompts and gives rivals no equal tuning, which would favour us.
9. The promised single T2I-CompBench setup does not exist. ToMe runs seed 0 with its own negative prompt, CoMat is designed for seeds 42 to 51, and RTD and CO3 use different guidance.
10. The CMNIST bar of 83.6% sits below the 98.15% that CoInD prints for the same AND rule on uniform support ([p8][coind8]), and the cause of that gap is unknown.

## Words this uses

**Most objects that succeed (Kmax).** The largest K where every count from 1 to K succeeds. A count succeeds when K objects appear in at least 50% of images and at least K-2 appear in at least 90% ([Local Mechanisms p23][loc23]).

**Conformity score.** The percentage of generated images a classifier reads as having every requested attribute, 0 to 100.

**ImageReward.** A learned score of how much a person would prefer an image for its prompt. Higher is better, and it can go below 0.

**BLIP-VQA.** The product of a question-answering model's yes-probabilities for each named phrase, 0 to 1.

**Checker.** Our trained classifier, whose gradient pushes the sampler during the early steps.

**GPU-hours.** Each cost below is marked *measured* when it comes from a run's log and *estimated* otherwise. The estimates use these rates: 7 to 9.4 GPU-hours per CLEVR model at 8 million images (two models shared one card for about 13h 50m, and a lone probe ran at 4.05 seconds per thousand images), 14 to 15 seconds per plain SDXL picture (measured), 82 seconds per ToMe picture (measured median), and 40 seconds per pushed picture (the checker design's budget).

## What each benchmark actually is

| Benchmark | What the rival trained | How it composes | Data and protocol | Scorer | Can our method run? |
|---|---|---|---|---|---|
| CLEVR, one object | Local patch model: one network. Projective: one location network plus a separate empty-background network ([p39][proj39]) | AND. Local conditions one pass on all positions, Projective adds scores at inference | 128 px, 500,000 or 1,550,000 single-object images, 134M images seen, no guidance | ResNet-18 count, 99.74%, 1,024 images, counts total objects ([p26][loc26]) | Paper 1 needs experts the rivals lack. Paper 2 needs a new checker |
| Shapes3D | One U-Net with a conditional-independence loss ([p22][coind22]) | AND over 6 attributes, learned loss plus inference | Orthogonal partial support, unseen combinations, 100 DDIM steps | ResNet-18 attribute heads | No: scale, shape and orientation belong to one object |
| Colored MNIST | One U-Net on all 100 combinations | AND of digit and colour, with a noisy-image classifier ([p5][lg5]) | 28 px, 50 steps, 100 queries of 100 images | ResNet-18 | No: digit and colour are one object |
| CelebA | SD1.5 fine-tuned on 3 of 4 attribute groups ([p27][coind27]) | AND of two prompts | Base SD1.5 weights predate the fine-tune; image count unstated | ResNet-18 at 128 px | No: one face |
| T2I-CompBench Color, Shape, Texture | ToMe trains nothing. CoMat fine-tunes on about 20,000 prompts ([p7][comat7]) | Joint prompt with attention or token changes | 300 validation prompts each; seeds and image counts unstated | BLIP-VQA | On 261 of 288, 261 of 293 and 240 of 300 prompts, those with two subjects |
| T2I-CompBench Complex | RTD trains nothing ([p4][rtd4]) | Joint prompt plus one start-noise update | 1,000 prompts, train and validation lists together, 3 seeds, guidance 5.0 | ImageReward | On 475 to 578 of 988 prompts |
| RareBench Concat (reported) | RTD trains nothing | As above | 33 of 40 prompts allowed, 3 seeds | ImageReward | Yes: every prompt joins two phrases, and 5 ask for several of one item |
| GenEval two objects | FKC trains nothing ([p38][fkc38]) | Classifier-free guidance plus resampling | 99 two-object prompts of the 553, 4 images each | Mask2Former at confidence 0.3 | Yes |

## The 10 targets

| Row | Verdict | Evidence | Fix, or the claim still allowed |
|---|---|---|---|
| Paper 1, CLEVR, 5 becomes 6 | FIXABLE | One-network rivals ([p3][loc3], [p39][proj39]); training gap ([p15][proj15]); 97 of 100 at 6 ([p17][proj17]) | Judge against the control and local patch model trained to the same length, and require 7 on the hand count. 5 models at 8M images: 69 GPU-hours, estimated from the measured rate |
| Paper 1, Shapes3D, 91.10% | NOT COMPARABLE | One U-Net on the union, AND over six attributes ([p20][coind20], [p21][coind21], [p22][coind22]) | Allowed: CoInD as context. The nearest fair rival is LACE, one model per attribute, 62.07% |
| Paper 1, CMNIST AND, 83.6% | NOT COMPARABLE | [Uniform support config][logdiff-cfg]; [Table 4, p7][lg7]; one model for every method ([p26][lg26]) | Allowed: context. A run about unseen combinations belongs on diagonal partial support, where CoInD prints 52.38% ([p8][coind8]) |
| Paper 1, CelebA, 49.15% | NOT COMPARABLE | One fine-tuned SD1.5, one face ([p27][coind27]) | Allowed: the CoInD reproduction as context |
| Paper 2, Color, 0.7827 | FIXABLE | SDXL prints 0.5879 in [CoMat][comat7] and 0.6369 in [ToMe][tome8]. 32 of the 288 allowed prompts are in [CoMat's training list][comat-prompts] | One setup for plain SDXL, CoMat, ToMe and ours; report the 256 prompts CoMat never trained on. About 17 GPU-hours for all three attribute rows, estimated |
| Paper 2, Shape, 0.6051 | FIXABLE | [ToMe Table 1, p8][tome8]; 4 allowed prompts in CoMat's list; 3 use a pair the checker trained on | Same setup; drop those prompts from the headline |
| Paper 2, Texture, 0.6894 | FIXABLE | [ToMe Table 1, p8][tome8]; 60 of 300 prompts have 3 or 4 subjects | Same setup; report the 240 two-subject prompts as a named subset |
| Paper 2, Complex, 0.4661 | FIXABLE | [RTD Table 2, p6][rtd6]; our rerun reads 0.2423 and sits 0.0214 below plain SDXL (`rtd-scores.jsonl`) | Read against the same-setup rerun, with the printed number beside it. Define the checker for 3 or more subjects, or report the two-subject subset. 33 GPU-hours, estimated |
| Paper 2, CLEVR, 5 becomes 6 | FIXABLE | The target row copies Paper 1's. None of the [three checker candidates][checker-design] can run as written on a pixel-space location model | Keep control against control plus checker, which differs in one thing. Call it a controlled count test with a new checker, and count it once with Paper 1's row |
| Paper 2, GenEval, 0.78 | FIXABLE | [FKC Table A4, p39][fkc39]; [evaluator][geneval-code]; [fused-object probe][scorer-probe] | Keep the rule to beat plain SDXL's 0.80. Add a check that the two class boxes are separate objects. About 6 GPU-hours, estimated |

## The 30 side-by-side rows

Every call needs both methods rendered on the same prompt and seed. No call has been made yet, and [the table of calls](side-by-side.csv) is empty.

| Row | Verdict | Evidence | Fix, or the claim still allowed |
|---|---|---|---|
| Paper 1, CLEVR, Bradley-Projective | FIXABLE | Rule is fully described ([p39][proj39]) | Render the rule on our models at the same positions and seeds |
| Paper 1, CLEVR, Bradley-Local | FIXABLE | No code, architecture described ([p29][loc29]) | Rebuild it and pass the authors' check that k = 8 removes the gain. One model: 7 to 9.4 GPU-hours at 8M, estimated |
| Paper 1, Shapes3D, CoInD | NOT COMPARABLE | Region method undefined ([p21][coind21]) | Allowed: a picture of the rival's failure only |
| Paper 1, Shapes3D, Liu | NOT COMPARABLE | Same | Same |
| Paper 1, CMNIST, Liu | NOT COMPARABLE | Same task on one object ([p26][lg26]) | Same |
| Paper 1, CMNIST, Logical Guidance | NOT COMPARABLE | Same | Same |
| Paper 1, CMNIST, SuperDiff | NOT COMPARABLE | Its failure line is about Stable Diffusion pairs, and the CMNIST run used one model ([p26][lg26]) | Remove the failure line from this row |
| Paper 1, CelebA, CoInD | NOT COMPARABLE | One face ([p27][coind27]) | Allowed: context, kept on the cluster under the CelebA licence |
| Paper 1, CelebA, Liu | NOT COMPARABLE | Same | Same |
| Paper 2, Color, Attend-and-Excite | NOT COMPARABLE | Printed run is on SD2 ([T2I-CompBench++ Table VIII, p9][t2ipp9]) | Allowed: printed context |
| Paper 2, Color, CoMat | FIXABLE | Failure line comes from prompts of an unstated set | Render on the shared setup and call only those pairs |
| Paper 2, Color, Liu | NOT COMPARABLE | SD2 ([p9][t2ipp9]) | Allowed: printed context |
| Paper 2, Color, ToMe | FIXABLE | ToMe has a printed Color number, 0.7656 ([p8][tome8]) | Use it; render ours at ToMe's seed, guidance and negative prompt |
| Paper 2, Shape, Attend-and-Excite | NOT COMPARABLE | Same value as the SD2-era table ([CoMat p7][comat7]) | Allowed: printed context |
| Paper 2, Shape, CoMat | FIXABLE | As Color | As Color |
| Paper 2, Shape, Liu | NOT COMPARABLE | As Attend-and-Excite | Allowed: printed context |
| Paper 2, Shape, ToMe | FIXABLE | 3 prompts use a trained pair | Leave those out of the calls |
| Paper 2, Texture, Attend-and-Excite | NOT COMPARABLE | As Shape | Allowed: printed context |
| Paper 2, Texture, CoMat | FIXABLE | As Color | As Color |
| Paper 2, Texture, Liu | NOT COMPARABLE | As Shape | Allowed: printed context |
| Paper 2, Texture, ToMe | FIXABLE | 60 of 300 prompts have 3 or 4 subjects, and the ladder of calls stops at two things | Call the 240 two-subject prompts |
| Paper 2, Complex, CO3 | FIXABLE | Same seeds and starting noise ([CO3 design](../reproductions/CO3/design.md)) | Add "all named objects present" to the ladder |
| Paper 2, Complex, RTD | FIXABLE | Rerun renders exist | Same |
| Paper 2, Complex, ToMe | FIXABLE | Seed 0 only ([render script][tome-render]) | Compare on seed 0 only |
| Paper 2, Concat, CO3 | FIXABLE | Reported row, so calls cannot support a win | Label as reported |
| Paper 2, Concat, RTD | FIXABLE | Same | Same |
| Paper 2, Concat, ToMe | FIXABLE | Same, on seed 0 | Same, on seed 0 |
| Paper 2, GenEval, FKC | FIXABLE | Failure line comes from coloured-object prompts in [Figure A5, p39][fkc39], another GenEval task | Replace it with two-object prompts from our run |
| Paper 2, CLEVR, Bradley-Projective | FIXABLE | Duplicate of the Paper 1 row | Share one set of control renders |
| Paper 2, CLEVR, Bradley-Local | FIXABLE | Duplicate of the Paper 1 row | Share one rebuilt model |

## The 28 reproduction rows

| Row | Verdict | Evidence | Fix, or the claim still allowed |
|---|---|---|---|
| Bradley-Projective, CLEVR, 4 | FIXABLE | [Table 2, p39][proj39] against [hand counts, p17][proj17] | Rerun as a same-model control and report both measures. Never label it reproduced below Bradley's training length |
| Bradley-Local, CLEVR, 5 | FIXABLE | [p3][loc3], [p29][loc29] | Rebuild from p29. Label it a reimplementation |
| CoInD, Shapes3D, 91.10 | NOT COMPARABLE | Target regime differs. Board question 34 on its evaluation script is still open | Allowed: a reproduction of CoInD |
| Liu, Shapes3D, 51.56 | NOT COMPARABLE | Printed as Composed GLIDE ([p20][coind20]) | Same |
| Liu, CMNIST | NOT COMPARABLE | The row says no number exists, while [Table 11, p26][lg26] prints 80.39 for Liu's rule and 26.61 for plain averaging | Correct the note. Allowed: context |
| Logical Guidance, CMNIST, 83.6 | NOT COMPARABLE | [p7][lg7]; the config ships with `use_neg_guidance: false` | Set it true for this row. Allowed: context |
| SuperDiff, CMNIST | NOT COMPARABLE | Run on one model outside its two-model design ([p26][lg26]) | Allowed: context |
| CoInD, CelebA, 49.15 | NOT COMPARABLE | [p27][coind27]; 2,500 images | Allowed: a reproduction, with per-group accuracy printed |
| Liu, CelebA, 14.19 | NOT COMPARABLE | Same | Same |
| Attend-and-Excite, Color, 0.6400 | NOT COMPARABLE | SD2 ([p9][t2ipp9]) | Allowed: printed context |
| CoMat, Color, 0.7827 | FIXABLE | Evaluation size is unstated, and it trained at 512 px ([design](../reproductions/CoMat/design.md)) | Score 1024 as approved, and add 512 on a 30-prompt sample at 10 images each. About 0.4 GPU-hours, estimated from the 11 hours the owner was quoted for all 8,810 pictures |
| Liu, Color, 0.4063 | NOT COMPARABLE | SD2 ([p9][t2ipp9]) | Allowed: printed context |
| ToMe, Color | FIXABLE | The row says no number exists. [ToMe Table 1][tome8] prints 0.7656 by BLIP-VQA | Fill in `ToMe-T1-052` |
| Attend-and-Excite, Shape | NOT COMPARABLE | Same printed value as the SD2-era table | Allowed: printed context |
| CoMat, Shape, 0.5329 | FIXABLE | As Color | As Color |
| Liu, Shape | NOT COMPARABLE | As Attend-and-Excite | Allowed: printed context |
| ToMe, Shape, 0.6051 | FIXABLE | One seed, a written merge rule and a repaired parser ([design](../reproductions/ToMe/design.md)) | Print scores with and without the 2 repaired prompts |
| Attend-and-Excite, Texture | NOT COMPARABLE | As Shape | Allowed: printed context |
| CoMat, Texture, 0.6468 | FIXABLE | As Color | As Color |
| Liu, Texture | NOT COMPARABLE | As Shape | Allowed: printed context |
| ToMe, Texture, 0.6894 | FIXABLE | The rerun is faithful, while our method covers 240 of 300 prompts | Report ToMe on all 300 and on the 240 |
| CO3, Complex, 0.4406 | FIXABLE | The row's protocol text says guidance 5.0, and its note says CFG++ at 0.8 | Make the protocol cell state what ran |
| RTD, Complex, 0.4661 | FIXABLE | Rerun 0.2423, plain SDXL 0.2637, recomputed from the score files | Keep "not reproduced", and read wins against the rerun |
| ToMe, Complex, 0.2891 | FIXABLE | One seed, guidance 7.5 and a negative prompt, against the row's guidance 5.0 | Correct the protocol cell, and compare on seed 0 |
| CO3, Concat, 0.1610 | FIXABLE | As CO3 Complex | As CO3 Complex |
| RTD, Concat, 0.4464 | FIXABLE | "Reproduced" with a band of plus or minus 0.443. RTD's printed gain over plain SDXL is +0.3446, and our rerun's is +0.1131 | Relabel as "not refuted, weak test" |
| ToMe, Concat, -0.2041 | FIXABLE | One seed; 4 prompts render as plain SDXL | Print both prompt counts |
| FKC, GenEval, 0.78 | FIXABLE | No SDXL code, particle count unstated ([p38][fkc38], [p39][fkc39]) | Label it a port. Under 2 GPU-hours, approved |

## The 16 judging rules

| Rule in [the protocol](comparison-protocol.md) | Verdict | What the check found | Fix |
|---|---|---|---|
| A target is one step past the best rival | FIXABLE | A step of 0.01 is well inside the noise: the margin on Complex is about 0.03 | Print the margin beside each target |
| A number counts when the protocol matches | FIXABLE | The CLEVR plan's training length and several protocol cells break it | Apply it row by row, as listed above |
| Rerun, and use the harder of printed and rerun | FAIR | This cannot favour us | None |
| A win inside printed uncertainty is noise | FIXABLE | No target has a printed interval, so the rule never fires | Rely on the rerun margin |
| T2I-CompBench needs one rerun setup | FIXABLE | The runs use different seeds, guidance and negative prompts | Fix one setup in writing |
| Guard prompts are shared by rival and ours | FIXABLE | The printed bar covers other prompts. On Concat, dropping 7 prompts goes with plain SDXL reading 0.2017 above its print | Treat the printed number as context on these rows |
| GenEval must also beat plain SDXL | FAIR | This closes the gap between FKC's 0.78 and plain SDXL's 0.80 ([p39][fkc39]) | None |
| A win clears 1.96 standard errors | FIXABLE | There is no correction for 10 targets, and a `blocked` rival can lose to a printed number from another setup | Apply Holm correction, and let a blocked row claim no win |
| Classifier within one point of the print | FIXABLE | The 90% per-group bar was replaced after the classifier scored 88.75% ([review](../reproductions/CoInD/review.md)) | Record that the bar moved after the result |
| CLEVR by most objects, on our models | FIXABLE | "Clean pictures" has no definition or threshold | Define it in code before the first run |
| CIFAR-10 is context | FAIR | SuperDiff's task is OR ([p7][sd7]) | None |
| Both papers compete on CLEVR | FIXABLE | One benchmark fills two target rows, and Paper 2's checker for it does not exist | Count it once, and name the new checker |
| RareBench Concat is a reported row | FAIR | This matches its 33-prompt power | None |
| A disputed number cannot set a target | FAIR | CoMat's own page prints 0.7827 ([p7][comat7]) | None |
| The picture calls | FIXABLE | The ladder ends at two things, while many prompts name more | Add "all named objects present" and attribute checks |
| What targets do not establish | FAIR | The limits are stated correctly | None |

## Other commitments in the route

| Commitment | Verdict | Evidence | Fix, or the claim still allowed |
|---|---|---|---|
| CIFAR-10 disjoint halves | NOT COMPARABLE | OR task ([p7][sd7]), while the route still lists it as a Paper 1 run | Allowed: context. Correct the route |
| Chest X-ray | CANNOT TELL | No rival number ([limits](limits.md), line 221) | Missing facts: the dataset, task, metric and rival. Drop it or specify it |
| Held-out Attend-and-Excite prompts | FIXABLE | 216 of 231 allowed prompts are in training | Hold out 25% of pairs by name hash and retrain. About 6 GPU-hours, estimated |
| Sample-and-paste control on digits | FIXABLE | [The problem statement](../problem-statements.md) says it has not been run | Run it on the 1,280 canvases. Under 1 GPU-hour, estimated |
| Tune rung | FIXABLE | No held-out split, and rivals stay frozen (route line 76) | Tune on separate prompts and seeds, and give rivals the same number of trials or use their published settings throughout |
| Full rung, three seed sets for every rival | FIXABLE | ToMe has one seed | Two more seeds cost about 86 GPU-hours, estimated, or state one-seed comparisons |
| CO3 against CO3 plus checker | FAIR | Exactly one thing differs, provided both use the released script | None. It has no run yet |

## The CLEVR scaling plan

### What to change before Session 5 trains

1. **Add a sample-and-paste control and an experts-without-checker run.** The design then has four clean cells: one network or experts, each with and without the checker.
2. **Compare only at equal training length.** Keep printed numbers as context unless a run reaches Bradley's 134 million images.
3. **Use Bradley's 12 printed positions, in printed order** ([p39][proj39]), and name the location conditioner and the 32-step Heun sampler ([p15][proj15]).
4. **Train the scorer on real rendered images with 1 to 12 objects, and keep it separate from every checker.** Define the picture-quality bar in code.
5. **Replace Du's cubes as the first rung, and correct the cost table.** Du's data holds 1 to 5 cubes per image, and each model costs 7 to 9.4 GPU-hours, so the plan's 7.5 is the low end.

### Row by row

| Plan item | Verdict | Evidence | Fix, or the claim still allowed |
|---|---|---|---|
| Control: one location network, one background network | FIXABLE | Matches the printed setup ([p39][proj39]), except for length | Require it to reach about 4 before anything is compared. If it does not, report the shortfall |
| One network plus checker (Paper 2) | FIXABLE | Clean single change, but [the candidate checkers][checker-design] read SDXL features or text attention | Specify a pixel-space checker for CLEVR |
| Experts trained apart plus checker (Paper 1) | FIXABLE | Differs from control in the training, the checker, the model count and the data each model sees | Add experts without a checker, and report total training images and weights |
| Paste control | FIXABLE | Absent from the plan; [claimed present](proposed-target-changes.md); expected to do well ([p2][proj2], [p6][proj6]) | Sample each expert once per seed and paste. 10,240 samples, render time unmeasured |
| Local patch model | FIXABLE | It sets the bar of 5 and is not among the plan's methods ([p29][loc29]) | Rebuild it. 7 to 9.4 GPU-hours at 8M images, estimated |
| Rung 1: Du's cubes | NOT COMPARABLE | 200,000 images with "between 1 to 5" cubes ([Du p18][du18]) | Allowed: a pipeline check. Filtering to one-cube images needs labels the plan has not confirmed |
| Rung 2: single-object CLEVR at 64 px | FIXABLE | No Blender on `mscluster84`'s path or in `/datasets/mmolefe` to depth 4; dataset size and per-position data are unstated | Install and time the renderer, and state the image count per expert |
| Rung 3: 128 px beats a printed number | NOT COMPARABLE | 8M against 134M images ([p15][proj15], [p26][loc26]). Matching costs 512 A100-hours per model, 6,144 for 12 | Allowed: "better than the same rule on models trained equally" |
| Stage 2: two experts, several objects | FIXABLE | Bans the halves that [Paper 1's method](../problem-statements.md) uses (line 28). Its models see 2 or 3 real objects, and "the blue tables" in rule 5 is undefined | Restate Paper 1's method, say which table Stage 2 belongs to, and add a colour-aware scorer. 83 GPU-hours, estimated |
| Stage 3: inclusive labels | FIXABLE | 9 is printed for the local patch models ([p3][loc3]). Target 10 sits at the classifier's ceiling, and Bradley tests to 12 ([p26][loc26]) | Extend the scorer to 12 |
| The bar of 6 | FIXABLE | 97 of 100 at 6 and 98 of 100 at 5 ([p17][proj17]); partial at 5 and 6, failing at 7 ([p3][proj3]) | Require 7 on the hand count, with 6 reported on the classifier |
| Score: most objects that succeed | FIXABLE | Bradley's classifier counts total objects ([p26][loc26]) | Score objects at the requested positions too |
| Score: all N in place, 100 by eye | FIXABLE | The rival's hand count stops at 6 ([p17][proj17]) | Use neutral file names and full-size images |
| Score: picture quality | FIXABLE | No measure or threshold | Fix both in code before the first run |
| Rule 1: each expert passes alone | FIXABLE | No pass mark, no limit on retraining, and no matching filter on the control | Set the mark, allow one training per expert, and report failures |
| Rule 2: position order written first | FIXABLE | Does not commit to Bradley's list | Use the printed list |
| Rule 3: stop after two failures or at 10 | FAIR | Kmax needs every smaller count to pass | None |
| Rule 4: test every smaller group | FIXABLE | At 10 experts: 1,012 groups, 1,036,288 images, 339 million model calls. At 6: 56 groups, 57,344 images | Test the N leave-one-out groups, 10,240 images at N = 10 |
| Rule 5: checker sees single objects and composites | FIXABLE | Composites give it multi-object pictures that the rival's rule never receives | Train on random positions, hold out the evaluated position sets, and report a no-composite checker |
| Cost table | FIXABLE | Two models shared one card for 13h 49m and about 13h 52m over 8.4M images (`log.txt`, measured), and a lone probe ran at 4.05 seconds per thousand images | 12 models cost about 83 to 113 GPU-hours where the plan says 90, and 21 to 28 at quarter length where it says 22.5 |
| "No rival prints experts trained apart" | FAIR | Holds across the 25 rival PDFs. [Wiedemer et al.][wiedemer] trains one decoder on two-sprite images | State the scope. CoInD's LACE trains one model per attribute ([p22][coind22]) and belongs in related work |
| "The two Bradley papers disagree on 1 to 3" | FAIR | 9 in [Projective Table 2][proj39], 7 in [Local Table 1][loc3] | None |
| "Bradley's colour model reaches 4" | FAIR | [Table 4, p30][loc30]. It uses a 25% rule on 64 hand-counted images ([p29][loc29]), and 33 of 64 also clears 50% | Add that note |

## Claims the sources contradict

| Claim | Where we say it | What the source says |
|---|---|---|
| The helper excludes the union of the two datasets | [Problem statement](../problem-statements.md), lines 23 and 24 | Checkers trained on digits of every class ([plan 32 review][digit-review], lines 97 to 103, 227 to 231) |
| Paper 1 gives each model its own region | [Problem statement](../problem-statements.md), line 28 | The [CLEVR plan](clevr-scaling-plan.md), lines 28 and 29, forbids it in Stage 2 |
| The scaling plan holds the paste-control table | [Target changes](proposed-target-changes.md), lines 54 and 55 | The [plan's methods table](clevr-scaling-plan.md), lines 11 to 15, has no such control |
| A model costs about 7.5 GPU-hours at 8M images | [CLEVR plan](clevr-scaling-plan.md), line 69 | Low end of 7 to 9.4: two models shared a card for about 13h 50m in `/datasets/mmolefe/projective-composition/checkpoints/*/log.txt` |
| 128 px is "Bradley's size", so a result there can beat a printed number | [CLEVR plan](clevr-scaling-plan.md), lines 21 and 22 | Bradley also trains 16 times longer ([p15][proj15]) |
| Du's cubes can start a ladder where no model sees two objects | [CLEVR plan](clevr-scaling-plan.md), lines 19, 20 and 40 | 1 to 5 cubes per image ([Du p18][du18]) |
| One scorer serves every method | [CLEVR plan figure](one-clevr-experiment-for-both-papers.png) | Colour cannot be judged by a count classifier ([Local p30][loc30]) |
| Both CLEVR rivals are "a single network" | [Protocol](comparison-protocol.md), lines 56 and 57 | Projective uses "a separate model trained only on empty images" ([p39][proj39]) |
| Paper 2's CLEVR target "tests the same method" | [Target changes](proposed-target-changes.md), line 40 | The candidates need SDXL features or a joint prompt ([checker design][checker-design]) |
| GenEval and CLEVR "count separate objects" | [Problem statement](../problem-statements.md), line 129 | GenEval can pass one fused object ([evaluator][geneval-code], [probe][scorer-probe]) |
| CO3 is the base that "misses a quarter" | [Route](../../../../plans/route-two-papers.md), line 21 | The joint prompt alone scores 0.7626 against CO3's 0.7500 on the same 396 images ([table][bench-table]), and CO3's settings there are [unconfirmed](../void-list.md) |
| Evidence of 1,200 and 54 | [Route](../../../../plans/route-two-papers.md), line 24, with no flag | Both are on the [void list](../void-list.md) |
| Liu has no CMNIST number under the protocol | [Rows with no number](../reproductions/rows-with-no-matching-number.md) | 80.39 for Liu's rule ([Table 11, p26][lg26]) |
| ToMe has no Color number under the protocol | [Rows with no number](../reproductions/rows-with-no-matching-number.md) | 0.7656 by BLIP-VQA ([p8][tome8]) |
| RTD "reproduced" on Concat | [Reproduction table](../reproductions/table.csv) | The level matches while the gain over plain SDXL falls from +0.3446 to +0.1131 |

## What could not be checked

**Du's released file.** The 6.2 GB `dataset_train_larger.npz` was not downloaded, so its contents rest on the paper's description.

**Runs still in progress.** CO3's and ToMe's full runs were rendering, so their scores are unknown. No run of our method exists on any target.

**CoMat's evaluation size.** Neither the paper nor the code settles whether Table 1 used 512 or 1024 pixels.

**Why RTD's Complex score misses.** The card, the scorer version and the printed number all remain possible causes.

**Render and Blender costs for CLEVR.** Neither has been measured. The Blender check covers one node's path and a depth-limited folder search.

**Literature beyond the 25 rivals.** "No rival prints experts trained apart" was checked in those PDFs and in one cited paper, with no new search.

[proj2]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/mechanisms-of-projective-composition-of-diffusion-models.pdf#page=2
[proj3]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/mechanisms-of-projective-composition-of-diffusion-models.pdf#page=3
[proj6]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/mechanisms-of-projective-composition-of-diffusion-models.pdf#page=6
[proj15]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/mechanisms-of-projective-composition-of-diffusion-models.pdf#page=15
[proj17]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/mechanisms-of-projective-composition-of-diffusion-models.pdf#page=17
[proj39]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/mechanisms-of-projective-composition-of-diffusion-models.pdf#page=39
[loc3]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/local-mechanisms-of-compositional-generalization-in-conditional-diffusion.pdf#page=3
[loc23]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/local-mechanisms-of-compositional-generalization-in-conditional-diffusion.pdf#page=23
[loc26]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/local-mechanisms-of-compositional-generalization-in-conditional-diffusion.pdf#page=26
[loc29]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/local-mechanisms-of-compositional-generalization-in-conditional-diffusion.pdf#page=29
[loc30]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/compositional-generalization/local-mechanisms-of-compositional-generalization-in-conditional-diffusion.pdf#page=30
[coind8]: pdfs/2503.01145.pdf#page=8
[coind20]: pdfs/2503.01145.pdf#page=20
[coind21]: pdfs/2503.01145.pdf#page=21
[coind22]: pdfs/2503.01145.pdf#page=22
[coind27]: pdfs/2503.01145.pdf#page=27
[lg5]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/logical-guidance-for-exact-composition-of-diffusion-models.pdf#page=5
[lg7]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/logical-guidance-for-exact-composition-of-diffusion-models.pdf#page=7
[lg26]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/logical-guidance-for-exact-composition-of-diffusion-models.pdf#page=26
[comat7]: pdfs/2404.03653.pdf#page=7
[tome8]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/region-and-binding-failures/token-merging-for-training-free-semantic-binding.pdf#page=8
[rtd4]: pdfs/2608.03135.pdf#page=4
[rtd6]: pdfs/2608.03135.pdf#page=6
[fkc38]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/feynman-kac-correctors-in-diffusion.pdf#page=38
[fkc39]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/feynman-kac-correctors-in-diffusion.pdf#page=39
[du18]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/compositional-generation-with-energy-based-diffusion-models-and-mcmc.pdf#page=18
[sd7]: ../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/sampling-composed-models/superposition-of-diffusion-models-using-the-ito-density-estimator.pdf#page=7
[t2ipp9]: ../../../../../replications/t2i-compbench/paper/T2I-CompBench++.pdf#page=9
[wiedemer]: https://arxiv.org/abs/2307.05596
[logdiff-cfg]: ../../../../../replications/logdiff/configs/cmnist_inference.yaml
[comat-prompts]: ../../../../../replications/comat/merged_data/abc5k_hrs10k_t2icompall_20k.txt
[tome-render]: ../../../../../replications/published-methods-on-our-pairs/scripts/reproductions/tome_generate.py
[geneval-code]: ../../../../../replications/geneval/evaluation/evaluate_images.py
[scorer-probe]: ../../../../1-lora-correction/report/measuring-composition/what-each-scorer-says-about-two-objects.md
[bench-table]: ../../../../1-lora-correction/report/against-other-methods/how-does-the-adapter-score-on-the-standard-benchmarks.md
[digit-review]: ../../../../2-classifier-guidance/plans/review/32-two-digit-experts-that-never-met.md
[digit-checkers]: ../../../../2-classifier-guidance/artifacts/results/can-two-digits-compose-on-mnist/03-composing-the-two-models-while-sampling/03-adding-the-checker-pushes/README.md
[training-list]: ../../../../2-classifier-guidance/scripts/rung2/training_set_multipair.json
[pool-script]: ../../../../2-classifier-guidance/scripts/rung2/collect_benchmark_pool.py
[checker-design]: ../reproductions/checker/design.md
