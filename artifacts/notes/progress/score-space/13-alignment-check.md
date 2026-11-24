# Alignment check: each loop's intent against its own account

On 2026-11-24 the owner asked each research loop to read a paragraph stating what it was meant to do, and to answer five questions: what it has done, whether it still works towards that intent, whether any hypothesis was falsified, whether a failure was the idea's fault or the setup's, and what would need to change. The paragraphs were written by Loop 10 from each loop's brief. The answers are each loop's own, collected here unedited in substance and shortened only to the table.

## The two papers, as the owner intends them

**Paper 1, experts that never met.** Two diffusion models trained on separate data are composed into one picture holding both concepts as separate objects. Each model draws only its own region, and a small checker trained without joint images guides the merged sample. No image or prompt naming both concepts, and no model trained on their union, may be used.

**Paper 2, better than CO3 inside one model.** One pretrained SDXL has seen both concepts and the joint prompt is available. The aim is to fix what CO3 still gets wrong (counting, binding, spatial relations, identity) by steering CO3's sampling with a small checker trained after the model, by a gradient push or by setting CO3's own knobs from the checker's reading. It stays on one trajectory per seed, leaves SDXL's weights untouched, and uses best-of-N only as a baseline.

## Side by side

| Loop | The intent, in brief | Still working towards it? | Hypothesis falsified? | Right reason, or wrong setup? | What would need to change |
|---|---|---|---|---|---|
| 4. Rivals and targets | read every rival in full; set targets from what they report | **partly.** 7 targets set from printed numbers, but by the owner's choice a win is now judged against the best rerun on our own pictures, because printed scores did not reproduce | **three assumptions.** Printed rival scores reproduce (RTD Complex 0.2423 against 0.4661 printed); beating paste on CLEVR by object count is reachable (paste reaches 7 objects at 90.6% per-object accuracy); GenEval counts two separate objects (it checks each class alone) | **mostly setup.** On CLEVR, position-conditioned experts make pasting optimal by construction, so object count cannot separate Paper 1 from paste | Paper 1 needs a target where pasting fails: objects that overlap or shade each other |
| 5. Reproduce the rivals | rerun the strongest rival per benchmark on our protocol | **yes,** for the strongest rival per benchmark; 13 rows kept as printed context by the owner's choice. Loop closed | **not reproduced:** ToMe Shape 0.5502 against 0.6051, CO3 Complex 0.3326 against 0.4406, RTD Complex 0.2423 against 0.4661; **reproduced:** CoMat Color 0.7696 against 0.7827, ToMe Texture 0.6503 against 0.6894 | **the rivals', not ours.** Plain SDXL reproduced its printed scores on Complex, Texture and Color, so the misses look like the rivals' own | nothing; Paper 2's Complex rival is CO3 at 0.4628 on the 463 two-subject prompts |
| 6. Find the best method | search checker, labels and settings per paper on tuning data; beat paste and CO3 alone; freeze one recipe each | **partly.** The baseline tests ran; the search itself (M5, D3, M2, M3) has not started, because M7 and M9 came first and M7 waits on question 66 | **yes, on tuning data.** Paper 2's push beats CO3 (7 against 11 of 96); Paper 1's push beats paste (lead of 8, margin 13.54); plain regions from step 0 separate cat and dog (0 of 8) | **mixed.** Paper 2: the checker's presence outputs learned on cat and dog only, so its data may be wrong for the question. Paper 1: with clean experts, paste sits near its ceiling, so this setup cannot separate a method from paste. M9 failed at its region reader, a setup failure | Paper 1 needs objects that interact; Paper 2 needs a method aimed at identity; both need answers to questions 65 and 66 |
| 7. Abstract-checker papers | design checkers for properties that hold across concepts | **yes.** All three steps done: 25 papers read, the score view derived, the five-checker design written | **one assumption, on a computed toy:** the five checkers' logs can simply be added. With nested labels the summed gradient is 27.9% too large | **the right reason:** it is mathematics, and the design now trains the checkers as a chain. Two setup problems found: the digit model never saw two digits together; one pair is both tuning and unseen | put a question head for identity and binding on the board, since CO3 fails on identity |
| 8. Test abstract checkers | on digits with exact labels, test a property checker on concepts it never trained on, then SDXL | **partly.** The digit half matches: count transfers to unseen digits (0.791 against 0.782 seen). The SDXL half (V4) has not started; it waits on Loop 6's D3 | **yes.** A map reader learns which digits are present at SDXL-matched noise: 0.555, bar 0.75. Each checker controls only its own property: removing one moved another score about 7 points, limit 4.2 | **more setup than idea.** The same two-map input reads identity at 0.563 early and 0.702 in a later window; Loop 10's signature reads 0.655 | run V4 on SDXL now, judged against CO3 as well as the plain product |
| 9. Concepts into scores | a noise-aware text-and-image score as a cheaper source of the five checkers | **partly.** Answered its question (no) and closed without building a checker; the push (W3) never ran | **yes.** Sentence scores read counting, prompt match and left-right order at best 0.574 at steps 0 to 15, bar 0.75; its own digit model reads unseen pairs at 0.590 against 0.758 seen | **the idea.** On the same pictures and labels, a readout on SDXL's U-Net features reaches 0.87 by step 15; position and size, and binding, failed on their labels instead (eye checks 37 of 57 and 26 of 32, against 90%) | nothing for this loop; read the five checkers from SDXL's own features (Loops 6 and 8) |
| 10. Score path as a third space | a space shared by text, noisy pictures and the model's own predictions, in which adding scores means adding text | **partly.** It closed, and its findings feed Paper 2's lead direction (regions read from each subject's $d_t$) | **narrowly.** On digits the score path read digit presence at up to 0.700 against a 0.75 bar ([the X5 review](05-signal/review.md)), so X6 to X8 did not run | **mostly the setup.** Digits have no joint label, so the owner's additivity question could not be posed there; the arithmetic tests sat behind the readability bar; the probe was weak and the window stopped while the score was still rising ([the independent audit](12-independent-audit.md)) | test additivity where a joint prompt exists, in the block-wise form Bradley et al.'s Theorem 6.1 allows, not readability on digits |
| 12. Where CO3 leaks | find where CO3's subjects borrow parts, test one fix, choose Paper 2's method, audit Loops 8 to 10 | **yes.** Every Z4 candidate is marked for whether it trains SDXL; best-of-N is baseline only | **nothing tested yet.** Z1 found no paper that measures part borrowing pair by pair | **nothing failed.** One setup risk caught before any run: reading only CO3's corrected steps 0 to 5, where the latent is 0.6% to 1.7% picture; the owner chose to read three windows across the path | no automatic label for part binding exists (DeLeaker: crop-score agreement with leakage labels near 0), so a trained checker needs eye labels, and the budget is the owner's |

## Each loop's full answer

Each answer below is the loop's own, as received.

### Loop 10

1. **Account.** Read 12 papers, described Loops 5, 6, 8 and 9, marked the ideas and their assumptions, designed one digit experiment, ran X5, and wrote the verdict ([09-verdict.md](09-verdict.md)); an independent audit narrowed its closures ([12-independent-audit.md](12-independent-audit.md)).
2. **Still working towards it?** Partly. The loop is closed; its block-wise reading of the shared-space idea is now Paper 2's lead direction.
3. **Falsified?** Only that a probe on the digit score path reads presence at 0.75: it reached 0.700 at timestep 480.
4. **Right reason?** Mostly the setup, for the reasons in the table.
5. **Change.** Pose the additivity question on SDXL, where a joint prompt exists, in the block-wise form.

### Loop 9

1. **Account.** Read 12 papers plus Latent CLIP; scored NoisyCLIP, frozen CLIP and Latent CLIP against right and wrong sentences on 220 labelled cat and dog pictures at steps 0 to 15 (W2), with a positive control and fair settings (W2c); ran a linear probe on frozen image vectors (W2b); trained its own noise-aware model on digits (W4); wrote the verdict (`noise-text-critic/05-verdict.md`). The loop is complete and stopped.
2. **Still working towards it?** Partly. It answered whether sentence scores give the five checkers cheaply: no. It closed without building a checker, and W3's push never ran.
3. **Falsified?** Yes. "A sentence score reads the properties at steps 0 to 15" is falsified for counting, prompt match and left-right order: best mean balanced accuracy 0.574 against 0.75 (W2c, Latent CLIP, real sampler states). "Our own model carries to new concepts" is falsified: unseen digit pairs 0.590 against seen 0.758 at digit steps 14 to 25 (W&B `en52x031`).
4. **Right reason?** The idea, not the setup. On the same pictures, labels and protocol, a readout on SDXL's U-Net features reaches 0.87 by step 15 on 195 real runs. The bar is formally inconclusive only because that control averaged 0.637 over steps 0 to 15 on noised pictures, under its 0.65 floor. Position and size, and binding, failed on labels: eye checks of 37 of 57 and 26 of 32, against 90%.
5. **Change.** Nothing for this loop. The five checkers should be read from SDXL's own features, which is Loops 6 and 8's route.

### Loop 12

1. **Account.** Z1 done: 13 leakage papers read in full (the 5 named plus 8 from a recorded search), 321 quotes checked (`leakage/01-reading/mechanism-table.md`). Z2 designed with bars committed (`z2_bars.py`: Spearman rho at least 0.706 over 12 pairs, 7 tests); smoke job 94366 submitted after the owner answered board question 82. Z4's candidates are in `leakage/04-paper-2-method.md`.
2. **Still working towards it?** Yes. Every Z4 candidate is marked for whether it trains SDXL, and seed restarts or best-of-N are baselines only. The reading went past the 5 named papers because none of them measures which pairs leak.
3. **Falsified?** Nothing tested yet. No paper measures part borrowing pair by pair; the nearest, DAAM (arXiv 2210.04885, p. 7), finds same-category pairs with low attention overlap drew both objects in 71.7% of images against 9.75% with high overlap, a different failure on SD 2.0.
4. **Right reason?** Nothing has failed. A setup risk was caught before any run: reading only CO3's corrected steps 0 to 5, where the latent is 0.6% to 1.7% picture on SDXL's schedule. The owner chose to read three windows: early 0 to 5, middle 10 to 20, late 25 to 45.
5. **Change.** No automatic label for part binding exists (DeLeaker: crop scores against leakage labels, Spearman between -0.027 and 0.027, p. 18), so a trained checker needs labels by eye, and the labelling budget is the owner's call.

### Loop 4

1. **Account.** Read the 25 rivals and took 6,686 printed numbers (`targets/numbers.csv`). Set 7 targets on 6 benchmarks in `scoreboard.json`: Paper 1 CLEVR 7 objects; Paper 2 Color 0.7927, Shape 0.6151, Texture 0.6994, Complex 0.4761, CLEVR 7 objects, GenEval 0.79. Wrote the fairness audit (114 rows: 11 fair, 70 fixable, 32 not comparable), the comparison rules and the CLEVR plan.
2. **Still working towards it?** Partly. The bars still come from printed numbers, but by the owner's answer to board question 47 a win is judged against the best finished rerun on our own pictures, because printed scores did not reproduce. By question 50 the headline is the two-subject prompts (Color 233, Shape 209, Texture 234, Complex 463).
3. **Falsified?** (a) Printed rival scores reproduce on our setup: RTD Complex 0.2423 against 0.4661, CO3 0.3326 against 0.4406, ToMe Shape 0.5502 against 0.6051. (b) Beating sample-and-paste on CLEVR by object count is reachable: by Bradley's rule paste reaches 7 objects at 90.6% per-object accuracy and the test's top of 12 at 94.4%, and M7's experts placed their object in 256 of 256 samples. (c) GenEval counts two separate objects: its evaluator checks each class alone, so one fused object passes.
4. **Right reason?** Mostly setup. CO3's released script differs from its paper's section 4.1. ToMe states no image count or seeds. On CLEVR, position-conditioned experts make pasting optimal by construction (Projective Composition, p. 2), so object count cannot separate Paper 1 from paste.
5. **Change.** Paper 1 needs a target where pasting fails: objects that overlap or shade each other, or the which-object models, pending M7's paste score and the owner's choice of replacement test.

### Loop 5

1. **Account.** Rival reruns from released code are in `reproductions/table.csv`: 20 rows, 2 reproduced, 3 not, 13 not comparable, 2 blocked. Paper 2's two-subject prompt lists are in `reproductions/two-subject-prompts.json`. The loop stopped on 2026-11-23.
2. **Still working towards it?** Yes for the strongest rival per benchmark, the owner's narrowing in board question 43. Partly overall: 13 rows kept as printed context, and no rebuild of Bradley's local patch model.
3. **Falsified?** Reproduced: CoMat Color 0.7696 against 0.7827 printed (BLIP-VQA, W&B `dk8g2tpx`); ToMe Texture 0.6503 against 0.6894. Not reproduced: ToMe Shape 0.5502 against 0.6051, CO3 Complex 0.3326 against 0.4406 (ImageReward), RTD Complex 0.2423 against 0.4661.
4. **Right reason?** Our setup checks out: plain SDXL reproduced its printed score on Complex (0.2637), Texture (0.5233) and Color (0.6172), though not on Shape (0.4825 against 0.5408). So the Complex misses look like the rivals' own: on the same 2,964 pictures RTD sits 0.0214 below plain SDXL, and CO3 beats RTD.
5. **Change.** Nothing; the loop is closed. Paper 2's Complex rival is CO3 at 0.4628 on the 463 two-subject prompts, above RTD's 0.3689.

### Loop 6

1. **Account.** M1 tested each method against its simplest baseline: Paper 1's push 295 clean pairs of 320 against paste's 287 (margin 13.54, W&B `vekm3sfh`); Paper 2's push 7 faithful pictures of 96 against CO3's 11 (`s7o6404z`). M7 rendered 22,000 CLEVR pictures, its scorer reads 100% on 400 held-out, and its 4 experts draw in place 256 of 256 each; the paste waits on board question 66. M9's regions drew 8 of 8 clear pairs with bootstrapping and 0 undamaged; rung 2 stopped when 11 of 24 mask reads failed.
2. **Still working towards it?** Partly. The baseline tests ran as intended, but the search for checker, labels and settings (M5, D3, M2, M3) has not started; M7 and M9 came first by owner order.
3. **Falsified?** On tuning data: Paper 2's kept push beats CO3 alone; Paper 1's push beats paste; plain regions from step 0 separate cat and dog (0 of 8).
4. **Right reason?** Mixed. Paper 2: CO3 drew two bodies in 96 of 96, so the open failure is identity, and the push's presence outputs learned on cat and dog only (1,603 pictures), so the checker's data may be wrong for the question. Paper 1: with clean experts, separated objects leave paste near its ceiling, so this setup cannot separate a method from paste. M9 rung 2 failed at its region reader, a setup failure.
5. **Change.** Paper 1 needs a test where objects interact so paste can fail; Paper 2 needs a method aimed at identity; both need the owner's answers to questions 65 and 66.

### Loop 7

1. **Account.** U1: 25 papers read from full PDFs, Mid-U Guidance marked not read, 463 quotes found on their pages. U2 derives when checker logs may be added (`abstract-checkers/02-score-view.md`). U3 specifies the five checkers, steps V2 to V5, the question head and the scaling from 2 to 9 objects (`03-design.md`).
2. **Still working towards it?** Yes. Each checker trains only on pictures with two subjects and pushes at its own fixed length, so removing one leaves the others unchanged.
3. **Falsified?** On a computed toy, not a run: that the five checkers' logs can simply be added. With nested labels the summed gradient is 2.619 against a true 2.048, 27.9% too large; a shared cap raises each remaining push by 11.8% when one is removed.
4. **Right reason?** Yes: it is mathematics, and the design now trains the checkers as a chain. Two setup problems: the digit model never saw two digits on one canvas (handled through question 58), and lighthouse and sailboat is both a D3 tuning pair and a V4 unseen pair at seeds 6001 to 6008 (reported to s6).
5. **Change.** Put the question head for identity and binding on the board after V3 and W2 report, because CO3 already draws two bodies in all 96 tuning pictures and fails on identity.

### Loop 8

1. **Account.** V1 built the digit canvas with exact labels. V2: a count checker reads two digits apart at 0.791 on unseen digits 5 to 9 against 0.782 on 0 to 4; the identity checker reads 0.555. V3: its push gives 43 clean pairs of 320 on unseen digits, against 0 plain, 1 random and 0 best of 4. V5 is null by its written rule. V6, an identity grid, is running.
2. **Still working towards it?** Partly. The digit half matches: count is learned on some concepts and used on others. The SDXL half (V4) waits on Loop 6's D3.
3. **Falsified?** "A map reader learns which digits are present at SDXL-matched early noise": 0.555 against 0.75. "Each checker controls only its own property": removing count or order moved another score about 7 points against a 4.2-point limit.
4. **Right reason?** The identity miss looks like setup more than idea: the same two-map input reads 0.563 early and 0.702 in a later window, SDXL's steps 16 to 31 (V6 grid, job 94336); Loop 10's signature read 0.655. V5's side effects come partly from its weak right-digits checker (0.556) and from scoring each column only on canvases that passed the one before.
5. **Change.** Run V4 on SDXL without waiting for D3, judged against CO3 as well as the plain product, since CO3 already draws two bodies in 96 of 96.

## What the answers say together

**Every loop still points at the owner's two papers.** None has drifted to another question. Three loops are partly off course for reasons the owner chose: printed targets replaced by our own reruns, the search in Loop 6 deferred behind M7 and M9, and Loop 8's SDXL step waiting on Loop 6.

**The same two gaps appear in four or more answers.** For Paper 1, Loops 4 and 6 and the audit all say the current setup cannot separate the method from pasting, because pasting is exact when each expert owns its region; the paper needs a task where objects interact. For Paper 2, Loops 6, 7, 8 and 12 all say the open failure is identity, since CO3 already draws two bodies in 96 of 96, and no method aimed at identity has run.

**Most failures were the setup's, not the idea's.** Only Loop 9 calls its result the idea's fault, and it names the replacement: read the checkers from SDXL's own features. Loops 4, 6, 8 and 10 each name a setup that could not answer the question it was given.
