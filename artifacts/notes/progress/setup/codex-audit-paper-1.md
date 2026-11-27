# Paper 1 independent audit

## 1. Code

**Verdict:** the first result's numerical failure is reproducible. The implementation largely does what its immediate experiment says. Several safeguards are weaker than their descriptions, and the evidence does not justify declaring classifier guidance generally unable to solve the task.

Audit date: 2026-11-27, Africa/Johannesburg. Repository: `/home/molef/PhD/concept-composition`, branch `exec/classifier-guidance-ladder`, inspected HEAD `5e88bb2dc607127823a8549595403e5606945067`. Existing unrelated working-tree changes were left alone. No skill was used. The two requested checkpoints were observed, and the owner approved continuation after each.

Path abbreviations used throughout:

- **S** = `2-classifier-guidance/scripts/paper1` in this repository.
- **P** = `/datasets/mmolefe/poe_repair_min/paper1` on mscluster84.
- **E** = `P/paper1-first-eye-sheet`; **F** = `P/paper1-baseline-sheet`.
- **R** = `artifacts/results/does-the-contact-push-beat-the-region-sampler`.
- **A** = `R/first-result-answer.json`; **B** = `R/no-training-baseline-blind.json`.
- **Supervisor** = `artifacts/notes/progress/setup/supervisor.md`.

Line references refer to the inspected working-tree code. “Correct” means the stated operation is implemented, not that its scientific assumptions have been proved.

| Item | Verdict | Evidence and limits |
|---|---|---|
| Push, D' = D + w sigma² grad(logit) | Correct | S/paper1_sample.py:710,725,731,733,740. Positive sign, current noisy image, gradient through both frozen experts and the bounded head. |
| Frozen decoder features and head | Correct | S/paper1_disc.py:127,163,173,201; S/paper1_sample.py:135. Frozen weights do not prevent image gradients. Features come from the named decoder blocks and are concatenated at the larger grid. |
| Bounded logit | Correct formula; wrong claimed guarantee | S/paper1_disc.py:189 implements 4 tanh(raw/4). Its derivative is sech²(raw/4) times the raw derivative. It can suppress a saturated gradient; it does not guarantee a useful gradient or impose a fixed bound on the gradient's norm. |
| Heun sampler | Correct | S/paper1_sample.py:99,106. Deterministic schedule, Euler prediction, correction except at the final step. “63 calls” means denoiser evaluations, each of which may call multiple experts. No claim of exact numerical equality to every EDM2 implementation follows. |
| Start step and guidance window | Correct | S/paper1_sample.py:113,680,689. Indices are zero-based. The final window step's correction is not pushed. Harmonise resumes at the selected schedule index, not at index zero. |
| Region denoiser and shared noise | Correct | S/paper1_sample.py:149,156,330,461,503. Same seed gives both lone experts the same initial noise. Both active masks give D_red + D_blue - D_empty. Purple from the first step is an inference, not a saved-state observation. |
| Lower object owns overlap | Correct heuristic | S/paper1_sample.py:256,269,283,443,454. Connected foreground components supply bottom rows; larger bottom row wins, ties go to blue. Foreground differences can include shadows, and connected objects can merge into one component. This is not renderer depth. |
| Separate starting noise | Correct, narrower than a fully independent sampler | S/paper1_sample.py:170,185,330,798. Blue's lone image uses seed s + offset. A single joint noise tensor is assembled by masks. With both masks active, blue supplies overlap noise; with exclusive masks, its owner does. Both experts subsequently read the same joint state. |
| Offset bookkeeping | Unclear as a complete guard | S/paper1_sample.py:176 checks blue-folder metadata, and :1053 prevents mixing offsets while constructing a split. But seeds_for at :425 does not check the selected split's offset against the sampler's offset. Inspected launchers use matching folders and splits; the general guard is incomplete. |
| Harmonise | Correct | S/paper1_sample.py:586,592,626. Encoder maps the finished image to model scale; sigma_S times unit noise is added; Heun resumes at S with fixed lone masks. It is a tested re-noising baseline, not a guarantee of preserving object identity. |
| Contact rule | Correct for its stated pixel proxy | S/paper1_sample.py:968,976. Foreground must differ from empty by more than 0.1 on the 0-to-1 channel scale. Red or blue dominance must exceed 0.1. Any qualifying red and blue pixels in a 3-by-3 neighbourhood suffice. Counts, object integrity and front/back order are not tested. |
| Purple share | Correct for its stated proxy | S/paper1_overlap_stats.py:40,49. Denominator is output foreground inside fixed lone-mask overlap. Similar red and blue channels qualify, including dark or grey foreground. Undefined shares are excluded; the pooled statistic is a median of image shares, not a pixel-weighted pooled share. |
| Overlap band matches first-result plan | Wrong | First-result plan :71 adds another pixel on either side of the intersection of dilated masks. S/paper1_sample.py:705 returns just that intersection. Band statistics and the weight anchor use this narrower code definition. |
| Training negatives and noise | Correct | S/paper1_disc.py:229,271,375,383. Finished PNG images are loaded, shifted, then independently forward-noised. There is no per-step sampler-state loader on this training path. |
| Alignment loss | Correct for first-result design; not identical to full-plan variant | S/paper1_disc.py:331,339,350,388. Whole-image cosine uses x0 - D_region with renderer masks and differentiates through the gradient. Full plan :379 instead describes a joint-model teacher's denoised output minus D_region. No such teacher was used here. |
| Negative touching reweighting | Correct in expectation, not exact in every batch | S/paper1_disc.py:297,379 samples with replacement using weights. It targets the positive touching proportion per count but does not force that proportion in each batch. If all negatives have the same status, it leaves weights unchanged. |
| Third reviewer | Correct for the written fieldwise rule | S/paper1_sheets.py:239. Agreement wins; otherwise a nonempty third field wins. This can assemble a record no single reviewer supplied. It is not majority vote over whole scenes. |
| The word “none” | Correct for final success scoring; different during settlement | S/paper1_sheets.py:229,251 preserves “none” and accepts it as an adjudication. An empty disputed third field remains unresolved. S/paper1_first_bars.py:99 requires “yes” for touching occlusion, so both “none” and empty fail that success condition. |
| Reader, paired test and real-scene threshold | Correct arithmetic, incomplete validation | S/paper1_first_bars.py:64,75,99,151. It implements the exact one-sided binomial test and amended threshold. It checks push/region pair equality but not the required number of pairs or random-control pair equality. No real scenes bypasses the gate at :165. Blank agreed calls are accepted by settlement. |
| Blind sheet construction | Correct anonymity mechanism, limited guarantee | S/paper1_sheets.py:135,183. Random letter names, re-encoded pixels and an external key hide explicit labels. This does not certify that reviewers lacked other context or had independent errors. |
| Leakage check | Correct within one pack; incomplete across experiments | S/paper1_pack.py:96,123,156. Checks exact render and packed-pixel hashes across that invocation's training and held-out partitions. Does not compare against earlier blind sheets or prevent later reuse of their scenes. |
| High-noise diagnostics labelled “10.5-3.8” | Wrong label on all-step runs | S/paper1_sample.py:748 checks lower edges only. Every sigma above 3.8, including sigma 80, is assigned to that label. This does not change sampling, but those push summaries are not restricted to sigma 10.5 through 3.8. Direction-check bands remain separately defined. |

**Verification limits.** I inspected all six `tests/test_paper1_*.py` files. Tests cover synthetic push arithmetic, Heun start behaviour, overlap ownership, offsets, harmonise, contact, purple shares, head gradients, alignment, packing, bars and settlement. They do not establish realism, human agreement or cross-experiment leakage freedom. I did not run their file-writing fixtures. Eight read-only assertions on the actual bars functions passed; a separate in-memory settlement check confirmed the empty/“none” distinction. No GPU or model inference ran.

**Run provenance.** The requested local `/home/molef/.claude/jobs/bb43a4bd/tmp/` directory is absent. I read its available launcher counterparts under P/launch, including every p1 shell launcher in filename order. Logs identify initial code as 71749b47, tuning as 5acbe69d, final rows as c2895e60, the first sheet as 65a0a3ee, lower-overlap as 97bcd5e6, harmonise/offset as 621bba59, most diagnostic runs as 6e9fb248, larger weights as 8ce5e30b, and separate-noise push as afde29f8. Git comparisons show no Paper 1 source changes from 71749b47 to c2895e60, from 621bba59 to 6e9fb248, or from 6e9fb248 to 8ce5e30b. The head source is unchanged from 71749b47 to inspected HEAD. These are commit and command records, not an independent hash of the historical live working tree.

The separate-noise measurement initially failed because its launcher passed an unsupported offset argument to the overlap-statistics script. P/logs/paper1-diag-push-sepnoise.log records that failure and the successful corrected measurement. P/launch/p1-sepnoise-checks.sh uses the offset blue folder and split without that unsupported argument. The existing final contact and overlap JSON files correspond to the corrected read.

## 2. Numbers

The ledger below covers every numerical claim in the owner's “What is already known” list, including the three forecasts. **41 claims match, 1 differs at printed precision, and 7 are unverifiable as independent numerical results.** Of those 7, three are future forecasts, three require missing completed second-sheet calls, and one lacks its contact record. The printed-precision discrepancy is not evidence of a material change in the colour result.

Methods: I independently parsed the first sheet's CSVs, settled each field using agreement or the third call, and counted success by the written rule and key. I also summed saved per-count contact counts and denominators, and took medians of saved per-seed purple shares. Thus contact values are verified aggregation of stored measurements, not a fresh image-pixel audit. Purple inputs have already been rounded to four decimal places. Settings were checked against train/push JSON and the executed commands. “Same” for a setting means its record matches, not that a model was retrained.

| ID | Claim | Recomputed or inspected value | Source | Verdict |
|---|---|---|---|---|
| N01 | First blind evaluation size: 96 touching pairs | 96 pairs for each generated row | E/key.json | same |
| N02 | First push: 1 right scene | 1 of 96 scenes | E/key.json + E/sheet/calls-{a,b,c}.csv; A | same |
| N03 | First region: 2 right scenes | 2 of 96 scenes | E calls; A | same |
| N04 | First lower-on-top paste: 44 right scenes | 44 of 96 scenes | E calls; A | same |
| N05 | First real control: 68 right of 76 scenes | 68 of 76 scenes | E calls; A | same |
| N06 | First head: 64 training scenes | 64 scenes, of which 36 touch | P/paper1-first-disc-k64/train.json; P/paper1-first-pack/k64-s1/touching.json | same |
| N07 | First push window: steps 10 to 22 | Zero-based steps 10 through 22; 25 denoiser evaluations | P/paper1-first-row-ours-k64/push.json | same |
| N08 | First push weight: 1 | Weight 1 | P/paper1-first-row-ours-k64/push.json | same |
| N09 | Logit cap: 4 tanh(raw/4) | Bound parameter 4; stated formula implemented | P/paper1-first-disc-k64/train.json; S/paper1_disc.py:189 | same |
| N10 | Old real bar: 82 of 96 | 82 scenes = ceiling(0.85 × 96) | git show 9f994f72^:2-classifier-guidance/scripts/paper1/paper1_first_bars.py | same |
| N11 | New real bar: share 0.85 | 0.85 of real scenes called | commit 9f994f72; S/paper1_first_bars.py:46 | same |
| N12 | New real bar: 65 of 76 | 65 scenes = ceiling(0.85 × 76) | S/paper1_first_bars.py:75; E/key.json | same |
| N13 | Region purple share: 0.986 | Median 0.98575 from saved per-seed shares; 624 valid of 626 pairs | P/paper1-diag-push-all-steps/overlap-stats.json, region entry | same |
| N14 | Lower-owner purple share: 0.154 | Median 0.1545; 617 valid of 626 pairs. Quoted 0.154 retained alongside it | P/paper1-next-overlap-stats/overlap-stats.json, lower-overlap entry | different |
| N15 | Second blind lower-owner: 1 right scene | Summary says 1 of 96; completed calls unavailable | B; F/sheet/calls-{a,b,c}.csv are blank | unverifiable |
| N16 | Second blind harmonise: 9 right scenes | Summary says 9 of 96; completed calls unavailable | B; F blank calls | unverifiable |
| N17 | Second blind paste: 43 right scenes | Summary says 43 of 96; completed calls unavailable | B; F blank calls | unverifiable |
| N18 | Original training noise lower bound: 0.28 | Sigma 0.28 | P/paper1-first-disc-k64/train.json | same |
| N19 | Original training noise upper bound: 10.5 | Sigma 10.5 | same train.json | same |
| N20 | Sampler starts at noise 80 | Sigma 80 | S/paper1_sample.py:81,99; diagnostic push.json records | same |
| N21 | Shared-noise touching set: 626 | 134 + 236 + 256 = 626 seed-and-count pairs | P/paper1-first-lone-red/split.json | same |
| N22 | Diagnostic guidance: all 32 steps | Steps 0 through 31; 63 denoiser evaluations | P/paper1-diag-push-{aligned,aligned-w,fullnoise,k384,k384-hw,sepnoise}/push.json | same |
| N23 | Aligned head, weight 1: 16 contacts | Not found. The available 16-contact entry is the unpushed region sampler | P/paper1-diag-push-aligned/contact.json missing; its log has no contact read | unverifiable |
| N24 | Aligned head, weight 4: 18 contacts | 18 of 626 pairs | P/paper1-diag-push-aligned-w/contact.json | same |
| N25 | Aligned head, weight 16: 35 contacts | 35 of 626 pairs | P/paper1-diag-push-aligned-w/contact.json | same |
| N26 | Aligned head, weight 64: 42 contacts | 42 of 626 pairs | P/paper1-diag-push-aligned-w/contact.json | same |
| N27 | Wider-noise head, weight 1: 16 contacts | 16 of 626 pairs | P/paper1-diag-push-fullnoise/contact.json | same |
| N28 | Wider-noise head, weight 4: 20 contacts | 20 of 626 pairs | P/paper1-diag-push-fullnoise/contact.json | same |
| N29 | Wider-noise head, weight 16: 36 contacts | 36 of 626 pairs | P/paper1-diag-push-fullnoise/contact.json | same |
| N30 | Wider-noise head, weight 64: 28 contacts | 28 of 626 pairs | P/paper1-diag-push-fullnoise/contact.json | same |
| N31 | Larger training set: 384 scenes | 384 scenes; 128 scenes in its own held-out partition | P/paper1-diag-disc-k384-aligned/train.json; P/paper1-diag-pack-k384/heldout/touching.json | same |
| N32 | 384-scene head, weight 1: 18 contacts | 18 of 626 pairs | P/paper1-diag-push-k384/contact.json | same |
| N33 | 384-scene head, weight 4: 22 contacts | 22 of 626 pairs | P/paper1-diag-push-k384/contact.json | same |
| N34 | 384-scene head, weight 16: 38 contacts | 38 of 626 pairs | P/paper1-diag-push-k384/contact.json | same |
| N35 | 384-scene head, weight 64: 68 contacts | 68 of 626 pairs | P/paper1-diag-push-k384/contact.json | same |
| N36 | 384-scene head, weight 128: 61 contacts | 61 of 626 pairs | P/paper1-diag-push-k384-hw/contact.json | same |
| N37 | 384-scene head, weight 256: 47 contacts | 47 of 626 pairs | P/paper1-diag-push-k384-hw/contact.json | same |
| N38 | Shared-noise lower paste: 517 contacts | 84 + 194 + 239 = 517 of 626 pairs | P/paper1-diag-push-aligned-w/contact.json | same |
| N39 | Separate blue seed offset: 100000 | 100,000 seed positions | P/paper1-diag-push-sepnoise/push.json | same |
| N40 | Separate-noise touching set: 527 | 78 + 200 + 249 = 527 pairs | P/paper1-next-lone-blue-o100000/split.json | same |
| N41 | Separate noise, weight 0: 83 contacts | 83 of 527 pairs | P/paper1-diag-push-sepnoise/contact.json | same |
| N42 | Separate noise, weight 16: 131 contacts | 131 of 527 pairs | P/paper1-diag-push-sepnoise/contact.json | same |
| N43 | Separate noise, weight 64: 107 contacts | 107 of 527 pairs | P/paper1-diag-push-sepnoise/contact.json | same |
| N44 | Separate noise, weight 256: 55 contacts | 55 of 527 pairs | P/paper1-diag-push-sepnoise/contact.json | same |
| N45 | Separate-noise paste: 461 contacts | 61 + 169 + 231 = 461 of 527 pairs | P/paper1-diag-push-sepnoise/contact.json | same |
| N46 | Best tested push near one quarter | 131 / 527 = 24.86% pixel-contact share | P/paper1-diag-push-sepnoise/contact.json | same |
| N47 | Method A on 384 scenes: 45% chance of beating paste | A forecast, not a measured frequency; attribution absent from current supervisor page | Owner-supplied audit brief; see section 6 | unverifiable |
| N48 | Method A on 64 scenes: 20% chance of beating paste | A forecast, not a measured frequency; attribution absent from current supervisor page | Owner-supplied audit brief; see section 6 | unverifiable |
| N49 | Head trained on sampler states: 10% chance of beating paste | A forecast, not a measured frequency; attribution absent from current supervisor page | Owner-supplied audit brief; see section 6 | unverifiable |

Additional checks needed to interpret the ledger:

| Quantity | Independent result | Source and meaning |
|---|---|---|
| First paired outcome | 1 pair both right, 0 push-only right, 1 region-only right, 94 neither right; gain minus 1 scene; one-sided p = 1.0 | E completed calls and key; agrees with A |
| Other first-sheet rows | Random push 3/96 right scenes; ordinary paste 27/96; plain sum 0/96; apart push 46/96 and apart paste 94/96 | E completed calls and key |
| First adjudication | 3,223 agreed fields and 153 third-review fields, out of 3,376 fields; 96 disputed pictures; no unresolved field | E completed calls; agreement 95.47% |
| First pack | 36/64 training scenes touch, or 56.25%; 227/448 held-out scenes touch, or 50.67% | P/paper1-first-pack/{k64-s1,heldout}/touching.json. R/README.md's description of “64 touching” training scenes is wrong |
| Larger pack | 197/384 training scenes touch; 66/128 held-out scenes touch | P/paper1-diag-pack-k384/{k384-s1,heldout}/touching.json |
| Real lower-is-nearer check | 341/346 eligible touching object pairs, or 98.55% | P/paper1-first-pack/depth-order.json; sum of saved per-count numerators and denominators. Equal-bottom pairs are excluded; this is not a generated-scene success rate |
| Shadow shortcut | Median ring darkness 11.133 grey levels for region samples versus 1.572 for real scenes; ratio 7.084 | P/paper1-first-shadow-check/shadow-check.json. The stop rule only rejects a ratio below 0.5 |
| Original head direction | Training medians, high to low noise: minus 0.0856, 0.1896, 0.2909, 0.3205 cosine units; held out: minus 0.1601, 0.0531, 0.2425, 0.3441 | P/paper1-first-disc-check/direction.json. These are whole-image directions, not contact-only directions |
| Original head accuracy | Held-out balanced accuracy: 62.95%, 73.66%, 81.14%, 89.40%; training: 64.84%, 82.81%, 96.48%, 99.61% | P/paper1-first-disc-check/accuracy.json |
| Training gauge qualification | Held-out-negative accuracy after update 500 spans 78.12% to 88.28%, rather than the supervisor's 80% to 88%; the recorded r_t reaches 0.3438 at update 1,250 | P/paper1-first-disc-k64/overfit.json. The diagnosis note's claimed r_t range of 0.53 to 0.94 misses that value |
| Smoke gate | Weight zero has zero image change; weight one median whole-image change is 0.9342 grey levels, below the 1-grey-level gate | P/paper1-first-push-smoke/push.json and supervisor's recorded override |
| Shared-noise paste distinction | Lower-on-top paste 517/626 contacts, or 82.59%; ordinary paste 546/626, or 87.22% | Diagnostic contact JSON. The diagnosis note's generic paste share of 0.87 must not be relabelled lower-on-top |
| Separate-noise improvement | Weight 16 adds 48 contacts over weight zero on the same 527 pairs, a 9.11-percentage-point increase | P/paper1-diag-push-sepnoise/contact.json. It still trails lower paste by 330 contacts |
| Overlap size | Shared-noise median 138 pixels; separate-noise median 71 pixels | Saved overlap statistics. Different touching cohorts are involved |
| Real controls reused later | 72/76 real scene identities on each sheet occur in the 384-scene training pack | Intersection described in section 3 |

The three pictures in R were inspected. They support the limited observations that the original push resembles the poor region output, and that lower ownership or harmonisation can remove blending while losing or separating visible parts. They do not establish a population success rate. They are selected examples, not a blind evaluation of every diagnostic weight.

**Missing evidence and exact read commands.** The second sheet's three CSV files exist on the cluster but contain no completed call fields. They must not be scored as an observed zero. The local paths requested in the brief are absent, so they cannot fill that gap. The aligned weight-one contact file is absent. Useful checks are:

```sh
timeout 120 ssh mscluster84 "cat /datasets/mmolefe/poe_repair_min/paper1/paper1-diag-push-aligned/contact.json"
timeout 120 ssh mscluster84 "cat /datasets/mmolefe/poe_repair_min/paper1/logs/paper1-diag-push-aligned.log"
timeout 120 ssh mscluster84 "cat /datasets/mmolefe/poe_repair_min/paper1/paper1-baseline-sheet/sheet/calls-a.csv"
timeout 120 ssh mscluster84 "cat /datasets/mmolefe/poe_repair_min/paper1/paper1-baseline-sheet/sheet/calls-b.csv"
timeout 120 ssh mscluster84 "cat /datasets/mmolefe/poe_repair_min/paper1/paper1-baseline-sheet/sheet/calls-c.csv"
```

The handoff's frozen-setting record was not available as a standalone chosen-setting.json in the inspected tuning output. The launch command and final push.json independently establish the setting that actually ran. This limits reconstruction of selection provenance, not verification of the final weight and window.

## 3. Hygiene

### Comparisons

“Clean” here means a controlled change in the stated implementation on a common comparison set. It does not mean preregistered, adequately powered or independently replicated.

| Comparison | Verdict | Reason |
|---|---|---|
| First push versus region | Clean | Same seed-and-count pairs, lone samples, masks and sampler; guidance is the intervention. The independent blind calls reproduce the negative result |
| Random push control | Clean for the implemented control | Random directions use local magnitude matching along their own trajectory. This is not a replay of the learned push's entire trajectory with only its direction replaced |
| Shared-noise region versus lower-owner region | Clean | Overlap ownership changes. The colour reduction supports the role of adding both experts inside the overlap |
| Paste versus harmonised paste | Clean as a procedure comparison | Re-noising and continuation are the intervention. It does not isolate whether any harm comes from noise, continuation length or the region denoiser |
| Lower-owner region versus harmonised paste | Mixed | Different starting image and start step, as well as the construction procedure |
| Harmonise step 10 versus step 15 | Clean | Continuation start changes within the same procedure; both saved median purple shares round to 0.163 |
| Late-window versus all-step original head | Clean for timing, limited interpretation | Extending the window also exposes this same head to noise levels outside its training range. A negative result cannot separate bad timing from an unsuitable head at those new levels |
| Original versus aligned head | Clean nominal loss intervention | Same recorded training size, bounds and training budget; alignment added. One training seed and whole-image alignment do not isolate a reliable contact mechanism |
| Aligned narrow-noise versus wider-noise head | Clean nominal noise-bound intervention | Upper training bound changes from sigma 10.5 to sigma 80. The distribution of training noise changes throughout, and sigma below 0.28 remains untrained |
| Aligned 64-scene versus aligned 384-scene head | Mixed under a strict one-change rule | Data size changes, but also positive composition, derived negative weights and exposure per scene under a fixed 4,000-update budget. For example, count-one positive touching share changes from 27.27% to 10.94%, and its touching-negative weight from about 0.4468 to 0.1792 |
| Weights within a fixed head and noise setup | Clean | Weight is the changed setting, with a common set of pairs. Selecting the best after looking at this evaluation set makes the maximum exploratory |
| Shared versus separate starting noise | Mixed population comparison | Blue images, masks, overlap geometry and the selected touching cohort all change. There are 626 shared-noise pairs versus 527 separate-noise pairs |
| Weights and paste within separate noise | Clean | Same 527 pairs. The 83-to-131 contact improvement is meaningful for this pixel rule |
| First versus second paste blind score | Mixed | Same method with fresh calls/adjudication; 44 versus 43 right scenes is not an intervention effect. The second underlying calls are unavailable |
| All diagnostics as a final test | Mixed | Repeated adaptive inspection of the evaluation seeds, changing settings and using a different outcome from blind scene correctness. These are exploratory diagnostics, not a fresh confirmatory test |

### Bars and timing

The first-result design and review prescribe a gain of at least 15 right touching scenes over region, the paired significance check, the random-control guard and the real-scene readability gate. The saved first result fails on the main outcome regardless of reasonable interpretation of the real-scene denominator.

There were nevertheless changes and overrides:

1. **Whole-image smoke gate waived after its result.** The supervisor entry at 19:37 records the measured 0.93-grey-level change, below the 1-grey-level bar, and an owner decision to proceed. This is an explicit post-measurement override. It is not a hidden change to the main success outcome, but “no bar moved after its result” is false without this qualification.
2. **Real-scene denominator amended.** The old requirement was 82 correct real scenes out of 96 expected. The sheet held only 76 because S/paper1_sheets.py:103 selects at most 32 touching scenes per count, and the held-out count-one pool supplies only 12. The supervisor records the owner decision at 03:06, calls at 03:13 and final reader result at 03:24. Commit 9f994f72 is timestamped 03:23:20 +0200. This is consistent with an owner-approved denominator correction before the final reader, and preserves the 85% target. Git alone cannot prove when the owner first saw the calls or gave consent. The original plan and review still state 82 of 96.
3. **The wider-window weight grid differs from the current design.** The anchor is weight 4. The plan at lines 507 to 508 specifies wider-window weights twice and four times the anchor, hence 8 and 16. The handoff and launcher actually use weights 1 and 2 on steps 10 through 22. The final weight 1 is real, but the requested wider-window weights were not the documented tuning comparison. Later high weights in other windows do not repair that omission.
4. **The contact-preservation guard was nonbinding.** Its baseline was about 3% contact and its allowed absolute loss was 10 percentage points. No nonnegative contact share could fail that guard. It could not protect against further separation in this regime.
5. **Outcome changed for diagnosis.** The later experiments use red-blue pixel contact and purple shares, not the original blind scene-success bars. That is acceptable for diagnosis if labelled exploratory. It cannot supply a pass or a final universal failure verdict under the original rule.

The required design and review markdown files are currently untracked. Their present text is not an independently timestamped preregistration in this repository's history. The initial code commit contains the bars before the recorded first launches, which is useful evidence, but weaker than an immutable archived design and approval record. The audit cannot certify every historical statement that no plan changed.

The reviewers were fresh Claude agents, not human readers. Explicit row names were hidden, but shared model errors remain possible. Three parts were returned to apply the purple-count rule. That is disclosed on the supervisor page; the retained CSVs cannot establish that the correction was applied consistently to all parts. “None” is equivalent to empty only for the final occlusion success test, not for adjudication.

### Cross-experiment scene reuse

**The requested no-reuse condition is violated across the experiment chain.** I extracted each real picture's scene identity from E/key.json and F/key.json and intersected it with the keys in P/paper1-diag-pack-k384/k384-s1/touching.json. Both sheets use the same 76 real scenes. Of those, 72 occur in the later training set. The four that do not are scene IDs 390, 402, 405 and 408.

The reused IDs are:

```text
64 65 67 68 71 73 74 77 79 80 83 85 86 88 89 92 94 95
96 97 98 100 101 104 106 107 109 110 112 113 114 115 116
119 122 125 127 128 130 131 132 133 136 137 146 148 149
152 155 157 158 159 160 163 164 166 167 169 170 175 184
193 196 199 202 211 220 226 234 279 348 366
```

The 64-scene pack trains on scene IDs 0 through 63 and holds out 64 through 511. The 384-scene pack trains on 0 through 383 and holds out 384 through 511. Both leakage.txt files correctly report no shared hashes *within their own pack*. They do not establish that the second pack excludes earlier held-out scenes.

This does not retroactively contaminate the first 64-scene result or the training-free baselines. It does mean those real controls and much of the old held-out evidence cannot remain “unseen” evidence for the later 384-scene model. No completed blind 384-scene evaluation is demonstrated here. A future comparison needs a new frozen scene partition and a real control drawn outside every training set used in that comparison.

## 4. Reasoning

These verdicts concern what follows from the inspected evidence, not whether an alternative method will work.

| ID | Supervisor claim | Verdict | Reason |
|---|---|---|---|
| R01 | First-result answer is no; the registered 64-scene route fails | Supported | Independently reproduced blind gain minus 1 scene and p = 1.0; the readability control passes the amended gate. This closes that registered experiment |
| R02 | The original pack passes its touching-share and internal leakage checks | Supported | The saved masks and within-pack hashes support those checks. This statement does not extend to the later pack |
| R03 | The shadow check does not stop the run | Supported | Its one-sided stop rule is not triggered by the observed ratio 7.084. Passing this rule is not absence of a shortcut |
| R04 | Passing the direction check proves the push follows contact rather than halo | Not supported | The target includes the whole image, including the halo. The supervisor later correctly retracts this interpretation. Positive cosine is not a contact-local causal measurement |
| R05 | Head training checks show a useful, noncollapsed discriminator | Partly supported | Loss falls and held-out discrimination exceeds chance in most bands. This does not rule out shortcuts or establish a useful guided sampler. The reported gauge ranges also omit recorded values |
| R06 | Shared starting noise causes aligned objects, and adding experts in overlap causes purple | Partly supported | Shared noise and overlap addition are confirmed; ownership and separate-noise interventions support their relevance. No saved step-zero trace establishes the precise onset, and noise changes also change the cohort |
| R07 | The original push is too small to repair the observed colour difference | Partly supported | Saved image-change summaries and examples support a small effect. A roughly 40-grey-level colour-distance proxy is not a necessary denoiser correction or a proof that a smaller early intervention cannot succeed |
| R08 | Lower ownership removes most of the measured purple | Supported | The median drops from about 0.986 to 0.1545 on the saved overlap measurements. It does not imply two correct touching objects |
| R09 | A learned owner can at best tie the lower-object rule | Not supported | 341/346 is an eligible real object-pair depth statistic. Generated masks can be wrong, generated scenes can lose objects, and learned layout could change the problem. The statistic is not a ceiling on scene success |
| R10 | After assigning one owner, the colour problem is solved and only a roughly 2-pixel edge remains | Not supported | The proxy can improve when a colour or object disappears. Blind failure and the examples contradict the reduction to a small boundary task |
| R11 | Withdrawing that reframe was warranted | Supported | The original blind results already distinguish low blending from correct counts, contact and occlusion |
| R12 | About 45 percentage points remain available to a learned interaction | Partly supported | The real-minus-paste observed gap is 43.64 percentage points on the first sheet. The second summary implies 44.68 percentage points. Different scene/count mixtures and an imperfect real control prevent treating this as guaranteed learnable headroom |
| R13 | The second blind baseline establishes 1, 9 and 43 right scenes | Partly supported | These values exist in B, but cannot be independently reconstructed from the retained blank calls. Its detailed failure counts are likewise summary claims |
| R14 | No training-free route beats paste; a learned interaction is necessary | Not supported | At most, the tested alternatives did not beat paste according to the saved second summary. This neither exhausts training-free procedures nor proves learning necessary |
| R15 | All-step weight-one failure rules out timing as a cause | Not supported | It tests one head and weight under out-of-range early noise. Timing interacts with head training, magnitude and trajectory |
| R16 | Removing the cap can increase the push at most 2.4 times, so strength is ruled out | Not supported | The derivative ratio is cosh²(raw/4), with no universal 2.4 bound. No exhaustive raw-logit bound or uncapped comparison is supplied. Fixed weight scaling is not the same as state-dependent uncapping |
| R17 | Alignment improves the tested direction score but does not rescue these samples | Supported | Recorded cosines improve and the measured contact ladder stays far below paste. The high band of the direction test ends at sigma 10.5, so it says nothing direct about sigma 80 |
| R18 | Extending training to sigma 80 does not rescue the tested contact ladder | Supported | Its largest contact count is 36/626, below the narrow aligned head's 42/626 in that ladder. This is a tested result, not a general claim about noise-aware guidance |
| R19 | More data is the only lever that moves anything | Not supported | Strength changes contact counts and separate noise makes a substantial difference. The data comparison also changes composition and reweighting |
| R20 | For the tested shared-noise 384-scene head, weight 64 is the best measured weight | Supported | 68 contacts at weight 64, then 61 and 47 at weights 128 and 256. This is a maximum over tested weights, not an optimized global bound |
| R21 | Across the tested ladder the best push has about one quarter pixel contact and remains poor relative to paste | Supported | Best measured share is 131/527, or 24.86%, versus 461/527 for the paired paste. The purple proxy remains high |
| R22 | Therefore classifier pushes never draw two clean objects, or the route is exhausted | Not supported | Contact is not scene correctness; at least one first-sheet push scene is judged right. The tested family is narrow, adaptively selected, and lacks a matched blind test of the best diagnostic. No universal impossibility follows |
| R23 | The broad negative verdict can substitute for testing the full plan's other models and joint ceiling | Not supported | The full plan's joint ceiling uses a substantially larger joint dataset and training run. No corresponding ceiling result appears in this evidence. A scalar feature head and its whole-image alignment are not every model in that plan |

There are **9 supported, 5 partly supported and 9 not-supported conclusions** in this reasoning ledger. “Not supported” does not mean the opposite was proved.

The change of account is not itself a fault: correcting the halo interpretation and withdrawing the boundary-only reframe were warranted. The problem is repeatedly turning a local diagnostic into an exclusion of an entire cause or model family before the relevant controlled test.

The outside reading at [the supplied ChatGPT share](https://chatgpt.com/share/6b0932ec-5838-83ea-8b1f-8964d6003e5a) did not expose its conversation through the available retrieval tools. Retrieved content contained only share-page metadata. I therefore cannot verify its four methods, its caveats, or the supervisor's correspondence to Method C. The current supervisor page also contains no 45%, 20%, 10% forecast or Method A/C wording. Those attributions in this audit come from the owner's brief, not an independently recovered transcript. Methods B and D are not reconstructed or guessed.

## 5. The two claimed causes

**Noise-range mismatch: confirmed in implementation, not established as the cause of the first no.** S/paper1_disc.py:69 and :894 to :895 set the default sigma bounds to 0.28 and 10.5. Training draws log-uniform noise at :276 and :383. S/paper1_sample.py:81 and :99 start sampling at sigma 80.

However, the original guidance is active only on steps 10 through 22. Its pushed evaluations are approximately sigma 10.521 through 0.283, almost exactly the intended training range. The small difference between 10.521 and 10.5 is a boundary rounding issue, not evidence of an early high-noise exposure. The claim “the sampler starts at 80” is true but does not explain that original result through an active high-noise head.

The mismatch is material when the same narrow-range head is pushed on all steps. Extending its training upper bound to sigma 80 addresses that early exposure and does not rescue the measured contact outcome. Calling this the “full noise range” is imprecise: the late sampler evaluations below sigma 0.28 remain outside training, down toward sigma 0.002. The evidence weakens the hypothesis that the upper bound alone explains the all-step failure; it does not prove that all training-distribution mismatch is irrelevant.

**Finished, re-noised negatives rather than sampler states: confirmed in implementation, causal effect unmeasured.** S/paper1_disc.py:229 loads finished image files. At :375, :380 and :383, training selects examples, shifts them and adds newly sampled noise. There is no collection or matching of the sampler's actual intermediate states along its guided trajectory. The negatives are not the half-drawn pictures the head encounters at each step.

This can create a distribution mismatch, especially after guidance changes the trajectory. It is not automatically a mathematical error: a forward-noised terminal distribution can be the intended density-ratio training distribution, depending on the sampling argument. The audit contains no measurement of those two distributions and no matched sampler-state-trained head experiment. Thus this is a credible untested explanation, not a demonstrated reason for failure. Its purported match to the outside reading's Method C caveat remains unverifiable because the shared conversation was unavailable.

## 6. Predictions

The three percentages are subjective forecasts, not quantities that can be recomputed from these runs. There is no resolved outcome with which to score their calibration. The session's qualitative tuning forecast of a first-result no was borne out; its stronger early predictions that ownership would leave only a small boundary problem were not supported by the subsequent evidence.

For comparability, I interpret “beats paste blind” as a higher paired blind scene-success count than lower-on-top paste on a fresh, frozen 96-pair touching evaluation, after a fixed development budget. Requiring a statistically significant win would lower these probabilities. The exact event and training budget were not supplied.

| Proposed experiment | Supervisor forecast | Assessment | My provisional forecast |
|---|---|---|---|
| Small network trained with denoising loss on 384 joint scenes, described as Method A | 45% chance of beating paste | Possible, but optimistic and unsupported as a calibrated estimate. Direct denoising supervision is a meaningful change, while 384 scenes remain a small, composition-sensitive dataset | 25% chance; subjective plausible range 10% to 45% |
| Same proposed method on 64 scenes | 20% chance | Plausible as an optimistic judgment, but current evidence does not justify that precision. Fewer touching examples and generalization risk argue for a lower estimate | 10% chance; subjective plausible range 3% to 25% |
| Head retrained on actual sampler states | 10% chance | A reasonable low-confidence forecast. It directly addresses an untested mismatch, but targets, state coverage and the scalar guidance mechanism still matter | 10% chance; subjective plausible range 2% to 25% |

These ranges are judgments, not confidence intervals. Their overlap is intentional. I have more confidence that the first proposal changes the supervision substantially than in any numerical ranking between the two smaller proposals. The failure of the present head cannot justify a very precise probability for a different model. Nor does the real-versus-paste gap guarantee that any of them can learn it.

Before resolving such a forecast, freeze the architecture, training budget, paired evaluation set, blind rubric and meaning of “beats”; exclude all training scenes from its real controls. None of these proposed experiments was run by this audit.

## 7. What the session got wrong or missed

1. **It treated weak proxies as the paper's outcome.** A lower purple median can accompany missing objects. Pixel contact can be supplied by a tiny accidental adjacency. Neither is a clean, correctly counted, correctly occluded scene. The claim that only a small boundary task remained, and the later universal failure claim, both exceed these measurements.
2. **It did not track held-out status across the chain.** Later training reused 72 of each blind sheet's 76 real scenes. Internal leakage checks did not catch this because each pack was checked in isolation. The first result survives, but those controls cannot be reused as unseen evidence for the later model.
3. **It ruled out interacting causes from insufficient comparisons.** Timing, strength and direction were declared settled before their relevant training-range and trajectory issues were controlled. The 2.4-times uncapping limit has no general code basis.
4. **It overstated what the direction test proves.** Its whole-image target rewards correcting shadows and background. Passing it cannot certify contact repair. The later correction on the supervisor page is the sound reading.
5. **The second blind result is not independently auditable from the retained calls.** Its summary may be correct, but blank cluster templates and absent local originals cannot substantiate it. The aligned weight-one contact count also lacks its claimed measurement file.
6. **“One change at a time” and “preregistered” need qualifications.** The larger-data intervention changes derived weights and exposure, the separate-noise experiment changes the touching cohort, and the evaluation set is repeatedly inspected. The smoke bar was explicitly overridden after measurement; the wide-window tuning weights differ from the present plan. The denominator correction is reasonable but lacks independent consent chronology.
7. **It converted a depth-order statistic into an unwarranted performance ceiling.** Near-perfect lower-is-nearer ordering on eligible real pairs does not establish that a learned owner or layout model can at best tie a generated-image paste.
8. **Several labels and summaries are too strong.** “64 touching” scenes should be 64 scenes with 36 touching; “full noise” omits the low-noise tail; the high-noise push bucket is mislabelled; reported gauge ranges omit values; and “none” matches empty only at final scoring. The 0.154 versus 0.1545 purple discrepancy is minor beside these interpretation issues.
9. **The account lacks the comparisons needed for a broad model judgment.** There is no independently verified blind score for the best diagnostic head, no sampler-state-negative ablation, no uncapped counterpart, no training-seed replication and no joint-ceiling result. These are limits on the conclusion, not instructions to run more jobs.

## 8. Questions for the owner

1. Can you identify the retained completed second-sheet calls and the aligned weight-one contact measurement, if they exist, so the four unresolved historical result claims can be checked?
2. Is there an immutable copy of the original design and the owner decisions on the smoke override, wider-window weights and real-scene denominator, with timestamps before the relevant results were inspected?
3. Can you provide the text of the shared Methods A to D discussion, including the forecast event and budget, so its attribution and the three probabilities can be assessed against the actual proposal?

