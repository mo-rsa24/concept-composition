```text
No skill. This is a read-only audit in the repository /home/molef/PhD/concept-composition, branch
exec/classifier-guidance-ladder.

## Background

Paper 1 composes two separately trained EDM2 colour experts on 64-pixel CLEVR scenes (one draws
red objects, one blue, plus an empty-scene model) and asks whether a learned interaction can make
red and blue objects that should touch come out cleanly, one in front of the other. A supervising
Claude session (Loop 18) ran the first pre-registered result, got a no, diagnosed it and ran a chain
of follow-up experiments, then wrote its reading on a supervisor page. The owner does not trust that
reading on its own word, because the session changed its account more than once. You are asked to
check the code, the numbers and the reasoning independently, and to say plainly where they hold
and where they do not.

## Objectives

1. Check that the code each experiment relied on does what the supervisor page says it does.
2. Recompute every number in the list under "What is already known" from the files that hold it.
3. Check the experimental hygiene: one change at a time, clean comparisons, no bar moved after its
   result, no training scene reused as a held-out or real test scene.
4. Judge whether each conclusion the supervisor drew follows from the evidence.
5. Check two claimed causes of failure against the code, and rate the session's predictions.
6. Write one audit report with a verdict on every claim.

## What to read

- plans/paper-1-clevr-touching/plans/02-first-result.md: the first result's design and its bars.
- plans/paper-1-clevr-touching/review/02-first-result.md: the verdict rule written before the run.
- plans/paper-1-clevr-touching/plans/03-full-plan.md: the full plan, its models and its joint ceiling.
- artifacts/notes/progress/setup/supervisor.md: read every paragraph that starts "Paper 1" under
  "The first runs"; these are the claims you are auditing.
- artifacts/notes/progress/setup/paper-1-handoff.md: every command that ran, and its output folder.
- artifacts/results/does-the-contact-push-beat-the-region-sampler/: README.md, first-result-answer.json,
  why-the-answer-was-no.md, pressure-test-of-the-next-design.md, no-training-baseline-blind.json and
  the three pictures.
- 2-classifier-guidance/scripts/paper1/paper1_sample.py: the samplers, the push, the overlap owner,
  separate starting noise, harmonise and the contact rule.
- 2-classifier-guidance/scripts/paper1/paper1_disc.py: the head, its training noise range, its
  negatives and the alignment loss.
- 2-classifier-guidance/scripts/paper1/paper1_first_bars.py and paper1_sheets.py: the bars, the
  reader, and how two reviewers' calls are settled by a third.
- 2-classifier-guidance/scripts/paper1/paper1_overlap_stats.py: the purple share inside the overlap.
- tests/test_paper1_*.py: the tests for all of the above.
- /datasets/mmolefe/poe_repair_min/paper1/launch/p1-*.sh on the cluster: every launcher that ran.
- /datasets/mmolefe/poe_repair_min/paper1/paper1-first-eye-sheet/: the first blind sheet, its key.json
  and the three reviewers' filled calls in sheet/calls-a.csv, calls-b.csv and calls-c.csv.
- /datasets/mmolefe/poe_repair_min/paper1/paper1-baseline-sheet/: the second blind sheet and its
  key.json. Its filled calls were lost when a laptop working folder was cleared; only the settled
  table survives, in no-training-baseline-blind.json. Report its numbers as not re-derivable.
- The run folders on the cluster under /datasets/mmolefe/poe_repair_min/paper1/, each holding
  push.json, overlap-stats.json, contact.json, train.json or direction.json, and the logs in
  /datasets/mmolefe/poe_repair_min/paper1/logs/. Reach them read-only with
  timeout 120 ssh mscluster84 "cat <file>" or "ls <folder>".
- https://chatgpt.com/share/6b0932ec-5838-83ea-8b1f-8964d6003e5a: an outside reading the owner
  shared, with four methods (A to D) for learning the interaction; it renders only in a browser.

## What is already known

These are the supervisor's claims. Treat each as a claim to check, never as given.

- The first result, on 96 touching evaluation seeds judged blind: the push (a head on frozen expert
  decoder features, 64 real scenes, steps 10 to 22, weight 1, logit capped at 4 tanh(raw/4)) was
  right on 1, the region sampler on 2, the paste with the lower object on top on 44, and real scenes
  on 68 of 76 (first-result-answer.json).
- The real-scene bar was changed from 82 of 96 to 0.85 of the scenes on the sheet (65 of 76) by the
  owner before the reader ran, because the sheet held only 76 real scenes (commit 9f994f72).
- The cause given: the red and blue lone samples share their starting noise, so their objects overlap,
  and inside the overlap the region sampler adds both experts, which draws purple; the purple share
  inside the overlap is 0.986 (why-the-answer-was-no.md).
- Giving the overlap to the lower object lowers that purple share to 0.154, but judged blind it is
  right on 1 of 96, harmonised paste on 9, the paste on 43 (no-training-baseline-blind.json).
- The head was trained only on noise from 0.28 to 10.5 (SIGMA_HI in paper1_disc.py), while the
  sampler starts at noise 80.
- The head's "drawn" examples are finished region-sampler pictures with noise added back, not the
  sampler's own half-drawn pictures at each step (the claimed match to the outside reading's
  Method C caveat).
- Red meeting blue by the contact rule, on 626 touching seeds, push on all 32 steps at weights 1, 4,
  16 and 64: aligned head 16, 18, 35, 42; aligned head on the full noise range 16, 20, 36, 28;
  aligned head on 384 joint scenes 18, 22, 38, 68, and 61 and 47 at weights 128 and 256; the paste
  517 (run folders paper1-diag-push-*).
- With separate starting noise per expert (blue offset 100000), 527 seeds touch; red meets blue on 83
  with no push, 131 at weight 16 with the 384-scene head, 107 at 64, 55 at 256, and 461 for the paste
  on the same pairs (paper1-diag-push-sepnoise, paper1-diag-paste-lower-o100000).
- The supervisor's conclusion: across timing, direction, noise range, data, strength and starting
  noise, the classifier push tops out near a quarter of seeds touching, so the push recolours and
  nudges but never draws two clean objects.
- The supervisor's predictions for the next experiments: Method A (a small network trained with the
  denoising loss on the 384 joint scenes) beats the paste blind with about 45% chance; Method A on
  64 scenes about 20%; the head retrained on the sampler's own half-drawn pictures about 10%.

## When something is missing

- If a run folder or log is missing on the cluster, say which, mark the claim unverifiable, and go on.
- If ssh to mscluster84 fails, audit the code and the laptop files, and list every cluster claim as
  unverifiable with the command that would check it.
- If the ChatGPT link does not render, rely on the supervisor's account of it, and say so.
- If two sources disagree on a number, report both with their paths; do not pick one.
- A question only the owner can answer goes into the report's last section, not to the cluster.

## Rules

- Read only. Change no file in the repository, write nothing on the cluster, and commit nothing.
- On the cluster run only read commands (cat, ls, head, python reading JSON). Never run find over
  / or /datasets, never start a job, and never touch a GPU.
- Write exactly one file: artifacts/notes/progress/setup/codex-audit-paper-1.md.
- Stop after the code check and show the list of code findings before recomputing numbers.
- Stop after recomputing the numbers and show the table before judging the conclusions.
- Use plain words, every number with its unit and meaning, and no em or en dash as a clause separator.

## What to hand back

- artifacts/notes/progress/setup/codex-audit-paper-1.md with these sections, in order:
  1. Code: each function relied on (the push D' = D + w sigma^2 grad logit, the Heun sampler and its
     start step, the overlap owner, separate starting noise, harmonise, the contact rule, the purple
     share, settling with a third reviewer and the word "none"), each marked correct, wrong or
     unclear, with file and line.
  2. Numbers: one row per number above, with the recomputed value, its source file, and same,
     different or unverifiable.
  3. Hygiene: each comparison marked clean (one change) or mixed, every bar change with its timing
     against the result, and whether any of the 384 training scenes appears among the real scenes on
     either blind sheet.
  4. Reasoning: each conclusion marked supported, partly supported or not supported, with the reason.
  5. The two claimed causes (the noise range and the mismatched drawn examples): confirmed or not,
     with code lines.
  6. Predictions: whether the 45%, 20% and 10% are reasonable, and what you would predict instead.
  7. What the session got wrong or missed, in order of how much it matters.
  8. Questions for the owner, at most three.
- The proof is the report itself: every claim listed above appears in it with a verdict.

## What you will see

- Along the way: after the code check, the session shows its code findings and asks whether to go
  on; after the numbers, it shows the table and asks the same.
- At the end, a closing report in chat: what was asked, in one sentence; what was done, with the
  report's path; the outcome, as counts of claims confirmed, wrong and unverifiable; what is still
  open; and at most one question for the owner.

The owner judges it from codex-audit-paper-1.md: done when every claim above has a verdict with
its evidence.
```
