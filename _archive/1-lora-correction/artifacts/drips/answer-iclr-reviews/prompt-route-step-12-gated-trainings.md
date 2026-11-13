# Prompt: route step 12, decide and launch the two gated trainings

Paste into a fresh session in `/home/molef/PhD/poe_repair_min`. It replaces the prompt block under step 12 of `plans/route-answer-the-iclr-reviews.md`.

```text
/drip-execute-plan --request --depth objective "Decide from the gate results whether to train the
early-only adapter and the SD 1.5 adapter, and launch only the ones whose gate passed, each matched
to the paper's recipe in everything except the axis under test"
Model: Opus, medium effort. Box: 30 minutes of your time. Filing: on.
Controls: piece 1 example (the matched config beside the paper's recipe, line by line), piece 3
debug
(a launch that does not start), piece 4 prove (that the new run differs from the old in one axis).
Side routes: piece 3 /calibrate --predict <the launch command> (80% intervals for runtime, memory
and
an outcome range); piece 4 /analyze-run <the W&B run> once the training is running.

Background
Reviewer UHz6 asked whether restricting the adapter to the early steps changes the failure types.
Both reviewers wanted a fair comparison with other methods on the same weights. Each training
answers one of these, and each opens only if a cheap test shows it can pay off. The repo is
/home/molef/PhD/poe_repair_min and cluster actions go through scripts/cluster.sh.

Objectives
1. Read the early-steps inference test (step 5, wave 1 job 4: the adapter over steps 0 to 10 and
   0 to 25 on the 53 held-out pairs) and the ceiling test (step 10) against their written bars and
   say, for each training, open or closed.
2. For each open training, write the matched config next to the paper's recipe, line by line, and
   show
   the user every difference.
3. Record a prediction for each open training, then launch it after the user's yes.
4. Set up the evaluation: the same checkpoint-choice rule that chose the adapter of record
   (protocol rule 1, the showcase rule on cat and dog and elephant and penguin), then sampling on
   the evaluation set from the same starting noise, and the new rows into R9 and, for SD 1.5, R5.

What to read
- 1-lora-correction/report/paper-review/protocol.md: the bars for both gates and the checkpoint-choice rule.
- 1-lora-correction/report/paper-review/prze-11-early-training.md and prze-1-scores.md: the outcome lines.
- 1-lora-correction/artifacts/drips/answer-iclr-reviews/ground-check.md: the paper's real recipe, pool and noise
  steps.
- environment/hpc/throughput.md: step times and the memory a rank-32 training needs.
- The training package under poe_repair/experiments/cross_pair_lora_pooling/ and its launchers
  (scripts/launch_sdlora.sh and the pooled-training launchers).
- The submitted paper, Appendix A: the stated training recipe.

What is already known
- Rank-32 training needs about 25 GB and takes about 0.236 seconds per step for rank 8 on the
  Blackwell
  (measured 2026-09-01); the user's check puts the full runs at 7 to 10 hours.
- The adapter of record is v58-03-self at step 30,000, which is 600 epochs of 50 steps
  (adapter_of_record.yaml). The paper's text says 1,000 epochs (step 50,000); route step 13
  corrects the text. A new training matches the recipe that made the adapter of record, and the
  config diff names the step it stops at.
- The paper's pair filter keeps a pair if the joint prompt or CO3 composes; on SD 1.5 only the joint
  prompt can be used, which makes that comparison mixed.
- biggpu allows one job per user; training runs on mscluster112 by the shared-device path.
  mscluster107's allocation (job 83423) ends around midday Thursday 12 November.
- Draft to supervisors is Thursday 12 or Friday 13 November; OpenReview closes Wednesday 18.

When something is missing
- If a gate's result is not there yet, say which step is owed and do nothing else.
- If the matched config cannot reproduce a line of the paper's recipe, show the line and ask the
  user.
- If the Blackwell is busy, ask before moving a training to a slower card.

Rules
- Launch nothing without both a passed gate and the user's yes to the exact command.
- Change exactly one axis per training and list every difference from the paper's recipe.
- Do not choose a checkpoint by looking at the evaluation set.
- Follow the execution protocol's safety rules for the shared-device path.

What to hand back
- For each training, open or closed with the number behind it, and for each open one the matched
  config, the prediction, the launch command and the W&B run id.
- Rows added to "What we know so far" in plans/route-answer-the-iclr-reviews.md.
- Proof: the board redraws with --step 12, and each launched run shows in pgrep and nvidia-smi on
  its node.

What you will see
Along the way: each gate's result against its bar, then the config beside the paper's recipe, then
the
prediction and launch command, each followed by one question. At the end: what was asked, what was
done with paths, OPEN or CLOSED for each training with its number, what is open, and at most one
question. You judge from the gate numbers and the config diff.
```

- `added:` the adapter of record and what its 600 epochs mean against the paper's 1,000 · `added:` where the early-steps test now lives (step 5, wave 1 job 4) · `added:` the checkpoint-choice rule by name and the slots the new rows fill · `added:` the 107 allocation's end time
- `dropped:` "to be resolved by ground-check.md" (why: step 4 resolved it)
