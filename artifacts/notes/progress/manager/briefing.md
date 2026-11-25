# 🧭 Briefing: where the PhD work stands

The one page to read. Loop 13 rewrites it every 6 hours, and at once when you say "updates", "brief me"
or "where are we". The full detail is on [the status page](status.md).

**Written:** 2026-11-25 10:40 (UTC+2).

## Contents

- [Where we are](#where-we-are)
- [For you to decide](#for-you-to-decide)
- [What the manager does for you](#what-the-manager-does-for-you)

## Where we are

<a href="../leakage/09-dcfm-colored-mnist/before-after-plain-product-and-checker-push.png"><img src="../leakage/09-dcfm-colored-mnist/before-after-plain-product-and-checker-push.png" width="200" alt="Colored MNIST, the 50 digit-colour combinations neither expert saw: the plain product beside the checker push, with the share right under each."></a>

1. **Paper 1 has its first result against its closest rival.** Two Colored MNIST experts, each from
   its own data holder and never retrained, composed on the 50 combinations neither saw: the checker
   push gets digit and colour right in 76.0% of 5,000 pictures against 46.9% for the plain product,
   and a novel-combination FID of 12.63 against 25.87 ([Z9](../leakage/09-dcfm-colored-mnist/design.md)).
   DCFM's own finished-expert baseline prints 25.19 and DCFM, which retrains, 11.41; those are
   context, since DCFM uses ten holders. The push here strengthens each attribute, so it does not yet
   show the interaction story, and Logical Guidance reports a similar setting on Colored MNIST, which a
   novelty check must compare. Arms 3 and 4 (the ratio head and the trained interaction model) run next.
2. **Paper 2's checker route is closed (your answer to 111).** On SDXL the kept checker's push points
   against the joint-minus-product term (median cosine -0.105 at step 0 on CO3, random about 0), and
   the pictures that came out two clear lean further against it (-0.139). Since the joint prompt also
   loses subjects, the joint's direction may be the wrong target, which supports controlling a
   relation's strength directly.
3. **What is learnable, from the toy.** For a pair nobody saw together, the data says only "never
   together": a push faithful to it reaches the pair 0.000 of the time, a low-rank head 0.844 (true
   joint 0.924), and the same head fails when the interaction has no structure (0.120)
   ([the toy](../../../results/can-a-classifier-reach-an-unseen-pair/README.md)).
4. **Loop 15 is reading** for directions that carry to new subjects and for your three dials
   (co-occurrence strength, count, binding strength); H1 and H2 are done, H3 is under way.
5. **Compute.** Everything ran on the cluster; RunPod has stayed unused since its two tests.

## For you to decide

Answer with `python3 scripts/progress_board.py answer <number> "<your answer>"`, or in this session.

| # | The question | Default |
|---|---|---|
| 96 | Should Paper 2's checkers be judged on ranking good pictures above bad ones instead of their yes-or-no call at zero? | keep the yes-or-no gate; now moot for the closed checker route |
| 104 | Which model Paper 2 runs on | **held by you** until Paper 2 works on SDXL |

## What the manager does for you

### Deciding questions

<a href="briefing/deciding-questions.png"><img src="briefing/deciding-questions.png" width="200" alt="A question goes through the reviewer, then a test: reversible, inside budget, no recorded bar changed. Yes, the manager decides and you can overrule; no, it stays with you."></a>

When a loop asks something whose every option is reversible, inside its budget and changes no recorded
bar or verdict, the manager runs the reviewer and decides, marked "Decided by Loop 13 for the owner".
Deletions, secrets, disagreeing records, runs over 3 GPU-hours and stopping a loop stay with you.

### Resources and finish times

<a href="briefing/resources.png"><img src="briefing/resources.png" width="200" alt="The hourly cluster read: our jobs against the caps, long waits, compute per loop, and moving a waiting job to an idle partition."></a>

Every hour the manager reads the queue, the finished jobs and the node states, compares our jobs with
the caps, flags waits over 2 hours, and tells a loop when the other partition has an idle card.

### Checking designs before they run

<a href="briefing/checking-designs.png"><img src="briefing/checking-designs.png" width="200" alt="A new design is checked for its data, controls, whether the input holds the information, and cost, then one request names the fix."></a>

A new or changed design is read before its first run, for its data, its controls, whether its input
holds what it asks for, and its cost. A design likely to waste more than 3 GPU-hours reaches you.

### Things that may be wrong

<a href="briefing/things-that-may-be-wrong.png"><img src="briefing/things-that-may-be-wrong.png" width="200" alt="Failed jobs, slow compute and stale steps go on the status page; disagreeing records go to you; two misses in a row trigger a method review."></a>

Failed and slow jobs and steps with nothing running go on the status page. Two records that disagree go
to you. Two misses in a row from one loop earn it a method review request: it pauses and writes why.

### The reviewer

<a href="briefing/reviewer.png"><img src="briefing/reviewer.png" width="200" alt="The reviewer checks a loop's question, designs, pictures and results against 84 papers and gives continue, fix or stop."></a>

Once a day, and before any method review, the reviewer reads each loop against the 84 papers the
project has read and gives one verdict: continue, fix or stop. A stop reaches you as a board question.

### Nudges and re-routing

<a href="briefing/nudges-and-rerouting.png"><img src="briefing/nudges-and-rerouting.png" width="200" alt="A changed condition or a request waiting over 6 hours earns one message; a request to a stopped loop is re-routed."></a>

A loop gets one message when something it waits on changes or a request to it has waited 6 hours. A
request addressed to a stopped loop goes to the loop that now holds that work, or to you.

### Closing sessions

<a href="briefing/closing-sessions.png"><img src="briefing/closing-sessions.png" width="200" alt="Busy sessions are never closed; idle sessions of finished, closed or duplicate loops are closed after their requests move; others need your yes."></a>

An idle session whose loop is finished or closed, or a second copy of a loop, is closed after its open
requests move on. A busy session never is, and a loop with steps left only with your yes.

### Briefing

<a href="briefing/briefing.png"><img src="briefing/briefing.png" width="200" alt="Every 6 hours or on 'updates', the briefing page is rewritten: where we are, what the manager does, and what you decide."></a>

This page, every 6 hours or when you ask.
