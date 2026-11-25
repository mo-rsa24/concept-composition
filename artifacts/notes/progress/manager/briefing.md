# 🧭 Briefing: where the PhD work stands

The one page to read. Loop 13 rewrites it every 6 hours, and at once when you say "updates", "brief me"
or "where are we". The full detail is on [the status page](status.md).

**Written:** 2026-11-25 03:16 (UTC+2).

## Contents

- [Where we are](#where-we-are)
- [For you to decide](#for-you-to-decide)
- [What the manager does for you](#what-the-manager-does-for-you)

## Where we are

<a href="briefing/where-we-are.png"><img src="briefing/where-we-are.png" width="200" alt="The running loops feeding Paper 1 and Paper 2, Loop 11 stopped until a new result, and your one open question on the left."></a>

1. **Paper 1** (experts that never met) is level with simply pasting the two experts' samples (295
   clean digit pairs of 320 against 287). Loop 6's CLEVR test where neither expert is told where to
   draw is still rendering, with a paste comparison beside it.
2. **Paper 2** (better than CO3) has no working fix yet. Loop 12's drawn-box fix took the cheetah's
   spots off the cougar on one seed, but over 48 pictures it gave 5 two-clear, the same as CO3 alone;
   it tries a text-side fix next. Loop 6's kept checker does not see CO3's failure (right 60% of the
   time against a 65% floor), and its two new checkers are being read.
3. **Loop 8** missed twice at reading digits from the sampler's own states (0.651, then 0.552 after
   memorising its training digits). The reviewer read it again, Loop 13 sent a method review request,
   and a free check now asks whether the digits are readable there at all before any more training.
   Its SDXL test against CO3 does not wait on this.
4. **The cluster** still has no idle node; 5 of our 6 `bigbatch` slots run, and the borrowed card on
   mscluster107 is no longer in use because the other user's job ended.
5. **Waiting on you:** one question, 96, on how Paper 2's checkers are judged. Since the last briefing you answered 79, 80, 86, 87, 88, 89 and 90, and
   Loop 13 decided 91 to 95 for you; each can be overruled by answering it again.

## For you to decide

Answer with `python3 scripts/progress_board.py answer <number> "<your answer>"`, or in this session.

| # | The question | Default |
|---|---|---|
| 102 | Loop 8's SDXL identity test now costs 5 to 14 GPU-hours. Loop 13 recommends waiting a few hours for Loop 6's 0.3 GPU-hour identity read, then the cheaper gated version if identity proves readable by step 15, otherwise closing it. | A: the cheaper gated version |
| 96 | Should Paper 2's checkers be judged on ranking good pictures above bad ones (what actually steers the push) instead of their yes-or-no call at zero? Both loops' checkers rank moderately but miss at zero. | keep the recorded gate, report ranking beside it, and bring a pass on ranking alone back to you |

**When Loop 11 restarts.** Loop 13 may not start sessions, so it tells you when Loop 8's identity result
or Loop 6's two-colour CLEVR test is done; restarting Loop 11 then puts the new result on a report page.

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
