# 🧪 Review: does one pooled adapter transfer at all?

**Answered yes: held-out 0.906 at step 60000 (n=64), level with the 0.909 on training pairs, where plain PoE composes near zero.** This file judges
[the plan that trains one pooled adapter](../plans/hypothesis/03-does-one-pooled-fix-transfer-at-all.md).
It exists to answer one cheap question before fifteen expensive runs are paid for: does the fix
reach unseen pairs at all? It does, so the fifteen runs are warranted.

Every checkpoint is scored from the renders alone with the current instance-count scorer. The
go-ahead for the fifteen runs is given below. One thing is still owed: the direction measures were
wired after this run finished. The paper's draft and figure F8a still carry 0.96; see
[What the write-up owes](#what-the-write-up-owes).

## Recommended prompt (to finish the read)

```
/analyze-run phase1_r8_100k
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis/03-does-one-pooled-fix-transfer-at-all.md) | the pooled adapter, the split it is tested on, the level it must beat |
| **this file** | **the verdict: it transfers, at step 60000, with three things still owed** |
| [the fifteen runs this unblocks](04-transfer-as-a-rate-over-fifteen-pairs.md) | transfer as a rate over fifteen held-out pairs |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Go-ahead for the fifteen runs](#go-ahead-for-the-fifteen-runs)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **Pooled**: one adapter trained on all eleven training pairs at once, rather than one adapter
  per pair.
- **Held-out**: a pair the adapter never trained on. The only kind that tests transfer. Its
  opposite is a pair it did train on.
- **The split it was tested on**: the unseen blend pairs, the known-failure reference pair, and
  the control pair that composes fine with no adapter at all.
- **With its step**: a transfer number is only meaningful beside the checkpoint it came from,
  because the run kept training after it was measured.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim** (does the fix transfer at all; it decides whether the 15 runs go ahead).

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| phase1_r8_100k, pooled rank-8 LoRA on 11 training pairs, 88 pair-and-seed runs | Tests the claim | config `all_groups` | 100000 steps | `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k`, checkpoints to step 100000 | finished; read incomplete |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ✅ Does held-out [compose rate](../../../../context/world/compose-rate.md) beat the level plain
      PoE reaches with no fix at all? Yes. At step 60000, scored from the renders alone with the
      instance-count scorer (two or more animal boxes, each spanning at least 25% of the image's
      long side): in-distribution 0.909 (80 of 88), held-out 0.906 (58 of 64). Per held-out pair, 8
      seeds each: cow×buffalo, goose×swan, leopard×jaguar and elephant×penguin (control) 1.0,
      seal×walrus and eagle×hawk 0.875, cat×dog and frog×toad 0.75. Plain PoE composes near zero.
      Scored 2026-11-13 (job 86591); numbers in the run folder's `compose_rate.json`.
      **How the number was cleaned.** From step 20000 on, the trainer saves each render twice: as
      itself and as a three-panel comparison strip (`__cmp`: joint-prompt target, plain PoE,
      adapter). The scorer used to count the strips as renders, and their target panel always holds
      two animals. The first scoring (0.961 held-out at step 60000) included them; so did a rescore
      with the size filter (0.742), because the filter drops the strips' small animals. Scored
      from renders alone, the filter changes 1 call in 64 (frog×toad seed 15, a speck it rightly
      drops). The held-out grids with every box drawn are in the run folder's
      `filter-calls-step-060000/` (job 86592).
      **Read by eye, cat×dog is weaker than 0.75.** Of its 8 held-out seeds, 3 show a clear cat
      beside a clear dog (9, 12, 16), seed 13 shows two cats, seeds 11 and 15 are layered sketches,
      and seeds 10 and 14 hold one animal. The scorer counts animals and cannot tell a cat from a dog.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

Nothing beyond the one question above. This run was deliberately cheap and asked one question.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become the question above**, because it was
written with the answer already visible.

- [x] ✅ Do steps 70000 to 100000 change the read? No. Held-out reads 0.88, 0.89, 0.83 and 0.91
      at steps 70000 to 100000 (n=64 each), in-distribution 0.92 to 0.94 (n=88). From step 20000
      on, held-out moves between 0.83 and 0.91 with no trend; one seed per pair is 0.016, so these
      swings are a few seeds.
- [ ] ⚠️ Does the direction axis agree? The two direction metrics (plan 02) were wired after this
      run's eval; a pair sitting at the no-fix level cannot yet be split delivery-null vs
      no-transfer.

      > A null on such a pair means it composed as rarely with the fix as without it.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Held-out and in-distribution pairs are scored by the same
      scorer at the same checkpoint, and the only thing differing between them is whether the
      adapter trained on the pair. Confirm the eleven training pairs and the held-out set do not
      overlap, from the realised split rather than the config.
- [ ] ⚠️ **Was the measuring tool sound?** The compose-rate scorer must have read only this run's
      output directory. This is the fault that has already produced convincing wrong numbers in
      this project once, in the run across correction amounts.
- [ ] ⚠️ **Did the run respect the environment?** Output under `/datasets`, checkpoints written
      through to step 100000, and no silent fp16 fallback in the eval hook.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#go-ahead-for-the-fifteen-runs) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| held-out compose rate 0.906 (n=64) at step 60000, level with 0.909 on training pairs, where plain PoE composes near zero | the step; that the scorer counts animals and cannot tell a cat from a dog (cat×dog reads 0.75 by the scorer, 3 of 8 clean by eye); and that `paper/iclr/DRAFT_MAP.md` and F8a still carry the 0.96 that counted the comparison strips |
| the pooled fix transfers to unseen pairs | that this is one pooled run, not a rate. The rate comes from the fifteen runs, and the sentence must not read as though this run supplied it |

## Go-ahead for the fifteen runs

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

**Go.** Step 11's fifteen training runs may launch. The backing number: one pooled rank-8 adapter
composes 0.906 of held-out renders at step 60000 (58 of 64, 8 unseen pairs, every pair at 0.75 or
above), level with 0.909 on its training pairs, where plain PoE composes near zero. Scored from the
renders alone by the instance-count scorer with its 25% size filter. Held-out stays between 0.83
and 0.91 from step 20000 to 100000, so training length is not what the fifteen runs need to vary.
Decided 2026-11-13.

One condition before launch, carried from [the live-curves review](02-three-live-curves-while-training.md#asked-after-the-result):
anchor images exist for only 2 of the 19 sampled pairs, so the in-training compose-rate curve
averages over those 2. Generate anchors for the other 17 first, or read the fifteen runs' compose
rate from a scoring pass like this one rather than from the live curve.

## Still open

Navigation: ⬅️ [Go-ahead](#go-ahead-for-the-fifteen-runs) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether the direction axis agrees | re-running eval with the two direction metrics from plan 02 wired in | telling a delivery-null pair apart from a no-transfer pair |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Update F8a's numbers and DRAFT_MAP's to the renders-only rates (F8a's claim itself holds), then
re-read the pairs that came out no better than plain PoE on the two direction measures.
