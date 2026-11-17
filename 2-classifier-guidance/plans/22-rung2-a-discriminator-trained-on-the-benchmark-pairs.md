# 🧪 Rung 2, second discriminator: "two separate things" learned from the benchmark's 48 pairs

## Recommended prompt

```text
/drip-execute-plan 2-classifier-guidance/plans/22-rung2-a-discriminator-trained-on-the-benchmark-pairs.md
```

## Position in the plan tree

[Root](../../MASTER_PLAN.md) › [the classifier-guided PoE route](route-classifier-guided-poe-from-toy-to-animals.md) ›
this plan, which follows [plan 21](21-rung2-a-small-discriminator-steers-cat-x-dog.md) on the
board's row A3q in [the classifier-guided PoE board](../report/classifier-guided-poe/board.md).

## Table of contents

- [Quick context](#quick-context)
- [Considerations](#considerations)
- [The claim under test](#the-claim-under-test)
- [Goal](#goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [Outputs](#outputs)
- [Error Matrix](#error-matrix)
- [Orchestration](#orchestration)
- [Next step](#next-step)

## Quick context

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**Why this plan exists**

Plan 21's discriminator, trained on 17 cat-and-dog positives, separated pair-prompt renders from
single-prompt and product renders rather than reading "a cat and a dog apart": hybrids and two of
one kind from the pair prompts scored as positives. Trained again with those failures as
negatives, it fell to 0.53 to 0.63 held-out balanced accuracy at every early step. Seventeen
positives cannot teach the concept with this design.

**What it does**

The benchmark's 48 training pairs (`pair_pool_v58.yaml`; never cat × dog, never seeds 9 to 16) are
rendered on seeds 1 to 8 with the joint prompt and with the plain product, called by eye, and used
to train two discriminators on identical data: the attention-pool head alone on frozen SDXL, and
the head with plan 18's LoRA adapters on SDXL's down and middle blocks.

## Considerations

Navigation: ⬅️ [Previous](#quick-context) | 📋 [TOC](#table-of-contents) | [Next](#the-claim-under-test) ➡️

**The renders are new pictures**

The benchmark's pictures used cached starting noise kept on `/datasets`, which cannot be regenerated
from the seed number and was unreachable on 2026-11-17 from about 13:10 UTC. Every render here uses
noise from the seed number, like every rung 2 render, so the benchmark's kept list and
`verdicts.json` do not transfer; every joint picture is called afresh.

**Labels**

Positive: the joint prompt's picture shows both named things, separate and intact. Negative: a joint
picture that does not (one thing, a hybrid, two of one kind, damaged), and every plain product
picture unless it shows both things separate (then it is left out). Claude calls by eye at full
resolution under the reader's delegation; the reader spot-checks a sample and can veto any call.
Plan 21's 17 cat × dog positives and 47 negatives are added unchanged.

**Held-out check by pair**

Eight whole pairs are left out of training (every 6th pair of the sorted list, starting at the
first), with cat × dog seeds 7 and 8. The bar is plan 21's: balanced accuracy at least 0.75 on at
least 10 of the first 20 sampler steps, 4 noise draws per tile.

**Where outputs go**

`/home-mscluster/mmolefe/poe_repair_min_scratch` (via `POE_REPAIR_MOUNT`) while `/datasets` is
down, moved to `/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_ladder/rung2/` once it
reads again.

## The claim under test

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#goal) ➡️

A discriminator trained on a few hundred joint-prompt successes and failures across 48 pairs reads
"two separate things" on pairs it never saw, and its push makes the cat × dog product draw two
separate animals on held-out seeds.

## Goal

Navigation: ⬅️ [Previous](#the-claim-under-test) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**The bar** for the discriminator is the held-out check above. If either discriminator passes, the
rung 2 sheet is rendered with it and judged by
[the rung 2 review file](review/a3q-cat-x-dog-quick-discriminator-test.md)'s seven readings,
unchanged. If both fail, the finding is that this discriminator design does not learn "two separate
things" from this data, and rung 2 is written up on that.

**Time box:** 2 sittings, about 3 hours of the reader's time, plus about 4 hours of GPU.

## Environment Facts This Plan Depends On

Navigation: ⬅️ [Previous](#goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- [The nodes](../../environment/hpc/nodes.md#partitions): `co3` on `mscluster107`, both Quadro RTX 8000 cards.
- [Storage](../../environment/storage.md): large outputs on `/datasets`; the scratch folder above is
  the reader's exception while `/datasets` is down.

## Tasks

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute. Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

◀ **Needs: [2. Test each piece](#2-test-each-piece)** walked.

- [ ] **0.1** `/verify-plan 2-classifier-guidance/plans/22-rung2-a-discriminator-trained-on-the-benchmark-pairs.md`, then `/xref-pillar` on it, at the walk's close.

▶ **Next: [the next step](#next-step)**

### 1. 🔧 Build the pieces

◀ **Needs: nothing first.**

▶ **Next: [2. Test each piece](#2-test-each-piece)**

- [ ] **1.1** Render the 48 pairs on seeds 1 to 8, joint and product, pictures and latents, one
  sheet per pair. `scripts/rung2/render_benchmark.py`, split across both cards.
- [ ] **1.2** Call every joint picture and screen every product picture; write the calls and the
  training list.
- [ ] **1.3** Train both discriminators on the list, each with the held-out check by pair.
- [ ] **1.4** For a discriminator that passes, the rung 2 sheet on seeds 1 to 4 and 9 to 12,
  `scripts/rung2/render_results.py`, unchanged columns and strengths.
- [ ] **1.5** The calls, the reading, the card and the board.

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#outputs) ➡️

For you to follow manually. Do these yourself.

### 2. Test each piece

◀ **Needs: [1. Build the pieces](#1--build-the-pieces)**

- [ ] **2.1** Spot-check Claude's calls on a sample of pair sheets; veto any you disagree with.
- [ ] **2.2** Read both discriminators' held-out accuracy per step against the bar.
- [ ] **2.3** Call the rung 2 sheet's tiles at full resolution, if one was rendered.

▶ **Next: [0. Check this plan before working from it](#0--check-this-plan-before-working-from-it)**, at the walk's close.

## Outputs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

768 renders with latents and 48 pair sheets; the calls file; two discriminator runs with their
held-out checks; possibly a rung 2 sheet; results under
`2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog/`.

## Error Matrix

Navigation: ⬅️ [Previous](#outputs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration) ➡️

| Symptom | Cause | Fix |
|---|---|---|
| `Input/output error` on `/datasets` | the mount is down | write to the scratch folder through `POE_REPAIR_MOUNT` |
| `cluster.sh run` refuses | another session's uncommitted edit | push and fast-forward the cluster checkout by hand |

## Orchestration

Navigation: ⬅️ [Previous](#error-matrix) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

The walk map is `2-classifier-guidance/artifacts/drips/rung2-pair-aware-discriminator/EXECUTE_WALK.md`,
shared with plan 21.

## Next step

Navigation: ⬅️ [Previous](#orchestration) | 📋 [TOC](#table-of-contents)

The route's step 10b if a discriminator passes and the rung 2 sheet lands on reading 1; otherwise
the write-up of rung 2.
