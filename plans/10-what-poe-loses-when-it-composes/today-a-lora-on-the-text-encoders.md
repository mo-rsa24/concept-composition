# 🎯 Today: can a small change to how "cat" and "dog" are encoded stop them fusing?

<!-- template: exempt. Reason: a two-sitting walk off a settled null, not a scope plan -->

> [!NOTE]
> **You will hold:** a rank-8 LoRA on SDXL's two text encoders, trained so that the product of the two
> single-prompt experts matches the joint prompt's prediction, with the UNet frozen throughout; its
> held-out cat-by-dog renders beside [plain PoE](../../context/world/poe-composition.md#what-poe-composition-is) and beside the existing rank-32 UNet adapter; a count
> of how many held-out seeds show two separate animals; and a fidelity number saying whether a cat
> still looks like a cat. **Time:** ~2 sittings, about 8 GPU-hours · **Stop point:** step 5's table

Walk it with:

```text
/drip-execute-plan plans/10-what-poe-loses-when-it-composes/today-a-lora-on-the-text-encoders.md
```

## Contents

- [Why this exists](#why-this-exists)
- [What is different from every adapter already trained here](#what-is-different-from-every-adapter-already-trained-here)
- [The test, written before any training](#the-test-written-before-any-training)
- [Where things land](#where-things-land)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What has to pass before this is done](#what-has-to-pass-before-this-is-done)
- [What it cannot claim](#what-it-cannot-claim)

## Why this exists

Navigation: 📋 [Contents](#contents) | [Next](#what-is-different-from-every-adapter-already-trained-here) ➡️

Eight third experts have been tried beside "dog" and "cat" and none restores identity. The reason is
structural: a third condition reaches the picture only as w·(ε_x − ε_∅), a whole-prompt-shaped pull
added to a sum, while the two animals are fused inside the "dog" and "cat" forward passes before that
sum is ever formed. The measured version of that argument is in
[the finding](../../report/can-a-text-space-third-expert-restore-composition.md): build x faithfully
as a difference and it sits near the empty prompt and does nothing (x_and, 0 of 7); build it far from
empty and it is simply another prompt (x_pos, 6 of 7 on count, 0 on identity).

So the correction has to reach inside those two passes. The repo already does that with a rank-32
LoRA on the UNet's cross-attention, which [composes on 7 of 8 held-out cat-by-dog seeds](../../report/does-training-longer-help-the-pooled-lora/which-checkpoint-composes-best-and-does-more-correction-help.md). **This plan
asks whether the same thing can be done one stage earlier and far more cheaply, by changing how the
two words are encoded rather than how the denoiser reads them.**

If it works, the claim is sharper than the UNet adapter's: single-concept embeddings are not
separable, and a small change to the encoder makes them so.

## What is different from every adapter already trained here

Navigation: ⬅️ [Previous](#why-this-exists) | 📋 [Contents](#contents) | [Next](#the-test-written-before-any-training) ➡️

| | Where the trainable weights sit | What it can change |
|---|---|---|
| the `<and>` token, four variants | one embedding row, or a small network writing it | the third expert only. All four null on the 10% rule |
| the token plus a rank-8 LoRA | the UNet's cross-attention keys and values, active only for the third expert | the third expert only. Null |
| the pooled rank-32 adapter | the UNet's cross-attention, active for every condition | ε_cat and ε_dog. **Works: 7 of 8** |
| **this plan** | **the two CLIP text encoders, active when encoding any prompt** | **ε_cat and ε_dog, one stage earlier** |

**The property that makes it worth doing.** The same encoder encodes both prompts and every future
pair, so it cannot memorise a cat-and-dog answer. Whatever it learns is a rule about how two concepts
should be encoded so their attention does not collide. Generalisation is forced by construction
rather than hoped for, which is what the `<and>` token's pair-written-rows variant could not promise.

## The test, written before any training

Navigation: ⬅️ [Previous](#what-is-different-from-every-adapter-already-trained-here) | 📋 [Contents](#contents) | [Next](#where-things-land) ➡️

**Training target.** The same one the pooled adapter uses: the joint prompt's prediction on the
curated cells of `artifacts/_shared/cross_pair_pool_configs/cells_v58.json`, 182 cells over 48 pairs,
cat by dog held out. No new curation.

**What is frozen.** The UNet entirely. Only the two text encoders carry trainable weights, rank 8 on
their attention projections.

**Evaluation**, on held-out cat by dog, seeds 9 to 16, which no training has seen:

| Measure | How | Why it is here |
|---|---|---|
| two separate animals | `count_instances` on each render | the co-occurrence question |
| one whole cat beside one whole dog | by eye at full resolution, user vetoes | the identity question the counter cannot answer |
| fidelity | the reward fine-tune's existing fidelity term, cat alone and dog alone under the adapter against under the frozen encoder | changing E("cat") can break what a cat looks like, and a layout win bought with a broken cat is not a win |

**Pass marks, set now:**

| Result | Reading |
|---|---|
| two separate animals on at least 6 of 8, and fidelity within 0.05 of the frozen encoder | **supported**: the encoder can be made separable, and cheaply |
| two separate animals on at least 6 of 8, fidelity worse than 0.05 | **traded, not solved**: it bought layout with identity, and the trade has to be reported as the result |
| two separate animals on 4 or fewer of 8 | **null**: the encoder cannot carry it and the correction belongs in the denoiser, where it already works |
| 5 of 8 | **inconclusive** at 8 seeds; rerun on the held-out pair set before calling it |

**What would surprise me.** The prediction is a partial result: some separation, bought with some
drift in what each animal looks like, because nothing in the objective says "keep the cat a cat"
until the fidelity term is switched on. A clean 8 of 8 with flat fidelity would say the two concepts
were separable in the encoder all along and only the training target was ever wrong.

## Where things land

Navigation: ⬅️ [Previous](#the-test-written-before-any-training) | 📋 [Contents](#contents) | [Next](#tasks) ➡️

```
poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py   the trainer this extends
  --lora-rank, --lora-alpha, --lora-targets                      exist; they attach to the UNet
  --text-encoder-lora                                            new: attach to the encoders instead
poe_repair/experiments/reward_finetune/                          the fidelity term to reuse
artifacts/_shared/cross_pair_pool_configs/cells_v58.json         the curated training cells
/datasets/mmolefe/poe_repair_min/outputs/text_encoder_lora/      checkpoints and renders, gitignored
artifacts/results/can-the-encoder-be-made-separable/             the sheet, the card, the numbers
W&B: prime_lab/poe-repair-animals-compose                        the three live curves
```

## Tasks

For Claude to execute. Ask Claude to do these.

### 1. 🔧 Attach the LoRA to the encoders instead of the UNet

- [ ] **1.1** Add a switch to `train_pooled.py` that attaches the LoRA to `text_encoder` and
  `text_encoder_2` and leaves the UNet frozen, using the same `LoraConfig` path
  `sample_crossbar.py` already uses for the UNet.
- [ ] **1.2** Assert at startup that no UNet parameter requires a gradient, and print the trainable
  parameter count for each encoder.
  - ✅ Two non-zero encoder counts and a UNet count of exactly 0.
  - ❌ Any trainable UNet parameter means the switch did not take; stop, because the run would then
    measure the adapter that already works.

▶ **Next: [task 2](#2--a-smoke-run-before-the-long-one)**.

### 2. 🔧 A smoke run before the long one

- [ ] **2.1** 300 updates on three cells, checkpoint, render one held-out seed.
  - Expected: a picture, not noise. This only proves the plumbing.
- [ ] **2.2** Check memory and step time, and compare against the pooled adapter's 0.7 s per step on
  an A6000, so the long run's walltime is a measurement rather than a guess.

◀ **Needs: [task 1.2](#1--attach-the-lora-to-the-encoders-instead-of-the-unet)**.
▶ **Next: [instruction 6](#6--read-the-smoke-render)**.

### 3. 🚀 The long run

- [ ] **3.1** Train to the pooled adapter's schedule on `cells_v58.json`, cat by dog held out, with
  the fidelity term on from the first update.
- [ ] **3.2** Log the three live curves to W&B so the run can be read while it is in flight.

◀ **Needs: [instruction 6](#6--read-the-smoke-render)**.
▶ **Next: [instruction 7](#7--watch-the-run)**.

### 4. 🔧 Render and score the held-out seeds

- [ ] **4.1** Render held-out cat by dog, seeds 9 to 16, under the adapter, beside plain PoE and
  beside the existing rank-32 UNet adapter on the same noise.
- [ ] **4.2** Count instances on every render, and compute the fidelity number on "a cat" and "a dog"
  alone under the adapter against under the frozen encoder.

◀ **Needs: [task 3.1](#3--the-long-run)**.
▶ **Next: [instruction 8](#8--call-the-renders-and-read-the-table)**.

### 5. 🔧 The card, the finding and the row

- [ ] **5.1** File the sheet and the numbers under
  `artifacts/results/can-the-encoder-be-made-separable/` with its card.
- [ ] **5.2** Write the finding against the pass marks above and add its row to the report index.

◀ **Needs: [instruction 8](#8--call-the-renders-and-read-the-table)**, so the verdict exists.
▶ **Next: the [what has to pass](#what-has-to-pass-before-this-is-done)**, which is where the plan ends.

## Instructions

For you to follow manually. Do these yourself.

### 6. 👀 Read the smoke render

◀ **Needs: [tasks 2.1 to 2.2](#2--a-smoke-run-before-the-long-one)**.

- [ ] **Open the one smoke tile.** You want a photograph of something, not grey and not noise. 300
  updates will not compose; this only says the gradient reaches the encoders and the renderer still
  works.
- [ ] **Say `go`** to spend the long run's GPU-hours, or `stop`.

▶ **Next: [task 3.1](#3--the-long-run)**.

### 7. 👀 Watch the run

◀ **Needs: [task 3.2](#3--the-long-run)**.

- [ ] **Open W&B**, project `prime_lab/poe-repair-animals-compose`, find the newest run by timestamp.
- [ ] **Watch the fidelity curve as much as the loss.** If fidelity falls while the loss falls, the
  adapter is buying layout with identity and the run should be stopped and restarted with the
  fidelity weight raised, rather than left to finish.

▶ **Next: [task 4.1](#4--render-and-score-the-held-out-seeds)**.

### 8. 👀 Call the renders and read the table

◀ **Needs: [tasks 4.1 to 4.2](#4--render-and-score-the-held-out-seeds)**.

- [ ] **Call each held-out tile at full resolution**: two whole animals, two animals at least one a
  hybrid, one hybrid, one plain animal, or no animal. Not from the sheet.
- [ ] **Read the three numbers against [the pass marks](#the-test-written-before-any-training)** and
  write one verdict line.

▶ **Next:** the verdict is the plan's answer; the walk closes on it.

## What has to pass before this is done

Navigation: ⬅️ [Previous](#instructions) | 📋 [Contents](#contents) | [Next](#what-it-cannot-claim) ➡️

Run these in order. Each prints its own answer; none needs a judgement call except the last.

**The UNet really is frozen.**

```bash
python -c "from poe_repair.run import make_ctx; c=make_ctx(); print(sum(p.requires_grad for p in c.models['unet'].parameters()))"
```

✅ `0`. ❌ Anything else means the switch in task 1.1 did not take and the run would measure the
adapter that already works.

**The three numbers exist and can be read off disk.**

```bash
python -c "import json; d=json.load(open('artifacts/results/can-the-encoder-be-made-separable/counts.json')); print(d['two_animals_of_8'], d['clear_pairs_of_8'], d['fidelity_delta'])"
```

✅ Three numbers. ❌ A missing file means task 4.2 did not run to completion.

**Task 2.2's measured step time and memory are written down, not just printed.** They belong in the
card at `artifacts/results/can-the-encoder-be-made-separable/README.md` under the run's entry, beside
the pooled adapter's 0.7 s per step on an A6000 that they are compared against.

**Instruction 8's verdict line is written into the finding**, not left in a chat message.

**The verdict matches [the pass marks](#the-test-written-before-any-training)**, read by you, not by
the count alone: a fidelity number inside the mark with animals that look wrong by eye is a miss.

## What it cannot claim

Navigation: ⬅️ [Previous](#what-has-to-pass-before-this-is-done) | 📋 [Contents](#contents)

**One held-out pair, eight seeds.** Cat by dog only. A result here says nothing about the 48-pair
pool until the same renders are made across it.

**It does not separate the two encoders.** CLIP ViT-L and OpenCLIP bigG are adapted together, so a
win does not say which one carried it. That is a follow-on, not this run.

**Nothing about the sampler.** Every render uses the project's own scheduler, step count and guidance
scale, unchanged, so any sampler-side contribution to the fusion is present here exactly as it is
everywhere else.

**A fidelity number is not a quality judgement.** It measures drift from the frozen encoder's render,
which a better picture would also show. The eye call in instruction 8 is what decides whether the
animals still look right.

## Cross-references

- The mention of **plan 10's text-encoder LoRA plan**, placed in the text-embedding project, in [the project table for regrouping the core repo](../split-into-idea-repos.md).
