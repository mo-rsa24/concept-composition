# 🎯 Today: does one prompt already say "one animal"?

<!-- template: exempt. Reason: a one-sitting walk settling two language-space claims, not a scope plan -->

> [!NOTE]
> **You will hold:** a one-vs-many direction built from 81 matched singular and plural prompts and
> checked on held-out nouns against a random-direction floor; where "a cat", "a dog", their implied
> sum and "a cat and a dog" land on it; the same read on PixArt's Flan-T5-XXL, which has no CLIP in
> it and fuses anyway; the share of the joint prompt's embedding lying outside the span of its own
> parts; and a report finding carrying a verdict for each of the two claims.
> **Time:** ~2.5 h over two sittings, minutes of it on a GPU · **Stop point:** step 5's finding

Walk it with:

```text
/drip-execute-plan plans/10-what-poe-loses-when-it-composes/today-does-one-prompt-already-say-one-animal.md
```

## Contents

- [Why this exists](#why-this-exists)
- [The two claims](#the-two-claims)
- [What would count, written before anything ran](#what-would-count-written-before-anything-ran)
- [What is already settled and is not re-run](#what-is-already-settled-and-is-not-re-run)
- [Environment facts this plan depends on](#environment-facts-this-plan-depends-on)
- [Where things land](#where-things-land)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What it cannot claim](#what-it-cannot-claim)

## Why this exists

Navigation: 📋 [Contents](#contents) | [Next](#the-two-claims) ➡️

A reader meeting the central result asks one question before any other: if "a cat" and "a dog" each
already say "exactly one animal is here", then summing their two guidance directions was never going
to draw two animals, and no dropped interaction term is needed to explain the fusion. The repository
has no answer to that. Every language-space measurement here so far asked a different question,
whether the prompt predicts how hard a pair is, and both came back null.

The second claim is the reverse direction, and it is the sentence the paper wants to be able to write:
the term the product drops is what the joint prompt's embedding supplies, so the failure is
composition in language space carrying something composition in score space does not. The arithmetic
for it exists in `scripts/and_vector_sheet.py` as `x_raw`. The argument does not, and parts of it are
already known to be wrong.

Neither claim needs a render. Both read text encoders only.

## The two claims

Navigation: ⬅️ [Previous](#why-this-exists) | 📋 [Contents](#contents) | [Next](#what-would-count-written-before-anything-ran) ➡️

**Claim one, load-bearing.** A single-prompt text embedding already encodes "one animal is present
here", strongly enough that summing two such predictions yields one animal.

This splits in two, and only the first half is measurable on embeddings. The premise is that the
embedding carries a readable singular signal. The inference is that summing therefore fuses, and that
runs through the UNet: product-of-experts composition sums noise predictions, never embeddings. So
the measurement can support the premise or kill it, and the inference stays an inference either way.
The plan says so rather than letting the two blur.

**Claim two, a nice-to-have.** `x_raw = E(joint) - E(a) - E(b)` is exactly the interaction term the
product drops.

The word "exactly" is already wrong. `x_raw` is dominated by the negated start token, norm 1190
against about 15 for a content word, because CLIP's causal mask makes position 0 identical in every
prompt. What has never been measured is the geometric fact underneath the claim: how much of the
joint prompt's embedding lies outside the span of its own two parts. That is a well-posed question on
the pooled view, where addition is a genuine vector-space operation and no token-alignment artefact
enters.

## What would count, written before anything ran

Navigation: ⬅️ [Previous](#the-two-claims) | 📋 [Contents](#contents) | [Next](#what-is-already-settled-and-is-not-re-run) ➡️

**The instrument has to pass its own check first.** The one-vs-many direction is built on half the
nouns and read on the other half. It is usable only if the held-out singular-to-plural separation is
at least 3x the separation a random direction of the same shape achieves on the same prompts. Below
that, the axis measures nothing and claim one is untestable by it, which is inconclusive rather than
null. `MIN_AXIS_OVER_FLOOR = 3.0` sits in the source, not here.

| Claim one | Reading |
|---|---|
| both solo prompts land on the singular side with a margin at least the mean held-out singular margin, and the joint prompt sits at least one such margin toward plural | supported: the prompt's own embedding distinguishes one animal from two, and a per-prompt singular prior is a live account of the fusion |
| the solos do not separate from the joint by more than the floor | null: nothing in the embedding tells a solo prompt from the joint on this axis, so the fusion cannot be laid at the prompt's door |
| the axis fails its held-out check, or the solos separate while the joint does not move | inconclusive |

| Claim two | Reading |
|---|---|
| the component of E(joint) orthogonal to span{E(a), E(b), E("")} exceeds 0.2 of ‖E(joint)‖ on the pooled view, for every pair in the pool | supported as a geometric statement: the joint carries a direction no sum of its parts can reach |
| that fraction is at or below 0.2 | null: the joint sits effectively inside the span, and there is nothing in language space the sum lacks |

`MIN_ORTHOGONAL_SHARE = 0.2` sits in the source for the same reason.

**The CLIP-free control decides what claim one is worth.** PixArt-alpha holds two clear animals on
0 of 8 seeds with Flan-T5-XXL and no CLIP in any role, per
[the CLIP-free finding](../../report/is-the-gap-the-samplers-or-the-models/does-a-clip-free-model-hybridise.md).
If the singular signal clears the floor in SDXL's encoders and misses it in Flan-T5-XXL, claim one is
removed as the account of the failure the same way CLIP was: the fusion happens where the signal is
absent. If it clears the floor in both, claim one survives and the paper has to own it.

## What is already settled and is not re-run

Navigation: ⬅️ [Previous](#what-would-count-written-before-anything-ran) | 📋 [Contents](#contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

Two nulls stand and are quoted, never re-measured. Re-running a failed bar in a new geometry after
seeing its answer is the move `~/.claude/EXPERIMENT_CONVENTIONS.md` exists to prevent.

- **The additivity gap does not predict how big a correction a pair needs.** Over 75 pairs the rank
  correlation against the pre-registered correction size is +0.29 pooled against a bar of 0.30, and
  one of four views squeaks past while the CLIP-L view sees nothing.
- **The leftover direction is not shared across pairs.**

Both in
[the three-sides review](../03-does-the-correction-cause-composition/review/06-the-same-story-from-three-sides.md).
Step 4 measures a different quantity, the orthogonal share, and makes no predictive claim.

## Environment facts this plan depends on

Navigation: ⬅️ [Previous](#what-is-already-settled-and-is-not-re-run) | 📋 [Contents](#contents) | [Next](#where-things-land) ➡️

- **No model weights and no embedding cache exist on the laptop.** `/datasets/mmolefe/poe_repair_min/`
  and `artifacts/caches/` are both absent here, so every step runs on the cluster through
  `scripts/cluster.sh`, which refuses to sync with uncommitted tracked changes.
- **Text encoders only, so no partition with a time limit is at risk.** Both steps are minutes on one
  GPU and need no `sbatch` header.
- **Cached and released tensors are fp16.** Anything that accumulates over many of them upcasts to
  fp32 first, which the axis mean and the span projection both do.
- **PixArt-alpha's weights are at** `/datasets/mmolefe/poe_repair_min/models/PixArt-XL-2-1024-MS`,
  with its `text_encoder` and `tokenizer` folders present, verified 2026-11-04.

## Where things land

Navigation: ⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [Contents](#contents) | [Next](#tasks) ➡️

| What | Path |
|---|---|
| the script | `scripts/one_animal_axis.py` |
| the axis and its held-out check | `artifacts/results/does-one-prompt-already-say-one-animal/axis-heldout-check.json` |
| where the four prompts land, as a figure | `artifacts/results/does-one-prompt-already-say-one-animal/where-the-prompts-land.png` |
| the Flan-T5-XXL read | `artifacts/results/does-one-prompt-already-say-one-animal/flan-t5-axis.json` |
| the orthogonal shares over the pool | `artifacts/results/does-one-prompt-already-say-one-animal/orthogonal-share.json` and `.png` |
| the card | `artifacts/results/does-one-prompt-already-say-one-animal/README.md` |
| the finding | `report/what-poe-loses-when-it-composes/does-one-prompt-already-say-one-animal.md` |

## Tasks

Navigation: ⬅️ [Previous](#where-things-land) | 📋 [Contents](#contents) | [Next](#instructions) ➡️

For Claude to execute. Ask Claude to do these.

### 1. 🔧 Build the one-vs-many axis and check it on held-out nouns

- [x] ✅ **1.1 Write `scripts/one_animal_axis.py`**, which encodes "a {noun}" and "two {plural}" for
      the 81 nouns of `artifacts/_shared/cross_pair_pool_configs/cells_v58.json` through SDXL's two
      text encoders, prints the generated plurals so they can be eyeballed, and reports all four
      views of the prompt the way `scripts/language_probes.py` does, with the pooled view as the
      primary read.
- [x] ✅ **1.2 Split the nouns in half, build the axis on one half, read it on the other**, and compare
      the held-out separation against the mean over 200 random directions of the same shape. Bar
      `MIN_AXIS_OVER_FLOOR = 3.0` in the source.
- [x] ✅ **1.3 Write `axis-heldout-check.json`** carrying the separation, the floor, their ratio, the
      generated plural list, and the per-view breakdown.
      ✓ verified 2026-11-04: pooled ratio 43.27 against the bar of 3.0, and all four views clear
      (seq_clipL 84.69, seq_bigG 118.93, seq_both 148.28). Ran from the committed blob
      `46026ce4`, md5 `6ca9a211753330726720a4dbd8776f9b`, on mscluster84.

▶ Next: instruction 6.

### 2. 📐 Read the four prompts off the axis

- [x] ✅ **2.1 Project E("a cat"), E("a dog"), the implied sum E(a) + E(b) - E("") and
      E("a cat and a dog")** onto the checked axis, in units of the mean held-out singular margin.
- [x] ✅ **2.2 Extend the read to every pair in the pool**, so the answer is not a cat-by-dog anecdote.
- [x] ✅ **2.3 Draw `where-the-prompts-land.png`**: x is the projection onto the one-vs-many axis in
      held-out-margin units, y is nothing (one row per prompt kind, jittered), one point per pool
      pair, with the held-out singular and plural bands shaded and the four cat-by-dog prompts
      labelled on the points themselves.

◀ Needs: task 1. ▶ Next: instruction 7.

### 3. 🧪 The same axis on Flan-T5-XXL

- [ ] **3.1 Add a `--encoder flan-t5` path** reading PixArt-alpha's `text_encoder` and `tokenizer`,
      and run steps 1 and 2 unchanged on it.
- [ ] **3.2 Write `flan-t5-axis.json`** with the held-out check, the floor ratio and the four
      projections, in the same shape as the SDXL file so the two can be read side by side.

◀ Needs: task 2. ▶ Next: instruction 8.

### 4. 📏 How far the joint sits outside the span of its parts

- [ ] **4.1 For every pool pair, compute the component of E(joint) orthogonal to
      span{E(a), E(b), E("")}** on the pooled view, and its share of ‖E(joint)‖. Bar
      `MIN_ORTHOGONAL_SHARE = 0.2` in the source.
- [ ] **4.2 Write `orthogonal-share.json` and its figure**: y is the orthogonal share on a 0-to-1
      scale, x is the pair, sorted, with the 0.2 bar drawn as a line and cat-by-dog labelled.

◀ Needs: task 1. ▶ Next: instruction 9.

### 5. ✍️ The argument, written

- [ ] **5.1 Write the finding** at
      `report/what-poe-loses-when-it-composes/does-one-prompt-already-say-one-animal.md` per
      `~/.claude/REPORT_FORMAT.md`: one verdict per claim against the criteria above, the figure and
      the sidecar file that back each, and the inference boundary stated for claim one.
- [ ] **5.2 File the results folder** with its card entry, and add the finding's row to
      `report/00-INDEX.md`.

◀ Needs: tasks 2, 3 and 4. ▶ Next: instruction 10.

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [Contents](#contents) | [Next](#what-it-cannot-claim) ➡️

For you to follow manually. Do these yourself.

### 6. 👀 Check the plurals and the floor ratio

- [x] ✅ **Read the generated plural list** printed by step 1.1.
  - `scripts/cluster.sh sh "python3 -m json.tool artifacts/results/does-one-prompt-already-say-one-animal/axis-heldout-check.json | head -120"`
  - What you want: 81 pairs, each plural the one you would say out loud. "sheep" stays "sheep",
    "wolf" becomes "wolves".
  - What failure looks like: "sheeps" or "wolfs" in the list. That is a wrong prompt, not a wrong
    axis, and it is `fix` on the irregular table rather than a verdict.
- [x] ✅ **Read the floor ratio.** Above 3.0 means the axis is usable and step 2's read means something.
    At or below 3.0 the verdict is inconclusive and tasks 2 to 4 still run, but claim one goes down
    as untested by this instrument.
  - ✓ read by Claude 2026-11-04, open to veto: all 13 plurals a bare `+s` would have got wrong are
    right, no article disagrees with its own rule, 81 pairs present. The pooled ratio is 43.24,
    which clears 3.0, so the axis is usable and step 2's read carries weight. An earlier run read
    43.27 on prompts carrying `an ukulele`; the article is now chosen by sound and the ratio moved
    by 0.03, so the slip changed nothing.

◀ Needs: task 1. ▶ Next: task 2.

### 7. 🖼️ Look at where the four prompts land

- [x] ✅ **Open `where-the-prompts-land.png`** and read the cat-by-dog points against the shaded bands.
  - ✓ read by Claude 2026-11-04, open to veto: four separated bands. The two solo rows sit inside
    or just right of the held-out singular band. The implied sum sits far left of every other row,
    around -0.95, further from plural than either solo. The joint row sits between the two bands at
    about +0.11, clear of the singular cloud and well short of the plural one. Cat by dog: a cat
    -0.40, a dog -0.38, the implied sum -0.66, the joint +0.05.
  - What you want: whichever the answer is, the four labelled points sit somewhere you can describe
    in a sentence.
  - What failure looks like: every point inside the singular band including the joint prompt, with no
    separation at all, which is the null and is a real answer.
- [ ] **Say which of the three readings the picture shows**, in your own words, before step 5 writes
      anything down.
  - ⚠️ Blocked on an ambiguity in the criterion itself, which is yours to settle and not Claude's.
    "The joint prompt sits at least one such margin toward plural" reads two ways. Read as an
    absolute position, the joint must reach the held-out plural position of +0.50, and it reaches
    +0.111, so claim one is **null**. Read as a displacement from the solos, the joint must move
    0.50 margin units and it moves +0.643, so claim one is **supported**. The number was not seen
    before the criterion was written; the wording was simply underspecified.

◀ Needs: task 2. ▶ Next: task 3.

### 8. 🔀 Compare the two encoder families side by side

- [ ] **Read `flan-t5-axis.json` beside `axis-heldout-check.json`** and say whether the singular
      signal clears the floor in one, both or neither.
  - What you want: a one-line call. Both clear means claim one survives; CLIP clears and Flan-T5 does
    not means claim one is removed as the account, since PixArt fuses on 0 of 8 anyway.
  - What failure looks like: the Flan-T5 axis failing its held-out check, which says nothing about
    claim one and only that the instrument does not transfer.

◀ Needs: task 3. ▶ Next: task 4.

### 9. 📊 Read the orthogonal shares

- [ ] **Open `orthogonal-share.png`** and check where the pairs sit against the 0.2 line.
  - What you want: a clear majority on one side, with cat-by-dog findable.
  - What failure looks like: shares at or near 1.0 for every pair, which usually means the empty
    prompt was left out of the span and the projection is wrong, not that the joint is orthogonal to
    everything.

◀ Needs: task 4. ▶ Next: task 5.

### 10. ✅ Approve the two verdicts

- [ ] **Read the finding** and check each verdict against the criteria written above, before it is
      indexed.
  - What you want: two verdicts, each naming the bar it was judged against, and claim one's
    inference boundary stated rather than skipped.
  - What failure looks like: a verdict stated more strongly than its bar allows, or claim two written
    as if the predictive question had been re-opened.

◀ Needs: task 5. The walk closes here.

## What it cannot claim

Navigation: ⬅️ [Previous](#instructions) | 📋 [Contents](#contents)

- **It cannot show that summing two predictions yields one animal.** That inference runs through the
  UNet and no embedding read reaches it. Claim one's premise is what is on test.
- **A null on claim one does not say the prompt is irrelevant.** It says this axis does not separate a
  solo prompt from the joint, and another instrument might.
- **The cross-family comparison is mixed.** SDXL and PixArt differ in encoder, denoiser, sampler, step
  count and guidance scale. Only the direction of the signal's absence licenses anything, exactly as
  in the CLIP-free finding.
- **The orthogonal share is a geometric fact and nothing more.** It says the joint carries a direction
  the sum of its parts cannot reach. It does not say that direction is the interaction term, and the
  two existing nulls are the reason the stronger wording is not available.
