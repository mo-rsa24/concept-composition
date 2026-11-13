# 💡 Is product-of-experts ill-posed for compositional diffusion?

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. The product's failure is not the binding failure those repairs fix | **holds** | **dog and dog: nothing to bind, two identical experts, and the plain product still returns one animal on 5 of 5 cells. The chimera also sits in none of the four published buckets (missing entity, wrong attribute, wrong spatial relation, wrong count)** |
| 2. The joint prompt really does have an advantage over the product | **holds, on animal pairs only** | **cat x dog: joint prompt 8 of 8 composed, plain product 0 of 8. Not universal: the joint prompt drops a concept on 11 of 32 rendered held-out cells, and on the chess cell at seed 10 the plain product holds both objects where the joint prompt holds one** |
| 3a. The product fails only when the target needs two mutually exclusive identities in separate regions | **wrong as stated** | **dog and dog has no mutual exclusion at all and still returns one animal on 5 of 5. Repaired: it fails when the target needs more than one object instance, whether or not the two identities conflict. Composing cases on the dog sheet: dog x wet, dog x rain, dog x rain x wet, plus butterfly x meadow at 8 of 8** |
| 3b. ⭐ What `r_t` carries is a count-and-layout fact, not a relation between the two concepts | **wrong as stated** | **the spatial map: `r_t` concentrates on one centred animal face, not on two regions; CV rises 0.358 to 1.564 over steps 0 to 30 while `ε̃_PoE` stays at 0.36, and the 8-seed mean of \|r_t\| renders a face where the 8-seed mean of \|ε̃_PoE\| is noise. Repaired: `r_t` is a single-subject identity correction, concentrated on the contested object** |
| 3c. A count-and-layout fact is expressible in score space | **holds, partially** | **the adapter runs with no joint prompt and returns two animals on 4 of 5 dog-and-dog cells, so something in score space does carry the count. Partial because it reaches 19% out-of-span against the target's 37%** |
| 4. `r_t` can be supplied without ever forming a joint prompt | **holds, at generation time** | **the project's headline: a rank-8 adapter on `attn2.{to_q,to_k,to_v}`, trained on the cached correction, drives the product toward the joint prompt at inference with no joint prompt encoded (G04 in the decision timeline). The word *ever* is wrong: every correction it learned was copied from a joint prompt during training. The reach is a pair problem on unseen pairs, not a drift** |
| 5. No attention-map repair has anywhere to act inside the product** | **wrong as stated** | **each pass has one noun, so nothing contends inside a single map and the published losses have no pair to separate. But the two maps both exist, one per pass, so a repair can compare them across passes and penalise their spatial overlap. Repaired: no *within-pass* repair has anywhere to act; a cross-pass one does, and nobody has built it** |
| 6. One measurement separates a cross-attention advantage from an interaction-term advantage | **wrong as stated** | **no single measurement does it. Two do, and both are routed: the overlap between the two passes' attention maps (route 1), and what the unaccounted share of the correction is made of (route 2)** |

Load-bearing: claim 3b. It is the fork. If the joint prompt's advantage is wholly `r_t`, the missing
thing is a measurable quantity in score space and the formulation is incomplete. If part of the
advantage sits somewhere `r_t` does not reach, that part lives in contact between the two names
inside one attention sequence, which the product can never have, and the formulation is ill-posed.

Claims 3 and 4 together are the answer. Claim 3 asks whether the missing thing is expressible in
score space. Claim 4 asks whether it can be supplied without the joint prompt. Ill-posed needs a no
to either.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [The claims](#the-claims)
- [What the words are](#what-the-words-are)
- [The case that it is ill-posed](#the-case-that-it-is-ill-posed)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Routes](#routes)
- [Sources](#sources)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the question is**

Every published repair for composition failure in text-to-image diffusion reaches inside one
forward pass and edits the cross-attention maps, under one prompt naming both things. This
project's method never has one prompt naming both things: it runs two UNet passes under two
separate prompts and adds the results. So none of those repairs has anywhere to act here. The
question is whether that is a hole in the formulation or a property of it.

**Where it came from**

The literature round of the [inclusive-captions walk](../../../../4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md#routes),
which recorded the observation and split it off rather than absorbing it.

**The bar and the box**

Every claim carries a mark other than `open`, and the walk says whether the formulation is
ill-posed with a reason rather than a verdict. Sixty minutes. The proof is that a grep of this
file for a row still marked open prints nothing.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

The project's combine, at `poe_repair/training_cache.py:11`, is

```
ε̃_PoE = ε_uncond + gs · (ε_a + ε_b − 2 · ε_uncond)
```

Each of `ε_a` and `ε_b` comes from its own UNet pass under its own prompt. The two prompts are
never tokens in one attention sequence, so no query from one name can ever attend to a key from
the other. SDXL's two text encoders, CLIP ViT-L and OpenCLIP ViT-bigG concatenated, each see one
prompt at a time.

The repairs this question is about all need that contact. They read the cross-attention map for
each noun, notice two nouns competing for the same spatial region or one noun receiving almost no
mass, and push the maps apart or up. With two separate passes there is no competition to read:
each pass has one noun and that noun owns the whole map.

What the project measures instead is `r_t = ε̃_J − ε̃_PoE`, the difference between what a joint
prompt predicts and what the product predicts. Seven papers were reconciled on the point that
this is the interaction term the product drops.

The question is whether `r_t` is the whole of what the joint prompt has, or whether some of the
joint prompt's advantage lives in a place the score space cannot reach.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

Cut finer at the open, before anything was walked, so nothing is renumbered.

### 1. The product's failure is not the binding failure those repairs fix

**Mark:** holds. Dog and dog has nothing to bind and the plain product still returns one animal
on 5 of 5 cells. The chimera also sits in none of the four published buckets.

### 2. The joint prompt really does have an advantage over the product

**Mark:** holds, on animal pairs only. Cat x dog: joint 8 of 8, product 0 of 8. The joint prompt
drops a concept on 11 of 32 rendered held-out cells, so the advantage is not universal.

### 3a. The product fails only when the target needs two mutually exclusive identities in separate regions

**Mark:** wrong as stated. Dog and dog has no mutual exclusion and still fails. Repaired: it fails
when the target needs more than one object instance, conflicting identities or not.

### 3b. ⭐ What `r_t` carries is a count-and-layout fact

**Mark:** wrong as stated. Averaged over 8 seeds the correction's magnitude renders one centred
animal face, not two regions. Repaired: it is a single-subject identity correction, concentrated
on the contested object. See
[where the correction lives in the frame](../../results/where-the-correction-lives-in-the-frame/README.md).

### 3c. A count fact is expressible in score space

**Mark:** holds, partially. The adapter carries the count with no joint prompt, reaching 19%
out-of-span against the target's 37%.

### 4. `r_t` can be supplied without ever forming a joint prompt

**Mark:** holds at generation time, partially in substance. No joint prompt is encoded at
inference, but every correction was copied off one during training, and about half the
out-of-reach direction is in nothing that ships.

### 5. No attention-map repair has anywhere to act inside the product

**Mark:** wrong as stated. No *within-pass* repair has a handle. The two passes' maps both exist
in the same spatial frame, so a cross-pass one does.

### 6. One measurement separates a cross-attention advantage from an interaction-term advantage

**Mark:** wrong as stated. No single measurement does it. Two would: the overlap between the two
passes' attention maps, and what the unaccounted share of the correction is made of.

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#the-case-that-it-is-ill-posed) ➡️

## The case that it is ill-posed

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

## Held claims

Navigation: ⬅️ [The case](#the-case-that-it-is-ill-posed) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

None yet.

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

None yet.

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

None yet.

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

**How much co-occurrence the product needs** (2026-11-04, exact mixture-of-Gaussians toy, 300 draws
per cell). The exact product needs no measurable co-occurrence: 0 crisp pairs of 300 at zero, 300 of
300 at a pair-image weight of 0.001, which is the sweep's smallest non-zero point, while its two
samplers cross half of 300 only between 0.3 and 0.5. Deriving the two experts from competing captions
instead gives the same implied weight of 0.046 and a different failure: about 296 single objects of
300 from every sampler including the exact one, no overlay, because both experts then spread over the
same picture bank and the surviving components are the single shapes. Counts, figures and the caveats
are in [the co-occurrence card](../../results/how-much-co-occurrence-the-product-needs/README.md).

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

Returns land in `routes/` beside this file.

**A separation loss that works across the two passes.** From claim 5's repair. Each pass holds one
noun's attention map, both in the same spatial frame, so the overlap between them is computable
without a joint prompt. No published repair does this, because every one of them assumes both
nouns are in one sequence. Routed to a fresh `/drip-idea`.

**What the unaccounted part of the correction is.** From claim 4's break. 37% of the early
correction lies outside anything re-weighting the three predictions can produce, the adapter
supplies 19%, and the remaining share is the only live candidate for something the product can
never have. Routed to `/frame-hypothesis`.

## Sources

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- The project's combine, `poe_repair/training_cache.py:11`, and the exact-recovery note in the
  same docstring that makes claim 2 a cached read.
- [The findings front door](../../../../report/00-INDEX.md).
- [Does the joint prompt always draw both concepts](../../../../1-lora-correction/report/is-the-adapter-just-copying-the-joint-prompt/does-the-joint-prompt-always-draw-both-concepts.md),
  which is claim 4's evidence.
- [Does the adapter return two when both prompts name the same concept](../../../../1-lora-correction/report/is-the-adapter-just-copying-the-joint-prompt/does-the-adapter-return-two-when-both-prompts-name-the-same-concept.md),
  which is claim 1's evidence.
- [The decision timeline](../../../../1-lora-correction/report/decision-timeline.md), whose G04 entry records that the
  project's own adapter is a cross-attention LoRA on `attn2.{to_q,to_k,to_v}`.
- [The inclusive-captions walk's route](../../../../4-inclusive-captions/artifacts/ideas/inclusive-captions-axis/IDEA_MAP.md#routes), where this
  question was split off.
- SynGen 2306.08877, the concept-omission work 2605.14270, MultiCompose 2608.03708: the repairs
  that act on cross-attention maps.
- StructureDiffusion 2212.05032 and CoBind 2607.16307, from the opening scan.

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Walk claim 1.

## The wet-dog counterexample

A sum of scores does change the picture, and the dog sheet at
`supervisor-meeting/0-update/dog-sheet.jpg` proves it over eight seeds. Column 6 is the product of
"a dog" and "wet", and the wetness lands on the dog rather than in a separate region. Column 8 is
the product of "a dog", "wet" and "wet", and the dog gets wetter still. Column 7 adds "rain" as a
third expert and composes a wet dog in rain.

So three experts compose when the three constraints are compatible at the same pixels. The failure
is not that a sum cannot carry a modifier. It is that "a cat and a dog" asks for two mutually
exclusive identities in two places, and neither marginal contains a fact about how many objects
there are or where each goes.

The dog-and-dog cell is the same point from the other side: two identical compatible constraints,
both satisfied at one location, give one dog. That is the product behaving correctly. Wanting two
is the thing it cannot express.
