# Idea map: plurality comes from the target distribution, not from the sampler

**Idea.** Two things in one picture (plurality) comes from the distribution being sampled. Product of experts plus a corrector plus classifier guidance only raises mass the product already has, so none of them can create a picture with one digit from each of two separately trained experts if the product has no mass there. Only a change of the target (training data, the inputs each expert sees, locality) can.

**Bar and time box.** A claim ledger in which every claim has a mark and a named next thing, and one runnable check for the load-bearing claim (5). This sitting.

**Position.** claim 1 of 5, round 1 on this claim.

| # | Claim | Mark | Settled by |
|---|---|---|---|
| 1 | A sampler, corrector or guidance push can only reweight or explore the target's existing mass; it cannot create mass where the product has none | open (first read: holds, with a threshold question) | |
| 2 | Whether two things appear together depends on the data the model was trained on, not on the sampler | open (first read: wrong as stated) | |
| 3 | Guidance on a product is only relevant when the product already has support, as with a single model that saw both | open (first read: holds, with a repair) | |
| 4 | Product of densities (AND over canvases) is not the same as both present (AND over contents); they coincide only when each expert is its own region times a shared background | open (first read: holds) | |
| 5 (load-bearing) | A target with support can be built at inference from separately trained experts without being told the partition of the canvas | open (first read: needs a check) | |

| 6 (appended) | Composition is decided in a short early window, the speciation window of the joint distribution, and the interaction term the product drops, a classifier push and a committed layout all act there and fail after it | open (first read: a hypothesis, three cheap tests named) | |
| 7 (appended) | E0e's layout is inferred from the experts' score gaps, and the inference is what helps, not only holding some layout | open (first read: untested; the missing control is a random held layout) | |

## Ground check

Plan 32: plain product 0 to 11 of 320, pushes 4. Step 7b: single model with 1% two-digit canvases, 285 of 320 by amplification. E0b masks 0. Full-canvas product with pushes: plan 32's experts 1, fragments experts 66. E0d-b 34, with pushes 56. E0e 259, with pushes 310 (other half occluded in each expert's input).

## Dead ends

E0e final read, seeds 128 to 191, 20 unseen pairs: 1,200 of 1,280 by the judge with pushes, 1,043 without. Held-layout with no hiding and random held layout were never run.

(no claim dead yet)

## Held claims

(none yet)
