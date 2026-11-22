# 🌍 The animal pair

An animal pair is the unit every experiment in the current scope is built from: two animal
prompts (for example "a cat" and "a dog"), tested together across several seeds and render
methods. Whether a pair "fails" (blends into a [chimera](chimera.md)) or "composes" is judged per
seed by the scorer in [compose-rate.md](compose-rate.md).

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [What an animal pair is](#what-an-animal-pair-is)
- [What it looks like](#what-it-looks-like)
- [Why the project cares](#why-the-project-cares)
- [How it shows up in the data](#how-it-shows-up-in-the-data)
- [What people get wrong](#what-people-get-wrong)
- [Where this came from](#where-this-came-from)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)
- [Cross-references](#cross-references)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-an-animal-pair-is) ➡️

- **Pair slug**: the on-disk and in-code name for a pair, always the form `a_cat__x__a_dog`
  (article, animal, double underscore, `x`, double underscore, article, animal). Kept in this
  exact form everywhere on disk even though prose says "a cat and a dog"; changing the slug form
  was costed and rejected (`plans/standing/retrofit-poe-repair-min.md`: "costs 3,049 files and 456
  directories and buys nothing a card does not already give").
- **Seed**: a re-roll of the starting noise only; the pair of concepts stays the same. Contrast
  with changing the pair itself, which is the harder generalisation test.
- **Group**: an older, six-way taxonomy (co-occurrence, factorisation, object-plus-scene,
  dual-object, entanglement, concept collision) that classified pairs by *why* composition is
  hard, not just by species. See [What people get wrong](#what-people-get-wrong) for its current
  status.
- **Run**: one pair-and-seed point, the smallest unit a figure or a claim is built from.

## What an animal pair is

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-it-looks-like) ➡️

**Two animal names, joined into one prompt pair, chosen so that PoE composition reliably fails on
them.** ✅

`1-lora-correction/report/experiments-log.md`'s selection warning states this plainly: pairs entered the current pool "by
failing" under plain PoE (8 seeds each, instance-count scorer). 15 of 17 pairs in
`1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/fail_rate.md` fail on all 8 seeds; the two least-reliable
failures are `a_donkey__x__a_pony` (6/8) and `a_crocodile__x__an_alligator` (5/8). ✅

**The pool is deliberately biased toward failure, which the project's own documents flag as a
scoping limit, not an oversight.** ✅

`1-lora-correction/report/experiments-log.md`: "No analysis inside this pool can say what predicts PoE failure, because the
pool has no successes to contrast against. Any such analysis is conditioned on the outcome." A
follow-on experiment (EXP-02, status pending as of this build) is designed to build a second pool
spanning both successes and failures. ✅

**One pair is a deliberate exception: it composes.** ✅

`plans/standing/retrofit-poe-repair-min.md`'s "Words this uses": `a_butterfly__x__a_flower_meadow` "is the
deliberate exception: it is the control that composes and the paper opens on it." ✍️ Its fail rate is not established, because `fail_rate.md` does not list
it. See [Still open](../00-INDEX.md#still-open).

## What it looks like

Navigation: ⬅️ [What an animal pair is](#what-an-animal-pair-is) | 📋 [TOC](#table-of-contents) | [Next](#why-the-project-cares) ➡️

The PoE and Mono renders in [poe-composition.md](poe-composition.md#what-it-looks-like) are the
`a_cat__x__a_dog` pair, seed 42: the project's own reference pair, present in almost every walk of
this repository.

## Why the project cares

Navigation: ⬅️ [What it looks like](#what-it-looks-like) | 📋 [TOC](#table-of-contents) | [Next](#how-it-shows-up-in-the-data) ➡️

**The pair is the axis every transfer claim is measured across.** ✅

Whether a LoRA trained on one pair (or one group of pairs) composes on a pair it never trained on
is Objective 3 and Objective 4 of `MASTER_PLAN.md`'s five-step ladder, and is the harder of the two
generalisation axes (pair vs seed) per `MASTER_PLAN.md`'s Glossary: "a new pair is the harder
test."

## How it shows up in the data

Navigation: ⬅️ [Why the project cares](#why-the-project-cares) | 📋 [TOC](#table-of-contents) | [Next](#what-people-get-wrong) ➡️

| Column | Stands for | Example | Entry |
|---|---|---|---|
| `pair_slug` | which two animals this run is about | `a_cat__x__a_dog` (example) | [Dictionary § pair_slug](../data/02-dictionary.md#pair_slug) |
| `seed` | which noise re-roll produced this run | `42` (example) | [Dictionary § seed](../data/02-dictionary.md#seed) |
| `fail_rate` | how often plain PoE fails this pair, over 8 seeds | `1.00` (example, 8 of 8) | [Dictionary § fail_rate](../data/02-dictionary.md#fail_rate--compose_rate) |
| `group` | the six-way taxonomy label some pairs still carry | `group6_coherent_collision` (example) | [Dictionary § group](../data/02-dictionary.md#group--group_label) |

## What people get wrong

Navigation: ⬅️ [How it shows up in the data](#how-it-shows-up-in-the-data) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**The pair count is not settled across the repo's own documents.** ⚠️

`1-lora-correction/report/experiments-log.md` (its axes table) says "17 in the current pool, plus a new spread set built in
EXP-02" (EXP-02 status: pending). `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/fail_rate.md` lists exactly 17
pairs (1 dissimilar control, 1 reference, 15 train). `plans/standing/retrofit-poe-repair-min.md`'s "Words
this uses" instead names "the 20 animal pairs" as the current scope boundary, plus 7 non-animal
pairs from an earlier era that stay on disk but are out of the live claim. This file cannot tell
which count is current without checking which pairs, beyond the 17, exist under the pool's
directory today; listed under [Still open](../00-INDEX.md#still-open).

**A pair from the older six-group taxonomy is not automatically part of the current pool.** ✍️

`artifacts/plans/archived/phases/09-lora-taxonomy-single-seed.md` (a shelved plan) describes representative
pairs for Groups 1-5 (dolphin×ocean-wave, dog×oil-painting-style, mailbox×snowfield,
typewriter×cactus, and a deferred, unresolved Group 5) that are not animal-vs-animal pairs and sit
outside the current animal-only scope, per the retrofit plan's animal-pair boundary. Group 6
(`a_cat__x__a_dog`) is the one taxonomy group that overlaps the live pool.

## Where this came from

Navigation: ⬅️ [What people get wrong](#what-people-get-wrong) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

| What | How it was established | When |
|---|---|---|
| The pool is biased toward failure by construction | Read in `1-lora-correction/report/experiments-log.md`, "The selection warning" | 2026-08-24 |
| The 17-row fail-rate table | Read in `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/fail_rate.md` | 2026-08-24 |
| The butterfly control pair and the animal-pair boundary | Read in `plans/standing/retrofit-poe-repair-min.md`, "Words this uses" | 2026-08-24 |
| The six-group taxonomy and its representative pairs | Read in `artifacts/plans/archived/phases/09-lora-taxonomy-single-seed.md` | 2026-08-24 |
| Pair vs seed, which is the harder test | Read in `MASTER_PLAN.md`, Glossary | 2026-08-24 |

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#cross-references) ➡️

A pair is two animal names, and the pictures below show what plain PoE draws for several of them and how each pair is scored over 8 seeds.

### How it fits together

*Picture held back.* `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/hard-vs-easy-transfer/hard_vs_easy_transfer.png` holds a frog and toad row, which the [pair ledger](../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off every sheet, so it is named here and not shown.

Six pairs, one row each (frog and toad, eagle and hawk, seal and walrus, leopard and jaguar, wolf and husky, lion and tiger), with three columns: the joint prompt (Mono, the target), plain PoE and the LoRA. The Mono tile shows two animals in every row, and the PoE tile shows one animal in the foreground in every row `verified` (opened and read by eye). The card calls the PoE column one blended creature in all six and says each row is one seed per pair, the first cached one, at training step 60,000 `stated` (the folder's query file). The card also notes that in the seal and walrus row the second LoRA animal has no tusks, so a pair can score as two animals while one of them is the wrong species `stated`.

### The code behind it

The fail rate per pair is the share of the 8 seeds where plain PoE is scored `blend`. Excerpt from `poe_repair/experiments/does_the_fix_reach_unseen_pairs/fail_rate.py` (lines 56 to 64):

```python
        for s in SEEDS:
            cell = cell_for(p["prompt_a"], p["prompt_b"], s)
            png = run_method("poe", cell, ctx)  # idempotent; fresh seeded init
            sc = score_output_instances(Path(png), p["prompt_a"], p["prompt_b"], device=device)
            is_blend = (sc.label == "blend")
            n_blend += is_blend
            per_seed.append({"seed": s, "png": str(png), "n_instances": sc.n_instances,
                             "label": sc.label})
        fail_rate = n_blend / len(SEEDS)
```

- `SEEDS` is `list(range(1, 9))` on line 34 of the same file, so 8 seeds per pair, as the page says.
- `run_method("poe", ...)` renders plain PoE with no corrector, so the rate describes the uncorrected method.
- The role of each pair (train, reference, control) is set on lines 49 and 65: three slugs are listed as controls there (`an_elephant__x__a_penguin`, `a_giraffe__x__a_crab`, `an_octopus__x__a_sparrow`), and the pair the page calls the reference is `a_cat__x__a_dog`.

### Further illustrations

**Evidence**

*Picture held back.* `1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/contact-sheet.png` holds a frog and toad row, which the [pair ledger](../../artifacts/_shared/cross_pair_pool_configs/pair_ledger.md) keeps off every sheet, so it is named here and not shown.

Eight pairs by four seeds (9 to 12), each tile labelled with the detector's count `n` `verified` (read from the labels). These are renders at full correction strength, not plain PoE: the card says they are every cell behind the 94% headline at λ=1 `stated` (the folder's card). Pairs in the sheet: cat and dog, cow and buffalo, frog and toad, goose and swan, leopard and jaguar, seal and walrus, eagle and hawk, elephant and penguin `verified`.

### Go deeper

| What | Link |
|---|---|
| The scorer behind each fail rate | [compose rate](compose-rate.md) |
| What a failed pair looks like | [chimera](chimera.md) |
| The plain PoE render of the reference pair | [PoE composition](poe-composition.md#what-it-looks-like) |
| The hard against easy pairs card | [query and method for the six-pair figure](../../1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/hard-vs-easy-transfer/QUERY.md) |
| The audit of the 32 corrected cells | do the successful cells contain both animals, at `1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md` (not linked, because the folder holds the held-back pair) |
| The pairs results group | [the pairs](../../1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-pairs/README.md) |

### Still open

- The page cites `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/fail_rate.md`. That file is not at that path now, the folder holds `hard-vs-easy-transfer` and `pooled_lora` only, and `RENAMES.md` has no entry for it. The script above writes `fail_rate.md` beside the pair pool, so the 17-row table cannot be checked from here.
- The line above in the code names three controls, while the page describes the table as holding one dissimilar control. Both are left as they are.
- Pictures of the control pair `a_butterfly__x__a_flower_meadow` exist and none is placed in this block. One is `corrector-butterfly-meadow-eight-seed-sheet.png` in `1-lora-correction/artifacts/results/is-it-the-sampler-or-the-model/is-the-gap-the-samplers-or-the-models/`, with the joint prompt and plain PoE on seeds 9 to 16 `verified`. The published-models grid card says that grid has a row for this pair, and the grid shows the cat and dog row only (see the note in [PoE composition](poe-composition.md)).
- `pair_list.csv` in the pairs folder has no card, so the meaning of its Joint and Product columns is not stated and it is not quoted here.
<!-- related:end -->

## Cross-references

Navigation: ⬅️ [Previous](#in-pictures-code-and-further-reading) | 📋 [TOC](#table-of-contents)

- The mention of **the pair that composes by default** (a butterfly and a flower meadow, used as the control a corrector must not break) in [does a corrector alone produce two animals](../../1-lora-correction/report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md) (relevance match).
- The mention of **animal pairs** (the 11 look-alike pairs behind `00`'s five runs, read as history rather than as a comparison) in [designing the correction loss](../../1-lora-correction/plans/09-designing-the-correction-loss/MASTER_PLAN.md) (exact match).
- The mention of **animal pairs** (the 11 look-alike pairs of the original 88-run training pool, whose adapters never draw a person into the picture) in [the known failure "a pooled adapter draws a person into a picture of two animals"](../../environment/known-failures.md#entry-id-poe-lora-005) (exact match).
- The mention of **animal pairs** (the 15 pairs each held out in turn, one LoRA trained without each) in [the plan measuring transfer as a rate over fifteen held-out pairs](../../1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/plans/hypothesis/04-transfer-as-a-rate-over-fifteen-pairs.md#quick-context-where-you-are) (exact match).
- The mention of **seed** (same pair and same seed across every corrector count) in [the whole corrector design](../../1-lora-correction/plans/06-is-the-gap-the-samplers-or-the-models/source/the-whole-corrector-design.md#the-claim).

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
