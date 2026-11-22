# 🌍 Context: concept-composition

What this project is about in the real world: why PoE composition on SDXL fails, what a chimera,
an animal pair, the interaction term, and a compose rate are, where the pipeline's data comes
from, and what every field in a `summary.json` or a scorer output means. What the system *is*
(the cluster, the two filesystems, the LaTeX build) lives in
the [environment index](../environment/00-INDEX.md); how to drive it lives in the
[runbook index](../runbook/00-INDEX.md). This folder is only the meaning.

If what you need is not in here, **stop and ask**. Do not infer a meaning from a column or symbol
name and do not proceed on a plausible assumption; a wrong guess about what `d_T` or `arm` means
gets built on by the next figure or claim. Say so and ask instead.

## Table of contents

- [I want to know...](#i-want-to-know)
- [The themes](#the-themes)
- [Still open](#still-open)
- [Pictures still missing](#pictures-still-missing)

## I want to know...

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-themes) ➡️

| The question | Where it is answered |
|---|---|
| What problem is this project actually solving, and for whom | [The problem](purpose/01-the-problem.md#who-has-the-problem) |
| What PoE composition is, and what Mono is | [PoE composition § What PoE composition is](world/poe-composition.md#what-poe-composition-is) |
| What a chimera looks like | [Chimera § What it looks like](world/chimera.md#what-it-looks-like) |
| What a failed render's look says about its cause (clipart, mush, sketch, invented people, wrong species) | [How a render fails § The looks](world/how-a-render-fails.md#the-looks) |
| What the interaction term (`r_t`) is | [Interaction term § What the interaction term is](world/interaction-term.md#what-the-interaction-term-is) |
| What the trained LoRA corrector does, and why it never sees the joint prompt | [LoRA corrector § What the LoRA corrector is](world/lora-corrector.md#what-the-lora-corrector-is) |
| What an animal pair is, and why the pool is biased toward failure | [Animal pair § What an animal pair is](world/animal-pair.md#what-an-animal-pair-is) |
| What a compose rate is, and what the scorer can and cannot tell you | [Compose rate § What a compose rate is](world/compose-rate.md#what-a-compose-rate-is) |
| How a prompt pair becomes a scored run | [Where it comes from § The journey](data/01-where-it-comes-from.md#the-journey) |
| What `pair_slug`, `seed`, `lambda`, `d_T`, or `n_instances` mean | [Dictionary](data/02-dictionary.md) |
| Who reads the output, and what they do with it | [What we produce § Who reads it](purpose/02-what-we-produce.md#who-reads-it) |
| How would we know this project is working | [What working means § The test](purpose/03-what-working-means.md#the-test) |

## The themes

Navigation: ⬅️ [I want to know...](#i-want-to-know) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| Folder | What it answers | Files |
|---|---|---|
| [purpose/](purpose) | Why this exists, what it produces, what working looks like | 3 |
| [world/](world) | The real things behind the data: PoE composition, chimera, the interaction term, the LoRA corrector, the animal pair, compose rate, how a render fails | 7 |
| [data/](data) | Where the data comes from, and what every field means | 2 |
| [plain-words.md](plain-words.md) | Which words this project replaced, which it kept, and why | 1 |
| [research-guidelines.md](research-guidelines.md) | Where the general research rules land in this repo, and the mistakes it has already made | 1 |

Picture prompts are in [diagram-prompts.md](diagram-prompts.md) (8 prompts, 0 rendered); rendered
images sit in `diagrams/`. Existing pipeline images already copied in sit in `images/world/`.
External documents this folder draws on are listed in [the sources register](sources.md) (5 entries).

## Still open

Navigation: ⬅️ [The themes](#the-themes) | 📋 [TOC](#table-of-contents) | [Next](#pictures-still-missing) ➡️

The things this pass could not settle by reading. Each names who or what would settle it.

- [ ] **The current animal-pair pool's size is not settled.** `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/fail_rate.md`
      lists 17 pairs; `1-lora-correction/report/experiments-log.md` also says 17 "in the current pool"; but
      `plans/standing/retrofit-poe-repair-min.md` (a later document) names "the 20 animal pairs" as the
      scope boundary. Would settle it: list the pair directories actually present under the
      current pool's output folder and recount. See
      [world/animal-pair.md § What people get wrong](world/animal-pair.md#what-people-get-wrong).
- [ ] **Whether `a_butterfly__x__a_flower_meadow` (named as the one pair that composes, and the
      pair the paper opens on) has a recorded fail-rate anywhere.** It does not appear in
      `1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/fail_rate.md`. Would settle it: find or run its scorer
      result. See [world/animal-pair.md § What an animal pair is](world/animal-pair.md#what-an-animal-pair-is).
- [ ] **Entry 1 in `sources.md`, the low-rank adaptation paper, has not been read past its
      title.** Would settle it: open the PDF, then log it in
      `plans/standing/literature/plans/01-reading-register.md`.
- [ ] **The interaction term's causal role is only backed, inside this repository, by this
      project's own paper draft.** The maintainer's working memory (not filed in the repo) refers
      to a seven-paper external literature reconciliation on this exact question, but
      `plans/standing/literature/plans/01-reading-register.md`'s own task list confirms the
      register is still empty. Would settle it: back-fill that register, then this file's citation
      of the paper's abstract can be joined by external corroboration.

## Pictures still missing

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

- [ ] 📷 [world/lora-corrector.md § What it looks like](world/lora-corrector.md#what-it-looks-like):
      one run's PoE render, LoRA-corrected render, and Mono render side by side. No search tried
      yet beyond noting a candidate source (`1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json`'s
      passing runs). Save as `images/world/05-lora-corrected-vs-poe-vs-mono.png`.
- [ ] 🖼️ All 8 pieces in [diagram-prompts.md](diagram-prompts.md) are written and unrendered: 4
      subject-lane (3 pieces plus a capstone) and 4 process-lane (3 pieces plus a capstone). Paste each into ChatGPT, save under `diagrams/`,
      and check against its faithfulness note before embedding, per that file's own instructions.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
