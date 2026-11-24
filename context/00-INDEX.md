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
| How the product of experts adds two models' noise predictions, and what the sum leaves out | [Product of experts § What one step does](methods/01-the-product-of-experts/01-overview.md#what-one-step-does) |
| How the readout checker pushes the sampler | [Readout checker § What one pushed step does](methods/02-the-readout-checker/01-overview.md#what-one-pushed-step-does) |
| How the digit-map checkers read a noisy digit canvas and push the digit product | [Digit-map checkers § What one step does](methods/03-the-digit-map-checkers/01-overview.md#what-one-step-does) |
| How a right and a wrong sentence become a push on the sampler | [Sentence-score push § What one step does](methods/04-the-sentence-score-push/01-overview.md#what-one-step-does) |
| What Latent CLIP changes when it reads the latent in place of NoisyCLIP | [Latent CLIP § What one step would do](methods/05-latent-clip/01-overview.md#what-one-step-would-do) |
| How the word-map checker reads where two subjects stand from their words' attention maps | [Word-map checker § What one step does](methods/06-the-word-map-checker/01-overview.md#what-one-step-does) |
| What a score path is, and how one canvas becomes one | [Score path § What one canvas goes through](methods/07-the-score-path/01-overview.md#what-one-canvas-goes-through) |
| What the BLIP-VQA score measures | [BLIP-VQA score § What one scoring pass does](methods/08-the-blip-vqa-score/01-overview.md#what-one-scoring-pass-does) |
| What the trained LoRA corrector does, and why it never sees the joint prompt | [LoRA corrector § What the LoRA corrector is](world/lora-corrector.md#what-the-lora-corrector-is) |
| What an animal pair is, and why the pool is biased toward failure | [Animal pair § What an animal pair is](world/animal-pair.md#what-an-animal-pair-is) |
| What a compose rate is, and what the scorer can and cannot tell you | [Compose rate § What a compose rate is](world/compose-rate.md#what-a-compose-rate-is) |
| How a prompt pair becomes a scored run | [Where it comes from § The journey](data/01-where-it-comes-from.md#the-journey) |
| What `pair_slug`, `seed`, `lambda`, `d_T`, or `n_instances` mean | Dictionary: [`pair_slug`](data/02-dictionary.md#pair_slug), [`seed`](data/02-dictionary.md#seed), [`lambda`](data/02-dictionary.md#lambda), [`d_T`](data/02-dictionary.md#d_t--d_t), [`n_instances`](data/02-dictionary.md#n_instances) |
| Who reads the output, and what they do with it | [What we produce § Who reads it](purpose/02-what-we-produce.md#who-reads-it) |
| How would we know this project is working | [What working means § The test](purpose/03-what-working-means.md#the-test) |

## The themes

Navigation: ⬅️ [I want to know...](#i-want-to-know) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| Folder | What it answers | Files |
|---|---|---|
| [purpose/](purpose) | Why this exists, what it produces, what working looks like | 3 |
| [world/](world) | The real things behind the data: PoE composition, chimera, the interaction term, the LoRA corrector, the animal pair, compose rate, how a render fails | 7 |
| [data/](data) | Where the data comes from, and what every field means | 2 |
| [methods/](methods/00-INDEX.md) | How each of the eight methods the loops use works, from the overview down to its parts: the product of experts, the readout checker, the digit-map checkers, the sentence-score push, Latent CLIP, the word-map checker, the score path and the BLIP-VQA score | 34 |
| [plain-words.md](plain-words.md) | Which words this project replaced, which it kept, and why | 1 |
| [research-guidelines.md](research-guidelines.md) | Where the general research rules land in this repo, and the mistakes it has already made | 1 |

Picture prompts are in [diagram-prompts.md](diagram-prompts.md) (8 prompts, 0 rendered); rendered
images sit in `diagrams/`. Existing pipeline images already copied in sit in `images/world/`.
External documents this folder draws on are listed in [the sources register](sources.md) (5 entries).

## Still open

Navigation: ⬅️ [The themes](#the-themes) | 📋 [TOC](#table-of-contents) | [Next](#pictures-still-missing) ➡️

`fail_rate.md` and `scorer_validated.json`, cited on this page, exist only in the cluster checkout (checked 2026-11-22), because their `results/` folders are git-ignored.

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
- [ ] **Whether the SDXL pipeline's `r_t` is the guidance scale times the digit scene's.** The
      pipeline computes `r_t` from guided predictions, so for the same model and step it should be
      the guidance scale (7.5 in the runs the world page quotes) times the unguided `r_t`. This
      follows from two quoted code lines and has not been checked on a run. Would settle it:
      compute `r_t` both ways at one cached SDXL step and compare the ratio with 7.5. See
      [Product of experts § The identity, by what moves](methods/01-the-product-of-experts/01-overview.md#the-identity-by-what-moves).
- [ ] **Whether CO3 steps through the same 50 timesteps as SDXL's own sampler.** The readout
      checker's overview places CO3's steps 0 and 15 at timesteps 981 and 681 because CO3 loads a
      DDIM scheduler from SDXL's scheduler folder (`Co3.py`, line 85 of the CO3 clone). Would
      settle it: print the scheduler's timesteps after CO3 sets 50 steps. See
      [Readout checker § The step, by what moves](methods/02-the-readout-checker/01-overview.md#the-step-by-what-moves).
- [ ] **Which batch settings the kept readout checker's training run used.** The trainer has four
      switches that change the batch, and no source read for the page records which the kept run
      set, so the page describes the trainer's default of steps drawn evenly over all 50. Would
      settle it: the kept run's saved config or its W&B run config. See
      [Inside the kept checker § What the head learned from](methods/02-the-readout-checker/02-inside-the-kept-checker.md#what-the-head-learned-from).
- [ ] **How large the clean part of Latent CLIP's pooled input is at steps 0 to 15.** The page's
      range, a ninth to nine tenths of the noise part, assumes the 2 × 2 average pool leaves the
      clean latent's spread about unchanged, and nobody measured it. Would settle it: the standard
      deviation of a few cached clean SDXL latents before and after the pool. See
      [Reading a latent with no VAE § How far the pushed steps sit from its training](methods/05-latent-clip/02-reading-a-latent-with-no-vae.md#how-far-the-pushed-steps-sit-from-its-training).
- [ ] **Whether the word-map checker can tell two animals from one fused body.** It reads only the
      two words' attention maps and its head has no class for "fused", so a fused body whose two
      maps sit apart may read as two animals. The ladder states this as untested, and the reading
      note infers that a head for the product would need a fused class. Would settle it: run a
      trained head on fused product renders, once a head exists on SDXL. See
      [Word-map checker § How it fails and how you know it worked](methods/06-the-word-map-checker/01-overview.md#how-it-fails-and-how-you-know-it-worked)
      and [its miniature § Stage 5: three ways it misleads](methods/06-the-word-map-checker/04-the-miniature.md#stage-5-three-ways-it-misleads).
- [ ] **Whether BLIP-VQA's per-question answers still exist on the cluster.** The scoring script
      writes each question's answers to `annotation<i>_blip/VQA/result/vqa_result.json` in its work
      folder (`BLIP_vqa.py`, line 98), and nobody has checked whether those folders survive. Four
      inferred claims on the BLIP-VQA pages wait on them: that the colour swap is why CO3's render
      of "a yellow bow and a pink bowl" scored 0.1899, which object the fused mug picture's "yes"
      came from, whether a fused object passes the score in general, and which object answered "a
      green bowl?" on the bench tile. Would settle it: list the work folders on the cluster and
      read one `vqa_result.json`. See
      [How the question reads the picture § How it fails and how you know it worked](methods/08-the-blip-vqa-score/02-how-the-question-reads-the-picture.md#how-it-fails-and-how-you-know-it-worked).
- [ ] **Whether ids 2748 and 2053 are "yes" and "no" in BERT's uncased word list.** The
      yes-probability is read from these two ids, and the page takes them from `vqa_prob` without
      checking the word list. Would settle it: the page's own one-line tokenizer check, run on a
      machine with the scorer's environment. See the same section.
- [ ] **How many pictures two rows of the BLIP-VQA target table rest on.** The 2,330 pictures for
      CoMat's two-subject Color prompts and the 200 for the adapter on T2I-CompBench colour are
      computed from prompt counts and seeds, not counted. Would settle it: count the scored
      pictures in each run's output or the rows of its score file. See
      [From answers to a target § From pictures to a category's number](methods/08-the-blip-vqa-score/03-from-answers-to-a-target.md#from-pictures-to-a-categorys-number).

## Pictures still missing

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

- [ ] 📷 [world/lora-corrector.md § What it looks like](world/lora-corrector.md#what-it-looks-like):
      one run's PoE render, LoRA-corrected render, and Mono render side by side. No search tried
      yet beyond noting a candidate source (`1-lora-correction/artifacts/results/measuring-composition/the-scorers/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json`'s
      passing runs). Save as `images/world/05-lora-corrected-vs-poe-vs-mono.png`.
- [ ] 🖼️ All 8 pieces in [diagram-prompts.md](diagram-prompts.md) are written and unrendered: 4
      subject-lane (3 pieces plus a capstone) and 4 process-lane (3 pieces plus a capstone). Paste each into ChatGPT, save under `diagrams/`,
      and check against its faithfulness note before embedding, per that file's own instructions.
- [ ] 🖼️ [Readout checker § The names, taken apart](methods/02-the-readout-checker/01-overview.md#the-names-taken-apart):
      balanced accuracy on the 20-picture toy as two overlapping circles, the pictures called clean
      by eye and by the checker, beside a panel for a checker that calls everything a reject. The
      slot names no save path yet.
- [ ] 🖼️ [The linear probe § What the probes read](methods/05-latent-clip/04-the-linear-probe-on-frozen-vectors.md#what-the-probes-read):
      balanced accuracy against step 0 to 15, one panel per property and one line per reader. The
      linear probe finding owns the slot and its data path; render it there, then embed it here.
- [ ] 🖼️ [Word-map checker's network view § One joint prompt against two prompts](methods/06-the-word-map-checker/02-the-network-view.md#one-joint-prompt-against-two-prompts):
      two words' brightest patches as overlapping circles, with the overlap over the union written
      on them, from the section's toy counts. The slot names no save path yet.
- [ ] 🖼️ [From answers to a target § From a printed number to a target](methods/08-the-blip-vqa-score/03-from-answers-to-a-target.md#from-a-printed-number-to-a-target):
      the coffee toy's resampled mean gaps as two histograms, matched and unmatched, each with its
      band of ±1.96 standard errors. The slot names no save path yet.

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
