# The submerge prompt: rank, and what low-rank buys

Paste the block below into a fresh session. It is written to `~/.claude/PROMPT_FORMAT.md`: a
session that has never seen the conversation that produced it should do the right thing from it the
first time.

```text
/submerge

## Background

I am training LoRAs on SDXL in /home/molef/PhD/concept-composition and I cannot explain the name. I
know how to attach one and I know it trains, and that is the whole of it. Today I wrote a trainer
that puts a rank-32 adapter on SDXL's attention projections and it reported 46,448,640 trainable
parameters, and I could not say where that number comes from, what the 32 buys me, or what the
adapter cannot represent because of it. I want to fix that from the linear algebra up.

Four words, in the order they build on each other: rank, low-rank, adapter, parameter-efficient
fine-tuning. I want the first one as mathematics I can do on paper, the second as the consequence
that follows from it, the third as the thing that gets bolted onto a frozen network, and the
fourth as the library I already import.

I am a visual, hands-on learner. Worked examples with small numbers I can do by hand, geometry I
can look at, then the real matrix from my own run. Drip cadence, one piece per message, iPad open.

This belongs under the existing linear-algebra journey at
/home/molef/goal-setting/learning/mathematics/linear-algebra/ (master plan only, plans/ is empty),
as a promoted child journey. Settle that in Phase 0 rather than taking it from me, but that is my
expectation. The deep-learning end of it already has a neighbour:
deep-learning/diffusion-models/where-the-lora-touches-the-unet-corridor/ covers WHERE a LoRA
attaches inside SDXL's UNet. This journey is the other half, WHAT the low-rank constraint does to
what the adapter can express, and the two should cross-reference rather than overlap.

## Objectives

1. Settle the journey's three roots and register it in
   /home/molef/goal-setting/learning/JOURNEYS.md with its parent cell filled, per
   ~/.claude/JOURNEY_FORMAT.md.
2. Run the diagram gate, with the failure-first annotation on every node: what breaks without this
   idea. Do not start planning until the integration picture matches what I say I imagined.
3. Cover the maths strand at full weight, not as a garnish. Rank is the journey's floor and I want
   it derived, not asserted: the column space, why the row rank equals the column rank, and what a
   rank-deficient matrix loses. Then why truncating the SVD at r is the best rank-r approximation
   there is, which is the result that turns "low-rank" from a saving into a claim.
4. Take every name apart in the gate's "Names on this map" table, each read back in words. "Rank",
   "low-rank", "adaptation", "adapter", "parameter-efficient". A name I cannot read aloud as a
   sentence is a name I have not learnt.
5. Find the teaching material through the pulses rather than from memory, per the list in "What is
   already known" below.
6. Build the geometry so I can operate it, not read it: a scene where I drag the rank of a matrix
   down and watch the ellipsoid it maps the unit sphere to collapse, one axis at a time.
7. Land the journey on my own number: decompose the 46,448,640 trainable parameters my trainer
   reported into the per-module r(d + k) terms, and say what the full-rank update would have cost.
8. End with the capstone: the whole approved diagram walked once, plus a written account of what a
   rank-32 update to an attention projection can and cannot do, short enough to defend out loud.

## What to read

- /home/molef/goal-setting/learning/mathematics/linear-algebra/MASTER_PLAN.md: the parent journey's
  mission and what it already claims to cover, so this child does not restate it.
- /home/molef/goal-setting/learning/JOURNEYS.md: the ledger, and the row for
  where-the-lora-touches-the-unet-corridor, which names the neighbouring journey's 420 adapted
  modules and 35,799,040 trainable parameters on a different checkpoint.
- /home/molef/Obsidian/Mathematics/Linear Algebra/: the notes that already exist and must be
  extended rather than duplicated, per ~/.claude/VAULT_NOTE_FORMAT.md. Singular Value
  Decomposition.md, Projection.md, Matrix.md, Linear Map.md, Norm.md, Eigenvalue.md and
  Orthogonality.md are written. There is no Rank.md, and that is the gap this journey fills.
- /home/molef/Obsidian/Artificial Intelligence/Large Language Models/LoRA.md: what I have already
  written about LoRA, which is the note the adapter rungs extend.
- /home/molef/PhD/poe_repair_min-dog-and-rain/poe_repair/experiments/inclusive_captions/trainer.py:
  the trainer that produced the 46,448,640 figure. Its build_lora() names the four adapted
  projections and the rank, and the module list is what objective 7 decomposes.
- ~/.claude/VIDEO_CREATORS.md: the creators every video search is judged against first.

## What is already known

- A rank-32 LoRA on SDXL's to_q, to_k, to_v and to_out.0 projections, via peft 0.19.1, reports
  46,448,640 trainable parameters. Source: the trainer's own startup line, printed on 2026-11-04 on
  mscluster111. This is the journey's real anchor and the number objective 7 must reproduce by hand.
- The neighbouring corridor journey measured 420 adapted modules and 35,799,040 trainable
  parameters on a different checkpoint. Source: its JOURNEYS.md row. The two numbers differ and
  working out why is a legitimate rung, not a discrepancy to hide.
- Part of this already exists and must be linked rather than rebuilt. The plan at
  /home/molef/goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/plans/08-lora-update.md,
  "The low-rank update, B times A", already derives the parameter count r(in + out). The LoRA card
  at /home/molef/Obsidian/Artificial Intelligence/Large Language Models/LoRA.md already cites two
  textbook passages that work the saving with numbers (a 9 by 9 matrix of 81 parameters against a
  9 by 1 and a 1 by 9 factor of 18; a 10 by 10 rebuilt from 20 weights) and two lecture segments
  that work it at d = 1000, r = 10. So the parameter-count rung is covered: this journey reads that
  plan and those passages instead of writing a third version, and spends its own weight on what
  they do not cover, which is the rank itself, why the truncation is the best approximation, and
  what the constrained update cannot express.
- The term index has no card for "rank". Source: the lookup run on 2026-11-04. Minting it is the
  journey's first durable output.
- My linear algebra is uneven. I can do a matrix product, a dot product, a projection onto a unit
  vector and a norm by hand. I have met the SVD and can say what U, S and V are, but I have never
  used it to build a best rank-r approximation, and I have never worked out why the truncation is
  the best one. Treat Eckart-Young as unknown and derive it.
- The pulses are how the material gets found, and I want all of them run, not one:
  - /video-pulse over the journey once the plan tree exists, so every word that needs a video gets
    a timestamped segment from a real caption rather than a guessed timecode. Code Emporium covers
    LoRA, and I expect 3Blue1Brown for the geometry of a linear map and Umar Jamil for the code
    walk, but judge that against VIDEO_CREATORS.md rather than against my guess, and report any
    node no listed creator covers as a gap instead of filling it from the open web.
  - /video-scout for the uncovered ground the pulse reports, positioned by function: orient before
    the build, unblock during it, confirm after it, with confirm the default.
  - /textbook-cadence for the pages that teach rank, the fundamental subspaces, the SVD and the
    best low-rank approximation. The linear algebra and machine-learning textbooks already staged
    under /home/molef/Obsidian/Mathematics/references/textbooks/ are the first place to look, and
    the existing JOURNEYS.md rows show the page-cited shape to match.
  - /paper-cadence for the papers: LoRA itself (arXiv 2106.09685), and whatever the shelf already
    holds on low-rank structure in trained weights. File anything new on the shelf before citing it.
  - /concept-link once the plans are written, so every use of "rank", "low-rank", "adapter" and
    "projection" in them links to its card. Rank has no card yet, so minting it is part of the work.
  - /vault-pulse at the close, so the notes this journey writes are reachable from the subject trees
    and not only from the journey folder.
- Reusable material (rank, low-rank approximation, the adapter pattern) goes to the Obsidian subject
  trees per ~/.claude/VAULT_NOTE_FORMAT.md. Only what belongs to this journey alone goes in its
  mirror folder.

## When something is missing

Try the step first, then:

- A video with no real caption or chapter list: do not invent a timecode. Report the node as
  uncovered and carry on; a wrong timestamp is worse than a gap because I will click it once and
  stop trusting the table.
- A textbook page that teaches the idea in notation the journey has not fixed yet: stage it anyway
  and note the notation clash in NOTATION.md, which gets minted before any teaching is planned.
- My own LoRA module list not reachable, because the repo is a worktree another session is using:
  read the rank and the target list out of trainer.py's build_lora() and LORA_TARGETS instead, and
  mark objective 7's per-module decomposition as needing the live checkpoint. Ask me for it at the
  close rather than blocking on it.
- Anything about where the journey sits in the hierarchy that Phase 0 cannot settle from the ledger:
  ask me once, with your recommendation first.

## Rules

- Stop at the diagram gate and wait. I render the prompts and paste the images back; planning starts
  only when the integration picture is what I pictured. Ask the two art-direction questions first.
- Every maths node gets a worked example with numbers small enough to do on paper, and every worked
  example names the component of the gate diagram it translates, per "A worked example is a picture
  I can do" in ~/.claude/CLAUDE.md.
- The toy stays a toy. The 46,448,640 is the run back, worked after the small example, never instead
  of it.
- Simple notation first and the paper's notation last, mapped onto the simple one. Do not open on
  $\Delta W = BA$.
- Show me the hour estimate per region before I commit to anything, and let me trim against the
  numbers.
- No scene before its drip. A scene is consolidation, not introduction.
- Do not write into /home/molef/PhD/concept-composition or its worktrees. Read trainer.py and nothing
  else there.

## What to hand back

- The journey at /home/molef/goal-setting/learning/mathematics/linear-algebra/<journey>/: the gate
  file with the three roots and the illustration style, MASTER_PLAN.md, NOTATION.md, the staged
  sources, and the plan tree.
- The row in /home/molef/goal-setting/learning/JOURNEYS.md, parent cell filled.
- The journey scope's diagram-prompts.md, seeded from the approved gate set, with every piece
  assigned to the plan whose claim it covers.
- The pulse results: the video table with real timestamps, the textbook page citations, the papers
  filed on the shelf, the concept cards linked, and one list of every node no source covers.
- The proof line: /verify-plan --walk over the journey prints its check count with no structural
  failures, and the Todoist bridge is offered but not run without my yes.

## What you will see

Along the way: the outcome and purpose sentences for me to correct, then the two art-direction
questions, then the component diagrams and the integration picture, which I iterate on until it
matches. Then the hour estimates per region, for me to trim. Then the plan tree, with the first
calendar-ready task.

At the end, a closing report: what I asked for in one sentence; every file written with its path;
whether every one of the eight objectives landed, naming which did not; what is still open, with
each question you sent and the fallback you used; and at most one question for me.

I judge it from the integration picture and the plan tree: the picture has to match what I
imagined, every gate node has to name the failure it fixes, rank has to be derived rather than
asserted, and the 46,448,640 has to be decomposed into terms I could have written myself.
```
