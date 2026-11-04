# Walk: does a clip-free model hybridise

Plan: [14-does-a-clip-free-model-hybridise](../../../plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/14-does-a-clip-free-model-hybridise.md)
· review file: [14](../../../plans/06-is-the-gap-the-samplers-or-the-models/review/14-does-a-clip-free-model-hybridise.md)
· branch `exec/does-a-clip-free-model-hybridise` · mode `--request`, depth full

## Stop point

- **Bar.** The card at `artifacts/results/does-a-clip-free-model-hybridise/README.md` states how many
  of eight PixArt-alpha product renders hold two clear animals, in three categories, with the SDXL
  counts quoted beside them. Judged by the user from the sheet and the card.
- **Time box.** 2 sittings, about 2.5 hours.
- Sitting 1 opened 2026-11-04.

## The toy

Seed 1, prompt "a cat", 20 steps at guidance 4.5. The identity check and the first product render
both use it, so the two paths' numbers check each other.

## Position

On piece 3 of 5, the identity check. Pieces 1 and 2 are done.

| Piece | What it is | State |
|---|---|---|
| 1 | the composition script, `scripts/pixart_multi_poe.py` | ✅ compiles, formulas matched |
| 2 | the weights on disk and the tokenizer loading | ✅ 21 GB at `/datasets/mmolefe/poe_repair_min/models/PixArt-XL-2-1024-MS`, sentencepiece 0.2.2 |
| 3 | the single-prompt identity check | ⚠️ next, and the stop the request asked for |
| 4 | the 32 renders and the four-column sheet | ⚠️ |
| 5 | the calls by eye, `calls.json`, the card | ⚠️ |

## Ticked this session

- Plan task 1.1, the script. `python3 -m py_compile` clean, and the guidance line reads as the same
  arithmetic as `concept_multi_poe.py:66` and `sd35_and_vector_render.py:103`.
- Plan task 1.2, the weights. 21 GB on `/datasets`, fetched with a disk guard on that mount.

## Decisions

| Decision | Chosen | Options there were | Who | Why |
|---|---|---|---|---|
| fetch the weights | yes, `PixArt-alpha/PixArt-XL-2-1024-MS`, 21.8 GB to `/datasets` | yes · no, piece 1 only | user, `defaults` | the question needs the model |
| sentencepiece | pip into `co3` | pip into co3 · clone co3 first | user, `defaults` | no torch pin involved |
| precision | fp16 throughout, reference included | fp16 · fp32 text encoder | user, `defaults` | an fp32-against-fp16 comparison of one loop differs on its own |
| where the plan lives | scope 06 hypothesis, number 14 | hypothesis 05 as first proposed · a new scope | me, correcting the open | review files are numbered across the whole scope and 05 to 13 were taken |
| stop point | the card's counts, 2 sittings | the request's own done-when · your own | user, `defaults` | it is the request's done-when |
| the reference for the identity check | the released pipeline from the raw prompt string | from precomputed embeddings | me | the raw string covers tokenisation too |
| new material routed out | `/submerge` on T5 conditioning, the diffusion transformer and the multistep solver | teach it inside the round | user | asked for it as a route |

## What left the session

| Row | Kind | Serves | Id | State |
|---|---|---|---|---|
| 1 | background run, the weights fetch | piece 2 | `bozmy3vtv` | ✅ returned, exit 0 |
| 2 | route, `/submerge` on the three new things | the reader, not the plan | emitted, not yet run | ⚠️ out |

## How to approach this next time

- **Piece 1.** When you reimplement a sampling loop to change one thing, copy the released loop and
  change only the line you came for. Every pipeline carries conditioning the maths does not mention,
  and here that was three things: a learned sigma on half the output channels, an attention mask on
  the text sequence, and resolution conditioning the 1024 model wants.
- **Piece 2.** A download is the long pole, so it goes in the background before the code is written,
  with its disk guard reading the mount it resolves to rather than an assumed one.
