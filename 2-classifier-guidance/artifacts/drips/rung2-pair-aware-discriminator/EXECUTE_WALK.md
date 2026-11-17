# Walk: rung 2, a small pair-aware discriminator steers cat × dog

Walking [plan 21](../../../plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md) with
`/drip-execute-plan --request`. Branch `exec/classifier-guidance-ladder` (shared with plan 20's
walk, which runs in another session).

## Where the walk is

Round 1, the timing, is measured and ticked (1.1): plain 27.5 s, guided 42.2 s, N = 2, about
2.1 GPU-hours for the rung. Waiting on the reader's yes (2.1) and the prompt wording before
round 2.

## The stop point

- **Bar:** one of the seven readings in the rung 2 review file printed with its counts, judged by
  the reader from full-resolution tiles.
- **Time box:** 3 sittings, about 4.5 hours.
- **Steps at the open:** 5 pieces, 5 tests (depth full).
- **Sittings:** 1 started 2026-11-17 08:41 UTC.

## The toy

Seed 1, one tile per column.

## Decisions

| Decision | Chosen | Options | Who | Why |
|---|---|---|---|---|
| card | `mscluster107` GPU 0, `co3`; fallback `mscluster108` GPU 1 | share `mscluster111` with plan 20; wait for the Blackwells | reader took the default | plan 20 holds 111, 112 and 107 GPU 1 |
| discriminator body | plan 18's trunk, fresh one-logit head | small network from scratch | reader took the default | 10 to 15 positives |
| best-of-N pick | log-odds on the final latent at the last timestep | log-odds summed over steps | reader took the default | |
| positives | one cat and one dog, apart and intact | any two separate animals | reader took the default | |
| tile calls | the review file's three calls; route instruction 16 rewritten to match | instruction 16's four levels | reader took the default | |
| time box | 3 sittings, about 4.5 hours | own | reader took the default | |

## Ticks this walk

- 1.1 the timing · ✓ verified 2026-11-17: Quadro RTX 8000 on `mscluster107` GPU 0; plain 27.2 to 27.7 s, guided 42.1 to 42.3 s, N = 2 · `rung2/compute/timing.json`

## Repairs and changes

- added: plan 21 written from the open (you approved, step 0)
- route instruction 16 rewritten to the review file's three calls (you approved, step 0)

## Out of the session

| Id | Kind | Serves | State |
|---|---|---|---|
| PID 713135 | run, `mscluster107` GPU 0, `scripts/rung2/time_renders.py`, log `/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_ladder/rung2/logs/time_renders.log` | 1.1, the timing | done 09:27 UTC, folded into plan 21 |

## Closing pass owed

The board's rung 2 row says "four readings" where the review file holds seven; offered as
`fix board`, not yet taken.

## How to approach this next time

Nothing yet.
