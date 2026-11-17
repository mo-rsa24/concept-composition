# Walk: rung 2, a small pair-aware discriminator steers cat × dog

Walking [plan 21](../../../plans/21-rung2-a-small-discriminator-steers-cat-x-dog.md) with
`/drip-execute-plan --request`. Branch `exec/classifier-guidance-ladder` (shared with plan 20's
walk, which runs in another session).

## Where the walk is

Plan 22 builds its training set from about 1,000 local 1024 px pair pictures (the benchmark's and
the evaluations' own renders; no cat x dog on seeds 9 to 16). 543 were called twice, by two
independent passes of helper agents (91% agreement on role); Claude settled the 37 disagreements in
the 87 two-things pairs at full size. Final, two-things pairs only: 64 positives across 38 pairs
(11 cat x dog), 315 negatives (105 hybrid, 36 single, 85 wrong pair, 89 damaged), 23 left out.
`results/does-a-small-discriminator-steer-cat-x-dog/dataset-v3/calls-final.csv`. Next: encode on
the cluster and train both discriminators. `/datasets` still down; the cap-0.3 rerun waits for it.
Training started 16:40 UTC: v3 head-only (held-out check and final) on `mscluster107`'s two cards,
v3 head with adapters (held-out check and final) on `mscluster111` and `112`. Held out: 7 whole
pairs plus cat x dog seeds 7 and 8 (20 positives, 23 negatives).

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
| discriminator body | SDXL trunk frozen, attention-pool head only, log-odds push | plan 18's adapters plus head; small network from scratch | reader ("both") | 41 tiles would let 11 M adapter weights memorise the positives |
| discriminator check | held out on seeds 7 and 8, then a final run on all eight | fresh noise over the training tiles | reader ("both") | the old check measured fit, not reading new pictures |
| best-of-N pick | log-odds on the final latent at the last timestep | log-odds summed over steps | reader took the default | |
| positives | one cat and one dog, apart and intact | any two separate animals | reader took the default | |
| tile calls | the review file's three calls; route instruction 16 rewritten to match | instruction 16's four levels | reader took the default | |
| fewer than 10 positives | 16 extra draws, fresh noise under seeds 5 to 8 (generator 10,000 + 100 × seed + draw), the two pair wordings | re-render the same seeds (identical pictures); CO3 renders | reader ("agree") | a seed's cached noise repeats its picture; CO3 positives would teach CO3's look |
| calls on the extra tiles | Claude calls, reader can veto after | reader calls every tile | reader (asked for a /loop) | keeps the loop unattended |
| outputs while `/datasets` is down | v2 path in `/home-mscluster/mmolefe/poe_repair_min_scratch`, about 1 GB, moved to `/datasets` once it is back | wait for `/datasets` | reader ("go") | the outage length is unknown; the v2 retrain is the more informative follow-up |
| plan 22's data | wait for `/datasets` and use the benchmark's original pictures and verdicts | re-render the 48 pairs from seed-number noise and call them afresh | reader ("stop and wait") | re-rendering duplicates pictures already judged |
| time box | 3 sittings, about 4.5 hours | own | reader took the default | |

## Ticks this walk

- 1.2 the training set · ✓ verified 2026-11-17: 48 renders, latents saved · `rung2/training_set/`
- 1.1 the timing · ✓ verified 2026-11-17: Quadro RTX 8000 on `mscluster107` GPU 0; plain 27.2 to 27.7 s, guided 42.1 to 42.3 s, N = 2 · `rung2/compute/timing.json`

## Repairs and changes

- added: plan 21 written from the open (you approved, step 0)
- the product's wording set to "a cat" × "a dog" in the review file (you approved, step 1)
- route instruction 16 rewritten to the review file's three calls (you approved, step 0)

## Out of the session

| Id | Kind | Serves | State |
|---|---|---|---|
| PID 713135 | run, `mscluster107` GPU 0, `scripts/rung2/time_renders.py`, log `/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_ladder/rung2/logs/time_renders.log` | 1.1, the timing | done 09:27 UTC, folded into plan 21 |
| PID 715633 | run, `mscluster107` GPU 0, `scripts/rung2/render_training_set.py`, log `/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_ladder/rung2/logs/training_set.log` | 1.2, the training set | done 09:51 UTC |
| PID 739713 | run, `mscluster107` GPU 0, `train_discriminator.py` check run then final run, log `rung2/logs/discriminator.log` | 1.3, the discriminator | done 11:58 UTC |
| PID 754977 | run, `mscluster107` GPU 0, `render_results.py`, log `rung2/logs/results.log` | 1.4, the result sheet | done 12:41 UTC |
| PID 768102 | run, `mscluster107` GPU 1, `diagnose.py` (part B rerun with `--only-kinds`) | the diagnostics the reader asked for | done 12:50 UTC |
| PID 727709 | run, `mscluster107` GPU 0, `render_training_set.py --seeds 5 6 7 8 --extra-draws 2`, log `rung2/logs/extra_draws.log` | 1.2, the 16 extra draws | done, called |

## Closing pass owed

The board's rung 2 row says "four readings" where the review file holds seven; offered as
`fix board`, not yet taken.

## How to approach this next time

Nothing yet.
