# Walk map: repair the classifier push in the SDXL product sampler

Plan: route step 5 in [the route](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md#5--repair-the-sdxl-product-samplers-push-and-rerun-the-canary) · mode `--request` · branch `exec/classifier-guidance-ladder`

## Stop point

- Bar: `canary.json`'s strength-0 md5s equal plain product's, cat × dog seed 9; judged by md5 equality.
- Time box: 1 sitting, about 90 minutes.
- steps at the open: 2 (depth full)
- Sitting 1: started 2026-11-17.

## Where the walk is

Closed. Both pieces ticked, the gate passed, the closing pass done. Retrospective: [the step 5 retro](../../notes/session-merges/2026-11-17-repair-the-classifier-push-retro.md).

## Ticks

- T1 · commit `445cef9f`, tick `3333b4fb` · cap shortens 0.871 of ε_PoE to 0.100 at cosine 1.000000, leaves 0.051 alone, a nonzero `weight` raises.
- T2 and the gate · job 89341 · plain, strength 0 and strength 0 with an untrained head share md5 `c291b259647d30a450e87b234423fe99`, 0 pixels differing.

## Changes to the plan

- added: T1 and T2 lines under step 5 (repair, step 1)
- fix: the canary runs as an sbatch job on an idle node; seed 9 has no cached latent, its noise is manual_seed(9) (you approved, step 2)

## Decisions

| Decision | Chosen | Options | Who | Why |
|---|---|---|---|---|
| the old rescaled branch | deleted, nonzero `weight` raises | keep beside strength · delete and move callers | you, by defaults | old callers fail loudly instead of drifting |
| κ | 0.1, `PUSH_CAP_FRACTION` | your own | you, by defaults | the paper gives none; step 9 sets the final value |
| what the cap measures | whole-latent norm, scale down | per-element clamp | you, by defaults | keeps the push's direction |
| plan file | route step 5 | a new plans/ file | you, by defaults | the repair's record is the commit |
| dirty route file | committed as its own commit `980c5ff4` | wait | you, by defaults | step 4's pictures, already closed |
| launch | sbatch on idle mscluster108 | nohup on a shared card | you, by fix | the launch recipe puts an idle node first |

## Out of the session

| id | what | serves | state |
|---|---|---|---|
| job 89341 | `sbatch --nodelist=mscluster108 presence_head_canary.sbatch 9` | T2 | completed, exit 0, folded in |

## Teaching lines for the close

- A cap that scales the whole vector keeps its direction; per-element clipping bends it.
- At strength 0 the classifier is never called, so the canary checks the wiring around the push.
