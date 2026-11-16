# Walk map: reproduce arXiv 2105.05233 (route step 4)

**Plan walked:** [route step 4](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md#4--reproduce-classifier-guidance-on-the-released-weights), mode `--reproduce`, branch `exec/classifier-guidance-ladder` (cut from `6f767cd1`).

**On step:** T1, scale 0 against plain sampling.

## Stop point

- **Bar:** T1 sha256 equal on 4 of 4 seeds; T2 to T4 at least 14 of 16 called the asked-for class at scale 10, fewer at scale 1 (the route's step 4 table).
- **Time box:** 2 sittings, about 4.5 hours.
- **Steps at the open:** 4 (depth full: T1 to T4).
- **Sittings:** 1 started 2026-11-16 21:56 SAST.

## Decisions

| Decision | Chosen | Options | Who | Why |
|---|---|---|---|---|
| a branch base | current commit `6f767cd1` | `main` | default you accepted | `main` lacks the route and 2-classifier-guidance |
| b uncommitted files | committed as `b9b463e4` | stash | default you accepted | `cluster.sh` refuses a dirty tree; route step 4 was among them |
| c mpi4py | no `dist_util` import in the wrapper | install into co3 | default you accepted | co3 is shared; sampling never needed `dist_util` |
| d cluster clone | `/home-mscluster/mmolefe/Playground/PhD/replications/guided-diffusion` | own path | default you accepted | folder exists beside the cluster checkout |
| e stop point | the bar above, 2 sittings | own | default you accepted | route's 270-minute budget |
| launch | `sbatch --nodelist=mscluster108` | `nohup` on a free card | Claude | no job held, so the one-job slot is free and Slurm sees the run |

## Ticks

(none yet)

## Repairs

- The cluster clone's first attempt left an empty folder ("File exists" from git on the network filesystem); the retry into the empty folder worked.
- Parallel downloads were reset by the server near 800 MB, and the server answers a range request with 200, so `curl -C -` cannot resume. Downloads now run one at a time from zero, up to 8 attempts each, checked against the server's Content-Length.

## Out of the session

| id | kind | serves | state |
|---|---|---|---|
| weight downloads | `fetch.sh` (nohup) on mscluster84, log `/datasets/mmolefe/poe_repair_min/weights/guided-diffusion/fetch.log` | T1 to T4, step 8 | both classifiers done; 256 px model then 64 px model, sequential, from 20:17 UTC |
| T1 submit | laptop loop: waits for the 256 px model, then `cluster.sh run "sbatch --nodelist=mscluster108 ... identity"` | T1 | waiting |
| the loop | `/loop` self-paced: settle each target, go next, closing pass | T1 to T4 | running, approved by you 2026-11-16 |

## Closing pass owed

Card, board G0 lines, route rows, retrospective.
