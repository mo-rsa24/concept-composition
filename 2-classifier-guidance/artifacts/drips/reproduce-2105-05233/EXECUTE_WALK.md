# Walk map: reproduce arXiv 2105.05233 (route step 4)

**Plan walked:** [route step 4](../../../plans/route-classifier-guided-poe-from-toy-to-animals.md#4--reproduce-classifier-guidance-on-the-released-weights), mode `--reproduce`, branch `exec/classifier-guidance-ladder` (cut from `6f767cd1`).

**On step:** closed 2026-11-17; all four targets reproduced.

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
| relaunch T3, T4 | `nohup` on mscluster107 card 1 | wait for mscluster108 | Claude | another user took all of mscluster108; 106 and 109 down, 110 to 112 Blackwell |

## Ticks

- T1 reproduced 2026-11-16: sha256 equal on 4 of 4 seeds, plain-vs-plain control equal; job 89232; sheet `artifacts/results/does-classifier-guidance-work-on-the-released-weights/sheet-identity-scale-0-against-plain.png`.
- T2 reproduced 2026-11-16: 16 of 16 called Pembroke at scale 10, 2 of 16 at scale 1; by eye 13 of 16 clear corgis at scale 10; job 89235; sheet `sheet-pembroke-welsh-corgi-scale-1-and-10.png`. Seed 0's push: applied push norm 0.18 at the first step and 0.32 at step 50 against a mean norm of about 429, falling to 0 at the last step; at scale 10 the raw gradient falls from 0.52 (step 50) to 0.11 (step 200) as the classifier is satisfied, at scale 1 it rises from 0.54 to 1.85.

- T3 reproduced 2026-11-17: 15 of 16 called tabby at scale 10, 2 of 16 at scale 1; by eye 12 of 16; nohup run on mscluster107 card 1.
- T4 reproduced 2026-11-17: 15 of 16 called golden retriever at scale 10, 5 of 16 at scale 1; by eye 14 of 16; same run.

## Repairs

- The cluster clone's first attempt left an empty folder ("File exists" from git on the network filesystem); the retry into the empty folder worked.
- `--gres=gpu:1` is rejected by this cluster's sbatch; the job script carries no gres line.
- A folder a job creates on mscluster108 can read as missing from mscluster84 for a minute (network filesystem cache); a retried `cluster.sh pull` finds it.
- Parallel downloads were reset by the server near 800 MB, and the server answers a range request with 200, so `curl -C -` cannot resume. Downloads now run one at a time from zero, up to 8 attempts each, checked against the server's Content-Length.

## Out of the session

| id | kind | serves | state |
|---|---|---|---|
| weight downloads | `fetch.sh` (nohup) on mscluster84, log `/datasets/mmolefe/poe_repair_min/weights/guided-diffusion/fetch.log` | T1 to T4, step 8 | all four done, sizes matching the server |
| T3 and T4 run (done 22:32 UTC) | `ssh mscluster107` then `CUDA_VISIBLE_DEVICES=1 nohup bash .../guided_diffusion_reproduce.sbatch guide "281 207" "1.0 10.0"`, log `logs/gd_reproduce_t3t4_mscluster107.log` | T3, T4 | done; job 89245 cancelled after 30 min pending |
| the loop | `/loop` self-paced: settle each target, go next, closing pass | T1 to T4 | ended at the close |

## Closing pass

Done: the card with three sheets and `classifier_calls.csv`, board G0 done and redrawn at step 4, four route facts rows, the retrospective at `artifacts/notes/session-merges/2026-11-17-reproduce-2105-05233-retro.md`. Open for you: instruction 13's eye counts.
