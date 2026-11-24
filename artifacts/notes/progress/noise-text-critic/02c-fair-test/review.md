# Review: is W2's null fair, and is the setup readable at all? (step W2c)

**State: running.** No W2c score exists yet. The [design](design.md) holds the groups and the bars, all fixed before any W2c score.

## Runs

| Job | Stage | Code | Outcome |
|---|---|---|---|
| 94113 | smoke test: 24 pictures, steps 0 and 15, 8 trajectory runs | `be6a9e0f` | Ran in 81 seconds on mscluster63. Every row scored in both groups and the readout checker loaded. Its numbers come from 24 pictures and carry no verdict |
| 94114 | full run | `be6a9e0f` | First attempt: mscluster63 failed at 07:09 UTC, 65 minutes in, with group A finished and group B at step 3 of 21. The script writes its results only at the end, so nothing was kept. Slurm requeued the job, and it restarted on mscluster49 at about 07:31 UTC. At that hour 31 `bigbatch` nodes were down and 5 drained |
| 94131 | species-free counting rescore (group 4) | `5294ca7c` | Cancelled before it started, to be resubmitted after W2c |
