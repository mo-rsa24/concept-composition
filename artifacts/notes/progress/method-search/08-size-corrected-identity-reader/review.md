# Review: does correcting for each animal's size let the per-place reader see borrowing? (step M12, second reader)

This file holds what the runs in [the second reader's design](design.md) produced. GPU outputs are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m12/size-corrected/`; the blind calls are on the laptop under
`outputs/loop-6/m12/size-corrected/blind/`, which git ignores.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Smoke: one fresh trajectory, puma and condor at seed 6009 | 95006 | `ef5e7c70-15d8bdd` | `size-corrected-smoke/` | passed in 2 minutes 33 seconds on mscluster46; the count checker read 2 animals on every step's clean estimate |
| Fresh CO3 read, 12 tuning pairs at seeds 6009 to 6016, states saved, then the per-place shares and per-step counts | 95007 | `ef5e7c70-15d8bdd` | `co3/`, `perplace/` | done in 1 hour 31 minutes on mscluster46: 96 trajectories, 96 finished pictures |
| Expected shares: each subject alone, plain SDXL, seeds 8001 to 8004 | 95008, then 95077, then 95081 | `ef5e7c70`, `84f3d05a`, `16cabcf2` | `expected/` | 95008 failed on an fp16 latent fed to the fp32 decoder, fixed by decoding with a separate fp32 copy; 95077 failed in 1 second on mscluster48, whose GPU is faulted and is now excluded; 95081 running |
| Blind calls on the 96 fresh finished pictures, two reviewers | laptop | none | `blind/calls-first.csv`, `-second.csv`, `-third.csv`, `calls-final.csv` | done: the two reviewers agreed on two clear or not for 84 of 96; a third blind call settled the other 12 (3 two clear, 9 not); 41 of 96 two clear in all. The pictures carry neutral ids, and the key (`blind/key.json`) is kept apart from them |
