# Review: does the experts' own vote place the digits, or would any held side do? (step M11)

This file holds what the runs in [the M11 design](design.md) produced. M11 has no verdict yet. Outputs are on the cluster under
`/datasets/mmolefe/poe_repair_min/loop-6/m11/`, and logs are in `/datasets/mmolefe/poe_repair_min/loop-6/logs/`.

## Runs

| Run | Job | Commits | Output | Outcome |
|---|---|---|---|---|
| Smoke: digits 4 and 7 at 8 seeds, every experiment, with the held-side control | 95225 | `084dfa7b-15d8bdd` | `m11/smoke/` | queued |
| Full: five tuning pairs at seeds 192 to 255, five experiments | 95226 | `084dfa7b-15d8bdd` | `m11/full/` | queued after the smoke |
