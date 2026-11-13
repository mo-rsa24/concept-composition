# Walk: why the second animal loses its species

plan: [the walk's plan](plan.md)
step: 1.1, launch command shown, waiting for a yes

## Stop point

bar: the plan's PASS line (level 3 on at least 6 more of 32 tiles than v58-03-self step 50,000 on
all 50 steps, at most 2 tiles dropping below level 2); judged by the user from labels.csv and sheets
time box: 2 sittings, about 3 hours
steps at the open: 9 tasks + 3 instructions (depth full)
toy: lion × meerkat, seed 9, windows 10, 25, 50

sitting 1: started 2026-11-10

## Decisions

| Decision | Options | Chosen | Who | Why |
|---|---|---|---|---|
| noise for seeds 9 to 16 | seed-number noise · drop seeds · read training cache | seed-number noise | user (defaults) | the renderer's cache has no cat × dog seed 9 to 16; seed-number noise is shared across every setting |
| where the walk's files live | artifacts/drips/second-animal-species/ · the results folder | artifacts/drips/second-animal-species/ | user (defaults) | nothing added under plans/ |
| node | mscluster111 co3_bw · mscluster112 co3_bw · mscluster108 co3 | mscluster112 | user | user wary of 111's fault history; both 111 and 112 probed healthy 2026-11-10 (cuda True, matmul 1e9) |
| plain-product row | keep · drop | keep | user (defaults) | reference for how often PoE alone gives two animals |
| curated training cells | v58's 182 · by-eye recurated | 161 cells, 48 pairs (cells_v58_bound.json) | user, by eye from curation sheets | keep only cells where the joint prompt draws both things, each its own species |
| first retrain | window 0-40 now · wait for controls | launch v58-06-bound-w40 on 107 card 0 after the sweep | user | baseline for ideas 2 and 3; brief's no-training rule lifted by the user |
| extra windows | add 15 and 35 · keep 10, 25, 50 | add 15 and 35 | user | the lion row shows the switch between 10 and 25 and between 25 and 50 |
| where piece 1 runs | inside interactive job on 107 · wait for 84113 · end 83423 | 107 card 0, nohup | user | biggpu allows one job per user and 83423 holds it; 84113 cancelled |

added: 1.4 windows 15 and 35 (you approved), step 1
added: 1.5 count adapter layers per checkpoint (repair), step 1

## Ticks

- 1.1 lion × meerkat 24 tiles, exit 0; user saw the row 2026-11-10
- 1.2 three pairs 72 tiles, exit 0 (17:42)
- 1.3 plain product 32 tiles, exit 0 (17:41)
- 1.5 layer counts: self 420/420, plain 0/420

## Round notes

- oracle (true correction steps 0-9, PoE after), 32 tiles, seeds 9-16: 0/1/2/3 = 2/14/16/0; adapter window 10 on the same cells = 1/13/18/0. The early correction carries two bodies, not two species, on these pairs. Labels in scratchpad sf/labels_oracle.csv until filed.
- owed when the window grid lands: answer "the LoRA gets co-occurrence but mixes the animals; how do we enforce binding?" across six levers the user named (sampling, attention, PoE, score addition and guidance scale, sampler, architecture), each with what the grid says about where binding is decided, cost, and the cheapest test on these 4 pairs.
- piece 4 launched 18:29: plain47k on card 1, self47k queued on card 0 behind the oracle; windows 10, 25, 50.

- lion × meerkat at sheet scale: window 50 turns the meerkat into a lion cub on 7 of 8 seeds; window 25 keeps a meerkat on most; window 10 loses the lion. Impression only, labels pending.

## Out of session

| id | kind | serves | state |
|---|---|---|---|
| v58-08-bound-w40-attnsep005, W&B fpz887kq (111 card 0, pid 40238, co3_bw, code from species-fidelity/code-idea3; --attn-sep-weight 0.05 --attn-sep-steps 0 20, else identical to v58-06; log cohort3/logs/v58-08-bound-w40-attnsep005.out). Weight picked from smoke run 2d56id4v in poe-repair-smoke: at 1.0 the term was ~70x the fit loss | training run | idea 3 | started ~03:50 2026-11-11 |
| conform_on_adapter.py on mscluster108 (shared-device path, no Slurm): card 0 push-apart pid 54628, card 1 no-fix control pid 54633; v58-03-self step 50,000, 4 pairs x seeds 9-16; out species-fidelity/conform/ | run | CONFORM-style push-apart across the two runs, training-free | started ~01:25 2026-11-11 |
| v58-07-bound-w40-inplace4, W&B z00vnnee (107 card 1, pid 99590, started 01:02 2026-11-11; code from worktree species-fidelity/code-idea2 at commit 2f4a6f24, --in-place-weight 4.0, else identical to v58-06) | training run | idea 2 | running |
| v58-06-bound-w40, W&B ot7qych4 (107 card 0, pid 97741, relaunched 00:16 after a W&B start failure) | training run | idea 1 | running |
| mscluster107 waiter 94340 → v58-06-bound-w40 (card 0, after sweep pid 78075; cells_v58_bound.json, steps 0-40, else v58-03-self recipe; log cohort3/logs/v58-06-bound-w40.out) | training run | idea 1, the window retrain | queued 2026-11-10 ~21:40 |
| mscluster107 waiter 93626 → live_split_controls.py (card 1, after attention maps) | run | damping vs ownership controls | queued |
| mscluster107 pid 64589 (nohup, card 0, inside interactive job 83423) | run | 1.1 lion × meerkat self50k w10/25/50 | started 2026-11-10 16:10 SAST |
