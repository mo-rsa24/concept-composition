# Progress: the two-paper route

**What this holds.** The live state of the [two-paper route](../../../plans/route-two-papers.md), for the three
looping sessions and the calendar agent.

| File | What it is |
|---|---|
| `board.json` | Every step, its state, outcome and proof, and the calendar blocks. Change it only through `scripts/progress_board.py` |
| `map.mmd`, `map.png` | The progress map, drawn from the board |
| `priority.mmd`, `priority.png` | Each open step by need against effort |
| `sessions.mmd`, `sessions.png` | One coloured lane per live session, with its steps left and ETA, redrawn at every board change |
| `timeline.png` | Each open step against the clock, with its median and longest ETA, redrawn at every board change |
| `scoreboard.json`, `scoreboard.png` | Each paper bar against the best result measured under its own conditions |
| `outcomes.png`, `waiting.png`, `burnup.png`, `history.jsonl` | What each step serves, what waits on the owner, and steps done against steps planned |
| Live page | All eight pictures at `http://127.0.0.1:8767/`, served from the `progress-live` tmux session |
| `prompts/` | The three loop prompts, their shared rules and the session closing prompts |
| `rival-datasets-first-pass.md` | What each rival benchmarks on, for step C3 to verify |

**How it was made.** The steps were agreed in the planning session `8959e932` on 2026-11-21.
