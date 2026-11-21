# Route walk: bind each body to its own concept

Builds the route for making classifier-guided PoE general one decision per round, with
`/drip-skill-route --research`. Opened 2026-11-19.

## Position

Round 1 settled. Round 2 (the outcomes as board rows) proposed: seven rows, N0, N1a, N1b, N2, N3a,
N3b and X2, drafted outside the board until settled.

## Round 1: the sentence and the disk

**The goal.** Classifier-guided PoE that works on pairs it never saw: the readout checker's push
already makes two bodies, and the method should also decide which body is which concept, so lion ×
meerkat stops ending as two meerkats. Stated in the sentence.

**The objective.** Three binding methods (regional binding, a cross-attention binding term, the two
MNIST checker changes), tried cheapest first, each with pass marks written before it renders, each
judged on cat × dog seeds 6001 to 6128, lion × meerkat seeds 3401 to 3432 and typewriter × cactus
seeds 3301 to 3316. A method is kept only if it beats the kept recipe on lion × meerkat without
lowering cat × dog. Stated; the numbers for "beats" and "without lowering" are inferred and asked.

**The outcome.** A verdict per method on lion × meerkat seeds 3401 to 3432, as PASS or MISS against
marks written before its render, with its sheets, eye-call CSVs, plan and review file, and new rows
in the classifier-guided route's "What we know so far". Stated.

**The kind of work.** Test your own idea, on the `--research` spine. Layers 1 to 5 (scout, read,
reproduce, rebuild) are done on disk for this question: the Readout Guidance read and reproduction
are steps 10f to 10h of the classifier-guided route, and the MNIST rebuild is step 7b. Layer 6 (own
ideas, small) is the MNIST loop. This route is layers 7 to 9: pass marks per method, the tests, and
an explanation at the close.

### What the disk holds

| Path | What it is | Bearing on the route |
|---|---|---|
| `artifacts/drips/rung2-pair-aware-discriminator/checker-loop.md` | every push round, read end to end; "Loop closed" on 2026-11-19 | the kept recipe and its counts; the next idea is regional binding from about step 10 |
| `artifacts/ideas/beyond-the-product-of-experts/where-we-are-stuck.md` | every line of work on one page | three levers restore two bodies, none restores identity; the regional sampler has never run on SDXL |
| `poe_repair/methods/_poe_regional.py` | `run_cfg_poe_regional` | masks fixed from step 0, no push, no mid-run switch: the masks have to go inside `run_poe_classifier_guided` (`poe_repair/methods/_classifier_guided.py:64`) |
| `poe_repair/composers/_regional_masks.py` | boxes to two soft masks summing to 1, two overlap readings | reusable as is for detected boxes |
| `3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/is-the-layout-the-missing-piece.md` | the oracle-layout design: joint-render boxes, random boxes, swapped prompts, same prompt in both, 8 seeds | never rendered; it is the upper bound for regional binding |
| `scripts/rung2/attention_push.py`, commit `ff028dd5` | a push that lowers the correlation of the cat expert's "cat" map and the dog expert's "dog" map at 32 × 32 | the overlap half of the cross-attention term already exists, and it stores maps through a custom attention processor |
| `scripts/rung2/attention_overlap.py` | measured "cat" and "dog" map overlap | top-10% overlap 0.53 to 0.81 in the product against about 0.25 in joint renders that composed (its docstring) |
| project memory, `discriminator-guidance-set-aside` | the attention push alone, drawn 2026-11-18 02:52 | null: 0 of 8 held-out seeds with two animals; strength 1,000 zooms into one face. Recorded only in session `1c41950d`'s transcript, in no plan file |
| `scripts/rung2/train_readout_pair.py:61` | `PairHead`, the readout checker's head | the pair enters only through SDXL's features read with the joint prompt; the head's own layers never see it |
| `plans/review/a2b-mnist-digits-on-a-canvas.md`, loop 2 | MNIST: the pair fed to every head layer took the tuning pairs from 39 to 81 of 192; two retraining passes on its own push 77 then 120; with eta 1.0, 217 of 320 held-out | eta 1.0 already failed on SDXL (`result-sheet-catdog-eta1`), so it stays out |
| cluster, `rung2/joint-catdog-eval` | joint prompt on cat × dog 6001 to 6128 | 128 of 128 drawn, not scored |
| cluster, `rung2/co3-catdog-eval/a_cat__x__a_dog` | CO3 on the same seeds | 64 drawn, none written in the last 20 minutes, no process found on mscluster111 or 112; not scored |
| `report/classifier-guided-poe/board.md` | the classifier-guided board, 23 rows | the board these method rows would join (round 2) |
| mscluster112 | Blackwell card | free (2 MiB used); mscluster111 holds another user's 58 GB job |

### Where the sentence and the disk disagree

1. The cross-attention method is described as new. Its separation half was drawn on 2026-11-18 as a
   push on its own and was null. Riding on the readout push, which already makes two bodies, is a
   different test, so the method stands; its review file has to cite that null.
2. The regional sampler cannot take the push or masks that switch on mid-run, so the "When something
   is missing" branch applies from the start: the masks go inside `run_poe_classifier_guided`.
3. "Beats the kept recipe on lion × meerkat" has no number, and the kept recipe has no render on
   lion × meerkat 3401 to 3432 or on typewriter × cactus 3301 to 3316. Those reference rows are drawn
   once, shared by all three methods.
4. The joint prompt landed and is unscored; CO3 may have stopped at 64 of 128.

## Decisions

| Round | What | Chosen | Options there were | Who chose | Why |
|---|---|---|---|---|---|
| 1 | the keep rule | scorer faithful on lion × meerkat 3401 to 3432 at least 4 above the kept recipe's count on the same seeds, and cat × dog faithful at least 33 of 128; eye counts beside, deciding only a tie | eye +4 with scorer beside; either count clearing +4 | user | recommended option taken |
| 1 | regional binding's first piece | the oracle-layout sheet on the 8 cat × dog seeds whose boxes and masks exist, beside random boxes and the same prompt in both; if the right boxes do not give a clean pair, method 1 stops there | straight to regions detected mid-push | user | recommended option taken |
| 1 | time | no date; each method closes when its sheets are judged; the schedule counts sittings and GPU-hours | this week; given windows | user | recommended option taken |
| 1 | depth | ship it and explain it, 1.5×: each review file says how the method works in plain words, and the close walks the three verdicts | rebuild it, 2×; master it with the maths, 3× | user | recommended option taken |
