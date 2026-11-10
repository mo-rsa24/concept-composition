# Walk map: route step 4, freeze the evaluation set, the tuning set and the one adapter

Plan: [route step 4](../../../plans/route-answer-the-iclr-reviews.md#4--choose-and-freeze-the-evaluation-set-the-tuning-set-and-the-one-checkpoint), walked with `/drip-execute-plan --request`.
Board: [the paper review board](../../../report/paper-review/board.md). Protocol: [the comparison protocol](../../../report/paper-review/protocol.md).

## Stop point

Bar: the frozen file committed with at least 8 pairs per group (or a smaller count you accept), the adapter named by the protocol's showcase rule, and no adapter result on any evaluation pair.
Time box: 60 minutes of your time; cluster renders run in the background and do not count.
Steps at the open: 5 (pieces 1 to 5, depth full).

| Sitting | Start | End |
|---|---|---|
| 1 | 2026-11-10 19:04 | 2026-11-10 21:13 |

Toy: bison and horse, seed 9.

## Position

Closed 2026-11-10 21:13: bar met, PASS; frozen set at `089abd6e`. Retrospective: [the retrospective](2026-11-10-frozen-set-and-adapter-retro.md).

## Ticks

| Piece | What | Evidence |
|---|---|---|
| 0 | steps 1 and 2 committed so the cluster checkout carries the protocol and the refill script | commit `084ad92e`, 47 files, no PDF, nothing under `figures/` or `build/`; the repo's only remote is GitHub |
| 1 | 62 candidates outside the 48 training pairs sorted by kind: 30 competing (3 held out), 20 object and place, 10 style, 2 showcase; tent and snowy mountain added to object and place from Figure 5 | pair_prompts*.yaml and pair_pool_v58.yaml read 2026-11-10 |
| 1 | Attend-and-Excite prompt sets fetched from the attachment the repository's README links: animals 66, animals_objects 144, objects 66; none names a training pair | `a.e_prompts.txt`, 10,146 bytes, in scratch |
| 2 | census rendered: 848 pictures (51 pairs plus the 2 showcase pairs, joint prompt and plain product, seeds 9 to 16), 0 errors | logs `census-a.log` (done 18:14 UTC) and `census-b.log` under `/datasets/mmolefe/poe_repair_min/outputs/review_step4/logs/` |
| 3 | animal pairs counted: 288 pictures, `census_counts.csv`; joint prompt composes on 6 to 8 of 8 seeds for all 18 animal pairs, plain product on 0 to 2 | `scripts/review_step4_count.py` on mscluster111, 0 missing |
| 3 | object pairs called by eye from full resolution, joint prompt, seeds composing of 8: bathtub × streetlamp 8, birdcage × watering can 8, briefcase × ceramic bowl 8, chessboard × lantern 8, drum set × snowman 8, feather pillow × cast iron pan 4 (10, 12, 14, 16), lab microscope × hay bale 8, microwave × potted plant 8, mug × wine glass 5 (11, 13 to 16), piano × candle 8, skateboard × helmet 8, suitcase × desk fan 3 (9, 11, 16) | Claude's calls, 2026-11-10; open to your overrule |
| 3 | species check by eye on the 18 animal pairs' joint pictures, seeds where both named species appear, of 8: baboon × gazelle 6, bear × salmon 8, bear × squirrel 8, bison × horse 6, cat × horse 7, cat × lion 7, cow × horse 2 (13, 15; the rest two cows, two horses or a cow-horse blend), deer × hedgehog 7, dog × horse 8, fox × wolf 4 (9, 10, 12, 14), giraffe × lion 7, lion × dog 7, lion × horse 4 (10, 11, 12, 15), peacock × sheep 7, raccoon × salmon 8, tiger × dog 4 (9, 10, 12, 13), wolf × horse 6, wolf × husky 8 | Claude's calls from the three sheets, 2026-11-10; open to your overrule |
| 3 | look-alike pairs' joint pictures by eye: seal × walrus 0 (no tusked walrus on any seed), goose × swan 6; eagle × hawk and leopard × jaguar show two birds or two spotted cats on every seed, but the species cannot be told apart by eye | Claude's calls, 2026-11-10; frog × toad not rendered, held by the subject guard's ask-first list |
| 4 | showcase calls by eye (sheets, then the contested seeds of candidates 3, 6 and 8 larger): seeds where cat and dog AND elephant and penguin both show two clear animals of the right species, of 8: v58-03-self 25k 3 (11, 13, 16), v58-03-self 27.5k 4 (10, 12, 13, 16), **v58-03-self 30k 5 (10, 11, 12, 13, 16)**, v58-03-self 50k 3 (11, 13, 16), v58-05-self-contrast 15k 3 (10, 13, 16), v57-11b-x0-self-decay 50k 3 (11, 12, 16), v58-01-plain 47.5k 3 (10, 15, 16), v58-02-contrast 27.5k 4 (10, 11, 15, 16), v58f-05-self-contrast 25k 3 (10, 13, 16). Seed 9 elephant and penguin fails for every candidate (a tall fused creature beside an elephant); seed 14 cat and dog fails for every candidate but the last (a boy with one animal) | Claude's calls, 2026-11-10; the cat and dog seed 10 call (a cat and a dog nuzzling, both faces clear, bodies overlapping) decides the shared seed |

## Facts found on the way

- `.gitignore` line 43 has `*.pdf` commented out, so PDFs are tracked unless a path rule excludes them; the export block added for `paper/iclr-56173/` excludes its PDFs. `environment/paper.md` says PDFs are gitignored at line 31, which is no longer true.
- The Attend-and-Excite prompt file is not in the repository; the README links it as a GitHub attachment.
- The animals set includes "a cat and a dog" (a showcase pair) and "a dog and a horse" (one of our competing candidates).

## Decisions

| Decision | Options | Who chose | Why |
|---|---|---|---|
| Commit steps 1 and 2 first | commit · launch without | you: commit only if it cannot reach Overleaf, then defaults | the repo's only remote is GitHub; the Overleaf clone is a separate repository nothing here touches |
| Style pairs | leave out · a fourth group | you, by `defaults` | they fit none of PRZE-3b's groups |
| Census scope | all 50 · 30 | you, by `defaults` | |
| Attend-and-Excite prompt sets | R3 only · also 66 animal prompts in the census | you: `a` | census stays at 51 pairs; file saved to `data/prompt-sets/attend-and-excite/a.e_prompts.json` |
| How the census is sorted (written before any count) | object pairs by eye · by per-name detector · left out | you: `a`, by eye | the validated instance count covers animals only. Rule: a competing pair is hard when its joint prompt composes on fewer than 4 of 8 seeds (animal pairs: 2 or more boxes from `count_instances`; object pairs: both objects present, called by eye from full resolution, you can overrule); easy but unseen otherwise; object-and-place pairs form the not-competing group by kind |
| Main table and ablation | R1 on the 55 census pairs · R1 on the 276 Attend-and-Excite prompts with the census as an R3 ablation, at 1 or 4 seeds | you: `yes 4, frog mouse yes` | the same prompts and metrics as CO3's Table 1; 4 seeds for precision; frog and mouse allowed for the 45 benchmark prompts only, recorded in the pair ledger |
| The one adapter | the showcase rule's winner, v58-03-self 30k at seed 10 · seed 11 if the seed-10 cat and dog call is overruled | you: agree, keep it accessible and easy to swap | the choice is `artifacts/_shared/cross_pair_pool_configs/adapter_of_record.yaml`, rewritten by `scripts/review_showcase_pick.py` from `report/paper-review/showcase-calls.csv` or an override |

## Runs, threads, handoffs

| Id | Serves | Command | State |
|---|---|---|---|
| census-a | piece 3 | `ROLE=census-a GPU=0 nohup bash scripts/review_step4_renders.sh` on mscluster111, PID 25435, commit `6a070f84`, log `/datasets/mmolefe/poe_repair_min/outputs/review_step4/logs/census-a.log` | running since 17:30 UTC, 480 tiles, 38 written by 17:37 |
| census-b | pieces 3 and 4 | `ROLE=census-b GPU=0 nohup bash scripts/review_step4_renders.sh` on mscluster112, PID 31439, commit `6a070f84`, log `.../logs/census-b.log` | running since 17:35 UTC, 368 census tiles then 144 showcase tiles |
