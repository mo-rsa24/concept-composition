# Table rows whose published number took the owner's choice

Step D2 fills one row of `table.csv` per rival per target. Each row needs the one published number for that rival under
the target's dataset, metric, protocol and task. Nine rows cannot get that number by the rule alone, and board
question 13 holds the owner's answer.

## Seven rows with no number under the target's protocol

These stay `not comparable`, and none of their candidate numbers is used. The candidates are the rival's rows on the same dataset in
`targets/numbers.csv`, and every one of them was printed under a different protocol from the target's.

| Rival | Target benchmark | Target's metric | Candidate row | Candidate's metric | Value | Where it is printed |
|---|---|---|---|---|---|---|
| Liu | CMNIST, AND | Conformity score (%) | `LogicalGuidance-T11-003` | CS (Conformity Score) | 26.61 | LogicalGuidance Table 11, p26 |
| SuperDiff | CMNIST, AND | Conformity score (%) | `LogicalGuidance-T11-004` | CS (Conformity Score) | 26.20 | LogicalGuidance Table 11, p26 |
| ToMe | T2I-CompBench Color | T2I-CompBench result | `ToMe-T1-052` | BLIP-VQA | 0.7656 | ToMe Table 1, p8 |
| Attend-and-Excite | T2I-CompBench Shape | BLIP-VQA | `ToMe-T4-036` | BLIP-VQA | 0.4517 | ToMe Table 4, p21 |
| Attend-and-Excite | T2I-CompBench Texture | BLIP-VQA | `ToMe-T4-035` | BLIP-VQA | 0.5963 | ToMe Table 4, p21 |
| Liu | T2I-CompBench Shape | BLIP-VQA | `ToMe-T4-030` | BLIP-VQA | 0.3299 | ToMe Table 4, p21 |
| Liu | T2I-CompBench Texture | BLIP-VQA | `ToMe-T4-029` | BLIP-VQA | 0.3645 | ToMe Table 4, p21 |

ToMe's Color row has five more rows beside the one listed: `ToMe-T1-055` is ImageReward 1.074, and `ToMe-T2-019`,
`ToMe-T3-034` and `ToMe-T4-052` repeat 0.7656, while `ToMe-T3-010` prints 0.7612. The last four Attend-and-Excite and
Liu rows are also printed in CoMat Table 1 and RPG Table 1 with the same values under those tables' protocols.

## Two rows where the rule finds a number on a different base model

The matching fields do not include the base model. On Shape and Texture the targets are ToMe on SDXL, and the one CoMat
row that matches is CoMat on SD1.5. ToMe prints CoMat on SDXL too, with a longer protocol string, because that value is
copied from CoMat's paper: ToMe's Appendix C.2 says CoMat's SDXL version is not open-sourced.

| Benchmark | Row the rule picks | Value | Base model | The SDXL row it passes over | Value |
|---|---|---|---|---|---|
| T2I-CompBench Shape | `ToMe-T1-035` | 0.4975 | SD1.5 | `ToMe-T1-051` | 0.5262 |
| T2I-CompBench Texture | `ToMe-T1-034` | 0.6190 | SD1.5 | `ToMe-T1-050` | 0.6591 |

The owner's session has put both CoMat rows on hold, because `CaraJ/CoMat_sdxl_ft_unet` holds SDXL weights that the answer to
question 13 assumed were unreleased. Nothing runs for them, including SD1.5 training, until the owner answers again.
