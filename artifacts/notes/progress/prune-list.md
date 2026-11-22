# 🧭 Progress: prune list

Artifact files that nothing in the repository names, so they can leave the repository once the owner approves. Nothing has moved. The [full list](prune-list.tsv) gives each file's path, size, reason, risk and proposed move to the laptop archive.

The search reads every page, card, plan, paper, script and data file for the file's name. It cannot see a name a script or scene builds at run time, so files under `scenes/`, `decks/` and `drips/` are marked `check first`.

## By project

| Project | Artifact files | Named by nothing | Size |
|---|---|---|---|
| 1-lora-correction | 9,425 | 4,019 | 2.06 GB |
| 2-classifier-guidance | 350 | 30 | 6.9 MB |
| 3-understanding-compositionality | 3,198 | 344 | 8.3 MB |
| 4-inclusive-captions | 648 | 233 | 26.1 MB |
| 5-text-embedding-third-expert | 85 | 14 | 6.3 MB |
| repository root | 3,771 | 1,954 | 679.8 MB |
| **All** | 17,477 | 6,594 | 2.79 GB |

## Largest folders

| Project | Folder | Size |
|---|---|---|
| 1-lora-correction | `results/the-adapter-on-cat-and-dog` | 2.03 GB |
| repository root | `drips/present-revision-to-supervisor` | 515.7 MB |
| repository root | `decks/supervisor-meeting` | 160.9 MB |
| 1-lora-correction | `scenes/where-each-condition-lands` | 19.9 MB |
| 1-lora-correction | `results/measuring-composition` | 14.8 MB |
| 4-inclusive-captions | `scenes/how-the-toy-relabelling-experiment-works` | 9.2 MB |
| 5-text-embedding-third-expert | `results/the-and-vector` | 6.3 MB |
| 4-inclusive-captions | `decks/supervisor-meeting` | 4.8 MB |
| 4-inclusive-captions | `scenes/how-the-inclusive-captions-pilot-works` | 4.7 MB |
| 4-inclusive-captions | `drips/inclusive-captions-mnist-stage` | 3.5 MB |
| 3-understanding-compositionality | `results/what-sdxl-already-knows` | 3.5 MB |
| 4-inclusive-captions | `scenes/how-the-inclusive-labels-mnist-stage-works` | 3.3 MB |

## By risk

| Risk | Size |
|---|---|
| low | 2.06 GB |
| check first: a built page may load it | 728.1 MB |
