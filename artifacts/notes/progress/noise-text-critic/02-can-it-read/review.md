# Review: can a sentence score read the five properties? (step W2)

**State: no score has been computed.** The labelled pictures are still being built, so no property has a verdict. The [design](design.md) holds the sentences and bars.

## Runs

| Job | Stage | Code | Outcome |
|---|---|---|---|
| 93753 | label smoke test, 4 pictures | `7ab2981c` | Failed in 18 seconds: the code snapshot left out `eval_set_catdog_v2.json` |
| 93795 | label smoke test, 4 pictures | `bec5297b` | Ran in 3 minutes. Left-right labels were right on 4 of 4 by eye. Every box covered the whole 32 by 32 grid, and the colour masks covered the background, so the region rule was replaced |
| 93851 | label smoke test, 8 pictures, new region rule | `9529388a` | Submitted. Outcome not read yet |

## Cluster access

On 2026-11-23 at 20:25 and again at about 20:30 (laptop time), `ssh mscluster84` refused a new session on the shared connection and then timed out after 200 and 300 seconds. Job 93851's state could not be read during that time.
