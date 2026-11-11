# To the step 8 walk: SuperDiff is out, CO3 runs on six shards

From the step 9 walk, 2026-11-11 08:45 UTC, on the author's word.

- **SuperDiff is no longer a baseline.** It does not claim to fix hybrids, so it is out of every table and figure (protocol and layout sheet, commit `ef5494c9`; dated in `report/paper-review/history.md`). Jobs 84592, 84593, 84595 and 84597 were cancelled; `superdiff_default/` holds the 373-odd tiles they wrote, unused. Do not relaunch them.
- **CO3 now runs on six shards.** The two original jobs (84590 `eval-co3-0`, 84591 `eval-co3-1`, `NSHARDS=2`) keep running. Four more were added with `NSHARDS=6`, shards 2 to 5 (84757, 84761, 84759, 84760), into the freed `bigbatch` slots. Each new shard is a third of one of the old ones, and `review_baselines_eval.sh` skips tiles that exist, so they meet without double work beyond an occasional tile both reach at once.
- **mscluster68 has a faulted GPU** ("Unable to determine the device handle for GPU0: 0000:17:00.0"); job 84758 landed there and the launcher's guard refused it. Add it to the exclude list.
