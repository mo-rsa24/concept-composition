# Review: can a sentence score read the five properties? (step W2)

**State: no score has been computed.** The labelled pictures are still being built, so no property has a verdict. The [design](design.md) holds the sentences and bars.

## Runs

| Job | Stage | Code | Outcome |
|---|---|---|---|
| 93753 | label smoke test, 4 pictures | `7ab2981c` | Failed in 18 seconds: the code snapshot left out `eval_set_catdog_v2.json` |
| 93795 | label smoke test, 4 pictures | `bec5297b` | Ran in 3 minutes. Left-right labels were right on 4 of 4 by eye. Every box covered the whole 32 by 32 grid, and the colour masks covered the background, so the region rule was replaced |
| 93851 | label smoke test, 8 pictures, new region rule | `9529388a` | Stalled: it started at 20:08 on mscluster89, printed only its start line, and was cancelled at its 90-minute limit. This matches the cluster storage outage below |
| 94092 | label smoke test, 8 pictures, new region rule | `9529388a` | Ran in under 5 minutes. By eye at sheet size: left-right right on 8 of 8, and each recoloured animal carries its own colour with the background unchanged. Size labels are unreliable: on a large cat the word map covers mostly the head, so pictures 04 and 05 read "dog larger" when the animals are about equal, and 03 is doubtful |

## Text-only word check

Before any image is read, [this check](word-check/word-check.json) asks whether CLIP ViT-L/14's text encoder, the one NoisyCLIP keeps frozen, can tell word order and colour binding apart at all. It averages each swapped pair over both word orders: what survives is the part of the swap a sum of word slots cannot produce. The ratio is that part's size over twice the swap's size, 0 for a pure sum of word slots and 1 when the whole swap survives. The script is `2-classifier-guidance/scripts/noise_text_critic/word_check.py`, run on the laptop's CPU, over six noun pairs.

| Property | Trained CLIP: ratio, mean (range) | Same encoder with random weights | Swapped sentences' cosine, trained | Swap size, trained, against changing one noun |
|---|---|---|---|---|
| Colour binding, red and blue | 0.72 (0.48 to 0.94) | 0.13 | 0.956 | 0.23 to 0.41 |
| Left-right order | 0.28 (0.22 to 0.33) | 0.08 | 0.965 | 0.19 to 0.34, against 0.63 to 0.87 |

**Binding is carried by the text side.** Most of the colour-swap difference survives averaging both word orders. With random weights the colour-swap direction is the same for every noun pair (mean cosine 0.96 across pairs, a pure slot term), and in trained CLIP it is not (0.24).

**Order is carried weakly.** Swapping the nouns moves a sentence about a third as far as changing one noun, and about a quarter of that move survives averaging. So the text side gives a left-right check a small target, and W2 measures whether the image side can hit it.

This check uses no image, so it bounds what any reader using this text encoder can do and says nothing about NoisyCLIP's image side.

## Cluster access

From 20:25 on 2026-11-23 (laptop time), `ssh mscluster84` refused a new session on the shared connection and then timed out after 200 and 300 seconds. At 21:05 it timed out again after 300 seconds, a direct connection with no shared socket hung for 45 seconds, and the login node `mscluster` hung for 70 seconds even though its port 22 accepts a TCP connection. At 21:44, 21:48, 22:19, 22:23, 22:54, 22:58, 23:31 and 23:34 the login node hung again, for 60 to 90 seconds each time. At 06:21 and 06:23 on 2026-11-24 the login node still failed. Verbose attempts show the network and the key exchange working, and the connection stalling at public-key authentication. With a 30-second keepalive the client gave up after 90 seconds; with none, at 06:29, the server closed the connection after 2 minutes 46 seconds at the same point. The owner's own interactive login succeeded after a few minutes. Once the owner opened a shared connection, sessions started through it still hung, even `hostname`, while `ssh mscluster107`, which only passes through the login node, answered in seconds. Cluster commands now go through `mscluster107`. A stall at that point usually means the server cannot read the home folder that holds `authorized_keys` (inferred). Job 93851's state could not be read during that time.
