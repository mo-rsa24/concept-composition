# Handoff: where step U1 stands

Loop 7 (`s7`) continues from this file in a session started from `/home/molef/PhD/concept-composition`. Its brief is [the session 7 prompt](../../prompts/session-7-read-the-abstract-checker-papers.md). Step U1 stays `doing` on the board until the closer papers below are settled.

## What is done

**All 18 listed papers are read from their full PDFs.** Each has a note in this folder and a row in the [reading table](reading-table.md). The row data is in `rows/`, one JSON file per paper.

**All 306 quotes are on their stated PDF pages.** `tools/check_quotes.py` found every one. Eleven values were read from figures or page images, and each is marked `read_from_image` in its row. The check proves a quote is on its page. It does not prove a reader interpreted it correctly, and no second reader has checked the interpretations.

**The sketch of what the papers cover** is [the five-checker picture](what-the-papers-cover.png), drawn from `what-the-papers-cover.mmd`.

## What is half done

**Closer papers are not read.** The brief allows up to 10 more when a paper trains a classifier for an abstract property or tests one on unseen concepts. The arXiv title lookup returned no ID for the first four and HTTP 429 (rate limited) for the last four, so none of these was downloaded.

| Paper | arXiv | Why it qualifies | Named by |
|---|---|---|---|
| Harnessing the Spatial-Temporal Attention of Diffusion Models | 2304.03869 | trains a layout predictor for noun pixels and optimises latents with it | SynGen, p. 10 |
| GLIDE | 2112.10741 | guides with a CLIP retrained on noisy inputs | Universal Guidance, p. 2 |
| Refining Generative Process with Discriminator Guidance | unknown | trains a real-against-generated classifier across concepts and adds its gradient | Correcting Diffusion, p. 9 |
| Mid-U Guidance | unknown | learns a guidance function on diffusion features for aesthetics | Readout Guidance, p. 2 |
| CLIP-Count | 2305.07304 | trained text-guided counter tested on held-out FSC-147 classes | T2ICount, p. 9 |
| Open-world Text-specified Object Counting (CounTX) | unknown | trained counter tested on held-out FSC-147 classes | T2ICount, p. 9 |
| VLCounter | unknown | trained counter tested on held-out FSC-147 classes | T2ICount, p. 9 |
| Teaching CLIP to Count to Ten | unknown | trains a counting-aware CLIP | Counting Guidance, p. 9 |

Four more were flagged and left off this list because they are less close to a guided checker: RCC (2205.10203), VA-Count (2407.04948), ImageReward and AFreeCA (2403.04943).

**A zoom ladder is being built by the session that wrote this file.** It teaches the word-map spatial head and lands in `2-classifier-guidance/artifacts/scenes/word-map-checker/`, local only, with nothing published. The replacement session does not start another one. That folder is outside `s7`'s commit list until the owner says where it belongs.

## The next action on U1

1. Fetch the papers into an empty folder: `artifacts/notes/progress/abstract-checkers/01-reading/tools/fetch_papers.sh "$CLAUDE_JOB_DIR/tmp/papers"`.
2. Look up the missing arXiv IDs with at least 10 seconds between requests, because arXiv rate-limited the first attempt at 3.5 seconds.
3. Read the closer papers under `tools/reader-brief.md`, adding each to `tools/papers.tsv` and to the groups in `tools/build_table.py`.
4. Run `python3 -I artifacts/notes/progress/abstract-checkers/01-reading/tools/check_quotes.py artifacts/notes/progress/abstract-checkers/01-reading/rows "$CLAUDE_JOB_DIR/tmp/papers"` and expect 0 not found.
5. Run `python3 -I artifacts/notes/progress/abstract-checkers/01-reading/tools/build_table.py`, check U1's proof, and set U1 to `done`.

## What the owner has decided for U2 and U3

**The design carries five checkers.** They are spatial relation, counting, attribute binding, position and size, and overall prompt match. The brief's equation and `method-search/abstract-checker-idea.png` show three terms, and the owner confirmed five on 2026-11-23, so U2 and U3 write the sum with five:

$$
\hat\epsilon_t = \epsilon_A + \epsilon_B - \epsilon_\varnothing - \sigma_t \nabla_{x_t}\big[\lambda_s \log C_{\text{spatial}} + \lambda_c \log C_{\text{count}} + \lambda_b \log C_{\text{bind}} + \lambda_p \log C_{\text{pos}} + \lambda_m \log C_{\text{match}}\big]
$$

**The main experiment removes one checker at a time.** The owner's words on 2026-11-23: "remove one gradient / abstract concept at a time. For example, we remove count and see if that affects counting, remove attribute binding and see if that confuses concepts". If each removal damages only its own property, composition is controllable, and that is the case for a paper on controllable composition through classifier guidance of abstract concepts. The owner gave that title as an example only.

U3 specifies this experiment, with these parts:

- **The table it fills.** Rows are the configurations: plain product, all five, and all five minus one (five rows). Columns are the five properties, each scored by a scorer that never steers. Control is shown when each removal lowers its own column and leaves the other four within a margin written before the run.
- **Adding back as well as removing.** Each checker alone on the plain product, so a property that needs two checkers shows up as a difference between the two tests.
- **The cap can hide the effect.** The push is capped at 0.4 of the product's prediction norm, so removing one checker hands its share of the cap to the other four. U3 fixes each checker's norm before the sum, and includes a row replacing the removed gradient with a size-matched random one.
- **Strength as a dial.** Sweeping one checker's λ with the others fixed tests whether its property rises with the dial, which is a stronger claim of control than on and off.

## Findings U2 and U3 must carry

- The first thing to test is whether word maps can be read this early. Detector Guidance's word-map detector falls to mAP@0.5 of 18.0 of 100 at t = 800 ([p. 7](https://arxiv.org/pdf/2306.02236#page=7)), and 10 of our 16 pushed steps sit above t = 800 under even spacing (inferred).
- Latent Classifier Guidance assumes independence and prints a proof for one condition only ([p. 4](https://arxiv.org/pdf/2304.12536#page=4)). The sum of logs needs the labels independent given $x_t$, which U2 derives and tests against the early steps (inferred).
- Dynamic CFG's one summed-gradient test scored below one evaluator alone: Gecko alignment 45.3 for CLIP plus discriminator against 46.1 for CLIP, higher is better ([p. 8](https://arxiv.org/pdf/2509.16131#page=8)).
- InfSplign's hand-written centroid loss lifts SDXL VISOR uncond (percent of images with the correct relation) from 45.10 to 76.91 ([p. 18](https://arxiv.org/pdf/2512.17851#page=18)), so our 0 of 8 attention push does not show that attention losses cannot move SDXL.
- Attend-and-Excite raises the weakest word's peak, and our correlation push has no such term ([p. 4](https://arxiv.org/pdf/2301.13826#page=4)). The dog token's peak on our 8 failed seeds is an untested explanation (inferred).
- Correcting Diffusion's best-of-15 selection with no resampling also reaches 0.744 object occurrence ([p. 6](https://arxiv.org/pdf/2312.06038#page=6)), so every comparison carries a best-of-K baseline (inferred).
