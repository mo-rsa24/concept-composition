# Handoff: where the abstract-checker reading stands

Loop 7 (`s7`) continues from this file in a session started from `/home/molef/PhD/concept-composition`. Its brief is [the session 7 prompt](../../prompts/session-7-read-the-abstract-checker-papers.md). Step U1 is done, and the next step is U2, the score view.

## What is done

**25 papers are read from their full PDFs:** the 18 listed papers and 7 closer papers they cite. Each has a note in this folder and a row in the [reading table](reading-table.md). The row data is in `rows/`, one JSON file per paper.

**All 463 quotes were checked against their stated PDF pages, and none is missing.** Sixteen values were read from figures or page images and are marked `read_from_image`, so they rest on a reader's eye. The check proves a quote is on its page. It does not prove a reader interpreted it correctly, and no second reader has checked the interpretations.

**Mid-U Guidance is not read.** It is a W&B report with no arXiv PDF or HTML page, so no claim uses it.

**Two pictures summarise the reading:** [the five-checker picture](what-the-papers-cover.png) and [what the closer papers read](what-the-closer-papers-read.png). [GLIDE's CLIP score against human ratings](glide-clip-score-against-people.png) sits in the GLIDE note.

**A zoom ladder is being built by an earlier session.** It teaches the word-map spatial head and lands in `2-classifier-guidance/artifacts/scenes/word-map-checker/`, local only, with nothing published. This session does not start another one, and that folder is outside `s7`'s commit list until the owner says where it belongs.

## The next action

Write `artifacts/notes/progress/abstract-checkers/02-score-view.md`. It derives the added gradients, when classifier logs may be added, what a classifier on attention maps differentiates through, and one worked example with small numbers. Read the files listed in the session 7 prompt under "What to read" first. To re-check quotes, fetch the PDFs with `tools/fetch_papers.sh "$CLAUDE_JOB_DIR/tmp/papers"`.

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
- GLIDE's CLIP guidance raised a CLIP score computed by a different CLIP network (ViT-B/16 against the guiding noised ViT-L), while people preferred classifier-free guidance on caption match, 110.9 against 29.3 Elo ([pp. 8, 9, 13](https://arxiv.org/pdf/2112.10741#page=8)). A scorer from the same family as a checker is therefore weak evidence (inferred).
- Discriminator Guidance's correction is exact only at the optimal discriminator and for a true model score, and its Theorem 2 "does not guarantee that the gain is strictly positive" ([p. 4](https://arxiv.org/pdf/2211.17091#page=4)). The product of experts has no true model score, so U2 cannot borrow this guarantee (inferred).
- FSC-147 holds no image below 7 objects ([CounTX p. 19](https://arxiv.org/pdf/2306.01851#page=19)), so its held-out counting results say nothing direct about a count of two.
- Counters given no text beat the text-guided ones on FSC-147: CounTR 14.71 against CLIP-Count 17.78 objects per image ([p. 6](https://arxiv.org/pdf/2305.07304#page=6)). VLCounter moves only from 17.05 to 17.19 when text is removed from its prompts ([p. 5](https://arxiv.org/pdf/2312.16580#page=5)). The count checker needs right-word, wrong-word and no-word tests (inferred).
- Spatial-Temporal Attention loses spatial precision from 66.7% to 51.7% when its assigned regions are removed ([p. 8](https://arxiv.org/pdf/2304.03869#page=8)), so assigning each subject a region is worth testing as a baseline (inferred).
