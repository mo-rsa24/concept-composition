# Handoff to the laptop: cat and dog captions

Written on the cluster on 2026-11-03 so a session on the laptop can carry on. Paste the block below
into a fresh Claude Code session started in `/home/molef/PhD/poe_repair_min-dog-and-rain`, after
`git pull` on the branch `exec/dog-and-rain`.

```text
Background
The project studies why product-of-experts (PoE) composition on SDXL fails to draw a cat and a dog
together. The missing piece is the interaction term r_t, the gap between the joint prompt's
prediction and the PoE of the two single prompts. On 2026-11-03 you asked whether a third prompt x
could stand in for that term, so that PoE(cat, dog, x) matches the joint prompt, and whether the
word "and" is rare in the training captions next to a cat and a dog. A cluster session searched all
2.1 billion captions of Re-LAION-2B-en (the public stand-in for SDXL's unpublished training set)
and built a page of sampled captions with clickable pictures for an eye check. You are continuing
that work on the laptop.

Objectives
1. Run git pull on the branch exec/dog-and-rain and confirm the files listed under What to read
   exist.
2. Open the ChatGPT share link https://chatgpt.com/share/6ae9602c-0378-83ea-a868-6df6c06a6c91 in
   the browser and write down in three to six lines what it says about the wet camel caption search.
3. Open 4-inclusive-captions/artifacts/results/does-any-caption-say-a-cat-and-a-dog/caption-check.md and walk the user
   through the eye check, one section per message, in the order bare, short, random, long.
4. After each section, write the user's letters into that section's last column, and print the
   running count of B, O and N with the number of pictures it covers.
5. When all four sections are done, write the table of B, O and N per section, with the share of B
   and its 95 percent interval, to 4-inclusive-captions/artifacts/results/does-any-caption-say-a-cat-and-a-dog/tally.md.
6. Say plainly what the tally shows about the user's guess that a clean "a cat and a dog" caption
   with both animals in the picture is missing from the training data.
7. Propose, in plain words and without running it, the next step toward finding a text x for
   PoE(cat, dog, x), using the words that join the two animals in the captions as candidates.

What to read
- _archive/artifacts/notes/session-merges/2026-11-03-cat-and-dog-captions-handoff.md: this file, with the
  first read by the cluster session below the block.
- 4-inclusive-captions/artifacts/results/does-any-caption-say-a-cat-and-a-dog/caption-check.md: the page to check by eye.
- 4-inclusive-captions/artifacts/results/does-any-caption-say-a-cat-and-a-dog/counts.json: sizes, forms and joiners.
- 4-inclusive-captions/artifacts/results/does-any-caption-say-a-cat-and-a-dog/sample.json: the 240 sampled captions.
- 3-understanding-compositionality/scripts/census/cat_and_dog_captions.py: the four stages that built the page.
- 3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-camel-and-dog-in-the-training-data.md: the earlier
  wet camel walk this one copies.
- context/world/poe-composition.md: what PoE composition, the joint prompt and r_t mean here.

What is already known
- The search covered all 128 files, 2,097,693,557 captions, and found 1,102,469 that name a cat word
  (cat, cats, kitten, kittens) and a dog word (dog, dogs, puppy, puppies). Source: counts.json.
- Caption length among those: median 14 words, 10th percentile 7, 90th percentile 27. Source:
  counts.json.
- The bare form, only the two animals joined by and, &, with or plus, is 4,971 captions, 0.45
  percent of the 1,102,469. Short captions of 12 words or fewer that are not bare are 423,432 (38.4
  percent). Captions of 30 words or more are 85,734 (7.8 percent). Source: counts.json.
- In 821,378 captions the two animal words sit next to each other. What joins them: nothing
  (368,780), and (261,475), & (99,367), a comma (52,126), or (32,695), vs (2,422 plus 1,362 as
  "vs."), with (2,094), + (893), versus (136), x (18), plus (10). Source: counts.json.
- Most bare captions come from stock photo sites and read "Cat and Dog". In the cluster session's
  own look at the first 24 loaded bare pictures, 17 were photos showing both animals, 3 were
  drawings or silhouettes showing both, and 4 were product or cover art. This is one reader's first
  look at 24 pictures and is not the user's count. Source: the sample in caption-check.md, tiles 1
  to 24 of the bare section.
- The pictures come from a sample of 60 captions per class, seed 20261103. 155 of 240 still load,
  so each section shows 27 to 31 pictures.
- Re-LAION-2B-en is the stand-in for SDXL's training data. SDXL's own captions are unpublished, so
  none of this describes SDXL directly. Source: the wet camel walk.
- A zero count is a bound, not "never". With 0 in n, quote "fewer than 3 in n" (rule of three).
  Source: the wet camel walk, step 6.
- On the interaction term: r_t is state specific. Its cosine between seeds is 0.002, and the part
  shared across seeds is about 19.6 percent at steps 0 to 2 and near zero after step 20. So a
  single text x cannot equal r_t at every step and seed; the fair target is how much of the shared
  early part some x recovers. Source: the project's notes on the r_t direction structure and the
  correction's shared early part.
- The unconditional prediction is already handled by writing the third expert as a difference from
  it: its contribution is w_x times (eps_x minus eps_empty), so x equal to the empty prompt adds
  nothing. The joint prompt must be computed in the same guidance form.
- On the cluster, the full caption set is at /datasets/mmolefe/relaion2b-en/parquet and the 128
  per-file results of this search are at /datasets/mmolefe/cat-and-dog-captions. The laptop has
  neither.
- The ChatGPT link was fetched from the cluster and returned only the page title "ChatGPT -
  Discuss Silicotuberculosis Imaging", with no body. The user described it as holding the wet camel
  caption search, so the link may be wrong or private.

When something is missing
- If the ChatGPT page does not load or shows a different topic, say so in one line, skip objective
  2, and ask the user to paste the wet camel part of that chat. The wet camel walk in the plans
  folder carries the same exercise meanwhile.
- If fewer than 20 pictures in a section load on the laptop, run the fetch stage of
  3-understanding-compositionality/scripts/census/cat_and_dog_captions.py again, then the page stage. Dead links change over time.
- If the user wants more pictures than the sample holds, the full per-file results are only on the
  cluster. Ask the user to run the sample stage there with a larger per-class number. Until then,
  work with the 240.
- If a question about what a term means is not answered by context/world/poe-composition.md, ask
  the user. Do not infer a meaning from the name.

Rules
- Show one section of the page per message and wait for the user's letters. Never propose a call.
- A call is about the picture: B both a cat and a dog are in it, O only one of them, N neither or
  not a photo. A drawing, a silhouette or product art with both animals is B only if the user says
  so; ask once at the start whether to count those, and record the answer in tally.md.
- Do not run any search over the full caption set, and do not start any run on the laptop's GPU.
- Do not commit or push until the user says done.
- Write plain sentences. Say what each number covers and its unit.

What to hand back
- 4-inclusive-captions/artifacts/results/does-any-caption-say-a-cat-and-a-dog/caption-check.md with the last column
  filled for every picture shown.
- 4-inclusive-captions/artifacts/results/does-any-caption-say-a-cat-and-a-dog/tally.md with the table from objective 5.
- A short note of what the ChatGPT link said, at the top of tally.md under its own heading.
- The proposal from objective 7 in the closing report, not in a file.

What you will see
Along the way: one section's table with its pictures, then a request for the letters, then the
running count. At the end, a closing report in this order: what was asked, what was done with each
output's path, the tally per section with its 95 percent interval, what is still open, and one
question for the user.

Done when the user has called every picture shown in all four sections and tally.md holds the
table; the user judges from tally.md.
```

added: the ChatGPT link and what the cluster fetch returned (from this session)
added: the interaction-term facts, r_t cosine and the unconditional handling (from the user's
question this session and the project's r_t notes)
added: the cluster session's own first read of 24 bare pictures, marked as not the user's count
dropped: the ChatGPT share's content (why: the cluster fetch returned only a title, so nothing
could be carried)
dropped: the optimisation of x itself as an objective (why: the user asked to continue this
session, and the next step is only proposed in objective 7 until the tally says what the captions
hold)

## What the cluster session found, in one place

The literal caption "Cat and Dog" exists 4,971 times in 2.1 billion captions, about one in 420,000,
so the user's doubt that a single such caption exists is answered: it does, mostly on stock photo
sites, and in the first look most of those pictures show both animals as a photo. The user's second
guess, that captions naming both animals are long and varied, held only partly: the median is 14
words and 74 percent have the two animal words next to each other. About a third of those adjacent
pairs (261,475 of 821,378) use "and". Nothing here says whether "and" binds the two animals in the text encoder; it says
only how real captions phrase them.

The first full-set search ran as one DuckDB query, never finished in over 20 minutes under load, and
was killed by a session restart. The rerun as 128 per-file jobs, 16 at a time, finished in about 10
to 15 minutes. A shell on the cluster node stalls while that many reads run.
