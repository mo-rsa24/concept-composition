# Prompt: route step 7, read the scorers together

Paste into a fresh session in `/home/molef/PhD/poe_repair_min`. It replaces the prompt block under step 7 of `plans/route-answer-the-iclr-reviews.md`, whose image paths predate step 4.

```text
/drip-read --combine --lean "how a benchmark decides that two named objects are both in the picture,
and what its number cannot tell you" 2509.25940 2307.06350 <ImageReward paper> <GenEval paper>
Model: Sonnet, medium effort. Box: 60 minutes of your time. Filing: tied.
Controls: each metric example (its number on the two small-object images), each metric break (an
image that fools it), the last round prove (a claim about where a score sits in our table).
Side routes: after the read, /submerge --names <journey folder from step 3> (the names pass over the
eleven words, once step 3 has made the journey).

Background
Reviewers of submission 56173 asked for scores on standard benchmarks. Before running them we need
to know what each one measures, its scale, and its blind spots, so each number lands in the right
place in the results and is not over-claimed. The repo is /home/molef/PhD/poe_repair_min. The
reader is a visual, hands-on learner with little time who needs to use each metric and place it,
not derive it.

Objectives
1. Resolve and show the arXiv ids of the ImageReward and GenEval papers before reading anything.
2. Read the one idea across the four sources, one chunk per round, on two small-object images.
3. Before each scorer answers, ask the user to predict which image it ranks higher.
4. Leave a one-page table: metric, what it measures, scale, blind spot, which rows of our results
   it feeds.

What to read
- arXiv 2509.25940 (CO3), Table 1 and the evaluation section.
- arXiv 2307.06350 (T2I-CompBench++), the BLIP-VQA description.
- The ImageReward and GenEval papers, once their ids are confirmed.
- composition/focus/eval_metrics.py: CLIP, EVA-CLIP and ImageReward scorers already written here.
- 1-lora-correction/report/paper-review/protocol.md and held-out-set.md, once steps 2 and 4 have left them.
- The step 4 census pictures under /datasets/mmolefe/poe_repair_min/outputs/review_step4/census/
  (joint prompt and product, 53 held-out pairs, seeds 9 to 16; read them with scripts/cluster.sh sh
  or pull a few with scripts/cluster.sh pull): the source of the two small-object images.
- 1-lora-correction/report/paper-review/layout-sheet.md, rows R1 and R14: which scores each table carries.

What is already known
- CO3 scores the Chefer et al. prompt sets animals, animals-objects and objects with BLIP-VQA and
  ImageReward, and does not state prompts per set, seeds, resolution or how the metrics were run.
- Our own count is GroundingDINO instances, validated on labelled images. It counts animals but
  cannot
  tell a cat from a dog (earlier notes, 64 of 64 false dog calls).
- CLIP separates pairs from fused renders when ranking (earlier notes); that is a ranking, not a
  verdict on whether two objects are present.
- The main table R1 carries, per prompt set: Comp. (the share of images with both animals, by the
  instance count; animals set only), CLIP score, BLIP-VQA and ImageReward; then network evaluations
  per image. R14 carries GenEval two-object and counting and T2I-CompBench spatial and complex, each
  by its official evaluator (layout-sheet.md).
- The user wants to use each metric and place it. Draft to supervisors Thursday 12 or Friday
  13 November; OpenReview closes Wednesday 18.

When something is missing
- If an arXiv id cannot be resolved, show the candidate titles and ask the user to confirm one.
- If no small-object pair is in the census, use two tiles from
  1-lora-correction/artifacts/results/the-adapter-on-cat-and-dog/how-far-one-adapter-reaches/does-one-adapter-compose-four-pairs/tiles/ (tent and snowy mountain, lighthouse and stormy
  sea) and say so.
- If a scorer cannot run on this machine, say which and run it on the cluster after the user says
  yes.

Rules
- One chunk per round. Predict before the answer, every time.
- Do not run any scorer on the evaluation set; only on the two small-object images.
- Do not copy a published number into our results; cite it as context only.

What to hand back
- 1-lora-correction/artifacts/drips/answer-iclr-reviews/read-the-scorers/COMBINE_MAP.md and the one-page table.
- A vault note written by the walk's own close, in the Obsidian vault.
- A line added to 1-lora-correction/report/paper-review/prze-1-scores.md linking the table.
- Proof: the table has a row for every metric, and each blind spot cites the image that showed it.

What you will see
Along the way: the ids, then each metric's chunk with your prediction first and its answer second,
each followed by one question. At the end: what was asked, what was made with paths, PASS or MISS,
what is open, and at most one question. PASS means every metric has a row with its blind spot shown
on one of your own images. You judge from the table.
```

- `added:` where the census pictures live and how to reach them (step 4 retro) · `added:` the scores R1 and R14 carry (`layout-sheet.md`) · `added:` the dates (route, "What we know so far")
- `dropped:` the folder `four-pairs-four-methods/` (why: renamed to `four-pairs-one-adapter/`)
