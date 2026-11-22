# CountLoop first read

CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance, arXiv 2508.16644v5, TMLR 08/2026, 17
PDF pages. Page numbers are PDF pages counted from 1, and they match the printed footer.

## What was read

Four tables and 335 rows. Table 2 has four panels, (a) to (d), under one label, so all its rows carry `CountLoop Table 2`
and the panel shows in `task` and `protocol`. Rows outside the tables are the count tags on the Figure 6 examples (page 9)
and 27 numbers stated in the running text or the Figure 8 caption (pages 2, 3, 10, 11, 12 and 13).

Every method except CountLoop is outside the rival list, so those rows are `baseline`. The two CountLoop critic variants
of Table 1 are both `CountLoop`, separated in `model`.

## Numbers that could not be read

The dashed reference line in Figure 8 (page 13) has a small printed label that the bars partly cover. The caption says the
line marks CountLoop's overall MAE on CountLoop-S. No row was written.

Figure 1 (page 1) prints count tags on example images for several colours, and which tag counts which object cannot be
told with certainty. No row was written.

## Plot-only, no rows written

Figure 7a (page 11), MAE against object count. Figure 7b (page 11), MAE against runtime at N of 25, 50 and 100, with values
only where the text on page 11 states them. Figure 8 (page 13), MAE bars by object size and N, with values only where the
caption and text state them.

Values on page 11 are written with an approximately sign or a tilde. They are copied as printed, and the metric name says
approximate.

## Protocol facts a comparison depends on

**Datasets (page 8).** COCO-Count (MS-COCO subset), T2I-CompBench Count (subset), CountLoop-S (200 prompts, 30 to 200
instances) and CountLoop-M (200 prompts, 30 to 200 instances). Prompt counts for the first two are not stated.

**Evaluators (page 8).** MAE uses OWLv2 at confidence 0.25 with class-agnostic NMS IoU 0.5, counting boxes of at least
3 x 3 whose center is visible. Exact-count match uses CountGD++. Spatial quality uses the CLIP-FlanT5 encoder from
VQAScore. The paper says these detectors are proxies and do not establish true physical counts.

**Default model (page 12).** SDXL backbone, Qwen3-VL as design and critic, GroundingDINO critic detector at confidence 0.3,
Q-Align aesthetic scorer.

**Baselines (page 9).** LMD and CountGen use their own layout modules. MIGC, InstanceDiffusion and 3DIS receive layouts from
LMD's generator. Agentic baselines run with their native self-correction loops (page 10).

**Commercial reference (page 10).** Nano Banana Pro was run on a 100-prompt subsample of CountLoop-S, a different prompt
count from the other CountLoop-S rows.

**Human study (page 12).** 30 participants (20 designers and 10 AI artists), each rating 15 blinded sets of 5 images.

Seeds, image resolution, sampler and guidance scale are not stated in the main text. Runtime breakdown, benchmark
construction and study details are deferred to a supplement that this PDF does not contain.

## Hardware and training time

The method is training-free. The acknowledgements name the NVIDIA Academic Grant Program and Isambard-AI as resources, and
no hardware is tied to a run. Page 11 gives wall-clock times of about 150 s (N = 100) and about 50 s (N = 25) for CountLoop.

## Contradictions inside the paper

**CountLoop-S is single-category on page 8 and under Multi Category in Table 1.** Table 1 (page 10) places both CountLoop-S
and CountLoop-M under the Multi Category header. Those rows keep the printed group and name the conflict in `task`.

**"Less than half" fails against 3DIS (page 9).** The text says 7.59 is less than half the error of Qwen-Image (17.30) and
3DIS (14.55). 7.59 is 43.9% of 17.30 and 52.2% of 14.55.

**The critic and evaluator overlap in one row.** Page 12 says the critic detector is kept distinct from the evaluation
detector so the critic cannot optimise against the metric. Exact-count match is scored by CountGD++ (page 8), and the
second CountLoop row of Table 1 uses CountGD++ as its critic, so its `Ex.` values (85, 70, 29 and 42) are scored by its own
critic.

**The rating scale does not reach an exact match (page 12).** Each rating point maps to 17% count precision and 5 is said
to indicate an exact match, but 5 x 17% is 85%.

**Table 4 rounds one value down.** MAE 9.21 against 14.55 is a 36.70% reduction, printed as +36%. The other two rows round
correctly (47.84 to 48, 41.10 to 41).

**"3x lower error" (page 11).** MAE about 13 against about 50 is about 3.8 times lower.

**Two Table 2b rows are identical.** PG alone and PG with IR both print MAE 14.98 and spatial 0.68, so IR changes nothing
without CA. Page 10 uses this value as the result of dropping CA.

**The density threshold differs.** Page 9 writes N at least 30, and page 14 writes N above 30.

Table 3 bolds SDXL at MAE 7.59 although SD 3.5 prints a lower 7.44. Bold marks the default in Table 2's caption, and
Table 3's caption does not say what bold means.
