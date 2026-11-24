# The search for papers on leakage between subjects

Step Z1 reads the papers on leakage between subjects, and searches for any paper that measures which subject pairs leak and why. This page records that search, run on 2026-11-24, and what it added to the reading list.

## What was searched

**The five named papers' ids were confirmed on the arXiv API first.** All five resolve to the titles [the loop's brief](../../prompts/session-12-where-co3-leaks.md) names: Be Yourself 2403.16990, Be Decisive 2505.21488, Conform 2312.06059, Divide & Bind 2307.10864, Structured Diffusion Guidance 2212.05032.

**arXiv API, abstract field, 12 queries, up to 25 results each, sorted by relevance.** The script and every hit with its abstract are kept in [the saved search](tools/search-2026-11-24/), as `q.py` and `arxiv_search.json`.

| Query (arXiv abstract field) | Matching papers |
|---|---|
| "semantic leakage" AND diffusion | 20 |
| "attribute leakage" AND diffusion | 16 |
| "subject leakage" OR "feature leakage" AND "text-to-image" | 5 |
| "similar subjects" AND "text-to-image" | 5 |
| "subject mixing" AND diffusion | 1 |
| "multi-subject" AND "text-to-image" AND leakage | 5 |
| "cross-attention" AND "attribute binding" AND "text encoder" | 1 |
| "text encoder" AND information AND mix AND "text-to-image" | 1 |
| "semantically similar" AND objects AND "text-to-image" AND attention | 3 |
| "self-attention" AND leakage AND diffusion | 16 |
| "catastrophic neglect" AND similar | 0 |
| entity AND leakage AND "text-to-image" | 7 |

The 80 hits are 64 distinct papers. Most use "leakage" for style transfer, editing, personalisation with reference images, or privacy, and were set aside from their saved abstracts.

**Web search, three queries, the same day:**

1. which subject pairs leak, text-to-image diffusion, semantic leakage, similarity measured, analysis
2. text embedding similarity predicts subject mixing, two animals, diffusion model, cross-attention overlap
3. semantic leakage benchmark, text-to-image, SDXL, animal pairs, leakage score

## What it found

**No paper found measures which subject pairs borrow parts and ties that to a property of the pair.** The three web searches each reported no such study. The closest is [DAAM](daam.md), found through a reader's citation trail: over 1,000 two-noun prompts on SD 2.0 it relates attention-map overlap and shared category to whether both objects appear, a different failure from borrowing (p. 7). The rest measure leakage on a set of prompts or argue for one cause from examples, and each note says whether it reports a per-pair breakdown.

**Eight papers were added to the five named ones** and read in full:

| Paper | arXiv | Why it was added |
|---|---|---|
| Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects | 2411.18936 | the only hit for "subject mixing"; penalises the overlap between one subject's cross-attention and another's self-attention, on similar-looking subjects |
| Enhancing MMDiT-Based Text-to-Image Models for Similar Subject Generation | 2411.18301 | names three places mixing of similar subjects enters on SD3, one of them the text encoder, and measures early cross-attention overlap per subject |
| DeLeaker: Dynamic Inference-Time Reweighting For Semantic Leakage Mitigation | 2510.15015 | builds a dataset for measuring semantic leakage (SLIM, 1,130 human-verified samples by its abstract), with animal-pair subsets |
| A Cat Is A Cat (Not A Dog!): Unraveling Information Mix-ups in Text-to-Image Encoders | 2410.00321 | a causal analysis of how the text encoder mixes two objects' information |
| Magnet: We Never Know How Text-to-Image Diffusion Models Work, Until We Learn How Vision-Language Models Function | 2409.19967 | examines CLIP's text encoder on multi-object prompts and blended embeddings |
| Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models | 2504.01137 | found by web search; measures by patching when one item's meaning flows into another item's token |
| DALLE-2 is Seeing Double: Flaws in Word-to-Concept Mapping in Text2Image Models | 2210.10606 | named by the web search as the origin of image-side semantic leakage (Rassin et al. 2022) |
| What the DAAM: Interpreting Stable Diffusion Using Cross Attention | 2210.04885 | cited by Conform (p. 3) for nouns of the same category tangling their cross-attention maps; the closest candidate for a measurement of which pairs leak |

**Two more papers cited by the readers were not added:** A-Star (2306.14544), an attention-overlap loss of the same kind as Conform's, and DenseDiffusion (ICCV 2023), which needs a layout.

**Four close papers were already read by other loops and are cited, not reread:** Attend-and-Excite (2301.13826) and SynGen (2306.08877) in [Loop 7's reading table](../../abstract-checkers/01-reading/reading-table.md), Detector Guidance (2306.02236) in the same table, and Token Merging (2411.07132) and Rectify Then Diffuse (2608.03135) in Loop 4's first reads under `targets/reads/first/`.

**Set aside from their abstracts:** Compositional Text-to-Image Synthesis with Attention Map Control (2305.13921) and Towards Better Text-to-Image Generation Alignment via Attention Modulation (2404.13899) name entity and attribute leakage but need boxes or fix it in cross-attention in the same way as the read papers; FOCUS (2510.02315) is a flow-matching control method; FaR (2504.03292), MuDI (2404.04243) and IdGlow (2603.00607) are personalisation from reference images, which this project does not do.
