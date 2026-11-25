# Who counts borrowed parts and extra subjects

The question is whether any paper counts how often a text-to-image method does either of two things:
draws one subject with the other subject's parts (a borrowed face, markings or head), or draws more
subjects than the prompt asked for. The search covered CO3 and its rivals, along with any benchmark built for these failures.

This project's only such count is its own hand labels of CO3 on SDXL, in
`artifacts/notes/progress/noise-text-critic/02d-binding/part-binding-calls.csv`. Of the 88 pictures of animal
pairs (11 pairs at 8 seeds; the lighthouse and sailboat pair is left out), 34 are clean, 33 borrow parts and 21
are not two subjects. In 20 of those 21 an extra animal is drawn.

**Marks.** `read` means read from the paper in this session. `note` means read from this project's
page-cited reading note for that paper, which was itself read from the full PDF. `inferred` means this page's own
reasoning.

## Conclusion

**Borrowed parts have been counted, but never on SDXL and never for CO3 or any SDXL rival.** Two papers
count them as a rate per picture. DeLeaker's annotators found leakage in 33% of 1,000 plain animal-pair images on
FLUX.1-dev, and Self-Cross's GPT-4o question finds a "mixture" in 22% to 30% of images on SD1.4 and SD3. Two
older papers count something narrower: A Cat Is A Cat counts fused creatures, and Seeing Double counts leaked
properties on prompts built to leak. CO3, ToMe, Attend-and-Excite, FKC, CONFORM, Divide-and-Bind, Bounded
Attention, Be Decisive and the MMDiT paper report question-answering, detector, image-quality or preference
scores, and none of those scores sees a borrowed part on two separate bodies (note; GenEval's own rule, read).

**No paper found counts extra subjects on a prompt that names two subjects.** The nearest measures count
instances on prompts that ask for a number (Bounded Attention and Be Decisive's counting precision and F1, FineGRAIN's
counting failure mode). DeLeaker sets its images with missing or extra subjects aside instead of reporting a rate. Seeing Double's
homonym rate counts a second object drawn from one ambiguous word.

**For Paper 2's measure on SDXL (inferred):** DeLeaker's written labelling rule is the closest published
definition of `borrowed`, and the project's two-labeller calls already follow the same idea, so the hand labels can
be cited as following an established protocol and reported as the first such count on SDXL. Self-Cross's GPT-4o
mixture question is the only cheap automatic check per picture, but it reads high when a subject is missing, so it needs a
both-present check beside it. The extra-animal count has no published counterpart, so it would be a new measure:
a detector count of animals above two per picture would do, since it needs no species name (the project's
detector can count animals but cannot name them).

## The table

| Paper (arXiv id, confirmed on the arXiv API) | Failure it counts, and how a picture is judged | Rate, model, prompt set, sample count | Where the number is | Could it be Paper 2's measure on SDXL | Mark |
|---|---|---|---|---|---|
| [DeLeaker](../leakage/01-reading/deleaker.md), 2510.15015 | Borrowed parts. Two authors label by a written rule: leakage if one subject carries a feature unique to the other, if the two look like two of one kind, or if a hybrid cannot be assigned to either. Blur, colour bleed and occlusion do not count. A GroundingDINO, SAM and Gemini filter runs first. Extra subjects are not in the rate | 33% of 1,000 plain animal-pair images (328 kept); 23% weighted over 4,000 images in five subsets. FLUX.1-dev, 20 steps; 200 GPT-4o-written look-alike animal-pair prompts × 5 seeds. 222 images with missing or extra subjects are set aside, with no rate | p. 36, Table 17; rule p. 30; filter p. 37; set-aside images p. 27 | Yes for the labelling rule, which matches the project's `borrowed` call apart from its "two of one kind" clause. Its automatic scorer compares a fixed image with the original and gives no rate; it agrees with people at Spearman 0.432 (p. 6, p. 7). CLIP and BLIP crop scores did not track leakage at all (p. 18) | note; usefulness inferred |
| [Self-Cross guidance](../leakage/01-reading/self-cross-guidance.md), 2411.18936 | Mixture, which covers fused and borrowed together. GPT-4o answers "Is the generated content a mixture of {class_A} and {class_B}?" per image | 27.8% mixed on SD1.4 Animal-Animal (66 prompts × 65 seeds); 22.5% on SD1.4 and 15.1% on SD2.1 on SSD-2 (31 look-alike pairs × 65 seeds); 29.9% on SD3-medium SSD-3 (21 three-subject prompts). Rates are 100 minus the printed "w/o M" | p. 7, Tables 4 and 5; p. 6, Table 1; question p. 12 | Partly. It is cheap and per picture, but the paper warns it reads high when a subject is missing (p. 7); plain SD1.4 shows both subjects in only 30.8% of SSD-2 images. It needs a both-present check beside it. The note records no agreement figure with people | note; rates inferred by subtraction |
| [A Cat Is A Cat](../leakage/01-reading/a-cat-is-a-cat.md), 2410.00321 | Fused creature only. OWL-ViT boxes for the two named animals overlap by more than 90%. Agrees with human labels on 81% of 400 images | 9.5% of images on SD1.4; Attend-and-Excite 11.5%, SynGen 10.5%. 400 two-animal prompts over 17 animals | p. 8, Table 4; metric p. 7 | No. It cannot see a borrowed part on two separately detected bodies (inferred), and CO3 draws no single or fused picture on 96 seeds ([the reviewer's check](../reviewer/2026-11-24-review.md), line 45) | note |
| [Seeing Double](../leakage/01-reading/seeing-double.md), 2210.10606 | Borrowed property (one animal's shell or fins on the other) and an extra subject from a homonym (both senses drawn). Who judged is not printed | Property leak in 92.5% of 120 images against 6.6% for minimally changed controls, 10 hand-written pairs × 12 images; homonym duplication 80.3% (164 of 204, 17 prompts × 12). DALLE-2 | p. 4; p. 3; Tables 1 and 3, pp. 6 and 7 | No. The prompts are built to leak, the model is DALLE-2 and the judge is unstated. Its challenge-and-control prompt design is worth borrowing (inferred) | note |
| [MMDiT similar subjects](../leakage/01-reading/mmdit-similar-subjects.md), 2411.18301 | Neither directly. Success means every named subject is found by Grounding DINO or GPT-4o-mini, so mixing counts only when a subject is lost | Plain SD3: 53.1% of two-look-alike-subject images succeed; 30 prompts × 100 seeds | p. 8, Table 1; definition p. 12 | No | note |
| [Bounded Attention](../leakage/01-reading/bounded-attention.md), 2403.16990 | Extra instances on counting prompts only: detector precision against the requested number. Leakage is judged only by a preference study | Counting precision 0.83, recall 0.88 on SD, DrawBench; the authors say these metrics "have limitations in capturing semantic leakage" | p. 9, Table 1 | No borrowed-parts rate. Counting precision is the nearest published form of an extra-subject count (inferred) | note |
| [Be Decisive](../leakage/01-reading/be-decisive.md), 2505.21488 | Extra instances on counting prompts only: GroundedSAM count F1. Leakage judged by preference only; the paper says its automatic metrics "fail to detect semantic leakage" | Single-class count F1 0.837, multi-class 0.723 on SDXL; T2I-CompBench, 200 prompts. Preference 0.74 over plain SDXL, 32 participants | p. 12, Table 2; p. 7, Table 1 | No | note |
| [DAAM](../leakage/01-reading/daam.md), 2210.04885 | Neither: a dropped subject only, three crowd annotators per image | Both present in 52% of same-category pairs against 61% for unrelated pairs; SD 2.0, 1,000 prompts × 1 image | p. 7 | No | note |
| [CO3](../gaps/notes/co3.md), 2509.25940 | Neither. BLIP-VQA, ImageReward, and human ratings 0 to 1 from 11 raters | Animal pairs BLIP-VQA 0.7441 on SDXL; no seed or image count printed | p. 7, Tables 1 and 2 | No. Its own appendix figure shows a tiny extra animal beside a horse, uncounted ([the failures catalogue](../targets/failures.md), Fig. 7 p. 19) | note |
| [ToMe](../gaps/notes/tome.md), 2411.07132 | Neither. BLIP-VQA on T2I-CompBench and a GPT-4o object-binding score | Object binding 0.9549 on SDXL, 50 prompts | p. 8, Table 1 | No | note |
| [Attend-and-Excite](../abstract-checkers/01-reading/attend-and-excite.md), 2301.13826 | Neither. CLIP and caption scores, plus a user study. It says Composable Diffusion "tends to generate images containing a mixture" (p. 6) without counting it | User study 90.70% of picks, 65 respondents, 10 prompts × 4 seeds; SD1.4 | p. 8, Table 2 | No | note |
| FKC, 2503.02819 | Neither. GenEval two objects, which marks a picture correct when both named objects are detected; extra objects and borrowed parts are not penalised (GenEval 2310.11513, p. 5) | 0.78 on SDXL, 553 prompts; images and seeds not stated | Table A4, p. 39 ([the gaps page](../gaps/01-is-paper-2-still-open.md)) | No | FKC figure from note; GenEval rule read |
| [CONFORM](../leakage/01-reading/conform.md), 2312.06059, and [Divide-and-Bind](../leakage/01-reading/divide-and-bind.md), 2307.10864 | Neither. TIFA question answering, CLIP caption similarity and user studies; the Divide-and-Bind note finds that no score catches a hybrid or a borrowed part | CONFORM TIFA 0.95 on SD; Divide-and-Bind TIFA about 32% above SD on Animal-Animal | CONFORM p. 7, Table 2; Divide-and-Bind p. 8, Fig. 6 | No | note |
| Benchmarks: GenEval, T2I-CompBench, FineGRAIN 2512.02161, MultiBind 2603.21937, When Identities Collapse 2603.26078, MuDI 2404.04243, T2LSC-Bench 2609.02255 | GenEval and T2I-CompBench check presence or attributes only. FineGRAIN's 27 failure modes include counting but no fusion or borrowed-part mode; it reports SDXL at 0.034 success on two-object counting prompts. MultiBind, When Identities Collapse and MuDI count identity mixing in personalisation from reference photos, a different setting. T2LSC-Bench counts leakage of rendered text | No rate of either failure on a plain two-subject prompt | GenEval p. 5; FineGRAIN Tables 7 and 9 | No | GenEval read; FineGRAIN from a fetched summary of its HTML; the rest abstract only |

## Searches made

- Loop 12's leakage search on 2026-11-24: 12 arXiv abstract queries and 3 web searches, recorded in
  [the search for leakage papers](../leakage/01-reading/search-for-leakage-papers.md). It found no study tying
  which pairs borrow to a property of the pair.
- This page, 2026-11-25. arXiv API abstract queries: "object duplication"; "extra objects" with
  "text-to-image"; "concept mixing" with "text-to-image"; "subject mixing"; hallucination with "text-to-image",
  benchmark and objects; "multi-subject" with benchmark and "text-to-image"; "identity mixing"; "attribute
  leakage" with benchmark; "semantic leakage" with benchmark; duplicate with "text-to-image" and objects. Two web
  searches: one for a benchmark rate of extra duplicated subjects on two-subject prompts, and one for a
  multi-subject leakage benchmark on SDXL with a VLM judge. Both web searches reported no such benchmark.
- Read in this session: the GenEval PDF's scoring rule (p. 5), FineGRAIN's failure-mode list through a fetch
  of its HTML, and the abstracts of the seven benchmark papers in the last row. All 15 arXiv ids in the table were
  confirmed against their titles on the arXiv API.
