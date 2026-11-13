# Claims from the three reviews of submission 56173

Every claim the three reviewers make, quoted in their words, with its page in `paper/iclr-56173/Reviews.pdf` (PrzE on page 2, UHz6 on page 3, k4nd on pages 4 and 5). Each claim is mapped to where the submitted paper (`56173_Restoring_plurality_in_c.pdf`, 13 pages) is weak and to where the revision (`paper/iclr-56173/iclr2027_conference.tex`, built as a 20-page PDF) answers it. Reply sections are in `paper/iclr-56173/replies/k4nd-and-uhz6.md` and `paper/iclr-56173/replies/prze.md`.

Claim ids are the reviewer's initial, then W for a weakness or Q for a question, then the reviewer's own number (a letter splits one bullet into its separate claims). "Sub p." is a page of the submitted PDF, and "Rev p." a page of the revised PDF.

**Status** says whether the revision answers the claim: **answered** means it does in full, **pending** means it waits on a result still rendering, **partly** means part of it is answered and the rest is named, and **unanswered** means nothing in the revision answers it.

## The twelve groups

| # | Group | Claims | Status |
|---|---|---|---|
| G1 | What the residual measures | K-W1a, K-W1b, K-Q1 | answered |
| G2 | What is new in the method | K-W2 | partly |
| G3 | How often the adapter works on unseen pairs | K-W3a, K-W3b, K-W3c, K-W3d, K-Q2, U-W1a, U-Q2 | answered, one link pending |
| G4 | Standard benchmarks and metrics | P-W1, P-Q1, U-W1b | pending (Table 2) |
| G5 | Baselines beyond CO3 | P-W2a, U-W3a, U-W3b | partly, ToMe row pending |
| G6 | The training set and the CO3 filter | P-W2b, U-W2a, U-W2b, U-Q1, K-Q3 | answered |
| G7 | The training loss and what the adapter returns | K-W4a, K-W4b, K-Q4 | answered |
| G8 | The same starting noise for every method | K-W5 | answered |
| G9 | A missing figure, a broken sentence, repeated claims | U-W4a, U-W4b, K-W4c, K-W4d | answered |
| G10 | Beyond two animals on SDXL | P-Q2, U-W3c | partly |
| G11 | The adapter on the early steps only | U-Q3 | answered |
| G12 | Pairs that do not compete for one region | U-Q4 | answered, CO3 row pending |

35 claims: PrzE 5, UHz6 13, k4nd 17.

## Every claim

| Claim | Rev. page | Their words | Group | Where the submission is weak | Where the revision answers | Reply | Status |
|---|---|---|---|---|---|---|---|
| P-W1 | 2 | "The paper does not report a single quantitative metric. Standard benchmarks such as T2I-Compench, GenEval and standard metrics such as CLIP, Imagereward are completely absent" | G4 | No table anywhere in the 13 pages; every result is an image grid (Sub p. 5 to 9) | Table 1, BLIP-VQA, CLIP score and ImageReward on 276 Attend-and-Excite prompts (Rev p. 9, tex 681 to 686); Table 2, GenEval and T2I-CompBench (Rev p. 16, tex 1276 to 1281, slot R14) | prze W1 | pending: Table 2 |
| P-W2a | 2 | "There is only CO3 baseline compared, the paper doesn't provide other corrected methods such as MCMC corrector" | G5 | Figure 5 has four columns: joint prompt, plain product, CO3, ours (Sub p. 7, lines 356 to 360) | Figure 5 adds ToMe and a Langevin corrector (Rev p. 8, tex 674 to 678; text at tex 818) | prze W2 | answered |
| P-W2b | 2 | "The 48 concept pairs are not listed so it's impossible to see what the adapter is trained on" | G6 | "This filtering leaves 48 concept pairs", none named (Sub p. 7, lines 344 to 346) | Section 4 training-set paragraph (Rev p. 7, tex 706); Figure 6 shows nine targets (Rev p. 13); full list in the UHz6 Q1 reply | prze W2, UHz6 Q1 | answered |
| P-Q1 | 2 | "what are the evaluation scores for the proposed method? Can you show the results on some benchmarks and compare your method with other standard baselines?" | G4 | As P-W1 | Tables 1 to 3 (Rev p. 9 and 16) | prze Q1 | pending: Table 2 |
| P-Q2 | 2 | "How does your method perform when composing more than two concepts?" | G10 | Pairs only throughout | Appendix C.5, Figure 13 (Rev p. 20, tex 1348 to 1357); Discussion (Rev p. 10, tex 888) | prze Q2 | answered |
| U-W1a | 3 | "The trained adapter … is only evaluated qualitatively. In Fig. 5 and 6 … it is difficult to evaluate its generalization … examples being limited to similar entities (cat, dog)" | G3 | Figures 5 and 6 are image grids, Figure 6 is cat and dog only (Sub p. 7 and 8) | Table 3, 53 held-out pairs in three groups, seeds 9 to 16, 95% intervals (Rev p. 16, tex 1288 to 1296) | UHz6 W1 | answered |
| U-W1b | 3 | "there are no standardized benchmark evaluations such as T2I-CompBench or GenEval counting" | G4 | As P-W1 | Table 2 (Rev p. 16, slot R14) | UHz6 W1 | pending: Table 2 |
| U-W2a | 3 | "the training set used for adapter training is not provided in sufficient detail" | G6 | Sub p. 7, lines 338 to 349 | Section 4 training-set paragraph (Rev p. 7, tex 706); Appendix A (tex 1116) | UHz6 training filter, Q1 | answered |
| U-W2b | 3 | "the training filter keeps a pair if either the joint prompt or CO3 succeeds, but the target is always the joint prompt (Eq. 11)" | G6 | "retain the pair if either image contains both concepts" (Sub p. 7, lines 343 to 344) | Section 4: a pair enters only when the joint prompt draws both (Rev p. 7, tex 706); Figure 7, joint prompt beside CO3 (Rev p. 13, tex 1125 to 1129) | UHz6 training filter | answered |
| U-W3a | 3 | "There are no quantitative comparisons with the compositional samplers discussed in Sec. 2.2, and CO3 appears only as one qualitative column" | G5 | Sub p. 7, Figure 5 | Tables 1 and 3 score CO3 (Rev p. 9, 16); the Langevin corrector appears in Figure 5 only, with no number | UHz6 W3 | partly: other samplers have no numbers; ToMe row of Table 1 pending |
| U-W3b | 3 | "Attention-based methods … (e.g., Attend-and-Excite, CONFORM) target the same failure mode and are not discussed" | G5 | Neither is cited in the submission | Section 2 (Rev p. 5, tex 484) | UHz6 W3 | answered |
| U-W3c | 3 | "The study is also limited to SDXL and to pairs of two concepts, mostly animals" | G10 | Sub p. 7 | Objects in Table 1 (Rev p. 9); three concepts in Figure 13 (Rev p. 20); nothing beyond SDXL since SD 1.5 left the paper | UHz6 W3 | partly: SDXL only |
| U-W4a | 3 | "Sec. 4 refers to a Figure 10 that does not exist" | G9 | "as shown in Figure 10" (Sub p. 7, line 337) | Reference removed (Section 4, Rev p. 7, tex 702) | UHz6 presentation | answered |
| U-W4b | 3 | "several sentences are broken (e.g., "inside create hybrid objects" in Sec. 5)" | G9 | Sub p. 9, line 465 | Discussion reads "and instead create hybrid objects" (Rev p. 10, tex 886) | UHz6 presentation | answered |
| U-Q1 | 3 | "Could the authors list the 48 training pairs and the held-out pairs, and clarify whether cat/dog is in the training set?" | G6 | Sub p. 7 | The 48 pairs are listed in the reply; held-out groups in Appendix C.2 (Rev p. 16, tex 1285) | UHz6 Q1 | answered (pairs in the reply, not the paper) |
| U-Q2 | 3 | "What is the success rate of the adapter compared with PoE and CO3 over all held-out pairs and multiple seeds?" | G3 | As U-W1a | Table 3 (Rev p. 16) | UHz6 Q2 | answered |
| U-Q3 | 3 | "does restricting the adapter to the early steps change the failure modes in Fig. 7?" | G11 | Submitted Figure 7 (Sub p. 9, line 452) | Figure 10 counts failure types at windows 0 to 10, 0 to 25 and all 50 (Rev p. 17); Figure 11 shows the images (Rev p. 18) | UHz6 Q3 | answered |
| U-Q4 | 3 | "Does the adapter degrade PoE on pairs that do not compete for the same region, such as … lighthouse and sea?" | G12 | One lighthouse-and-sea image (Sub p. 8, line 384) | Table 3, not-competing group (Rev p. 16) | UHz6 Q4 | answered; CO3's row (slot R3) pending in the reply |
| K-W1a | 4 | "The proposed residual does not establish a distinct representation of plurality … does not establish that it specifically measures whether the named concepts appear as distinct objects" | G1 | Definition 1.1 (Sub p. 2, line 96); "We measure the abstract concept of plurality" (Sub p. 2, line 106); Eq. 9 (Sub p. 4, line 206) | Section 3: "$r_t$ is the whole interaction that the product drops" and the paper is scoped to plurality (Rev p. 5, tex 587); contributions rewritten (Rev p. 2, tex 221 to 223) | k4nd W1 | answered |
| K-W1b | 4 | "the full correction recovers the joint-prompt prediction by construction … The stronger claims about measuring and learning an abstract concept therefore exceed the evidence" | G1 | Eq. 10 (Sub p. 4) | As K-W1a | k4nd W1 | answered |
| K-W2 | 4 | "The methodological novelty is insufficiently established … Stronger comparisons and ablations are needed to identify which benefits arise from joint-prompt supervision" | G2 | Sub p. 3, contributions | Section 2: prior work keeps the product as target, we change the target (Rev p. 5, tex 484) | k4nd W2 | partly: the supervision-versus-pairs ablation is named as future work |
| K-W3a | 4 | "The principal results in Section 4.1 are qualitative … There are no aggregate benchmark results" | G3 | Sub p. 7 to 9 | Tables 1 and 3 (Rev p. 9, 16); protocol fixed before scoring (Rev p. 8, tex 794) | k4nd W3 | answered |
| K-W3b | 4 | "detector-based counts … in Figure 4 … concern an oracle correction … for one cat–dog pair and four seeds" | G3 | Figure 4 (Sub p. 6, line 314) | Table 3 scores the learned adapter on 53 pairs, 8 seeds | k4nd W3 | answered |
| K-W3c | 4 | "The paper needs aggregate success and failure rates … together with uncertainty estimates and image-quality evaluation" | G3 | Sub p. 9, Figure 7 shows failures without counts | Figure 10 counts failure types (Rev p. 17); 95% intervals in Table 3; CLIP score and ImageReward in Table 1 | k4nd W3 | answered |
| K-W3d | 5 | "the complete splits and aggregate results needed to assess this claim are missing" | G3 | "Neither concept in a held-out pair appears in any training pair" (Sub p. 7, lines 349 to 351) | Section 4 now says "unseen as pairs"; Appendix C.2 describes the held-out groups (Rev p. 16) | k4nd W3 | answered; the anonymous code link is a TODO in the reply |
| K-W4a | 5 | "it does not explicitly formulate the training loss. The squared-error objective appears in the training diagram in Appendix A, Figure 8" | G7 | Sub p. 7, lines 352 to 356; Figure 8 (Sub p. 12, line 621) | Equation 12 in Section 4 (Rev p. 8, tex 741 to 743) | k4nd W4 | answered |
| K-W4b | 5 | "The main text describes individual adapted network evaluations as returning the correction, whereas Appendix B defines them as adapted denoising predictions" | G7 | "returns the predicted correction defined in Equation 9" (Sub p. 7, line 334) | "returns a noise prediction, as the frozen model does" (Rev p. 7, tex 702) | k4nd W4 | answered |
| K-W4c | 5 | "Section 4 refers to "Figure 10," although the actual training diagram is Figure 8" | G9 | Sub p. 7, line 337 | Reference removed; diagram hidden (Rev p. 7) | k4nd W4 | answered |
| K-W4d | 5 | "The repetitive claims about learning an abstract concept occupy space that could instead clarify the objective, training data, evaluation protocol, and limitations" | G9 | Sub p. 2 bullet, Sub p. 7 lines 331 to 332, Sub p. 9 line 480 | Claims cut from Sections 3, 4 and the Discussion; Section 4.1 opens with the protocol (Rev p. 8, tex 794) | k4nd W4 | answered |
| K-W5 | 5 | "The Figure 5 caption states that CO3 uses different starting noise, while Section 4.1 states that all examples use the same initial seeds" | G8 | Caption, "CO3 uses different starting noise" (Sub p. 7, line 360); "all examples are generated from the same initial seeds" (Sub p. 8, line 381) | Figure 5 caption and Section 4.1: every column starts from the same latent, CO3 marked with a dagger (Rev p. 8, tex 677; Rev p. 8, tex 794) | k4nd W5 | answered |
| K-Q1 | 5 | "What evidence establishes that the residual in Equation (9) specifically measures plurality rather than the broader difference between joint and separate conditioning?" | G1 | As K-W1a | As K-W1a | k4nd Q1 | answered |
| K-Q2 | 5 | "Can the authors provide aggregate results … over documented held-out concepts and multiple seeds, including identity errors, additional objects, and style-dependent failures?" | G3 | As K-W3a | Tables 1 and 3; Figure 10 (Rev p. 17) | k4nd Q2 | answered |
| K-Q3 | 5 | "How are pairs handled when only CO3 succeeds?" | G6 | As U-W2b | As U-W2b | k4nd Q3 | answered |
| K-Q4 | 5 | "Can the authors state the optimization objective explicitly in the main text and resolve the inconsistencies concerning network outputs, figure references, and baseline starting noise?" | G7 | As K-W4a, K-W4b, K-W4c, K-W5 | Equation 12 (Rev p. 8); see K-W4b, K-W4c, K-W5 | k4nd Q4 | answered |

## Found while mapping, for Molefe to decide

**A misquote in the k4nd W4 reply.** The reply quotes k4nd as writing "Appendix A, Figure 7" and "the actual training diagram is Figure 7". paper/iclr-56173/Reviews.pdf page 5 says Figure 8 both times.

**A table that no longer exists.** The PrzE Q1 reply says "Tables 1 to 4", but Table 4 was removed with SD 1.5, so the revision has three tables.

**Groups that stay partly answered.** G2 leaves the supervision-versus-pairs ablation as future work. G5 has no numbers for samplers other than CO3. G10 is SDXL only.
