"""One spec per claim group. Read by draw_review_figures.py.

A panel's crop runs from the line holding `from` to the line holding `to`. A mark boxes the
words of `phrase` (one box per line), or the lines from `lines_from` to `lines_to`, or an
explicit `box` in 200 dpi page pixels where the source has no text layer. Crop names follow
<source>-p<page>-<claim>.
"""

K4ND_REPLY = "replies/k4nd-and-uhz6.md"
PRZE_REPLY = "replies/prze.md"

G7 = {
    "group": "G7",
    "file": "g07-the-training-loss-and-what-the-adapter-returns",
    "title": "G7: The training loss and what the adapter returns",
    "subtitle": "k4nd weakness 4 (two parts) and question 4  ·  status: answered",
    "columns": {
        "review": [
            {
                "id": "v1", "title": "k4nd, weakness 4",
                "src": {"kind": "ocr", "source": "reviews", "page": 5},
                "crop": {"from": "4. Essential method details are poorly presented",
                         "to": "although the actual training diagram is Figure 8"},
                "crop_name": "reviews-p5-k4nd-W4a-W4b",
                "marks": [
                    {"key": "loss", "phrase": "it does not explicitly formulate the training loss.",
                     "label": "K-W4a"},
                    {"key": "sq", "phrase": "The squared-error objective appears in the training diagram in Appendix A, Figure 8.",
                     "label": "K-W4a"},
                    {"key": "ret", "phrase": "individual adapted network evaluations as returning the correction,",
                     "label": "K-W4b"},
                ],
            },
            {
                "id": "v2", "title": "k4nd, question 4",
                "src": {"kind": "ocr", "source": "reviews", "page": 5},
                "crop": {"from": "Can the authors state the optimization objective",
                         "to": "and baseline starting noise?"},
                "crop_name": "reviews-p5-k4nd-Q4",
                "marks": [
                    {"key": "q4", "phrase": "state the optimization objective explicitly in the main text",
                     "label": "K-Q4"},
                ],
            },
        ],
        "submitted": [
            {
                "id": "s1", "title": "Section 4, opening paragraph",
                "src": {"kind": "pdf", "source": "submitted", "page": 7},
                "crop": {"from": "We train a LoRA module", "to": "correction term as shown in Figure"},
                "crop_name": "submitted-p7-k4nd-W4b",
                "marks": [
                    {"key": "ret", "phrase": "returns the predicted correction defined in Equation 9.",
                     "label": "returns the correction"},
                ],
            },
            {
                "id": "s2", "title": "Section 4, how the adapter is trained",
                "src": {"kind": "pdf", "source": "submitted", "page": 7},
                "crop": {"from": "to the frozen model’s guided prediction under the",
                         "to": "training target and the prediction used during sampling."},
                "crop_name": "submitted-p7-k4nd-W4a",
                "marks": [
                    {"key": "noloss", "phrase": "Only the adapter parameters ψ are updated.",
                     "label": "no loss written"},
                    {"key": "lam", "lines_from": "pling, the learned correction is", "lines_to": "(12)",
                     "full": True, "pad_bottom": 40, "label": "applied as a correction at strength λ"},
                ],
            },
            {
                "id": "s3", "title": "Appendix A, Figure 8",
                "src": {"kind": "pdf", "source": "submitted", "page": 12},
                "crop": {"from": "Figure 8 shows how these runs are trained.",
                         "to": "the adapted network produces the composition matched to it."},
                "crop_name": "submitted-p12-k4nd-W4a",
                "marks": [
                    {"key": "diag", "box": (1205, 688, 1397, 1084),
                     "label": "the squared error"},
                ],
            },
        ],
        "revised": [
            {
                "id": "r1", "title": "Section 4, opening paragraph",
                "src": {"kind": "pdf", "source": "revised", "page": 7},
                "crop": {"from": "Once trained, the module is used at",
                         "to": "and never the joint prompt."},
                "crop_name": "revised-p7-k4nd-W4b",
                "marks": [
                    {"key": "noise", "phrase": "returns a noise prediction, as the frozen model does.",
                     "label": "returns a noise prediction"},
                    {"key": "inf", "phrase": "we run the usual product-of-experts step on",
                     "label": "inference: PoE, adapter on"},
                ],
            },
            {
                "id": "r2", "title": "Section 4, Equation 12",
                "src": {"kind": "pdf", "source": "revised", "page": 8},
                "crop": {"from": "Only the adapter parameters",
                         "to": "Appendix A gives the training and sampling details."},
                "crop_name": "revised-p8-k4nd-W4a-Q4",
                "marks": [
                    {"key": "eq12", "lines_from": "Only the adapter parameters",
                     "lines_to": "over the cached states of every training pair",
                     "full": True, "label": "Equation 12, the training loss"},
                    {"key": "impl", "lines_from": "The joint prompt’s prediction is the product’s prediction plus",
                     "lines_to": "on adds it.", "full": True, "label": "trains the adapter to add rₜ"},
                ],
            },
        ],
    },
    "arrows": [
        {"from": ("v1", "ret"), "to": ("s1", "ret"), "label": "this sentence", "role": "review"},
        {"from": ("v1", "loss"), "to": ("s2", "noloss"), "label": "no loss here", "role": "review"},
        {"from": ("v1", "sq"), "to": ("s3", "diag"), "label": "the diagram", "role": "review"},
        {"from": ("s1", "ret"), "to": ("r1", "noise"), "label": "now reads"},
        {"from": ("s2", "lam"), "to": ("r1", "inf"), "label": "replaced by"},
        {"from": ("s2", "noloss"), "to": ("r2", "eq12"), "label": "Equation 12 added"},
        {"from": ("s3", "diag"), "to": ("r2", "eq12"), "label": "stated in Section 4 instead"},
    ],
    "reply": {
        "title": "The reply we plan to send (replies/k4nd-and-uhz6.md: k4nd W4, its first two parts, and Q4)",
        "file": K4ND_REPLY,
        "sections": [
            "W4: method details > The training loss is not stated in Section 4",
            "W4: method details > Evaluations described as returning the correction",
            "Q4: the objective in the main text",
        ],
        "status": "answered",
        "status_color": "#1e8449",
    },
    "footer": [
        "Changes behind this page: 8ef00a17 Section 4 states the training loss as Equation 12  ·  88f02a45 and ead813ab each adapted evaluation returns a noise prediction and inference is the usual PoE step with the adapter on  ·  2e9af75c the training diagram is hidden.",
        "Question 4 also asks about figure references and starting noise; those are shown on the G9 and G8 pages. The reviewer's quotations are left out of the reply panel because the left column shows them.",
    ],
}




# ---------------------------------------------------------------- helpers for the other groups

UHZ6_REPLY = K4ND_REPLY  # both reviewers' replies live in one file


def mk(key, label, phrase=None, **kw):
    m = {"key": key, "label": label, **kw}
    if phrase:
        m["phrase"] = phrase
    return m


def panel(pid, title, source, page, frm, to, name, marks, **crop):
    kind = "ocr" if source == "reviews" else "pdf"
    return {"id": pid, "title": title, "src": {"kind": kind, "source": source, "page": page},
            "crop": {"from": frm, "to": to, **crop}, "crop_name": name, "marks": marks}


def note(pid, title, text, keys, height=300):
    """A dashed panel for something the revision does not contain or a result not landed yet."""
    return {"id": pid, "title": title, "pending": text, "height": height,
            "src_note": "no crop: this panel states what the claim table records",
            "marks": [{"key": k} for k in keys]}


def spec(group, file, title, subtitle, review, submitted, revised, arrows, reply, footer,
         status="answered", status_color="#1e8449"):
    return {"group": group, "file": file, "title": title, "subtitle": subtitle,
            "columns": {"review": review, "submitted": submitted, "revised": revised},
            "arrows": arrows,
            "reply": {"title": reply[0], "file": reply[1], "sections": reply[2],
                      "status": status, "status_color": status_color},
            "footer": footer}


AMBER = "#b9770e"
SOURCES_LINE = "Every excerpt is a crop of a page render; the file and page are printed under it. Boxes mark text already on the page."

G1 = spec(
    "G1", "g01-what-the-residual-measures", "G1: What the residual measures",
    "k4nd weakness 1 and question 1  ·  status: answered",
    [panel("v1", "k4nd, weakness 1", "reviews", 4,
           "1. The proposed residual does not establish", "therefore exceed the evidence.",
           "reviews-p4-k4nd-W1", [
               mk("a", "K-W1a", "specifically measures whether the named concepts appear as distinct objects."),
               mk("b", "K-W1b", "the full correction recovers the joint-prompt prediction by construction,"),
               mk("c", "K-W1b", "The stronger claims about measuring and learning an abstract concept therefore exceed the evidence."),
           ]),
     panel("v2", "k4nd, question 1", "reviews", 5,
           "What evidence establishes that the residual", "joint and separate conditioning?",
           "reviews-p5-k4nd-Q1", [mk("q1", "K-Q1", "specifically measures plurality rather than the broader difference")])],
    [panel("s1", "Introduction, the contributions", "submitted", 2,
           "as “plurality” and define this concept", "a pretrained LDM such as Stable Diffusion XL.",
           "submitted-p2-k4nd-W1a", [
               mk("meas", "measures plurality itself", "We measure the abstract concept of plurality"),
           ]),
     panel("s2", "Section 3, the plurality term", "submitted", 4,
           "We define the plurality term", "the strength of the correction with",
           "submitted-p4-k4nd-W1b", [
               mk("eq9", "whole difference, named plurality", lines_from="We define the plurality term", lines_to="diction and evaluating both", full=True),
           ])],
    [panel("r1", "Section 3, what the term is", "revised", 5,
           "We define the plurality term", "The training pairs are pairs the product",
           "revised-p5-k4nd-W1", [mk("whole", "the whole interaction", "is the whole interaction that the product drops:"),
                                  mk("scope", "scoped to plurality", "This paper studies one part of it, plurality:")]),
     panel("r2", "Introduction, the contributions", "revised", 3,
           "We measure, at every denoising step", "plurality on concept pairs it never trained on.",
           "revised-p3-k4nd-W1", [mk("contrib", "claims only what is measured", lines_from="We measure, at every denoising step", lines_to="plurality on concept pairs it never trained on.", full=True)])],
    [{"from": ("v1", "a"), "to": ("s1", "meas"), "label": "this claim", "role": "review"},
     {"from": ("v1", "b"), "to": ("s2", "eq9"), "label": "by construction", "role": "review"},
     {"from": ("s1", "meas"), "to": ("r2", "contrib"), "label": "rewritten"},
     {"from": ("s2", "eq9"), "to": ("r1", "whole"), "label": "now named for what it is"}],
    ("The reply we plan to send (replies/k4nd-and-uhz6.md: k4nd W1 and Q1)", K4ND_REPLY,
     ["W1: the residual does not establish", "Q1: evidence that the residual"]),
    ["Changes behind this page: Section 3 says r_t is the whole interaction the product drops and scopes the paper to plurality; the contributions claim only what the experiments show.", SOURCES_LINE],
)

G2 = spec(
    "G2", "g02-what-is-new-in-the-method", "G2: What is new in the method",
    "k4nd weakness 2  ·  status: partly answered (the supervision ablation is future work)",
    [panel("v1", "k4nd, weakness 2", "reviews", 4,
           "2. The methodological novelty is insufficiently established.", "require the proposed plurality formulation.",
           "reviews-p4-k4nd-W2", [
               mk("std", "K-W2", "beyond supervised prediction matching with a standard adapter."),
               mk("abl", "K-W2", "Stronger comparisons and ablations are needed"),
           ])],
    [panel("s1", "Introduction, the contributions", "submitted", 3,
           "We show that adding a plurality term to PoE", "scenes through compositional generalisation",
           "submitted-p3-k4nd-W2", [mk("c", "framed as a new concept", lines_from="We show that adding a plurality term", lines_to="scenes through compositional generalisation", full=True)])],
    [panel("r1", "Section 2, prior corrections", "revised", 5,
           "These methods keep the product as the target", "plurality emerges on concept pairs the adapter never trained on",
           "revised-p5-k4nd-W2", [mk("tgt", "what is new: the target", "we change the target instead,")]),
     note("r2", "Not in the revision",
          "The ablation k4nd asks for (the same joint-prompt supervision on pairs\nthe product does not fuse) was not run. The reply names it as future work.",
          ["none"], height=220)],
    [{"from": ("v1", "std"), "to": ("s1", "c"), "label": "the novelty claimed", "role": "review"},
     {"from": ("s1", "c"), "to": ("r1", "tgt"), "label": "stated as"},
     {"from": ("s1", "c"), "to": ("r2", "none"), "label": "not run", "role": "submitted"}],
    ("The reply we plan to send (replies/k4nd-and-uhz6.md: k4nd W2)", K4ND_REPLY, ["W2: methodological novelty"]),
    ["Changes behind this page: Section 2 says prior corrections keep the product as the target while this paper changes the target.", SOURCES_LINE],
    status="partly answered", status_color=AMBER,
)

G3 = spec(
    "G3", "g03-how-often-the-adapter-works-on-unseen-pairs", "G3: How often the adapter works on unseen pairs",
    "k4nd weakness 3 and question 2, UHz6 weakness 1 and question 2  ·  status: answered",
    [panel("v1", "k4nd, weakness 3", "reviews", 4,
           "3. The learned adapter lacks systematic quantitative evaluation.", "estimates and image-quality evaluation.",
           "reviews-p4-k4nd-W3", [
               mk("agg", "K-W3a", "There are no aggregate benchmark results"),
               mk("orc", "K-W3b", "these concern an oracle correction derived from the joint prompt"),
               mk("rate", "K-W3c", "The paper needs aggregate success and failure rates"),
           ]),
     panel("v2", "k4nd, weakness 3 (continued)", "reviews", 5,
           "The manuscript states that evaluation concepts", "generally beyond joint prompting.",
           "reviews-p5-k4nd-W3d", [mk("split", "K-W3d", "the complete splits and aggregate results needed to assess this claim are missing.")]),
     panel("v3", "UHz6, weakness 1 and question 2", "reviews", 3,
           "The trained adapter, which serves as the main contribution", "it is difficult to evaluate its generalization.",
           "reviews-p3-uhz6-W1a", [mk("qual", "U-W1a", "is only evaluated qualitatively.")]),
     panel("v4", "UHz6, question 2", "reviews", 3,
           "What is the success rate of the adapter", "What is the success rate of the adapter",
           "reviews-p3-uhz6-Q2", [mk("q2", "U-Q2", "What is the success rate of the adapter compared with PoE and CO3")])],
    [panel("s1", "Figure 4, the only counts", "submitted", 6,
           "Figure 4: Composition survives only while", "Figure 4: Composition survives only while",
           "submitted-p6-k4nd-W3b", [mk("f4", "oracle correction, one pair", lines_from="Figure 4: Composition survives only while", lines_to="Figure 4: Composition survives only while")],
           figure_above=True),
     panel("s2", "Section 4, the held-out claim", "submitted", 7,
           "pair where joint prompting cannot even produce", "also excluded from training.",
           "submitted-p7-k4nd-W3d", [mk("nei", "no concept shared?", "Neither concept in a held-out pair appears in any train-")],
           x=(840, 1415))],
    [panel("r1", "Table 1, Attend-and-Excite prompts", "revised", 9,
           "Table 1: Composition and image quality", "Ours",
           "revised-k4nd-W3-table1", [mk("t1", "every image scored", lines_from="Table 1: Composition and image quality", lines_to="Table 1: Composition and image quality")]),
     panel("r2", "Figure 11, failure types counted", "revised", 17,
           "Figure 11: Failure types as the adapter", "Figure 11: Failure types as the adapter",
           "revised-p17-k4nd-W3c", [mk("f10", "failures counted", lines_from="Figure 11: Failure types as the adapter", lines_to="Figure 11: Failure types as the adapter")],
           figure_above=True),
     panel("r3", "Section 4.1, the protocol", "revised", 8,
           "We evaluate one adapter, chosen once by a fixed rule", "We evaluate one adapter, chosen once by a fixed rule",
           "revised-p8-k4nd-W3a", [mk("fix", "fixed before scoring", "chosen once by a fixed rule before any held-out score was read,")])],
    [{"from": ("v1", "orc"), "to": ("s1", "f4"), "label": "these counts", "role": "review"},
     {"from": ("v2", "split"), "to": ("s2", "nei"), "label": "this claim", "role": "review"},
     {"from": ("s1", "f4"), "to": ("r1", "t1"), "label": "learned adapter, 276 prompts"},
     {"from": ("s1", "f4"), "to": ("r2", "f10"), "label": "failure rates"},
     {"from": ("s2", "nei"), "to": ("r3", "fix"), "label": "protocol stated"}],
    ("The reply we plan to send (replies/k4nd-and-uhz6.md: k4nd W3 and Q2, UHz6 Q2)", K4ND_REPLY,
     ["W3: no systematic quantitative", "Q2: aggregate results", "Q2: success rate against"]),
    ["Changes behind this page: Table 1 scores every method on the 276 Attend-and-Excite prompts, Figure 11 counts failure types on 53 held-out pairs, and Section 4.1 opens with the protocol.",
     "UHz6 weakness 1 also asks for standard benchmarks; that half is on the G4 page. The reply's anonymous code link is still a TODO. " + SOURCES_LINE],
)

G4 = spec(
    "G4", "g04-standard-benchmarks-and-metrics", "G4: Standard benchmarks and metrics",
    "PrzE weakness 1 and question 1, UHz6 weakness 1  ·  status: answered",
    [panel("v1", "PrzE, weakness 1", "reviews", 2,
           "Lack of quantitative benchmarks:", "insufficient for the claim in this paper.",
           "reviews-p2-prze-W1", [mk("w1", "P-W1", "The paper does not report a single quantitative metric.")]),
     panel("v2", "PrzE, question 1", "reviews", 2,
           "1. Quantitative Benchmark:", "compare your method with other standard baselines?",
           "reviews-p2-prze-Q1", [mk("q1", "P-Q1", "what are the evaluation scores for the proposed method?")]),
     panel("v3", "UHz6, weakness 1", "reviews", 3,
           "In addition to the examples being limited", "such as T2I-CompBench or GenEval counting.",
           "reviews-p3-uhz6-W1b", [mk("bench", "U-W1b", "there are no standardized benchmark evaluations")])],
    [panel("s1", "Section 4.1, the results", "submitted", 7,
           "Figure 5 shows the results of our method compared", "the null prompt where the LoRA is active",
           "submitted-p7-prze-W1", [mk("grid", "results are image grids only", "Figure 5 shows the results of our method compared to joint prompting, PoE and CO3")])],
    [panel("r1", "Table 1, Attend-and-Excite prompts", "revised", 9,
           "Table 1: Composition and image quality", "Ours",
           "revised-p9-prze-W1", [mk("t1", "CLIP, BLIP-VQA, ImageReward", lines_from="Table 1: Composition and image quality", lines_to="Table 1: Composition and image quality")]),
     panel("r2", "Table 2, GenEval and T2I-CompBench", "revised", 15,
           "Table 2: Standard benchmarks on their own prompts", "Ours",
           "revised-prze-W1-table2", [mk("t2", "GenEval, T2I-CompBench", lines_from="Table 2: Standard benchmarks on their own prompts", lines_to="Table 2: Standard benchmarks on their own prompts")])],
    [{"from": ("v1", "w1"), "to": ("s1", "grid"), "label": "no metric", "role": "review"},
     {"from": ("s1", "grid"), "to": ("r1", "t1"), "label": "scored"},
     {"from": ("s1", "grid"), "to": ("r2", "t2"), "label": "benchmarks"}],
    ("The reply we plan to send (replies/prze.md: W1 and Q1)", PRZE_REPLY,
     ["W1: lack of quantitative benchmarks", "Q1: quantitative benchmark"]),
    ["Table 2 scores GenEval two-object, attribute binding and position, and T2I-CompBench colour and shape, each with its official evaluator.",
     "The PrzE Q1 reply says \"Tables 1 to 4\"; the revision has three tables. UHz6's reply to weakness 1 is on the G3 page. " + SOURCES_LINE],
    
)

G5 = spec(
    "G5", "g05-baselines-beyond-co3", "G5: Baselines beyond CO3",
    "PrzE weakness 2, UHz6 weakness 3  ·  status: partly answered (no number for the Langevin corrector)",
    [panel("v1", "PrzE, weakness 2", "reviews", 2,
           "Narrow baselines and Missing details:", "without the corrector they mention.",
           "reviews-p2-prze-W2a", [mk("mcmc", "P-W2a", "the paper doesn't provide other corrected methods such as MCMC corrector.")]),
     panel("v2", "UHz6, weakness 3", "reviews", 3,
           "There are no quantitative comparisons with the compositional samplers", "are not discussed.",
           "reviews-p3-uhz6-W3ab", [
               mk("samp", "U-W3a", "There are no quantitative comparisons with the compositional samplers"),
               mk("att", "U-W3b", "Attention-based methods that address missing or merged objects"),
           ])],
    [panel("s1", "Section 4.1, the results", "submitted", 7,
           "Figure 5 shows the results of our method compared", "uses the joint prompt).",
           "submitted-p7-prze-W2a", [mk("co3", "CO3 the only baseline", "compared to joint prompting, PoE and CO3")])],
    [panel("r1", "Figure 5, six columns", "revised", 8,
           "Figure 5: The adapter repairs the composition", "every other column uses DDIM",
           "revised-p8-prze-W2a", [mk("cols", "ToMe, Langevin added", "product with a Langevin corrector,")],
           figure_above=True),
     panel("r2", "Section 2, attention-based methods", "revised", 5,
           "Attention-based methods such as Attend-and-Excite", "inference, only the separate concept prompts.",
           "revised-p5-uhz6-W3b", [mk("att", "now discussed", lines_from="Attention-based methods such as Attend-and-Excite", lines_to="inference, only the separate concept prompts.", full=True)]),
     note("r3", "Not in the revision",
          "The Langevin corrector appears in Figure 5 only,\nwith no number in a table.",
          ["tome"], height=230)],
    [{"from": ("v1", "mcmc"), "to": ("s1", "co3"), "label": "only CO3", "role": "review"},
     {"from": ("s1", "co3"), "to": ("r1", "cols"), "label": "more columns"},
     {"from": ("s1", "co3"), "to": ("r2", "att"), "label": "discussed"},
     {"from": ("s1", "co3"), "to": ("r3", "tome"), "label": "no number", "role": "submitted"}],
    ("The reply we plan to send (replies/prze.md: W2; replies/k4nd-and-uhz6.md: UHz6 W3)", PRZE_REPLY,
     ["W2: narrow baselines"]),
    ["UHz6's weakness 3 reply (replies/k4nd-and-uhz6.md) says the same: CO3 in Tables 1 and 3, a Langevin corrector in Figure 5, Attend-and-Excite and CONFORM in Section 2.",
     SOURCES_LINE],
    status="partly answered", status_color=AMBER,
)

G6 = spec(
    "G6", "g06-the-training-set-and-the-co3-filter", "G6: The training set and the CO3 filter",
    "PrzE weakness 2, UHz6 weakness 2 and question 1, k4nd question 3  ·  status: answered",
    [panel("v1", "UHz6, weakness 2", "reviews", 3,
           "For reproducibility concerns, the training set", "supervise toward a target without plurality.",
           "reviews-p3-uhz6-W2", [
               mk("det", "U-W2a", "the training set used for adapter training is not provided in sufficient detail."),
               mk("flt", "U-W2b", "the training filter keeps a pair if either the joint prompt or CO3 succeeds,"),
           ]),
     panel("v2", "UHz6, question 1", "reviews", 3,
           "Could the authors list the 48 training pairs", "Could the authors list the 48 training pairs",
           "reviews-p3-uhz6-Q1", [mk("q1", "U-Q1", "Could the authors list the 48 training pairs and the held-out pairs,")]),
     panel("v3", "k4nd, question 3", "reviews", 5,
           "Training pairs are retained when either joint", "How are pairs handled when only CO3 succeeds?",
           "reviews-p5-k4nd-Q3", [mk("q3", "K-Q3", "How are pairs handled when only CO3 succeeds?")]),
     panel("v4", "PrzE, weakness 2", "reviews", 2,
           "The 48 concept pairs are not listed", "The 48 concept pairs are not listed",
           "reviews-p2-prze-W2b", [mk("list", "P-W2b", "The 48 concept pairs are not listed")])],
    [panel("s1", "Section 4, the training pool", "submitted", 7,
           "More formally, we construct the training pool", "the generalisability of our LoRA module.",
           "submitted-p7-uhz6-W2", [
               mk("either", "either image passes", "and retain the pair if ei-"),
               mk("48", "48 pairs, none named", "leaves 48 concept pairs which demonstrate plu-"),
           ], x=(840, 1415))],
    [panel("r1", "Section 4, the training set", "revised", 8,
           "48 pairs (28 of two animals", "Seeds 9 to 16 are never trained on.",
           "revised-p8-uhz6-W2", [
               mk("tgt", "every target is the joint prompt", "every target is the frozen model’s joint-prompt prediction."),
               mk("co3", "CO3 a second view only", "no CO3 picture is a training target."),
           ]),
     panel("r2", "Figure 7, screening with CO3 beside", "revised", 13,
           "Figure 7: Screening candidate pairs with the joint prompt", "Figure 7: Screening candidate pairs with the joint prompt",
           "revised-p13-uhz6-W2", [mk("f7", "joint prompt picks the pair", lines_from="Figure 7: Screening candidate pairs", lines_to="Figure 7: Screening candidate pairs")],
           figure_above=True)],
    [{"from": ("v1", "flt"), "to": ("s1", "either"), "label": "this filter", "role": "review"},
     {"from": ("v4", "list"), "to": ("s1", "48"), "label": "not listed", "role": "review"},
     {"from": ("s1", "either"), "to": ("r1", "co3"), "label": "now"},
     {"from": ("s1", "48"), "to": ("r1", "tgt"), "label": "described"},
     {"from": ("s1", "either"), "to": ("r2", "f7"), "label": "shown"}],
    ("The reply we plan to send (replies/k4nd-and-uhz6.md: UHz6 training filter and Q1, k4nd Q3)", K4ND_REPLY,
     ["The training filter", "Q1: the 48 training pairs", "Q3: pairs retained when only CO3"]),
    ["The 48 pairs are listed in the UHz6 Q1 reply and released with the code; the paper describes them by kind (28 two animals, 3 animal and object, 17 two objects).", SOURCES_LINE],
)

G8 = spec(
    "G8", "g08-the-same-starting-noise-for-every-method", "G8: The same starting noise for every method",
    "k4nd weakness 5  ·  status: answered",
    [panel("v1", "k4nd, weakness 5", "reviews", 5,
           "5. The qualitative baseline comparison contains", "over multiple seeds is necessary.",
           "reviews-p5-k4nd-W5", [
               mk("cap", "K-W5", "The Figure 5 caption states that CO3 uses different starting noise,"),
               mk("sec", "K-W5", "while Section 4.1 states that all examples use the same initial seeds."),
           ])],
    [panel("s1", "Figure 5 caption", "submitted", 7,
           "Figure 5: The adapter repairs the composi-", "columns share it.",
           "submitted-p7-k4nd-W5-caption", [mk("cap", "CO3 different noise", "CO3 uses different starting noise;")],
           x=(180, 845)),
     panel("s2", "Section 4.1", "submitted", 8,
           "uncommon pairing. Importantly, all examples", "column shows the corresponding product-of-experts",
           "submitted-p8-k4nd-W5-text", [mk("same", "same seeds", "all examples are generated from the same initial seeds.")])],
    [panel("r1", "Figure 5 caption", "revised", 8,
           "Figure 5: The adapter repairs the composition", "every other column uses DDIM",
           "revised-p8-k4nd-W5-caption", [mk("lat", "same latent, every column", "every column starts from the same latent.")]),
     panel("r2", "Section 4.1, the protocol", "revised", 8,
           "We evaluate one adapter, chosen once by a fixed rule", "CO3 starts from the same latent but draws its later",
           "revised-p8-k4nd-W5-text", [mk("dag", "CO3 from the same latent", "CO3 starts from the same latent")]),
     panel("r3", "Section 4.1, continued", "revised", 9,
           "resampling noise from the seed, so its column", "resampling noise from the seed, so its column",
           "revised-p9-k4nd-W5-dagger", [mk("dg", "CO3 column marked †", "resampling noise from the seed, so its column is marked")])],
    [{"from": ("v1", "cap"), "to": ("s1", "cap"), "label": "this caption", "role": "review"},
     {"from": ("v1", "sec"), "to": ("s2", "same"), "label": "this sentence", "role": "review"},
     {"from": ("s1", "cap"), "to": ("r1", "lat"), "label": "now"},
     {"from": ("s2", "same"), "to": ("r2", "dag"), "label": "now"}],
    ("The reply we plan to send (replies/k4nd-and-uhz6.md: k4nd W5)", K4ND_REPLY,
     ["W5: the qualitative baseline comparison"]),
    ["k4nd's question 4 also asks about starting noise; it is answered here. " + SOURCES_LINE],
)

G9 = spec(
    "G9", "g09-a-missing-figure-a-broken-sentence-repeated-claims", "G9: A missing figure, a broken sentence, repeated claims",
    "UHz6 weakness 4, k4nd weakness 4 (last two parts)  ·  status: answered",
    [panel("v1", "UHz6, weakness 4", "reviews", 3,
           "There are presentation issues:", "in Sec. 5).",
           "reviews-p3-uhz6-W4", [
               mk("f10", "U-W4a", "Sec. 4 refers to a Figure 10 that does not exist,"),
               mk("brk", "U-W4b", "several sentences are broken"),
           ]),
     panel("v2", "k4nd, weakness 4 (continued)", "reviews", 5,
           "Section 4 refers to", "unnecessarily difficult to assess.",
           "reviews-p5-k4nd-W4cd", [
               mk("f10", "K-W4c", "Section 4 refers to"),
               mk("rep", "K-W4d", "The repetitive claims about learning an abstract concept"),
           ])],
    [panel("s1", "Section 4, opening paragraph", "submitted", 7,
           "the prompt tokens. The module is trained", "correction term as shown in Figure",
           "submitted-p7-uhz6-W4a", [
               mk("abs", "abstract notion of plurality", "learns the abstract notion of plurality,"),
               mk("f10", "Figure 10 does not exist", "correction term as shown in Figure"),
           ]),
     panel("s2", "Discussion", "submitted", 9,
           "In this work we introduce the concept of plurality", "dividual objects being prompted for.",
           "submitted-p9-uhz6-W4b", [mk("brk", "broken sentence", "inside create hybrid objects")])],
    [panel("r1", "Section 4, opening paragraph", "revised", 7,
           "Once trained, the module is used at", "and never the joint prompt.",
           "revised-p7-uhz6-W4a", [mk("open", "no Figure 10, no abstract notion", lines_from="Once trained, the module is used at", lines_to="and never the joint prompt.", full=True)]),
     panel("r2", "Discussion", "revised", 10,
           "similar compositional diffusion methods are unable to create images", "instead create hybrid objects",
           "revised-p10-uhz6-W4b", [mk("fix", "sentence repaired", "instead create hybrid objects")])],
    [{"from": ("v1", "f10"), "to": ("s1", "f10"), "label": "this reference", "role": "review"},
     {"from": ("v2", "rep"), "to": ("s1", "abs"), "label": "repeated claim", "role": "review"},
     {"from": ("v1", "brk"), "to": ("s2", "brk"), "label": "this sentence", "role": "review"},
     {"from": ("s1", "f10"), "to": ("r1", "open"), "label": "both removed"},
     {"from": ("s2", "brk"), "to": ("r2", "fix"), "label": "now reads"}],
    ("The reply we plan to send (replies/k4nd-and-uhz6.md: UHz6 presentation issues, k4nd W4 third part)", K4ND_REPLY,
     ["Presentation issues", "W4: method details > Repetitive claims about learning an abstract concept"]),
    ["The repeated abstract-concept claims were cut from Sections 3 and 4 and the Discussion (06a9cf98); Section 4.1 opens with the evaluation protocol (508d2647). " + SOURCES_LINE],
)

G10 = spec(
    "G10", "g10-beyond-two-animals-on-sdxl", "G10: Beyond two animals on SDXL",
    "PrzE question 2, UHz6 weakness 3 (scope)  ·  status: partly answered (SDXL only)",
    [panel("v1", "PrzE, question 2", "reviews", 2,
           "2. Multiple concepts:", "results for more than two concepts?",
           "reviews-p2-prze-Q2", [mk("q2", "P-Q2", "How does your method perform when composing more than two concepts?")]),
     panel("v2", "UHz6, weakness 3 (scope)", "reviews", 3,
           "The study is also limited to SDXL", "and to pairs of two concepts, mostly animals.",
           "reviews-p3-uhz6-W3c", [mk("sc", "U-W3c", "The study is also limited to SDXL and to pairs of two concepts,")])],
    [panel("s1", "Section 4, the training pool", "submitted", 7,
           "More formally, we construct the training pool", "compose under a joint prompt.",
           "submitted-p7-prze-Q2", [mk("pairs", "pairs on SDXL only", "from concept pairs that Stable Diffusion XL can")],
           x=(840, 1415))],
    [panel("r1", "Figure 16, three concepts", "revised", 20,
           "Figure 16: Three concepts.", "Figure 16: Three concepts.",
           "revised-prze-Q2", [mk("f13", "one triple of each kind", lines_from="Figure 16: Three concepts.", lines_to="Figure 16: Three concepts.")],
           figure_above=True),
     panel("r2", "Discussion, three concepts", "revised", 10,
           "The adapter is trained on pairs, yet it carries", "are open questions.",
           "revised-p10-prze-Q2", [mk("disc", "stops at two animals", "When the triple has two or three animals it stops at two,")]),
     note("r3", "Not in the revision",
          "No model other than SDXL. The SD 1.5 results were removed on 2026-11-12.", ["sdxl"], height=200)],
    [{"from": ("v1", "q2"), "to": ("s1", "pairs"), "label": "pairs only", "role": "review"},
     {"from": ("s1", "pairs"), "to": ("r1", "f13"), "label": "three concepts"},
     {"from": ("s1", "pairs"), "to": ("r2", "disc"), "label": "discussed"},
     {"from": ("s1", "pairs"), "to": ("r3", "sdxl"), "label": "still SDXL", "role": "submitted"}],
    ("The reply we plan to send (replies/prze.md: Q2)", PRZE_REPLY, ["Q2: multiple concepts"]),
    ["Objects beyond animals are scored in Table 1 (animals with objects, objects); see G4. UHz6's weakness 3 reply is on the G5 page. " + SOURCES_LINE],
    status="partly answered", status_color=AMBER,
)

G11 = spec(
    "G11", "g11-the-adapter-on-the-early-steps-only", "G11: The adapter on the early steps only",
    "UHz6 question 3  ·  status: answered",
    [panel("v1", "UHz6, question 3", "reviews", 3,
           "Given the finding in Sec. 3", "Given the finding in Sec. 3",
           "reviews-p3-uhz6-Q3", [mk("q3", "U-Q3", "does restricting the adapter to the early steps change the failure modes in Fig. 7?")])],
    [panel("s1", "Figure 7, limitations", "submitted", 9,
           "Figure 7: Limitations of the Plurality LoRA:", "dependent loss of plurality (bottom left) or instability (top right).",
           "submitted-p9-uhz6-Q3", [mk("f7", "failures, no count", lines_from="Figure 7: Limitations of the Plurality LoRA:", lines_to="Figure 7: Limitations of the Plurality LoRA:")],
           figure_above=True)],
    [panel("r1", "Figure 11, failure types by window", "revised", 17,
           "Figure 11: Failure types as the adapter", "Figure 11: Failure types as the adapter",
           "revised-p17-uhz6-Q3", [mk("f10", "steps 0 to 10, 0 to 25, all 50", lines_from="Figure 11: Failure types as the adapter", lines_to="Figure 11: Failure types as the adapter")],
           figure_above=True),
     panel("r2", "Figure 13, the images behind it", "revised", 18,
           "Figure 13: The images behind Figure 11.", "Figure 13: The images behind Figure 11.",
           "revised-uhz6-Q3", [mk("f11", "the renders", lines_from="Figure 13: The images behind Figure 11.", lines_to="Figure 13: The images behind Figure 11.")],
           figure_above=True)],
    [{"from": ("v1", "q3"), "to": ("s1", "f7"), "label": "these failures", "role": "review"},
     {"from": ("s1", "f7"), "to": ("r1", "f10"), "label": "counted per window"},
     {"from": ("s1", "f7"), "to": ("r2", "f11"), "label": "shown per window"}],
    ("The reply we plan to send (replies/k4nd-and-uhz6.md: UHz6 Q3)", K4ND_REPLY, ["Q3: restricting the adapter"]),
    ["The submitted Figure 7 is Figure 14 in the revision. " + SOURCES_LINE],
)

G12 = spec(
    "G12", "g12-pairs-that-do-not-compete-for-one-region", "G12: Pairs that do not compete for one region",
    "UHz6 question 4  ·  status: answered in the reply",
    [panel("v1", "UHz6, question 4", "reviews", 3,
           "Does the adapter degrade PoE on pairs", "pairs like lighthouse and sea?",
           "reviews-p3-uhz6-Q4", [mk("q4", "U-Q4", "Does the adapter degrade PoE on pairs that do not compete for the same region,")])],
    [panel("s1", "Section 4.1, one lighthouse", "submitted", 8,
           "preserved as distinct entities", "reasonably well when there is a distinction",
           "submitted-p8-uhz6-Q4", [mk("lh", "one example, no count", "The lighthouse and sea appears structurally correct")])],
    [note("r1", "Answered in the reply",
          "The revision has no per-group table. The reply answers from the\n21 held-out pairs whose concepts do not compete for one region.", ["nc"], height=220)],
    [{"from": ("v1", "q4"), "to": ("s1", "lh"), "label": "one image", "role": "review"},
     {"from": ("s1", "lh"), "to": ("r1", "nc"), "label": "answered", "role": "submitted"}],
    ("The reply we plan to send (replies/k4nd-and-uhz6.md: UHz6 Q4)", K4ND_REPLY, ["Q4: pairs that do not compete"]),
    ["The held-out groups were removed from the paper; the hard, easy and not-competing split is named as future work in the Discussion. " + SOURCES_LINE],
    status="answered in the reply", status_color=AMBER,
)

SPECS = [G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12]
