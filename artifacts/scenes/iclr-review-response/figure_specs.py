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
                         "to": "(Appendix B). Appendix A gives the training and sampling"},
                "crop_name": "revised-p8-k4nd-W4a-Q4",
                "marks": [
                    {"key": "eq12", "lines_from": "Only the adapter parameters", "lines_to": "(12)",
                     "full": True, "label": "Equation 12, the training loss"},
                    {"key": "impl", "phrase": "The adapter is trained to reproduce the joint prompt’s prediction, not the correction.",
                     "label": "learns the joint prompt"},
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

SPECS = [G7]
