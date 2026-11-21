import sys, pymupdf

SRC = sys.argv[1]
DST = sys.argv[2]

BLUE = ((0.82, 0.90, 1.0), (0.12, 0.30, 0.65))    # answers a reviewer point
AMBER = ((1.0, 0.93, 0.74), (0.70, 0.45, 0.05))   # reviewer point plus our own improvement
GREY = ((0.90, 0.90, 0.90), (0.40, 0.40, 0.40))   # our own change, no reviewer

# (revised page, y_top of anchor, y_bottom of anchor, kind, text)
N = [
 (5, 673, 740, BLUE, "k4nd W1, W2\nConcern: the residual does not isolate plurality, and novelty is unclear.\nResponse: the contributions are rewritten to claim only what the experiments show."),
 (9, 480, 500, BLUE, "k4nd W2\nConcern: methodological novelty is insufficiently established.\nResponse: this paragraph now says earlier corrections keep the product as the target and change how it is sampled, while we change the target."),
 (9, 596, 612, BLUE, "UHz6 W3\nConcern: attention methods (Attend-and-Excite, CONFORM) target the same failure and are not discussed.\nResponse: added here. They adjust cross-attention under one joint prompt; our setting has none at inference."),
 (11, 417, 515, BLUE, "k4nd W1, Q1\nConcern: r_t may carry more than plurality.\nResponse: agreed. r_t is now stated to be the whole interaction the product drops (relations, effects, layout, style), and the paper is scoped to its plurality part."),
 (15, 357, 567, BLUE, "PrzE W1, W2, Q1; k4nd W3, Q2; UHz6 W1, W3, Q2\nConcern: no quantitative metrics, benchmarks or baselines beyond CO3.\nResponse: new table scores every method on 276 Attend-and-Excite prompts (BLIP-VQA, CLIP, ImageReward), with CO3 and ToMe as baselines, seeds 9 to 12."),
 (15, 592, 614, BLUE, "k4nd W4, Q4\nConcern: Section 4 and Appendix B disagree on what each network evaluation returns.\nResponse: each adapted evaluation now returns a noise prediction, and inference is the usual product-of-experts step with the adapter on."),
 (15, 621, 719, BLUE, "k4nd Q3; UHz6 training filter\nConcern: pairs kept when only CO3 succeeds would train toward a target without plurality.\nResponse: every target is the frozen joint-prompt prediction; CO3 was a second view during screening only (Figure 7)."),
 (17, 410, 500, BLUE, "k4nd W4, Q4\nConcern: the training loss appeared only in an appendix diagram.\nResponse: the squared-error objective is now Equation 12 in Section 4."),
 (17, 585, 640, BLUE, "k4nd W3, W4\nConcern: no systematic evaluation protocol.\nResponse: Section 4.1 opens with the protocol: one adapter chosen by a fixed rule before any held-out score was read, the same seed for every method."),
 (19, 549, 614, BLUE, "k4nd limitations; PrzE Q2; UHz6 W3\nConcern: only pairs, mostly animals. What about three concepts?\nResponse: new paragraph on Figure 16. The adapter separates every triple but stops at two when the triple has two or three animals."),
 (25, 632, 700, BLUE, "UHz6 Q1; PrzE W2\nConcern: the 48 training pairs are not listed, and is cat and dog trained on?\nResponse: counts by kind and examples are given here and Figure 6 shows nine targets. The full list is in the reply and released with the code; cat and dog is held out."),
 (27, 170, 202, AMBER, "PrzE W2; UHz6 Q1; our addition\nNew figure: the joint prompt's picture for nine of the 48 training pairs, so a reader can see what the adapter trains on."),
 (27, 510, 553, BLUE, "k4nd Q3; UHz6 training filter\nNew figure: the joint prompt beside CO3 on five candidate pairs, one green pick per seed. Crow and raven are dropped because the two birds are hard to tell apart."),
 (27, 657, 710, BLUE, "k4nd W4\nAppendix B now matches Section 4: the adapter gets c1, c2 and the null prompt in three separate passes, never the joint prompt."),
 (29, 466, 478, GREY, "New in this revision\nAppendix C holds what the reviews called for: benchmark prompts and image sheets, failure counts by adapter window, the judge's labels, limitations and three concepts."),
 (31, 262, 294, AMBER, "PrzE W1; our addition\nThe images behind Table 1, one prompt per set, so the scores can be checked against pictures."),
 (31, 307, 485, BLUE, "PrzE W1, Q1; UHz6 W1\nConcern: no standard benchmarks (GenEval, T2I-CompBench).\nResponse: Table 2 reports GenEval's two-object, attribute-binding and position tasks and T2I-CompBench's colour and shape sets, each with its own scorer."),
 (31, 500, 724, BLUE, "k4nd W3, Q2; UHz6 Q2, Q3; PrzE Q2\nConcern: aggregate results over held-out pairs and seeds, the failure types, and whether the early steps alone change them.\nResponse: new figure counts failure types over 53 held-out pairs at seeds 9 to 16 with the adapter off, or on for the first 10, 25 or all 50 steps."),
 (33, 415, 458, AMBER, "PrzE W1; our addition\nThe images behind Table 2, one prompt per column and one row per method."),
 (33, 481, 524, BLUE, "UHz6 Q3; k4nd Q2\nConcern: does restricting the adapter to early steps change the failures, including extra objects?\nResponse: hybrids fall with every step added, while extra objects rise and are the most common failure at 50 steps."),
 (33, 537, 671, AMBER, "k4nd Q2; our addition\nHow the failure labels were made: each image labelled once by a fixed-prompt Claude judge that was not told the setting. The prompt is given here and forty labels are in Figure 15."),
 (35, 511, 605, BLUE, "k4nd Q2; UHz6 Q2, Q3\nSame 53 held-out pairs and seeds as Figure 11, scored by a detector and ImageReward. With the adapter on for 25 steps the detector finds two animals as often as with 50; with 10 only, far less often."),
 (37, 327, 348, BLUE, "UHz6 Q3\nThe images behind Figure 11, so each count can be seen. Species substitution persists at every window, so it does not come from the late steps."),
 (39, 644, 665, AMBER, "k4nd Q2; our addition\nForty of the judge's labels, drawn at random from the 1,696 images behind Figure 11, so a reader can check the labelling."),
 (41, 105, 137, BLUE, "PrzE Q2; UHz6 W1, W3; k4nd limitations\nConcern: how does it do with more than two concepts?\nResponse: one triple of each kind at a shared seed. The product fuses or drops a concept; the adapter, trained on pairs only, separates them, but draws two when the triple has animals."),
]

FS = 5.6
X0, X1 = 507.0, 594.0
LEAD = 1.22
PAD = 3.0
helv = pymupdf.Font("helv")

def wrap(text, width):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split(" "):
            t = (cur + " " + w).strip()
            if helv.text_length(t, FS) <= width:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines

doc = pymupdf.open(SRC)
placed = {}
for pg_no, y0, y1, (fill, ink), text in N:
    page = doc[pg_no - 1]
    nlines = len(wrap(text, X1 - X0 - 2 * PAD - 2))
    h = nlines * FS * LEAD + 2 * PAD + 3
    top = y0
    for (a, b) in sorted(placed.get(pg_no, [])):
        if top < b + 3 and top + h > a - 3:
            top = b + 3
    placed.setdefault(pg_no, []).append((top, top + h))
    rect = pymupdf.Rect(X0, top, X1, top + h)
    a = page.add_freetext_annot(rect, text, fontsize=FS, fontname="helv",
                                text_color=ink, fill_color=fill)
    a.set_info(title="Reviewer response", subject="Revision note")
    a.update(fontsize=FS, fontname="helv", text_color=ink, fill_color=fill)
    # bracket on the anchored text, and a short tie to the note box
    bar = page.add_line_annot(pymupdf.Point(505.5, y0), pymupdf.Point(505.5, y1))
    bar.set_colors(stroke=ink); bar.set_border(width=1.6); bar.update()
    if top > y0 + 1 or True:
        tie = page.add_line_annot(pymupdf.Point(505.5, (y0 + y1) / 2), pymupdf.Point(X0, min(max(top + 6, 0), top + h - 4)))
        tie.set_colors(stroke=ink); tie.set_border(width=0.5); tie.update()

# cover page: legend and what is not in this comparison
cov = doc[0]
x = 56
y = 470
cov.insert_text((x, y), "Reviewer-response notes", fontsize=13, fontname="hebo")
y += 18
body = ("Notes sit in the right margin of the revised pages (the odd pages), beside the text each one is about. "
        "Each note names the reviewer and point, the concern, and how the revision responded. "
        "The bar on the text edge marks the passage; the note can be selected, moved or deleted like any PDF comment.")
cov.insert_textbox(pymupdf.Rect(x, y, 540, y + 50), body, fontsize=9, fontname="helv")
y += 56
for (fill, ink), label in ((BLUE, "A reviewer point the revision answers"),
                           (AMBER, "A reviewer point, plus an improvement we added to support it"),
                           (GREY, "Our own change; no reviewer asked for it")):
    cov.draw_rect(pymupdf.Rect(x, y, x + 26, y + 11), color=ink, fill=fill, width=0.6)
    cov.insert_text((x + 34, y + 9), label, fontsize=9, fontname="helv")
    y += 17
y += 6
note = ("Reviewers: PrzE, k4nd and UHz6 (submission 56173). Labels such as W1, Q2 are the weakness and question numbers "
        "in each review. Figure numbers are those of the revised paper in this comparison.\n"
        "Not in this comparison: UHz6 Q4 (pairs that do not compete for one region, such as lighthouse and sea). "
        "It is answered in the reply only, and the revised text has no matching passage.")
cov.insert_textbox(pymupdf.Rect(x, y, 540, y + 90), note, fontsize=9, fontname="helv")

doc.save(DST, garbage=3, deflate=True)
print("saved", DST, "notes", len(N))
