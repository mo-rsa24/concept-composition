"""Write review.tex: the ICLR 56173 reviews point by point, each with the reply and what changed.

One section per reviewer (k4nd, UHz6, PrzE). Under each point: the reviewer's words as quoted in
the reply files, the reply itself, and a note on what changed in the revision (CHANGES below).
The reviewer quotes and replies are read from paper/iclr-56173/replies/k4nd-and-uhz6.md and paper/iclr-56173/replies/prze.md, so the
document follows the replies whenever they change. The twelve claim-group figures are a separate PDF,
claim-figures.pdf, joined with pdfunite. Build with Tectonic: `tectonic review.tex` in this folder.

Usage, from the repo root:
  python3 1-lora-correction/artifacts/scenes/iclr-review-response/make_review_tex.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from figure_specs import SPECS  # noqa: E402


# What changed in the revision, per reply heading (the text before the colon), per reviewer.
CHANGES = {
    ("k4nd", "W1"): "Section 3 now says $r_t$ is the whole interaction the product drops and scopes the paper to plurality; the contributions claim only what the experiments show.",
    ("k4nd", "W2"): "Section 2 now says prior corrections keep the product as the target while this paper changes the target. The ablation separating joint-prompt supervision from the choice of pairs is left as future work.",
    ("k4nd", "W3"): "Table 1 scores every method on the 276 Attend-and-Excite prompts with BLIP-VQA, CLIP score and ImageReward. Figures 11 and 12 score 53 held-out pairs as the adapter acts on fewer steps. Section 4.1 opens with the evaluation protocol.",
    ("k4nd", "W4"): "Section 4 states the training loss as Equation 12 and says each adapted evaluation returns a noise prediction; inference is the usual product-of-experts step with the adapter switched on. The training diagram and the ``Figure 10'' reference are gone, and the repeated abstract-concept claims are cut.",
    ("k4nd", "W5"): "Figure 5's caption and Section 4.1 say every column starts from the same latent; CO3 is marked with a dagger for its later resampling noise.",
    ("k4nd", "Q1"): "As for W1.",
    ("k4nd", "Q2"): "Table 1, and Figures 11 and 12 on the held-out pairs.",
    ("k4nd", "Q3"): "Section 4 describes the training set; Figure 7 shows the joint prompt beside CO3 during screening. No CO3 picture is a training target.",
    ("k4nd", "Q4"): "As for W4 and W5.",
    ("UHz6", "W1"): "Table 1 adds three scorers on 276 prompts; Table 2 adds GenEval and T2I-CompBench.",
    ("UHz6", "W3"): "Figure 5 adds ToMe and a Langevin corrector; Table 1 compares with CO3 and ToMe; Section 2 discusses Attend-and-Excite and CONFORM; Appendix C.4 shows three concepts.",
    ("UHz6", "The training filter"): "As for k4nd Q3.",
    ("UHz6", "Presentation issues"): "The ``Figure 10'' reference is removed, and the Discussion reads ``and instead create hybrid objects''.",
    ("UHz6", "Q1"): "Section 4 describes the training set by kind; the 48 pairs are listed in the reply and released with the code.",
    ("UHz6", "Q2"): "Table 1, and Figures 11 and 12 on the held-out pairs.",
    ("UHz6", "Q3"): "Figures 11, 12 and 13 show the adapter switched on for the first 10, the first 25 or all 50 steps.",
    ("UHz6", "Q4"): "No change to the paper; the reply answers from the held-out pairs.",
    ("PrzE", "W1"): "Table 1 adds BLIP-VQA, CLIP score and ImageReward on 276 prompts; Table 2 adds GenEval and T2I-CompBench.",
    ("PrzE", "W2"): "Figure 5 adds ToMe and a Langevin corrector; Table 1 adds ToMe; Section 4 describes the training set.",
    ("PrzE", "Q1"): "As for W1 and W2.",
    ("PrzE", "Q2"): "Appendix C.4 and a Discussion paragraph show the adapter on three concepts.",
}


TRANSCRIBED = REPO / "_archive/artifacts/drips/present-revision-to-supervisor/reviews-transcribed.md"


def full_review(name: str) -> str:
    """The reviewer's whole review, from the checked transcription, as a framed block."""
    md = TRANSCRIBED.read_text()
    start = md.index(f"\n## {name}\n")
    nxt = md.find("\n## ", start + 4)
    block = md[start: nxt if nxt != -1 else len(md)].split("\n")[2:]
    out, mode = [], None

    def close():
        nonlocal mode
        if mode:
            out.append(f"\\end{{{mode}}}")
        mode = None

    para = []

    def flush():
        if para:
            out.append(tex(" ".join(para)) + "\n")
            para.clear()

    for line in block:
        st = line.strip()
        if st.startswith("Page") or st.startswith("Posted"):
            flush(); out.append("{\\small\\color{gray}" + tex(st) + "}\\par"); continue
        if st.startswith("### "):
            flush(); close(); out.append("\\paragraph{" + tex(st[4:]) + "}"); continue
        m = __import__("re").match(r"^(- |\d+\. )(.*)", st)
        if m:
            flush()
            want = "enumerate" if m.group(1)[0].isdigit() else "itemize"
            if mode != want:
                close(); out.append(f"\\begin{{{want}}}"); mode = want
            out.append("\\item " + tex(m.group(2)))
            continue
        if not st:
            flush(); continue
        if mode and line.startswith("   "):
            out.append("\\par " + tex(st)); continue
        close(); para.append(st)
    flush(); close()
    return ("\\begin{tcolorbox}[breakable,colback=red!3,colframe=red!45!black,title={The review by " + name
            + ", as posted on OpenReview}]\n" + "\n".join(out) + "\n\\end{tcolorbox}\n")


def tex(text: str) -> str:
    """Escape LaTeX specials outside $...$, and turn markdown emphasis into LaTeX."""
    parts = re.split(r"(\$[^$]*\$)", text)
    out = []
    for p in parts:
        if p.startswith("$") and p.endswith("$") and len(p) > 1:
            out.append(p)
            continue
        p = p.replace("\\", r"\textbackslash{}")
        for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
            p = p.replace(a, b)
        p = p.replace("\u2026", r"\ldots{}")
        p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p)
        p = re.sub(r"`([^`]+)`", r"\\texttt{\1}", p)
        p = re.sub(r'"([^"]+)"', r"``\1''", p)
        out.append(p)
    return "".join(out)


def sections(md: str, reviewer: str) -> list[dict]:
    """The ### points under one reviewer: heading, quoted words, reply paragraphs."""
    if "## Reviewer" in md:
        start = md.index(f"## Reviewer {reviewer}")
        nxt = md.find("\n## ", start + 5)
        md = md[start: nxt if nxt != -1 else len(md)]
    points = []
    for block in re.split(r"\n### ", "\n" + md)[1:]:
        lines = block.split("\n")
        head = lines[0].strip()
        quote, reply, para = [], [], []
        for l in lines[1:] + [""]:
            if l.startswith(">"):
                quote.append(l.lstrip("> ").rstrip())
            elif not l.strip():
                if para:
                    reply.append(" ".join(para))
                    para = []
            else:
                para.append(l.strip())
        points.append({"head": head, "quote": " ".join(q for q in quote if q), "reply": reply})
    return points


def main() -> None:
    k = (REPO / "paper/iclr-56173/replies/k4nd-and-uhz6.md").read_text()
    p = (REPO / "paper/iclr-56173/replies/prze.md").read_text()
    # review-text.tex: everything the reviewers wrote, stored under names; nobody edits it.
    store = [r"""% The reviewers' own words, stored under names that review.tex calls.
% \fullreview{k4nd} prints k4nd's whole review; \reviewquote{k4nd}{W1} prints the part of it
% that the reply to W1 answers. Generated from paper/iclr-56173/Reviews.pdf (transcribed) and the reply files.
"""]
    # review.tex: our replies and the changes; this is the file to edit.
    out = [r"""\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage[most]{tcolorbox}
\usepackage[hidelinks]{hyperref}
\definecolor{rev}{HTML}{C0392B}\definecolor{chg}{HTML}{1E8449}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}

% --- what to show: put % in front of a line to hide that text everywhere, remove it to show it ---
\def\showfullreviews{}   % every reviewer's whole review, in a box
\def\showquotes{}        % the red quote above each reply
% To hide one review or one quote only, put % in front of its \fullreview or \reviewquote line below.

\newcommand{\fullreview}[1]{\ifdefined\showfullreviews\csname fullreview-#1\endcsname\fi}
\newcommand{\reviewquote}[2]{\ifdefined\showquotes\begin{quote}\color{rev}\itshape\csname quote-#1-#2\endcsname\end{quote}\fi}
\newcommand{\changed}[1]{\par{\color{chg}\textbf{What changed in the revision.}} #1\par}
\input{review-text}

\title{ICLR 2027 submission 56173: the reviews, our replies and the revision}
\date{}
\begin{document}
\maketitle
Each reviewer's section opens with the full review as posted on OpenReview (transcribed from paper/iclr-56173/Reviews.pdf). Then each point quotes the reviewer (in red), gives the reply we plan to send, and says what changed in the revised paper (in green). Figure and table numbers refer to the revised paper. The reviewers' text lives in \texttt{review-text.tex}; this file holds our replies.
"""]
    for name, md in (("k4nd", k), ("UHz6", k), ("PrzE", p)):
        store.append(f"\n% ===== {name} =====\n\\expandafter\\long\\expandafter\\def\\csname fullreview-{name}\\endcsname{{%\n"
                     + full_review(name) + "}\n")
        out.append(f"\n\\section*{{Reviewer {name}}}\n\\fullreview{{{name}}}\n\n\\subsection*{{Our replies, point by point}}\n")
        for pt in sections(md, name):
            key = pt["head"].split(":")[0].strip()
            out.append(f"\n\\subsubsection*{{{tex(pt['head'])}}}\n")
            if pt["quote"]:
                store.append(f"\\expandafter\\long\\expandafter\\def\\csname quote-{name}-{key}\\endcsname{{{tex(pt['quote'])}}}\n")
                out.append(f"\\reviewquote{{{name}}}{{{key}}}\n\n")
            for para in pt["reply"]:
                out.append(tex(para) + "\n\n")
            if (name, key) in CHANGES:
                out.append("\\changed{" + CHANGES[(name, key)] + "}\n")
    out.append("\n\\end{document}\n")
    (HERE / "review-text.tex").write_text("".join(store))
    (HERE / "review.tex").write_text("".join(out))
    print(HERE / "review.tex", HERE / "review-text.tex")

if __name__ == "__main__":
    main()
