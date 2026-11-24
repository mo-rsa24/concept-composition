"""Build mechanism-table.md for Loop 12 (step Z1) from the quote-checked row files.

The opening prose is hand-written in tools/head.md; everything below it is generated from rows/*.json.
Usage: python3 -I tools/build_table.py   (from leakage/01-reading/)
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent
ROWS = HERE / "rows"
OUT = HERE / "mechanism-table.md"
HEAD = (HERE / "tools" / "head.md").read_text()
GROUPS = [
    ("Text encoder", ["a-cat-is-a-cat", "magnet", "follow-the-flow", "structured-diffusion-guidance"]),
    ("Cross-attention", ["conform", "divide-and-bind", "deleaker", "daam"]),
    ("Cross-attention and self-attention", ["bounded-attention", "self-cross-guidance"]),
    ("Layout, with attention", ["be-decisive", "mmdit-similar-subjects"]),
    ("Not located", ["seeing-double"]),
]
ELSEWHERE = [
    ("Attend-and-Excite", "2301.13826", "Loop 7", "../../abstract-checkers/01-reading/attend-and-excite.md",
     "cross-attention: raises the most neglected subject word's attention by a gradient on the latent"),
    ("Linguistic Binding in Diffusion Models (SynGen)", "2306.08877", "Loop 7", "../../abstract-checkers/01-reading/syngen-linguistic-binding.md",
     "cross-attention: pulls each noun's map toward its modifiers and away from other nouns"),
    ("Detector Guidance for Multi-Object Text-to-Image Generation", "2306.02236", "Loop 7", "../../abstract-checkers/01-reading/detector-guidance.md",
     "cross-attention, with a detector reading the predicted clean image to separate the subjects' maps"),
    ("Token Merging for Training-Free Semantic Binding (ToMe)", "2411.07132", "Loop 4", "../../targets/reads/first/ToMe.notes.md",
     "text encoder: merges each object's tokens into one composite token, then refines it in early steps"),
    ("Rectify Then Diffuse (RTD)", "2608.03135", "Loops 4 and 5", "../../targets/reads/first/RTD.notes.md",
     "layout: one norm-bounded update to the starting noise that reduces overlap between the concepts' attention maps"),
    ("Steer Away From Mode Collisions (CO3)", "2509.25940", "Loops 4 and 5", "../../reproductions/CO3/design.md",
     "the sum of scores: the only paper here that acts on how noise predictions are combined; it targets one concept dominating, not part borrowing"),
]


def cell(s):
    return " ".join(str(s).split()).replace("|", "\\|")


def page(r, key):
    p = r.get(key)
    return f" ([p. {p}]({r['page_url']}#page={p}))" if p else ""


rows, disagreements, closer, missing = [], [], [], []
quotes = images = pages = 0
for group, slugs in GROUPS:
    for slug in slugs:
        f = ROWS / f"{slug}.json"
        if not f.exists():
            missing.append(slug)
            rows.append(f"| {group} | {slug} | not read yet | | | | | | | | |")
            continue
        r = json.loads(f.read_text())
        quotes += len(r["quotes"])
        images += sum(bool(q.get("read_from_image")) for q in r["quotes"])
        pages += int(r["pages_read"])
        rows.append("| " + " | ".join([
            group,
            f"[{cell(r['title'])}]({slug}.md), {cell(r['source'])}, {r['pages_read']} pages",
            cell(r["enters"]) + ". " + cell(r["enters_evidence"]) + page(r, "enters_page"),
            cell(r["changes"]) + page(r, "changes_page"),
            cell(r["training"]) + page(r, "training_page"),
            cell(r["model"]),
            cell(r["similar_subjects"]) + page(r, "similar_page"),
            cell(r["which_pairs"]) + page(r, "which_pairs_page"),
            cell(r["when"]) + page(r, "when_page"),
            cell(r["cost"]) + page(r, "cost_page"),
        ]) + " |")
        for d in r.get("disagreements", []):
            disagreements.append(f"| [{cell(r['title'])}]({slug}.md) | {cell(d.get('claim', ''))} | {cell(d.get('source', ''))} | "
                                 f"{cell(d.get('paper_says', ''))} | {d.get('page', '')} |")
        for e in r.get("extra_papers", []):
            closer.append(f"| {cell(e.get('title', ''))} | {cell(e.get('arxiv', ''))} | [{slug}]({slug}.md), p. {e.get('page', '')} | {cell(e.get('why', ''))} |")

out = [HEAD.rstrip(), "", "## The table", "",
       "Navigation: [Table of contents](#table-of-contents)", "",
       "One row per paper read in full for this step. Every cell's page links open the PDF at that page.", "",
       "| Where leakage enters | Paper | The paper's place, and its evidence | What it changes there | Training and inputs it needs | Model | Similar subjects | Which pairs leak and why | When in sampling | Cost |",
       "|---|---|---|---|---|---|---|---|---|---|", *rows, "",
       "## Papers read by other loops and cited here", "",
       "Navigation: [Table of contents](#table-of-contents)", "",
       "| Paper | arXiv | Read by | Where it acts |", "|---|---|---|---|",
       *[f"| [{t}]({p}) | {a} | {who} | {w} |" for t, a, who, p, w in ELSEWHERE], "",
       "## Closer papers these cite, not read", "",
       "Navigation: [Table of contents](#table-of-contents)", "",
       "| Paper | arXiv | Cited in | Why |", "|---|---|---|---|", *(closer or ["| None | | | |"]), "",
       "## Where sources disagree", "",
       "Navigation: [Table of contents](#table-of-contents)", "",
       "Each row is a place where a paper contradicts itself or contradicts a claim this loop started from. None is resolved here.", "",
       "| Paper | Claim | Source of the claim | What the paper says | Page |", "|---|---|---|---|---|", *(disagreements or ["| None | | | | |"]), "",
       "## How the rows were checked", "",
       "Navigation: [Table of contents](#table-of-contents)", "",
       f"Each reader followed [the reader brief](tools/reader-brief.md) and wrote a row file under `rows/`. "
       f"Every number in a row's headline and similar-subject cell, and its place, change, training and which-pairs claims, carries a verbatim quote. "
       f"`tools/check_quotes.py` found all {quotes} quotes on their stated pages of the extracted text, {images} of them read from a page image. "
       f"{pages} PDF pages were read in all. The table is rebuilt by `python3 -I tools/build_table.py` from this folder.",
       ""]
if missing:
    out.append(f"Not read yet: {', '.join(missing)}.")
OUT.write_text("\n".join(out) + "\n")
print(f"{len(rows)} rows, {len(missing)} missing, {quotes} quotes, {pages} pages, {len(disagreements)} disagreements, {len(closer)} closer papers")
