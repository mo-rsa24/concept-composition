"""Build reading-table.md for Loop 10 (step X1) from the quote-checked row files.

The opening prose is hand-written in tools/head.md; everything below it is generated from rows/*.json.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent
ROWS = HERE / "rows"
OUT = HERE / "reading-table.md"
HEAD = (HERE / "tools" / "head.md").read_text()
GROUPS = [
    ("Scoring a caption with noise predictions", ["diffusion-classifier", "diffusion-zero-shot-classifiers", "diffusion-itm"]),
    ("Training on noise predictions or diffusion residuals", ["diffusion-contrastive-reconstruction", "noiseclr", "dire", "sampling-path-tells-more"]),
    ("One space for many inputs", ["imagebind", "repa"]),
    ("Adding scores to compose", ["composable-diffusion", "reduce-reuse-recycle", "superdiff"]),
]
ELSEWHERE = [
    ("CLIP goes 3D", "2303.11313", "Loop 9", "../../noise-text-critic/01-reading/clip-goes-3d.md",
     "adds a third encoder beside a frozen CLIP and trains it with two contrastive losses"),
    ("Mechanisms of Projective Composition of Diffusion Models", "2502.04549", "Loop 9", "../../noise-text-critic/01-reading/projective-composition.md",
     "says when adding scores is exact"),
    ("Concept Algebra for (Score-Based) Text-Controlled Generative Models", "2302.03693", "Loop 9", "../../noise-text-critic/01-reading/concept-algebra.md",
     "treats concepts as subspaces of the score"),
    ("Steer Away From Mode Collisions (CO3)", "2509.25940", "Loops 4 and 5", "../../reproductions/CO3/design.md",
     "the sampler the CO3 paper tries to improve; Loop 5's design compares its printed settings with the released script"),
]


def cell(s):
    return " ".join(str(s).split()).replace("|", "\\|")


def page(r, key):
    p = r.get(key)
    return f" ([p. {p}]({r['page_url']}#page={p}))" if p else ""


rows, disagreements, closer = [], [], []
quotes = images = 0
for group, slugs in GROUPS:
    for slug in slugs:
        f = ROWS / f"{slug}.json"
        if not f.exists():
            rows.append(f"| {group} | {slug} | not read | | | | | | | | | | |")
            continue
        r = json.loads(f.read_text())
        quotes += len(r["quotes"])
        images += sum(bool(q.get("read_from_image")) for q in r["quotes"])
        rows.append("| " + " | ".join([
            group,
            f"[{cell(r['title'])}]({slug}.md), {cell(r['source'])}",
            cell(r["model"]),
            str(r["pages_read"]),
            cell(r["reads"]) + page(r, "reads_page"),
            cell(r["trains"]) + page(r, "trains_page"),
            cell(r["combines"]) + page(r, "combines_page"),
            cell(r["adding"]) + page(r, "adding_page"),
            cell(r["noise_and_seed"]) + page(r, "noise_page"),
            cell(r["timesteps"]) + page(r, "timesteps_page"),
            cell(r["unseen"]) + page(r, "unseen_page"),
            cell(r["headline"]) + page(r, "headline_page"),
            cell(r["suggests"]),
        ]) + " |")
        for d in r.get("disagreements", []):
            disagreements.append(f"| [{cell(r['title'])}]({slug}.md) | {cell(d['claim'])} | {cell(d['source'])} | {cell(d['paper_says'])} | [p. {d['page']}]({r['page_url']}#page={d['page']}) |")
        for e in r.get("extra_papers", []):
            closer.append(f"| {cell(e['title'])} | {cell(e.get('arxiv') or 'not printed')} | [{cell(r['title'])}]({slug}.md), p. {e.get('page')} | {cell(e['why'])} |")

n = sum(1 for g in GROUPS for s in g[1] if (ROWS / f"{s}.json").exists())
elsewhere = "\n".join(f"| {t} | {a} | {who} | [its note]({p}) | {why} |" for t, a, who, p, why in ELSEWHERE)
TABLE = f"""
## The table

{n} papers read in full for this step. Each row links to its note.

| Group | Paper | Model | Pages read | What it reads | What it trains | How noise predictions are combined | When adding is valid | Noise and seed | Which timesteps | Unseen concepts | Headline number | What it suggests here (inferred) |
|---|---|---|---:|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

## Papers read by other loops and cited here

These were not read again. Their notes are the source for any claim this loop makes about them.

| Paper | arXiv | Read by | Note | Why it matters here |
|---|---|---|---|---|
{elsewhere}

## Closer papers these cite, not read

No claim on this page relies on these.

| Paper | arXiv | Named by | Why it is close |
|---|---|---|---|
""" + "\n".join(closer) + """

## Where sources disagree

Each row gives the claim, where it came from and what the page says. No result was changed.

| Paper | Claim | Claim's source | What the page says | Page |
|---|---|---|---|---|
""" + "\n".join(disagreements) + f"""

## How the rows were checked

<details>
<summary>The quote check, and what it cannot catch</summary>

Each row carries word-for-word quotes for its claims and numbers. `tools/check_quotes.py` checked all {quotes} quotes against their stated PDF pages. The {images} values marked as read from figures rest on a reader's eye, because the script cannot match them in extracted text. The script confirms that a quote is on its page, and it cannot confirm that a reader interpreted it correctly, so anything a reader concluded is marked inferred.

</details>
"""
OUT.write_text(HEAD.rstrip() + "\n" + TABLE)
print(n, "rows,", len(disagreements), "disagreements,", len(closer), "closer papers,", quotes, "quotes,", images, "from images")
