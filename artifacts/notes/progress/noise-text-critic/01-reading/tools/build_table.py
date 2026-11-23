"""Build reading-table.md for Loop 9 from the quote-checked row files."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent
ROWS = HERE / "rows"
OUT = HERE / "reading-table.md"
GROUPS = [
    ("The model under test", ["noisyclip"]),
    ("Other scores on noisy or generated images", ["noise-aware-latent-reward", "noise-diffusion-semantic-faithfulness"]),
    ("Score decompositions", ["cfgcomposer", "memory-experts-world-models"]),
    ("When adding scores or embeddings is valid", ["projective-composition", "local-compositional-generalization", "concept-algebra", "linear-spaces-of-meanings"]),
    ("Training a new encoder", ["clip-goes-3d", "rince"]),
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
        r = json.loads((ROWS / f"{slug}.json").read_text())
        quotes += len(r["quotes"])
        images += sum(bool(q.get("read_from_image")) for q in r["quotes"])
        rows.append("| " + " | ".join([
            group,
            f"[{cell(r['title'])}]({slug}.md), {cell(r['source'] if r['source'].startswith('arXiv') else 'ICLR 2026')}",
            cell(r["model"]),
            str(r["pages_read"]),
            cell(r["reads"]) + page(r, "reads_page"),
            cell(r["trained_on"]) + page(r, "trained_page"),
            cell(r["score_enters"]) + page(r, "score_page"),
            cell(r["adding_scores"]) + page(r, "adding_page"),
            cell(r["properties"]) + page(r, "properties_page"),
            cell(r["unseen"]) + page(r, "unseen_page"),
            cell(r["headline"]) + page(r, "headline_page"),
            cell(r["copy_or_avoid"]),
        ]) + " |")
        for d in r.get("disagreements", []):
            disagreements.append(f"| [{cell(r['title'])}]({slug}.md) | {cell(d['claim'])} | {cell(d['source'])} | {cell(d['paper_says'])} | [p. {d['page']}]({r['page_url']}#page={d['page']}) |")
        for e in r.get("extra_papers", []):
            closer.append(f"| {cell(e['title'])} | {cell(e.get('arxiv') or 'not printed')} | [{cell(r['title'])}]({slug}.md), p. {e.get('page')} | {cell(e['why'])} |")

n = len(rows)
HEAD = f"""# Reading table: can a sentence score on a noisy latent act as a checker?

This table holds one row for each of the {n} papers this loop was asked to read, each read from its full PDF with appendices. It answers what a noise-aware text and image score can already do on the five abstract properties, and what step W2 should expect. Each row links to that paper's note.

<a href="sentence-score-push.png"><img src="sentence-score-push.png" width="620" alt="A half-drawn SDXL latent passes through a fixed linear map into the NoisyCLIP image encoder. A right and a wrong sentence pass through the frozen CLIP text encoder. The two scores are subtracted, and the gradient of that difference is added to the base noise prediction."></a>

## Table of contents

- [What the reading changes for W2](#what-the-reading-changes-for-w2)
- [What each property has behind it](#what-each-property-has-behind-it)
- [The table](#the-table)
- [Closer papers these cite, not read](#closer-papers-these-cite-not-read)
- [Where sources disagree](#where-sources-disagree)
- [How the rows were checked](#how-the-rows-were-checked)

## What the reading changes for W2

**NoisyCLIP reads worst on the steps this loop pushes.** Recall@1 is the share of images whose right caption ranks first among five captions, so 0.20 is chance. The right caption starts near 0.03 at latent 0 and reaches 0.20 near latent 10 ([Figure 4, p. 10](https://arxiv.org/pdf/2512.08505#page=10)). The caption with the wrong count ranks first more often than the right one until about latent 13 ([Figure 10, p. 23](https://arxiv.org/pdf/2512.08505#page=23)). Both values are read from the plots.

**No listed paper steers a sampler with a sentence-score gradient and measures layout.** The closest noise-aware score, the latent reward model, uses its score only to choose training pairs ([p. 6](https://arxiv.org/pdf/2502.01051#page=6)). Noise Diffusion's own comparison shows direct gradient updates making "limited changes to the image structure" on one prompt ([p. 7](https://arxiv.org/pdf/2411.16503#page=7)).

**A word-swap pair may carry no signal in the additive part of the embedding (inferred).** If CLIP's text embedding were a sum of word vectors, "a cat left of a dog" and "a dog left of a cat" would differ by a fixed vector that flips sign with the order. Only the part that does not add up could then separate them. W2 can measure that part from the text alone, following [the Linear Spaces note](linear-spaces-of-meanings.md).

**The published weights match the brief.** Both projection heads equal base CLIP's exactly, and the vision tower differs ([the comparison output](tools/checkpoint-compare.txt)). The released code leaves out the resize and normalisation, so W2 takes them from the checkpoint's processor config.

## What each property has behind it

| Property | What the {n} papers measure | Gap W2 fills |
|---|---|---|
| Counting | NoisyCLIP's wrong-count caption outranks the right caption until about latent 13 ([p. 23](https://arxiv.org/pdf/2512.08505#page=23)). Two theory papers count objects on CLEVR ([Projective Composition, p. 17](https://arxiv.org/pdf/2502.04549#page=17); [Local Mechanisms, p. 3](https://arxiv.org/pdf/2509.16447#page=3)) | One against two animals on early SDXL latents |
| Prompt match | NoisyCLIP Recall@1 over latents 21 to 30 is 0.506 against 0.145 for frozen CLIP ([p. 14](https://arxiv.org/pdf/2512.08505#page=14)) | Steps 0 to 15, on this project's pairs |
| Spatial relation | Only as a GenAI-Bench skill on ranked images ([NoisyCLIP, p. 24](https://arxiv.org/pdf/2512.08505#page=24)). No paper tests a left-right swap | The first swap test on noisy latents |
| Position and size | Position is measured on CLEVR ([Local Mechanisms, p. 3](https://arxiv.org/pdf/2509.16447#page=3)). No paper measures size | Both, on SDXL |
| Attribute binding | NoisyCLIP's printed failure picks a blue bicycle on a red wall for "a red bicycle against a blue wall" ([p. 30](https://arxiv.org/pdf/2512.08505#page=30)). No paper scores a colour swap | The first colour-swap test on noisy latents |

## The table

| Group | Paper | Model | Pages read | What it reads | What it trained on | How its score enters | When adding is valid | Properties measured | Unseen concepts | Headline number | Copy or avoid (inferred) |
|---|---|---|---:|---|---|---|---|---|---|---|---|
"""
TAIL = f"""
## Closer papers these cite, not read

These sit outside this loop's list, so no claim here relies on them. *Controlling Latent Diffusion Using Latent CLIP* is the nearest: its title describes a CLIP that reads latents and controls a diffusion model.

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

Each row carries word-for-word quotes for its claims and numbers. `tools/check_quotes.py` checked all {quotes} quotes against their stated PDF pages and none was missing. The {images} values marked as read from figures rest on a reader's eye, because the script cannot match them in extracted text. Figures 4 and 10 of NoisyCLIP were re-read from the rendered pages. The script confirms that a quote is on its page, and it cannot confirm that a reader interpreted it correctly, so anything a reader concluded is marked inferred. The argument in the Projective Composition note that a right-minus-wrong push falls outside the theorem's conditions is a reader's own short proof, and nobody else has checked it.

</details>
"""
OUT.write_text(HEAD + "\n".join(rows) + "\n" + TAIL)
print(n, "rows,", len(disagreements), "disagreements,", len(closer), "closer papers,", quotes, "quotes,", images, "from images")
