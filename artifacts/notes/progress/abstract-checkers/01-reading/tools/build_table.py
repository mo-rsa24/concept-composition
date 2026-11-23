"""Build reading-table.md from the checked row files."""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent.parent
J = HERE / "rows"
OUT = HERE / "reading-table.md"
groups = [
 ("Spatial relations", ["learn-to-steer", "infsplign"]),
 ("Counting", ["counting-guidance", "iterative-object-count-optimization", "countdiffusion", "make-it-count", "t2icount"]),
 ("Attribute binding", ["syngen-linguistic-binding"]),
 ("Missing subjects", ["attend-and-excite"]),
 ("Position, size, shape", ["diffusion-self-guidance"]),
 ("Learned heads on diffusion features", ["readout-guidance"]),
 ("Overall prompt match", ["discriminative-probing-and-tuning"]),
 ("Guidance from any classifier", ["universal-guidance"]),
 ("Guidance from a detector", ["detector-guidance"]),
 ("Detector and discriminator signals", ["correcting-diffusion-through-resampling"]),
 ("Evaluators on noisy latents", ["dynamic-cfg-online-feedback"]),
 ("Classifier gradients on noisy inputs", ["gradcheck"]),
 ("Latent classifiers for composition", ["latent-classifier-guidance"]),
 ("Closer work: layout before sampling", ["spatial-temporal-attention"]),
 ("Closer work: classifier on noisy pixels", ["glide"]),
 ("Closer work: discriminator gradient", ["discriminator-guidance"]),
 ("Closer work: trained counters", ["clip-count", "countx", "vlcounter", "teaching-clip-to-count"]),
]
LISTED = 18
def cell(s): return " ".join(str(s).split()).replace("|", "\\|")
def pg(r, key, a):
    p = r.get(key)
    return f" ([p. {p}](https://arxiv.org/pdf/{a}#page={p}))" if p else ""
rows, dis, closer = [], [], []
n = quotes = images = 0
for prop, slugs in groups:
    for s in slugs:
        r = json.loads((J / f"{s}.json").read_text()); a = r["arxiv"]; n += 1
        quotes += len(r.get("quotes", [])); images += sum(bool(q.get("read_from_image")) for q in r.get("quotes", []))
        if r.get("why_added"):
            closer.append(f"| [{cell(r['title'])}]({s}.md) | {a} | read in full, {r['pages_read']} pages | {cell(r['why_added'])} |")
        rows.append("| " + " | ".join([
            prop, f"[{cell(r['title'])}]({s}.md), arXiv {a}", cell(r.get("model", "")), str(r["pages_read"]),
            cell(r["reads"]) + pg(r, "reads_page", a), cell(r["trained_on"]) + pg(r, "trained_page", a),
            cell(r["gradient_enters"]) + pg(r, "gradient_page", a), cell(r["unseen"]) + pg(r, "unseen_page", a),
            cell(r["headline"]) + pg(r, "headline_page", a), cell(r["copy_or_avoid"]),
        ]) + " |")
        for d in r.get("disagreements", []):
            dis.append(f"| [{cell(r['title'])}]({s}.md) | {cell(d['claim'])} | {cell(d['source'])} | {cell(d['paper_says'])} | [p. {d['page']}](https://arxiv.org/pdf/{a}#page={d['page']}) |")
head = f"""# Reading table: the closest work on abstract checkers

This table holds one row per paper read, {n} in all: the {LISTED} listed papers and the {n - LISTED} closer papers they cite. Each was read from its full PDF with appendices. It answers which of the five proposed checkers (spatial relation, counting, binding, position and size, prompt match) already exists as a trained classifier tested on concepts it never saw. Each row links to that paper's note, which holds the detail.

<a href="what-the-papers-cover.png"><img src="what-the-papers-cover.png" width="900" alt="Word attention maps feed five proposed checkers. Only the spatial relation checker has a trained precedent, Learn-to-Steer on FLUX.1-dev. Counting, binding, position and size, and prompt match are gaps. Their gradients are summed, capped and added to the product of experts on steps 0 to 15 of 50, and a dashed risk box asks whether word maps can be read above t = 800."></a>

## Table of contents

- [Which checkers already exist](#which-checkers-already-exist)
- [The table](#the-table)
- [Closer papers added](#closer-papers-added)
- [Where sources disagree](#where-sources-disagree)
- [How the rows were checked](#how-the-rows-were-checked)

## Which checkers already exist

| Proposed checker | Trained and tested on unseen concepts? | Closest paper | What is missing |
|---|---|---|---|
| Spatial relation | Yes, on word attention maps | Learn-to-Steer: GenEval position 0.60 of 1 on 16 unseen categories, FLUX.1-dev ([p. 7](https://arxiv.org/pdf/2509.02295#page=7)) | Never run on SDXL. The unseen words were chosen as absent from training, without a planned split |
| Counting | Only for counters that do not steer early sampling | T2ICount: held-out FSC-147 classes, mean error 11.76 objects per image ([p. 6](https://arxiv.org/pdf/2502.20625#page=6)). Iterative Object Count Optimization: one sentence on unseen classes, mean error 8.2 against 7.72 ([p. 7](https://arxiv.org/pdf/2408.11721#page=7)). CLIP-Count, CounTX and VLCounter test held-out FSC-147 classes on clean photos, with mean errors of 17.78 ([p. 6](https://arxiv.org/pdf/2305.07304#page=6)), 15.73 ([p. 7](https://arxiv.org/pdf/2306.01851#page=7)) and 17.05 ([p. 5](https://arxiv.org/pdf/2312.16580#page=5)) objects per image | No trained counter on word maps, and no count-2 result for two different subjects. FSC-147 images hold at least 7 objects ([CounTX p. 19](https://arxiv.org/pdf/2306.01851#page=19)) |
| Binding | No | SynGen's loss is hand-written ([p. 4](https://arxiv.org/pdf/2306.08877#page=4)) | Any trained binding checker |
| Position and size | No | Diffusion Self-Guidance's formulas are hand-written and print no score ([p. 13](https://arxiv.org/pdf/2306.00986#page=13)) | Any trained checker and any number |
| Prompt match | No | Discriminative Probing trains on U-Net features and holds out no concepts ([p. 15](https://arxiv.org/pdf/2403.04321#page=15)). GLIDE's noised CLIP reads noisy pixels and the caption, and holds out no concepts ([p. 6](https://arxiv.org/pdf/2112.10741#page=6)) | A held-out test, and an input free of the prompt |

## The table

| Property | Paper | Model | Pages read | What it reads | What it trained on | How its gradient enters | Unseen concepts | Headline number | Copy or avoid (inferred) |
|---|---|---|---:|---|---|---|---|---|---|
"""
tail = f"""
## Closer papers added

The brief allows up to 10 papers beyond the {LISTED} listed, when a paper trains a classifier for an abstract property or tests one on unseen concepts.

<a href="what-the-closer-papers-read.png"><img src="what-the-closer-papers-read.png" width="800" alt="The seven closer papers grouped by whether they read a noisy input or clean images, and by how each reaches a generator. None trains a checker on word maps from a noisy latent, tests it on held-out concepts and steers a sampler with it."></a>

| Paper | arXiv | Status | Why it was added |
|---|---|---|---|
""" + "\n".join(closer) + f"""
| Mid-U Guidance: Fast Classifier Guidance for Latent Diffusion Models | none | not read | learns a guidance function on diffusion features for aesthetics; named by Readout Guidance, p. 2. It is a W&B report with no arXiv PDF or HTML page, so no claim here uses it |

RCC (2205.10203), VA-Count (2407.04948), ImageReward and AFreeCA (2403.04943) were flagged and left off because they are less close to a guided checker.

## Where sources disagree

Each row quotes the claim, where it came from and what the paper's page says. No result was changed.

| Paper | Claim | Claim's source | What the page says | Page |
|---|---|---|---|---|
""" + "\n".join(dis) + f"""

## How the rows were checked

<details>
<summary>The quote check, and what it cannot catch</summary>

Each row carries word-for-word quotes for its claims and numbers. A script checked all {quotes} quotes against their stated PDF pages and none was missing. The {images} values marked as read from figures or page images rest on a reader's eye, because the script cannot match them in the extracted text. For every other quote, the script confirms that it is on its page. It cannot confirm that a reader interpreted the quote correctly, so anything a reader concluded is marked inferred.

</details>
"""
OUT.write_text(head + "\n".join(rows) + "\n" + tail)
print(n, "rows,", len(dis), "disagreements")
