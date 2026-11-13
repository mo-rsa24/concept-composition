"""Where every file this question reads and writes lives.

One module so the five scripts cannot disagree about a path. The question is whether a fused
product render fits both single-animal captions better than a real photograph of a cat beside a
dog does; the sets, the embedding and the figure are the three steps that answer it.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Local artifact folder, named for the question it answers.
OUT = REPO / "3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/what-the-captions-say/does-the-hybrid-fit-both-captions"
JOINT_TILES = OUT / "joint-tiles"
SETS = OUT / "sets.json"
SIMILARITIES = OUT / "similarities.json"
SUMMARY = OUT / "summary.json"
PLANE = OUT / "cat-dog-similarity-plane.png"
COLUMNS = OUT / "cat-dog-caption-fit.png"
PHOTO_THUMBS = OUT / "photo-thumbs"

# Sources already on disk here.
JOINT_SHEET = REPO / "artifacts/decks/supervisor-meeting/1-inference-only/0-reference-where-the-joint-prompt-succeeds.png"
TILES = REPO / "3-understanding-compositionality/artifacts/results/what-sdxl-already-knows/poe-against-the-joint/does-poe-of-cats-and-dogs-match-the-joint/tiles"
PRODUCT_TILES = TILES / "cat__x__dog"
PLURAL_TILES = TILES / "cats__x__dogs"

# Cluster side. The laptop has no /datasets mount, so nothing here is openable locally.
CLUSTER_OUT = "/datasets/mmolefe/poe_repair_min/outputs/hybrid_caption_fit"
CLUSTER_IMAGES = f"{CLUSTER_OUT}/images"
POOL_MANIFEST = "/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions_pool/manifest.json"

# The contact sheet is a regular 8 x 4 grid. Each row band is 348 px tall and opens with a 28 px
# label strip naming the seed, so the tile itself is the 320 px below it.
SHEET_COLS, SHEET_ROWS = 8, 4
SHEET_TILE = 320
SHEET_LABEL = 28

CAPTIONS = {
    "cat": "a photo of a cat",
    "dog": "a photo of a dog",
    "both": "a photo of a cat and a dog",
}

CLIP_MODEL = "openai/clip-vit-base-patch32"
