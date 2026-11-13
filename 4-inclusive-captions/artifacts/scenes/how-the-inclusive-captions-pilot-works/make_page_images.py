#!/usr/bin/env python3
"""Build the pictures the scene embeds, from the files the pilot and its neighbours already wrote.

Two jobs. It shrinks every picture to at most 1800 pixels wide as a JPEG, so nine of them fit in a
page that has to stay under 16 MB. And it burns a source line onto the top of any picture that is
not this pilot's own output, because a caption can be read past and a label on the pixels cannot.

Run from the repo root:
    python3 artifacts/scenes/how-the-inclusive-captions-pilot-works/make_page_images.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "page-images"
AUDIT_SRC = Path("/tmp/claude-1000/-home-molef-PhD-poe-repair-min-dog-and-rain/"
                 "7b180fb3-eb17-4df9-a4c7-c5063044d5dd/scratchpad/audit/audit_page_00.png")
MAX_W = 1800
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

OWN = [  # this pilot's own output: no banner
    (REPO / "artifacts/results/the-inclusive-captions-photo-pool/both_sheet_00.png", "both-sheet-00.jpg"),
    (REPO / "artifacts/results/the-inclusive-captions-photo-pool/both_sheet_02.png", "both-sheet-02.jpg"),
    (AUDIT_SRC, "audit-page-00.jpg"),
]
BORROWED = [  # from another experiment in this repo: banner burnt on
    (REPO / "artifacts/results/does-poe-of-cats-and-dogs-match-the-joint/cats-and-dogs-sheet.jpg",
     "borrowed-cats-and-dogs-sheet.jpg",
     "FROM ANOTHER EXPERIMENT, NOT THIS PILOT   base SDXL with no adapter attached   "
     "artifacts/results/does-poe-of-cats-and-dogs-match-the-joint"),
    (REPO / "artifacts/results/does-poe-of-cats-and-dogs-match-the-joint/tiles/cat__x__dog/seed02/poe.png",
     "borrowed-poe-seed02.jpg",
     "FROM ANOTHER EXPERIMENT, NOT THIS PILOT   base SDXL, cat x dog product, seed 2   "
     "artifacts/results/does-poe-of-cats-and-dogs-match-the-joint"),
    (REPO / "artifacts/results/does-the-cat-prompt-already-draw-dogs/contact_sheet.png",
     "borrowed-contact-sheet.jpg",
     "FROM ANOTHER EXPERIMENT, NOT THIS PILOT   base SDXL, the cat prompt on 64 seeds   "
     "artifacts/results/does-the-cat-prompt-already-draw-dogs"),
    (REPO / "artifacts/results/does-the-cat-prompt-already-draw-dogs/crop_classifier_sheet.png",
     "borrowed-crop-classifier-sheet.jpg",
     "FROM ANOTHER EXPERIMENT, NOT THIS PILOT   the same 64 crops, named by zero-shot CLIP   "
     "artifacts/results/does-the-cat-prompt-already-draw-dogs"),
]


def shrink(im):
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    return im.convert("RGB")


def banner(im, text):
    """A dark strip across the top of the picture, saying where the picture came from."""
    size = max(15, round(im.width / 66))
    font = ImageFont.truetype(FONT, size)
    pad = round(size * 0.8)
    strip = size + 2 * pad
    out = Image.new("RGB", (im.width, im.height + strip), (24, 27, 34))
    out.paste(im, (0, strip))
    d = ImageDraw.Draw(out)
    while d.textlength(text, font=font) > im.width - 2 * pad and size > 9:
        size -= 1
        font = ImageFont.truetype(FONT, size)
    d.text((pad, (strip - size) / 2 - size * 0.12), text, font=font, fill=(255, 214, 102))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for src, name in OWN:
        shrink(Image.open(src)).save(OUT / name, quality=80, optimize=True)
        print(f"  {name}  {(OUT / name).stat().st_size // 1024} KB  from {src}")
    for src, name, text in BORROWED:
        banner(shrink(Image.open(src)), text).save(OUT / name, quality=80, optimize=True)
        print(f"  {name}  {(OUT / name).stat().st_size // 1024} KB  banner  from {src}")


if __name__ == "__main__":
    main()


# --- The picture the two trainer boxes carry: one training photo and the caption each run draws for it.
# The photograph is tile 6 of the pilot's own pool sheet. The caption strings are the literal ones in
# poe_repair/experiments/inclusive_captions/captions.py:54-57 and the shares are BOTH_SHARES at :69.
# Nothing here is invented: the manifest itself lives on /datasets and is not on this machine, so the
# figure shows the rule the trainer applies rather than a row read out of the manifest.

TILE_BOX = (1196, 55, 1388, 256)   # tile 6 on both_sheet_00.png, with its title line
RUN_A = [("a photo of a cat and a dog", "90%"), ("(blank)", "10%")]
RUN_B = [("a photo of a cat", "40%"), ("a photo of a dog", "40%"),
         ("a photo of a cat and a dog", "10%"), ("(blank)", "10%")]


def trainer_figure():
    reg = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    sheet = Image.open(REPO / "artifacts/results/the-inclusive-captions-photo-pool/both_sheet_00.png")
    tile = sheet.crop(TILE_BOX).convert("RGB")
    tile = tile.resize((tile.width * 3, tile.height * 3), Image.LANCZOS)
    W, H = 1500, max(620, tile.height + 180)
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f_head = ImageFont.truetype(FONT, 26)
    f_run = ImageFont.truetype(FONT, 22)
    f_cap = ImageFont.truetype(reg, 21)
    f_small = ImageFont.truetype(reg, 16)
    d.text((34, 28), "One both-animal training photo, and the caption each run draws for it",
           font=f_head, fill=(28, 31, 38))
    d.text((34, 64), "The photograph is tile 6 of this pilot's own pool sheet. The four caption strings and the "
                     "shares are read out of captions.py:54-57 and :69.", font=f_small, fill=(110, 116, 126))
    im.paste(tile, (34, 100))
    x = 34 + tile.width + 56
    y = 110
    for run, rows, tint in (("Run A draws", RUN_A, (90, 143, 230)), ("Run B draws", RUN_B, (224, 139, 75))):
        d.rectangle([x, y, x + 10, y + 26], fill=tint)
        d.text((x + 24, y), run, font=f_run, fill=(28, 31, 38))
        y += 44
        for text, share in rows:
            d.rectangle([x + 24, y + 4, x + 24 + 660, y + 40], outline=(222, 226, 232), width=1)
            d.text((x + 38, y + 11), text, font=f_cap, fill=(28, 31, 38))
            d.text((x + 24 + 600, y + 11), share, font=f_cap, fill=(110, 116, 126))
            y += 46
        y += 26
    d.text((x + 24, y - 10), "Run B resamples on every draw, so the same photo trains under all four over a run.",
           font=f_small, fill=(110, 116, 126))
    d.text((x + 24, y + 14), "0.169 of run B's cat-captioned draws hold a dog. Run A's hold none. That is the experiment.",
           font=f_small, fill=(110, 116, 126))
    im.save(OUT / "training-photo-and-its-captions.jpg", quality=86, optimize=True)
    print(f"  training-photo-and-its-captions.jpg  {(OUT / 'training-photo-and-its-captions.jpg').stat().st_size // 1024} KB  drawn")


trainer_figure()
