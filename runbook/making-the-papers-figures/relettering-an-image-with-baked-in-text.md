# 🧭 Runbook: re-lettering an image whose text is baked in

For a figure made outside this repo (an image generator, a screenshot) whose labels are part of the pixels, so they print too small and in the wrong face. The fix is to keep the artwork and set every label again as real type at its printed size. Figures 1, 7 and 8 of the ICLR 56173 paper (`final_figures/molefe_fig1.png`, `final_figures/molefe_fig2.png`, `training-the-compositional-adapter.jpg`) need this; the lettering audit measured their text at about 1 to 4 pt against the 7 pt floor.

The rule every figure meets is on [the layout sheet](../../1-lora-correction/report/paper-review/layout-sheet.md#before-a-slot-is-called-finished): 8 pt labels, 9 pt titles, a Times-matching serif, drawn at the final width. The two-panel recipe beside this one, [composing a two-panel figure](composing-a-two-panel-figure-in-inkscape.md), covers placing and exporting in more depth.

Recipes 1, 2, 4 and 6 give a ✅ line as their success check. Recipes 3 and 5 state none yet.

## Table of contents

- [Fill in](#fill-in)
- [1. Set the page to the printed width](#1-set-the-page-to-the-printed-width)
- [2. Bring in the artwork without its text](#2-bring-in-the-artwork-without-its-text)
- [3. Fit the page to the artwork](#3-fit-the-page-to-the-artwork)
- [4. Set every label as real type](#4-set-every-label-as-real-type)
- [5. Export the PDF](#5-export-the-pdf)
- [6. Check it](#6-check-it)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)

## Fill in

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-set-the-page-to-the-printed-width) ➡️

| What | Figure 1 | Figure 7 | Figure 8 |
|---|---|---|---|
| file | `writing/restoring-plurality-in-compositional-diffusion_overleaf/final_figures/molefe_fig1.png` | `writing/restoring-plurality-in-compositional-diffusion_overleaf/final_figures/molefe_fig2.png` | `writing/restoring-plurality-in-compositional-diffusion_overleaf/figures/training-the-compositional-adapter.jpg` |
| picture | picture held back, because a tile holds a shape that could not be identified at full size. It shows: Panel a: a red and blue surface with a cat photo on one peak, a dog photo on the other, a blended cat-dog in the valley marked 'PoE lands here', and a cat-and-dog photo below; panel b: joint prompt against PoE renders for SD 1.4, SD 2.1, SDXL and SD 3.5 | picture held back, because a tile holds a shape that could not be identified at full size. It shows: Panel a: four prompt pairs (cat and dog, tiger and dog, chess board and hourglass, picnic basket and watermelon) rendered by the base model and through the adapter; panel b: six failure renders with red zoom boxes, from one fused animal to two animals of the wrong species | <a href="../../writing/restoring-plurality-in-compositional-diffusion_overleaf/figures/training-the-compositional-adapter.jpg"><img src="../../writing/restoring-plurality-in-compositional-diffusion_overleaf/figures/training-the-compositional-adapter.jpg" width="200" alt="Three-box training diagram: before training the frozen SDXL U-Net caches a joint-prompt target; every training step the U-Net with LoRA adapter composes the two single-concept predictions; the loss compares the two and updates only the LoRA weights"></a> |
| printed width | 5.45 in | 4.95 in | 5.50 in |
| where it prints | main text, page 2 | appendix | appendix |

## 1. Set the page to the printed width

Navigation: ⬅️ [Previous](#fill-in) | 📋 [TOC](#table-of-contents) | [Next](#2-bring-in-the-artwork-without-its-text) ➡️

`unverified`

1. Open Inkscape. **File → Document Properties** (Ctrl+Shift+D).
2. Display units and page size units: **in**. Width: the printed width from the table. Leave height for step 3.
3. Close the dialog.

✅ Expected: the white page rectangle at that width; the ruler along the top reads up to it.

## 2. Bring in the artwork without its text

Navigation: ⬅️ [Previous](#1-set-the-page-to-the-printed-width) | 📋 [TOC](#table-of-contents) | [Next](#3-fit-the-page-to-the-artwork) ➡️

`unverified`

1. **File → Import** (Ctrl+I), pick the image, choose Embed, image rendering Default.
2. With the image selected, type its width in the toolbar's **W** box as the printed width, with the lock on so height follows.
3. Remove the baked-in words. Easiest: regenerate the image without any text in it. Otherwise, cover each word with a rectangle filled with the background colour (**Rectangle tool**, F4; pick the colour with the Dropper, F7).

✅ Expected: the artwork filling the page width, no words left in it.
What failure looks like: the image is under 300 pixels per inch of printed width (Figure 1 needs at least 1,635 px across); it will print soft, so regenerate it larger.

## 3. Fit the page to the artwork

Navigation: ⬅️ [Previous](#2-bring-in-the-artwork-without-its-text) | 📋 [TOC](#table-of-contents) | [Next](#4-set-every-label-as-real-type) ➡️

`unverified`

**Edit → Resize Page to Selection** (Ctrl+Shift+R) with the image selected.

## 4. Set every label as real type

Navigation: ⬅️ [Previous](#3-fit-the-page-to-the-artwork) | 📋 [TOC](#table-of-contents) | [Next](#5-export-the-pdf) ➡️

`unverified`

1. **Text tool** (T). In the toolbar, font TeX Gyre Termes (or Times New Roman), unit pt.
2. Labels at **8 pt**, panel titles at 9 pt, panel letters (a), (b) at 9 pt bold. Never smaller than 7 pt.
3. Use the names the tables use: joint, PoE, CO3, ours; "cat and dog", not slugs.
4. Align with **Object → Align and Distribute** (Ctrl+Shift+A), never by eye.

✅ Expected: labels the same height as the paper's caption text when the page is viewed at 100% zoom.

## 5. Export the PDF

Navigation: ⬅️ [Previous](#4-set-every-label-as-real-type) | 📋 [TOC](#table-of-contents) | [Next](#6-check-it) ➡️

`unverified`

1. **File → Save As** (Ctrl+Shift+S), type PDF.
2. In the PDF options: text output **Embed fonts** (not "Convert text to paths"), rasterize filter effects at 300 dpi, use document's page size.
3. Save beside the old file with the same name and `.pdf`, for example `final_figures/molefe_fig1.pdf`.

## 6. Check it

Navigation: ⬅️ [Previous](#5-export-the-pdf) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

`unverified`

Tell Claude the file is there. It switches the `\includegraphics` in `writing/restoring-plurality-in-compositional-diffusion_overleaf/iclr2027_conference.tex` to the PDF and runs:

```bash
python3 1-lora-correction/scripts/figure_text_audit.py
```

✅ Expected: the figure's row reads a measured size of at least 7 pt (8 for labels) and PASS, with no "OCR estimate".
What failure looks like: "no text found" means the text was converted to paths in step 5; save again with Embed fonts.

Provenance: unverified (written 2026-11-11 from the Inkscape 1.x menus and this repo's audit script; not yet walked end to end).


<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#6-check-it) | 📋 [TOC](#table-of-contents)

The recipe re-letters three figures; their input pictures are in the Fill in table. This block adds the check that decides whether the result passes, and links the neighbouring pages.

### The code behind it

Step 6 runs `1-lora-correction/scripts/figure_text_audit.py`. It sets the 7 pt floor on line 26 (`MIN_PT = 7.0`) and decides PASS or FAIL here. Excerpt (lines 106 to 113):

```python
        raster = p.suffix.lower() in (".png", ".jpg", ".jpeg")
        sizes = raster_sizes(p, printed) if raster else pdf_sizes(p, printed)
        if not sizes:
            print(f"{name:58s} {printed:8.0f} {0:6d} {'':>7s} {'':>7s}  no text found")
            continue
        lo, med = min(sizes), statistics.median(sizes)
        verdict = ("PASS" if lo >= MIN_PT else "FAIL") + (" (OCR estimate)" if raster else "")
        print(f"{name:58s} {printed:8.0f} {len(sizes):6d} {lo:7.1f} {med:7.1f}  {verdict}")
```

- A raster file (png, jpg) is measured by OCR and its verdict carries "(OCR estimate)", which is why step 6 wants a vector PDF with no such tag.
- A file with no text layer prints "no text found", the failure step 6 names for text saved as paths.
- Columns printed: file, printed width in points, words measured, smallest and median size in points, verdict.

### What comes out

The audit prints one row per figure with the columns above. No printed output of it on these three figures is filed. The prompt for revising the paper, in the same drip as the retro below, records sizes for two of them: 3 to 4 pt for Figure 1 and 1.3 to 1.8 pt for Figure 7 `stated`. A note records that 6 of 8 figures passed the audit after the paper's presentation pass, up from 3 of 8 `stated` (the presentation retro in the reviewer-answer drip). That note calls it the 8 pt audit, while the script's PASS floor is 7 pt, so the two differ.

### Go deeper

| What | Link |
|---|---|
| The two-panel recipe, placing and exporting in depth | [composing a two-panel figure](composing-a-two-panel-figure-in-inkscape.md) |
| The lettering rule | [the layout sheet](../../1-lora-correction/report/paper-review/layout-sheet.md#before-a-slot-is-called-finished) |
| The before-and-after count of figures passing the audit | [the presentation retro](../../1-lora-correction/artifacts/drips/answering-the-reviewers/answer-iclr-reviews/2026-11-11-presentation-retro.md) |

### Still open

No picture of a finished re-lettered figure is filed, and no screenshot of Inkscape's menus. The recipe's own provenance line says it has not been walked end to end.
<!-- related:end -->

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
