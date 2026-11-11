# Re-lettering an image whose text is baked in

For a figure made outside this repo (an image generator, a screenshot) whose labels are part of the pixels, so they print too small and in the wrong face. The fix is to keep the artwork and set every label again as real type at its printed size. Figures 1, 7 and 8 of the ICLR 56173 paper (`final_figures/molefe_fig1.png`, `final_figures/molefe_fig2.png`, `training-the-compositional-adapter.jpg`) need this; the lettering audit measured their text at about 1 to 4 pt against the 7 pt floor.

The rule every figure meets is on [the layout sheet](../../report/paper-review/layout-sheet.md#before-a-slot-is-called-finished): 8 pt labels, 9 pt titles, a Times-matching serif, drawn at the final width. The two-panel recipe beside this one, [composing a two-panel figure](composing-a-two-panel-figure-in-inkscape.md), covers placing and exporting in more depth.

## Fill in

| What | Figure 1 | Figure 7 | Figure 8 |
|---|---|---|---|
| file | `paper/iclr-56173/final_figures/molefe_fig1.png` | `paper/iclr-56173/final_figures/molefe_fig2.png` | `paper/iclr-56173/figures/training-the-compositional-adapter.jpg` |
| printed width | 5.45 in | 4.95 in | 5.50 in |
| where it prints | main text, page 2 | appendix | appendix |

## 1. Set the page to the printed width

1. Open Inkscape. **File → Document Properties** (Ctrl+Shift+D).
2. Display units and page size units: **in**. Width: the printed width from the table. Leave height for step 3.
3. Close the dialog.

You want: the white page rectangle at that width; the ruler along the top reads up to it.

## 2. Bring in the artwork without its text

1. **File → Import** (Ctrl+I), pick the image, choose **Embed**, image rendering **Default**.
2. With the image selected, type its width in the toolbar's **W** box as the printed width, with the lock on so height follows.
3. Remove the baked-in words. Easiest: regenerate the image without any text in it. Otherwise, cover each word with a rectangle filled with the background colour (**Rectangle tool**, F4; pick the colour with the **Dropper**, F7).

You want: the artwork filling the page width, no words left in it.
What failure looks like: the image is under 300 pixels per inch of printed width (Figure 1 needs at least 1,635 px across); it will print soft, so regenerate it larger.

## 3. Fit the page to the artwork

**Edit → Resize Page to Selection** (Ctrl+Shift+R) with the image selected.

## 4. Set every label as real type

1. **Text tool** (T). In the toolbar, font **TeX Gyre Termes** (or Times New Roman), unit **pt**.
2. Labels at **8 pt**, panel titles at **9 pt**, panel letters (a), (b) at **9 pt bold**. Never smaller than 7 pt.
3. Use the names the tables use: joint, PoE, CO3, ours; "cat and dog", not slugs.
4. Align with **Object → Align and Distribute** (Ctrl+Shift+A), never by eye.

You want: labels the same height as the paper's caption text when the page is viewed at 100% zoom.

## 5. Export the PDF

1. **File → Save As** (Ctrl+Shift+S), type **PDF**.
2. In the PDF options: text output **Embed fonts** (not "Convert text to paths"), rasterize filter effects at **300 dpi**, use document's page size.
3. Save beside the old file with the same name and `.pdf`, for example `final_figures/molefe_fig1.pdf`.

## 6. Check it

Tell Claude the file is there. It switches the `\includegraphics` in `paper/iclr-56173/iclr2027_conference.tex` to the PDF and runs:

```bash
python3 scripts/figure_text_audit.py
```

You want: the figure's row reads a measured size of at least 7 pt (8 for labels) and PASS, with no "OCR estimate".
What failure looks like: "no text found" means the text was converted to paths in step 5; save again with Embed fonts.

Provenance: unverified (written 2026-11-11 from the Inkscape 1.x menus and this repo's audit script; not yet walked end to end).
