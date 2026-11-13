# Putting the revision into the supervisor's Overleaf project

You do every step here by hand in the browser; nothing is pushed by git. The revision goes into the
supervisor's existing project beside the original: one new `.tex` at the project root and 26 files in
a new `revision/` folder. No file of the original is replaced. A correct result is a 20-page PDF that
matches the local build.

**[src]** is the path on the laptop. **[dest]** is the path in the Overleaf project's file tree, where
`/` is the project root. On Windows, a [src] path opens in File Explorer as
`\\wsl$\Ubuntu` followed by the same path with backslashes, for example
`\\wsl$\Ubuntu\home\molef\PhD\poe_repair_min\artifacts\drips\present-revision-to-supervisor\overleaf-bundle\revision`.

Upload the files themselves, never `overleaf-bundle.zip`: the zip is 71 MB, above Overleaf's 50 MB
limit for one upload, and Overleaf does not unpack a zip into an existing project.

## Steps

1. **Open the project.**
   [dest] overleaf.com → sign in → the supervisor's ICLR 56173 project. The file tree is on the left.

2. **Check nothing will be overwritten.**
   [dest] `/` (the top level of the file tree). If `/iclr2027_conference_revision.tex`, `/review.tex`,
   `/review-text.tex` or a folder `/revision` already exists, stop and tell Claude.

3. **Create the folder.**
   [dest] `/revision`: click the **New folder** icon at the top of the file tree, type `revision`,
   click **Create**.

4. **Upload the 26 files into the folder in one drag.**
   [src] `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/` (open it, press Ctrl+A, drag everything)
   [dest] `/revision/` (click the `revision` folder once, then the **Upload** icon, then drop)
   When every file shows as uploaded, `revision` should list exactly the 26 files below. A file that
   shows an error usually means a dropped connection; upload that one again on its own.

   | # | [src] | [dest] | used by | size |
   |---|---|---|---|---|
   | 1 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/R1.tex` | `revision/R1.tex` | Table 1 | 0.00 MB |
   | 2 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/R10.pdf` | `revision/R10.pdf` | Figure 13 | 3.52 MB |
   | 3 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/R11.png` | `revision/R11.png` | Figure 16 | 2.51 MB |
   | 4 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/R12.pdf` | `revision/R12.pdf` | Figure 8 | 1.50 MB |
   | 5 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/R14.tex` | `revision/R14.tex` | Table 2 | 0.00 MB |
   | 6 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/R15.pdf` | `revision/R15.pdf` | Figure 9 | 2.35 MB |
   | 7 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/R2.pdf` | `revision/R2.pdf` | Figure 5 | 19.81 MB |
   | 8 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/adapter-limitations-stacked.pdf` | `revision/adapter-limitations-stacked.pdf` | Figure 15 | 2.39 MB |
   | 9 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/adapter-settings-full-width.pdf` | `revision/adapter-settings-full-width.pdf` | Figure 10 | 1.60 MB |
   | 10 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/fancyhdr.sty` | `revision/fancyhdr.sty` | template | 0.02 MB |
   | 11 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/iclr2027_conference.bib` | `revision/iclr2027_conference.bib` | bibliography | 0.01 MB |
   | 12 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/iclr2027_conference.bst` | `revision/iclr2027_conference.bst` | template | 0.03 MB |
   | 13 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/iclr2027_conference.sty` | `revision/iclr2027_conference.sty` | template | 0.01 MB |
   | 14 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/joint-prompt-screening.pdf` | `revision/joint-prompt-screening.pdf` | Figure 7 | 2.05 MB |
   | 15 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/judge_sample_40.png` | `revision/judge_sample_40.png` | Figure 14 | 6.03 MB |
   | 16 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/math_commands.tex` | `revision/math_commands.tex` | template | 0.01 MB |
   | 17 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/molefe_fig1.png` | `revision/molefe_fig1.png` | Figure 1 | 1.62 MB |
   | 18 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/natbib.sty` | `revision/natbib.sty` | template | 0.05 MB |
   | 19 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/panel_a_failure_types.pdf` | `revision/panel_a_failure_types.pdf` | Figure 11 | 0.01 MB |
   | 20 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/panel_c_two_animals.pdf` | `revision/panel_c_two_animals.pdf` | Figure 12a | 0.01 MB |
   | 21 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/panel_d_imagereward.pdf` | `revision/panel_d_imagereward.pdf` | Figure 12b | 0.01 MB |
   | 22 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/panel_legend.pdf` | `revision/panel_legend.pdf` | Figures 11 and 12, legend | 0.01 MB |
   | 23 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/strength-grid-seven-columns.pdf` | `revision/strength-grid-seven-columns.pdf` | Figure 2 | 1.89 MB |
   | 24 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/training-target-examples.pdf` | `revision/training-target-examples.pdf` | Figure 6 | 0.48 MB |
   | 25 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/when-the-correction-arrives.pdf` | `revision/when-the-correction-arrives.pdf` | Figure 3 | 14.11 MB |
   | 26 | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/where-the-correction-lands.pdf` | `revision/where-the-correction-lands.pdf` | Figure 4 | 11.23 MB |

5. **Upload the main file at the project root.**
   [src] `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/iclr2027_conference_revision.tex`
   [dest] `/iclr2027_conference_revision.tex`
   Click an empty spot in the file tree so no folder is selected, then **Upload**. It must sit at `/`,
   not inside `/revision/`: Overleaf compiles from the root and every path in the file starts with
   `revision/`.

5b. **Upload the reviews document and its reviewer text at the project root.**
   [src] `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/review.tex` → [dest] `/review.tex`
   [src] `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/review-text.tex` → [dest] `/review-text.tex`
   `review.tex` holds our replies and what changed, and is the file you and the supervisor edit. The
   reviewers' words live in `review-text.tex`, which nobody needs to open: `review.tex` calls them by
   name. To hide the reviewers' text, put `%` in front of a line in `review.tex`: in front of
   `\def\showfullreviews{}` or `\def\showquotes{}` near the top to hide every full review or every
   quote, or in front of one `\fullreview{...}` or `\reviewquote{...}{...}` line to hide just that one. Like the revision, it compiles only when it is the
   main document (step 6 with `review.tex` selected); a correct build is 13 pages, or 6 with both
   switches off.

6. **Make it the main document.**
   [dest] **Menu** (top left) → **Settings** → **Main document** → `iclr2027_conference_revision.tex`.
   Leave **Compiler** as it is for now. This setting is shared with the original, so the project shows
   the revision until it is set back in step 10.

7. **Compile.** Click **Recompile** above the PDF pane. The first compile is slow; the figures are large.

8. **Check the result.** A correct build:
   - shows **20 pages**, each with the header "Under review as a conference paper at ICLR 2027";
   - page 9: Table 1 with rows Joint prompt, CO3, ToMe, PoE and Ours;
   - page 10: the main text ends with the AI Declaration, then the references begin;
   - page 15: Figure 8 (three rows by five columns), Table 2 with five numeric columns, and Figure 11;
   - no "??" anywhere, and no red errors under the log button beside **Recompile**.

   If it is wrong:
   - **"Timed out"**: the plan's compile time is too short for these figures; tell Claude, who can make
     lighter copies of rows 7, 25 and 26 of the table above.
   - **An error naming fontspec, a font or an unknown character**: **Menu** → **Compiler** →
     **XeLaTeX**, then **Recompile**. The revision was checked with a XeTeX-based build.
   - **"File not found: revision/…"**: that file is missing from `/revision/`; upload it from its [src]
     row in the table and recompile.
   - **No running header**: tell Claude.

9. **Compare with the local build.**
   [src] `/home/molef/PhD/poe_repair_min/paper/iclr-56173/build/iclr2027_conference.pdf`
   Open it beside the Overleaf PDF; pages 1, 9, 10 and 15 should look the same.

10. **Leave the project as the supervisor expects.**
    [dest] **Menu** → **Main document** → `iclr2027_conference.tex` (and **Compiler** back to what it
    was, if you changed it in step 8). The revision stays in the project; switching the main document
    is all it takes to view either one.

11. **Send the supervisor the claim figures.** They are not uploaded to Overleaf.
    [src] `/home/molef/PhD/poe_repair_min/1-lora-correction/artifacts/scenes/iclr-review-response/claim-figures.pdf`
    with one line: the revision is `/iclr2027_conference_revision.tex` with its files in `/revision/`,
    and the reviews with our replies are `/review.tex`; set either as the main document to compile it.
