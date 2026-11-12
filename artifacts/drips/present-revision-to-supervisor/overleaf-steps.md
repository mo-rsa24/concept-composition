# Putting the revision into the supervisor's Overleaf project

You do every step here by hand in the browser; nothing is pushed by git. The revision goes into the
supervisor's existing project beside the original: one new `.tex` at the project root and every file
it reads in a new `revision/` folder. No file of the original is replaced.

The files come from the bundle folder on the laptop (paste a path into the file manager's address
bar with Ctrl+L):

| What | Path |
|---|---|
| the bundle folder | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/` |
| the main file, uploaded at the project root (step 5) | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/iclr2027_conference_revision.tex` |
| the 26 files uploaded into `revision/` (step 4), about 71 MB, the largest 19.8 MB | `/home/molef/PhD/poe_repair_min/artifacts/drips/present-revision-to-supervisor/overleaf-bundle/revision/` |
| the local build to compare against (step 9) | `/home/molef/PhD/poe_repair_min/paper/iclr-56173/build/iclr2027_conference.pdf` |
| the walk-through PDF for the supervisor (step 11) | `/home/molef/PhD/poe_repair_min/artifacts/scenes/iclr-review-response/supervisor-review-response.pdf` |

On Windows through WSL, the same folders open from File Explorer as
`\\wsl$\<your distribution>\home\molef\PhD\poe_repair_min\...`, or run `explorer.exe .` inside the
folder from a WSL terminal.

Upload the files themselves, never `overleaf-bundle.zip`: the zip is 71 MB, above Overleaf's 50 MB
limit for one upload, and Overleaf does not unpack a zip into an existing project.

A correct result is a 20-page PDF that matches `paper/iclr-56173/build/iclr2027_conference.pdf`.

## Steps

1. **Open the project.** In a browser, go to overleaf.com, sign in, and click the supervisor's ICLR
   56173 project in your project list. The editor opens with the file tree on the left.

2. **Check nothing will be overwritten.** In the file tree, look at the files at the top level.
   You should see the original's files (for example `iclr2027_conference.tex`). If a file named
   `iclr2027_conference_revision.tex` or a folder named `revision` already exists, stop and tell
   Claude, because uploading would replace it.

3. **Create the folder.** At the top of the file tree, click the **New folder** icon (a folder with a
   plus). Type `revision` and click **Create**. A `revision` folder appears in the tree.

4. **Upload the 26 files into it.** Click the `revision` folder once so it is selected, then click
   the **Upload** icon (an arrow pointing up) at the top of the file tree. In the upload window,
   drag in all 26 files from `overleaf-bundle/revision/` at once (in the laptop's file manager, open
   that folder, press Ctrl+A, and drag). Wait until every file shows as uploaded and close the window.
   What you should see: expanding `revision` lists 26 files, among them `R2.pdf`, `R14.tex`,
   `R15.pdf`, `iclr2027_conference.bib` and `panel_a_failure_types.pdf`.
   If a file shows an error: it is usually a dropped connection; upload that file again on its own.

5. **Upload the main file at the root.** Click an empty spot in the file tree so no folder is
   selected, click **Upload** again, and drag in `iclr2027_conference_revision.tex` from
   `overleaf-bundle/`. It must sit at the top level, not inside `revision/`, because Overleaf compiles
   from the project root and the file's paths all start with `revision/`.

6. **Make it the main document.** Click **Menu** (top left). Under **Settings**, set **Main document**
   to `iclr2027_conference_revision.tex`. Leave **Compiler** as it is for now (usually pdfLaTeX).
   Close the menu. This setting applies to the whole project, so the supervisor sees the revision
   when he opens it until it is set back (step 10).

7. **Compile.** Click **Recompile** above the PDF pane. The first compile takes longer than usual,
   because the figures are large.

8. **Check the result.** What a correct build looks like:
   - the PDF pane shows **20 pages**;
   - every page header reads "Under review as a conference paper at ICLR 2027";
   - page 9 holds Table 1, with rows Joint prompt, CO3, ToMe, PoE and Ours;
   - page 10 ends the main text with the AI Declaration, then the references begin;
   - page 15 holds Figure 8 (three rows by five columns), Table 2 (five numeric columns, no
     "result pending" box) and Figure 11 (failure types);
   - no "??" anywhere in the text, and the log button (next to Recompile) shows no red errors.

   If something is wrong:
   - **"Timed out"**: the project's plan allows too little compile time for these figures. Tell
     Claude; the fallback is lighter copies of the three largest PDFs (`R2.pdf`,
     `when-the-correction-arrives.pdf`, `where-the-correction-lands.pdf`).
   - **An error naming fontspec, an unknown character, or a font**: click **Menu**, set **Compiler**
     to **XeLaTeX**, and click **Recompile**. The revision was checked with a XeTeX-based build,
     so XeLaTeX matches it.
   - **"File not found" naming a `revision/...` file**: that file is missing from step 4; upload it
     into `revision/` and recompile.
   - **A missing running header**: tell Claude; it means a different `fancyhdr` was picked up.

9. **Compare with the local build.** Open `paper/iclr-56173/build/iclr2027_conference.pdf` on the
   laptop beside the Overleaf PDF and check that pages 1, 9, 10 and 15 look the same.

10. **Leave the project as the supervisor expects.** If he should open the original by default, set
    **Menu → Main document** back to `iclr2027_conference.tex` (and **Compiler** back to what it was,
    if you changed it in step 8). The revision stays in the project; switching the main document is
    all it takes to view either one.

11. **Tell the supervisor where it is.** One line is enough: the revision is
    `iclr2027_conference_revision.tex` with its files in `revision/`; set it as the main document to
    compile it; the claim-by-claim walk-through is the PDF you send him separately
    (`artifacts/scenes/iclr-review-response/supervisor-review-response.pdf`).
