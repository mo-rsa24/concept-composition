The largest review costs are the visibly unfinished captions and references, R2’s unreadable labels, R8’s unexplained visual encoding, and tables that do not yet implement the promised comparison structure. The centered tables themselves are reasonably balanced. Their rule style is a smaller problem.

I read the [layout sheet](../../../../../report/paper-review/layout-sheet.md), both generators, and the slot definitions. No files were changed. For empty slots, I distinguish visible faults from problems the generator would introduce when filled.

One important correction: the layout sheet calls `\footnotesize` 8 pt, but this paper’s [style file](/home/molef/PhD/concept-composition/paper/iclr-56173/iclr2027_conference.sty:175) defines it as **9 pt with 10 pt leading**. The unscaled R1 and R5 tables therefore use 9 pt Times, as do the placeholder labels. Preserve the conference definition and correct the layout sheet’s description.

1. All unfinished slots: highest review cost.

R1 and R3 through R14 have captions beginning “Reserved for”; the empty boxes additionally say “see the layout sheet.” These are production instructions addressed to the authors. R8 retains them even though its image is filled. A reviewer sees an incomplete manuscript, not an intentional presentation.

Replace each caption with a descriptive subject followed by the information needed to interpret the result. Match the existing paper’s bold opening sentence where appropriate, without inventing a result claim. For example: `\caption{\textbf{Adapter selection on two showcase pairs.} ...}`.

The blue outlines and grey fills are useful drafting aids but must disappear with the placeholders. In a submission build, the missing-file branch of `\reviewslot` should fail explicitly, for example `\PackageError{reviewslots}{Missing result for #2}{Supply the final slot asset}`. Checking that a slot is wired is insufficient to establish that it is finished.

R12 and R13 show unresolved table references in the gallery. Their source labels exist, so this may be a stale or insufficiently compiled build. Resolve the references and inspect the final PDF; red boxed question marks are particularly conspicuous evidence of unfinished production.

2. R2: severe print-legibility failure.

The submitted fallback image has tiny sans-serif method headings and rotated pair labels. At 2.64 inches wide, the roughly 20 px lettering in its 1276 px canvas is approximately 3 pt. It falls far below the 7 pt minimum. Rotating the long pair names makes an already difficult reading task worse.

Rebuild the labels at their final physical size. For this canvas, 7 pt requires approximately 47 px type: `font_px = math.ceil(7 / 72 * 1276 / 2.64)`. That will require rearranging the labels, not simply enlarging them in their current margins. Use short horizontal pair labels or panel letters with a caption key. Typesetting labels in LaTeX over the image grid would provide sharp Times lettering.

The narrow wrapped caption also produces a dense, heavily broken block. The fixed `[26]` wrap length leaves an empty left-hand region beside continuing text on the R1 page. Recalculate the wrap length after rebuilding R2, or use a normal figure if the wrap cannot end cleanly.

The caption explains CO3’s different noise, but the column itself lacks the promised dagger. Use `CO3†` and define the exception once. Standardize “Plain product” with “PoE” elsewhere, for example “PoE” in every panel and “plain product of experts” at its first definition.

3. R8: serious interpretation and accessibility faults.

The labels are approximately **7.04 pt**, calculated from the actual 2080 px image width and 37 px font at 5.5 inches. They meet the minimum. However, they use DejaVu Sans against the paper’s Times body. Sans-serif figure labels are not inherently wrong, but here the visual treatment differs markedly from the tables and captions. In `draw`, use a Times-compatible serif family such as Liberation Serif, with its matching bold face, then recompute the dimensions.

Each thumbnail is only **0.254 inches wide**. That is too small to independently judge subtle fusion, species identity, or extra animals reliably. Higher pixel resolution will not fix physical size. Within the existing slot, present a readable call matrix and reserve representative image enlargements for a companion panel. If all 144 photographs must remain independently inspectable, the slot needs more space.

Red and green alone encode the calls, and no legend explains them. This is difficult for red-green colour-blind readers and ambiguous in monochrome printing. Use redundant symbols or border patterns as well as a distinguishable palette. A palette change such as `colour = "#0072B2" if clear else "#D55E00"` helps, but does not replace a symbol and legend. Put symbols in dedicated margins or cells rather than obscuring the already small photographs.

The layout sketch specifies four call categories, while the generator renders only “clear” and “not clear.” Reconcile the figure and its specification. If the underlying annotations are binary, explain that binary classification rather than presenting an unsupported four-category legend.

The seed numbers are missing. “Seeds 9 to 16” forces readers to count columns, and the thicker border marking the selected seed is unexplained. Add individual seed headings to both groups and a separate selection indicator. Use a black heading marker or column outline so selection does not alter the same border that encodes the call.

The winning row’s black outline is only about 0.38 pt and competes with coloured cell borders. Its bold label helps, but neither treatment is explained. Add “selected” to the label or use a clear bracket and caption key. Separate winner selection, seed selection, and call classification visually.

Candidate names are opaque experiment identifiers. “self,” “contrast,” “plain,” and the unexplained “v57-11b” are not parallel descriptions. `short()` also removes meaningful configuration text from some identifiers. Use explicit display labels such as “Self-supervised, 30k updates,” with a mapping to full run IDs in accompanying text. Define “k” and distinguish training updates from sampling steps.

Finally, the group headings sit tightly against the first row, and the blocks are separated by only a narrow gutter. Add a genuine header band containing centered group titles and seed labels. Recheck the final printed font size afterward: increasing image height can make `\reviewslot` scale by its 2.6-inch height limit, reducing every label.

4. R1 and R5: substantial table-structure faults.

Both test tables have one flat sequence of metrics instead of the required Animals, Animals + objects, and Objects groups. They also include aggregate benchmark columns that do not match the layout sheet’s planned R1/R5 structure. Readers cannot tell which population each score describes.

In `csv_to_tabular`, replace “every non-axis CSV column” with slot-specific schemas and pivot the `group` field into grouped columns. Use a two-row header with `\multicolumn`, followed by `\cmidrule(lr){2-5}` and equivalent rules for the other groups. Keep BLIP and IR in the prescribed order across both tables.

The first column lacks a “Method” heading. Comp., IR, CB++, GE, and NFE are not explained by the shown captions, despite the generator comment saying that the caption carries the key. Add metric definitions, score direction, units or scale, and the meaning of the uncertainty interval. Explain the Langevin dagger and label its subset.

R1 also lacks the planned cited CO3 reference row and showcase marker. Distinguish the cited row with spacing, a separating rule, and an explicit “reported” label. Pale grey text alone risks poor print contrast.

The test zeros are correctly understood here as stand-ins, but they must never reach the submission. `000` is especially inappropriate for NFE; print an integer without zero padding. The generator currently copies numeric strings verbatim, so mixed precision and inconsistent uncertainty formatting will survive into the paper.

Right alignment looks acceptable with identical zeros but will not reliably align heterogeneous decimal values. Use `siunitx` numeric columns with formats chosen per metric, for example `S[table-format=1.2]`, and a separate integer format for NFE. Format estimates and intervals consistently before rendering, with spaced `\pm` notation. Do not indiscriminately impose two decimals on every metric.

5. R1, R3, R5, R7 and R14: generator and table-style faults.

The generator uses three identical `\hline` rules. They lack the visual hierarchy and breathing room of booktabs. This is a polish issue, not a scientific defect. Add `\usepackage{booktabs}` and generate `\toprule`, `\midrule`, and `\bottomrule`, with no vertical rules.

Natural-width centering is already correct in the supplied table examples. Preserve it. Use `@{}l...@{}` to remove unnecessary outer padding while retaining readable intercolumn spacing.

The automatic `\resizebox` fallback violates the requirement to keep tables in `\footnotesize`. Any shrinking changes that size; sufficiently large tables also fall below 7 pt. Replace the resize branch with an explicit overflow error. Fit tables through grouped headers, controlled column spacing, wrapped method names, or additional panels.

The `.tex` branch of `\reviewslot` ignores the height argument entirely. Consequently, filling a table does not preserve the placeholder’s reserved height. The image branch likewise preserves aspect ratio without reserving unused space. The claimed stable pagination is therefore not guaranteed. Measure completed tables against their height budgets and review pagination after insertion rather than trusting the placeholder geometry.

All five table captions currently appear below their tables. Putting table captions above and figure captions below would provide a clearer conventional distinction. Move `\caption` and its `\label` before `\reviewslot` for tables.

6. Remaining slot-specific faults and constraints.

R3 is still a blank box, so its eventual numeric alignment cannot be judged. Its promised group counts sum to 49, not the caption’s 55 held-out pairs. Resolve that discrepancy before generating the grouped header. The generic renderer’s unconditional NFE column also does not match R3’s specified three-group summary.

R4’s caption names Chefer et al. prompts, whereas the layout sheet assigns it the held-out pairs behind R3. Correct the caption’s population and identify the groups and seeds. The large empty box currently contributes substantially to the unfinished appearance.

R6’s caption, “the samplers that correct the product,” neither identifies the six settings nor explains their comparison. Name the settings, shared seed, and any exceptions. Define Langevin and uHMC abbreviations consistently with R1. There are no visible plot labels yet to assess.

R7 is particularly vulnerable to the generic CSV renderer: it excludes axis fields such as `pair` and `group`, yet those are central to this inventory, and it always adds NFE. Implement a dedicated schema for pair, split, composition calls, and retention status. Use wrapped text columns at natural width. Check whether all rows fit at the required font size; an 8-inch placeholder is not evidence that they will.

R9 has no chart yet. Its caption needs to define the sampling window and distinguish the all-steps-trained adapter from applying the selected adapter at all steps. At only two inches high, five settings across five failure categories will need carefully wrapped labels and a compact legend. Use patterns or other redundant distinctions as well as colour, and label the count denominator.

R10’s “early-steps renders” caption is too vague to connect the images to R9. Reuse the exact five setting names and identify each pair and seed. The reader should not have to translate between different labels for the same setting.

R11’s “three concepts” caption gives no comparison key. Specify the four triples and three method columns, using the same method names as R2. Its actual font size and image clarity remain unassessable while empty.

R12 and R13 have both the unresolved-reference problem and sparse, vertically centered float pages. R13 leaves particularly large blank regions. In the TeX, consider local float-page glue such as `\makeatletter\setlength{\@fptop}{0pt}\setlength{\@fpbot}{0pt plus 1fil}\makeatother`, then reassess page packing with the completed figures. Captions should identify the prompt sets, seed, method order, and noise exceptions rather than merely calling the images “renders behind” another table.

R14 needs its own two-level benchmark header and four explicit task columns. The generic renderer will otherwise add NFE and shorten task names into cryptic abbreviations. “Every method” is also unnecessarily vague; identify the model and comparison scope. Keep its table natural-width and centered.

The three changes that would most improve the paper are: replace every production caption, empty box, and broken reference with finished reader-facing material; rebuild R2 and R8 around actual print-size legibility and explicit visual keys; and replace the generic CSV dump with slot-specific, grouped, decimal-aligned booktabs tables that never shrink automatically.