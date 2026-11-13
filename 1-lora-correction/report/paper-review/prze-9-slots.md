# PRZE-9: figure and table slots that refill themselves

Every new figure and table is planned before its result exists: its size in the ICLR layout, its labels, its place in the paper. A script rebuilds it when its result file changes, so a good-enough early result can be placed and replaced as better ones arrive. It builds on the existing figure register (`paper/iclr/figures.md`, statuses reserved, fillable, built, broken) and the placeholder macros already in the paper's source.

## Outcomes

- [x] 📝 markdown created: a layout sheet with one row per new figure and table: column or full text width, aspect ratio, smallest font at final size, panel order, axis and column labels, and the result file it reads ✓ verified 2026-11-10: [the layout sheet](layout-sheet.md), 13 slots R1 to R13
- [x] ✏️ markdown updated: each slot has a row in the figure register with status reserved and the claim its caption will make ✓ verified 2026-11-10: 13 reserved rows in `paper/iclr-56173/figures.md` (the export's register; `paper/iclr/` is the older draft)
- [x] 💻 code ran: every slot compiles in the paper as a placeholder at its final size, so no page shifts when a result arrives ✓ verified 2026-11-10: Tectonic on mscluster, 22 pages, exit 0, no overfull box, log at `paper/iclr-56173/build/iclr2027_conference.log`
- [x] 💻 code ran: a refresh script rebuilds a slot's figure or table from its result file and the paper still compiles ✓ verified 2026-11-10: `1-lora-correction/scripts/refresh_paper_slots.py` refilled R1 from a test CSV with rule 9's columns in a copy of the paper; the copy built with exit 0 and every float on its placeholder page; `--check` prints PASS: 13 slots
- [ ] ✅ check passed: figures are cropped tight with no extra margin, and text in each is legible at final printed size Open: R1 fits the text width at 8 pt with its short labels (test refill 2026-11-10); every image slot is checked when its result lands, and `--compile` prints any table it had to shrink
- [ ] 🖼️ screenshot: a rendered paper page with every slot filled
