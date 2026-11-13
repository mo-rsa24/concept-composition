# The composition notebook, cell by cell

**what it is** the worked-examples notebook for claims 1 and 7 of the overlap-dial idea map, drawn
as a scene by hld-diagram's notebook level: an outline of its six sections (cut at its own `##`
headings), each opening into its cells and functions. Every line can be walked; each numeric line
opens on a name for what it computes and its maths in numbered steps; each cell shows the figure it
drew behind its "out" badge; the words in its markdown link to the concept cards and notes that
teach them; `one_step_grads` carries a "Run it for real" `/plumb` prompt.

**published** https://claude.ai/artifact/DBmJtn6xy75sPtQwPEvQZU, and listed in the scene library
(`Obsidian/Mathematics/references/scenes/Scenes.md`, "Kinds of composition notebook").

**built from** the notebook
`artifacts/ideas/poe-overlap-dial/what-an-expert-learns-and-kinds-of-composition.ipynb` (its cells,
its stored figures from its last run, its markdown); `notebook-maths.json`, the name and stepped maths
for all 52 numeric lines, each checked against the code it describes; `top.mmd` and `sec1.mmd` to
`sec6.mmd`, drafted by `read_config.py --draft-mmd` and rendered to the matching `.svg` with
mermaid-cli 11.4.2; the spec `what-an-expert-learns-cell-by-cell.json`.

**how to rebuild**

```bash
cd artifacts/scenes/what-an-expert-learns-cell-by-cell
python3 -I ~/.claude/skills/hld-diagram/scripts/build_scene.py \
  --spec what-an-expert-learns-cell-by-cell.json \
  --out "$PWD/what-an-expert-learns-cell-by-cell-scene.html" --no-link-check
```

After the notebook changes, redraft the drawings first (`read_config.py --file <notebook>
--draft-mmd top.mmd`, then `--section N` for each), render them, and run `--maths-todo` against
`notebook-maths.json` for any line that lost its maths. Rerun the notebook in Jupyter before building
when its code changed, so the figures behind the badges are current.

**why** to read the notebook by its argument (six sections) rather than as a wall of cells, with the
maths of every line one click away and each idea linked to where the learning tree teaches it.

**depends on** the notebook at its path above, read at build time; `~/.claude/cache/term-index.json`
for the word links; KaTeX 0.16.9 from cdnjs, drawing MathML, so the maths shows on the published page
and not on this file opened straight from disk.
