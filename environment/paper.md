# 🗺️ Environment: where the paper's LaTeX lives and how it builds

Navigation: 📋 [Index](00-INDEX.md) | [Overview](overview.md#paper-latex-build)

The manuscript is `writing/iclr/`, owned by the
`1-lora-correction/plans/07-writing-the-paper/` scope. Editing and building it
is a different loop from running experiments: no GPU, no queue, no job.

## Table of contents

- [Where to make changes](#where-to-make-changes)
- [No system LaTeX exists on this cluster](#no-system-latex-exists-on-this-cluster)
- [Building it, the terminal route (what Claude and any script uses)](#building-it-the-terminal-route-what-claude-and-any-script-uses)
- [Building it, the VS Code route](#building-it-the-vs-code-route)
- [🖼️ In pictures, code and further reading](#in-pictures-code-and-further-reading)

## Where to make changes

Navigation: 📋 [TOC](#table-of-contents) | [Next](#no-system-latex-exists-on-this-cluster) ➡️

All prose goes in `writing/iclr/iclr2027_conference.tex`, the single manuscript file. The other
files in that folder are the ICLR distribution and are not edited: `iclr2027_conference.sty`
(layout), `.bst` (bibliography style), `natbib.sty`, `fancyhdr.sty`, and `math_commands.tex`
(macro definitions, worth reading before defining a new macro since it likely already exists).
References go in `iclr2027_conference.bib`. Figures are referenced, never copied in: they are
produced by `1-lora-correction/plans/03-does-the-correction-cause-composition/`
and `1-lora-correction/plans/04-does-the-fix-reach-unseen-pairs/`, and
`writing/iclr/README.md` holds the path rule.

**The cluster checkout still holds the old `paper/` folder.** On 2026-11-21 it sat at `ccf76f13`,
before the move to `writing/`. Its `paper/iclr/figures/` holds 188 files: 132 tracked and 56
untracked PDFs that no `.gitignore` rule covers. A fast-forward past the move carries the 132 to
`writing/` and leaves the 56 PDFs in `paper/iclr/figures/`, where the rewritten links no longer
look. `verified` (`git ls-files` on the cluster checkout and `git check-ignore` against the current
`.gitignore`, 2026-11-21)

## No system LaTeX exists on this cluster

Navigation: ⬅️ [Previous](#where-to-make-changes) | 📋 [TOC](#table-of-contents) | [Next](#building-it-the-terminal-route-what-claude-and-any-script-uses) ➡️

Confirmed live on 2026-08-24: `pdflatex`, `xelatex`, `latexmk`, and `bibtex` are all absent from
PATH (`which` returns nothing for each). Every LaTeX recipe except `tectonic` fails here,
including the extension's own default recipe (`latexmk`, see below). Any instruction, script, or
CI step that assumes a normal TeX install is wrong on this cluster.

## Building it, the terminal route (what Claude and any script uses)

Navigation: ⬅️ [Previous](#no-system-latex-exists-on-this-cluster) | 📋 [TOC](#table-of-contents) | [Next](#building-it-the-vs-code-route) ➡️

```bash
cd writing/iclr
/home-mscluster/mmolefe/.local/bin/tectonic iclr2027_conference.tex
```
Confirmed present and executable at that path live on 2026-08-24. `tectonic` is a
self-contained engine: it needs no TeX installation and fetches packages and fonts on demand
into `~/.cache/Tectonic`. The FIRST build on a node therefore needs network access. Later builds
are offline and fast.

**Reading the output.** The build writes `iclr2027_conference.pdf` next to the `.tex`. The
repository-wide `*.pdf` rule in `.gitignore` is commented out, so a PDF is kept out only by a path
rule: `writing/restoring-plurality-in-compositional-diffusion_overleaf/**/*.pdf` covers the export. The paper is read by building
it, either through the VS Code PDF viewer or from a laptop over the same SSH tunnel the LoRA
Inspector is reached through.

**Build noise that is not failure.** A successful build still prints underfull `\vbox` warnings
and `TeX rerun seems needed, but stopping at 6 passes`. Both are normal here. Judge a build by
whether the PDF was written, not by whether the output was silent.

## Building it, the VS Code route

Navigation: ⬅️ [Previous](#building-it-the-terminal-route-what-claude-and-any-script-uses) | 📋 [TOC](#table-of-contents) | [Next](#in-pictures-code-and-further-reading) ➡️

LaTeX Workshop (`james-yu.latex-workshop`, version `10.15.2`, installed server-side, confirmed
present live via `ls ~/.vscode-server/extensions/` on 2026-08-24) is the extension.
`Ctrl+Shift+P` -> `LaTeX Workshop: Build with recipe` -> `tectonic`, then `LaTeX Workshop: View
LaTeX PDF` to see it.

**The extension's default recipe is `latexmk`, which does not exist on this cluster.** Plain
"Build LaTeX project" (no recipe chosen) therefore fails, not because anything is wrong with the
document. The recipe has to be picked by hand as above. Nine recipes ship by default;
`latexmk` is first, `tectonic` is last (confirmed by reading
`~/.vscode-server/extensions/james-yu.latex-workshop-10.15.2/package.json` on 2026-08-05).

**A tracked `.vscode/settings.json` defines a custom `tectonic` recipe and tool.** The file was added
in commit `2c4d0b6` ("Working-tree state before the retrofit rename") and read live on 2026-08-24.
It points at the absolute `tectonic` path above and does not set
`latex-workshop.latex.recipe.default`. Whether plain "Build LaTeX project" uses `tectonic`
automatically, or still tries `latexmk` first and fails, is untested: running the build would write
a PDF outside the environment folder's scope. It is listed under "Still open" in `00-INDEX.md`.

Absence of a project or machine settings file was confirmed by `ls` on 2026-08-05. Since the
`.vscode/settings.json` addition a project settings file exists.

The VS Code keystroke route above is user-stated (2026-08-05) and consistent with the recipe
list read from the extension's `package.json`, but the command-palette action itself cannot be
driven from a shell, so it has never been machine-verified.

<!-- related:start (drain) -->
<a id="in-pictures-code-and-further-reading"></a>

## 🖼️ In pictures, code and further reading

Navigation: ⬅️ [Previous](#building-it-the-vs-code-route) | 📋 [TOC](#table-of-contents)

No picture of the paper build exists in any `artifacts/` folder. The one piece of code behind this page is the tracked settings file that adds the `tectonic` recipe, shown whole so the open question above can be read against it.

### The code behind it

Whole file `.vscode/settings.json` (lines 1 to 22):

```json
{
  "latex-workshop.latex.recipes": [
    {
      "name": "tectonic",
      "tools": ["tectonic"]
    }
  ],
  "latex-workshop.latex.tools": [
    {
      "name": "tectonic",
      "command": "/home-mscluster/mmolefe/.local/bin/tectonic",
      "args": [
        "-X",
        "compile",
        "%DOC%"
      ],
      "env": {}
    }
  ],
  "latex-workshop.view.pdf.viewer": "tab",
  "latex-workshop.synctex.afterBuild.enabled": true
}
```

- `recipes` defines one recipe named `tectonic` that runs the tool of the same name.
- `tools` points that tool at the absolute `tectonic` path and runs `tectonic -X compile` on the open document.
- No `latex-workshop.latex.recipe.default` key appears, which is the fact behind the unverified question on this page: nothing in the file says plain "Build LaTeX project" should pick `tectonic`.
- The file's `viewer` and `synctex` lines are about the PDF view and are not part of the build.

### Go deeper

| What | Link |
|---|---|
| The environment overview of the paper build | [overview, paper build](overview.md#paper-latex-build) |
| How the `tectonic` and settings facts were established | [provenance](provenance.md) |
| Syncing the paper with Overleaf | [the Overleaf sync recipe](../runbook/syncing-the-paper-with-overleaf.md) |
| The manuscript folder's own readme | [writing readme](../writing/iclr/README.md) |

### Still open

- A recorded `tectonic` build of `iclr2027_conference_revision` sits in the git-ignored folder `artifacts/drips/present-revision-to-supervisor/bundle-build-proof/`. No recorded run of the plain "Build LaTeX project" command exists in `artifacts/`, and the page itself says no build was run to check the default recipe.

<!-- related:end -->

<!-- built-to: PAGE_LOOK_FORMAT v1 -->
