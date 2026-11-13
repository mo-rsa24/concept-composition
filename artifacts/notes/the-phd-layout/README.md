# The PhD layout, as one picture

The PhD folder sorted by kind of work, as the [PhD organogram](/home/molef/PhD/ORGANOGRAM.md)
stands on 2026-11-13: two research portfolios with their projects, writing, replications with the
published-methods hub, cluster bring-up, and the archive.

## the-phd-layout.png

<a href="the-phd-layout.png"><img src="the-phd-layout.png" width="400" alt="The PhD folder sorted by kind of work"></a>

**What it shows**

A tree read left to right: the PhD splits into five kinds of work (research portfolios, writing,
replications, admin, archive), each kind into its folders, and the two portfolios into their
projects. Colour marks only the kind: blue research, green writing, purple other people's code,
grey admin and archive. The symlinks at old folder names are left out, so the picture shows only
where things live now.

**Where it came from**

Drawn by `make_the_phd_layout.py` in this folder, which holds the tree as data and writes
`the-phd-layout.svg` and `the-phd-layout.png`. Its prompt entry is in [the prompt file](diagram-prompts.md).
It is embedded in the repository README under "Where this repo sits in the PhD" and at the top of
the PhD organogram. When the organogram's folders change, edit the tree in the script and run it
again.
