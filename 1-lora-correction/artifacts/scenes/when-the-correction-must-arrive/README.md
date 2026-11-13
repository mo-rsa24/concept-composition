# When in the run the correction is needed, as a page you can drive

The inspector's correction-timing tab, ported to a static page that opens from this folder with no
server and no tunnel. Pick a pair and seed, slide the 10-step window from step 0 to step 49, and
the picture, the scorer's call and the marker on the compose-rate curve move together. "Pin this
one" holds a window beside the current one, so an early window and a late one can be compared by eye.

<a href="index.html"><img src="img/a_cat__x__a_dog/seed_9/0-10.jpg" width="200" alt="Cat and dog, seed 9, correction injected over steps 0 to 10: two animals"></a>

**What it shows**

8 animal pairs × 4 seeds × 9 window positions (width 10 steps, stride 5, 50 DDIM steps). Outside
the window nothing is injected and the prompt stays the same, so the only thing that changes across
the slider is when the correction acts. The curve is the compose rate per window centre over 32
scored renders each: 0.66 at centre 5, 0.25 at 10, 0.09 at 15, 0.03 at 20, and 0 to 0.03 from 25 on.
The scorer is the instance count (two or more animals counts as composing). This is the evidence
behind step 6 of the running order, judged in
[its review](../../plans/completed/03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md).

**How to open it**

Double-click `index.html`, or serve the folder (`python3 -m http.server` here, then
`http://localhost:8000`). The manifest is inlined in the page, and the pictures are 512 px JPEG
copies in `img/<pair>/seed_<n>/<window>.jpg`.

**Where it came from**

[`loader/build_scene.py`](loader/build_scene.py), run on the cluster on 2026-11-13. It rebuilds the
timing manifest from the window sweep on disk with `1-lora-correction/scripts/build_window_manifest.py`,
copies every picture the manifest names (originals under
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window/pairs/`; each cell's
`source_image` in `manifest.json` names its original), and renders the inspector's own template
from `1-lora-correction/scripts/window_tab.py`. The only change to that template is that pictures
load from these copies instead of the inspector's Flask route. The header's "544 of 288 cells
generated" is the builder's own count: the sweep folder holds cells beyond the planned grid, and
the curve uses only the 288 planned ones.
