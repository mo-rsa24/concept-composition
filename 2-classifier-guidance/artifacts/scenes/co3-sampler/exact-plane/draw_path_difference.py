"""Where CO3 changes the path: at each of the 50 sampler steps, how far the CO3 latent is from the latent the same seed
and prompt reach with CO3 switched off, as a share of the plain latent's length. Both pairs; CO3's six steps shaded.

y: ||x_CO3 - x_plain|| / ||x_plain|| over the latent's 65,536 numbers, from passes.npz (co3_path) and
plain-joint-path.npy, via exact_plane_numbers.json. x: sampler step 0 to 49 with its timestep.
At steps 0 to 5 the CO3 latent is recorded after CO3's passes at that step, so step 0 already carries the resampler.

  python3 draw_path_difference.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from matplotlib.patches import Rectangle  # noqa: E402

from co3_figkit import FILL, INK, KEY, Canvas  # noqa: E402

J = json.loads((HERE / "exact_plane_numbers.json").read_text())
CC, EF = J["pairs"]["a_cheetah__x__a_cougar"], J["pairs"]["an_eagle__x__a_fox"]
W, H = 17.0, 9.6
c = Canvas(W, H, scale=0.82)
X0, X1, Y0, Y1 = 2.0, 13.4, 1.6, 7.6
YMAX = 1.2
fx = lambda i: X0 + (X1 - X0) * i / 49
fy = lambda v: Y0 + (Y1 - Y0) * v / YMAX

c.text(0.3, H - 0.4, "Where CO3 changes the path: the latent leaves the plain run during CO3's six steps and keeps leaving after", size=14.5, weight="bold", ha="left")
c.text(0.3, H - 0.95, r"y: $\|x^{\mathrm{CO3}}-x^{\mathrm{plain}}\|\,/\,\|x^{\mathrm{plain}}\|$ at each step, over the latent's "
       "65,536 numbers (0 means the same latent, 1 means the gap is as long as the plain latent itself).   "
       "x: sampler step i, timestep t = 981 - 20i.", size=10, ha="left", color="#3b4552")
c.text(0.3, H - 1.35, "Same seed (6001) and joint prompt; the plain run has CO3's correction switched off and is otherwise the "
       "same sampler. Source: passes.npz (co3_path) and plain-joint-path.npy.", size=10, ha="left", color="#3b4552")

shade = Rectangle((fx(-0.5), Y0), fx(5.5) - fx(-0.5), Y1 - Y0, fc=FILL["method"], ec="none", zorder=0)
c.ax.add_patch(shade)
c.text((fx(-0.5) + fx(5.5)) / 2, Y1 - 0.25, "CO3's\nsix steps", size=9.5, color=KEY["method"], weight="bold", va="top")
# axes
c.line([(X0, Y0), (X1 + 0.2, Y0)], color=INK, lw=1.2, arrow=False)
c.line([(X0, Y0), (X0, Y1)], color=INK, lw=1.2, arrow=False)
for v in (0, 0.25, 0.5, 0.75, 1.0):
    c.line([(X0 - 0.1, fy(v)), (X0, fy(v))], color=INK, lw=1.0, arrow=False)
    c.text(X0 - 0.17, fy(v), f"{v:g}", size=9.5, ha="right")
    if v:
        c.ax.plot([X0, X1], [fy(v)] * 2, color="#e6e8ec", lw=0.7, zorder=0)
for i in (0, 5, 10, 20, 30, 40, 49):
    c.line([(fx(i), Y0 - 0.1), (fx(i), Y0)], color=INK, lw=1.0, arrow=False)
    c.text(fx(i), Y0 - 0.2, f"{i}\nt={981 - 20 * i}", size=9, va="top")
c.text((X0 + X1) / 2, Y0 - 0.95, "sampler step i", size=10.5)
c.text(X0 - 1.25, (Y0 + Y1) / 2, "gap between the CO3 and\nplain latents, over the\nplain latent's length", size=10,
       rotation=90)

for P, col, ls, name in ((CC, INK, "-", "cheetah x cougar"), (EF, KEY["fixed"], "--", "eagle x fox")):
    v = P["path"]["relative_distance"]
    c.curve([fx(i) for i in range(50)], [fy(x) for x in v], color=col, lw=2.0, ls=ls)
    c.text(fx(49) + 0.15, fy(v[-1]) + (0.12 if P is CC else -0.12), f"{name}: {v[-1]:.2f}", size=10, ha="left", color=col,
           weight="bold")

# the step-5 value, labelled on the thing
v5c, v5e = CC["path"]["relative_distance"][5], EF["path"]["relative_distance"][5]
v0c, v0e = CC["path"]["relative_distance"][0], EF["path"]["relative_distance"][0]
c.line([(fx(9), fy(0.5)), (fx(5) + 0.05, fy(max(v5c, v5e)) + 0.08)], color=KEY["method"], lw=1.0)
c.text(fx(9) + 0.05, fy(0.5) + 0.05,
       f"after step 0 (10 resampler passes):\n{v0c:.3f} cheetah x cougar, {v0e:.3f} eagle x fox\n"
       f"after CO3's last step (i = 5):\n{v5c:.3f} cheetah x cougar, {v5e:.3f} eagle x fox",
       size=9.5, ha="left", va="bottom", color=KEY["method"])
c.text(fx(8), fy(0.97), "steps 6 to 49 run the same plain sampler in both runs,\nyet the gap keeps growing, to over 1 at the end",
       size=9.5, ha="left", va="bottom")

c.save("exact-plane/co3-path-difference")
