"""Level 0.5: the co3-hybrid schedule over the 50 DDIM steps: which steps resample, which correct, how many passes,
which beta, and the alpha-bar the code uses at each step, with the branch's rules boxed beneath. Every number from
co3_numbers.json, every rule's line asserted against Co3.py. Writes co3-schedule.png and co3-schedule.svg.
"""
from co3_figkit import FILL, INK, KEY, MONO, N, Canvas, code_line, fmt

CO3 = "/home/molef/PhD/replications/co3/composers/Co3.py"
steps, s, T = N["steps"], N["settings"], N["totals"]
ts = N["timesteps"]
c = Canvas(17.6, 10.6)

c.text(0.3, 10.25, "The co3-hybrid schedule: CO3 runs on the first 6 of 50 steps, resampling 3 then correcting 3",
       size=14.5, weight="bold", ha="left")
c.text(0.3, 9.85, "One cell per DDIM step, left to right from pure noise (t = 981) to the last step (t = 1). Settings are the "
       "released ones in co3_common.py; every number is from co3_numbers.json.", size=9.3, ha="left")

# ---- the strip of 50 steps --------------------------------------------------------------------------------
X0, CW, SY, SH = 0.9, 0.318, 5.25, 0.62
colour = {"resampler": (KEY["method"], KEY["method"]), "corrector": (KEY["method"], "#f4c79a"), "plain": ("#b9c0c8", FILL["fixed"])}
for r in steps:
    ec, fc = colour[r["algorithm"]]
    c.ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (X0 + r["step"] * CW, SY), CW, SH, facecolor=fc, edgecolor="white", lw=0.8, zorder=3))
c.ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((X0, SY), 50 * CW, SH, fill=False, ec=INK, lw=1.0, zorder=4))
for i in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49):
    c.text(X0 + (i + 0.5) * CW, SY - 0.2, str(i), size=8.2)
    c.text(X0 + (i + 0.5) * CW, SY - 0.48, str(ts[i]), size=8.2, color=KEY["fixed"])
c.text(X0 - 0.12, SY - 0.2, "step i", size=8.2, ha="right")
c.text(X0 - 0.12, SY - 0.48, "t", size=8.2, ha="right", color=KEY["fixed"])
c.text(X0 + 28 * CW, SY + SH / 2, f"steps 6 to 49 (t = {ts[6]} down to {ts[-1]}): no CO3, only the plain DDIM step with CFG++",
       size=9.2, color=KEY["fixed"], bg=True)

# ---- the zoomed six CO3 steps -------------------------------------------------------------------------------
ZX, ZW, ZY, ZH = 0.9, 1.55, 6.75, 1.55
c.box(ZX - 0.12, ZY - 0.12, 6 * ZW + 0.24, ZH + 0.24, "", kind="white", ec=KEY["method"], lw=1.0, container=True)
c.ax.plot([X0, ZX - 0.12], [SY + SH, ZY - 0.12], color=KEY["method"], lw=0.9, ls=":", zorder=1)
c.ax.plot([X0 + 6 * CW, ZX + 6 * ZW + 0.12], [SY + SH, ZY - 0.12], color=KEY["method"], lw=0.9, ls=":", zorder=1)
for r in steps[:6]:
    x = ZX + r["step"] * ZW
    ec, fc = colour[r["algorithm"]]
    word = "resample" if r["algorithm"] == "resampler" else "correct"
    c.box(x + 0.04, ZY, ZW - 0.08, ZH, "", kind=fc, ec=ec, lw=1.4, round_=0.05)
    tc = "white" if r["algorithm"] == "resampler" else INK
    c.ax.text(x + ZW / 2, ZY + ZH - 0.25, f"t = {r['t']}", ha="center", va="center", fontsize=10, weight="bold", color=tc, zorder=4)
    c.ax.text(x + ZW / 2, ZY + ZH - 0.58, word, ha="center", va="center", fontsize=9.5, color=tc, zorder=4)
    c.ax.text(x + ZW / 2, ZY + 0.52, f"P = {r['iterations']} passes", ha="center", va="center", fontsize=9, color=tc, zorder=4)
    c.ax.text(x + ZW / 2, ZY + 0.22, f"$\\beta$ = {r['beta']:g}", ha="center", va="center", fontsize=9, color=tc, zorder=4)
c.bracket(ZX, ZX + 3 * ZW, 8.62, "resampler, Algorithm 2 ($\\sum w_k = 0$)", size=10.5, color=KEY["method"])
c.bracket(ZX + 3 * ZW, ZX + 6 * ZW, 8.62, "corrector, Algorithm 3 ($\\sum w_k = 1$)", size=10.5, color=KEY["method"])

# the cost, beside the zoom
c.box(10.6, 6.9, 6.7, 1.52, "", kind="white", ec=INK, lw=1.0)
c.ax.text(10.8, 8.13, "What the six steps cost in U-Net images", ha="left", va="center", fontsize=10, weight="bold", color=INK, zorder=4)
c.ax.text(10.8, 7.5, f"CO3 passes: {T['resampler_iterations']} resampling + {T['corrector_iterations']} correcting "
          f"= {T['co3_iterations']},\neach a batch of {N['weights']['K'] + 2} (empty, joint, {N['concepts'][0]}, {N['concepts'][1]}).\n"
          f"Whole run: {T['unet_images_co3']} images, against {T['unet_images_plain_cfg']} for plain CFG\n(50 steps × 2).",
          ha="left", va="center", fontsize=9, color=INK, zorder=4, linespacing=1.3)

# ---- alpha-bar under the strip --------------------------------------------------------------------------------
BY, BH = 2.55, 1.75
c.line([(X0, BY), (X0 + 50 * CW + 0.05, BY)], INK, lw=0.9, arrow=False)
for r in steps:
    col = KEY["method"] if r["co3"] else "#b9c0c8"
    c.ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (X0 + r["step"] * CW + 0.04, BY), CW - 0.08, BH * r["alpha_bar"], facecolor=col, edgecolor="none", zorder=3))
c.text(X0 - 0.12, BY + BH, "1", size=8.2, ha="right")
c.text(X0 - 0.12, BY, "0", size=8.2, ha="right")
c.text(X0 + 0.55, BY + BH + 0.08, "$\\bar\\alpha_t$ as CO3 uses it: the share of the clean picture's variance still in $x_t$ "
       "(Co3.py line 91 prepends 1.0, so this is the scheduler's value at t − 1)", size=8.6, ha="left", va="bottom")
for i, dx in ((0, 0.0), (5, 0.0), (49, -0.1)):
    r = steps[i]
    c.text(X0 + (i + 0.5) * CW + dx, BY + BH * r["alpha_bar"] + 0.2, fmt(r["alpha_bar"], 4), size=8.2,
           color=KEY["method"] if r["co3"] else INK, ha="center" if i < 49 else "right")
fs = N["first_step"]
c.text(5.4, BY + 1.15, f"at t = {fs['t']}: {fmt(fs['alpha_bar'], 5)} here, {fmt(fs['alpha_bar_without_prepend'], 5)} in the "
       f"scheduler at t ({100 * fs['relative_shift']:.1f}% apart)", size=8.4, color=KEY["fixed"])

# ---- the rules, boxed beneath ------------------------------------------------------------------------------
rules = [(103, "self.scheduler.timesteps[:t_cond]", f"IF t is one of the first {s['num_ts_to_correct']}\ntimesteps ({ts[0]} … {ts[5]})\nTHEN run CO3 first", "Co3.py 103, 279"),
         (309, "if t >= self.scheduler.timesteps[self.config.num_resampling_steps-1]:", f"IF t ≥ timesteps[{s['num_resampling_steps'] - 1}] = {N['resample_from_t']}\nTHEN resampler\nELSE corrector", "Co3.py 309"),
         (307, "num_corrector_steps = 10 if t >= self.scheduler.timesteps[1]", f"IF t ≥ timesteps[1] = {N['ten_iterations_from_t']}\nTHEN P = {N['iterations_early']}\nELSE P = {s['num_latent_corrector_steps']}", "Co3.py 307"),
         (306, "beta = 1.0 if t == self.start_t else self.config.beta", f"IF t = timesteps[0] = {ts[0]}\nTHEN $\\beta$ = {N['beta_first_step']:g}\nELSE $\\beta$ = {s['beta']:g}", "Co3.py 306")]
RW = 4.1
for k, (n, must, rule, where) in enumerate(rules):
    code_line(CO3, n, must)
    x = 0.3 + k * (RW + 0.17)
    c.box(x, 0.2, RW, 1.72, "", kind="white", ec=KEY["method"], lw=1.2)
    c.ax.text(x + 0.15, 1.08, rule, ha="left", va="center", fontsize=9.4, color=INK, zorder=4, linespacing=1.3)
    c.ax.text(x + RW - 0.15, 0.36, where, ha="right", va="center", fontsize=8, family=MONO, color=KEY["fixed"], zorder=4)
c.text(0.3, 2.05, f"The paper's own settings differ: CO3 on 10 steps, $\\beta$ = 0.8 and guidance $\\lambda$ = 5.0 (pp. 6 and 17); "
       f"the release runs {s['num_ts_to_correct']} steps, $\\beta$ = {s['beta']:g} and guidance {s['guidance_scale']:g} with CFG++.",
       size=8.8, ha="left", color=KEY["broken"])

c.save("co3-schedule")
