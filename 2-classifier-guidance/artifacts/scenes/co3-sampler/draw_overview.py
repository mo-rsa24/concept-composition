"""Level 0: one CO3 run on SDXL, annotated stage by stage, with code boxes beside the stages and a small forward
diagram of one step. Every number comes from co3_numbers.json; every code line is read from the authors'
checkout and asserted. Writes co3-overview.png and co3-overview.svg.
"""
from co3_figkit import BG, FILL, INK, KEY, MONO, N, Canvas, code_line, fmt, wrap_code, HERE

CO3 = "/home/molef/PhD/replications/co3/composers/Co3.py"
ATTN = "/home/molef/PhD/replications/co3/composers/utils/attn_utils.py"
steps, s = N["steps"], N["settings"]
c = Canvas(18.4, 11.2)

# ---- the x and y maps: steps 0 to 5 drawn wider than steps 6 to 49 (said on the face) --------------------
X0, DX = 1.75, 0.95
XB = 7.25
def X(i):
    return X0 + i * DX if i <= 5 else XB + (i - 6) * (12.2 - XB) / 43
Y0, YH = 4.35, 4.15
def Y(v):
    return Y0 + v * YH

c.text(0.25, 10.85, "CO3 on SDXL: one run of 50 DDIM steps at the released settings", size=15.5, weight="bold", ha="left")
c.text(0.25, 10.4, "y: how much of the clean picture $x_t$ still carries, $\\sqrt{\\bar\\alpha_t}$ (0 is pure noise, 1 is clean).   "
       "x: sampling step i with its timestep t.   Steps 0 to 5 are drawn wider than steps 6 to 49.", size=9.6, ha="left")

c.bracket(X(0) - 0.42, X(2) + 0.42, 9.45, "CO3 resampler\nAlgorithm 2", size=11, color=KEY["method"])
c.bracket(X(3) - 0.42, X(5) + 0.42, 9.45, "CO3 corrector\nAlgorithm 3", size=11, color=KEY["method"])
c.bracket(X(6) - 0.15, X(49) + 0.15, 9.45, "plain DDIM step\nwith CFG++ only", size=11, color=KEY["fixed"])

# axes
c.line([(1.2, Y0), (6.85, Y0)], INK, lw=1.1, arrow=False)
c.line([(7.0, Y0), (12.45, Y0)], INK, lw=1.1, arrow=False)
c.line([(1.2, Y0), (1.2, Y(1.0) + 0.05)], INK, lw=1.1, arrow=False)
for xx in (6.85, 7.0):
    c.ax.plot([xx - 0.05, xx + 0.05], [Y0 - 0.1, Y0 + 0.1], color=INK, lw=1.1, zorder=2)
for v in (0.0, 1.0):
    c.ax.plot([1.12, 1.2], [Y(v), Y(v)], color=INK, lw=1.1)
    c.text(1.05, Y(v), f"{v:.0f}", size=9, ha="right")
c.text(0.42, Y(0.5), "$\\sqrt{\\bar\\alpha_t}$", size=13, rotation=90)

# per-step table above the first six steps
rows = (("passes P", "iterations"), ("$\\beta$ in eq. 14", "beta"), ("U-Net\nimages", "unet_images"))
for k, (head, key) in enumerate(rows):
    yy = 8.2 - k * 0.5
    c.text(1.05, yy, head, size=9, ha="right", color=KEY["method"] if k < 2 else INK)
    for i in range(6):
        v = steps[i][key]
        c.text(X(i), yy, f"{v:g}" if isinstance(v, float) else str(v), size=10.5,
               weight="bold" if k == 0 else "normal", color=KEY["method"] if k < 2 else INK)
c.text((X(0) + X(5)) / 2, 6.75, "each of these six steps then also runs the plain DDIM step", size=8.8, style="italic")
c.text((X(0) + X(5)) / 2, 6.38, f"U-Net images per pass: {N['weights']['K'] + 2} (empty prompt, joint prompt, "
       f"{N['concepts'][0]}, {N['concepts'][1]})", size=8.8, style="italic")

# the curve and its dots
xs = [X(i) for i in range(6)]
c.curve(xs, [Y(r["sqrt_alpha_bar"]) for r in steps[:6]], INK, lw=1.6)
c.curve([X(5), X(6)], [Y(steps[5]["sqrt_alpha_bar"]), Y(steps[6]["sqrt_alpha_bar"])], INK, lw=1.2, ls=":")
c.curve([X(i) for i in range(6, 50)], [Y(r["sqrt_alpha_bar"]) for r in steps[6:]], INK, lw=1.6)
for i, r in enumerate(steps):
    col = KEY["method"] if r["co3"] else KEY["fixed"]
    fc = col if r["algorithm"] == "resampler" else ("white" if r["co3"] else col)
    c.ax.plot([X(i)], [Y(r["sqrt_alpha_bar"])], marker="o", ms=7 if r["co3"] else 3.5, mfc=fc, mec=col, mew=1.6, zorder=5)
for i in range(6):
    c.text(X(i), Y0 - 0.33, f"{i}\nt={steps[i]['t']}", size=8.2, linespacing=1.1)
for i in (6, 20, 35, 49):
    c.text(X(i), Y0 - 0.33, f"{i}\nt={steps[i]['t']}", size=8.2, linespacing=1.1)
c.text(6.7, Y0 - 0.95, "sampling step i (timestep t), 20 timesteps apart", size=9.2)
c.text(X(0) + 0.62, Y(steps[0]["sqrt_alpha_bar"]) + 0.42, f"{fmt(steps[0]['sqrt_alpha_bar'], 3)} at t=981", size=8.4)
c.text(10.55, 5.05, f"44 steps, {steps[6]['unet_images']} U-Net images each:\nempty prompt and joint prompt,\n"
       f"guidance {s['guidance_scale']} with CFG++", size=8.8, color=KEY["fixed"])
c.text(11.75, 9.0, f"step 49, t = 1: returns $\\hat x_0$ directly", size=8.6, ha="right")

# ---- code boxes beside the stages -------------------------------------------------------------------------
def code_box(x, y, w, h, title, items, kind, width, plain=""):
    body = "\n".join(wrap_code(n, code_line(path, n, must), width) for path, n, must in items)
    c.box(x, y, w, h, "", kind="white", ec=KEY[kind], lw=1.6)
    c.ax.text(x + 0.14, y + h - 0.22, title, ha="left", va="center", fontsize=9.4, weight="bold", color=KEY[kind], zorder=4)
    c.ax.text(x + 0.14, y + h - 0.45, body, ha="left", va="top", fontsize=7.4, family=MONO, color=INK, zorder=4, linespacing=1.25)
    c.ax.text(x + 0.14, y + 0.16, plain, ha="left", va="bottom", fontsize=8.4, color=INK, zorder=4, style="italic", linespacing=1.25)

code_box(0.2, 0.2, 4.35, 2.95, "Which CO3 runs at this step (Co3.py)", [
    (CO3, 279, "if t in self.config.latent_corrector_ts:"),
    (CO3, 306, "beta = 1.0 if t == self.start_t"),
    (CO3, 307, "num_corrector_steps = 10 if"),
    (CO3, 309, "if t >= self.scheduler.timesteps[self.config.num_resampling_steps-1]:")], "method", 50,
    "t = 981 and 961: resample, 10 passes.\nt = 941: resample, 5 passes.\nt = 921, 901, 881: correct, 5 passes.")
code_box(4.7, 0.2, 3.95, 2.95, "One resampler pass (attn_utils.py)", [
    (ATTN, 183, "noise_pred = unet(latent_model_input"),
    (ATTN, 191, "concept_weights = self._get_concept_weights_from_dists"),
    (ATTN, 196, "denoised_tweedie_at = x - (1 - at).sqrt() * composed_noise_lmda"),
    (ATTN, 198, "x_2 = denoised_tweedie_at + (1 - at).sqrt() * noise_pred_uncond")], "method", 46,
    "x cancels in line 196, so the new latent\nis built only from the four predictions.")
code_box(8.8, 0.2, 3.75, 2.95, "Plain DDIM step, CFG++ (Co3.py)", [
    (CO3, 354, "noise_pred = noise_pred_uncond + self.config.guidance_scale"),
    (CO3, 355, "denoised_tweedie = (x - (1-at).sqrt() * noise_pred) / at.sqrt()"),
    (CO3, 357, "denoised_latent = at_next.sqrt() * denoised_tweedie + (1-at_next).sqrt() * noise_pred_uncond")], "fixed", 44,
    "Guidance 0.8 builds the clean estimate;\nthe renoise uses the empty prompt's\nprediction (the CFG++ choice).")

# ---- the small forward diagram of one step --------------------------------------------------------------
c.text(15.55, 10.0, "One step, as the code runs it", size=11.5, weight="bold")
MX, BW = 14.55, 2.3
c.box(MX - BW / 2, 9.0, BW, 0.6, "$x_t$, 4 × 128 × 128", kind="input", size=10)
c.box(MX - BW / 2, 7.8, BW, 0.78, "one of the first 6 steps?\nAlgorithm 1 line 2", kind="white", ec=KEY["method"], size=8.8)
c.box(16.25, 7.65, 1.95, 1.08, "CO3: P passes\nresampler or\ncorrector", kind="method", size=8.8)
c.box(MX - BW / 2, 6.45, BW, 0.85, "U-Net on $[x_t, x_t]$\nempty and joint prompt", kind="denoiser", size=8.8)
c.box(MX - BW / 2, 5.2, BW, 0.85, "CFG++: $\\hat x_0$ from guided $\\epsilon$,\nrenoise with $\\epsilon^{\\phi}$", kind="white", ec=KEY["fixed"], size=8.8)
c.box(MX - BW / 2, 4.15, BW, 0.6, "$x_{t-20}$", kind="output", size=10.5)
c.line([(MX, 9.0), (MX, 8.58)], INK)
c.line([(MX + BW / 2, 8.25), (16.25, 8.25)], KEY["method"], lw=1.6)
c.text(15.95, 8.42, "yes", size=8.4, color=KEY["method"])
c.line([(MX, 7.8), (MX, 7.3)], INK)
c.text(MX - 0.25, 7.55, "no", size=8.4, ha="right")
c.line([(17.22, 7.65), (17.22, 6.88), (MX + BW / 2, 6.88)], KEY["method"], lw=1.6)
c.text(17.32, 7.25, "$x_t^{(2)}$", size=10, ha="left", color=KEY["method"])
c.line([(MX, 6.45), (MX, 6.05)], INK)
c.line([(MX, 5.2), (MX, 4.75)], INK)
c.line([(MX - BW / 2, 4.45), (13.08, 4.45), (13.08, 9.3), (MX - BW / 2, 9.3)], KEY["fixed"], lw=1.4)
c.text(12.86, 6.9, "50 steps", size=9, rotation=90, color=KEY["fixed"])
c.line([(MX, 4.15), (MX, 3.62)], INK)
c.text(MX + 0.12, 3.88, "after step 49: VAE decode", size=8.4, ha="left")
c.image(HERE / "source" / "co3-eagle-fox-seed-6001.png", MX - 1.0, 1.6, 2.0, 2.0)
r = N["render"]
c.text(MX, 1.12, f"\"{r['prompt']}\", seed {r['seed']}\nloop-6 render, {r['seconds']} s", size=8.2)
c.box(16.0, 0.35, 2.25, 3.25, "", kind="white", ec=KEY["method"], lw=1.4, container=True)
# one panel of exact-plane/co3-exact-plane.png (eagle x fox, step 4), cropped by the fractions its layout gives:
# canvas 23.95 x 12.9 units, that panel and its lines from x 16.75 to 20.15 and y 0.95 to 5.5
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

_im = Image.open(HERE / "exact-plane" / "co3-exact-plane.png").convert("RGB")
_w, _h = _im.size
_crop = np.asarray(_im.crop((int(_w * 16.75 / 23.95), int(_h * (12.9 - 5.5) / 12.9),
                             int(_w * 20.15 / 23.95), int(_h * (12.9 - 0.95) / 12.9))))
_iw = 2.35 * 3.4 / 4.55
c.ax.imshow(_crop, extent=(17.125 - _iw / 2, 17.125 + _iw / 2, 1.15, 3.5), zorder=3, interpolation="bilinear")
c.ax.set_xlim(0, c.w)
c.ax.set_ylim(0, c.h)
_frame = Rectangle((17.125 - _iw / 2, 1.15), _iw, 2.35, fill=False, ec=KEY["method"], lw=1.4, zorder=4)
c.ax.add_patch(_frame)
c.obstacles.append((_frame, "exact-plane thumbnail", "rect"))
c.ax.text(17.125, 0.72, "Worked case, real latents:\nexact-plane/ (one of its\ntwelve panels)", ha="center", va="center",
          fontsize=7.9, color=KEY["method"], zorder=4, linespacing=1.25)

c.save("co3-overview")
