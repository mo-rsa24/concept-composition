"""Level 0.5c: the noise guess and the clean guess (the Tweedie mean). The real eagle and fox picture noised to SDXL's
alpha-bar at steps 0, 5, 20 and 45 (a pixel-space illustration: the real x_t is the 4 x 128 x 128 latent), the
U-Net's noise guess turned into a clean guess, and the slot for the decoded clean guesses at CO3's corrected steps,
which waits for job 94450. Every number from sdxl_numbers.json. Writes l05c-noise-guess-and-clean-guess.png and .svg.
"""
import numpy as np
from PIL import Image

from sdxl_figkit import FILL, INK, KEY, N, Canvas

S, U, Z = N["schedule"], N["unet"], N["z2"]
steps = S["steps"]
shown = S["shown_steps"]
c = Canvas(21.0, 11.4)
c.text(0.3, 11.05, "The noise guess and the clean guess: how much of $x_t$ is picture at steps 0, 5, 20 and 45, "
       "and how the noise guess becomes a clean guess", size=13.6, weight="bold", ha="left")
c.text(0.3, 10.6, f"SDXL's schedule: scaled-linear betas from {S['beta_start']} to {S['beta_end']} over "
       f"{S['train_steps']:,} training steps; {S['sampling_steps']} sampling steps with leading spacing, timesteps "
       f"{steps[0]['t']} down to {steps[-1]['t']}.   $\\bar\\alpha_t$ is the share of $x_t$'s variance that is picture.",
       size=9.2, ha="left")

# ---- row 1: the forward equation and the real picture at four steps ---------------------------------------
c.box(0.3, 8.35, 4.7, 1.0, "$x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$", kind="white",
      ec=INK, size=14, lw=1.0)
c.text(0.35, 7.75, "$x_0$: the clean latent.\n$\\epsilon$: Gaussian noise of the same size,\n"
       f"{U['in_channels']} × {U['sample_size']} × {U['sample_size']}.\n"
       "Under each picture: its step, its\ntimestep, $\\bar\\alpha_t$ and the two weights.", size=8.6, ha="left", va="top")
c.text(0.35, 5.55, "These four are the picture itself\nnoised in pixels, to show the share.\nThe sampler noises the latent.",
       size=8.0, ha="left", va="top", style="italic", color=KEY["fixed"])

pic = np.asarray(Image.open(N["sources"]["picture"]).convert("RGB").resize((384, 384)), dtype=np.float64) / 127.5 - 1
noise = np.random.default_rng(0).standard_normal(pic.shape)
X = [5.5, 9.4, 13.3, 17.2]
for x, k in zip(X, shown):
    s = steps[k]
    xt = s["sqrt_alpha_bar"] * pic + s["sqrt_one_minus"] * noise
    c.array(np.clip((xt + 1) / 2, 0, 1), x, 6.75, 3.3, 3.3, ec=KEY["input"], name=f"step {k}")
    pct = f"{s['share_pct']:.1f}" if s["share_pct"] < 10 else f"{s['share_pct']:.0f}"
    c.text(x + 1.65, 6.15, f"step {k}, t = {s['t']}:  {pct}% picture\n"
           f"$\\bar\\alpha_t$ = {s['alpha_bar']:.4f}\n"
           f"$\\sqrt{{\\bar\\alpha_t}}$ = {s['sqrt_alpha_bar']:.3f},  $\\sqrt{{1-\\bar\\alpha_t}}$ = {s['sqrt_one_minus']:.3f}",
           size=8.6, va="center")
c.line([(X[0], 5.62), (X[1] + 3.3, 5.62)], KEY["method"], lw=2.0, arrow=False)
c.text(X[0] + 3.5, 5.4, "CO3 acts at steps 0 to 5", size=8.2, color=KEY["method"])

# ---- row 2: from the noise guess to the clean guess -------------------------------------------------------
Y = 3.55
c.box(0.3, Y - 0.5, 1.9, 1.0, f"$x_t$\n{U['in_channels']} × {U['sample_size']} × {U['sample_size']}", kind="input", size=9)
c.box(2.9, Y - 0.6, 2.4, 1.2, "U-Net\nguesses the noise", kind="denoiser", size=9.2)
c.box(6.0, Y - 0.5, 2.3, 1.0, "noise guess $\\hat\\epsilon$\nsame size as $x_t$", kind="output", size=9)
c.box(9.1, Y - 0.75, 6.0, 1.5, "$\\hat x_0 = \\dfrac{x_t - \\sqrt{1-\\bar\\alpha_t}\\,\\hat\\epsilon}{\\sqrt{\\bar\\alpha_t}}$",
      kind="white", ec=INK, size=15, lw=1.0)
c.box(15.8, Y - 0.5, 2.3, 1.0, "clean guess $\\hat x_0$\nthe Tweedie mean", kind="output", size=9)
c.line([(2.2, Y), (2.9, Y)])
c.line([(5.3, Y), (6.0, Y)])
c.line([(8.3, Y), (9.1, Y)])
c.text(8.7, Y + 0.32, "$\\hat\\epsilon$", size=10)
c.line([(1.25, Y - 0.5), (1.25, 2.3), (12.1, 2.3), (12.1, Y - 0.75)])
c.text(6.7, 2.12, "$x_t$ is used again: subtract the guessed noise at its weight, then undo the picture's weight", size=8.4)
c.line([(15.1, Y), (15.8, Y)])
c.text(15.3, 2.62, "the DDIM step builds the next latent (CO3's code puts\nthe empty prompt's $\\hat\\epsilon$ in the second term):", size=8.2, ha="left")
c.text(15.3, 2.0, "$x_{t-20} = \\sqrt{\\bar\\alpha_{t-20}}\\,\\hat x_0 + \\sqrt{1-\\bar\\alpha_{t-20}}\\,\\hat\\epsilon$",
       size=10, ha="left")
s0, s45 = steps[0], steps[45]
m0 = s0["sqrt_one_minus"] / s0["sqrt_alpha_bar"]
m45 = s45["sqrt_one_minus"] / s45["sqrt_alpha_bar"]
c.text(12.1, 4.75, f"an error in $\\hat\\epsilon$ reaches $\\hat x_0$ multiplied by $\\sqrt{{1-\\bar\\alpha_t}}/\\sqrt{{\\bar\\alpha_t}}$: "
       f"{m0:.1f} at step 0, {m45:.2f} at step 45", size=8.6)

# ---- row 3: the slot for the decoded clean guesses at CO3's corrected steps -----------------------------
c.box(0.3, 0.25, 20.4, 1.45, "", kind="white", ec=KEY["fixed"], ls="--", lw=1.3, container=True)
for k in range(6):
    x = 0.6 + k * 1.55
    c.box(x, 0.4, 1.25, 1.15, f"step {k}\nt = {steps[k]['t']}", kind="white", ec=KEY["method"], ls="--", size=8.2)
c.text(10.1, 0.97, "Waiting: the decoded clean guess $\\hat x_0$ at each step CO3 corrects, before and after its passes,\n"
       "from job 94450 (pending in the bigbatch queue on 2026-11-24).  Source once it lands:\n"
       "/datasets/mmolefe/poe_repair_min/loop-12/co3-inside/.  No picture until then.", size=8.6, ha="left",
       color=KEY["fixed"])
c.save("l05c-noise-guess-and-clean-guess")
