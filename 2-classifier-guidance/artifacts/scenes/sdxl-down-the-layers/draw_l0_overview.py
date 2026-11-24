"""Level 0: one real SDXL run followed from the prompt to the picture, with CO3's box and Z2's three readings placed
where they act. Every size comes from sdxl_numbers.json; every code line is read from its file and asserted.
Writes l0-sdxl-from-prompt-to-picture.png and .svg.
"""
from sdxl_figkit import FILL, INK, KEY, MONO, N, Canvas, code_line, wrap_code, HERE

U, T, Z, V = N["unet"], N["text"], N["z2"], N["vae"]
DIF = HERE / "source" / "diffusers-0.29.2"
PIPE = DIF / "pipeline_stable_diffusion_xl.py.copy"
ZP = N["sources"]["z2_code"]
e1, e2 = T["encoder_1"], T["encoder_2"]
c = Canvas(20.4, 12.6)

c.text(0.3, 12.25, "SDXL from the prompt to the picture: where CO3 changes the latent and where Z2 takes its three readings",
       size=15, weight="bold", ha="left")
c.text(0.3, 11.78, f"One real run: \"{Z['prompt']}\", seed {Z['seed']}, {N['schedule']['sampling_steps']} sampling steps.   "
       "Blue: what goes in.  Green: SDXL's frozen networks.  Red: what comes out.  Orange: CO3.  "
       "Gold: Loop 12's step Z2, which reads what predicts borrowed parts.", size=9.4, ha="left")

# ---- the text path ------------------------------------------------------------------------------------------
c.box(0.3, 9.65, 2.3, 1.0, f"prompt\n\"{Z['prompt']}\"", kind="input", size=9.5)
c.box(3.0, 9.4, 2.75, 1.5, f"tokenizer\n{T['tokens']} slots: start, an, eagle,\nand, a, fox, end, then\n"
      f"{T['tokens'] - 7} padding slots", kind="input", size=8.6)
c.box(6.15, 10.1, 2.85, 0.8, f"{e1['name']}\n{e1['layers']} layers, {e1['width']} wide", kind="denoiser", size=8.6)
c.box(6.15, 9.05, 2.85, 0.8, f"{e2['name']}\n{e2['layers']} layers, {e2['width']} wide", kind="denoiser", size=8.6)
c.box(9.45, 9.9, 3.1, 1.0, f"word vectors {T['word_vectors'][0]} × {T['word_vectors'][1]}\none per slot: layer before last\n"
      f"of each encoder, {e1['width']} + {e2['width']}", kind="input", size=8.6)
c.box(10.35, 8.55, 2.2, 0.95, f"pooled vector\n{e2['pooled']}, from bigG's\nend slot", kind="input", size=8.6)
c.box(13.0, 9.9, 3.4, 1.0, "Z2 reading 1: text similarity\ncosine of the eagle and fox\n"
      f"vectors: {Z['text_cosine_joint']:.3f} here, {Z['text_cosine_alone']:.3f} alone", kind="reading", size=8.6)
c.line([(2.6, 10.15), (3.0, 10.15)])
c.line([(5.75, 10.5), (6.15, 10.5)])
c.line([(5.75, 9.45), (6.15, 9.45)])
c.line([(9.0, 10.5), (9.45, 10.5)])
c.line([(9.0, 9.6), (9.45, 10.0)])
c.line([(9.0, 9.2), (10.35, 9.0)])
c.line([(12.55, 10.4), (13.0, 10.4)], KEY["reading"], lw=1.6)

# ---- the conditioning that enters the U-Net ------------------------------------------------------------------
c.box(6.3, 7.6, 2.6, 0.8, f"timestep t: 981, ..., 1\n→ {U['time_features']} waves\n→ time vector {U['time_vector']}",
      kind="input", size=8.4)
c.box(10.35, 7.55, 3.1, 0.75, f"pooled {e2['pooled']} + {U['size_numbers']} size numbers = {U['added_input']}\n"
      f"→ {U['time_vector']}, added to the time vector", kind="input", size=8.2)
c.line([(11.45, 8.55), (11.45, 8.3)])
c.line([(7.6, 7.6), (7.6, 7.3)])
c.line([(11.9, 7.55), (11.9, 7.3)])
c.line([(9.75, 9.9), (9.75, 7.3)], KEY["input"], lw=1.6)
c.text(9.66, 8.0, "by cross-\nattention", size=7.6, ha="right", color=KEY["input"])

# ---- the denoising loop --------------------------------------------------------------------------------------
lv = {l["part"]: l for l in U["levels"]}
g = U["transformer_blocks_by_grid"]
c.box(6.3, 4.3, 7.15, 3.0, "", kind="denoiser")
c.ax.text(9.875, 6.75, "SDXL U-Net, run on [empty prompt, prompt]", ha="center", va="center", fontsize=10.5, weight="bold",
          color=KEY["denoiser"], zorder=4)
c.ax.text(9.875, 5.55, f"in: $x_t$, {U['in_channels']} × {U['sample_size']} × {U['sample_size']}      "
          f"out: noise guess $\\hat\\epsilon$, {U['out_channels']} × {U['sample_size']} × {U['sample_size']}\n"
          f"widths {U['widths'][0]}, {U['widths'][1]}, {U['widths'][2]} on grids {U['grids'][0]}, {U['grids'][1]}, {U['grids'][2]}\n"
          f"{U['resblocks']} ResBlocks each add the time vector\n"
          f"{U['transformer_blocks']} transformer blocks read the word vectors\n"
          f"({g['64']} on the 64 × 64 grid, {g['32']} on 32 × 32, none on 128 × 128)\n"
          "opened in figures 0.5a and 0.5b", ha="center", va="center", fontsize=8.8, color=INK, zorder=4, linespacing=1.35)

c.box(0.3, 5.45, 2.3, 0.9, f"$x_T$: pure noise\n{U['in_channels']} × {U['sample_size']} × {U['sample_size']}", kind="input", size=9)
c.box(3.0, 5.0, 2.95, 1.8, "CO3, steps 0 to 5 only\nreplaces $x_t$ with $x_t^{(2)}$\nbefore the ordinary step\n"
      "(opened in the CO3 ladder)", kind="method", size=8.6)
c.line([(2.6, 5.9), (3.0, 5.9)], KEY["method"], lw=1.6)
c.line([(5.95, 5.9), (6.3, 5.9)], KEY["method"], lw=1.6)
c.line([(1.45, 5.45), (1.45, 4.62), (6.3, 4.62)], INK)
c.text(4.1, 4.42, "steps 6 to 49 go straight in", size=8.2)

c.box(13.85, 6.15, 3.55, 1.15, "Z2 reading 2: attention overlap\nthe eagle and fox columns of all\n"
      f"{Z['layers_captured']} cross-attention layers, summed", kind="reading", size=8.6)
c.box(13.85, 4.5, 3.55, 1.3, "Z2 reading 3: the $d_t$ maps\n$d_t(c_1) = \\epsilon^{c_1} - \\epsilon^{\\phi}$ (eagle minus empty),\n"
      f"{U['out_channels']} × {U['sample_size']} × {U['sample_size']}; cosine of the two", kind="reading", size=8.6)
c.line([(13.45, 6.7), (13.85, 6.7)], KEY["reading"], lw=1.6)
c.line([(13.45, 5.15), (13.85, 5.15)], KEY["reading"], lw=1.6)
c.text(15.62, 4.2, f"read in one extra U-Net call at {len(Z['read_steps'])} steps;\nthe picture is unchanged by it",
       size=7.9, color=KEY["reading"])

c.box(7.3, 3.15, 5.15, 0.7, "noise guess $\\hat\\epsilon = \\epsilon^{\\phi} + \\lambda\\,(\\epsilon^{C} - \\epsilon^{\\phi})$:  $\\phi$ empty prompt, $C$ the prompt",
      kind="output", size=8.6)
c.box(7.3, 2.1, 5.15, 0.7, "DDIM step: clean guess $\\hat x_0$ (figure 0.5c), then $x_{t-20}$", kind="output", size=8.8)
c.line([(9.875, 4.3), (9.875, 3.85)])
c.line([(9.875, 3.15), (9.875, 2.8)])
c.line([(7.3, 2.45), (0.75, 2.45), (0.75, 5.45)], KEY["output"], lw=1.4)
c.text(4.0, 2.27, f"next step, {N['schedule']['sampling_steps']} in all", size=8.2, color=KEY["output"])
c.box(13.0, 1.95, 2.75, 1.0, f"VAE decoder\n{V['latent'][0]} × {V['latent'][1]} × {V['latent'][2]} → "
      f"{V['pixels'][0]} × {V['pixels'][1]} × {V['pixels'][2]}", kind="denoiser", size=8.6)
c.line([(12.45, 2.45), (13.0, 2.45)])
c.text(12.73, 2.68, "after\nstep 49", size=7.4, va="bottom")
c.image(N["sources"]["picture"], 16.55, 0.3, 3.3, 3.3)
c.line([(15.75, 2.45), (16.55, 2.45)], KEY["output"])
c.text(14.4, 1.35, f"the picture, {V['pixels'][1]} × {V['pixels'][2]} pixels", size=8.2, color=KEY["output"])
c.text(14.4, 1.0, f"{Z['prompt']}, seed {Z['seed']}", size=8.2, color=KEY["output"])

# ---- the step strip --------------------------------------------------------------------------------------
c.text(0.3, 1.62, "the 50 steps:", size=8.6, ha="left", weight="bold")
X0, DX = 0.35, 0.236
for i in range(50):
    k = "method" if i < 6 else "fixed"
    c.ax.add_patch(__import__("matplotlib").patches.Rectangle((X0 + i * DX, 0.95), DX * 0.86, 0.35, fc=FILL[k],
                   ec=KEY[k], lw=0.8, zorder=3))
    if i in Z["read_steps"]:
        c.ax.plot([X0 + i * DX + DX * 0.43], [1.125], marker="o", ms=4.2, color=KEY["reading"], zorder=5)
for i in (0, 5, 10, 20, 30, 40, 49):
    c.text(X0 + i * DX + DX * 0.43, 0.72, str(i), size=7.6)
c.text(0.3, 0.32, "orange: CO3 acts (steps 0 to 5).  gold dot: Z2 reads (steps 0 to 5, then 10, 15, ..., 45).", size=8.2, ha="left")

# ---- code boxes ------------------------------------------------------------------------------------------
def code_box(x, y, w, h, title, items, kind, width, plain=""):
    body = "\n".join(wrap_code(n, code_line(path, n, must), width) for path, n, must in items)
    c.box(x, y, w, h, "", kind="white", ec=KEY[kind], lw=1.5)
    c.ax.text(x + 0.12, y + h - 0.1, title, ha="left", va="top", fontsize=8.6, weight="bold", color=KEY[kind], zorder=4)
    c.ax.text(x + 0.12, y + h - 0.62, body, ha="left", va="top", fontsize=6.9, family=MONO, color=INK, zorder=4, linespacing=1.2)
    if plain:
        c.ax.text(x + 0.12, y + 0.12, plain, ha="left", va="bottom", fontsize=7.8, color=INK, zorder=4, style="italic",
                  linespacing=1.2)

code_box(16.8, 8.75, 3.45, 2.15, "Prompt to vectors\n(pipeline_stable_diffusion_xl.py)", [
    (PIPE, 402, "pooled_prompt_embeds = prompt_embeds[0]"),
    (PIPE, 404, "prompt_embeds = prompt_embeds.hidden_states[-2]"),
    (PIPE, 411, "prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)")], "input", 44,
    "[-2]: the layer before last.\nconcat: 768 + 1280 = 2048 per slot.")
code_box(17.65, 4.0, 2.6, 3.3, "Z2 keeps the maps\n(z2_predictors.py)", [
    (ZP, 65, "p = probs.view(batch, attn.heads, seq, -1)"),
    (ZP, 102, 'name.endswith("attn2.processor")'),
    (ZP, 113, '"dA": (eps[2] - eps[0]).cpu()')], "reading", 33,
    "attn2 is cross-attention;\nmean(0) averages heads.")
c.save("l0-sdxl-from-prompt-to-picture")
