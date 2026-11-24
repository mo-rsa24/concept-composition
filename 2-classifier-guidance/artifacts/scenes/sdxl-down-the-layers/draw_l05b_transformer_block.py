"""Level 0.5b: one transformer block at 32 x 32 (1280 channels), with cross-attention opened beneath it and the
point where Z2's KeepCrossAttention keeps the attention weights. Sizes from sdxl_numbers.json.
Writes l05b-one-transformer-block.png and .svg.
"""
from sdxl_figkit import FILL, INK, KEY, N, Canvas

B, U, T, Z = N["block"]["32"], N["unet"], N["text"], N["z2"]
B64 = N["block"]["64"]
P, C, Hh, D = B["patches"], B["channels"], B["heads"], B["head_dim"]
W77, W2048 = T["word_vectors"]
c = Canvas(22.8, 10.6)
c.text(0.25, 10.25, f"One transformer block at 32 × 32: patches ask patches, then ask the words, then each patch is "
       "reworked alone; Z2 reads the middle step", size=14.5, weight="bold", ha="left")
c.text(0.25, 9.8, f"{U['transformer_blocks_by_grid']['32']} blocks like this one sit at {C} channels on the 32 × 32 grid "
       f"(down 2, middle, up 0); {U['transformer_blocks_by_grid']['64']} more sit at {B64['channels']} channels on 64 × 64 "
       "(down 1, up 1), drawn in journey 02.  Each + circle adds a step's input back to its output.", size=9.0, ha="left")

Y, H = 7.2, 1.1
def sub(x, w, text, kind="denoiser", size=8.4, ec=None, lw=1.5, h=H):
    return c.box(x, Y - h / 2, w, h, text, kind=kind, size=size, ec=ec, lw=lw)

sub(0.2, 1.75, f"h from the\nResBlock\n{C} × 32 × 32", kind="input")
sub(2.3, 2.25, f"GroupNorm, flatten\nto {P} patches,\nlinear {C} → {C}", size=8.0)
sub(4.9, 0.85, "Layer\nNorm", kind="fixed", size=7.6, h=0.75)
sub(6.1, 1.95, f"self-attention\npatch to patch\n{P} × {P} per head")
c.node(8.5, Y, 0.2, "+", size=11, kind="white", ec=INK)
sub(9.05, 0.85, "Layer\nNorm", kind="fixed", size=7.6, h=0.75)
sub(10.25, 1.95, f"cross-attention\npatch to word\n{P} × {W77} per head", ec=KEY["reading"], lw=2.4)
c.node(12.65, Y, 0.2, "+", size=11, kind="white", ec=INK)
sub(13.2, 0.85, "Layer\nNorm", kind="fixed", size=7.6, h=0.75)
sub(14.4, 2.05, f"feed-forward\n{C} → 2 × {B['ff_inner']},\none half gates\nthe other → {C}")
c.node(16.95, Y, 0.2, "+", size=11, kind="white", ec=INK)
sub(17.5, 2.2, f"linear {C} → {C},\nback to\n{C} × 32 × 32", size=8.0)
c.node(20.2, Y, 0.2, "+", size=11, kind="white", ec=INK)
sub(20.75, 1.85, f"to the next\nResBlock\n{C} × 32 × 32", kind="output")

for a, b in ((1.95, 2.3), (4.55, 4.9), (5.75, 6.1), (8.05, 8.3), (8.7, 9.05), (9.9, 10.25), (12.2, 12.45), (12.85, 13.2),
             (14.05, 14.4), (16.45, 16.75), (17.15, 17.5), (19.7, 20.0), (20.4, 20.75)):
    c.line([(a, Y), (b, Y)])
for x0, x1 in ((4.72, 8.5), (8.88, 12.65), (13.02, 16.95)):
    c.line([(x0, Y), (x0, 8.25), (x1, 8.25), (x1, Y + 0.2)], KEY["fixed"], lw=1.3)
    c.text((x0 + x1) / 2, 8.4, "adds its input back", size=7.6, color=KEY["fixed"])
c.line([(1.07, Y + H / 2), (1.07, 9.0), (20.2, 9.0), (20.2, Y + 0.2)], KEY["fixed"], lw=1.3)
c.text(10.6, 9.15, "the whole stack adds its input back too", size=7.6, color=KEY["fixed"])

c.line([(4.9, 6.42), (4.9, 6.28), (17.15, 6.28), (17.15, 6.42)], INK, lw=1.2, arrow=False)
c.text(7.6, 6.03, f"one transformer block: repeats {U['transformer_layers_per_block'][2]} times in a row in each "
       f"32 × 32 stack ({U['transformer_layers_per_block'][1]} at 64 × 64)", size=8.2)

# ---- cross-attention opened ---------------------------------------------------------------------------------
c.box(0.2, 0.2, 22.4, 5.35, "", kind="white", ec=KEY["reading"], ls="--", lw=1.2, container=True)
c.line([(11.22, Y - H / 2), (11.22, 5.55)], KEY["reading"], lw=1.4)
c.text(0.5, 5.15, "cross-attention opened: every patch asks every word slot", size=11, weight="bold", ha="left",
       color=KEY["reading"])

c.box(0.5, 3.35, 2.4, 1.0, f"word vectors\n{W77} × {W2048}", kind="input", size=8.6)
c.box(0.5, 1.4, 2.4, 1.0, f"patches after\nLayerNorm\n{P} × {C}", kind="denoiser", size=8.4)
c.box(3.4, 3.95, 2.2, 0.6, f"to_v: {W2048} → {C}", kind="white", ec=KEY["denoiser"], size=8.2)
c.box(3.4, 3.15, 2.2, 0.6, f"to_k: {W2048} → {C}", kind="white", ec=KEY["denoiser"], size=8.2)
c.box(3.4, 1.6, 2.2, 0.6, f"to_q: {C} → {C}", kind="white", ec=KEY["denoiser"], size=8.2)
c.box(6.1, 3.95, 2.3, 0.6, f"V: {W77} × {C}", kind="denoiser", size=8.4)
c.box(6.1, 3.15, 2.3, 0.6, f"K: {W77} × {C}", kind="denoiser", size=8.4)
c.box(6.1, 1.6, 2.3, 0.6, f"Q: {P} × {C}", kind="denoiser", size=8.4)
c.line([(2.9, 4.0), (3.4, 4.25)])
c.line([(2.9, 3.7), (3.4, 3.45)])
c.line([(2.9, 1.9), (3.4, 1.9)])
for y in (4.25, 3.45, 1.9):
    c.line([(5.6, y), (6.1, y)])
c.box(9.0, 1.75, 3.75, 1.7, f"split into {Hh} heads of {D} numbers;\nper head, scores = $QK^\\top/\\sqrt{{{D}}}$\n"
      f"A = softmax of each row\nA: {P} patches × {W77} slots,\neach row sums to 1", kind="denoiser", size=8.3)
c.line([(8.4, 3.45), (9.0, 3.05)])
c.line([(8.4, 1.9), (9.0, 2.2)])
c.box(13.35, 2.9, 2.75, 1.05, f"A V per head: {P} × {D}\n{Hh} heads joined:\n{P} × {C}", kind="denoiser", size=8.3)
c.line([(8.4, 4.25), (14.7, 4.25), (14.7, 3.95)])
c.line([(12.75, 3.2), (13.35, 3.2)])
c.box(16.6, 2.9, 2.4, 1.05, f"to_out: {C} → {C}\nback up to the\nblock's + ", kind="denoiser", size=8.3)
c.line([(16.1, 3.42), (16.6, 3.42)])
c.box(19.4, 3.0, 2.9, 0.85, "$\\mathrm{softmax}(QK^\\top/\\sqrt{d})\\,V$", kind="white", ec=INK, size=11, lw=1.0)
c.box(13.35, 0.45, 8.95, 1.95, "Z2's KeepCrossAttention keeps A here, for the prompt's entry of the batch only:\n"
      f"the mean over the {Hh} heads gives {P} × {W77};\n"
      f"it keeps 4 columns: eagle (slot {Z['token_columns'][0]}), fox (slot {Z['token_columns'][1]}), "
      f"end (slot {Z['token_columns'][2]}), mean of padding (slots {Z['token_columns'][3]} to {W77 - 1});\n"
      f"each column is a 32 × 32 map, upscaled to {U['sample_size']} × {U['sample_size']} and summed over "
      f"{Z['layers_captured']} layers.\nThe sampler's own call computes the same A inside one fused kernel and never stores it;\nthat is why Z2 makes one extra call with this keeping processor.", kind="reading", size=8.4)
c.line([(10.9, 1.75), (10.9, 1.4), (13.35, 1.4)], KEY["reading"], lw=1.5)
c.save("l05b-one-transformer-block")
