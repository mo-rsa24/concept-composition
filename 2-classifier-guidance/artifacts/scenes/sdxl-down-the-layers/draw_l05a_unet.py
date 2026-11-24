"""Level 0.5a: inside the SDXL U-Net. Three grid sizes down, a middle, three back up; real channels, grids, skip
widths, ResBlock and transformer-block counts, where the time vector and the word vectors enter, and which blocks
hold the 70 cross-attention layers Z2 reads. Every number from sdxl_numbers.json.
Writes l05a-inside-the-unet.png and .svg.
"""
from sdxl_figkit import INK, KEY, N, Canvas

U, T = N["unet"], N["text"]
L = {l["part"]: l for l in U["levels"]}
c = Canvas(22.6, 11.0)
c.text(0.3, 10.65, "Inside the SDXL U-Net: three grid sizes down, a middle, three back up; the words are read only on the "
       "64 × 64 and 32 × 32 grids", size=14.5, weight="bold", ha="left")
c.text(0.3, 10.2, "One box per block: its grid, its width in channels, what it holds.  Lines across the top are the skip "
       "connections, each with the channels it carries.  Gold: the cross-attention layers Z2 reads, "
       f"{U['cross_attention_layers']} in all; Z2 upscales each layer's map to {U['sample_size']} × {U['sample_size']} and sums them.",
       size=9.0, ha="left")

CX = dict(zip(["down 0", "down 1", "down 2", "middle", "up 0", "up 1", "up 2"], [3.2, 5.9, 8.6, 11.3, 14.0, 16.7, 19.4]))
CY = {128: 7.9, 64: 6.0, 32: 4.1}
W, H = 2.3, 1.5


def ypos(l):
    return 2.2 if l["part"] == "middle" else CY[l["grid"]]


def body(l):
    s = f"{l['part']}  ·  {l['grid']} × {l['grid']}\n{l['channels']} channels\n{l['resblocks']} ResBlocks"
    if l["transformer_blocks"]:
        s += (f"\n{l['transformer_stacks']} stack{'s' if l['transformer_stacks'] > 1 else ''} × {l['blocks_per_stack']} transformer\nblocks, "
              f"{l['heads']} heads of {l['head_dim']}")
    else:
        s += "\nno transformer blocks"
    return s


for part, l in L.items():
    x, y = CX[part], ypos(l)
    c.box(x - W / 2, y - H / 2, W, H, body(l), kind="denoiser", size=8.1, name=part,
          ec=KEY["reading"] if l["transformer_blocks"] else KEY["denoiser"], lw=2.2 if l["transformer_blocks"] else 1.5)
    if l["transformer_blocks"]:
        c.text(x, y - H / 2 - 0.05, f"Z2 reads {l['transformer_blocks']}\ncross-attention layers", size=7.8, color=KEY["reading"], va="top")
    if "resblock_inputs" in l:
        ins = ", ".join(str(v) for v in l["resblock_inputs"])
        c.text(x, y + H / 2 + 0.05, f"ResBlock inputs\n{ins}", size=7.4, color=INK, va="bottom")

# input and output
c.box(0.2, 7.3, 1.55, 1.2, f"$x_t$\n{U['in_channels']} × {U['sample_size']} × {U['sample_size']}\nconv_in → {U['widths'][0]}", kind="input", size=8.2)
c.box(20.95, 7.3, 1.55, 1.2, f"conv_out → {U['out_channels']}\n$\\hat\\epsilon$\n{U['out_channels']} × {U['sample_size']} × {U['sample_size']}", kind="output", size=8.2)
c.line([(1.75, 7.9), (CX["down 0"] - W / 2, 7.9)])
c.line([(CX["up 2"] + W / 2, 7.9), (20.95, 7.9)])

# down and up arrows between levels
order = ["down 0", "down 1", "down 2", "middle", "up 0", "up 1", "up 2"]
for a, b in zip(order[:-1], order[1:]):
    la, lb = L[a], L[b]
    xa, ya, xb, yb = CX[a], ypos(la), CX[b], ypos(lb)
    yb_in = yb + 0.45 if b == "middle" else yb
    ya_out = ya + 0.45 if a == "middle" else ya
    if yb < ya:   # going down: from the bottom edge of a to the left edge of b
        c.line([(xa - 1.0, ya - H / 2), (xa - 1.0, yb_in), (xb - W / 2, yb_in)])
    else:         # going up: from the right edge of a to the bottom edge of b
        c.line([(xa + W / 2, ya_out), (xb + 1.0, ya_out), (xb + 1.0, yb - H / 2)])
c.text(0.3, 5.0, "a line down then right: the grid\nis halved (downsample).\na line right then up: the\ngrid is doubled (upsample).",
       size=8.0, ha="left")

# skip connections across the top, one per grid, with the channels each carries
skips = U["skips"]
for grid, (a, b), y in ((128, ("down 0", "up 2"), 9.35), (64, ("down 1", "up 1"), 7.05), (32, ("down 2", "up 0"), 5.15)):
    chans = [s["channels"] for s in skips if s["grid"] == grid]
    ya, yb = CY[grid] + H / 2, CY[grid] + H / 2
    c.line([(CX[a] + 0.85, ya), (CX[a] + 0.85, y), (CX[b] - 0.85, y), (CX[b] - 0.85, yb)], KEY["fixed"], lw=1.4)
    c.text((CX[a] + CX[b]) / 2, y + 0.17, f"skips at {grid} × {grid}: {', '.join(map(str, chans))} channels, "
           "joined to the up block's input", size=7.9, color=KEY["fixed"])

# conditioning at the bottom
c.box(0.2, 0.2, 6.2, 1.25, f"time vector {U['time_vector']}: a small network on t ({U['time_features']} waves)\n"
      f"+ a small network on pooled text {T['encoder_2']['pooled']} and {U['size_numbers']} size numbers ({U['added_input']})\n"
      f"added inside every one of the {U['resblocks']} ResBlocks", kind="input", size=8.4)
c.box(15.6, 0.2, 6.8, 1.25, f"word vectors {T['word_vectors'][0]} × {T['word_vectors'][1]}\n"
      f"the keys and values of every one of the {U['cross_attention_layers']} cross-attention layers\n"
      f"({U['transformer_blocks_by_grid']['64']} on 64 × 64, {U['transformer_blocks_by_grid']['32']} on 32 × 32)",
      kind="input", size=8.4)
mx, my = CX["middle"], 2.2
c.line([(6.4, 0.82), (8.6, 0.82), (8.6, my - 0.4), (mx - W / 2, my - 0.4)], KEY["input"], lw=1.5)
c.text(8.75, 1.15, "into every\nResBlock", size=7.6, color=KEY["input"], ha="left")
c.line([(15.6, 0.82), (14.0, 0.82), (14.0, my - 0.4), (mx + W / 2, my - 0.4)], KEY["input"], lw=1.5)
c.text(13.85, 1.2, "into every\ntransformer\nblock", size=7.6, color=KEY["input"], ha="right")
c.save("l05a-inside-the-unet")
