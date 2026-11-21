"""Draws the four pictures in ../diagram-prompts.md: the two films, the gap step by step, right pictures through
the wrong film, and the three side by side.

Frames: the project's seed-42 SDXL renders of "a cat" and "a dog", noised by x_t = sqrt(abar) x0 + sqrt(1 - abar) eps
with a fixed seed. Bell curves: one pixel x0 ~ N(1, 0.5^2) under the read's VP schedule (beta 0.1 to 20) at
t = 0, 0.3, 0.5, 1.0. Gap numbers: the read map's chunks 5 and 6; the start gap 0.135 is recomputed here.
Run: python3 make_figures.py   (writes 01- to 04-*.svg beside this script)
"""
import base64, io, math, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/molef/.claude/skills/render-diagrams/scripts')
from mathsvg import T, R, Ln, Ar, wrap, tex, INK, MUTE

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = '/home/molef/PhD/concept-composition/data/pilot/seed_42/a_cat__x__a_dog/'
BLUE, ORANGE = '#1f5fd0', '#c86a0a'
W, H = 1400, 788
STYLE = "<style>text{font-family:'DejaVu Sans',sans-serif}</style>"

# --- the numbers -------------------------------------------------------------------------------
abar_t = lambda t: math.exp(-(0.1 * t + 9.95 * t * t))
FILM_T = [0.0, 0.3, 0.5, 1.0]
FILM_AB = [abar_t(t) for t in FILM_T]
SPREAD = [(math.sqrt(a) * 1.0, a * 0.25 + (1 - a)) for a in FILM_AB]       # (mean, variance) of one pixel
TWO_AB = [1.0, 0.5, 0.25]                                                  # two steps with b = 0.5
m2, v2 = math.sqrt(0.25) * 1.0, 0.25 * 0.25 + 0.75
START = 0.5 * (v2 + m2 ** 2 - 1 - math.log(v2))                            # KL(N(m2, v2) || N(0, 1))
assert abs(START - 0.1351) < 5e-4, START
WRONG = [('start', 0.135), ('step 1', 0.185), ('step 2', 0.382)]; WRONG_TOTAL = 0.702
PERFECT = [('start', 0.135), ('step 1', 0.0), ('step 2', 0.0)]; PERFECT_TOTAL = 0.135
END_GAP, FILM_GAP = 0.000, 0.805

# --- the frames --------------------------------------------------------------------------------
def load(name, size=150):
    im = Image.open(PILOT + name).convert('RGB').resize((size, size), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 127.5 - 1.0
CAT, DOG = load('solo_a.png'), load('solo_b.png')
def noised(x0, ab, seed):
    eps = np.random.default_rng(seed).standard_normal(x0.shape).astype(np.float32)
    x = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * eps
    return np.clip((x + 1) * 127.5, 0, 255).astype(np.uint8)
def uri(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
def frame(arr, x, y, s, col):
    return f'<image href="{uri(arr)}" x="{x:.1f}" y="{y:.1f}" width="{s}" height="{s}"/>' + R(x, y, s, s, 'none', col, 3, 4)

def bell(o, cx, y, w, h, mean, var, col):
    xs = [-3 + 6 * i / 120 for i in range(121)]
    pk = 1 / math.sqrt(2 * math.pi * 0.25)
    pts = [(cx - w / 2 + (x + 3) / 6 * w, y + h - h * (math.exp(-(x - mean) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var)) / pk) for x in xs]
    o.append(Ln(cx - w / 2, y + h, cx + w / 2, y + h, MUTE, 1))
    o.append(Ln(cx, y + h, cx, y + h + 5, MUTE, 1)); o.append(T(cx, y + h + 17, '0', 10, MUTE, 400, 'middle'))
    o.append('<polyline points="' + ' '.join(f'{a:.1f},{b:.1f}' for a, b in pts) + f'" fill="none" stroke="{col}" stroke-width="2.2"/>')
    o.append(T(cx, y + h + 32, f'mean {mean:.2f}, variance {var:.2f}', 11, col, 600, 'middle'))

def save(name, body, desc, w=W, h=H):
    open(os.path.join(HERE, name + '.svg'), 'w').write(wrap(w, h, STYLE + body, desc))

# --- picture 1 ---------------------------------------------------------------------------------
def fig1():
    o = [T(40, 48, 'The two films', 24, INK, 700)]
    s, gap, x0 = 150, 190, 150
    names = ['clean', 'grainier', 'speckled', 'static']
    frames = [noised(CAT, a, 7) for a in FILM_AB]
    o.append(T(40, 96, 'noising film: fixed, nothing learnt', 16, BLUE, 700))
    for i in range(4):
        x = x0 + i * (s + gap)
        o.append(frame(frames[i], x, 110, s, BLUE)); o.append(T(x + s / 2, 280, names[i], 13, BLUE, 600, 'middle'))
        bell(o, x + s / 2, 290, 220, 50, *SPREAD[i], BLUE)
        if i < 3: o.append(Ar(x + s + 12, 185, x + s + gap - 12, 185, BLUE, 2.5, 12))
    o.append(T(40, 448, "the model's film: from static back to a cat", 16, ORANGE, 700))
    for i in range(4):
        x = x0 + i * (s + gap)
        o.append(frame(frames[i], x, 462, s, ORANGE)); o.append(T(x + s / 2, 632, names[i], 13, ORANGE, 600, 'middle'))
        bell(o, x + s / 2, 642, 220, 50, *SPREAD[i], ORANGE)
        if i < 3: o.append(Ar(x + s + gap - 12, 537, x + s + 12, 537, ORANGE, 2.5, 12))
    o.append(T(40, 776, 'under each frame: where one pixel sits across many pictures; the model drawn here matches the noising film exactly', 12, MUTE))
    save('01-the-two-films', '\n'.join(o), 'The noising film left to right in blue and the model film right to left in orange, each frame with one pixel\'s spread')

# --- picture 2 ---------------------------------------------------------------------------------
def strip(o, y, pieces, total, title):
    s = 130
    o.append(T(40, y - 52, title, 17, ORANGE, 700))
    pos = {'x2': 80, 'x1': 300, 'x0': 520}
    for lab, ab in zip(['x0', 'x1', 'x2'], TWO_AB):
        x = pos[lab]; o.append(frame(noised(CAT, ab, 11), x, y, s, ORANGE)); o.append(tex(f'x_{lab[1]}', x + s / 2, y + s + 26, 20, ORANGE)[0])
    lab = {p: v for p, v in pieces}
    o.append(Ar(80 + s + 8, y + s / 2, 300 - 8, y + s / 2, ORANGE, 2, 10)); o.append(Ar(300 + s + 8, y + s / 2, 520 - 8, y + s / 2, ORANGE, 2, 10))
    for x, nm in ((80 + s / 2, 'start'), (80 + s + 45, 'step 1'), (300 + s + 45, 'step 2')):
        o.append(T(x, y + s + 62 if nm == 'start' else y - 12, f'{nm} {lab[nm]:.3f}' if lab[nm] else f'{nm} 0', 15, INK, 700, 'middle'))
    # stacked bar
    bx, by, bh = 760, y - 10, s + 40; scale = bh / 0.8
    cols = {'start': '#7a99d6', 'step 1': ORANGE, 'step 2': '#e8a868'}
    cur = by + bh
    for nm, v in pieces:
        hgt = v * scale
        if hgt > 0:
            o.append(R(bx, cur - hgt, 80, hgt, cols[nm], 'white', 1, 0)); o.append(T(bx + 92, cur - hgt / 2 + 5, f'{nm} {v:.3f}', 13, INK, 600))
        cur -= hgt
    o.append(Ln(bx - 6, cur, bx + 86, cur, INK, 2)); o.append(T(bx + 92, cur - 8, f'total {total:.3f} nats', 15, INK, 700))
    o.append(Ln(bx - 6, by + bh, bx + 86, by + bh, MUTE, 1))

def fig2():
    o = [T(40, 48, 'The gap, step by step', 24, INK, 700)]
    o.append(tex(r'\text{gap} = \text{start} + \text{step 1} + \text{step 2}', 1060, 120, 30, INK)[0])
    strip(o, 150, WRONG, WRONG_TOTAL, 'a wrong model')
    strip(o, 500, PERFECT, PERFECT_TOTAL, 'every step perfect')
    o.append(T(1060, 400, 'The start piece is the same 0.135', 15, MUTE, 600, 'middle'))
    o.append(T(1060, 422, 'in both: the model starts from', 15, MUTE, 600, 'middle'))
    o.append(T(1060, 444, 'static, and x2 is not quite static.', 15, MUTE, 600, 'middle'))
    o.append(T(40, 776, 'one pixel, x0 ~ N(1, 0.5²), two noising steps with b = 0.5; numbers from the read map, chunks 5 and 6', 12, MUTE))
    save('02-the-gap-step-by-step', '\n'.join(o), 'The gap between the noising film and a model film split into start, step 1 and step 2, for a wrong model (0.702 nats) and a perfect one (0.135)')

# --- picture 3 ---------------------------------------------------------------------------------
def fig3():
    o = [T(40, 48, 'Right pictures, wrong film', 24, INK, 700)]
    s, gap, x0 = 150, 130, 160
    blue = [noised(CAT, a, 7) for a in FILM_AB]
    orange = [noised(CAT, 1.0, 7)] + [noised(DOG, a, 7) for a in FILM_AB[1:]]
    for row, (frs, col, lab, y) in enumerate([(blue, BLUE, "the right film: the noising film's frames, run back to the cat", 170), (orange, ORANGE, 'a wrong film: through a dog', 450)]):
        o.append(T(40, y - 20, lab, 16, col, 700))
        for i in range(4):
            x = x0 + i * (s + gap); o.append(frame(frs[i], x, y, s, col))
            if i < 3: o.append(Ar(x + s + gap - 10, y + s / 2, x + s + 10, y + s / 2, col, 2.2, 10))
    o.append(Ln(x0 + s / 2, 170 + s + 8, x0 + s / 2, 450 - 30, INK, 2, '5 4'))
    o.append(T(x0 + s / 2 + 12, 400, f'end-picture gap {END_GAP:.3f}', 17, INK, 700))
    bx = x0 + 3 * (s + gap) + s + 30
    o.append(Ln(bx, 170, bx + 14, 170, INK, 2)); o.append(Ln(bx + 14, 170, bx + 14, 450 + s, INK, 2)); o.append(Ln(bx, 450 + s, bx + 14, 450 + s, INK, 2))
    o.append(T(bx + 26, 450, 'whole-film gap', 17, INK, 700)); o.append(T(bx + 26, 476, f'{FILM_GAP:.3f} nats', 17, INK, 700))
    o.append(T(40, 776, 'both films end on the same cat; the orange middle frames are the seed-42 dog, noised; numbers from the read map, chunk 5', 12, MUTE))
    save('03-right-pictures-wrong-film', '\n'.join(o), 'Two films ending on the same cat, one through a dog: end-picture gap 0.000, whole-film gap 0.805 nats')

# --- picture 4: the capstone -------------------------------------------------------------------
def fig4():
    import cairosvg
    o = []; cw = 620; ch = int(cw * H / W)
    for i, n in enumerate(['01-the-two-films', '02-the-gap-step-by-step', '03-right-pictures-wrong-film']):
        png = cairosvg.svg2png(url=os.path.join(HERE, n + '.svg'), output_width=cw * 2)
        o.append(f'<image href="data:image/png;base64,{base64.b64encode(png).decode()}" x="{20 + i * (cw + 20)}" y="20" width="{cw}" height="{ch}"/>')
    save('04-capstone', '\n'.join(o), 'The two films, the gap step by step, and right pictures through the wrong film, side by side', w=3 * cw + 80, h=ch + 40)

for fn in (fig1, fig2, fig3, fig4): fn()
print('wrote 4 SVGs; start gap recomputed', round(START, 4))
