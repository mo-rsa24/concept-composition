"""Level 2 of the NoisyCLIP ladder: the miniature's figures, every number read from miniature_answer_key.json.

Run from this folder: PYTHONDONTWRITEBYTECODE=1 python3 make_miniature_figures.py [figure stems]
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_kit import *  # noqa: E402,F403
from figure_kit import _width  # noqa: E402

K = json.loads((HERE / 'miniature_answer_key.json').read_text())
SAMP, BR, TR = K['sampler'], K['breaks'], K['training']
TOY = 'Toy with hand-set weights: it shows the mechanics and predicts nothing about SDXL.'


def n(v, d=4):
    """A key value printed to d decimals with a true minus sign, and whole numbers without decimals."""
    if float(v) == int(v) and d != 4.5:
        s = str(int(v))
    else:
        s = f'{v:.{d}f}'.rstrip('0').rstrip('.') if d == 1 else f'{v:.{d}f}'
    return s.replace('-', '−')


def go(f, *pts, color=INK, width=1.4, head=10, allowed=(), name='link', gap=4.6):
    pts = [tuple(map(float, p)) for p in pts]
    for idx, nb in ((0, 1), (-1, -2)):
        (x, y), (u, v) = pts[idx], pts[nb]
        d = math.hypot(u - x, v - y)
        pts[idx] = (x + (u - x) * gap / d, y + (v - y) * gap / d)
    if len(pts) == 2:
        f.arrow(*pts[0], *pts[1], color, width, head, allowed, name)
    else:
        f.route(pts, color, width, head, allowed, name)


def node(f, x, y, text_, fill, stroke, r=31, size=15, oid=None, color=INK):
    oid = f.obstacle(oid, x - r, y - r, 2 * r, 2 * r, 'circle')
    f.raw(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    f.words(x, y + size * .35, text_, size, color, anchor='middle')
    return oid


def edge(f, p, q, ids, r1=31, r2=31, color='#aab3bc', width=1.0, dash=None):
    d = math.hypot(q[0] - p[0], q[1] - p[1])
    a = (p[0] + (q[0] - p[0]) * (r1 + 4) / d, p[1] + (q[1] - p[1]) * (r1 + 4) / d)
    b = (q[0] - (q[0] - p[0]) * (r2 + 4) / d, q[1] - (q[1] - p[1]) * (r2 + 4) / d)
    f.line(*a, *b, color, width, ids, 'edge', dash)


def footer(f, y):
    f.words(30, y, TOY, 16, FROZEN_TEXT)


def mm0():
    f = Fig('mm0-the-whole-miniature', 1780, 1010)
    f.words(30, 44, 'The whole method on 2 pixels', 28, INK, 700)
    lx, cx, hx, vx, sx, dx = 220, 540, 860, 1090, 1370, 1640
    ly = [[150, 225, 300, 375], [520, 595, 670, 745]]
    cy = [[185, 262, 339], [555, 632, 709]]
    cname = 'RGB'
    for p in range(2):
        for c in range(3):
            for l in range(4):
                edge(f, (lx, ly[p][l]), (cx, cy[p][c]), (f'l{p}{l}', f'c{p}{c}'), color='#c7ced5', width=.8)
    hy, vy = [300, 590], [300, 590]
    links = [(0, 0, 0, '2'), (0, 2, 1, '2'), (1, 0, 0, '1'), (1, 1, 1, '1')]
    for p, c, h, w in links:
        dead = p == 1 and c == 0
        edge(f, (cx, cy[p][c]), (hx, hy[h]), (f'c{p}{c}', f'h{h}'), color=FIXED if dead else INK, width=1.6 if dead else 1.2, dash='6 5' if dead else None)
    for h in range(2):
        edge(f, (hx, hy[h]), (vx, vy[h]), (f'h{h}', f'v{h}'), color=INK, width=1.3)
    for p in range(2):
        f.words(28, (ly[p][0] + ly[p][3]) / 2 + 7, 'pixel ' + 'AB'[p], 18, INK, 700)
        for l in range(4):
            node(f, lx, ly[p][l], n(K['latent'][p][l], 1), WHITE, INK, oid=f'l{p}{l}')
            f.maths(f'z_{l}', lx - 50, ly[p][l], 16)
        for c in range(3):
            before, after = K['rgb_before_clamp'][p][c], K['rgb_after_clamp'][p][c]
            dead = before != after
            node(f, cx, cy[p][c], f'{cname[c]} {n(after, 1)}', FIXED_FILL if dead else mix(IMAGE, .45), FIXED if dead else IMAGE_EDGE, r=36, size=13.5, oid=f'c{p}{c}')
    f.words(cx, 800, 'pixel B red: 305 clamped to 255', 16, FIXED, 700, 'middle')
    f.words(lx, 110, 'latent', 19, INK, 700, 'middle')
    f.words(cx, 110, 'colours', 19, INK, 700, 'middle')
    f.words(hx, 228, 'image layer', 19, INK, 700, 'middle')
    f.words(vx, 228, 'unit length', 19, INK, 700, 'middle')
    for i in range(2):
        node(f, hx, hy[i], n(K['h'][i]), IMAGE, IMAGE_EDGE, oid=f'h{i}')
        node(f, vx, vy[i], n(K['v'][i]), mix(IMAGE, .3), IMAGE_EDGE, oid=f'v{i}')
    chips = [(190, 'S right', K['s_right'], MATCH, MATCH_EDGE, K['t_right']), (610, 'S wrong', K['s_wrong'], '#f3f4f6', FIXED, K['t_wrong'])]
    for y, name, val, fill, edge_col, tv in chips:
        f.rect(sx - 96, y, 192, 106, fill, edge_col, 1.4, 9, oid=name)
        f.words(sx, y + 42, name, 19, INK, anchor='middle')
        f.words(sx, y + 78, n(val), 21, INK, 700, 'middle')
        f.words(sx, y - 12, f'sentence vector ({tv[0]}, {tv[1]})', 15, FROZEN_TEXT, anchor='middle')
    for j in range(2):
        for y, nm in ((243, 'S right'), (663, 'S wrong')):
            f.line(vx + 35, vy[j], sx - 100, y, INK, 1.0, (f'v{j}', nm), 'vs')
    f.rect(dx - 112, 380, 224, 126, '#e9ebee', FIXED, 1.4, 10, oid='d')
    f.words(dx, 420, 'S right − S wrong', 18, INK, anchor='middle')
    f.words(dx, 468, n(K['difference']), 27, INK, 700, 'middle')
    go(f, (sx + 96, 243), (dx, 243), (dx, 380))
    go(f, (sx + 96, 663), (dx, 663), (dx, 506))
    ly2 = 868
    f.route([(dx + 124, 443), (1760, 443), (1760, ly2), (lx, ly2), (lx, 785)], PUSH, 3.2, 14, name='back')
    f.words(910, ly2 - 14, 'backward pass: the gradient g is read at the 8 latent numbers', 19, PUSH, anchor='middle')
    footer(f, 945)
    f.write(HERE, 'Two pixels of four latent channels pass through the real linear map, a hand-set image layer and two sentence vectors, giving a score difference whose gradient returns to the latent.')


def mm1():
    f = Fig('mm1-from-latent-to-colours', 1640, 790)
    f.words(30, 44, "The released map, on pixel A and pixel B", 28, INK, 700)
    heads = ((70, 'pixel'), (210, 'colour'), (640, '60 z₀ − 60 z₁ … + bias'), (1070, 'before clamp'), (1270, 'after clamp'), (1460, 'gradient?'))
    for x, t in heads:
        f.words(x if x != 640 else 640, 124, t, 18, FROZEN_TEXT, 700, 'start' if x < 300 else 'middle')
    f.raw(Ln(40, 142, 1600, 142, INK, 1.2))
    y = 188
    for p in range(2):
        z = K['latent'][p]
        f.words(60, y + 62, 'AB'[p], 28, INK, 700)
        f.words(60, y + 92, '(' + ', '.join(n(v, 1) for v in z) + ')', 13, FROZEN_TEXT)
        for c in range(3):
            w = K['weights'][c]
            terms = ' '.join((('+ ' if t * zz >= 0 else '− ') if i else ('' if t * zz >= 0 else '−')) + f'{abs(t):g}({n(zz, 1)})' for i, (t, zz) in enumerate(zip(w, z)))
            f.words(236, y + 28, 'RGB'[c], 21, ('#d1495b', '#3f9b5f', '#3f73c9')[c], 700, 'middle')
            f.words(640, y + 28, terms + f' + {K["biases"][c]:g}', 18, INK, anchor='middle')
            before, after = K['rgb_before_clamp'][p][c], K['rgb_after_clamp'][p][c]
            f.words(1070, y + 28, n(before, 1), 20, INK, anchor='middle')
            dead = before != after
            f.chip(1210, y + 4, 120, 36, n(after, 1), FIXED_FILL if dead else mix(IMAGE, .45), FIXED if dead else IMAGE_EDGE, 18, oid=f'a{p}{c}')
            f.words(1460, y + 28, 'no, clamped' if dead else 'yes', 18, FIXED if dead else '#3b7d50', 700 if dead else 400, 'middle')
            y += 56
        y += 28
        if p == 0:
            f.raw(Ln(40, y - 8, 1600, y - 8, FAINT_LINE, 1.0))
            y += 16
    f.rect(40, 616, 1560, 86, FAINT, FAINT_LINE, 1.1, 9, oid='note')
    f.words(820, 668, 'Pixel B red lands at 305. The clamp holds it at 255, so nothing passes back along that colour.', 20, INK, anchor='middle')
    footer(f, 750)
    f.write(HERE, 'The real linear map worked for two pixels. Pixel B red is 305 before the clamp and 255 after it, so it passes no gradient.')


def card(f, x, y, w, h, title, tex_lines, fill=WHITE, stroke=INK, tsize=18, msize=20, oid=None):
    oid = oid or f'card-{len(f.obstacles)}'
    f.rect(x, y, w, h, fill, stroke, 1.3, 10, oid=oid)
    f.words(x + w / 2, y + 34, title, tsize, INK, 700, 'middle')
    step = (h - 54) / (len(tex_lines) + 0)
    for i, t in enumerate(tex_lines):
        f.maths(t, x + w / 2, y + 54 + step * (i + .5), msize)
    return oid


def mm2():
    f = Fig('mm2-forward-to-the-two-scores', 1720, 800)
    f.words(30, 44, 'Forward: from colours to one score difference', 28, INK, 700)
    p = K['scaled']
    y, h = 120, 300
    card(f, 30, y, 390, h, 'scale each colour', [r'p=c/255-0.5', rf'p_{{A,R}}={p[0][0]:.4f}', rf'p_{{B,R}}={p[1][0]:.4f}'], fill=mix(IMAGE, .6), stroke=IMAGE_EDGE, oid='a')
    card(f, 470, y, 390, h, 'hand-set image layer', [rf'h_1=2p_{{A,R}}+p_{{B,R}}={K["h"][0]:.4f}', rf'h_2=2p_{{A,B}}+p_{{B,G}}={K["h"][1]:.4f}'], fill=IMAGE, stroke=IMAGE_EDGE, msize=19, oid='b')
    card(f, 910, y, 350, h, 'make it unit length', [rf'\lVert h\rVert={K["h_length"]:.4f}', rf'I=({K["v"][0]:.4f},\ {K["v"][1]:.4f})'], fill=mix(IMAGE, .3), stroke=IMAGE_EDGE, oid='c')
    card(f, 1310, y, 380, h, 'two dot products', [rf'S_{{\text{{right}}}}=I\cdot({K["t_right"][0]},{K["t_right"][1]})={K["s_right"]:.4f}', rf'S_{{\text{{wrong}}}}=I\cdot({K["t_wrong"][0]},{K["t_wrong"][1]})={K["s_wrong"]:.4f}'], fill=MATCH, stroke=MATCH_EDGE, msize=18, oid='d')
    for a, b in ((420, 470), (860, 910), (1260, 1310)):
        go(f, (a, y + h / 2), (b, y + h / 2))
    f.rect(420, 500, 880, 130, '#e9ebee', FIXED, 1.4, 10, oid='diff')
    f.maths(rf'S_{{\text{{right}}}}-S_{{\text{{wrong}}}}={K["s_right"]:.4f}-{K["s_wrong"]:.4f}={K["difference"]:.4f}'.replace('-0.0308', '-0.0308'), 860, 565, 27)
    go(f, (1500, y + h), (1500, 565), (1300, 565))
    f.words(860, 680, 'Negative: in this toy the picture sits closer to the wrong sentence, so the push has work to do.', 19, INK, anchor='middle')
    footer(f, 750)
    f.write(HERE, 'The colours are scaled, read by a hand-set layer and normalised, then dotted with a right and a wrong sentence vector to give a score difference of minus 0.0308.')


def mm3():
    f = Fig('mm3-the-gradient-back-to-the-latent', 1720, 880)
    f.words(30, 44, 'Backward: the chain rule hands each latent number its share', 28, INK, 700)
    y, h = 112, 240
    cards_ = [(30, 400, 'from the difference', [r'\frac{\partial D}{\partial h}=\frac{\Delta T-(\Delta T\cdot I)I}{\lVert h\rVert}', rf'=({K["dD_dh"][0]:.4f},\ {K["dD_dh"][1]:.4f})']),
              (480, 400, 'back through the layer', [r'\frac{\partial D}{\partial p}=M^{\top}\frac{\partial D}{\partial h}', rf'p_{{A,R}}:\ {K["dD_dp"][0][0]:.4f}']),
              (930, 340, 'the clamp mask', [r'\text{alive}=1\text{ or }0', r'\text{pixel B red}=0']),
              (1320, 370, 'back through the map', [r'g_{\text{pixel},l}=\sum_c \frac{W_{cl}}{255}\,\frac{\partial D}{\partial p_c}', r'\text{live colours only}'])]
    for i, (x, w, t, lines) in enumerate(cards_):
        card(f, x, y, w, h, t, lines, fill='#fff4e8', stroke=PUSH, msize=19, oid=f'k{i}')
    for a, b in ((430, 480), (880, 930), (1270, 1320)):
        go(f, (b, y + h / 2), (a, y + h / 2), color=PUSH, width=2.8, head=13)
    tx, ty, cw, rh = 410, 470, 170, 76
    f.words(tx + 2 * cw, ty - 52, 'g: the gradient at each of the 8 latent numbers', 21, PUSH, 700, 'middle')
    for l in range(4):
        f.maths(f'z_{l}', tx + (l + .5) * cw, ty - 18, 21)
    for p in range(2):
        f.words(tx - 26, ty + (p + .5) * rh + 8, 'pixel ' + 'AB'[p], 20, INK, 700, 'end')
        for l in range(4):
            f.chip(tx + l * cw, ty + p * rh, cw, rh, n(K['g'][p][l]), WHITE, PUSH, 23, oid=f'g{p}{l}')
    f.rect(170, 666, 1380, 112, FAINT, FAINT_LINE, 1.1, 9, oid='n')
    f.words(860, 712, 'Pixel B only reaches the layer through green now, because its red is clamped.', 20, INK, anchor='middle')
    f.words(860, 748, f"Hand chain rule and automatic differentiation agree to {K['checks']['hand_vs_autograd_max']:.0e}.", 18, FROZEN_TEXT, anchor='middle')
    footer(f, 830)
    f.write(HERE, 'Four orange cards walk the chain rule from the score difference back to the latent, ending in a table of eight gradient values.')


def mm4():
    f = Fig('mm4-the-push-joins-the-noise-prediction', 1720, 960)
    f.words(30, 44, 'The push changes the noise prediction, then one sampler step moves the latent', 26, INK, 700)
    def grid(x, y, title, a, fill, stroke, cw=88, rh=48, color=INK):
        f.words(x + 2 * cw, y - 16, title, 18, color, 700, 'middle')
        for p in range(2):
            for l in range(4):
                f.chip(x + l * cw, y + p * rh, cw, rh, n(a[p][l]), fill, stroke, 16, oid=f'{title[:3]}{x}{p}{l}')
    grid(40, 130, 'g', K['g'], WHITE, PUSH, color=PUSH)
    f.rect(420, 136, 150, 84, WHITE, PUSH, 1.5, 10, oid='times')
    f.maths(rf'\times\,{SAMP["lambda"]:g}\times{SAMP["sigma"]:g}', 495, 178, 19, PUSH)
    go(f, (392, 178), (420, 178), color=PUSH, width=3, head=13)
    grid(620, 130, 'the push', SAMP['push'], WHITE, PUSH, color=PUSH)
    go(f, (570, 178), (620, 178), color=PUSH, width=3, head=13)
    grid(40, 330, 'ε base', SAMP['eps_base'], '#e9ebee', FIXED)
    f.circle(1120, 178, 25, WHITE, PUSH, 2.0, oid='join')
    f.raw(Ln(1106, 178, 1134, 178, PUSH, 2.6))
    go(f, (972, 178), (1095, 178), color=PUSH, width=3, head=13)
    go(f, (392, 378), (1120, 378), (1120, 203))
    grid(1230, 130, 'ε̂ = ε base − push', SAMP['eps_hat'], WHITE, INK, cw=108)
    go(f, (1145, 178), (1230, 178))
    f.words(1446, 300, 'the prediction the sampler uses', 17, FROZEN_TEXT, anchor='middle')
    f.rect(40, 500, 780, 300, FAINT, FAINT_LINE, 1.1, 10, oid='ddim')
    f.words(430, 540, 'One DDIM step', 21, INK, 700, 'middle')
    f.maths(rf'x_{{\text{{next}}}}={SAMP["signal_next"]}\,\frac{{x_t-{SAMP["sigma"]}\,\hat\epsilon_t}}{{{SAMP["signal"]}}}+{SAMP["sigma_next"]}\,\hat\epsilon_t', 430, 610, 25)
    f.words(430, 690, f'Subtracting the push from ε adds {abs(SAMP["coefficient_on_eps"]):.4f} × push to the next latent,', 17, INK, anchor='middle')
    f.words(430, 720, 'which is a step along g.', 17, INK, anchor='middle')
    # score difference, plain against pushed
    bx0, bx1, by = 980, 1640, 700
    lo, hi = -0.12, 0.0
    px = lambda v: bx0 + (v - lo) / (hi - lo) * (bx1 - bx0)
    f.words((bx0 + bx1) / 2, 520, 'Score difference after one step (toy)', 21, INK, 700, 'middle')
    for i, (name, val, col) in enumerate((('no push', SAMP['difference_plain'], FIXED), ('with the push', SAMP['difference_pushed'], PUSH))):
        y = 556 + i * 92
        f.words(bx0 - 16, y + 38, name, 18, INK, anchor='end')
        f.raw(R(px(val), y, px(0) - px(val), 58, mix(col, .35), col, 1.2, 0))
        f.obstacle(f'b{i}', px(val), y, px(0) - px(val), 58, 'rect')
        f.words(px(val) - 10, y + 38, n(val), 20, INK, 700, 'end') if px(val) - bx0 > 90 else f.words(px(0) + 8, y + 38, n(val), 20, INK, 700)
    f.raw(Ln(px(0), 546, px(0), by + 26, INK, 1.4) + Ln(bx0, by + 26, px(0), by + 26, INK, 1.2))
    for v in (-0.12, -0.08, -0.04, 0):
        f.raw(Ln(px(v), by + 26, px(v), by + 33, INK, 1.2))
        f.words(px(v), by + 56, n(v, 2), 16, INK, anchor='middle')
    f.words((bx0 + bx1) / 2, by + 90, 'S right minus S wrong, cosine units', 17, INK, anchor='middle')
    footer(f, 880)
    f.write(HERE, 'The gradient times strength and noise level is subtracted from the base noise prediction. One DDIM step then gives a score difference of minus 0.0791 with the push and minus 0.0984 without it, on the toy.')


def mm5():
    f = Fig('mm5-four-ways-the-gradient-dies', 1720, 980)
    f.words(30, 44, 'Four ways the push can get no signal', 28, INK, 700)
    panels = [(30, 96, '.byte() on the colours', '#fff4e8', PUSH), (880, 96, 'both sentences get the same vector', FAINT, FAINT_LINE),
              (30, 500, 'all three colours clamped', FAINT, FAINT_LINE), (880, 500, 'a hard clamp against a soft clamp', FAINT, FAINT_LINE)]
    for x, y, t, fill, st in panels:
        f.card(x, y, 810, 376, fill, st, 1.2, 14, oid=f'p{x}{y}')
        f.words(x + 405, y + 42, t, 22, INK, 700, 'middle')
    b = BR['byte']
    f.chip(120, 190, 200, 76, f"{b['example_in']}", WHITE, INK, 25, oid='b0')
    f.chip(540, 190, 200, 76, f"{b['example_out']}", FIXED_FILL, FIXED, 25, oid='b1')
    go(f, (320, 228), (540, 228))
    f.words(430, 212, 'truncates', 17, INK, anchor='middle')
    f.words(435, 326, 'The value drops its decimals and leaves the graph.', 19, INK, anchor='middle')
    f.words(435, 362, f"has a gradient: {'yes' if b['has_gradient'] else 'no'}", 22, PUSH, 700, 'middle')
    f.words(435, 410, 'The push path uses a copy of the map without it.', 18, FROZEN_TEXT, anchor='middle')
    s = BR['same_sentence']
    f.maths(r'T_{\text{right}}=T_{\text{wrong}}', 1285, 210, 28)
    f.maths(r'D=I\cdot(T_{\text{right}}-T_{\text{wrong}})=0', 1285, 268, 25)
    f.words(1285, 334, f"score difference {n(s['difference'])}, largest gradient {n(s['gradient_max'])}", 20, INK, anchor='middle')
    f.words(1285, 386, 'A text encoder that cannot tell two word orders apart', 17, FROZEN_TEXT, anchor='middle')
    f.words(1285, 412, 'gives this push nothing to follow.', 17, FROZEN_TEXT, anchor='middle')
    a = BR['all_three_clamped']
    f.words(435, 586, 'pixel B set to (' + ', '.join(n(v, 1) for v in a['latent_pixel']) + ')', 19, INK, anchor='middle')
    f.words(435, 628, 'colours before the clamp: ' + ', '.join(n(v, 1) for v in a['rgb_before_clamp']), 19, INK, anchor='middle')
    for l in range(4):
        f.chip(155 + l * 140, 664, 140, 62, n(a['g_row'][l]), FIXED_FILL, FIXED, 21, oid=f'z{l}')
    f.words(435, 780, "pixel B's gradient row: 0 on all four channels", 20, FIXED, 700, 'middle')
    f.words(435, 828, 'Every channel of this pixel is cut off.', 18, FROZEN_TEXT, anchor='middle')
    h = BR['hard_vs_soft']
    f.words(1285, 582, f"pixel B red at {n(h['red_b_before_clamp'], 1)}, gradient on its z₀:", 19, INK, anchor='middle')
    f.chip(1000, 616, 260, 72, 'hard: ' + n(h['g_hard'][1][0]), WHITE, INK, 22, oid='hh')
    f.chip(1310, 616, 260, 72, 'soft: ' + n(h['g_soft'][1][0]), WHITE, INK, 22, oid='ss')
    f.words(1285, 742, f"The soft clamp keeps slope {h['soft_slope_there']} at that value (sharpness {h['beta']}).", 18, INK, anchor='middle')
    f.words(1285, 786, 'It lets some signal through and changes every other value too.', 18, INK, anchor='middle')
    f.words(1285, 830, 'Step W3 records this choice before any picture is pushed.', 17, FROZEN_TEXT, anchor='middle')
    footer(f, 930)
    f.write(HERE, 'Four panels on the toy: byte conversion removes the gradient, identical sentence vectors give zero difference, a pixel with all colours clamped gets zero gradient, and a soft clamp passes a small slope where a hard clamp passes none.')


def mm6():
    f = Fig('mm6-one-training-step', 1720, 860)
    f.words(30, 44, 'One training step moves the image layer and leaves the text vectors alone', 26, INK, 700)
    def matrix(x, y, title, logits, loss, img):
        cw, rh = 150, 92
        f.words(x + cw + 80, y - 86, title, 22, INK, 700, 'middle')
        for j in range(2):
            f.words(x + 80 + (j + .5) * cw, y - 50, f'T{j + 1}', 18, INK, 700, 'middle')
            f.strip(x + 80 + j * cw + 25, y - 40, cw - 50, 14, TEXT, oid=f'ts{x}{j}')
        for i in range(2):
            f.strip(x + 18, y + i * rh + 14, 14, rh - 28, IMAGE, vertical=True, oid=f'is{x}{i}')
            f.words(x + 54, y + (i + .5) * rh + 7, f'I{i + 1}', 18, INK, 700, 'middle')
            for j in range(2):
                f.chip(x + 80 + j * cw, y + i * rh, cw, rh, n(logits[i][j]), MATCH if i == j else '#f8f9fa', MATCH_EDGE if i == j else FAINT_LINE, 21, oid=f'm{x}{i}{j}')
        f.words(x + 80 + cw, y + 2 * rh + 44, f'loss {n(loss)}', 25, INK, 700, 'middle')
    matrix(90, 190, 'before', TR['logits_before'], TR['loss_before'], TR['image_vectors_before'])
    matrix(1090, 190, 'after one step', TR['logits_after'], TR['loss_after'], TR['image_vectors_after'])
    f.rect(600, 160, 450, 268, WHITE, IMAGE_EDGE, 1.5, 12, oid='up')
    f.words(825, 200, 'update the image layer', 20, INK, 700, 'middle')
    f.maths(rf'M\leftarrow M-{TR["learning_rate"]}\,\frac{{\partial \mathcal{{L}}}}{{\partial M}}', 825, 268, 25)
    f.words(825, 340, f"first weight: 2 becomes {TR['layer_after'][0][0]:.4f}", 18, INK, anchor='middle')
    f.words(825, 376, 'text vectors: unchanged', 18, FROZEN_TEXT, anchor='middle')
    go(f, (548, 284), (600, 284))
    go(f, (1050, 284), (1100, 284))
    f.rect(90, 540, 1540, 170, FAINT, FAINT_LINE, 1.1, 10, oid='n')
    f.words(860, 584, 'Each cell is the score times a toy scale of 10. Real NoisyCLIP multiplies by 98.86.', 20, INK, anchor='middle')
    f.words(860, 626, 'I1 and I2 are the same two pixels in swapped order, each paired with its own sentence T1 or T2.', 19, INK, anchor='middle')
    f.words(860, 668, 'The loss is the mean of the cross-entropy over rows and over columns.', 19, INK, anchor='middle')
    footer(f, 790)
    f.write(HERE, 'A two by two similarity matrix before and after one gradient step on the image layer. The toy loss falls from 0.9264 to 0.7929 and the text vectors do not move.')


ALL = {'mm0': mm0, 'mm1': mm1, 'mm2': mm2, 'mm3': mm3, 'mm4': mm4, 'mm5': mm5, 'mm6': mm6}
if __name__ == '__main__':
    bad = 0
    for k in (sys.argv[1:] or ALL):
        try:
            ALL[k]()
        except RuntimeError as e:
            bad += 1
            lines = str(e).split('\n')
            print(lines[0])
            for line in lines[1:7]:
                print('  ', line[:185])
    print('failed' if bad else 'all written', bad)
