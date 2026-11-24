"""Level 0.5 of the NoisyCLIP ladder: the network view, in the clip-contrastive style.

Every size and share is read from ../numbers/numbers.json. Noise thumbnails are synthetic Gaussian noise, never a generated
picture. Run from this folder: PYTHONDONTWRITEBYTECODE=1 python3 make_network_view.py [figure stems]
"""
import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_kit import *  # noqa: E402,F403
from figure_kit import _width  # noqa: E402

N = json.loads((HERE.parent / 'numbers' / 'numbers.json').read_text())
C, P_, M, S, K = N['clip'], N['parameters'], N['linear_map'], N['schedule'], N['checkpoint']


def go(f, *pts, color=INK, width=1.4, head=10, allowed=(), name='link', gap=4.6):
    """An arrow or elbow that stops just short of the things it joins, so it never touches them."""
    pts = [tuple(map(float, p)) for p in pts]
    for idx, nb in ((0, 1), (-1, -2)):
        (x, y), (u, v) = pts[idx], pts[nb]
        d = math.hypot(u - x, v - y)
        pts[idx] = (x + (u - x) * gap / d, y + (v - y) * gap / d)
    if len(pts) == 2:
        f.arrow(*pts[0], *pts[1], color, width, head, allowed, name)
    else:
        f.route(pts, color, width, head, allowed, name)


def wire(f, *pts, color=INK, width=1.4, gap=4.6, allowed=()):
    """A line with no arrowhead."""
    pts = [tuple(map(float, p)) for p in pts]
    for idx, nb in ((0, 1), (-1, -2)):
        (x, y), (u, v) = pts[idx], pts[nb]
        d = math.hypot(u - x, v - y)
        pts[idx] = (x + (u - x) * gap / d, y + (v - y) * gap / d)
    for p, q in zip(pts, pts[1:]):
        f.line(*p, *q, color, width, allowed, 'wire')


def boxed(f, x, y, w, h, lines, fill=FIXED_FILL, stroke=FIXED, size=18, sub=15, oid=None):
    """A box holding one main line and optional smaller grey lines."""
    oid = oid or f'box-{len(f.obstacles)}'
    f.rect(x, y, w, h, fill, stroke, 1.3, 9, oid=oid)
    total = size + sum(sub + 6 for _ in lines[1:])
    yy = y + (h - total) / 2 + size * 0.85
    f.words(x + w / 2, yy, lines[0], size, INK, anchor='middle')
    for extra in lines[1:]:
        yy += sub + 6
        f.words(x + w / 2, yy, extra, sub, FROZEN_TEXT, anchor='middle')
    return oid


def noise_uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).resize((180, 180), Image.NEAREST).save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def tile(f, x, y, s, arr, oid=None):
    oid = f.obstacle(oid, x, y, s, s, 'image')
    f.raw(f'<image href="{noise_uri(arr)}" x="{x}" y="{y}" width="{s}" height="{s}" preserveAspectRatio="none"/>')
    f.raw(R(x, y, s, s, 'none', INK, 1.0, 0))
    return oid


def mini_matrix(f, x, y, cw, rh, n=3):
    for i in range(n):
        for j in range(n):
            f.raw(R(x + j * cw, y + i * rh, cw, rh, MATCH if i == j else '#f8f9fa', '#c9ced4', 0.9, 0))
    f.obstacle(None, x, y, n * cw, n * rh, 'rect')


def towers(f, y, text_label, tag='tuned'):
    """One lane's two rows: caption over text encoder, latent through the map and image encoder."""
    f.cards(34, y + 46, 188, 62, text_label, 16)
    f.encoder(312, y + 30, 210, 100, 'Text Encoder', TEXT, frozen=True, size=19, oid=f'te{y}')
    f.pill(417, y + 12, 'frozen')
    f.strip(585, y + 66, 130, 26, TEXT)
    go(f, (234, y + 80), (312, y + 80))
    go(f, (522, y + 80), (585, y + 80))
    f.stack(46, y + 188, 74, 74)
    f.words(92, y + 166, 'noisy latent', 15, INK, anchor='middle')
    boxed(f, 172, y + 190, 108, 74, ['linear map', 'no VAE'], size=15, sub=13)
    f.encoder(312, y + 176, 210, 100, 'Image Encoder', IMAGE, size=19, oid=f'ie{y}')
    f.pill(417, y + 158, tag, color=INK if tag == 'tuned' else FIXED)
    f.strip(585, y + 212, 130, 26, IMAGE)
    go(f, (136, y + 226), (172, y + 226))
    go(f, (280, y + 226), (312, y + 226))
    go(f, (522, y + 226), (585, y + 226))


def nc1():
    f = Fig('nc1-one-model-three-uses', 1760, 1300)
    f.words(30, 40, 'The same two encoders do three jobs', 28, INK, 700)
    y1, y2, y3 = 62, 410, 758
    # lane one: training
    towers(f, y1, 'caption')
    mini_matrix(f, 860, y1 + 54, 62, 56)
    f.words(953, y1 + 240, 'similarity matrix', 16, INK, anchor='middle')
    go(f, (715, y1 + 80), (820, y1 + 80), (820, y1 + 22), (953, y1 + 22), (953, y1 + 54))
    go(f, (715, y1 + 226), (800, y1 + 226), (800, y1 + 138), (860, y1 + 138))
    boxed(f, 1130, y1 + 92, 300, 96, ['contrastive loss', 'updates the image encoder only'], fill=WHITE, stroke=INK, size=19, sub=15)
    go(f, (1046, y1 + 138), (1130, y1 + 138))
    f.words(1720, y1 + 146, 'Training', 24, INK, 700, 'end')
    f.dotted(20, y2 - 18, 1740)
    # lane two: reading
    towers(f, y2, 'two sentences', tag='weights fixed')
    f.chip(880, y2 + 54, 150, 52, 'S right', MATCH, MATCH_EDGE, oid='r2')
    f.chip(880, y2 + 186, 150, 52, 'S wrong', '#f3f4f6', FIXED, oid='w2')
    go(f, (715, y2 + 80), (880, y2 + 80))
    go(f, (715, y2 + 226), (800, y2 + 226), (800, y2 + 212), (880, y2 + 212))
    boxed(f, 1130, y2 + 92, 300, 96, ['S right minus S wrong', 'a number for this latent'], fill='#e9ebee', size=19, sub=15)
    go(f, (1030, y2 + 80), (1080, y2 + 80), (1080, y2 + 124), (1130, y2 + 124))
    go(f, (1030, y2 + 212), (1080, y2 + 212), (1080, y2 + 156), (1130, y2 + 156))
    f.words(1720, y2 + 146, 'Reading', 24, INK, 700, 'end')
    f.dotted(20, y3 - 18, 1740)
    # lane three: pushing
    towers(f, y3, 'two sentences', tag='weights fixed')
    f.chip(880, y3 + 54, 150, 52, 'S right', MATCH, MATCH_EDGE, oid='r3')
    f.chip(880, y3 + 186, 150, 52, 'S wrong', '#f3f4f6', FIXED, oid='w3')
    go(f, (715, y3 + 80), (880, y3 + 80))
    go(f, (715, y3 + 226), (800, y3 + 226), (800, y3 + 212), (880, y3 + 212))
    boxed(f, 1130, y3 + 92, 300, 96, ['S right minus S wrong'], fill='#e9ebee', size=19)
    go(f, (1030, y3 + 80), (1080, y3 + 80), (1080, y3 + 124), (1130, y3 + 124))
    go(f, (1030, y3 + 212), (1080, y3 + 212), (1080, y3 + 156), (1130, y3 + 156))
    f.words(1720, y3 + 146, 'Pushing', 24, INK, 700, 'end')
    # the backward pass ends at the latent; the gradient then leaves for the noise prediction
    by = y3 + 330
    wire(f, (1430, y3 + 140), (1490, y3 + 140), (1490, by), (64, by), (64, y3 + 262), color=PUSH, width=3.2, allowed=('dback',))
    f.dot(64, y3 + 262, oid='dback')
    f.words(780, by - 10, 'backward pass: the gradient is taken at the latent', 18, PUSH, anchor='middle')
    f.dot(46, y3 + 225, oid='dout')
    gy = by + 56
    wire(f, (46, y3 + 225), (20, y3 + 225), (20, gy), (560, gy), color=PUSH, width=3.2, allowed=('dout',))
    f.maths(r'g=\nabla_{x_t}[S_{\text{right}}-S_{\text{wrong}}]', 300, gy - 24, 19, PUSH)
    f.rect(560, gy - 25, 120, 50, WHITE, PUSH, 1.3, 8, oid='scale')
    f.maths(r'\times\,\lambda\,\sigma_t', 620, gy, 20, PUSH)
    f.circle(790, gy, 19, WHITE, PUSH, 1.8, oid='join')
    f.raw(Ln(780, gy, 800, gy, PUSH, 2.2))
    go(f, (680, gy), (771, gy), color=PUSH, width=3.2, head=14)
    boxed(f, 330 + 1, gy + 60, 300, 66, ['ε base', 'product or CO3'], fill='#e9ebee', size=18, sub=14, oid='base')
    go(f, (631, gy + 93), (790, gy + 93), (790, gy + 19))
    f.rect(880, gy - 44, 560, 88, WHITE, INK, 1.3, 9, oid='hat')
    f.maths(r'\hat\epsilon_t=\epsilon_{\text{base}}-\lambda\,\sigma_t\,g', 1160, gy, 26)
    go(f, (809, gy), (880, gy))
    boxed(f, 1500, gy - 44, 220, 88, ['next sampler step'], fill='#e9ebee', size=18, oid='next')
    go(f, (1440, gy), (1500, gy))
    f.write(HERE, 'One CLIP model in three horizontal lanes: training with a contrastive loss, reading a latent against two sentences, and pushing the noise prediction with the gradient taken at the latent.')


def nc2():
    u = 58
    f = Fig('nc2-the-whole-read-and-push-path', 1760, 1370)
    X = lambda x: round(x * u + 2, 1)
    Y = lambda y: round((19.4 - y) * u, 1)
    f.words(14, 40, 'A noisy latent read against two sentences, with every size', 28, INK, 700)
    f.frame(X(1.9), Y(15.05), 26.2 * u, 6.35 * u, 'CLIP ViT-L/14')
    f.rect(X(3), Y(18.2), 24 * u, u, '#e9ebee', FIXED, 1.3, 10, oid='bar')
    f.words(X(15), Y(17.7) + 9, 'S right minus S wrong (a number between −2 and 2)', 23, INK, anchor='middle')
    for x, name, fill, edge in ((10.5, 'S right', MATCH, MATCH_EDGE), (19.5, 'S wrong', '#f3f4f6', FIXED)):
        f.chip(X(x - 1.5), Y(16.25), 3 * u, 0.9 * u, name, fill, edge, 21, oid=f'c{x}')
        go(f, (X(x), Y(16.25)), (X(x), Y(17.2) + 0))
        f.product(X(x), Y(13.3), 22, oid=f'p{x}')
        go(f, (X(x), Y(13.3) - 22), (X(x), Y(15.35)))
    for x0, x1, tex_, colour, key in ((3.5, 8.0, r'T_{\text{right}}', TEXT, 'tr'), (12.7, 17.3, 'I', IMAGE, 'im'), (22.0, 26.5, r'T_{\text{wrong}}', TEXT, 'tw')):
        f.strip(X(x0), Y(13.7), (x1 - x0) * u, 0.8 * u, colour, oid=key)
        f.maths(tex_ + r'\ \ ' + str(C['shared_width']) + r'\text{ numbers}', X((x0 + x1) / 2), Y(14.14), 19)
    go(f, (X(8.0), Y(13.3)), (X(10.5) - 22, Y(13.3)))
    go(f, (X(12.7), Y(13.3)), (X(10.5) + 22, Y(13.3)))
    go(f, (X(17.3), Y(13.3)), (X(19.5) - 22, Y(13.3)))
    go(f, (X(22.0), Y(13.3)), (X(19.5) + 22, Y(13.3)))
    enc = ((2.8, 5.9, 'Text Encoder', TEXT, True, 'frozen, same encoder'), (11.6, 6.8, 'Image Encoder', IMAGE, False, f"tuned, {C['vision_layers']} blocks"),
           (21.3, 5.9, 'Text Encoder', TEXT, True, 'frozen, same encoder'))
    for i, (x, w, name, colour, frozen, tag) in enumerate(enc):
        f.encoder(X(x), Y(11.3), w * u, 1.9 * u, name, colour, frozen, tag, shape='round', size=24, oid=f'e{i}')
        go(f, (X(x + w / 2), Y(11.3)), (X(x + w / 2), Y(12.9)))
    for cx, sentence, group in ((5.75, 'a cat to the left of a dog', 'right sentence'), (24.25, 'a dog to the left of a cat', 'wrong sentence')):
        f.rect(X(cx - 3.15), Y(6.7), 6.3 * u, 2.3 * u, 'none', INK, 1.1, 10, dash='5 4', oid=f'dash{cx}')
        f.rect(X(cx - 2.85), Y(6.35), 5.7 * u, 1.6 * u, WHITE, INK, 1.0, 0, oid=f'card{cx}')
        f.words(X(cx), Y(5.55) + 7, sentence, 20, INK, anchor='middle')
        go(f, (X(cx), Y(6.7)), (X(cx), Y(9.4)))
        f.brace(X(cx - 3.15), X(cx + 3.15), Y(4.2), group, 19)
    boxed(f, X(10.55), Y(8.42), 8.9 * u, 1.5 * u, [f"linear map to RGB, no VAE decode: 3 × 128 × 128", f"resize to 3 × {C['image_size']} × {C['image_size']} and normalise (not in the released code)"], size=18, sub=15, oid='map')
    f.stack(X(13.17), Y(5.9), 3.3 * u, 2.5 * u, oid='xt')
    f.maths(r'x_t', X(17.45), Y(5.75), 25)
    f.words(X(17.1), Y(5.1), '4 × 128 × 128', 15, FROZEN_TEXT)
    go(f, (X(15), Y(5.9) - 14), (X(15), Y(6.92)))
    go(f, (X(15), Y(8.42)), (X(15), Y(9.4)))
    f.brace(X(13.17), X(16.83), Y(3.3), 'the noisy input', 19)
    # backward pass
    wire(f, (X(27), Y(17.7)), (X(29.2), Y(17.7)), (X(29.2), Y(1.25)), (X(18.2), Y(1.25)), (X(18.2), Y(4.65)), (X(13.17) + 3.3 * u + 14, Y(4.65)), color=PUSH, width=3.4, allowed=('dback',))
    f.dot(X(13.17) + 3.3 * u + 14, Y(4.65), oid='dback')
    f.maths(r'\text{backward pass: the gradient is taken at }x_t', X(23.7), Y(1.75), 20, PUSH)
    f.dot(X(13.17), Y(4.65), oid='dout')
    wire(f, (X(13.17), Y(4.65)), (X(11.6), Y(4.65)), (X(11.6), Y(-0.12)), color=PUSH, width=3.4, allowed=('dout',))
    f.maths(r'g=\nabla_{x_t}[S_{\text{right}}-S_{\text{wrong}}]', X(14.7), Y(0.55), 21, PUSH)
    f.rect(X(10.45), Y(-0.12), 2.3 * u, 0.86 * u, WHITE, PUSH, 1.3, 8, oid='scale')
    f.maths(r'\times\,\lambda\,\sigma_t', X(11.6), Y(-0.55), 21, PUSH)
    go(f, (X(11.6), Y(-0.98)), (X(11.6), Y(-2.35) - 24), color=PUSH, width=3.4, head=15)
    f.circle(X(11.6), Y(-2.35), 24, WHITE, PUSH, 2.0, oid='join')
    f.raw(Ln(X(11.6) - 13, Y(-2.35), X(11.6) + 13, Y(-2.35), PUSH, 2.6))
    f.rect(X(2.6), Y(-1.45), 6.3 * u, 1.8 * u, '#e9ebee', FIXED, 1.3, 9, oid='base')
    f.maths(r'\epsilon_{\text{base}}', X(5.75), Y(-1.98), 24)
    f.words(X(5.75), Y(-2.72) + 5, 'product or CO3 noise prediction', 16, FROZEN_TEXT, anchor='middle')
    go(f, (X(8.9), Y(-2.35)), (X(11.6) - 24, Y(-2.35)))
    f.rect(X(13.6), Y(-1.45), 11.6 * u, 1.8 * u, WHITE, INK, 1.3, 9, oid='hat')
    f.maths(r'\hat\epsilon_t=\epsilon_{\text{base}}-\lambda\,\sigma_t\,\nabla_{x_t}[S_{\text{right}}-S_{\text{wrong}}]', X(19.4), Y(-2.35), 24)
    go(f, (X(11.6) + 24, Y(-2.35)), (X(13.6), Y(-2.35)))
    boxed(f, X(26.1), Y(-1.45), 3.75 * u, 1.8 * u, ['next', 'sampler step'], fill='#e9ebee', size=17, sub=17, oid='next')
    go(f, (X(25.2), Y(-2.35)), (X(26.1), Y(-2.35)))
    f.write(HERE, 'The read and push path with real sizes: the latent through a fixed map into a tuned image encoder, scored against a right and a wrong sentence, with the gradient taken at the latent and subtracted from the base noise prediction.')


def nc3():
    f = Fig('nc3-four-latent-channels-to-rgb', 1720, 830)
    f.words(30, 44, 'A fixed linear map turns 4 latent channels into 3 colours', 28, INK, 700)
    rng = np.random.default_rng(3)
    z = rng.standard_normal((4, 24, 24))
    W, b = np.array(M['weights'], float), np.array(M['biases'], float)
    rgb = np.moveaxis(np.tensordot(W, z, axes=(1, 0)), 0, -1) + b
    rgb8 = np.clip(rgb, 0, 255).astype('uint8')
    for l in range(4):
        y = 110 + l * 142
        g = (np.clip(0.5 + z[l] / 5, 0, 1) * 255).astype('uint8')
        tile(f, 90, y, 112, np.stack([g] * 3, -1), oid=f'z{l}')
        f.maths(f'z_{l}', 52, y + 56, 24)
    f.brace(70, 222, 684, None)
    f.words(146, 742, 'latent: 4 × 128 × 128', 18, INK, anchor='middle')
    f.words(146, 768, '(unit noise here)', 16, FROZEN_TEXT, anchor='middle')
    cx, cy, cw, rh = 400, 214, 92, 76
    f.words(cx + 2 * cw, cy - 66, 'one weight per latent channel', 18, FROZEN_TEXT, anchor='middle')
    for j in range(4):
        f.maths(f'z_{j}', cx + (j + .5) * cw, cy - 24, 22)
    f.words(cx + 4 * cw + 74, cy - 16, 'bias', 19, INK, anchor='middle')
    cols = ('#d1495b', '#3f9b5f', '#3f73c9')
    for i in range(3):
        f.words(cx - 32, cy + (i + .5) * rh + 8, 'RGB'[i], 24, cols[i], 700, 'middle')
        for j in range(4):
            v = M['weights'][i][j]
            f.chip(cx + j * cw, cy + i * rh, cw, rh, f'{v:+d}'.replace('-', '−'), WHITE, FAINT_LINE, 23, oid=f'w{i}{j}')
        f.words(cx + 4 * cw + 26, cy + (i + .5) * rh + 8, '+', 22, INK, anchor='middle')
        f.chip(cx + 4 * cw + 44, cy + i * rh + 8, 72, rh - 16, str(M['biases'][i]), FIXED_FILL, FIXED, 21, oid=f'b{i}')
    go(f, (232, 342), (352, 342))
    go(f, (cx + 4 * cw + 128, 328), (1016, 328))
    for i in range(3):
        y = 122 + i * 142
        a = np.zeros((24, 24, 3), 'uint8')
        a[..., i] = rgb8[..., i]
        tile(f, 1020, y, 112, a, oid=f'c{i}')
        f.words(1148, y + 64, 'RGB'[i], 21, cols[i], 700)
    f.line(1190, 122, 1190, 518, INK, 1.1, name='bracket')
    go(f, (1190, 320), (1300, 320))
    tile(f, 1304, 186, 268, rgb8, oid='rgb')
    f.words(1438, 490, 'colour input to the image encoder', 17, INK, anchor='middle')
    f.words(1438, 518, '3 × 128 × 128, values 0 to 255', 16, FROZEN_TEXT, anchor='middle')
    f.rect(400, 500, 560, 214, FAINT, FAINT_LINE, 1.1, 10, oid='eq')
    f.words(680, 540, 'one pixel, one colour', 18, FROZEN_TEXT, anchor='middle')
    f.maths(r'R = 60\,z_0 - 60\,z_1 + 25\,z_2 - 70\,z_3 + 150', 680, 592, 25)
    f.words(680, 650, 'No VAE decoder: each colour is a weighted sum of', 17, INK, anchor='middle')
    f.words(680, 677, 'the 4 channels at the same latent position', 17, INK, anchor='middle')
    f.words(1438, 620, 'Thumbnails: synthetic Gaussian noise', 15, FROZEN_TEXT, anchor='middle')
    f.words(1438, 646, 'through the real map, not a picture', 15, FROZEN_TEXT, anchor='middle')
    f.write(HERE, 'Four latent channels multiply a fixed 3 by 4 weight matrix and add a bias to give red, green and blue, with no VAE decoder.')


def nc4():
    f = Fig('nc4-clamping-kills-the-gradient', 1700, 800)
    f.words(30, 44, 'Where the map leaves 0 to 255, the clamp passes no gradient', 28, INK, 700)
    x0, x1, y0, y1 = 150, 820, 660, 180
    lo, hi = -120.0, 375.0
    px = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    py = lambda v: y0 - v / 255 * (y0 - y1)
    f.raw(R(x0, y1 - 20, px(0) - x0, y0 - y1 + 20, FIXED_FILL, 'none', 0, 0))
    f.raw(R(px(255), y1 - 20, x1 - px(255), y0 - y1 + 20, FIXED_FILL, 'none', 0, 0))
    f.raw(Ln(x0, y0, x1, y0, INK, 1.4) + Ln(x0, y0, x0, y1 - 20, INK, 1.4))
    for v in (0, 100, 200, 255):
        f.raw(Ln(px(v), y0, px(v), y0 + 7, INK, 1.2))
        f.words(px(v), y0 + 30, str(v), 16, INK, anchor='middle')
    for v in (0, 255):
        f.raw(Ln(x0 - 7, py(v), x0, py(v), INK, 1.2))
        f.words(x0 - 13, py(v) + 6, str(v), 16, INK, anchor='end')
    f.words((x0 + x1) / 2, y0 + 68, 'value from the linear map, before the clamp (RGB units)', 19, INK, anchor='middle')
    f.words(30, y1 - 42, 'value after the clamp (RGB units)', 19, INK)
    f.raw(Ln(x0, py(0), px(0), py(0), FIXED, 4) + Ln(px(0), py(0), px(255), py(255), IMAGE_EDGE, 4) + Ln(px(255), py(255), x1, py(255), FIXED, 4))
    f.words((x0 + px(0)) / 2, 330, 'slope 0', 19, FIXED, 700, 'middle')
    f.words((x0 + px(0)) / 2, 358, 'no gradient', 17, FIXED, anchor='middle')
    f.words((px(255) + x1) / 2, 408, 'slope 0', 19, FIXED, 700, 'middle')
    f.words((px(255) + x1) / 2, 436, 'no gradient', 17, FIXED, anchor='middle')
    f.words(px(112), py(52), 'slope 1', 19, IMAGE_EDGE, 700)
    f.words(px(112), py(52) + 25, 'gradient passes', 17, IMAGE_EDGE)
    bx, bw = 1070, 500
    f.words(bx - 90, 160, 'Share of pixels clamped, unit-variance noise', 21, INK, 700)
    f.words(bx - 90, 188, 'computed from the weights, independent channels', 16, FROZEN_TEXT)
    scale = bw / 30.0
    for i, c in enumerate('RGB'):
        d = M['per_colour'][c]
        y = 230 + i * 92
        f.words(bx - 22, y + 34, {'R': 'red', 'G': 'green', 'B': 'blue'}[c], 20, INK, anchor='end')
        lw, hw = d['below_0_percent'] * scale, d['above_255_percent'] * scale
        f.raw(R(bx, y, lw, 50, mix(FIXED, .55), 'none', 0, 0) + R(bx + lw, y, hw, 50, FIXED, 'none', 0, 0))
        f.obstacle(f'bar{c}', bx, y, lw + hw, 50, 'rect')
        f.words(bx + lw + hw + 12, y + 33, f"{d['clamped_percent']}%", 22, INK, 700)
    f.raw(Ln(bx, 214, bx, 500, INK, 1.2))
    f.raw(R(bx, 520, 18, 18, mix(FIXED, .55), 'none', 0, 0))
    f.words(bx + 26, 536, 'below 0', 16, INK)
    f.raw(R(bx + 130, 520, 18, 18, FIXED, 'none', 0, 0))
    f.words(bx + 156, 536, 'above 255', 16, INK)
    f.rect(bx - 90, 575, 600, 112, FAINT, FAINT_LINE, 1.1, 9, oid='sum')
    f.words(bx + 210, 618, f"{M['pixels_with_any_colour_clamped_percent']}% of pixels have at least one colour clamped", 19, INK, anchor='middle')
    f.words(bx + 210, 656, f"{M['pixels_with_all_three_clamped_percent']}% have all three clamped and send back nothing", 19, INK, anchor='middle')
    f.write(HERE, 'The clamp has slope one between 0 and 255 and slope zero outside it. On unit-variance noise it clamps 26.8 percent of red, 11.4 percent of green and 7.0 percent of blue values.')


def nc5():
    f = Fig('nc5-what-is-tuned-and-what-is-frozen', 1760, 985)
    f.words(30, 44, 'The image encoder is tuned, and the text encoder and both projections stay frozen', 28, INK, 700)
    f.frame(30, 86, 1120, 860, 'CLIP ViT-L/14: the published NoisyCLIP weights')
    ty, iy = 260, 650
    f.words(70, ty - 88, 'Text lane', 21, FROZEN_TEXT, 700)
    f.cards(66, ty - 42, 170, 80, 'sentence', 18)
    f.encoder(320, ty - 92, 310, 190, 'Text Encoder', TEXT, frozen=True, size=25, oid='te')
    f.pill(475, ty - 113, 'frozen')
    f.words(475, ty + 124, f"{P_['text_model']:,} numbers", 18, FROZEN_TEXT, anchor='middle')
    f.rect(720, ty - 44, 190, 88, mix(TEXT, 0.55), FIXED, 1.3, 12, oid='tp')
    f.words(815, ty - 4, 'text projection', 18, FROZEN_TEXT, anchor='middle')
    f.words(815, ty + 22, f"768 to {C['shared_width']}", 15, FROZEN_TEXT, anchor='middle')
    f.pill(815, ty - 66, 'frozen')
    f.words(815, ty + 72, f"{P_['text_projection']:,} numbers", 16, FROZEN_TEXT, anchor='middle')
    f.strip(962, ty - 14, 150, 30, TEXT, oid='ts')
    f.words(1037, ty - 28, f"T, {C['shared_width']} numbers", 16, INK, anchor='middle')
    go(f, (248, ty), (320, ty))
    go(f, (630, ty), (720, ty))
    go(f, (910, ty), (962, ty))
    f.words(70, iy - 112, 'Image lane', 21, INK, 700)
    f.stack(66, iy - 54, 98, 98)
    f.words(122, iy + 88, 'noisy latent', 16, INK, anchor='middle')
    boxed(f, 202, iy - 48, 92, 92, ['fixed', 'map'], size=17, sub=17, oid='map')
    f.encoder(320, iy - 100, 310, 200, 'Image Encoder', IMAGE, size=26, oid='ie')
    f.pill(475, iy - 121, 'tuned', color=INK)
    f.words(475, iy + 128, f"{P_['vision_model']:,} numbers", 19, INK, 700, 'middle')
    f.rect(720, iy - 44, 190, 88, mix(IMAGE, 0.55), FIXED, 1.3, 12, oid='vp')
    f.words(815, iy - 4, 'image projection', 18, FROZEN_TEXT, anchor='middle')
    f.words(815, iy + 22, f"{C['vision_width']} to {C['shared_width']}", 15, FROZEN_TEXT, anchor='middle')
    f.pill(815, iy - 66, 'frozen')
    f.words(815, iy + 72, f"{P_['visual_projection']:,} numbers", 16, FROZEN_TEXT, anchor='middle')
    f.strip(962, iy - 14, 150, 30, IMAGE, oid='is')
    f.words(1037, iy - 28, f"I, {C['shared_width']} numbers", 16, INK, anchor='middle')
    go(f, (178, iy), (202, iy))
    go(f, (294, iy), (320, iy))
    go(f, (630, iy), (720, iy))
    go(f, (910, iy), (962, iy))
    f.words(590, 860, f"The image encoder holds {P_['tuned_share_percent']}% of CLIP's {P_['total']:,} numbers", 21, INK, 700, 'middle')
    f.words(590, 894, f"Score scale: exp({K['logit_scale_noisyclip']}) = {K['score_multiplier_noisyclip']}, against {K['score_multiplier_base']:.2f} in base CLIP", 18, FROZEN_TEXT, anchor='middle')
    # the opened encoder
    tx, tw = 1270, 400
    f.words(tx + tw / 2, 106, 'Inside the green block', 22, INK, 700, 'middle')
    rows = [(f"patch cut: {C['patch_size']} × {C['patch_size']} pixels", f"{C['patches']} patches, {C['vision_width']} numbers each", IMAGE),
            ('add summary token and positions', f"{C['image_tokens']} tokens", IMAGE),
            ('multi-head attention', f"{C['vision_heads']} heads, then add and norm", IMAGE),
            ('MLP', f"{C['vision_mlp']} wide, then add and norm", IMAGE),
            ('read the summary token', f"{C['vision_width']} numbers", IMAGE)]
    y = 140
    for i, (a, b_, col) in enumerate(rows):
        h = 104
        f.rect(tx, y, tw, h, col if i in (2, 3) else mix(col, 0.35), IMAGE_EDGE, 1.3, 9, oid=f'tower{i}')
        f.words(tx + tw / 2, y + 44, a, 19, INK, anchor='middle')
        f.words(tx + tw / 2, y + 74, b_, 16, FROZEN_TEXT, anchor='middle')
        if i < len(rows) - 1:
            go(f, (tx + tw / 2, y + h), (tx + tw / 2, y + h + 34))
        y += h + 34
    f.raw(f'<path d="M{tx - 16},{140 + 2 * 138} C{tx - 42},{140 + 2 * 138} {tx - 42},{140 + 3 * 138 + 104} {tx - 16},{140 + 3 * 138 + 104}" fill="none" stroke="{INK}" stroke-width="1.3"/>')
    f.words(tx - 50, 140 + 2 * 138 + 112, f"× {C['vision_layers']}", 21, INK, 700, 'end')
    f.words(tx + tw / 2, 870, 'Against base CLIP, the first vision block differs by', 17, FROZEN_TEXT, anchor='middle')
    f.words(tx + tw / 2, 896, f"{K['first_vision_layer_relative_change_percent']}% of its norm. Both projections and the first text", 17, FROZEN_TEXT, anchor='middle')
    f.words(tx + tw / 2, 922, 'block differ by 0. Those are the tensors compared.', 17, FROZEN_TEXT, anchor='middle')
    f.write(HERE, 'The published weights change the image encoder blocks. The text encoder, the text projection and the image projection are drawn faded and tagged frozen.')


def nc6():
    f = Fig('nc6-training-on-noisy-latents', 1760, 930)
    f.words(30, 44, 'Training pulls each noisy latent toward its own caption', 28, INK, 700)
    ty, iy = 190, 560
    mx, my, cw, rh = 880, 382, 118, 78
    f.cards(34, ty - 50, 186, 100, 'caption', 21)
    f.encoder(350, ty - 92, 300, 184, 'Text Encoder', TEXT, frozen=True, size=25, oid='te')
    f.pill(500, ty - 113, 'frozen')
    go(f, (232, ty), (350, ty))
    f.stack(40, iy - 70, 138, 138)
    f.words(116, iy + 104, 'noisy SDXL latents', 17, INK, anchor='middle')
    f.words(116, iy + 130, 'latents 20 to 29 of 50 (p. 6)', 15, FROZEN_TEXT, anchor='middle')
    boxed(f, 212, iy - 44, 110, 88, ['fixed', 'map'], size=17, sub=17, oid='map')
    go(f, (192, iy), (212, iy))
    go(f, (322, iy), (350, iy))
    f.encoder(350, iy - 92, 300, 184, 'Image Encoder', IMAGE, size=25, oid='ie')
    f.pill(500, iy - 113, 'tuned', color=INK)
    names = ['1', '2', '3', 'N']
    cols = [mx + cw * (j + .5) for j in range(4)]
    rows = [my + rh * (i + .5) for i in range(4)]
    f.line(650, ty, cols[-1], ty, INK, 1.4, name='bus', allowed=('te',))
    for j, x in enumerate(cols):
        go(f, (x, ty - 4.6), (x, 298))
        f.strip(x - 48, 298, 96, 24, TEXT)
        f.maths(f'T_{{{names[j]}}}', x, 350, 20)
    f.line(650, iy, 760, iy, INK, 1.4, name='ibus', allowed=('ie',))
    f.line(760, rows[0], 760, rows[-1], INK, 1.4, name='ibus2')
    for i, y in enumerate(rows):
        go(f, (755.4, y), (792, y))
        f.strip(792, y - 32, 24, 64, IMAGE, vertical=True)
        f.maths(f'I_{{{names[i]}}}', 850, y, 20)
        for j, x in enumerate(cols):
            f.raw(R(x - cw / 2, y - rh / 2, cw, rh, MATCH if i == j else '#f8f9fa', '#c9ced4', 0.9, 0))
            f.maths(rf'I_{{{names[i]}}}\cdot T_{{{names[j]}}}', x, y, 19)
    f.obstacle('matrix', mx, my, 4 * cw, 4 * rh, 'card')
    f.words(mx + 2 * cw, my + 4 * rh + 34, 'the diagonal holds each latent with its own caption', 18, INK, anchor='middle')
    bx, bw = 1400, 330
    f.rect(bx, 278, bw, 142, WHITE, INK, 1.2, 9, oid='loss')
    f.words(bx + bw / 2, 316, 'Contrastive loss:', 20, INK, 700, 'middle')
    f.words(bx + bw / 2, 352, 'maximise the diagonal,', 18, INK, anchor='middle')
    f.words(bx + bw / 2, 384, 'minimise the off-diagonal', 18, INK, anchor='middle')
    go(f, (mx + 4 * cw, 400), (bx, 400))
    f.card(bx, 450, bw, 380, FAINT, FAINT_LINE, 1.2, 16, oid='panel')
    f.words(bx + bw / 2, 490, 'attract and repel', 20, FROZEN_TEXT, 700, 'middle')
    a, p = (bx + 160, 660), (bx + 272, 556)
    others = ((bx + 56, 552), (bx + 74, 776), (bx + 268, 770))
    for q in others:
        dx, dy = q[0] - a[0], q[1] - a[1]
        n = math.hypot(dx, dy)
        f.arrow(a[0] + 28 * dx / n, a[1] + 28 * dy / n, q[0] - 26 * dx / n, q[1] - 26 * dy / n, FIXED, 1.6, 10, ('na', f'n{others.index(q)}'), 'rep')
    dx, dy = p[0] - a[0], p[1] - a[1]
    n = math.hypot(dx, dy)
    f.arrow(a[0] + 28 * dx / n, a[1] + 28 * dy / n, p[0] - 26 * dx / n, p[1] - 26 * dy / n, '#3f93bd', 2.4, 13, ('na', 'np'), 'att1')
    f.raw(Ar(p[0] - 26 * dx / n, p[1] - 26 * dy / n, a[0] + 28 * dx / n, a[1] + 28 * dy / n, '#3f93bd', 2.4, 13))
    for (x, y), c, t_, nid in ((a, IMAGE, 'I_1', 'na'), (p, TEXT, 'T_1', 'np'), (others[0], mix(TEXT, .45), 'T_2', 'n0'), (others[1], mix(TEXT, .45), 'T_3', 'n1'), (others[2], mix(TEXT, .45), 'T_N', 'n2')):
        f.circle(x, y, 22, c, INK, 1.1, oid=nid)
        f.maths(t_, x, y, 16)
    f.words(bx + 282, 648, 'attract', 16, '#2f7fa8', anchor='middle')
    f.words(bx + 74, 672, 'repel', 16, FIXED, anchor='middle')
    f.words(30, 800, f"The paper fine-tunes on {N['paper']['train_latents']['value']:,} latents in batches of {N['paper']['batch']['value']}, for {N['paper']['epochs']['value']} epochs (p. 8).", 18, FROZEN_TEXT)
    f.words(30, 830, 'Its loss treats every other caption in the batch as a wrong one (p. 6).', 18, FROZEN_TEXT)
    f.words(30, 860, 'The released script loads ready-made inputs and draws no step itself.', 18, FROZEN_TEXT)
    f.write(HERE, 'Noisy latents and captions enter two encoders. A similarity matrix with a pale blue diagonal feeds a contrastive loss. Only the image encoder is tuned.')


def nc7():
    f = Fig('nc7-trained-noise-levels-and-pushed-steps', 1600, 880)
    f.words(30, 44, 'The model was trained on steps 20 to 29, and this loop works at steps 0 to 15', 26, INK, 700)
    x0, x1, y0, y1 = 150, 1530, 730, 160
    px = lambda k: x0 + k / 49 * (x1 - x0)
    py = lambda v: y0 - v * (y0 - y1)
    a, b = S['push_window'], S['train_window']
    f.raw(R(px(a[0]), y1 - 28, px(a[1]) - px(a[0]), y0 - y1 + 28, mix(PUSH, .82), 'none', 0, 0))
    f.raw(R(px(b[0]), y1 - 28, px(b[1]) - px(b[0]), y0 - y1 + 28, mix(IMAGE, .35), 'none', 0, 0))
    f.words((px(a[0]) + px(a[1])) / 2, y1 - 42, 'this loop reads, and may push: 0 to 15', 20, PUSH, 700, 'middle')
    f.words((px(b[0]) + px(b[1])) / 2, y1 - 42, 'NoisyCLIP trained: 20 to 29', 20, '#3b7d50', 700, 'middle')
    f.raw(Ln(x0, y0, x1, y0, INK, 1.4) + Ln(x0, y0, x0, y1 - 28, INK, 1.4))
    for k in (0, 10, 15, 20, 29, 40, 49):
        f.raw(Ln(px(k), y0, px(k), y0 + 7, INK, 1.2))
        f.words(px(k), y0 + 30, str(k), 17, INK, anchor='middle')
    for v in (0, .25, .5, .75, 1):
        f.raw(Ln(x0 - 7, py(v), x0, py(v), INK, 1.2) + Ln(x0, py(v), x1, py(v), '#eceff2', 1.0))
        f.words(x0 - 13, py(v) + 6, f'{v:g}', 17, INK, anchor='end')
    f.words((x0 + x1) / 2, y0 + 72, 'sampler step of 50 (0 is the noisiest, 49 is almost finished)', 20, INK, anchor='middle')
    sig, sg = S['sigma'], S['signal']
    pts = ' '.join(f'{px(k):.1f},{py(sig[k]):.1f}' for k in range(50))
    f.raw(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="3"/>')
    pts = ' '.join(f'{px(k):.1f},{py(sg[k]):.1f}' for k in range(50))
    f.raw(f'<polyline points="{pts}" fill="none" stroke="{FIXED}" stroke-width="3" stroke-dasharray="9 6"/>')
    f.words(px(39), py(sig[39]) - 34, 'noise scale σ', 21, INK, 700, 'start')
    f.words(px(31.5), py(.93), 'signal scale √ᾱ', 21, FIXED, 700)
    f.words(30, 148, 'scale,', 18, INK)
    f.words(30, 172, '0 to 1', 16, FROZEN_TEXT)
    f.words((x0 + x1) / 2, y0 + 112, "Computed from SDXL's scheduler config: 50 steps, scaled-linear betas, leading spacing.", 17, FROZEN_TEXT, anchor='middle')
    f.write(HERE, 'Noise scale and signal scale against sampler step, from SDXL scheduler configuration, with the steps NoisyCLIP trained on and the steps this loop pushes marked.')


def nc8():
    f = Fig('nc8-right-minus-wrong', 1660, 800)
    f.words(30, 44, 'The score difference only sees where the two sentences differ', 28, INK, 700)
    mx, my, cw, rh = 330, 260, 210, 150
    for j, (t_, name, col) in enumerate((('T_{\\text{right}}', 'a cat to the left of a dog', MATCH), ('T_{\\text{wrong}}', 'a dog to the left of a cat', '#f3f4f6'))):
        x = mx + j * cw
        f.strip(x + 22, 186, cw - 44, 26, TEXT, oid=f'ts{j}')
        f.maths(t_, x + cw / 2, 158, 23)
        f.words(x + cw / 2, 120, name, 15, INK, anchor='middle')
        f.rect(x, my, cw, rh, col, MATCH_EDGE if j == 0 else FIXED, 1.4, 0, oid=f'cell{j}')
        f.maths(rf'I\cdot {t_}', x + cw / 2, my + rh / 2 - 10, 24)
        f.words(x + cw / 2, my + rh / 2 + 36, 'S right' if j == 0 else 'S wrong', 18, FROZEN_TEXT, anchor='middle')
        go(f, (x + cw / 2, 212), (x + cw / 2, my))
    f.strip(250, my + 28, 28, 94, IMAGE, vertical=True, oid='istrip')
    f.maths('I', 214, my + rh / 2, 25)
    go(f, (278, my + rh / 2), (mx, my + rh / 2))
    f.rect(mx, my + rh + 46, 2 * cw, 86, '#e9ebee', FIXED, 1.3, 9, oid='diff')
    f.maths(r'S_{\text{right}}-S_{\text{wrong}}=I\cdot(T_{\text{right}}-T_{\text{wrong}})', mx + cw, my + rh + 89, 23)
    go(f, (mx + cw / 2, my + rh), (mx + cw / 2, my + rh + 46))
    go(f, (mx + 3 * cw / 2, my + rh), (mx + 3 * cw / 2, my + rh + 46))
    # geometry
    cx, cy, r = 1190, 400, 260
    f.raw(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{FAINT_LINE}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    f.obstacle('o', cx - 4, cy - 4, 8, 8, 'circle')
    ang = lambda d: (cx + r * math.cos(math.radians(d)), cy - r * math.sin(math.radians(d)))
    tr, tw, im = ang(62), ang(118), ang(40)
    f.arrow(cx, cy, *tr, '#8d76d1', 3, 15, ('o',), 'tr')
    f.arrow(cx, cy, *tw, '#8d76d1', 3, 15, ('o',), 'tw')
    f.arrow(cx, cy, *im, IMAGE_EDGE, 3, 15, ('o',), 'im')
    f.raw(Ln(tw[0], tw[1], tr[0], tr[1], INK, 1.5, '6 5'))
    f.maths(r'T_{\text{right}}', tr[0] + 8, tr[1] - 24, 22)
    f.maths(r'T_{\text{wrong}}', tw[0] - 16, tw[1] - 24, 22)
    f.maths('I', im[0] + 22, im[1], 24)
    f.words(cx, cy - r - 60, 'T right minus T wrong is the dashed line', 18, INK, anchor='middle')
    f.words(cx, cy + r + 44, 'Sketch in 2 dimensions: the real vectors have 768 numbers.', 16, FROZEN_TEXT, anchor='middle')
    f.words(cx, cy + r + 70, 'If the two sentences get the same vector, the difference is 0 everywhere.', 16, FROZEN_TEXT, anchor='middle')
    f.write(HERE, 'One image embedding is multiplied with a right and a wrong sentence embedding. The difference of the two scores depends only on the difference of the two sentence vectors.')


def nc9():
    f = Fig('nc9-the-gradient-back-to-the-latent', 1760, 760)
    f.words(30, 44, 'The gradient walks back through every step to the latent', 28, INK, 700)
    stages = [('the latent', '4 × 128 × 128', WHITE, INK), ('linear map', 'fixed weights', FIXED_FILL, FIXED), ('clamp', '0 to 255', FIXED_FILL, FIXED),
              ('resize', f"to {C['image_size']}", FIXED_FILL, FIXED), ('normalise', 'CLIP mean, std', FIXED_FILL, FIXED),
              ('ViT', 'weights held', IMAGE, IMAGE_EDGE), ('projection', f"to {C['shared_width']}", mix(IMAGE, .55), FIXED),
              ('unit length', 'gives I', WHITE, INK), ('two scores', 'S right, S wrong', MATCH, MATCH_EDGE), ('difference', 'one number', '#e9ebee', FIXED)]
    n, sw, gap, x0, y = len(stages), 142, 32, 34, 150
    xs = [x0 + i * (sw + gap) for i in range(n)]
    f.words(34, 124, 'Forward', 21, INK, 700)
    for i, (a, b_, fill, edge) in enumerate(stages):
        f.rect(xs[i], y, sw, 120, fill, edge, 1.4, 10, oid=f's{i}')
        f.words(xs[i] + sw / 2, y + 52, a, 18, INK, anchor='middle')
        f.words(xs[i] + sw / 2, y + 82, b_, 13.5, FROZEN_TEXT, anchor='middle')
        if i:
            go(f, (xs[i - 1] + sw, y + 60), (xs[i], y + 60), gap=3.4, head=8)
    by = 400
    f.words(34, by - 20, 'Backward: the chain rule, one factor per box', 21, PUSH, 700)
    for i in range(n):
        f.dot(xs[i] + sw / 2, by + 26, r=7, oid=f'd{i}')
        if i:
            go(f, (xs[i] + sw / 2 - 7, by + 26), (xs[i - 1] + sw / 2 + 7, by + 26), color=PUSH, width=3.2, head=13, gap=5)
    notes = {0: ['g, same shape', 'as the latent'], 1: ['each latent', 'channel gets', 'its colours back'], 2: ['0 on any', 'clamped colour'],
             3: ['spreads over', 'nearby pixels'], 4: ['÷ CLIP std'], 5: ['weights held,', 'gradient flows'], 6: ['frozen,', 'gradient flows'],
             7: ['cosine', 'similarity'], 8: ['one dot product', 'per sentence'], 9: ['start: +1 from', 'S right, −1', 'from S wrong']}
    for i, lines in notes.items():
        for j, line in enumerate(lines):
            f.words(xs[i] + sw / 2, by + 68 + j * 24, line, 15.5, INK, anchor='middle')
    bx = 34
    f.rect(bx, 590, 840, 132, '#fff4e8', PUSH, 1.5, 10, oid='brk')
    f.words(bx + 22, 628, 'Two breaks in the released map', 20, PUSH, 700)
    f.words(bx + 22, 660, '.byte() cuts values to 8-bit integers (127.9 becomes 127), so no gradient passes.', 17, INK)
    f.words(bx + 22, 690, 'A clamped colour passes zero. The push needs a copy of the map without .byte().', 17, INK)
    f.rect(900, 590, 826, 132, FAINT, FAINT_LINE, 1.2, 10, oid='hold')
    f.words(922, 628, 'What the backward pass leaves alone', 20, INK, 700)
    f.words(922, 660, 'Every weight stays fixed while the push is computed.', 17, INK)
    f.words(922, 690, 'Only the gradient at the latent is read.', 17, INK)
    f.write(HERE, 'A forward row of ten stages from the latent to the score difference, and an orange backward row applying the chain rule, with the two points where the released map blocks the gradient.')


def nc10():
    f = Fig('nc10-how-the-push-joins-co3', 1760, 900)
    f.words(30, 44, "Where the push enters our sampler's noise prediction", 28, INK, 700)
    y = 250
    f.stack(46, y - 56, 112, 112)
    f.maths('x_t', 106, y + 92, 24)
    boxed(f, 214, y - 76, 230, 152, ['frozen SDXL U-Net', "CO3's main call", 'batch of 2'], fill='#e9ebee', size=19, sub=15, oid='unet')
    go(f, (172, y), (214, y))
    f.rect(520, y - 150, 190, 100, WHITE, INK, 1.2, 9, oid='u')
    f.maths(r'\epsilon_{\text{uncond}}', 615, y - 100, 23)
    f.rect(520, y + 50, 190, 100, WHITE, INK, 1.2, 9, oid='c')
    f.maths(r'\epsilon_{\text{cond}}', 615, y + 100, 23)
    go(f, (444, y - 30), (482, y - 30), (482, y - 100), (520, y - 100))
    go(f, (444, y + 30), (482, y + 30), (482, y + 100), (520, y + 100))
    for yy, key in ((y - 100, 'a'), (y + 100, 'b')):
        f.circle(800, yy, 22, WHITE, PUSH, 1.9, oid=f'j{key}')
        f.raw(Ln(788, yy, 812, yy, PUSH, 2.5))
        go(f, (710, yy), (778, yy))
    f.rect(690, y - 18, 220, 36, WHITE, PUSH, 1.3, 8, oid='p')
    f.maths(r'p=\lambda\,\sigma_t\,g', 800, y, 19, PUSH)
    go(f, (800, y - 18), (800, y - 78), color=PUSH, width=3, head=13)
    go(f, (800, y + 18), (800, y + 78), color=PUSH, width=3, head=13)
    f.rect(980, y - 88, 330, 176, WHITE, INK, 1.3, 10, oid='g')
    f.words(1145, y - 46, 'guided prediction', 19, INK, anchor='middle')
    f.maths(r'(\epsilon_{u}-p)+w\,[(\epsilon_{c}-p)-(\epsilon_{u}-p)]', 1145, y + 2, 16.5)
    f.maths(r'=\epsilon_{u}+w\,(\epsilon_{c}-\epsilon_{u})-p', 1145, y + 46, 19)
    go(f, (822, y - 100), (900 + 40, y - 100), (940, y - 34), (980, y - 34))
    go(f, (822, y + 100), (940, y + 100), (940, y + 34), (980, y + 34))
    boxed(f, 1400, y - 60, 310, 120, ["CO3's next step", 'with the pushed prediction'], fill='#e9ebee', size=20, sub=15, oid='next')
    go(f, (1310, y), (1400, y))
    f.rect(46, 480, 820, 348, FAINT, FAINT_LINE, 1.2, 12, oid='left')
    f.words(70, 524, 'What the existing push code does', 21, INK, 700)
    lines = ["On CO3's main call only, the batch of two, it subtracts p from both entries.",
             'The push length is capped at c times the length of the guided prediction.',
             'A random direction of matched length is the control.',
             'Zero strength must reproduce the plain sampler.']
    for i, line in enumerate(lines):
        f.words(70, 572 + i * 40, line, 18, INK)
    f.words(70, 762, 'from 2-classifier-guidance/scripts/method_search/co3_push.py', 16, FROZEN_TEXT)
    f.rect(900, 480, 810, 348, FAINT, FAINT_LINE, 1.2, 12, oid='right')
    f.words(924, 524, 'What is proposed here', 21, INK, 700)
    lines = ['The checker in that code is swapped for S right minus S wrong.',
             'Its gradient g is taken at the latent.',
             'The step W3 decides the strength, cap and window.',
             'No sentence score has been run on our pictures.']
    for i, line in enumerate(lines):
        f.words(924, 572 + i * 40, line, 18, INK)
    f.words(924, 762, 'steps W2 and W3 measure whether it reads or steers', 16, FROZEN_TEXT)
    f.write(HERE, "The push p is subtracted from both of CO3's two noise predictions, which lowers the guided prediction by p before the next sampler step.")


ALL = {'nc1': nc1, 'nc2': nc2, 'nc3': nc3, 'nc4': nc4, 'nc5': nc5, 'nc6': nc6, 'nc7': nc7, 'nc8': nc8, 'nc9': nc9, 'nc10': nc10}


WC = json.loads(Path('/home/molef/PhD/concept-composition/artifacts/notes/progress/noise-text-critic/02-can-it-read/word-check/word-check.json').read_text())


def _cd(model, prop):
    return next(r for r in WC[model][prop] if r['pair'] == 'cat, dog')


def nc11():
    f = Fig('nc11-the-text-only-word-check', 1760, 1010)
    f.words(30, 44, 'Average both word orders, and keep only what survives', 28, INK, 700)
    cards = [('a cat to the left of a dog', 190), ('a dog to the right of a cat', 285), ('a dog to the left of a cat', 455), ('a cat to the right of a dog', 550)]
    f.words(34, 150, 'same meaning', 18, FROZEN_TEXT, 700)
    f.words(34, 415, 'nouns swapped', 18, FROZEN_TEXT, 700)
    f.encoder(540, 160, 250, 430, 'Text Encoder', TEXT, frozen=True, shape='round', size=24, oid='enc')
    f.pill(665, 138, 'frozen, CLIP ViT-L/14')
    labels = ['u₁', 'u₂', 'u₃', 'u₄']
    for i, (t, y) in enumerate(cards):
        f.rect(34, y - 30, 400, 60, WHITE, INK, 1.1, 6, oid=f'card{i}')
        f.words(234, y + 7, t, 18, INK, anchor='middle')
        go(f, (434, y), (540, y))
        f.strip(870, y - 15, 150, 30, TEXT, oid=f'u{i}')
        f.words(1045, y + 8, labels[i], 20, INK)
        go(f, (790, y), (870, y))
    for k, (a, b, ym, name) in enumerate(((190, 285, 237, 'sum, same meaning'), (455, 550, 502, 'sum, nouns swapped'))):
        f.circle(1140, ym, 19, WHITE, INK, 1.5, oid=f'plus{k}')
        f.raw(Ln(1130, ym, 1150, ym, INK, 2) + Ln(1140, ym - 10, 1140, ym + 10, INK, 2))
        go(f, (1080, a), (1140, a), (1140, ym - 19))
        go(f, (1080, b), (1140, b), (1140, ym + 19))
        f.strip(1200, ym - 15, 150, 30, TEXT, oid=f's{k}')
        f.words(1275, ym - 26, name, 16, FROZEN_TEXT, anchor='middle')
        go(f, (1159, ym), (1200, ym))
    ym = 370
    f.circle(1440, ym, 21, WHITE, INK, 1.5, oid='minus')
    f.raw(Ln(1429, ym, 1451, ym, INK, 2.2))
    go(f, (1350, 237), (1440, 237), (1440, ym - 21))
    go(f, (1350, 502), (1440, 502), (1440, ym + 21))
    f.strip(1520, ym - 15, 150, 30, TEXT, oid='w')
    f.maths('w', 1700, ym, 26)
    go(f, (1461, ym), (1520, ym))
    f.rect(1150, 620, 580, 210, WHITE, INK, 1.3, 10, oid='ratio')
    f.maths(r'\text{ratio}=\frac{\lVert w\rVert}{2\,\lVert u_1-u_3\rVert}', 1440, 670, 26)
    o, b = _cd('trained', 'order')['ratio'], _cd('trained', 'bind')['ratio']
    f.words(1440, 746, f'cat × dog, order: {o:.2f}     binding: {b:.2f}', 21, INK, 700, 'middle')
    f.words(1440, 788, '0 for a pure sum of word slots, 1 when the whole swap survives', 15, FROZEN_TEXT, anchor='middle')
    f.words(34, 650, 'The binding set swaps colours the same way:', 17, FROZEN_TEXT)
    f.words(34, 678, '"a red cat and a blue dog" and "a blue dog and a red cat"', 17, FROZEN_TEXT)
    f.words(34, 706, 'against "a blue cat and a red dog" and "a red dog and a blue cat".', 17, FROZEN_TEXT)
    f.dotted(20, 858, 1740)
    y = 930
    f.words(34, 892, 'Control: the same encoder with random weights', 19, FADED_TEXT, 700)
    f.rect(34, y - 26, 400, 52, '#f7f8f9', FAINT_LINE, 1.0, 6, oid='tcard')
    f.words(234, y + 7, 'the same four sentences', 18, FADED_TEXT, anchor='middle')
    f.rect(540, y - 30, 250, 60, FAINT, FAINT_LINE, 1.2, 14, oid='tenc')
    f.words(665, y + 7, 'random weights', 20, FADED_TEXT, anchor='middle')
    f.raw(R(870, y - 15, 150, 30, FAINT, FAINT_LINE, 1.0, 0))
    f.obstacle('tw', 870, y - 15, 150, 30, 'rect')
    f.maths('w', 1050, y, 22, FADED_TEXT)
    go(f, (434, y), (540, y), color=FAINT_LINE)
    go(f, (790, y), (870, y), color=FAINT_LINE)
    ro, rb = _cd('random_weights', 'order')['ratio'], _cd('random_weights', 'bind')['ratio']
    f.words(1440, y + 8, f'cat × dog, order: {ro:.2f}     binding: {rb:.2f}', 21, FADED_TEXT, 700, 'middle')
    f.words(30, 990, "Text only: no image is read, so this says nothing about NoisyCLIP's image side. Values from word-check.json.", 16, FROZEN_TEXT)
    f.write(HERE, 'Four sentences of one swap set pass the frozen text encoder. The two same-meaning vectors are summed, the two swapped ones are summed, and their difference w is what a sum of word slots cannot make. Cat and dog give 0.33 for order and 0.61 for binding, against 0.08 and 0.11 with random weights.')


def nc12():
    f = Fig('nc12-what-survives-for-six-noun-pairs', 1600, 700)
    f.words(30, 44, 'Binding survives averaging in trained CLIP, and order survives a little', 26, INK, 700)
    x0, x1 = 360, 1530
    px = lambda v: x0 + v * (x1 - x0)
    rows = [('binding, trained', 'trained', 'bind'), ('binding, random weights', 'random_weights', 'bind'),
            ('order, trained', 'trained', 'order'), ('order, random weights', 'random_weights', 'order')]
    for i, (name, m, p) in enumerate(rows):
        y = 150 + i * 105
        faded = m == 'random_weights'
        f.words(x0 - 24, y + 7, name, 19, FADED_TEXT if faded else INK, 400 if faded else 700, 'end')
        f.raw(Ln(x0, y, x1, y, '#eceff2', 1.0))
        for r_ in WC[m][p]:
            v = r_['ratio']
            cat = r_['pair'] == 'cat, dog'
            f.raw(f'<circle cx="{px(v):.1f}" cy="{y}" r="{11 if cat else 8}" fill="{FAINT if faded else ("#8d76d1" if cat else TEXT)}" stroke="{FADED_TEXT if faded else INK}" stroke-width="{1.6 if cat else 1}"/>')
        mean = WC[m][f'{p}_mean_ratio']
        f.raw(Ln(px(mean), y - 24, px(mean), y + 24, FADED_TEXT if faded else INK, 2.4))
        f.words(px(mean), y - 32, f'mean {mean:.2f}', 15, FADED_TEXT if faded else INK, 700, 'middle')
    ya = 150 + 4 * 105 - 40
    f.raw(Ln(x0, ya, x1, ya, INK, 1.3))
    for v in (0, .25, .5, .75, 1):
        f.raw(Ln(px(v), ya, px(v), ya + 7, INK, 1.2))
        f.words(px(v), ya + 30, f'{v:g}', 16, INK, anchor='middle')
    f.words(px(0), ya + 58, 'pure sum of word slots', 15, FROZEN_TEXT)
    f.words(px(1), ya + 58, 'whole swap survives', 15, FROZEN_TEXT, anchor='end')
    f.words((x0 + x1) / 2, ya + 90, 'ratio ‖w‖ / (2 ‖swap‖), one dot per noun pair, the larger dot is cat × dog', 17, INK, anchor='middle')
    f.write(HERE, 'Ratio of what survives averaging, from 0 to 1, for six noun pairs: binding near 0.72 and order near 0.28 in trained CLIP, against 0.13 and 0.08 with random weights.')


ALL['nc11'] = nc11
ALL['nc12'] = nc12

if __name__ == '__main__':
    bad = 0
    for key in (sys.argv[1:] or ALL):
        try:
            ALL[key]()
        except RuntimeError as e:
            bad += 1
            text = str(e).split('\n')
            print(text[0])
            for line in text[1:9]:
                print('  ', line[:190])
    print('failed' if bad else 'all written', bad)
