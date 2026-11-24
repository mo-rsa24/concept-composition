"""Levels 0 and 0.5 of the score-path ladder, in the clip-contrastive style, drawn with figure_kit's layout checks.

Every size comes from ../numbers/numbers.json (read from the repository's own code). The canvas pictures are the real
canvas from ../overview/make_canvas_3_7.py. Anything belonging to the score-path encoder is dashed rose and tagged
"illustrative, not yet designed". Run from this folder: PYTHONDONTWRITEBYTECODE=1 python3 make_network_view.py [stems]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ladder_kit import *  # noqa: E402,F403

OV = HERE.parent / 'overview'
U, SP, W4, TR = N['unet'], N['sampler'], N['w4_text_with_latent'], N['training_of_the_digit_models']
WIN = SP['window_steps_0_to_15']
CLEAN, NOISY0, NOISY15 = OV / 'canvas-3-7.png', OV / 'canvas-3-7-step0-t999.png', OV / 'canvas-3-7-step15-t693.png'


def dashed_trapezoid(f, x, y, w, h, oid, label, sub=None):
    n = h * 0.22
    f.obstacle(oid, x, y, w, h, 'rect')
    pts = ' '.join(f'{a:.1f},{b:.1f}' for a, b in [(x, y), (x + w, y + n), (x + w, y + h - n), (x, y + h)])
    f.raw(f'<polygon points="{pts}" fill="{mix(SCORE, 0.55)}" stroke="{SCORE_EDGE}" stroke-width="1.8" stroke-dasharray="{PLANNED_DASH}"/>')
    f.words(x + w / 2, y + h / 2 + (0 if sub else 7), label, 19, SCORE_INK, 600, anchor='middle')
    if sub:
        f.words(x + w / 2, y + h / 2 + 24, sub, 14, SCORE_INK, anchor='middle')


def sys0():
    f = Fig('sys0-where-the-score-path-sits', 2200, 1000)
    title(f, 'Where a score path comes from, and where Loop 10 would send it',
          'Solid parts exist in the repository. Dashed rose parts are planned for step X4 and have no design yet.')
    # input
    f.image(CLEAN, 40, 520, 180, 90, oid='clean')
    f.words(130, 640, 'x0: a real 7 and 3', 15, INK, anchor='middle')
    f.words(130, 662, 'make_canvas, 32 x 64', 14, FROZEN_TEXT, anchor='middle')
    f.image(NOISY0, 300, 420, 180, 90, oid='n0')
    f.words(390, 405, 'x_t, noised', 15, INK, anchor='middle')
    f.image(NOISY15, 300, 730, 180, 90, oid='n15')
    f.words(390, 850, 'x_t, noised', 15, INK, anchor='middle')
    go(f, (220, 545), (260, 545), (260, 465), (300, 465), name='c-n0')
    go(f, (220, 590), (260, 590), (260, 775), (300, 775), name='c-n15')
    # text lane
    f.cards(300, 180, 180, 70, f"{W4['caption_bank']} captions", oid='caps')
    f.encoder(580, 150, 320, 130, 'Text encoder', TEXT, frozen=True, tag='frozen, CLIP-L', oid='tenc')
    go(f, (494, 215), (580, 215), name='caps-tenc')
    f.strip(1000, 200, 200, 30, TEXT, oid='T')
    f.words(1100, 186, f"T, {W4['text_width']} numbers", 15, INK, anchor='middle')
    go(f, (900, 215), (1000, 215), name='tenc-T')
    # score lane
    boxed(f, 580, 400, 320, 130, ['Digit U-Net', f"{U['weights']:,} weights, frozen", 'share-0.0 or share-0.01, on the cluster'], oid='unet')
    go(f, (480, 465), (580, 465), name='n0-unet')
    f.pill(740, 372, 'step t and one of 11 labels')
    boxed(f, 980, 410, 240, 110, ['d_t(a), d_t(b)', 'label call minus', 'no-digit call, 1 x 32 x 64'], fill=mix(SCORE, 0.6), stroke=SCORE_EDGE, oid='dt')
    go(f, (900, 465), (980, 465), name='unet-dt')
    boxed(f, 1290, 410, 240, 110, ['score path', 'steps 0 to 15 stacked', f"16 x 32 x 64 = {N['score_path']['numbers']:,}"], fill=mix(SCORE, 0.4), stroke=SCORE_EDGE, oid='stack')
    go(f, (1220, 465), (1290, 465), name='dt-stack')
    dashed_trapezoid(f, 1600, 395, 220, 140, 'senc', 'Score-path', 'encoder, planned')
    go(f, (1530, 465), (1600, 465), name='stack-senc')
    f.strip(1860, 450, 60, 30, SCORE, oid='S')
    go(f, (1820, 465), (1860, 465), name='senc-S')
    planned_tag(f, 1710, 566)
    # sampler loop
    boxed(f, 980, 600, 240, 70, ['e_PoE = e_null + d(a) + d(b)'], fill=WHITE, stroke=INK, size=16, oid='poe')
    go(f, (1100, 520), (1100, 600), name='dt-poe')
    boxed(f, 580, 600, 320, 70, ['DDIM step: the next x_t', '50 steps, eta 0'], fill=FIXED_FILL, size=16, sub=13, oid='ddim')
    go(f, (980, 635), (900, 635), name='poe-ddim')
    go(f, (580, 635), (520, 635), (520, 540), (440, 540), (440, 510), name='ddim-loop')
    f.words(740, 588, 'or a test canvas noised once per step: X4 decides', 14, SCORE_INK, anchor='middle')
    # canvas lane
    f.encoder(580, 710, 320, 130, 'W4 canvas encoder', IMAGE, tag='Loop 9, 4 conv blocks', oid='ienc')
    go(f, (480, 775), (580, 775), name='n15-ienc')
    f.strip(1000, 760, 200, 30, IMAGE, oid='I')
    f.words(1100, 820, f"I, {W4['text_width']} numbers", 15, INK, anchor='middle')
    go(f, (900, 775), (1000, 775), name='ienc-I')
    # shared space
    f.card(1960, 150, 220, 700, mix(SCORE, 0.85), SCORE_EDGE, 1.8, 16, oid='space', dash=PLANNED_DASH)
    f.words(2070, 190, 'one shared', 18, SCORE_INK, 700, anchor='middle')
    f.words(2070, 214, 'space', 18, SCORE_INK, 700, anchor='middle')
    f.words(2070, 240, '(planned)', 15, SCORE_INK, anchor='middle')
    for i, line in enumerate(('T against S', 'I against S', 'T against I', '', 'two tests:', 'sum lands', 'on joint;', 'gap predicts', 'the term')):
        f.words(2070, 330 + i * 30, line, 15, INK, anchor='middle')
    go(f, (1200, 215), (1960, 215), name='T-space')
    go(f, (1920, 465), (1960, 465), name='S-space', allowed=('space',))
    go(f, (1200, 775), (1960, 775), name='I-space')
    footer(f, 'Sizes: numbers.json (canvas_unet.py, w4_train.py, Loop 9 design.md). Canvases: build_env.make_canvas on MNIST t10k; the noise on them is one numpy draw.')
    f.write(HERE, 'The system around a score path: a real canvas noised, the frozen digit U-Net giving d_t, sixteen steps stacked, a planned encoder, and a planned shared space beside CLIP-L text and Loop 9 canvas encoders.')


def sp1():
    f = Fig('sp1-inside-the-digit-unet', 2200, 980)
    title(f, 'Inside the digit U-Net: three grid sizes down and back up',
          f"CanvasUNet with its trainer's defaults: {U['weights']:,} weights. Each box names its output, channels x height x width.")
    f.image(NOISY0, 40, 195, 180, 90, oid='in')
    f.words(130, 312, 'x_t, 1 x 32 x 64', 15, INK, anchor='middle')
    L = U['levels']
    def lv(x, y, w, h, shape, lines, oid, glow=False):
        if glow:
            f.halo(x, y, w, h, IMAGE)
        boxed(f, x, y, w, h, [' x '.join(map(str, shape))] + lines, fill=mix(IMAGE, 0.35 if glow else 0.6), stroke=IMAGE_EDGE, size=19, sub=14, oid=oid)
    lv(280, 180, 280, 120, L[0], ['input conv, 2 ResBlocks'], 'd1')
    lv(400, 380, 280, 100, L[1], ['stride-2 conv, 2 ResBlocks'], 'd2')
    lv(520, 560, 280, 90, L[2], ['stride-2 conv, 2 ResBlocks'], 'd3')
    lv(650, 740, 450, 100, L[2], ['ResBlock, 4-head attention, ResBlock'], 'mid', glow=True)
    lv(950, 560, 280, 90, L[2], ['3 ResBlocks'], 'u3')
    lv(1070, 380, 280, 100, L[1], ['upsample, conv, 3 ResBlocks'], 'u2')
    lv(1190, 180, 280, 120, L[0], ['upsample, conv, 3 ResBlocks'], 'u1')
    go(f, (220, 240), (280, 240), name='in-d1')
    for a, b, x in (((300, 380), 'd2', 480), ((480, 560), 'd3', 600), ((650, 740), 'mid', 700)):
        go(f, (x, a[0]), (x, a[1]), name=f'down-{b}')
    for a, x in (((740, 650), 1050), ((560, 480), 1150), ((380, 300), 1270)):
        go(f, (x, a[0]), (x, a[1]), name=f'up-{x}')
    for y, x0, x1 in ((240, 560, 1190), (430, 680, 1070), (605, 800, 950)):
        go(f, (x0, y), (x1, y), color=FIXED, width=1.2, name=f'skip-{y}')
    f.words(875, 228, 'skip: joined by channels', 14, FROZEN_TEXT, anchor='middle')
    boxed(f, 1540, 195, 180, 90, ['output conv', '1 x 32 x 64'], fill=WHITE, stroke=INK, size=16, oid='out')
    go(f, (1470, 240), (1540, 240), name='u1-out')
    f.maths(r'\epsilon_\theta(x_t, t, c)', 1630, 330, 22, INK)
    # the join of step and label
    f.card(1780, 380, 390, 470, FAINT, FAINT_LINE, 1.2, 14, oid='join')
    f.words(1800, 418, 'How the step and the label enter', 17, INK, 700)
    f.words(1800, 455, f"step t: {U['base_channels']} sines and cosines,", 15, INK)
    f.words(1800, 478, f"then a 2-layer MLP to {U['vector_width']}", 15, INK)
    f.words(1800, 515, f"label c: one row of an {U['labels']} x {U['vector_width']} table", 15, INK)
    f.words(1800, 538, 'rows 0 to 9 are digits, row 10 is no digit', 15, INK)
    f.maths(r'\mathrm{emb} = \mathrm{MLP}(\mathrm{sin}(t)) + \mathrm{table}[c]', 1975, 585, 19, INK)
    f.words(1800, 630, 'emb is added inside every one of the', 15, INK)
    f.words(1800, 653, f"{sum(U['res_blocks'].values())} ResBlocks, after its first conv", 15, INK)
    f.words(1800, 700, 'A pair has no row of its own: two', 15, SCORE_INK, 600)
    f.words(1800, 723, 'digits enter only as two calls whose', 15, SCORE_INK, 600)
    f.words(1800, 746, 'differences are added (the product).', 15, SCORE_INK, 600)
    f.words(1800, 800, 'canvas_unet.py, CanvasUNet.forward', 14, FROZEN_TEXT)
    # weights by part
    wp = U['weights_by_part']
    f.words(60, 885, 'Weights by part: ' + ', '.join(f"{k.replace('_', ' ')} {v:,}" for k, v in wp.items()), 15, INK)
    footer(f, 'Shapes traced through the real forward pass on the CPU with untrained weights (numbers/make_numbers.py); no trained weight was read.')
    f.write(HERE, 'The digit U-Net drawn as a U: 64 x 32 x 64 down to 128 x 8 x 16 with attention in the middle and back up, skips across, the step and label entering as one 256-number vector.')


def sp2():
    f = Fig('sp2-sixteen-steps-make-a-score-path', 2200, 960)
    title(f, 'Sixteen sampler steps, one d per label each, stacked into a score path',
          'Steps 0 to 15 are the first 16 of DDIM\'s 50, the window W4 trains on. Rose tiles are d_t maps: none has been computed yet.')
    x0, cw = 260, 118
    rows = [('step', 150), ('t', 190), ('signal scale', 240), ('noise scale', 290)]
    for name, y in rows:
        f.words(x0 - 20, y + 22, name, 16, INK, anchor='end')
    for k, r in enumerate(WIN):
        x = x0 + k * cw
        f.words(x + cw / 2, 172, str(k), 16, INK, 700, anchor='middle')
        cells(f, x + 4, 190, [r['t']], cw=cw - 8, ch=34, size=15, oid=f't{k}')
        cells(f, x + 4, 240, [f"{r['sqrt_abar']:.3f}"], cw=cw - 8, ch=34, size=14, fill=mix(IMAGE, 0.6), oid=f's{k}')
        cells(f, x + 4, 290, [f"{r['sqrt_one_minus_abar']:.3f}"], cw=cw - 8, ch=34, size=14, fill=FIXED_FILL, oid=f'n{k}')
    f.words(x0 - 20, 395, 'x_t', 16, INK, anchor='end')
    f.image(NOISY0, x0 + 4, 355, cw - 8, 55, oid='img0')
    f.image(NOISY15, x0 + 15 * cw + 4, 355, cw - 8, 55, oid='img15')
    f.words(x0 + 8 * cw, 390, 'the same real canvas with one noise draw, shown at steps 0 and 15', 15, FROZEN_TEXT, anchor='middle')
    for j, (lab, y) in enumerate(((r'd_t(3)', 450), (r'd_t(7)', 530))):
        f.maths(lab, x0 - 20, y + 32, 21, INK, anchor='end')
        for k in range(16):
            f.rect(x0 + k * cw + 4, y, cw - 8, 55, mix(SCORE, 0.55), SCORE_EDGE, 1.1, 4, oid=f'd{j}{k}', dash='4 3')
    f.words(x0 + 8 * cw, 625, 'each tile is 1 x 32 x 64: the 3-call minus the no-digit call at that step (needs the U-Net on a GPU)', 15, SCORE_INK, anchor='middle')
    # the stacks
    bx = 260
    for i, (name, sub) in enumerate((('S(3)', 'd_t(3) at steps 0 to 15'), ('S(7)', 'd_t(7) at steps 0 to 15'),
                                     ('S(3) + S(7)', "what the product adds"))):
        x = bx + i * 520
        for k in range(4, 0, -1):
            f.raw(R(x + k * 10, 690 - k * 10, 300, 110, mix(SCORE, 0.75 - 0.08 * k), SCORE_EDGE, 0.9, 6))
        boxed(f, x, 690, 300, 110, [name, sub, f"16 x 32 x 64 = {N['score_path']['numbers']:,}"], fill=mix(SCORE, 0.35), stroke=SCORE_EDGE, size=19, sub=14, oid=f'S{i}')
    f.words(bx + 1560, 720, 'All three are read at the same x_t,', 15, INK)
    f.words(bx + 1560, 743, 'so S(3)+S(7) is exact, step by step.', 15, INK)
    f.words(bx + 1560, 790, 'Signal scale = sqrt(abar_t): the clean canvas is', 15, FROZEN_TEXT)
    f.words(bx + 1560, 813, f"x {WIN[0]['sqrt_abar']} of x_t at step 0, x {WIN[15]['sqrt_abar']} at step 15.", 15, FROZEN_TEXT)
    footer(f, "t, signal and noise scales: ddim_timesteps(50)[:16] and ABAR in canvas_unet.py, via numbers.json. v2_learn.py's other window, t = 712 to 480, is SDXL's noise level; X4 picks one.")
    f.write(HERE, 'The 16 steps of the window with their timestep and signal and noise scales, empty rose d_t tiles for two labels, and three stacks: S(3), S(7) and their sum.')


def sp3():
    f = Fig('sp3-the-planned-score-path-encoder', 2200, 900)
    title(f, 'Three encoders into one width: two exist, the score-path one is planned',
          'Every size on the dashed rose encoder is illustrative, not yet designed. Its input size is real.')
    rows = [(190, 'text'), (420, 'canvas'), (650, 'score')]
    f.cards(60, 200, 220, 70, '"the digit three ..."', size=14, oid='cap')
    f.words(170, 300, f"one of {W4['caption_bank']} captions", 14, FROZEN_TEXT, anchor='middle')
    f.encoder(400, 170, 380, 130, 'CLIP-L text encoder', TEXT, frozen=True, tag='frozen', oid='te')
    go(f, (294, 235), (400, 235), name='cap-te')
    f.words(840, 220, f"{W4['text_layers']} layers, {W4['text_width']} out", 15, INK)
    f.image(NOISY15, 60, 405, 220, 110, oid='cv')
    f.words(170, 540, 'noisy canvas, 1 x 32 x 64', 14, FROZEN_TEXT, anchor='middle')
    f.encoder(400, 395, 380, 130, 'W4 canvas encoder', IMAGE, tag='trained by Loop 9', oid='ie')
    go(f, (280, 460), (400, 460), name='cv-ie')
    f.words(840, 420, 'conv widths ' + ', '.join(map(str, W4['encoder_widths'])) + ', step told,', 15, INK)
    f.words(840, 444, f"flatten, then {W4['text_width']} out", 15, INK)
    for k in range(4, 0, -1):
        f.raw(R(60 + k * 8, 640 - k * 8, 220, 110, mix(SCORE, 0.75 - 0.08 * k), SCORE_EDGE, 0.9, 6))
    boxed(f, 60, 640, 220, 110, ['score path', '16 x 32 x 64'], fill=mix(SCORE, 0.35), stroke=SCORE_EDGE, oid='sp')
    dashed_trapezoid(f, 400, 625, 380, 140, 'se', 'Score-path encoder', 'planned')
    go(f, (280, 695), (400, 695), name='sp-se')
    planned_tag(f, 590, 600)
    f.words(840, 655, 'width out drawn as 768 to meet the others:', 15, SCORE_INK)
    f.words(840, 679, 'illustrative, not yet designed', 15, SCORE_INK, 600)
    for y, colour, oid in ((235, TEXT, 'sT'), (460, IMAGE, 'sI'), (695, SCORE, 'sS')):
        f.strip(1300, y - 15, 200, 30, colour, oid=oid)
    go(f, (780, 235), (1300, 235), name='te-s', allowed=())
    go(f, (780, 460), (1300, 460), name='ie-s')
    go(f, (780, 695), (1300, 695), name='se-s')
    f.card(1560, 150, 600, 640, FAINT, FAINT_LINE, 1.2, 14, oid='qs')
    f.words(1585, 190, 'What step X4 has to decide for it', 18, INK, 700)
    qs = [["Which x_t the 16 steps read: the sampler's", 'own path, or a test canvas noised per step.'],
          ['Which window: t = 999 to 693, or 712 to 480.'],
          ["Raw d_t, or scaled like V2's maps (each", 'map adds to 2,048).'],
          ['Whether it is told the step, as W4 is.'],
          ['What a joint path is: the models have no', 'pair label, so only product paths exist.'],
          ['How it reads 16 steps: as 16 channels, or', 'one step at a time, then pooled.']]
    y = 235
    for q in qs:
        for line in q:
            f.words(1585, y, line, 15, INK)
            y += 24
        y += 16
    f.words(1585, 650, 'These are open in the code as of', 14, FROZEN_TEXT)
    f.words(1585, 672, f"commit {N['source_commit'][:8]}; none is answered here.", 14, FROZEN_TEXT)
    footer(f, 'Real sizes: numbers.json (CLIP-L text width and caption bank from w4_train.py; W4 widths from Loop 9 design.md). The score-path encoder has no design.')
    f.write(HERE, 'Three lanes: a caption through the frozen CLIP-L text encoder, a noisy canvas through Loop 9 W4 encoder, and a score path through a planned dashed encoder, each to a strip, beside the questions step X4 must settle.')


def sp4():
    f = Fig('sp4-three-spaces-three-matrices', 2200, 980)
    title(f, 'All three in one space: three similarity matrices, one per pair of kinds',
          'Every cell value is unknown until X6 trains; cells show only which product sits where. Pale blue cells are matching pairs.')
    names = ['1', '2', '3', 'N']
    def matrix(mx, my, top, side, topc, sidec, label, sub, oid):
        cw, rh = 95, 62
        for j in range(4):
            f.strip(mx + j * cw + 17, my - 40, cw - 34, 20, topc, oid=f'{oid}t{j}')
            f.words(mx + j * cw + cw / 2, my - 52, f'{top}{names[j]}', 16, INK, anchor='middle')
        for i in range(4):
            f.strip(mx - 40, my + i * rh + 12, 20, rh - 24, sidec, vertical=True, oid=f'{oid}s{i}')
            f.words(mx - 52, my + i * rh + rh / 2 + 6, f'{side}{names[i]}', 16, INK, anchor='end')
            for j in range(4):
                f.rect(mx + j * cw, my + i * rh, cw, rh, MATCH if i == j else '#f8f9fa', '#c9ced4', 0.8, 0, oid=f'{oid}{i}{j}')
                f.words(mx + j * cw + cw / 2, my + i * rh + rh / 2 + 6, f'{side}{names[i]}.{top}{names[j]}', 14, INK, anchor='middle')
        f.words(mx + 2 * cw, my + 4 * rh + 40, label, 18, INK, 700, anchor='middle')
        f.words(mx + 2 * cw, my + 4 * rh + 66, sub, 15, FROZEN_TEXT, anchor='middle')
    matrix(160, 250, 'T', 'S', TEXT, SCORE, 'captions against score paths', 'planned: X6', 'ts')
    matrix(890, 250, 'I', 'S', IMAGE, SCORE, 'canvases against score paths', 'planned: X6', 'is')
    matrix(1620, 250, 'T', 'I', TEXT, IMAGE, 'captions against canvases', "W4's task: Loop 9, not yet landed", 'ti')
    f.rect(140, 700, 1920, 150, WHITE, INK, 1.0, 12, oid='loss')
    f.maths(r'\mathcal{L}_{\mathrm{three}} = \mathcal{L}(T,S) + \mathcal{L}(I,S) + \mathcal{L}(T,I), \qquad \mathcal{L}(A,B) = \tfrac12\left(\mathrm{CE}_{rows} + \mathrm{CE}_{cols}\right)', 1100, 755, 24, INK)
    f.words(1100, 815, 'One form the "all three" row of the session note could take, in the way CLIP goes 3D adds a third encoder. X4 sets the real one.', 15, SCORE_INK, anchor='middle')
    footer(f, 'Layout follows the clip-contrastive style reference of the NoisyCLIP ladder. No number in a cell: no score path has been embedded yet.')
    f.write(HERE, 'Three 4 by 4 similarity matrices, captions against score paths, canvases against score paths and captions against canvases, with pale blue diagonals and one summed loss marked as a planned form.')


FIGS = {'sys0': sys0, 'sp1': sp1, 'sp2': sp2, 'sp3': sp3, 'sp4': sp4}
if __name__ == '__main__':
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
