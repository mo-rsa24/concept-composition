"""Level 2 of the score-path ladder: the miniature's figures, every number read from miniature_answer_key.json.

Run from this folder: PYTHONDONTWRITEBYTECODE=1 python3 make_miniature_figures.py [figure stems]
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ladder_kit import *  # noqa: E402,F403

K = json.loads((HERE / 'miniature_answer_key.json').read_text())
S0, S1 = K['steps']
Wt = K['weights']
TOY = 'A hand-set toy: 4 pixels, 2 steps, integer weights. It shows where each term sits and predicts nothing about the digit models.'
CW = 50


def label_row(f, x, y, label, values, cx, fill=WHITE, signed=False, size=18, maths=False, oid=None):
    """One row of an equation laid out in cells: a label ending at cx - 18, then the four cells."""
    if maths:
        f.maths(label, cx - 18, y + 26, size + 4, INK, anchor='end')
    else:
        f.words(cx - 18, y + 25, label, size, INK, anchor='end')
    return cells(f, cx, y, values, fill, signed=signed, cw=CW, oid=oid)


def heads(f, cx, y):
    for i, name in enumerate(('L1', 'L2', 'R1', 'R2')):
        f.words(cx + i * CW + CW / 2, y, name, 14, FROZEN_TEXT, anchor='middle')


def rule(f, cx, y):
    f.raw(Ln(cx - 30, y, cx + 4 * CW, y, INK, 1.6))


def mm1():
    f = Fig('mm1-one-step-three-calls', 1900, 900)
    title(f, 'One toy step: three calls, two differences, one sum',
          'Rose cells are what a label adds to the noise guess, d. Green would be a positive d; every d here is negative or zero.')
    # panel A: the three calls
    f.card(30, 120, 620, 690, FAINT, FAINT_LINE, 1.2, 14, oid='pa')
    f.words(52, 160, '1. Three calls on the same 4 pixels', 20, INK, 700)
    cx = 380
    heads(f, cx, 205)
    label_row(f, 0, 220, 'x at step 0 (noise)', S0['x'], cx)
    label_row(f, 0, 300, r'\epsilon_\varnothing = x', S0['eps_null'], cx, FIXED_FILL, maths=True)
    label_row(f, 0, 370, r'\epsilon_3 = x + u_3 + v_3 \odot x', S0['eps_3'], cx, maths=True)
    label_row(f, 0, 440, r'\epsilon_7 = x + u_7 + v_7 \odot x', S0['eps_7'], cx, maths=True)
    f.words(52, 545, 'The weights, set by hand:', 17, INK, 700)
    for i, (k, lab) in enumerate((('3', 'label 3 acts on the left pixels'), ('7', 'label 7 acts on the right pixels'))):
        f.words(52, 585 + i * 70, f"u{k} = {Wt['u'][k]},  v{k} = {Wt['v'][k]}", 17, INK)
        f.words(52, 612 + i * 70, lab, 15, FROZEN_TEXT)
    f.words(52, 740, 'No label adds nothing: u and v are zero,', 15, FROZEN_TEXT)
    f.words(52, 764, 'so the no-label guess is x itself.', 15, FROZEN_TEXT)
    go(f, (650, 405), (700, 405), name='a-b')
    # panel B: what each label adds
    f.card(700, 120, 520, 690, FAINT, FAINT_LINE, 1.2, 14, oid='pb')
    f.words(722, 160, '2. What each label adds', 20, INK, 700)
    f.maths(r'd(c) = \epsilon_c - \epsilon_\varnothing', 960, 200, 22, INK)
    cx = 960
    heads(f, cx, 250)
    label_row(f, 0, 265, r'\epsilon_3', S0['eps_3'], cx, maths=True)
    label_row(f, 0, 315, r'-\,\epsilon_\varnothing', [-v for v in S0['eps_null']], cx, FIXED_FILL, maths=True)
    rule(f, cx, 362)
    label_row(f, 0, 372, r'd(3)', S0['d_3'], cx, signed=True, maths=True)
    label_row(f, 0, 495, r'\epsilon_7', S0['eps_7'], cx, maths=True)
    label_row(f, 0, 545, r'-\,\epsilon_\varnothing', [-v for v in S0['eps_null']], cx, FIXED_FILL, maths=True)
    rule(f, cx, 592)
    label_row(f, 0, 602, r'd(7)', S0['d_7'], cx, signed=True, maths=True)
    f.words(722, 700, 'd(3) is non-zero only on the left,', 15, FROZEN_TEXT)
    f.words(722, 724, 'd(7) only on the right. The noise x', 15, FROZEN_TEXT)
    f.words(722, 748, 'cancels out of the u part of each d.', 15, FROZEN_TEXT)
    go(f, (1220, 405), (1270, 405), name='b-c')
    # panel C: the product
    f.card(1270, 120, 600, 690, FAINT, FAINT_LINE, 1.2, 14, oid='pc')
    f.words(1292, 160, "3. The product's noise guess, then one step", 20, INK, 700)
    f.maths(r'\epsilon_{PoE} = \epsilon_\varnothing + d(3) + d(7)', 1570, 200, 22, INK)
    cx = 1560
    heads(f, cx, 250)
    label_row(f, 0, 265, r'\epsilon_\varnothing', S0['eps_null'], cx, FIXED_FILL, maths=True)
    label_row(f, 0, 315, r'+\,d(3)', S0['d_3'], cx, signed=True, maths=True)
    label_row(f, 0, 365, r'+\,d(7)', S0['d_7'], cx, signed=True, maths=True)
    rule(f, cx, 412)
    label_row(f, 0, 422, r'\epsilon_{PoE}', S0['eps_poe'], cx, maths=True)
    label_row(f, 0, 520, 'x at step 0', S0['x'], cx)
    label_row(f, 0, 570, r'-\,\epsilon_{PoE}', [-v for v in S0['eps_poe']], cx, maths=True)
    rule(f, cx, 617)
    label_row(f, 0, 627, 'x at step 1', S0['x_next'], cx)
    f.words(1292, 720, 'Toy step: subtract the guess. The real sampler', 15, FROZEN_TEXT)
    f.words(1292, 744, 'is DDIM, 50 steps, which scales each part.', 15, FROZEN_TEXT)
    footer(f, TOY + ' Numbers: miniature_answer_key.json, steps[0].')
    f.write(HERE, 'One toy sampler step on 4 pixels: three calls, the two label differences d(3) and d(7), and their sum with the no-label guess.')


def mm2():
    f = Fig('mm2-two-steps-make-a-score-path', 1900, 820)
    title(f, 'Two steps stack into a score path of 8 numbers',
          "Each label's d is read at every step along the product's own path, then the steps are laid end to end.")
    cx0, cx1 = 330, 650
    for k, (s, cx) in enumerate(((S0, cx0), (S1, cx1))):
        f.words(cx + 100, 150, f'step {k}', 20, INK, 700, anchor='middle')
        heads(f, cx, 185)
        cells(f, cx, 200, s['x'], cw=CW, oid=f'x{k}')
        cells(f, cx, 290, s['d_3'], signed=True, cw=CW, oid=f'd3{k}')
        cells(f, cx, 380, s['d_7'], signed=True, cw=CW, oid=f'd7{k}')
    f.words(cx0 - 20, 225, 'x', 20, INK, anchor='end')
    f.maths(r'd(3)', cx0 - 20, 316, 22, INK, anchor='end')
    f.maths(r'd(7)', cx0 - 20, 406, 22, INK, anchor='end')
    go(f, (cx0 + 200, 218), (cx1, 218), name='step')
    f.words((cx0 + 200 + cx1) / 2, 268, r'x - ePoE', 14, FROZEN_TEXT, anchor='middle')
    f.words(120, 500, 'd(7) at step 1 is [0, 0, -1, 0], not [0, 0, -1, -2]: its v part reads x,', 16, FROZEN_TEXT)
    f.words(120, 524, 'and x has moved. A path depends on where the canvas has been.', 16, FROZEN_TEXT)
    # the stacked paths
    px = 1050
    f.words(px, 150, 'The score paths: step 0 then step 1', 20, INK, 700)
    for i, name in enumerate(('L1', 'L2', 'R1', 'R2', 'L1', 'L2', 'R1', 'R2')):
        f.words(px + 120 + i * CW + CW / 2, 185, name, 14, FROZEN_TEXT, anchor='middle')
    f.raw(Ln(px + 120 + 4 * CW, 192, px + 120 + 4 * CW, 560, FIXED, 1.0, '4 4'))
    P = K['score_paths']
    rows = (('3', r'S(3)', 200), ('7', r'S(7)', 290), ('poe', r'S(3)+S(7)', 380), ('joint', r'S(joint)', 490))
    for key, lab, y in rows:
        f.maths(lab, px + 105, y + 26, 22, INK, anchor='end')
        cells(f, px + 120, y, P[key], signed=True, cw=CW, oid=f'p-{key}')
    go(f, (cx1 + 205, 350), (px + 40, 350), color=SCORE_EDGE, name='stack')
    f.words((cx1 + 205 + px + 40) / 2, 340, 'stack', 15, SCORE_INK, anchor='middle')
    f.words(px, 600, 'S(3)+S(7) is the product\'s path: what PoE adds, step by step.', 16, INK)
    f.words(px, 628, 'S(joint) uses a pair label the real digit models do not have;', 16, SCORE_INK)
    f.words(px, 652, 'its d is d(3)+d(7)+r with r = ' + str(Wt['r_interaction']) + ', the toy interaction term.', 16, SCORE_INK)
    f.words(px, 700, 'Real size: 16 steps x 32 x 64 pixels = ' + format(N['score_path']['numbers'], ',') + ' numbers per path.', 16, FROZEN_TEXT)
    footer(f, TOY + ' Numbers: miniature_answer_key.json, steps and score_paths.')
    f.write(HERE, 'Two toy steps: the canvas moves, each label difference is read at each step, and the eight numbers are laid end to end into a score path.')


def mm3():
    f = Fig('mm3-the-encoder-and-the-contrastive-score', 1900, 940)
    title(f, 'A toy encoder turns each path into 2 numbers, and the matrix scores them against captions',
          'The encoder E is a hand-set integer matrix standing in for the planned score-path encoder, which step X4 has not designed.')
    E = Wt['encoder_E']
    ex = 300
    f.words(60, 165, 'Toy encoder E, 2 x 8', 19, INK, 700)
    planned_tag(f, 540, 158)
    for r, row in enumerate(E):
        cells(f, ex, 190 + r * 46, row, fill=mix(SCORE, 0.5), cw=44, ch=40, oid=f'E{r}')
    f.words(ex - 16, 217, 'row 1: minus the left d', 15, FROZEN_TEXT, anchor='end')
    f.words(ex - 16, 263, 'row 2: minus the right d', 15, FROZEN_TEXT, anchor='end')
    f.maths(r'z = E\,S', 480, 340, 24, INK)
    Z = K['embeddings']
    for i, (k, lab) in enumerate((('3', r'z(3)'), ('7', r'z(7)'), ('poe', r'z(3)+z(7)'))):
        y = 380 + i * 58
        f.maths(lab, 400, y + 27, 22, INK, anchor='end')
        cells(f, 420, y, Z[k], signed=False, fill=mix(SCORE, 0.35), cw=56, oid=f'z{k}')
    f.words(60, 590, 'E is linear, so the product\'s z is exactly z(3)+z(7).', 15, FROZEN_TEXT)
    f.words(60, 614, 'A real encoder with a nonlinearity would break that.', 15, FROZEN_TEXT)
    f.words(60, 680, 'Captions, hand-set 2-number text vectors:', 17, INK, 700)
    for i, (k, v) in enumerate(K['weights']['captions'].items()):
        f.words(60, 718 + i * 30, f'"{k}"  {v}', 16, INK)
    # the matrix
    C = K['contrastive']
    mx, my, cw, rh = 900, 250, 210, 92
    cols = ['"one 3"', '"one 7"', '"3 and 7, apart"']
    rows = [r'S(3)', r'S(7)', r'S(3)+S(7)']
    for j, c in enumerate(cols):
        f.strip(mx + j * cw + 45, my - 62, cw - 90, 22, TEXT, oid=f'ts{j}')
        f.words(mx + j * cw + cw / 2, my - 74, c, 16, INK, anchor='middle')
    for i, r in enumerate(rows):
        f.strip(mx - 60, my + i * rh + 18, 22, rh - 36, SCORE, vertical=True, oid=f'ss{i}')
        f.maths(r, mx - 74, my + i * rh + rh / 2 + 8, 20, INK, anchor='end')
        for j in range(3):
            x, y = mx + j * cw, my + i * rh
            f.rect(x, y, cw, rh, MATCH if i == j else '#f8f9fa', '#c9ced4', 0.8, 0, oid=f'c{i}{j}')
            f.words(x + cw / 2, y + 38, f"cos {C['cos'][i][j]:.2f}", 16, INK, anchor='middle')
            f.words(x + cw / 2, y + 66, f"/ tau = {C['logits'][i][j]:.1f}", 16, FROZEN_TEXT, anchor='middle')
    f.words(mx + 1.5 * cw, my + 3 * rh + 34, 'pale blue: each path with its own caption', 15, FROZEN_TEXT, anchor='middle')
    f.words(mx + 1.5 * cw, my + 3 * rh + 58, f"tau = {K['weights']['tau']}, W4's real temperature (w4_train.py TAU)", 15, FROZEN_TEXT, anchor='middle')
    lx = mx + 3 * cw + 60
    f.rect(lx, my, 290, 3 * rh, WHITE, INK, 1.0, 10, oid='lossbox')
    f.words(lx + 145, my + 34, 'Softmax along each row,', 15, INK, anchor='middle')
    f.words(lx + 145, my + 56, 'diagonal share:', 15, INK, anchor='middle')
    f.words(lx + 145, my + 84, ', '.join(f'{C["softmax_rows"][i][i]:.3f}' for i in range(3)), 16, INK, 600, anchor='middle')
    f.words(lx + 145, my + 124, 'down each column:', 15, INK, anchor='middle')
    f.words(lx + 145, my + 152, ', '.join(f'{C["softmax_cols"][i][i]:.3f}' for i in range(3)), 16, INK, 600, anchor='middle')
    f.words(lx + 145, my + 200, 'loss, mean of the two', 15, INK, anchor='middle')
    f.words(lx + 145, my + 224, f'cross-entropies: {C["loss"]:.4f}', 16, INK, 600, anchor='middle')
    go(f, (mx + 3 * cw, my + 1.5 * rh), (lx, my + 1.5 * rh), name='toloss')
    f.maths(r'\mathcal{L} = \tfrac12\left(\mathrm{CE}_{rows} + \mathrm{CE}_{cols}\right), \quad \mathrm{cell}_{ij} = \cos(z_i, e_j)/\tau', 1320, 660, 22, INK)
    f.words(900, 740, 'Training would move E so the pale blue cells win their rows and columns. Here E is set by hand,', 15, FROZEN_TEXT)
    f.words(900, 764, 'so the loss shows only where each number sits.', 15, FROZEN_TEXT)
    go(f, (545, 438), (650, 438), color=SCORE_EDGE, name='z-matrix')
    footer(f, TOY + ' Numbers: miniature_answer_key.json, embeddings and contrastive.')
    f.write(HERE, 'The toy encoder E maps each 8-number path to 2 numbers; a 3 by 3 matrix scores the three paths against three captions with W4\'s temperature, and the loss is read off the diagonal.')


def mm4():
    f = Fig('mm4-the-two-tests-in-miniature', 1900, 900)
    title(f, 'The two arithmetic tests, drawn in the toy\'s 2-number space',
          'Rose arrows are path embeddings z, lilac arrows are caption vectors e. Every vector is from the toy answer key.')
    ox, oy, sc = 330, 650, 95
    def P(v):
        return ox + v[0] * sc, oy - v[1] * sc
    f.raw(Ln(ox - 40, oy, ox + 6.2 * sc, oy, FIXED, 1.0) + Ln(ox, oy + 40, ox, oy - 5.0 * sc, FIXED, 1.0))
    f.words(ox + 6.2 * sc, oy + 30, 'number 1', 14, FROZEN_TEXT, anchor='end')
    f.words(ox - 14, oy - 5.0 * sc + 6, 'number 2', 14, FROZEN_TEXT, anchor='end')
    Z, Cap = K['embeddings'], K['weights']['captions']
    for k, v in Cap.items():
        x, y = P(v)
        f.raw(Ar(ox, oy, x, y, TEXT, 2.6, 12))
    for k in ('3', '7', 'poe'):
        x, y = P(Z[k])
        f.raw(Ar(ox, oy, x, y, SCORE_EDGE, 2.6, 12))
    labels = {('cap', 'one 3'): (12, 30), ('cap', 'one 7'): (-92, -8), ('cap', 'a 3 and a 7, apart'): (12, -6),
              ('cap', 'a 3 on top of a 7'): (12, 26), ('z', '3'): (12, 30), ('z', '7'): (-60, -12), ('z', 'poe'): (12, -8)}
    for (kind, k), (dx, dy) in labels.items():
        v = Cap[k] if kind == 'cap' else Z[k]
        x, y = P(v)
        if kind == 'cap':
            f.words(x + dx, y + dy, f'e("{k}")', 15, '#6f5bb0')
        else:
            f.words(x + dx, y + dy, {'3': 'z(3)', '7': 'z(7)', 'poe': 'z(3)+z(7)'}[k], 15, SCORE_INK, 600)
    # test 1
    T1, T2 = K['test_sum_lands_on_joint'], K['test_gap_predicts_term']
    bx = 1000
    f.card(bx, 130, 860, 330, FAINT, FAINT_LINE, 1.2, 14, oid='t1')
    f.words(bx + 22, 170, "Test 1: does the product's path land on the joint caption?", 19, INK, 700)
    f.words(bx + 22, 205, "Rank every caption by its cosine with z(3)+z(7):", 16, INK)
    for i, (k, c) in enumerate(T1['ranking']):
        f.words(bx + 40, 245 + i * 34, f'{i + 1}.  "{k}"', 16, INK, 600 if i == 0 else 400)
        f.words(bx + 560, 245 + i * 34, f'cos {c:.3f}', 16, INK, 600 if i == 0 else 400)
    f.words(bx + 22, 400, f'Top 1 is "{T1["top1"]}". In the real test this is one row of the ranking', 15, FROZEN_TEXT)
    f.words(bx + 22, 424, 'over unseen pairs, against a shuffled-label control.', 15, FROZEN_TEXT)
    # test 2
    f.card(bx, 490, 860, 330, FAINT, FAINT_LINE, 1.2, 14, oid='t2')
    f.words(bx + 22, 530, 'Test 2: does the path gap point the way the text gap does?', 19, INK, 700)
    f.maths(r'z(\mathrm{joint}) - z(3) - z(7) = ' + str(T2['path_gap']).replace(' ', ''), bx + 300, 585, 21, INK)
    f.maths(r'e(\mathrm{joint}) - e(3) - e(7) = ' + str(T2['text_gap']).replace(' ', ''), bx + 300, 640, 21, INK)
    f.words(bx + 22, 700, f"cosine between the two gaps: {T2['cos']:.3f}", 17, INK, 600)
    f.words(bx + 22, 740, 'z(joint) comes from the toy pair label, which the real models lack.', 15, SCORE_INK)
    f.words(bx + 22, 764, 'What a joint path means on the digit models is for step X4 to define.', 15, SCORE_INK)
    footer(f, TOY + ' Numbers: miniature_answer_key.json, test_sum_lands_on_joint and test_gap_predicts_term.')
    f.write(HERE, "The toy's 2-number space with caption arrows and path arrows; the product's path ranks the joint caption first, and the path gap and text gap are compared by cosine.")


FIGS = {'mm1': mm1, 'mm2': mm2, 'mm3': mm3, 'mm4': mm4}
if __name__ == '__main__':
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
