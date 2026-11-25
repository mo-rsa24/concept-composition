"""The single source of every value the DCFM ladder prints.

Writes dcfm_numbers.json (printed numbers with their page, values computed from the paper's formulas, worked numbers
for the maths page, and the toy's results) and toy_arrays.npz (the toy's fields and samples, read by the figures).

Usage:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/.claude/venvs/figures/bin/python make_dcfm_numbers.py [--printed-only]
The toy part trains small numpy MLPs on the CPU (about ten minutes in all); --printed-only skips it and keeps any toy
section already in the JSON; --non-iid-only reruns only the non-IID part of (iv). A later run must leave every value already in the JSON unchanged (seeds are fixed).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toy_dcfm as T  # noqa: E402

OUT = os.path.join(HERE, "dcfm_numbers.json")
NPZ = os.path.join(HERE, "toy_arrays.npz")
PDF = "artifacts/notes/progress/targets/pdfs/2606.10153.pdf"


def printed():
    table1 = {  # p. 7, Table 1: FID on Colored MNIST, o = known (some client saw it), * = novel (no client saw it)
        "FedFlow": [9.41, 20.83, 15.02, 20.02],
        "FedFlow+L": [12.37, 18.65, 15.81, 17.12],
        "DDM": [8.19, 25.19, 7.46, 17.98],
        "DDM+L": [9.03, 16.38, 7.99, 18.84],
        "DFD": [8.81, 29.71, 7.02, 38.27],
        "DFD+L": [9.58, 19.13, 8.17, 31.36],
        "DCFM-A": [8.53, 11.41, 7.33, 9.29],
        "DCFM-B": [9.32, 12.24, 8.49, 9.15],
    }
    table2 = {  # p. 7, Table 2: success rate % on cube-single-play, mean +- spread over 3 groups
        "FedFlow+L": {"iid": [[58.33, 2.62], [39.67, 4.11]], "non_iid": [[27.67, 7.76], [9.0, 3.74]]},
        "DDM+L": {"iid": [[53.33, 2.05], [38.33, 5.73]], "non_iid": [[67.67, 2.87], [29.67, 2.87]]},
        "DFD+L": {"iid": [[58.0, 2.16], [40.0, 4.9]], "non_iid": [[68.67, 1.25], [18.33, 5.44]]},
        "DCFM-A": {"iid": [[57.67, 2.62], [56.33, 5.31]], "non_iid": [[68.33, 2.87], [53.0, 2.94]]},
        "DCFM-B": {"iid": [[54.0, 3.74], [53.33, 2.36]], "non_iid": [[65.67, 2.49], [54.67, 2.49]]},
    }
    c_min_mnist = (1 + (10 - 1) + (10 - 1)) / 100
    c_min_robot = (1 + (4 - 1) + (4 - 1)) / 16
    return {
        "source": {"paper": "Morshed and Boddeti, Compositional Generative Modeling from Decentralized Data, arXiv 2606.10153, ICML 2026",
                   "pdf": PDF, "pages": 13, "code": "not public (checked 2026-11-25)",
                   "coind_code": "github.com/sachit3022/compositional-generation (CoInD's public code, the independence penalty DCFM carries; cited, not run)"},
        "colored_mnist": {
            "page": "p. 6 to 7", "image_size": "28 x 28 x 3", "k_attributes": 2, "digits": 10, "colours": 10,
            "combinations": 100, "clients_n": 10, "total_coverage_C": 0.5, "dirichlet_alpha_non_iid": 0.1,
            "c_min_eq2": round(c_min_mnist, 4), "c_min_working": "(1 + (10 - 1) + (10 - 1)) / 100 = 19 / 100",
            "c_min_source": "computed from Eq. 2 (p. 3); the paper does not print this value",
            "novel_examples_fig5": ["(0, yellow)", "(4, red)"],
            "network": "not stated", "fid_sample_count": "not stated",
            "table1_columns": ["IID FID known", "IID FID novel", "Non-IID FID known", "Non-IID FID novel"],
            "table1_page": "p. 7, Table 1 (lower is better)", "table1": table1,
        },
        "robot_planning": {
            "page": "p. 7 to 8", "dataset": "OGBench cube-single-play", "clients_n": 2, "combinations": 16,
            "total_coverage_C": 0.75, "held_out": ["(TL, BR)", "(TR, BL)", "(BR, TL)", "(BL, TR)"],
            "c_min_eq2": round(c_min_robot, 4), "c_min_working": "(1 + (4 - 1) + (4 - 1)) / 16 = 7 / 16",
            "c_min_source": "computed from Eq. 2 (p. 3); the paper does not print this value",
            "network": "not stated", "table2_page": "p. 7, Table 2 (success rate %, [known, novel], each [mean, spread])",
            "table2": table2,
        },
        "chest_xray": {
            "page": "p. 8 to 9", "dataset": "NIH Chest X-ray 14", "clients_n": 4, "image_size": "256 x 256",
            "model": "a latent flow matching model on the latent space of a pretrained autoencoder (which autoencoder: not stated)",
            "labels_k": 14, "feasible_combinations": 54, "feasible_rule": "frequency at least 0.1% of the data",
            "share_with_3_or_fewer": "about 99%", "max_co_occurring": 9, "count_with_9": 2,
            "classifier": "ResNet-18", "results": "plot points only (Fig. 9), no table",
        },
        "cost": {
            "page": "p. 9, Tables 3 and 4",
            "table3_communication_per_client": {"FL": "2T", "DDM": "N - 1", "DFD": "N - 1", "DCFM (DCFM-A)": "RN", "DCFM-B": "2"},
            "dcfm_a_rounds_R": "1 to 3 (about 2 on average)",
            "dcfm_b_exchange": "every peer uploads its local model to the server and receives the distilled model back",
            "table4_per_step": {
                "Baselines (FL, DDM, DFD)": {"P": "n", "fwd": "1 x B", "bwd": "1 x B", "avg_novel_fid": 25.33},
                "Baselines + local CI": {"P": "n", "fwd": "~4 x B", "bwd": "~4 x B", "avg_novel_fid": 20.24},
                "DCFM-A": {"P": "n", "fwd": "~4 x B", "bwd": "~2 x B", "avg_novel_fid": 10.35},
                "DCFM-B": {"P": "1", "fwd": "~5 x B", "bwd": "~4 x B", "avg_novel_fid": 10.69},
            },
        },
        "not_stated": [
            "the network architecture for Colored MNIST and for robot planning",
            "which pretrained autoencoder the X-ray latent flow uses",
            "the ODE solver and its step count",
            "the penalty weight lambda (Eq. 11, 14, 17)",
            "the masking weights pi_full, pi_marg, pi_uncond (Eq. 8)",
            "the guidance weights w_i used in the experiments (Eq. 5); Eq. 7 and Eq. 10 set w_i = 1",
            "how the routing vector r is drawn (Eq. 12)",
            "the number of samples behind each FID",
            "the code (not public, checked 2026-11-25)",
        ],
        "printing_notes": [
            "Eq. 10 and Eq. 12 (p. 5) print the penalty inside plain square brackets with no norm; Eq. 7 (p. 3) and Eq. 16 (p. 6) print a squared norm. The ladder reads all four as the squared distance.",
            "Eq. 5 and Eq. 6 write v(x_t, y); A.2 writes v_t(x | y): the same velocity, two spellings.",
        ],
    }


def worked():
    """Small worked numbers for the maths page, all from the toy's exact fields (cell std 0.15, four-cell grid unless named)."""
    four = T.exact_expert(list(T.CENTRE))
    iid = T.exact_expert(T.SPLITS["iid"][1])
    x = np.array([[0.2, 0.3]])
    t = 0.5
    v_u, v_r, v_top = four(x, t)[0], four(x, t, y1=2)[0], four(x, t, y2=1)[0]
    v_hat = v_r + v_top - v_u
    v_true = T.exact_velocity(x, t, [(2, 1)])[0]
    # A.2 at t = 0.5 on the straight path: alpha_t = t, sigma_t = 1 - t, a_t = 1/t, b_t = (1 - t)/t.
    a_t, b_t = 1 / t, (1 - t) / t
    mus = np.array([T.CENTRE[(2, 1)]])
    score = -(x[0] - t * mus[0]) / T.tau2(t)  # one cell: grad log N(x; t mu, tau^2 I)
    v_from_score = a_t * x[0] + b_t * score
    # Eq. 7 on the IID silo's exact expert at the same point, for the seen label (1, 1).
    j11 = iid(x, t, 1, 1)[0]
    s11 = iid(x, t, 1, None)[0] + iid(x, t, None, 1)[0] - iid(x, t)[0]
    # Eq. 9: one training pair.
    x0, x1, tt = np.array([0.4, -1.2]), np.array([-1.0, 1.0]), 0.25
    xt = (1 - tt) * x0 + tt * x1
    # Eq. 12: non-IID, silo 1 training label (2, 2) with r = (2, 1, 2), exact experts.
    e1, e2 = T.exact_expert(T.SPLITS["non_iid"][1]), T.exact_expert(T.SPLITS["non_iid"][2])
    u2, m_right_1, m_bottom_2 = e2(x, t)[0], e1(x, t, y1=2)[0], e2(x, t, y2=2)[0]
    z = u2 + (m_right_1 - u2) + (m_bottom_2 - u2)
    j22 = e1(x, t, 2, 2)[0]
    # Eq. 3: the integral of the four-cell field's (2, 1) velocity from one starting noise.
    rng = np.random.default_rng(7)
    x_start = rng.standard_normal((1, 2))
    xs = x_start.copy()
    for k in range(200):
        xs = xs + (1 / 200) * T.exact_velocity(xs, k / 200, [(2, 1)])
    r = lambda v: [round(float(a), 3) for a in v]
    # Eq. 4 in noise form: on the straight path eps = E[x_0 | x_t] = x_t - t v (from x_t = t x_1 + (1 - t) x_0).
    eps = lambda v: x[0] - t * v
    # Eq. 15: the teacher (silo 1, non-IID, exact) carries one noise to t = 0.5 under label (2, 2); its velocity there is the target.
    xh = np.array([[0.4, -1.2]])
    for k in range(100):
        xh = xh + (0.5 / 100) * e1(xh, k * 0.5 / 100, 2, 2)
    teacher_v = e1(xh, 0.5, 2, 2)[0]
    i_sq = float(np.sum((j11 - s11) ** 2))
    # The overview's inset: IID silo, label (1, 2) (bottom-left), where joint, marginals and unconditional all differ.
    in_u, in_l, in_b, in_j = iid(x, t)[0], iid(x, t, y1=1)[0], iid(x, t, y2=2)[0], iid(x, t, 1, 2)[0]
    # Eq. 5 at the same point on the IID silo (three cells), asked for the missing (2, 1).
    ii_u, ii_r, ii_t = iid(x, t)[0], iid(x, t, y1=2)[0], iid(x, t, y2=1)[0]
    eq5_iid = {"v_uncond": r(ii_u), "v_right": r(ii_r), "v_top": r(ii_t), "v_hat_sum_rule": r(ii_r + ii_t - ii_u),
               "v_true_cell_2_1": r(v_true), "gap_norm": round(float(np.linalg.norm(ii_r + ii_t - ii_u - v_true)), 4)}
    inset = {"v_uncond": r(in_u), "v_left": r(in_l), "v_bottom": r(in_b), "joint": r(in_j),
             "sum_rule": r(in_l + in_b - in_u), "I_t": r(in_j - (in_l + in_b - in_u)),
             "I_t_norm": round(float(np.linalg.norm(in_j - (in_l + in_b - in_u))), 4)}
    ab = {f"{tt:.2f}": {"a_t": round(1 / tt, 4), "b_t": round((1 - tt) / tt, 4)} for tt in (0.25, 0.5, 0.75)}
    # Eq. 1 at the point: posterior of each label from the cells' densities at x_t (equal cell weights).
    def post(cells):
        T2 = T.tau2(t)
        dens = {c: np.exp(-np.sum((x[0] - t * np.array(T.CENTRE[c])) ** 2) / (2 * T2)) for c in cells}
        z = sum(dens.values())
        p = {c: d / z for c, d in dens.items()}
        p_r = sum(v for c, v in p.items() if c[0] == 2)
        p_top = sum(v for c, v in p.items() if c[1] == 1)
        return {"p_y1_right": round(p_r, 4), "p_y2_top": round(p_top, 4), "product": round(p_r * p_top, 4),
                "p_joint_2_1": round(p.get((2, 1), 0.0), 4)}
    eq1 = {"four_cells": post(list(T.CENTRE)), "iid_silo_three_cells": post(T.SPLITS["iid"][1])}
    # A.1: scores from velocities by Eq. 28 on the straight path at t = 0.5: s = (v - a_t x) / b_t.
    sc_ = lambda v: (v - a_t * x[0]) / b_t
    a1 = {"s_uncond": r(sc_(v_u)), "s_right": r(sc_(v_r)), "s_top": r(sc_(v_top)),
          "s_sum_rule": r(sc_(v_r) + sc_(v_top) - sc_(v_u)), "s_joint_2_1": r(score),
          "a_x_plus_b_s_hat": r(a_t * x[0] + b_t * (sc_(v_r) + sc_(v_top) - sc_(v_u)))}
    out_extra = {
        "eq1_posteriors_at_point": eq1,
        "a1_scores_four_cells": a1,
        "eq4_noise_form": {"rule": "eps = x_t - t v on the straight path", "eps_uncond": r(eps(v_u)), "eps_right": r(eps(v_r)),
                           "eps_top": r(eps(v_top)), "eps_hat": r(eps(v_r) + eps(v_top) - eps(v_u)), "x_minus_t_v_hat": r(eps(v_hat))},
        "eq11_one_point": {"I_t_norm_squared": round(i_sq, 4), "lambda": T.TOY_CHOICES["lambda_ci"],
                           "added_to_loss": round(T.TOY_CHOICES["lambda_ci"] * i_sq, 4)},
        "eq13_r_2_1_2_for_silo1": {"r0": "silo 2, frozen", "r1": "silo 1, trained", "r2": "silo 2, frozen"},
        "eq14_one_point": {"residual_norm_squared": round(float(np.sum((j22 - z) ** 2)), 4)},
        "eq15_teacher": {"x0": [0.4, -1.2], "label": "(2, 2)", "teacher": "silo 1 non-IID, exact", "x_hat_t_at_0.5": r(xh[0]),
                         "teacher_velocity_target": r(teacher_v)},
        "a2_ab_table_straight_path": ab,
        "inset_iid_label_1_2": inset,
        "eq5_iid_silo_missing_2_1": eq5_iid,
    }
    return {**out_extra, **{
        "point_x": r(x[0]), "t": t, "direction": "t runs from 0 (noise) to 1 (data)",
        "eq5_four_cells": {"v_uncond": r(v_u), "v_right": r(v_r), "v_top": r(v_top), "v_hat_sum_rule": r(v_hat),
                           "v_true_cell_2_1": r(v_true), "gap_norm": round(float(np.linalg.norm(v_hat - v_true)), 4)},
        "eq6_a2_straight_path": {"alpha_t": t, "sigma_t": 1 - t, "a_t": a_t, "b_t": b_t, "cell": "(2, 1) alone",
                                 "score": r(score), "a_x_plus_b_score": r(v_from_score),
                                 "velocity_direct": r(T.exact_velocity(x, t, [(2, 1)])[0])},
        "eq7_iid_silo_label_1_1": {"joint": r(j11), "sum_rule": r(s11), "I_t": r(j11 - s11),
                                   "I_t_norm": round(float(np.linalg.norm(j11 - s11)), 4)},
        "eq8_pi_equal": {"P_full": round(1 / 3, 4), "P_each_marginal": round(1 / 6, 4), "P_uncond": round(1 / 3, 4)},
        "eq9_pair": {"x0": r(x0), "x1": r(x1), "t": tt, "x_t": r(xt), "target_x1_minus_x0": r(x1 - x0)},
        "eq12_non_iid_silo1_label_2_2_r_2_1_2": {"v_uncond_silo2_frozen": r(u2), "v_right_silo1": r(m_right_1),
                                                 "v_bottom_silo2_frozen": r(m_bottom_2), "z": r(z), "joint_silo1": r(j22), "residual": r(j22 - z),
                                                 "residual_norm": round(float(np.linalg.norm(j22 - z)), 4)},
        "eq3_one_path": {"x0": r(x_start[0]), "end_after_200_euler_steps": r(xs[0]), "cell_centre": [1.0, 1.0]},
    }}


def toy():
    """Parts (i) to (iv) of the toy. Returns the JSON section and the arrays for the figures."""
    out, arr = {"choices": T.TOY_CHOICES, "cells": {f"{k[0]},{k[1]}": v for k, v in T.CENTRE.items()},
                "missing": "2,1", "splits": {s: {str(a): [f"{c[0]},{c[1]}" for c in cs] for a, cs in d.items()}
                                              for s, d in T.SPLITS.items()}}, {}
    grid = np.linspace(-2, 2, 17)
    GX, GY = np.meshgrid(grid, grid)
    G = np.stack([GX.ravel(), GY.ravel()], 1)
    arr["grid"] = G
    shares = lambda xs: {k: round(v, 4) for k, v in T.nearest_cell_shares(xs).items()}

    # (i) each silo's expert alone: exact unconditional samples.
    for split in ("iid", "non_iid"):
        for a, cells in T.SPLITS[split].items():
            e = T.exact_expert(cells)
            xs = T.ode_sample(lambda x, t: e(x, t), n=1500, seed=10 + a)
            arr[f"i_{split}_silo{a}"] = xs
            out.setdefault("i_silo_alone", {})[f"{split}_silo{a}"] = shares(xs)

    # (ii) the sum rule of exact marginal velocities for the missing (2, 1).
    e_iid = T.exact_expert(T.SPLITS["iid"][1])
    e1, e2 = T.exact_expert(T.SPLITS["non_iid"][1]), T.exact_expert(T.SPLITS["non_iid"][2])
    e4 = T.exact_expert(list(T.CENTRE))
    rules = {
        "iid_within_silo1": T.sum_rule(lambda x, t: e_iid(x, t), lambda x, t: e_iid(x, t, y1=2), lambda x, t: e_iid(x, t, y2=1)),
        "non_iid_across_silos": T.sum_rule(lambda x, t: e1(x, t), lambda x, t: e1(x, t, y1=2), lambda x, t: e2(x, t, y2=1)),
        "four_cell_reference": T.sum_rule(lambda x, t: e4(x, t), lambda x, t: e4(x, t, y1=2), lambda x, t: e4(x, t, y2=1)),
    }
    for name, f in rules.items():
        xs = T.ode_sample(f, n=1500, seed=20)
        arr[f"ii_{name}"] = xs
        out.setdefault("ii_sum_rule_missing", {})[name] = {"shares": shares(xs), "mean": [round(float(v), 3) for v in xs.mean(0)]}
    rng = np.random.default_rng(0)
    xp = rng.standard_normal((4000, 2)) * 1.5
    gap = max(float(np.abs(e4(xp, t, 1, 1) + e4(xp, t, 2, 2) - e4(xp, t, 1, 2) - T.exact_velocity(xp, t, [(2, 1)])).max())
              for t in (0.1, 0.3, 0.5, 0.7, 0.9))
    out["parallelogram_max_abs_gap"] = gap
    out["parallelogram_note"] = ("on the four-cell grid, v(x|1,1) + v(x|2,2) - v(x|1,2) equals v(x|2,1) at every x and t "
                                 "(max abs gap over 4,000 points and 5 times), because each cell's velocity is affine in its centre")

    # (iii) I_t as a field, exact, IID silo, label (1, 1); and the curve of its mean size against t.
    for t in (0.3, 0.6, 0.9):
        j = e_iid(G, t, 1, 1)
        s = e_iid(G, t, 1, None) + e_iid(G, t, None, 1) - e_iid(G, t)
        arr[f"iii_I_iid_t{t}"] = j - s
    vj = lambda x, t, c: e_iid(x, t, *c)
    vs = lambda x, t, c: e_iid(x, t, c[0], None) + e_iid(x, t, None, c[1]) - e_iid(x, t)
    ts, cur = T.interaction_norm(vj, vs, T.SPLITS["iid"][1])
    arr["ts"], arr["iii_curve_exact_iid"] = ts, cur
    vj4 = lambda x, t, c: e4(x, t, *c)
    vs4 = lambda x, t, c: e4(x, t, c[0], None) + e4(x, t, None, c[1]) - e4(x, t)
    _, cur4 = T.interaction_norm(vj4, vs4, list(T.CENTRE))
    arr["iii_curve_exact_four"] = cur4
    out["iii_I_t_mean_norm"] = {"exact_iid_silo": round(float(cur.mean()), 4), "four_cell_reference": float(cur4.max()),
                                "at_t": {f"{t:.2f}": round(float(c), 4) for t, c in zip(ts, cur)},
                                "unit": "velocity units (data coordinates per unit t), averaged over x_t on the seen cells' paths and over t in 0.05..0.95"}

    # (iv) what the penalty does: trained numpy MLPs.
    nets = {}
    for lam in (0.0, 0.3, 1.0, 3.0, 10.0):
        t0 = time.time()
        net = T.train(T.SPLITS["iid"][1], lam=lam, mode="local" if lam else "none", seed=0)
        nets[("iid", lam)] = net
        f = lambda x, t, n=net: n.velocity(x, t, 2, None) + n.velocity(x, t, None, 1) - n.velocity(x, t)
        xs = T.ode_sample(f, n=1500, seed=20)
        vj = lambda x, t, c, n=net: n.velocity(x, t, *c)
        vs = lambda x, t, c, n=net: n.velocity(x, t, c[0], None) + n.velocity(x, t, None, c[1]) - n.velocity(x, t)
        _, c = T.interaction_norm(vj, vs, T.SPLITS["iid"][1])
        arr[f"iv_iid_lam{lam}_missing"], arr[f"iv_iid_lam{lam}_curve"] = xs, c
        tt = 0.6
        arr[f"iv_iid_lam{lam}_I_t0.6"] = vj(G, tt, (1, 1)) - vs(G, tt, (1, 1))
        seen = T.ode_sample(lambda x, t, n=net: n.velocity(x, t), n=1500, seed=11)
        arr[f"iv_iid_lam{lam}_uncond"] = seen
        out.setdefault("iv_iid_local_penalty", {})[str(lam)] = {
            "I_t_mean_norm": round(float(c.mean()), 4), "missing_shares_r0.5": shares(xs),
            "missing_within_1_of_cell": round(T.fraction_near(xs, (1, 1), 1.0), 4),
            "missing_mean": [round(float(v), 3) for v in xs.mean(0)],
            "uncond_shares_r0.5": shares(seen), "train_seconds": round(time.time() - t0, 1)}
        print("iid", lam, out["iv_iid_local_penalty"][str(lam)], flush=True)

    # Sensitivity: the same lambda = 1 run trained three times longer.
    t0 = time.time()
    net = T.train(T.SPLITS["iid"][1], lam=1.0, mode="local", seed=0, steps=12000)
    f = lambda x, t, n=net: n.velocity(x, t, 2, None) + n.velocity(x, t, None, 1) - n.velocity(x, t)
    xs = T.ode_sample(f, n=1500, seed=20)
    vj = lambda x, t, c, n=net: n.velocity(x, t, *c)
    vs = lambda x, t, c, n=net: n.velocity(x, t, c[0], None) + n.velocity(x, t, None, c[1]) - n.velocity(x, t)
    _, c = T.interaction_norm(vj, vs, T.SPLITS["iid"][1])
    arr["iv_iid_lam1_12000_missing"], arr["iv_iid_lam1_12000_curve"] = xs, c
    out["iv_iid_local_penalty_12000_steps"] = {
        "lambda": 1.0, "steps": 12000, "I_t_mean_norm": round(float(c.mean()), 4), "missing_shares_r0.5": shares(xs),
        "missing_within_1_of_cell": round(T.fraction_near(xs, (1, 1), 1.0), 4),
        "missing_mean": [round(float(v), 3) for v in xs.mean(0)], "train_seconds": round(time.time() - t0, 1)}
    print("iid 12000", out["iv_iid_local_penalty_12000_steps"], flush=True)

    non_iid(out, arr, shares)
    return out, arr


def non_iid(out, arr, shares):
    """Non-IID: Stage I with the local penalty (Eq. 11), then one DCFM-A round (Eq. 14) against frozen Stage I peers.

    Two gaps are measured on each silo's own labels. "routed": |J - z| averaged over every routing r that Eq. 12 allows (each
    r_i from a silo whose data hold y_i, r_0 either silo, at least one entry the silo itself), the gap the peer penalty trains.
    "local": |J - (M_1 + M_2 - U)| with all four calls from the silo's own model, the gap the local penalty trains.
    """
    import itertools
    lam = T.TOY_CHOICES["lambda_ci"]
    sp = T.SPLITS["non_iid"]
    seen_by = {(i, v): [a for a, cs in sp.items() if any(c[i] == v for c in cs)] for i in (0, 1) for v in (1, 2)}
    stage1 = {a: T.train(sp[a], lam=lam, mode="local", seed=a) for a in (1, 2)}
    dcfm_a = {a: T.train(sp[a], lam=lam, mode="peer", seed=10 + a, init=stage1[a], peers={b: stage1[b] for b in (1, 2) if b != a}, me=a)
              for a in (1, 2)}
    for name, m in (("stage1_local_penalty", stage1), ("dcfm_a_one_round", dcfm_a)):
        f = lambda x, t, m=m: m[1].velocity(x, t, 2, None) + m[2].velocity(x, t, None, 1) - m[1].velocity(x, t)
        xs = T.ode_sample(f, n=1500, seed=20)
        arr[f"iv_non_iid_{name}_missing"] = xs
        routed, local = [], []
        for a in (1, 2):
            for c in sp[a]:
                vj = lambda x, t, cc, a=a: m[a].velocity(x, t, *cc)
                for r0, r1, r2 in itertools.product((1, 2), seen_by[(0, c[0])], seen_by[(1, c[1])]):
                    if a not in (r0, r1, r2):
                        continue
                    vs = lambda x, t, cc, r=(r0, r1, r2): (m[r[1]].velocity(x, t, cc[0], None) + m[r[2]].velocity(x, t, None, cc[1])
                                                           - m[r[0]].velocity(x, t))
                    routed.append(T.interaction_norm(vj, vs, [c])[1])
                vl = lambda x, t, cc, a=a: m[a].velocity(x, t, cc[0], None) + m[a].velocity(x, t, None, cc[1]) - m[a].velocity(x, t)
                local.append(T.interaction_norm(vj, vl, [c])[1])
        cr, cl = np.mean(routed, 0), np.mean(local, 0)
        arr[f"iv_non_iid_{name}_curve_routed"], arr[f"iv_non_iid_{name}_curve_local"] = cr, cl
        out.setdefault("iv_non_iid", {})[name] = {
            "I_t_mean_norm_routed": round(float(cr.mean()), 4), "I_t_mean_norm_local": round(float(cl.mean()), 4),
            "routings_averaged": len(routed), "missing_shares_r0.5": shares(xs),
            "missing_within_1_of_cell": round(T.fraction_near(xs, (1, 1), 1.0), 4),
            "missing_mean": [round(float(v), 3) for v in xs.mean(0)]}
        print("non_iid", name, out["iv_non_iid"][name], flush=True)


def main():
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    data["printed"] = printed()
    data["worked"] = worked()
    if "--non-iid-only" in sys.argv:
        arr = dict(np.load(NPZ))
        for k in [k for k in arr if k.startswith("iv_non_iid_")]:
            del arr[k]
        data["toy"].pop("iv_non_iid", None)
        non_iid(data["toy"], arr, lambda xs: {k: round(v, 4) for k, v in T.nearest_cell_shares(xs).items()})
        np.savez_compressed(NPZ, **arr)
    elif "--printed-only" not in sys.argv:
        data["toy"], arr = toy()
        np.savez_compressed(NPZ, **arr)
    data["colour_key"] = {"inputs": "#1f5fd0", "experts_and_hidden_units": "#1d6b3a", "outputs": "#c0262d",
                          "interaction_term_penalty_gradient_push": "#c8680a", "fixed_frozen_off": "#6b7a88"}
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
