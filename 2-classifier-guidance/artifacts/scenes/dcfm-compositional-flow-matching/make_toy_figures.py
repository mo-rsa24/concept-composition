"""Level 2: the figures of DCFM's own 2-D toy (Figs. 2 to 3, p. 4), drawn from toy_arrays.npz and dcfm_numbers.json.

  03a-each-silo-alone           (i)   each silo's exact expert, sampled with no label
  03b-sum-rule-missing-cell     (ii)  the sum rule asked for the missing (2, 1): within a silo, across silos, four-cell grid
  03c-interaction-term          (iii) I_t = joint minus sum rule, as a field at three times, and its mean size against t
  03d-penalty-on-one-silo       (iv)  the local penalty (Eq. 11) on trained networks: I_t and the (2, 1) samples as lambda grows
  03e-penalty-across-silos      (iv)  non-IID: Stage I with the local penalty against one DCFM-A round (Eq. 14)

Run make_dcfm_numbers.py first. Usage: ~/.claude/venvs/figures/bin/python make_toy_figures.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toy_dcfm as T  # noqa: E402
from figcheck import check  # noqa: E402

N = json.load(open(os.path.join(HERE, "dcfm_numbers.json")))
A = np.load(os.path.join(HERE, "toy_arrays.npz"))
TOY = N["toy"]
K = N["colour_key"]
BLUE, GREEN, RED, ORANGE, GREY = (K["inputs"], K["experts_and_hidden_units"], K["outputs"],
                                  K["interaction_term_penalty_gradient_push"], K["fixed_frozen_off"])
INK = "#222222"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c7cdd3", "axes.labelcolor": INK, "xtick.color": "#555555",
                     "ytick.color": "#555555"})


def cells(ax, present, label=True):
    for c, (x, y) in T.CENTRE.items():
        here = c in present
        p = Circle((x, y), 0.5, fill=False, ec=GREEN if here else GREY, lw=1.0, ls="-" if here else "--", zorder=1)
        p.set_gid("free")
        ax.add_patch(p)
        if label:
            ax.text(x, y + (0.62 if y > 0 else -0.62), f"({c[0]},{c[1]})" + ("" if here else " missing"), ha="center", va="center",
                    fontsize=8, color=INK if here else GREY, gid="free")


def frame(ax, title):
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_title(title, fontsize=9.5, color=INK)


def scatter(ax, xs, colour, n=900, s=3, alpha=0.45):
    ax.scatter(xs[:n, 0], xs[:n, 1], s=s, color=colour, alpha=alpha, lw=0, zorder=2)


def share(d, key="2,1"):
    return 100 * d[key]


def save(fig, name):
    check(fig, name)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(HERE, "figures", f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote figures/" + name)


def fig_a():
    fig, axs = plt.subplots(1, 4, figsize=(15.5, 4.6), dpi=130)
    sp = TOY["splits"]
    for ax, (split, a) in zip(axs, [("iid", 1), ("iid", 2), ("non_iid", 1), ("non_iid", 2)]):
        present = [tuple(int(v) for v in c.split(",")) for c in sp[split][str(a)]]
        cells(ax, present)
        scatter(ax, A[f"i_{split}_silo{a}"], RED)
        sh = TOY["i_silo_alone"][f"{split}_silo{a}"]
        frame(ax, f"{'IID' if split == 'iid' else 'non-IID'} split, silo {a}: holds {', '.join('(' + c + ')' for c in sp[split][str(a)])}\n"
              f"{100 - share(sh, 'none'):.1f}% of samples within 0.5 of a held cell")
        ax.set_xlabel("first coordinate (attribute 1: left -1, right +1)")
    axs[0].set_ylabel("second coordinate (attribute 2: top +1, bottom -1)")
    fig.suptitle("(i) Each silo's expert alone, sampled with no label: it reproduces only the cells its silo holds "
                 "(1,500 samples, 900 drawn; exact velocities, Euler 200 steps, t from 0 noise to 1 data)", fontsize=10.5, y=1.02)
    save(fig, "03a-each-silo-alone")


def fig_b():
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.9), dpi=130)
    rows = [("iid_within_silo1", T.SPLITS["iid"][1], "one IID silo, its own right, top and unconditional"),
            ("non_iid_across_silos", [(1, 1), (1, 2), (2, 2)], "non-IID: right from silo 1, top from silo 2,\nunconditional from silo 1"),
            ("four_cell_reference", list(T.CENTRE), "reference: all four cells in the data")]
    for ax, (key, present, title) in zip(axs, rows):
        cells(ax, present)
        scatter(ax, A[f"ii_{key}"], RED)
        d = TOY["ii_sum_rule_missing"][key]
        frame(ax, title)
        ax.text(0, -1.85, f"{share(d['shares']):.1f}% land within 0.5 of the (2,1) centre; mean ({d['mean'][0]:.2f}, {d['mean'][1]:.2f})",
                ha="center", fontsize=8, color=INK)
        ax.set_xlabel("first coordinate (left -1, right +1)")
    axs[0].set_ylabel("second coordinate (top +1, bottom -1)")
    fig.suptitle("(ii) The sum rule asked for the missing (2,1): v(x_t | right) + v(x_t | top) - v(x_t), exact velocities "
                 "(1,500 samples, 900 drawn)", fontsize=10.5, y=1.02)
    save(fig, "03b-sum-rule-missing-cell")


def fig_c():
    fig = plt.figure(figsize=(17.5, 5.0), dpi=130)
    fig.subplots_adjust(wspace=0.38)
    G = A["grid"]
    for k, t in enumerate((0.3, 0.6, 0.9)):
        ax = fig.add_subplot(1, 4, k + 1)
        cells(ax, T.SPLITS["iid"][1], label=False)
        I = A[f"iii_I_iid_t{t}"]
        mag = np.linalg.norm(I, axis=1)
        q = ax.quiver(G[:, 0], G[:, 1], I[:, 0], I[:, 1], color=ORANGE, angles="xy", scale_units="xy",
                      scale=max(mag.max(), 1e-9) / 0.45, width=0.004, zorder=3)
        q.set_gid("free")
        frame(ax, f"t = {t} (0 noise, 1 data)\nlongest arrow |I_t| = {mag.max():.2f}")
        ax.set_xlabel("first coordinate")
    fig.axes[0].set_ylabel("second coordinate")
    ax = fig.add_subplot(1, 4, 4)
    ts = A["ts"]
    ax.plot(ts, A["iii_curve_exact_iid"], color=ORANGE, lw=1.8, label="one silo, cells (1,1), (1,2), (2,2)")
    ax.plot(ts, A["iii_curve_exact_four"], color=GREY, lw=1.4, ls="--", label="all four cells (zero)")
    ax.set_xlabel("t (0 noise, 1 data)")
    ax.set_ylabel("mean |I_t| on the held cells' paths\n(velocity units)")
    ax.set_title("size of I_t against time", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("(iii) The interaction term I_t = v(x_t | 1,1) - [v(x_t | left) + v(x_t | top) - v(x_t)], exact velocities of one IID "
                 "silo, label (1,1); arrows scaled per panel", fontsize=10.5, y=1.02)
    save(fig, "03c-interaction-term")


def fig_d():
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.9), dpi=130)
    lams = ["0.0", "0.3", "1.0", "3.0", "10.0"]
    d = TOY["iv_iid_local_penalty"]
    ts = A["ts"]
    shades = ["#f3c9a1", "#eaa766", "#d9822b", "#b3600f", "#7a3d05"]
    ax = axs[0]
    for lam, col in zip(lams, shades):
        ax.plot(ts, A[f"iv_iid_lam{float(lam)}_curve"], color=col, lw=1.6, label=f"$\\lambda_{{CI}}$ = {float(lam):g}: mean {d[lam]['I_t_mean_norm']:.3f}")
    ax.plot(ts, A["iii_curve_exact_iid"], color=GREY, lw=1.2, ls="--", label=f"exact expert: mean {TOY['iii_I_t_mean_norm']['exact_iid_silo']:.3f}")
    ax.set_xlabel("t (0 noise, 1 data)")
    ax.set_ylabel("mean |I_t| on the held cells' paths (velocity units)")
    ax.set_title("the penalty shrinks I_t where it is measured", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")
    ax = axs[1]
    cells(ax, T.SPLITS["iid"][1])
    for lam, col in (("0.0", "#f3c9a1"), ("1.0", "#d9822b"), ("10.0", "#7a3d05")):
        xs = A[f"iv_iid_lam{float(lam)}_missing"]
        ax.scatter(xs[:600, 0], xs[:600, 1], s=4, color=col, alpha=0.6, lw=0, zorder=2,
                   label=f"$\\lambda_{{CI}}$ = {float(lam):g}: mean ({d[lam]['missing_mean'][0]:.2f}, {d[lam]['missing_mean'][1]:.2f})")
    frame(ax, "where the sum rule for (2,1) sends samples")
    ax.legend(frameon=False, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    ax.set_xlabel("first coordinate (left -1, right +1)")
    ax.set_ylabel("second coordinate (top +1, bottom -1)")
    ax = axs[2]
    lv = [float(l) for l in lams]
    w1 = [100 * d[l]["missing_within_1_of_cell"] for l in lams]
    w05 = [share(d[l]["missing_shares_r0.5"]) for l in lams]
    ax.plot(lv, w1, "o-", color=RED, lw=1.6, label="within 1.0 of the (2,1) centre")
    ax.plot(lv, w05, "s-", color=RED, lw=1.0, alpha=0.6, label="within 0.5 of the (2,1) centre, the cell itself")
    long = TOY["iv_iid_local_penalty_12000_steps"]
    ax.plot([1.0], [100 * long["missing_within_1_of_cell"]], marker="D", color=INK, ls="none",
            label=f"$\\lambda_{{CI}}$ = 1 trained 3x longer: {100 * long['missing_within_1_of_cell']:.1f}%")
    ax.set_xscale("symlog", linthresh=0.3)
    ax.set_xticks(lv)
    ax.set_xticklabels([f"{v:g}" for v in lv])
    ax.set_ylim(-3, 103)
    ax.set_xlabel("$\\lambda_{CI}$ (symmetric log scale)")
    ax.set_ylabel("% of the 1,500 (2,1) samples")
    ax.set_title("how close they get", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    fig.suptitle("(iv) The local penalty on one IID silo (Eq. 11), numpy MLP 15-128-128-128-2, 4,000 Adam steps per run, "
                 "one seed: I_t falls, the (2,1) samples move toward the cell but stop short", fontsize=10.5, y=1.02)
    save(fig, "03d-penalty-on-one-silo")


def fig_e():
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.9), dpi=130)
    d = TOY["iv_non_iid"]
    present = [(1, 1), (1, 2), (2, 2)]
    for ax, key, title in ((axs[0], "stage1_local_penalty", "Stage I, local penalty, then composed across silos"),
                           (axs[1], "dcfm_a_one_round", "after one DCFM-A round (peer penalty, Eq. 14)")):
        cells(ax, present)
        scatter(ax, A[f"iv_non_iid_{key}_missing"], RED, n=900)
        frame(ax, title)
        r = d[key]
        ax.text(0, -1.85, f"{share(r['missing_shares_r0.5']):.1f}% within 0.5 of the (2,1) centre, {100 * r['missing_within_1_of_cell']:.1f}% within 1;"
                f" mean ({r['missing_mean'][0]:.2f}, {r['missing_mean'][1]:.2f})", ha="center", fontsize=8, color=INK)
        ax.set_xlabel("first coordinate (left -1, right +1)")
    axs[0].set_ylabel("second coordinate (top +1, bottom -1)")
    ax = axs[2]
    ts = A["ts"]
    for key, col, lab in (("stage1_local_penalty", "#eaa766", "Stage I"), ("dcfm_a_one_round", "#7a3d05", "DCFM-A, one round")):
        ax.plot(ts, A[f"iv_non_iid_{key}_curve_routed"], color=col, lw=1.8,
                label=f"{lab}, routed across silos: mean {d[key]['I_t_mean_norm_routed']:.3f}")
        ax.plot(ts, A[f"iv_non_iid_{key}_curve_local"], color=col, lw=1.2, ls="--",
                label=f"{lab}, own model only: mean {d[key]['I_t_mean_norm_local']:.3f}")
    ax.set_xlabel("t (0 noise, 1 data)")
    ax.set_ylabel("mean |I_t| on each silo's own labels\n(velocity units)")
    ax.set_title("routed: averaged over the routings Eq. 12 allows", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")
    fig.suptitle("(iv) Non-IID split, silo 1 holds (1,2), (2,2); silo 2 holds (1,1), (1,2); sum rule for (2,1) = silo 1's right + silo 2's top "
                 "- silo 1's unconditional ($\\lambda_{CI}$ = 1, one seed)", fontsize=10.5, y=1.02)
    save(fig, "03e-penalty-across-silos")


if __name__ == "__main__":
    for f in (fig_a, fig_b, fig_c, fig_d, fig_e):
        f()
