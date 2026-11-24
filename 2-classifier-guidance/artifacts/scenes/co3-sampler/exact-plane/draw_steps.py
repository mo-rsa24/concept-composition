"""What CO3 sees and does at each of its six steps, on one pair: the decoded clean-picture guesses of the empty, joint
and each animal's prompt at the step's last pass, the two distances and two weights CO3 computed there, and the
finished pictures with CO3 on and off.

  python3 draw_steps.py a_cheetah__x__a_cougar      (full size)
  python3 draw_steps.py an_eagle__x__a_fox --small  (smaller version)

Every number is read from exact_plane_numbers.json; the pictures from the run's folder recorded there.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from co3_figkit import KEY, Canvas  # noqa: E402

J = json.loads((HERE / "exact_plane_numbers.json").read_text())
pair = sys.argv[1]
small = "--small" in sys.argv
P = J["pairs"][pair]
folder = Path(J["source"]["local"]) / pair / f"seed-{J['source']['seed']}"
A, B = P["single_prompts"]
NAMES = {"a_cheetah__x__a_cougar": "cheetah x cougar", "an_eagle__x__a_fox": "eagle x fox"}
LABEL = {"borrowed": "the W2d call: borrowed (\"" + J["source"]["borrowed_note"] + "\")",
         "own_parts": "the W2d call: each animal keeps its own parts"}

S = 1.25 if small else 1.75     # thumbnail side
G = 0.12                        # gap between thumbnails
X0 = 1.75                       # left edge of the first thumbnail column
TOP = 2.6                      # space above the grid for the title and headers
cols = [("empty", "empty\nprompt", "fixed"), ("joint", "joint\nprompt", "input"),
        ("a", f"\"{A}\"\nalone", "concept1"), ("b", f"\"{B}\"\nalone", "concept2")]
TXT_W = 4.6 if small else 5.0
FIN = 3.0 if small else 4.1     # finished picture side
W = X0 + 4 * S + 3 * G + 0.3 + TXT_W + 0.4 + FIN + 0.5
H = TOP + 0.55 + 6 * (S + G) + 0.6
H = max(H, TOP + 0.55 + 2 * FIN + 1.6)
c = Canvas(W, H, scale=0.78 if small else 0.82)

title = f"CO3's six steps on \"{P['prompt']}\": each prompt's guess of the finished picture, and CO3's push"
c.text(0.3, H - 0.4, title, size=13 if small else 15, weight="bold", ha="left")
c.text(0.3, H - 0.95,
       "Each picture is the model's guess of the finished picture from the latent at the step's last pass: "
       r"$(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon^{\lambda,c})/\sqrt{\bar\alpha_t}$, decoded. "
       f"Seed {J['source']['seed']}, {LABEL[P['label']]}.",
       size=9 if small else 10, ha="left", color="#3b4552")
c.text(0.3, H - 1.62,
       r"$d_k$: length of $\epsilon^{C}-\epsilon^{c_k}$ over the latent's 65,536 numbers (how far apart the joint prompt's and "
       r"the animal's noise predictions are)." + "\n" + r"$w_k$: the weight equation 15 gives that animal, always negative, "
       r"the two summing to $-1$; the closer animal gets the larger push away.",
       size=9 if small else 10, ha="left", color="#3b4552")

y_head = H - TOP - 0.25
for j, (_, head, kind) in enumerate(cols):
    cx = X0 + j * (S + G) + S / 2
    c.text(cx, y_head, head, size=8.5 if small else 9.5, color=KEY[kind], weight="bold")
tx = X0 + 4 * S + 3 * G + 0.3
c.text(tx, y_head, "what CO3 computed at this pass", size=8.5 if small else 9.5, ha="left", weight="bold")

for r, row in enumerate(P["steps"]):
    y = H - TOP - 0.55 - (r + 1) * (S + G) + G
    c.text(X0 - 0.15, y + S / 2, f"step {row['step']}\nt = {row['t']}", size=9 if small else 10, ha="right")
    for j, (key, _, kind) in enumerate(cols):
        c.image(folder / f"guess-step{row['step']}-last-{key}.png", X0 + j * (S + G), y, S, S, ec=KEY[kind])
    lp = row["last_pass"]
    verb = "resamples (Algorithm 2)" if row["kind"] == "resample" else "corrects (Algorithm 3)"
    c.text(tx, y + S * 0.80, f"{verb}, pass {row['passes']} of {row['passes']}", size=9 if small else 10, ha="left",
           color=KEY["method"], weight="bold")
    c.text(tx, y + S * 0.50, f"$d_1$ = {lp['dists'][0]:.2f} ({A})    $d_2$ = {lp['dists'][1]:.2f} ({B})",
           size=9 if small else 10, ha="left")
    c.text(tx, y + S * 0.20, f"$w_1$ = {lp['w'][0]:.2f} ({A})    $w_2$ = {lp['w'][1]:.2f} ({B})",
           size=9 if small else 10, ha="left", color=KEY["method"])

fx = tx + TXT_W + 0.4
fy_top = H - TOP - 0.55 - FIN
c.image(folder / "co3.png", fx, fy_top, FIN, FIN, ec=KEY["output"])
c.text(fx + FIN / 2, fy_top - 0.3, "finished picture, CO3 on", size=9.5 if small else 10.5, weight="bold", color=KEY["output"])
c.image(folder / "plain-joint.png", fx, fy_top - FIN - 0.85, FIN, FIN, ec=KEY["fixed"])
c.text(fx + FIN / 2, fy_top - FIN - 1.15, "same seed and prompt, CO3 off", size=9.5 if small else 10.5, weight="bold",
       color=KEY["fixed"])
c.text(fx + FIN / 2, y_head, "after all 50 steps", size=8.5 if small else 9.5, weight="bold")

stem = {"a_cheetah__x__a_cougar": "co3-steps-cheetah-cougar", "an_eagle__x__a_fox": "co3-steps-eagle-fox"}[pair]
c.save(f"exact-plane/{stem}")
