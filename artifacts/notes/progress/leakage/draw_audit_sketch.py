"""Z5 draft audit on one picture: one dot per issue in 05-design-audit.md, by loop and by whether fixing it could change a verdict."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = {  # issue number: (loop, verdict at risk, fundamental?)
 1:(8,"yes",0),2:(8,"no",0),3:(8,"possibly",1),4:(8,"no",1),5:(8,"no",0),6:(8,"possibly",0),7:(8,"possibly",0),8:(8,"possibly",0),
 9:(8,"possibly",0),10:(8,"possibly",0),11:(8,"possibly",0),12:(8,"no",0),13:(8,"possibly",0),
 14:(9,"no",1),15:(9,"no",0),16:(9,"possibly",0),17:(9,"no",0),18:(9,"possibly",0),19:(9,"yes",0),20:(9,"no",0),21:(9,"no",0),22:(9,"possibly",1),
 23:(10,"possibly",0),24:(10,"unclear",0),25:(10,"no",0),26:(10,"possibly",0),27:(10,"unclear",0),28:(10,"unclear",0),29:(10,"yes",1),30:(10,"possibly",0)}
reruns = {8: "1. where the digit sampler settles\nits digits (0.04 GPU-h, measured)", 11: "2. an untrained rule against\nthe identity checker (about 0.1)",
          30: "3. the readout checker on CO3's path,\nsteps 0 to 30 (about 1.2)", 29: "4. each animal's own region from its\nd_t map on CO3 (under 0.1)"}
yes_note = {1: "Loop 8: 'removing one checker' null\nread with the old identity checker", 19: "Loop 9: binding on CO3 never scored\n(7 and 15 pictures, under the minimum)",
            29: "Loop 10: block-wise shared space\nkept open, never tested"}
L = {8: "Loop 8\nabstract checkers on digits", 9: "Loop 9\nsentence scores", 10: "Loop 10\nthe score path"}
V = ["yes", "possibly", "no", "unclear"]; VL = {"yes": "yes", "possibly": "possibly", "no": "no", "unclear": "unclear\nin the draft"}
col = {"yes": "#c92a2a", "possibly": "#e8590c", "no": "#868e96", "unclear": "#adb5bd"}
fig, ax = plt.subplots(figsize=(15, 8.2))
X = {8: 0, 9: 1.35, 10: 2.7}; Y = {v: 3 - i for i, v in enumerate(V)}
for v in V:
    ax.axhspan(Y[v] - 0.42, Y[v] + 0.42, color={"yes": "#fff5f5", "possibly": "#fff4e6", "no": "#f8f9fa", "unclear": "#ffffff"}[v], zorder=0)
for loop in L:
    for v in V:
        ids = [i for i, r in rows.items() if r[0] == loop and r[1] == v]
        for j, i in enumerate(ids):
            x = X[loop] - 0.5 + (j % 7) * 0.16; y = Y[v] + (0.17 if j < 7 else -0.17) * (1 if len(ids) > 7 else 0)
            ax.scatter(x, y, s=520, color=col[v], edgecolor="#212529" if rows[i][2] else "white", linewidth=2.6 if rows[i][2] else 1, zorder=3)
            ax.text(x, y, str(i), ha="center", va="center", color="white", fontsize=10, weight="bold", zorder=4)
            if i in reruns:
                ax.scatter(x, y, s=1150, facecolor="none", edgecolor="#1c7ed6", linewidth=2.2, zorder=2)
for i, txt in yes_note.items():
    ax.text(X[rows[i][0]] - 0.5 + 0.2, Y["yes"], txt, fontsize=8.6, va="center")
ax.set_xticks([X[k] for k in L]); ax.set_xticklabels([L[k] for k in L], fontsize=11)
ax.set_yticks([Y[v] for v in V]); ax.set_yticklabels([VL[v] for v in V], fontsize=11)
ax.set_ylabel("could fixing it change the loop's recorded verdict?", fontsize=11)
ax.set_xlim(-0.75, 3.45); ax.set_ylim(-0.5, 3.5)
for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
side = "\n\n".join(reruns.values())
fig.text(0.80, 0.86, "blue ring: one of the 4 reruns proposed,\nabout 1.4 GPU-hours in all\n\n" + side + "\n\nthick black edge: a fundamental issue\n(the question itself needs changing)",
         fontsize=9.5, va="top", color="#1c3d6e")
fig.suptitle("Z5 draft: 30 design issues in Loops 8, 9 and 10, one dot each (numbers are the rows of 05-design-audit.md)", fontsize=12.5, x=0.02, ha="left")
plt.subplots_adjust(left=0.13, right=0.78, top=0.9, bottom=0.12)
plt.savefig("audit-sketch.png", dpi=120)
