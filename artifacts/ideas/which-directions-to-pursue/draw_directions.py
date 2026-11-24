import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = [("Placement for never-met experts\n(Paper 1: 310 of 320 on digits; ties paste)", 0.62, 0.86, "#1971c2"),
     ("Why never-met experts fail\n(Paper 1 analysis: 0 of 320 without placement)", 0.80, 0.56, "#1971c2"),
     ("Interaction-term analysis\n(LoRA paper: causal, early, small)", 0.86, 0.71, "#1971c2"),
     ("Identity reward LoRA\n(Paper 2: score the finished picture)", 0.30, 0.80, "#e8590c"),
     ("Regions name each body\n(Paper 2: M9, 8 of 8 but damaged)", 0.40, 0.62, "#e8590c"),
     ("Identity checker push\n(Paper 2: reads 0.70 at best early)", 0.16, 0.66, "#e8590c"),
     ("Sampler corrector\n(257 to 306 with placement; SDXL null)", 0.66, 0.32, "#2b8a3e"),
     ("Two-body push on CO3\n(7 against CO3's 11 of 96)", 0.80, 0.16, "#2b8a3e"),
     ("Free judges, seed ranking\n(baselines, no training)", 0.55, 0.08, "#2b8a3e"),
     ("Score path as third space\n(closed: no new information)", 0.14, 0.12, "#868e96")]
fig, ax = plt.subplots(figsize=(11, 8))
for x0, y0, w, h, c, t in [(0,0.5,0.5,0.5,"#fff4e6","Worth one decisive test"),(0.5,0.5,0.5,0.5,"#eaf1ff","Build on these"),
                           (0,0,0.5,0.5,"#f8f9fa","Park"),(0.5,0,0.5,0.5,"#ebfbee","Keep as tools or baselines")]:
    ax.add_patch(plt.Rectangle((x0,y0),w,h,color=c,zorder=0)); ax.text(x0+0.015,y0+h-0.03,t,fontsize=12,weight="bold",va="top",color="#444")
for lab,x,y,c in P:
    ax.scatter(x,y,s=90,color=c,zorder=3); ax.text(x+0.015,y,lab,fontsize=9,va="center",zorder=4)
ax.set_xlim(0,1.32); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("evidence so far that it works  →", fontsize=11); ax.set_ylabel("what a win would add to a paper  →", fontsize=11)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.set_title("Where each direction stands (placements are a judgement from the records, not a measurement)", fontsize=11.5)
fig.tight_layout(); fig.savefig("directions-evidence-and-payoff.png", dpi=140); fig.savefig("directions-evidence-and-payoff.svg")  # run from this folder
