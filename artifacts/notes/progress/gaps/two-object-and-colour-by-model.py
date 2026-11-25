import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = [  # label, two-object, colour attribution, kind
 ("SDXL\n(GenEval p8)",0.74,0.23,"base"),
 ("SD3.5 Medium\n(Flow-GRPO p7)",0.78,0.52,"base"),
 ("FLUX.1 dev\n(Flow-GRPO p7)",0.81,0.45,"base"),
 ("FLUX.1 dev\n(ReNO p7)",0.85,0.48,"base"),
 ("SD3.5 Large\n(Flow-GRPO p7)",0.89,0.47,"base"),
 ("FLUX schnell + ReNO\nfrozen, noise search",0.90,0.56,"frozen"),
 ("SD3 8B + DPO\nnot released (SD3 p11)",0.94,0.60,"trained"),
 ("SD3.5 M + Flow-GRPO\nRL on GenEval-style",0.99,0.86,"trained"),
]
col={"base":"#9aa5b1","frozen":"#3b82c4","trained":"#d9822b"}
fig,axes=plt.subplots(1,2,figsize=(13,5.2),sharey=True)
for ax,idx,title in [(axes[0],1,"GenEval two objects"),(axes[1],2,"GenEval colour attribution")]:
    y=range(len(rows))
    ax.barh(y,[r[idx] for r in rows],color=[col[r[3]] for r in rows])
    for i,r in enumerate(rows): ax.text(r[idx]+0.01,i,f"{r[idx]:.2f}",va="center",fontsize=9)
    ax.set_xlim(0,1.08); ax.set_title(title+"\nfraction of images judged correct, 0 to 1",fontsize=11)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows],fontsize=8.5); ax.set_ylim(-1.4,7.6)
    ax.axvline(0.7626,ls="--",c="k",lw=1) if idx==1 else ax.axvline(0.2050,ls="--",c="k",lw=1)
axes[0].text(0.75,-1.0,"dashed: this project's SDXL joint prompt, 0.7626 on 396 images",fontsize=8,ha="right")
axes[1].text(0.22,-1.0,"dashed: this project's SDXL joint prompt, 0.2050 on 400 images",fontsize=8)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=col[k],label=l) for k,l in [("base","model as released"),("frozen","frozen weights, test-time noise search"),("trained","weights trained further")]],loc="lower center",ncol=3,fontsize=9)
fig.suptitle("Two different objects are nearly solved on newer models; binding a colour to each is not",fontsize=12)
plt.tight_layout(rect=(0,0.06,1,0.95))
plt.savefig("/home/molef/PhD/concept-composition/artifacts/notes/progress/gaps/two-object-and-colour-by-model.png",dpi=130)
