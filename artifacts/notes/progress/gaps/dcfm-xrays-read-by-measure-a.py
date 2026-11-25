# Draws dcfm-xrays-read-by-measure-a.png from DCFM's Figure 8 (arXiv 2606.10153, p. 8).
# Run from this folder: ~/.claude/venvs/figures/bin/python dcfm-xrays-read-by-measure-a.py
# Needs pdfimages (poppler) and the DCFM PDF at ../targets/pdfs/2606.10153.pdf.
import os, subprocess, tempfile
from PIL import Image
tmp = tempfile.mkdtemp()
subprocess.run(["pdfimages", "-f", "8", "-l", "8", "-png", "../targets/pdfs/2606.10153.pdf", os.path.join(tmp, "img")], check=True)
fig8 = Image.open(os.path.join(tmp, "img-000.png")).convert("RGB")   # 1761 x 1007, Figure 8
for name, box in {"a_clean.png": (440, 517, 860, 937), "b_hazy.png": (20, 517, 440, 937), "c_ribs.png": (897, 517, 1316, 937)}.items():
    fig8.crop(box).resize((630, 630)).save(os.path.join(tmp, name))
os.chdir(tmp)
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcfm-xrays-read-by-measure-a.png"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
tiles=[("a_clean.png","DCFM-A, Atelectasis + Nodule","anatomy readable: measure A can run"),
       ("b_hazy.png","DCFM-A, Effusion + Mass","heart edge not found: measure C has no border to read"),
       ("c_ribs.png","DCFM-B, Effusion + Mass","anatomy broken: measure A should fail it")]
fig,axes=plt.subplots(1,3,figsize=(15,6.0))
G="#1a7f37"; R="#c0392b"
def note(ax,txt,xy,xytext,c):
    ax.annotate(txt,xy=xy,xytext=xytext,color="white",fontsize=9,fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3",fc=c,ec="none",alpha=0.92),
        arrowprops=dict(arrowstyle="->",color=c,lw=2.2))
for ax,(f,title,verdict) in zip(axes,tiles):
    ax.imshow(Image.open(f),cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title,fontsize=11)
    ax.text(315,655,verdict,ha="center",va="top",fontsize=10,color=G if f.startswith("a") else R,fontweight="bold")
a,b,c=axes
note(a,"heart's left edge: visible",(395,300),(420,120),G)
note(a,"diaphragm domes",(470,385),(470,470),G)
a.annotate("",xy=(25,365),xytext=(590,365),arrowprops=dict(arrowstyle="<->",color="#f1c40f",lw=1.8))
a.annotate("",xy=(190,330),xytext=(415,330),arrowprops=dict(arrowstyle="<->",color="#f39c12",lw=1.8))
a.text(300,560,"heart width ÷ chest width:\nmeasure A checks it falls in\nthe range of real images",ha="center",fontsize=8.5,color="black",bbox=dict(fc="#fff6d5",ec="#f39c12"))
note(b,"no heart edge found\n(effusion, or haze everywhere?)",(400,380),(330,520),R)
note(b,"straight brightness step\nacross the image",(560,213),(330,80),R)
note(c,"ribs drawn below\nthe diaphragm",(480,450),(330,560),R)
note(c,"spine bends into\nthe abdomen",(300,520),(40,600),R)
note(c,"no heart outline",(330,300),(80,120),R)
fig.suptitle("What Paper 1's X-ray measures would read on DCFM's own generated X-rays (DCFM Figure 8, p. 8)",fontsize=12.5)
fig.text(0.5,0.012,"Marks drawn by eye by a reader without radiology training; no segmenter or classifier was run, and no number is read here.\n"
         "Measures A (anatomy kept) and C (the heart's left edge) are from Loop 4's evaluation plan. Images: arXiv 2606.10153, Figure 8, generated, not patients.",
         ha="center",fontsize=8.5,color="#444")
plt.tight_layout(rect=(0,0.07,1,0.95))
plt.savefig(OUT,dpi=120)
