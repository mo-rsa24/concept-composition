import re,json,sys
b=open(sys.argv[2],'rb').read(); h=re.match(rb'P6\s+(\d+)\s+(\d+)\s+255\s',b); WW=int(h[1]); o=h.end(); s=200/72
class I:
    def getpixel(self,p):
        i=o+3*(p[1]*WW+p[0]); return (b[i],b[i+1],b[i+2])
im=I()
W=[(float(a),float(b),float(c),float(d),t) for a,b,c,d,t in re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]+)</word>',open(sys.argv[1]).read())]
M=['CFG++','CFG-Zero*','APG','TCFG','SAG','SEG','OSEG','PAG','no-CFG','CFG']; K=['GE','DPG','Obj','Txt','Rsn']
out={}
for m in M:
    r=[w for w in W if w[4]==m and w[0]<130 and w[1]>230]
    y=r[0][1]; c=sorted([w for w in W if abs(w[1]-y)<2.5 and re.fullmatch(r'\d+\.\d+',w[4])])
    assert len(c)==10,(m,len(c))
    for j,w in enumerate(c):
        px=im.getpixel((int((w[0]-2.2)*s),int((w[1]+w[3])/2*s))); R,G,B=px
        cl='gain' if G>R+8 else 'loss' if R>G+8 else 'none'
        out[f"{m}|{'SD' if j<5 else 'FLUX'}|{K[j%5]}"]=[float(w[4]),cl,px]
json.dump(out,open('cells.json','w'))
mg={'SD':dict(zip(K,[.027,.76,.019,.044,.005])),'FLUX':dict(zip(K,[.028,.85,.018,.042,.004]))}
n=0
for k,v in out.items():
    m,a,b=k.split('|')
    if m in('CFG','no-CFG'):continue
    d=round(v[0]-out[f'CFG|{a}|{b}'][0],3); exp='gain' if d>mg[a][b] else 'loss' if d<-mg[a][b] else 'none'
    if exp!=v[1]: n+=1; print('differs',k,'value',v[0],'diff',d,'margin',mg[a][b],'printed',v[1],v[2])
print('cells',len(out),'differs',n); print('CFG colours',[out[f'CFG|SD|{x}'][1] for x in K])
