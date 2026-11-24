import urllib.request, urllib.parse, re, time, json, sys
queries = [
 'abs:"noise prediction" AND abs:contrastive AND abs:encoder AND abs:diffusion',
 'abs:"denoising trajectory" AND abs:contrastive',
 'abs:"diffusion trajectory" AND abs:embedding AND abs:encoder',
 'abs:"sampling trajectory" AND abs:representation AND abs:diffusion AND abs:encoder',
 'abs:"score function" AND abs:"representation learning" AND abs:encoder AND abs:diffusion',
 'abs:"noise predictions" AND abs:features AND abs:classifier AND abs:diffusion',
 'abs:"epsilon prediction" AND abs:representation AND abs:diffusion',
 'abs:"score field" AND abs:encoder AND abs:contrastive',
 'abs:"denoising trajectories" AND abs:features',
 'abs:"trajectory" AND abs:"text-to-image" AND abs:contrastive AND abs:"noise"  AND abs:embedding',
 'abs:"classifier-free guidance" AND abs:"difference" AND abs:"representation" AND abs:encoder',
 'abs:"diffusion classifier" AND abs:"noise" AND abs:"learned" AND abs:"embedding"',
]
out=[]
for q in queries:
    url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':q,'max_results':25,'sortBy':'relevance'})
    t=urllib.request.urlopen(url,timeout=60).read().decode()
    tot=re.search(r'<opensearch:totalResults>(\d+)',t).group(1)
    ents=re.findall(r'<entry>.*?<id>http://arxiv.org/abs/(.*?)</id>.*?<title>(.*?)</title>.*?<summary>(.*?)</summary>',t,re.S)
    out.append({'query':q,'total':int(tot),'hits':[{'id':i,'title':' '.join(ti.split()),'abs':' '.join(s.split())[:600]} for i,ti,s in ents]})
    print(f'## {q}  total={tot}')
    for i,ti,s in ents: print('  ',i,'|',' '.join(ti.split()))
    time.sleep(3.5)
json.dump(out,open('arxiv_search.json','w'),indent=1)
