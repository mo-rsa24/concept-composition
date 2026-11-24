"""Recorded search for Loop 12 step Z1: papers on leakage between subjects, and papers that measure which pairs leak and why."""
import urllib.request, urllib.parse, re, time, json
ids = ['2403.16990','2505.21488','2312.06059','2307.10864','2212.05032']
queries = [
 'abs:"semantic leakage" AND abs:diffusion',
 'abs:"attribute leakage" AND abs:diffusion',
 'abs:"subject leakage" OR abs:"feature leakage" AND abs:"text-to-image"',
 'abs:"similar subjects" AND abs:"text-to-image"',
 'abs:"subject mixing" AND abs:diffusion',
 'abs:"multi-subject" AND abs:"text-to-image" AND abs:leakage',
 'abs:"cross-attention" AND abs:"attribute binding" AND abs:"text encoder"',
 'abs:"text encoder" AND abs:"information" AND abs:"mix" AND abs:"text-to-image"',
 'abs:"semantically similar" AND abs:"objects" AND abs:"text-to-image" AND abs:"attention"',
 'abs:"self-attention" AND abs:"leakage" AND abs:"diffusion"',
 'abs:"catastrophic neglect" AND abs:"similar"',
 'abs:"entity" AND abs:"leakage" AND abs:"text-to-image"',
]
out = {'ids': [], 'queries': []}
url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'id_list': ','.join(ids), 'max_results': 10})
t = urllib.request.urlopen(url, timeout=60).read().decode()
for i, ti in re.findall(r'<entry>.*?<id>http://arxiv.org/abs/(.*?)</id>.*?<title>(.*?)</title>', t, re.S):
    out['ids'].append({'id': i, 'title': ' '.join(ti.split())}); print('ID', i, '|', ' '.join(ti.split()))
time.sleep(3.5)
for q in queries:
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'max_results': 25, 'sortBy': 'relevance'})
    t = urllib.request.urlopen(url, timeout=60).read().decode()
    tot = re.search(r'<opensearch:totalResults>(\d+)', t).group(1)
    ents = re.findall(r'<entry>.*?<id>http://arxiv.org/abs/(.*?)</id>.*?<title>(.*?)</title>.*?<summary>(.*?)</summary>', t, re.S)
    out['queries'].append({'query': q, 'total': int(tot), 'hits': [{'id': i, 'title': ' '.join(ti.split()), 'abs': ' '.join(s.split())[:700]} for i, ti, s in ents]})
    print(f'## {q}  total={tot}')
    for i, ti, s in ents: print('  ', i, '|', ' '.join(ti.split()))
    time.sleep(3.5)
json.dump(out, open('arxiv_search.json', 'w'), indent=1)
