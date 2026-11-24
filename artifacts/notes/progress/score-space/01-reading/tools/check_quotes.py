"""Check every quote in each row file against its stated PDF page. Usage: check_quotes.py <rows folder> <papers folder>."""
import json, re, sys, pathlib, unicodedata
ROWS, PAPERS = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"-\s*\n\s*", "", s)
    return re.sub(r"[^a-z0-9.%+]", "", s)
def pages(p):
    return {int(n): norm(t) for n, t in re.findall(r"=== PAGE (\d+) ===\n(.*?)(?=\n=== PAGE \d+ ===|\Z)", p.read_text(), re.S)}
tot = bad = 0
for f in sorted(ROWS.glob("*.json")):
    r = json.loads(f.read_text()); slug = f.stem
    a, b = pages(PAPERS/slug/"paged.txt"), pages(PAPERS/slug/"paged-layout.txt")
    for q in r.get("quotes", []):
        tot += 1; n = norm(q["text"]); pg = int(q["page"])
        if n and (n in a.get(pg, "") or n in b.get(pg, "")): continue
        if q.get("read_from_image"): print(f"IMAGE {slug} p{pg}: {q['text'][:80]}"); continue
        bad += 1; print(f"MISS  {slug} p{pg}: {q['text'][:100]}")
print(f"{tot} quotes, {bad} not found on their page")
