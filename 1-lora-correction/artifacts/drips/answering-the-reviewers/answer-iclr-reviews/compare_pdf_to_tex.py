import re, sys, difflib
pdf_pages = open(sys.argv[1]).read().split("\f")
tex = open(sys.argv[2]).read()
def clean_tex(s):
    s = re.sub(r"(?m)%.*$", "", s)
    s = re.sub(r"\\begin\{(figure|table|align|equation)\*?\}.*?\\end\{\1\*?\}", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\\(cite[pt]?|ref|eqref|label|includegraphics)(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}~]", " ", s)
    return re.sub(r"\s+", " ", s)
def clean_pdf(s):
    s = re.sub(r"(?m)^\s*\d{3}\s*$", " ", s)
    s = s.replace("-\n", "").replace("\n", " ")
    s = re.sub(r"Under review as a conference paper at ICLR 2027", " ", s)
    return re.sub(r"\s+", " ", s)
def sents(s):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", s) if len(x.split()) >= 8]
norm = lambda x: re.sub(r"[^a-z0-9 ]", "", x.lower())
tex_s = sents(clean_tex(tex)); tex_n = [norm(x) for x in tex_s]
pages = [int(p) for p in sys.argv[3].split(",")]
total = diff = missing = 0
for p in pages:
    for s in sents(clean_pdf(pdf_pages[p-1])):
        n = norm(s); total += 1
        best = max(tex_n, key=lambda t: difflib.SequenceMatcher(None, n, t).quick_ratio() and difflib.SequenceMatcher(None, n, t).ratio(), default="")
        r = difflib.SequenceMatcher(None, n, best).ratio()
        if r < 0.9:
            kind = "absent" if r < 0.6 else "changed"
            diff += 1; missing += kind == "absent"
            print(f"p{p}\t{kind}\t{r:.2f}\t{s[:150]}")
print(f"TOTAL pdf sentences {total}, differing {diff}, of which absent from tex {missing}", file=sys.stderr)
