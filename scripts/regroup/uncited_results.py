"""List each project's result folders that nothing cites (ORGANOGRAM.md's rule).

A result is cited when a file outside it names it by a path through results/, or when the paper,
the learning tree, Obsidian or ~/.claude names it at all. Generated indexes, the archive, the
quarantine and the regroup's own inventories do not count.

    python3 scripts/regroup/uncited_results.py [out.json]
"""
import os, re, json, collections, sys, subprocess
ROOT = "/home/molef/PhD/concept-composition"
GROUPS = {l.strip() for l in open(f"{ROOT}/scripts/regroup/result-groups.txt") if l.strip()}
MARK = "index: written by scripts/regroup/results_index.py"
ROOTS = [r for r in ["1-lora-correction", "2-classifier-guidance", "3-understanding-compositionality",
         "4-inclusive-captions", "5-text-embedding-third-expert", ""] if os.path.isdir(os.path.join(ROOT, r, "artifacts/results"))]
results = []
for r in ROOTS:
    base = os.path.join(r, "artifacts/results") if r else "artifacts/results"
    stack = [base]
    while stack:
        d = stack.pop()
        for c in sorted(os.listdir(os.path.join(ROOT, d))):
            p = os.path.join(d, c)
            if not os.path.isdir(os.path.join(ROOT, p)) or c.startswith((".", "_")): continue
            if p in GROUPS: stack.append(p)
            else: results.append((r or "root", p))
SKIPD = {".git", "outputs", "wandb", "logs", "temp", "node_modules", "caches", "__pycache__", "_archive", "_quarantine", "archived"}
SKIPF = {"plans/prune-the-portfolio.md", "plans/regroup-into-project-folders-inventory.md", "RENAMES.md",
         "scripts/regroup/inventory.json", "scripts/regroup/moves.tsv", "scripts/regroup/later-moves.tsv"}
EXT = (".md", ".tex", ".html", ".py", ".sh", ".sbatch", ".yaml", ".yml", ".json", ".txt")
PATH = re.compile(r"results/([A-Za-z0-9_.%/-]+)")
cited = collections.defaultdict(set)
for dp, dns, fns in os.walk(ROOT):
    dns[:] = [d for d in dns if d not in SKIPD]
    for fn in fns:
        if not fn.endswith(EXT): continue
        fp = os.path.join(dp, fn); rel = os.path.relpath(fp, ROOT)
        if rel in SKIPF: continue
        try:
            if os.path.getsize(fp) > 3_000_000: continue
            t = open(fp, errors="ignore").read()
        except OSError: continue
        if MARK in t: continue
        for m in PATH.finditer(t):
            for comp in m.group(1).split("/")[:4]:
                cited[comp].add(rel)
OUTSIDE = ["/home/molef/goal-setting/learning", "/home/molef/Obsidian/", "/home/molef/.claude", os.path.join(ROOT, "paper")]
def named_outside(name):
    r = subprocess.run(["grep", "-rlIF", "--exclude-dir=.git", "--exclude-dir=temp", "--exclude-dir=projects",
                        "--exclude-dir=cadence-queue", "-m1", "--", name] + OUTSIDE, capture_output=True, text=True)
    return bool(r.stdout.strip())
out = collections.defaultdict(list)
for proj, p in results:
    name = os.path.basename(p)
    srcs = {s for s in cited.get(name, ()) if not s.startswith(p + "/")}
    if not srcs and not named_outside(name): out[proj].append(p)
print("result folders:", len(results))
for proj in sorted(out): print(f"{len(out[proj]):4d} uncited of {sum(1 for x in results if x[0]==proj):4d}  {proj}")
print("uncited total:", sum(len(v) for v in out.values()))
for proj in sorted(out):
    for p in out[proj]: print("  ", p)
if len(sys.argv) > 1: json.dump(out, open(sys.argv[1], "w"), indent=1)
