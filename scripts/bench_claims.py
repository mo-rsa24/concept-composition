"""Chunk ledger for Table 2's renders (compute job 13): many workers share the remaining images without
rendering any image twice, by claiming chunks with mkdir on /datasets.

A chunk is about 25 images of one method on one set. The joint prompt's chunks are `--shard k/N` of
scripts/review_bench_joint.py; PoE, ours and CO3 chunks are the k-th of N slices of geneval_two's pair
list (for CO3, of its cell list), taken the same way (every N-th from k), so a chunk names exactly the
images it owns and the renderers' own skip-if-exists behaviour does the rest.

Layout under CLAIMS (default /datasets/mmolefe/poe_repair_min/outputs/paper_review/claims):
    chunks.txt          one line per chunk: id method set k N images
    held/<id>/owner     a claim: mkdir held/<id> succeeding makes the caller the owner (host pid time)
    done/<id>           every image of the chunk exists
    released/<id>.*     a claim given back (stale, failed, or yielded), kept for the record
    skip.txt            chunk ids no worker may take, one per line with the reason (the pairs PoE and
                        ours refuse are put in one chunk per method and listed here)

A claim is stale when it is older than STALE_MIN minutes and no image of its chunk was written in the
last STALE_MIN minutes; `claim` releases stale claims before taking a new one.

    python bench_claims.py build                 write chunks.txt and skip.txt (refuses if chunks.txt exists; --force)
    python bench_claims.py claim --owner H:P --methods joint,poe     prints "id method set k N" or exits 1
    python bench_claims.py pairs <id>            "a|b" lines for a poe/ours chunk
    python bench_claims.py cells <id>            review_baseline_cell.py --cells lines for a co3 chunk
    python bench_claims.py missing <id>          image paths of the chunk not yet on disk
    python bench_claims.py finish <id>           mark done if every image exists (exit 0), else release (exit 1)
    python bench_claims.py release <id> <why>    give a claim back
    python bench_claims.py status                counts per method: done / claimed / waiting / skipped, images
"""
from __future__ import annotations

import argparse
import os
import re
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = Path(os.environ.get("BENCH", "/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers/bench13/scripts"))
sys.path.insert(0, str(BENCH))
from review_bench_sets import SETS, load, split  # noqa: E402

OUT = Path(os.environ.get("REVIEW_BASELINE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/paper_review"))
CLAIMS = Path(os.environ.get("CLAIMS", OUT / "claims"))
STALE_MIN = float(os.environ.get("STALE_MIN", "20"))
OURS_TAG = os.environ.get("OURS_TAG", "v58-03-self_s030000")  # adapter_of_record.yaml's candidate
PER_CHUNK = 25
PAIRED_SET = "geneval_two"


def cslug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", t.strip().lower()).strip("_")


def pair_slug(a: str, b: str) -> str:
    return f"{cslug(a)}__x__{cslug(b)}"


def pairs_all() -> list[tuple[str, str]]:
    kept, _ = load(PAIRED_SET)
    return [split(x["prompt"]) for x in kept]


def refused(a: str, b: str) -> bool:
    """PoE and ours refuse a pair naming a listed subject (1-lora-correction/scripts/showcase/disallowed_subjects.py). geneval_two
    keeps "computer mouse" and "baseball bat" pairs by setting the object phrase aside, which
    concept_pair_sheet.py and figure_candidates.py do not do, so such a pair dies at argument parsing."""
    from disallowed_subjects import check
    try:
        check(a), check(b)
        return False
    except ValueError:
        return True


def cells_all() -> list[str]:
    """review_bench_sets.py --co3-cells order: each pair, then each seed."""
    return [f"{PAIRED_SET}|{a}|{b}|and|{s}" for a, b in pairs_all() for s in SETS[PAIRED_SET]["seeds"]]


def chunk_items(method: str, set_: str, k: int, n: int) -> list:
    if method == "joint":
        kept, _ = load(set_)
        return kept[k::n]
    if method in ("poe", "ours"):
        ps = [p for p in pairs_all() if refused(*p) == set_.endswith(":refused")]
        return ps[k::n]
    if method == "co3":
        return cells_all()[k::n]
    raise ValueError(method)


def images(method: str, set_: str, k: int, n: int) -> list[Path]:
    items = chunk_items(method, set_, k, n)
    set_ = set_.split(":")[0]
    seeds = SETS[set_]["seeds"]
    if method == "joint":
        return [OUT / "bench_joint" / set_ / f"{x['idx']:04d}" / f"seed{s:02d}" / "joint.png" for x in items for s in seeds]
    if method == "poe":
        return [OUT / "bench_poe" / set_ / pair_slug(a, b) / f"seed{s:02d}" / "poe.png" for a, b in items for s in seeds]
    if method == "ours":
        return [OUT / "bench_ours" / set_ / pair_slug(a, b) / f"seed{s:02d}" / f"ours_{OURS_TAG}_w50.png"
                for a, b in items for s in seeds]
    out = []
    for line in items:
        g, a, b, _, s = line.split("|")
        out.append(OUT / "co3_default" / g / pair_slug(a, b) / f"seed{int(s):02d}" / "co3.png")
    return out


def read_chunks() -> list[dict]:
    rows = []
    for ln in (CLAIMS / "chunks.txt").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            cid, method, set_, k, n, nimg = ln.split()
            rows.append({"id": cid, "method": method, "set": set_, "k": int(k), "n": int(n), "nimg": int(nimg)})
    return rows


def skipped() -> set[str]:
    p = CLAIMS / "skip.txt"
    return {ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")} if p.exists() else set()


def get(cid: str) -> dict:
    for r in read_chunks():
        if r["id"] == cid:
            return r
    raise SystemExit(f"no chunk {cid}")


def imgs(r: dict) -> list[Path]:
    return images(r["method"], r["set"], r["k"], r["n"])


def build(force: bool) -> None:
    if (CLAIMS / "chunks.txt").exists() and not force:
        raise SystemExit(f"{CLAIMS / 'chunks.txt'} exists; pass --force to rewrite it (ids change if N changes)")
    lines = ["# id method set k N images   (scripts/bench_claims.py build)"]
    plan = [("joint", s) for s in ("geneval_two", "geneval_count", "t2i_spatial", "t2i_complex")]
    plan += [("poe", PAIRED_SET), ("ours", PAIRED_SET)]
    for method, set_ in plan:
        total = len(chunk_items(method, set_, 0, 1)) * len(SETS[set_]["seeds"])
        n = max(1, round(total / PER_CHUNK))
        for k in range(n):
            lines.append(f"{method}-{set_}-{k:03d}of{n:03d} {method} {set_} {k} {n} {len(images(method, set_, k, n))}")
    # CO3's cost per image differs, so its chunk size is CO3_PER_CHUNK
    n = max(1, round(len(cells_all()) / int(os.environ.get("CO3_PER_CHUNK", PER_CHUNK))))
    for k in range(n):
        lines.append(f"co3-{PAIRED_SET}-{k:03d}of{n:03d} co3 {PAIRED_SET} {k} {n} {len(images('co3', PAIRED_SET, k, n))}")
    # The pairs PoE and ours refuse (see refused()) get one chunk per method, listed in skip.txt.
    bad = [f"{a}|{b}" for a, b in pairs_all() if refused(a, b)]
    skip = ["# chunks no worker takes, and why"]
    for method in ("poe", "ours"):
        cid = f"{method}-{PAIRED_SET}-refused"
        lines.append(f"{cid} {method} {PAIRED_SET}:refused 0 1 {len(images(method, PAIRED_SET + ':refused', 0, 1))}")
        skip.append(f"{cid}  the renderer refuses these {len(bad)} pairs: " + "; ".join(bad))
    CLAIMS.mkdir(parents=True, exist_ok=True)
    for d in ("held", "done", "released"):
        (CLAIMS / d).mkdir(exist_ok=True)
    (CLAIMS / "chunks.txt").write_text("\n".join(lines) + "\n")
    (CLAIMS / "skip.txt").write_text("\n".join(skip) + "\n")
    print(f"{len(lines) - 1} chunks; {len(bad)} refused pairs per method in one skipped chunk each")


def newest_mtime(paths: list[Path]) -> float:
    m = 0.0
    for p in paths:
        try:
            m = max(m, p.stat().st_mtime)
        except OSError:
            pass
    return m


def release(cid: str, why: str) -> bool:
    try:
        os.rename(CLAIMS / "held" / cid, CLAIMS / "released" / f"{cid}.{why}.{int(time.time())}.{os.getpid()}")
        return True
    except OSError:
        return False


def release_stale() -> None:
    now = time.time()
    by_id = {r["id"]: r for r in read_chunks()}
    for d in (CLAIMS / "held").iterdir():
        r = by_id.get(d.name)
        try:
            t_claim = d.stat().st_mtime
        except OSError:
            continue
        if r is None or now - t_claim < STALE_MIN * 60:
            continue
        if now - newest_mtime(imgs(r)) >= STALE_MIN * 60:
            if release(d.name, "stale"):
                print(f"released stale claim {d.name} (claimed {(now - t_claim) / 60:.0f} min ago, no image in {STALE_MIN:.0f} min)",
                      file=sys.stderr)


def claim(owner: str, methods: set[str]) -> int:
    release_stale()
    skip = skipped()
    for r in read_chunks():
        if r["method"] not in methods or r["id"] in skip or (CLAIMS / "done" / r["id"]).exists():
            continue
        if (CLAIMS / "held" / r["id"]).exists():
            continue
        if all(p.exists() for p in imgs(r)):  # finished by an earlier launcher before the ledger existed
            (CLAIMS / "done" / r["id"]).mkdir(exist_ok=True)
            continue
        try:
            os.mkdir(CLAIMS / "held" / r["id"])
        except FileExistsError:
            continue
        (CLAIMS / "held" / r["id"] / "owner").write_text(f"{owner} {socket.gethostname()} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        print(r["id"], r["method"], r["set"], r["k"], r["n"])
        return 0
    return 1


def finish(cid: str) -> int:
    r = get(cid)
    miss = [p for p in imgs(r) if not p.exists()]
    if miss:
        release(cid, "incomplete")
        print(f"{cid}: {len(miss)} of {r['nimg']} images missing, claim released", file=sys.stderr)
        return 1
    (CLAIMS / "done" / cid).mkdir(exist_ok=True)
    try:
        os.rename(CLAIMS / "held" / cid, CLAIMS / "released" / f"{cid}.done.{int(time.time())}")
    except OSError:
        pass
    return 0


def status(count_images: bool) -> None:
    rows, skip = read_chunks(), skipped()
    held = {d.name for d in (CLAIMS / "held").iterdir()}
    done = {d.name for d in (CLAIMS / "done").iterdir()}
    print(f"{'method':6s} {'set':14s} {'chunks':>6s} {'done':>5s} {'claimed':>7s} {'waiting':>7s} {'skipped':>7s}"
          + ("  images on disk / needed" if count_images else ""))
    tot = [0, 0, 0, 0, 0]
    seen = []
    for r in rows:
        key = (r["method"], r["set"])
        if key not in seen:
            seen.append(key)
    for method, set_ in seen:
        rs = [r for r in rows if (r["method"], r["set"]) == (method, set_)]
        c = [len(rs), sum(r["id"] in done for r in rs), sum(r["id"] in held and r["id"] not in done for r in rs), 0,
             sum(r["id"] in skip and r["id"] not in done for r in rs)]
        c[3] = c[0] - c[1] - c[2] - c[4]
        tot = [a + b for a, b in zip(tot, c)]
        extra = ""
        if count_images:
            allp = [p for r in rs for p in imgs(r)]
            extra = f"  {sum(p.exists() for p in allp):5d} / {len(allp)}"
        print(f"{method:6s} {set_:14s} {c[0]:6d} {c[1]:5d} {c[2]:7d} {c[3]:7d} {c[4]:7d}{extra}")
    print(f"{'all':6s} {'':14s} {tot[0]:6d} {tot[1]:5d} {tot[2]:7d} {tot[3]:7d} {tot[4]:7d}")
    now = time.time()
    for h in sorted(held - done):
        try:
            own = (CLAIMS / "held" / h / "owner").read_text().strip()
            age = (now - (CLAIMS / "held" / h).stat().st_mtime) / 60
        except OSError:
            own, age = "?", 0
        print(f"  held {h:32s} {age:5.0f} min  {own}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--force", action="store_true")
    c = sp.add_parser("claim"); c.add_argument("--owner", required=True); c.add_argument("--methods", required=True)
    for name in ("pairs", "cells", "missing", "finish"):
        sp.add_parser(name).add_argument("id")
    r = sp.add_parser("release"); r.add_argument("id"); r.add_argument("why")
    s = sp.add_parser("status"); s.add_argument("--images", action="store_true", help="also stat every image (slow)")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.force)
    elif a.cmd == "claim":
        sys.exit(claim(a.owner, set(a.methods.split(","))))
    elif a.cmd == "pairs":
        r_ = get(a.id)
        for x, y in chunk_items(r_["method"], r_["set"], r_["k"], r_["n"]):
            print(f"{x}|{y}")
    elif a.cmd == "cells":
        r_ = get(a.id)
        print("\n".join(chunk_items("co3", r_["set"], r_["k"], r_["n"])))
    elif a.cmd == "missing":
        for p in imgs(get(a.id)):
            if not p.exists():
                print(p)
    elif a.cmd == "finish":
        sys.exit(finish(a.id))
    elif a.cmd == "release":
        sys.exit(0 if release(a.id, a.why) else 1)
    elif a.cmd == "status":
        status(a.images)


if __name__ == "__main__":
    main()
