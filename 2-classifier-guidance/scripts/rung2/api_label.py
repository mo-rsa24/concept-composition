#!/usr/bin/env python
"""Label pair pictures through the Claude API: is this a faithful pair, a hybrid, a single, a wrong pair, damaged?

Runs on the laptop (the cluster has no API key and no `anthropic` package). The key is read from
~/.config/anthropic/api_key and passed to the client; it is never printed or written anywhere.

Each picture is sent with its pair's two names and a fixed answer schema (structured output). A
picture is a faithful pair when it shows exactly two separate bodies, one recognisable A and one
recognisable B, no body mixing features of both, as an intact photo: the same rule as the hand calls.

--validate: label plan 22's 443 hand-called pictures and report agreement with the hand calls.
--folder DIR: label every JPEG/PNG in DIR, named <pair>__seedNN__<joint|product>.<ext>.
Results are appended to a JSONL file (one line per picture), so a stopped run resumes.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import re
from collections import Counter
from pathlib import Path

import anthropic

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS = REPO / "2-classifier-guidance/artifacts/results/does-a-small-discriminator-steer-cat-x-dog"
MODEL = "claude-opus-5-5"
VERDICTS = ["faithful pair", "hybrid", "single", "wrong pair", "damaged"]
SCHEMA = {
    "type": "object",
    "properties": {
        "bodies": {"type": "integer", "description": "number of separate main animals or objects"},
        "a_present": {"type": "boolean"},
        "b_present": {"type": "boolean"},
        "hybrid": {"type": "boolean"},
        "photo_intact": {"type": "boolean"},
        "verdict": {"type": "string", "enum": VERDICTS},
        "note": {"type": "string"},
    },
    "required": ["bodies", "a_present", "b_present", "hybrid", "photo_intact", "verdict", "note"],
    "additionalProperties": False,
}


def names(pair: str) -> tuple[str, str]:
    a, b = pair.split("__x__")
    return a.replace("_", " "), b.replace("_", " ")


def prompt(a: str, b: str) -> str:
    return (f"This picture was generated for the two things \"{a}\" and \"{b}\". Look at it closely and answer:\n"
            f"- bodies: how many separate main animals or objects are shown (count each distinct body once; "
            f"background crowds or tiny distant figures do not count).\n"
            f"- a_present: is there a body that is clearly and only {a}?\n"
            f"- b_present: is there a body that is clearly and only {b}?\n"
            f"- hybrid: does any body mix features of both (for example {a}'s head on {b}'s body, or one's "
            f"markings, ears, snout, tail or legs on the other)?\n"
            f"- photo_intact: is it a photograph-like picture with no melted, torn, duplicated or broken parts "
            f"(a drawing, painting, cartoon or split-frame collage is not intact)?\n"
            f"- verdict: \"faithful pair\" only if bodies is 2, a_present and b_present are true, hybrid is "
            f"false and photo_intact is true. Otherwise the first that applies: \"hybrid\" if any body mixes "
            f"both; \"single\" if there is only one body; \"wrong pair\" if there are two or more bodies but "
            f"not exactly one {a} and one {b}; \"damaged\" otherwise.\n"
            f"- note: under 15 words on what you see.")


async def label_one(client, sem, path: Path, pair: str):
    a, b = names(pair)
    media = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode()
    async with sem:
        resp = await client.beta.messages.create(
            model=MODEL, max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"], extra_body={"fallbacks": "default"},
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
                {"type": "text", "text": prompt(a, b)}]}])
    if resp.stop_reason == "refusal":
        return {"error": "refusal"}
    text = next(bl.text for bl in resp.content if bl.type == "text")
    out = json.loads(text)
    out["model"] = resp.model
    return out


async def run(items, out_path: Path, concurrency: int):
    key = (Path.home() / ".config/anthropic/api_key").read_text().strip()
    client = anthropic.AsyncAnthropic(api_key=key, max_retries=6)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} pictures, {len(done)} already labelled, {len(todo)} to go", flush=True)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    n = 0

    async def worker(it):
        nonlocal n
        try:
            res = await label_one(client, sem, it["path"], it["pair"])
        except anthropic.BadRequestError as e:
            res = {"error": f"bad request: {e.message}"}
        except anthropic.APIStatusError as e:
            res = {"error": f"status {e.status_code}"}
        except anthropic.APIConnectionError:
            res = {"error": "connection"}
        row = {k: v for k, v in it.items() if k != "path"} | res
        async with lock:
            with out_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"{n} of {len(todo)} labelled", flush=True)

    await asyncio.gather(*(worker(it) for it in todo))
    await client.close()


def validate_items():
    rows = list(csv.DictReader(open(RESULTS / "dataset-v3/calls-final.csv")))
    by_md5 = {r["md5"]: r for r in rows}
    items = []
    for t in json.loads((HERE / "training_set_v3.json").read_text())["tiles"]:
        if t["source"] == "local":
            path = REPO / by_md5[t["id"]]["path"].lstrip("./")
        else:
            path = RESULTS / "training-set/tiles" / f"{t['id']}.png"
        items.append({"id": t["id"], "pair": t["pair"], "path": path, "label": t["label"], "kind": t["kind"]})
    return items


def report(out_path: Path):
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "verdict" in r]
    errs = Counter(r.get("error") for r in rows if "verdict" not in r)
    tp = sum(1 for r in ok if r["label"] == 1 and r["verdict"] == "faithful pair")
    tn = sum(1 for r in ok if r["label"] == 0 and r["verdict"] != "faithful pair")
    pos, neg = sum(r["label"] == 1 for r in ok), sum(r["label"] == 0 for r in ok)
    print(f"labelled {len(ok)} (errors {dict(errs)}); hand positives called faithful pair {tp} of {pos}; "
          f"hand negatives called otherwise {tn} of {neg}; balanced agreement {(tp / pos + tn / neg) / 2:.3f}")
    by = {}
    for r in ok:
        by.setdefault(r["kind"], Counter())[r["verdict"]] += 1
    for k, c in by.items():
        print(f"  hand {k:22s} -> {dict(c)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    if args.validate:
        items = validate_items()
    else:
        items = []
        for p in sorted(args.folder.glob("*")):
            m = re.match(r"(.+?)__seed(\d+)__(joint|product|photo)$", p.stem)
            if m and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append({"id": p.stem, "pair": m.group(1), "seed": int(m.group(2)), "method": m.group(3), "path": p})
    asyncio.run(run(items, args.out, args.concurrency))
    if args.validate:
        report(args.out)
    print("API LABELS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
