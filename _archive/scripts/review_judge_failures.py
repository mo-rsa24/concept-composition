"""Label each Figure 10 image with one failure type, using Claude as a fixed-prompt judge.

One Messages API call per image: the image (JPEG, 1024 px) and PROMPT below with the image's own two
concepts filled in. The model sees a random id and the image only, never the adapter setting. Output
is constrained to a JSON schema, so every answer is one of the labels. Results append to
<out>/judge_labels.csv as they arrive, so a stopped run resumes where it left off.

The key is read from ANTHROPIC_API_KEY, or from ~/.config/anthropic/api_key; it is never printed.

Usage:
  python3 scripts/review_judge_failures.py --key <key.csv> --images <dir> --out <dir> [--limit 5]
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

MODEL = "claude-opus-5-5"
LABELS = ["hybrid", "count error", "one", "intrusion", "two", "broken"]
PROMPT = (
    'You see one image generated for the prompt "a {x} and a {y}". Reply with JSON '
    '{{"label": ..., "style_loss": ...}}. label is exactly one of, checked in this order: '
    '"hybrid" if a single creature or object combines features of {x} and {y}; '
    '"count error" if a named thing appears more than once while the other is missing or miscounted; '
    '"one" if only one of {x} or {y} is drawn; '
    '"intrusion" if both are drawn and an unrequested object, creature or extra copy is prominent; '
    '"two" if both are drawn as separate, recognisable things; '
    '"broken" if no subject can be read. '
    'style_loss is "yes" if the image is a painting, drawing or illustration rather than a photograph, else "no".'
)
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": LABELS},
                   "style_loss": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["label", "style_loss"],
    "additionalProperties": False,
}
FIELDS = ["id", "label", "style_loss", "stop_reason", "model", "input_tokens", "output_tokens", "request_id"]


def concepts(pair: str) -> tuple[str, str]:
    a, b = (p.split("_", 1)[1].replace("_", " ") for p in pair.split("__x__"))
    return a, b


def make_client() -> anthropic.Anthropic:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic(max_retries=6)
    key = (Path.home() / ".config/anthropic/api_key").read_text().strip()
    return anthropic.Anthropic(api_key=key, max_retries=6)


def judge(client: anthropic.Anthropic, row: dict, images: Path) -> dict:
    x, y = concepts(row["pair"])
    data = base64.standard_b64encode((images / f"{row['id']}.jpg").read_bytes()).decode()
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        # the installed SDK predates these two fields, so they go in the body as sent
        extra_body={"fallbacks": "default",
                    "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
            {"type": "text", "text": PROMPT.format(x=x, y=y)},
        ]}],
    )
    out = {"id": row["id"], "stop_reason": resp.stop_reason, "model": resp.model,
           "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
           "request_id": resp._request_id, "label": "", "style_loss": ""}
    if resp.stop_reason != "refusal":
        text = next(b.text for b in resp.content if b.type == "text")
        out.update(json.loads(text))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.key)))
    out = Path(a.out) / "judge_labels.csv"
    done = {r["id"] for r in csv.DictReader(open(out))} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    client = make_client()
    lock = threading.Lock()
    new = not out.exists()
    n = 0
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ThreadPoolExecutor(a.workers) as pool:
            futs = {pool.submit(judge, client, r, Path(a.images)): r for r in todo}
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except anthropic.APIStatusError as e:
                    print(f"{futs[fut]['id']}: API error {e.status_code}: {e.message}")
                    continue
                except anthropic.APIConnectionError:
                    print(f"{futs[fut]['id']}: connection error, left for the next run")
                    continue
                with lock:
                    w.writerow(res)
                    f.flush()
                n += 1
                if n % 50 == 0:
                    print(f"{n}/{len(todo)}")
    print(f"labelled {n} this run; {len(done) + n} of {len(rows)} in {out}")


if __name__ == "__main__":
    main()
