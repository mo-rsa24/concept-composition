#!/usr/bin/env bash
# Usage: fetch_papers.sh <empty folder under $CLAUDE_JOB_DIR/tmp>. Downloads each listed PDF into its own folder and writes page-marked text.
set -u; HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="$1"; mkdir -p "$OUT"
while IFS=$'\t' read -r slug id name; do d="$OUT/$slug"; mkdir -p "$d"; [ -s "$d/paper.pdf" ] || { curl -sS -L --max-time 90 -o "$d/paper.pdf" "https://arxiv.org/pdf/$id"; sleep 3; }; [ "$(head -c4 "$d/paper.pdf")" = "%PDF" ] || echo "NOT A PDF: $slug $id"; done < "$HERE/papers.tsv"
python3 -I "$HERE/page_split.py" "$OUT"
