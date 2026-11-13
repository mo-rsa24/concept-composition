#!/usr/bin/env python3
"""Is poe_repair_min quiet enough to regroup? Prints CLEAR or WAIT, and why.

The regroup in plans/regroup-into-project-folders.md moves hundreds of files across three
checkouts. It must not run while another Claude session is writing into them, or while files sit
uncommitted, because a move under a live session breaks that session's paths and a merge over
uncommitted work mixes it in half-finished.

For each checkout it lists the uncommitted files (ignoring .playwright-mcp/, browser-tool
scratch), matches each to the most recent session whose edits or commands name it, and lists every
session active in the last hour. CLEAR means: no session but the caller active in the last hour,
and no uncommitted file left.

    python3 scripts/quiet_window_check.py [--me <session id>] [--minutes 60]
"""
import argparse
import collections
import glob
import json
import os
import subprocess
import time

PHD = "/home/molef/PhD"
CHECKOUTS = {
    "poe_repair_min": "-home-molef-PhD-poe-repair-min",
    "poe_repair_min-dog-and-rain": "-home-molef-PhD-poe-repair-min-dog-and-rain",
    "poe_repair_min-reviews": "-home-molef-PhD-poe-repair-min-reviews",
}
IGNORED_PREFIXES = (".playwright-mcp/",)
LOOKBACK_DAYS = 3


def uncommitted(checkout):
    out = subprocess.run(["git", "-C", f"{PHD}/{checkout}", "status", "--porcelain", "-uall"],
                         capture_output=True, text=True).stdout.splitlines()
    paths = [line[3:].split(" -> ")[-1].strip('"') for line in out]
    return [p for p in paths if not p.startswith(IGNORED_PREFIXES)]


def sessions(project_dir):
    found = {}
    cutoff = time.time() - LOOKBACK_DAYS * 86400
    for f in glob.glob(os.path.expanduser(f"~/.claude/projects/{project_dir}/*.jsonl")):
        if os.path.getmtime(f) < cutoff:
            continue
        named = []
        for line in open(f, errors="ignore"):
            if '"tool_use"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for part in record.get("message", {}).get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    tool_input = part.get("input", {})
                    named.append(f"{tool_input.get('file_path', '')} {tool_input.get('command', '')}")
        found[os.path.basename(f)[:-6]] = (" ".join(named), os.path.getmtime(f))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--me", default="", help="the calling session's id, left out of the live count")
    parser.add_argument("--minutes", type=int, default=60, help="how recent counts as live")
    args = parser.parse_args()
    live_cutoff = time.time() - args.minutes * 60
    blocking_files = 0
    live = []
    for checkout, project_dir in CHECKOUTS.items():
        files = uncommitted(checkout)
        known = sessions(project_dir)
        blocking_files += len(files)
        owner = collections.defaultdict(list)
        for path in files:
            hits = [sid for sid, (text, _) in known.items() if path in text]
            owner[max(hits, key=lambda s: known[s][1]) if hits else "no recent session"].append(path)
        print(f"== {checkout}: {len(files)} uncommitted")
        for sid, paths in sorted(owner.items(), key=lambda kv: -len(kv[1])):
            print(f"   {sid[:8]:18s} {len(paths):5d} files, e.g. {paths[0]}")
        for sid, (_, mtime) in known.items():
            if mtime >= live_cutoff and not sid.startswith(args.me or "\0"):
                live.append((checkout, sid, time.strftime("%H:%M", time.localtime(mtime))))
    print("== live in the last", args.minutes, "minutes:",
          ", ".join(f"{s[:8]} ({c}, {t})" for c, s, t in live) or "none")
    clear = not live and blocking_files == 0
    print("CLEAR" if clear else f"WAIT: {len(live)} live session(s), {blocking_files} uncommitted file(s)")


if __name__ == "__main__":
    main()
