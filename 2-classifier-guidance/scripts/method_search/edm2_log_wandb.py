#!/usr/bin/env python3
"""Copy one EDM2 run's stats.jsonl into a W&B run, so step M7's trainings sit beside the project's other runs.

Usage: python edm2_log_wandb.py <run folder> <run name>
"""
import json
import sys
from pathlib import Path


def main():
    folder, name = Path(sys.argv[1]), sys.argv[2]
    rows = [json.loads(line) for line in (folder / "stats.jsonl").read_text().splitlines() if line.strip()]
    try:
        import wandb
        run = wandb.init(entity="prime_lab", project="poe-repair-animals-compose", group="loop-6-m7", name=name,
                         config={"folder": str(folder)}, settings=wandb.Settings(x_service_wait=300))
    except Exception as error:   # the stats stay on disk either way
        print(f"W&B unavailable: {error}", flush=True)
        return
    for row in rows:
        values = {key: value for key, value in row.items() if isinstance(value, (int, float)) and key != "timestamp"}
        if "Progress/kimg" in values:
            run.log(values, step=int(round(values["Progress/kimg"] * 1000)))
    losses = [row["Loss/loss"] for row in rows if isinstance(row.get("Loss/loss"), (int, float))]
    if losses:
        run.summary.update({"first_loss": losses[0], "last_loss": losses[-1]})
    print(f"W&B run {run.id}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
