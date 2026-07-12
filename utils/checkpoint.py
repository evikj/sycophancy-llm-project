"""
checkpoint.py
=============
This is the file that solves the Colab session-limit problem.

Idea: instead of collecting results in an in-memory list and saving them
at the end (which means - if the session disconnects at 90% of the loop,
you lose ALL results) - every single result is written immediately, one
line at a time, to a .jsonl file on Drive. When you resume work later
(new session, same day or a week later), the loop automatically skips
everything that was already processed.

Usage inside a notebook:

    from utils.checkpoint import run_resumable
    from config import RAW_RESULTS_DIR

    out_path = RAW_RESULTS_DIR / "baseline_llama3_propositional.jsonl"

    def process_one(item):
        response = generate(model, tokenizer, None, item["prompt"])
        return {"prompt": item["prompt"], "response": response}

    run_resumable(
        items=my_dataset,
        id_fn=lambda item: item["question_id"],
        process_fn=process_one,
        out_path=out_path,
    )

If the session disconnects at, say, item 340/1000 - the next time you run
the same cell, it reads that 340 are already done and continues from
item 341, without repeating them.
"""

import json
from pathlib import Path


def load_done_ids(path: Path) -> set:
    """Return the set of IDs that have already been processed and saved."""
    if not path.exists():
        return set()

    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row["id"])
            except (json.JSONDecodeError, KeyError):
                # Partial/corrupted line (e.g. the session disconnected
                # mid-write) - just skip it.
                continue
    return done


def append_result(path: Path, row: dict):
    """Append one result to disk IMMEDIATELY - don't wait for the loop to finish."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_all_results(path: Path) -> list:
    """Read all results saved so far (e.g. for analysis/visualization)."""
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def run_resumable(items, id_fn, process_fn, out_path: Path, progress_every: int = 20):
    """
    Generic resumable loop.

    items          -- list of examples to process
    id_fn          -- function item -> unique string id
                       (must be stable: the same item always yields the same id)
    process_fn     -- function item -> dict with the result (WITHOUT an "id"
                       key; that is added automatically)
    out_path       -- Path to the .jsonl file where results are stored
    progress_every -- print a status line every N processed items
    """
    done = load_done_ids(out_path)
    total = len(items)
    print(f"[checkpoint] Already done: {len(done)} / {total}")

    processed_this_run = 0
    for i, item in enumerate(items):
        item_id = id_fn(item)
        if item_id in done:
            continue

        try:
            result = process_fn(item)
            result["id"] = item_id
            append_result(out_path, result)
            processed_this_run += 1
        except Exception as e:
            # One bad example (e.g. the model returned an empty output)
            # should not crash a loop of hundreds of examples. Log and continue.
            print(f"[checkpoint] ERROR at id={item_id}: {e}")
            continue

        if (i + 1) % progress_every == 0:
            print(f"[checkpoint] ...processed {i + 1}/{total} "
                  f"(new this run: {processed_this_run})")

    print(f"[checkpoint] Done. New this run: {processed_this_run}. "
          f"Total in file: {len(load_done_ids(out_path))}/{total}")
