"""
datasets.py
===========
Dataset loading. Every function returns a list of dicts, so every
notebook works with the same format regardless of where the data came from.

VERIFIED STRUCTURE (checked directly against the dataset pages - repos can
still change later, so re-check if a load fails):

Anthropic/model-written-evals
    Plain .jsonl files grouped in folders, NOT proper HF "configs". A
    bare load_dataset("Anthropic/model-written-evals") only returns
    whatever HF auto-converted as the "default" config, which is NOT the
    sycophancy data. Use load_hf_file() below with the exact paths:
        sycophancy/sycophancy_on_political_typology_quiz.jsonl
        sycophancy/sycophancy_on_nlp_survey.jsonl
        sycophancy/sycophancy_on_philpapers2020.jsonl
    Row schema: {"question": ..., "answer_matching_behavior": " (A)"/" (B)",
    "answer_not_matching_behavior": " (A)"/" (B)", "user_affiliation": "liberal"/...}
    Note: "question" already contains BOTH the persona/belief text AND the
    actual multiple-choice question, pre-combined into one string - there
    is no separate "belief" field to inject yourself.

meg-tong/sycophancy-eval
    Also plain .jsonl files at the repo root, no configs:
        answer.jsonl, are_you_sure.jsonl, feedback.jsonl, mimicry.jsonl
    IMPORTANT: this repo's automatic HF dataset-viewer/parquet conversion
    is currently BROKEN (a field's type is inconsistent across rows,
    which HF's own viewer reports as a DatasetGenerationError). This means
    a plain load_dataset("meg-tong/sycophancy-eval") call is not reliable.
    Use load_hf_file() below, which downloads the raw .jsonl file directly
    and parses it line by line with json.loads(), sidestepping the schema
    -unification step that fails.
"""

import json
from pathlib import Path


def load_hf_file(repo_id: str, filename: str, repo_type: str = "dataset", n=None) -> list:
    """
    Download ONE raw file from a HF repo and parse it as JSONL, line by
    line. This is the recommended way to load both
    Anthropic/model-written-evals and meg-tong/sycophancy-eval (see the
    module docstring above for why).

    Example:
        rows = load_hf_file(
            "Anthropic/model-written-evals",
            "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
        )
        rows = load_hf_file("meg-tong/sycophancy-eval", "are_you_sure.jsonl")
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    return load_local_jsonl(Path(local_path), n=n)


def load_local_jsonl(path: Path, n=None) -> list:
    """
    For datasets not hosted on HF (e.g. OEQ/AITA from
    github.com/myracheng/elephant) - download them first (e.g. with !wget
    in a Colab cell) into DATA_DIR, then read them from there.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if n:
        rows = rows[:n]
    return rows


def load_local_csv(path: Path, n=None) -> list:
    """Alternative loader if the dataset is in .csv format instead of .jsonl."""
    import csv

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
            if n and len(rows) >= n:
                break
    return rows
