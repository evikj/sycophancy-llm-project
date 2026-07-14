# Sycophancy in LLMs — Personalization & Emotional Intelligence

Code for a course project studying how persona-based emotional
intelligence prompting affects propositional and social sycophancy in
open-weight instruction-tuned LLMs (Llama-3-8B, Mistral-7B).

Scope is deliberately narrow - see [`docs/plan.md`](docs/plan.md) (in
Macedonian) for the frozen research plan, including what was cut and why.
The full experimental report (results, discussion) will also live in
[`docs/`](docs/), in Macedonian - this README covers setup and code
structure only.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                  <- single source of truth for paths/models/params
├── utils/
│   ├── colab_setup.py         <- Drive mount + repo clone/update + install deps
│   ├── checkpoint.py          <- resumable experiment loops (see below)
│   ├── model_loader.py        <- 4-bit model loading + generate()
│   ├── prompts.py             <- all prompt templates
│   ├── datasets.py            <- dataset loaders (HF + local jsonl/csv)
│   └── evaluate.py            <- response parsing + metrics
├── notebooks/
│   ├── 00_connection_test.ipynb
│   └── 01_propositional_baseline_EXAMPLE.ipynb
└── docs/
    └── plan.md                <- frozen research plan (Macedonian)
```

## Two storage tiers: GitHub vs. Google Drive

This project deliberately splits storage into two tiers:

| | Lives on | Tracked by git? | Why |
|---|---|---|---|
| Code (`config.py`, `utils/`, `notebooks/`) | GitHub, cloned into Colab's local disk (`/content/repo`) each session | Yes | Small, text-only, worth versioning/diffing |
| Data, model weights, experiment results | Google Drive (`MyDrive/sycophancy_project_data/`) | No | Large, binary, or frequently appended - not meaningful to diff, would bloat the repo |

Colab's local disk is wiped every time the runtime resets, but that's
fine for the code: it's re-cloned from GitHub in a few seconds at the
start of each session. Drive persists across sessions and holds
everything too large or too dynamic for git.

## Setup

1. In a new Colab notebook, first cell:

```python
import sys, subprocess
from google.colab import drive

drive.mount("/content/drive")
subprocess.run(["git", "clone", "https://github.com/<you>/<repo>.git", "/content/repo"], check=False)
subprocess.run(["pip", "install", "-q", "-r", "/content/repo/requirements.txt"], check=True)
sys.path.insert(0, "/content/repo")
```

2. From then on: `from config import ...` and `from utils.xxx import ...`
   work normally.

3. Gated models (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`) require
   requesting access on Hugging Face and logging in with a token
   (`huggingface-cli login` or `huggingface_hub.login()`), once per
   session.

## Workflow

- One notebook per model x phase combination. Copy
  `01_propositional_baseline_EXAMPLE.ipynb` and rename it for each new
  experiment.
- Each experiment writes to its own `.jsonl` file under
  `results/raw/` on Drive.
- If Colab disconnects (idle timeout ~90 min, or the free-tier session
  cap of ~12h), just re-run the notebook from the top -
  `run_resumable()` skips everything already saved.
- To persist code edits made inside `/content/repo` during a session,
  commit and push before the session ends (or before you close the
  tab) - anything not pushed is lost when the runtime resets.
