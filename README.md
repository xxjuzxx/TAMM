# TAMM Experiment Code Snapshot

This directory is a GitHub-ready code snapshot for the TAMM ICDM Applied Track
submission:

**TAMM: Tail-aware Behavioral Motif-memory Mining for Open-world Traffic Anomaly
Diagnosis**

The snapshot contains public code, configurations, tests, paper-facing wrapper
scripts, and adapted baseline wrappers. It intentionally excludes raw datasets,
packet captures, generated result CSVs, logs, model checkpoints, LaTeX build
outputs, and large intermediate artifacts.

## Contents

- `src/`: data adapters, flow/packet feature extraction, behavior tokens, motif
  selection, evaluation metrics, model utilities, and training helpers.
- `scripts/`: preprocessing, split construction, tokenization, leakage checks,
  unknown-attack evaluation, calibration, memory tests, external validation, and
  paper-facing experiment wrappers.
- `configs/`: dataset, experiment, calibration, and model configuration files.
- `tests/`: lightweight unit and smoke tests for leakage controls, tokenization,
  adapters, thresholding, online replay utilities, and table-generation helpers.
- `experiments/`: adapted baseline wrappers for BSTS-Net, Kitsune, and
  CCF-A-style proxy baselines.
- `paper_scripts/`: auxiliary historical figure-building helpers that are
  code-only and do not include generated figures. The active manuscript figures
  and source SVG/PDF files are tracked through the artifact package.
- `docs/reproducibility_checklist.md`: detailed paper-order reproduction map and
  artifact checklist.
- `requirements.txt`: Python dependency snapshot from the research workspace.

## Data and Artifacts

Raw third-party traffic datasets are not redistributed. Obtain CICIDS2017,
UNSW-NB15, USTC-TFC2016, and CSE-CIC-IDS2018 from their official public sources
listed in `docs/reproducibility_checklist.md`.

The paper-facing results also require an artifact package containing split
manifests, generated CSV/JSON summaries, active manuscript figures/tables, P4
measurement notes/logs, and the LaTeX manuscript files. In the checklist, paths
are denoted as:

- `repo:/...`: relative to this GitHub repository root.
- `artifact:/...`: relative to the separate artifact package.

## Quick Checks

```bash
python -m pip install -r requirements.txt

python -m py_compile \
  src/features/motif_selection.py \
  scripts/12_eval_leave_one_attack_out.py \
  scripts/15_eval_threshold_sweep.py \
  scripts/label_ids2018_official_victim_streaming.py

bash -n scripts/run_ids2018_schedule_processing.sh

pytest -q tests/test_structural_primitives.py tests/test_tokenizer.py
```

## Paper Reproduction

Start from `docs/reproducibility_checklist.md`. It gives the paper-order mapping
from RQ1--RQ5 to scripts, result summaries, split manifests, and table/figure
files.

Some full-paper wrappers expect an accompanying artifact package because the raw
PCAP reruns and generated summaries are too large for the code repository.
Examples include:

- `scripts/60_run_unknown_multiseed_best_settings.py`
- `scripts/run_motif_selection_experiments.py`
- `scripts/run_memory_optimization_experiments.py`
- `scripts/build_tamm_paper_artifacts.py`
- `scripts/59_build_icdm_figures.py`

Set local data roots via command-line arguments or environment variables instead
of hard-coding workstation paths.

## Boundaries

TAMM is evaluated as a behavior-only motif-memory workflow for low-FPR
open-world traffic diagnosis. The P4 component is an implemented online
preprocessing path in an experimental testbed; motif transaction construction,
KNN scoring, thresholding, and evidence records remain host-side. 
