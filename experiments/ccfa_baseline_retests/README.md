# CCF-A Baseline Retests for TAMM/TMDD Protocol

This directory contains same-protocol adapted retests for high-quality CCF-A traffic/anomaly-detection baselines.

The retests use the local TAMM/TMDD leave-one-unknown artifacts:

- IDS2017: `paper_icdm_applied_2026/experiments/unknown`
- IDS2018 official-victim external: `paper_icdm_applied_2026/experiments/ids2018_official_victim_external/unknown`

Important: most rows are adapted reproductions, not official full reproductions. The scripts keep the target paper protocol fixed: train/validation are benign-only, the held-out attack family appears only in test, and all methods report anomaly scores with the same AUROC/FPR95/low-FPR recall metrics.

