# BSTS-Net Reproduction Sandbox

This directory contains an isolated reproduction/adaptation of the public
BSTS-Net code for the local FlowPrim project datasets.

The upstream repository is available locally at `third_party/BSTS-Net`.
The upstream README provides the model code and a Git LFS pointer for the
Patator demonstration data, but does not include complete IDS2017/IDS2018
experiment configs or full intermediate features. The scripts here therefore
run a documented core-flow adaptation:

- use local labeled flow JSONL artifacts;
- train the same 128->64->32->16 triplet embedding network shape;
- perform source-IP/destination-port behavior-window anomaly detection;
- report FPR, precision, recall, F1, AUROC, and fixed-threshold diagnostics.

All generated artifacts are kept under this sandbox.
