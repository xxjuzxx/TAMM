# Kitsune Flow-artifact Baseline

This directory evaluates a Kitsune-style autoencoder ensemble under the same TAMM CICIDS2017 leave-one unknown split protocol.

The baseline is an adapted flow-artifact implementation, not an official packet-level Kitsune reproduction. It keeps the Kitsune idea of feature mapping, small ensemble autoencoders, and an output autoencoder anomaly score, but uses the local JSONL flow behavior artifacts available to TAMM rather than live packet feature extraction.

