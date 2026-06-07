from __future__ import annotations

import numpy as np

from src.evaluation.robustness import PER_FLOW_PERTURBATION_MODES, perturb_flow, perturb_flows


def _flow() -> dict:
    return {
        "flow_id": "f1",
        "service_key": ["10.0.0.2", "443", "TCP"],
        "label": "Benign",
        "binary_label": "BENIGN",
        "lens": [50, 60, 70, 80, 90, 100],
        "dirs": [True, True, False, False, True, False],
        "tss": [0.0, 0.001, 0.002, 0.5, 0.501, 1.0],
        "packet_count": 6,
        "start_ts": 0.0,
        "end_ts": 1.0,
        "duration": 1.0,
    }


def test_all_perturbations_preserve_flow_shape() -> None:
    for mode in sorted(PER_FLOW_PERTURBATION_MODES):
        out = perturb_flow(_flow(), mode=mode, strength=0.3, rng=np.random.default_rng(7))
        assert out["lens"]
        assert len(out["lens"]) == len(out["dirs"]) == len(out["tss"]) == out["packet_count"]
        assert out["start_ts"] == out["tss"][0]
        assert out["end_ts"] == out["tss"][-1]
        assert out["duration"] >= 0.0


def test_length_alignment_uses_expected_boundary() -> None:
    out = perturb_flow(_flow(), mode="length_align", strength=0.5, rng=np.random.default_rng(7))
    assert all(length % 128 == 0 for length in out["lens"])


def test_direction_flip_changes_at_least_one_direction() -> None:
    base = _flow()
    out = perturb_flow(base, mode="direction_flip", strength=0.3, rng=np.random.default_rng(7))
    assert out["dirs"] != base["dirs"]


def test_short_flow_delete_removes_short_flows() -> None:
    short = _flow()
    short["flow_id"] = "short"
    short["lens"] = short["lens"][:3]
    short["dirs"] = short["dirs"][:3]
    short["tss"] = short["tss"][:3]
    short["packet_count"] = 3
    long = _flow()
    long["flow_id"] = "long"
    out = perturb_flows([short, long], mode="short_flow_delete", strength=1.0, seed=7)
    assert [flow["flow_id"] for flow in out] == ["long"]


def test_benign_flow_insert_adds_benign_flows() -> None:
    base = _flow()
    out = perturb_flows([base], mode="benign_flow_insert", strength=1.0, seed=7)
    assert len(out) == 2
    assert sum(1 for flow in out if flow["binary_label"] == "BENIGN") == 2
