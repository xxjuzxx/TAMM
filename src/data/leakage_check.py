from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path
from typing import Any


class LeakageError(RuntimeError):
    pass


PORT_TOKEN_RE = re.compile(r"(?:^|[_=])(?:PORT|DPORT|SPORT|SRC_PORT|DST_PORT)(?:[_=]|$)", re.IGNORECASE)
TIME_TOKEN_RE = re.compile(r"(?:^|[_=])(?:TS|TIME|TIMESTAMP|START_TS|END_TS)(?:[_=]|$)", re.IGNORECASE)


def _status(ok: bool, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": bool(ok), **(detail or {})}


def _splits(payload: dict[str, Any]) -> dict[str, list[str]]:
    return payload.get("splits", payload)


def check_flow_id_disjoint(split_payload: dict[str, Any]) -> dict[str, Any]:
    splits = _splits(split_payload)
    seen: dict[str, str] = {}
    overlaps: list[dict[str, str]] = []
    for split_name in ("train", "val", "test"):
        for flow_id in splits.get(split_name, []):
            flow_id = str(flow_id)
            if flow_id in seen:
                overlaps.append({"flow_id": flow_id, "first": seen[flow_id], "second": split_name})
            else:
                seen[flow_id] = split_name
    return _status(not overlaps, {"overlaps": overlaps, "num_unique": len(seen)})


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_vocab(path: str | Path | None) -> dict[str, int] | None:
    payload = _read_json(path)
    if payload is None:
        return None
    if "token_to_id" in payload:
        payload = payload["token_to_id"]
    return {str(key): int(value) for key, value in payload.items()}


def _looks_like_ip(token: str) -> bool:
    value = token.strip("<>[](){} ,;")
    if not value or "." not in value and ":" not in value:
        return False
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        pass
    for part in re.split(r"[^0-9a-fA-F:.]+", value):
        if not part:
            continue
        try:
            ipaddress.ip_address(part)
            return True
        except ValueError:
            continue
    return False


def check_vocab_tokens(vocab: dict[str, int], *, allow_port_tokens: bool = False) -> dict[str, Any]:
    ip_tokens = sorted(token for token in vocab if _looks_like_ip(token))
    time_tokens = sorted(token for token in vocab if TIME_TOKEN_RE.search(token))
    port_tokens = sorted(token for token in vocab if PORT_TOKEN_RE.search(token))
    ok = not ip_tokens and not time_tokens and (allow_port_tokens or not port_tokens)
    return _status(
        ok,
        {
            "num_tokens": len(vocab),
            "ip_tokens": ip_tokens[:20],
            "time_tokens": time_tokens[:20],
            "port_tokens": port_tokens[:20],
            "allow_port_tokens": bool(allow_port_tokens),
        },
    )


def check_token_meta(token_data: dict[str, Any], *, allow_port_tokens: bool = False) -> dict[str, Any]:
    bad: list[dict[str, Any]] = []
    for meta in token_data.get("meta", []):
        flags = {
            "has_ip_token": bool(meta.get("has_ip_token", False)),
            "has_abs_time_token": bool(meta.get("has_abs_time_token", False)),
            "has_port_token": bool(meta.get("has_port_token", False)),
        }
        if flags["has_ip_token"] or flags["has_abs_time_token"] or (flags["has_port_token"] and not allow_port_tokens):
            bad.append({"flow_id": meta.get("flow_id"), **flags})
    return _status(not bad, {"bad_rows": bad[:20], "bad_count": len(bad), "allow_port_tokens": bool(allow_port_tokens)})


def check_train_only_provenance(manifest: dict[str, Any] | None, artifact: str) -> dict[str, Any]:
    if manifest is None:
        return _status(False, {"artifact": artifact, "reason": "manifest_missing"})
    provenance = str(manifest.get("provenance") or manifest.get("vocab_provenance") or "")
    train_only = bool(manifest.get("train_only", False) or provenance == "train_only")
    split_path = manifest.get("splits") or manifest.get("split_path")
    return _status(train_only and bool(split_path), {"artifact": artifact, "provenance": provenance, "split_path": split_path})


def build_leakage_report(
    *,
    split_payload: dict[str, Any],
    profile_manifest: dict[str, Any] | None = None,
    token_manifest: dict[str, Any] | None = None,
    vocab: dict[str, int] | None = None,
    token_data: dict[str, Any] | None = None,
    allow_port_tokens: bool = False,
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "L1_flow_id_disjoint": check_flow_id_disjoint(split_payload),
        "L2_profile_primitives_train_only": check_train_only_provenance(profile_manifest, "profile_primitives"),
        "L3_vocab_train_only": check_train_only_provenance(token_manifest, "vocab"),
        "L7_val_test_not_used_for_training_stats": _status(
            bool(token_manifest and token_manifest.get("train_only", False)),
            {"source": "token_manifest"},
        ),
        "L8_test_not_used_for_threshold_tuning": _status(
            bool(token_manifest is None or token_manifest.get("threshold_tuning_split", "val") == "val"),
            {"threshold_tuning_split": None if token_manifest is None else token_manifest.get("threshold_tuning_split", "val")},
        ),
    }
    if vocab is not None:
        token_check = check_vocab_tokens(vocab, allow_port_tokens=allow_port_tokens)
        checks["L4_vocab_no_raw_ip"] = _status(token_check["ok"] and not token_check["ip_tokens"], {"ip_tokens": token_check["ip_tokens"]})
        checks["L5_vocab_no_abs_time"] = _status(
            token_check["ok"] and not token_check["time_tokens"],
            {"time_tokens": token_check["time_tokens"]},
        )
        checks["L6_vocab_no_raw_port"] = _status(
            allow_port_tokens or not token_check["port_tokens"],
            {"port_tokens": token_check["port_tokens"], "allow_port_tokens": bool(allow_port_tokens)},
        )
    if token_data is not None:
        meta_check = check_token_meta(token_data, allow_port_tokens=allow_port_tokens)
        checks["L4_L5_L6_token_meta_flags"] = meta_check
    passed = all(bool(item.get("ok")) for item in checks.values())
    return {"passed": passed, "checks": checks}


def assert_no_leakage(report: dict[str, Any]) -> None:
    if report.get("passed"):
        return
    failed = [name for name, item in report.get("checks", {}).items() if not item.get("ok")]
    raise LeakageError(f"Leakage check failed: {', '.join(failed)}")
