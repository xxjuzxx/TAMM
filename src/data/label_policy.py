from __future__ import annotations

import re
from typing import Literal


AttemptedPolicy = Literal["keep", "drop", "attack", "benign"]

ATTEMPTED_POLICIES: tuple[str, ...] = ("keep", "drop", "attack", "benign")

MERGED_CICIDS_LABEL_MAP = {
    "benign": "BENIGN",
    "normal": "BENIGN",
    "ddos": "DDoS",
    "ddos attack-hoic": "DDoS",
    "ddos attack-loic-udp": "DDoS",
    "ddos attacks-loic-http": "DDoS",
    "portscan": "PortScan",
    "bot": "Bot",
    "ftp-patator": "BruteForce",
    "ssh-patator": "BruteForce",
    "ftp-bruteforce": "BruteForce",
    "ssh-bruteforce": "BruteForce",
    "brute force-web": "WebAttack",
    "brute force-xss": "WebAttack",
    "bruteforce": "WebAttack",
    "brute force": "WebAttack",
    "xss": "WebAttack",
    "sqlinjection": "WebAttack",
    "sql injection": "WebAttack",
    "slowhttptest": "DoS",
    "slowloris": "DoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowhttptest": "DoS",
    "dos slowloris": "DoS",
    "dos attacks-hulk": "DoS",
    "dos attacks-goldeneye": "DoS",
    "dos attacks-slowhttptest": "DoS",
    "dos attacks-slowloris": "DoS",
    "infiltration": "Infiltration",
    "infilteration": "Infiltration",
    "infiltration - portscan": "Infiltration",
    "heartbleed": "Heartbleed",
}


def normalize_label(value: object) -> str:
    label = str(value).strip()
    label = re.sub(r"\s+", " ", label)
    return label if label else "UNKNOWN"


def is_benign_label(label: object) -> bool:
    return normalize_label(label).lower() in {"benign", "normal", "benign traffic"}


def binary_label_for(label: object) -> str:
    return "BENIGN" if is_benign_label(label) else "ATTACK"


def is_attempted_label(label: object) -> bool:
    normalized = normalize_label(label).lower()
    return normalized == "attempted" or normalized.startswith("attempted") or " attempted" in normalized


def attempted_base_label(label: object) -> str:
    normalized = normalize_label(label)
    base = re.sub(r"(?i)\battempted\b", "", normalized)
    base = re.sub(r"^[\s:_/\\-]+|[\s:_/\\-]+$", "", base)
    base = re.sub(r"\s+", " ", base)
    return base or "Attempted"


def apply_attempted_policy(label: object, policy: AttemptedPolicy = "keep") -> str | None:
    if policy not in ATTEMPTED_POLICIES:
        raise ValueError(f"Unsupported attempted policy: {policy}")
    normalized = normalize_label(label)
    if not is_attempted_label(normalized):
        return normalized
    if policy == "drop":
        return None
    if policy == "attack":
        return attempted_base_label(normalized)
    if policy == "benign":
        return "BENIGN"
    return normalized


def merged_cicids_label(label: object) -> str:
    normalized = normalize_label(label)
    key = normalized.strip().lower()
    if key.startswith("web attack") or key.startswith("webattack"):
        return "WebAttack"
    key = key.replace("_", " ")
    key = re.sub(r"\s*-\s*", "-", key)
    key = re.sub(r"\s+", " ", key)
    compact = key.replace(" ", "")
    if key in {"benign", "normal"} or compact in {"benign", "normal"}:
        return "BENIGN"
    if "ddos" in key or "ddos" in compact:
        return "DDoS"
    if any(term in key for term in ("dos attacks", "dos hulk", "dos goldeneye", "dos slowhttptest", "dos slowloris", "slowhttptest", "slowloris", "goldeneye", "hulk")):
        return "DoS"
    if "infiltration" in key or "infilteration" in key:
        return "Infiltration"
    if "bot" in key:
        return "Bot"
    if "ftp-bruteforce" in compact or "ssh-bruteforce" in compact or "ftp-patator" in compact or "ssh-patator" in compact:
        return "BruteForce"
    if ("brute force" in key or "bruteforce" in compact) and any(term in key for term in ("web", "xss", "sql")):
        return "WebAttack"
    if "sql injection" in key or "web attack" in key or "webattack" in compact:
        return "WebAttack"
    if "portscan" in key:
        return "PortScan"
    if "heartbleed" in key:
        return "Heartbleed"
    return MERGED_CICIDS_LABEL_MAP.get(key, MERGED_CICIDS_LABEL_MAP.get(compact, normalized))
