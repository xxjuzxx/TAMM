from __future__ import annotations

import math


def log_bin(value: float | int, max_bin: int = 15) -> int:
    if value <= 0:
        return 0
    return min(int(math.floor(math.log2(float(value) + 1.0))), max_bin)


def iat_bin(seconds: float, max_bin: int = 15) -> int:
    millis = max(seconds, 0.0) * 1000.0
    return log_bin(millis, max_bin=max_bin)


def count_bin(value: int, max_bin: int = 15) -> int:
    return log_bin(value, max_bin=max_bin)


def ratio_bin(value: float, bins: int = 10) -> int:
    value = max(0.0, min(1.0, value))
    return min(int(value * bins), bins)
