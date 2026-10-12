#!/usr/bin/env python3
"""Shared measurement and analysis helpers for MPS recovery experiments."""

from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import math
import platform
import statistics
import time
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def recovery_threshold(
    latency_ms: list[float], *, multiplier: float = 1.25,
    mad_multiplier: float = 3.0, minimum_margin_ms: float = 0.10,
) -> dict[str, float]:
    """Calculate a robust readiness threshold from idle probe samples."""
    if len(latency_ms) < 3:
        raise ValueError("at least three calibration samples are required")
    median = statistics.median(latency_ms)
    deviations = [abs(value - median) for value in latency_ms]
    mad = statistics.median(deviations)
    threshold = max(
        median * multiplier,
        median + max(mad * mad_multiplier, minimum_margin_ms),
    )
    return {
        "median_ms": round(median, 6),
        "mad_ms": round(mad, 6),
        "p95_ms": round(percentile(latency_ms, 0.95), 6),
        "threshold_ms": round(threshold, 6),
        "multiplier": multiplier,
        "mad_multiplier": mad_multiplier,
        "minimum_margin_ms": minimum_margin_ms,
    }


def wait_for_recovery(
    measure: Callable[[], dict[str, Any]], *, threshold_ms: float,
    consecutive_required: int, interval_seconds: float, max_wait_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Probe until enough consecutive samples satisfy the readiness threshold."""
    if consecutive_required < 1 or interval_seconds <= 0 or max_wait_seconds <= 0:
        raise ValueError("recovery wait parameters must be positive")
    started = monotonic()
    consecutive = 0
    samples: list[dict[str, Any]] = []
    while True:
        sample = dict(measure())
        elapsed = monotonic() - started
        sample["wait_elapsed_seconds"] = round(elapsed, 3)
        sample["ready"] = sample["latency_ms"] <= threshold_ms
        samples.append(sample)
        if on_sample is not None:
            on_sample(sample)
        consecutive = consecutive + 1 if sample["ready"] else 0
        if consecutive >= consecutive_required:
            return {
                "ready": True,
                "timed_out": False,
                "wait_seconds": round(elapsed, 3),
                "samples": samples,
            }
        if elapsed >= max_wait_seconds:
            return {
                "ready": False,
                "timed_out": True,
                "wait_seconds": round(elapsed, 3),
                "samples": samples,
            }
        sleep(min(interval_seconds, max_wait_seconds - elapsed))


def fixed_recovery_window(
    measure: Callable[[], dict[str, Any]], *, cooldown_seconds: float,
    interval_seconds: float, threshold_ms: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Observe a fixed cooldown without allowing probe state to shorten it."""
    if cooldown_seconds < 0 or interval_seconds <= 0:
        raise ValueError("fixed recovery parameters are invalid")
    started = monotonic()
    samples: list[dict[str, Any]] = []
    while True:
        elapsed = monotonic() - started
        sample = dict(measure())
        sample["wait_elapsed_seconds"] = round(elapsed, 3)
        sample["ready"] = sample["latency_ms"] <= threshold_ms
        samples.append(sample)
        if on_sample is not None:
            on_sample(sample)
        remaining = cooldown_seconds - (monotonic() - started)
        if remaining <= 0:
            break
        sleep(min(interval_seconds, remaining))
    total = monotonic() - started
    return {
        "ready": bool(samples[-1]["ready"]),
        "timed_out": False,
        "wait_seconds": round(total, 3),
        "samples": samples,
    }


class AppleThermalProbe:
    """Read Apple's public system thermal-pressure state without PyObjC."""

    LABELS = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}

    def __init__(self) -> None:
        self.available = False
        self.error: str | None = None
        self._process_info = None
        self._thermal_selector = None
        self._low_power_selector = None
        self._send_long = None
        if platform.system() != "Darwin":
            self.error = "not_macos"
            return
        try:
            ctypes.CDLL(
                "/System/Library/Frameworks/Foundation.framework/Foundation"
            )
            objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
            objc.objc_getClass.argtypes = [ctypes.c_char_p]
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            objc.sel_registerName.restype = ctypes.c_void_p
            send_id = ctypes.CFUNCTYPE(
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            )(("objc_msgSend", objc))
            self._send_long = ctypes.CFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
            )(("objc_msgSend", objc))
            process_info_class = objc.objc_getClass(b"NSProcessInfo")
            self._process_info = send_id(
                process_info_class, objc.sel_registerName(b"processInfo"),
            )
            self._thermal_selector = objc.sel_registerName(b"thermalState")
            self._low_power_selector = objc.sel_registerName(
                b"isLowPowerModeEnabled"
            )
            self.available = bool(self._process_info)
            if not self.available:
                self.error = "process_info_unavailable"
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def measure(self) -> dict[str, Any]:
        if not self.available or self._send_long is None:
            return {
                "thermal_state": "unavailable",
                "thermal_state_raw": None,
                "thermal_performance_boundary_reached": None,
                "low_power_mode": None,
                "thermal_error": self.error,
            }
        raw = int(self._send_long(self._process_info, self._thermal_selector))
        state = self.LABELS.get(raw, "unknown")
        low_power = bool(
            self._send_long(self._process_info, self._low_power_selector)
        )
        return {
            "thermal_state": state,
            "thermal_state_raw": raw,
            "thermal_performance_boundary_reached": state in {"serious", "critical"},
            "low_power_mode": low_power,
            "thermal_error": None,
        }

    @staticmethod
    def metadata() -> dict[str, Any]:
        return {
            "source": "Foundation ProcessInfo.thermalState",
            "scope": "system thermal pressure, not GPU-only",
            "states": ["nominal", "fair", "serious", "critical"],
            "performance_impact_boundary": "serious",
            "temperature_celsius": None,
            "gpu_throttle_temperature_celsius": None,
            "temperature_note": "Apple exposes no supported GPU junction-temperature or fixed throttle-temperature API",
        }


class MPSLatencyProbe:
    """A deterministic, retained matrix multiplication on the shared MPS device."""

    def __init__(self, *, matrix_size: int = 4096, iterations: int = 1) -> None:
        if matrix_size < 256 or iterations < 1:
            raise ValueError("probe matrix size and iterations must be positive")
        import psutil
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        self.torch = torch
        self.psutil = psutil
        self.thermal = AppleThermalProbe()
        self.matrix_size = matrix_size
        self.iterations = iterations
        generator = torch.Generator(device="cpu").manual_seed(360)
        self.left = torch.randn(
            matrix_size, matrix_size, generator=generator, dtype=torch.float16,
        ).to("mps")
        self.right = torch.randn(
            matrix_size, matrix_size, generator=generator, dtype=torch.float16,
        ).to("mps")
        self._warmup()

    def _warmup(self) -> None:
        output = self.left @ self.right
        self.torch.mps.synchronize()
        del output

    def measure(self) -> dict[str, Any]:
        torch = self.torch
        torch.mps.synchronize()
        started = time.perf_counter()
        output = None
        for _ in range(self.iterations):
            output = self.left @ self.right
        torch.mps.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000 / self.iterations
        if output is None:
            raise RuntimeError("probe did not produce an output")
        checksum = float(output[0, 0].float().cpu())
        del output
        return {
            "recorded_at": utc_now(),
            "latency_ms": round(latency_ms, 6),
            "checksum": round(checksum, 6),
            "matrix_size": self.matrix_size,
            "iterations": self.iterations,
            "torch_current_allocated_bytes": torch.mps.current_allocated_memory(),
            "torch_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "torch_recommended_max_bytes": torch.mps.recommended_max_memory(),
            "client_process_rss_bytes": self.psutil.Process().memory_info().rss,
            "system_available_bytes": self.psutil.virtual_memory().available,
            **self.thermal.measure(),
        }


def summarize_runs(
    runs: list[dict[str, Any]], recovery_events: list[dict[str, Any]],
    *, measured_after: int = 2,
) -> dict[str, Any]:
    measured = runs[measured_after:]
    if not measured:
        return {"measured_runs": 0}
    totals = [float(item["total_seconds"]) for item in measured]
    started = min(item["started_monotonic"] for item in measured)
    ended = max(item["completed_monotonic"] for item in measured)
    elapsed = max(ended - started, 1e-9)
    waits = [float(item["wait_seconds"]) for item in recovery_events[measured_after - 1:]]
    profiles: dict[str, Any] = {}
    for profile in sorted({item["profile"] for item in measured}):
        values = [float(item["total_seconds"]) for item in measured if item["profile"] == profile]
        profiles[profile] = {
            "runs": len(values),
            "median_seconds": round(statistics.median(values), 3),
            "min_seconds": round(min(values), 3),
            "max_seconds": round(max(values), 3),
            "p95_seconds": round(percentile(values, 0.95), 3),
            "p95_to_median": round(percentile(values, 0.95) / statistics.median(values), 6),
        }
    return {
        "measured_runs": len(measured),
        "elapsed_seconds": round(elapsed, 3),
        "outputs_per_hour": round(len(measured) * 3600 / elapsed, 3),
        "median_run_seconds": round(statistics.median(totals), 3),
        "p95_run_seconds": round(percentile(totals, 0.95), 3),
        "total_recovery_wait_seconds": round(sum(waits), 3),
        "median_recovery_wait_seconds": round(statistics.median(waits), 3) if waits else 0.0,
        "profiles": profiles,
    }
