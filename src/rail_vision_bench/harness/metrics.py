"""Prometheus metrics of the harness.

The metrics live in a dedicated registry rather than the process-global one so
that tests can create many applications in one process without duplicate
registration errors, and so ``/metrics`` never exposes unrelated collectors.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY: Final[CollectorRegistry] = CollectorRegistry()

FRAME_LATENCY: Final[Histogram] = Histogram(
    "rvb_frame_latency_seconds",
    "Wall-clock time of one frame inference.",
    labelnames=["model", "mode"],
    registry=REGISTRY,
)
FRAMES_TOTAL: Final[Counter] = Counter(
    "rvb_frames_total",
    "Frames processed, by outcome (ok, invalid, error).",
    labelnames=["model", "mode", "status"],
    registry=REGISTRY,
)
COST_USD: Final[Counter] = Counter(
    "rvb_cost_usd_total",
    "Accumulated provider cost in US dollars.",
    labelnames=["model"],
    registry=REGISTRY,
)
TOKENS: Final[Counter] = Counter(
    "rvb_tokens_total",
    "Tokens exchanged with the provider, by direction (in, out).",
    labelnames=["model", "direction"],
    registry=REGISTRY,
)
