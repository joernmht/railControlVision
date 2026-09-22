"""Real-time harness: the FastAPI service and its async client."""

from __future__ import annotations

from rail_vision_bench.harness.app import create_app
from rail_vision_bench.harness.client import HarnessClient, HarnessError

__all__ = ["HarnessClient", "HarnessError", "create_app"]
