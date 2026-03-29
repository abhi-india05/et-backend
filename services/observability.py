from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any, Dict


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._request_metrics: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "errors": 0, "total_duration_ms": 0.0}
        )
        self._agent_metrics: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"success": 0, "failure": 0, "escalated": 0}
        )

    def record_request(self, *, method: str, path: str, status_code: int, duration_ms: float) -> None:
        key = f"{method.upper()} {path}"
        with self._lock:
            metric = self._request_metrics[key]
            metric["count"] += 1
            metric["total_duration_ms"] += max(0.0, duration_ms)
            if status_code >= 400:
                metric["errors"] += 1

    def record_agent_result(self, *, agent_name: str, status: str) -> None:
        bucket = status if status in {"success", "failure", "escalated"} else "failure"
        with self._lock:
            self._agent_metrics[agent_name][bucket] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            request_metrics = {
                key: {
                    **value,
                    "avg_duration_ms": round(value["total_duration_ms"] / value["count"], 2)
                    if value["count"]
                    else 0.0,
                }
                for key, value in self._request_metrics.items()
            }
            return {"requests": request_metrics, "agents": dict(self._agent_metrics)}


_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    return _registry
