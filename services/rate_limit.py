from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from time import time


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after_seconds: int


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, *, limit: int, window_seconds: int) -> RateLimitResult:
        now = time()
        floor = now - window_seconds
        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] <= floor:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = int(max(1, window_seconds - (now - bucket[0])))
                return RateLimitResult(allowed=False, remaining=0, retry_after_seconds=retry_after)
            bucket.append(now)
            remaining = max(0, limit - len(bucket))
            return RateLimitResult(allowed=True, remaining=remaining, retry_after_seconds=0)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


_rate_limiter = InMemoryRateLimiter()


def get_rate_limiter() -> InMemoryRateLimiter:
    return _rate_limiter

