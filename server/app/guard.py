"""
Abuse guard: a per-IP sliding-window rate limit, plus a global daily cap
that protects the actual dollar cost regardless of how many different IPs
show up. Two layers, because a per-IP limit alone does nothing against a
hundred different IPs - see server/README.md for the reasoning.

In-memory, single-process. That is enough for one Cloud Run instance
pinned with min-instances=1/max-instances=1 (a proof-of-concept setup).
Running more than one instance needs a shared store (Redis, Firestore) or
the caps become per-instance instead of global - flagged in the README.
"""
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time


@dataclass(frozen=True)
class GuardConfig:
    per_ip_limit: int = 10
    per_ip_window_s: int = 3600
    daily_cap: int = 100


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: str = ""  # "" | "per_ip_limit" | "daily_cap"


class RateGuard:
    def __init__(self, config: GuardConfig = GuardConfig(), now=time.time):
        self._config = config
        self._now = now
        self._per_ip: dict[str, deque] = defaultdict(deque)
        self._global: deque = deque()

    def _prune(self, dq: deque, window_s: float, current: float):
        while dq and current - dq[0] > window_s:
            dq.popleft()

    def check(self, client_ip: str) -> GuardResult:
        """Record one request attempt and say whether it's allowed.
        client_ip must be the real transport-layer IP - never a
        client-supplied header - or this guarantees nothing."""
        current = self._now()

        day_bucket = self._global
        self._prune(day_bucket, 24 * 3600, current)
        if len(day_bucket) >= self._config.daily_cap:
            return GuardResult(allowed=False, reason="daily_cap")

        ip_bucket = self._per_ip[client_ip]
        self._prune(ip_bucket, self._config.per_ip_window_s, current)
        if len(ip_bucket) >= self._config.per_ip_limit:
            return GuardResult(allowed=False, reason="per_ip_limit")

        ip_bucket.append(current)
        day_bucket.append(current)
        return GuardResult(allowed=True)
