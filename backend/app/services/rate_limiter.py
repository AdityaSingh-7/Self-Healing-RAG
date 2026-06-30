"""
services/rate_limiter.py — Rate Limiting

Simple in-memory rate limiter using sliding window.
Limits requests per user to prevent abuse.

Default: 30 requests per minute per user.
"""

import time
from collections import defaultdict


class RateLimiter:
    """
    Sliding window rate limiter.

    Tracks request timestamps per user and rejects requests
    that exceed the configured rate.

    Usage:
        limiter = RateLimiter(max_requests=30, window_seconds=60)

        if not limiter.allow("user123"):
            raise HTTPException(429, "Rate limit exceeded")
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def allow(self, user_id: str) -> bool:
        """
        Check if a request is allowed for this user.

        Returns True if allowed, False if rate limit exceeded.
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Remove expired timestamps
        self._requests[user_id] = [
            t for t in self._requests[user_id] if t > window_start
        ]

        # Check if under limit
        if len(self._requests[user_id]) >= self.max_requests:
            return False

        # Record this request
        self._requests[user_id].append(now)
        return True

    def remaining(self, user_id: str) -> int:
        """How many requests this user has left in the current window."""
        now = time.time()
        window_start = now - self.window_seconds
        recent = [t for t in self._requests[user_id] if t > window_start]
        return max(0, self.max_requests - len(recent))

    def reset_time(self, user_id: str) -> float:
        """Seconds until the oldest request in the window expires."""
        if not self._requests[user_id]:
            return 0
        oldest = min(self._requests[user_id])
        return max(0, (oldest + self.window_seconds) - time.time())


# Global instances for different endpoint groups
query_limiter = RateLimiter(max_requests=30, window_seconds=60)  # 30 queries/min
ingest_limiter = RateLimiter(max_requests=10, window_seconds=60)  # 10 uploads/min
