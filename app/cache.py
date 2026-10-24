import time
import hashlib
from typing import Optional, Dict


class ResponseCache:
    """A simple in‑memory cache with TTL support."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Dict[str, object]] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str) -> str:
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[str]:
        key = self._make_key(query)
        entry = self._cache.get(key)
        if entry and (time.time() - entry["timestamp"] < self.ttl):
            self._hits += 1
            return entry["response"]
        # Miss or expired
        if entry:
            # Remove stale entry
            del self._cache[key]
        self._misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        key = self._make_key(query)
        self._cache[key] = {"response": response, "timestamp": time.time()}

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) * 100 if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_entries": len(self._cache),
        }
