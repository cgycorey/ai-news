"""Time-based cache for spaCy digest analysis results."""

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any


class DigestCache:
    """Cache spaCy analysis results with TTL expiration."""

    DEFAULT_TTL_HOURS = 6  # Configurable

    def __init__(self, db_path: str = "ai_news.db", ttl_hours: Optional[int] = None):
        """
        Initialize cache with optional cleanup.

        Args:
            db_path: Path to SQLite database
            ttl_hours: Time-to-live in hours (default: DEFAULT_TTL_HOURS)
        """
        self.db_path = db_path
        self.ttl_hours = ttl_hours or self.DEFAULT_TTL_HOURS
        self._init_cache_table()
        self._cleanup_expired()

    def _init_cache_table(self):
        """Create cache table if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS digest_spacy_cache (
                    topics_key TEXT PRIMARY KEY,
                    results_json TEXT,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)

    def _make_key(self, topics: List[str], days: int) -> str:
        """
        Generate cache key from topics and days.

        Uses MD5 hash for consistent keys regardless of topic order.
        """
        # Sort topics for consistent keys
        normalized = json.dumps(sorted(topics) + [days])
        return hashlib.md5(normalized.encode()).hexdigest()

    def _generate_cache_key(self, topics: List[str], days: int) -> str:
        """
        Public alias for _make_key for testing purposes.

        Args:
            topics: List of topic keywords
            days: Number of days for digest

        Returns:
            Cache key string
        """
        return self._make_key(topics, days)
        """
        Generate cache key from topics and days.

        Uses MD5 hash for consistent keys regardless of topic order.
        """
        # Sort topics for consistent keys
        normalized = json.dumps(sorted(topics) + [days])
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, topics: List[str], days: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached results if valid.

        Args:
            topics: List of topic keywords
            days: Number of days for digest

        Returns:
            Cached results dict if valid and not expired, None otherwise
        """
        key = self._make_key(topics, days)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT results_json, expires_at FROM digest_spacy_cache WHERE topics_key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                if row:
                    expires_at = datetime.fromisoformat(row[1])
                    if datetime.now() < expires_at:
                        return json.loads(row[0])
        except (sqlite3.Error, json.JSONDecodeError, ValueError) as e:
            # Log error but don't crash - treat as cache miss
            print(f"Cache retrieval error: {e}")
        return None

    def set(self, topics: List[str], days: int, results: Dict[str, Any]):
        """
        Store analysis results with expiration.

        Args:
            topics: List of topic keywords
            days: Number of days for digest
            results: Analysis results to cache
        """
        key = self._make_key(topics, days)

        # Use results' generated_at if available, otherwise use now
        created_at = datetime.now()
        if 'generated_at' in results:
            try:
                created_at = datetime.fromisoformat(results['generated_at'])
            except (ValueError, TypeError):
                pass  # Use current time if parsing fails

        expires_at = created_at + timedelta(hours=self.ttl_hours)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO digest_spacy_cache VALUES (?, ?, ?, ?)",
                    (key, json.dumps(results), created_at.isoformat(), expires_at.isoformat())
                )
        except sqlite3.Error as e:
            # Log error but don't crash - cache failure is non-critical
            print(f"Cache storage error: {e}")

    def invalidate(self, topics: List[str], days: int):
        """
        Invalidate specific cache entry.

        Args:
            topics: List of topic keywords
            days: Number of days for digest
        """
        key = self._make_key(topics, days)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM digest_spacy_cache WHERE topics_key = ?", (key,))
        except sqlite3.Error as e:
            print(f"Cache invalidation error: {e}")

    def clear(self):
        """Clear all cache entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM digest_spacy_cache")
        except sqlite3.Error as e:
            print(f"Cache clear error: {e}")

    def cleanup_expired(self):
        """Remove expired entries from cache (public method for testing)."""
        self._cleanup_expired()

    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM digest_spacy_cache WHERE expires_at < ?", (datetime.now().isoformat(),))
        except sqlite3.Error as e:
            print(f"Cache cleanup error: {e}")

    def put(self, topics: List[str], days: int, results: Dict[str, Any]):
        """
        Alias for set() - store analysis results with expiration.

        Args:
            topics: List of topic keywords
            days: Number of days for digest
            results: Analysis results to cache
        """
        self.set(topics, days, results)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache size and expiration info
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*), COUNT(CASE WHEN expires_at < ? THEN 1 END), MIN(created_at), MAX(created_at)
                    FROM digest_spacy_cache
                """, (datetime.now().isoformat(),))
                total, expired, oldest, newest = cursor.fetchone()
                return {
                    'total_entries': total,
                    'expired_entries': expired,
                    'valid_entries': total - expired if total else 0,
                    'oldest_entry': oldest,
                    'newest_entry': newest,
                    'ttl_hours': self.ttl_hours
                }
        except sqlite3.Error:
            return {'total_entries': 0, 'expired_entries': 0, 'valid_entries': 0, 'ttl_hours': self.ttl_hours}
