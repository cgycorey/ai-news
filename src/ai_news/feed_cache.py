"""Feed cache management for dynamic feed discovery."""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from .database import Database

logger = logging.getLogger(__name__)


class FeedCache:
    """Manages persistent caching of discovered feeds."""
    
    CACHE_FRESHNESS_DAYS = 7  # Cache considered fresh for 7 days
    CACHE_EXPIRY_DAYS = 30    # Cache expires after 30 days
    
    def __init__(self, database: Database):
        """Initialize feed cache.
        
        Args:
            database: Database instance
        """
        self.db = database
    
    def check_cache(self, topic: str) -> Optional[List[Dict]]:
        """
        Return cached feeds if fresh (< CACHE_FRESHNESS_DAYS old).
        
        Args:
            topic: Topic to search for
        
        Returns:
            List of feed dicts if cache hit and fresh, None if cache miss or stale
        """
        cached = self.db.check_feed_cache(topic, self.CACHE_FRESHNESS_DAYS)
        
        if not cached:
            logger.debug(f"Cache miss for topic: {topic}")
            return None
        
        # Convert database rows to expected dict format
        feeds = [self._row_to_dict(row) for row in cached]
        logger.info(f"✅ Cache hit for '{topic}': {len(feeds)} feeds")
        
        # Mark as accessed
        self.mark_accessed(topic)
        
        return feeds
    
    def save_to_cache(self, topic: str, feeds: List[Dict]) -> bool:
        """Store discovered feeds for future instant access.
        
        Args:
            topic: Topic name
            feeds: List of feed dictionaries
        
        Returns:
            True if successful, False otherwise
        """
        if not feeds:
            logger.warning(f"No feeds to cache for topic: {topic}")
            return False
        
        success = self.db.save_discovered_feeds(topic, feeds)
        
        if success:
            logger.info(f"💾 Cached {len(feeds)} feeds for '{topic}'")
        else:
            logger.error(f"Failed to cache feeds for '{topic}'")
        
        return success
    
    def mark_accessed(self, topic: str) -> bool:
        """Update last_seen timestamp.
        
        Args:
            topic: Topic name
        
        Returns:
            True if successful, False otherwise
        """
        return self.db.mark_feed_accessed(topic)
    
    def clear_stale(self, days: Optional[int] = None) -> int:
        """Remove entries older than N days.
        
        Args:
            days: Days threshold (defaults to CACHE_EXPIRY_DAYS)
        
        Returns:
            Number of feeds deleted
        """
        if days is None:
            days = self.CACHE_EXPIRY_DAYS
        
        deleted = self.db.clear_stale_feeds(days)
        
        if deleted > 0:
            logger.info(f"🧹 Cleared {deleted} stale feed entries (> {days} days)")
        else:
            logger.debug("No stale entries to clear")
        
        return deleted
    
    def get_all_cached_topics(self) -> List[str]:
        """Get list of all cached topics.
        
        Returns:
            List of topic names
        """
        return self.db.get_cached_topics()
    
    def is_cache_fresh(self, topic: str) -> bool:
        """Check if topic has fresh cache.
        
        Args:
            topic: Topic name
        
        Returns:
            True if cache exists and is fresh, False otherwise
        """
        cached = self.db.check_feed_cache(topic, self.CACHE_FRESHNESS_DAYS)
        return cached is not None and len(cached) > 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        return self.db.get_feed_cache_stats()
    
    @staticmethod
    def _row_to_dict(row) -> Dict:
        """Convert database row to dictionary.
        
        Args:
            row: Database row (dict from sqlite3.Row)
        
        Returns:
            Standardized feed dictionary
        """
        return {
            'id': row.get('id'),
            'url': row.get('feed_url', ''),
            'title': row.get('title', ''),
            'description': row.get('description', ''),
            'relevance_score': row.get('relevance_score', 0.0),
            'intersection_score': row.get('intersection_score', 0.0),
            'validated': bool(row.get('validated', 0)),
            'discovered_at': row.get('discovered_at'),
            'last_seen': row.get('last_seen'),
            'last_updated': row.get('last_updated'),
            'article_count': row.get('article_count', 0)
        }
