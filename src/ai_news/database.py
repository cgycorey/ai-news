"""Database models for AI News."""

import sqlite3
import shutil
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Article:
    """Represents a news article."""
    id: Optional[int] = None
    title: str = ""
    content: str = ""
    summary: str = ""
    url: str = ""
    author: str = ""
    published_at: Optional[datetime] = None
    source_name: str = ""
    category: str = ""
    region: str = "global"
    ai_relevant: bool = False
    ai_keywords_found: Optional[List[str]] = None


@dataclass
class Entity:
    """Represents an entity (company, product, technology, person)."""
    id: Optional[int] = None
    name: str = ""
    entity_type: str = ""  # company, product, technology, person
    description: Optional[str] = None
    aliases: Optional[List[str]] = None  # Alternative names/spellings
    metadata: Optional[Dict[str, Any]] = None  # Additional structured data
    confidence_score: float = 0.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Topic:
    """Represents a trending topic or theme."""
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    topic_cluster_id: Optional[int] = None  # For hierarchical topics
    weight: float = 0.0  # Importance/trending score
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class EntityMention:
    """Represents a mention of an entity in an article."""
    id: Optional[int] = None
    article_id: int = 0
    entity_id: int = 0
    mention_count: int = 1
    sentiment_score: Optional[float] = None  # -1.0 to 1.0
    context_snippets: Optional[List[str]] = None  # Text snippets around mentions
    confidence_score: float = 0.0
    mention_positions: Optional[List[int]] = None  # Character positions in article
    created_at: Optional[datetime] = None


class Database:
    """Simple SQLite database for storing articles."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.init_database()

        # Automatic deduplication on first init
        self._auto_deduplicate()

    def _auto_deduplicate(self):
        """Automatically remove duplicate articles from database.

        This runs once on database initialization to clean up existing duplicates
        caused by tracking URLs. Uses canonical URL extraction.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if we've already run deduplication
                cursor = conn.execute("""
                    SELECT value FROM metadata WHERE key = 'deduplication_run'
                """).fetchone()

                if cursor:
                    # Already ran, skip
                    return

                # Find and remove duplicates
                logger.info("🔍 Running automatic deduplication...")

                # Get all articles with Bing tracking URLs
                cursor = conn.execute("""
                    SELECT id, url, title
                    FROM articles
                    WHERE url LIKE '%apiclick.aspx%'
                    ORDER BY id
                """).fetchall()

                if not cursor:
                    # No Bing articles, mark as done
                    conn.execute("""
                        INSERT INTO metadata (key, value) VALUES ('deduplication_run', '1')
                    """)
                    return

                # Group by canonical URL
                from urllib.parse import urlparse, parse_qs, unquote
                from collections import defaultdict

                canonical_groups = defaultdict(list)
                for article_id, url, title in cursor:
                    try:
                        parsed = urlparse(url)
                        if 'apiclick.aspx' in parsed.path:
                            params = parse_qs(parsed.query)
                            canonical = unquote(params.get('url', [url])[0])
                        else:
                            canonical = url
                        canonical_groups[canonical].append((article_id, title))
                    except:
                        canonical_groups[url].append((article_id, title))

                # Find duplicates (keep first, delete rest)
                duplicates_to_delete = []
                for canonical, articles in canonical_groups.items():
                    if len(articles) > 1:
                        # Sort by ID, keep oldest
                        articles.sort(key=lambda x: x[0])
                        # Keep first, mark others for deletion
                        duplicates_to_delete.extend([a[0] for a in articles[1:]])

                if duplicates_to_delete:
                    # Delete duplicates
                    for article_id in duplicates_to_delete:
                        conn.execute("DELETE FROM articles WHERE id = ?", (article_id,))

                    logger.info(f"✅ Auto-removed {len(duplicates_to_delete)} duplicate articles")

                # Mark deduplication as complete
                conn.execute("""
                    INSERT INTO metadata (key, value) VALUES ('deduplication_run', '1')
                """)
                conn.commit()

        except Exception as e:
            logger.warning(f"Auto-deduplication failed: {e}")

    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    summary TEXT,
                    url TEXT UNIQUE NOT NULL,
                    author TEXT,
                    published_at TIMESTAMP,
                    source_name TEXT,
                    category TEXT,
                    region TEXT DEFAULT 'global',
                    ai_relevant BOOLEAN DEFAULT FALSE,
                    ai_keywords_found TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_url ON articles(url)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_published_at ON articles(published_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ai_relevant ON articles(ai_relevant)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_region ON articles(region)
            """)
    
    def save_article(self, article: Article) -> Optional[int]:
        """Save an article to the database with automatic deduplication."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Extract canonical URL to prevent duplicates
                canonical_url = self._extract_canonical_url(article.url)

                # Check if article already exists (by canonical URL)
                existing = conn.execute(
                    "SELECT id FROM articles WHERE url = ?", (canonical_url,)
                ).fetchone()

                if existing:
                    # Article already exists, skip
                    logger.debug(f"Duplicate article skipped: {article.title[:50]}...")
                    return existing[0]

                # Save new article with canonical URL
                cursor = conn.execute("""
                    INSERT INTO articles 
                    (title, content, summary, url, author, published_at, 
                     source_name, category, region, ai_relevant, ai_keywords_found)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.title,
                    article.content,
                    article.summary,
                    canonical_url,  # Use canonical URL
                    article.author,
                    article.published_at,
                    article.source_name,
                    article.category,
                    article.region,
                    article.ai_relevant,
                    ",".join(article.ai_keywords_found or [])
                ))
                return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error saving article: {e}")
            return None

    def _extract_canonical_url(self, url: str) -> str:
        """Extract canonical URL from tracking/redirect URLs.

        Handles:
        - Bing News: apiclick.aspx?tid=...&url=...
        - Other redirect services

        Args:
            url: URL that might contain tracking parameters

        Returns:
            Canonical URL (actual article URL)
        """
        if not url:
            return url

        try:
            from urllib.parse import urlparse, parse_qs, unquote

            parsed = urlparse(url)

            # Handle Bing News redirects
            if 'apiclick.aspx' in parsed.path:
                params = parse_qs(parsed.query)
                canonical = params.get('url', [url])[0]
                return unquote(canonical)

            # Handle other common redirect patterns
            # Add more as needed

            return url

        except Exception as e:
            logger.warning(f"Failed to extract canonical URL from {url}: {e}")
            return url

    def get_article_by_id(self, article_id: int) -> Optional[Article]:
        """Get article by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
            
            if row:
                return Article(
                    id=row['id'],
                    title=row['title'],
                    content=row['content'],
                    summary=row['summary'],
                    url=row['url'],
                    author=row['author'] or "",
                    published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                    source_name=row['source_name'] or "",
                    category=row['category'] or "",
                    region=row['region'] or "global",
                    ai_relevant=bool(row['ai_relevant']),
                    ai_keywords_found=row['ai_keywords_found'].split(",") if row['ai_keywords_found'] else []
                )
            return None

    def add_article(self, article: Article) -> bool:
        """Add article to database, returns True if added (new), False if exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO articles 
                    (title, content, summary, url, author, published_at, 
                     source_name, category, region, ai_relevant, ai_keywords_found)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.title,
                    article.content,
                    article.summary,
                    article.url,
                    article.author,
                    article.published_at,
                    article.source_name,
                    article.category,
                    article.region,
                    article.ai_relevant,
                    ",".join(article.ai_keywords_found or [])
                ))
                return conn.total_changes > 0
        except sqlite3.Error as e:
            print(f"Error adding article: {e}")
            return False
    
    def get_article_count(self) -> int:
        """Get the total number of articles in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("SELECT COUNT(*) FROM articles").fetchone()
                return result[0] if result else 0
        except sqlite3.Error as e:
            print(f"Error getting article count: {e}")
            return 0


    def get_articles(self, limit: int = 20, ai_only: bool = False, region: Optional[str] = None) -> List[Article]:
        """Get articles from database with optional region filtering and fallback."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Base query
            query = "SELECT * FROM articles WHERE 1=1"
            params = []
            
            if ai_only:
                query += " AND ai_relevant = 1"
            
            # Regional filtering with fallback mechanism
            if region:
                region = region.lower()
                if region in ['us', 'eu', 'uk', 'apac']:
                    # Try specific region first, then fallback to global if no results
                    region_query = query + f" AND region = ? ORDER BY published_at DESC LIMIT ?"
                    region_params = params + [region, limit]
                    
                    region_rows = conn.execute(region_query, region_params).fetchall()
                    
                    # If no results for specific region, fallback to global
                    if not region_rows:
                        fallback_query = query + " AND region = 'global' ORDER BY published_at DESC LIMIT ?"
                        fallback_params = params + [limit]
                        rows = conn.execute(fallback_query, fallback_params).fetchall()
                    else:
                        rows = region_rows
                else:
                    # For 'global' or other regions, use exact match
                    query += " AND region = ?"
                    params.append(region)
                    query += " ORDER BY published_at DESC LIMIT ?"
                    params.append(limit)
                    rows = conn.execute(query, params).fetchall()
            else:
                # No region filter
                query += " ORDER BY published_at DESC LIMIT ?"
                params.append(limit)
                rows = conn.execute(query, params).fetchall()
            
            return [
                Article(
                    id=row['id'],
                    title=row['title'],
                    content=row['content'],
                    summary=row['summary'],
                    url=row['url'],
                    author=row['author'] or "",
                    published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                    source_name=row['source_name'] or "",
                    category=row['category'] or "",
                    region=row['region'] or "global",
                    ai_relevant=bool(row['ai_relevant']),
                    ai_keywords_found=row['ai_keywords_found'].split(",") if row['ai_keywords_found'] else []
                )
                for row in rows
            ]
    
    def search_articles(self, query: str, limit: int = 20, region: Optional[str] = None) -> List[Article]:
        """Search articles with optional region filtering and fallback."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            search_query = f"%{query}%"
            
            # Regional filtering with fallback mechanism
            if region:
                region = region.lower()
                if region in ['us', 'eu', 'uk', 'apac']:
                    # Try specific region first
                    region_sql = """
                        SELECT * FROM articles 
                        WHERE (title LIKE ? OR content LIKE ? OR summary LIKE ?)
                        AND region = ?
                        ORDER BY published_at DESC LIMIT ?
                    """
                    region_params = [search_query, search_query, search_query, region, limit]
                    
                    region_rows = conn.execute(region_sql, region_params).fetchall()
                    
                    # If no results for specific region, fallback to global
                    if not region_rows:
                        fallback_sql = """
                            SELECT * FROM articles 
                            WHERE (title LIKE ? OR content LIKE ? OR summary LIKE ?)
                            AND region = 'global'
                            ORDER BY published_at DESC LIMIT ?
                        """
                        fallback_params = [search_query, search_query, search_query, limit]
                        rows = conn.execute(fallback_sql, fallback_params).fetchall()
                    else:
                        rows = region_rows
                else:
                    # For 'global' or other regions, use exact match
                    sql = """
                        SELECT * FROM articles 
                        WHERE (title LIKE ? OR content LIKE ? OR summary LIKE ?)
                    """
                    params = [search_query, search_query, search_query]
                    
                    if region:
                        sql += " AND region = ?"
                        params.append(region.lower())
                    
                    sql += " ORDER BY published_at DESC LIMIT ?"
                    params.append(limit)
                    rows = conn.execute(sql, params).fetchall()
            else:
                # No region filter
                sql = """
                    SELECT * FROM articles 
                    WHERE (title LIKE ? OR content LIKE ? OR summary LIKE ?)
                    ORDER BY published_at DESC LIMIT ?
                """
                params = [search_query, search_query, search_query, limit]
                rows = conn.execute(sql, params).fetchall()
            
            return [
                Article(
                    id=row['id'],
                    title=row['title'],
                    content=row['content'],
                    summary=row['summary'],
                    url=row['url'],
                    author=row['author'] or "",
                    published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
                    source_name=row['source_name'] or "",
                    category=row['category'] or "",
                    region=row['region'] or "global",
                    ai_relevant=bool(row['ai_relevant']),
                    ai_keywords_found=row['ai_keywords_found'].split(",") if row['ai_keywords_found'] else []
                )
                for row in rows
            ]
    
    def get_stats(self, region: Optional[str] = None) -> Dict[str, Any]:
        """Get database statistics with optional region filtering."""
        query = "SELECT COUNT(*), SUM(ai_relevant), COUNT(DISTINCT source_name) FROM articles"
        params = []
        
        if region:
            query += " WHERE region = ?"
            params.append(region.lower())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            total, ai_relevant, sources = cursor.fetchone()
            
            return {
                'total_articles': total or 0,
                'ai_relevant_articles': ai_relevant or 0,
                'sources_count': sources or 0,
                'ai_relevance_rate': f"{((ai_relevant or 0) / (total or 1) * 100):.1f}%" if total else "0%"
            }
    
    def cleanup_old_articles(self, days: int = 90, dry_run: bool = False) -> dict:
        """Remove articles older than specified number of days.
        
        Args:
            days: Remove articles older than this many days
            dry_run: If True, only report what would be deleted without actually deleting
            
        Returns:
            Dict with cleanup statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Count articles that would be deleted
            count_result = conn.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE published_at < ? AND published_at IS NOT NULL
            """, (cutoff_date.isoformat(),)).fetchone()
            
            articles_to_delete = count_result[0] if count_result else 0
            
            # Get detailed stats for reporting
            stats_query = """
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN ai_relevant = 1 THEN 1 END) as ai_relevant,
                    COUNT(DISTINCT source_name) as sources
                FROM articles 
                WHERE published_at < ? AND published_at IS NOT NULL
            """
            
            stats = conn.execute(stats_query, (cutoff_date.isoformat(),)).fetchone()
            
            result = {
                "articles_to_delete": articles_to_delete,
                "ai_relevant_to_delete": stats[1] if stats else 0,
                "sources_affected": stats[2] if stats else 0,
                "dry_run": dry_run
            }
            
            if not dry_run and articles_to_delete > 0:
                # Actually delete the articles
                conn.execute("""
                    DELETE FROM articles 
                    WHERE published_at < ? AND published_at IS NOT NULL
                """, (cutoff_date.isoformat(),))
                
                result["articles_deleted"] = conn.total_changes
                
            return result
    
    def remove_duplicate_articles(self, dry_run: bool = False) -> dict:
        """Find and remove duplicate articles based on URL similarity.
        
        Args:
            dry_run: If True, only report what would be deleted without actually deleting
            
        Returns:
            Dict with cleanup statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            # Find duplicates by exact URL match
            duplicate_query = """
                SELECT url, COUNT(*) as count, GROUP_CONCAT(id) as ids
                FROM articles 
                WHERE url != ''
                GROUP BY url 
                HAVING COUNT(*) > 1
            """
            
            duplicates = conn.execute(duplicate_query).fetchall()
            
            total_duplicates = 0
            articles_to_delete = []
            
            for url, count, ids in duplicates:
                article_ids = [int(id_str) for id_str in ids.split(',')]
                # Keep the oldest one, delete the rest
                keep_id = min(article_ids)
                delete_ids = [aid for aid in article_ids if aid != keep_id]
                articles_to_delete.extend(delete_ids)
                total_duplicates += count - 1
            
            result = {
                "duplicate_groups": len(duplicates),
                "articles_to_delete": len(articles_to_delete),
                "dry_run": dry_run
            }
            
            if not dry_run and articles_to_delete:
                # Delete duplicates (keep oldest)
                placeholders = ','.join('?' * len(articles_to_delete))
                conn.execute(f"""
                    DELETE FROM articles 
                    WHERE id IN ({placeholders})
                """, articles_to_delete)
                
                result["articles_deleted"] = conn.total_changes
                
            return result
    
    def remove_empty_articles(self, dry_run: bool = False) -> dict:
        """Remove articles with empty titles or content.
        
        Args:
            dry_run: If True, only report what would be deleted without actually deleting
            
        Returns:
            Dict with cleanup statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            # Find articles with empty or whitespace-only titles/summaries
            empty_query = """
                SELECT COUNT(*) FROM articles 
                WHERE (title IS NULL OR title = '' OR TRIM(title) = '')
                OR (summary IS NULL OR summary = '' OR TRIM(summary) = '')
            """
            
            count_result = conn.execute(empty_query).fetchone()
            articles_to_delete = count_result[0] if count_result else 0
            
            result = {
                "articles_to_delete": articles_to_delete,
                "dry_run": dry_run
            }
            
            if not dry_run and articles_to_delete > 0:
                conn.execute("""
                    DELETE FROM articles 
                    WHERE (title IS NULL OR title = '' OR TRIM(title) = '')
                    OR (summary IS NULL OR summary = '' OR TRIM(summary) = '')
                """)
                
                result["articles_deleted"] = conn.total_changes
                
            return result
    
    def optimize_database(self) -> dict:
        """Optimize database performance by vacuuming and analyzing.
        
        Returns:
            Dict with optimization results
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get database size before optimization
            size_before = self.db_path.stat().st_size
            
            # VACUUM to reclaim space and defragment
            conn.execute("VACUUM")
            
            # ANALYZE to update query planner statistics
            conn.execute("ANALYZE")
            
            # Get database size after optimization
            size_after = self.db_path.stat().st_size
            space_saved = size_before - size_after
            
            return {
                "vacuum_completed": True,
                "analyze_completed": True,
                "size_before_mb": round(size_before / (1024 * 1024), 2),
                "size_after_mb": round(size_after / (1024 * 1024), 2),
                "space_saved_mb": round(space_saved / (1024 * 1024), 2),
                "space_saved_percent": round((space_saved / size_before * 100), 2) if size_before > 0 else 0
            }
    
    def backup_database(self, backup_path: Optional[str] = None) -> dict:
        """Create a backup of the database.
        
        Args:
            backup_path: Path for backup file, if None uses timestamp
            
        Returns:
            Dict with backup results
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backup_ai_news_{timestamp}.db"
        
        backup_path = Path(backup_path)
        
        try:
            # Copy database file
            shutil.copy2(self.db_path, backup_path)
            
            size = backup_path.stat().st_size
            
            return {
                "backup_path": str(backup_path),
                "size_mb": round(size / (1024 * 1024), 2),
                "success": True
            }
            
        except Exception as e:
            return {
                "backup_path": str(backup_path),
                "success": False,
                "error": str(e)
            }
    
    def get_cleanup_preview(self) -> dict:
        """Get a preview of what cleanup operations would affect.
        
        Returns:
            Dict with cleanup preview statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get basic stats
            total_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            
            # Old articles (older than 90 days)
            cutoff_90 = datetime.now() - timedelta(days=90)
            old_90_days = conn.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE published_at < ? AND published_at IS NOT NULL
            """, (cutoff_90.isoformat(),)).fetchone()[0]
            
            # Old articles (older than 180 days)
            cutoff_180 = datetime.now() - timedelta(days=180)
            old_180_days = conn.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE published_at < ? AND published_at IS NOT NULL
            """, (cutoff_180.isoformat(),)).fetchone()[0]
            
            # Duplicates
            duplicates = conn.execute("""
                SELECT COUNT(*) - COUNT(DISTINCT url) FROM articles 
                WHERE url != ''
            """).fetchone()[0]
            
            # Empty articles
            empty_articles = conn.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE (title IS NULL OR title = '' OR TRIM(title) = '')
                OR (summary IS NULL OR summary = '' OR TRIM(summary) = '')
            """).fetchone()[0]
            
            # Database size
            db_size = self.db_path.stat().st_size
            
            return {
                "total_articles": total_articles,
                "articles_older_90_days": old_90_days,
                "articles_older_180_days": old_180_days,
                "duplicate_articles": duplicates,
                "empty_articles": empty_articles,
                "database_size_mb": round(db_size / (1024 * 1024), 2)
            }
    
    # ========== Feed Discovery Cache Methods ==========
    
    def check_feed_cache(self, topic: str, max_age_days: int = 7) -> List[Dict[str, Any]]:
        """Check if topic has cached feeds within max_age_days.
        
        Args:
            topic: Topic to search for
            max_age_days: Maximum age of cache in days (default 7)
        
        Returns:
            List of cached feed dicts, or empty list if not found/stale
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                query = """
                    SELECT * FROM discovered_feeds
                    WHERE topic = ? AND last_seen >= date('now', '-' || ? || ' days')
                    ORDER BY relevance_score DESC
                """
                cursor = conn.execute(query, (topic, max_age_days))
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error checking feed cache: {e}")
            return []
    
    def save_discovered_feeds(self, topic: str, feeds: List[Dict[str, Any]]) -> bool:
        """Save discovered feeds to cache.
        
        Args:
            topic: Topic name
            feeds: List of feed dictionaries with keys: url, title, description,
                   relevance_score, intersection_score, validated, article_count
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                for feed in feeds:
                    conn.execute("""
                        INSERT OR REPLACE INTO discovered_feeds
                        (topic, feed_url, title, description, relevance_score,
                         intersection_score, validated, last_seen, article_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                    """, (
                        topic,
                        feed.get('url', ''),
                        feed.get('title', ''),
                        feed.get('description', ''),
                        feed.get('relevance_score', 0.0),
                        feed.get('intersection_score', 0.0),
                        1 if feed.get('validated', False) else 0,
                        feed.get('article_count', 0)
                    ))
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error saving discovered feeds: {e}")
            return False
    
    def mark_feed_accessed(self, topic: str) -> bool:
        """Update last_seen timestamp for topic.
        
        Args:
            topic: Topic name
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE discovered_feeds
                    SET last_seen = CURRENT_TIMESTAMP
                    WHERE topic = ?
                """, (topic,))
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error marking feed accessed: {e}")
            return False
    
    def clear_stale_feeds(self, days: int = 30) -> int:
        """Remove feeds older than N days.
        
        Args:
            days: Number of days after which feeds are considered stale
        
        Returns:
            Number of feeds deleted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM discovered_feeds
                    WHERE last_seen < date('now', '-' || ? || ' days')
                """, (days,))
                conn.commit()
                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Error clearing stale feeds: {e}")
            return 0
    
    def get_cached_topics(self) -> List[str]:
        """Get list of all cached topics.
        
        Returns:
            List of unique topic names
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT topic FROM discovered_feeds
                    ORDER BY last_seen DESC
                """)
                return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error getting cached topics: {e}")
            return []
    
    def get_feed_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the feed cache.
        
        Returns:
            Dict with cache statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total cached topics
                total_topics = conn.execute("""
                    SELECT COUNT(DISTINCT topic) FROM discovered_feeds
                """).fetchone()[0]
                
                # Total cached feeds
                total_feeds = conn.execute("SELECT COUNT(*) FROM discovered_feeds").fetchone()[0]
                
                # Fresh feeds (< 7 days)
                fresh_feeds = conn.execute("""
                    SELECT COUNT(*) FROM discovered_feeds
                    WHERE last_seen >= date('now', '-7 days')
                """).fetchone()[0]
                
                # Stale feeds (> 30 days)
                stale_feeds = conn.execute("""
                    SELECT COUNT(*) FROM discovered_feeds
                    WHERE last_seen < date('now', '-30 days')
                """).fetchone()[0]
                
                return {
                    "total_cached_topics": total_topics,
                    "total_cached_feeds": total_feeds,
                    "fresh_feeds": fresh_feeds,
                    "stale_feeds": stale_feeds
                }
        except sqlite3.Error as e:
            logger.error(f"Error getting feed cache stats: {e}")
            return {}

    def remove_orphaned_entities(self, dry_run: bool = False) -> dict:
        """Remove entities that are not referenced by any articles.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Dict with orphaned counts by type
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Find orphaned entities (not in article_entities)
                orphaned = conn.execute("""
                    SELECT
                        e.entity_type,
                        COUNT(*) as count
                    FROM entities e
                    LEFT JOIN article_entities ae ON e.id = ae.entity_id
                    WHERE ae.entity_id IS NULL
                    GROUP BY e.entity_type
                """).fetchall()

                # Count by type
                by_type = {row['entity_type']: row['count'] for row in orphaned}
                total_orphaned = sum(by_type.values())

                if not dry_run and total_orphaned > 0:
                    # Delete orphaned entities
                    conn.execute("""
                        DELETE FROM entities
                        WHERE id IN (
                            SELECT e.id FROM entities e
                            LEFT JOIN article_entities ae ON e.id = ae.entity_id
                            WHERE ae.entity_id IS NULL
                        )
                    """)
                    conn.commit()
                    logger.info(f"Deleted {total_orphaned} orphaned entities")

                return {
                    'total_orphaned': total_orphaned,
                    'by_type': by_type
                }

        except sqlite3.Error as e:
            logger.error(f"Error removing orphaned entities: {e}")
            return {
                'total_orphaned': 0,
                'by_type': {}
            }
