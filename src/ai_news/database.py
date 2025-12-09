"""Database models for AI News."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from pathlib import Path


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
    ai_relevant: bool = False
    ai_keywords_found: Optional[List[str]] = None


class Database:
    """Simple SQLite database for storing articles."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.init_database()
    
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
                    ai_relevant BOOLEAN DEFAULT FALSE,
                    ai_keywords_found TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    
    def add_article(self, article: Article) -> bool:
        """Add article to database, returns True if added (new), False if exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO articles 
                    (title, content, summary, url, author, published_at, 
                     source_name, category, ai_relevant, ai_keywords_found)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.title,
                    article.content,
                    article.summary,
                    article.url,
                    article.author,
                    article.published_at,
                    article.source_name,
                    article.category,
                    article.ai_relevant,
                    ",".join(article.ai_keywords_found or [])
                ))
                return conn.total_changes > 0
        except sqlite3.Error as e:
            print(f"Error adding article: {e}")
            return False
    
    def get_articles(self, limit: int = 20, ai_only: bool = False) -> List[Article]:
        """Get articles from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM articles"
            params = []
            
            if ai_only:
                query += " WHERE ai_relevant = 1"
            
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
                    ai_relevant=bool(row['ai_relevant']),
                    ai_keywords_found=row['ai_keywords_found'].split(",") if row['ai_keywords_found'] else []
                )
                for row in rows
            ]
    
    def search_articles(self, query: str, limit: int = 20) -> List[Article]:
        """Search articles by title and content."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            rows = conn.execute("""
                SELECT * FROM articles 
                WHERE (title LIKE ? OR content LIKE ? OR summary LIKE ?)
                ORDER BY published_at DESC 
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", limit)).fetchall()
            
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
                    ai_relevant=bool(row['ai_relevant']),
                    ai_keywords_found=row['ai_keywords_found'].split(",") if row['ai_keywords_found'] else []
                )
                for row in rows
            ]
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            ai_relevant = conn.execute("SELECT COUNT(*) FROM articles WHERE ai_relevant = 1").fetchone()[0]
            sources = conn.execute("SELECT COUNT(DISTINCT source_name) FROM articles").fetchone()[0]
            
            return {
                "total_articles": total,
                "ai_relevant_articles": ai_relevant,
                "sources_count": sources,
                "ai_relevance_rate": f"{(ai_relevant/total*100):.1f}%" if total > 0 else "0%"
            }