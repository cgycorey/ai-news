#!/usr/bin/env python3
"""Initialize database with all required tables for entity extraction."""

import sqlite3
import sys
from pathlib import Path

def init_database(db_path: str = "data/production/ai_news.db"):
    """Initialize database with complete schema."""
    
    db_path_obj = Path(db_path)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if db_path_obj.exists():
        db_path_obj.unlink()
        print(f"Removed existing database: {db_path}")
    
    conn = sqlite3.connect(str(db_path_obj))
    cursor = conn.cursor()
    
    print(f"Creating database: {db_path}")
    
    # Articles table
    cursor.execute("""
        CREATE TABLE articles (
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
    print("✓ Created articles table")
    
    # Metadata table
    cursor.execute("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✓ Created metadata table")
    
    # Article entity tags table (for auto-tagging)
    cursor.execute("""
        CREATE TABLE article_entity_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            entity_text TEXT NOT NULL,
            entity_type TEXT NOT NULL CHECK (entity_type IN ('company', 'product', 'technology', 'person')),
            confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            source TEXT NOT NULL CHECK (source IN ('spacy', 'pattern', 'known', 'discovered')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(article_id, entity_text, entity_type),
            FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
        )
    """)
    print("✓ Created article_entity_tags table")
    
    # Indexes for article_entity_tags
    cursor.execute("CREATE INDEX idx_aet_article_id ON article_entity_tags(article_id)")
    cursor.execute("CREATE INDEX idx_aet_entity_text ON article_entity_tags(entity_text)")
    cursor.execute("CREATE INDEX idx_aet_entity_type ON article_entity_tags(entity_type)")
    print("✓ Created indexes for article_entity_tags")
    
    # Entities table (for entity manager)
    cursor.execute("""
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            normalized_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            description TEXT DEFAULT '',
            aliases TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            confidence REAL DEFAULT 0.8,
            confidence_score REAL DEFAULT 0.8,
            mention_count INTEGER DEFAULT 0,
            last_seen TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✓ Created entities table")
    
    # Schema version
    cursor.execute("""
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("INSERT INTO schema_version (version, description) VALUES (6, 'Complete schema with entity support')")
    print("✓ Created schema_version table")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Database initialized successfully!")
    print(f"   Location: {Path(db_path).absolute()}")
    print(f"   Tables: articles, metadata, article_entity_tags, entities, schema_version")
    print(f"\n✅ Auto-tagging is enabled by default for new articles")
    print(f"✅ Entity-aware digest will work once articles are tagged")
    
    return 0

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/production/ai_news.db"
    sys.exit(init_database(db_path))
