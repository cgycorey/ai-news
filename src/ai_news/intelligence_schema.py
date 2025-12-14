"""Create database schema for intelligence layer."""

import sqlite3
from datetime import datetime
from pathlib import Path

def create_intelligence_schema(db_path: str) -> bool:
    """Create all required tables for the intelligence layer."""
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Create entities table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    entity_type TEXT NOT NULL,
                    description TEXT,
                    aliases TEXT,  -- JSON array
                    metadata TEXT,     -- JSON object
                    confidence_score REAL DEFAULT 0.0,
                    mention_count INTEGER DEFAULT 0,
                    last_seen TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CHECK (
                        confidence_score >= 0.0 AND confidence_score <= 1.0
                    )
                )
            """)
            
            # Create topics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    keywords TEXT,  -- JSON array
                    topic_cluster_id INTEGER,
                    weight REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create article_entities table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS article_entities (
                    id INTEGER PRIMARY KEY,
                    article_id INTEGER NOT NULL,
                    entity_id INTEGER NOT NULL,
                    relevance_score REAL DEFAULT 1.0,
                    mention_positions TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
            
            # Create entity_mentions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_mentions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER NOT NULL,
                    entity_id INTEGER NOT NULL,
                    mention_count INTEGER DEFAULT 1,
                    sentiment_score REAL,
                    context_snippets TEXT,  -- JSON array
                    confidence_score REAL DEFAULT 0.0,
                    mention_positions TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
            
            # Create product_ideas table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS product_ideas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    problem_statement TEXT,
                    target_market TEXT,
                    key_features TEXT,  -- JSON array
                    tech_stack TEXT,      -- JSON array
                    similar_products TEXT,    -- JSON array
                    market_opportunity TEXT,
                    confidence_score REAL DEFAULT 0.0,
                    source_entities TEXT,    -- JSON array
                    source_articles TEXT,    -- JSON array
                    generated_by TEXT,
                    metadata TEXT,       -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CHECK (
                        confidence_score >= 0.0 AND confidence_score <= 1.0
                    )
                )
            """)
            
            # Create competitive_analysis table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS competitive_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    focus_entity_id INTEGER,
                    competitor_entities TEXT,  -- JSON array
                    analysis_type TEXT,
                    insights TEXT,
                    key_findings TEXT,    -- JSON array
                    opportunities TEXT,      -- JSON array
                    threats TEXT,         -- JSON array
                    strengths TEXT,        -- JSON array
                    weaknesses TEXT,       -- JSON array
                    market_position TEXT,
                    confidence_score REAL DEFAULT 0.0,
                    data_sources TEXT,       -- JSON array
                    generated_by TEXT,
                    metadata TEXT,           -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CHECK (
                        confidence_score >= 0.0 AND confidence_score <= 1.0
                    )
                )
            """)
            
            # Create entity_cache table for text processing
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_cache (
                    text_hash TEXT PRIMARY KEY,
                    processed_data TEXT,     -- JSON ProcessedText object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_confidence ON entities (confidence_score DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_mention_count ON entities (mention_count DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_updated_at ON entities (updated_at DESC)")
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topics_weight ON topics (weight DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topics_name ON topics (name ASC)")
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_product_ideas_confidence ON product_ideas (confidence_score DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_competitive_analysis_date ON competitive_analysis (created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_article_entities_article_id ON article_entities (article_id, entity_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity_id ON entity_mentions (entity_id, article_id, created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_mentions_sentiment_score ON entity_mentions (sentiment_score DESC)")
            
            print("✅ Intelligence layer database schema created successfully")
            return True
            
    except sqlite3.Error as e:
        print(f"❌ Error creating intelligence schema: {e}")
        return False

def add_intelligence_columns_to_main_articles():
    """Add intelligence layer columns to existing articles table."""
    try:
        with sqlite3.connect("ai_news.db") as conn:
            # Check if columns already exist
            cursor = conn.execute("PRAGMA table_info(articles)")
            columns = [row[1] for row in cursor.fetchall()]
            existing_columns = set(columns)
            
            # Add new columns if they don't exist
            new_columns = [
                ('entities', 'TEXT', 'JSON array'),
                ('sentiment_score', 'REAL DEFAULT 0.0'),
                ('classification', 'TEXT'),
                ('trend_score', 'REAL DEFAULT 0.0')
            ]
            
            for col_name, col_type in new_columns:
                if col_name not in existing_columns:
                    cursor.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_type}")
                    print(f"  ✓ Added column: {col_name}")
                else:
                    print(f"  ✅ Column {col_name} already exists")
            
            conn.commit()
            print("✅ Added intelligence columns to articles table")
            return True
            
    except sqlite3.Error as e:
        print(f"❌ Error adding columns: {e}")
        return False

if __name__ == "__main__":
    # Test the schema creation
    db_path = "ai_news.db"
    
    # Remove existing database if it exists
    if Path(db_path).exists():
        Path(db_path).unlink()
    
    # Create fresh database with intelligence schema
    success = create_intelligence_schema(db_path)
    
    if success:
        print("✅ Intelligence database schema created")
        
        # Add intelligence columns to main articles
        add_intelligence_columns_to_main_articles()
        
        print("✅ Intelligence columns added to articles table")
        print("\n✅ Intelligence layer ready for use")
    else:
        print("❌ Failed to create intelligence database schema")

