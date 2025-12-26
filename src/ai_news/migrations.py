"""Database migration management."""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any


class MigrationManager:
    """Manages database schema migrations."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
    
    def get_current_version(self) -> int:
        """Get the current database schema version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
                return result[0] if result and result[0] else 0
        except sqlite3.Error:
            return 0
    
    def run_migrations(self, target_version: Optional[int] = None) -> bool:
        """Run pending migrations to reach target version."""
        current_version = self.get_current_version()
        
        if target_version is None:
            target_version = self.get_latest_version()
        
        if current_version >= target_version:
            print(f"Database is already at version {current_version}")
            return True
        
        print(f"Migrating database from version {current_version} to {target_version}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Ensure schema_version table exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Run migrations sequentially
                for version in range(current_version + 1, target_version + 1):
                    migration = self._get_migration(version)
                    if migration:
                        print(f"Applying migration {version}: {migration['description']}")
                        conn.executescript(migration['sql'])
                        conn.execute(
                            "INSERT INTO schema_version (version) VALUES (?)",
                            (version,)
                        )
                        print(f"Applied migration {version}")
                    else:
                        print(f"No migration found for version {version}")
                        return False
                
                print(f"Successfully migrated to version {target_version}")
                return True
                
        except sqlite3.Error as e:
            print(f"Migration failed: {e}")
            return False
    
    def get_latest_version(self) -> int:
        """Get the latest available migration version."""
        return max(self._get_available_migrations().keys())
    
    def _get_available_migrations(self) -> Dict[int, Dict[str, str]]:
        """Get all available migrations."""
        return {
            1: {
                "description": "Initialize base schema with articles table",
                "sql": """
                    -- Base articles table (already exists in most databases)
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
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_url ON articles(url);
                    CREATE INDEX IF NOT EXISTS idx_published_at ON articles(published_at);
                    CREATE INDEX IF NOT EXISTS idx_ai_relevant ON articles(ai_relevant);
                """
            },
            2: {
                "description": "Add intelligence layer tables",
                "sql": """
                    -- Entities table
                    CREATE TABLE IF NOT EXISTS entities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        entity_type TEXT NOT NULL CHECK (entity_type IN ('company', 'product', 'technology', 'person')),
                        description TEXT,
                        aliases TEXT,  -- JSON array
                        metadata TEXT,  -- JSON object
                        confidence_score REAL DEFAULT 0.0 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(name, entity_type)
                    );
                    
                    -- Topics table
                    CREATE TABLE IF NOT EXISTS topics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT,
                        keywords TEXT,  -- JSON array
                        topic_cluster_id INTEGER,
                        weight REAL DEFAULT 0.0 CHECK (weight >= 0.0),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (topic_cluster_id) REFERENCES topics(id) ON DELETE SET NULL
                    );
                    
                    -- Article-Entities many-to-many relationship
                    CREATE TABLE IF NOT EXISTS article_entities (
                        article_id INTEGER NOT NULL,
                        entity_id INTEGER NOT NULL,
                        relevance_score REAL DEFAULT 1.0 CHECK (relevance_score >= 0.0),
                        mention_positions TEXT,  -- JSON array of character positions
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (article_id, entity_id),
                        FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
                        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
                    );
                    
                    -- Entity mentions table
                    CREATE TABLE IF NOT EXISTS entity_mentions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id INTEGER NOT NULL,
                        entity_id INTEGER NOT NULL,
                        mention_count INTEGER DEFAULT 1 CHECK (mention_count > 0),
                        sentiment_score REAL CHECK (sentiment_score >= -1.0 AND sentiment_score <= 1.0),
                        context_snippets TEXT,  -- JSON array of text snippets
                        confidence_score REAL DEFAULT 0.0 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
                        mention_positions TEXT,  -- JSON array
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
                        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
                    );
                    
                    -- Academic tables removed (product_ideas, competitive_analysis)
                    -- Focus on practical business intelligence:
                    -- articles, entities, topics, entity_mentions only
                    
                    -- Create indexes for performance
                    CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
                    CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
                    CREATE INDEX IF NOT EXISTS idx_entities_confidence ON entities(confidence_score);
                    
                    CREATE INDEX IF NOT EXISTS idx_topics_weight ON topics(weight);
                    CREATE INDEX IF NOT EXISTS idx_topics_cluster ON topics(topic_cluster_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_article_entities_entity ON article_entities(entity_id);
                    CREATE INDEX IF NOT EXISTS idx_article_entities_relevance ON article_entities(relevance_score);
                    
                    CREATE INDEX IF NOT EXISTS idx_entity_mentions_article ON entity_mentions(article_id);
                    CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
                    CREATE INDEX IF NOT EXISTS idx_entity_mentions_sentiment ON entity_mentions(sentiment_score);
                    CREATE INDEX IF NOT EXISTS idx_entity_mentions_created ON entity_mentions(created_at);
                    
                    -- Academic feature indexes removed
                    -- Focus on business intelligence indexes only
                """
            },
            3: {
                "description": "Add region column and index to articles table",
                "sql": """
                    -- Add region column to articles table if it doesn't exist
                    -- SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we check first
                    -- This migration is safe to run multiple times
                    ALTER TABLE articles ADD COLUMN region TEXT DEFAULT 'global';
                    
                    -- Create region index if it doesn't exist
                    CREATE INDEX IF NOT EXISTS idx_region ON articles(region);
                """
            },
            5: {
                "description": "Add article auto-tagging table for entity extraction",
                "sql": """
                    -- Article auto-tagging table
                    -- Stores entity tags extracted from articles during collection
                    CREATE TABLE IF NOT EXISTS article_entity_tags (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id INTEGER NOT NULL,
                        entity_text TEXT NOT NULL,
                        entity_type TEXT NOT NULL CHECK (entity_type IN ('company', 'product', 'technology', 'person')),
                        confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
                        source TEXT NOT NULL CHECK (source IN ('spacy', 'pattern', 'known', 'discovered')),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(article_id, entity_text, entity_type),
                        FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
                    );
                    
                    -- Indexes for performance
                    CREATE INDEX IF NOT EXISTS idx_article_entity_tags_article_id ON article_entity_tags(article_id);
                    CREATE INDEX IF NOT EXISTS idx_article_entity_tags_entity_text ON article_entity_tags(entity_text);
                    CREATE INDEX IF NOT EXISTS idx_article_entity_tags_entity_type ON article_entity_tags(entity_type);
                    CREATE INDEX IF NOT EXISTS idx_article_entity_tags_confidence ON article_entity_tags(confidence);
                    CREATE INDEX IF NOT EXISTS idx_article_entity_tags_source ON article_entity_tags(source);
                """
            },
            6: {
                "description": "Add dynamic feed discovery cache tables",
                "sql": """
                    -- Discovered feeds cache table
                    CREATE TABLE IF NOT EXISTS discovered_feeds (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic VARCHAR(255) NOT NULL,
                        feed_url TEXT NOT NULL,
                        title TEXT,
                        description TEXT,
                        relevance_score REAL,
                        intersection_score REAL,
                        validated BOOLEAN DEFAULT 0,
                        discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP,
                        article_count INTEGER DEFAULT 0,
                        UNIQUE(topic, feed_url)
                    );
                    
                    -- Indexes for discovered_feeds
                    CREATE INDEX IF NOT EXISTS idx_discovered_feeds_topic ON discovered_feeds(topic);
                    CREATE INDEX IF NOT EXISTS idx_discovered_feeds_last_seen ON discovered_feeds(last_seen);
                    CREATE INDEX IF NOT EXISTS idx_discovered_feeds_relevance ON discovered_feeds(relevance_score);
                    
                    -- Search queries tracking table (for learning)
                    CREATE TABLE IF NOT EXISTS search_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_text TEXT NOT NULL,
                        search_engine VARCHAR(50),
                        results_count INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0,
                        last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Index for search_queries
                    CREATE INDEX IF NOT EXISTS idx_search_queries_text ON search_queries(query_text);
                    CREATE INDEX IF NOT EXISTS idx_search_queries_last_used ON search_queries(last_used);
                """
            }
        }
    
    def _get_migration(self, version: int) -> Optional[Dict[str, str]]:
        """Get a specific migration by version."""
        migrations = self._get_available_migrations()
        return migrations.get(version)
    
    def validate_schema(self) -> Dict[str, Any]:
        """Validate the current database schema."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_tables": [],
            "missing_indexes": [],
            "current_version": self.get_current_version()
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check required tables for current version
                current_version = self.get_current_version()
                required_tables = self._get_required_tables_for_version(current_version)
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}
                
                missing_tables = required_tables - existing_tables
                if missing_tables:
                    validation_result["valid"] = False
                    validation_result["missing_tables"] = list(missing_tables)
                    validation_result["errors"].append(f"Missing tables: {missing_tables}")
                
                # Check foreign key constraints
                if current_version >= 2:
                    cursor.execute("PRAGMA foreign_key_check")
                    fk_violations = cursor.fetchall()
                    if fk_violations:
                        validation_result["valid"] = False
                        validation_result["errors"].append(f"Foreign key violations: {fk_violations}")
                
        except sqlite3.Error as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Database validation error: {e}")
        
        return validation_result
    
    def _get_required_tables_for_version(self, version: int) -> set:
        """Get required tables for a specific version."""
        tables = {"schema_version", "articles"}
        
        if version >= 2:
            # Academic tables removed - focus on business intelligence
            tables.update({
                "entities", "topics", "article_entities", "entity_mentions"
                # "product_ideas", "competitive_analysis" removed - academic features
            })
        
        if version >= 4:
            tables.update({
                "discovered_feeds", "search_queries"
            })
        
        return tables
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the current database."""
        if backup_path is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(self.db_path.parent / f"{self.db_path.stem}_backup_{timestamp}{self.db_path.suffix}")
        
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            print(f"Database backed up to: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Backup failed: {e}")
            raise
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get detailed migration status."""
        current_version = self.get_current_version()
        latest_version = self.get_latest_version()
        
        return {
            "current_version": current_version,
            "latest_version": latest_version,
            "needs_migration": current_version < latest_version,
            "available_migrations": list(self._get_available_migrations().keys()),
            "pending_migrations": list(range(current_version + 1, latest_version + 1)),
            "validation": self.validate_schema()
        }


def migrate_database(db_path: str, target_version: Optional[int] = None, 
                    backup_before: bool = True) -> bool:
    """Convenience function to migrate a database.
    
    Args:
        db_path: Path to the database file
        target_version: Target version to migrate to (defaults to latest)
        backup_before: Whether to create a backup before migration
    
    Returns:
        True if migration was successful, False otherwise
    """
    manager = MigrationManager(db_path)
    
    # Create backup if requested
    if backup_before and manager.get_current_version() > 0:
        try:
            manager.backup_database()
        except Exception as e:
            print(f"Warning: Backup failed: {e}")
    
    # Run migrations
    return manager.run_migrations(target_version)


def validate_database_schema(db_path: str) -> Dict[str, Any]:
    """Convenience function to validate database schema."""
    manager = MigrationManager(db_path)
    return manager.validate_schema()


def get_database_migration_status(db_path: str) -> Dict[str, Any]:
    """Convenience function to get migration status."""
    manager = MigrationManager(db_path)
    return manager.get_migration_status()