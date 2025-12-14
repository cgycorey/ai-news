"""AI News Collector - Simple RSS-based news feeder with intelligence layer."""

__version__ = "0.1.0"

# Simplified approach - import functions for lazy loading
# This avoids import-time overhead while maintaining API compatibility

def _import_database_classes():
    """Import database classes when needed."""
    from .database import (
        Article,
        Entity,
        Topic,
        EntityMention,
        Database
    )
    return Article, Entity, Topic, EntityMention, Database

def _import_intelligence_classes():
    """Import intelligence classes when needed."""
    from .intelligence_db import IntelligenceDB
    return IntelligenceDB

def _import_migration_classes():
    """Import migration classes when needed."""
    from .migrations import (
        MigrationManager,
        migrate_database,
        validate_database_schema,
        get_database_migration_status
    )
    return MigrationManager, migrate_database, validate_database_schema, get_database_migration_status

# Direct imports for immediate access - much simpler and more reliable
from .database import (
    Article,
    Database
)

from .intelligence_db import IntelligenceDB

from .migrations import (
    MigrationManager,
    migrate_database,
    validate_database_schema,
    get_database_migration_status
)

__all__ = [
    # Core models
    "Article",
    "Entity", 
    "Topic",
    "EntityMention",
    "Database",
    
    # Intelligence layer
    "IntelligenceDB",
    
    # Migrations
    "MigrationManager",
    "migrate_database",
    "validate_database_schema", 
    "get_database_migration_status"
]