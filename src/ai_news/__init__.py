"""AI News Collector - Simple RSS-based news feeder with intelligence layer."""

__version__ = "0.1.0"

# Direct imports for immediate access
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