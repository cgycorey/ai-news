"""Dynamic entity management system for AI news."""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, Counter
import hashlib

from .database import Database

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entity with metadata."""
    name: str
    entity_type: str = ""
    normalized_name: Optional[str] = None  # Normalized/lowercase version for matching
    confidence: float = 0.8
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    mention_count: int = 0
    last_seen: Optional[datetime] = None
    
    def __post_init__(self):
        """Generate normalized_name if not provided."""
        if self.normalized_name is None and self.name:
            # Normalize: lowercase, remove extra whitespace, remove special chars
            self.normalized_name = re.sub(r'[^a-z0-9\s]', '', self.name.lower().strip())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'entity_type': self.entity_type,
            'confidence': self.confidence,
            'aliases': self.aliases,
            'description': self.description,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'mention_count': self.mention_count,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create from dictionary."""
        # Handle datetime parsing
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if isinstance(data.get('last_seen'), str):
            data['last_seen'] = datetime.fromisoformat(data['last_seen'])
        
        return cls(**data)


@dataclass
class EntityPattern:
    """Pattern for entity extraction."""
    name: str
    pattern: str
    entity_type: str
    confidence: float
    description: str = ""
    examples: List[str] = field(default_factory=list)
    compiled_pattern: Optional[re.Pattern] = None

    def __post_init__(self):
        """Compile regex pattern."""
        try:
            self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
        except re.error as e:
            logger.warning(f"Invalid regex pattern {self.name}: {e}")


class EntityManager:
    """Manages entities dynamically with learning capabilities."""
    
    def __init__(self, entities_dir: str = "entities", db_path: str = "data/production/ai_news.db"):
        """Initialize entity manager.
        
        Args:
            entities_dir: Directory containing entity configuration files
            db_path: Path to SQLite database
        """
        self.entities_dir = Path(entities_dir)
        self.db_path = db_path
        
        # In-memory entity storage
        self.entities: Dict[str, Entity] = {}
        self.entities_by_type: Dict[str, Dict[str, Entity]] = defaultdict(dict)
        self.patterns: List[EntityPattern] = []
        self.exclusion_patterns: List[re.Pattern] = []
        
        # Configuration
        self.min_confidence_threshold = 0.3
        self.learning_enabled = True
        self.auto_discover_entities = True
        
        # Load initial entities
        self._load_entities()
        self._load_patterns()
        self._load_entities_from_db()
    
    def _load_entities(self):
        """Load entities from JSON configuration files."""
        logger.info("Loading entities from configuration files...")
        
        # Load companies
        self._load_entity_file(
            self.entities_dir / "companies" / "ai_companies.json",
            "companies",
            "company"
        )
        
        # Load technologies
        self._load_entity_file(
            self.entities_dir / "technologies" / "ai_technologies.json",
            "technologies",
            "technology"
        )
        
        # Load products
        self._load_entity_file(
            self.entities_dir / "products" / "ai_products.json",
            "products",
            "product"
        )
        
        logger.info(f"Loaded {len(self.entities)} entities from configuration")
    
    def _load_entity_file(self, file_path: Path, data_key: str, entity_type: str):
        """Load entities from a specific JSON file."""
        try:
            if not file_path.exists():
                logger.warning(f"Entity file not found: {file_path}")
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entities_data = data.get(data_key, {})
            
            for name, info in entities_data.items():
                entity = Entity(
                    name=name,
                    entity_type=entity_type,
                    confidence=info.get('confidence', 0.8),
                    aliases=info.get('aliases', []),
                    description=info.get('description', ''),
                    metadata=info
                )
                
                self._add_entity(entity)
        
        except Exception as e:
            logger.error(f"Error loading entities from {file_path}: {e}")
    
    def _load_patterns(self):
        """Load entity extraction patterns."""
        patterns_file = self.entities_dir / "patterns" / "entity_patterns.json"
        
        try:
            if not patterns_file.exists():
                logger.warning(f"Patterns file not found: {patterns_file}")
                return
            
            with open(patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load inclusion patterns
            for pattern_data in data.get('patterns', []):
                pattern = EntityPattern(**pattern_data)
                self.patterns.append(pattern)
            
            # Load exclusion patterns
            for exclusion_data in data.get('exclusion_patterns', []):
                try:
                    pattern = re.compile(exclusion_data['pattern'], re.IGNORECASE)
                    self.exclusion_patterns.append(pattern)
                except re.error as e:
                    logger.warning(f"Invalid exclusion pattern: {e}")
            
            # Load validation rules
            validation_rules = data.get('validation_rules', {})
            self.min_confidence_threshold = validation_rules.get('min_confidence_threshold', 0.3)
            
            logger.info(f"Loaded {len(self.patterns)} patterns and {len(self.exclusion_patterns)} exclusions")
        
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
    
    def _load_entities_from_db(self):
        """Load entities from database."""
        try:
            database = Database(self.db_path)
            
            # For now, create a simple connection
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT name, entity_type, description, aliases, metadata, 
                           confidence_score, mention_count
                    FROM entities
                    ORDER BY mention_count DESC
                """)
                
                for row in cursor.fetchall():
                    name, entity_type, description, aliases_json, metadata_json, confidence, mention_count = row
                    
                    aliases = json.loads(aliases_json) if aliases_json else []
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    entity = Entity(
                        name=name,
                        entity_type=entity_type,
                        description=description or "",
                        aliases=aliases,
                        metadata=metadata,
                        confidence=confidence,
                        mention_count=mention_count
                    )
                    
                    # Only add if not already loaded from config
                    if name not in self.entities:
                        self._add_entity(entity)
                    else:
                        # Update mention count from database
                        self.entities[name].mention_count = mention_count
                
                logger.info(f"Loaded {len([e for e in self.entities.values() if e.mention_count > 0])} entities from database")
        
        except Exception as e:
            logger.error(f"Error loading entities from database: {e}")
    
    def _add_entity(self, entity: Entity):
        """Add entity to internal storage."""
        self.entities[entity.name] = entity
        self.entities_by_type[entity.entity_type][entity.name] = entity
        
        # Add aliases lookup
        for alias in entity.aliases:
            alias_key = f"{alias.lower()}_{entity.entity_type}"
            self.entities[alias_key] = entity
    
    def get_entity(self, name: str, entity_type: Optional[str] = None) -> Optional[Entity]:
        """Get entity by name and optionally type."""
        # Try exact match first
        if name in self.entities:
            entity = self.entities[name]
            if entity_type is None or entity.entity_type == entity_type:
                return entity
        
        # Try case-insensitive match
        name_lower = name.lower()
        for entity in self.entities.values():
            if (entity.name.lower() == name_lower and 
                (entity_type is None or entity.entity_type == entity_type)):
                return entity
            
            # Check aliases
            if (name_lower in [alias.lower() for alias in entity.aliases] and 
                (entity_type is None or entity.entity_type == entity_type)):
                return entity
        
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return list(self.entities_by_type[entity_type].values())
    
    def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Entity]:
        """Search entities by name, description, or aliases."""
        query_lower = query.lower()
        results = []
        
        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            
            # Search in name
            if query_lower in entity.name.lower():
                results.append(entity)
                continue
            
            # Search in aliases
            if any(query_lower in alias.lower() for alias in entity.aliases):
                results.append(entity)
                continue
            
            # Search in description
            if query_lower in entity.description.lower():
                results.append(entity)
                continue
            
            # Search in keywords
            keywords = entity.metadata.get('keywords', [])
            if any(query_lower in keyword.lower() for keyword in keywords):
                results.append(entity)
        
        return results
    
    def add_entity(self, entity: Entity, save_to_db: bool = True) -> bool:
        """Add a new entity."""
        try:
            # Check if entity already exists
            if entity.name in self.entities:
                logger.info(f"Entity {entity.name} already exists, updating...")
                existing = self.entities[entity.name]
                existing.aliases.extend([a for a in entity.aliases if a not in existing.aliases])
                existing.updated_at = datetime.now()
                entity = existing
            else:
                self._add_entity(entity)
            
            # Save to database if requested
            if save_to_db:
                self._save_entity_to_db(entity)
            
            logger.info(f"Added entity: {entity.name} ({entity.entity_type})")
            return True
        
        except Exception as e:
            logger.error(f"Error adding entity {entity.name}: {e}")
            return False
    
    def _save_entity_to_db(self, entity: Entity):
        """Save entity to database."""
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO entities 
                    (name, normalized_name, entity_type, description, aliases, metadata, confidence_score, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.name,
                    entity.normalized_name or re.sub(r'[^a-z0-9\s]', '', entity.name.lower().strip()),
                    entity.entity_type,
                    entity.description,
                    json.dumps(entity.aliases),
                    json.dumps(entity.metadata),
                    entity.confidence,
                    entity.updated_at
                ))
        
        except Exception as e:
            logger.error(f"Error saving entity to database: {e}")
    
    def update_entity_mention(self, entity_name: str, confidence_boost: float = 0.01):
        """Update entity mention count and confidence."""
        entity = self.get_entity(entity_name)
        if entity:
            entity.mention_count += 1
            entity.last_seen = datetime.now()
            entity.updated_at = datetime.now()
            
            # Boost confidence based on frequency (capped at 0.95)
            entity.confidence = min(0.95, entity.confidence + confidence_boost)
            
            # Update in database
            try:
                import sqlite3
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE entities 
                        SET mention_count = ?, confidence_score = ?, last_seen = ?, updated_at = ?
                        WHERE name = ?
                    """, (
                        entity.mention_count,
                        entity.confidence,
                        entity.last_seen,
                        entity.updated_at,
                        entity.name
                    ))
            
            except Exception as e:
                logger.error(f"Error updating entity mention: {e}")
    
    def discover_new_entities(self, text: str, existing_entities: List[str]) -> List[Entity]:
        """Discover new entities from text that aren't in our knowledge base."""
        if not self.auto_discover_entities:
            return []
        
        new_entities = []
        existing_names = {e.lower() for e in existing_entities}
        existing_names.update(self.entities.keys())
        
        # Use patterns to find potential entities
        for pattern in self.patterns:
            if not pattern.compiled_pattern:
                continue
            
            for match in pattern.compiled_pattern.finditer(text):
                entity_text = match.group().strip()
                
                # Skip if already known or too short
                if (entity_text.lower() in existing_names or 
                    len(entity_text) < 3 or
                    self._is_excluded(entity_text)):
                    continue
                
                # Validate entity
                if self._validate_potential_entity(entity_text, pattern.entity_type):
                    # Create new entity with lower confidence
                    new_entity = Entity(
                        name=entity_text,
                        entity_type=pattern.entity_type,
                        confidence=pattern.confidence * 0.7,  # Lower confidence for discovered
                        description=f"Auto-discovered {pattern.entity_type} from article",
                        metadata={'source': 'auto_discovery', 'pattern': pattern.name}
                    )
                    
                    new_entities.append(new_entity)
        
        return new_entities
    
    def _is_excluded(self, text: str) -> bool:
        """Check if text matches exclusion patterns."""
        for pattern in self.exclusion_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _validate_potential_entity(self, text: str, entity_type: str) -> bool:
        """Validate a potential entity."""
        # Length validation
        if len(text) < 2 or len(text) > 100:
            return False
        
        # Word count validation
        if len(text.split()) > 6:
            return False
        
        # Type-specific validation
        if entity_type == "company" and not any(c.isupper() for c in text):
            return False
        
        return True
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed entities."""
        stats = {
            'total_entities': len(self.entities),
            'entities_by_type': defaultdict(int),
            'high_confidence_entities': 0,
            'most_mentioned': [],
            'recently_discovered': [],
            'patterns_count': len(self.patterns),
            'exclusion_patterns_count': len(self.exclusion_patterns)
        }
        
        # Count by type
        for entity in self.entities.values():
            stats['entities_by_type'][entity.entity_type] += 1
            
            if entity.confidence >= 0.8:
                stats['high_confidence_entities'] += 1
        
        # Most mentioned entities
        all_entities = list(self.entities.values())
        all_entities.sort(key=lambda x: x.mention_count, reverse=True)
        stats['most_mentioned'] = [(e.name, e.mention_count) for e in all_entities[:10]]
        
        # Recently discovered
        recent = [e for e in all_entities if e.metadata.get('source') == 'auto_discovery']
        recent.sort(key=lambda x: x.created_at, reverse=True)
        stats['recently_discovered'] = [e.name for e in recent[:10]]
        
        return stats
    
    def export_entities(self, file_path: str, entity_type: Optional[str] = None):
        """Export entities to JSON file."""
        try:
            entities_to_export = []
            for entity in self.entities.values():
                if entity_type is None or entity.entity_type == entity_type:
                    entities_to_export.append(entity.to_dict())
            
            export_data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'total_entities': len(entities_to_export),
                    'entity_type': entity_type
                },
                'entities': entities_to_export
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(entities_to_export)} entities to {file_path}")
        
        except Exception as e:
            logger.error(f"Error exporting entities: {e}")
    
    def import_entities(self, file_path: str):
        """Import entities from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_count = 0
            for entity_data in data.get('entities', []):
                entity = Entity.from_dict(entity_data)
                if self.add_entity(entity, save_to_db=False):  # Don't auto-save to DB during import
                    imported_count += 1
            
            logger.info(f"Imported {imported_count} entities from {file_path}")
        
        except Exception as e:
            logger.error(f"Error importing entities: {e}")


# Global entity manager instance
_entity_manager: Optional[EntityManager] = None


def get_entity_manager() -> EntityManager:
    """Get global entity manager instance."""
    global _entity_manager
    if _entity_manager is None:
        _entity_manager = EntityManager()
    return _entity_manager


def reset_entity_manager():
    """Reset global entity manager instance."""
    global _entity_manager
    _entity_manager = None