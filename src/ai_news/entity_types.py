"""Shared entity types and data classes."""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class EntityType(Enum):
    """Entity types for AI news."""
    COMPANY = "company"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    PERSON = "person"
    EVENT = "event"
    LOCATION = "location"
    ORGANIZATION = "organization"


class SentimentLabel(Enum):
    """Sentiment labels for text analysis."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class EntityRelationship:
    """Relationship between two entities."""
    source_entity: str
    target_entity: str
    relationship_type: str  # e.g., 'CEO_OF', 'ACQUIRED', 'PARTNERED_WITH'
    confidence: float
    context: str
    start_char: int
    end_char: int
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_entity': self.source_entity,
            'target_entity': self.target_entity,
            'relationship_type': self.relationship_type,
            'confidence': self.confidence,
            'context': self.context,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'evidence': self.evidence
        }


@dataclass
class ExtractedEntity:
    """Simplified extracted entity for business intelligence."""
    text: str
    entity_type: EntityType
    confidence: float
    start_position: int
    end_position: int
    extraction_method: str = "pattern_matching"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'text': self.text,
            'entity_type': self.entity_type.value,
            'confidence': self.confidence,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'extraction_method': self.extraction_method,
            'metadata': self.metadata or {}
        }