"""Article auto-tagging with entity extraction.

This module provides automatic article tagging during collection.
Zero configuration needed - works transparently in the background.
"""

import logging
import sqlite3
import threading
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .entity_extractor import EntityExtractor, ExtractedEntity
from .entity_types import EntityType
from .database import Database, Article

logger = logging.getLogger(__name__)


@dataclass
class EntityTag:
    """Represents an entity tag attached to an article."""
    entity_text: str
    entity_type: str  # 'company', 'product', 'technology', 'person'
    confidence: float
    source: str  # 'spacy', 'pattern', 'known', 'discovered'


class ArticleTagger:
    """Auto-tag articles with entities during collection.
    
    This class handles automatic entity extraction and tagging
    for articles as they are saved to the database.
    
    Usage:
        tagger = ArticleTagger(entity_extractor)
        tags = tagger.tag_article(article)
        tagger.save_tags(article_id, tags, database)
    """
    
    # Class-level lock for thread-safe tag operations
    _tag_save_lock = threading.Lock()
    
    def __init__(self, entity_extractor: EntityExtractor,
                 min_confidence: float = 0.6):
        """Initialize the article tagger.
        
        Args:
            entity_extractor: EntityExtractor instance
            min_confidence: Minimum confidence for auto-tagging (default: 0.6)
        """
        self.entity_extractor = entity_extractor
        self.min_confidence = min_confidence
        logger.info(f"ArticleTagger initialized with min_confidence={min_confidence}")
    
    def tag_article(self, article: Article) -> List[EntityTag]:
        """Extract and tag entities from an article.

        Args:
            article: Article object with title and content

        Returns:
            List of EntityTag objects
        """
        # Combine title and content for entity extraction
        text = f"{article.title} {article.content or ''}"

        if not text.strip():
            logger.debug("Empty article text, skipping entity extraction")
            return []

        try:
            # Extract entities using the entity extractor
            entities = self.entity_extractor.extract_entities(text)
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            entities = []

        # Filter by minimum confidence and valid entity types
        valid_entity_types = {'company', 'product', 'technology', 'person'}

        # Extended garbage filter - common words misclassified as entities
        garbage_words = {
            # Pronouns and determiners
            'it', 'its', 'this', 'that', 'these', 'those', 'he', 'she', 'they', 'them',
            'his', 'her', 'their', 'our', 'your', 'my', 'who', 'which', 'what', 'whom',

            # Common verbs
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'make', 'made', 'take', 'took', 'get', 'got', 'go', 'went',
            'come', 'came', 'say', 'said', 'see', 'saw', 'know', 'knew', 'think',
            'thought', 'want', 'wanted', 'use', 'used', 'find', 'found', 'give', 'gave',

            # Common adjectives and adverbs
            'more', 'most', 'less', 'least', 'very', 'really', 'just', 'still', 'also',
            'even', 'only', 'some', 'many', 'much', 'few', 'all', 'any', 'each', 'every',
            'good', 'better', 'best', 'bad', 'worse', 'worst', 'new', 'old', 'big',
            'small', 'large', 'little', 'long', 'short', 'high', 'low', 'first', 'last',

            # Common nouns (often misclassified)
            'people', 'person', 'things', 'something', 'nothing', 'anything', 'everything',
            'time', 'times', 'way', 'ways', 'part', 'parts', 'work', 'works', 'world',
            'life', 'lives', 'case', 'cases', 'point', 'points', 'place', 'places',
            'right', 'rights', 'reason', 'reasons', 'problem', 'problems', 'service',
            'services', 'company', 'companies', 'business', 'businesses', 'industry',
            'industries', 'market', 'markets', 'system', 'systems', 'process', 'processes',
            'product', 'products', 'technology', 'technologies', 'research', 'researchers',

            # Temporal and spatial words
            'now', 'then', 'today', 'tomorrow', 'yesterday', 'here', 'there', 'where',
            'when', 'while', 'during', 'after', 'before', 'until', 'since', 'again',

            # Logical and connector words
            'and', 'or', 'but', 'if', 'because', 'although', 'though', 'however',
            'therefore', 'thus', 'hence', 'otherwise', 'whether', 'either', 'neither',

            # Common prepositions
            'of', 'in', 'on', 'at', 'by', 'for', 'from', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',

            # Numbers and quantities
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'ten', 'hundred', 'thousand', 'million', 'billion', 'once', 'twice',

            # Specific garbage found in DB
            'learning', 'companies', 'human', 'here', 'there', 'two', 'even', 'key',
            'people', 'still', 'among', 'few', 'comments', 'world', 'according',
            'future', 'not', 'after', 'when', 'some', 'why', 'everything', 'full',
            'such', 'all', 'build', 'most', 'now', 'many', 'often', 'think', 'review',
            'top', 'chinese', 'make', 'take', 'come', 'like', 'while', 'what', 'how',
            'one', 'said', 'say'
        }

        tags = []
        for entity in entities:
            if entity.confidence >= self.min_confidence:
                entity_type_value = entity.entity_type.value

                # Skip entity types that aren't in the database schema
                if entity_type_value not in valid_entity_types:
                    logger.debug(f"Skipping entity '{entity.text}' with invalid type '{entity_type_value}'")
                    continue

                # Skip garbage words misclassified as entities
                entity_lower = entity.text.lower().strip()
                if entity_lower in garbage_words:
                    logger.debug(f"Skipping garbage entity '{entity.text}'")
                    continue

                # Skip very short (< 3 chars) or very long (> 50 chars) entities
                if len(entity.text) < 3 or len(entity.text) > 50:
                    logger.debug(f"Skipping entity '{entity.text}' (invalid length)")
                    continue

                # Skip single-character entities
                if len(entity.text.strip()) == 1:
                    logger.debug(f"Skipping single-character entity '{entity.text}'")
                    continue
                
                # Map extraction method to database source values
                source_map = {
                    'pattern_matching': 'pattern',
                    'known_entity_matching': 'known',
                    'spacy_matching': 'spacy',
                    'discovered_entity': 'discovered'
                }
                source = source_map.get(entity.extraction_method, 'pattern')
                
                tag = EntityTag(
                    entity_text=entity.text,
                    entity_type=entity_type_value,
                    confidence=entity.confidence,
                    source=source
                )
                tags.append(tag)
        
        logger.info(f"Tagged article '{article.title[:50]}...' with {len(tags)} entities")
        return tags
    
    def save_tags(self, article_id: int, tags: List[EntityTag], 
                  database: Database) -> int:
        """Save entity tags to database (thread-safe).
        
        Uses global database write lock to prevent concurrent write conflicts.
        
        Args:
            article_id: Article ID
            tags: List of EntityTag objects
            database: Database instance
            
        Returns:
            Number of tags saved
        """
        print(f"DEBUG save_tags: article_id={article_id}, tags={len(tags)}")  # DEBUG
        
        if not tags:
            logger.debug(f"No tags to save for article {article_id}")
            return 0
        
        # Use global database write lock to prevent concurrent tag saves (SQLite limitation)
        with database._write_lock:
            try:
                with sqlite3.connect(database.db_path) as conn:
                    # Use WAL mode for better concurrency
                    conn.execute('PRAGMA journal_mode=WAL')

                    saved_count = 0

                    for tag in tags:
                        try:
                            # Use UPSERT (INSERT OR REPLACE) to handle duplicates
                            conn.execute("""
                                INSERT OR REPLACE INTO article_entity_tags
                                (article_id, entity_text, entity_type, confidence, source)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                article_id,
                                tag.entity_text,
                                tag.entity_type,
                                tag.confidence,
                                tag.source
                            ))
                            saved_count += 1
                            print(f"DEBUG: Saved tag {tag.entity_text}")

                        except sqlite3.IntegrityError as e:
                            logger.debug(f"Duplicate tag for article {article_id}: {tag.entity_text}")
                            print(f"DEBUG: Duplicate tag {tag.entity_text}: {e}")
                        except Exception as e:
                            print(f"DEBUG: Error saving tag {tag.entity_text}: {e}")

                    conn.commit()
                    logger.info(f"Saved {saved_count} entity tags for article {article_id}")
                    return saved_count

            except sqlite3.Error as e:
                logger.error(f"Database error saving tags for article {article_id}: {e}")
                print(f"DEBUG: Database error: {e}")
                return 0
    
    def suggest_category(self, tags: List[EntityTag]) -> Optional[str]:
        """Suggest article category based on entity tags.
        
        Only suggests if category is empty - never overrides existing.
        
        Args:
            tags: List of EntityTag objects
            
        Returns:
            Suggested category or None
        """
        if not tags:
            return None
        
        entity_types = {tag.entity_type for tag in tags}
        
        # Simple heuristic-based categorization
        if 'company' in entity_types and 'product' in entity_types:
            return 'business'
        elif 'technology' in entity_types:
            return 'technology'
        elif 'person' in entity_types:
            return 'people'
        elif 'company' in entity_types:
            return 'business'
        elif 'product' in entity_types:
            return 'technology'
        else:
            return 'general'


# Singleton instance for app-wide use
_tagger_instance: Optional[ArticleTagger] = None


def get_article_tagger() -> ArticleTagger:
    """Get singleton ArticleTagger instance.
    
    Creates instance on first call and reuses thereafter.
    """
    global _tagger_instance
    
    if _tagger_instance is None:
        from .text_processor import TextProcessor
        from .spacy_utils import is_model_available
        
        # Initialize entity extractor
        text_processor = TextProcessor()
        # Only use spaCy if model is available
        use_spacy = is_model_available()
        entity_extractor = EntityExtractor(text_processor, use_spacy=use_spacy)
        
        # Create tagger with default settings
        _tagger_instance = ArticleTagger(entity_extractor, min_confidence=0.6)
        logger.info(f"ArticleTagger singleton initialized (spaCy={use_spacy})")
    
    return _tagger_instance


def batch_tag_articles(database: Database, 
                       limit: Optional[int] = None,
                       batch_size: int = 100,
                       skip_tagged: bool = True) -> Dict[str, Any]:
    """Batch tag existing articles.
    
    Args:
        database: Database instance
        limit: Maximum number of articles to tag (None = all)
        batch_size: Number of articles per batch
        skip_tagged: Skip articles that already have tags
        
    Returns:
        Dictionary with statistics
    """
    tagger = get_article_tagger()
    
    # Get articles to tag
    articles = database.get_articles(limit=limit or 10000)
    
    if skip_tagged:
        # Filter out articles that already have tags
        untagged = []
        for article in articles:
            existing_tags = database.get_article_entity_tags(article.id)
            if not existing_tags:
                untagged.append(article)
        articles = untagged
    
    total_articles = len(articles)
    tagged_count = 0
    total_tags = 0
    failed_count = 0
    
    logger.info(f"Starting batch tagging of {total_articles} articles")
    
    for i, article in enumerate(articles):
        try:
            # Extract tags
            tags = tagger.tag_article(article)
            
            if tags:
                # Save tags
                saved = tagger.save_tags(article.id, tags, database)
                total_tags += saved
                tagged_count += 1
            
            # Progress logging
            if (i + 1) % batch_size == 0:
                logger.info(f"Progress: {i + 1}/{total_articles} articles processed")
                
        except Exception as e:
            logger.error(f"Error tagging article {article.id}: {e}")
            failed_count += 1
    
    stats = {
        'total_articles': total_articles,
        'tagged_count': tagged_count,
        'total_tags': total_tags,
        'failed_count': failed_count,
        'success_rate': tagged_count / total_articles if total_articles > 0 else 0.0
    }
    
    logger.info(f"Batch tagging complete: {stats}")
    return stats