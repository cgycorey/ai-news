"""Enhanced topic digest with entity-aware matching.

This module enhances topic digest generation to use entity extraction
for better article matching, grouping, and insights.

Zero flags needed - works transparently.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3

from .entity_extractor import EntityExtractor, ExtractedEntity
from .entity_types import EntityType
from .text_processor import TextProcessor
from .database import Database, Article

logger = logging.getLogger(__name__)


@dataclass
class ScoredArticle:
    """Article with relevance score and entities."""
    article: Article
    confidence: float
    matched_entities: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'article': self.article,
            'confidence': self.confidence,
            'matched_entities': self.matched_entities
        }


class EnhancedTopicDigest:
    """Enhanced topic digest with entity-aware matching.
    
    This class provides entity-aware topic matching and digest generation.
    Automatically uses entities when available, falls back gracefully.
    """
    
    def __init__(self, database: Database):
        """Initialize enhanced topic digest.
        
        Args:
            database: Database instance
        """
        self.database = database
        
        # Initialize entity extractor (lazy loading)
        self._entity_extractor: Optional[EntityExtractor] = None
        
        logger.info("EnhancedTopicDigest initialized")
    
    @property
    def entity_extractor(self) -> Optional[EntityExtractor]:
        """Lazy load entity extractor."""
        if self._entity_extractor is None:
            try:
                text_processor = TextProcessor()
                self._entity_extractor = EntityExtractor(text_processor, use_spacy=True)
                logger.info("Entity extractor loaded for topic digest")
            except Exception as e:
                logger.warning(f"Failed to load entity extractor: {e}")
                self._entity_extractor = None
        return self._entity_extractor
    
    def extract_topic_entities(self, topic: str) -> List[ExtractedEntity]:
        """Extract entities from topic string.
        
        Args:
            topic: Topic string (e.g., "LLM", "OpenAI GPT-4")
            
        Returns:
            List of extracted entities
        """
        if not self.entity_extractor:
            return []
        
        try:
            entities = self.entity_extractor.extract_entities(topic)
            logger.info(f"Extracted {len(entities)} entities from topic '{topic}'")
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities from topic: {e}")
            return []
    
    def find_articles_for_topic(self, 
                                topics: List[str], 
                                days: int,
                                ai_only: bool = True,
                                min_confidence: float = 0.3) -> List[ScoredArticle]:
        """Find articles for topic using entity-aware matching.
        
        Args:
            topics: List of topic strings
            days: Number of days to look back
            ai_only: Only include AI-relevant articles
            min_confidence: Minimum confidence for entity match
            
        Returns:
            List of ScoredArticle objects sorted by relevance
        """
        # Extract entities from topics
        topic_entities = []
        for topic in topics:
            entities = self.extract_topic_entities(topic)
            topic_entities.extend(entities)
        
        # Get articles from database
        articles = self.database.get_articles(limit=5000, ai_only=ai_only)
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_articles = []
        for article in articles:
            if not article.published_at:
                recent_articles.append(article)
            else:
                if article.published_at.tzinfo:
                    article_date = article.published_at.astimezone(None).replace(tzinfo=None)
                else:
                    article_date = article.published_at
                if article_date >= cutoff_date:
                    recent_articles.append(article)
        
        # Score articles by entity match
        scored_articles = []
        for article in recent_articles:
            # Skip articles without ID (can't query entity tags)
            if article.id is None:
                continue
                
            score, matched_entities = self._score_article(article, topic_entities, topics)
            
            if score >= min_confidence:
                scored_article = ScoredArticle(
                    article=article,
                    confidence=score,
                    matched_entities=matched_entities
                )
                scored_articles.append(scored_article)
        
        # Sort by confidence (highest first)
        scored_articles.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Found {len(scored_articles)} articles for topics {topics}")
        return scored_articles
    
    def _score_article(self, 
                      article: Article, 
                      topic_entities: List[ExtractedEntity],
                      topic_keywords: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
        """Score article based on entity and keyword matching.
        
        Args:
            article: Article to score
            topic_entities: Entities extracted from topics
            topic_keywords: Original topic keywords (for fallback)
            
        Returns:
            Tuple of (confidence_score, matched_entities_list)
        """
        # Get article's entity tags
        article_tags = self.database.get_article_entity_tags(article.id)
        
        matched_entities = []
        entity_score = 0.0
        
        # Entity-based matching (higher weight)
        for topic_entity in topic_entities:
            for tag in article_tags:
                # Check for exact match
                if tag['entity_text'].lower() == topic_entity.text.lower():
                    entity_score += 0.4  # High weight for exact match
                    matched_entities.append({
                        'matched_text': tag['entity_text'],
                        'entity_type': tag['entity_type'],
                        'match_type': 'exact',
                        'confidence': tag['confidence']
                    })
                # Check for partial match (substring)
                elif topic_entity.text.lower() in tag['entity_text'].lower() or \
                     tag['entity_text'].lower() in topic_entity.text.lower():
                    entity_score += 0.2  # Medium weight for partial match
                    matched_entities.append({
                        'matched_text': tag['entity_text'],
                        'entity_type': tag['entity_type'],
                        'match_type': 'partial',
                        'confidence': tag['confidence']
                    })
        
        # Keyword-based matching (fallback, lower weight)
        keyword_score = 0.0
        article_text = f"{article.title} {article.content or ''} {article.summary or ''}".lower()
        
        for keyword in topic_keywords:
            keyword_lower = keyword.lower()
            # Title matches get higher weight
            if keyword_lower in article.title.lower():
                keyword_score += 0.3  # High weight for title match
            elif keyword_lower in article_text:
                keyword_score += 0.15  # Medium weight for content match
        
        # Calculate final confidence
        # Cap entity score at 1.0, keyword score at 0.5
        final_score = min(entity_score, 1.0) + min(keyword_score, 0.5)
        final_score = min(final_score, 1.0)  # Cap at 1.0
        
        return final_score, matched_entities
    
    def generate_entity_insights(self, scored_articles: List[ScoredArticle]) -> Dict[str, Any]:
        """Generate entity insights from scored articles.
        
        Args:
            scored_articles: List of scored articles
            
        Returns:
            Dictionary with entity insights
        """
        insights = {
            'companies': {},
            'products': {},
            'technologies': {},
            'people': {}
        }
        
        for scored_article in scored_articles:
            for match in scored_article.matched_entities:
                entity_type = match['entity_type']
                entity_text = match['matched_text']
                
                if entity_type in insights:
                    if entity_text not in insights[entity_type]:
                        insights[entity_type][entity_text] = 0
                    insights[entity_type][entity_text] += 1
        
        # Sort by count
        for entity_type in insights:
            insights[entity_type] = dict(
                sorted(insights[entity_type].items(), 
                      key=lambda x: x[1], 
                      reverse=True)
            )
        
        return insights
    
    def group_by_entity(self, 
                       scored_articles: List[ScoredArticle], 
                       entity_type: str = 'company') -> Dict[str, List[ScoredArticle]]:
        """Group articles by entity.
        
        Args:
            scored_articles: List of scored articles
            entity_type: Type of entity to group by
            
        Returns:
            Dictionary mapping entity names to article lists
        """
        groups = {}
        
        for scored_article in scored_articles:
            # Find entities of specified type
            entities_for_type = [
                match['matched_text'] for match in scored_article.matched_entities
                if match['entity_type'] == entity_type
            ]
            
            if entities_for_type:
                # Use first matching entity as group key
                group_key = entities_for_type[0]
                
                if group_key not in groups:
                    groups[group_key] = []
                
                groups[group_key].append(scored_article)
            else:
                # No entity of this type, put in "Other" group
                if 'Other' not in groups:
                    groups['Other'] = []
                groups['Other'].append(scored_article)
        
        return groups


def create_enhanced_topic_digest(database: Database) -> EnhancedTopicDigest:
    """Factory function to create EnhancedTopicDigest instance.
    
    Args:
        database: Database instance
        
    Returns:
        EnhancedTopicDigest instance
    """
    return EnhancedTopicDigest(database)
