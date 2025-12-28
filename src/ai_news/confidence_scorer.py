"""Multi-layer AI relevance confidence scoring.

Combines entity matching, learned phrases, and contextual signals
to calculate confidence score for article AI relevance.
"""

import logging
import time
from typing import Dict, List
from .database import Database, Article
from .entity_extractor import EntityExtractor
from .phrase_learner import PhraseLearner
from .text_processor import TextProcessor
from .config import get_performance_config
from .performance_metrics import get_metrics

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Calculate AI relevance confidence using multiple signals."""

    # Entity types that indicate AI relevance
    AI_ENTITY_TYPES = {'company', 'product', 'technology', 'person'}

    # Negative indicators (reduce confidence)
    NEGATIVE_CATEGORIES = {'gaming', 'consumer electronics', 'entertainment', 'sports'}

    # Positive sources (boost confidence)
    POSITIVE_SOURCE_PATTERNS = {'AI', 'artificial intelligence', 'machine learning'}

    def __init__(self, database: Database):
        self.database = database
        text_processor = TextProcessor()
        self.entity_extractor = EntityExtractor(text_processor, use_spacy=False)
        self.phrase_learner = PhraseLearner(database)
        self._learned_phrases = None

    def calculate_confidence(self, article: Article) -> float:
        """
        Calculate AI relevance confidence score (0.0-1.0).

        Layers:
        1. Entity matching (0.0-0.7)
        2. Learned phrases (0.0-0.3)
        3. Source reputation (0.0-0.1)
        4. Category adjustments (-0.2 to +0.1)

        Returns:
            Confidence score 0.0-1.0
        """
        confidence = 0.0

        # Layer 1: Entity matching
        confidence += self._score_entities(article)

        # Layer 2: Learned phrases
        confidence += self._score_phrases(article)

        # Layer 3: Source reputation
        confidence += self._score_source(article)

        # Layer 4: Category adjustments
        confidence += self._score_category(article)

        # Normalize to 0.0-1.0
        return max(0.0, min(1.0, confidence))

    def _score_entities(self, article: Article) -> float:
        """Score based on AI entities found in article (0.0-0.7).
        
        Hybrid mode: Uses fast pattern matching, then spaCy for uncertain cases.
        """
        metrics = get_metrics()
        start = time.time()
        
        text = f"{article.title or ''} {article.content or ''}"
        perf_config = get_performance_config()
        
        try:
            # Step 1: Fast pattern matching (always)
            entities = self.entity_extractor.extract_entities(text)
            ai_entities = [
                e for e in entities
                if e.entity_type.value in self.AI_ENTITY_TYPES
            ]
            
            # Step 2: Calculate base confidence from patterns
            entity_score = min(0.7, len(ai_entities) * 0.2)
            
            # Step 3: Decide if spaCy enhancement needed
            needs_spacy = False
            if perf_config.use_spacy_in_collection == "full":
                needs_spacy = True
            elif perf_config.use_spacy_in_collection == "hybrid":
                # Use spaCy for uncertain cases
                needs_spacy = (
                    (perf_config.spacy_on_low_confidence and entity_score < 0.5) or
                    (perf_config.spacy_on_no_entities and len(ai_entities) == 0) or
                    (perf_config.spacy_on_high_value_sources and 
                     article.source_name.lower() in perf_config.high_value_sources)
                )
            # else "pattern-only": needs_spacy = False
            
            # Step 4: Apply spaCy if needed
            if needs_spacy and self.entity_extractor.use_spacy and self.entity_extractor.nlp:
                logger.debug(f"Using spaCy for article (confidence={entity_score:.2f}): {article.title[:40]}...")
                
                # Re-extract with spaCy
                doc = self.entity_extractor.nlp(text)
                spacy_entities = self._extract_entities_from_spacy(doc)
                
                # Use spaCy results if better
                if len(spacy_entities) > len(ai_entities):
                    ai_entities = spacy_entities
                    entity_score = min(0.7, len(ai_entities) * 0.2)
            
            duration = time.time() - start
            method = "spacy" if needs_spacy else "pattern"
            metrics.record_entity_extraction(duration, method)
            
            logger.debug(f"Entity score: {entity_score:.2f} ({len(ai_entities)} AI entities, spaCy_used={needs_spacy})")
            return entity_score
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return 0.0

    def _extract_entities_from_spacy(self, doc) -> List:
        """Extract entities from spaCy doc using EntityExtractor."""
        # This is handled by EntityExtractor.extract_entities_with_spacy
        # For now, delegate to the entity extractor
        text = doc.text if hasattr(doc, 'text') else str(doc)
        return self.entity_extractor.extract_entities_with_spacy(text)

    def _score_phrases(self, article: Article) -> float:
        """Score based on learned AI phrases (0.0-0.3)."""
        if self._learned_phrases is None:
            self._learned_phrases = self.phrase_learner.learn_from_database()

        if not self._learned_phrases:
            return 0.0

        text = f"{article.title or ''} {article.content or ''}".lower()

        # Find highest-scoring matching phrase
        best_score = 0.0
        for phrase, confidence in self._learned_phrases.items():
            if phrase in text:
                best_score = max(best_score, confidence)
                logger.debug(f"Matched phrase: '{phrase}' (conf={confidence:.2f})")

        # Cap phrase contribution at 0.3
        return min(0.3, best_score)

    def _score_source(self, article: Article) -> float:
        """Score based on source name (0.0-0.1)."""
        if not article.source_name:
            return 0.0

        source_lower = article.source_name.lower()

        # Check if source name contains AI indicators
        for pattern in self.POSITIVE_SOURCE_PATTERNS:
            if pattern.lower() in source_lower:
                logger.debug(f"Source boost: '{article.source_name}' contains '{pattern}'")
                return 0.1

        return 0.0

    def _score_category(self, article: Article) -> float:
        """Score based on category (-0.2 to +0.1)."""
        if not article.category:
            return 0.0

        category_lower = article.category.lower()

        # Negative categories
        if category_lower in self.NEGATIVE_CATEGORIES:
            logger.debug(f"Category penalty: '{article.category}'")
            return -0.2

        # Positive categories
        positive_categories = {'technology', 'tech', 'science', 'research'}
        if category_lower in positive_categories:
            logger.debug(f"Category boost: '{article.category}'")
            return 0.1

        return 0.0

    def get_review_status(self, confidence: float) -> str:
        """Map confidence score to review status.

        Args:
            confidence: Score from calculate_confidence()

        Returns:
            'auto' (>=0.7), 'reviewed' (0.5-0.69), or 'rejected' (<0.5)
        """
        if confidence >= 0.7:
            return 'auto'
        elif confidence >= 0.5:
            return 'reviewed'
        else:
            return 'rejected'
