"""
SpacyDigestAnalyzer - Analyze articles for topic relevance using spaCy semantic scoring.

This module provides intelligent article filtering that combines:
1. Fast keyword-based pre-filtering
2. spaCy semantic analysis for confidence scoring
3. DigestCache integration for performance
4. Graceful fallback when spaCy unavailable
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from datetime import datetime

from .digest_cache import DigestCache
from .intersection_optimization import IntersectionOptimizer


logger = logging.getLogger(__name__)


@dataclass
class ScoredArticle:
    """Article with confidence score and matched entities."""
    article: Dict
    confidence: float
    matched_entities: Set[str]


class SpacyDigestAnalyzer:
    """
    Analyze articles for topic relevance using spaCy semantic scoring.

    Combines fast keyword filtering with spaCy-based semantic analysis
    to provide confidence-scored article recommendations.
    """

    CONFIDENCE_THRESHOLD = 0.7

    def __init__(self, cache_db_path: str = "ai_news.db", ttl_hours: int = 6):
        """
        Initialize analyzer with cache and dependencies.

        Args:
            cache_db_path: Path to SQLite database for cache
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache = DigestCache(db_path=cache_db_path, ttl_hours=ttl_hours)
        self.optimizer = IntersectionOptimizer()
        self._extractor = None
        self._spacy_available = self._init_spacy_extractor()

    def _init_spacy_extractor(self) -> bool:
        """
        Initialize spaCy term extractor if available.

        Returns:
            True if spaCy is available, False otherwise
        """
        try:
            from .spacy_term_extractor import SpaCyTermExtractor
            self._extractor = SpaCyTermExtractor()
            available = self._extractor.is_available()
            if not available:
                logger.warning("spaCy term extractor initialized but not available")
            return available
        except ImportError as e:
            logger.warning(f"spaCy not installed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize spaCy extractor: {e}")
            return False

    def analyze(
        self,
        articles: List[Dict],
        topics: List[str],
        days: int = 7
    ) -> List[ScoredArticle]:
        """
        Analyze articles for topic relevance.

        Args:
            articles: List of article dictionaries with title, content, summary
            topics: List of topic keywords
            days: Number of days for digest (affects cache key)

        Returns:
            List of ScoredArticle with confidence ≥ 0.7
        """
        if not articles or not topics:
            return []

        # Check cache first
        cached = self.cache.get(topics, days)
        if cached:
            return [
                ScoredArticle(
                    article=item["article"],
                    confidence=item["confidence"],
                    matched_entities=set(item.get("matched_entities", set()))
                )
                for item in cached.get("scored_articles", [])
            ]

        # Step 1: Fast keyword-based filtering
        filtered_articles = self._filter_by_keywords(articles, topics)
        if not filtered_articles:
            return []

        # Step 2: Semantic analysis and scoring
        scored_articles = self._analyze_articles(filtered_articles, topics)

        # Step 3: Filter by confidence threshold
        high_confidence = [
            article for article in scored_articles
            if article.confidence >= self.CONFIDENCE_THRESHOLD
        ]

        # Step 4: Cache results
        self._cache_results(topics, days, high_confidence)

        return high_confidence

    def _filter_by_keywords(
        self,
        articles: List[Dict],
        topics: List[str]
    ) -> List[Dict]:
        """
        Fast initial keyword matching before expensive spaCy analysis.

        Args:
            articles: List of article dictionaries
            topics: List of topic keywords

        Returns:
            Articles containing at least one topic keyword
        """
        filtered = []

        for article in articles:
            title = article.get("title", "").lower()
            content = article.get("content", "").lower()
            summary = article.get("summary", "").lower()

            combined_text = f"{title} {content} {summary}"

            if any(topic.lower() in combined_text for topic in topics):
                filtered.append(article)

        return filtered

    def _analyze_articles(
        self,
        articles: List[Dict],
        topics: List[str]
    ) -> List[ScoredArticle]:
        """
        Perform semantic analysis and scoring on filtered articles.

        Args:
            articles: Pre-filtered articles
            topics: Topic keywords

        Returns:
            List of ScoredArticle with confidence scores
        """
        scored = []

        for article in articles:
            confidence = 0.0
            matched_entities = set()

            # Use spaCy if available
            if self._spacy_available and self._extractor:
                try:
                    # Extract terms from title, content, summary
                    title_terms = self._extractor.extract_terms(
                        article.get("title", ""),
                        article.get("id")
                    )
                    content_terms = self._extractor.extract_terms(
                        article.get("content", ""),
                        article.get("id")
                    )

                    all_terms = title_terms | content_terms

                    # Calculate confidence based on matched entities
                    if all_terms:
                        topic_lower = [t.lower() for t in topics]
                        for term in all_terms:
                            if any(topic in term.text.lower() for topic in topic_lower):
                                matched_entities.add(term.text)
                                confidence += term.confidence * 0.5

                        # Normalize confidence
                        confidence = min(1.0, confidence)

                except Exception as e:
                    logger.debug(f"spaCy extraction error: {e}")
                    # Fall through to keyword-based scoring

            # Fallback: Use IntersectionOptimizer for keyword-based scoring
            if confidence < 0.3:
                try:
                    result = self.optimizer.detect_weighted_intersections(
                        article,
                        topics
                    )
                    confidence = result.get("confidence", 0.0)
                except Exception as e:
                    logger.debug(f"Intersection optimizer error: {e}")
                    confidence = 0.0

            scored.append(ScoredArticle(
                article=article,
                confidence=confidence,
                matched_entities=matched_entities
            ))

        return scored

    def _cache_results(
        self,
        topics: List[str],
        days: int,
        scored_articles: List[ScoredArticle]
    ):
        """
        Store analysis results in cache.

        Args:
            topics: Topic keywords
            days: Number of days
            scored_articles: Results to cache
        """
        if not scored_articles:
            return

        try:
            results = {
                "scored_articles": [
                    {
                        "article": article.article,
                        "confidence": article.confidence,
                        "matched_entities": list(article.matched_entities)
                    }
                    for article in scored_articles
                ],
                "generated_at": datetime.now().isoformat()
            }
            self.cache.set(topics, days, results)
        except Exception as e:
            logger.debug(f"Cache storage error: {e}")


def create_spacy_digest_analyzer(
    cache_db_path: str = "ai_news.db",
    ttl_hours: int = 6
) -> Optional[SpacyDigestAnalyzer]:
    """
    Factory function to create a SpacyDigestAnalyzer.

    Args:
        cache_db_path: Path to SQLite database for cache
        ttl_hours: Time-to-live for cache entries in hours

    Returns:
        SpacyDigestAnalyzer instance or None if initialization fails
    """
    try:
        analyzer = SpacyDigestAnalyzer(
            cache_db_path=cache_db_path,
            ttl_hours=ttl_hours
        )
        return analyzer
    except Exception as e:
        logger.error(f"Failed to create SpacyDigestAnalyzer: {e}")
        return None
