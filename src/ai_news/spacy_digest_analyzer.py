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
        days: int = 7,
        use_and_logic: bool = True
    ) -> List[ScoredArticle]:
        """
        Analyze articles for topic relevance.

        Args:
            articles: List of article dictionaries with title, content, summary
            topics: List of topic keywords
            days: Number of days for digest (affects cache key)
            use_and_logic: If True, articles must match ALL topics. If False, any topic matches.

        Returns:
            List of ScoredArticle with confidence ≥ 0.7
        """
        if not articles or not topics:
            return []

        # Check cache first (include AND logic in cache key)
        cache_key_suffix = "_and" if use_and_logic else "_or"
        cached = self.cache.get(topics, days, cache_key_suffix)
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
        filtered_articles = self._filter_by_keywords(articles, topics, use_and_logic)
        if not filtered_articles:
            return []

        # Step 2: Semantic analysis and scoring
        scored_articles = self._analyze_articles(filtered_articles, topics, use_and_logic)

        # Step 3: Filter by confidence threshold
        high_confidence = [
            article for article in scored_articles
            if article.confidence >= self.CONFIDENCE_THRESHOLD
        ]

        # Step 4: Cache results
        self._cache_results(topics, days, high_confidence, cache_key_suffix)

        return high_confidence

    def _filter_by_keywords(
        self,
        articles: List[Dict],
        topics: List[str],
        use_and_logic: bool = True
    ) -> List[Dict]:
        """
        Fast initial keyword matching before expensive spaCy analysis.

        Args:
            articles: List of article dictionaries
            topics: List of topic keywords
            use_and_logic: If True, require ALL topics. If False, ANY topic.

        Returns:
            Articles matching the keyword criteria
        """
        filtered = []

        for article in articles:
            title = article.get("title", "").lower()
            content = article.get("content", "").lower()
            summary = article.get("summary", "").lower()

            combined_text = f"{title} {content} {summary}"

            if use_and_logic:
                # AND logic: all topics must be present
                if all(topic.lower() in combined_text for topic in topics):
                    filtered.append(article)
            else:
                # OR logic: any topic can match
                if any(topic.lower() in combined_text for topic in topics):
                    filtered.append(article)

        return filtered

    def _analyze_articles(
        self,
        articles: List[Dict],
        topics: List[str],
        use_and_logic: bool = True
    ) -> List[ScoredArticle]:
        """
        Perform semantic analysis and scoring on filtered articles.

        Args:
            articles: Pre-filtered articles
            topics: Topic keywords
            use_and_logic: If True, score each topic individually and require all to pass

        Returns:
            List of ScoredArticle with confidence scores
        """
        scored = []

        for article in articles:
            final_confidence = 0.0
            matched_entities = set()

            if use_and_logic and len(topics) > 1:
                # AND logic: score each topic individually, take minimum confidence
                topic_scores = []
                
                for topic in topics:
                    topic_conf = self._score_article_for_topic(article, topic)
                    topic_scores.append(topic_conf)
                
                # Final confidence is the minimum (strictest requirement)
                final_confidence = min(topic_scores) if topic_scores else 0.0
                
                # Collect matched entities from all topics
                if self._spacy_available and self._extractor:
                    try:
                        title_terms = self._extractor.extract_terms(
                            article.get("title", ""),
                            article.get("id")
                        )
                        content_terms = self._extractor.extract_terms(
                            article.get("content", ""),
                            article.get("id")
                        )
                        all_terms = title_terms | content_terms
                        
                        if all_terms:
                            topic_lower = [t.lower() for t in topics]
                            for term in all_terms:
                                if any(topic in term.text.lower() for topic in topic_lower):
                                    matched_entities.add(term.text)
                    except Exception:
                        pass
                        
            else:
                # OR logic: score against all topics collectively
                final_confidence = self._score_article_for_topic(article, " ".join(topics))
                
                # Collect matched entities
                if self._spacy_available and self._extractor:
                    try:
                        title_terms = self._extractor.extract_terms(
                            article.get("title", ""),
                            article.get("id")
                        )
                        content_terms = self._extractor.extract_terms(
                            article.get("content", ""),
                            article.get("id")
                        )
                        all_terms = title_terms | content_terms
                        
                        if all_terms:
                            topic_lower = [t.lower() for t in topics]
                            for term in all_terms:
                                if any(topic in term.text.lower() for topic in topic_lower):
                                    matched_entities.add(term.text)
                                    final_confidence += term.confidence * 0.5
                            
                            final_confidence = min(1.0, final_confidence)
                    except Exception:
                        pass

            scored.append(ScoredArticle(
                article=article,
                confidence=final_confidence,
                matched_entities=matched_entities
            ))

        return scored

    def _score_article_for_topic(
        self,
        article: Dict,
        topic: str
    ) -> float:
        """
        Score a single article against a single topic.

        Args:
            article: Article dictionary
            topic: Single topic keyword

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0
        
        # Use spaCy if available
        if self._spacy_available and self._extractor:
            try:
                title_terms = self._extractor.extract_terms(
                    article.get("title", ""),
                    article.get("id")
                )
                content_terms = self._extractor.extract_terms(
                    article.get("content", ""),
                    article.get("id")
                )
                
                all_terms = title_terms | content_terms
                
                if all_terms:
                    topic_lower = topic.lower()
                    for term in all_terms:
                        if topic_lower in term.text.lower():
                            confidence += term.confidence * 0.5
                    
                    confidence = min(1.0, confidence)
                    
            except Exception as e:
                logger.debug(f"spaCy extraction error: {e}")
        
        # Fallback: Use IntersectionOptimizer for keyword-based scoring
        if confidence < 0.3:
            try:
                result = self.optimizer.detect_weighted_intersections(
                    article,
                    [topic]
                )
                confidence = result.get("confidence", 0.0)
            except Exception as e:
                logger.debug(f"Intersection optimizer error: {e}")
                confidence = 0.0
        
        return confidence

    def _cache_results(
        self,
        topics: List[str],
        days: int,
        scored_articles: List[ScoredArticle],
        cache_key_suffix: str = ""
    ):
        """
        Store analysis results in cache.

        Args:
            topics: Topic keywords
            days: Number of days
            scored_articles: Results to cache
            cache_key_suffix: Suffix to differentiate AND vs OR caching
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
            self.cache.set(topics, days, results, cache_key_suffix)
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
