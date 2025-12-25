"""
Topic Discovery System for AI News Collector

Dynamically discovers related terms from collected articles to build
keyword associations without manual maintenance.
"""

import re
import sqlite3
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging

from .database import Database
from .entity_extractor import EntityExtractor
from .text_processor import TextProcessor
from .spacy_term_extractor import SpaCyTermExtractor, Term
from .domain_term_filter import DomainTermFilter

logger = logging.getLogger(__name__)


class TopicDiscovery:
    """
    Discovers related terms from articles to build dynamic keyword associations.

    Learns from articles that contain topic keywords to find frequently
    co-occurring terms, entities, and phrases.
    """

    def __init__(self, database: Database, use_spacy: bool = True):
        """Initialize topic discovery with optional spaCy support."""
        self.database = database
        self.text_processor = TextProcessor()
        self.use_spacy = use_spacy
        self.entity_extractor = None

        try:
            self.entity_extractor = EntityExtractor(self.text_processor, use_spacy=False)
        except Exception as e:
            logger.warning(f"Entity extractor not available: {e}")
            self.entity_extractor = None

        # Initialize spaCy components if requested
        if use_spacy:
            try:
                self.term_extractor = SpaCyTermExtractor()
                self.domain_filter = DomainTermFilter()
                logger.info("SpaCy term extraction enabled")
            except Exception as e:
                logger.warning(f"SpaCy unavailable, using basic extraction: {e}")
                self.term_extractor = None
                self.domain_filter = None
        else:
            self.term_extractor = None
            self.domain_filter = None

        # Initialize discovery schema
        self._init_discovery_schema()

    def _init_discovery_schema(self):
        """Create database tables for topic discovery."""
        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()

        # Table for discovered term relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_name TEXT NOT NULL,
                discovered_term TEXT NOT NULL,
                occurrence_count INTEGER DEFAULT 1,
                confidence_score REAL DEFAULT 0.0,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(topic_name, discovered_term)
            )
        """)

        # Table for article analysis tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                topics_analyzed TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES articles(id)
            )
        """)

        # Indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_topic_discoveries_topic
            ON topic_discoveries(topic_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_topic_discoveries_confidence
            ON topic_discoveries(confidence_score DESC)
        """)

        conn.commit()
        conn.close()

    def analyze_co_occurrences(
        self,
        articles: List,
        base_terms: List[str],
        topic_name: str,
        min_occurrence: int = 3
    ) -> List[Tuple[str, float, int]]:
        """
        Analyze articles to find terms that frequently appear with base terms.

        Uses spaCy NER if available, falls back to basic extraction.

        Args:
            articles: List of article objects
            base_terms: Base topic keywords to find associations with
            topic_name: Name of the topic for storage
            min_occurrence: Minimum times a term must appear to be considered

        Returns:
            List of (discovered_term, confidence_score, occurrence_count)
        """
        logger.info(f"Analyzing {len(articles)} articles for topic: {topic_name}")

        # Normalize base terms
        base_terms_lower = [term.lower() for term in base_terms]

        if self.term_extractor and self.domain_filter:
            # Use spaCy-based extraction
            return self._analyze_with_spacy(articles, base_terms_lower, topic_name, min_occurrence)
        else:
            # Use basic extraction
            return self._analyze_basic(articles, base_terms_lower, topic_name, min_occurrence)

    def _analyze_with_spacy(
        self,
        articles: List,
        base_terms_lower: List[str],
        topic_name: str,
        min_occurrence: int
    ) -> List[Tuple[str, float, int]]:
        """Analyze using spaCy NER and domain filtering."""
        all_terms = set()
        article_count = 0

        for article in articles:
            # Check if article contains base term
            text = f"{article.title} {article.content}".lower()
            has_base_term = any(term in text for term in base_terms_lower)

            if not has_base_term:
                continue

            article_count += 1

            # Extract terms using spaCy
            try:
                terms = self.term_extractor.extract_terms(
                    text,
                    article_id=article.id if hasattr(article, 'id') else None
                )
                all_terms.update(terms)
            except Exception as e:
                logger.debug(f"SpaCy extraction failed for article: {e}")

        # Filter and score
        scored_terms = self.domain_filter.filter_and_score(all_terms, article_count)

        # Filter by min_occurrence
        filtered_terms = [
            (term, confidence, count)
            for term, confidence, count in scored_terms
            if count >= min_occurrence
        ]

        logger.info(f"SpaCy discovered {len(filtered_terms)} terms for topic: {topic_name}")
        return filtered_terms

    def _analyze_basic(
        self,
        articles: List,
        base_terms_lower: List[str],
        topic_name: str,
        min_occurrence: int
    ) -> List[Tuple[str, float, int]]:
        """Analyze using basic word frequency (original method)."""
        # Track co-occurrences
        term_counter = Counter()
        article_mentions = defaultdict(set)

        for article in articles:
            article_id = article.id if hasattr(article, 'id') else None
            text = f"{article.title} {article.content}".lower()

            # Check if article contains any base term
            has_base_term = any(term in text for term in base_terms_lower)

            if not has_base_term:
                continue

            # Extract terms from this article
            terms = self._extract_terms_from_article(article)

            # Count co-occurrences
            for term in terms:
                term_lower = term.lower()

                # Skip if it's a base term itself
                if term_lower in base_terms_lower:
                    continue

                term_counter[term_lower] += 1
                if article_id:
                    article_mentions[term_lower].add(article_id)

        # Filter by minimum occurrence
        discovered = []
        for term, count in term_counter.items():
            if count >= min_occurrence:
                # Calculate confidence based on frequency and article distribution
                article_diversity = len(article_mentions[term])
                confidence = self._calculate_confidence(count, article_diversity, len(articles))

                discovered.append((term, confidence, count))

        # Sort by confidence
        discovered.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Basic discovered {len(discovered)} terms for topic: {topic_name}")
        return discovered

    def _extract_terms_from_article(self, article) -> Set[str]:
        """Extract relevant terms from an article."""
        terms = set()

        text = f"{article.title} {article.content}"

        # Extract entities if available
        if self.entity_extractor:
            try:
                entities = self.entity_extractor.extract_entities(text)
                for entity in entities:
                    terms.add(entity.text)
            except Exception as e:
                logger.debug(f"Entity extraction failed: {e}")

        # Extract key phrases using text processor
        try:
            # Use text processor to extract meaningful phrases
            words = text.split()
            # Extract capitalized phrases (potential entities/topics)
            phrases = []
            for i in range(len(words) - 1):
                if words[i][0].isupper() and words[i+1][0].isupper():
                    phrases.append(f"{words[i]} {words[i+1]}")
            terms.update(phrases[:10])
        except Exception as e:
            logger.debug(f"Phrase extraction failed: {e}")

        # Extract significant words (filter out common words)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        significant_words = [w for w in words if len(w) > 3 and w.lower() not in self._get_stopwords()]
        terms.update(significant_words[:30])  # Limit to top 30

        return terms

    def _calculate_confidence(self, occurrence_count: int, article_diversity: int, total_articles: int) -> float:
        """
        Calculate confidence score for a discovered term.

        Higher confidence for terms that:
        - Appear frequently (occurrence_count)
        - Appear in many different articles (article_diversity)
        """
        # Frequency score (0-1)
        frequency_score = min(occurrence_count / 20.0, 1.0)

        # Diversity score (0-1)
        diversity_score = min(article_diversity / 10.0, 1.0)

        # Combined confidence (weighted average)
        confidence = (frequency_score * 0.6 + diversity_score * 0.4)

        return round(confidence, 3)

    def learn_from_articles(
        self,
        articles: List,
        topic_name: str,
        base_keywords: List[str],
        min_occurrence: int = 3
    ) -> int:
        """
        Analyze articles and persist discovered term relationships.

        Args:
            articles: List of article objects
            topic_name: Name of the topic
            base_keywords: Base keywords for this topic
            min_occurrence: Minimum occurrence threshold

        Returns:
            Number of new discoveries saved
        """
        # Analyze co-occurrences
        discovered = self.analyze_co_occurrences(articles, base_keywords, topic_name, min_occurrence)

        if not discovered:
            return 0

        # Save to database
        new_discoveries = 0
        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()

        for term, confidence, count in discovered:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO topic_discoveries
                    (topic_name, discovered_term, occurrence_count, confidence_score, last_seen)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (topic_name, term, count, confidence))

                if cursor.rowcount > 0:
                    new_discoveries += 1

            except sqlite3.Error as e:
                logger.error(f"Failed to save discovery {term}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"Saved {new_discoveries} new discoveries for topic: {topic_name}")
        return new_discoveries

    def get_expanded_keywords(
        self,
        topic_name: str,
        base_keywords: List[str],
        min_confidence: float = 0.3,
        max_keywords: int = 50
    ) -> List[str]:
        """
        Get base keywords expanded with discovered terms.

        Args:
            topic_name: Name of the topic
            base_keywords: Base keywords defined in config
            min_confidence: Minimum confidence score for discovered terms
            max_keywords: Maximum total keywords to return

        Returns:
            List of base + discovered keywords
        """
        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()

        # Fetch discovered terms above confidence threshold
        cursor.execute("""
            SELECT discovered_term, confidence_score
            FROM topic_discoveries
            WHERE topic_name = ? AND confidence_score >= ?
            ORDER BY confidence_score DESC, occurrence_count DESC
            LIMIT ?
        """, (topic_name, min_confidence, max_keywords))

        discovered_terms = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Combine base and discovered
        all_keywords = base_keywords + discovered_terms

        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in all_keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique_keywords.append(kw)

        return unique_keywords[:max_keywords]

    def get_discovery_stats(self, topic_name: str) -> Dict:
        """Get statistics about discovered terms for a topic."""
        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*), AVG(confidence_score), MAX(last_seen)
            FROM topic_discoveries
            WHERE topic_name = ?
        """, (topic_name,))

        row = cursor.fetchone()
        conn.close()

        if row and row[0] > 0:
            return {
                "topic": topic_name,
                "total_discovered": row[0],
                "avg_confidence": round(row[1], 3) if row[1] else 0,
                "last_updated": row[2]
            }

        return {
            "topic": topic_name,
            "total_discovered": 0,
            "avg_confidence": 0,
            "last_updated": None
        }

    def prune_stale_discoveries(
        self,
        topic_name: str,
        days_threshold: int = 30,
        min_confidence: float = 0.2
    ) -> int:
        """
        Remove stale discoveries that haven't been seen recently or have low confidence.

        Args:
            topic_name: Topic to prune
            days_threshold: Remove terms not seen in this many days
            min_confidence: Remove terms below this confidence

        Returns:
            Number of pruned discoveries
        """
        cutoff_date = datetime.now() - timedelta(days=days_threshold)

        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM topic_discoveries
            WHERE topic_name = ?
            AND (last_seen < ? OR confidence_score < ?)
        """, (topic_name, cutoff_date.strftime('%Y-%m-%d %H:%M:%S'), min_confidence))

        pruned = cursor.rowcount
        conn.commit()
        conn.close()

        if pruned > 0:
            logger.info(f"Pruned {pruned} stale discoveries from topic: {topic_name}")

        return pruned

    def suggest_related_topics(self, topic_name: str, base_keywords: List[str]) -> List[str]:
        """
        Suggest related topics based on discovered terms.

        For example, if "AI" topic discovers many "healthcare" terms,
        suggest creating an "AI+Healthcare" combination topic.
        """
        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()

        # Get top discovered terms
        cursor.execute("""
            SELECT discovered_term, occurrence_count
            FROM topic_discoveries
            WHERE topic_name = ?
            ORDER BY occurrence_count DESC
            LIMIT 20
        """, (topic_name,))

        discovered = cursor.fetchall()
        conn.close()

        # Simple suggestion: if discovered terms match other topic keywords, suggest combination
        # This is a basic implementation - could be enhanced with NLP
        suggestions = []

        # Common topic indicators
        topic_indicators = {
            'healthcare': ['healthcare', 'medical', 'hospital', 'clinical', 'patient'],
            'finance': ['finance', 'banking', 'trading', 'investment', 'fintech'],
            'insurance': ['insurance', 'underwriting', 'claims', 'risk', 'insurtech'],
            'automotive': ['automotive', 'vehicle', 'car', 'autonomous', 'driving'],
            'education': ['education', 'learning', 'student', 'teaching', 'edtech']
        }

        discovered_terms = [term.lower() for term, _ in discovered]

        for suggested_topic, indicators in topic_indicators.items():
            if suggested_topic.lower() == topic_name.lower():
                continue

            overlap = len(set(discovered_terms) & set(indicators))
            if overlap >= 2:  # At least 2 overlapping terms
                suggestions.append(f"{topic_name}+{suggested_topic}")

        return suggestions

    def _get_stopwords(self) -> Set[str]:
        """Get set of common stopwords to filter out."""
        return {
            # Pronouns & articles
            'the', 'a', 'an', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose',

            # Prepositions
            'with', 'from', 'at', 'by', 'on', 'off', 'in', 'out', 'over', 'under',
            'about', 'after', 'before', 'between', 'into', 'through', 'during',

            # Conjunctions
            'and', 'or', 'but', 'if', 'because', 'although', 'though', 'while',

            # Verbs
            'have', 'has', 'had', 'been', 'were', 'was', 'is', 'are', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'cannot',

            # Adverbs
            'not', 'now', 'then', 'there', 'here', 'when', 'where', 'why', 'how',
            'also', 'very', 'more', 'most', 'some', 'such', 'only', 'own', 'same',
            'so', 'than', 'too', 'just', 'first', 'last',

            # Common nouns
            'news', 'article', 'articles', 'report', 'reports', 'says', 'said',
            'according', 'time', 'year', 'day', 'way', 'make', 'take',

            # Numbers & quantities
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'ten', 'hundred', 'thousand', 'million', 'billion'
        }


def create_topic_discovery(database: Database) -> TopicDiscovery:
    """Factory function to create TopicDiscovery instance."""
    return TopicDiscovery(database)
