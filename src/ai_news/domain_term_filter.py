"""Filter and score terms by domain relevance."""

import json
import logging
from pathlib import Path
from typing import Set, List, Tuple, Dict
from collections import Counter
from .spacy_term_extractor import Term

logger = logging.getLogger(__name__)


class DomainTermFilter:
    """Filter generic terms and score by domain relevance."""

    def __init__(self, config_path: str | None = None):
        """Load domain dictionaries and stopwords."""
        self.tech_keywords = self._load_tech_keywords(config_path)
        self.stopwords = self._load_extended_stopwords()
        self.term_frequency = Counter()  # Track term frequency for specificity

    def _load_tech_keywords(self, config_path: str | None = None) -> Set[str]:
        """Load AI/tech keywords from config."""
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config" / "domain_keywords.json")

        try:
            with open(config_path, 'r') as f:
                data = json.load(f)

            keywords = set()
            for category, terms in data.items():
                keywords.update(terms)

            logger.info(f"Loaded {len(keywords)} tech keywords")
            return keywords

        except Exception as e:
            logger.error(f"Failed to load tech keywords: {e}")
            return set()

    def _load_extended_stopwords(self) -> Set[str]:
        """Load extended stopword list."""
        return {
            # Articles, pronouns, prepositions
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'with', 'from', 'at', 'by', 'on', 'off', 'in', 'out',
            'about', 'after', 'before', 'between', 'into', 'through',

            # Conjunctions
            'and', 'or', 'but', 'if', 'because', 'although', 'though', 'while',

            # Verbs
            'have', 'has', 'had', 'been', 'were', 'was', 'is', 'are',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cannot',

            # Generic tech/non-tech words
            'tech', 'technology', 'digital', 'system', 'platform',
            'service', 'company', 'business', 'market', 'industry',
            'report', 'news', 'article', 'update', 'release',
            'make', 'take', 'get', 'use', 'help', 'work', 'need',

            # Time/frequency
            'today', 'yesterday', 'recent', 'current', 'latest',
            'year', 'month', 'week', 'day', 'time'
        }

    def filter_and_score(self, terms: Set[Term], total_articles: int) -> List[Tuple[str, float, int]]:
        """
        Filter generic terms and calculate scores.

        Returns: List of (term, confidence, count)
        """
        scored_terms = []

        for term in terms:
            text = term.text.lower()

            # Filter stopwords
            if text in self.stopwords:
                continue

            # Filter single short words (except tech entities)
            if len(text) <= 3 and ' ' not in text and term.term_type != "ENTITY":
                continue

            # Calculate scores
            domain_relevance = self._calculate_domain_relevance(term)
            specificity = self._calculate_specificity(term, total_articles)

            # Final confidence
            confidence = (domain_relevance * 0.6) + (specificity * 0.4)

            # Filter low confidence (but not for entities)
            if confidence < 0.3 and term.term_type != "ENTITY":
                continue

            # Track frequency
            self.term_frequency[term.text] += 1

            scored_terms.append((term.text, confidence, 1))

        # Sort by confidence
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        return scored_terms

    def _calculate_domain_relevance(self, term: Term) -> float:
        """Calculate domain relevance score (0.0 - 1.0)."""
        score = 0.0
        text = term.text.lower()

        # Factor 1: Contains tech keyword (50%)
        if any(kw in text for kw in self.tech_keywords):
            score += 0.5

        # Factor 2: Entity type bonus (20%)
        if term.term_type == "ENTITY":
            score += 0.2

        # Factor 3: Multi-word technical term (30%)
        if len(term.text.split()) > 1 and term.term_type == "NOUN_PHRASE":
            score += 0.3

        return min(score, 1.0)

    def _calculate_specificity(self, term: Term, total_articles: int) -> float:
        """Calculate specificity (TF-IDF style)."""
        freq = self.term_frequency.get(term.text, 0)

        if total_articles == 0:
            return 0.5

        # TF-IDF style: rare terms = higher specificity
        tf = freq / total_articles
        specificity = 1.0 - min(tf, 1.0)

        # Bonus for multi-word terms
        if len(term.text.split()) > 1:
            specificity = min(specificity + 0.1, 1.0)

        return specificity
