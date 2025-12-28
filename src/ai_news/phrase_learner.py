"""Automatic phrase learning from database.

Analyzes existing AI-relevant articles to extract discriminative phrases
that distinguish AI content from general content.
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Tuple
from .database import Database

logger = logging.getLogger(__name__)


class PhraseLearner:
    """Learn AI-related phrases from existing articles in database."""

    def __init__(self, database: Database):
        self.database = database
        self._cached_phrases = None
        self._cache_timestamp = None

    def learn_from_database(self, min_ai_frequency: float = 0.05,
                           max_non_ai_frequency: float = 0.01) -> Dict[str, float]:
        """
        Extract discriminative phrases from AI articles vs non-AI articles.

        Args:
            min_ai_frequency: Minimum % of AI articles that must contain phrase (0.05 = 5%)
            max_non_ai_frequency: Maximum % of non-AI articles that can contain phrase (0.01 = 1%)

        Returns:
            Dict mapping phrase -> confidence score (0.0-1.0)
        """
        if self._cached_phrases:
            return self._cached_phrases

        logger.info("Starting phrase learning from database...")

        # Get AI and non-AI articles
        all_articles = self.database.get_articles(limit=3000)
        ai_articles = [a for a in all_articles if a.ai_relevant][:1000]
        non_ai_articles = [a for a in all_articles if not a.ai_relevant][:2000]

        if len(ai_articles) < 10:
            logger.warning("Not enough AI articles for learning (need at least 10)")
            return {}

        # Extract phrases from both sets
        ai_phrases = self._extract_ngrams(ai_articles, n=3)
        non_ai_phrases = self._extract_ngrams(non_ai_articles, n=3)

        # Calculate discriminative score
        discriminative_phrases = {}

        for phrase, ai_count in ai_phrases.items():
            ai_freq = ai_count / len(ai_articles)
            non_ai_count = non_ai_phrases.get(phrase, 0)
            non_ai_freq = non_ai_count / len(non_ai_articles) if non_ai_articles else 0

            # Phrase must be common in AI articles AND rare in non-AI articles
            if ai_freq >= min_ai_frequency and non_ai_freq <= max_non_ai_frequency:
                # Score = frequency in AI articles * rarity in non-AI articles
                confidence = ai_freq * (1.0 - non_ai_freq)
                discriminative_phrases[phrase] = min(1.0, confidence)

        self._cached_phrases = discriminative_phrases
        logger.info(f"Learnt {len(discriminative_phrases)} phrases from {len(ai_articles)} AI articles")

        return discriminative_phrases

    def _extract_ngrams(self, articles: List, n: int = 3) -> Counter:
        """Extract n-gram phrases from article titles and content.

        Args:
            articles: List of Article objects
            n: Maximum n-gram size (1=unigram, 2=bigram, 3=trigram)

        Returns:
            Counter of phrase -> frequency
        """
        phrases = Counter()

        for article in articles:
            # Combine title and content
            text = f"{article.title or ''} {article.content or ''}".lower()

            # Extract phrases (1-gram to n-gram)
            words = re.findall(r'\b[a-z]{3,}\b', text)  # Words with 3+ letters

            for i in range(len(words)):
                for gram_size in range(1, min(n + 1, len(words) - i + 1)):
                    phrase = ' '.join(words[i:i + gram_size])

                    # Filter: must contain at least one meaningful word
                    if len(phrase.split()) >= 2:
                        phrases[phrase] += 1

        return phrases

    def refresh_cache(self):
        """Force refresh of cached phrases."""
        self._cached_phrases = None
        self.learn_from_database()
