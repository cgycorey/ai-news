"""Automatic phrase learning from database.

Analyzes existing AI-relevant articles to extract discriminative phrases
that distinguish AI content from general content.
"""

import logging
import re
import time
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
        if self._cached_phrases and self._cache_timestamp:
            age = time.time() - self._cache_timestamp
            if age < 300:
                return self._cached_phrases

        logger.info("Starting phrase learning from database...")

        # Get AI and non-AI articles
        # Note: ai_only=False returns ALL articles, so we filter in Python
        ai_articles = self.database.get_articles(ai_only=True, limit=1000)
        all_articles = self.database.get_articles(limit=3000)
        non_ai_articles = [a for a in all_articles if not a.ai_relevant][:2000]

        if len(ai_articles) < 10:
            logger.debug("Not enough AI articles for learning (need at least 10)")
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
        self._cache_timestamp = time.time()
        logger.info(f"Learnt {len(discriminative_phrases)} phrases from {len(ai_articles)} AI articles")

        return discriminative_phrases

    def _extract_ngrams(self, articles: List, n: int = 3) -> Counter:
        """Extract n-gram phrases from article titles and content.

        Args:
            articles: List of Article objects
            n: Maximum n-gram size (1=gram, 2=bigram, 3=trigram)

        Returns:
            Counter of phrase -> frequency
        """
        # Comprehensive stopword filter to exclude common phrases
        stopwords = {
            # Articles and determiners
            'the', 'a', 'an', 'this', 'that', 'these', 'those',

            # Prepositions
            'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about',
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',

            # Conjunctions
            'and', 'or', 'but', 'if', 'because', 'although', 'though', 'however',

            # Common verbs
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',

            # Pronouns
            'it', 'its', 'he', 'she', 'they', 'them', 'his', 'her', 'their', 'our', 'your',

            # Common adjectives
            'more', 'most', 'less', 'least', 'very', 'really', 'just', 'still', 'also',
            'even', 'only', 'some', 'many', 'much', 'few', 'all', 'any', 'each', 'every',

            # Common nouns (non-technical)
            'people', 'person', 'things', 'something', 'time', 'way', 'part', 'work',
            'world', 'life', 'case', 'point', 'place', 'right', 'reason', 'problem',

            # Numbers
            'one', 'two', 'first', 'second', 'next', 'last',

            # Additional common words
            'company', 'new', 'can', 'but'
        }

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
                        # STOPWORD FILTER: Skip phrases with only stopwords
                        words_in_phrase = phrase.split()
                        meaningful_words = [w for w in words_in_phrase if w not in stopwords]

                        # Must have at least 50% meaningful words
                        if len(meaningful_words) / len(words_in_phrase) >= 0.5:
                            phrases[phrase] += 1

        return phrases

    def refresh_cache(self):
        """Force refresh of cached phrases."""
        self._cached_phrases = None
        self.learn_from_database()
