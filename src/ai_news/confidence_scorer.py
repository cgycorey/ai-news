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

    # Negative indicators (reduce confidence) - REMOVED
    # Rely on AI keyword gate instead of category penalties
    # This allows AI content in ANY domain (sports, gaming, healthcare, etc.)
    NEGATIVE_CATEGORIES = set()

    # Positive sources (boost confidence)
    POSITIVE_SOURCE_PATTERNS = {'artificial intelligence', 'machine learning', 'deep learning'}

    # Core AI technology keywords (compact, fundamental terms)
    # These are auto-expanded by phrase learner
    CORE_AI_KEYWORDS = {
        'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
        'gpt', 'llm', 'large language model', 'transformer', 'diffusion model',
        'computer vision', 'natural language processing', 'nlp', 'reinforcement learning',
        'generative ai', 'chatgpt', 'openai', 'anthropic', 'claude', 'gemini', 'bard'
    }

    # Auto-learned phrases (updated from database)
    _AUTO_LEARNED_KEYWORDS = set()

    def __init__(self, database: Database):
        self.database = database
        text_processor = TextProcessor()
        self.entity_extractor = EntityExtractor(text_processor, use_spacy=False)
        self.phrase_learner = PhraseLearner(database)
        self._learned_phrases = None
        self._auto_learned_keywords = set()
        self._update_learned_keywords()

    def calculate_confidence(self, article: Article) -> float:
        """
        Calculate AI relevance confidence score (0.0-1.0).

        Layers:
        0. Mandatory AI keyword check (hard gate)
        1. Entity matching (0.0-0.4)
        2. Learned phrases (0.0-0.3)
        3. Source reputation (0.0-0.1)
        4. Category adjustments (-0.2 to +0.1)

        Returns:
            Confidence score 0.0-1.0
        """
        # Layer 0: Mandatory AI keyword check
        if not self._has_ai_keyword(article):
            logger.debug(f"Failed mandatory AI keyword check: {article.title[:40]}...")
            return 0.0

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

    def _update_learned_keywords(self):
        """Auto-update AI keywords from phrase learner."""
        try:
            learned_phrases = self.phrase_learner.learn_from_database()
            # Extract top single-word and two-word phrases as keywords
            new_keywords = set()
            for phrase, confidence in learned_phrases.items():
                if confidence >= 0.3:  # Only high-confidence phrases
                    words = phrase.split()
                    if len(words) <= 2:  # Unigrams and bigrams
                        new_keywords.add(phrase)
            
            self._auto_learned_keywords = new_keywords
            
            if self._auto_learned_keywords:
                logger.info(f"Auto-learned {len(self._auto_learned_keywords)} AI keywords")
        except Exception as e:
            logger.warning(f"Failed to update learned keywords: {e}")

    def refresh_learned_keywords(self):
        """Manually trigger refresh of learned keywords from database."""
        logger.info("Refreshing learned keywords...")
        self._update_learned_keywords()
        return len(self._auto_learned_keywords)

    def _has_ai_keyword(self, article: Article) -> bool:
        """Check if article contains AI context using dynamic pattern detection.
        
        Works with ANY domain dynamically:
        1. Core AI/ML technical terms
        2. Auto-learned keywords from database (updated automatically)
        3. 'AI' + action verb (uses, powered by, leverages, applies, implements)
        4. 'AI' + technical noun (model, algorithm, system, platform, technology)
        5. 'AI' + 'for' (e.g., "AI for healthcare", "AI for legal")
        
        No need to maintain domain-specific keyword lists.
        """
        text = f"{article.title or ''} {article.content or ''}".lower()
        
        # Check 1: Core AI keywords
        for keyword in self.CORE_AI_KEYWORDS:
            if keyword in text:
                logger.debug(f"Found core AI keyword: '{keyword}'")
                return True
        
        # Check 2: Auto-learned keywords (dynamic, updates from DB)
        for keyword in self._auto_learned_keywords:
            if keyword in text:
                logger.debug(f"Found learned keyword: '{keyword}'")
                return True
        
        # Check 3: 'ai' with action verbs (indicates actual AI usage)
        import re
        ai_action_patterns = [
            r'\bai\s+(uses|used|using|use|powered|powers|applies|applied|applies|'
            r'leverages|leveraged|deployed|implements|implemented|implements|'
            r'enables|enabled|enhances|enhanced|drives|driven|based|helps)',
            r'^(uses|powered|driven|enabled|based|built|developed)\s+on\s+\bai',
        ]
        
        for pattern in ai_action_patterns:
            if re.search(pattern, text):
                logger.debug(f"Found AI with action pattern")
                return True
        
        # Check 4: 'ai' + technical noun (indicates AI technology)
        ai_tech_patterns = [
            r'\bai\s+(model|models|algorithm|algorithms|system|systems|platform|platforms|'
            r'technology|technologies|solution|solutions|tool|tools|assistant|assistants|'
            r'agent|agents|bot|bots|engine|framework|library|api)',
        ]
        
        for pattern in ai_tech_patterns:
            if re.search(pattern, text):
                logger.debug(f"Found AI with technical noun pattern")
                return True
        
        # Check 5: 'AI for [any domain]' (universal pattern)
        if re.search(r'\bai\s+for\s+\w+', text):
            logger.debug(f"Found 'AI for [domain]' pattern")
            return True
        
        return False

    def _score_entities(self, article: Article) -> float:
        """Score based on AI entities found in article (0.0-0.4).
        
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
            entity_score = min(0.6, len(ai_entities) * 0.25)
            
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
                    entity_score = min(0.6, len(ai_entities) * 0.25)
            
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
        """Score based on source name (0.0-0.1).
        
        Only boost if source explicitly mentions AI/ML in name.
        """
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
        """Score based on category (-0.2 to +0.15).
        
        Smart categorization:
        - Negative for pure entertainment/consumer tech
        - Neutral/positive for professional/technical domains (including sports tech)
        """
        if not article.category:
            return 0.0

        category_lower = article.category.lower()

        # Negative categories (only consumer/entertainment fluff)
        if category_lower in self.NEGATIVE_CATEGORIES:
            logger.debug(f"Category penalty: '{article.category}'")
            return -0.2

        # High-value AI categories
        if category_lower in {'artificial intelligence', 'machine learning', 'deep learning'}:
            return 0.15

        # AI-adjacent technical categories
        if category_lower in {'ai', 'data science', 'computer vision', 'nlp', 'sports tech',
                             'sports analytics', 'sports science', 'esports'}:
            return 0.1

        # Professional/technical domains (dynamic - works for any domain)
        # Exclude generic/fluff categories
        generic_categories = {'news', 'general', 'latest', 'updates', 'articles', 'blog'}
        if category_lower not in generic_categories:
            # Any substantive category gets small boost
            logger.debug(f"Category boost (domain): '{article.category}'")
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
