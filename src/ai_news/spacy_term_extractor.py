"""
SpaCy-based Term Extractor for High-Quality Topic Discovery

Uses spaCy's NER (Named Entity Recognition) to extract:
- Named entities (ORG, PRODUCT, FAC) - companies, products, facilities
- Technical noun phrases - domain-specific multi-word terms
- Domain-specific terms - filtered using AI/tech dictionaries
"""

import logging
from dataclasses import dataclass
from typing import Set, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Term:
    """Represents an extracted term with metadata."""
    text: str
    term_type: str  # ENTITY, NOUN_PHRASE, TECH_TERM
    confidence: float
    source_article_id: Optional[int] = None


class SpaCyTermExtractor:
    """
    Extract high-quality terms from text using spaCy NER.

    Uses en_core_web_sm model for fast, accurate entity recognition.
    """

    # Entity types to extract
    TARGET_ENTITY_TYPES = {'ORG', 'PRODUCT', 'FAC', 'PERSON', 'GPE'}

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize spaCy term extractor.

        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
        """
        self.model_name = model_name
        self.nlp = None
        self._model_loaded = False

    def _load_model(self) -> bool:
        """
        Lazy load spaCy model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model_loaded:
            return True

        try:
            import spacy
            logger.debug(f"Loading spaCy model: {self.model_name}")

            try:
                self.nlp = spacy.load(self.model_name)
                self._model_loaded = True
                logger.info(f"Successfully loaded spaCy model: {self.model_name}")
                return True
            except OSError:
                # Model not downloaded
                logger.warning(f"spaCy model '{self.model_name}' not found. "
                             f"Download with: python -m spacy download {self.model_name}")
                return False

        except ImportError:
            logger.warning("spaCy not installed. Install with: pip install spacy")
            return False
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            return False

    def extract_terms(self, text: str, article_id: Optional[int] = None) -> Set[Term]:
        """
        Extract terms from text using spaCy.

        Args:
            text: Input text to analyze
            article_id: Optional article ID for tracking

        Returns:
            Set of Term objects
        """
        if not self._load_model():
            return set()

        if not text or not text.strip():
            return set()

        try:
            doc = self.nlp(text)

            terms = set()

            # Extract named entities
            entities = self._extract_entities(doc)
            for entity_text in entities:
                terms.add(Term(
                    text=entity_text,
                    term_type="ENTITY",
                    confidence=0.8,  # Base confidence for entities
                    source_article_id=article_id
                ))

            # Extract technical noun phrases
            noun_phrases = self._extract_noun_phrases(doc)
            for phrase_text in noun_phrases:
                terms.add(Term(
                    text=phrase_text,
                    term_type="NOUN_PHRASE",
                    confidence=0.6,  # Base confidence for noun phrases
                    source_article_id=article_id
                ))

            logger.debug(f"Extracted {len(terms)} terms from text (length: {len(text)})")
            return terms

        except Exception as e:
            logger.error(f"Error during spaCy term extraction: {e}")
            return set()

    def _extract_entities(self, doc) -> Set[str]:
        """
        Extract named entities from spaCy doc.

        Focuses on ORG, PRODUCT, FAC entities which are most relevant
        for tech/AI topics.
        """
        entities = set()

        for ent in doc.ents:
            # Only extract target entity types
            if ent.label_ in self.TARGET_ENTITY_TYPES:
                entity_text = ent.text.strip()

                # Filter out single characters and very short entities
                if len(entity_text) < 2:
                    continue

                # Filter out numeric entities
                if entity_text.replace('.', '').replace(',', '').isdigit():
                    continue

                entities.add(entity_text)

        return entities

    def _extract_noun_phrases(self, doc) -> Set[str]:
        """
        Extract technical noun phrases from spaCy doc.

        Uses noun chunking to find multi-word technical terms.
        Filters for tech-relevant phrases.
        """
        phrases = set()

        # Tech keywords that indicate a phrase is technical
        tech_indicators = {
            'learning', 'network', 'model', 'system', 'algorithm', 'data',
            'intelligence', 'computer', 'software', 'hardware', 'platform',
            'service', 'application', 'technology', 'framework', 'library',
            'neural', 'deep', 'machine', 'artificial', 'automated', 'autonomous',
            'language', 'vision', 'recognition', 'processing', 'analysis',
            'optimization', 'training', 'inference', 'generation', 'detection'
        }

        for chunk in doc.noun_chunks:
            phrase_text = chunk.text.strip()

            # Only consider phrases with 2-4 words (single words are usually entities)
            word_count = len(phrase_text.split())
            if word_count < 2 or word_count > 4:
                continue

            # Must contain at least one tech indicator
            phrase_lower = phrase_text.lower()
            if not any(indicator in phrase_lower for indicator in tech_indicators):
                continue

            # Filter out phrases starting with stopword-like words
            first_word = phrase_text.split()[0].lower()
            if first_word in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
                continue

            # Capitalization check - should have at least one capitalized word
            # (excluding all lowercase phrases which are often generic)
            if not any(word[0].isupper() for word in phrase_text.split()):
                continue

            phrases.add(phrase_text)

        return phrases

    def is_available(self) -> bool:
        """Check if spaCy model is available."""
        return self._load_model()

    def get_model_info(self) -> dict:
        """Get information about the spaCy model."""
        if not self._load_model():
            return {
                "loaded": False,
                "model_name": self.model_name
            }

        return {
            "loaded": True,
            "model_name": self.model_name,
            "version": self.nlp.meta.get("version", "unknown"),
            "vectors": self.nlp.meta.get("vectors", {}).get("width", 0)
        }


def create_spacy_extractor(model_name: str = "en_core_web_sm") -> Optional[SpaCyTermExtractor]:
    """
    Factory function to create a SpaCyTermExtractor.

    Args:
        model_name: spaCy model to use

    Returns:
        SpaCyTermExtractor instance or None if unavailable
    """
    extractor = SpaCyTermExtractor(model_name)
    if extractor.is_available():
        return extractor
    return None
