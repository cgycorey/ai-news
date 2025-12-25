"""Tests for SpaCyTermExtractor"""

import pytest
from ai_news.spacy_term_extractor import SpaCyTermExtractor, Term


def test_extractor_initialization():
    """Test extractor loads spaCy model"""
    extractor = SpaCyTermExtractor(model_name="en_core_web_sm")
    assert extractor.model_name == "en_core_web_sm"
    assert extractor.is_available()


def test_extract_entities():
    """Test named entity extraction"""
    extractor = SpaCyTermExtractor()
    text = "OpenAI released GPT-4 and Google announced PaLM 2."
    terms = extractor.extract_terms(text)

    # Should extract organizations and products
    term_texts = {t.text for t in terms}
    assert "OpenAI" in term_texts
    assert "Google" in term_texts

    # GPT-4 might be extracted as GPT or GPT-4
    assert any("GPT" in t for t in term_texts)


def test_extract_noun_phrases():
    """Test technical noun phrase extraction"""
    extractor = SpaCyTermExtractor()
    # Use proper capitalization for technical terms
    text = "Neural networks use Deep Learning for Computer Vision tasks."
    terms = extractor.extract_terms(text)

    term_texts = {t.text for t in terms}
    # Should extract technical phrases
    assert any("neural network" in t.lower() for t in term_texts)
    assert any("deep learning" in t.lower() for t in term_texts)
    assert any("computer vision" in t.lower() for t in term_texts)


def test_filter_stopwords():
    """Test that common words are filtered"""
    extractor = SpaCyTermExtractor()
    text = "The model was released with new features and capabilities."
    terms = extractor.extract_terms(text)

    term_texts = {t.text for t in terms}
    # These should NOT be in results
    assert "the" not in term_texts
    assert "and" not in term_texts
    assert "with" not in term_texts


def test_term_dataclass():
    """Test Term dataclass structure"""
    term = Term(
        text="OpenAI",
        term_type="ENTITY",
        confidence=0.8,
        source_article_id=123
    )

    assert term.text == "OpenAI"
    assert term.term_type == "ENTITY"
    assert term.confidence == 0.8
    assert term.source_article_id == 123


def test_empty_text():
    """Test extraction from empty text"""
    extractor = SpaCyTermExtractor()
    terms = extractor.extract_terms("")
    assert len(terms) == 0

    terms = extractor.extract_terms("   ")
    assert len(terms) == 0


def test_model_info():
    """Test getting model information"""
    extractor = SpaCyTermExtractor()
    info = extractor.get_model_info()

    assert "loaded" in info
    assert "model_name" in info
    assert info["model_name"] == "en_core_web_sm"
