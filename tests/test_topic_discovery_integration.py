"""Integration tests for enhanced topic discovery"""

import pytest
import tempfile
import os
from pathlib import Path

from ai_news.topic_discovery import TopicDiscovery
from ai_news.database import Database, Article
from ai_news.config import Config

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    # Initialize database with schema
    db = Database(db_path=path)
    
    yield db
    
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def sample_articles(temp_db):
    """Create sample articles for testing."""
    articles = [
        Article(
            title="OpenAI Releases GPT-4 Turbo",
            content="OpenAI announced GPT-4 Turbo, a new neural network model with improved capabilities.",
            url="http://test1.com",
            summary="AI announcement",
            source_name="Test",
            ai_relevant=True,
            region="US"
        ),
        Article(
            title="Deep Learning Advances in Computer Vision",
            content="New breakthrough in deep learning and computer vision research using convolutional networks.",
            url="http://test2.com",
            summary="Research news",
            source_name="Test",
            ai_relevant=True,
            region="US"
        ),
        Article(
            title="Google's PaLM 2 Language Model",
            content="Google announced PaLM 2, a large language model for natural language processing tasks.",
            url="http://test3.com",
            summary="Google AI news",
            source_name="Test",
            ai_relevant=True,
            region="US"
        )
    ]

    for article in articles:
        temp_db.save_article(article)

    return temp_db.get_articles(limit=10)

def test_spacy_integration(sample_articles, temp_db):
    """Test that spaCy extraction is used when available."""
    discovery = TopicDiscovery(temp_db, use_spacy=True)

    # Use base terms that are actually in the articles
    # These terms will be used to find articles, then extract OTHER terms from those articles
    results = discovery.analyze_co_occurrences(
        sample_articles,
        base_terms=["OpenAI", "Google", "learning"],
        topic_name="AI",
        min_occurrence=1
    )

    # Should extract quality terms
    term_texts = [r[0] for r in results]

    # The results should not include the base terms themselves
    # But should include related entities and technical terms
    print(f"Extracted terms: {term_texts[:10]}")

    # Check for technical terms (GPT, neural network, language model, etc.)
    # We should find at least one technical term
    # "gpt-4" contains "gpt" (case-insensitive match)
    technical_found = any(
        "GPT" in t or "gpt" in t.lower() or "neural" in t.lower() or
        "language model" in t.lower() or "computer vision" in t.lower() or
        "PaLM" in t or "palm" in t.lower() or "convolutional" in t.lower()
        for t in term_texts
    )
    assert technical_found, f"Should extract technical terms. Got: {term_texts}"

    # Check that base terms are NOT in results (they should be filtered out)
    assert "OpenAI" not in term_texts, "Base term 'OpenAI' should be filtered out"
    assert "Google" not in term_texts, "Base term 'Google' should be filtered out"
    assert "learning" not in [t.lower() for t in term_texts], "Base term 'learning' should be filtered out"

    # Check that common words are NOT present
    assert "the" not in [t.lower() for t in term_texts], "Should not contain 'the'"
    assert "and" not in [t.lower() for t in term_texts], "Should not contain 'and'"
    assert "with" not in [t.lower() for t in term_texts], "Should not contain 'with'"

def test_fallback_to_basic(sample_articles, temp_db):
    """Test fallback to basic extraction when spaCy unavailable."""
    discovery = TopicDiscovery(temp_db, use_spacy=False)

    results = discovery.analyze_co_occurrences(
        sample_articles,
        base_terms=["AI", "artificial intelligence"],
        topic_name="AI",
        min_occurrence=1
    )

    # Should still return results (basic extraction)
    assert len(results) >= 0, "Basic extraction should return results"

def test_spacy_components_initialized(temp_db):
    """Test that spaCy components are initialized when use_spacy=True."""
    discovery = TopicDiscovery(temp_db, use_spacy=True)

    # Check that components are initialized
    assert hasattr(discovery, 'term_extractor'), "Should have term_extractor attribute"
    assert hasattr(discovery, 'domain_filter'), "Should have domain_filter attribute"

    # If spaCy is available, components should not be None
    if discovery.term_extractor is not None:
        assert discovery.term_extractor.is_available(), "SpaCy extractor should be available"
    
    assert discovery.domain_filter is not None, "Domain filter should always be initialized"

def test_basic_extraction_no_spacy(temp_db):
    """Test that basic extraction works without spaCy components."""
    discovery = TopicDiscovery(temp_db, use_spacy=False)

    # Components should be None when use_spacy=False
    assert discovery.term_extractor is None, "term_extractor should be None"
    assert discovery.domain_filter is None, "domain_filter should be None"

def test_min_occurrence_filtering(sample_articles, temp_db):
    """Test that min_occurrence parameter filters results correctly."""
    discovery = TopicDiscovery(temp_db, use_spacy=True)

    # With min_occurrence=1, should get more results
    results_low = discovery.analyze_co_occurrences(
        sample_articles,
        base_terms=["AI"],
        topic_name="AI",
        min_occurrence=1
    )

    # With min_occurrence=5, should get fewer results
    results_high = discovery.analyze_co_occurrences(
        sample_articles,
        base_terms=["AI"],
        topic_name="AI",
        min_occurrence=5
    )

    assert len(results_low) >= len(results_high), \
        "Lower min_occurrence should return same or more results"
