"""Tests for DomainTermFilter"""

from ai_news.domain_term_filter import DomainTermFilter
from ai_news.spacy_term_extractor import Term
import pytest

def test_filter_generic_terms():
    """Test that generic terms are filtered out"""
    filter_obj = DomainTermFilter()

    terms = {
        Term(text="the", term_type="NOUN_PHRASE", confidence=0.5),
        Term(text="and", term_type="NOUN_PHRASE", confidence=0.5),
        Term(text="GPT", term_type="ENTITY", confidence=0.8),
        Term(text="neural network", term_type="NOUN_PHRASE", confidence=0.7)
    }

    results = filter_obj.filter_and_score(terms, total_articles=100)

    term_texts = [t[0] for t in results]
    assert "GPT" in term_texts
    assert "neural network" in term_texts
    assert "the" not in term_texts
    assert "and" not in term_texts

def test_domain_relevance_scoring():
    """Test that tech terms get higher scores"""
    filter_obj = DomainTermFilter()

    terms = {
        Term(text="GPT", term_type="ENTITY", confidence=0.8),
        Term(text="company", term_type="NOUN_PHRASE", confidence=0.5)
    }

    results = filter_obj.filter_and_score(terms, total_articles=100)

    # GPT should have higher confidence than "company"
    gpt_score = [r[1] for r in results if r[0] == "GPT"][0]
    assert gpt_score > 0.5  # Should be high

def test_specificity_scoring():
    """Test that rare terms get higher specificity scores"""
    filter_obj = DomainTermFilter()

    # Common term (entity so it passes the filter)
    terms_common = {
        Term(text="AI Corp", term_type="ENTITY", confidence=0.5)
    }
    # Set high article frequency for "AI Corp"
    filter_obj.term_frequency["AI Corp"] = 50

    results = filter_obj.filter_and_score(terms_common, total_articles=100)
    model_score = [r[1] for r in results if r[0] == "AI Corp"][0]

    # Rare term
    terms_rare = {
        Term(text="backpropagation", term_type="NOUN_PHRASE", confidence=0.5)
    }
    filter_obj.term_frequency["backpropagation"] = 2

    results = filter_obj.filter_and_score(terms_rare, total_articles=100)
    backprop_score = [r[1] for r in results if r[0] == "backpropagation"][0]

    # Backpropagation (rare) should score higher than AI Corp (common)
    assert backprop_score > model_score
