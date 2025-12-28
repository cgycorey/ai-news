import pytest
import tempfile
import os
from ai_news.confidence_scorer import ConfidenceScorer
from ai_news.database import Database, Article


def test_high_confidence_for_ai_entities():
    """Test that articles with AI entities get high confidence."""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    try:
        db = Database(db_path)
        scorer = ConfidenceScorer(db)

        # Article with known AI entities
        article = Article(
            title="OpenAI releases GPT-4",
            content="OpenAI announced GPT-4, a large language model",
            url="http://test",
            source_name="Test",
            ai_relevant=False
        )

        confidence = scorer.calculate_confidence(article)

        assert confidence >= 0.7, "Should be high confidence with AI entities"
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_low_confidence_for_gaming():
    """Test that gaming articles get low confidence."""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    try:
        db = Database(db_path)
        scorer = ConfidenceScorer(db)

        article = Article(
            title="Rainbow Six Siege Tournament",
            content="Esports competition starts tonight",
            url="http://test",
            source_name="Gaming Weekly",
            category="gaming"
        )

        confidence = scorer.calculate_confidence(article)

        assert confidence < 0.5, "Should be low confidence for gaming content"
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_boost_from_learned_phrases():
    """Test that learned phrases boost confidence."""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    try:
        db = Database(db_path)

        # Add AI articles with "machine learning"
        for i in range(10):
            article = Article(
                title=f"ML Research {i}",
                content="Advances in machine learning and neural networks",
                url=f"http://test{i}",
                source_name="AI News",
                ai_relevant=True,
                ai_keywords_found=["machine learning"]
            )
            db.save_article(article, auto_tag=False)

        scorer = ConfidenceScorer(db)

        # New article with same phrase but no AI entities
        article = Article(
            title="New ML Framework",
            content="A new machine learning framework released",
            url="http://new",
            source_name="Tech Blog"
        )

        confidence = scorer.calculate_confidence(article)

        assert confidence >= 0.6, "Should be boosted by learned 'machine learning' phrase"
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_review_status_mapping():
    """Test that confidence scores map to correct review status."""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    try:
        db = Database(db_path)
        scorer = ConfidenceScorer(db)

        # Test auto status (>= 0.7)
        assert scorer.get_review_status(0.7) == 'auto'
        assert scorer.get_review_status(0.8) == 'auto'
        assert scorer.get_review_status(1.0) == 'auto'

        # Test reviewed status (0.5 - 0.69)
        assert scorer.get_review_status(0.5) == 'reviewed'
        assert scorer.get_review_status(0.6) == 'reviewed'
        assert scorer.get_review_status(0.69) == 'reviewed'

        # Test rejected status (< 0.5)
        assert scorer.get_review_status(0.0) == 'rejected'
        assert scorer.get_review_status(0.3) == 'rejected'
        assert scorer.get_review_status(0.49) == 'rejected'
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_source_boost():
    """Test that AI-related sources get confidence boost."""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    try:
        db = Database(db_path)
        scorer = ConfidenceScorer(db)

        # AI-related source
        article = Article(
            title="Tech News",
            content="Some technology article",
            url="http://test",
            source_name="AI News Weekly"
        )

        confidence = scorer.calculate_confidence(article)

        # Should get at least 0.1 from source boost
        assert confidence >= 0.1, "Should get boost from AI-related source"
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
