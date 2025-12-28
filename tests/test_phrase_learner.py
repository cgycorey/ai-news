import pytest
import tempfile
import os
from ai_news.phrase_learner import PhraseLearner
from ai_news.database import Database, Article

def test_learn_phrases_from_ai_articles():
    """Test that we extract discriminative phrases from AI articles."""
    # Use temp file database instead of :memory: to avoid initialization issues
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    try:
        db = Database(db_path)

        # Add AI articles (need at least 10 for learning)
        ai_articles = [
            Article(title="GPT-4 Launch", content="OpenAI launches GPT-4 large language model", url="http://test1", source_name="Test", ai_relevant=True, ai_keywords_found=["GPT", "language model"]),
            Article(title="Neural Network Breakthrough", content="New neural network architecture for computer vision", url="http://test2", source_name="Test", ai_relevant=True, ai_keywords_found=["neural network", "computer vision"]),
            Article(title="Machine Learning in Healthcare", content="Applying machine learning to medical diagnosis", url="http://test3", source_name="Test", ai_relevant=True, ai_keywords_found=["machine learning"]),
            Article(title="Deep Learning Advances", content="Deep learning techniques improve accuracy", url="http://test6", source_name="Test", ai_relevant=True, ai_keywords_found=["deep learning"]),
            Article(title="AI Research Progress", content="Artificial intelligence research shows promise", url="http://test7", source_name="Test", ai_relevant=True, ai_keywords_found=["artificial intelligence"]),
            Article(title="NLP Innovation", content="Natural language processing breakthrough", url="http://test8", source_name="Test", ai_relevant=True, ai_keywords_found=["natural language processing"]),
            Article(title="Robotics AI", content="Robotics uses machine learning for navigation", url="http://test9", source_name="Test", ai_relevant=True, ai_keywords_found=["robotics", "machine learning"]),
            Article(title="Computer Vision AI", content="Computer vision applications expand", url="http://test10", source_name="Test", ai_relevant=True, ai_keywords_found=["computer vision"]),
            Article(title="Transformer Models", content="Transformer architecture revolutionizes NLP", url="http://test11", source_name="Test", ai_relevant=True, ai_keywords_found=["transformer", "nlp"]),
            Article(title="Reinforcement Learning", content="Reinforcement learning masters games", url="http://test12", source_name="Test", ai_relevant=True, ai_keywords_found=["reinforcement learning"]),
            Article(title="Large Language Model", content="Large language models scale up", url="http://test13", source_name="Test", ai_relevant=True, ai_keywords_found=["large language model"]),
        ]
        for article in ai_articles:
            db.save_article(article, auto_tag=False)

        # Add non-AI articles
        non_ai_articles = [
            Article(title="Stock Market Update", content="Markets rally as earnings beat expectations", url="http://test4", source_name="Test", ai_relevant=False, ai_keywords_found=[]),
            Article(title="Baseball Game Tonight", content="The game starts at 7pm", url="http://test5", source_name="Test", ai_relevant=False, ai_keywords_found=[]),
        ]
        for article in non_ai_articles:
            db.save_article(article, auto_tag=False)

        # Learn phrases
        learner = PhraseLearner(db)
        phrases = learner.learn_from_database()

        # Verify high-confidence AI phrases
        assert 'large language model' in phrases or 'language model' in phrases
        assert 'neural network' in phrases
        assert 'machine learning' in phrases

        # Verify non-AI words are not in phrases
        assert 'stock market' not in phrases or phrases.get('stock market', 0) < 0.3
        assert 'baseball' not in phrases
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
