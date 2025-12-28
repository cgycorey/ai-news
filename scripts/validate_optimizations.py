#!/usr/bin/env python3
"""Validate NLP optimization performance and accuracy."""

import sys
import time
import tempfile
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_news.database import Database, Article
from ai_news.confidence_scorer import ConfidenceScorer
from ai_news.performance_metrics import get_metrics
from ai_news.config import get_performance_config


def test_collection_performance():
    """Test collection speed with optimizations."""
    print("\n=== Testing Collection Performance ===")
    
    fd, db_path = tempfile.mkstemp(suffix='.db')
    try:
        db = Database(db_path)
        db.init_database()
        metrics = get_metrics()
        
        # Create test articles
        test_articles = [
            Article(
                title=f'AI Article {i}',
                content=f'OpenAI GPT-{i} and machine learning advances',
                url=f'http://test{i}',
                source_name='TechCrunch AI' if i % 3 == 0 else 'Random Blog'
            )
            for i in range(50)
        ]
        
        start = time.time()
        
        scorer = ConfidenceScorer(db)
        # Initialize phrases to empty to avoid DB query
        scorer._learned_phrases = {}
        
        for article in test_articles:
            confidence = scorer.calculate_confidence(article)
        
        duration = time.time() - start
        
        summary = metrics.get_summary()
        print(f"Processed {len(test_articles)} articles in {duration:.2f}s")
        print(f"Average: {duration/len(test_articles):.3f}s per article")
        print(f"spaCy calls: {summary['spaCy_calls']}/{summary['articles_processed']} ({summary['spaCy_usage_percent']:.1f}%)")
        
        return duration / len(test_articles)
    finally:
        import os
        try:
            os.close(fd)
        except:
            pass
        try:
            os.unlink(db_path)
        except:
            pass


def test_accuracy_preservation():
    """Test that accuracy is maintained."""
    print("\n=== Testing Accuracy Preservation ===")
    
    fd, db_path = tempfile.mkstemp(suffix='.db')
    try:
        db = Database(db_path)
        db.init_database()
        scorer = ConfidenceScorer(db)
        # Initialize phrases to empty to avoid DB query
        scorer._learned_phrases = {}
        
        # Known AI articles
        ai_articles = [
            Article(title='OpenAI releases GPT-4', content='OpenAI announces GPT-4 large language model', url='http://ai1', source_name='AI News'),
            Article(title='Google Gemini Launch', content='Google launches Gemini AI model', url='http://ai2', source_name='TechCrunch'),
            Article(title='Transformer Architecture', content='Attention is all you need for NLP', url='http://ai3', source_name='MIT Tech Review'),
        ]
        
        # Known non-AI articles
        non_ai_articles = [
            Article(title='Stock Market Rally', content='Markets surge on earnings', url='http://non1', source_name='Finance News'),
            Article(title='Baseball Game', content='The game starts at 7pm', url='http://non2', source_name='Sports News'),
        ]
        
        # Test AI articles
        ai_correct = 0
        for article in ai_articles:
            confidence = scorer.calculate_confidence(article)
            if confidence >= 0.7:
                ai_correct += 1
                print(f"✅ {article.title[:30]:30} | Conf={confidence:.2f}")
            else:
                print(f"❌ {article.title[:30]:30} | Conf={confidence:.2f} (should be >=0.7)")
        
        # Test non-AI articles
        non_ai_correct = 0
        for article in non_ai_articles:
            confidence = scorer.calculate_confidence(article)
            if confidence < 0.7:
                non_ai_correct += 1
                print(f"✅ {article.title[:30]:30} | Conf={confidence:.2f}")
            else:
                print(f"❌ {article.title[:30]:30} | Conf={confidence:.2f} (should be <0.7)")
        
        accuracy = (ai_correct + non_ai_correct) / (len(ai_articles) + len(non_ai_articles))
        print(f"\nAccuracy: {accuracy*100:.1f}% ({ai_correct + non_ai_correct}/{len(ai_articles) + len(non_ai_articles)})")
        
        return accuracy >= 0.80  # Accept 80%+ accuracy (test is small)
    finally:
        import os
        try:
            os.close(fd)
        except:
            pass
        try:
            os.unlink(db_path)
        except:
            pass


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("NLP Optimization Validation")
    print("=" * 60)
    
    # Show current config
    config = get_performance_config()
    print(f"\nCurrent config: use_spacy_in_collection = '{config.use_spacy_in_collection}'")
    
    # Run tests
    avg_time = test_collection_performance()
    accuracy_ok = test_accuracy_preservation()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Average time per article: {avg_time:.3f}s")
    print(f"Accuracy maintained: {'✅ PASS' if accuracy_ok else '❌ FAIL'}")
    
    # Performance targets
    if avg_time < 0.5:  # Target: < 0.5s per article
        print(f"✅ Performance target met (< 0.5s)")
    else:
        print(f"⚠️  Performance slow ({avg_time:.3f}s > 0.5s target)")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
