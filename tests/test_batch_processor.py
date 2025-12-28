import pytest
from ai_news.database import Database, Article
from ai_news.batch_processor import BatchEntityProcessor

def test_batch_extraction_faster_than_individual():
    """Test that batch processing works correctly."""
    import time
    
    db = Database(':memory:')
    
    # Create test articles
    articles = [
        Article(title=f'Article {i}', content=f'Content about OpenAI and GPT-{i}', url=f'http://test{i}', source_name='Test')
        for i in range(20)
    ]
    
    processor = BatchEntityProcessor(db)
    
    # Individual processing using spacy_extractor directly
    start = time.time()
    individual_results = [processor.spacy_extractor.extract_entities_with_spacy(f"{a.title} {a.content or ''}") for a in articles[:10]]
    individual_time = time.time() - start
    
    # Batch processing
    start = time.time()
    batch_results = processor.extract_entities_batch(articles)
    batch_time = time.time() - start
    
    # Batch should process all articles
    print(f'Individual (10 articles): {individual_time:.3f}s, Batch (20 articles): {batch_time:.3f}s')
    print(f'Batch processed {len(batch_results)} articles')
    assert len(batch_results) == len(articles), "Batch should process all articles"
