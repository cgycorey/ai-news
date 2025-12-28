"""Batch processing for efficient NLP operations."""

import logging
from typing import List, Dict, Any
from .database import Database, Article
from .entity_extractor import EntityExtractor
from .spacy_utils import get_spacy_model
from .config import get_performance_config

logger = logging.getLogger(__name__)


class BatchEntityProcessor:
    """Process multiple articles efficiently using batch operations."""
    
    def __init__(self, database: Database):
        self.database = database
        from .text_processor import TextProcessor
        self.text_processor = TextProcessor()
        self.entity_extractor = EntityExtractor(self.text_processor, use_spacy=False)
        self.spacy_extractor = EntityExtractor(self.text_processor, use_spacy=True)
        
    def extract_entities_batch(self, articles: List[Article]) -> Dict[int, List[Any]]:
        """
        Extract entities from multiple articles using batch processing.
        
        Args:
            articles: List of Article objects
            
        Returns:
            Dict mapping article ID to list of entities
        """
        perf_config = get_performance_config()
        results = {}
        
        # Separate articles that need spaCy vs pattern-only
        needs_spacy = []
        pattern_only = []
        
        for article in articles:
            if perf_config.use_spacy_in_collection == "full":
                needs_spacy.append(article)
            elif perf_config.use_spacy_in_collection == "hybrid":
                # Quick pattern check first
                pattern_entities = self.entity_extractor.extract_entities(
                    f"{article.title} {article.content or ''}"
                )
                
                # Use spaCy for uncertain cases
                if len(pattern_entities) == 0:
                    needs_spacy.append(article)
                else:
                    pattern_only.append((article, pattern_entities))
            else:  # pattern-only
                pattern_only.append((article, []))
        
        # Process pattern-only articles
        for article, pattern_entities in pattern_only:
            if not pattern_entities:
                pattern_entities = self.entity_extractor.extract_entities(
                    f"{article.title} {article.content or ''}"
                )
            results[article.id or id(article)] = pattern_entities
        
        # Batch process spaCy articles
        if needs_spacy and self.spacy_extractor.nlp:
            results.update(self._batch_spacy_process(needs_spacy))
        
        return results
    
    def _batch_spacy_process(self, articles: List[Article]) -> Dict[int, List[Any]]:
        """Process articles with spaCy in batches."""
        perf_config = get_performance_config()
        batch_size = perf_config.spaCy_batch_size
        nlp = self.spacy_extractor.nlp
        
        results = {}
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            texts = [f"{a.title} {a.content or ''}" for a in batch]
            
            # Process batch with spaCy
            docs = nlp.pipe(texts, n_process=1, batch_size=batch_size)
            
            for article, doc in zip(batch, docs):
                entities = self.spacy_extractor.extract_entities_with_spacy(doc.text)
                results[article.id or id(article)] = entities
        
        return results
