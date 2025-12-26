"""Example: How to use EntityExtractor in other features.

This file demonstrates how to reuse the entity extraction system
across different parts of the ai-news application.
"""

from ai_news.entity_extractor import EntityExtractor
from ai_news.text_processor import TextProcessor
from ai_news.entity_types import EntityType, ExtractedEntity
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Initialize once (can be a singleton or app-level instance)
_text_processor = None
_entity_extractor = None


def get_entity_extractor() -> EntityExtractor:
    """Get a singleton instance of EntityExtractor.
    
    This should be called at app startup and reused throughout.
    """
    global _text_processor, _entity_extractor
    if _entity_extractor is None:
        _text_processor = TextProcessor()
        _entity_extractor = EntityExtractor(_text_processor, use_spacy=True)
        logger.info("EntityExtractor initialized")
    return _entity_extractor


# =============================================================================
# USAGE EXAMPLES FOR DIFFERENT FEATURES
# =============================================================================

def example_article_tagging(article_content: str) -> Dict[str, List[str]]:
    """Feature: Auto-tag articles with entities.
    
    Use this in the article processing pipeline to add entity tags.
    """
    extractor = get_entity_extractor()
    entities = extractor.extract_entities(article_content)
    
    tags = {
        'companies': [],
        'products': [],
        'technologies': [],
        'people': []
    }
    
    for entity in entities:
        if entity.entity_type == EntityType.COMPANY:
            tags['companies'].append(entity.text)
        elif entity.entity_type == EntityType.PRODUCT:
            tags['products'].append(entity.text)
        elif entity.entity_type == EntityType.TECHNOLOGY:
            tags['technologies'].append(entity.text)
        elif entity.entity_type == EntityType.PERSON:
            tags['people'].append(entity.text)
    
    return tags


def example_content_recommendation(user_read_articles: List[str], 
                                    candidate_article: str) -> float:
    """Feature: Recommend articles based on entity overlap.
    
    Calculate similarity between what user has read and candidate articles.
    """
    extractor = get_entity_extractor()
    
    # Extract entities from user's read articles
    user_entities = set()
    for article in user_read_articles:
        entities = extractor.extract_entities(article)
        user_entities.update(e.text.lower() for e in entities)
    
    # Extract entities from candidate article
    candidate_entities = extractor.extract_entities(candidate_article)
    candidate_entity_set = set(e.text.lower() for e in candidate_entities)
    
    # Calculate overlap score
    if not candidate_entity_set:
        return 0.0
    
    overlap = user_entities & candidate_entity_set
    score = len(overlap) / len(candidate_entity_set)
    
    return score


def example_trending_entities(articles: List[str], 
                               min_occurrences: int = 3) -> Dict[str, int]:
    """Feature: Find trending entities across recent articles.
    
    Use this for dashboards, trending topics, or analytics.
    """
    extractor = get_entity_extractor()
    entity_counts = {}
    
    for article in articles:
        entities = extractor.extract_entities(article)
        for entity in entities:
            # Only count high-confidence entities
            if entity.confidence >= 0.7:
                key = f"{entity.text}|{entity.entity_type.value}"
                entity_counts[key] = entity_counts.get(key, 0) + 1
    
    # Filter by minimum occurrences
    trending = {k: v for k, v in entity_counts.items() if v >= min_occurrences}
    
    # Sort by count
    return dict(sorted(trending.items(), key=lambda x: x[1], reverse=True))


def example_duplicate_detection(article1: str, article2: str) -> bool:
    """Feature: Detect duplicate articles using entity overlap.
    
    Two articles are likely duplicates if they share many entities.
    """
    extractor = get_entity_extractor()
    
    entities1 = extractor.extract_entities(article1)
    entities2 = extractor.extract_entities(article2)
    
    set1 = set(e.text.lower() for e in entities1 if e.confidence >= 0.7)
    set2 = set(e.text.lower() for e in entities2 if e.confidence >= 0.7)
    
    if not set1 or not set2:
        return False
    
    overlap = set1 & set2
    union = set1 | set2
    
    # High overlap threshold for duplicate detection
    similarity = len(overlap) / len(union)
    return similarity > 0.6


def example_personalized_feed(user_interests: List[str], 
                              articles: List[Dict[str, str]]) -> List[Dict]:
    """Feature: Personalize feed based on user's entity interests.
    
    Rank articles by relevance to user's interests.
    """
    extractor = get_entity_extractor()
    
    # Normalize user interests
    interest_set = set(i.lower() for i in user_interests)
    
    scored_articles = []
    for article in articles:
        entities = extractor.extract_entities(article['content'])
        entity_set = set(e.text.lower() for e in entities)
        
        # Calculate relevance score
        matches = interest_set & entity_set
        score = len(matches)
        
        scored_articles.append({**article, 'relevance_score': score})
    
    # Sort by relevance
    scored_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
    return scored_articles


def example_search_enhancement(query: str, articles: List[str]) -> List[str]:
    """Feature: Enhance search by expanding search with entities.
    
    Find articles that match query OR related entities.
    """
    extractor = get_entity_extractor()
    
    # Extract entities from query
    query_entities = extractor.extract_entities(query)
    query_terms = set(query.lower().split())
    query_terms.update(e.text.lower() for e in query_entities)
    
    # Find matching articles
    matching_articles = []
    for article in articles:
        article_lower = article.lower()
        # Check if any query term or entity appears in article
        if any(term in article_lower for term in query_terms):
            matching_articles.append(article)
    
    return matching_articles


def example_topic_clustering(articles: List[str]) -> Dict[str, List[str]]:
    """Feature: Cluster articles by their key entities.
    
    Group articles that talk about the same companies/products.
    """
    extractor = get_entity_extractor()
    
    clusters = {}
    
    for article in articles:
        entities = extractor.extract_entities(article)
        # Use top entity as cluster key
        if entities:
            top_entity = max(entities, key=lambda e: e.confidence)
            key = f"{top_entity.text}|{top_entity.entity_type.value}"
            
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(article[:100] + '...')  # Store preview
    
    return clusters


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================

class EntityAwareFeature:
    """Base class for features that need entity extraction.
    
    Inherit from this to easily add entity awareness to any feature.
    """
    
    def __init__(self):
        self.extractor = get_entity_extractor()
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Convenience method for subclasses."""
        return self.extractor.extract_entities(text)
    
    def get_companies(self, text: str) -> List[str]:
        """Get only company entities."""
        entities = self.extract_entities(text)
        return [e.text for e in entities if e.entity_type == EntityType.COMPANY]
    
    def get_products(self, text: str) -> List[str]:
        """Get only product entities."""
        entities = self.extract_entities(text)
        return [e.text for e in entities if e.entity_type == EntityType.PRODUCT]


if __name__ == '__main__':
    # Quick test
    test_article = """
    OpenAI announced GPT-4 Turbo, an improved version of their language model.
    Sam Altman, CEO of OpenAI, said the model will compete with Google's Gemini
    and Anthropic's Claude. Microsoft has been integrating these models into Copilot.
    """
    
    tags = example_article_tagging(test_article)
    print("Article Tags:")
    for category, items in tags.items():
        if items:
            print(f"  {category}: {', '.join(items)}")
    
    print("\nEntity extraction is ready to use across all features! 🐶")
